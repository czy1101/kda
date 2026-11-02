import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "record_knowledge.py"
_SPEC = importlib.util.spec_from_file_location("record_knowledge", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
KnowledgeError = _MODULE.KnowledgeError
create_knowledge_record = _MODULE.create_knowledge_record
publish = _MODULE.publish
_validate = _MODULE.validate_record


def _workspace(tmp_path, *, correctness="pass", decision="keep", candidate_values=None):
    workspace = tmp_path / "task"
    (workspace / ".kernelpilot").mkdir(parents=True)
    (workspace / "runs").mkdir()
    task = {
        "version": 1,
        "task": {"name": "gla-test", "description": "test"},
        "target": {"backend": "hygon", "device": "BW3000", "arch": "gfx936", "language": "triton"},
        "implementation": {"path": "src/gla.py", "entrypoint": "gla.chunk_gla"},
        "goal": {"metric": "latency", "direction": "minimize", "aggregation": "geomean_ratio", "max_regression": 0.02},
        "benchmark": {"shapes": ["small", "large"]},
    }
    (workspace / ".kernelpilot/task.yaml").write_text(yaml.safe_dump(task), encoding="utf-8")
    candidate = {
        "id": "c001",
        "hypothesis": "reduce redundant staging",
        "correctness": correctness,
        "decision": decision,
        "metric": "latency",
        "unit": "ms",
    }
    (workspace / "runs/candidates.jsonl").write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    values = candidate_values or {"small": 0.9, "large": 0.8}
    with (workspace / "runs/benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate", "shape", "metric_value", "baseline_value", "unit", "status"])
        writer.writeheader()
        for shape, value in values.items():
            writer.writerow({"candidate": "c001", "shape": shape, "metric_value": value, "baseline_value": 1.0, "unit": "ms", "status": "pass"})
    return workspace


def _record(workspace, **kwargs):
    params = {
        "workspace": workspace,
        "candidate_id": "c001",
        "title": "LDS tiling improvement",
        "mechanism": "Reuses an operand tile in LDS.",
        "applicability": "Validated for the listed cases.",
        "lesson": "Check resource pressure with the latency result.",
        "limitations": "Other shapes are not validated.",
    }
    params.update(kwargs)
    return create_knowledge_record(**params)


def test_publish_requires_pass_and_significant_benchmark(tmp_path):
    workspace = _workspace(tmp_path)
    record, relative, markdown = _record(workspace)
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/optimization-knowledge.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(record)
    out = publish(record, relative, markdown, tmp_path / "knowledge")
    assert out.is_file()
    assert out.with_suffix(".json").is_file()
    assert "gfx936" in out.read_text(encoding="utf-8")


def test_rejects_candidate_not_promoted(tmp_path):
    workspace = _workspace(tmp_path, decision="revise")
    with pytest.raises(KnowledgeError, match="decision=keep"):
        _record(workspace)


def test_rejects_correctness_failure(tmp_path):
    workspace = _workspace(tmp_path, correctness="fail")
    with pytest.raises(KnowledgeError, match="correctness must be pass"):
        _record(workspace)


def test_rejects_insignificant_or_regressed_candidate(tmp_path):
    workspace = _workspace(tmp_path, candidate_values={"small": 0.99, "large": 0.99})
    with pytest.raises(KnowledgeError, match="below the required"):
        _record(workspace)

    workspace = _workspace(tmp_path / "regressed", candidate_values={"small": 0.8, "large": 1.1})
    with pytest.raises(KnowledgeError, match="regresses"):
        _record(workspace)


def test_requires_every_configured_shape(tmp_path):
    workspace = _workspace(tmp_path, candidate_values={"small": 0.8})
    with pytest.raises(KnowledgeError, match="lacks configured benchmark shapes"):
        _record(workspace)


def test_record_binds_ledger_digests_and_workspace(tmp_path):
    workspace = _workspace(tmp_path)
    record, _, _ = _record(workspace)
    ledger = record["evidence_ledger"]
    assert ledger["candidates_sha256"] == hashlib.sha256(
        (workspace / "runs/candidates.jsonl").read_bytes()
    ).hexdigest()
    assert ledger["benchmark_sha256"] == hashlib.sha256(
        (workspace / "runs/benchmark.csv").read_bytes()
    ).hexdigest()
    assert record["task_workspace"] == workspace.as_posix()


def test_validate_record_rejects_missing_required_field(tmp_path):
    workspace = _workspace(tmp_path)
    record, _, _ = _record(workspace)
    broken = dict(record)
    broken.pop("mechanism")
    with pytest.raises(KnowledgeError, match="violates its schema"):
        _validate(broken)
    broken = dict(record, status="maybe")
    with pytest.raises(KnowledgeError, match="violates its schema"):
        _validate(broken)


def test_record_is_validated_against_the_schema(tmp_path):
    workspace = _workspace(tmp_path)
    with pytest.raises(KnowledgeError, match="mechanism must not be empty"):
        _record(workspace, mechanism="   ")


def test_publish_regenerates_the_index(tmp_path):
    workspace = _workspace(tmp_path)
    record, relative, markdown = _record(workspace)
    root = tmp_path / "note"
    out = publish(record, relative, markdown, root)
    index = (root / "INDEX.md").read_text(encoding="utf-8")
    assert "do not edit by hand" in index
    assert out.relative_to(root).as_posix() in index
    assert "+33.33% aggregate" in index or "aggregate" in index


def test_update_revises_a_note_in_place(tmp_path):
    workspace = _workspace(tmp_path)
    record, relative, markdown = _record(workspace)
    root = tmp_path / "note"
    publish(record, relative, markdown, root)
    with pytest.raises(KnowledgeError, match="refusing to overwrite"):
        publish(record, relative, markdown, root)

    created_at = record["created_at"]
    revised, relative, markdown = _record(workspace, limitations="Revised after re-validation.")
    out = publish(revised, relative, markdown, root, update=True)
    stored = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert stored["version"] == 2
    assert stored["created_at"] == created_at
    assert stored["updated_at"]
    assert "Revised after re-validation." in out.read_text(encoding="utf-8")


def test_note_dir_controls_the_note_location(tmp_path):
    workspace = _workspace(tmp_path)
    _, relative, _ = _record(workspace, note_dir="gla")
    assert relative.startswith("gla/")
    with pytest.raises(KnowledgeError, match="inside the knowledge root"):
        _record(workspace, note_dir="../escape")


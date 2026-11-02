"""Lifecycle tests: retiring notes, replacing them, and auditing freshness."""

import hashlib
import importlib.util
import json
from datetime import date
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


knowledge_index = _load("knowledge_index", "scripts/knowledge_index.py")
knowledge_audit = _load("knowledge_audit", "scripts/knowledge_audit.py")
record_rejection = _load("record_rejection", "scripts/record_rejection.py")
record_knowledge = _load("record_knowledge", "scripts/record_knowledge.py")


GUIDE = """---
id: {id}
kind: guide
status: {status}
device: BW
arch: gfx936
scope: "{scope}"
revalidate_after: {due}
---

# {title}

Body text that must survive a metadata rewrite.
"""


def _guide(root: Path, name: str, **overrides) -> Path:
    values = {
        "id": name,
        "status": "confirmed",
        "scope": "scope",
        "due": "2027-01-31",
        "title": name.replace("-", " ").title(),
    }
    values.update(overrides)
    path = root / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(GUIDE.format(**values), encoding="utf-8")
    return path


def _result(
    root: Path,
    relative: str,
    *,
    note_id: str = "c001",
    status: str = "confirmed",
    operator: str = "demo.op",
    extra: dict | None = None,
) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": note_id,
        "title": "Demo result",
        "status": status,
        "backend": "hygon",
        "device": "BW",
        "arch": "gfx936",
        "operator": operator,
        "improvement_percent": 12.0,
    }
    record.update(extra or {})
    path.with_suffix(".json").write_text(json.dumps(record), encoding="utf-8")
    path.write_text("# Demo result\n\nNumbers.\n", encoding="utf-8")
    return path


def _codes(findings):
    return [item.code for item in findings]


def test_mark_superseded_retires_a_result_note(tmp_path):
    _result(tmp_path, "op/c001-demo.md")
    path = knowledge_index.mark_superseded(tmp_path, "c001", "op/c002-demo.md")
    record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert record["status"] == "superseded"
    assert record["superseded_by"] == "op/c002-demo.md"
    assert "Status: superseded by `op/c002-demo.md`" in path.read_text(encoding="utf-8")
    index = (tmp_path / "INDEX.md").read_text(encoding="utf-8")
    assert "## Superseded results" in index


def test_mark_superseded_rewrites_guide_front_matter(tmp_path):
    path = _guide(tmp_path, "demo-guide")
    knowledge_index.mark_superseded(tmp_path, "demo-guide", "demo-guide-v2.md")
    text = path.read_text(encoding="utf-8")
    front = knowledge_index.parse_front_matter(path)
    assert front["status"] == "superseded"
    assert front["superseded_by"] == "demo-guide-v2.md"
    assert "Body text that must survive" in text


def test_find_note_rejects_unknown_and_ambiguous_ids(tmp_path):
    _guide(tmp_path, "one")
    with pytest.raises(knowledge_index.KnowledgeIndexError, match="no note with id"):
        knowledge_index.find_note(tmp_path, "missing")
    _guide(tmp_path, "two", id="one")
    with pytest.raises(knowledge_index.KnowledgeIndexError, match="ambiguous"):
        knowledge_index.find_note(tmp_path, "one")


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "task"
    (workspace / ".kernelpilot").mkdir(parents=True)
    (workspace / "runs").mkdir()
    (workspace / ".kernelpilot/task.yaml").write_text(
        "\n".join(
            [
                "version: 1",
                "task: {name: demo, description: demo}",
                "target: {backend: hygon, device: BW, arch: gfx936, language: triton}",
                "implementation: {path: src/kernel.py, entrypoint: demo.op}",
                "goal: {metric: latency, direction: minimize, aggregation: sum_ratio, max_regression: 0.02}",
                "benchmark: {shapes: [s1]}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (workspace / "runs/candidates.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "id": candidate,
                    "hypothesis": "h",
                    "correctness": "pass",
                    "decision": "keep",
                    "metric": "latency",
                    "unit": "ms",
                }
            )
            for candidate in ("c001", "c002")
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "runs/benchmark.csv").write_text(
        "candidate,shape,metric_value,baseline_value,unit,status\n"
        "c001,s1,0.8,1.0,ms,pass\n"
        "c002,s1,0.7,1.0,ms,pass\n",
        encoding="utf-8",
    )
    return workspace


def _publish(tmp_path: Path, workspace: Path, candidate: str = "c001", title="Reuse an operand tile", **kwargs):
    record, relative, markdown = record_knowledge.create_knowledge_record(
        workspace=workspace,
        candidate_id=candidate,
        title=title,
        mechanism="Reuses a tile.",
        applicability="One shape.",
        lesson="Measure resources.",
        limitations="Others untested.",
    )
    root = tmp_path / "note"
    return record_knowledge.publish(record, relative, markdown, root, **kwargs), root


def test_publish_can_supersede_a_previous_note(tmp_path):
    workspace = _workspace(tmp_path)
    _, root = _publish(tmp_path, workspace)
    old = next(path for path in root.rglob("*.md") if path.name != "INDEX.md")
    _publish(
        tmp_path,
        workspace,
        candidate="c002",
        title="Reuse two operand tiles",
        supersede=old.name,
    )
    retired = json.loads(old.with_suffix(".json").read_text(encoding="utf-8"))
    assert retired["status"] == "superseded"
    assert retired["superseded_by"].endswith("c002-reuse-two-operand-tiles.md")
    assert "Superseded results" in (root / "INDEX.md").read_text(encoding="utf-8")


def test_rejection_can_supersede_a_previous_rejection(tmp_path):
    root = tmp_path / "note"
    root.mkdir()
    common = dict(
        device="BW",
        arch="gfx936",
        operator="demo.op",
        scope="scope",
        hypothesis="h",
        observed="o",
        reason="r",
        retry_when="w",
        rejected_on="2026-11-02",
    )
    first = record_rejection.render_rejection(
        title="First rejection", evidence=[], **common
    )
    record_rejection.write_rejection(root, first[0]["id"], first[1])
    code = record_rejection.main(
        [
            "--root",
            str(root),
            "--title",
            "Second rejection",
            "--device",
            "BW",
            "--arch",
            "gfx936",
            "--operator",
            "demo.op",
            "--scope",
            "scope",
            "--hypothesis",
            "h",
            "--observed",
            "o2",
            "--reason",
            "r2",
            "--retry-when",
            "w2",
            "--date",
            "2026-11-02",
            "--supersede",
            first[0]["id"],
        ]
    )
    assert code == 0
    index = (root / "INDEX.md").read_text(encoding="utf-8")
    assert "**superseded**" in index
    retired = knowledge_index.parse_front_matter(
        root / "rejected" / f"{first[0]['id']}.md"
    )
    assert retired["status"] == "superseded"
    assert retired["superseded_by"].endswith("second-rejection.md")


def test_audit_flags_a_changed_ledger(tmp_path):
    workspace = _workspace(tmp_path)
    root = tmp_path / "note"
    root.mkdir()
    ledger = {
        "candidates_sha256": hashlib.sha256(
            (workspace / "runs/candidates.jsonl").read_bytes()
        ).hexdigest(),
        "benchmark_sha256": hashlib.sha256(
            (workspace / "runs/benchmark.csv").read_bytes()
        ).hexdigest(),
    }
    _result(
        root,
        "op/c001-demo.md",
        extra={"task_workspace": workspace.as_posix(), "evidence_ledger": ledger},
    )
    assert knowledge_audit.audit(root) == []

    (workspace / "runs/benchmark.csv").write_text(
        "candidate,shape,metric_value,baseline_value,unit,status\nc001,s1,0.9,1.0,ms,pass\n",
        encoding="utf-8",
    )
    findings = knowledge_audit.audit(root)
    assert _codes(findings) == ["ledger-changed"]
    assert findings[0].severity == "error"
    assert knowledge_audit.main(["--root", str(root)]) == 1


def test_audit_flags_overdue_and_stale_notes(tmp_path):
    _guide(tmp_path, "overdue", due="2026-01-01")
    _guide(tmp_path, "stale", status="provisional", due="2027-01-31")
    findings = knowledge_audit.audit(tmp_path, as_of=date(2026, 6, 1))
    codes = _codes(findings)
    assert "revalidate-overdue" in codes
    assert knowledge_audit.main(
        ["--root", str(tmp_path), "--as-of", "2026-06-01"]
    ) == 0
    assert knowledge_audit.main(
        ["--root", str(tmp_path), "--as-of", "2026-06-01", "--strict"]
    ) == 1


def test_audit_flags_conflicting_confirmed_results(tmp_path):
    _result(tmp_path, "op/c001-a.md", note_id="c001")
    _result(tmp_path, "op/c002-b.md", note_id="c002")
    findings = knowledge_audit.audit(tmp_path)
    assert "conflicting-confirmed" in _codes(findings)


def test_audit_reports_an_unknown_workspace(tmp_path):
    _result(
        tmp_path,
        "op/c001-demo.md",
        extra={
            "task_workspace": (tmp_path / "gone").as_posix(),
            "evidence_ledger": {"candidates_sha256": "0" * 64, "benchmark_sha256": "0" * 64},
        },
    )
    findings = knowledge_audit.audit(tmp_path)
    assert _codes(findings) == ["workspace-missing"]
    assert findings[0].severity == "warning"
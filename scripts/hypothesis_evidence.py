"""Pre-registered, falsifiable mechanism hypotheses. Not performance approval."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import candidate_evidence as E
import compiler_evidence as C

ROOT = Path("runs/hypotheses")
ASSESSMENTS = Path("runs/hypothesis-assessments")
Error = E.EvidenceError


def policy(task):
    analysis = task.get("analysis") or {}
    if not isinstance(analysis, dict):
        raise Error("analysis must be a mapping")
    value = analysis.get("hypotheses", "optional")
    if value not in ("optional", "required"):
        raise Error("analysis.hypotheses must be optional or required")
    if value == "required" and C.config(task) is None:
        raise Error("required hypotheses need analysis.compiler")
    return value


def read(workspace, candidate):
    E.validate_candidate(candidate)
    path = C.safe_file(workspace, (ROOT / candidate / "record.json").as_posix())
    if not path.is_file():
        raise Error("register hypothesis before editing candidate " + candidate)
    value = json.loads(path.read_text())
    if value["spec"]["id"] != candidate:
        raise Error("hypothesis candidate identity mismatch")
    parent_dir, parent = C.load(workspace, value["spec"]["parent_capture"])
    if E.digest(parent_dir / "record.json") != value["parent_record_sha256"]:
        raise Error("hypothesis parent evidence changed")
    return path, value, parent


def register(workspace, task, spec):
    fields = {
        "id",
        "parent_capture",
        "statement",
        "change",
        "expected_effect",
        "risks",
        "disproof",
        "predictions",
    }
    if not isinstance(spec, dict) or set(spec) != fields:
        raise Error("hypothesis requires " + ", ".join(sorted(fields)))
    E.validate_candidate(spec["id"])
    if spec["id"] == "baseline":
        raise Error("baseline is reserved for initial capture")
    for field in fields - {"predictions"}:
        if not isinstance(spec[field], str) or not spec[field].strip():
            raise Error("hypothesis " + field + " must be nonempty text")
    predictions = spec["predictions"]
    if not isinstance(predictions, list) or not predictions:
        raise Error("hypothesis needs at least one measurable prediction")
    parent_dir, parent = C.load(workspace, spec["parent_capture"])
    if E.source_snapshot(workspace, task) != parent["before"]:
        raise Error(
            "register against current parent source before editing; stale parent capture"
        )
    parent_data = json.loads((parent_dir / "capture.json").read_text())
    cases = {(x["workload"], x["kernel"]): x for x in parent_data["cases"]}
    seen = set()
    for p in predictions:
        if not isinstance(p, dict) or set(p) != {
            "workload",
            "kernel",
            "kind",
            "name",
            "relation",
        }:
            raise Error("prediction requires workload/kernel/kind/name/relation")
        if any(not isinstance(x, str) or not x for x in p.values()):
            raise Error("prediction fields must be nonempty strings")
        key = (p["workload"], p["kernel"])
        if key not in cases:
            raise Error("prediction workload/kernel is absent from parent")
        identity = key + (p["kind"], p["name"])
        if identity in seen:
            raise Error("duplicate prediction target")
        seen.add(identity)
        if p["kind"] == "metric":
            if p["relation"] not in (
                "decrease",
                "increase",
                "equal",
                "nonincrease",
                "nondecrease",
            ):
                raise Error("invalid metric prediction relation")
            if p["name"] not in cases[key]["metrics"]:
                raise Error(
                    "prediction metric unavailable in parent; gather evidence first"
                )
        elif p["kind"] == "artifact":
            if (
                p["relation"] not in ("changed", "unchanged")
                or p["name"] not in cases[key]["artifacts"]
            ):
                raise Error("invalid artifact prediction")
        else:
            raise Error("prediction kind must be metric or artifact")
    entry = workspace / ROOT / spec["id"]
    entry.mkdir(parents=True, exist_ok=False)
    record = dict(
        version=1,
        registered_at=datetime.now(timezone.utc).isoformat(),
        spec=spec,
        parent_record_sha256=E.digest(parent_dir / "record.json"),
        parent_identity=parent["before"],
    )
    (entry / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def assess(workspace, task, candidate, capture_id):
    path, registered, parent = read(workspace, candidate)
    candidate_dir, captured = C.load(workspace, capture_id)
    if captured["candidate"] != candidate or captured["before"] != E.source_snapshot(
        workspace, task
    ):
        raise Error("candidate capture does not match current source/task/candidate")
    comparison = C.compare(workspace, registered["spec"]["parent_capture"], capture_id)
    index = {(x["workload"], x["kernel"]): x for x in comparison["cases"]}
    rows = []
    for prediction in registered["spec"]["predictions"]:
        case = index[(prediction["workload"], prediction["kernel"])]
        name, relation = prediction["name"], prediction["relation"]
        outcome = "unknown"
        if prediction["kind"] == "metric":
            observed = case["metrics"].get(name, {"status": "unavailable"})
            if observed["status"] == "comparable":
                d = observed["delta"]
                supported = {
                    "decrease": d < 0,
                    "increase": d > 0,
                    "equal": d == 0,
                    "nonincrease": d <= 0,
                    "nondecrease": d >= 0,
                }[relation]
                outcome = "supported" if supported else "not_supported"
        else:
            observed = case["stages"].get(name, {"status": "unavailable"})
            if observed["status"] in ("identical_bytes", "different_bytes"):
                supported = (observed["status"] == "different_bytes") == (
                    relation == "changed"
                )
                outcome = "supported" if supported else "not_supported"
        rows.append(dict(prediction=prediction, observed=observed, outcome=outcome))
    states = [r["outcome"] for r in rows]
    conclusion = (
        "not_supported"
        if "not_supported" in states
        else "unknown" if "unknown" in states else "supported"
    )
    result = dict(
        version=1,
        assessment_id=uuid.uuid4().hex,
        candidate=candidate,
        hypothesis_sha256=E.digest(path),
        candidate_capture=capture_id,
        candidate_record_sha256=E.digest(candidate_dir / "record.json"),
        comparison_id=comparison["comparison_id"],
        comparison_sha256=E.digest(
            workspace / C.COMPARISONS / comparison["comparison_id"] / "record.json"
        ),
        predictions=rows,
        mechanism_status=conclusion,
        performance_status="not_assessed",
        promotion_authorized=False,
        assessment_scope="declared_compiler_predictions_only",
        purpose="diagnostic_only_not_promotion",
    )
    entry = workspace / ASSESSMENTS / result["assessment_id"]
    entry.mkdir(parents=True)
    (entry / "record.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def require(workspace, task, candidate, section):
    """Optional workflow gate: assessment existence, never a performance verdict."""
    if policy(task) != "required" or candidate == "baseline":
        return
    path, record, parent = read(workspace, candidate)
    current = E.source_snapshot(workspace, task)
    if current["task_sha256"] != parent["before"]["task_sha256"] or C.protected(
        current
    ) != C.protected(parent["before"]):
        raise Error("hypothesis contract or protected inputs changed")
    if section not in ("benchmark", "promotion"):
        return
    for entry in sorted((workspace / ASSESSMENTS).glob("*/record.json"), reverse=True):
        value = json.loads(entry.read_text())
        if value.get("candidate") != candidate or value.get(
            "hypothesis_sha256"
        ) != E.digest(path):
            continue
        capture_dir, captured = C.load(workspace, value["candidate_capture"])
        if captured["before"] != current:
            continue
        if E.digest(capture_dir / "record.json") != value["candidate_record_sha256"]:
            raise Error("hypothesis assessment capture changed")
        comparison_path = C.safe_file(
            workspace,
            (C.COMPARISONS / value["comparison_id"] / "record.json").as_posix(),
        )
        E.load_record(comparison_path.parent)
        if E.digest(comparison_path) != value["comparison_sha256"]:
            raise Error("hypothesis assessment comparison changed")
        # Unsupported/unknown hypotheses may still be timed to measure a tradeoff;
        # the ordinary correctness, benchmark and confirmation gates still apply.
        return
    raise Error("assess hypothesis for this exact source before benchmark/promotion")

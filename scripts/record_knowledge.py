#!/usr/bin/env python3
"""Publish an evidence-gated, reusable optimization note to the KDA library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from knowledge_index import (  # noqa: E402
    KnowledgeIndexError,
    mark_superseded,
    write_index,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas/optimization-knowledge.schema.json"
RECORD_VERSION = 1


def _require_text(name: str, value: str) -> str:
    text = (value or "").strip()
    if not text:
        raise KnowledgeError(f"{name} must not be empty")
    return text


class KnowledgeError(ValueError):
    """Raised when a candidate does not meet the knowledge-promotion gates."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_record(record: dict[str, Any]) -> None:
    """Validate a record against the published schema before anything is written."""
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:  # pragma: no cover - repo damage
        raise KnowledgeError(f"cannot read knowledge schema: {error}") from error
    errors = sorted(
        Draft202012Validator(schema).iter_errors(record), key=lambda error: list(error.path)
    )
    if errors:
        details = "; ".join(
            f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise KnowledgeError(f"knowledge record violates its schema: {details}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise KnowledgeError(f"missing evidence ledger: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise KnowledgeError(f"{path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise KnowledgeError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def _load_workspace(workspace: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    task_path = workspace / ".kernelpilot/task.yaml"
    if not task_path.is_file():
        raise KnowledgeError(f"missing task contract: {task_path}")
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    if not isinstance(task, dict):
        raise KnowledgeError("task contract must be a YAML mapping")
    for section in ("task", "target", "implementation", "goal", "benchmark"):
        if not isinstance(task.get(section), dict):
            raise KnowledgeError(f"task contract is missing mapping: {section}")
    return task, task["target"]


def _candidate_record(workspace: Path, candidate_id: str) -> dict[str, Any]:
    records = _read_jsonl(workspace / "runs/candidates.jsonl")
    matches = [record for record in records if record.get("id") == candidate_id]
    if len(matches) != 1:
        raise KnowledgeError(
            f"candidate {candidate_id!r} must appear exactly once in candidates.jsonl"
        )
    record = matches[0]
    if record.get("decision") != "keep":
        raise KnowledgeError("only a KDA-approved candidate (decision=keep) can be published")
    if record.get("correctness") != "pass":
        raise KnowledgeError("candidate correctness must be pass")
    return record


def _ratio(candidate: float, baseline: float, direction: str) -> float:
    if candidate <= 0 or baseline <= 0:
        raise KnowledgeError("benchmark values must be positive")
    return baseline / candidate if direction == "minimize" else candidate / baseline


def _aggregate(rows: list[dict[str, Any]], direction: str, aggregation: str) -> float:
    ratios = [row["directional_ratio"] for row in rows]
    if aggregation == "sum_ratio":
        candidate_sum = sum(row["metric_value"] for row in rows)
        baseline_sum = sum(row["baseline_value"] for row in rows)
        return baseline_sum / candidate_sum if direction == "minimize" else candidate_sum / baseline_sum
    if aggregation == "mean_ratio":
        raw = sum(row["metric_value"] / row["baseline_value"] for row in rows) / len(rows)
        return 1.0 / raw if direction == "minimize" else raw
    if aggregation == "geomean_ratio":
        return math.exp(sum(math.log(value) for value in ratios) / len(ratios))
    if aggregation == "all":
        return min(ratios)
    if aggregation == "any":
        return max(ratios)
    raise KnowledgeError(f"unsupported aggregation: {aggregation}")


def _benchmark_rows(
    workspace: Path,
    task: dict[str, Any],
    candidate_id: str,
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    path = workspace / "runs/benchmark.csv"
    if not path.is_file():
        raise KnowledgeError(f"missing benchmark ledger: {path}")
    direction = task["goal"].get("direction", "minimize")
    if direction not in ("minimize", "maximize"):
        raise KnowledgeError(f"unsupported metric direction: {direction}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"candidate", "shape", "metric_value", "baseline_value", "unit", "status"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KnowledgeError("benchmark.csv missing columns: " + ", ".join(sorted(missing)))
        raw_rows = [row for row in reader if row["candidate"] == candidate_id]

    expected_shapes = task["benchmark"].get("shapes") or []
    by_shape: dict[str, dict[str, str]] = {}
    for row in raw_rows:
        shape = row.get("shape", "")
        if shape in by_shape:
            raise KnowledgeError(f"duplicate benchmark row for shape {shape!r}")
        by_shape[shape] = row
    if expected_shapes:
        missing_shapes = [shape for shape in expected_shapes if shape not in by_shape]
        if missing_shapes:
            raise KnowledgeError(f"candidate lacks configured benchmark shapes: {missing_shapes}")
        selected = [by_shape[shape] for shape in expected_shapes]
    else:
        selected = list(by_shape.values())
    if not selected:
        raise KnowledgeError(f"no benchmark rows found for candidate {candidate_id!r}")

    units = {row.get("unit", "") for row in selected}
    if len(units) != 1 or not next(iter(units)):
        raise KnowledgeError("all rows must use one non-empty metric unit")
    record_metric = candidate.get("metric")
    task_metric = task["goal"].get("metric")
    metric = record_metric or task_metric or "performance metric"
    record_unit = candidate.get("unit")
    if record_unit and record_unit not in units:
        raise KnowledgeError("candidate record unit does not match benchmark ledger")

    normalized = []
    for row in selected:
        if row.get("status") != "pass":
            raise KnowledgeError(f"benchmark row did not pass for shape {row.get('shape')!r}")
        try:
            value = float(row["metric_value"])
            baseline_value = float(row["baseline_value"])
        except (TypeError, ValueError) as error:
            raise KnowledgeError("benchmark values must be numeric") from error
        directional_ratio = _ratio(value, baseline_value, direction)
        normalized.append(
            {
                "shape": row["shape"],
                "metric_value": value,
                "baseline_value": baseline_value,
                "unit": next(iter(units)),
                "directional_ratio": directional_ratio,
                "improvement_percent": 100.0 * (directional_ratio - 1.0),
                "status": "pass",
            }
        )
    return normalized


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:64] or "optimization"


def create_knowledge_record(
    *,
    workspace: Path,
    candidate_id: str,
    title: str,
    mechanism: str,
    applicability: str,
    lesson: str,
    limitations: str,
    minimum_improvement_percent: float = 5.0,
    evidence_paths: list[str] | None = None,
    note_dir: str | None = None,
) -> tuple[dict[str, Any], str, str]:
    task, target = _load_workspace(workspace)
    title = _require_text("title", title)
    mechanism = _require_text("mechanism", mechanism)
    applicability = _require_text("applicability", applicability)
    lesson = _require_text("lesson", lesson)
    limitations = _require_text("limitations", limitations)
    candidate = _candidate_record(workspace, candidate_id)
    rows = _benchmark_rows(workspace, task, candidate_id, candidate)
    direction = task["goal"].get("direction", "minimize")
    aggregation = task["goal"].get("aggregation", "sum_ratio")
    aggregate_ratio = _aggregate(rows, direction, aggregation)
    improvement_percent = (aggregate_ratio - 1.0) * 100.0
    if improvement_percent < minimum_improvement_percent:
        raise KnowledgeError(
            f"measured aggregate improvement {improvement_percent:.2f}% is below "
            f"the required {minimum_improvement_percent:.2f}%"
        )

    max_regression = task["goal"].get("max_regression")
    if max_regression is not None:
        for row in rows:
            if row["improvement_percent"] < -100.0 * float(max_regression) - 1e-9:
                raise KnowledgeError(
                    f"shape {row['shape']!r} regresses by "
                    f"{-row['improvement_percent']:.2f}%, beyond the task's "
                    f"{100.0 * float(max_regression):.2f}% limit"
                )

    extras = []
    for evidence in evidence_paths or []:
        evidence_path = Path(evidence)
        if evidence_path.is_absolute() or ".." in evidence_path.parts:
            raise KnowledgeError(f"evidence path must stay inside task workspace: {evidence}")
        if not (workspace / evidence_path).is_file():
            raise KnowledgeError(f"evidence file does not exist: {evidence}")
        extras.append(evidence_path.as_posix())

    evidence = ["runs/candidates.jsonl", "runs/benchmark.csv", *extras]
    for field in ("profile_evidence", "generalization_evidence"):
        value = candidate.get(field)
        if isinstance(value, str) and value and value not in evidence:
            evidence.append(value)

    backend = str(target.get("backend", "unspecified"))
    device = str(target.get("device", "unspecified"))
    arch = str(target.get("arch", "unspecified"))
    operator = str(task["implementation"].get("entrypoint") or task["task"]["name"])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ledger = {
        "candidates_sha256": _sha256(workspace / "runs/candidates.jsonl"),
        "benchmark_sha256": _sha256(workspace / "runs/benchmark.csv"),
    }
    record = {
        "version": RECORD_VERSION,
        "id": candidate_id,
        "title": title,
        "status": "confirmed",
        "created_at": now,
        "task": task["task"]["name"],
        "task_workspace": workspace.as_posix(),
        "backend": backend,
        "device": device,
        "arch": arch,
        "language": target.get("language", "unspecified"),
        "operator": operator,
        "implementation": task["implementation"].get("path", "unspecified"),
        "metric": candidate.get("metric") or task["goal"].get("metric", "performance metric"),
        "direction": direction,
        "aggregation": aggregation,
        "unit": rows[0]["unit"],
        "aggregate_ratio": aggregate_ratio,
        "improvement_percent": improvement_percent,
        "minimum_improvement_percent": minimum_improvement_percent,
        "mechanism": mechanism,
        "applicability": applicability,
        "transferable_lesson": lesson,
        "limitations": limitations,
        "tested_shapes": rows,
        "evidence": evidence,
        "evidence_ledger": ledger,
    }
    validate_record(record)
    slug = _safe_slug(title)
    relative_dir = _note_directory(note_dir, operator)
    filename = f"{_safe_slug(candidate_id)}-{slug}"
    markdown = _render_markdown(record)
    return record, f"{relative_dir}/{filename}.md", markdown


def _note_directory(note_dir: str | None, operator: str) -> str:
    """Resolve the note subdirectory, defaulting to the operator's slug."""
    candidate = (note_dir or _safe_slug(operator)).strip("/")
    path = Path(candidate)
    if path.is_absolute() or ".." in path.parts:
        raise KnowledgeError(f"note directory must stay inside the knowledge root: {candidate}")
    normalized = "/".join(_safe_slug(part) for part in candidate.split("/") if part)
    if not normalized:
        raise KnowledgeError("note directory cannot be empty")
    return normalized


def _render_markdown(record: dict[str, Any]) -> str:
    lines = [
        f"# {record['title']}",
        "",
        "> Status: confirmed. This note was generated only after the KDA candidate passed correctness and its configured benchmark gates.",
        "",
        "## Target",
        "",
        f"- Backend/device/architecture: `{record['backend']}` / `{record['device']}` / `{record['arch']}`",
        f"- Operator: `{record['operator']}`",
        f"- Implementation: `{record['implementation']}`",
        f"- Language: `{record['language']}`",
        f"- Candidate: `{record['id']}` from task `{record['task']}`",
        "",
        "## Confirmed result",
        "",
        f"- Metric: `{record['metric']}` ({record['direction']}, `{record['unit']}`)",
        f"- Aggregate improvement: **{record['improvement_percent']:.2f}%**",
        f"- Aggregation: `{record['aggregation']}`; directional ratio `{record['aggregate_ratio']:.6f}`",
        "",
        "| Shape | Candidate | Baseline | Improvement | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in record["tested_shapes"]:
        lines.append(
            f"| `{row['shape']}` | {row['metric_value']:.6g} {row['unit']} | "
            f"{row['baseline_value']:.6g} {row['unit']} | "
            f"{row['improvement_percent']:+.2f}% | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism",
            "",
            record["mechanism"],
            "",
            "## Applicability",
            "",
            record["applicability"],
            "",
            "## Transferable lesson",
            "",
            record["transferable_lesson"],
            "",
            "## Limitations and guardrails",
            "",
            record["limitations"],
            "",
            "## Evidence",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in record["evidence"])
    ledger = record.get("evidence_ledger") or {}
    if ledger:
        lines.extend(
            [
                "",
                "Ledger digests at publication (a changed ledger means this note is stale):",
                "",
                f"- `runs/candidates.jsonl` sha256 `{ledger['candidates_sha256']}`",
                f"- `runs/benchmark.csv` sha256 `{ledger['benchmark_sha256']}`",
            ]
        )
    if record.get("task_workspace"):
        lines.extend(["", f"Task workspace: `{record['task_workspace']}`"])
    lines.extend(["", f"Recorded at: `{record['created_at']}`"])
    if record.get("updated_at"):
        lines.append(
            f"Updated at: `{record['updated_at']}` (version {record.get('version', 1)})"
        )
    lines.append("")
    return "\n".join(lines)


def publish(
    record: dict[str, Any],
    markdown_path: str,
    markdown: str,
    knowledge_root: Path,
    *,
    update: bool = False,
    supersede: str | None = None,
) -> Path:
    note_path = knowledge_root / markdown_path
    json_path = note_path.with_suffix(".json")
    if (note_path.exists() or json_path.exists()) and not update:
        raise KnowledgeError(
            f"refusing to overwrite existing knowledge note: {note_path} "
            "(pass --update to revise a published note)"
        )
    if update:
        if not json_path.is_file():
            raise KnowledgeError(f"--update requires an existing record: {json_path}")
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        record["created_at"] = existing.get("created_at", record["created_at"])
        record["version"] = int(existing.get("version", 1)) + 1
        record["status"] = existing.get("status", record.get("status", "confirmed"))
        record["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        validate_record(record)
        markdown = _render_markdown(record)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        write_index(knowledge_root)
    except (KnowledgeIndexError, OSError) as error:
        # The note itself is valid; a stale index is reported by
        # `knowledge_index.py --check` in CI, so do not lose the publication.
        print(f"record_knowledge: warning: index not regenerated: {error}", file=sys.stderr)
    if supersede:
        try:
            retired = mark_superseded(knowledge_root, supersede, markdown_path)
        except KnowledgeIndexError as error:
            raise KnowledgeError(
                f"note published, but `{supersede}` could not be superseded: {error}"
            ) from error
        print(f"superseded: {retired.relative_to(knowledge_root).as_posix()}")
    return note_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="KDA task workspace")
    parser.add_argument("--candidate", required=True, help="KDA candidate id")
    parser.add_argument("--title", required=True)
    parser.add_argument("--mechanism", required=True, help="Why this change improves the target device/operator")
    parser.add_argument("--applicability", required=True, help="Validated workload range and guard condition")
    parser.add_argument("--lesson", required=True, help="Transferable device/operator optimization lesson")
    parser.add_argument("--limitations", required=True, help="Known regressions, risks, and untested ranges")
    parser.add_argument("--evidence", action="append", default=[], help="Additional task-workspace evidence file; repeatable")
    parser.add_argument("--min-improvement", type=float, default=5.0, help="Minimum aggregate improvement percent (default: 5)")
    parser.add_argument(
        "--note-dir",
        help="Note subdirectory under the knowledge root; defaults to the operator slug",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Revise an existing note for this candidate, incrementing its version",
    )
    retirement = parser.add_mutually_exclusive_group()
    retirement.add_argument(
        "--revalidate-of",
        help="Note id or path that this note replaces; the old note is marked superseded",
    )
    retirement.add_argument(
        "--supersede",
        help="Note id or path to retire as superseded by this note",
    )
    parser.add_argument(
        "--knowledge-root",
        type=Path,
        help="Output root; defaults to backends/<backend>/note/ when present, otherwise knowledge/",
    )
    args = parser.parse_args(argv)
    if args.min_improvement < 0:
        parser.error("--min-improvement must be non-negative")
    try:
        record, relative_path, markdown = create_knowledge_record(
            workspace=args.workspace.resolve(),
            candidate_id=args.candidate,
            title=args.title,
            mechanism=args.mechanism,
            applicability=args.applicability,
            lesson=args.lesson,
            limitations=args.limitations,
            minimum_improvement_percent=args.min_improvement,
            evidence_paths=args.evidence,
            note_dir=args.note_dir,
        )
        vendor_notes = REPO_ROOT / "backends" / _safe_slug(record["backend"]) / "note"
        root = args.knowledge_root or (
            vendor_notes if vendor_notes.is_dir() else REPO_ROOT / "knowledge"
        )
        output = publish(
            record,
            relative_path,
            markdown,
            root.resolve(),
            update=args.update,
            supersede=args.supersede or args.revalidate_of,
        )
    except (KnowledgeError, OSError, yaml.YAMLError) as error:
        print(f"record_knowledge: {error}", file=sys.stderr)
        return 2
    print(output)
    print(f"aggregate improvement: {record['improvement_percent']:.2f}%")
    print(f"shapes recorded: {len(record['tested_shapes'])}")
    print(f"record version: {record.get('version', 1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

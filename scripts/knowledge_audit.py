#!/usr/bin/env python3
"""Audit a KDA knowledge root for notes that have gone stale or disagree.

The index answers "what do we know"; this answers "what should we re-check".
It reads the same notes as ``knowledge_index.py`` and reports:

* ``revalidate-overdue``  — a note's ``revalidate_after`` date has passed;
* ``ledger-changed``      — the task ledger no longer matches the digests the
  note was published from, so its measurements are superseded by the ledger;
* ``workspace-missing``   — the recorded task workspace is gone, so the note's
  evidence can no longer be re-derived;
* ``provisional-stale``   — provisional guidance older than the configured age;
* ``conflicting-confirmed`` — more than one confirmed result for the same
  operator/device, which needs a human decision, not silent coexistence;
* ``ledger-unbound``      — a result written before evidence binding existed.

Errors (``ledger-changed``) fail the command. Warnings do not, unless
``--strict`` is given; CI runs the default mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from knowledge_index import (  # noqa: E402
    INDEX_NAME,
    KnowledgeIndexError,
    default_roots,
    parse_front_matter,
)


SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
DEFAULT_PROVISIONAL_DAYS = 30


class Finding(NamedTuple):
    severity: str
    code: str
    path: str
    message: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        # YAML parses an unquoted ``2027-01-31`` into a date already.
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _audit_results(
    root: Path, as_of: date, provisional_days: int
) -> tuple[list[Finding], dict[tuple[str, str, str], list[str]]]:
    findings: list[Finding] = []
    by_operator: dict[tuple[str, str, str], list[str]] = {}
    for json_path in sorted(root.rglob("*.json")):
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            findings.append(
                Finding("error", "invalid-record", _relative(json_path, root), str(error))
            )
            continue
        if not isinstance(record, dict) or "id" not in record:
            continue
        relative = _relative(json_path.with_suffix(".md"), root)
        status = record.get("status", "confirmed")
        if status == "superseded":
            # Historical record: keep it listed and readable, but its ledgers
            # are expected to move on after retirement, so do not re-verify them.
            continue
        if status == "confirmed":
            # Group by note family directory (for example ``gla/``) so two
            # entrypoints of the same operator family are still compared.
            family = relative.split("/", 1)[0] if "/" in relative else relative
            key = (
                str(record.get("backend", "")),
                str(record.get("device", "")),
                family,
            )
            by_operator.setdefault(key, []).append(relative)

        ledger = record.get("evidence_ledger")
        workspace = record.get("task_workspace")
        if not isinstance(ledger, dict) or not ledger:
            findings.append(
                Finding(
                    "info",
                    "ledger-unbound",
                    relative,
                    "record has no ledger digests; re-publish with --update to bind it",
                )
            )
        elif not isinstance(workspace, str) or not workspace:
            findings.append(
                Finding("warning", "workspace-missing", relative, "record has no task workspace")
            )
        else:
            workspace_path = Path(workspace)
            if not workspace_path.is_dir():
                findings.append(
                    Finding(
                        "warning",
                        "workspace-missing",
                        relative,
                        f"task workspace is gone: {workspace}",
                    )
                )
            else:
                for filename, key in (
                    ("runs/candidates.jsonl", "candidates_sha256"),
                    ("runs/benchmark.csv", "benchmark_sha256"),
                ):
                    ledger_path = workspace_path / filename
                    if not ledger_path.is_file():
                        findings.append(
                            Finding(
                                "warning",
                                "workspace-missing",
                                relative,
                                f"ledger is gone: {filename}",
                            )
                        )
                        continue
                    actual = _sha256(ledger_path)
                    if actual != ledger.get(key):
                        findings.append(
                            Finding(
                                "error",
                                "ledger-changed",
                                relative,
                                f"{filename} changed since publication "
                                f"(recorded {ledger.get(key)}, now {actual})",
                            )
                        )

        if status == "provisional":
            created = _parse_date(record.get("created_at"))
            if created and (as_of - created).days > provisional_days:
                findings.append(
                    Finding(
                        "warning",
                        "provisional-stale",
                        relative,
                        f"provisional for {(as_of - created).days} days; promote or retire it",
                    )
                )
    return findings, by_operator


def _audit_guides(root: Path, as_of: date, provisional_days: int) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(root.rglob("*.md")):
        if path.name == INDEX_NAME or path.with_suffix(".json").is_file():
            continue
        front = parse_front_matter(path)
        if front is None:
            findings.append(
                Finding("error", "missing-front-matter", _relative(path, root), "no metadata")
            )
            continue
        relative = _relative(path, root)
        due = _parse_date(front.get("revalidate_after"))
        if due and due < as_of:
            findings.append(
                Finding(
                    "warning",
                    "revalidate-overdue",
                    relative,
                    f"was due for re-validation on {due.isoformat()}",
                )
            )
        if front.get("status") == "provisional":
            anchor = _parse_date(front.get("last_validated"))
            if anchor and (as_of - anchor).days > provisional_days:
                findings.append(
                    Finding(
                        "warning",
                        "provisional-stale",
                        relative,
                        f"provisional since {anchor.isoformat()}; promote or retire it",
                    )
                )
    return findings


def audit(
    root: Path,
    *,
    as_of: date | None = None,
    provisional_days: int = DEFAULT_PROVISIONAL_DAYS,
) -> list[Finding]:
    """Return every finding for one knowledge root, most severe first."""
    when = as_of or date.today()
    if not root.is_dir():
        raise KnowledgeIndexError(f"knowledge root does not exist: {root}")
    result_findings, by_operator = _audit_results(root, when, provisional_days)
    findings = result_findings + _audit_guides(root, when, provisional_days)
    for (backend, device, family), notes in sorted(by_operator.items()):
        if len(notes) > 1:
            findings.append(
                Finding(
                    "warning",
                    "conflicting-confirmed",
                    ", ".join(notes),
                    f"{len(notes)} confirmed results in note family `{family}` on "
                    f"{backend}/{device}; supersede the older one or record why both hold",
                )
            )
    return sorted(findings, key=lambda item: (SEVERITY_ORDER[item.severity], item.path, item.code))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=None,
        help="Knowledge root to audit; repeatable (default: every vendor note root)",
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=None,
        help="Evaluate dates as of YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--provisional-days",
        type=int,
        default=DEFAULT_PROVISIONAL_DAYS,
        help="Warn when provisional guidance is older than this many days (default: 30)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on warnings as well as errors",
    )
    args = parser.parse_args(argv)
    roots = [path.resolve() for path in (args.root or default_roots())]
    if not roots:
        print("knowledge_audit: no knowledge roots found", file=sys.stderr)
        return 2
    findings: list[Finding] = []
    try:
        for root in roots:
            findings.extend(
                audit(root, as_of=args.as_of, provisional_days=args.provisional_days)
            )
    except (KnowledgeIndexError, OSError, yaml.YAMLError) as error:
        print(f"knowledge_audit: {error}", file=sys.stderr)
        return 2

    if not findings:
        print("knowledge: no findings")
        return 0
    for item in findings:
        print(f"{item.severity:<7} {item.code:<21} {item.path}  {item.message}")
    counts = {severity: 0 for severity in SEVERITY_ORDER}
    for item in findings:
        counts[item.severity] += 1
    print(
        f"\n{counts['error']} error(s), {counts['warning']} warning(s), "
        f"{counts['info']} info"
    )
    if counts["error"] or (args.strict and counts["warning"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

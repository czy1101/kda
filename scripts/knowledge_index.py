#!/usr/bin/env python3
"""Regenerate ``INDEX.md`` for a KDA knowledge root from the notes themselves.

Each vendor note collection (``backends/<vendor>/note/``) and the generic
``knowledge/<backend>-<device>-<arch>/`` fallback are indexed the same way:

* confirmed results are ``*.json`` records with a sibling Markdown report;
* guides are Markdown notes that start with a YAML front matter block;
* rejections are guides whose ``kind`` is ``rejection`` and that live under a
  ``rejected/`` directory.

The index is generated, never hand-edited, so the catalog cannot drift from the
files. ``--check`` exits non-zero when a committed index is stale, which is what
CI runs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_NAME = "INDEX.md"
FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.S)

REQUIRED_FRONT_MATTER = ("id", "kind", "status", "device", "arch", "scope")
KINDS = ("guide", "rejection")
STATUSES = ("confirmed", "provisional", "superseded", "rejected")
RESULT_STATUSES = ("confirmed", "provisional", "superseded")

TITLE = "KDA optimization knowledge"


class KnowledgeIndexError(ValueError):
    """Raised when a note cannot be indexed."""


def parse_front_matter(path: Path) -> dict[str, Any] | None:
    text = path.read_text(encoding="utf-8")
    match = FRONT_MATTER.match(text)
    if not match:
        return None
    data = yaml.safe_load(match.group(1))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise KnowledgeIndexError(f"{path}: front matter must be a mapping")
    return data


def _title_from_heading(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def collect(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Classify every note under ``root`` into results, guides, and rejections."""
    if not root.is_dir():
        raise KnowledgeIndexError(f"knowledge root does not exist: {root}")
    results: list[dict[str, Any]] = []
    guides: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for json_path in sorted(root.rglob("*.json")):
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise KnowledgeIndexError(f"{json_path}: {error}") from error
        if not isinstance(record, dict) or "id" not in record:
            raise KnowledgeIndexError(f"{json_path}: not a knowledge record")
        status = record.get("status", "confirmed")
        if status not in RESULT_STATUSES:
            raise KnowledgeIndexError(f"{json_path}: unsupported result status {status!r}")
        markdown = json_path.with_suffix(".md")
        if not markdown.is_file():
            raise KnowledgeIndexError(f"{json_path}: missing sibling Markdown report")
        results.append(
            {
                "kind": "result",
                "id": str(record["id"]),
                "title": str(record.get("title") or markdown.stem),
                "status": status,
                "backend": record.get("backend", ""),
                "device": record.get("device", ""),
                "arch": record.get("arch", ""),
                "operator": record.get("operator", ""),
                "improvement_percent": record.get("improvement_percent"),
                "path": markdown.relative_to(root).as_posix(),
            }
        )
    for markdown in sorted(root.rglob("*.md")):
        if markdown.name == INDEX_NAME:
            continue
        if markdown.with_suffix(".json").is_file():
            continue  # indexed as a result
        front = parse_front_matter(markdown)
        relative = markdown.relative_to(root)
        if front is None:
            raise KnowledgeIndexError(
                f"{relative}: missing YAML front matter and no sibling JSON record"
            )
        missing = [field for field in REQUIRED_FRONT_MATTER if not front.get(field)]
        if missing:
            raise KnowledgeIndexError(
                f"{relative}: front matter is missing {', '.join(missing)}"
            )
        kind = front["kind"]
        status = front["status"]
        if kind not in KINDS:
            raise KnowledgeIndexError(f"{relative}: unsupported kind {kind!r}")
        if status not in STATUSES:
            raise KnowledgeIndexError(f"{relative}: unsupported status {status!r}")
        entry = {
            "kind": kind,
            "id": str(front["id"]),
            "title": str(front.get("title") or _title_from_heading(markdown)),
            "status": status,
            "device": str(front["device"]),
            "arch": str(front["arch"]),
            "operator": str(front.get("operator") or ""),
            "scope": str(front["scope"]),
            "last_validated": str(front.get("last_validated") or ""),
            "revalidate_after": str(front.get("revalidate_after") or ""),
            "path": relative.as_posix(),
        }
        if kind == "rejection":
            if "rejected" not in relative.parts:
                raise KnowledgeIndexError(
                    f"{relative}: kind 'rejection' must live under rejected/"
                )
            entry["retry_when"] = str(front.get("retry_when") or "")
            rejections.append(entry)
        else:
            guides.append(entry)
    return {"results": results, "guides": guides, "rejections": rejections}


def _result_line(entry: dict[str, Any]) -> str:
    device = "/".join(item for item in (entry["backend"], entry["device"], entry["arch"]) if item)
    detail = [f"`{device}`" if device else "", f"candidate `{entry['id']}`"]
    if isinstance(entry["improvement_percent"], (int, float)):
        detail.insert(1, f"{entry['improvement_percent']:+.2f}% aggregate")
    suffix = ""
    if entry["status"] != "confirmed":
        suffix = f" — **{entry['status']}**"
    return f"- [{entry['title']}]({entry['path']}) — {', '.join(item for item in detail if item)}{suffix}"


def _guide_line(entry: dict[str, Any]) -> str:
    detail = [f"`{entry['device']}/{entry['arch']}`"]
    if entry["last_validated"]:
        detail.append(f"validated {entry['last_validated']}")
    if entry["revalidate_after"]:
        detail.append(f"revalidate after {entry['revalidate_after']}")
    if entry["status"] != "confirmed":
        detail.append(f"**{entry['status']}**")
    scope = entry["scope"].rstrip(".")
    return (
        f"- [{entry['title']}]({entry['path']}) — {scope}. "
        f"Scope: {', '.join(detail)}."
    )


def _rejection_line(entry: dict[str, Any]) -> str:
    detail = [f"`{entry['device']}/{entry['arch']}`"]
    if entry["operator"]:
        detail.insert(0, f"`{entry['operator']}`")
    if entry["retry_when"]:
        detail.append(f"retry when {entry['retry_when'].rstrip('.')}")
    if entry["status"] != "rejected":
        detail.append(f"**{entry['status']}**")
    return f"- [{entry['title']}]({entry['path']}) — {'; '.join(detail)}."


def render(root: Path) -> str:
    collected = collect(root)
    results = sorted(collected["results"], key=lambda item: (item["id"], item["path"]))
    guides = sorted(collected["guides"], key=lambda item: (item["id"], item["path"]))
    rejections = sorted(collected["rejections"], key=lambda item: (item["id"], item["path"]))

    lines = [
        f"# {TITLE}",
        "",
        "<!-- Generated by scripts/knowledge_index.py; do not edit by hand. -->",
        "",
        "Notes are published only after KDA correctness, benchmark, and promotion",
        "gates pass. Guides record reusable method; rejections record what did not",
        "work so the same experiment is not repeated.",
        "",
    ]
    confirmed = [item for item in results if item["status"] == "confirmed"]
    provisional = [item for item in results if item["status"] == "provisional"]
    superseded = [item for item in results if item["status"] == "superseded"]

    lines.append("## Confirmed operator results")
    lines.append("")
    lines.extend(_result_line(item) for item in confirmed)
    if not confirmed:
        lines.append("_None yet._")
    lines.append("")

    if provisional:
        lines.append("## Provisional operator results")
        lines.append("")
        lines.append(
            "_Not yet re-validated on the recorded device/toolchain; treat as a"
            " starting hypothesis, not a rule._"
        )
        lines.append("")
        lines.extend(_result_line(item) for item in provisional)
        lines.append("")

    if superseded:
        lines.append("## Superseded results")
        lines.append("")
        lines.extend(_result_line(item) for item in superseded)
        lines.append("")

    lines.append("## Optimization guides")
    lines.append("")
    lines.extend(_guide_line(item) for item in guides)
    if not guides:
        lines.append("_None yet._")
    lines.append("")

    if rejections:
        lines.append("## Rejected approaches")
        lines.append("")
        lines.extend(_rejection_line(item) for item in rejections)
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def write_index(root: Path) -> tuple[Path, bool]:
    """Write ``root/INDEX.md``; return the path and whether it changed."""
    content = render(root)
    index = root / INDEX_NAME
    previous = index.read_text(encoding="utf-8") if index.is_file() else None
    if previous == content:
        return index, False
    index.write_text(content, encoding="utf-8")
    return index, True


def find_note(root: Path, identifier: str) -> Path:
    """Resolve a note by recorded id, relative path, or bare filename."""
    requested = PurePosixPath(identifier)
    if requested.suffix == ".md":
        direct = root / identifier
        if direct.is_file():
            return direct
        same_name = [
            path
            for path in sorted(root.rglob("*.md"))
            if path.name == requested.name and path.name != INDEX_NAME
        ]
        if len(same_name) == 1:
            return same_name[0]
        if not same_name:
            raise KnowledgeIndexError(f"no note named {identifier!r} under {root}")
        listed = ", ".join(path.relative_to(root).as_posix() for path in same_name)
        raise KnowledgeIndexError(
            f"ambiguous note name {identifier!r}: {listed}; pass a note path instead"
        )
    matches: list[Path] = []
    for path in sorted(root.rglob("*.md")):
        if path.name == INDEX_NAME:
            continue
        sibling = path.with_suffix(".json")
        if sibling.is_file():
            try:
                record = json.loads(sibling.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise KnowledgeIndexError(f"{sibling}: {error}") from error
            if isinstance(record, dict) and record.get("id") == identifier:
                matches.append(path)
        else:
            front = parse_front_matter(path) or {}
            if str(front.get("id")) == identifier:
                matches.append(path)
    if not matches:
        raise KnowledgeIndexError(f"no note with id {identifier!r} under {root}")
    if len(matches) > 1:
        listed = ", ".join(path.relative_to(root).as_posix() for path in matches)
        raise KnowledgeIndexError(
            f"ambiguous note id {identifier!r}: {listed}; pass a note path instead"
        )
    return matches[0]


def _rewrite_front_matter(path: Path, front: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    match = FRONT_MATTER.match(text)
    if not match:
        raise KnowledgeIndexError(f"{path}: no front matter to rewrite")
    body = text[match.end() :]
    block = yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip()
    path.write_text(f"---\n{block}\n---\n{body}", encoding="utf-8")


def mark_superseded(
    root: Path,
    identifier: str,
    superseded_by: str,
) -> Path:
    """Retire a note without deleting it, then refresh the index.

    Result notes keep their JSON record and get ``status: superseded`` plus
    ``superseded_by``; guides and rejections get the same pair in front matter.
    """
    path = find_note(root, identifier)
    json_path = path.with_suffix(".json")
    if json_path.is_file():
        record = json.loads(json_path.read_text(encoding="utf-8"))
        record["status"] = "superseded"
        record["superseded_by"] = superseded_by
        json_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        status_line = (
            f"> Status: superseded by `{superseded_by}`. The measurements below are "
            "historical; do not cite them as current guidance."
        )
        text = path.read_text(encoding="utf-8")
        replacement, count = re.subn(r"^> Status:.*$", status_line, text, count=1, flags=re.M)
        path.write_text(replacement if count else status_line + "\n\n" + text, encoding="utf-8")
    else:
        front = parse_front_matter(path)
        if front is None:
            raise KnowledgeIndexError(f"{path}: note has no front matter or JSON record")
        front["status"] = "superseded"
        front["superseded_by"] = superseded_by
        _rewrite_front_matter(path, front)
    write_index(root)
    return path


def default_roots(repo_root: Path = REPO_ROOT) -> list[Path]:
    roots: list[Path] = []
    notes = sorted((repo_root / "backends").glob("*/note"))
    roots.extend(path for path in notes if path.is_dir())
    generic = repo_root / "knowledge"
    if generic.is_dir():
        roots.extend(sorted(path for path in generic.iterdir() if path.is_dir()))
    return roots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=None,
        help="Knowledge root to index; repeatable (default: every vendor note root)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when an index is stale instead of rewriting it",
    )
    args = parser.parse_args(argv)
    roots = [path.resolve() for path in (args.root or default_roots())]
    if not roots:
        print("knowledge_index: no knowledge roots found", file=sys.stderr)
        return 2
    stale = 0
    try:
        for root in roots:
            if args.check:
                content = render(root)
                index = root / INDEX_NAME
                current = index.read_text(encoding="utf-8") if index.is_file() else None
                if current != content:
                    print(f"stale index: {index}", file=sys.stderr)
                    stale += 1
                else:
                    print(f"fresh index: {index}")
            else:
                index, changed = write_index(root)
                print(f"{'updated' if changed else 'unchanged'}: {index}")
    except (KnowledgeIndexError, OSError, yaml.YAMLError) as error:
        print(f"knowledge_index: {error}", file=sys.stderr)
        return 2
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())

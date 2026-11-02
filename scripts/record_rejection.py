#!/usr/bin/env python3
"""Record a rejected optimization attempt as negative knowledge.

Confirmed notes answer "what works". This recorder answers "what did not, and
under which condition it is worth retrying", so the same experiment is not
repeated by the next person. A rejection is deliberately *not* evidence-gated
the way a confirmed result is: the value is the documented failure, not a win.

The note is Markdown with YAML front matter under
``<knowledge-root>/rejected/`` and is indexed by ``scripts/knowledge_index.py``.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from knowledge_index import (  # noqa: E402
    INDEX_NAME,
    KnowledgeIndexError,
    mark_superseded,
    parse_front_matter,
    write_index,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class RejectionError(ValueError):
    """Raised when a rejection note is incomplete or would overwrite another note."""


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:64] or "rejection"


def _require(name: str, value: str) -> str:
    if not value or not value.strip():
        raise RejectionError(f"{name} must not be empty")
    return " ".join(value.split())


def render_rejection(
    *,
    title: str,
    device: str,
    arch: str,
    operator: str,
    scope: str,
    hypothesis: str,
    observed: str,
    reason: str,
    retry_when: str,
    evidence: list[str] | None = None,
    rejected_on: str,
) -> tuple[dict[str, Any], str]:
    front = {
        "id": _slug(title),
        "kind": "rejection",
        "status": "rejected",
        "device": device,
        "arch": arch,
        "operator": operator,
        "scope": scope,
        "rejected_on": rejected_on,
        "retry_when": retry_when,
    }
    body = [
        "---",
        yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip(),
        "---",
        "",
        f"# {title}",
        "",
        f"> Status: rejected on `{rejected_on}`. This is negative knowledge: it records a measured",
        "> failure, not an optimization result. Do not cite it as a win.",
        "",
        "## Attempted",
        "",
        f"- Device/architecture: `{device}` / `{arch}`",
        f"- Operator: `{operator}`",
        f"- Scope: {scope}",
        "",
        "## Hypothesis",
        "",
        hypothesis,
        "",
        "## What happened",
        "",
        observed,
        "",
        "## Why it was rejected",
        "",
        reason,
        "",
        "## When to retry",
        "",
        retry_when,
        "",
        "## Evidence",
        "",
    ]
    if evidence:
        body.extend(f"- `{item}`" for item in evidence)
    else:
        body.append("- No separately stored artifact; the observation above is the record.")
    body.append("")
    return front, "\n".join(body)


def write_rejection(
    root: Path,
    title: str,
    markdown: str,
    *,
    update: bool = False,
) -> Path:
    directory = root / "rejected"
    path = directory / f"{_slug(title)}.md"
    if path.exists() and not update:
        raise RejectionError(
            f"refusing to overwrite existing rejection note: {path} (pass --update)"
        )
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    try:
        write_index(root)
    except (KnowledgeIndexError, OSError) as error:
        print(f"record_rejection: warning: index not regenerated: {error}", file=sys.stderr)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="Knowledge root (default: backends/hygon/note)",
    )
    parser.add_argument("--title", required=True, help="One-line description of the rejected idea")
    parser.add_argument("--device", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--operator", default="", help="Operator or operator family")
    parser.add_argument("--scope", required=True, help="What exactly was tried, in one line")
    parser.add_argument("--hypothesis", required=True, help="The idea that was expected to help")
    parser.add_argument("--observed", required=True, help="Measured outcome, with numbers where available")
    parser.add_argument("--reason", required=True, help="Why the approach was rejected")
    parser.add_argument("--retry-when", required=True, help="Concrete condition that would justify retrying")
    parser.add_argument("--evidence", action="append", default=[], help="Artifact or note path; repeatable")
    parser.add_argument("--date", dest="rejected_on", default=date.today().isoformat())
    parser.add_argument("--update", action="store_true", help="Revise an existing rejection note")
    parser.add_argument(
        "--supersede",
        help="Note id or path that this rejection replaces; the old note is marked superseded",
    )
    parser.add_argument(
        "--revalidate-of",
        help="Alias of --supersede for a re-measured answer to the same question",
    )
    args = parser.parse_args(argv)
    try:
        root = (args.root or (REPO_ROOT / "backends/hygon/note")).resolve()
        if not root.is_dir():
            raise RejectionError(f"knowledge root does not exist: {root}")
        front, markdown = render_rejection(
            title=_require("--title", args.title),
            device=_require("--device", args.device),
            arch=_require("--arch", args.arch),
            operator=_require_optional(args.operator),
            scope=_require("--scope", args.scope),
            hypothesis=_require("--hypothesis", args.hypothesis),
            observed=_require("--observed", args.observed),
            reason=_require("--reason", args.reason),
            retry_when=_require("--retry-when", args.retry_when),
            evidence=args.evidence,
            rejected_on=_require("--date", args.rejected_on),
        )
        path = write_rejection(root, front["id"], markdown, update=args.update)
        retired = args.supersede or args.revalidate_of
        if retired:
            by = path.relative_to(root).as_posix()
            try:
                old = mark_superseded(root, retired, by)
            except KnowledgeIndexError as error:
                raise RejectionError(
                    f"note published, but `{retired}` could not be superseded: {error}"
                ) from error
            print(f"superseded: {old.relative_to(root).as_posix()}")
    except (RejectionError, KnowledgeIndexError, OSError) as error:
        print(f"record_rejection: {error}", file=sys.stderr)
        return 2
    print(path)
    return 0


def _require_optional(value: str) -> str:
    return " ".join(value.split()) if value else ""


if __name__ == "__main__":
    raise SystemExit(main())

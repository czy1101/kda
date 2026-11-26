#!/usr/bin/env python3
"""Offline identity manifest and layered comparison for compiler artifacts.

The tool answers one question: *are these two compiler-artifact sets the same
bytes, and which named stages do they contain?* It deliberately answers nothing
else.

Scope and non-goals
-------------------
- No device, no ``torch``/``torch_gcu``/``triton`` import, no network. It reads
  files from a directory the caller names.
- It never writes to, renames, normalizes, or deletes the input artifacts. Only
  the ``--out`` file is written.
- The stage label of an artifact is derived from its filename suffix and is
  reported as ``stage_hint`` with ``stage_hint_basis``. It is a filing label,
  not proof of an IR dialect, a producer, or a pass boundary.
- ``content_kind`` is a bounded byte sniff (magic bytes / JSON parse / text
  decodability). It is a container classification, not a semantic one.
- The tool reports no operation counts and no dialect detection. A token that
  is absent from a text artifact is not evidence that the corresponding
  operation is absent; that inference has produced wrong conclusions before
  (see ``knowledge/enflame/compiler-artifact-identity.md``).
- A ``differ`` or ``identical`` verdict says nothing about performance. Identical
  bytes do not imply an unchanged timed region, and differing bytes do not
  imply differing machine instructions. When no device image exists on both
  sides, the comparison says so in ``unresolved`` rather than letting an
  ``identical`` count on text stages stand in for an emitted-code result.

Typical use::

    python3 scripts/compile_artifacts.py index \
        --root /path/to/cache-entry --expect ttir,ttgir,gcuir,llir,fatbin

    python3 scripts/compile_artifacts.py compare \
        --left /path/to/baseline --right /path/to/variant \
        --expect ttir,ttgir,gcuir,llir,fatbin --out runs/artifact-diff.json

Fixture directories used in tests are test data, not S60 compilation evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

TOOL_NAME = "compile_artifacts"
TOOL_VERSION = 1

BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
ELF_MAGIC = b"\x7fELF"
SNIFF_BYTES = 4096
MAX_LINE_STATS_BYTES = 32 * 1024 * 1024
CHUNK = 1024 * 1024

# Filename suffix -> stage label. This is a filing convention only.
STAGE_SUFFIXES = {
    ".ttir": "ttir",
    ".ttgir": "ttgir",
    ".gcuir": "gcuir",
    ".llir": "llir",
    ".mlir": "mlir",
    ".ll": "llvm-ir",
    ".bc": "llvm-bitcode",
    ".fatbin": "fatbin",
    ".cubin": "cubin",
    ".ptx": "ptx",
    ".sass": "sass",
    ".json": "metadata",
    ".s": "asm",
    ".txt": "text",
    ".log": "log",
}

LOC_PATTERNS = (
    re.compile(r"loc\((?:[^()]|\([^()]*\))*\)"),
    re.compile(r"#loc\d+"),
)


class ArtifactError(ValueError):
    """Raised for unusable input; reported as a contract-style error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
            total += len(block)
    return digest.hexdigest(), total


def _sniff(path: Path, size: int) -> tuple[str, list[str]]:
    """Return (content_kind, flags) from a bounded read of the file head."""
    flags: list[str] = []
    if size == 0:
        return "empty", flags
    with path.open("rb") as handle:
        head = handle.read(SNIFF_BYTES)
    if head.startswith(BUNDLE_MAGIC):
        return "offload-bundle", flags
    if head.startswith(ELF_MAGIC):
        return "elf", flags
    if head.lstrip()[:1] == b"{":
        try:
            with path.open("rb") as handle:
                json.loads(handle.read().decode("utf-8"))
            return "json", flags
        except (UnicodeDecodeError, json.JSONDecodeError):
            flags.append("json-parse-failed")
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return "binary", flags
    if b"\x00" in head:
        return "binary", flags
    return "text", flags


def _parse_bundle(path: Path, size: int) -> dict[str, Any]:
    """Parse the clang offload-bundle header. Never guesses on failure."""
    result: dict[str, Any] = {"parsed": False, "format_version": None, "targets": []}
    try:
        with path.open("rb") as handle:
            blob = handle.read()
    except OSError as error:  # pragma: no cover - defensive
        result["error"] = f"read failed: {error}"
        return result
    if not blob.startswith(BUNDLE_MAGIC):
        result["error"] = "magic missing"
        return result
    position = len(BUNDLE_MAGIC)
    if len(blob) < position + 8:
        result["error"] = "truncated before entry count"
        return result
    count = int.from_bytes(blob[position : position + 8], "little")
    position += 8
    if count > 64:
        result["error"] = f"implausible entry count {count}"
        return result
    targets: list[dict[str, Any]] = []
    for index in range(count):
        if len(blob) < position + 24:
            result["error"] = f"truncated in entry {index}"
            result["targets"] = targets
            return result
        offset = int.from_bytes(blob[position : position + 8], "little")
        length = int.from_bytes(blob[position + 8 : position + 16], "little")
        triple_size = int.from_bytes(blob[position + 16 : position + 24], "little")
        position += 24
        if len(blob) < position + triple_size:
            result["error"] = f"truncated triple in entry {index}"
            result["targets"] = targets
            return result
        triple = blob[position : position + triple_size].decode("utf-8", "replace")
        position += triple_size
        targets.append(
            {
                "triple": triple,
                "offset": offset,
                "size": length,
                "inside_file": offset + length <= size,
            }
        )
    result.update({"parsed": True, "format_version": 2, "targets": targets})
    return result


def _cache_key(relative: Path) -> str | None:
    """Return a Triton-style cache directory name when the layout matches."""
    for part in relative.parts[:-1]:
        if len(part) >= 16 and re.fullmatch(r"[0-9A-Za-z]{16,}", part):
            return part
    return None


def _stage_hint(path: Path) -> tuple[str | None, str, list[str]]:
    suffix = path.suffix.lower()
    stage = STAGE_SUFFIXES.get(suffix)
    if stage is None:
        return None, "unknown-suffix", ["unsupported-suffix"]
    return stage, "filename-suffix", []


def describe_file(path: Path, root: Path) -> dict[str, Any]:
    """Describe one artifact by identity only."""
    relative = path.relative_to(root)
    record: dict[str, Any] = {
        "path": relative.as_posix(),
        "size": 0,
        "sha256": None,
        "stage_hint": None,
        "stage_hint_basis": "filename-suffix",
        "content_kind": "unknown",
        "cache_key": _cache_key(relative),
        "flags": [],
    }
    try:
        digest, size = _sha256_and_size(path)
    except OSError as error:
        record["flags"].append("unreadable")
        record["error"] = str(error)
        return record
    record["size"] = size
    record["sha256"] = digest
    stage, basis, flags = _stage_hint(path)
    record["stage_hint"] = stage
    record["stage_hint_basis"] = basis
    record["flags"].extend(flags)
    kind, sniff_flags = _sniff(path, size)
    record["content_kind"] = kind
    record["flags"].extend(sniff_flags)
    if kind == "offload-bundle":
        record["bundle"] = _parse_bundle(path, size)
    if stage is not None:
        expected_kind = {
            "ttir": "text",
            "ttgir": "text",
            "gcuir": "text",
            "llir": "text",
            "metadata": "json",
            "fatbin": "offload-bundle",
            "cubin": "elf",
            "ptx": "text",
        }.get(stage)
        if expected_kind is not None and kind != expected_kind:
            # Filing label disagrees with container bytes; report, do not fix.
            record["flags"].append(f"suffix-content-mismatch:{expected_kind}->{kind}")
    return record


def index_root(root: Path, *, label: str | None, expect: Iterable[str]) -> dict[str, Any]:
    if not root.exists():
        raise ArtifactError(f"input path does not exist: {root}")
    if not root.is_dir():
        raise ArtifactError(f"input path is not a directory: {root}")
    artifacts = [
        describe_file(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    present = sorted({item["stage_hint"] for item in artifacts if item["stage_hint"]})
    expected = [stage for stage in expect if stage]
    manifest = {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "generated_at": _utc_now(),
        "source_label": label or root.name,
        "root_kind": "directory",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "stages": {
            "present": present,
            "expected": expected,
            "missing": [stage for stage in expected if stage not in present],
        },
        "unsupported": [
            item["path"] for item in artifacts if "unsupported-suffix" in item["flags"]
        ],
        "notes": [
            "stage_hint is derived from the filename suffix only; it is not proof of IR dialect or producer.",
            "content_kind is a byte-level container sniff; it carries no semantic claim.",
            "Identical hashes do not prove the same code was timed; differing hashes do not prove differing machine instructions.",
        ],
    }
    if not artifacts:
        manifest["notes"].append("empty directory: no artifacts were described")
    return manifest


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"invalid manifest {path}: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("artifacts"), list):
        raise ArtifactError(
            f"manifest {path} is not a compile_artifacts manifest (no artifacts list)"
        )
    return data


def load_side(value: str, expect: Iterable[str]) -> dict[str, Any]:
    path = Path(value)
    if path.is_file() and path.suffix.lower() == ".json":
        manifest = _read_manifest(path)
        manifest["root_kind"] = "manifest"
        return manifest
    return index_root(path, label=None, expect=expect)


def _byte_compare(left: Path, right: Path) -> tuple[int | None, int]:
    """Return (first differing offset, number of differing bytes)."""
    first: int | None = None
    differing = 0
    offset = 0
    with left.open("rb") as lh, right.open("rb") as rh:
        while True:
            lb = lh.read(CHUNK)
            rb = rh.read(CHUNK)
            if lb == rb:
                if not lb:
                    break
                offset += len(lb)
                continue
            limit = min(len(lb), len(rb))
            for index in range(limit):
                if lb[index] != rb[index]:
                    differing += 1
                    if first is None:
                        first = offset + index
            if len(lb) != len(rb):
                differing += abs(len(lb) - len(rb))
                if first is None:
                    first = offset + limit
            offset += max(len(lb), len(rb))
            if not lb or not rb:
                break
    return first, differing


def _strip_locations(line: str) -> str:
    """Collapse debug-location syntax so only structural text remains."""
    line = LOC_PATTERNS[0].sub("loc()", line)
    return LOC_PATTERNS[1].sub("#loc", line)


def _location_stats(left: Path, right: Path, left_size: int, right_size: int) -> dict[str, Any] | None:
    """Count lines that differ only in ``loc(...)``/``#locN`` references.

    This is a byte-level line statistic. It is not a semantic-equivalence
    proof and must not be reported as one.
    """
    if max(left_size, right_size) > MAX_LINE_STATS_BYTES:
        return {"skipped": "size-limit"}
    try:
        left_lines = left.read_text(encoding="utf-8", errors="strict").splitlines()
        right_lines = right.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeDecodeError):
        return {"skipped": "not-utf8-text"}
    differing = 0
    location_only = 0
    for left_line, right_line in zip(left_lines, right_lines):
        if left_line == right_line:
            continue
        differing += 1
        if _strip_locations(left_line) == _strip_locations(right_line):
            location_only += 1
    differing += abs(len(left_lines) - len(right_lines))
    return {
        "lines": [len(left_lines), len(right_lines)],
        "lines_differing": differing,
        "lines_differing_only_in_location_refs": location_only,
        "claim": "byte-level line statistic only; not a semantic-equivalence proof",
    }


def _pair_artifacts(
    left: dict[str, Any], right: dict[str, Any]
) -> list[tuple[str, dict[str, Any] | None, dict[str, Any] | None, str]]:
    """Pair artifacts by relative path, then by unique basename.

    The basename fallback exists because a Triton cache entry is keyed by a
    hash directory: recompiling the same kernel produces the same artifact
    filenames under a different directory. Pairing by path alone would report
    "cache key differs, everything only on one side", which is the case this
    tool exists to disprove or confirm. Ambiguous basenames are never guessed.
    """
    left_by_path = {item["path"]: item for item in left["artifacts"]}
    right_by_path = {item["path"]: item for item in right["artifacts"]}
    ordered: list[tuple[str, dict[str, Any] | None, dict[str, Any] | None, str]] = []
    matched_left: set[str] = set()
    matched_right: set[str] = set()

    for path in sorted(set(left_by_path) & set(right_by_path)):
        ordered.append((path, left_by_path[path], right_by_path[path], "path"))
        matched_left.add(path)
        matched_right.add(path)

    def _unique_by_name(
        items: dict[str, dict[str, Any]], skip: set[str]
    ) -> dict[str, str]:
        counts: dict[str, list[str]] = {}
        for path in items:
            if path in skip:
                continue
            counts.setdefault(Path(path).name, []).append(path)
        return {
            name: paths[0]
            for name, paths in counts.items()
            if len(paths) == 1
        }

    left_names = _unique_by_name(left_by_path, matched_left)
    right_names = _unique_by_name(right_by_path, matched_right)
    for name in sorted(set(left_names) & set(right_names)):
        left_path = left_names[name]
        right_path = right_names[name]
        ordered.append(
            (left_path, left_by_path[left_path], right_by_path[right_path], "basename")
        )
        matched_left.add(left_path)
        matched_right.add(right_path)

    for path in sorted(set(left_by_path) - matched_left):
        ordered.append((path, left_by_path[path], None, "none"))
    for path in sorted(set(right_by_path) - matched_right):
        ordered.append((path, None, right_by_path[path], "none"))
    return ordered


def compare_sides(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_root: Path | None = None,
    right_root: Path | None = None,
    locations: bool = False,
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for path, lhs, rhs, matched_by in _pair_artifacts(left, right):
        pair: dict[str, Any] = {
            "path": path,
            "right_path": None if rhs is None else rhs["path"],
            "matched_by": matched_by,
            "stage_hint": (lhs or rhs or {}).get("stage_hint"),
            "content_kind": (lhs or rhs or {}).get("content_kind"),
            "left": _side_summary(lhs),
            "right": _side_summary(rhs),
            "left_cache_key": (lhs or {}).get("cache_key"),
            "right_cache_key": (rhs or {}).get("cache_key"),
            "status": "only_left" if rhs is None else "only_right" if lhs is None else "identical",
        }
        if lhs and rhs:
            if lhs.get("sha256") != rhs.get("sha256"):
                pair["status"] = "differ"
            pair["size_delta"] = rhs.get("size", 0) - lhs.get("size", 0)
            if pair["status"] == "differ" and left_root and right_root:
                left_file = left_root / path
                right_file = right_root / rhs["path"]
                first, count = _byte_compare(left_file, right_file)
                pair["first_diff_offset"] = first
                pair["differing_bytes"] = count
                if locations and (lhs or {}).get("content_kind") == "text":
                    pair["location_stats"] = _location_stats(
                        left_file,
                        right_file,
                        lhs.get("size", 0),
                        rhs.get("size", 0),
                    )
        pairs.append(pair)

    kinds: dict[str, dict[str, int]] = {}
    for pair in pairs:
        bucket = kinds.setdefault(
            str(pair["content_kind"]), {"identical": 0, "differ": 0, "only_left": 0, "only_right": 0}
        )
        bucket[pair["status"]] = bucket.get(pair["status"], 0) + 1

    cache_key = {"same": 0, "differ": 0, "unknown": 0}
    for pair in pairs:
        left_key = pair["left_cache_key"]
        right_key = pair["right_cache_key"]
        if not left_key or not right_key:
            cache_key["unknown"] += 1
        elif left_key == right_key:
            cache_key["same"] += 1
        else:
            cache_key["differ"] += 1

    left_stages = set(left["stages"]["present"])
    right_stages = set(right["stages"]["present"])
    unresolved: list[str] = []
    if cache_key["differ"] and any(
        pair["status"] == "identical" for pair in pairs
    ):
        unresolved.append(
            "cache keys differ while at least one artifact is byte-identical; "
            "a cache key is not device code (see S60-EVIDENCE-001)"
        )
    if any(pair["status"] == "differ" for pair in pairs):
        unresolved.append(
            "differing bytes do not establish differing machine instructions; "
            "inspect the device image or disassembly before claiming an instruction change"
        )
    for stage in sorted(left_stages - right_stages):
        unresolved.append(f"stage {stage!r} exists only on the left side")
    for stage in sorted(right_stages - left_stages):
        unresolved.append(f"stage {stage!r} exists only on the right side")
    # Text stages can be byte-identical while the emitted machine code differs, so
    # a comparison that never reached a device image must say so instead of
    # letting an "identical" count read as a statement about compiled code.
    device_kinds = {"elf", "offload-bundle"}
    if not any(
        pair["content_kind"] in device_kinds
        and pair["status"] in ("identical", "differ")
        for pair in pairs
    ):
        unresolved.append(
            "no device binary was compared on both sides; identical text stages do "
            "not show that the emitted machine code is identical"
        )

    return {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "generated_at": _utc_now(),
        "left_label": left.get("source_label"),
        "right_label": right.get("source_label"),
        "pairs": pairs,
        "summary": {
            "counts": {
                status: sum(1 for pair in pairs if pair["status"] == status)
                for status in ("identical", "differ", "only_left", "only_right")
            },
            "by_content_kind": kinds,
            "cache_key": cache_key,
            "stages": {
                "left": sorted(left_stages),
                "right": sorted(right_stages),
                "only_in_left": sorted(left_stages - right_stages),
                "only_in_right": sorted(right_stages - left_stages),
            },
        },
        "unresolved": unresolved,
        "notes": [
            "Descriptive identity comparison only; no performance or causality claim.",
            "Inputs were read, never written. Normalized or rewritten artifacts are not produced here.",
        ],
    }


def _side_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    summary = {
        "size": record.get("size"),
        "sha256": record.get("sha256"),
        "content_kind": record.get("content_kind"),
        "flags": record.get("flags", []),
    }
    bundle = record.get("bundle")
    if isinstance(bundle, dict):
        summary["bundle"] = {
            "parsed": bundle.get("parsed"),
            "targets": [target.get("triple") for target in bundle.get("targets", [])],
        }
    return summary


def summarize_index(manifest: dict[str, Any]) -> str:
    lines = [
        f"{TOOL_NAME}: {manifest['source_label']} ({manifest['artifact_count']} files)",
        f"  stages present  : {', '.join(manifest['stages']['present']) or 'none'}",
    ]
    if manifest["stages"]["expected"]:
        lines.append(
            f"  stages missing  : {', '.join(manifest['stages']['missing']) or 'none'}"
        )
    unsupported = manifest.get("unsupported") or []
    if unsupported:
        lines.append(f"  unsupported     : {len(unsupported)} file(s), see manifest")
    mismatch = [
        item["path"]
        for item in manifest["artifacts"]
        if any(flag.startswith("suffix-content-mismatch") for flag in item["flags"])
    ]
    if mismatch:
        lines.append(f"  label conflicts : {len(mismatch)} file(s), see manifest")
    return "\n".join(lines) + "\n"


def summarize_compare(diff: dict[str, Any]) -> str:
    counts = diff["summary"]["counts"]
    cache_key = diff["summary"]["cache_key"]
    lines = [
        f"{TOOL_NAME}: {diff['left_label']} vs {diff['right_label']}",
        f"  identical={counts['identical']} differ={counts['differ']} "
        f"only_left={counts['only_left']} only_right={counts['only_right']}",
        f"  cache key: same={cache_key['same']} differ={cache_key['differ']} "
        f"unknown={cache_key['unknown']}",
    ]
    for stage in diff["summary"]["stages"]["only_in_left"]:
        lines.append(f"  missing on right: {stage}")
    for stage in diff["summary"]["stages"]["only_in_right"]:
        lines.append(f"  missing on left : {stage}")
    if diff["unresolved"]:
        lines.append("  unresolved:")
        for item in diff["unresolved"]:
            lines.append(f"    - {item}")
    return "\n".join(lines) + "\n"


def _write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _expect_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description=(
            "Describe and compare compiler artifacts by identity only. "
            "No device, no framework import, no writes to inputs."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--root", required=True)
    index_parser.add_argument("--out")
    index_parser.add_argument("--label")
    index_parser.add_argument("--expect", help="comma-separated stage labels")
    index_parser.add_argument("--json", action="store_true", help="print the manifest")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.add_argument("--out")
    compare_parser.add_argument("--expect", help="comma-separated stage labels")
    compare_parser.add_argument("--left-label")
    compare_parser.add_argument("--right-label")
    compare_parser.add_argument(
        "--locations",
        action="store_true",
        help="count text lines differing only in loc()/#locN references",
    )
    compare_parser.add_argument("--json", action="store_true", help="print the diff")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expect = _expect_list(getattr(args, "expect", None))
    try:
        if args.command == "index":
            manifest = index_root(Path(args.root), label=args.label, expect=expect)
            _write_json(Path(args.out) if args.out else None, manifest)
            print(json.dumps(manifest, indent=2) if args.json else summarize_index(manifest), end="")
            return 0

        left_root = Path(args.left)
        right_root = Path(args.right)
        left = load_side(args.left, expect)
        right = load_side(args.right, expect)
        if args.left_label:
            left["source_label"] = args.left_label
        if args.right_label:
            right["source_label"] = args.right_label
        diff = compare_sides(
            left,
            right,
            left_root=None if left["root_kind"] == "manifest" else left_root,
            right_root=None if right["root_kind"] == "manifest" else right_root,
            locations=args.locations,
        )
        _write_json(Path(args.out) if args.out else None, diff)
        print(json.dumps(diff, indent=2) if args.json else summarize_compare(diff), end="")
        return 0
    except ArtifactError as error:
        print(f"{TOOL_NAME}: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

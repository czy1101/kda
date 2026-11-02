"""Conformance tests for the committed knowledge library.

These guard the library itself, not the recorder: every recorded result must
still match the schema, every guide and rejection must carry metadata, and every
INDEX.md must be exactly what the generator would produce.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads(
    (REPO_ROOT / "schemas/optimization-knowledge.schema.json").read_text(encoding="utf-8")
)
NOTE_ROOTS = sorted((REPO_ROOT / "backends").glob("*/note"))


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


knowledge_index = _load("knowledge_index", "scripts/knowledge_index.py")
record_rejection = _load("record_rejection", "scripts/record_rejection.py")


def _result_records(root: Path) -> list[Path]:
    return sorted(root.rglob("*.json"))


def _metadata_notes(root: Path) -> list[Path]:
    notes = []
    for path in sorted(root.rglob("*.md")):
        if path.name == "INDEX.md" or path.with_suffix(".json").is_file():
            continue
        notes.append(path)
    return notes


def test_note_roots_exist():
    assert NOTE_ROOTS, "no vendor note root found under backends/*/note"


@pytest.mark.parametrize("root", NOTE_ROOTS, ids=lambda path: path.parent.name)
def test_result_records_match_schema(root):
    records = _result_records(root)
    assert records, f"{root} has no recorded results"
    validator = Draft202012Validator(SCHEMA)
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        assert not errors, f"{path}: " + "; ".join(error.message for error in errors)
        assert path.with_suffix(".md").is_file(), f"{path}: missing Markdown report"


@pytest.mark.parametrize("root", NOTE_ROOTS, ids=lambda path: path.parent.name)
def test_metadata_notes_carry_front_matter(root):
    for path in _metadata_notes(root):
        front = knowledge_index.parse_front_matter(path)
        assert front is not None, f"{path}: missing YAML front matter"
        for field in knowledge_index.REQUIRED_FRONT_MATTER:
            assert front.get(field), f"{path}: front matter is missing {field}"
        assert front["kind"] in knowledge_index.KINDS, f"{path}: bad kind"
        assert front["status"] in knowledge_index.STATUSES, f"{path}: bad status"
        if front["kind"] == "rejection":
            assert "rejected" in path.parts, f"{path}: rejection outside rejected/"


@pytest.mark.parametrize("root", NOTE_ROOTS, ids=lambda path: path.parent.name)
def test_index_is_fresh(root):
    index = root / "INDEX.md"
    assert index.is_file(), f"{index}: missing; run scripts/knowledge_index.py"
    assert index.read_text(encoding="utf-8") == knowledge_index.render(root), (
        f"{index}: stale; run scripts/knowledge_index.py"
    )


def test_index_lists_every_note():
    for root in NOTE_ROOTS:
        text = (root / "INDEX.md").read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.md")):
            if path.name == "INDEX.md":
                continue
            relative = path.relative_to(root).as_posix()
            assert f"]({relative})" in text, f"{path}: not linked from INDEX.md"


def test_rejected_approaches_are_indexed():
    collected = knowledge_index.collect(NOTE_ROOTS[0])
    assert collected["rejections"], "expected at least one documented rejection"
    index = (NOTE_ROOTS[0] / "INDEX.md").read_text(encoding="utf-8")
    assert "## Rejected approaches" in index
    for entry in collected["rejections"]:
        assert entry["retry_when"], f"{entry['path']}: rejection must say when to retry"
        assert f"]({entry['path']})" in index


def test_rejection_recorder_writes_an_indexed_note(tmp_path):
    root = tmp_path / "note"
    root.mkdir()
    front, markdown = record_rejection.render_rejection(
        title="Split-K beyond one wave is slower",
        device="BW",
        arch="gfx936",
        operator="demo.operator",
        scope="Split-K with four segments on a small reduction",
        hypothesis="More parallelism shortens the tail.",
        observed="1.2x slower than one segment.",
        reason="Combine overhead exceeds the saved tail.",
        retry_when="The reduction has at least four waves of work.",
        evidence=["runs/benchmark.csv"],
        rejected_on="2026-11-02",
    )
    path = record_rejection.write_rejection(root, front["id"], markdown)
    assert path.is_file()
    assert path.parent.name == "rejected"
    index = (root / "INDEX.md").read_text(encoding="utf-8")
    assert "rejected/split-k-beyond-one-wave-is-slower.md" in index
    with pytest.raises(record_rejection.RejectionError):
        record_rejection.write_rejection(root, front["id"], markdown)


def test_generated_index_sections_are_ordered():
    root = NOTE_ROOTS[0]
    text = (root / "INDEX.md").read_text(encoding="utf-8")
    sections = [
        "## Confirmed operator results",
        "## Optimization guides",
        "## Rejected approaches",
    ]
    positions = [text.index(section) for section in sections]
    assert positions == sorted(positions)

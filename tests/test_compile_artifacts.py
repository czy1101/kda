"""Tests for the offline compiler-artifact identity tool.

Fixtures here are synthetic. They exist to exercise parsing, pairing and
comparison logic. They are not S60 compilation evidence, and a passing test says
nothing about a real toolchain.
"""

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compile_artifacts.py"
SPEC = importlib.util.spec_from_file_location("compile_artifacts", SCRIPT)
compile_artifacts = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compile_artifacts)

TTIR = """module {
  tt.func public @k(%a: !tt.ptr<f32>) loc(#loc1) {
    tt.return loc(#loc1)
  }
}
#loc1 = loc("k.py":10:5)
"""

TTIR_SHIFTED = """module {
  tt.func public @k(%a: !tt.ptr<f32>) loc(#loc2) {
    tt.return loc(#loc2)
  }
}
#loc2 = loc("k.py":12:10)
"""

TTIR_DIFFERENT = """module {
  tt.func public @k(%a: !tt.ptr<f32>) loc(#loc1) {
    %0 = tt.dot %a, %a loc(#loc1)
    tt.return loc(#loc1)
  }
}
#loc1 = loc("k.py":10:5)
"""


def write_bundle(path, targets):
    """Write a minimal clang offload bundle with the given triples."""
    header = bytearray(compile_artifacts.BUNDLE_MAGIC)
    payload = b"D" * (8 * len(targets))
    entries = []
    offset = 0
    for triple in targets:
        encoded = triple.encode()
        entries.append((offset, 8, len(encoded), encoded))
        offset += 8
    body = bytearray(struct.pack("<Q", len(targets)))
    for offset_value, size, triple_size, encoded in entries:
        body += struct.pack("<QQQ", offset_value, size, triple_size)
        body += encoded
    path.write_bytes(bytes(header) + bytes(body) + payload)


def make_capture(root, cache_name, files):
    entry = root / cache_name
    entry.mkdir(parents=True)
    for name, content in files.items():
        target = entry / name
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    return entry


class CompileArtifactTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _index(self, path, expect=("ttir", "gcuir", "llir", "fatbin")):
        return compile_artifacts.index_root(path, label=None, expect=expect)

    def test_identical_captures_are_matched_across_cache_directories(self):
        left = self.root / "left"
        right = self.root / "right"
        make_capture(left, "1" * 16, {"k.ttir": TTIR, "k.fatbin": b"same-bytes"})
        make_capture(right, "2" * 16, {"k.ttir": TTIR, "k.fatbin": b"same-bytes"})
        diff = compile_artifacts.compare_sides(
            self._index(left), self._index(right), left_root=left, right_root=right
        )
        self.assertEqual(diff["summary"]["counts"]["identical"], 2)
        self.assertEqual(diff["summary"]["counts"]["only_left"], 0)
        self.assertEqual(diff["summary"]["counts"]["only_right"], 0)
        self.assertTrue(all(pair["matched_by"] == "basename" for pair in diff["pairs"]))

    def test_differing_bytes_report_first_offset(self):
        left = self.root / "left"
        right = self.root / "right"
        make_capture(left, "1" * 16, {"k.gcuir": TTIR})
        make_capture(right, "2" * 16, {"k.gcuir": TTIR_DIFFERENT})
        diff = compile_artifacts.compare_sides(
            self._index(left), self._index(right), left_root=left, right_root=right
        )
        pair = diff["pairs"][0]
        self.assertEqual(pair["status"], "differ")
        self.assertIsNotNone(pair["first_diff_offset"])
        self.assertGreater(pair["differing_bytes"], 0)

    def test_two_specializations_of_one_kernel_are_not_paired(self):
        # One side holds two specializations of the same kernel, the other
        # holds one. There is no fact in the manifests that says which one
        # corresponds, so the tool must refuse rather than pick.
        left = self.root / "left"
        right = self.root / "right"
        make_capture(left, "a" * 16, {"gla.ttir": TTIR})
        make_capture(left, "b" * 16, {"gla.ttir": TTIR_SHIFTED})
        make_capture(right, "c" * 16, {"gla.ttir": TTIR})
        diff = compile_artifacts.compare_sides(
            self._index(left), self._index(right), left_root=left, right_root=right
        )
        self.assertEqual(diff["summary"]["counts"]["only_left"], 2)
        self.assertEqual(diff["summary"]["counts"]["only_right"], 1)
        self.assertEqual(diff["summary"]["counts"]["identical"], 0)
        self.assertEqual(diff["summary"]["match_quality"]["name-only"], 0)
        self.assertEqual(
            diff["summary"]["ambiguous_names"],
            [{"name": "gla.ttir", "left": 2, "right": 1}],
        )
        self.assertTrue(
            any("could not be paired" in note for note in diff["unresolved"]),
            diff["unresolved"],
        )

    def test_single_artifact_each_side_is_labelled_name_only(self):
        # The one-to-one case still pairs, but the record says the match rests
        # on a filename and warns that the two files may be different
        # specializations. Pairing is not silently presented as a hit.
        left = self.root / "left"
        right = self.root / "right"
        make_capture(left, "a" * 16, {"gla.ttir": TTIR})
        make_capture(right, "c" * 16, {"gla.ttir": TTIR})
        diff = compile_artifacts.compare_sides(
            self._index(left), self._index(right), left_root=left, right_root=right
        )
        self.assertEqual(diff["pairs"][0]["match_quality"], "name-only")
        self.assertEqual(diff["summary"]["match_quality"]["name-only"], 1)
        self.assertEqual(diff["summary"]["ambiguous_names"], [])
        self.assertTrue(
            any("matched by filename only" in note for note in diff["unresolved"]),
            diff["unresolved"],
        )

    def test_same_relative_path_is_labelled_exact_and_raises_no_name_warning(self):
        left = self.root / "left"
        right = self.root / "right"
        (left / "entry").mkdir(parents=True)
        (right / "entry").mkdir(parents=True)
        (left / "entry" / "gla.ttir").write_text(TTIR, encoding="utf-8")
        (right / "entry" / "gla.ttir").write_text(TTIR, encoding="utf-8")
        diff = compile_artifacts.compare_sides(
            self._index(left), self._index(right), left_root=left, right_root=right
        )
        self.assertEqual(diff["pairs"][0]["match_quality"], "exact-path")
        self.assertFalse(
            any("matched by filename only" in note for note in diff["unresolved"]),
            diff["unresolved"],
        )

    def test_cache_key_difference_does_not_hide_identical_device_binary(self):
        left = self.root / "left"
        right = self.root / "right"
        device = self.root / "device.o"
        write_bundle(device, ["tops-dtu-enflame-tops--gcu300"])
        binary = device.read_bytes()
        make_capture(left, "a" * 16, {"k.fatbin": binary})
        make_capture(right, "b" * 16, {"k.fatbin": binary})
        diff = compile_artifacts.compare_sides(
            self._index(left), self._index(right), left_root=left, right_root=right
        )
        self.assertEqual(diff["pairs"][0]["status"], "identical")
        self.assertEqual(diff["summary"]["cache_key"]["differ"], 1)
        self.assertTrue(
            any("cache key" in note for note in diff["unresolved"]),
            diff["unresolved"],
        )

    def test_missing_stage_is_reported_on_both_sides(self):
        left = self.root / "left"
        right = self.root / "right"
        make_capture(left, "1" * 16, {"k.ttir": TTIR, "k.gcuir": TTIR})
        make_capture(right, "2" * 16, {"k.ttir": TTIR})
        manifest = self._index(left)
        self.assertEqual(manifest["stages"]["missing"], ["llir", "fatbin"])
        diff = compile_artifacts.compare_sides(manifest, self._index(right))
        self.assertIn("gcuir", diff["summary"]["stages"]["only_in_left"])
        self.assertTrue(any("gcuir" in note for note in diff["unresolved"]))

    def test_empty_directory_is_described_not_crashed(self):
        empty = self.root / "empty"
        empty.mkdir()
        manifest = self._index(empty)
        self.assertEqual(manifest["artifact_count"], 0)
        self.assertEqual(manifest["artifacts"], [])
        self.assertTrue(any("empty" in note for note in manifest["notes"]))

    def test_invalid_inputs_raise_a_contract_style_error(self):
        with self.assertRaises(compile_artifacts.ArtifactError):
            self._index(self.root / "does-not-exist")
        file_path = self.root / "a-file"
        file_path.write_text("x", encoding="utf-8")
        with self.assertRaises(compile_artifacts.ArtifactError):
            self._index(file_path)
        bad_manifest = self.root / "bad.json"
        bad_manifest.write_text('{"not": "a manifest"}', encoding="utf-8")
        with self.assertRaises(compile_artifacts.ArtifactError):
            compile_artifacts.load_side(str(bad_manifest), ())

    def test_cli_returns_two_on_invalid_input(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "index", "--root", str(self.root / "nope")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("error", result.stderr)

    def test_input_files_are_not_modified(self):
        left = self.root / "left"
        right = self.root / "right"
        make_capture(left, "1" * 16, {"k.ttir": TTIR, "k.fatbin": b"abc"})
        make_capture(right, "2" * 16, {"k.ttir": TTIR_DIFFERENT, "k.fatbin": b"abd"})

        def snapshot():
            entries = {}
            for path in sorted(self.root.rglob("*")):
                if path.is_file():
                    entries[path.as_posix()] = (
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                        path.stat().st_mtime_ns,
                        path.stat().st_size,
                    )
            return entries

        before = snapshot()
        diff = compile_artifacts.compare_sides(
            self._index(left),
            self._index(right),
            left_root=left,
            right_root=right,
            locations=True,
        )
        compile_artifacts.summarize_compare(diff)
        self.assertEqual(before, snapshot())

    def test_unsupported_format_is_reported(self):
        capture = self.root / "capture"
        make_capture(capture, "1" * 16, {"k.ttir": TTIR, "notes.weird": "text"})
        manifest = self._index(capture)
        self.assertEqual(manifest["unsupported"], ["1111111111111111/notes.weird"])

    def test_suffix_conflict_is_flagged_not_silently_trusted(self):
        capture = self.root / "capture"
        bundle = self.root / "device.o"
        write_bundle(bundle, ["tops-dtu-enflame-tops--gcu300"])
        make_capture(capture, "1" * 16, {"k.ttir": bundle.read_bytes()})
        manifest = self._index(capture)
        record = manifest["artifacts"][0]
        self.assertEqual(record["stage_hint"], "ttir")
        self.assertEqual(record["content_kind"], "offload-bundle")
        self.assertIn("suffix-content-mismatch:text->offload-bundle", record["flags"])

    def test_bundle_targets_are_parsed(self):
        capture = self.root / "capture"
        bundle = self.root / "device.o"
        write_bundle(
            bundle,
            ["host-x86_64-unknown-linux", "tops-dtu-enflame-tops--gcu300"],
        )
        make_capture(capture, "1" * 16, {"k.fatbin": bundle.read_bytes()})
        manifest = self._index(capture)
        parsed = manifest["artifacts"][0]["bundle"]
        self.assertTrue(parsed["parsed"])
        self.assertEqual(
            parsed["targets"][1]["triple"], "tops-dtu-enflame-tops--gcu300"
        )
        self.assertTrue(all(item["inside_file"] for item in parsed["targets"]))

    def test_truncated_bundle_reports_an_error_instead_of_guessing(self):
        capture = self.root / "capture"
        truncated = compile_artifacts.BUNDLE_MAGIC + struct.pack("<Q", 3)
        make_capture(capture, "1" * 16, {"k.fatbin": truncated})
        manifest = self._index(capture)
        parsed = manifest["artifacts"][0]["bundle"]
        self.assertFalse(parsed["parsed"])
        self.assertIn("truncated", parsed["error"])

    def test_bundle_larger_than_the_parse_limit_is_reported_unparsed(self):
        # Support range, not a guess: the parser states the limit it works
        # within instead of reading an unbounded file into memory.
        capture = self.root / "capture"
        bundle = self.root / "device.o"
        write_bundle(bundle, ["tops-dtu-enflame-tops--gcu300"])
        make_capture(capture, "1" * 16, {"k.fatbin": bundle.read_bytes()})
        original = compile_artifacts.MAX_PARSE_BYTES
        compile_artifacts.MAX_PARSE_BYTES = 8
        try:
            manifest = self._index(capture)
        finally:
            compile_artifacts.MAX_PARSE_BYTES = original
        parsed = manifest["artifacts"][0]["bundle"]
        self.assertFalse(parsed["parsed"])
        self.assertIn("parse limit", parsed["error"])

    def test_parsed_bundle_names_the_layout_not_a_version(self):
        capture = self.root / "capture"
        bundle = self.root / "device.o"
        write_bundle(bundle, ["tops-dtu-enflame-tops--gcu300"])
        make_capture(capture, "1" * 16, {"k.fatbin": bundle.read_bytes()})
        parsed = self._index(capture)["artifacts"][0]["bundle"]
        self.assertEqual(parsed["format_version"], "counted-entries")

    def test_location_statistic_counts_only_debug_reference_differences(self):
        left = self.root / "left"
        right = self.root / "right"
        make_capture(left, "1" * 16, {"k.ttir": TTIR})
        make_capture(right, "2" * 16, {"k.ttir": TTIR_SHIFTED})
        diff = compile_artifacts.compare_sides(
            self._index(left), self._index(right), left_root=left, right_root=right,
            locations=True,
        )
        stats = diff["pairs"][0]["location_stats"]
        self.assertEqual(stats["lines_differing"], 3)
        self.assertEqual(stats["lines_differing_only_in_location_refs"], 3)
        self.assertIn("not a semantic-equivalence proof", stats["claim"])

    def test_manifest_round_trip_matches_directory_comparison(self):
        left = self.root / "left"
        right = self.root / "right"
        make_capture(left, "1" * 16, {"k.ttir": TTIR})
        make_capture(right, "2" * 16, {"k.ttir": TTIR})
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(
            json.dumps(self._index(left)), encoding="utf-8"
        )
        from_manifest = compile_artifacts.load_side(str(manifest_path), ())
        self.assertEqual(from_manifest["root_kind"], "manifest")
        diff = compile_artifacts.compare_sides(from_manifest, self._index(right))
        self.assertEqual(diff["summary"]["counts"]["identical"], 1)
        # No byte comparison is possible without the files themselves.
        self.assertNotIn("first_diff_offset", diff["pairs"][0])

    def test_summary_is_short_and_states_the_counts(self):
        left = self.root / "left"
        right = self.root / "right"
        make_capture(left, "1" * 16, {"k.ttir": TTIR})
        make_capture(right, "2" * 16, {"k.ttir": TTIR_DIFFERENT})
        diff = compile_artifacts.compare_sides(
            self._index(left), self._index(right), left_root=left, right_root=right
        )
        summary = compile_artifacts.summarize_compare(diff)
        self.assertIn("identical=0 differ=1", summary)
        self.assertLessEqual(len(summary.splitlines()), 12)

    def test_absent_device_binary_refuses_the_machine_code_reading(self):
        # Degraded but honest: when the dump was not enabled for this stage, or
        # the capture stopped before linking, the comparison still runs on text
        # stages and states that it never looked at emitted code.
        left = self.root / "left"
        right = self.root / "right"
        make_capture(left, "1" * 16, {"k.ttir": TTIR, "k.gcuir": TTIR})
        make_capture(right, "2" * 16, {"k.ttir": TTIR, "k.gcuir": TTIR})
        diff = compile_artifacts.compare_sides(
            self._index(left), self._index(right), left_root=left, right_root=right
        )
        self.assertEqual(diff["summary"]["counts"]["identical"], 2)
        self.assertTrue(
            any("no device binary" in note for note in diff["unresolved"]),
            diff["unresolved"],
        )

    def test_absent_device_binary_note_disappears_once_a_fatbin_is_compared(self):
        left = self.root / "left"
        right = self.root / "right"
        bundle = self.root / "device.o"
        write_bundle(bundle, ["tops-dtu-enflame-tops--gcu300"])
        payload = bundle.read_bytes()
        make_capture(left, "1" * 16, {"k.ttir": TTIR, "k.fatbin": payload})
        make_capture(right, "2" * 16, {"k.ttir": TTIR, "k.fatbin": payload})
        diff = compile_artifacts.compare_sides(
            self._index(left), self._index(right), left_root=left, right_root=right
        )
        self.assertEqual(diff["summary"]["by_content_kind"]["offload-bundle"]["identical"], 1)
        self.assertFalse(
            any("no device binary" in note for note in diff["unresolved"]),
            diff["unresolved"],
        )


if __name__ == "__main__":
    unittest.main()

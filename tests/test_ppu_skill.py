"""Software fixtures only: none of these timings are PPU measurements."""
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills/ppu-kernel-optimizer/scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "scripts"))
import kernelpilot as kp
from common import summarize, write_json
from benchmark_kernel import measure
from collect_compiler_artifacts import collect
from analyze_compiler_artifacts import analyze, source_analysis
from compare_experiments import compare
from generate_report import report
from probe_capabilities import validate_export, verify


def task():
    return {"version": 1, "task": {"name": "ppu-fixture", "description": "software fixture"},
            "target": {"backend": "ppu", "arch": "ppu001", "device": "PPU-ZW810E", "language": "triton"},
            "implementation": {"path": "kernel.py"}, "correctness": {"command": "check"},
            "benchmark": {"command": "bench", "shapes": ["fixture"]},
            "baseline": {"name": "original"},
            "goal": {"metric": "latency", "direction": "minimize", "aggregation": "sum_ratio", "relative_to_baseline": .9},
            "constraints": {"allowed_paths": ["kernel.py"], "forbidden_paths": ["test.py"]},
            "profile": {"tools": {"acu": {"command": "capture-acu", "export_command": "export-acu"},
                                  "asys": {"command": "capture-asys", "export_command": "export-asys"}}}}


def discovery(w, t):
    result = {"version": 1, "generated_at": kp._utc_now(), "backend_profile": "thead-zw810e",
              "transport": "local", "environment_fingerprint": kp._analysis_environment_fingerprint(w, t),
              "policy": "auto", "status": "complete", "selected": "acu",
              "tools": [{"id": name, "available": True, "kind": kind, "priority": 1,
                         "probe_exit_code": 0, "capabilities": []}
                        for name, kind in (("acu", "kernel-profiler"), ("asys", "timeline-profiler"))]}
    write_json(w / kp.ANALYSIS_CAPABILITIES, result)
    return result


class FixtureAdapter:
    def __init__(self):
        self.syncs = 0
        self.launches = 0

    def prepare(self):
        self.ready = True

    def check(self):
        assert self.ready

    def launch(self):
        self.launches += 1

    def synchronize(self):
        self.syncs += 1

    def events(self):
        event = lambda: SimpleNamespace(record=lambda: None, synchronize=lambda: None,
                                         elapsed_time=lambda other: .01)
        return event(), event()

    def describe(self):
        return {"shape": [64], "stride": [1], "dtype": "fixture", "device": "fixture",
                "environment_fingerprint": "fixture", "config": {"block": 64}, "config_fixed": True,
                "scope": "fixture", "cache_protocol": "fixture", "execution_kind": "fixture"}


class PPUSkillTests(unittest.TestCase):
    def test_profile_routes_and_schema(self):
        t = task()
        self.assertEqual(kp._backend_profile(t)["id"], "thead-zw810e")
        self.assertEqual(kp.validate_backend_profile(t), [])
        self.assertIn("skills/ppu-kernel-optimizer", kp._backend_profile(t)["skills"]["always"])
        self.assertEqual(kp._backend_profile({"target": {"backend": "corex", "arch": "ivcore11", "device": "BI-V150 OAM"}})["id"], "iluvatar-ivcore11")
        self.assertEqual(kp._backend_profile({"target": {"backend": "cuda", "arch": "sm90", "device": "H100"}})["id"], "cuda-sm90")
        import jsonschema
        schema = yaml.safe_load((ROOT / "schemas/backend.schema.yaml").read_text(encoding="utf-8"))
        jsonschema.validate(kp._backend_profile(t), schema)

    def test_capture_failure_downgrades_and_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            w, t = Path(tmp), task()
            (w / "kernel.py").write_text("pass")
            discovery(w, t)
            bad = subprocess.CompletedProcess([], 101, "", "HGGC PCM API fail, hggc_result: 600 (HGGC_ERROR_NOT_READY)")
            with patch.object(kp.subprocess, "run", return_value=bad) as run:
                result = verify(w, t, "acu", "json", "target", "name", "duration_us")
                self.assertEqual(result["status"], "failed")
                self.assertEqual(run.call_count, 1)
                verify(w, t, "acu", "json", "target", "name", "duration_us")
                self.assertEqual(run.call_count, 1)
                with self.assertRaises(kp.ContractError):
                    kp.run_contract_command(w, t, "profile", tool="acu")
                self.assertEqual(run.call_count, 1)
                write_json(w / kp.ANALYSIS_STRATEGY, {
                    "environment_fingerprint": kp._analysis_environment_fingerprint(w, t),
                    "selected_profiler": "acu"})
                with self.assertRaises(kp.ContractError):
                    kp.run_contract_command(w, t, "profile")
                self.assertEqual(run.call_count, 1)
            strategy = kp.select_analysis_strategy(t, kp.load_analysis_capabilities(w, t))
            self.assertEqual(strategy["selected_profiler"], "asys")
            self.assertEqual(strategy["analysis_level"], 3)  # asys untested
            kp.record_analysis_function(w, t, "asys", "verified", {"fixture": True})
            self.assertEqual(kp.select_analysis_strategy(t, kp.load_analysis_capabilities(w, t))["analysis_level"], 2)
            kp.record_analysis_function(w, t, "asys", "failed", {"reason": "fixture denied"})
            self.assertEqual(kp.select_analysis_strategy(t, kp.load_analysis_capabilities(w, t))["route"], "benchmark_only")
            t["analysis"] = {"profiling": "required"}
            with self.assertRaises(kp.ContractError):
                kp.select_analysis_strategy(t, kp.load_analysis_capabilities(w, t))

    def test_real_fake_cli_error_and_empty_report(self):
        # A real child process emits failure; no hardware/PCM API is touched.
        result = subprocess.run([sys.executable, "-c", "import sys; print('HGGC_ERROR_NOT_READY (600)',file=sys.stderr); sys.exit(101)"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 101)
        for text in ("No kernel in report.", '{"kernels":[]}', '{"kernels":[{"name":"other","duration_us":1}]}'):
            with self.assertRaises(ValueError):
                validate_export(text, "json", "target", "name", "duration_us")
        rows = validate_export('{"kernels":[{"name":"target","duration_us":1}]}', "json", "target", "name", "duration_us")
        self.assertEqual(len(rows), 1)
        rows = validate_export('Loading report\nRow#,Name,Med (ns),\n1,vector,2160,\n', "csv", "^vector$", "Name", "Med (ns)")
        self.assertEqual(rows[0]["Med (ns)"], "2160")

    def test_functional_export_success_and_empty_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            w, t = Path(tmp), task()
            (w / "kernel.py").write_text("pass")
            discovery(w, t)
            results = [subprocess.CompletedProcess([], 0, "captured", ""),
                       subprocess.CompletedProcess([], 0, '{"kernels":[{"name":"target","duration_us":1}]}', "")]
            with patch.object(kp.subprocess, "run", side_effect=results):
                self.assertEqual(verify(w, t, "asys", "json", "target", "name", "duration_us")["status"], "verified")
            results[-1] = subprocess.CompletedProcess([], 0, '{"kernels":[]}', "")
            with patch.object(kp.subprocess, "run", side_effect=results):
                self.assertEqual(verify(w, t, "asys", "json", "target", "name", "duration_us")["status"], "failed")

    def test_disabled_cannot_run_and_fingerprint_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            w, t = Path(tmp), task()
            discovery(w, t)
            kp.record_analysis_function(w, t, "acu", "failed", {"fixture": True})
            with patch.object(kp, "_run_environment_probe", return_value=subprocess.CompletedProcess([], 0, "fixture version", "")):
                kp.discover_analysis_tools(w, t)
            self.assertEqual(kp.load_analysis_capabilities(w, t)["tools"][0]["functional_status"], "failed")
            t["analysis"] = {"profiling": "disabled"}
            with self.assertRaises(kp.ContractError), patch.object(kp.subprocess, "run") as run:
                kp.run_contract_command(w, t, "profile", tool="acu")
            run.assert_not_called()
            t["environment"] = {"setup": "export UPDATED=1"}
            discovery(w, t)
            self.assertNotIn("functional_status", kp.load_analysis_capabilities(w, t)["tools"][0])

    def test_probe_missing_shell_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(kp, "_run_environment_probe", side_effect=FileNotFoundError("missing shell")):
                result = kp.discover_analysis_tools(Path(tmp), task())
                self.assertEqual(result["status"], "no_tool_available")
                self.assertEqual(len(result["tools"]), 3)

    def test_timing_statistics_protocol_and_failure(self):
        adapter = FixtureAdapter()
        result = measure(adapter, warmup=2, repeat=5, timer="event", batch=2)
        self.assertEqual(result["statistics"]["median_us"], 5)
        self.assertEqual(result["samples_us"], [5] * 5)
        self.assertEqual(adapter.launches, 12)
        self.assertGreater(adapter.syncs, 5)
        host = FixtureAdapter()
        measure(host, warmup=0, repeat=3, timer="host")
        self.assertEqual(host.syncs, 8)
        candidate = copy.deepcopy(result)
        self.assertFalse(compare(result, candidate)["separated_p10_p90"])
        candidate["protocol"]["dtype"] = "different"
        with self.assertRaises(ValueError):
            compare(result, candidate)
        candidate = copy.deepcopy(result)
        candidate["config_changed"] = True
        with self.assertRaises(ValueError):
            compare(result, candidate)
        bad = FixtureAdapter()
        bad.check = lambda: (_ for _ in ()).throw(ValueError("incorrect"))
        with self.assertRaises(ValueError):
            measure(bad)
        self.assertEqual(bad.launches, 0)
        with self.assertRaises(ValueError):
            summarize([0])

    def test_compiler_archive_integrity_and_no_device_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp)
            provenance = {key: "fixture" for key in ("source_sha256", "signature", "shape", "dtype", "stride", "grid", "config", "target", "versions", "candidate")}
            kernel = SimpleNamespace(asm={"ttir": "tt.load\nconvert_layout", "hgbin": b"\x00\xff"}, metadata={"shared": 0})
            manifest = collect(kernel, w / "compiler", provenance)
            self.assertIsNone(manifest["resources"]["registers"])
            self.assertEqual(manifest["resources"]["shared_bytes"], 0)
            self.assertIn("llir", manifest["missing"])
            analyzed = analyze(w / "compiler/manifest.json")
            self.assertEqual(analyzed["static_features"]["ttir"]["layout_conversion"], 1)
            (w / "kernel.py").write_text("def kernel(x):\n    return x + 1\n")
            source = source_analysis(w / "kernel.py")
            self.assertIsNone(source["latency_us"])
            text = report(w, {"object": source})
            self.assertIn("未测", text)
            artifact = w / "compiler" / manifest["artifacts"]["ttir"]["path"]
            artifact.write_text("tampered")
            with self.assertRaises(ValueError):
                analyze(w / "compiler/manifest.json")

    def test_closed_loop_rejected_candidate_restores_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            w, t = Path(tmp), task()
            (w / "kernel.py").write_text("correct baseline")
            kp.save_checkpoint(w, t, "baseline", "baseline")
            kp.save_checkpoint(w, t, "best", "baseline")
            hypothesis = {"grade": "D", "claim": "fixture candidate", "refutation": "correctness failure"}
            (w / "kernel.py").write_text("incorrect candidate")
            with patch.object(kp.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "fixture correctness failed")):
                self.assertEqual(kp.run_contract_command(w, t, "correctness", capture=True).returncode, 1)
            restored = kp.restore_checkpoint(w, t)
            self.assertEqual(restored["candidate"], "baseline")
            self.assertEqual((w / "kernel.py").read_text(), "correct baseline")
            (w / "runs").mkdir(exist_ok=True)
            record = {"id": "bad", "parent": "baseline", "hypothesis": hypothesis["claim"], "correctness": "fail", "decision": "reject", "reason": "fixture correctness failure", "rollback": "restored_best"}
            (w / "runs/candidates.jsonl").write_text(json.dumps(record) + "\n")
            self.assertIn("restored_best", report(w, {"hypotheses": [hypothesis]}))


if __name__ == "__main__":
    unittest.main()

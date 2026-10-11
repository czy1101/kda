import base64
import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from test_kernelpilot import kernelpilot as K, valid_task
import compiler_evidence as C
import compiler_backends as B
import hypothesis_evidence as H

PROBE = """from pathlib import Path
import hashlib
v=float(Path('kernel.py').read_text())
cases=[]
for workload in ('reduce/fp32','matmul/fp16'):
 cases.append(dict(workload=workload,kernel='synthetic',input_id=hashlib.sha256(workload.encode()).hexdigest(),config={'tile':int(v)},artifacts={'ir':kda_compiler['artifact']('temporary_size='+str(v)), 'binary':kda_compiler['artifact'](b'fixed-code')},raw_resources={},metrics={'temporary_bytes':dict(value=v,unit='B',scope='kernel',definition='Synthetic allocation size',provenance='synthetic compiler fixture')}))
kda_compiler['emit'](cases,dict(name='synthetic',version='1',backend='cpu'))
"""
GUARD = "python3 -c 'import json;print(json.dumps(dict(idle=True,environment_id=\"fixture-v1\")))'"


class CompilerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        (self.ws / "kernel.py").write_text("32")
        (self.ws / "probe.py").write_text(PROBE)
        self.task = valid_task()
        self.task["target"] = {"backend": "cpu", "device": "synthetic"}
        self.task["constraints"]["forbidden_paths"] = ["probe.py"]
        self.task["analysis"] = {
            "compiler": {
                "probe": "probe.py",
                "guard_command": GUARD,
                "workloads": ["reduce/fp32", "matmul/fp16"],
            }
        }

    def execute(self, command):
        shell, script = K._command_script(self.task, command)
        return subprocess.run(
            [shell, "-lc", script], cwd=self.ws, text=True, capture_output=True
        )

    def capture(self, candidate="baseline", execute=None):
        return C.capture(self.ws, self.task, candidate, execute or self.execute)

    def spec(self, parent, candidate="c1"):
        return dict(
            id=candidate,
            parent_capture=parent["run_id"],
            statement="Temporary allocation is excessive",
            change="Reduce temporary allocation",
            expected_effect="Fewer temporary bytes",
            risks="May increase repeated computation",
            disproof="Allocation remains unchanged or grows",
            predictions=[
                dict(
                    workload="reduce/fp32",
                    kernel="synthetic",
                    kind="metric",
                    name="temporary_bytes",
                    relation="decrease",
                ),
                dict(
                    workload="matmul/fp16",
                    kernel="synthetic",
                    kind="artifact",
                    name="binary",
                    relation="unchanged",
                ),
            ],
        )

    def test_capture_compare_and_assess_do_not_authorize_promotion(self):
        a = self.capture()
        H.register(self.ws, self.task, self.spec(a))
        (self.ws / "kernel.py").write_text("16")
        b = self.capture("c1")
        result = H.assess(self.ws, self.task, "c1", b["run_id"])
        self.assertEqual(result["mechanism_status"], "supported")
        self.assertFalse(result["promotion_authorized"])
        self.assertEqual(result["performance_status"], "not_assessed")
        self.assertFalse((self.ws / ".kernelpilot/checkpoints").exists())
        comparison = json.loads(
            (
                self.ws / C.COMPARISONS / result["comparison_id"] / "record.json"
            ).read_text()
        )
        self.assertEqual(
            comparison["cases"][0]["metrics"]["temporary_bytes"]["delta"], -16
        )
        self.assertEqual(
            comparison["cases"][0]["stages"]["binary"]["status"], "identical_bytes"
        )
        self.assertTrue(comparison["artifacts"])

    def test_unchanged_codegen_disproves_changed_prediction(self):
        a = self.capture()
        spec = self.spec(a)
        spec["predictions"] = [
            dict(
                workload="reduce/fp32",
                kernel="synthetic",
                kind="artifact",
                name="binary",
                relation="changed",
            )
        ]
        H.register(self.ws, self.task, spec)
        (self.ws / "kernel.py").write_text("16")
        b = self.capture("c1")
        self.assertEqual(
            H.assess(self.ws, self.task, "c1", b["run_id"])["mechanism_status"],
            "not_supported",
        )

    def test_registration_requires_parent_before_edit_and_cannot_overwrite(self):
        a = self.capture()
        spec = self.spec(a)
        (self.ws / "kernel.py").write_text("16")
        with self.assertRaisesRegex(C.Error, "before editing"):
            H.register(self.ws, self.task, spec)
        (self.ws / "kernel.py").write_text("32")
        H.register(self.ws, self.task, spec)
        with self.assertRaises(FileExistsError):
            H.register(self.ws, self.task, spec)

    def test_assessment_does_not_replace_correctness_gate(self):
        self.task["analysis"]["hypotheses"] = "required"
        a = self.capture()
        H.register(self.ws, self.task, self.spec(a))
        (self.ws / "kernel.py").write_text("16")
        b = self.capture("c1")
        H.assess(self.ws, self.task, "c1", b["run_id"])
        with self.assertRaisesRegex(C.Error, "correctness"):
            K.run_contract_command(
                self.ws, self.task, "benchmark", candidate="c1", capture=True
            )
        self.assertFalse((self.ws / ".kernelpilot/checkpoints").exists())

    def test_metric_goal_does_not_change_mechanism_semantics(self):
        self.task["goal"].update(
            metric="throughput", direction="maximize", relative_to_baseline=1.1
        )
        a = self.capture()
        H.register(self.ws, self.task, self.spec(a))
        (self.ws / "kernel.py").write_text("16")
        b = self.capture("c1")
        result = H.assess(self.ws, self.task, "c1", b["run_id"])
        self.assertEqual(result["mechanism_status"], "supported")
        self.assertFalse(result["promotion_authorized"])

    def test_unsupported_prediction_remains_visible_and_is_not_performance_failure(
        self,
    ):
        self.task["analysis"]["hypotheses"] = "required"
        a = self.capture()
        H.register(self.ws, self.task, self.spec(a))
        (self.ws / "kernel.py").write_text("64")
        b = self.capture("c1")
        result = H.assess(self.ws, self.task, "c1", b["run_id"])
        self.assertEqual(result["mechanism_status"], "not_supported")
        H.require(self.ws, self.task, "c1", "benchmark")
        self.assertEqual(result["performance_status"], "not_assessed")

    def test_cli_reports_invalid_options_without_running_probe(self):
        (self.ws / ".kernelpilot").mkdir()
        (self.ws / ".kernelpilot/task.yaml").write_text(yaml.safe_dump(self.task))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                K.main(["compile", "capture", "--workspace", str(self.ws)]), 2
            )
        self.assertFalse((self.ws / C.ROOT).exists())

    def test_different_collector_implementations_rejected(self):
        a = self.capture()
        b = self.capture("c1")
        entry, _ = C.load(self.ws, b["run_id"])
        (entry / "exporter.py").write_text("# a different exporter")
        record = json.loads((entry / "record.json").read_text())
        record["artifacts"]["exporter.py"] = K.evidence.digest(entry / "exporter.py")
        (entry / "record.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(C.Error, "collector/adapter"):
            C.compare(self.ws, a["run_id"], b["run_id"])

    def test_new_contract_options_fail_closed_without_extra_dependencies(self):
        self.task["analysis"]["hypotheses"] = "enabled"
        errors = K.validate_task(self.ws, self.task)
        self.assertTrue(
            any("hypotheses must be optional or required" in x for x in errors)
        )
        self.task["analysis"]["hypotheses"] = "required"
        del self.task["analysis"]["compiler"]
        errors = K.validate_task(self.ws, self.task)
        self.assertTrue(any("required hypotheses need" in x for x in errors))

    def test_unknown_prediction_cannot_be_registered(self):
        a = self.capture()
        spec = self.spec(a)
        spec["predictions"][0]["name"] = "occupancy"
        with self.assertRaisesRegex(C.Error, "unavailable"):
            H.register(self.ws, self.task, spec)

    def test_gate_requires_registration_and_current_assessment(self):
        self.task["analysis"]["hypotheses"] = "required"
        a = self.capture()
        with self.assertRaisesRegex(C.Error, "register"):
            K.run_contract_command(
                self.ws, self.task, "correctness", candidate="c1", capture=True
            )
        H.register(self.ws, self.task, self.spec(a))
        (self.ws / "kernel.py").write_text("16")
        self.assertEqual(
            K.run_contract_command(
                self.ws, self.task, "correctness", candidate="c1", capture=True
            ).returncode,
            0,
        )
        with self.assertRaisesRegex(C.Error, "assess"):
            K.run_contract_command(
                self.ws, self.task, "benchmark", candidate="c1", capture=True
            )
        b = self.capture("c1")
        H.assess(self.ws, self.task, "c1", b["run_id"])
        H.require(self.ws, self.task, "c1", "benchmark")
        (self.ws / "kernel.py").write_text("15")
        with self.assertRaisesRegex(C.Error, "assess"):
            H.require(self.ws, self.task, "c1", "benchmark")

    def test_metric_semantics_mismatch_is_unknown_not_improvement(self):
        a = self.capture()
        H.register(self.ws, self.task, self.spec(a))
        (self.ws / "kernel.py").write_text("16")

        def execute(command):
            result = self.execute(command)
            if "KDA_COMPILE_RUN_ID=" in command:
                value = json.loads(result.stdout)
                value["cases"][0]["metrics"]["temporary_bytes"]["unit"] = "KiB"
                result.stdout = json.dumps(value)
            return result

        b = self.capture("c1", execute)
        self.assertEqual(
            H.assess(self.ws, self.task, "c1", b["run_id"])["mechanism_status"],
            "unknown",
        )

    def test_tampered_compiler_artifact_is_rejected(self):
        a = self.capture()
        entry, _ = C.load(self.ws, a["run_id"])
        (entry / "cases/0/binary").write_bytes(b"changed")
        with self.assertRaisesRegex(C.Error, "hash mismatch"):
            C.load(self.ws, a["run_id"])

    def test_source_mutation_and_process_failure_are_recorded(self):
        def mutate(command):
            result = self.execute(command)
            if "KDA_COMPILE_RUN_ID=" in command:
                (self.ws / "kernel.py").write_text("99")
            return result

        with self.assertRaisesRegex(C.Error, "inputs changed"):
            self.capture(execute=mutate)
        records = [
            json.loads(p.read_text()) for p in (self.ws / C.ROOT).glob("*/record.json")
        ]
        self.assertEqual(records[-1]["status"], "error")

        def fail(command):
            return (
                subprocess.CompletedProcess([], 9, "", "compile error")
                if "KDA_COMPILE_RUN_ID=" in command
                else self.execute(command)
            )

        with self.assertRaisesRegex(C.Error, "probe failed"):
            self.capture(execute=fail)
        self.assertEqual(len(list((self.ws / C.ROOT).glob("*/stderr.log"))), 2)

    def test_bad_envelopes_rejected(self):
        mutations = [
            lambda d: d.update(run_id="stale"),
            lambda d: d.update(cases=[]),
            lambda d: d["cases"].append(d["cases"][0]),
            lambda d: d["cases"][0]["artifacts"].update(
                {"../escape": dict(encoding="utf-8", data="x")}
            ),
            lambda d: d["cases"][0]["metrics"]["temporary_bytes"].update(
                value=float("nan")
            ),
            lambda d: d["cases"][0]["artifacts"]["binary"].update(data="!!"),
            lambda d: d["cases"][0].update(input_id="shape-only"),
            lambda d: d["cases"][0]["artifacts"]["ir"].update(data=""),
        ]
        for change in mutations:
            with self.subTest(change=change):

                def execute(command):
                    result = self.execute(command)
                    if "KDA_COMPILE_RUN_ID=" in command:
                        value = json.loads(result.stdout)
                        change(value)
                        result.stdout = json.dumps(value)
                    return result

                with self.assertRaises(C.Error):
                    self.capture(execute=execute)

    def test_environment_compiler_input_and_coverage_comparability(self):
        a = self.capture()
        for kind in ["environment", "compiler", "input", "kernel"]:

            def execute(command):
                result = self.execute(command)
                if kind == "environment" and "KDA_COMPILE_RUN_ID=" not in command:
                    result.stdout = json.dumps(
                        dict(idle=True, environment_id="different")
                    )
                if "KDA_COMPILE_RUN_ID=" in command:
                    d = json.loads(result.stdout)
                    if kind == "compiler":
                        d["compiler"]["version"] = "2"
                    if kind == "input":
                        d["cases"][0]["input_id"] = "0" * 64
                    if kind == "kernel":
                        d["cases"][0]["kernel"] = "another"
                    result.stdout = json.dumps(d)
                return result

            b = self.capture("c1", execute)
            with self.subTest(kind=kind), self.assertRaises(C.Error):
                C.compare(self.ws, a["run_id"], b["run_id"])

    def test_protected_input_change_blocks_comparison(self):
        a = self.capture()
        (self.ws / "probe.py").write_text(PROBE + "\n# changed\n")
        b = self.capture("c1")
        with self.assertRaisesRegex(C.Error, "protected"):
            C.compare(self.ws, a["run_id"], b["run_id"])

    def test_limits_busy_guard_and_probe_allowlist(self):
        self.task["analysis"]["compiler"]["max_artifact_bytes"] = 1
        with self.assertRaisesRegex(C.Error, "oversized"):
            self.capture()
        self.task["analysis"]["compiler"].pop("max_artifact_bytes")

        def busy(command):
            return subprocess.CompletedProcess(
                [], 0, '{"idle":false,"environment_id":"fixture"}', ""
            )

        with self.assertRaisesRegex(C.Error, "busy"):
            self.capture(execute=busy)
        self.task["constraints"]["forbidden_paths"] = []
        with self.assertRaisesRegex(C.Error, "protected"):
            C.config(self.task)

    def test_cli_and_simulated_controlled_ssh(self):
        remote = tempfile.TemporaryDirectory()
        self.addCleanup(remote.cleanup)
        executor = Path(remote.name)
        (executor / "kernel.py").write_text("32")
        (executor / "probe.py").write_text(PROBE)
        self.task["execution"] = {
            "transport": "ssh",
            "host": "fake",
            "workspace": str(executor),
        }
        (self.ws / ".kernelpilot").mkdir()
        (self.ws / ".kernelpilot/task.yaml").write_text(yaml.safe_dump(self.task))
        calls = []

        def remote_run(instance, script, shell="bash", capture=True):
            calls.append(script)
            return subprocess.run(
                [shell, "-lc", script], cwd=executor, text=True, capture_output=True
            )

        with patch.object(
            K.ControlledSSH, "run_script", remote_run
        ), contextlib.redirect_stdout(io.StringIO()) as output:
            code = K.main(
                [
                    "compile",
                    "capture",
                    "--workspace",
                    str(self.ws),
                    "--candidate",
                    "baseline",
                ]
            )
        self.assertEqual(code, 0)
        self.assertTrue(calls)
        record = json.loads(output.getvalue())
        self.assertEqual(record["status"], "pass")
        self.assertEqual(
            record["before"]["files"]["kernel.py"],
            K.evidence.digest(executor / "kernel.py"),
        )


class BackendSemanticsTests(unittest.TestCase):
    def test_metax_local_bytes_require_installed_driver_evidence(self):
        case = dict(raw_resources=dict(n_regs=64, n_spills=7, shared=1024), metrics={})
        m, notes = B.normalize("metax-triton", case, {})
        self.assertNotIn("local_memory_bytes", m)
        self.assertTrue(notes)
        source = b"mcFuncGetAttribute(&n_spills, MC_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun);\nn_spills /= 4;"
        m, notes = B.normalize("metax-triton", case, {"driver_source": source})
        self.assertEqual(m["local_memory_bytes"]["value"], 28)
        self.assertEqual(m["local_memory_bytes"]["unit"], "B")
        self.assertIn("not spill counts", m["local_memory_bytes"]["definition"])
        m, _ = B.normalize(
            "metax-triton", case, {"driver_source": b"/* " + source + b" */"}
        )
        self.assertNotIn("local_memory_bytes", m)

    def test_exporter_keeps_exact_instance_metadata(self):
        import compiler_export as X
        from types import SimpleNamespace

        compiled = SimpleNamespace(
            metadata={"name": "f", "shared": 2048, "num_warps": 4},
            asm={"ttgir": "IR", "binary": b"abc"},
            n_regs=48,
            n_spills=0,
        )
        case = X.triton_case("a", "0" * 64, compiled)
        self.assertEqual(case["config"], {"num_warps": 4})
        self.assertEqual(case["raw_resources"]["shared"], 2048)
        self.assertEqual(base64.b64decode(case["artifacts"]["binary"]["data"]), b"abc")


if __name__ == "__main__":
    unittest.main()

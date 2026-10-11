import contextlib
import io
import json
import tempfile
import subprocess
from unittest.mock import patch
import unittest
from pathlib import Path
from test_kernelpilot import kernelpilot as K, valid_task

M = K.measurement
E = K.evidence
HARNESS = """import csv,json,os,statistics
from pathlib import Path
value=float(Path("kernel.py").read_text())
options=json.loads(Path("options.json").read_text())
count_path=Path("count.txt")
count=int(count_path.read_text()) if count_path.exists() else 0
count_path.write_text(str(count+1))
scales=options.get("scales",[1])
value*=scales[count%len(scales)]
samples=options.get("samples",[value,value,value])
conditions=json.loads(os.environ["KERNELPILOT_MEASUREMENT_CONDITIONS"])
if options.get("wrong_conditions"): conditions["iterations"]+=1
run_id=os.environ["KERNELPILOT_RUN_ID"]
candidate=os.environ["KERNELPILOT_CANDIDATE"]
unit=options.get("unit","ms")
Path("runs").mkdir(exist_ok=True)
with open("runs/benchmark.csv","w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=["candidate","run_id","shape","metric_value","baseline_value","unit","status"])
 w.writeheader()
 for shape in ["forward/fp16/a","backward/bf16/b"]:
  w.writerow(dict(candidate=candidate,run_id=run_id,shape=shape,metric_value=statistics.median(samples),baseline_value=999,unit=unit,status="pass"))
data=dict(run_id=run_id,candidate=candidate,conditions=conditions,unit=unit,samples={s:samples for s in ["forward/fp16/a","backward/bf16/b"]})
if options.get("wrong_id"): data["run_id"]="stale"
Path("runs/samples.json").write_text(json.dumps(data))
if options.get("busy_after"): Path("device.json").write_text(json.dumps(dict(idle=False,environment_id="cpu-fixture")))
if options.get("env_after"): Path("device.json").write_text(json.dumps(dict(idle=True,environment_id="changed")))
"""


class MeasurementReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        (self.ws / "kernel.py").write_text("10")
        (self.ws / "check.py").write_text(
            'from pathlib import Path\nassert float(Path("kernel.py").read_text())>0\n'
        )
        (self.ws / "bench.py").write_text(HARNESS)
        (self.ws / "guard.py").write_text(
            'from pathlib import Path\nprint(Path("device.json").read_text())\n'
        )
        (self.ws / "options.json").write_text("{}")
        (self.ws / "device.json").write_text(
            json.dumps(dict(idle=True, environment_id="cpu-fixture"))
        )
        self.task = valid_task()
        self.task["baseline"]["role"] = "initial_implementation"
        self.task["correctness"]["command"] = "python3 check.py"
        self.task["benchmark"].update(
            command="python3 bench.py", shapes=["forward/fp16/a", "backward/bf16/b"]
        )
        self.task["constraints"]["forbidden_paths"] = [
            "check.py",
            "bench.py",
            "guard.py",
        ]
        self.task["goal"].update(
            aggregation="geomean_ratio", relative_to_baseline=0.8, max_regression=0.02
        )
        self.task["measurement"] = dict(
            conditions=dict(
                warmup=2,
                iterations=3,
                timing_scope="kernel",
                timer="fixture",
                input_id="seed0",
            ),
            process_repeats=2,
            min_samples=3,
            max_relative_mad=0.05,
            min_improvement=0.02,
            confirmation_pairs=2,
            guard_command="python3 guard.py",
            samples_path="runs/samples.json",
        )

    def run_stage(self, stage, candidate):
        return K.run_contract_command(
            self.ws, self.task, stage, candidate=candidate, capture=True
        )

    def keep(self, candidate):
        with (self.ws / "runs/candidates.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        id=candidate,
                        hypothesis="synthetic",
                        correctness="pass",
                        decision="keep",
                    )
                )
                + "\n"
            )

    def measure(self, candidate):
        self.assertEqual(self.run_stage("correctness", candidate).returncode, 0)
        return M.measure(self.ws, self.task, candidate, self.run_stage, K.evaluate_goal)

    def baseline(self):
        measured = self.measure("baseline")
        self.keep("baseline")
        K.save_checkpoint(self.ws, self.task, "best", "baseline")
        return measured

    def candidate(self, value=9, candidate="c1"):
        (self.ws / "kernel.py").write_text(str(value))
        measured = self.measure(candidate)
        self.keep(candidate)
        return measured

    def test_full_repeat_confirm_promote_and_preserve_original_baseline(self):
        original = self.baseline()
        frozen = (self.ws / M.FROZEN).read_bytes()
        candidate = self.candidate()
        confirmed = K.confirm_candidate(self.ws, self.task, "c1")
        self.assertEqual(confirmed["status"], "pass")
        self.assertEqual(len(confirmed["run_ids"]), 4)
        self.assertEqual(
            [M.run_record(self.ws, r)["candidate"] for r in confirmed["run_ids"]],
            ["baseline", "c1", "baseline", "c1"],
        )
        approved = K.save_checkpoint(self.ws, self.task, "best", "c1")
        self.assertFalse(approved["approval"]["evaluation"]["target_reached"])
        self.assertEqual(approved["approval"]["measurement_id"], candidate["id"])
        self.assertEqual((self.ws / M.FROZEN).read_bytes(), frozen)
        self.assertEqual(M.read_frozen(self.ws)["id"], original["id"])
        self.assertEqual(candidate["evaluation"]["per_shape"][0]["baseline_value"], 10)

    def test_baseline_cannot_be_refrozen(self):
        self.baseline()
        with self.assertRaisesRegex(E.EvidenceError, "already frozen"):
            M.measure(self.ws, self.task, "baseline", self.run_stage, K.evaluate_goal)

    def test_busy_guard_prevents_benchmark_execution(self):
        (self.ws / "device.json").write_text(
            json.dumps(dict(idle=False, environment_id="cpu-fixture"))
        )
        with self.assertRaisesRegex(E.EvidenceError, "busy"):
            self.measure("baseline")
        self.assertFalse((self.ws / "count.txt").exists())
        _, record = M.latest(self.ws, M.ROOT, "baseline")
        self.assertNotEqual(record["status"], "pass")

    def test_busy_after_run_preserves_raw_samples_but_blocks_measurement(self):
        (self.ws / "options.json").write_text(json.dumps(dict(busy_after=True)))
        with self.assertRaisesRegex(E.EvidenceError, "busy"):
            self.measure("baseline")
        entry, record = E.latest(self.ws, "baseline", "benchmark")
        self.assertTrue((entry / "samples.json").exists())
        self.assertNotEqual(record["status"], "pass")

    def test_changed_runtime_environment_rejected(self):
        (self.ws / "options.json").write_text(json.dumps(dict(env_after=True)))
        with self.assertRaisesRegex(E.EvidenceError, "environment changed"):
            self.measure("baseline")

    def test_within_process_noise_is_inconclusive(self):
        (self.ws / "options.json").write_text(json.dumps(dict(samples=[5, 10, 15])))
        with self.assertRaisesRegex(E.EvidenceError, "dispersion"):
            self.measure("baseline")

    def assert_attempt_link(self, attempt, status):
        self.assertEqual(attempt["status"], status)
        path = self.ws / E.ROOT / attempt["run_id"] / "record.json"
        self.assertEqual(attempt["record_sha256"], E.digest(path))
        self.assertEqual(json.loads(path.read_text())["status"], status)

    def test_second_noisy_repeat_is_linked_without_becoming_accepted(self):
        self.run_stage("correctness", "baseline")
        count = 0

        def changing(stage, candidate):
            nonlocal count
            if count:
                (self.ws / "options.json").write_text(
                    json.dumps(dict(samples=[5, 10, 15]))
                )
            count += 1
            return self.run_stage(stage, candidate)

        with self.assertRaisesRegex(E.EvidenceError, "dispersion"):
            M.measure(self.ws, self.task, "baseline", changing, K.evaluate_goal)
        _, record = M.latest(self.ws, M.ROOT, "baseline")
        self.assertEqual(len(record["run_ids"]), 1)
        self.assertEqual(len(record["attempts"]), 2)
        self.assert_attempt_link(record["attempts"][0], "pass")
        self.assert_attempt_link(record["attempts"][1], "inconclusive")
        self.assertEqual(record["attempts"][1]["repeat"], 1)
        self.assertFalse((self.ws / M.FROZEN).exists())

    def test_failed_benchmark_process_is_linked(self):
        self.task["benchmark"]["command"] = 'python3 -c "raise SystemExit(7)"'
        with self.assertRaisesRegex(E.EvidenceError, "benchmark process failed"):
            self.measure("baseline")
        _, record = M.latest(self.ws, M.ROOT, "baseline")
        self.assert_attempt_link(record["attempts"][0], "fail")
        self.assertFalse(record["run_ids"])

    def test_busy_guard_rejection_is_linked(self):
        (self.ws / "device.json").write_text(
            json.dumps(dict(idle=False, environment_id="cpu-fixture"))
        )
        with self.assertRaisesRegex(E.EvidenceError, "busy"):
            self.measure("baseline")
        _, record = M.latest(self.ws, M.ROOT, "baseline")
        self.assert_attempt_link(record["attempts"][0], "inconclusive")
        self.assertFalse((self.ws / "count.txt").exists())

    def test_failed_confirmation_arm_is_linked_and_restores_best(self):
        self.baseline()
        self.candidate()
        (self.ws / "options.json").write_text(json.dumps(dict(samples=[5, 10, 15])))
        with self.assertRaisesRegex(E.EvidenceError, "dispersion"):
            K.confirm_candidate(self.ws, self.task, "c1")
        _, record = M.latest(self.ws, M.CONFIRM, "c1")
        self.assert_attempt_link(record["attempts"][0], "inconclusive")
        self.assertEqual(record["attempts"][0]["arm"], "best")
        self.assertEqual(record["attempts"][0]["pair"], 0)
        self.assertEqual((self.ws / "kernel.py").read_text(), "10")

    def test_sequence_summaries_preserve_input_order(self):
        decreasing = M.stats([12, 11, 10, 9, 8, 7])
        increasing = M.stats([7, 8, 9, 10, 11, 12])
        self.assertEqual(decreasing["median"], increasing["median"])
        self.assertEqual(decreasing["relative_mad"], increasing["relative_mad"])
        self.assertEqual(decreasing["sequence_window_size"], 2)
        self.assertEqual(decreasing["sequence_first_median"], 11.5)
        self.assertEqual(decreasing["sequence_last_median"], 7.5)
        self.assertEqual(increasing["sequence_first_median"], 7.5)
        self.assertEqual(increasing["sequence_last_median"], 11.5)
        constant = M.stats([10, 10, 10])
        self.assertEqual(
            constant["sequence_first_median"], constant["sequence_last_median"]
        )

    def test_noisy_record_retains_all_shape_diagnostics(self):
        (self.ws / "options.json").write_text(json.dumps(dict(samples=[5, 10, 15])))
        with self.assertRaisesRegex(E.EvidenceError, "dispersion"):
            self.measure("baseline")
        _, record = E.latest(self.ws, "baseline", "benchmark")
        diag = record["diagnostics"]
        self.assertEqual(
            set(diag["exceeded_shapes"]), set(self.task["benchmark"]["shapes"])
        )
        self.assertEqual(set(diag["per_shape"]), set(self.task["benchmark"]["shapes"]))
        self.assertEqual(
            diag["per_shape"]["forward/fp16/a"]["sequence_first_median"], 5
        )
        self.assertFalse((self.ws / M.FROZEN).exists())

    def test_between_process_noise_retains_ordered_process_summaries(self):
        (self.ws / "options.json").write_text(json.dumps(dict(scales=[1, 1.4])))
        with self.assertRaisesRegex(E.EvidenceError, "between-process"):
            self.measure("baseline")
        _, record = M.latest(self.ws, M.ROOT, "baseline")
        diag = record["diagnostics"]
        self.assertEqual(
            set(diag["exceeded_shapes"]), set(self.task["benchmark"]["shapes"])
        )
        self.assertEqual(
            diag["per_shape"]["forward/fp16/a"]["sequence_first_median"], 10
        )
        self.assertEqual(
            diag["per_shape"]["forward/fp16/a"]["sequence_last_median"], 14
        )

    def test_noise_does_not_hide_later_invalid_csv_summary(self):
        shapes = self.task["benchmark"]["shapes"]
        raw = json.dumps(
            dict(
                run_id="r1",
                candidate="baseline",
                conditions=self.task["measurement"]["conditions"],
                unit="ms",
                samples={shapes[0]: [5, 10, 15], shapes[1]: [10, 10, 10]},
            )
        )
        rows = [
            dict(shape=shapes[0], unit="ms", metric_value=10),
            dict(shape=shapes[1], unit="ms", metric_value=11),
        ]
        with self.assertRaisesRegex(E.EvidenceError, "CSV summary"):
            M.accept_samples(raw, self.task, "baseline", "r1", rows)

    def test_between_process_noise_is_inconclusive(self):
        (self.ws / "options.json").write_text(json.dumps(dict(scales=[1, 1.4])))
        with self.assertRaisesRegex(E.EvidenceError, "between-process"):
            self.measure("baseline")

    def test_raw_samples_must_match_conditions_and_run(self):
        for flag in ["wrong_conditions", "wrong_id"]:
            (self.ws / "options.json").write_text(json.dumps({flag: True}))
            with self.assertRaisesRegex(
                E.EvidenceError, "frozen measurement conditions"
            ):
                self.measure("baseline")

    def test_insufficient_samples_rejected(self):
        (self.ws / "options.json").write_text(json.dumps(dict(samples=[10, 10])))
        with self.assertRaisesRegex(E.EvidenceError, "insufficient"):
            self.measure("baseline")

    def test_single_bound_run_cannot_replace_repeated_measurement(self):
        self.run_stage("correctness", "baseline")
        self.run_stage("benchmark", "baseline")
        self.keep("baseline")
        with self.assertRaisesRegex(E.EvidenceError, "no measurement"):
            K.save_checkpoint(self.ws, self.task, "best", "baseline")

    def test_no_confirmation_no_promotion(self):
        self.baseline()
        self.candidate()
        with self.assertRaisesRegex(E.EvidenceError, "no measurement/confirmation"):
            K.save_checkpoint(self.ws, self.task, "best", "c1")

    def test_unstable_apparent_winner_rejected_and_best_restored(self):
        self.baseline()
        self.candidate()
        # Both arms become equally fast: the apparent screen winner is noise.
        (self.ws / "options.json").write_text(json.dumps(dict(samples=[10, 10, 10])))
        with self.assertRaisesRegex(K.ContractError, "interleaved pair"):
            K.confirm_candidate(self.ws, self.task, "c1")
        self.assertEqual((self.ws / "kernel.py").read_text(), "10")
        _, record = M.latest(self.ws, M.CONFIRM, "c1")
        self.assertEqual(record["rollback"], "restored_best")
        self.assertEqual(len(record["pairs"]), 1)

    def test_small_gain_does_not_meet_confirmation_margin(self):
        self.baseline()
        self.candidate(9.99)
        with self.assertRaisesRegex(K.ContractError, "interleaved pair"):
            K.confirm_candidate(self.ws, self.task, "c1")
        self.assertEqual((self.ws / "kernel.py").read_text(), "10")

    def test_hardware_identity_change_across_measurements_rejected(self):
        self.baseline()
        (self.ws / "device.json").write_text(
            json.dumps(dict(idle=True, environment_id="different-device"))
        )
        with self.assertRaisesRegex(E.EvidenceError, "differs from frozen baseline"):
            self.candidate()

    def test_confirmation_becomes_stale_after_new_benchmark(self):
        self.baseline()
        self.candidate()
        K.confirm_candidate(self.ws, self.task, "c1")
        self.run_stage("benchmark", "c1")
        with self.assertRaisesRegex(E.EvidenceError, "latest candidate benchmark"):
            K.save_checkpoint(self.ws, self.task, "best", "c1")

    def test_raw_artifact_tampering_blocks_promotion(self):
        self.baseline()
        measured = self.candidate()
        run = measured["run_ids"][0]
        (self.ws / E.ROOT / run / "samples.json").write_text("{}")
        with self.assertRaisesRegex(E.EvidenceError, "hash mismatch"):
            K.confirm_candidate(self.ws, self.task, "c1")

    def test_policy_rejects_invalid_counts_and_thresholds(self):
        for key, value in [
            ("process_repeats", 1),
            ("min_samples", 2),
            ("confirmation_pairs", 1),
            ("min_improvement", float("nan")),
            ("max_relative_mad", -1),
        ]:
            original = self.task["measurement"][key]
            self.task["measurement"][key] = value
            with self.assertRaises(E.EvidenceError):
                M.policy(self.task)
            self.task["measurement"][key] = original

    def test_reference_cannot_replace_original_baseline(self):
        self.task["baseline"]["role"] = "external_reference"
        with self.assertRaisesRegex(E.EvidenceError, "external references"):
            self.measure("baseline")

    def test_candidate_sample_count_must_match_baseline(self):
        self.baseline()
        (self.ws / "options.json").write_text(json.dumps(dict(samples=[9, 9, 9, 9, 9])))
        with self.assertRaisesRegex(E.EvidenceError, "sample counts changed"):
            self.candidate()

    def test_frozen_baseline_record_tampering_detected(self):
        measured = self.baseline()
        path = self.ws / M.ROOT / measured["id"] / "record.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(E.EvidenceError, "record hash mismatch"):
            self.candidate()

    def test_failure_after_success_does_not_reuse_old_measurement(self):
        self.baseline()
        self.candidate()
        (self.ws / "device.json").write_text(
            json.dumps(dict(idle=False, environment_id="cpu-fixture"))
        )
        with self.assertRaises(E.EvidenceError):
            M.measure(self.ws, self.task, "c1", self.run_stage, K.evaluate_goal)
        (self.ws / "device.json").write_text(
            json.dumps(dict(idle=True, environment_id="cpu-fixture"))
        )
        with self.assertRaisesRegex(E.EvidenceError, "failed/stale"):
            K.confirm_candidate(self.ws, self.task, "c1")

    def test_measure_cli_runs_controller_repeats(self):
        import yaml

        contract = self.ws / "task.yaml"
        contract.write_text(yaml.safe_dump(self.task))
        self.run_stage("correctness", "baseline")
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = K.main(
                [
                    "measure",
                    "--workspace",
                    str(self.ws),
                    "--task",
                    str(contract),
                    "--candidate",
                    "baseline",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(stream.getvalue())["run_ids"]), 2)

    def test_interleaved_drift_is_inconclusive_and_restores_best(self):
        self.baseline()
        self.candidate()
        (self.ws / "count.txt").write_text("0")
        (self.ws / "options.json").write_text(json.dumps(dict(scales=[1, 1, 2, 2])))
        with self.assertRaisesRegex(E.EvidenceError, "drift/dispersion"):
            K.confirm_candidate(self.ws, self.task, "c1")
        _, record = M.latest(self.ws, M.CONFIRM, "c1")
        self.assertEqual(record["status"], "inconclusive")
        self.assertEqual(record["category"], "noise")
        self.assertEqual(record["rollback"], "restored_best")
        self.assertEqual((self.ws / "kernel.py").read_text(), "10")

    def test_controlled_remote_measurement_collects_raw_samples(self):
        self.task["execution"] = dict(
            transport="ssh", host="fixture", workspace=str(self.ws)
        )

        def execute_script(adapter, script, shell=None, capture=False, **kwargs):
            return subprocess.run(
                [shell or "bash", "-lc", script], capture_output=capture, text=True
            )

        with patch.object(
            E.ControlledSSH, "run_script", autospec=True, side_effect=execute_script
        ):
            record = self.measure("baseline")
        self.assertEqual(record["status"], "pass")
        self.assertEqual(record["environment_id"], "cpu-fixture")
        self.assertEqual(len(record["run_ids"]), 2)
        for run_id in record["run_ids"]:
            self.assertTrue((self.ws / E.ROOT / run_id / "samples.json").is_file())

    def test_maximized_throughput_confirmation(self):
        self.task["goal"].update(
            metric="throughput", direction="maximize", relative_to_baseline=1.2
        )
        (self.ws / "options.json").write_text(json.dumps(dict(unit="items/s")))
        self.baseline()
        self.candidate(11)
        K.confirm_candidate(self.ws, self.task, "c1")
        approved = K.save_checkpoint(self.ws, self.task, "best", "c1")
        self.assertFalse(approved["approval"]["evaluation"]["target_reached"])


if __name__ == "__main__":
    unittest.main()

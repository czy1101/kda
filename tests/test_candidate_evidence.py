import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from test_kernelpilot import kernelpilot, valid_task

E = kernelpilot.evidence
HARNESS = """import csv, os
from pathlib import Path
value=float(Path("kernel.py").read_text())
Path("runs").mkdir(exist_ok=True)
with open("runs/benchmark.csv", "w", newline="") as f:
 w=csv.DictWriter(f,fieldnames=["candidate","run_id","shape","metric_value","baseline_value","unit","status"])
 w.writeheader()
 for shape in ("forward/fp16/a", "backward/bf16/b"):
  w.writerow(dict(candidate=os.environ["KERNELPILOT_CANDIDATE"],run_id=os.environ["KERNELPILOT_RUN_ID"],shape=shape,metric_value=value,baseline_value=10,unit="ms",status="pass"))
"""


class BoundCandidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        (self.ws / "kernel.py").write_text("10")
        (self.ws / "check.py").write_text(
            'from pathlib import Path\nassert float(Path("kernel.py").read_text()) > 0\n'
        )
        (self.ws / "bench.py").write_text(HARNESS)
        self.task = valid_task()
        self.task["correctness"]["command"] = "python3 check.py"
        self.task["benchmark"].update(
            command="python3 bench.py", shapes=["forward/fp16/a", "backward/bf16/b"]
        )
        self.task["constraints"]["forbidden_paths"] = ["check.py", "bench.py"]
        self.task["goal"].update(aggregation="geomean_ratio", max_regression=0.02)

    def run_stage(self, stage, candidate="baseline"):
        return kernelpilot.run_contract_command(
            self.ws, self.task, stage, candidate=candidate, capture=True
        )

    def keep(self, candidate):
        (self.ws / "runs").mkdir(exist_ok=True)
        with (self.ws / "runs/candidates.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        id=candidate,
                        hypothesis="synthetic scalar workload",
                        correctness="pass",
                        decision="keep",
                    )
                )
                + "\n"
            )

    def measure(self, candidate="baseline"):
        self.assertEqual(self.run_stage("correctness", candidate).returncode, 0)
        self.assertEqual(self.run_stage("benchmark", candidate).returncode, 0)
        self.keep(candidate)

    def baseline(self):
        self.measure()
        return kernelpilot.save_checkpoint(self.ws, self.task, "best", "baseline")

    def test_can_promote_small_improvement_before_final_target(self):
        self.baseline()
        (self.ws / "kernel.py").write_text("9.9")
        self.measure("c1")
        metadata = kernelpilot.save_checkpoint(self.ws, self.task, "best", "c1")
        self.assertFalse(metadata["recovery_only"])
        self.assertFalse(metadata["approval"]["evaluation"]["target_reached"])
        (self.ws / "kernel.py").write_text("-1")
        kernelpilot.restore_checkpoint(self.ws, self.task)
        self.assertEqual((self.ws / "kernel.py").read_text(), "9.9")

    def test_source_change_after_correctness_blocks_benchmark(self):
        self.run_stage("correctness")
        (self.ws / "kernel.py").write_text("9")
        with self.assertRaises(E.EvidenceError):
            self.run_stage("benchmark")
        _, record = E.latest(self.ws, "baseline", "benchmark")
        self.assertEqual(record["status"], "error")
        self.assertNotIn("returncode", record)

    def test_source_change_after_benchmark_blocks_checkpoint(self):
        self.measure()
        (self.ws / "kernel.py").write_text("9")
        with self.assertRaises(E.EvidenceError):
            kernelpilot.save_checkpoint(self.ws, self.task, "best", "baseline")

    def test_contract_change_invalidates_evidence(self):
        self.measure()
        self.task["environment"] = {"setup": "export DIFFERENT_SETTING=1"}
        with self.assertRaises(E.EvidenceError):
            kernelpilot.save_checkpoint(self.ws, self.task, "best", "baseline")

    def test_protected_test_change_invalidates_evidence(self):
        self.measure()
        (self.ws / "check.py").write_text("pass")
        with self.assertRaises(E.EvidenceError):
            kernelpilot.save_checkpoint(self.ws, self.task, "best", "baseline")

    def test_failed_latest_correctness_supersedes_old_success(self):
        self.run_stage("correctness")
        with patch.object(
            kernelpilot.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 1, "", "bad"),
        ):
            self.assertEqual(self.run_stage("correctness").returncode, 1)
        with self.assertRaises(E.EvidenceError):
            self.run_stage("benchmark")

    def test_old_csv_cannot_be_reused_even_when_command_exits_zero(self):
        self.run_stage("correctness")
        self.run_stage("benchmark")
        with patch.object(
            kernelpilot.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, "", ""),
        ):
            with self.assertRaisesRegex(E.EvidenceError, "run_id"):
                self.run_stage("benchmark")

    def test_modified_frozen_artifact_is_rejected(self):
        self.measure()
        entry, _ = E.latest(self.ws, "baseline", "benchmark")
        (entry / "runs/benchmark.csv").write_text("modified")
        with self.assertRaisesRegex(E.EvidenceError, "hash mismatch"):
            kernelpilot.save_checkpoint(self.ws, self.task, "best", "baseline")

    def test_mutating_source_during_command_records_error(self):
        self.task["correctness"]["command"] = "printf 9 > kernel.py"
        with self.assertRaisesRegex(E.EvidenceError, "during execution"):
            self.run_stage("correctness")
        entry, record = E.latest(self.ws, "baseline", "correctness")
        self.assertEqual(record["status"], "error")
        self.assertTrue((entry / "stdout.log").exists())

    def test_benchmark_before_correctness_is_not_executed(self):
        with patch.object(kernelpilot.subprocess, "run") as execute:
            with self.assertRaises(E.EvidenceError):
                self.run_stage("benchmark")
            execute.assert_not_called()

    def test_unbound_legacy_results_cannot_approve_best(self):
        self.assertEqual(
            kernelpilot.run_contract_command(
                self.ws, self.task, "correctness", capture=True
            ).returncode,
            0,
        )
        with self.assertRaises(E.EvidenceError):
            kernelpilot.save_checkpoint(self.ws, self.task, "best", "c1")

    def test_slow_candidate_does_not_replace_best(self):
        self.baseline()
        best = self.ws / ".kernelpilot/checkpoints/best/checkpoint.json"
        before = best.read_bytes()
        (self.ws / "kernel.py").write_text("11")
        self.measure("slow")
        with self.assertRaises(kernelpilot.ContractError):
            kernelpilot.save_checkpoint(self.ws, self.task, "best", "slow")
        self.assertEqual(best.read_bytes(), before)

    def test_corrupted_checkpoint_is_not_restored(self):
        self.baseline()
        (self.ws / ".kernelpilot/checkpoints/best/kernel.py").write_text("tampered")
        (self.ws / "kernel.py").write_text("current")
        with self.assertRaisesRegex(kernelpilot.ContractError, "hash mismatch"):
            kernelpilot.restore_checkpoint(self.ws, self.task)
        self.assertEqual((self.ws / "kernel.py").read_text(), "current")

    def test_new_correctness_requires_new_benchmark(self):
        self.measure()
        self.run_stage("correctness")
        with self.assertRaisesRegex(kernelpilot.ContractError, "latest passing"):
            kernelpilot.save_checkpoint(self.ws, self.task, "best", "baseline")

    def test_symlink_source_is_rejected(self):
        (self.ws / "kernel.py").unlink()
        (self.ws / "kernel.py").symlink_to("check.py")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.run_stage("correctness")

    def test_changed_test_cannot_be_approved_by_rerunning_it(self):
        self.baseline()
        (self.ws / "check.py").write_text("pass")
        (self.ws / "kernel.py").write_text("8")
        self.measure("c1")
        with self.assertRaisesRegex(
            kernelpilot.ContractError, "protected inputs changed"
        ):
            kernelpilot.save_checkpoint(self.ws, self.task, "best", "c1")

    def test_generalization_must_match_current_benchmark(self):
        self.task["generalization"] = {"command": "python3 check.py"}
        self.measure()
        with self.assertRaises(E.EvidenceError):
            kernelpilot.save_checkpoint(self.ws, self.task, "best", "baseline")
        self.run_stage("generalization")
        kernelpilot.save_checkpoint(self.ws, self.task, "best", "baseline")
        self.run_stage("benchmark")
        with self.assertRaisesRegex(
            kernelpilot.ContractError, "generalization must follow"
        ):
            kernelpilot.save_checkpoint(self.ws, self.task, "best", "baseline")

    def test_baseline_numbers_cannot_change_between_candidates(self):
        self.baseline()

        # Change the emitted data without changing the frozen benchmark source.
        def altered(command):
            result = subprocess.run(
                ["bash", "-lc", command], cwd=self.ws, capture_output=True, text=True
            )
            p = self.ws / "runs/benchmark.csv"
            p.write_text(p.read_text().replace(",10,ms,", ",20,ms,"))
            return result

        (self.ws / "kernel.py").write_text("8")
        self.run_stage("correctness", "c1")
        E.run(self.ws, self.task, "benchmark", "c1", altered, kernelpilot.evaluate_goal)
        self.keep("c1")
        with self.assertRaisesRegex(
            kernelpilot.ContractError, "baseline values changed"
        ):
            kernelpilot.save_checkpoint(self.ws, self.task, "best", "c1")

    def test_throughput_candidate_uses_maximize_direction(self):
        self.task["goal"].update(
            metric="throughput", direction="maximize", relative_to_baseline=1.2
        )
        self.baseline()
        (self.ws / "kernel.py").write_text("11")
        self.measure("c1")
        result = kernelpilot.save_checkpoint(self.ws, self.task, "best", "c1")
        self.assertFalse(result["approval"]["evaluation"]["target_reached"])
        self.assertGreater(
            result["approval"]["comparison_to_best"]["aggregate_ratio"], 1
        )

    def test_checkpoint_install_failure_preserves_previous_best(self):
        self.baseline()
        before = (
            self.ws / ".kernelpilot/checkpoints/best/checkpoint.json"
        ).read_bytes()
        (self.ws / "kernel.py").write_text("8")
        self.measure("c1")
        rename = Path.rename

        def fail_install(path, target):
            if path.name == ".best.next":
                raise OSError("simulated installation failure")
            return rename(path, target)

        with patch.object(Path, "rename", fail_install):
            with self.assertRaises(OSError):
                kernelpilot.save_checkpoint(self.ws, self.task, "best", "c1")
        self.assertEqual(
            (self.ws / ".kernelpilot/checkpoints/best/checkpoint.json").read_bytes(),
            before,
        )

    def test_remote_snapshot_uses_executor_not_mirror(self):
        remote_task = dict(
            self.task,
            execution=dict(transport="ssh", host="fake", workspace=str(self.ws)),
        )

        def execute_script(adapter, script, shell=None, capture=False, **kwargs):
            return subprocess.run(
                [shell or "bash", "-lc", script], capture_output=capture, text=True
            )

        with patch.object(
            E.ControlledSSH, "run_script", autospec=True, side_effect=execute_script
        ):
            identity = E.source_snapshot(Path("/different/controller"), remote_task)
            self.assertEqual(
                identity["files"]["kernel.py"], E.digest(self.ws / "kernel.py")
            )
            self.assertEqual(
                kernelpilot.run_contract_command(
                    self.ws,
                    remote_task,
                    "correctness",
                    candidate="baseline",
                    capture=True,
                ).returncode,
                0,
            )
            self.assertEqual(
                kernelpilot.run_contract_command(
                    self.ws,
                    remote_task,
                    "benchmark",
                    candidate="baseline",
                    capture=True,
                ).returncode,
                0,
            )
            self.keep("baseline")
            mirror = self.ws / "remote-worktree"
            mirror.mkdir()
            (mirror / "kernel.py").write_text("9")
            with self.assertRaisesRegex(kernelpilot.ContractError, "mirror"):
                kernelpilot.save_checkpoint(self.ws, remote_task, "best", "baseline")
            (mirror / "kernel.py").write_text("10")
            metadata = kernelpilot.save_checkpoint(
                self.ws, remote_task, "best", "baseline"
            )
            self.assertFalse(metadata["recovery_only"])


if __name__ == "__main__":
    unittest.main()

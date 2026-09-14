import importlib.util
import csv
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/kernelpilot.py"
SPEC = importlib.util.spec_from_file_location("kernelpilot", SCRIPT)
kernelpilot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(kernelpilot)


def valid_task():
    return {
        "version": 1,
        "task": {"name": "demo", "description": "demo"},
        "target": {"backend": "cuda", "device": "H100"},
        "implementation": {"path": "kernel.py"},
        "correctness": {"command": "true"},
        "benchmark": {"command": "true"},
        "goal": {
            "metric": "latency",
            "direction": "minimize",
            "aggregation": "sum_ratio",
            "relative_to_baseline": 0.9,
        },
        "constraints": {
            "allowed_paths": ["kernel.py"],
            "forbidden_paths": ["tests"],
        },
    }


class KernelPilotTests(unittest.TestCase):
    def test_valid_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("pass\n", encoding="utf-8")
            self.assertEqual(kernelpilot.validate_task(workspace, valid_task()), [])

    def test_placeholder_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("pass\n", encoding="utf-8")
            task = valid_task()
            task["task"]["description"] = "<fill in>"
            errors = kernelpilot.validate_task(workspace, task)
            self.assertIn("task contract still contains <placeholder> values", errors)

    def test_environment_setup_is_applied(self):
        task = valid_task()
        task["environment"] = {"setup": "export KDA_SENTINEL=ready"}
        shell, script = kernelpilot._command_script(task, "test \"$KDA_SENTINEL\" = ready")
        self.assertEqual(shell, "bash")
        self.assertIn("export KDA_SENTINEL=ready", script)

    def test_forbidden_change_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tests = workspace / "tests"
            tests.mkdir()
            frozen = tests / "frozen.py"
            frozen.write_text("before\n", encoding="utf-8")
            task = valid_task()
            before = kernelpilot.snapshot_forbidden(workspace, task)
            frozen.write_text("after\n", encoding="utf-8")
            self.assertEqual(
                kernelpilot.compare_forbidden(workspace, task, before),
                ["tests/frozen.py"],
            )

    def test_goal_is_recomputed_from_required_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            runs = workspace / "runs"
            runs.mkdir()
            task = valid_task()
            task["benchmark"]["shapes"] = ["a", "b"]
            with (runs / "benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "candidate",
                        "shape",
                        "metric_value",
                        "baseline_value",
                        "unit",
                        "status",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "candidate": "c001",
                        "shape": "a",
                        "metric_value": 8,
                        "baseline_value": 10,
                        "unit": "ms",
                        "status": "pass",
                    }
                )
                writer.writerow(
                    {
                        "candidate": "c001",
                        "shape": "b",
                        "metric_value": 9,
                        "baseline_value": 10,
                        "unit": "ms",
                        "status": "pass",
                    }
                )
            evaluation, errors = kernelpilot.evaluate_goal(workspace, task, "c001")
            self.assertEqual(errors, [])
            self.assertTrue(evaluation["target_reached"])
            self.assertAlmostEqual(evaluation["aggregate_ratio"], 0.85)

    def test_any_aggregation_accepts_one_reaching_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            runs = workspace / "runs"
            runs.mkdir()
            task = valid_task()
            task["baseline"] = {"name": "reference", "command": "true"}
            task["goal"]["aggregation"] = "any"
            task["benchmark"]["shapes"] = ["a", "b"]
            with (runs / "benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "candidate",
                        "shape",
                        "metric_value",
                        "baseline_value",
                        "unit",
                        "status",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "candidate": "c001",
                        "shape": "a",
                        "metric_value": 8,
                        "baseline_value": 10,
                        "unit": "ms",
                        "status": "pass",
                    }
                )
                writer.writerow(
                    {
                        "candidate": "c001",
                        "shape": "b",
                        "metric_value": 12,
                        "baseline_value": 10,
                        "unit": "ms",
                        "status": "pass",
                    }
                )
            evaluation, errors = kernelpilot.evaluate_goal(workspace, task, "c001")
            self.assertEqual(errors, [])
            self.assertTrue(evaluation["target_reached"])
            self.assertEqual(evaluation["comparator"], "reference")
            self.assertAlmostEqual(evaluation["aggregate_ratio"], 0.8)

    def test_remote_contract_does_not_require_local_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            task = valid_task()
            task["execution"] = {
                "transport": "ssh",
                "host": "gpu-alias",
                "workspace": "/srv/operator",
            }
            self.assertEqual(kernelpilot.validate_task(workspace, task), [])

    def test_remote_adapter_rejects_undeclared_path(self):
        task = valid_task()
        task["execution"] = {
            "transport": "ssh",
            "host": "gpu-alias",
            "workspace": "/srv/operator",
        }
        adapter = kernelpilot.ControlledSSH(task)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(kernelpilot.SSHAdapterError):
                adapter.fetch("tests/test_kernel.py", Path(directory) / "test_kernel.py")

    def test_remote_prompt_requires_controlled_adapter(self):
        task = valid_task()
        task["execution"] = {
            "transport": "ssh",
            "host": "gpu-alias",
            "workspace": "/srv/operator",
        }
        prompt = kernelpilot.build_prompt(Path("/controller"), task)
        self.assertIn("Never invoke ssh, scp, rsync", prompt)
        self.assertIn("remote pull", prompt)
        self.assertIn("remote push", prompt)


if __name__ == "__main__":
    unittest.main()

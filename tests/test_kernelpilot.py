import importlib.util
import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_analysis_strategy_selects_reference_profile_comparison(self):
        task = valid_task()
        task["target"].update(
            {"profile": "cuda-sm90", "arch": "sm90", "language": "triton"}
        )
        task["profile"] = {
            "tool": "ncu",
            "candidate_command": "profile-candidate",
            "reference_command": "profile-reference",
            "export_command": "export-report",
        }
        task["analysis"] = {
            "profiling": "auto",
            "reference_profile_comparison": "auto",
            "low_level_artifacts": "auto",
            "ai_profile_fallback": True,
            "extensions": {"tle": "auto"},
        }
        discovery = {
            "status": "complete",
            "tools": [
                {"id": "ncu", "available": True},
                {"id": "nsys", "available": False},
            ],
        }
        strategy = kernelpilot.select_analysis_strategy(task, discovery)
        self.assertEqual(strategy["route"], "profile_compare")
        self.assertEqual(strategy["selected_profiler"], "ncu")
        self.assertEqual(strategy["low_level_artifacts"], ["triton-ir", "ptx", "sass"])
        self.assertTrue(strategy["ai_profile_fallback"])
        self.assertEqual(strategy["active_extensions"], [])
        self.assertIn("tle", strategy["skipped_extensions"])

    def test_cuda_backend_profile_and_discovery_skills_validate(self):
        task = valid_task()
        task["target"].update(
            {"profile": "cuda-sm90", "arch": "sm90", "language": "triton"}
        )
        self.assertEqual(kernelpilot.validate_backend_profile(task), [])

    def test_analysis_strategy_can_disable_profiling(self):
        task = valid_task()
        task["target"].update(
            {"profile": "cuda-sm90", "arch": "sm90", "language": "triton"}
        )
        task["profile"] = {"tool": "ncu", "command": "profile-candidate"}
        task["analysis"] = {"profiling": "disabled"}
        strategy = kernelpilot.select_analysis_strategy(task)
        self.assertEqual(strategy["route"], "benchmark_only")

    def test_analysis_strategy_requires_live_discovery_before_auto_profile(self):
        task = valid_task()
        task["target"].update(
            {"profile": "cuda-sm90", "arch": "sm90", "language": "triton"}
        )
        task["profile"] = {
            "tool": "auto",
            "tools": {"ncu": {"candidate_command": "profile-candidate"}},
        }
        task["analysis"] = {"tool_discovery": "auto", "profiling": "auto"}
        strategy = kernelpilot.select_analysis_strategy(task)
        self.assertEqual(strategy["route"], "discover_tools")
        self.assertTrue(strategy["discovery_required"])

    def test_live_discovery_records_tools_and_strategy_uses_available_one(self):
        task = valid_task()
        task["target"].update(
            {"profile": "cuda-sm90", "arch": "sm90", "language": "triton"}
        )
        task["profile"] = {
            "tool": "auto",
            "tools": {
                "ncu": {"candidate_command": "profile-with-ncu"},
                "nsys": {"candidate_command": "profile-with-nsys"},
            },
        }
        task["analysis"] = {
            "tool_discovery": "auto",
            "preferred_tools": ["nsys", "ncu"],
            "profiling": "auto",
        }
        completed = [
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "nsys 1.0", ""),
            subprocess.CompletedProcess([], 1, "", "not found"),
            subprocess.CompletedProcess([], 1, "", "not found"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with patch.object(kernelpilot, "_run_environment_probe", side_effect=completed):
                discovery = kernelpilot.discover_analysis_tools(workspace, task)
            self.assertEqual(discovery["selected"], "nsys")
            self.assertTrue((workspace / "runs/analysis-capabilities.json").is_file())
            strategy = kernelpilot.select_analysis_strategy(task, discovery)
            self.assertEqual(strategy["selected_profiler"], "nsys")
            self.assertEqual(strategy["route"], "profile_candidate")
            self.assertIn("ncu", strategy["unavailable_profilers"])

            changed_task = dict(task)
            changed_task["environment"] = {"setup": "export DIFFERENT_ENV=1"}
            with self.assertRaises(kernelpilot.ContractError):
                kernelpilot.load_analysis_capabilities(workspace, changed_task)

    def test_workflow_transitions_reference_existing_stages(self):
        workflow_path = SCRIPT.parents[1] / "workflows/kernel-optimization.yaml"
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        stages = workflow["stages"]
        ids = {stage["id"] for stage in stages}
        self.assertEqual(len(ids), len(stages))
        for stage in stages:
            for key, target in stage.items():
                if key == "next" or key == "otherwise" or key.startswith("on_"):
                    self.assertIn(target, ids, f"{stage['id']}.{key} -> {target}")

    def test_structured_result_requires_stop_reason_and_workflow_issues(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "stopped",
                        "best_candidate": "baseline",
                        "target_reached": False,
                        "correctness": "pass",
                        "summary": "No actionable bottleneck remained.",
                    }
                ),
                encoding="utf-8",
            )
            _, errors = kernelpilot.load_final_result(path)
            self.assertTrue(errors)
            self.assertIn("stop_reason", errors[0])


if __name__ == "__main__":
    unittest.main()

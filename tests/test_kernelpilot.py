import importlib.util
import contextlib
import csv
import io
import hashlib
import json
import shlex
import subprocess
import sys
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
controlled_ssh = sys.modules["controlled_ssh"]


def valid_task():
    return {
        "version": 1,
        "task": {"name": "demo", "description": "demo"},
        "target": {"backend": "cuda", "device": "H100"},
        "implementation": {"path": "kernel.py"},
        "correctness": {"command": "true"},
        "benchmark": {"command": "true"},
        "baseline": {"name": "original"},
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

    def test_quality_contract_requires_consistent_operator_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("pass\n", encoding="utf-8")
            task = valid_task()
            task["target"] = {"backend": "test", "device": "test"}
            task["quality"] = {
                "required": True,
                "public_api": {"command": "true"},
                "module_layers": {"command": "true"},
                "naming": {
                    "op_name": "example_attention",
                    "marker": "example_attention",
                    "registry": {"path": "operators.yaml", "id": "example_attention"},
                    "command": "true",
                },
                "pytest": {"marker": "example_attention", "command": "true"},
                "dispatch": {"operator_id": "example_attention", "command": "true"},
            }
            task["constraints"]["allowed_paths"].append("operators.yaml")
            self.assertEqual(kernelpilot.validate_task(workspace, task), [])
            task["quality"]["dispatch"]["operator_id"] = "other_attention"
            self.assertIn(
                "quality.dispatch.operator_id must equal quality.naming.op_name",
                kernelpilot.validate_task(workspace, task),
            )

    def test_quality_checks_write_machine_readable_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("pass\n", encoding="utf-8")
            task = valid_task()
            task["quality"] = {
                "required": True,
                "public_api": {"command": "true"},
                "module_layers": {"command": "true"},
                "import_paths": {},
                "naming": {
                    "op_name": "example_attention",
                    "marker": "example_attention",
                    "registry": {"path": "operators.yaml", "id": "example_attention"},
                    "command": "true",
                },
                "pytest": {"marker": "example_attention", "command": "true"},
                "dispatch": {"operator_id": "example_attention", "command": "true"},
            }
            task["constraints"]["allowed_paths"].append("operators.yaml")
            result = kernelpilot.run_quality_checks(workspace, task)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(set(result["checks"]), set(kernelpilot.QUALITY_CHECK_NAMES))
            evidence = json.loads((workspace / "runs/quality.json").read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "pass")

    def test_quality_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            task = valid_task()
            task["quality"] = {
                "required": True,
                "public_api": {"command": "false"},
                "module_layers": {"command": "true"},
                "import_paths": {},
                "naming": {
                    "op_name": "example_attention",
                    "marker": "example_attention",
                    "registry": {"path": "operators.yaml", "id": "example_attention"},
                    "command": "true",
                },
                "pytest": {"marker": "example_attention", "command": "true"},
                "dispatch": {"operator_id": "example_attention", "command": "true"},
            }
            task["constraints"]["allowed_paths"].append("operators.yaml")
            result = kernelpilot.run_quality_checks(workspace, task)
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["checks"]["public_api"]["status"], "fail")

    def test_import_path_quality_gate_rejects_relative_import(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "src/flag_attn/runtime/backend/_hygon/FLA"
            source.mkdir(parents=True)
            (source / "index.py").write_text("VALUE = 1\n", encoding="utf-8")
            implementation = source / "operator.py"
            implementation.write_text("from .index import VALUE\n", encoding="utf-8")
            task = valid_task()
            task["implementation"]["path"] = "src/flag_attn/runtime/backend/_hygon/FLA/operator.py"
            task["constraints"]["allowed_paths"] = [
                "src/flag_attn/runtime/backend/_hygon/FLA"
            ]
            task["quality"] = {
                "required": True,
                "public_api": {"command": "true"},
                "module_layers": {"command": "true"},
                "import_paths": {},
                "naming": {
                    "op_name": "example_attention",
                    "marker": "example_attention",
                    "registry": {"path": "operators.yaml", "id": "example_attention"},
                    "command": "true",
                },
                "pytest": {"marker": "example_attention", "command": "true"},
                "dispatch": {"operator_id": "example_attention", "command": "true"},
            }
            task["constraints"]["allowed_paths"].append("operators.yaml")

            result = kernelpilot.run_quality_checks(workspace, task)

            self.assertEqual(result["checks"]["import_paths"]["status"], "fail")
            self.assertIn("operator.py:1", result["checks"]["import_paths"]["stderr"])
            self.assertIn("from .index import", result["checks"]["import_paths"]["stderr"])

    def test_import_path_audit_runs_without_quality_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("from .index import VALUE\n", encoding="utf-8")

            result = kernelpilot.run_quality_checks(workspace, valid_task())

            self.assertEqual(result["status"], "fail")
            self.assertTrue(result["required"])
            self.assertEqual(set(result["checks"]), {"import_paths"})

    def test_import_path_quality_gate_accepts_repository_absolute_import(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "src/flag_attn/runtime/backend/_hygon/FLA"
            source.mkdir(parents=True)
            (source / "operator.py").write_text(
                "from flag_attn.runtime.backend._hygon.FLA.index import VALUE\n",
                encoding="utf-8",
            )
            task = valid_task()
            task["implementation"]["path"] = "src/flag_attn/runtime/backend/_hygon/FLA/operator.py"
            task["constraints"]["allowed_paths"] = [
                "src/flag_attn/runtime/backend/_hygon/FLA"
            ]
            task["quality"] = {
                "required": True,
                "public_api": {"command": "true"},
                "module_layers": {"command": "true"},
                "import_paths": {},
                "naming": {
                    "op_name": "example_attention",
                    "marker": "example_attention",
                    "registry": {"path": "operators.yaml", "id": "example_attention"},
                    "command": "true",
                },
                "pytest": {"marker": "example_attention", "command": "true"},
                "dispatch": {"operator_id": "example_attention", "command": "true"},
            }
            task["constraints"]["allowed_paths"].append("operators.yaml")

            result = kernelpilot.run_quality_checks(workspace, task)

            self.assertEqual(result["checks"]["import_paths"]["status"], "pass")

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

    def test_remote_container_contract_does_not_require_local_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            task = valid_task()
            task["target"] = {"backend": "test", "device": "test"}
            task["execution"] = {
                "transport": "ssh",
                "host": "hygon-bastion",
                "workspace": "/workspace/FlagAttention",
                "container": "jiangrenkai-dev",
            }
            self.assertEqual(kernelpilot.validate_task(workspace, task), [])

    def test_remote_container_wraps_commands_and_stdin_sync(self):
        task = valid_task()
        task["execution"] = {
            "transport": "ssh",
            "host": "hygon-bastion",
            "workspace": "/workspace/FlagAttention",
            "container": "jiangrenkai-dev",
        }
        adapter = kernelpilot.ControlledSSH(task)
        script = "set -euo pipefail\ncd -- /workspace/FlagAttention\ncat > candidate.py"
        argv = adapter._argv(script)
        self.assertEqual(argv[:4], ["ssh", "hygon-bastion", "bash", "-lc"])
        quoted_remote_command = shlex.split(argv[4])
        self.assertEqual(len(quoted_remote_command), 1)
        remote_argv = shlex.split(quoted_remote_command[0])
        self.assertEqual(
            remote_argv[:4], ["docker", "exec", "-i", "jiangrenkai-dev"]
        )
        self.assertEqual(remote_argv[4:], ["bash", "-lc", script])

    def test_remote_container_fetch_and_push_preserve_binary_bytes(self):
        task = valid_task()
        task["execution"] = {
            "transport": "ssh",
            "host": "hygon-bastion",
            "workspace": "/workspace/FlagAttention",
            "container": "jiangrenkai-dev",
        }
        adapter = kernelpilot.ControlledSSH(task)
        payload = b"\x00\xffkernel\n"
        digest = hashlib.sha256(payload).hexdigest()
        completed = [
            subprocess.CompletedProcess([], 0, payload, b""),
            subprocess.CompletedProcess([], 0, f"{digest} kernel.py\n", ""),
            subprocess.CompletedProcess([], 0, b"", b""),
        ]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "kernel.py"
            with patch.object(controlled_ssh.subprocess, "run", side_effect=completed) as run:
                self.assertEqual(adapter.fetch("kernel.py", destination), digest)
                self.assertEqual(destination.read_bytes(), payload)
                source = Path(directory) / "candidate.py"
                source.write_bytes(payload)
                self.assertEqual(adapter.push("kernel.py", source, digest), digest)

        calls = run.call_args_list
        fetch_argv = calls[0].args[0]
        remote_fetch = shlex.split(shlex.split(fetch_argv[-1])[0])
        self.assertEqual(remote_fetch[:4], ["docker", "exec", "-i", "jiangrenkai-dev"])
        self.assertFalse(calls[0].kwargs["text"])
        hash_argv = calls[1].args[0]
        remote_hash = shlex.split(shlex.split(hash_argv[-1])[0])
        self.assertEqual(remote_hash[:4], ["docker", "exec", "-i", "jiangrenkai-dev"])
        push_argv = calls[2].args[0]
        remote_push = shlex.split(shlex.split(push_argv[-1])[0])
        self.assertEqual(remote_push[:4], ["docker", "exec", "-i", "jiangrenkai-dev"])
        self.assertEqual(calls[2].kwargs["input"], payload)

    def test_remote_container_name_rejects_shell_syntax(self):
        task = valid_task()
        task["execution"] = {
            "transport": "ssh",
            "host": "hygon-bastion",
            "workspace": "/workspace/FlagAttention",
            "container": "jiangrenkai-dev; touch /tmp/pwned",
        }
        with self.assertRaises(kernelpilot.SSHAdapterError):
            kernelpilot.ControlledSSH(task)._argv("true")

    def test_remote_container_requires_ssh_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            task = valid_task()
            task["execution"] = {"transport": "local", "container": "gpu"}
            errors = kernelpilot.validate_task(Path(directory), task)
            self.assertIn("execution.container requires execution.transport: ssh", errors)

    def test_remote_state_is_bound_to_container(self):
        task = valid_task()
        task["execution"] = {
            "transport": "ssh",
            "host": "hygon-bastion",
            "workspace": "/workspace/FlagAttention",
            "container": "jiangrenkai-dev",
        }
        with tempfile.TemporaryDirectory() as directory:
            control_workspace = Path(directory)
            state_file = controlled_ssh.state_path(control_workspace)
            state_file.parent.mkdir(parents=True)
            state_file.write_text(
                json.dumps(
                    {
                        "host": "hygon-bastion",
                        "workspace": "/workspace/FlagAttention",
                        "container": "another-container",
                        "hashes": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                kernelpilot.SSHAdapterError, "target does not match"
            ):
                controlled_ssh._load_state(control_workspace, task)

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

    def test_remote_container_prompt_explains_container_boundary(self):
        task = valid_task()
        task["execution"] = {
            "transport": "ssh",
            "host": "hygon-bastion",
            "workspace": "/workspace/FlagAttention",
            "container": "jiangrenkai-dev",
        }
        prompt = kernelpilot.build_prompt(Path("/controller"), task)
        self.assertIn("Docker container jiangrenkai-dev", prompt)
        self.assertIn("file synchronization", prompt)

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
        self.assertIn("tle", strategy["pending_extensions"])

    def test_cuda_backend_profile_and_discovery_skills_validate(self):
        task = valid_task()
        task["target"].update(
            {"profile": "cuda-sm90", "arch": "sm90", "language": "triton"}
        )
        self.assertEqual(kernelpilot.validate_backend_profile(task), [])

    def test_hygon_backend_profile_and_discovery_skills_validate(self):
        task = valid_task()
        task["target"] = {
            "backend": "hygon",
            "device": "BW",
            "arch": "gfx936",
            "profile": "hygon-gfx936",
            "language": "triton",
        }
        self.assertEqual(kernelpilot.validate_backend_profile(task), [])

    def test_hygon_strategy_prefers_rocprofv2_for_kernel_profiles(self):
        task = valid_task()
        task["target"] = {
            "backend": "hygon",
            "device": "BW",
            "arch": "gfx936",
            "profile": "hygon-gfx936",
            "language": "triton",
        }
        task["profile"] = {
            "tool": "auto",
            "tools": {
                "rocprofv2": {
                    "candidate_command": "trace-candidate",
                    "reference_command": "trace-reference",
                },
                "hipprof": {"candidate_command": "timeline-candidate"},
            },
        }
        task["analysis"] = {
            "tool_discovery": "auto",
            "profiling": "auto",
            "reference_profile_comparison": "auto",
        }
        discovery = {
            "status": "complete",
            "tools": [
                {"id": "rocprofv2", "available": True},
                {"id": "hipprof", "available": True},
            ],
        }
        strategy = kernelpilot.select_analysis_strategy(task, discovery)
        self.assertEqual(strategy["selected_profiler"], "rocprofv2")
        self.assertEqual(strategy["route"], "profile_compare")

    def test_analysis_strategy_can_disable_profiling(self):
        task = valid_task()
        task["target"].update(
            {"profile": "cuda-sm90", "arch": "sm90", "language": "triton"}
        )
        task["profile"] = {"tool": "ncu", "command": "profile-candidate"}
        task["analysis"] = {"profiling": "disabled"}
        strategy = kernelpilot.select_analysis_strategy(task)
        self.assertEqual(strategy["route"], "benchmark_only")

    def test_unique_backend_profile_is_selected_when_omitted(self):
        profile = kernelpilot._backend_profile(valid_task())
        self.assertEqual(profile["id"], "cuda-sm90")

    def test_ambiguous_backend_profile_is_not_guessed(self):
        with patch.object(
            kernelpilot,
            "_matching_backend_profiles",
            return_value=[{"id": "a"}, {"id": "b"}],
        ):
            with self.assertRaises(kernelpilot.ContractError):
                kernelpilot._backend_profile(valid_task())

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

    def test_absolute_target_is_evaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            runs = workspace / "runs"
            runs.mkdir()
            task = valid_task()
            task["goal"].pop("relative_to_baseline")
            task["goal"]["target_value"] = 9.0
            task["goal"]["aggregation"] = "all"
            task["benchmark"]["shapes"] = ["a", "b"]
            with (runs / "benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("candidate", "shape", "metric_value", "baseline_value", "unit", "status"),
                )
                writer.writeheader()
                writer.writerow({"candidate": "c1", "shape": "a", "metric_value": 8, "baseline_value": 10, "unit": "ms", "status": "pass"})
                writer.writerow({"candidate": "c1", "shape": "b", "metric_value": 9, "baseline_value": 10, "unit": "ms", "status": "pass"})
            evaluation, errors = kernelpilot.evaluate_goal(workspace, task, "c1")
            self.assertEqual(errors, [])
            self.assertTrue(evaluation["target_reached"])

    def test_remote_profile_uses_selected_reference_command(self):
        task = valid_task()
        task["execution"] = {"transport": "ssh", "host": "gpu", "workspace": "/srv/op"}
        task["profile"] = {
            "tool": "auto",
            "tools": {"ncu": {"reference_command": "profile-reference"}},
        }
        completed = subprocess.CompletedProcess([], 0, "", "")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(kernelpilot.ControlledSSH, "run_script", return_value=completed) as run:
                kernelpilot.run_contract_command(
                    Path(directory), task, "profile", tool="ncu", variant="reference"
                )
        self.assertIn("profile-reference", run.call_args.args[0])

    def test_checkpoint_restores_best_local_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            kernel = workspace / "kernel.py"
            kernel.write_text("baseline\n", encoding="utf-8")
            task = valid_task()
            kernelpilot.save_checkpoint(workspace, task, "baseline", "baseline")
            kernel.write_text("best\n", encoding="utf-8")
            kernelpilot.save_checkpoint(workspace, task, "best", "c1")
            kernel.write_text("failed\n", encoding="utf-8")
            restored = kernelpilot.restore_checkpoint(workspace, task)
            self.assertEqual(restored["candidate"], "c1")
            self.assertEqual(kernel.read_text(encoding="utf-8"), "best\n")

    def test_required_tle_cannot_use_failed_assessment(self):
        task = valid_task()
        task["analysis"] = {"extensions": {"tle": "required"}}
        with self.assertRaises(kernelpilot.ContractError):
            kernelpilot.select_analysis_strategy(
                task,
                discovery={"status": "no_tool_available", "tools": []},
                tle_assessment={"status": "unavailable", "reason": "wiki missing"},
            )

    def test_tle_assessment_requires_wiki_and_api(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "tle-wiki").mkdir()
            (workspace / "tle-wiki/README.md").write_text("api\n", encoding="utf-8")
            task = valid_task()
            task["analysis"] = {"extensions": {"tle": "auto"}}
            task["extension_config"] = {
                "tle": {"wiki_path": "tle-wiki", "api_probe_command": "true"}
            }
            assessment = kernelpilot.assess_tle(workspace, task)
            self.assertEqual(assessment["status"], "available")
            self.assertTrue((workspace / "runs/tle-assessment.json").is_file())

    def test_remote_tle_wiki_read_is_confined_to_configured_root(self):
        task = valid_task()
        task["execution"] = {"transport": "ssh", "host": "gpu", "workspace": "/srv/op"}
        task["analysis"] = {"extensions": {"tle": "auto"}}
        task["extension_config"] = {"tle": {"wiki_path": "tle-wiki"}}
        listed = subprocess.CompletedProcess([], 0, "tle-wiki/README.md\n", "")
        read = subprocess.CompletedProcess([], 0, "wiki body\n", "")
        with patch.object(
            kernelpilot.ControlledSSH, "run_script", side_effect=[listed, read]
        ):
            self.assertEqual(kernelpilot.list_tle_wiki(task), ["README.md"])
            self.assertEqual(kernelpilot.read_tle_wiki(task, "README.md"), "wiki body\n")
        with self.assertRaises(kernelpilot.SSHAdapterError):
            kernelpilot.read_tle_wiki(task, "../secret")

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

    def test_contract_requires_goal_metric_and_valid_stop_block(self):
        task = valid_task()
        del task["goal"]["metric"]
        self.assertIn(
            "goal.metric must be a non-empty string",
            kernelpilot.validate_task(Path("."), task),
        )

        task = valid_task()
        task["stop"] = {"max_candidates": 0, "target_reached": "yes"}
        errors = kernelpilot.validate_task(Path("."), task)
        self.assertIn("stop.max_candidates must be an integer >= 1", errors)
        self.assertIn("stop.target_reached must be boolean", errors)

    def test_contract_validates_target_benchmark_and_reference(self):
        task = valid_task()
        task["target"]["device"] = ""
        task["benchmark"]["warmup"] = "20"
        task["benchmark"]["shapes"] = [""]
        task["reference"] = {"path": ""}
        errors = kernelpilot.validate_task(Path("."), task)
        self.assertIn("target.device must be a non-empty string", errors)
        self.assertIn("benchmark.warmup must be an integer >= 0", errors)
        self.assertIn("benchmark.shapes must be a list of non-empty strings", errors)
        self.assertIn("reference.path must be a non-empty string when configured", errors)

    def test_contract_reports_wrong_section_types_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("pass\n", encoding="utf-8")
            task = valid_task()
            task["constraints"] = "kernel.py"
            task["goal"] = "latency"
            task["stop"] = 3
            errors = kernelpilot.validate_task(workspace, task)
        self.assertIn("constraints must be a mapping", errors)
        self.assertIn("goal must be a mapping", errors)
        self.assertIn("stop must be a mapping", errors)

    def test_verify_uses_the_requested_task_file(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / ".kernelpilot").mkdir()
            (workspace / "kernel.py").write_text("pass\n", encoding="utf-8")
            task = valid_task()
            task["analysis"] = {"extensions": {"tle": "required"}}
            (workspace / ".kernelpilot/custom.yaml").write_text(
                yaml.safe_dump(task), encoding="utf-8"
            )
            buffer = io.StringIO()
            with contextlib.redirect_stderr(buffer):
                code = kernelpilot.main(
                    [
                        "verify",
                        "--workspace",
                        str(workspace),
                        "--task",
                        ".kernelpilot/custom.yaml",
                    ]
                )
        self.assertEqual(code, 2)
        self.assertIn("TLE assessment", buffer.getvalue())

    def _evidence_workspace(self, directory: str) -> Path:
        """A workspace with every evidence file verify_evidence requires."""
        workspace = Path(directory)
        (workspace / "kernel.py").write_text("pass\n", encoding="utf-8")
        (workspace / "docs").mkdir(exist_ok=True)
        for name in ("draft.md", "plan.md", "final-report.md"):
            (workspace / "docs" / name).write_text("x\n", encoding="utf-8")
        (workspace / "runs").mkdir(exist_ok=True)
        for name in ("workflow-state.json", "stage-events.jsonl"):
            (workspace / "runs" / name).write_text("{}\n", encoding="utf-8")
        (workspace / "runs/candidates.jsonl").write_text(
            json.dumps(
                {
                    "id": "baseline",
                    "hypothesis": "baseline",
                    "correctness": "pass",
                    "decision": "keep",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (workspace / "runs/benchmark.csv").write_text(
            "candidate,shape,metric_value,baseline_value,unit,status\n"
            "baseline,s1,1.0,1.0,ms,pass\n",
            encoding="utf-8",
        )
        (workspace / "runs/quality.json").write_text(
            json.dumps({"status": "pass", "checks": {"import_paths": {"status": "pass"}}}),
            encoding="utf-8",
        )
        return workspace

    def test_verify_requires_generalization_evidence_when_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = self._evidence_workspace(directory)
            task = valid_task()
            task["generalization"] = {"command": "true", "max_regression": 0.02}
            errors = kernelpilot.verify_evidence(workspace, task)
            self.assertTrue(
                any("missing generalization evidence" in error for error in errors),
                errors,
            )

            (workspace / "runs/generalization.json").write_text(
                json.dumps({"exit_code": 0, "command": "true"}), encoding="utf-8"
            )
            errors = kernelpilot.verify_evidence(workspace, task)
            self.assertFalse(
                any("generalization" in error for error in errors), errors
            )

    def test_verify_rejects_stale_or_failed_generalization_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = self._evidence_workspace(directory)
            task = valid_task()
            task["generalization"] = {"command": "python panel.py"}
            path = workspace / "runs/generalization.json"

            path.write_text(json.dumps({"exit_code": 1, "command": "python panel.py"}))
            self.assertIn(
                "generalization panel did not pass",
                kernelpilot.verify_evidence(workspace, task),
            )

            path.write_text(json.dumps({"exit_code": 0, "command": "python old.py"}))
            self.assertTrue(
                any(
                    "different command" in error
                    for error in kernelpilot.verify_evidence(workspace, task)
                )
            )

    def test_run_generalization_records_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            task = valid_task()
            task["generalization"] = {"command": "python panel.py", "max_regression": 0.02}
            completed = subprocess.CompletedProcess([], 0, "guard ok\n", "")
            with patch.object(
                kernelpilot, "run_contract_command", return_value=completed
            ) as runner:
                record = kernelpilot.run_generalization(workspace, task)
            self.assertTrue(record["passed"])
            self.assertEqual(record["max_regression"], 0.02)
            stored = json.loads(
                (workspace / "runs/generalization.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stored["exit_code"], 0)
            self.assertIn("KDA_GENERALIZATION_MAX_REGRESSION=0.02",
                          runner.call_args.args[1]["generalization"]["command"])

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
            self.assertIn("knowledge_publication", errors[0])

    def _write_ledgers(self, workspace, candidate, decision, metric_value="0.8"):
        (workspace / "runs").mkdir(exist_ok=True)
        (workspace / "runs/candidates.jsonl").write_text(
            json.dumps(
                {
                    "id": candidate,
                    "hypothesis": "tile reuse",
                    "correctness": "pass",
                    "decision": decision,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with (workspace / "runs/benchmark.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "candidate",
                    "shape",
                    "metric_value",
                    "baseline_value",
                    "unit",
                    "status",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "candidate": candidate,
                    "shape": "demo-shape",
                    "metric_value": metric_value,
                    "baseline_value": "1.0",
                    "unit": "ms",
                    "status": "pass",
                }
            )

    def test_save_best_refuses_candidate_without_keep_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("pass\n", encoding="utf-8")
            self._write_ledgers(workspace, "c1", "revise")
            with self.assertRaises(kernelpilot.ContractError) as raised:
                kernelpilot.save_checkpoint(
                    workspace, valid_task(), "best", "c1", require_evidence=True
                )
            self.assertIn("not 'keep'", str(raised.exception))

    def test_save_best_refuses_candidate_without_benchmark_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("pass\n", encoding="utf-8")
            self._write_ledgers(workspace, "c1", "keep")
            (workspace / "runs/benchmark.csv").unlink()
            with self.assertRaises(kernelpilot.ContractError) as raised:
                kernelpilot.save_checkpoint(
                    workspace, valid_task(), "best", "c1", require_evidence=True
                )
            self.assertIn("no measured evidence", str(raised.exception))

    def test_save_best_records_promotion_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("best\n", encoding="utf-8")
            self._write_ledgers(workspace, "c1", "keep")
            metadata = kernelpilot.save_checkpoint(
                workspace, valid_task(), "best", "c1", require_evidence=True
            )
            self.assertTrue(metadata["promotion_evidence"]["target_reached"])
            self.assertAlmostEqual(metadata["promotion_evidence"]["aggregate_ratio"], 0.8)
            checkpoint = json.loads(
                (workspace / ".kernelpilot/checkpoints/best/checkpoint.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(checkpoint["promotion_evidence"]["shapes"], 1)

    def test_save_best_ignores_unrelated_malformed_record(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("best\n", encoding="utf-8")
            self._write_ledgers(workspace, "c1", "keep")
            with (workspace / "runs/candidates.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                # A rejected candidate without its required reason/rollback must
                # not block promotion of a different, fully evidenced candidate.
                handle.write(
                    json.dumps(
                        {
                            "id": "c2",
                            "hypothesis": "regressed",
                            "correctness": "pass",
                            "decision": "reject",
                        }
                    )
                    + "\n"
                )
            metadata = kernelpilot.save_checkpoint(
                workspace, valid_task(), "best", "c1", require_evidence=True
            )
            self.assertTrue(metadata["promotion_evidence"]["target_reached"])

    def test_save_best_refuses_candidate_with_own_ledger_problem(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("pass\n", encoding="utf-8")
            self._write_ledgers(workspace, "c1", "pass")
            with self.assertRaises(kernelpilot.ContractError) as raised:
                kernelpilot.save_checkpoint(
                    workspace, valid_task(), "best", "c1", require_evidence=True
                )
            self.assertIn("invalid decision", str(raised.exception))

    def test_verify_requires_tle_assessment_when_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "docs").mkdir()
            for name in ("draft.md", "plan.md", "final-report.md"):
                (workspace / "docs" / name).write_text("x\n", encoding="utf-8")
            (workspace / "runs").mkdir()
            for name in ("workflow-state.json", "stage-events.jsonl"):
                (workspace / "runs" / name).write_text("{}\n", encoding="utf-8")
            (workspace / "runs/candidates.jsonl").write_text(
                json.dumps(
                    {
                        "id": "baseline",
                        "hypothesis": "baseline",
                        "correctness": "pass",
                        "decision": "keep",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (workspace / "runs/benchmark.csv").write_text(
                "candidate,shape,metric_value,baseline_value,unit,status\n"
                "baseline,s1,1.0,1.0,ms,pass\n",
                encoding="utf-8",
            )
            (workspace / "runs/quality.json").write_text(
                json.dumps(
                    {"status": "pass", "checks": {"import_paths": {"status": "pass"}}}
                ),
                encoding="utf-8",
            )
            task = valid_task()
            task["analysis"] = {"extensions": {"tle": "required"}}
            errors = kernelpilot.verify_evidence(workspace, task)
            self.assertTrue(
                any("missing TLE assessment evidence" in error for error in errors),
                errors,
            )

    def test_tle_policy_is_read_from_analysis_extensions(self):
        task = valid_task()
        task["analysis"] = {"extensions": {"tle": "auto"}}
        self.assertEqual(kernelpilot._tle_policy(task), "auto")
        task["analysis"]["extensions"]["tle"] = "required"
        self.assertEqual(kernelpilot._tle_policy(task), "required")
        self.assertEqual(kernelpilot._tle_policy(valid_task()), "disabled")

    def test_remote_forbidden_snapshot_parses_files_dirs_and_missing(self):
        task = valid_task()
        task["execution"] = {"transport": "ssh", "host": "gpu", "workspace": "/srv/op"}
        task["constraints"]["forbidden_paths"] = [
            "tests/bench.py",
            "tests/data",
            "tests/none.py",
        ]
        completed = subprocess.CompletedProcess(
            [],
            0,
            "file\ttests/bench.py\tdeadbeef\n"
            "dir\ttests/data\n"
            "file\ttests/data/a.py\tcafe\n"
            "missing\ttests/none.py\n",
            "",
        )
        with patch.object(kernelpilot.ControlledSSH, "run_script", return_value=completed):
            snapshot = controlled_ssh.hash_forbidden(task)
        self.assertEqual(snapshot["tests/bench.py"], "deadbeef")
        self.assertEqual(snapshot["tests/data/"], "dir")
        self.assertEqual(snapshot["tests/data/a.py"], "cafe")
        self.assertEqual(snapshot["tests/none.py"], "<missing>")

    def test_remote_forbidden_change_is_detected(self):
        task = valid_task()
        task["execution"] = {"transport": "ssh", "host": "gpu", "workspace": "/srv/op"}
        task["constraints"]["forbidden_paths"] = ["tests/bench.py"]
        before_run = subprocess.CompletedProcess(
            [], 0, "file\ttests/bench.py\tdeadbeef\n", ""
        )
        after_run = subprocess.CompletedProcess(
            [], 0, "file\ttests/bench.py\tfeedface\n", ""
        )
        with patch.object(
            kernelpilot.ControlledSSH, "run_script", side_effect=[before_run, after_run]
        ):
            before = kernelpilot.snapshot_forbidden_remote(task)
            changed = kernelpilot.compare_forbidden(
                Path("."), task, before, remote=True
            )
        self.assertEqual(changed, ["tests/bench.py"])


if __name__ == "__main__":
    unittest.main()

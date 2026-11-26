"""Tests for the two-step diagnosis gate and its backward compatibility.

These tests exercise the structural validator, the opt-in evidence policy, the
generated prompt routing and the workflow declaration. They do not measure a
diagnosis, and a passing run says nothing about whether any real diagnosis was
correct.
"""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/kernelpilot.py"
SPEC = importlib.util.spec_from_file_location("kernelpilot", SCRIPT)
kernelpilot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(kernelpilot)

LAYER_IDS = (
    "algorithm_numerics",
    "operator_dataflow",
    "kernel_implementation",
    "launch_config_autotune",
    "compiler_codegen",
    "runtime_system",
)


def valid_screen():
    layers = []
    for index, layer_id in enumerate(LAYER_IDS):
        layers.append(
            {
                "id": layer_id,
                "evidence": f"evidence for {layer_id}",
                "continue": "yes" if index == 4 else "no",
                "key_unknown": "none" if index != 4 else "which stage differs",
                "decision_supported": "whether to rewrite the source or the config",
            }
        )
    return {
        "version": 1,
        "layers": layers,
        "deep_dives": [
            {
                "direction": "compiler",
                "hypothesis": "the FP32 tile is transposed through shared memory",
                "control": "same shape, dtype, config and device",
                "expected_change": "one fewer transpose pair in gcuir",
                "supports_if": "the transpose count drops and the median drops",
                "refutes_if": "the transpose count is unchanged",
                "next_action": "rewrite the source, or drop the compiler direction",
                "stop": "one capture pair or two profiling runs",
            }
        ],
        "change_class": {"kind": "kernel", "evidence": "source-only diff"},
    }


def base_task(**analysis):
    block = {"tool_discovery": "disabled", "profiling": "disabled"}
    block.update(analysis)
    return {
        "version": 1,
        "task": {"name": "demo", "description": "demo"},
        "target": {"backend": "gcu", "device": "S60", "arch": "gcu300"},
        "implementation": {"path": "kernel.py"},
        "correctness": {"command": "true"},
        "benchmark": {"command": "true"},
        "baseline": {"name": "original"},
        "goal": {
            "metric": "latency",
            "direction": "minimize",
            "aggregation": "sum_ratio",
            "relative_to_baseline": 0.95,
        },
        "constraints": {"allowed_paths": ["kernel.py"], "forbidden_paths": ["tests"]},
        "analysis": block,
    }


def prepare_workspace(workspace: Path) -> None:
    (workspace / "kernel.py").write_text("pass\n", encoding="utf-8")
    (workspace / "docs").mkdir()
    (workspace / "runs").mkdir()
    for name in ("draft.md", "plan.md", "final-report.md"):
        (workspace / "docs" / name).write_text("# x\n", encoding="utf-8")
    (workspace / "runs" / "workflow-state.json").write_text("{}\n", encoding="utf-8")
    (workspace / "runs" / "stage-events.jsonl").write_text("{}\n", encoding="utf-8")
    (workspace / "runs" / "candidates.jsonl").write_text(
        json.dumps(
            {
                "id": "baseline",
                "hypothesis": "unchanged",
                "correctness": "pass",
                "decision": "keep",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "runs" / "benchmark.csv").write_text(
        "candidate,shape,metric_value,baseline_value,unit,status\n"
        "baseline,1x1,1.0,1.0,ms,pass\n",
        encoding="utf-8",
    )


def write_screen(workspace: Path, screen) -> None:
    path = workspace / kernelpilot.BOTTLENECK_SCREEN
    text = screen if isinstance(screen, str) else json.dumps(screen)
    path.write_text(text, encoding="utf-8")


class DiagnosisGateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        prepare_workspace(self.workspace)

    def tearDown(self):
        self._tmp.cleanup()

    def _errors(self, task):
        return kernelpilot.verify_evidence(self.workspace, task)

    def test_absent_screen_passes_by_default_for_existing_tasks(self):
        # Default policy is `auto`: a workspace written before this record
        # existed must keep validating unchanged.
        self.assertEqual(kernelpilot.verify_evidence(self.workspace, base_task()), [])
        self.assertEqual(kernelpilot.verify_evidence(self.workspace, None), [])

    def test_required_policy_rejects_a_missing_screen(self):
        errors = self._errors(base_task(bottleneck_screen="required"))
        self.assertTrue(any("bottleneck_screen is required" in error for error in errors))

    def test_required_policy_accepts_a_valid_screen(self):
        write_screen(self.workspace, valid_screen())
        errors = self._errors(base_task(bottleneck_screen="required"))
        self.assertEqual(errors, [])

    def test_malformed_screen_is_rejected_even_under_auto(self):
        write_screen(self.workspace, "{not json")
        errors = self._errors(base_task())
        self.assertTrue(any("unreadable bottleneck screen" in error for error in errors))

    def test_disabled_policy_ignores_even_a_malformed_screen(self):
        write_screen(self.workspace, "{not json")
        errors = self._errors(base_task(bottleneck_screen="disabled"))
        self.assertEqual(errors, [])

    def test_missing_layer_is_reported_by_name(self):
        screen = valid_screen()
        screen["layers"] = screen["layers"][:-1]
        errors = kernelpilot.validate_bottleneck_screen(screen)
        self.assertTrue(any("missing layers: runtime_system" in e for e in errors), errors)

    def test_duplicate_layer_is_rejected(self):
        screen = valid_screen()
        screen["layers"][1]["id"] = screen["layers"][0]["id"]
        errors = kernelpilot.validate_bottleneck_screen(screen)
        self.assertTrue(any("duplicate layer ids" in e for e in errors), errors)

    def test_unknown_disposition_is_rejected(self):
        screen = valid_screen()
        screen["layers"][0]["continue"] = "maybe"
        errors = kernelpilot.validate_bottleneck_screen(screen)
        self.assertTrue(any("continue must be one of" in e for e in errors), errors)

    def test_continue_yes_requires_a_deep_dive(self):
        screen = valid_screen()
        screen["deep_dives"] = []
        errors = kernelpilot.validate_bottleneck_screen(screen)
        self.assertTrue(any("deep_dives must name" in e for e in errors), errors)

    def test_deep_dive_requires_every_falsification_field(self):
        screen = valid_screen()
        del screen["deep_dives"][0]["refutes_if"]
        errors = kernelpilot.validate_bottleneck_screen(screen)
        self.assertTrue(
            any("deep_dives[0].refutes_if" in e for e in errors), errors
        )

    def test_all_layers_not_applicable_needs_no_deep_dive(self):
        screen = valid_screen()
        for layer in screen["layers"]:
            layer["continue"] = "not_applicable"
        screen["deep_dives"] = []
        screen["deep_dives"] = [
            {
                "direction": "none",
                "hypothesis": "no actionable direction remains",
                "control": "n/a",
                "expected_change": "none",
                "supports_if": "no candidate is justified",
                "refutes_if": "a new piece of evidence changes a disposition",
                "next_action": "finish with an evidence-backed stop",
                "stop": "immediately",
            }
        ]
        self.assertEqual(kernelpilot.validate_bottleneck_screen(screen), [])

    def test_change_class_must_be_one_of_the_known_kinds(self):
        screen = valid_screen()
        screen["change_class"] = {"kind": "magic", "evidence": "x"}
        errors = kernelpilot.validate_bottleneck_screen(screen)
        self.assertTrue(any("change_class.kind" in e for e in errors), errors)

    def test_unavailable_tools_can_be_recorded_and_do_not_block_the_screen(self):
        # A missing profiler or a dump that was not enabled is a normal outcome.
        # The screen must be able to say so and still pass, otherwise the gate
        # would push the agent to invent evidence or to skip the screen.
        screen = valid_screen()
        screen["layers"][4]["continue"] = "no"
        screen["layers"][4]["evidence"] = (
            "capability unavailable: MLIR dump not enabled for this run; no stage "
            "artifact was captured, so this layer stays unmeasured"
        )
        screen["layers"][5]["evidence"] = (
            "capability unavailable: no profiler discovered for this device"
        )
        screen["deep_dives"][0]["direction"] = "kernel"
        self.assertEqual(kernelpilot.validate_bottleneck_screen(screen), [])
        write_screen(self.workspace, screen)
        self.assertEqual(self._errors(base_task(bottleneck_screen="required")), [])

    def test_blank_layer_evidence_is_rejected_so_absence_must_be_stated(self):
        screen = valid_screen()
        screen["layers"][5]["evidence"] = "   "
        errors = kernelpilot.validate_bottleneck_screen(screen)
        self.assertTrue(any("layers[5].evidence" in e for e in errors), errors)

    def test_screen_does_not_require_a_capability_file(self):
        # The screen is a written decision, not a tool report. A workspace with
        # no analysis-capabilities.json still validates, so a device-free run is
        # not blocked by a missing tool inventory.
        self.assertFalse((self.workspace / "runs/analysis-capabilities.json").exists())
        write_screen(self.workspace, valid_screen())
        self.assertEqual(self._errors(base_task(bottleneck_screen="required")), [])

    def test_counter_observation_requires_a_scope_statement(self):
        # A counter reading is an observation, not a causal claim, and it is only
        # meaningful inside the documented scope of that counter.
        screen = valid_screen()
        screen["attribution"] = {"control_level": "L2", "observation": "counter"}
        errors = kernelpilot.validate_bottleneck_screen(screen)
        self.assertTrue(any("scope_note" in e for e in errors), errors)
        screen["attribution"]["scope_note"] = "unit activity, not latency; counts inclusive"
        self.assertEqual(kernelpilot.validate_bottleneck_screen(screen), [])

    def test_attribution_axes_are_validated_separately(self):
        screen = valid_screen()
        screen["attribution"] = {"control_level": "L4", "observation": "telemetry"}
        errors = kernelpilot.validate_bottleneck_screen(screen)
        self.assertTrue(any("control_level" in e for e in errors), errors)
        self.assertTrue(any("observation must be one of" in e for e in errors), errors)

    def test_decomposition_must_state_three_numbers_and_a_relationship(self):
        screen = valid_screen()
        screen["decomposition"] = {
            "source_at_fixed_config": "-9.1%",
            "config_at_fixed_source": "-5.6%",
            "deployed_combination": "-14.75%",
            "relationship": "not-additive",
        }
        self.assertEqual(kernelpilot.validate_bottleneck_screen(screen), [])
        del screen["decomposition"]["deployed_combination"]
        errors = kernelpilot.validate_bottleneck_screen(screen)
        self.assertTrue(
            any("decomposition.deployed_combination" in e for e in errors), errors
        )
        screen["decomposition"]["deployed_combination"] = "-14.75%"
        screen["decomposition"]["relationship"] = "adds-up"
        errors = kernelpilot.validate_bottleneck_screen(screen)
        self.assertTrue(any("relationship must be one of" in e for e in errors), errors)


class ContractCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        prepare_workspace(self.workspace)

    def tearDown(self):
        self._tmp.cleanup()

    def test_new_policy_is_accepted_and_bad_values_are_rejected(self):
        for value in ("auto", "disabled", "required"):
            self.assertEqual(
                kernelpilot.validate_task(
                    self.workspace, base_task(bottleneck_screen=value)
                ),
                [],
            )
        errors = kernelpilot.validate_task(
            self.workspace, base_task(bottleneck_screen="sometimes")
        )
        self.assertTrue(
            any("analysis.bottleneck_screen" in error for error in errors), errors
        )

    def test_task_schema_declares_the_new_field_as_optional(self):
        with (ROOT / "schemas/task.schema.yaml").open(encoding="utf-8") as handle:
            schema = yaml.safe_load(handle)
        field = schema["properties"]["analysis"]["properties"]["bottleneck_screen"]
        self.assertEqual(field["enum"], ["auto", "disabled", "required"])
        self.assertEqual(field["default"], "auto")
        # Optional: analysis declares no required members, and this field must
        # not become one, or every task written before it existed would fail.
        self.assertNotIn(
            "bottleneck_screen", schema["properties"]["analysis"].get("required", [])
        )

    def test_old_task_without_the_field_still_validates(self):
        task = base_task()
        task["analysis"].pop("bottleneck_screen", None)
        self.assertEqual(kernelpilot.validate_task(self.workspace, task), [])

    def test_workflow_declares_the_screen_evidence_and_keeps_the_action(self):
        with (ROOT / "workflows/kernel-optimization.yaml").open(encoding="utf-8") as handle:
            workflow = yaml.safe_load(handle)
        self.assertEqual(
            workflow["evidence"]["bottleneck_screen"], "runs/bottleneck-screen.json"
        )
        stages = {stage["id"]: stage for stage in workflow["stages"]}
        # The stage action is unchanged: the screen is required content of the
        # existing diagnose stage, not a renamed action pretending to automate it.
        self.assertEqual(stages["diagnose"]["action"], "diagnose_bottleneck")
        self.assertEqual(stages["diagnose"]["on_known_cause"], "select_optimization_strategy")

    def test_schema_file_documents_the_six_layers(self):
        schema = json.loads(
            (ROOT / "schemas/bottleneck-screen.schema.json").read_text(encoding="utf-8")
        )
        ids = schema["properties"]["layers"]["items"]["properties"]["id"]["enum"]
        self.assertEqual(tuple(ids), LAYER_IDS)


class PromptRoutingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        prepare_workspace(self.workspace)

    def tearDown(self):
        self._tmp.cleanup()

    def test_gcu_prompt_points_at_the_two_diagnosis_documents(self):
        prompt = kernelpilot.build_prompt(self.workspace, base_task())
        self.assertIn("bottleneck-localization.md", prompt)
        self.assertIn("enflame-compiler-diagnostics.md", prompt)
        self.assertIn("bottleneck-screen.json", prompt)
        self.assertIn("compile_artifacts.py", prompt)

    def test_non_gcu_prompt_does_not_inject_enflame_documents(self):
        task = copy.deepcopy(base_task())
        task["target"] = {"backend": "cuda", "device": "H100", "arch": "sm90"}
        prompt = kernelpilot.build_prompt(self.workspace, task)
        self.assertNotIn("enflame-compiler-diagnostics.md", prompt)
        self.assertNotIn("bottleneck-localization.md", prompt)
        # The offline artifact tool is backend-neutral and stays available.
        self.assertIn("compile_artifacts.py", prompt)

    def test_disabled_policy_does_not_ask_for_the_screen_in_the_prompt(self):
        # `disabled` must not instruct the agent to produce a record the gate
        # will ignore; the compiler diagnostics document stays available.
        prompt = kernelpilot.build_prompt(
            self.workspace, base_task(bottleneck_screen="disabled")
        )
        self.assertNotIn("bottleneck-screen.json", prompt)
        self.assertNotIn("bottleneck-localization.md", prompt)
        self.assertIn("enflame-compiler-diagnostics.md", prompt)

    def test_prompt_keeps_the_existing_evidence_wrappers(self):
        prompt = kernelpilot.build_prompt(self.workspace, base_task())
        for wrapper in ("run correctness", "run benchmark", "checkpoint save-best"):
            self.assertIn(wrapper, prompt)


if __name__ == "__main__":
    unittest.main()

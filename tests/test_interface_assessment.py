"""Tests for the nine-step loop routing and the interface-assessment guidance.

The prompt and the documents are checked for content and cross-reference
integrity. Nothing here validates a means, compiles anything, or measures a
kernel: a passing run says nothing about whether Gluon, TLE or an intrinsic works
on any device.
"""

import copy
import importlib.util
import re
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

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "test_diagnosis_gate", ROOT / "tests/test_diagnosis_gate.py"
)
helpers = importlib.util.module_from_spec(_HELPER_SPEC)
assert _HELPER_SPEC.loader is not None
_HELPER_SPEC.loader.exec_module(helpers)

MEANS_DOC = ROOT / "docs/enflame-interface-assessment.md"
SOP_DOC = ROOT / "docs/agent-flow.md"
STATIC_PROMPT = ROOT / "prompts/optimize-kernel.md"
RUNBOOK = ROOT / "docs/enflame-s60-runbook.md"

GATES = ("G1", "G2", "G3", "G4", "G5")


def relative_links(path: Path):
    """Yield the in-repository link targets of one markdown file."""
    text = path.read_text(encoding="utf-8")
    for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
        target = match.group(1).strip()
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        target = target.split("#", 1)[0]
        if target:
            yield target


class PromptRoutingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        helpers.prepare_workspace(self.workspace)

    def tearDown(self):
        self._tmp.cleanup()

    def gcu_prompt(self, **analysis):
        return kernelpilot.build_prompt(
            self.workspace, helpers.base_task(**analysis)
        )

    def test_gcu_prompt_names_the_interface_assessment(self):
        prompt = self.gcu_prompt()
        self.assertIn("enflame-interface-assessment.md", prompt)
        self.assertIn("eight items", prompt)
        self.assertIn("five capability gates", prompt)
        self.assertIn("five questions", prompt)

    def test_gcu_prompt_keeps_the_three_vocabularies_apart(self):
        prompt = self.gcu_prompt()
        self.assertIn("three separate vocabularies", prompt)
        self.assertIn("do not merge them", prompt)
        # The classes are enumerated so the layer list cannot be read as an edit
        # list. `none` is not a class: it means no change was selected.
        for kind in kernelpilot.CHANGE_CLASSES:
            if kind == "none":
                continue
            self.assertIn(kind, prompt)

    def test_non_gcu_prompt_gets_the_strategy_step_without_the_enflame_documents(self):
        task = copy.deepcopy(helpers.base_task())
        task["target"] = {"backend": "cuda", "device": "H100", "arch": "sm90"}
        prompt = kernelpilot.build_prompt(self.workspace, task)
        self.assertIn("agent-flow.md", prompt)
        self.assertIn("Select the strategy before implementing", prompt)
        self.assertIn("docs/plan.md", prompt)
        self.assertNotIn("enflame-interface-assessment.md", prompt)
        self.assertNotIn("Gluon", prompt)
        self.assertNotIn("bottleneck-localization.md", prompt)

    def test_disabled_tle_policy_forbids_tle_work_in_the_prompt(self):
        task = helpers.base_task(extensions={"tle": "disabled"})
        prompt = kernelpilot.build_prompt(self.workspace, task)
        self.assertIn("TLE is disabled for this task", prompt)
        self.assertNotIn("TLE is auto for this task", prompt)
        self.assertNotIn("run `assess-tle` before planning", prompt)

    def test_enabled_tle_policy_asks_for_the_assessment_first(self):
        for policy in ("auto", "required"):
            task = helpers.base_task(extensions={"tle": policy})
            prompt = kernelpilot.build_prompt(self.workspace, task)
            self.assertIn(f"TLE is {policy} for this task", prompt)
            self.assertIn("`assess-tle` before planning", prompt)
            self.assertNotIn("TLE is disabled for this task", prompt)

    def test_disabled_screen_policy_still_asks_for_the_strategy_step(self):
        # The two policies are independent: turning the screen off must not turn
        # the strategy step off, and must not reintroduce the ignored record.
        prompt = self.gcu_prompt(bottleneck_screen="disabled")
        self.assertNotIn("bottleneck-screen.json", prompt)
        self.assertNotIn("bottleneck-localization.md", prompt)
        self.assertIn("enflame-interface-assessment.md", prompt)
        self.assertIn("Select the strategy before implementing", prompt)

    def test_unprobed_means_are_described_as_unavailable_not_as_detected(self):
        prompt = self.gcu_prompt()
        self.assertIn("nothing here probes or routes", prompt)
        self.assertIn("Gluon or inline assembly", prompt)
        self.assertIn("unavailable", prompt)
        # No sentence may imply a probe, a version check or skill routing exists.
        for claim in ("Gluon is available", "detected Gluon", "Gluon probe"):
            self.assertNotIn(claim, prompt)

    def test_prompt_keeps_the_evidence_and_checkpoint_wrappers(self):
        prompt = self.gcu_prompt()
        for wrapper in ("run correctness", "run benchmark", "checkpoint save-best"):
            self.assertIn(wrapper, prompt)
        self.assertIn("Modify only constraints.allowed_paths", prompt)
        self.assertIn("Never modify", prompt)


class DocumentTests(unittest.TestCase):
    def test_assessment_lists_the_five_gates_in_order(self):
        text = MEANS_DOC.read_text(encoding="utf-8")
        positions = [text.index(f"| {gate} ") for gate in GATES]
        self.assertEqual(positions, sorted(positions))

    def test_assessment_lists_the_eight_candidate_items(self):
        text = MEANS_DOC.read_text(encoding="utf-8")
        for index in range(1, 9):
            self.assertRegex(text, rf"\| {index} \| ")

    def test_assessment_states_that_nothing_reads_it(self):
        text = MEANS_DOC.read_text(encoding="utf-8")
        self.assertIn("Agent guidance only", text)
        self.assertIn("no gate reads it", text)
        self.assertIn("It is guidance for the agent", text)

    def test_assessment_separates_the_three_vocabularies(self):
        text = MEANS_DOC.read_text(encoding="utf-8")
        self.assertIn("must stay separate", text)
        for layer in kernelpilot.SCREEN_LAYER_IDS:
            self.assertIn(layer, text)
        for kind in kernelpilot.CHANGE_CLASSES:
            if kind == "none":
                continue
            self.assertIn(kind, text)

    def test_assessment_does_not_claim_a_device_result(self):
        text = MEANS_DOC.read_text(encoding="utf-8")
        self.assertIn("not supported, probed or routed", text)
        self.assertIn("No interface described here was validated on an S60 device", text)

    def test_every_relative_link_in_the_changed_documents_resolves(self):
        documents = [
            ROOT / "AGENTS.md",
            SOP_DOC,
            MEANS_DOC,
            ROOT / "docs/bottleneck-localization.md",
            ROOT / "docs/enflame-compiler-diagnostics.md",
            RUNBOOK,
            ROOT / "docs/evidence-format.md",
            ROOT / "prompts/README.md",
            ROOT / "templates/final-report.md",
            ROOT / "knowledge/enflame/README.md",
        ]
        missing = []
        for document in documents:
            for target in relative_links(document):
                if not (document.parent / target).exists():
                    missing.append(f"{document.relative_to(ROOT)} -> {target}")
        self.assertEqual(missing, [], missing)

    def test_prompt_documents_all_exist(self):
        for name in (
            "docs/agent-flow.md",
            "docs/bottleneck-localization.md",
            "docs/enflame-compiler-diagnostics.md",
            "docs/enflame-interface-assessment.md",
            "docs/enflame-s60-runbook.md",
            "docs/evidence-format.md",
            "knowledge/enflame/README.md",
            "prompts/optimize-kernel.md",
            "workflows/kernel-optimization.yaml",
            "scripts/kernelpilot.py",
            "scripts/compile_artifacts.py",
        ):
            self.assertTrue((ROOT / name).is_file(), name)

    def test_static_prompt_carries_the_steps_the_means_and_the_gates(self):
        text = STATIC_PROMPT.read_text(encoding="utf-8")
        self.assertIn("## The nine-step loop", text)
        self.assertIn("## Strategy selection and implementation means", text)
        self.assertIn("## Capability validation before a candidate depends on it", text)
        self.assertIn("enflame-interface-assessment.md", text)
        for gate in ("Documented semantics", "Local interface available",
                     "Target backend compiles", "Runs correctly",
                     "Real workload gains"):
            self.assertIn(gate, text)
        # TLE stays one means among several and keeps its three policies.
        for policy in ("disabled", "auto", "required"):
            self.assertIn(policy, text)

    def test_runbook_sections_are_numbered_in_order(self):
        text = RUNBOOK.read_text(encoding="utf-8")
        numbers = [
            int(match.group(1))
            for match in re.finditer(r"^## (\d+)\. ", text, flags=re.MULTILINE)
        ]
        self.assertEqual(numbers, list(range(1, len(numbers) + 1)))
        self.assertIn("## 5. Select the implementation means", text)
        self.assertIn("## 6. TLE", text)

    def test_final_report_template_asks_for_means_and_gates(self):
        text = (ROOT / "templates/final-report.md").read_text(encoding="utf-8")
        self.assertIn("## Strategy and interface assessment", text)
        self.assertIn("Capability gates reached", text)

    def test_workspace_template_has_a_strategy_step(self):
        text = (ROOT / "templates/operator-workspace/AGENTS.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Select the strategy:", text)
        self.assertIn("Validate that the chosen means works", text)
        self.assertIn("runs/bottleneck-screen.json", text)


class WorkflowDeclarationTests(unittest.TestCase):
    def setUp(self):
        self.text = (ROOT / "workflows/kernel-optimization.yaml").read_text(
            encoding="utf-8"
        )
        self.workflow = yaml.safe_load(self.text)
        self.stages = {stage["id"]: stage for stage in self.workflow["stages"]}

    def test_the_nine_steps_are_documented_in_the_workflow(self):
        lowered = self.text.lower()
        for step, phrase in enumerate(
            (
                "fix the task",
                "verify correctness",
                "establish the baseline",
                "locate the bottleneck",
                "select strategy",
                "validate, implement",
                "same-condition a/b",
                "keep or drop",
                "commit and accumulate",
            ),
            start=1,
        ):
            self.assertIn(f"step {step} {phrase}", lowered)

    def test_step_five_is_declared_unenforced(self):
        self.assertIn("Step 5 has no stage of its own", self.text)
        self.assertIn("no evidence gate reads them", self.text)

    def test_the_stage_skeleton_is_unchanged(self):
        # Comments were added, actions were not renamed and stages were not
        # inserted: a diagram of the loop must not become a new gate.
        self.assertEqual(len(self.workflow["stages"]), 36)
        self.assertEqual(self.stages["diagnose"]["action"], "diagnose_bottleneck")
        self.assertEqual(
            self.stages["select_optimization_strategy"]["action"],
            "select_available_optimization_extension_from_validated_assessment",
        )
        self.assertEqual(self.stages["optimize"]["action"], "implement_candidate")
        self.assertEqual(self.stages["report"]["terminal"], True)

    def test_the_means_document_is_declared_as_the_reference(self):
        self.assertIn("docs/enflame-interface-assessment.md", self.text)


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        helpers.prepare_workspace(self.workspace)

    def tearDown(self):
        self._tmp.cleanup()

    def test_task_schema_gained_no_required_field(self):
        with (ROOT / "schemas/task.schema.yaml").open(encoding="utf-8") as handle:
            schema = yaml.safe_load(handle)
        analysis = schema["properties"]["analysis"]
        self.assertEqual(analysis.get("required", []), [])
        # The means and the gates are recorded in docs/plan.md on purpose; a new
        # contract field would make every existing task invalid.
        self.assertNotIn("interface_assessment", analysis["properties"])
        self.assertNotIn("implementation_means", analysis["properties"])

    def test_existing_task_still_validates_and_prompts(self):
        task = helpers.base_task()
        self.assertEqual(kernelpilot.validate_task(self.workspace, task), [])
        prompt = kernelpilot.build_prompt(self.workspace, task)
        self.assertIn("kernel-optimization.yaml", prompt)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_all_normal_final_outcomes_review_knowledge_before_reporting():
    workflow = yaml.safe_load(
        (ROOT / "workflows/kernel-optimization.yaml").read_text(encoding="utf-8")
    )
    stages = {stage["id"]: stage for stage in workflow["stages"]}

    assert stages["promote"]["next"] == "knowledge_review"
    assert stages["finalize_without_target"]["next"] == "knowledge_review"
    assert stages["workflow_error"]["next"] == "knowledge_review"

    review = stages["knowledge_review"]
    assert review["action"] == "publish_reusable_knowledge_if_eligible_or_record_skip_reason"
    assert review["on_published"] == "report"
    assert review["on_not_eligible"] == "report"
    assert review["on_publish_error"] == "report"


def test_codex_optimization_prompt_requires_evidence_gated_publication():
    prompt = (ROOT / "prompts/optimize-kernel.md").read_text(encoding="utf-8")
    assert "Default reusable-knowledge publication" in prompt
    assert "scripts/record_knowledge.py" in prompt
    assert "keep" in prompt
    assert "at least the recorder's default 5%" in prompt
    assert "skip result" in prompt


def test_structured_result_requires_knowledge_publication_outcome():
    schema = yaml.safe_load(
        (ROOT / "schemas/run-result.schema.json").read_text(encoding="utf-8")
    )
    assert "knowledge_publication" in schema["required"]
    publication = schema["properties"]["knowledge_publication"]
    assert publication["required"] == ["status", "notes", "reason"]
    assert publication["properties"]["status"]["enum"] == [
        "published", "skipped", "error"
    ]

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from deerflow.config.skill_evolution_config import SkillEvolutionConfig
from deerflow.skill_evolution.eligibility import (
    evaluate_evolution_eligibility,
)
from deerflow.skill_evolution.extractor import (
    DeterministicExtraction,
    pre_extract_evolution,
)
from deerflow.skill_evolution.models import (
    EvolutionEventKind,
    EvolutionTraceSnapshot,
    OutcomeEvidence,
    SkillGapCategory,
    TraceSkillEvent,
)
from deerflow.skill_evolution.store.memory import (
    InMemorySkillEvolutionStore,
)
from deerflow.skill_evolution.structured_extractor import (
    STRUCTURED_EXTRACTION_PROMPT_VERSION,
    StructuredEvolutionExtractor,
    StructuredExtractionOutput,
    build_structured_extraction_messages,
)

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "skill_evolution" / "recovered_package_run.json"


class SequenceModel:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[Any] = []

    async def ainvoke(
        self,
        messages,
        config=None,
    ):
        self.requests.append((messages, config))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, (dict, list)):
            response = json.dumps(response)
        if isinstance(response, str):
            return AIMessage(content=response)
        return response


def _pre_extraction(
    *,
    with_skill: bool = False,
) -> DeterministicExtraction:
    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    snapshot = EvolutionTraceSnapshot.model_validate(payload["snapshot"])
    outcome = OutcomeEvidence.model_validate(payload["outcome"])
    if with_skill:
        snapshot = snapshot.model_copy(
            update={
                "skill_events": [
                    TraceSkillEvent(
                        skill_name="package-repair",
                        skill_path=("/mnt/skills/custom/package-repair/SKILL.md"),
                        content_hash="b" * 64,
                        activation_source="read",
                        category="custom",
                    )
                ]
            }
        )
    eligibility = evaluate_evolution_eligibility(
        snapshot,
        outcome,
    )
    return pre_extract_evolution(
        snapshot,
        outcome,
        eligibility,
    )


def _new_skill_output() -> dict[str, Any]:
    return {
        "task_signature": "python-package-install",
        "task_goal": ("Install a Python package in the project environment and verify the import."),
        "task_goal_evidence_segment_ids": ["task:input"],
        "successful_path": [
            {
                "step": "Inspect the project environment.",
                "evidence_segment_ids": ["tool:call-inspect"],
            },
            {
                "step": "Install with the project package manager.",
                "evidence_segment_ids": ["tool:call-local-install"],
            },
            {
                "step": "Run the focused tests.",
                "evidence_segment_ids": ["tool:call-tests"],
            },
        ],
        "failed_attempts": [
            {
                "action": "Install into the global interpreter.",
                "error": "Permission denied.",
                "lesson": "Use the project environment.",
                "evidence_segment_ids": ["tool:call-global-install"],
            }
        ],
        "user_corrections": [
            {
                "correction": ("Use the project environment instead."),
                "effective_change": ("The package was installed with uv."),
                "evidence_segment_ids": ["correction:0"],
            }
        ],
        "reusable_lessons": [
            {
                "lesson": ("Inspect the active project environment before installing dependencies."),
                "evidence_segment_ids": [
                    "tool:call-inspect",
                    "tool:call-local-install",
                ],
            }
        ],
        "skill_gaps": [],
        "candidate_target": None,
        "candidate_target_evidence_segment_ids": [],
    }


def _patch_output() -> dict[str, Any]:
    output = _new_skill_output()
    output["skill_gaps"] = [
        {
            "category": (SkillGapCategory.missing_prerequisite.value),
            "evidence": ("The original workflow did not inspect the active project environment."),
            "recommended_change": ("Check the project package manager before install."),
            "evidence_segment_ids": [
                "tool:call-global-install",
                "tool:call-inspect",
            ],
        }
    ]
    output["candidate_target"] = {
        "name": "package-repair",
        "content_hash": "b" * 64,
    }
    output["candidate_target_evidence_segment_ids"] = ["tool:call-inspect"]
    return output


def test_structured_output_schema_is_strict() -> None:
    schema = StructuredExtractionOutput.model_json_schema()

    assert schema["additionalProperties"] is False
    assert {
        "task_signature",
        "task_goal",
        "successful_path",
        "failed_attempts",
        "user_corrections",
        "reusable_lessons",
        "skill_gaps",
        "candidate_target",
    }.issubset(schema["required"])

    with pytest.raises(Exception):
        StructuredExtractionOutput.model_validate(
            {
                **_new_skill_output(),
                "unexpected": "not allowed",
            }
        )


@pytest.mark.anyio
async def test_valid_new_skill_output_builds_and_persists_event() -> None:
    pre_extraction = _pre_extraction()
    model = SequenceModel([_new_skill_output()])
    store = InMemorySkillEvolutionStore()
    extractor = StructuredEvolutionExtractor(
        model=model,
        model_name="qwen3-local",
        max_attempts=2,
        retry_delay_seconds=0,
    )

    result = await extractor.extract_and_persist(
        pre_extraction,
        store,
    )

    assert result is not None
    assert result.created is True
    event = result.value
    assert event.event_kind is EvolutionEventKind.new_skill_evidence
    assert event.skill_usage.used is False
    assert event.task_signature == "python-package-install"
    assert event.reusable_lessons == [("Inspect the active project environment before installing dependencies.")]
    assert event.extractor_model_name == "qwen3-local"
    assert event.extractor_prompt_version == STRUCTURED_EXTRACTION_PROMPT_VERSION
    assert event.source_extraction_hash == pre_extraction.extraction_hash
    assert event.source_snapshot_hash == pre_extraction.source_snapshot_hash
    assert len(event.provenance) >= 5
    lesson_link = next(link for link in event.evidence_links if link.semantic_field == "reusable_lesson")
    assert lesson_link.item_index == 0
    assert lesson_link.evidence_segment_ids == [
        "tool:call-inspect",
        "tool:call-local-install",
    ]
    assert set(lesson_link.evidence_segment_ids).issubset({reference.source for reference in event.provenance})
    assert len(model.requests) == 1
    assert await store.list_events("user-1") == [event]


@pytest.mark.anyio
async def test_new_skill_output_discards_patch_only_fields() -> None:
    output = _new_skill_output()
    output["candidate_target"] = {
        "name": "invented-target",
        "content_hash": "none",
    }
    output["candidate_target_evidence_segment_ids"] = ["tool:call-inspect"]
    output["skill_gaps"] = [
        {
            "category": SkillGapCategory.missing_prerequisite.value,
            "evidence": "A target Skill should be updated.",
            "recommended_change": "Patch the target Skill.",
            "evidence_segment_ids": ["tool:call-inspect"],
        }
    ]
    extractor = StructuredEvolutionExtractor(
        model=SequenceModel([output]),
        model_name="qwen3-local",
        max_attempts=1,
        retry_delay_seconds=0,
    )

    event = await extractor.extract(_pre_extraction())

    assert event is not None
    assert event.event_kind is EvolutionEventKind.new_skill_evidence
    assert event.target_skill is None
    assert event.skill_gaps == []


@pytest.mark.anyio
async def test_optional_claims_without_typed_evidence_are_discarded() -> None:
    output = _new_skill_output()
    output["failed_attempts"][0]["evidence_segment_ids"] = ["tool:call-inspect"]
    output["user_corrections"][0]["evidence_segment_ids"] = ["tool:call-inspect"]
    extractor = StructuredEvolutionExtractor(
        model=SequenceModel([output]),
        model_name="qwen3-local",
        max_attempts=1,
        retry_delay_seconds=0,
    )

    event = await extractor.extract(_pre_extraction())

    assert event is not None
    assert event.failed_attempts == []
    assert event.user_corrections == []


@pytest.mark.anyio
async def test_valid_patch_output_selects_observed_skill() -> None:
    pre_extraction = _pre_extraction(with_skill=True)
    model = SequenceModel([_patch_output()])
    extractor = StructuredEvolutionExtractor(
        model=model,
        model_name="qwen3-local",
        retry_delay_seconds=0,
    )

    event = await extractor.extract(pre_extraction)

    assert event is not None
    assert event.event_kind is EvolutionEventKind.skill_patch_evidence
    assert event.skill_usage.used is True
    assert event.skill_usage.skill_name == "package-repair"
    assert event.target_skill is not None
    assert event.target_skill.name == "package-repair"
    assert event.skill_gaps[0].category is (SkillGapCategory.missing_prerequisite)


@pytest.mark.anyio
async def test_unknown_provenance_retries_then_succeeds() -> None:
    malformed = _new_skill_output()
    malformed["reusable_lessons"][0]["evidence_segment_ids"] = ["tool:not-present"]
    model = SequenceModel([malformed, _new_skill_output()])
    extractor = StructuredEvolutionExtractor(
        model=model,
        model_name="qwen3-local",
        max_attempts=2,
        retry_delay_seconds=0,
    )

    event = await extractor.extract(_pre_extraction())

    assert event is not None
    assert len(model.requests) == 2


@pytest.mark.anyio
async def test_repeated_malformed_output_fails_closed_without_store_write() -> None:
    model = SequenceModel(["not json", "still not json"])
    store = InMemorySkillEvolutionStore()
    extractor = StructuredEvolutionExtractor(
        model=model,
        model_name="qwen3-local",
        max_attempts=2,
        retry_delay_seconds=0,
    )

    result = await extractor.extract_and_persist(
        _pre_extraction(),
        store,
    )

    assert result is None
    assert len(model.requests) == 2
    assert await store.list_events("user-1") == []


@pytest.mark.anyio
async def test_transient_model_error_is_retried() -> None:
    model = SequenceModel([TimeoutError("provider timeout"), _new_skill_output()])
    extractor = StructuredEvolutionExtractor(
        model=model,
        model_name="qwen3-local",
        max_attempts=2,
        retry_delay_seconds=0,
    )

    event = await extractor.extract(_pre_extraction())

    assert event is not None
    assert len(model.requests) == 2


@pytest.mark.anyio
async def test_non_transient_model_error_is_not_retried() -> None:
    model = SequenceModel([RuntimeError("invalid model configuration")])
    extractor = StructuredEvolutionExtractor(
        model=model,
        model_name="qwen3-local",
        max_attempts=3,
        retry_delay_seconds=0,
    )

    event = await extractor.extract(_pre_extraction())

    assert event is None
    assert len(model.requests) == 1


@pytest.mark.anyio
async def test_markdown_fenced_json_is_rejected() -> None:
    response = "```json\n" + json.dumps(_new_skill_output()) + "\n```"
    extractor = StructuredEvolutionExtractor(
        model=SequenceModel([response]),
        model_name="qwen3-local",
        max_attempts=1,
        retry_delay_seconds=0,
    )

    assert await extractor.extract(_pre_extraction()) is None


@pytest.mark.anyio
async def test_patch_target_must_match_observed_skill() -> None:
    output = _patch_output()
    output["candidate_target"]["name"] = "other-skill"
    model = SequenceModel([output])
    extractor = StructuredEvolutionExtractor(
        model=model,
        model_name="qwen3-local",
        max_attempts=1,
        retry_delay_seconds=0,
    )

    assert await extractor.extract(_pre_extraction(with_skill=True)) is None


def test_prompt_contains_schema_and_only_bounded_pre_extraction() -> None:
    pre_extraction = _pre_extraction()

    messages = build_structured_extraction_messages(pre_extraction)
    serialized = "\n".join(str(message.content) for message in messages)

    assert "additionalProperties" in serialized
    assert pre_extraction.extraction_hash in serialized
    assert "tool:call-global-install" in serialized
    assert "Permission denied" in serialized
    assert "data:image" not in serialized


def test_config_exposes_optional_extraction_model() -> None:
    config = SkillEvolutionConfig(extraction_model_name="qwen3-local")

    assert config.extraction_model_name == "qwen3-local"

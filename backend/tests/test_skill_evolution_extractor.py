from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from deerflow.skill_evolution.eligibility import (
    evaluate_evolution_eligibility,
)
from deerflow.skill_evolution.extractor import (
    CandidateSegmentKind,
    DeterministicExtraction,
    IneligibleTraceError,
    pre_extract_evolution,
)
from deerflow.skill_evolution.models import (
    EvolutionEventKind,
    EvolutionTraceSnapshot,
    OutcomeEvidence,
    TraceSkillEvent,
    TraceToolEvent,
)

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "skill_evolution" / "recovered_package_run.json"


def _fixture() -> tuple[EvolutionTraceSnapshot, OutcomeEvidence]:
    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    return (
        EvolutionTraceSnapshot.model_validate(payload["snapshot"]),
        OutcomeEvidence.model_validate(payload["outcome"]),
    )


def _extract(
    snapshot: EvolutionTraceSnapshot,
    outcome: OutcomeEvidence,
    **kwargs,
) -> DeterministicExtraction:
    eligibility = evaluate_evolution_eligibility(
        snapshot,
        outcome,
    )
    return pre_extract_evolution(
        snapshot,
        outcome,
        eligibility,
        **kwargs,
    )


def _tool(
    sequence: int,
    *,
    status: str = "success",
    result: str = "ok",
    error_type: str | None = None,
) -> TraceToolEvent:
    return TraceToolEvent(
        sequence=sequence,
        tool_call_id=f"call-{sequence}",
        tool_name="bash",
        arguments=(f'{{"command":"step-{sequence}"}}'),
        result=result,
        status=status,
        error_type=error_type,
    )


def test_fixture_pre_extraction_is_deterministic_and_round_trips() -> None:
    snapshot, outcome = _fixture()

    first = _extract(snapshot, outcome)
    second = _extract(snapshot, outcome)
    restored = DeterministicExtraction.model_validate_json(first.model_dump_json())

    assert first == second
    assert restored == first
    assert first.extraction_hash == second.extraction_hash
    assert first.task_input_hash == hashlib.sha256(snapshot.task_input.encode("utf-8")).hexdigest()

    changed = first.model_dump(mode="json")
    changed["artifact_paths"].append("/mnt/user-data/outputs/untracked.json")
    with pytest.raises(ValidationError, match="extraction_hash"):
        DeterministicExtraction.model_validate(changed)


def test_fixture_derives_deterministic_fields_without_llm() -> None:
    snapshot, outcome = _fixture()

    extraction = _extract(snapshot, outcome)

    assert extraction.event_kind is EvolutionEventKind.new_skill_evidence
    assert extraction.environment == snapshot.environment
    assert extraction.outcome == outcome
    assert extraction.complexity.tool_calls == 6
    assert extraction.complexity.had_recoverable_errors is True
    assert extraction.complexity.had_user_correction is True
    assert extraction.tool_signature.tool_names == [
        "bash",
        "read_file",
        "bash",
        "write_file",
        "bash",
        "bash",
    ]
    assert extraction.tool_signature.error_types == ["permission"]
    assert [(entry.sequence, entry.tool_name, entry.status) for entry in extraction.tool_sequence] == [
        (1, "bash", "error"),
        (2, "read_file", "success"),
        (3, "bash", "success"),
        (4, "write_file", "success"),
        (5, "bash", "success"),
        (6, "bash", "success"),
    ]
    assert all(len(entry.arguments_hash) == 64 and len(entry.result_hash) == 64 for entry in extraction.tool_sequence)
    assert extraction.observed_skill_usages == []
    assert extraction.artifact_paths == snapshot.artifacts
    assert {segment.kind for segment in extraction.candidate_segments} == {
        CandidateSegmentKind.task,
        CandidateSegmentKind.tool,
        CandidateSegmentKind.correction,
        CandidateSegmentKind.artifact,
        CandidateSegmentKind.final_answer,
    }


def test_skill_usage_selects_patch_evidence_and_preserves_identity() -> None:
    snapshot, outcome = _fixture()
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

    extraction = _extract(snapshot, outcome)

    assert extraction.event_kind is EvolutionEventKind.skill_patch_evidence
    assert len(extraction.observed_skill_usages) == 1
    usage = extraction.observed_skill_usages[0]
    assert usage.used is True
    assert usage.skill_name == "package-repair"
    assert usage.content_hash == "b" * 64
    assert usage.activation_source == "read"


def test_ineligible_trace_is_rejected_before_extraction() -> None:
    snapshot, outcome = _fixture()
    snapshot = snapshot.model_copy(
        update={
            "tool_events": snapshot.tool_events[:1],
            "user_corrections": [],
        }
    )
    eligibility = evaluate_evolution_eligibility(
        snapshot,
        outcome,
    )
    assert eligibility.eligible is False

    with pytest.raises(
        IneligibleTraceError,
        match="not eligible",
    ):
        pre_extract_evolution(
            snapshot,
            outcome,
            eligibility,
        )


def test_outcome_must_match_eligibility_decision() -> None:
    snapshot, outcome = _fixture()
    eligibility = evaluate_evolution_eligibility(
        snapshot,
        outcome,
    )
    changed_outcome = outcome.model_copy(update={"confidence": 0.9})

    with pytest.raises(ValueError, match="outcome"):
        pre_extract_evolution(
            snapshot,
            changed_outcome,
            eligibility,
        )


def test_tool_segments_are_bounded_while_errors_and_tail_survive() -> None:
    snapshot, outcome = _fixture()
    tools = [
        _tool(
            index,
            status="error" if index == 2 else "success",
            error_type="permission" if index == 2 else None,
        )
        for index in range(1, 51)
    ]
    snapshot = snapshot.model_copy(update={"tool_events": tools})

    extraction = _extract(
        snapshot,
        outcome,
        max_tool_segments=8,
    )
    tool_segments = [segment for segment in extraction.candidate_segments if segment.kind is CandidateSegmentKind.tool]

    assert extraction.source_tool_count == 50
    assert extraction.included_tool_count == 8
    assert extraction.segments_truncated is True
    assert len(tool_segments) == 8
    assert any(segment.sequence == 2 for segment in tool_segments)
    assert any(segment.sequence == 50 for segment in tool_segments)


def test_probable_binary_or_base64_tool_content_is_omitted() -> None:
    snapshot, outcome = _fixture()
    encoded = "data:image/png;base64," + ("A" * 5_000)
    tools = list(snapshot.tool_events)
    tools[1] = tools[1].model_copy(update={"result": encoded})
    tools[3] = tools[3].model_copy(
        update={
            "arguments": json.dumps(
                {
                    "path": "/mnt/user-data/outputs/image.txt",
                    "content": encoded,
                }
            )
        }
    )
    snapshot = snapshot.model_copy(update={"tool_events": tools})

    extraction = _extract(snapshot, outcome)
    segment = next(item for item in extraction.candidate_segments if item.segment_id == "tool:call-inspect")

    assert segment.omitted is True
    assert segment.truncated is True
    assert "base64" not in segment.content.lower()
    assert encoded not in extraction.model_dump_json()
    assert len(segment.content_hash) == 64
    argument_segment = next(item for item in extraction.candidate_segments if item.segment_id == "tool:call-write")
    assert argument_segment.omitted is True
    assert "base64" not in argument_segment.content.lower()


def test_large_plain_text_is_bounded_and_hashed() -> None:
    snapshot, outcome = _fixture()
    large_result = "validation line\n" * 2_000
    tools = list(snapshot.tool_events)
    tools[2] = tools[2].model_copy(update={"result": large_result})
    snapshot = snapshot.model_copy(update={"tool_events": tools})

    extraction = _extract(snapshot, outcome)
    segment = next(item for item in extraction.candidate_segments if item.segment_id == "tool:call-local-install")

    assert segment.truncated is True
    assert segment.omitted is False
    assert len(segment.content) <= 2_000
    assert large_result not in extraction.model_dump_json()


def test_pre_extraction_keeps_redaction_and_never_embeds_artifacts() -> None:
    snapshot, outcome = _fixture()

    extraction = _extract(snapshot, outcome)
    serialized = extraction.model_dump_json()

    assert "[redacted]" in serialized
    assert "supersecret" not in serialized
    for path in snapshot.artifacts:
        artifact_segment = next(item for item in extraction.candidate_segments if item.kind is CandidateSegmentKind.artifact and item.content == path)
        assert artifact_segment.omitted is False
        assert artifact_segment.content_hash == hashlib.sha256(path.encode("utf-8")).hexdigest()

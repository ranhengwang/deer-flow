from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from deerflow.config.skill_evolution_config import (
    SkillEvolutionEvidenceConfig,
)
from deerflow.skill_evolution.eligibility import (
    EligibilityBranch,
    EligibilityHints,
    evaluate_evolution_eligibility,
)
from deerflow.skill_evolution.models import (
    EnvironmentSignature,
    EvolutionTraceSnapshot,
    OutcomeEvidence,
    OutcomeStatus,
    TraceRunStatus,
    TraceSkillEvent,
    TraceToolEvent,
)


def _outcome(
    status: OutcomeStatus = OutcomeStatus.success,
    confidence: float = 0.9,
) -> OutcomeEvidence:
    return OutcomeEvidence(
        status=status,
        confidence=confidence,
        sources=["deterministic-verifier"],
    )


def _tool(
    sequence: int,
    *,
    status: str = "success",
    name: str = "bash",
) -> TraceToolEvent:
    return TraceToolEvent(
        sequence=sequence,
        tool_call_id=f"call-{sequence}",
        tool_name=name,
        arguments="{}",
        result="ok" if status == "success" else "Error: failed",
        status=status,
        error_type="transient" if status == "error" else None,
    )


def _snapshot(
    *,
    tools: list[TraceToolEvent] | None = None,
    task_input: str = "Complete the task.",
    corrections: list[str] | None = None,
    skills: list[TraceSkillEvent] | None = None,
    stop_reason: str | None = None,
) -> EvolutionTraceSnapshot:
    return EvolutionTraceSnapshot(
        snapshot_hash="a" * 64,
        run_id="run-1",
        thread_id="thread-1",
        user_id="user-1",
        model_name="test-model",
        run_status=TraceRunStatus.success,
        stop_reason=stop_reason,
        task_input=task_input,
        final_answer="Done.",
        tool_events=tools or [],
        skill_events=skills or [],
        user_corrections=corrections or [],
        artifacts=[],
        environment=EnvironmentSignature(
            os="macos",
            shell="zsh",
            runtime="python3.12",
        ),
        source_event_count=8,
        included_event_count=8,
        truncated=False,
        created_at=datetime(2026, 8, 16, tzinfo=UTC),
    )


def test_evidence_config_defaults_and_validation() -> None:
    config = SkillEvolutionEvidenceConfig()

    assert config.min_success_confidence == 0.8
    assert config.tool_call_complexity_threshold == 5
    assert config.accept_recovered_errors is True
    assert config.accept_user_corrections is True
    assert config.accept_explicit_remember_requests is True
    assert config.accept_non_trivial_workflow is True

    with pytest.raises(ValidationError):
        SkillEvolutionEvidenceConfig(min_success_confidence=1.1)
    with pytest.raises(ValidationError):
        SkillEvolutionEvidenceConfig(
            tool_call_complexity_threshold=0,
        )


def test_tool_call_threshold_admits_high_confidence_success() -> None:
    decision = evaluate_evolution_eligibility(
        _snapshot(tools=[_tool(index) for index in range(1, 6)]),
        _outcome(confidence=0.8),
    )

    assert decision.eligible is True
    assert decision.branch is EligibilityBranch.no_skill
    assert decision.complexity.tool_calls == 5
    assert decision.qualifying_signals == ["tool_call_threshold"]
    assert decision.rejection_reasons == []


@pytest.mark.parametrize(
    "status",
    [OutcomeStatus.failure, OutcomeStatus.unknown],
)
def test_failed_or_unknown_outcome_is_rejected(
    status: OutcomeStatus,
) -> None:
    decision = evaluate_evolution_eligibility(
        _snapshot(tools=[_tool(index) for index in range(1, 7)]),
        _outcome(status=status, confidence=0.99),
    )

    assert decision.eligible is False
    assert "outcome_not_success" in decision.rejection_reasons


def test_success_below_configured_confidence_is_rejected() -> None:
    decision = evaluate_evolution_eligibility(
        _snapshot(tools=[_tool(index) for index in range(1, 7)]),
        _outcome(confidence=0.79),
    )

    assert decision.eligible is False
    assert decision.rejection_reasons == [
        "confidence_below_threshold",
    ]


def test_simple_success_is_rejected() -> None:
    decision = evaluate_evolution_eligibility(
        _snapshot(tools=[_tool(1), _tool(2)]),
        _outcome(),
    )

    assert decision.eligible is False
    assert decision.rejection_reasons == [
        "insufficient_complexity",
    ]


def test_error_followed_by_success_is_recovered_complexity() -> None:
    recovered = evaluate_evolution_eligibility(
        _snapshot(
            tools=[
                _tool(1, status="error"),
                _tool(2, status="success", name="read_file"),
            ]
        ),
        _outcome(),
    )
    unrecovered = evaluate_evolution_eligibility(
        _snapshot(
            tools=[
                _tool(1, status="success"),
                _tool(2, status="error"),
            ]
        ),
        _outcome(),
    )

    assert recovered.eligible is True
    assert recovered.complexity.had_recoverable_errors is True
    assert recovered.qualifying_signals == ["recovered_error"]
    assert unrecovered.eligible is False
    assert unrecovered.complexity.had_recoverable_errors is False


def test_recovered_error_signal_can_be_disabled() -> None:
    decision = evaluate_evolution_eligibility(
        _snapshot(
            tools=[
                _tool(1, status="error"),
                _tool(2, status="success"),
            ]
        ),
        _outcome(),
        config=SkillEvolutionEvidenceConfig(
            accept_recovered_errors=False,
        ),
    )

    assert decision.eligible is False
    assert decision.complexity.had_recoverable_errors is True
    assert decision.rejection_reasons == [
        "insufficient_complexity",
    ]


@pytest.mark.parametrize(
    "task_input",
    [
        "Remember this workflow for future repository repairs.",
        "Create a skill for this package repair process.",
        "请记住这个处理流程，以后继续使用。",
        "把这个方法沉淀成一个 skill。",
    ],
)
def test_explicit_remember_or_create_skill_request_bypasses_tool_threshold(
    task_input: str,
) -> None:
    decision = evaluate_evolution_eligibility(
        _snapshot(task_input=task_input),
        _outcome(),
    )

    assert decision.eligible is True
    assert decision.complexity.explicit_remember_request is True
    assert decision.qualifying_signals == [
        "explicit_remember_request",
    ]


@pytest.mark.parametrize(
    "task_input",
    [
        "Do not create a skill for this one-off task.",
        "不要记住这个流程，也不要创建 skill。",
    ],
)
def test_negated_skill_request_is_not_a_complexity_signal(
    task_input: str,
) -> None:
    decision = evaluate_evolution_eligibility(
        _snapshot(task_input=task_input),
        _outcome(),
    )

    assert decision.eligible is False
    assert decision.complexity.explicit_remember_request is False


def test_user_correction_from_snapshot_or_extractor_is_accepted() -> None:
    snapshot_signal = evaluate_evolution_eligibility(
        _snapshot(
            corrections=["Use the project environment instead."],
        ),
        _outcome(),
    )
    extracted_signal = evaluate_evolution_eligibility(
        _snapshot(),
        _outcome(),
        hints=EligibilityHints(had_user_correction=True),
    )

    assert snapshot_signal.eligible is True
    assert snapshot_signal.complexity.had_user_correction is True
    assert extracted_signal.eligible is True
    assert extracted_signal.complexity.had_user_correction is True


def test_user_correction_signal_can_be_disabled() -> None:
    decision = evaluate_evolution_eligibility(
        _snapshot(
            corrections=["Use the project environment instead."],
        ),
        _outcome(),
        config=SkillEvolutionEvidenceConfig(
            accept_user_corrections=False,
        ),
    )

    assert decision.eligible is False
    assert decision.complexity.had_user_correction is True
    assert decision.rejection_reasons == [
        "insufficient_complexity",
    ]


def test_explicit_remember_signal_can_be_disabled() -> None:
    decision = evaluate_evolution_eligibility(
        _snapshot(
            task_input="Create a skill for this workflow.",
        ),
        _outcome(),
        config=SkillEvolutionEvidenceConfig(
            accept_explicit_remember_requests=False,
        ),
    )

    assert decision.eligible is False
    assert decision.complexity.explicit_remember_request is True
    assert decision.rejection_reasons == [
        "insufficient_complexity",
    ]


def test_non_trivial_workflow_hint_is_accepted_and_configurable() -> None:
    hints = EligibilityHints(non_trivial_workflow=True)
    accepted = evaluate_evolution_eligibility(
        _snapshot(),
        _outcome(),
        hints=hints,
    )
    rejected = evaluate_evolution_eligibility(
        _snapshot(),
        _outcome(),
        hints=hints,
        config=SkillEvolutionEvidenceConfig(
            accept_non_trivial_workflow=False,
        ),
    )

    assert accepted.eligible is True
    assert accepted.qualifying_signals == [
        "non_trivial_workflow",
    ]
    assert rejected.eligible is False


def test_used_skill_selects_patch_branch() -> None:
    decision = evaluate_evolution_eligibility(
        _snapshot(
            tools=[_tool(index) for index in range(1, 6)],
            skills=[
                TraceSkillEvent(
                    skill_name="repo-repair",
                    skill_path=("/mnt/skills/custom/repo-repair/SKILL.md"),
                    content_hash="b" * 64,
                    activation_source="read",
                    category="custom",
                )
            ],
        ),
        _outcome(),
    )

    assert decision.eligible is True
    assert decision.branch is EligibilityBranch.skill_used


@pytest.mark.parametrize(
    "stop_reason",
    [
        "safety_capped",
        "loop_capped",
        "token_capped",
        "subagent_limit_capped",
    ],
)
def test_capped_run_is_rejected_by_default(
    stop_reason: str,
) -> None:
    decision = evaluate_evolution_eligibility(
        _snapshot(
            tools=[_tool(index) for index in range(1, 7)],
            stop_reason=stop_reason,
        ),
        _outcome(),
    )

    assert decision.eligible is False
    assert decision.excluded_stop_reason == stop_reason
    assert "excluded_stop_reason" in decision.rejection_reasons


def test_custom_tool_threshold_is_consumed() -> None:
    decision = evaluate_evolution_eligibility(
        _snapshot(tools=[_tool(1), _tool(2), _tool(3)]),
        _outcome(),
        config=SkillEvolutionEvidenceConfig(
            tool_call_complexity_threshold=3,
        ),
    )

    assert decision.eligible is True
    assert decision.qualifying_signals == ["tool_call_threshold"]


def test_explicit_request_does_not_bypass_success_requirement() -> None:
    decision = evaluate_evolution_eligibility(
        _snapshot(task_input="Create a skill for this workflow."),
        _outcome(
            status=OutcomeStatus.failure,
            confidence=0.99,
        ),
    )

    assert decision.eligible is False
    assert decision.complexity.explicit_remember_request is True
    assert decision.rejection_reasons == ["outcome_not_success"]

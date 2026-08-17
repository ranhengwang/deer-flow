from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from deerflow.skill_evolution.models import (
    EVOLUTION_EVENT_SCHEMA_VERSION,
    ClusterStatus,
    ComplexitySignals,
    EnvironmentSignature,
    EvaluationDecision,
    EvaluationMetrics,
    EvolutionCluster,
    EvolutionEvent,
    EvolutionEventKind,
    FailedAttempt,
    GroupingEvidence,
    OutcomeCheck,
    OutcomeEvidence,
    OutcomeStatus,
    ProposalOperation,
    ProposalStatus,
    ProposedSkillFile,
    SkillEvaluation,
    SkillGap,
    SkillGapCategory,
    SkillProposal,
    SkillTarget,
    SkillUsage,
    TaskEvaluationResult,
    ToolSignature,
)


def _success_outcome() -> OutcomeEvidence:
    return OutcomeEvidence(
        status=OutcomeStatus.success,
        confidence=0.95,
        sources=["focused-tests"],
        checks=[
            OutcomeCheck(
                source="pytest",
                passed=True,
                detail="12 focused tests passed",
            )
        ],
    )


def _no_skill_event(**overrides) -> EvolutionEvent:
    values = {
        "event_id": "event-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_id": "user-1",
        "extractor_version": "extractor-v1",
        "source_snapshot_hash": "d" * 64,
        "task_input_hash": "e" * 64,
        "event_kind": EvolutionEventKind.new_skill_evidence,
        "task_signature": "python-package-install",
        "task_goal": "Install a dependency and verify that it imports.",
        "environment": EnvironmentSignature(
            os="macos",
            shell="zsh",
            runtime="python3.12",
        ),
        "outcome": _success_outcome(),
        "complexity": ComplexitySignals(
            tool_calls=7,
            had_recoverable_errors=True,
        ),
        "tool_signature": ToolSignature(
            tool_names=["bash", "bash", "read_file"],
            error_types=["permission"],
        ),
        "skill_usage": SkillUsage(used=False),
        "successful_path": [
            "Inspect the active Python environment.",
            "Install into the project environment.",
            "Verify the package import.",
        ],
        "failed_attempts": [
            FailedAttempt(
                action="Install into the global interpreter.",
                error="Permission denied.",
                lesson="Use the project virtual environment.",
            )
        ],
        "reusable_lessons": ["Detect the active environment before installing dependencies."],
        "created_at": datetime(2026, 8, 16, tzinfo=UTC),
    }
    values.update(overrides)
    return EvolutionEvent(**values)


def test_event_round_trips_through_json() -> None:
    event = _no_skill_event()

    restored = EvolutionEvent.model_validate_json(event.model_dump_json())

    assert restored == event
    assert restored.schema_version == EVOLUTION_EVENT_SCHEMA_VERSION
    assert restored.event_kind is EvolutionEventKind.new_skill_evidence


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_outcome_confidence_must_be_between_zero_and_one(
    confidence: float,
) -> None:
    with pytest.raises(ValidationError):
        OutcomeEvidence(
            status=OutcomeStatus.success,
            confidence=confidence,
            sources=["tests"],
        )


def test_success_outcome_requires_evidence_source() -> None:
    with pytest.raises(ValidationError, match="evidence source"):
        OutcomeEvidence(
            status=OutcomeStatus.success,
            confidence=0.9,
            sources=[],
        )


def test_structured_extractor_metadata_must_be_complete() -> None:
    with pytest.raises(
        ValidationError,
        match="extractor metadata",
    ):
        _no_skill_event(
            extractor_model_name="qwen3-local",
        )


def test_new_skill_event_rejects_skill_usage() -> None:
    with pytest.raises(ValidationError, match="must not use a skill"):
        _no_skill_event(
            skill_usage=SkillUsage(
                used=True,
                skill_name="existing-skill",
                skill_path="/mnt/skills/custom/existing-skill/SKILL.md",
                content_hash="a" * 64,
            )
        )


def test_patch_event_requires_used_skill_and_matching_target() -> None:
    with pytest.raises(ValidationError, match="requires a used skill"):
        _no_skill_event(
            event_kind=EvolutionEventKind.skill_patch_evidence,
            target_skill=SkillTarget(
                name="python-package-manager",
                content_hash="b" * 64,
            ),
        )

    with pytest.raises(ValidationError, match="must match"):
        _no_skill_event(
            event_kind=EvolutionEventKind.skill_patch_evidence,
            skill_usage=SkillUsage(
                used=True,
                skill_name="python-package-manager",
                skill_path="/mnt/skills/custom/python-package-manager/SKILL.md",
                content_hash="b" * 64,
            ),
            target_skill=SkillTarget(
                name="other-skill",
                content_hash="c" * 64,
            ),
            skill_gaps=[
                SkillGap(
                    category=SkillGapCategory.missing_prerequisite,
                    evidence="The package manager was unavailable.",
                    recommended_change="Check the package manager first.",
                )
            ],
        )


def test_patch_event_requires_at_least_one_skill_gap() -> None:
    with pytest.raises(ValidationError, match="skill gap"):
        _no_skill_event(
            event_kind=EvolutionEventKind.skill_patch_evidence,
            skill_usage=SkillUsage(
                used=True,
                skill_name="python-package-manager",
                skill_path="/mnt/skills/custom/python-package-manager/SKILL.md",
                content_hash="b" * 64,
            ),
            target_skill=SkillTarget(
                name="python-package-manager",
                content_hash="b" * 64,
            ),
        )


def test_event_rejects_oversized_text_and_collections() -> None:
    with pytest.raises(ValidationError):
        _no_skill_event(task_goal="x" * 4001)

    with pytest.raises(ValidationError):
        _no_skill_event(successful_path=["step"] * 65)


def test_cluster_requires_unique_members_and_consistent_count() -> None:
    with pytest.raises(ValidationError, match="unique"):
        EvolutionCluster(
            cluster_id="cluster-1",
            user_id="user-1",
            event_kind=EvolutionEventKind.new_skill_evidence,
            canonical_signature="python-package-install",
            member_event_ids=["event-1", "event-1"],
            independent_run_count=2,
            status=ClusterStatus.collecting,
            grouping_evidence=[
                GroupingEvidence(
                    method="deterministic",
                    score=0.9,
                    reason="Same goal and tool signature.",
                )
            ],
        )

    with pytest.raises(ValidationError, match="cannot exceed"):
        EvolutionCluster(
            cluster_id="cluster-1",
            user_id="user-1",
            event_kind=EvolutionEventKind.new_skill_evidence,
            canonical_signature="python-package-install",
            member_event_ids=["event-1"],
            independent_run_count=2,
            status=ClusterStatus.collecting,
        )


def test_patch_proposal_requires_base_hash() -> None:
    with pytest.raises(ValidationError, match="base_skill_hash"):
        SkillProposal(
            proposal_id="proposal-1",
            cluster_id="cluster-1",
            user_id="user-1",
            operation=ProposalOperation.patch,
            skill_name="python-package-manager",
            proposed_files=[
                ProposedSkillFile(
                    path="SKILL.md",
                    content="updated",
                    executable=False,
                )
            ],
            supporting_event_ids=["event-1", "event-2", "event-3"],
            rationale="Add a missing prerequisite.",
            expected_improvements=["Avoid a known failure."],
            risks=[],
            status=ProposalStatus.staged,
        )


def test_create_proposal_rejects_base_hash_and_unsafe_paths() -> None:
    with pytest.raises(ValidationError, match="must not set base_skill_hash"):
        SkillProposal(
            proposal_id="proposal-1",
            cluster_id="cluster-1",
            user_id="user-1",
            operation=ProposalOperation.create,
            skill_name="python-package-manager",
            base_skill_hash="a" * 64,
            proposed_files=[
                ProposedSkillFile(
                    path="SKILL.md",
                    content="content",
                    executable=False,
                )
            ],
            supporting_event_ids=["event-1", "event-2", "event-3"],
            rationale="Create a reusable workflow.",
            expected_improvements=["Reuse successful installation steps."],
            risks=[],
            status=ProposalStatus.staged,
        )

    with pytest.raises(ValidationError):
        ProposedSkillFile(
            path="../SKILL.md",
            content="content",
            executable=False,
        )


def test_evaluation_validates_quality_and_decision() -> None:
    result = TaskEvaluationResult(
        task_id="task-1",
        split="held_out",
        condition="candidate_skill",
        success=True,
        metrics=EvaluationMetrics(
            tool_calls=4,
            input_tokens=100,
            output_tokens=25,
            latency_seconds=1.2,
        ),
    )
    evaluation = SkillEvaluation(
        evaluation_id="evaluation-1",
        proposal_id="proposal-1",
        user_id="user-1",
        source_replay_results=[],
        held_out_results=[result],
        baseline_results=[],
        candidate_results=[result],
        regression_results=[],
        safety_results={"static_scan": "allow", "llm_scan": "allow"},
        quality_score=0.8,
        decision=EvaluationDecision.approve,
        created_at=datetime(2026, 8, 16, tzinfo=UTC),
    )

    restored = SkillEvaluation.model_validate_json(evaluation.model_dump_json())
    assert restored == evaluation

    with pytest.raises(ValidationError):
        SkillEvaluation.model_validate(
            {
                **evaluation.model_dump(),
                "quality_score": 1.1,
            }
        )

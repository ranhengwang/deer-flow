from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from deerflow.config.skill_evolution_config import (
    SkillEvolutionPublicationConfig,
)
from deerflow.skill_evolution.approval import (
    SKILL_APPROVAL_POLICY_VERSION,
    ApprovalOutcome,
    ManualApprovalRequest,
    ProposalApprovalPolicy,
)
from deerflow.skill_evolution.models import (
    EvaluationDecision,
    EvaluationMetrics,
    ProposalOperation,
    ProposalStatus,
    ProposalStatusSource,
    ProposedSkillFile,
    SkillEvaluation,
    SkillPatchOperation,
    SkillProposal,
    TaskEvaluationResult,
)
from deerflow.skill_evolution.observability import (
    EvolutionLifecycleKind,
    EvolutionObservability,
)
from deerflow.skill_evolution.store.memory import (
    InMemorySkillEvolutionStore,
)

_CREATED = datetime(2026, 1, 1, tzinfo=UTC)
_NOW = datetime(2026, 8, 17, tzinfo=UTC)


def _result(
    task_id: str,
    *,
    split: str = "held_out",
    condition: str = "candidate_skill",
    success: bool = True,
) -> TaskEvaluationResult:
    return TaskEvaluationResult(
        task_id=task_id,
        split=split,
        condition=condition,
        success=success,
        metrics=EvaluationMetrics(
            tool_calls=3,
            input_tokens=100,
            output_tokens=20,
            latency_seconds=1.0,
        ),
        failure_reason=None if success else "verification_failed",
    )


def _proposal(
    *,
    operation: ProposalOperation = ProposalOperation.patch,
    executable: bool = False,
    risks: list[str] | None = None,
    requires_manual_review: bool = False,
    expires_at: datetime | None = None,
) -> SkillProposal:
    proposed_files = [
        ProposedSkillFile(
            path="SKILL.md",
            content="---\nname: fixture-transform\n---\n\n# Fixture Transform\n",
            executable=False,
        )
    ]
    if executable:
        proposed_files.append(
            ProposedSkillFile(
                path="scripts/run.sh",
                content="#!/bin/sh\nprintf ok\n",
                executable=True,
            )
        )
    kwargs = {}
    if operation is ProposalOperation.patch:
        kwargs.update(
            base_skill_hash="a" * 64,
            patch_operations=[
                SkillPatchOperation(
                    find="# Fixture Transform",
                    replace="# Safer Fixture Transform",
                    reason="Use the verified workflow.",
                    supporting_event_ids=[
                        "event-1",
                        "event-2",
                        "event-3",
                    ],
                )
            ],
            source_skill_hashes=["b" * 64],
        )
    return SkillProposal(
        proposal_id="proposal-1",
        cluster_id="cluster-1",
        user_id="user-1",
        operation=operation,
        skill_name="fixture-transform",
        proposed_files=proposed_files,
        supporting_event_ids=[
            "event-1",
            "event-2",
            "event-3",
        ],
        rationale="Capture the verified workflow.",
        expected_improvements=["Avoid the repeated failure."],
        risks=risks or [],
        requires_manual_review=requires_manual_review,
        review_reasons=(["unresolved_conflicts"] if requires_manual_review else []),
        status=ProposalStatus.validating,
        created_at=_CREATED,
        expires_at=expires_at or (_CREATED + timedelta(days=180)),
        **kwargs,
    )


def _evaluation(
    *,
    decision: EvaluationDecision = EvaluationDecision.approve,
    held_out_success: bool = True,
    safety_results: dict[str, str] | None = None,
) -> SkillEvaluation:
    held_out = _result(
        "held-out-1",
        success=held_out_success,
    )
    candidate_results = [held_out]
    held_out_results = [held_out]
    if decision is EvaluationDecision.manual_review:
        candidate_results = []
        held_out_results = []
    return SkillEvaluation(
        evaluation_id="evaluation-1",
        proposal_id="proposal-1",
        user_id="user-1",
        source_replay_results=[],
        held_out_results=held_out_results,
        baseline_results=[],
        candidate_results=candidate_results,
        regression_results=[],
        safety_results=safety_results
        or {
            "held_out": "1/1",
            "side_effects": "allow",
        },
        quality_score=0.8,
        decision=decision,
        created_at=_CREATED + timedelta(days=1),
    )


async def _persist(
    proposal: SkillProposal,
    evaluation: SkillEvaluation,
) -> InMemorySkillEvolutionStore:
    store = InMemorySkillEvolutionStore()
    await store.put_proposal(proposal)
    await store.put_evaluation(evaluation)
    return store


class RacingApprovalStore(InMemorySkillEvolutionStore):
    def __init__(self) -> None:
        super().__init__()
        self._transition_waiters = 0
        self._transition_ready = asyncio.Event()

    async def transition_proposal(self, **kwargs):
        if kwargs["new_status"] is ProposalStatus.approved:
            self._transition_waiters += 1
            if self._transition_waiters == 2:
                self._transition_ready.set()
            await self._transition_ready.wait()
        return await super().transition_proposal(**kwargs)


@pytest.mark.asyncio
async def test_default_policy_requires_manual_approval_without_state_change() -> None:
    proposal = _proposal()
    evaluation = _evaluation()
    store = await _persist(proposal, evaluation)
    policy = ProposalApprovalPolicy()

    result = await policy.assess_and_persist(
        user_id="user-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        store=store,
        now=_CREATED + timedelta(days=2),
    )

    assert result.assessment.outcome is ApprovalOutcome.manual_review
    assert result.assessment.reason_codes == ["publication_mode_manual"]
    assert result.changed is False
    assert result.proposal.status is ProposalStatus.validating
    assert result.proposal.status_history == []


@pytest.mark.asyncio
async def test_opt_in_auto_approval_only_marks_safe_patch_approved() -> None:
    proposal = _proposal()
    evaluation = _evaluation()
    store = await _persist(proposal, evaluation)
    observer = EvolutionObservability()
    policy = ProposalApprovalPolicy(
        config=SkillEvolutionPublicationConfig(
            mode="eligible_auto",
            allow_non_executable_auto_publish=True,
        ),
        observability=observer,
    )

    result = await policy.assess_and_persist(
        user_id="user-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        store=store,
        now=_CREATED + timedelta(days=2),
    )

    assert result.assessment.outcome is ApprovalOutcome.approved
    assert result.assessment.auto_publish_eligible is True
    assert result.changed is True
    assert result.proposal.status is ProposalStatus.approved
    assert result.proposal.status is not ProposalStatus.published
    transition = result.proposal.status_history[-1]
    assert transition.reason_code == "auto_approval_eligible"
    assert transition.source is ProposalStatusSource.approval_policy
    assert transition.evaluation_id == "evaluation-1"
    assert transition.policy_version == SKILL_APPROVAL_POLICY_VERSION
    assert [event.kind for event in observer.recent_events()] == [EvolutionLifecycleKind.approved]


@pytest.mark.asyncio
async def test_approved_but_unpublished_proposal_still_expires() -> None:
    proposal = _proposal()
    evaluation = _evaluation()
    store = await _persist(proposal, evaluation)
    policy = ProposalApprovalPolicy(
        config=SkillEvolutionPublicationConfig(
            mode="eligible_auto",
            allow_non_executable_auto_publish=True,
        )
    )
    approved = await policy.assess_and_persist(
        user_id="user-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        store=store,
        now=_CREATED + timedelta(days=2),
    )

    expired = await policy.assess_and_persist(
        user_id="user-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        store=store,
        now=_CREATED + timedelta(days=181),
    )

    assert approved.proposal.status is ProposalStatus.approved
    assert expired.proposal.status is ProposalStatus.expired
    assert [item.reason_code for item in expired.proposal.status_history] == [
        "auto_approval_eligible",
        "proposal_expired",
    ]
    assert expired.proposal.status_history[-1].from_status is ProposalStatus.approved


@pytest.mark.asyncio
async def test_identical_concurrent_auto_approval_is_idempotent() -> None:
    proposal = _proposal()
    evaluation = _evaluation()
    store = RacingApprovalStore()
    await store.put_proposal(proposal)
    await store.put_evaluation(evaluation)
    policy = ProposalApprovalPolicy(
        config=SkillEvolutionPublicationConfig(
            mode="eligible_auto",
            allow_non_executable_auto_publish=True,
        )
    )

    first, second = await asyncio.gather(
        policy.assess_and_persist(
            user_id="user-1",
            proposal_id="proposal-1",
            evaluation_id="evaluation-1",
            store=store,
            now=_CREATED + timedelta(days=2),
        ),
        policy.assess_and_persist(
            user_id="user-1",
            proposal_id="proposal-1",
            evaluation_id="evaluation-1",
            store=store,
            now=_CREATED + timedelta(days=2),
        ),
    )

    assert sorted([first.changed, second.changed]) == [
        False,
        True,
    ]
    assert first.proposal == second.proposal
    assert first.proposal.status is ProposalStatus.approved
    assert len(first.proposal.status_history) == 1


@pytest.mark.asyncio
async def test_new_skill_remains_manual_even_when_auto_mode_is_enabled() -> None:
    proposal = _proposal(operation=ProposalOperation.create)
    evaluation = _evaluation()
    store = await _persist(proposal, evaluation)
    policy = ProposalApprovalPolicy(
        config=SkillEvolutionPublicationConfig(
            mode="eligible_auto",
            allow_non_executable_auto_publish=True,
        )
    )

    result = await policy.assess_and_persist(
        user_id="user-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        store=store,
        now=_CREATED + timedelta(days=2),
    )

    assert result.assessment.outcome is ApprovalOutcome.manual_review
    assert "new_skill_requires_manual_approval" in result.assessment.reason_codes
    assert result.proposal.status is ProposalStatus.validating


@pytest.mark.asyncio
async def test_executable_change_can_never_be_auto_approved() -> None:
    proposal = _proposal(executable=True)
    evaluation = _evaluation()
    store = await _persist(proposal, evaluation)
    policy = ProposalApprovalPolicy(
        config=SkillEvolutionPublicationConfig(
            mode="eligible_auto",
            allow_non_executable_auto_publish=True,
        )
    )

    result = await policy.assess_and_persist(
        user_id="user-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        store=store,
        now=_CREATED + timedelta(days=2),
    )

    assert result.assessment.outcome is ApprovalOutcome.manual_review
    assert "executable_files_require_manual_review" in result.assessment.reason_codes
    assert result.assessment.auto_publish_eligible is False
    assert result.proposal.status is ProposalStatus.validating

    with pytest.raises(ValidationError):
        SkillEvolutionPublicationConfig(
            mode="eligible_auto",
            allow_non_executable_auto_publish=True,
            allow_executable_auto_publish=True,
        )


@pytest.mark.asyncio
async def test_risk_or_failed_held_out_blocks_auto_approval() -> None:
    policy = ProposalApprovalPolicy(
        config=SkillEvolutionPublicationConfig(
            mode="eligible_auto",
            allow_non_executable_auto_publish=True,
        )
    )
    risky = _proposal(risks=["May alter an environment-specific command."])
    risky_evaluation = _evaluation()
    risky_store = await _persist(risky, risky_evaluation)

    risky_result = await policy.assess_and_persist(
        user_id="user-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        store=risky_store,
        now=_CREATED + timedelta(days=2),
    )

    assert risky_result.assessment.outcome is ApprovalOutcome.manual_review
    assert "proposal_risks_require_manual_review" in risky_result.assessment.reason_codes

    failed = _proposal()
    failed_evaluation = _evaluation(
        decision=EvaluationDecision.reject,
        held_out_success=False,
    )
    failed_store = await _persist(failed, failed_evaluation)
    failed_result = await policy.assess_and_persist(
        user_id="user-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        store=failed_store,
        now=_CREATED + timedelta(days=2),
    )

    assert failed_result.assessment.outcome is ApprovalOutcome.rejected
    assert failed_result.proposal.status is ProposalStatus.rejected
    transition = failed_result.proposal.status_history[-1]
    assert transition.reason_code == "evaluation_rejected"
    assert "evaluation-1" in transition.reason


@pytest.mark.asyncio
async def test_expiration_precedes_automatic_or_manual_approval() -> None:
    proposal = _proposal(
        expires_at=_CREATED + timedelta(days=30),
    )
    evaluation = _evaluation()
    store = await _persist(proposal, evaluation)
    policy = ProposalApprovalPolicy(
        config=SkillEvolutionPublicationConfig(
            mode="eligible_auto",
            allow_non_executable_auto_publish=True,
        )
    )

    result = await policy.assess_and_persist(
        user_id="user-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        store=store,
        now=_CREATED + timedelta(days=31),
    )

    assert result.assessment.outcome is ApprovalOutcome.expired
    assert result.proposal.status is ProposalStatus.expired
    transition = result.proposal.status_history[-1]
    assert transition.reason_code == "proposal_expired"
    assert transition.reason

    with pytest.raises(ValueError, match="expired"):
        await policy.review_and_persist(
            user_id="user-1",
            proposal_id="proposal-1",
            evaluation_id="evaluation-1",
            request=ManualApprovalRequest(
                decision=ApprovalOutcome.approved,
                reviewer_id="reviewer-1",
                reason="Reviewed the candidate.",
                decided_at=_CREATED + timedelta(days=31),
            ),
            store=store,
        )


@pytest.mark.asyncio
async def test_manual_reviewer_can_approve_review_only_executable_proposal() -> None:
    proposal = _proposal(
        executable=True,
        requires_manual_review=True,
    )
    evaluation = _evaluation(
        decision=EvaluationDecision.manual_review,
        safety_results={
            "executable_support": "manual_review_required",
        },
    )
    store = await _persist(proposal, evaluation)
    policy = ProposalApprovalPolicy()
    request = ManualApprovalRequest(
        decision=ApprovalOutcome.approved,
        reviewer_id="reviewer-1",
        reason="Executable content was inspected and accepted.",
        decided_at=_CREATED + timedelta(days=2),
    )

    first = await policy.review_and_persist(
        user_id="user-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        request=request,
        store=store,
    )
    second = await policy.review_and_persist(
        user_id="user-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        request=request.model_copy(
            update={
                "decided_at": (request.decided_at + timedelta(minutes=1)),
            }
        ),
        store=store,
    )

    assert first.changed is True
    assert second.changed is False
    assert second.proposal == first.proposal
    assert second.assessment.assessed_at == request.decided_at
    assert first.proposal.status is ProposalStatus.approved
    assert first.proposal.status is not ProposalStatus.published
    transition = first.proposal.status_history[-1]
    assert transition.source is ProposalStatusSource.manual_reviewer
    assert transition.actor_id == "reviewer-1"
    assert transition.reason == request.reason


@pytest.mark.asyncio
async def test_manual_rejection_persists_explicit_reason() -> None:
    proposal = _proposal()
    evaluation = _evaluation()
    store = await _persist(proposal, evaluation)
    policy = ProposalApprovalPolicy()

    result = await policy.review_and_persist(
        user_id="user-1",
        proposal_id="proposal-1",
        evaluation_id="evaluation-1",
        request=ManualApprovalRequest(
            decision=ApprovalOutcome.rejected,
            reviewer_id="reviewer-1",
            reason="The workflow is too environment-specific.",
            decided_at=_CREATED + timedelta(days=2),
        ),
        store=store,
    )

    assert result.proposal.status is ProposalStatus.rejected
    transition = result.proposal.status_history[-1]
    assert transition.reason_code == "manual_rejection"
    assert transition.reason == "The workflow is too environment-specific."


@pytest.mark.asyncio
async def test_policy_rejects_cross_user_or_wrong_evaluation() -> None:
    proposal = _proposal()
    evaluation = _evaluation()
    store = await _persist(proposal, evaluation)
    policy = ProposalApprovalPolicy()

    with pytest.raises(ValueError, match="not found"):
        await policy.assess_and_persist(
            user_id="other-user",
            proposal_id="proposal-1",
            evaluation_id="evaluation-1",
            store=store,
            now=_CREATED + timedelta(days=2),
        )

    wrong_evaluation = _evaluation().model_copy(
        update={
            "evaluation_id": "evaluation-2",
            "proposal_id": "other-proposal",
        }
    )
    await store.put_evaluation(wrong_evaluation)
    with pytest.raises(ValueError, match="does not belong"):
        await policy.assess_and_persist(
            user_id="user-1",
            proposal_id="proposal-1",
            evaluation_id="evaluation-2",
            store=store,
            now=_CREATED + timedelta(days=2),
        )

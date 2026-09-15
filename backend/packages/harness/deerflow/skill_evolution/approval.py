"""Deterministic approval policy for evaluated Skill Proposals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from pydantic import Field, model_validator

from deerflow.config.skill_evolution_config import (
    SkillEvolutionPublicationConfig,
)
from deerflow.skill_evolution.models import (
    DetailText,
    EvaluationDecision,
    EvolutionModel,
    Identifier,
    ProposalOperation,
    ProposalStatus,
    ProposalStatusSource,
    ProposalStatusTransition,
    SkillEvaluation,
    SkillProposal,
)
from deerflow.skill_evolution.observability import (
    EvolutionLifecycleKind,
    EvolutionObservability,
    get_evolution_observability,
    safe_emit_evolution_event,
)
from deerflow.skill_evolution.store.base import (
    EvolutionStoreConflict,
    SkillEvolutionStore,
)

SKILL_APPROVAL_POLICY_VERSION = "skill-approval-v1"
DEFAULT_PROPOSAL_TTL_DAYS = 180


class ApprovalOutcome(StrEnum):
    manual_review = "manual_review"
    approved = "approved"
    rejected = "rejected"
    expired = "expired"


class ProposalApprovalAssessment(EvolutionModel):
    policy_version: Identifier
    proposal_id: Identifier
    evaluation_id: Identifier
    outcome: ApprovalOutcome
    reason_codes: list[Identifier] = Field(min_length=1, max_length=16)
    reason: DetailText
    auto_publish_eligible: bool
    expires_at: datetime
    assessed_at: datetime

    @model_validator(mode="after")
    def _validate_outcome(self):
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("approval reason codes must be unique")
        if self.auto_publish_eligible and self.outcome is not ApprovalOutcome.approved:
            raise ValueError("auto_publish_eligible requires an approved outcome")
        return self


class ManualApprovalRequest(EvolutionModel):
    decision: ApprovalOutcome
    reviewer_id: Identifier
    reason: DetailText
    decided_at: datetime

    @model_validator(mode="after")
    def _validate_decision(self):
        if self.decision not in {
            ApprovalOutcome.approved,
            ApprovalOutcome.rejected,
        }:
            raise ValueError("manual decision must approve or reject")
        return self


@dataclass(frozen=True, slots=True)
class ProposalApprovalResult:
    proposal: SkillProposal
    assessment: ProposalApprovalAssessment
    changed: bool


_REASON_TEXT = {
    "publication_mode_manual": "Publication mode requires an explicit human decision.",
    "auto_publish_disabled": "Automatic approval of non-executable changes is disabled.",
    "new_skill_requires_manual_approval": "New Skills require human approval in the research policy.",
    "evaluation_requires_manual_review": "The evaluation requires human review.",
    "executable_files_require_manual_review": "Executable Skill files always require human review.",
    "proposal_risks_require_manual_review": "The Proposal contains unresolved risks.",
    "proposal_review_flag": "The distiller marked the Proposal for human review.",
    "unsafe_evaluation_result": "The evaluation contains a safety or review blocker.",
    "held_out_evaluation_missing": "Automatic approval requires candidate held-out results.",
    "held_out_not_fully_successful": "Every candidate held-out task must pass before automatic approval.",
    "auto_approval_eligible": "All skill-approval-v1 automatic approval gates passed.",
    "proposal_expired": "The Proposal reached its approval deadline.",
    "evaluation_rejected": "The persisted evaluation rejected the candidate.",
    "manual_approval": "A human reviewer approved the Proposal.",
    "manual_rejection": "A human reviewer rejected the Proposal.",
}
_UNSAFE_SAFETY_MARKERS = (
    "blocked",
    "failed",
    "incomplete",
    "manual_review",
    "missing",
    "reject",
    "violation",
)


def _reason_text(
    reason_codes: list[str],
    *,
    evaluation_id: str,
) -> str:
    if reason_codes == ["evaluation_rejected"]:
        return f"Evaluation {evaluation_id} rejected the candidate."
    return " ".join(_REASON_TEXT[code] for code in reason_codes)


def _assessment(
    *,
    proposal: SkillProposal,
    evaluation: SkillEvaluation,
    outcome: ApprovalOutcome,
    reason_codes: list[str],
    auto_publish_eligible: bool,
    expires_at: datetime,
    now: datetime,
    reason: str | None = None,
) -> ProposalApprovalAssessment:
    return ProposalApprovalAssessment(
        policy_version=SKILL_APPROVAL_POLICY_VERSION,
        proposal_id=proposal.proposal_id,
        evaluation_id=evaluation.evaluation_id,
        outcome=outcome,
        reason_codes=reason_codes,
        reason=reason
        or _reason_text(
            reason_codes,
            evaluation_id=evaluation.evaluation_id,
        ),
        auto_publish_eligible=auto_publish_eligible,
        expires_at=expires_at,
        assessed_at=now,
    )


@dataclass(frozen=True, slots=True)
class ProposalApprovalPolicy:
    config: SkillEvolutionPublicationConfig = field(
        default_factory=SkillEvolutionPublicationConfig,
    )
    observability: EvolutionObservability = field(
        default_factory=get_evolution_observability,
        repr=False,
        compare=False,
    )

    def _observe_transition(
        self,
        *,
        proposal: SkillProposal,
        evaluation_id: str,
        assessment: ProposalApprovalAssessment,
        changed: bool,
    ) -> None:
        if not changed:
            return
        kind = EvolutionLifecycleKind.approved if assessment.outcome is ApprovalOutcome.approved else EvolutionLifecycleKind.rejected
        safe_emit_evolution_event(
            self.observability,
            kind=kind,
            stage="approval_policy",
            user_id=proposal.user_id,
            skill_name=proposal.skill_name,
            proposal_id=proposal.proposal_id,
            evaluation_id=evaluation_id,
            decision=assessment.outcome.value,
            reason_codes=assessment.reason_codes,
            occurred_at=assessment.assessed_at,
        )

    def _expires_at(
        self,
        proposal: SkillProposal,
    ) -> datetime:
        if proposal.expires_at is not None:
            return proposal.expires_at
        return proposal.created_at + timedelta(
            days=self.config.proposal_ttl_days,
        )

    @staticmethod
    def _validate_pair(
        proposal: SkillProposal,
        evaluation: SkillEvaluation,
    ) -> None:
        if evaluation.user_id != proposal.user_id:
            raise ValueError("evaluation user does not match proposal owner")
        if evaluation.proposal_id != proposal.proposal_id:
            raise ValueError("evaluation does not belong to proposal")

    @staticmethod
    def _evaluation_has_safety_blocker(
        evaluation: SkillEvaluation,
    ) -> bool:
        return any(marker in value.lower() for value in evaluation.safety_results.values() for marker in _UNSAFE_SAFETY_MARKERS)

    def assess(
        self,
        proposal: SkillProposal,
        evaluation: SkillEvaluation,
        *,
        now: datetime | None = None,
    ) -> ProposalApprovalAssessment:
        self._validate_pair(
            proposal,
            evaluation,
        )
        assessed_at = now or datetime.now(UTC)
        expires_at = self._expires_at(proposal)
        if assessed_at >= expires_at:
            return _assessment(
                proposal=proposal,
                evaluation=evaluation,
                outcome=ApprovalOutcome.expired,
                reason_codes=["proposal_expired"],
                auto_publish_eligible=False,
                expires_at=expires_at,
                now=assessed_at,
            )
        if evaluation.decision is EvaluationDecision.reject:
            return _assessment(
                proposal=proposal,
                evaluation=evaluation,
                outcome=ApprovalOutcome.rejected,
                reason_codes=["evaluation_rejected"],
                auto_publish_eligible=False,
                expires_at=expires_at,
                now=assessed_at,
            )
        if self.config.mode == "manual":
            return _assessment(
                proposal=proposal,
                evaluation=evaluation,
                outcome=ApprovalOutcome.manual_review,
                reason_codes=["publication_mode_manual"],
                auto_publish_eligible=False,
                expires_at=expires_at,
                now=assessed_at,
            )

        reason_codes: list[str] = []
        if not self.config.allow_non_executable_auto_publish:
            reason_codes.append("auto_publish_disabled")
        if proposal.operation is ProposalOperation.create:
            reason_codes.append("new_skill_requires_manual_approval")
        if evaluation.decision is EvaluationDecision.manual_review:
            reason_codes.append("evaluation_requires_manual_review")
        if any(item.executable for item in proposal.proposed_files):
            reason_codes.append("executable_files_require_manual_review")
        if proposal.risks:
            reason_codes.append("proposal_risks_require_manual_review")
        if proposal.requires_manual_review:
            reason_codes.append("proposal_review_flag")
        if self._evaluation_has_safety_blocker(evaluation):
            reason_codes.append("unsafe_evaluation_result")

        held_out = [result for result in evaluation.held_out_results if result.condition == "candidate_skill"]
        if not held_out:
            reason_codes.append("held_out_evaluation_missing")
        elif not all(result.success for result in held_out):
            reason_codes.append("held_out_not_fully_successful")

        if reason_codes:
            return _assessment(
                proposal=proposal,
                evaluation=evaluation,
                outcome=ApprovalOutcome.manual_review,
                reason_codes=reason_codes,
                auto_publish_eligible=False,
                expires_at=expires_at,
                now=assessed_at,
            )
        return _assessment(
            proposal=proposal,
            evaluation=evaluation,
            outcome=ApprovalOutcome.approved,
            reason_codes=["auto_approval_eligible"],
            auto_publish_eligible=True,
            expires_at=expires_at,
            now=assessed_at,
        )

    async def _load_pair(
        self,
        *,
        user_id: str,
        proposal_id: str,
        evaluation_id: str,
        store: SkillEvolutionStore,
    ) -> tuple[SkillProposal, SkillEvaluation]:
        proposal = await store.get_proposal(
            user_id,
            proposal_id,
        )
        if proposal is None:
            raise ValueError("proposal was not found")
        evaluation = await store.get_evaluation(
            user_id,
            evaluation_id,
        )
        if evaluation is None:
            raise ValueError("evaluation was not found")
        self._validate_pair(
            proposal,
            evaluation,
        )
        return proposal, evaluation

    @staticmethod
    def _target_status(
        outcome: ApprovalOutcome,
    ) -> ProposalStatus:
        return {
            ApprovalOutcome.approved: ProposalStatus.approved,
            ApprovalOutcome.rejected: ProposalStatus.rejected,
            ApprovalOutcome.expired: ProposalStatus.expired,
        }[outcome]

    @staticmethod
    async def _transition_or_read_winner(
        *,
        store: SkillEvolutionStore,
        user_id: str,
        proposal_id: str,
        expected_status: ProposalStatus,
        target_status: ProposalStatus,
        transition: ProposalStatusTransition,
    ) -> tuple[SkillProposal, bool]:
        try:
            updated = await store.transition_proposal(
                user_id=user_id,
                proposal_id=proposal_id,
                expected_status=expected_status,
                new_status=target_status,
                transition=transition,
            )
            return updated, True
        except EvolutionStoreConflict:
            winner = await store.get_proposal(
                user_id,
                proposal_id,
            )
            if winner is not None and winner.status is target_status and winner.status_history and winner.status_history[-1] == transition:
                return winner, False
            raise

    async def assess_and_persist(
        self,
        *,
        user_id: str,
        proposal_id: str,
        evaluation_id: str,
        store: SkillEvolutionStore,
        now: datetime | None = None,
    ) -> ProposalApprovalResult:
        proposal, evaluation = await self._load_pair(
            user_id=user_id,
            proposal_id=proposal_id,
            evaluation_id=evaluation_id,
            store=store,
        )
        assessment = self.assess(
            proposal,
            evaluation,
            now=now,
        )
        if assessment.outcome is ApprovalOutcome.manual_review:
            if proposal.status is not ProposalStatus.validating:
                raise ValueError("manual review requires a validating proposal")
            return ProposalApprovalResult(
                proposal=proposal,
                assessment=assessment,
                changed=False,
            )

        target_status = self._target_status(assessment.outcome)
        if proposal.status is target_status:
            return ProposalApprovalResult(
                proposal=proposal,
                assessment=assessment,
                changed=False,
            )
        expected_status = proposal.status
        if assessment.outcome is ApprovalOutcome.expired:
            eligible_statuses = {
                ProposalStatus.staged,
                ProposalStatus.validating,
                ProposalStatus.approved,
            }
        else:
            eligible_statuses = {ProposalStatus.validating}
        if expected_status not in eligible_statuses:
            raise ValueError(f"proposal is already {proposal.status.value}")
        transition = ProposalStatusTransition(
            from_status=expected_status,
            to_status=target_status,
            source=ProposalStatusSource.approval_policy,
            reason_code=assessment.reason_codes[0],
            reason=assessment.reason,
            occurred_at=assessment.assessed_at,
            evaluation_id=evaluation_id,
            policy_version=SKILL_APPROVAL_POLICY_VERSION,
        )
        updated, changed = await self._transition_or_read_winner(
            store=store,
            user_id=user_id,
            proposal_id=proposal_id,
            expected_status=expected_status,
            target_status=target_status,
            transition=transition,
        )
        self._observe_transition(
            proposal=updated,
            evaluation_id=evaluation_id,
            assessment=assessment,
            changed=changed,
        )
        return ProposalApprovalResult(
            proposal=updated,
            assessment=assessment,
            changed=changed,
        )

    @staticmethod
    def _manual_assessment(
        *,
        proposal: SkillProposal,
        evaluation: SkillEvaluation,
        request: ManualApprovalRequest,
        expires_at: datetime,
    ) -> ProposalApprovalAssessment:
        reason_code = "manual_approval" if request.decision is ApprovalOutcome.approved else "manual_rejection"
        return _assessment(
            proposal=proposal,
            evaluation=evaluation,
            outcome=request.decision,
            reason_codes=[reason_code],
            auto_publish_eligible=False,
            expires_at=expires_at,
            now=request.decided_at,
            reason=request.reason,
        )

    async def review_and_persist(
        self,
        *,
        user_id: str,
        proposal_id: str,
        evaluation_id: str,
        request: ManualApprovalRequest,
        store: SkillEvolutionStore,
    ) -> ProposalApprovalResult:
        proposal, evaluation = await self._load_pair(
            user_id=user_id,
            proposal_id=proposal_id,
            evaluation_id=evaluation_id,
            store=store,
        )
        target_status = self._target_status(request.decision)
        expires_at = self._expires_at(proposal)
        if proposal.status is target_status and proposal.status_history:
            latest = proposal.status_history[-1]
            if latest.source is ProposalStatusSource.manual_reviewer and latest.evaluation_id == evaluation_id and latest.actor_id == request.reviewer_id and latest.reason == request.reason:
                persisted_request = request.model_copy(
                    update={
                        "decided_at": latest.occurred_at,
                    }
                )
                return ProposalApprovalResult(
                    proposal=proposal,
                    assessment=self._manual_assessment(
                        proposal=proposal,
                        evaluation=evaluation,
                        request=persisted_request,
                        expires_at=expires_at,
                    ),
                    changed=False,
                )
        assessment = self._manual_assessment(
            proposal=proposal,
            evaluation=evaluation,
            request=request,
            expires_at=expires_at,
        )
        if proposal.status is ProposalStatus.expired or request.decided_at >= expires_at:
            if proposal.status is ProposalStatus.validating:
                transition = ProposalStatusTransition(
                    from_status=ProposalStatus.validating,
                    to_status=ProposalStatus.expired,
                    source=ProposalStatusSource.approval_policy,
                    reason_code="proposal_expired",
                    reason=_REASON_TEXT["proposal_expired"],
                    occurred_at=request.decided_at,
                    evaluation_id=evaluation_id,
                    policy_version=SKILL_APPROVAL_POLICY_VERSION,
                )
                expired, changed = await self._transition_or_read_winner(
                    store=store,
                    user_id=user_id,
                    proposal_id=proposal_id,
                    expected_status=ProposalStatus.validating,
                    target_status=ProposalStatus.expired,
                    transition=transition,
                )
                expired_assessment = _assessment(
                    proposal=proposal,
                    evaluation=evaluation,
                    outcome=ApprovalOutcome.expired,
                    reason_codes=["proposal_expired"],
                    auto_publish_eligible=False,
                    expires_at=expires_at,
                    now=request.decided_at,
                )
                self._observe_transition(
                    proposal=expired,
                    evaluation_id=evaluation_id,
                    assessment=expired_assessment,
                    changed=changed,
                )
            raise ValueError("proposal is expired")
        if proposal.status is not ProposalStatus.validating:
            raise EvolutionStoreConflict(f"proposal is already {proposal.status.value}")
        if request.decision is ApprovalOutcome.approved and evaluation.decision is EvaluationDecision.reject:
            raise ValueError("a rejected evaluation cannot be manually approved")

        transition = ProposalStatusTransition(
            from_status=ProposalStatus.validating,
            to_status=target_status,
            source=ProposalStatusSource.manual_reviewer,
            reason_code=assessment.reason_codes[0],
            reason=request.reason,
            occurred_at=request.decided_at,
            evaluation_id=evaluation_id,
            actor_id=request.reviewer_id,
            policy_version=SKILL_APPROVAL_POLICY_VERSION,
        )
        updated, changed = await self._transition_or_read_winner(
            store=store,
            user_id=user_id,
            proposal_id=proposal_id,
            expected_status=ProposalStatus.validating,
            target_status=target_status,
            transition=transition,
        )
        self._observe_transition(
            proposal=updated,
            evaluation_id=evaluation_id,
            assessment=assessment,
            changed=changed,
        )
        return ProposalApprovalResult(
            proposal=updated,
            assessment=assessment,
            changed=changed,
        )

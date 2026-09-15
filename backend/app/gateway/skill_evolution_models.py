"""Compact API models for owner-scoped Skill evolution operations."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _ApiModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class EvolutionPage(_ApiModel):
    user_id: str
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
    has_more: bool
    next_offset: int | None = Field(default=None, ge=0)


class EvolutionEventSummary(_ApiModel):
    event_id: str
    run_id: str
    thread_id: str
    event_kind: str
    task_signature_hash: str
    target_skill_name: str | None
    source_snapshot_hash: str
    outcome_status: str
    outcome_confidence: float
    tool_calls: int
    had_recoverable_errors: bool
    had_user_correction: bool
    non_trivial_workflow: bool
    explicit_remember_request: bool
    created_at: datetime


class EvolutionEventPage(EvolutionPage):
    data: list[EvolutionEventSummary]


class EvolutionClusterSummary(_ApiModel):
    cluster_id: str
    event_kind: str
    target_skill_name: str | None
    canonical_signature_hash: str
    member_event_count: int
    independent_run_count: int
    status: str
    contradictory_member_count: int
    created_at: datetime
    updated_at: datetime


class EvolutionClusterPage(EvolutionPage):
    data: list[EvolutionClusterSummary]


class SkillProposalSummary(_ApiModel):
    proposal_id: str
    cluster_id: str
    operation: str
    skill_name: str
    base_skill_hash: str | None
    status: str
    proposed_file_count: int
    executable_file_count: int
    supporting_event_count: int
    risk_count: int
    requires_manual_review: bool
    review_reason_codes: list[str]
    expires_at: datetime | None
    created_at: datetime


class SkillProposalPage(EvolutionPage):
    data: list[SkillProposalSummary]


class SkillEvaluationSummary(_ApiModel):
    evaluation_id: str
    proposal_id: str
    decision: str
    quality_score: float
    quality_designation: str | None
    sample_sufficient: bool | None
    sample_counts: dict[str, int]
    blocker_codes: list[str]
    candidate_result_count: int
    candidate_success_count: int
    regression_result_count: int
    regression_success_count: int
    created_at: datetime


class SkillEvaluationPage(EvolutionPage):
    data: list[SkillEvaluationSummary]


class SkillVersionSummary(_ApiModel):
    publication_id: str
    proposal_id: str
    evaluation_id: str | None
    skill_name: str
    operation: str
    status: str
    base_package_hash: str
    published_package_hash: str | None
    base_skill_hash: str | None
    published_skill_hash: str | None
    rollback_package_hash: str | None
    created_at: datetime
    published_at: datetime | None
    rolled_back_at: datetime | None


class SkillVersionPage(EvolutionPage):
    data: list[SkillVersionSummary]


class ProposalReviewRequest(_ApiModel):
    evaluation_id: str = Field(
        min_length=1,
        max_length=128,
    )
    decision: Literal["approve", "reject"]
    reason: str = Field(
        min_length=1,
        max_length=2_000,
    )


class ProposalPublishRequest(_ApiModel):
    evaluation_id: str = Field(
        min_length=1,
        max_length=128,
    )


class ProposalApprovalAssessmentSummary(_ApiModel):
    policy_version: str
    outcome: str
    reason_codes: list[str]
    auto_publish_eligible: bool
    expires_at: datetime
    assessed_at: datetime


class ProposalReviewResponse(_ApiModel):
    user_id: str
    proposal: SkillProposalSummary
    assessment: ProposalApprovalAssessmentSummary
    changed: bool


class SkillPublicationMutationResponse(_ApiModel):
    user_id: str
    proposal: SkillProposalSummary
    version: SkillVersionSummary
    changed: bool


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _target_skill_name(value: Any) -> str | None:
    target = getattr(value, "target_skill", None)
    name = getattr(target, "name", None)
    return str(name) if name is not None else None


def compact_event(value: Any) -> EvolutionEventSummary:
    complexity = value.complexity
    return EvolutionEventSummary(
        event_id=value.event_id,
        run_id=value.run_id,
        thread_id=value.thread_id,
        event_kind=_enum_value(value.event_kind),
        task_signature_hash=_sha256(value.task_signature),
        target_skill_name=_target_skill_name(value),
        source_snapshot_hash=value.source_snapshot_hash,
        outcome_status=_enum_value(value.outcome.status),
        outcome_confidence=value.outcome.confidence,
        tool_calls=complexity.tool_calls,
        had_recoverable_errors=(complexity.had_recoverable_errors),
        had_user_correction=complexity.had_user_correction,
        non_trivial_workflow=complexity.non_trivial_workflow,
        explicit_remember_request=(complexity.explicit_remember_request),
        created_at=value.created_at,
    )


def compact_cluster(value: Any) -> EvolutionClusterSummary:
    member_evidence = getattr(value, "member_evidence", ())
    return EvolutionClusterSummary(
        cluster_id=value.cluster_id,
        event_kind=_enum_value(value.event_kind),
        target_skill_name=_target_skill_name(value),
        canonical_signature_hash=_sha256(value.canonical_signature),
        member_event_count=len(value.member_event_ids),
        independent_run_count=value.independent_run_count,
        status=_enum_value(value.status),
        contradictory_member_count=sum(bool(getattr(item, "contradictory", False)) for item in member_evidence),
        created_at=value.created_at,
        updated_at=value.updated_at,
    )


def compact_proposal(value: Any) -> SkillProposalSummary:
    proposed_files = value.proposed_files
    return SkillProposalSummary(
        proposal_id=value.proposal_id,
        cluster_id=value.cluster_id,
        operation=_enum_value(value.operation),
        skill_name=value.skill_name,
        base_skill_hash=value.base_skill_hash,
        status=_enum_value(value.status),
        proposed_file_count=len(proposed_files),
        executable_file_count=sum(bool(item.executable) for item in proposed_files),
        supporting_event_count=len(value.supporting_event_ids),
        risk_count=len(value.risks),
        requires_manual_review=value.requires_manual_review,
        review_reason_codes=list(value.review_reasons),
        expires_at=value.expires_at,
        created_at=value.created_at,
    )


def compact_evaluation(value: Any) -> SkillEvaluationSummary:
    quality = value.quality
    candidate_results = value.candidate_results
    regression_results = value.regression_results
    return SkillEvaluationSummary(
        evaluation_id=value.evaluation_id,
        proposal_id=value.proposal_id,
        decision=_enum_value(value.decision),
        quality_score=value.quality_score,
        quality_designation=(_enum_value(quality.designation) if quality is not None else None),
        sample_sufficient=(quality.sample_sufficient if quality is not None else None),
        sample_counts=(dict(quality.sample_counts) if quality is not None else {}),
        blocker_codes=(list(quality.blockers) if quality is not None else []),
        candidate_result_count=len(candidate_results),
        candidate_success_count=sum(bool(item.success) for item in candidate_results),
        regression_result_count=len(regression_results),
        regression_success_count=sum(bool(item.success) for item in regression_results),
        created_at=value.created_at,
    )


def compact_version(value: Any) -> SkillVersionSummary:
    published_snapshot = value.published_snapshot
    rollback_snapshot = value.rollback_snapshot
    return SkillVersionSummary(
        publication_id=value.publication_id,
        proposal_id=value.proposal_id,
        evaluation_id=value.evaluation_id,
        skill_name=value.skill_name,
        operation=_enum_value(value.operation),
        status=_enum_value(value.status),
        base_package_hash=value.base_snapshot.snapshot_hash,
        published_package_hash=(published_snapshot.snapshot_hash if published_snapshot is not None else None),
        base_skill_hash=value.base_skill_hash,
        published_skill_hash=value.published_skill_hash,
        rollback_package_hash=(rollback_snapshot.snapshot_hash if rollback_snapshot is not None else None),
        created_at=value.created_at,
        published_at=value.published_at,
        rolled_back_at=value.rolled_back_at,
    )


def compact_approval_assessment(
    value: Any,
) -> ProposalApprovalAssessmentSummary:
    return ProposalApprovalAssessmentSummary(
        policy_version=value.policy_version,
        outcome=_enum_value(value.outcome),
        reason_codes=list(value.reason_codes),
        auto_publish_eligible=value.auto_publish_eligible,
        expires_at=value.expires_at,
        assessed_at=value.assessed_at,
    )

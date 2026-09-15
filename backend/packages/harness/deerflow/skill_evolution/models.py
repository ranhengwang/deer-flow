"""Versioned, JSON-safe contracts for evidence-based skill evolution."""

from __future__ import annotations

import base64
import binascii
import posixpath
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

if TYPE_CHECKING:
    from deerflow.skills.package import SkillPackageFile

EVOLUTION_EVENT_SCHEMA_VERSION = "deerflow.skill-evolution.event.v1"
EVOLUTION_CLUSTER_SCHEMA_VERSION = "deerflow.skill-evolution.cluster.v1"
SKILL_PROPOSAL_SCHEMA_VERSION = "deerflow.skill-evolution.proposal.v1"
SKILL_EVALUATION_SCHEMA_VERSION = "deerflow.skill-evolution.evaluation.v1"
EVOLUTION_TRACE_SCHEMA_VERSION = "deerflow.skill-evolution.trace-snapshot.v1"
SKILL_PACKAGE_SNAPSHOT_SCHEMA_VERSION = "deerflow.skill-evolution.package-snapshot.v1"
SKILL_PUBLICATION_SCHEMA_VERSION_V1 = "deerflow.skill-evolution.publication.v1"
SKILL_PUBLICATION_SCHEMA_VERSION = "deerflow.skill-evolution.publication.v2"
EVOLUTION_JOB_SCHEMA_VERSION = "deerflow.skill-evolution.job.v1"
SELECTION_CREDIT_SCHEMA_VERSION = "deerflow.skill-evolution.selection-credit.v1"
UTILIZATION_CREDIT_SCHEMA_VERSION = "deerflow.skill-evolution.utilization-credit.v1"
DISTILLATION_CREDIT_SCHEMA_VERSION = "deerflow.skill-evolution.distillation-credit.v1"

Identifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
SkillName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    ),
]
Sha256 = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    ),
]
ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
]
DetailText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000),
]
TaskGoalText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4_000),
]
SkillFileContent = Annotated[
    str,
    StringConstraints(min_length=1, max_length=262_144),
]


class EvolutionModel(BaseModel):
    """Strict immutable base for persisted evolution records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class OutcomeStatus(StrEnum):
    success = "success"
    failure = "failure"
    unknown = "unknown"


class EvolutionEventKind(StrEnum):
    new_skill_evidence = "new_skill_evidence"
    skill_patch_evidence = "skill_patch_evidence"


class SkillGapCategory(StrEnum):
    outdated_instruction = "outdated_instruction"
    os_incompatibility = "os_incompatibility"
    missing_prerequisite = "missing_prerequisite"
    uncovered_failure_mode = "uncovered_failure_mode"
    weak_verification = "weak_verification"
    wrong_tool_guidance = "wrong_tool_guidance"
    ambiguous_trigger = "ambiguous_trigger"
    unnecessary_step = "unnecessary_step"


class ClusterStatus(StrEnum):
    collecting = "collecting"
    ready = "ready"
    distilled = "distilled"
    rejected = "rejected"


class ProposalOperation(StrEnum):
    create = "create"
    patch = "patch"
    edit = "edit"
    write_file = "write_file"


class ProposalStatus(StrEnum):
    staged = "staged"
    validating = "validating"
    approved = "approved"
    publishing = "publishing"
    rejected = "rejected"
    published = "published"
    rolled_back = "rolled_back"
    expired = "expired"


class ProposalStatusSource(StrEnum):
    system = "system"
    evaluator = "evaluator"
    approval_policy = "approval_policy"
    manual_reviewer = "manual_reviewer"
    publisher = "publisher"
    rollback = "rollback"


class EvaluationDecision(StrEnum):
    approve = "approve"
    reject = "reject"
    manual_review = "manual_review"


class PublicationStatus(StrEnum):
    preparing = "preparing"
    published = "published"
    rolled_back = "rolled_back"


class EvolutionJobStatus(StrEnum):
    pending = "pending"
    running = "running"
    retry = "retry"
    completed = "completed"
    dead = "dead"


class QualityDesignation(StrEnum):
    insufficient_evidence = "insufficient_evidence"
    evaluated = "evaluated"
    high_quality = "high_quality"


class CreditKind(StrEnum):
    selection = "selection"
    utilization = "utilization"
    distillation = "distillation"


class SelectionDecisionSource(StrEnum):
    user_slash = "user_slash"
    model_read = "model_read"
    mixed = "mixed"
    no_skill = "no_skill"


class DistillationCreditStatus(StrEnum):
    collecting = "collecting"
    mature = "mature"
    rolled_back = "rolled_back"


class TraceRunStatus(StrEnum):
    success = "success"
    error = "error"
    timeout = "timeout"
    interrupted = "interrupted"


class EvolutionJob(EvolutionModel):
    """Durable run-level work item claimed through a renewable lease."""

    schema_version: Literal["deerflow.skill-evolution.job.v1"] = EVOLUTION_JOB_SCHEMA_VERSION
    job_id: Identifier
    idempotency_key: Sha256
    user_id: Identifier
    thread_id: Identifier
    run_id: Identifier
    snapshot_hash: Sha256
    pipeline_version: Identifier
    status: EvolutionJobStatus = EvolutionJobStatus.pending
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(ge=1, le=32)
    next_attempt_at: datetime | None
    lease_owner: Identifier | None = None
    lease_token: Identifier | None = None
    lease_expires_at: datetime | None = None
    last_error_code: Identifier | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    revision: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_lifecycle(self) -> Self:
        lease_values = (
            self.lease_owner,
            self.lease_token,
            self.lease_expires_at,
        )
        if self.status is EvolutionJobStatus.running:
            if any(value is None for value in lease_values):
                raise ValueError("running evolution job requires a complete lease")
            if self.next_attempt_at is not None:
                raise ValueError("running evolution job cannot have next_attempt_at")
            if self.completed_at is not None:
                raise ValueError("running evolution job cannot be completed")
        else:
            if any(value is not None for value in lease_values):
                raise ValueError("non-running evolution job cannot retain a lease")
        if self.status in {
            EvolutionJobStatus.pending,
            EvolutionJobStatus.retry,
        }:
            if self.next_attempt_at is None:
                raise ValueError("claimable evolution job requires next_attempt_at")
            if self.completed_at is not None:
                raise ValueError("claimable evolution job cannot be completed")
        elif self.status in {
            EvolutionJobStatus.completed,
            EvolutionJobStatus.dead,
        }:
            if self.next_attempt_at is not None:
                raise ValueError("terminal evolution job cannot be scheduled")
            if self.completed_at is None:
                raise ValueError("terminal evolution job requires completed_at")
        if self.attempt_count > self.max_attempts:
            raise ValueError("attempt_count cannot exceed max_attempts")
        return self


class SelectionCandidate(EvolutionModel):
    skill_name: SkillName
    content_hash: Sha256
    selected: bool
    rank: int | None = Field(default=None, ge=1, le=1_000)
    score: float | None = None


class CreditToolCall(EvolutionModel):
    tool_call_id: Identifier
    tool_name: Identifier
    status: Literal["success", "error", "unknown"]
    error_type: Identifier | None = None


class CreditCost(EvolutionModel):
    tool_call_count: int = Field(ge=0, le=10_000)
    error_tool_call_count: int = Field(ge=0, le=10_000)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_seconds: float | None = Field(default=None, ge=0.0)
    unavailable_fields: list[Identifier] = Field(
        default_factory=list,
        max_length=8,
    )

    @model_validator(mode="after")
    def _validate_counts(self) -> Self:
        if self.error_tool_call_count > self.tool_call_count:
            raise ValueError("error tool-call count cannot exceed total")
        if len(set(self.unavailable_fields)) != len(self.unavailable_fields):
            raise ValueError("unavailable cost fields must be unique")
        return self


class CreditOutcomeSample(EvolutionModel):
    run_id: Identifier
    reward: float | None = None
    outcome_status: OutcomeStatus
    environment_fingerprint: Sha256 | None = None
    observed_at: datetime


class SelectionCredit(EvolutionModel):
    schema_version: Literal["deerflow.skill-evolution.selection-credit.v1"] = SELECTION_CREDIT_SCHEMA_VERSION
    credit_id: Identifier
    kind: Literal[CreditKind.selection] = CreditKind.selection
    user_id: Identifier
    thread_id: Identifier
    run_id: Identifier
    snapshot_hash: Sha256
    search_query: ShortText | None = None
    query_unavailable_reason: Identifier | None = None
    candidates: list[SelectionCandidate] = Field(
        default_factory=list,
        max_length=32,
    )
    selected_skill_name: SkillName | None = None
    selected_skill_hash: Sha256 | None = None
    no_skill_selected: bool
    decision_source: SelectionDecisionSource
    policy_model_name: Identifier | None = None
    policy_prompt_version: Identifier | None = None
    policy_metadata_unavailable_reason: Identifier | None = None
    log_probability: float | None = Field(default=None, le=0.0)
    log_probability_unavailable_reason: Identifier | None = None
    outcome_reward: float | None = None
    credit_value: float | None = None
    formula_version: Identifier
    utility_sample_count: int = Field(ge=0)
    created_at: datetime
    revision: Literal[0] = 0

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        selected_identity = (
            self.selected_skill_name,
            self.selected_skill_hash,
        )
        if self.no_skill_selected:
            if any(value is not None for value in selected_identity):
                raise ValueError("no-skill selection cannot name a Skill")
            if self.decision_source is not (SelectionDecisionSource.no_skill):
                raise ValueError("no-skill selection requires no-skill source")
            if any(item.selected for item in self.candidates):
                raise ValueError("no-skill selection cannot select a candidate")
        elif any(value is None for value in selected_identity):
            raise ValueError("Skill selection requires name and content hash")
        elif not any(item.selected and item.skill_name == self.selected_skill_name and item.content_hash == self.selected_skill_hash for item in self.candidates):
            raise ValueError("selected Skill must resolve to a candidate")
        if (self.search_query is None) == (self.query_unavailable_reason is None):
            raise ValueError("search query requires exactly one value or reason")
        if (self.log_probability is None) == (self.log_probability_unavailable_reason is None):
            raise ValueError("log probability requires exactly one value or reason")
        if self.policy_model_name is None and (self.policy_metadata_unavailable_reason is None):
            raise ValueError("missing policy model requires an unavailable reason")
        if self.credit_value is not None and (self.utility_sample_count < 1):
            raise ValueError("selection credit requires utility samples")
        return self


class UtilizationCredit(EvolutionModel):
    schema_version: Literal["deerflow.skill-evolution.utilization-credit.v1"] = UTILIZATION_CREDIT_SCHEMA_VERSION
    credit_id: Identifier
    kind: Literal[CreditKind.utilization] = CreditKind.utilization
    user_id: Identifier
    thread_id: Identifier
    run_id: Identifier
    snapshot_hash: Sha256
    skill_name: SkillName
    skill_content_hash: Sha256
    activation_source: Literal["slash", "read"]
    relevant_tool_calls: list[CreditToolCall] = Field(
        default_factory=list,
        max_length=256,
    )
    tool_attribution_scope: Literal["run_level"]
    deviation_codes: list[Identifier] = Field(
        default_factory=list,
        max_length=64,
    )
    outcome_status: OutcomeStatus
    outcome_confidence: float = Field(ge=0.0, le=1.0)
    outcome_reward: float | None = None
    credit_value: float | None = None
    formula_version: Identifier
    cost: CreditCost
    instruction_adherence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    adherence_unavailable_reason: Identifier | None = None
    created_at: datetime
    revision: Literal[0] = 0

    @model_validator(mode="after")
    def _validate_utilization(self) -> Self:
        if len(set(self.deviation_codes)) != len(self.deviation_codes):
            raise ValueError("deviation codes must be unique")
        if (self.instruction_adherence is None) == (self.adherence_unavailable_reason is None):
            raise ValueError("instruction adherence requires value or reason")
        if self.credit_value != self.outcome_reward:
            raise ValueError("utilization credit must equal verified reward")
        return self


class DistillationCredit(EvolutionModel):
    schema_version: Literal["deerflow.skill-evolution.distillation-credit.v1"] = DISTILLATION_CREDIT_SCHEMA_VERSION
    credit_id: Identifier
    kind: Literal[CreditKind.distillation] = CreditKind.distillation
    user_id: Identifier
    proposal_id: Identifier
    publication_id: Identifier
    skill_name: SkillName
    published_skill_hash: Sha256
    source_event_ids: list[Identifier] = Field(
        min_length=1,
        max_length=64,
    )
    baseline_outcomes: list[CreditOutcomeSample] = Field(
        min_length=1,
        max_length=64,
    )
    future_outcomes: list[CreditOutcomeSample] = Field(
        default_factory=list,
        max_length=256,
    )
    minimum_future_samples: int = Field(ge=1, le=64)
    status: DistillationCreditStatus
    formula_version: Identifier
    credit_value: float | None = None
    created_at: datetime
    updated_at: datetime
    revision: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_distillation(self) -> Self:
        if len(set(self.source_event_ids)) != len(self.source_event_ids):
            raise ValueError("source event IDs must be unique")
        baseline_runs = [item.run_id for item in self.baseline_outcomes]
        future_runs = [item.run_id for item in self.future_outcomes]
        if len(set(baseline_runs)) != len(baseline_runs):
            raise ValueError("baseline outcome runs must be unique")
        if len(set(future_runs)) != len(future_runs):
            raise ValueError("future outcome runs must be unique")
        known_future = [item for item in self.future_outcomes if item.reward is not None]
        if self.status is DistillationCreditStatus.mature and len(known_future) < self.minimum_future_samples:
            raise ValueError("mature distillation credit needs future samples")
        if self.status is DistillationCreditStatus.collecting and self.credit_value is not None:
            raise ValueError("collecting distillation credit has no value")
        if self.status is DistillationCreditStatus.mature and self.credit_value is None:
            raise ValueError("mature distillation credit requires a value")
        return self


SkillCredit = Annotated[
    SelectionCredit | UtilizationCredit | DistillationCredit,
    Field(discriminator="kind"),
]


class TraceToolEvent(EvolutionModel):
    sequence: int = Field(ge=0)
    tool_call_id: Identifier
    tool_name: Identifier
    arguments: Annotated[
        str,
        StringConstraints(max_length=16_000),
    ]
    result: Annotated[
        str,
        StringConstraints(max_length=32_000),
    ]
    status: Literal["success", "error", "unknown"]
    error_type: Identifier | None = None
    result_truncated: bool = False


class TraceSkillEvent(EvolutionModel):
    skill_name: SkillName
    skill_path: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1_024),
    ]
    content_hash: Sha256
    activation_source: Literal["slash", "read"]
    category: Identifier | None = None


class EvolutionTraceSnapshot(EvolutionModel):
    schema_version: Literal["deerflow.skill-evolution.trace-snapshot.v1"] = EVOLUTION_TRACE_SCHEMA_VERSION
    snapshot_hash: Sha256
    run_id: Identifier
    thread_id: Identifier
    user_id: Identifier
    model_name: Identifier | None = None
    run_status: TraceRunStatus
    stop_reason: Identifier | None = None
    task_input: Annotated[
        str,
        StringConstraints(max_length=8_000),
    ]
    final_answer: Annotated[
        str,
        StringConstraints(max_length=12_000),
    ]
    tool_events: list[TraceToolEvent] = Field(
        default_factory=list,
        max_length=256,
    )
    skill_events: list[TraceSkillEvent] = Field(
        default_factory=list,
        max_length=32,
    )
    user_corrections: list[TaskGoalText] = Field(
        default_factory=list,
        max_length=16,
    )
    artifacts: list[
        Annotated[
            str,
            StringConstraints(
                strip_whitespace=True,
                min_length=1,
                max_length=1_024,
            ),
        ]
    ] = Field(default_factory=list, max_length=256)
    environment: EnvironmentSignature
    source_event_count: int = Field(ge=0)
    included_event_count: int = Field(ge=0)
    truncated: bool
    created_at: datetime

    @model_validator(mode="after")
    def _validate_event_counts(self) -> Self:
        if self.included_event_count > self.source_event_count:
            raise ValueError("included_event_count cannot exceed source_event_count")
        if self.truncated != (self.included_event_count < self.source_event_count):
            raise ValueError("truncated must agree with event counts")
        return self


class OutcomeCheck(EvolutionModel):
    source: Identifier
    passed: bool
    detail: DetailText
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    authoritative: bool = True
    priority: int | None = Field(default=None, ge=0, le=1_000)


class OutcomeEvidence(EvolutionModel):
    status: OutcomeStatus
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[ShortText] = Field(default_factory=list, max_length=32)
    checks: list[OutcomeCheck] = Field(default_factory=list, max_length=64)
    final_reward: float | None = None

    @model_validator(mode="after")
    def _success_requires_evidence(self) -> Self:
        if self.status is OutcomeStatus.success and not self.sources:
            raise ValueError("successful outcome requires at least one evidence source")
        return self


class EnvironmentSignature(EvolutionModel):
    os: ShortText
    shell: ShortText | None = None
    runtime: ShortText | None = None
    metadata: dict[Identifier, ShortText] = Field(default_factory=dict, max_length=32)


class ComplexitySignals(EvolutionModel):
    tool_calls: int = Field(ge=0, le=10_000)
    had_recoverable_errors: bool = False
    had_user_correction: bool = False
    non_trivial_workflow: bool = False
    explicit_remember_request: bool = False


class ToolSignature(EvolutionModel):
    tool_names: list[Identifier] = Field(default_factory=list, max_length=256)
    error_types: list[Identifier] = Field(default_factory=list, max_length=64)


class SkillUsage(EvolutionModel):
    used: bool
    skill_name: SkillName | None = None
    skill_path: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=1_024),
        ]
        | None
    ) = None
    content_hash: Sha256 | None = None
    activation_source: Literal["slash", "read", "configured", "unknown"] | None = None

    @model_validator(mode="after")
    def _validate_usage_fields(self) -> Self:
        details = (self.skill_name, self.skill_path, self.content_hash)
        if self.used and any(value is None for value in details):
            raise ValueError("used skill requires skill_name, skill_path, and content_hash")
        if not self.used and any(value is not None for value in details):
            raise ValueError("unused skill must not include skill details")
        if not self.used and self.activation_source is not None:
            raise ValueError("unused skill must not include activation_source")
        return self


class SkillTarget(EvolutionModel):
    name: SkillName
    content_hash: Sha256 | None = None


class FailedAttempt(EvolutionModel):
    action: DetailText
    error: DetailText
    lesson: DetailText


class UserCorrection(EvolutionModel):
    correction: DetailText
    effective_change: DetailText


class SkillGap(EvolutionModel):
    category: SkillGapCategory
    evidence: DetailText
    recommended_change: DetailText


class EvidenceReference(EvolutionModel):
    source: Identifier
    index: int | None = Field(default=None, ge=0)
    excerpt: DetailText | None = None
    content_hash: Sha256 | None = None


class SemanticEvidenceLink(EvolutionModel):
    semantic_field: Literal[
        "task_goal",
        "successful_path",
        "failed_attempt",
        "user_correction",
        "reusable_lesson",
        "skill_gap",
        "target_skill",
    ]
    item_index: int | None = Field(default=None, ge=0)
    evidence_segment_ids: list[Identifier] = Field(
        min_length=1,
        max_length=16,
    )

    @model_validator(mode="after")
    def _validate_link(self) -> Self:
        singleton_fields = {"task_goal", "target_skill"}
        if self.semantic_field in singleton_fields and self.item_index is not None:
            raise ValueError("singleton semantic field must not set item_index")
        if self.semantic_field not in singleton_fields and self.item_index is None:
            raise ValueError("collection semantic field requires item_index")
        if len(set(self.evidence_segment_ids)) != len(self.evidence_segment_ids):
            raise ValueError("semantic evidence segment IDs must be unique")
        return self


class EvolutionEvent(EvolutionModel):
    schema_version: Literal["deerflow.skill-evolution.event.v1"] = EVOLUTION_EVENT_SCHEMA_VERSION
    event_id: Identifier
    run_id: Identifier
    thread_id: Identifier
    user_id: Identifier
    extractor_version: Identifier
    extractor_model_name: Identifier | None = None
    extractor_prompt_version: Identifier | None = None
    source_snapshot_hash: Sha256
    source_extraction_hash: Sha256 | None = None
    task_input_hash: Sha256
    event_kind: EvolutionEventKind
    task_signature: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
    ]
    task_goal: TaskGoalText
    environment: EnvironmentSignature
    outcome: OutcomeEvidence
    complexity: ComplexitySignals
    tool_signature: ToolSignature
    skill_usage: SkillUsage
    successful_path: list[DetailText] = Field(min_length=1, max_length=64)
    failed_attempts: list[FailedAttempt] = Field(
        default_factory=list,
        max_length=32,
    )
    user_corrections: list[UserCorrection] = Field(
        default_factory=list,
        max_length=16,
    )
    reusable_lessons: list[DetailText] = Field(
        default_factory=list,
        max_length=32,
    )
    skill_gaps: list[SkillGap] = Field(default_factory=list, max_length=16)
    target_skill: SkillTarget | None = None
    provenance: list[EvidenceReference] = Field(
        default_factory=list,
        max_length=128,
    )
    evidence_links: list[SemanticEvidenceLink] = Field(
        default_factory=list,
        max_length=128,
    )
    created_at: datetime

    @model_validator(mode="after")
    def _validate_evolution_branch(self) -> Self:
        if self.outcome.status is not OutcomeStatus.success:
            raise ValueError("evolution event requires a successful outcome")
        extraction_metadata = (
            self.extractor_model_name,
            self.extractor_prompt_version,
            self.source_extraction_hash,
        )
        if any(value is not None for value in extraction_metadata) and any(value is None for value in extraction_metadata):
            raise ValueError("structured extractor metadata must be complete")
        if self.extractor_model_name is not None:
            provenance_ids = {reference.source for reference in self.provenance}
            linked_ids = {evidence_id for link in self.evidence_links for evidence_id in link.evidence_segment_ids}
            if not linked_ids.issubset(provenance_ids):
                raise ValueError("evidence links must resolve to provenance")

            expected_links = {("task_goal", None)}
            expected_links.update(("successful_path", index) for index in range(len(self.successful_path)))
            expected_links.update(("failed_attempt", index) for index in range(len(self.failed_attempts)))
            expected_links.update(("user_correction", index) for index in range(len(self.user_corrections)))
            expected_links.update(("reusable_lesson", index) for index in range(len(self.reusable_lessons)))
            expected_links.update(("skill_gap", index) for index in range(len(self.skill_gaps)))
            if self.target_skill is not None:
                expected_links.add(("target_skill", None))
            actual_links = {(link.semantic_field, link.item_index) for link in self.evidence_links}
            if actual_links != expected_links or len(actual_links) != len(self.evidence_links):
                raise ValueError("structured event evidence links are incomplete")

        if self.event_kind is EvolutionEventKind.new_skill_evidence:
            if self.skill_usage.used:
                raise ValueError("new-skill evidence must not use a skill")
            if self.target_skill is not None:
                raise ValueError("new-skill evidence must not target an existing skill")
            if self.skill_gaps:
                raise ValueError("new-skill evidence must not include skill gaps")
            return self

        if not self.skill_usage.used:
            raise ValueError("skill-patch evidence requires a used skill")
        if self.target_skill is None:
            raise ValueError("skill-patch evidence requires a target skill")
        if self.target_skill.name != self.skill_usage.skill_name:
            raise ValueError("target skill name must match the used skill")
        if not self.skill_gaps:
            raise ValueError("skill-patch evidence requires at least one skill gap")
        return self


class GroupingEvidence(EvolutionModel):
    method: Literal["deterministic", "semantic", "llm"]
    score: float = Field(ge=0.0, le=1.0)
    reason: DetailText


class ClusterEnvironmentRelationship(StrEnum):
    prototype = "prototype"
    same_workflow = "same_workflow"
    conditional_environment_branch = "conditional_environment_branch"
    different_workflow = "different_workflow"


class ClusterMemberEvidence(EvolutionModel):
    event_id: Identifier
    run_id: Identifier
    relationship: ClusterEnvironmentRelationship
    environment_condition: DetailText | None = None
    contradictory: bool = False
    grouping_evidence: list[GroupingEvidence] = Field(
        min_length=1,
        max_length=3,
    )

    @model_validator(mode="after")
    def _validate_relationship(self) -> Self:
        if self.relationship is ClusterEnvironmentRelationship.different_workflow:
            raise ValueError("different-workflow events cannot be cluster members")
        if self.relationship is ClusterEnvironmentRelationship.conditional_environment_branch and self.environment_condition is None:
            raise ValueError("conditional environment branch requires environment_condition")
        if self.relationship is not ClusterEnvironmentRelationship.conditional_environment_branch and self.environment_condition is not None:
            raise ValueError("environment_condition requires a conditional environment branch")
        methods = [evidence.method for evidence in self.grouping_evidence]
        if len(set(methods)) != len(methods):
            raise ValueError("cluster member grouping methods must be unique")
        return self


class EvolutionCluster(EvolutionModel):
    schema_version: Literal["deerflow.skill-evolution.cluster.v1"] = EVOLUTION_CLUSTER_SCHEMA_VERSION
    cluster_id: Identifier
    user_id: Identifier
    event_kind: EvolutionEventKind
    target_skill: SkillTarget | None = None
    canonical_signature: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
    ]
    member_event_ids: list[Identifier] = Field(min_length=1, max_length=64)
    independent_run_count: int = Field(ge=1, le=64)
    status: ClusterStatus
    grouping_evidence: list[GroupingEvidence] = Field(
        default_factory=list,
        max_length=64,
    )
    member_evidence: list[ClusterMemberEvidence] = Field(
        default_factory=list,
        max_length=64,
    )
    confirmation_model_name: Identifier | None = None
    confirmation_prompt_version: Identifier | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_members(self) -> Self:
        if len(set(self.member_event_ids)) != len(self.member_event_ids):
            raise ValueError("cluster member event IDs must be unique")
        if self.independent_run_count > len(self.member_event_ids):
            raise ValueError("independent_run_count cannot exceed member event count")
        if self.event_kind is EvolutionEventKind.skill_patch_evidence and self.target_skill is None:
            raise ValueError("skill-patch cluster requires a target skill")
        if self.event_kind is EvolutionEventKind.new_skill_evidence and self.target_skill is not None:
            raise ValueError("new-skill cluster must not target an existing skill")
        if self.member_evidence:
            evidence_ids = [evidence.event_id for evidence in self.member_evidence]
            if evidence_ids != self.member_event_ids:
                raise ValueError("cluster member evidence must match member event order")
            if self.confirmation_model_name is None or self.confirmation_prompt_version is None:
                raise ValueError("confirmed cluster members require confirmation metadata")
            if self.status is ClusterStatus.ready and any(evidence.contradictory for evidence in self.member_evidence):
                raise ValueError("ready cluster cannot contain contradictory evidence")
        elif self.confirmation_model_name is not None or self.confirmation_prompt_version is not None:
            raise ValueError("confirmation metadata requires member evidence")
        return self


def _normalize_relative_path(path: str) -> str:
    raw = path.replace("\\", "/").strip()
    if not raw:
        raise ValueError("path must not be empty")
    pure = PurePosixPath(raw)
    if pure.is_absolute():
        raise ValueError("absolute paths are not allowed")
    normalized = posixpath.normpath(raw)
    if normalized in {"", "."}:
        raise ValueError("path must identify a file")
    if any(part in {"", ".."} for part in PurePosixPath(normalized).parts):
        raise ValueError("path must not contain parent-directory traversal")
    return normalized


class ProposedSkillFile(EvolutionModel):
    path: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1_024),
    ]
    content: SkillFileContent
    executable: bool

    @model_validator(mode="after")
    def _validate_path(self) -> Self:
        normalized = _normalize_relative_path(self.path)
        if normalized != self.path.replace("\\", "/"):
            raise ValueError("path must already be normalized")
        return self


class ProposalEvidenceMapping(EvolutionModel):
    file_path: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1_024),
    ]
    section: Identifier
    supporting_event_ids: list[Identifier] = Field(
        min_length=1,
        max_length=64,
    )

    @model_validator(mode="after")
    def _validate_mapping(self) -> Self:
        normalized = _normalize_relative_path(self.file_path)
        if normalized != self.file_path.replace("\\", "/"):
            raise ValueError("evidence file_path must already be normalized")
        if len(set(self.supporting_event_ids)) != len(self.supporting_event_ids):
            raise ValueError("evidence supporting event IDs must be unique")
        return self


class SkillPatchOperation(EvolutionModel):
    path: Literal["SKILL.md"] = "SKILL.md"
    find: SkillFileContent
    replace: SkillFileContent
    expected_count: int = Field(default=1, ge=1, le=16)
    reason: DetailText
    environment_condition: DetailText | None = None
    supporting_event_ids: list[Identifier] = Field(
        min_length=1,
        max_length=64,
    )

    @model_validator(mode="after")
    def _validate_patch(self) -> Self:
        if self.find == self.replace:
            raise ValueError("patch find and replace must differ")
        if len(set(self.supporting_event_ids)) != len(self.supporting_event_ids):
            raise ValueError("patch supporting event IDs must be unique")
        return self


class ProposalStatusTransition(EvolutionModel):
    from_status: ProposalStatus
    to_status: ProposalStatus
    source: ProposalStatusSource
    reason_code: Identifier
    reason: DetailText
    occurred_at: datetime
    evaluation_id: Identifier | None = None
    publication_id: Identifier | None = None
    actor_id: Identifier | None = None
    policy_version: Identifier | None = None

    @model_validator(mode="after")
    def _validate_source_metadata(self) -> Self:
        if self.from_status is self.to_status:
            raise ValueError("proposal status transition must change status")
        if self.source is ProposalStatusSource.evaluator and self.evaluation_id is None:
            raise ValueError("evaluator transition requires evaluation_id")
        if self.source is ProposalStatusSource.manual_reviewer and self.actor_id is None:
            raise ValueError("manual reviewer transition requires actor_id")
        if self.source is ProposalStatusSource.approval_policy and self.policy_version is None:
            raise ValueError("approval policy transition requires policy_version")
        if (
            self.source
            in {
                ProposalStatusSource.publisher,
                ProposalStatusSource.rollback,
            }
            and self.publication_id is None
        ):
            raise ValueError("publication transition requires publication_id")
        if self.source is ProposalStatusSource.rollback and self.actor_id is None:
            raise ValueError("rollback transition requires actor_id")
        return self


class SkillProposal(EvolutionModel):
    schema_version: Literal["deerflow.skill-evolution.proposal.v1"] = SKILL_PROPOSAL_SCHEMA_VERSION
    proposal_id: Identifier
    cluster_id: Identifier
    user_id: Identifier
    operation: ProposalOperation
    skill_name: SkillName
    base_skill_hash: Sha256 | None = None
    proposed_files: list[ProposedSkillFile] = Field(min_length=1, max_length=64)
    supporting_event_ids: list[Identifier] = Field(min_length=1, max_length=64)
    rationale: TaskGoalText
    expected_improvements: list[DetailText] = Field(
        min_length=1,
        max_length=32,
    )
    risks: list[DetailText] = Field(default_factory=list, max_length=32)
    evidence_mapping: list[ProposalEvidenceMapping] = Field(
        default_factory=list,
        max_length=128,
    )
    distiller_model_name: Identifier | None = None
    distiller_prompt_version: Identifier | None = None
    source_cluster_hash: Sha256 | None = None
    requires_manual_review: bool = False
    review_reasons: list[Identifier] = Field(
        default_factory=list,
        max_length=16,
    )
    patch_operations: list[SkillPatchOperation] = Field(
        default_factory=list,
        max_length=64,
    )
    source_skill_hashes: list[Sha256] = Field(
        default_factory=list,
        max_length=16,
    )
    status: ProposalStatus
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    status_history: list[ProposalStatusTransition] = Field(
        default_factory=list,
        max_length=16,
    )

    @model_validator(mode="after")
    def _validate_operation(self) -> Self:
        if len(set(self.supporting_event_ids)) != len(self.supporting_event_ids):
            raise ValueError("supporting event IDs must be unique")
        paths = [item.path for item in self.proposed_files]
        if len(set(paths)) != len(paths):
            raise ValueError("proposed file paths must be unique")
        if self.operation is ProposalOperation.create:
            if self.base_skill_hash is not None:
                raise ValueError("create proposal must not set base_skill_hash")
            if self.patch_operations or self.source_skill_hashes:
                raise ValueError("create proposal must not contain patch metadata")
        elif self.base_skill_hash is None:
            raise ValueError(f"{self.operation.value} proposal requires base_skill_hash")
        if self.operation is ProposalOperation.patch:
            if not self.patch_operations:
                raise ValueError("patch proposal requires structured patch operations")
            if not self.source_skill_hashes:
                raise ValueError("patch proposal requires source Skill hashes")
            if "SKILL.md" not in paths:
                raise ValueError("patch proposal requires rendered SKILL.md")
            supporting_ids = set(self.supporting_event_ids)
            if any(not set(operation.supporting_event_ids) <= supporting_ids for operation in self.patch_operations):
                raise ValueError("patch operation cites unsupported events")
        elif self.patch_operations or self.source_skill_hashes:
            raise ValueError("patch metadata requires patch operation")
        if len(set(self.source_skill_hashes)) != len(self.source_skill_hashes):
            raise ValueError("source Skill hashes must be unique")
        metadata = (
            self.distiller_model_name,
            self.distiller_prompt_version,
            self.source_cluster_hash,
        )
        if any(value is not None for value in metadata):
            if not all(value is not None for value in metadata):
                raise ValueError("distiller metadata must be complete")
            if not self.evidence_mapping:
                raise ValueError("distilled proposal requires evidence mapping")
            proposed_paths = set(paths)
            mapped_paths = {mapping.file_path for mapping in self.evidence_mapping}
            if not proposed_paths <= mapped_paths:
                raise ValueError("every proposed file requires evidence mapping")
            supporting_ids = set(self.supporting_event_ids)
            if any(not set(mapping.supporting_event_ids) <= supporting_ids for mapping in self.evidence_mapping):
                raise ValueError("proposal evidence mapping cites unsupported events")
            mapping_keys = [(mapping.file_path, mapping.section) for mapping in self.evidence_mapping]
            if len(set(mapping_keys)) != len(mapping_keys):
                raise ValueError("proposal evidence mappings must be unique by file and section")
        elif self.evidence_mapping:
            raise ValueError("evidence mapping requires distiller metadata")
        if self.requires_manual_review and not self.review_reasons:
            raise ValueError("manual review requires at least one reason")
        if self.review_reasons and not self.requires_manual_review:
            raise ValueError("review reasons require manual review")
        if len(set(self.review_reasons)) != len(self.review_reasons):
            raise ValueError("review reasons must be unique")
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("proposal expires_at must be later than created_at")
        if self.status_history:
            for previous, current in zip(
                self.status_history,
                self.status_history[1:],
                strict=False,
            ):
                if previous.to_status is not current.from_status:
                    raise ValueError("proposal status history must form a continuous chain")
            if self.status_history[-1].to_status is not self.status:
                raise ValueError("proposal status must match the latest status history entry")
        return self


class SkillPackageSnapshotFile(EvolutionModel):
    path: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=1_024,
        ),
    ]
    content_base64: Annotated[
        str,
        StringConstraints(
            min_length=0,
            max_length=11_184_812,
        ),
    ]
    content_hash: Sha256
    size_bytes: int = Field(
        ge=0,
        le=8 * 1024 * 1024,
    )
    executable: bool = False

    @model_validator(mode="after")
    def _validate_content(self) -> Self:
        from deerflow.skills.package import normalize_package_path

        normalized = normalize_package_path(self.path)
        if normalized != self.path.replace("\\", "/"):
            raise ValueError("snapshot package path must already be normalized")
        try:
            content = base64.b64decode(
                self.content_base64,
                validate=True,
            )
        except (binascii.Error, ValueError) as exc:
            raise ValueError("snapshot file content_base64 is invalid") from exc
        if len(content) != self.size_bytes:
            raise ValueError("snapshot file size does not match content")
        import hashlib

        if hashlib.sha256(content).hexdigest() != self.content_hash:
            raise ValueError("snapshot file hash does not match content")
        return self

    @property
    def content_bytes(self) -> bytes:
        return base64.b64decode(
            self.content_base64,
            validate=True,
        )

    def to_package_file(self) -> SkillPackageFile:
        from deerflow.skills.package import SkillPackageFile

        return SkillPackageFile(
            path=self.path,
            content=self.content_bytes,
            executable=self.executable,
        )


class SkillPackageSnapshot(EvolutionModel):
    schema_version: Literal["deerflow.skill-evolution.package-snapshot.v1"] = SKILL_PACKAGE_SNAPSHOT_SCHEMA_VERSION
    snapshot_hash: Sha256
    user_id: Identifier
    skill_name: SkillName
    exists: bool
    skill_md_hash: Sha256 | None = None
    files: list[SkillPackageSnapshotFile] = Field(
        default_factory=list,
        max_length=256,
    )
    created_at: datetime

    @model_validator(mode="after")
    def _validate_snapshot(self) -> Self:
        from deerflow.skills.package import compute_skill_package_hash

        paths = [item.path for item in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("snapshot file paths must be sorted and unique")
        if self.exists:
            skill_md = next(
                (item for item in self.files if item.path == "SKILL.md"),
                None,
            )
            if skill_md is None:
                raise ValueError("existing Skill snapshot requires SKILL.md")
            if self.skill_md_hash != skill_md.content_hash:
                raise ValueError("snapshot skill_md_hash does not match SKILL.md")
        elif self.files or self.skill_md_hash is not None:
            raise ValueError("absent Skill snapshot cannot contain files or SKILL.md hash")
        package_hash = compute_skill_package_hash([item.to_package_file() for item in self.files])
        if package_hash != self.snapshot_hash:
            raise ValueError("snapshot hash does not match package files")
        return self

    def package_files(self) -> tuple[SkillPackageFile, ...]:
        return tuple(item.to_package_file() for item in self.files)

    @property
    def package_hash(self) -> str:
        return self.snapshot_hash


class SkillPublication(EvolutionModel):
    schema_version: Literal[
        "deerflow.skill-evolution.publication.v1",
        "deerflow.skill-evolution.publication.v2",
    ] = SKILL_PUBLICATION_SCHEMA_VERSION
    publication_id: Identifier
    user_id: Identifier
    proposal_id: Identifier
    evaluation_id: Identifier | None = None
    skill_name: SkillName
    operation: ProposalOperation
    status: PublicationStatus
    base_snapshot: SkillPackageSnapshot
    candidate_snapshot: SkillPackageSnapshot
    published_snapshot: SkillPackageSnapshot | None = None
    rollback_snapshot: SkillPackageSnapshot | None = None
    base_skill_hash: Sha256 | None = None
    published_skill_hash: Sha256 | None = None
    created_at: datetime
    published_at: datetime | None = None
    rolled_back_at: datetime | None = None
    rollback_actor_id: Identifier | None = None

    @model_validator(mode="after")
    def _validate_publication(self) -> Self:
        if self.schema_version == SKILL_PUBLICATION_SCHEMA_VERSION_V1 and self.evaluation_id is None:
            raise ValueError("publication v1 requires evaluation_id")
        snapshots = [
            self.base_snapshot,
            self.candidate_snapshot,
            *([self.published_snapshot] if self.published_snapshot is not None else []),
            *([self.rollback_snapshot] if self.rollback_snapshot is not None else []),
        ]
        if any(snapshot.user_id != self.user_id or snapshot.skill_name != self.skill_name for snapshot in snapshots):
            raise ValueError("publication snapshots must match user and Skill")
        if self.base_skill_hash != self.base_snapshot.skill_md_hash:
            raise ValueError("publication base_skill_hash must match base snapshot")
        if not self.candidate_snapshot.exists:
            raise ValueError("publication candidate snapshot must contain a Skill")
        if self.status is PublicationStatus.preparing:
            if self.published_snapshot is not None or self.rollback_snapshot is not None:
                raise ValueError("preparing publication cannot contain terminal snapshots")
            if any(
                value is not None
                for value in (
                    self.published_skill_hash,
                    self.published_at,
                    self.rolled_back_at,
                    self.rollback_actor_id,
                )
            ):
                raise ValueError("preparing publication cannot contain terminal metadata")
            return self
        if self.published_snapshot is None or self.published_at is None:
            raise ValueError("published publication requires the actual published snapshot and time")
        if self.published_snapshot.snapshot_hash != self.candidate_snapshot.snapshot_hash:
            raise ValueError("published snapshot must match the persisted candidate snapshot")
        if self.published_skill_hash != self.published_snapshot.skill_md_hash:
            raise ValueError("published_skill_hash must match the published snapshot")
        if self.status is PublicationStatus.published:
            if self.rollback_snapshot is not None or self.rolled_back_at is not None or self.rollback_actor_id is not None:
                raise ValueError("published record cannot contain rollback metadata")
            return self
        if self.rollback_snapshot is None or self.rolled_back_at is None or self.rollback_actor_id is None:
            raise ValueError("rolled-back publication requires rollback snapshot, actor, and time")
        if self.rollback_snapshot.snapshot_hash != self.base_snapshot.snapshot_hash:
            raise ValueError("rollback snapshot must match the persisted base snapshot")
        return self


class EvaluationMetrics(EvolutionModel):
    tool_calls: int = Field(ge=0, le=10_000)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_seconds: float = Field(ge=0.0)


class EvaluationArtifact(EvolutionModel):
    path: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=1_024,
        ),
    ]
    kind: Literal["file", "missing", "symlink"]
    content_hash: Sha256 | None = None
    size_bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_artifact(self) -> Self:
        normalized = _normalize_relative_path(self.path)
        if normalized != self.path.replace("\\", "/"):
            raise ValueError("evaluation artifact path must already be normalized")
        if self.kind == "file":
            if self.content_hash is None or self.size_bytes is None:
                raise ValueError("file artifact requires hash and size")
        elif self.content_hash is not None or self.size_bytes is not None:
            raise ValueError("non-file artifact cannot contain hash or size")
        return self


class TaskEvaluationResult(EvolutionModel):
    task_id: Identifier
    split: Literal["source", "held_out", "regression"]
    condition: Literal["no_skill", "base_skill", "candidate_skill"]
    success: bool
    metrics: EvaluationMetrics
    failure_reason: DetailText | None = None
    errors: list[DetailText] = Field(
        default_factory=list,
        max_length=32,
    )
    artifacts: list[EvaluationArtifact] = Field(
        default_factory=list,
        max_length=128,
    )
    environment_fingerprint: Sha256 | None = None

    @model_validator(mode="after")
    def _validate_failure_reason(self) -> Self:
        if self.success:
            if self.failure_reason is not None:
                raise ValueError("successful evaluation must not include failure_reason")
            if self.errors:
                raise ValueError("successful evaluation must not include errors")
        if len(set(self.errors)) != len(self.errors):
            raise ValueError("evaluation errors must be unique")
        if len({artifact.path for artifact in self.artifacts}) != len(self.artifacts):
            raise ValueError("evaluation artifact paths must be unique")
        return self


class SkillQualityDimension(EvolutionModel):
    raw_value: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    weight: float = Field(gt=0.0, le=1.0)
    sample_count: int = Field(ge=0)
    available: bool

    @model_validator(mode="after")
    def _validate_availability(self) -> Self:
        if self.available:
            if self.raw_value is None or self.score is None:
                raise ValueError("available quality dimension requires values")
            if self.sample_count < 1:
                raise ValueError("available quality dimension requires samples")
        elif self.raw_value is not None or self.score is not None:
            raise ValueError("unavailable quality dimension cannot contain values")
        return self


class SkillQualityReport(EvolutionModel):
    formula_version: Identifier
    dimensions: dict[Identifier, SkillQualityDimension] = Field(
        min_length=1,
        max_length=16,
    )
    aggregate_score: float = Field(ge=0.0, le=1.0)
    high_quality_threshold: float = Field(ge=0.0, le=1.0)
    designation: QualityDesignation
    sample_sufficient: bool
    sample_counts: dict[Identifier, int] = Field(
        min_length=1,
        max_length=16,
    )
    blockers: list[Identifier] = Field(
        default_factory=list,
        max_length=32,
    )

    @model_validator(mode="after")
    def _validate_designation(self) -> Self:
        if len(set(self.blockers)) != len(self.blockers):
            raise ValueError("quality blockers must be unique")
        if any(value < 0 for value in self.sample_counts.values()):
            raise ValueError("quality sample counts must be non-negative")
        if self.designation is QualityDesignation.high_quality:
            if not self.sample_sufficient:
                raise ValueError("high quality requires sufficient samples")
            if self.blockers:
                raise ValueError("high quality cannot contain blockers")
            if self.aggregate_score < self.high_quality_threshold:
                raise ValueError("high quality requires threshold score")
        if self.designation is QualityDesignation.insufficient_evidence and self.sample_sufficient:
            raise ValueError("insufficient designation requires insufficient samples")
        return self

    @property
    def high_quality(self) -> bool:
        return self.designation is QualityDesignation.high_quality


class SkillEvaluation(EvolutionModel):
    schema_version: Literal["deerflow.skill-evolution.evaluation.v1"] = SKILL_EVALUATION_SCHEMA_VERSION
    evaluation_id: Identifier
    proposal_id: Identifier
    user_id: Identifier
    source_replay_results: list[TaskEvaluationResult] = Field(max_length=256)
    held_out_results: list[TaskEvaluationResult] = Field(max_length=256)
    baseline_results: list[TaskEvaluationResult] = Field(max_length=256)
    candidate_results: list[TaskEvaluationResult] = Field(max_length=256)
    regression_results: list[TaskEvaluationResult] = Field(max_length=256)
    safety_results: dict[Identifier, ShortText] = Field(max_length=32)
    quality_score: float = Field(ge=0.0, le=1.0)
    quality: SkillQualityReport | None = None
    decision: EvaluationDecision
    created_at: datetime

    @model_validator(mode="after")
    def _validate_approval(self) -> Self:
        if self.quality is not None and self.quality_score != self.quality.aggregate_score:
            raise ValueError("quality_score must match quality report")
        if self.quality is not None and self.quality.high_quality and self.decision is not EvaluationDecision.approve:
            raise ValueError("high quality requires approved evaluation")
        if self.decision is EvaluationDecision.approve:
            if not self.candidate_results:
                raise ValueError("approved evaluation requires candidate results")
            if not self.held_out_results and not self.regression_results:
                raise ValueError("approved evaluation requires held-out or regression results")
        return self

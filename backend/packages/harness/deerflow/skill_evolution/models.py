"""Versioned, JSON-safe contracts for evidence-based skill evolution."""

from __future__ import annotations

import posixpath
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

EVOLUTION_EVENT_SCHEMA_VERSION = "deerflow.skill-evolution.event.v1"
EVOLUTION_CLUSTER_SCHEMA_VERSION = "deerflow.skill-evolution.cluster.v1"
SKILL_PROPOSAL_SCHEMA_VERSION = "deerflow.skill-evolution.proposal.v1"
SKILL_EVALUATION_SCHEMA_VERSION = "deerflow.skill-evolution.evaluation.v1"
EVOLUTION_TRACE_SCHEMA_VERSION = "deerflow.skill-evolution.trace-snapshot.v1"

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
    rejected = "rejected"
    published = "published"
    expired = "expired"


class EvaluationDecision(StrEnum):
    approve = "approve"
    reject = "reject"
    manual_review = "manual_review"


class TraceRunStatus(StrEnum):
    success = "success"
    error = "error"
    timeout = "timeout"
    interrupted = "interrupted"


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
    decision: EvaluationDecision
    created_at: datetime

    @model_validator(mode="after")
    def _validate_approval(self) -> Self:
        if self.decision is EvaluationDecision.approve:
            if not self.candidate_results:
                raise ValueError("approved evaluation requires candidate results")
            if not self.held_out_results:
                raise ValueError("approved evaluation requires held-out results")
        return self

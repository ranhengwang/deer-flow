from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class SkillEvolutionEvidenceConfig(BaseModel):
    """Admission and complexity thresholds for evidence-based evolution."""

    min_success_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum verified success confidence required for automatic evidence admission.",
    )
    tool_call_complexity_threshold: int = Field(
        default=5,
        ge=1,
        le=256,
        description="Tool-call count that independently qualifies a successful run as complex.",
    )
    min_cluster_events: int = Field(
        default=3,
        ge=2,
        le=64,
        description="Minimum confirmed events required before a cluster is ready.",
    )
    min_distinct_runs: int = Field(
        default=3,
        ge=2,
        le=64,
        description="Minimum distinct successful runs required before a cluster is ready.",
    )
    max_events_per_cluster: int = Field(
        default=20,
        ge=3,
        le=64,
        description="Maximum events retained in one confirmed cluster snapshot.",
    )
    accept_recovered_errors: bool = Field(
        default=True,
        description="Treat an error followed by a successful tool path as a complexity signal.",
    )
    accept_user_corrections: bool = Field(
        default=True,
        description="Treat an effective user correction as a complexity signal.",
    )
    accept_explicit_remember_requests: bool = Field(
        default=True,
        description="Treat an explicit remember/create-skill request as a complexity signal.",
    )
    accept_non_trivial_workflow: bool = Field(
        default=True,
        description="Treat a trusted extractor's non-trivial workflow flag as a complexity signal.",
    )

    @model_validator(mode="after")
    def _validate_cluster_thresholds(self) -> Self:
        if self.min_cluster_events > self.max_events_per_cluster:
            raise ValueError("min_cluster_events cannot exceed max_events_per_cluster")
        if self.min_distinct_runs > self.max_events_per_cluster:
            raise ValueError("min_distinct_runs cannot exceed max_events_per_cluster")
        return self


class SkillEvolutionEmbeddingConfig(BaseModel):
    """Optional embedding provider used for semantic candidate retrieval."""

    enabled: bool = Field(
        default=False,
        description="Enable semantic candidate retrieval. Disabled mode remains fully deterministic.",
    )
    provider: Literal["ollama"] = Field(
        default="ollama",
        description="Embedding provider implementation.",
    )
    base_url: str = Field(
        default="http://127.0.0.1:11434",
        min_length=1,
        max_length=2_048,
        description="Ollama HTTP base URL.",
    )
    model_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Embedding model name. Required when semantic retrieval is enabled.",
    )
    model_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Immutable embedding model version. Null resolves the Ollama model digest.",
    )
    timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        le=300.0,
        description="Embedding request timeout.",
    )

    @model_validator(mode="after")
    def _validate_enabled_model(self) -> Self:
        if self.enabled and self.model_name is None:
            raise ValueError("embedding model_name is required when semantic retrieval is enabled")
        return self


class SkillEvolutionVectorStoreConfig(BaseModel):
    """Qdrant settings for optional semantic event vectors."""

    provider: Literal["qdrant"] = Field(
        default="qdrant",
        description="Semantic vector-store implementation.",
    )
    url: str = Field(
        default="http://127.0.0.1:6333",
        min_length=1,
        max_length=2_048,
        description="Qdrant HTTP base URL.",
    )
    collection_name: str = Field(
        default="deerflow_skill_evolution_events",
        pattern=r"^[A-Za-z0-9_-]+$",
        min_length=1,
        max_length=128,
        description="Base collection name. A model-version suffix is added automatically.",
    )
    api_key_env: str | None = Field(
        default="QDRANT_API_KEY",
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        max_length=128,
        description="Environment variable containing the optional Qdrant API key.",
    )
    timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        le=120.0,
        description="Qdrant HTTP request timeout.",
    )


class SkillEvolutionGroupingConfig(BaseModel):
    """Deterministic candidate-grouping settings."""

    deterministic_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum deterministic fingerprint score for candidate cluster membership.",
    )
    semantic_threshold: float = Field(
        default=0.82,
        ge=0.0,
        le=1.0,
        description="Minimum vector similarity for semantic candidate retrieval.",
    )
    semantic_top_k: int = Field(
        default=16,
        ge=1,
        le=64,
        description="Maximum semantic candidates returned for one event.",
    )
    llm_confirmation: bool = Field(
        default=True,
        description="Require structured LLM confirmation before cluster readiness.",
    )
    confirmation_model_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Optional cluster-confirmation model. Null selects the primary model.",
    )
    embedding: SkillEvolutionEmbeddingConfig = Field(
        default_factory=SkillEvolutionEmbeddingConfig,
        description="Optional embedding provider settings.",
    )
    vector_store: SkillEvolutionVectorStoreConfig = Field(
        default_factory=SkillEvolutionVectorStoreConfig,
        description="Semantic vector-store settings.",
    )


class SkillEvolutionQualityConfig(BaseModel):
    """Pass thresholds and sample gates for versioned quality scoring."""

    min_source_replay_success_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Minimum candidate success rate across all source replay tasks.",
    )
    min_held_out_success_rate: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum candidate success rate across held-out same-family tasks.",
    )
    max_regression_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Maximum old-success/new-failure rate across historical regression tasks.",
    )
    high_quality_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum aggregate v1 score required for a high-quality designation.",
    )
    min_total_candidate_tasks: int = Field(
        default=5,
        ge=1,
        le=256,
        description="Minimum candidate task count required for high-quality designation.",
    )
    min_held_out_tasks: int = Field(
        default=2,
        ge=1,
        le=256,
        description="Minimum held-out tasks required for a high-quality new Skill.",
    )
    min_regression_tasks: int = Field(
        default=2,
        ge=1,
        le=256,
        description="Minimum old-Skill-success regression pairs required for a high-quality patch.",
    )
    min_distinct_environments: int = Field(
        default=2,
        ge=1,
        le=64,
        description="Minimum distinct replay environments required for high-quality designation.",
    )


class SkillEvolutionPublicationConfig(BaseModel):
    """Select evaluated or direct Skill publication behavior."""

    mode: Literal["manual", "eligible_auto", "direct"] = Field(
        default="manual",
        description=("Publication workflow: manual requires explicit review, eligible_auto retains evaluation gates, and direct publishes a staged Proposal immediately after distillation while retaining mutation security checks."),
    )
    allow_non_executable_auto_publish: bool = Field(
        default=False,
        description="Allow policy approval of low-risk non-executable updates. This never publishes the Skill by itself.",
    )
    allow_executable_auto_publish: Literal[False] = Field(
        default=False,
        description="Executable auto-publication is forbidden in the first policy version.",
    )
    require_held_out_evaluation: Literal[True] = Field(
        default=True,
        description="Require successful candidate held-out results before a Proposal can be automatically approved.",
    )
    proposal_ttl_days: int = Field(
        default=180,
        ge=1,
        le=730,
        description="Fallback lifetime for legacy Proposals that do not persist an explicit expires_at value.",
    )


def is_skill_manage_enabled(skill_evolution_config: object | None) -> bool:
    """Return whether lead agents may mutate Skills during their own run."""
    if not getattr(skill_evolution_config, "enabled", False):
        return False

    publication = getattr(skill_evolution_config, "publication", None)
    return getattr(publication, "mode", "manual") != "direct"


class SkillEvolutionCoordinatorConfig(BaseModel):
    """Restart-safe background job scheduling and shutdown bounds."""

    queue_capacity: int = Field(
        default=64,
        ge=1,
        le=4_096,
        description="Maximum in-memory wake-up hints; durable SQL jobs are never dropped when this queue is full.",
    )
    max_concurrent_jobs: int = Field(
        default=2,
        ge=1,
        le=32,
        description="Maximum evolution jobs processed concurrently by one Gateway process.",
    )
    poll_interval_seconds: float = Field(
        default=1.0,
        gt=0.0,
        le=60.0,
        description="Database polling interval used for pending, retry, and expired-lease recovery.",
    )
    lease_seconds: float = Field(
        default=120.0,
        gt=0.0,
        le=3_600.0,
        description="Renewable claim lease for one background evolution job.",
    )
    max_attempts: int = Field(
        default=5,
        ge=1,
        le=32,
        description="Maximum claimed processing attempts before a job enters the dead state.",
    )
    retry_base_delay_seconds: float = Field(
        default=5.0,
        ge=0.0,
        le=3_600.0,
        description="Initial deterministic exponential-backoff delay after a processing failure.",
    )
    retry_max_delay_seconds: float = Field(
        default=300.0,
        ge=0.0,
        le=86_400.0,
        description="Maximum deterministic exponential-backoff delay.",
    )
    shutdown_timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        le=300.0,
        description="Maximum Gateway shutdown drain time before local work is cancelled and left durable for recovery.",
    )

    @model_validator(mode="after")
    def _validate_retry_delays(self) -> Self:
        if self.retry_max_delay_seconds < self.retry_base_delay_seconds:
            raise ValueError("retry_max_delay_seconds cannot be less than retry_base_delay_seconds")
        return self


class SkillEvolutionConfig(BaseModel):
    """Configuration for agent-managed skill evolution."""

    enabled: bool = Field(
        default=False,
        description="Whether the agent can create and modify skills under skills/custom.",
    )
    moderation_model_name: str | None = Field(
        default=None,
        description="Optional model name for skill security moderation. Defaults to the primary chat model.",
    )
    extraction_model_name: str | None = Field(
        default=None,
        description="Optional model name for structured evolution-event extraction. Defaults to the primary chat model.",
    )
    distillation_model_name: str | None = Field(
        default=None,
        description="Optional model name for cross-trajectory Skill proposal distillation. Defaults to the primary chat model.",
    )
    security_fail_closed: bool = Field(
        default=True,
        description=("When the moderation model is unavailable, block skill writes if True (fail-closed). If False, non-executable content is allowed with a warning while executable content is still blocked."),
    )
    evidence: SkillEvolutionEvidenceConfig = Field(
        default_factory=SkillEvolutionEvidenceConfig,
        description="Evidence admission and complexity detection settings.",
    )
    grouping: SkillEvolutionGroupingConfig = Field(
        default_factory=SkillEvolutionGroupingConfig,
        description="Event fingerprint and candidate-grouping settings.",
    )
    quality: SkillEvolutionQualityConfig = Field(
        default_factory=SkillEvolutionQualityConfig,
        description="Candidate evaluation thresholds. Aggregate scoring is configured separately when implemented.",
    )
    publication: SkillEvolutionPublicationConfig = Field(
        default_factory=SkillEvolutionPublicationConfig,
        description="Proposal approval and future publication gates.",
    )
    coordinator: SkillEvolutionCoordinatorConfig = Field(
        default_factory=SkillEvolutionCoordinatorConfig,
        description="Durable background coordinator settings captured at Gateway startup.",
    )

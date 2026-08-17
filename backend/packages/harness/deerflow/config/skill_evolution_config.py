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
    """Pass thresholds used before aggregate quality scoring exists."""

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

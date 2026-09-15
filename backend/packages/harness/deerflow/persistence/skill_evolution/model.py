"""ORM rows for skill-evolution evidence, proposals, and evaluations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.persistence.base import Base


class SkillEvolutionEventRow(Base):
    __tablename__ = "skill_evolution_events"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False)
    extractor_version: Mapped[str] = mapped_column(String(128), nullable=False)
    event_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    task_signature: Mapped[str] = mapped_column(String(256), nullable=False)
    target_skill_name: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    source_snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    task_input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "run_id",
            "extractor_version",
            name="uq_skill_evolution_event_user_run_extractor",
        ),
        Index(
            "ix_skill_evolution_events_user_created",
            "user_id",
            "created_at",
        ),
        Index(
            "ix_skill_evolution_events_user_kind",
            "user_id",
            "event_kind",
        ),
        Index(
            "ix_skill_evolution_events_user_task_signature",
            "user_id",
            "task_signature",
        ),
        Index(
            "ix_skill_evolution_events_user_target_skill",
            "user_id",
            "target_skill_name",
        ),
    )


class SkillEvolutionClusterRow(Base):
    __tablename__ = "skill_evolution_clusters"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    event_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    target_skill_name: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    canonical_signature: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )
    independent_run_count: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_skill_evolution_clusters_user_status",
            "user_id",
            "status",
        ),
        Index(
            "ix_skill_evolution_clusters_user_target_skill",
            "user_id",
            "target_skill_name",
        ),
        Index(
            "ix_skill_evolution_clusters_user_updated",
            "user_id",
            "updated_at",
        ),
    )


class SkillEvolutionProposalRow(Base):
    __tablename__ = "skill_evolution_proposals"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    cluster_id: Mapped[str] = mapped_column(String(128), nullable=False)
    operation: Mapped[str] = mapped_column(String(32), nullable=False)
    skill_name: Mapped[str] = mapped_column(String(128), nullable=False)
    base_skill_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_skill_evolution_proposals_user_status",
            "user_id",
            "status",
        ),
        Index(
            "ix_skill_evolution_proposals_user_skill",
            "user_id",
            "skill_name",
        ),
        Index(
            "ix_skill_evolution_proposals_user_cluster",
            "user_id",
            "cluster_id",
        ),
        Index(
            "ix_skill_evolution_proposals_user_created",
            "user_id",
            "created_at",
        ),
    )


class SkillEvolutionEvaluationRow(Base):
    __tablename__ = "skill_evolution_evaluations"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    proposal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_skill_evolution_evaluations_user_proposal",
            "user_id",
            "proposal_id",
        ),
        Index(
            "ix_skill_evolution_evaluations_user_decision",
            "user_id",
            "decision",
        ),
        Index(
            "ix_skill_evolution_evaluations_user_created",
            "user_id",
            "created_at",
        ),
    )


class SkillEvolutionPublicationRow(Base):
    __tablename__ = "skill_evolution_publications"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    proposal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    evaluation_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    skill_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    base_package_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    published_package_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "proposal_id",
            name="uq_skill_evolution_publication_user_proposal",
        ),
        Index(
            "ix_skill_evolution_publications_user_status",
            "user_id",
            "status",
        ),
        Index(
            "ix_skill_evolution_publications_user_skill",
            "user_id",
            "skill_name",
        ),
        Index(
            "ix_skill_evolution_publications_user_updated",
            "user_id",
            "updated_at",
        ),
    )


class SkillEvolutionCreditRow(Base):
    __tablename__ = "skill_evolution_credits"

    user_id: Mapped[str] = mapped_column(
        String(128),
        primary_key=True,
    )
    id: Mapped[str] = mapped_column(
        String(128),
        primary_key=True,
    )
    kind: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    run_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    skill_name: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    publication_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    revision: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_skill_evolution_credits_user_kind_created",
            "user_id",
            "kind",
            "created_at",
        ),
        Index(
            "ix_skill_evolution_credits_user_skill_created",
            "user_id",
            "skill_name",
            "created_at",
        ),
        Index(
            "ix_skill_evolution_credits_user_publication",
            "user_id",
            "publication_id",
        ),
    )


class SkillEvolutionJobRow(Base):
    __tablename__ = "skill_evolution_jobs"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    next_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_token: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_error_code: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "idempotency_key",
            name="uq_skill_evolution_job_user_idempotency",
        ),
        Index(
            "ix_skill_evolution_jobs_status_due",
            "status",
            "next_attempt_at",
        ),
        Index(
            "ix_skill_evolution_jobs_status_lease",
            "status",
            "lease_expires_at",
        ),
        Index(
            "ix_skill_evolution_jobs_user_run",
            "user_id",
            "run_id",
        ),
    )

"""skill evolution evidence and proposal lifecycle.

Revision ID: 0012_skill_evolution
Revises: 0011_mcp_tasks
Create Date: 2026-08-16
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012_skill_evolution"
down_revision: str | Sequence[str] | None = "0011_mcp_tasks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _ensure_indexes(
    table_name: str,
    indexes: tuple[tuple[str, list[str]], ...],
) -> None:
    bind = op.get_bind()
    existing = {item["name"] for item in sa.inspect(bind).get_indexes(table_name) if item.get("name")}
    missing = [item for item in indexes if item[0] not in existing]
    if not missing:
        return
    with op.batch_alter_table(table_name, schema=None) as batch_op:
        for name, columns in missing:
            batch_op.create_index(name, columns, unique=False)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("skill_evolution_events"):
        op.create_table(
            "skill_evolution_events",
            sa.Column("user_id", sa.String(length=128), nullable=False),
            sa.Column("id", sa.String(length=128), nullable=False),
            sa.Column("run_id", sa.String(length=128), nullable=False),
            sa.Column("thread_id", sa.String(length=128), nullable=False),
            sa.Column(
                "extractor_version",
                sa.String(length=128),
                nullable=False,
            ),
            sa.Column("event_kind", sa.String(length=32), nullable=False),
            sa.Column(
                "task_signature",
                sa.String(length=256),
                nullable=False,
            ),
            sa.Column(
                "target_skill_name",
                sa.String(length=128),
                nullable=True,
            ),
            sa.Column(
                "source_snapshot_hash",
                sa.String(length=64),
                nullable=False,
            ),
            sa.Column(
                "task_input_hash",
                sa.String(length=64),
                nullable=False,
            ),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("user_id", "id"),
            sa.UniqueConstraint(
                "user_id",
                "run_id",
                "extractor_version",
                name="uq_skill_evolution_event_user_run_extractor",
            ),
        )
    _ensure_indexes(
        "skill_evolution_events",
        (
            (
                "ix_skill_evolution_events_user_created",
                ["user_id", "created_at"],
            ),
            (
                "ix_skill_evolution_events_user_kind",
                ["user_id", "event_kind"],
            ),
            (
                "ix_skill_evolution_events_user_task_signature",
                ["user_id", "task_signature"],
            ),
            (
                "ix_skill_evolution_events_user_target_skill",
                ["user_id", "target_skill_name"],
            ),
        ),
    )

    inspector = sa.inspect(bind)
    if not inspector.has_table("skill_evolution_clusters"):
        op.create_table(
            "skill_evolution_clusters",
            sa.Column("user_id", sa.String(length=128), nullable=False),
            sa.Column("id", sa.String(length=128), nullable=False),
            sa.Column("event_kind", sa.String(length=32), nullable=False),
            sa.Column(
                "target_skill_name",
                sa.String(length=128),
                nullable=True,
            ),
            sa.Column(
                "canonical_signature",
                sa.String(length=512),
                nullable=False,
            ),
            sa.Column(
                "independent_run_count",
                sa.Integer(),
                nullable=False,
            ),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("user_id", "id"),
        )
    _ensure_indexes(
        "skill_evolution_clusters",
        (
            (
                "ix_skill_evolution_clusters_user_status",
                ["user_id", "status"],
            ),
            (
                "ix_skill_evolution_clusters_user_target_skill",
                ["user_id", "target_skill_name"],
            ),
            (
                "ix_skill_evolution_clusters_user_updated",
                ["user_id", "updated_at"],
            ),
        ),
    )

    inspector = sa.inspect(bind)
    if not inspector.has_table("skill_evolution_proposals"):
        op.create_table(
            "skill_evolution_proposals",
            sa.Column("user_id", sa.String(length=128), nullable=False),
            sa.Column("id", sa.String(length=128), nullable=False),
            sa.Column(
                "cluster_id",
                sa.String(length=128),
                nullable=False,
            ),
            sa.Column("operation", sa.String(length=32), nullable=False),
            sa.Column(
                "skill_name",
                sa.String(length=128),
                nullable=False,
            ),
            sa.Column(
                "base_skill_hash",
                sa.String(length=64),
                nullable=True,
            ),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("user_id", "id"),
        )
    _ensure_indexes(
        "skill_evolution_proposals",
        (
            (
                "ix_skill_evolution_proposals_user_status",
                ["user_id", "status"],
            ),
            (
                "ix_skill_evolution_proposals_user_skill",
                ["user_id", "skill_name"],
            ),
            (
                "ix_skill_evolution_proposals_user_cluster",
                ["user_id", "cluster_id"],
            ),
            (
                "ix_skill_evolution_proposals_user_created",
                ["user_id", "created_at"],
            ),
        ),
    )

    inspector = sa.inspect(bind)
    if not inspector.has_table("skill_evolution_evaluations"):
        op.create_table(
            "skill_evolution_evaluations",
            sa.Column("user_id", sa.String(length=128), nullable=False),
            sa.Column("id", sa.String(length=128), nullable=False),
            sa.Column(
                "proposal_id",
                sa.String(length=128),
                nullable=False,
            ),
            sa.Column("decision", sa.String(length=32), nullable=False),
            sa.Column("quality_score", sa.Float(), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("user_id", "id"),
        )
    _ensure_indexes(
        "skill_evolution_evaluations",
        (
            (
                "ix_skill_evolution_evaluations_user_proposal",
                ["user_id", "proposal_id"],
            ),
            (
                "ix_skill_evolution_evaluations_user_decision",
                ["user_id", "decision"],
            ),
            (
                "ix_skill_evolution_evaluations_user_created",
                ["user_id", "created_at"],
            ),
        ),
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for table_name in (
        "skill_evolution_evaluations",
        "skill_evolution_proposals",
        "skill_evolution_clusters",
        "skill_evolution_events",
    ):
        if inspector.has_table(table_name):
            op.drop_table(table_name)

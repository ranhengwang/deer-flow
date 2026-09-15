"""durable background jobs for skill evolution.

Revision ID: 0014_skill_evolution_jobs
Revises: 0013_skill_publications
Create Date: 2026-08-17
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014_skill_evolution_jobs"
down_revision: str | Sequence[str] | None = "0013_skill_publications"
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
    with op.batch_alter_table(
        table_name,
        schema=None,
    ) as batch_op:
        for name, columns in missing:
            batch_op.create_index(
                name,
                columns,
                unique=False,
            )


def upgrade() -> None:
    bind = op.get_bind()
    table_name = "skill_evolution_jobs"
    if not sa.inspect(bind).has_table(table_name):
        op.create_table(
            table_name,
            sa.Column(
                "user_id",
                sa.String(length=128),
                nullable=False,
            ),
            sa.Column(
                "id",
                sa.String(length=128),
                nullable=False,
            ),
            sa.Column(
                "idempotency_key",
                sa.String(length=64),
                nullable=False,
            ),
            sa.Column(
                "run_id",
                sa.String(length=128),
                nullable=False,
            ),
            sa.Column(
                "thread_id",
                sa.String(length=128),
                nullable=False,
            ),
            sa.Column(
                "snapshot_hash",
                sa.String(length=64),
                nullable=False,
            ),
            sa.Column(
                "pipeline_version",
                sa.String(length=128),
                nullable=False,
            ),
            sa.Column(
                "status",
                sa.String(length=32),
                nullable=False,
            ),
            sa.Column(
                "attempt_count",
                sa.Integer(),
                nullable=False,
            ),
            sa.Column(
                "max_attempts",
                sa.Integer(),
                nullable=False,
            ),
            sa.Column(
                "next_attempt_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
            sa.Column(
                "lease_owner",
                sa.String(length=128),
                nullable=True,
            ),
            sa.Column(
                "lease_token",
                sa.String(length=128),
                nullable=True,
            ),
            sa.Column(
                "lease_expires_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
            sa.Column(
                "last_error_code",
                sa.String(length=128),
                nullable=True,
            ),
            sa.Column(
                "revision",
                sa.Integer(),
                nullable=False,
            ),
            sa.Column(
                "payload",
                sa.JSON(),
                nullable=False,
            ),
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
            sa.Column(
                "completed_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
            sa.PrimaryKeyConstraint(
                "user_id",
                "id",
            ),
            sa.UniqueConstraint(
                "user_id",
                "idempotency_key",
                name="uq_skill_evolution_job_user_idempotency",
            ),
        )
    _ensure_indexes(
        table_name,
        (
            (
                "ix_skill_evolution_jobs_status_due",
                [
                    "status",
                    "next_attempt_at",
                ],
            ),
            (
                "ix_skill_evolution_jobs_status_lease",
                [
                    "status",
                    "lease_expires_at",
                ],
            ),
            (
                "ix_skill_evolution_jobs_user_run",
                [
                    "user_id",
                    "run_id",
                ],
            ),
        ),
    )


def downgrade() -> None:
    bind = op.get_bind()
    table_name = "skill_evolution_jobs"
    if sa.inspect(bind).has_table(table_name):
        op.drop_table(table_name)

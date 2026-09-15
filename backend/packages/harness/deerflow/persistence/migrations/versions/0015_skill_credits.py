"""persist selection, utilization, and distillation Credits.

Revision ID: 0015_skill_credits
Revises: 0014_skill_evolution_jobs
Create Date: 2026-08-18
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015_skill_credits"
down_revision: str | Sequence[str] | None = "0014_skill_evolution_jobs"
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
    table_name = "skill_evolution_credits"
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
                "kind",
                sa.String(length=32),
                nullable=False,
            ),
            sa.Column(
                "run_id",
                sa.String(length=128),
                nullable=True,
            ),
            sa.Column(
                "skill_name",
                sa.String(length=128),
                nullable=True,
            ),
            sa.Column(
                "publication_id",
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
            sa.PrimaryKeyConstraint(
                "user_id",
                "id",
            ),
        )
    _ensure_indexes(
        table_name,
        (
            (
                "ix_skill_evolution_credits_user_kind_created",
                [
                    "user_id",
                    "kind",
                    "created_at",
                ],
            ),
            (
                "ix_skill_evolution_credits_user_skill_created",
                [
                    "user_id",
                    "skill_name",
                    "created_at",
                ],
            ),
            (
                "ix_skill_evolution_credits_user_publication",
                [
                    "user_id",
                    "publication_id",
                ],
            ),
        ),
    )


def downgrade() -> None:
    bind = op.get_bind()
    table_name = "skill_evolution_credits"
    if sa.inspect(bind).has_table(table_name):
        op.drop_table(table_name)

"""skill publication snapshots and rollback lifecycle.

Revision ID: 0013_skill_publications
Revises: 0012_skill_evolution
Create Date: 2026-08-17
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013_skill_publications"
down_revision: str | Sequence[str] | None = "0012_skill_evolution"
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
    inspector = sa.inspect(bind)
    table_name = "skill_evolution_publications"
    if not inspector.has_table(table_name):
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
                "proposal_id",
                sa.String(length=128),
                nullable=False,
            ),
            sa.Column(
                "evaluation_id",
                sa.String(length=128),
                nullable=False,
            ),
            sa.Column(
                "skill_name",
                sa.String(length=128),
                nullable=False,
            ),
            sa.Column(
                "status",
                sa.String(length=32),
                nullable=False,
            ),
            sa.Column(
                "base_package_hash",
                sa.String(length=64),
                nullable=False,
            ),
            sa.Column(
                "published_package_hash",
                sa.String(length=64),
                nullable=True,
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
            sa.UniqueConstraint(
                "user_id",
                "proposal_id",
                name="uq_skill_evolution_publication_user_proposal",
            ),
        )
    _ensure_indexes(
        table_name,
        (
            (
                "ix_skill_evolution_publications_user_status",
                [
                    "user_id",
                    "status",
                ],
            ),
            (
                "ix_skill_evolution_publications_user_skill",
                [
                    "user_id",
                    "skill_name",
                ],
            ),
            (
                "ix_skill_evolution_publications_user_updated",
                [
                    "user_id",
                    "updated_at",
                ],
            ),
        ),
    )


def downgrade() -> None:
    bind = op.get_bind()
    table_name = "skill_evolution_publications"
    if sa.inspect(bind).has_table(table_name):
        op.drop_table(table_name)

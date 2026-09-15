"""allow direct Skill publication without an Evaluation.

Revision ID: 0016_direct_skill_publication
Revises: 0015_skill_credits
Create Date: 2026-08-30
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016_direct_skill_publication"
down_revision: str | Sequence[str] | None = "0015_skill_credits"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "skill_evolution_publications"
_COLUMN = "evaluation_id"


def _column() -> dict | None:
    bind = op.get_bind()
    if not sa.inspect(bind).has_table(_TABLE):
        return None
    return next(
        (column for column in sa.inspect(bind).get_columns(_TABLE) if column["name"] == _COLUMN),
        None,
    )


def upgrade() -> None:
    column = _column()
    if column is None or column.get("nullable", True):
        return
    with op.batch_alter_table(_TABLE) as batch:
        batch.alter_column(
            _COLUMN,
            existing_type=sa.String(length=128),
            nullable=True,
        )


def downgrade() -> None:
    column = _column()
    if column is None or not column.get("nullable", True):
        return
    bind = op.get_bind()
    null_count = bind.execute(sa.text("SELECT COUNT(*) FROM skill_evolution_publications WHERE evaluation_id IS NULL")).scalar_one()
    if null_count:
        raise RuntimeError("Cannot downgrade while direct Skill publications exist.")
    with op.batch_alter_table(_TABLE) as batch:
        batch.alter_column(
            _COLUMN,
            existing_type=sa.String(length=128),
            nullable=False,
        )

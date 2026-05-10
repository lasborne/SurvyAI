"""Add grace period and reactivation columns to users (Postgres).

Revision ID: 20260503_002
Revises: 20250411_001
Create Date: 2026-05-03

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260503_002"
down_revision: Union[str, None] = "20250411_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("grace_period_ends_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("last_reactivation_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "last_reactivation_at")
    op.drop_column("users", "grace_period_ends_at")

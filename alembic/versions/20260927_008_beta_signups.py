"""Store first-user signups from the marketing site.

Revision ID: 20260927_008
Revises: 20260803_007
Create Date: 2026-09-27
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260927_008"
down_revision: Union[str, None] = "20260803_007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    if "beta_signups" in set(insp.get_table_names()):
        return
    op.create_table(
        "beta_signups",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("profession", sa.String(length=80), nullable=True),
        sa.Column("profession_other", sa.String(length=200), nullable=True),
        sa.Column("use_for", sa.String(length=120), nullable=True),
        sa.Column("use_for_other", sa.String(length=400), nullable=True),
        sa.Column("windows_specs", sa.String(length=400), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email", name="uq_beta_signups_email"),
    )
    op.create_index("ix_beta_signups_email", "beta_signups", ["email"], unique=False)


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    if "beta_signups" not in set(insp.get_table_names()):
        return
    op.drop_index("ix_beta_signups_email", table_name="beta_signups")
    op.drop_table("beta_signups")

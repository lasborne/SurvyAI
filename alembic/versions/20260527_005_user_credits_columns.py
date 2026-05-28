"""Add subscription credit columns to users and cost_usd to usage_events.

Revision ID: 20260527_005
Revises: 20260517_004
Create Date: 2026-05-27

The ORM expects these columns; without them /v1/auth/login returns HTTP 500.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260527_005"
down_revision: Union[str, None] = "20260517_004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    user_cols = {c["name"] for c in insp.get_columns("users")}
    if "monthly_credits_usd" not in user_cols:
        op.add_column(
            "users",
            sa.Column("monthly_credits_usd", sa.Float(), nullable=False, server_default="0"),
        )
    if "monthly_credits_used_usd" not in user_cols:
        op.add_column(
            "users",
            sa.Column("monthly_credits_used_usd", sa.Float(), nullable=False, server_default="0"),
        )
    if "credits_billing_interval" not in user_cols:
        op.add_column(
            "users",
            sa.Column(
                "credits_billing_interval",
                sa.String(length=16),
                nullable=False,
                server_default="monthly",
            ),
        )

    ev_cols = {c["name"] for c in insp.get_columns("usage_events")}
    if "cost_usd" not in ev_cols:
        op.add_column(
            "usage_events",
            sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
        )


def downgrade() -> None:
    op.drop_column("usage_events", "cost_usd")
    op.drop_column("users", "credits_billing_interval")
    op.drop_column("users", "monthly_credits_used_usd")
    op.drop_column("users", "monthly_credits_usd")

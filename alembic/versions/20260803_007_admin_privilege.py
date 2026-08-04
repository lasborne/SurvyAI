"""Add admin privilege flags on users.

Revision ID: 20260803_007
Revises: 20260711_006
Create Date: 2026-08-03
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260803_007"
down_revision: Union[str, None] = "20260711_006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    user_cols = {c["name"] for c in insp.get_columns("users")}
    if "admin_privilege_active" not in user_cols:
        op.add_column(
            "users",
            sa.Column(
                "admin_privilege_active",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
        )
    if "admin_privilege_note" not in user_cols:
        op.add_column(
            "users",
            sa.Column("admin_privilege_note", sa.String(length=500), nullable=True),
        )


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    user_cols = {c["name"] for c in insp.get_columns("users")}
    if "admin_privilege_note" in user_cols:
        op.drop_column("users", "admin_privilege_note")
    if "admin_privilege_active" in user_cols:
        op.drop_column("users", "admin_privilege_active")

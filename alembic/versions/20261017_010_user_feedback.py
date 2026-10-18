"""Desktop beta feedback.

Revision ID: 20261017_010
Revises: 20261011_009
Create Date: 2026-10-17
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20261017_010"
down_revision: Union[str, None] = "20261011_009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    if "user_feedback" not in set(insp.get_table_names()):
        op.create_table(
            "user_feedback",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("email", sa.String(length=320), nullable=True),
            sa.Column("name", sa.String(length=120), nullable=False),
            sa.Column("trying_to_do", sa.String(length=120), nullable=False),
            sa.Column("trying_other", sa.String(length=400), nullable=True),
            sa.Column("what_happened", sa.Text(), nullable=False),
            sa.Column("expected", sa.Text(), nullable=False),
            sa.Column("rating", sa.Integer(), nullable=False),
            sa.Column("context", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_user_feedback_email", "user_feedback", ["email"])
        op.create_index("ix_user_feedback_user_id", "user_feedback", ["user_id"])
        return
    cols = {c["name"] for c in insp.get_columns("user_feedback")}
    if "context" not in cols:
        op.add_column("user_feedback", sa.Column("context", sa.Text(), nullable=True))
    if "email" in cols:
        op.alter_column("user_feedback", "email", existing_type=sa.String(length=320), nullable=True)


def downgrade() -> None:
    op.drop_index("ix_user_feedback_user_id", table_name="user_feedback")
    op.drop_index("ix_user_feedback_email", table_name="user_feedback")
    op.drop_table("user_feedback")

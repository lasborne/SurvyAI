"""Anonymous marketing-site visits and installer clicks.

Revision ID: 20261011_009
Revises: 20260927_008
Create Date: 2026-10-11
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261011_009"
down_revision: Union[str, None] = "20260927_008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    if "site_events" in set(insp.get_table_names()):
        return
    op.create_table(
        "site_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("visitor_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("path", sa.String(length=80), nullable=False),
        sa.Column("event_day", sa.Date(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "visitor_id",
            "kind",
            "path",
            "event_day",
            name="uq_site_event_visitor_day",
        ),
    )
    op.create_index("ix_site_events_kind_created", "site_events", ["kind", "created_at"], unique=False)


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    if "site_events" not in set(insp.get_table_names()):
        return
    op.drop_index("ix_site_events_kind_created", table_name="site_events")
    op.drop_table("site_events")

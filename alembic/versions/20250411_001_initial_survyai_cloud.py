"""Initial SurvyAI Cloud schema (Postgres).

Revision ID: 20250411_001
Revises:
Create Date: 2025-04-11

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20250411_001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=200), nullable=True),
        sa.Column("paystack_customer_code", sa.String(length=64), nullable=True),
        sa.Column("paystack_subscription_code", sa.String(length=64), nullable=True),
        sa.Column("paystack_email_token", sa.String(length=128), nullable=True),
        sa.Column("last_payment_reference", sa.String(length=128), nullable=True),
        sa.Column("stripe_customer_id", sa.String(length=64), nullable=True),
        sa.Column("stripe_subscription_id", sa.String(length=64), nullable=True),
        sa.Column("plan_slug", sa.String(length=64), nullable=False, server_default="free"),
        sa.Column("subscription_status", sa.String(length=64), nullable=False, server_default="none"),
        sa.Column("subscription_current_period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("max_devices", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("monthly_agent_runs_quota", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("monthly_agent_runs_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("usage_period_anchor", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_paystack_customer_code", "users", ["paystack_customer_code"], unique=False)
    op.create_index("ix_users_paystack_subscription_code", "users", ["paystack_subscription_code"], unique=False)
    op.create_index("ix_users_last_payment_reference", "users", ["last_payment_reference"], unique=False)
    op.create_index("ix_users_stripe_customer_id", "users", ["stripe_customer_id"], unique=False)
    op.create_index("ix_users_stripe_subscription_id", "users", ["stripe_subscription_id"], unique=False)

    op.create_table(
        "devices",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fingerprint", sa.String(length=128), nullable=False),
        sa.Column("label", sa.String(length=200), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "fingerprint", name="uq_device_user_fingerprint"),
    )

    op.create_table(
        "refresh_tokens",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_refresh_tokens_token_hash", "refresh_tokens", ["token_hash"], unique=False)

    op.create_table(
        "payment_event_logs",
        sa.Column("event_id", sa.String(length=128), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False, server_default="paystack"),
        sa.Column("type", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("event_id"),
    )

    op.create_table(
        "diagnostics_bundles",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("client_version", sa.String(length=64), nullable=True),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "usage_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("meta", postgresql.JSONB(), nullable=True),
        sa.Column("device_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["device_id"], ["devices.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_usage_user_created", "usage_events", ["user_id", "created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_usage_user_created", table_name="usage_events")
    op.drop_table("usage_events")
    op.drop_table("diagnostics_bundles")
    op.drop_table("payment_event_logs")
    op.drop_index("ix_refresh_tokens_token_hash", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")
    op.drop_table("devices")
    op.drop_index("ix_users_stripe_subscription_id", table_name="users")
    op.drop_index("ix_users_stripe_customer_id", table_name="users")
    op.drop_index("ix_users_last_payment_reference", table_name="users")
    op.drop_index("ix_users_paystack_subscription_code", table_name="users")
    op.drop_index("ix_users_paystack_customer_code", table_name="users")
    op.drop_table("users")

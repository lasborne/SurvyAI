from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


# Use JSONB on Postgres, JSON elsewhere
def _json_type():
    return JSON().with_variant(JSONB(), "postgresql")


class SubscriptionStatus(str, enum.Enum):
    none = "none"
    trialing = "trialing"
    active = "active"
    non_renewing = "non_renewing"
    past_due = "past_due"
    canceled = "canceled"
    unpaid = "unpaid"
    incomplete = "incomplete"


class EnumAsString(TypeDecorator):
    """Store Python enums as plain strings (SQLite + Postgres friendly)."""

    impl = String(64)
    cache_ok = True

    def __init__(self, enum_type: type[enum.Enum]):
        super().__init__()
        self._enum_type = enum_type

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, enum.Enum):
            return value.value
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return self._enum_type(value)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(200))

    # Billing provider identifiers (Paystack first; old Stripe fields kept for compatibility)
    paystack_customer_code: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    paystack_subscription_code: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    paystack_email_token: Mapped[Optional[str]] = mapped_column(String(128))
    last_payment_reference: Mapped[Optional[str]] = mapped_column(String(128), index=True)

    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)  # legacy
    stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)  # legacy

    plan_slug: Mapped[str] = mapped_column(String(64), default="free", nullable=False)
    subscription_status: Mapped[SubscriptionStatus] = mapped_column(
        EnumAsString(SubscriptionStatus),
        default=SubscriptionStatus.none,
        nullable=False,
    )
    subscription_current_period_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Added by migration 20260503_002
    grace_period_ends_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_reactivation_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    max_devices: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    monthly_agent_runs_quota: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    monthly_agent_runs_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    monthly_credits_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    monthly_credits_used_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    credits_billing_interval: Mapped[str] = mapped_column(String(16), default="monthly", nullable=False)
    usage_period_anchor: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    password_changed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    # Set when support grants/adjusts Pro or credits via admin UI (not Paystack purchase).
    admin_privilege_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    admin_privilege_note: Mapped[Optional[str]] = mapped_column(String(500))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    devices: Mapped[list["Device"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    password_reset_tokens: Mapped[list["PasswordResetToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Device(Base):
    __tablename__ = "devices"
    __table_args__ = (UniqueConstraint("user_id", "fingerprint", name="uq_device_user_fingerprint"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)
    label: Mapped[Optional[str]] = mapped_column(String(200))
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="devices")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="refresh_tokens")


class PasswordResetToken(Base):
    """One-time emailed reset codes (store hash only)."""

    __tablename__ = "password_reset_tokens"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    request_ip: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="password_reset_tokens")


class UsageEvent(Base):
    __tablename__ = "usage_events"
    __table_args__ = (Index("ix_usage_user_created", "user_id", "created_at"),)

    # NOTE (SQLite): autoincrement only works reliably when the PK column is
    # declared as INTEGER PRIMARY KEY. BigInteger can map to BIGINT which breaks
    # autoincrement and causes NOT NULL constraint failures on insert.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    meta: Mapped[Optional[dict[str, Any]]] = mapped_column(_json_type())
    device_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("devices.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PaymentEventLog(Base):
    """Stores processed payment event ids for idempotent webhook handling."""

    __tablename__ = "payment_event_logs"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="paystack")
    type: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class DiagnosticsBundle(Base):
    __tablename__ = "diagnostics_bundles"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    client_version: Mapped[Optional[str]] = mapped_column(String(64))
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


__all__ = [
    "Base",
    "User",
    "Device",
    "RefreshToken",
    "UsageEvent",
    "PaymentEventLog",
    "DiagnosticsBundle",
    "SubscriptionStatus",
]

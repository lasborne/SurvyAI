from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# --- Auth ---
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: Optional[str] = Field(default=None, max_length=200)


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    display_name: Optional[str] = None
    plan_slug: str
    subscription_status: str

    model_config = {"from_attributes": True}

    @field_validator("subscription_status", mode="before")
    @classmethod
    def _subscription_status_str(cls, v: object) -> str:
        return getattr(v, "value", v) if v is not None else "none"


class MeOut(BaseModel):
    """Current authenticated user + subscription snapshot."""

    id: uuid.UUID
    email: str
    display_name: Optional[str] = None
    plan_slug: str
    subscription_status: str
    subscription_current_period_end: Optional[datetime] = None
    can_manage_paystack_subscription: bool = False
    max_devices: int
    monthly_agent_runs_quota: int
    monthly_agent_runs_used: int
    monthly_credits_usd: float = 0.0
    monthly_credits_used_usd: float = 0.0
    credit_markup_multiplier: float = 1.5
    credits_billing_interval: str = "monthly"
    can_use_platform_llm: bool
    primary_llm: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class RefreshIn(BaseModel):
    refresh_token: str


# --- Devices ---
class DeviceRegisterIn(BaseModel):
    fingerprint: str = Field(min_length=8, max_length=128)
    label: Optional[str] = Field(default=None, max_length=200)


class DeviceOut(BaseModel):
    id: uuid.UUID
    fingerprint: str
    label: Optional[str]
    last_seen_at: Optional[datetime]
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Billing ---
class PaystackInitializeIn(BaseModel):
    plan_code: Optional[str] = None


class PaystackInitializeOut(BaseModel):
    authorization_url: str
    access_code: Optional[str] = None
    reference: str


class BillingPlanOption(BaseModel):
    slug: str
    label: str
    plan_code: str


class BillingPlansOut(BaseModel):
    plans: list[BillingPlanOption]


class PaystackVerifyIn(BaseModel):
    reference: str = Field(min_length=3, max_length=128)


class PaystackVerifyOut(BaseModel):
    ok: bool
    plan_slug: Optional[str] = None
    subscription_status: Optional[str] = None
    detail: Optional[str] = None


class PaystackManageLinkOut(BaseModel):
    url: str


# --- Entitlements / bootstrap ---
class EntitlementsOut(BaseModel):
    plan_slug: str
    subscription_status: str
    max_devices: int
    monthly_agent_runs_quota: int
    monthly_agent_runs_used: int
    monthly_credits_usd: float = 0.0
    monthly_credits_used_usd: float = 0.0
    credit_markup_multiplier: float = 1.5
    credits_billing_interval: str = "monthly"
    can_use_platform_llm: bool
    primary_llm: Optional[str] = None


class BootstrapOut(BaseModel):
    """Secrets for desktop merge_settings — transport over TLS only."""

    access_token_hint: str = Field(
        default="Use Authorization: Bearer <access_token> on API calls.",
    )
    primary_llm: str
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    openai_model_nano: Optional[str] = None
    openai_model_mini: Optional[str] = None
    openai_model_complex: Optional[str] = None
    enable_tiered_models: Optional[bool] = None
    gemini_model: Optional[str] = None
    claude_model: Optional[str] = None
    deepseek_base_url: Optional[str] = None


# --- Usage ---
class UsageEventIn(BaseModel):
    kind: str = Field(min_length=1, max_length=64)
    quantity: int = Field(default=1, ge=1)
    cost_usd: float = Field(default=0.0, ge=0.0)
    meta: Optional[dict[str, Any]] = None


class UsageBatchIn(BaseModel):
    events: list[UsageEventIn]


class UsageBatchOut(BaseModel):
    accepted: int
    monthly_agent_runs_used: int
    monthly_agent_runs_quota: int
    monthly_credits_used_usd: float = 0.0
    monthly_credits_usd: float = 0.0


# --- Updates ---
class UpdateManifestOut(BaseModel):
    channel: str
    latest_version: str
    min_supported_version: Optional[str] = None
    download_url: Optional[str] = None
    sha256: Optional[str] = None
    release_notes_url: Optional[str] = None
    mandatory: bool = False


# --- Diagnostics ---
class DiagnosticsOut(BaseModel):
    id: uuid.UUID
    filename: str
    byte_size: int
    created_at: datetime

    model_config = {"from_attributes": True}

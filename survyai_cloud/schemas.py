from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# --- Auth ---
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)
    display_name: Optional[str] = Field(default=None, max_length=200)

    @field_validator("password")
    @classmethod
    def _password_strength(cls, v: str, info) -> str:
        from survyai_cloud.security import validate_password_strength

        email = None
        if info.data and "email" in info.data:
            email = str(info.data.get("email") or "")
        err = validate_password_strength(v, email=email)
        if err:
            raise ValueError(err)
        return v


class ForgotPasswordIn(BaseModel):
    email: EmailStr


class ForgotPasswordOut(BaseModel):
    detail: str = (
        "If an account exists for that email, a one-time reset code has been sent. "
        "Check your inbox and enter the code in SurvyAI."
    )


class ResetPasswordIn(BaseModel):
    email: EmailStr
    code: str = Field(min_length=6, max_length=32)
    new_password: str = Field(min_length=10, max_length=128)

    @field_validator("new_password")
    @classmethod
    def _password_strength(cls, v: str, info) -> str:
        from survyai_cloud.security import validate_password_strength

        email = None
        if info.data and "email" in info.data:
            email = str(info.data.get("email") or "")
        err = validate_password_strength(v, email=email)
        if err:
            raise ValueError(err)
        return v


class ChangePasswordIn(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=10, max_length=128)

    @field_validator("new_password")
    @classmethod
    def _password_strength(cls, v: str) -> str:
        from survyai_cloud.security import validate_password_strength

        err = validate_password_strength(v)
        if err:
            raise ValueError(err)
        return v


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
    usage_period_anchor: Optional[datetime] = None
    can_manage_paystack_subscription: bool = False
    max_devices: int
    monthly_agent_runs_quota: int
    monthly_agent_runs_used: int
    monthly_credits_usd: float = 0.0
    monthly_credits_used_usd: float = 0.0
    credit_markup_multiplier: float = 2.0
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
    credit_markup_multiplier: float = 2.0
    credits_billing_interval: str = "monthly"
    usage_period_anchor: Optional[datetime] = None
    subscription_current_period_end: Optional[datetime] = None
    can_use_platform_llm: bool
    primary_llm: Optional[str] = None


class AgentConfigOut(BaseModel):
    version: str
    system_prompt: str
    primary_llm: Optional[str] = None
    fallback_llm: Optional[str] = None
    openai_model: Optional[str] = None
    openai_model_nano: Optional[str] = None
    openai_model_mini: Optional[str] = None
    openai_model_complex: Optional[str] = None
    enable_tiered_models: Optional[bool] = None
    gemini_model: Optional[str] = None
    claude_model: Optional[str] = None
    deepseek_base_url: Optional[str] = None
    agent_temperature: Optional[float] = None
    agent_max_tokens: Optional[int] = None


class BootstrapOut(BaseModel):
    """Secrets for desktop merge_settings — transport over TLS only."""

    access_token_hint: str = Field(
        default="Use Authorization: Bearer <access_token> on API calls.",
    )
    llm_proxy_enabled: bool = False
    llm_proxy_path: str = "/v1/llm/chat"
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
    agent_config: Optional[AgentConfigOut] = None


# --- Usage ---
class UsageEventIn(BaseModel):
    kind: str = Field(min_length=1, max_length=64)
    quantity: int = Field(default=1, ge=1)
    cost_usd: float = Field(default=0.0, ge=0.0)
    model_name: Optional[str] = Field(default=None, max_length=200)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
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
    platform: str = "windows-x64"
    latest_version: str
    min_supported_version: Optional[str] = None
    download_url: Optional[str] = None
    sha256: Optional[str] = None
    artifact_kind: str = "full-installer"
    signature: Optional[str] = None
    signing_scheme: Optional[str] = None
    rollback_version: Optional[str] = None
    release_notes_url: Optional[str] = None
    mandatory: bool = False


class LlmToolCallIn(BaseModel):
    id: Optional[str] = None
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class LlmMessageIn(BaseModel):
    role: str = Field(min_length=1, max_length=32)
    content: Any = ""
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: list[LlmToolCallIn] = Field(default_factory=list)


class LlmProxyChatIn(BaseModel):
    provider: Literal["openai", "claude", "gemini", "deepseek"]
    model: Optional[str] = Field(default=None, max_length=200)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=200000)
    messages: list[LlmMessageIn]
    tools: list[dict[str, Any]] = Field(default_factory=list)


class LlmToolCallOut(BaseModel):
    id: Optional[str] = None
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class LlmProxyChatOut(BaseModel):
    provider: str
    model: str
    content: Any = ""
    tool_calls: list[LlmToolCallOut] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    billing: dict[str, Any] = Field(default_factory=dict)


# --- Diagnostics ---
class DiagnosticsOut(BaseModel):
    id: uuid.UUID
    filename: str
    byte_size: int
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Admin (support) ---
class AdminUserBillingPatch(BaseModel):
    """
    Support override payload. Protected by CLOUD_ADMIN_API_KEY + X-SurvyAI-Admin-Key.
    """

    plan_slug: Optional[str] = Field(default=None, max_length=64)
    subscription_status: Optional[str] = Field(
        default=None,
        description="Must match server enum values: none, trialing, active, past_due, canceled, unpaid, ...",
    )
    grace_period_ends_at: Optional[datetime] = None
    clear_grace: bool = False
    last_reactivation_at: Optional[datetime] = None
    touch_reactivated_now: bool = False
    apply_free_defaults: bool = False
    apply_pro_defaults: bool = False
    monthly_credits_usd: Optional[float] = Field(default=None, ge=0.0)
    monthly_credits_used_usd: Optional[float] = Field(default=None, ge=0.0)
    reset_credits_used: bool = False
    max_devices: Optional[int] = Field(default=None, ge=1, le=50)
    subscription_current_period_end: Optional[datetime] = None
    clear_period_end: bool = False


class AdminUserSnapshot(BaseModel):
    """Read-only support view of a user (no password/hash/token material)."""

    id: uuid.UUID
    email: str
    display_name: Optional[str] = None
    plan_slug: str
    subscription_status: str
    subscription_current_period_end: Optional[datetime] = None
    grace_period_ends_at: Optional[datetime] = None
    last_reactivation_at: Optional[datetime] = None
    max_devices: int
    monthly_agent_runs_quota: int
    monthly_agent_runs_used: int
    monthly_credits_usd: float
    monthly_credits_used_usd: float
    credits_billing_interval: str
    usage_period_anchor: Optional[datetime] = None
    paystack_customer_code: Optional[str] = None
    paystack_subscription_code: Optional[str] = None
    last_payment_reference: Optional[str] = None
    device_count: int = 0
    created_at: datetime
    updated_at: datetime


class AdminUsageEventOut(BaseModel):
    id: int
    kind: str
    quantity: int
    cost_usd: float
    meta: Optional[dict[str, Any]] = None
    device_id: Optional[uuid.UUID] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AdminDiagnosticsOut(BaseModel):
    id: uuid.UUID
    filename: str
    byte_size: int
    client_version: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}

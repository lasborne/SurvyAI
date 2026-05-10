"""
Backend-only settings (separate from desktop ``config.settings``).

Loaded from environment variables / ``.env`` when the cloud process starts.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CloudSettings(BaseSettings):
    model_config = SettingsConfigDict(
        # Later files override earlier; use `.env.cloud` for server-only secrets in a mixed repo.
        env_file=(".env", ".env.cloud"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="SurvyAI Cloud API")
    debug: bool = Field(default=False)

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/survyai_cloud.db",
        description="SQLAlchemy async URL (postgresql+asyncpg://... in production).",
    )
    run_migrations_on_startup: bool = Field(
        default=True,
        description="When using Postgres, run ``alembic upgrade head`` on startup (idempotent).",
    )

    # Redis (rate limits + future cache). Empty = in-process limiter (single worker only).
    redis_url: str = Field(default="", description="redis://... or rediss://... for TLS.")
    rate_limit_window_seconds: int = Field(default=60, ge=10, le=3600)
    rate_limit_bootstrap_per_window: int = Field(
        default=30,
        ge=0,
        le=100_000,
        description="Max GET /v1/bootstrap per user+IP per window; 0 disables.",
    )
    rate_limit_usage_events_per_window: int = Field(
        default=240,
        ge=0,
        le=1_000_000,
        description="Max POST /v1/usage/events per user+IP per window; 0 disables.",
    )

    # Auth
    jwt_secret: str = Field(
        default="change-me-in-production-use-openssl-rand-hex-32",
        min_length=16,
    )
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(
        default=7200,
        ge=5,
        le=10080,
        description="Access JWT lifetime in minutes (default 120h). Max 7 days.",
    )
    refresh_token_expire_days: int = Field(default=30, ge=1, le=365)
    bcrypt_rounds: int = Field(default=12, ge=4, le=31)

    # CORS (comma-separated origins; * for dev only)
    cors_origins: str = Field(default="*")

    # Paystack (Nigeria-first)
    paystack_secret_key: str = Field(default="", description="sk_live_* or sk_test_*")
    paystack_public_key: str = Field(default="", description="pk_live_* or pk_test_* (optional for backend)")
    paystack_plan_code_pro_monthly: str = Field(
        default="",
        description="Plan code PLN_* for Pro monthly subscriptions.",
    )
    paystack_plan_code_pro_annual: str = Field(
        default="",
        description="Plan code PLN_* for Pro annual subscriptions (from Paystack Dashboard).",
    )
    paystack_pro_monthly_amount_ngn: int = Field(
        default=15000,
        ge=0,
        description="Display/label only; actual charge is set on the Paystack plan (default ₦15,000/mo).",
    )
    paystack_pro_annual_amount_ngn: int = Field(
        default=162000,
        ge=0,
        description="Display/label only; actual charge is set on the Paystack plan (default ₦162,000/yr).",
    )
    paystack_callback_url: str = Field(
        default="http://127.0.0.1/survyai/paystack/callback",
        description="Shown after Paystack checkout; desktop can deep-link or just show success.",
    )
    paystack_webhook_enabled: bool = Field(default=True)

    # Platform LLM keys (injected to paying desktop clients via /v1/bootstrap only)
    platform_primary_llm: Literal["deepseek", "gemini", "claude", "openai"] = Field(
        default="openai",
    )
    platform_openai_api_key: str = Field(default="")
    platform_anthropic_api_key: str = Field(default="")
    platform_google_api_key: str = Field(default="")
    platform_deepseek_api_key: str = Field(default="")
    platform_openai_model: str = Field(
        default="gpt-4o-mini",
        description="Legacy single OpenAI model name (used when tiered models are disabled).",
    )
    platform_openai_model_nano: str = Field(
        default="gpt-5-nano",
        description="OpenAI model for trivial tasks (desktop tiered selection).",
    )
    platform_openai_model_mini: str = Field(
        default="gpt-5-mini",
        description="OpenAI model for normal tasks (desktop tiered selection).",
    )
    platform_openai_model_complex: str = Field(
        default="gpt-5.1",
        description="OpenAI model for very complex tasks (desktop tiered selection).",
    )
    platform_enable_tiered_models: bool = Field(
        default=True,
        description="If True, desktop can use nano/mini/complex model routing like local .env.",
    )
    platform_gemini_model: str = Field(default="gemini-2.0-flash")
    platform_claude_model: str = Field(default="claude-3-5-sonnet-20241022")
    platform_deepseek_base_url: str = Field(default="https://api.deepseek.com/v1")

    # Plans & quotas
    pro_plan_slug: str = Field(default="pro")
    free_monthly_agent_runs: int = Field(default=0, ge=0)
    pro_monthly_agent_runs: int = Field(default=500, ge=0)
    default_max_devices_free: int = Field(default=1, ge=1)
    default_max_devices_pro: int = Field(default=2, ge=1)

    # Credits: NGN→USD conversion and user-facing markup
    ngn_to_usd_rate: float = Field(
        default=0.00062,
        description=(
            "NGN to USD exchange rate used to convert subscription payment to a USD credit balance. "
            "Update periodically or fetch from a live API.  Default ≈ ₦1,600/$1."
        ),
    )
    credit_markup_multiplier: float = Field(
        default=1.5,
        ge=1.0,
        description=(
            "Multiplier applied to raw LLM cost before showing it to the user. "
            "Covers infrastructure, overhead, and margin."
        ),
    )

    # Updates (static manifest; point url_base at CDN or blob storage)
    updates_manifest_path: str = Field(
        default="",
        description="Optional path to JSON file for /v1/updates/manifest.",
    )
    updates_default_channel: str = Field(default="stable")

    # Diagnostics uploads
    diagnostics_storage_dir: str = Field(default="./data/diagnostics_uploads")

    # Usage batch limits
    usage_batch_max_events: int = Field(default=100, ge=1, le=1000)

    @field_validator("database_url")
    @classmethod
    def _normalize_db_url(cls, v: str) -> str:
        return v.strip()

    def cors_origin_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


@lru_cache
def get_cloud_settings() -> CloudSettings:
    return CloudSettings()


def reset_cloud_settings_cache() -> None:
    get_cloud_settings.cache_clear()

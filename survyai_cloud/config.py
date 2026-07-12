"""
Backend-only settings (separate from desktop ``config.settings``).

Loaded from environment variables / ``.env`` when the cloud process starts.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from survyai.database_urls import ResolvedDatabaseUrls, resolve_database_urls

DEFAULT_JWT_SECRET = "change-me-in-production-use-openssl-rand-hex-32"


class CloudSettings(BaseSettings):
    model_config = SettingsConfigDict(
        # Later files override earlier; use `.env.cloud` for server-only secrets in a mixed repo.
        env_file=(".env", ".env.cloud"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = Field(default="SurvyAI Cloud API")
    debug: bool = Field(default=False)
    deployment_env: Literal["development", "staging", "production"] = Field(
        default="development",
        validation_alias=AliasChoices("SURVYAI_ENV", "APP_ENV", "ENVIRONMENT", "environment"),
    )
    enforce_safe_production_startup: bool = Field(default=True)

    # Database — see survyai/database_urls.py and .env.example
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/survyai_cloud.db",
        description=(
            "Primary DB URL (DATABASE_URL). Sync ``postgresql://`` for Alembic; "
            "``postgresql+asyncpg://`` also accepted (async derived for API)."
        ),
    )
    async_database_url: str = Field(
        default="",
        validation_alias=AliasChoices("ASYNC_DATABASE_URL", "async_database_url"),
        description=(
            "FastAPI-only async URL (postgresql+asyncpg://…). When set, overrides "
            "async driver for the cloud API; Alembic still uses sync form of DATABASE_URL."
        ),
    )
    run_migrations_on_startup: bool = Field(
        default=True,
        description="When using Postgres, run ``alembic upgrade head`` on startup (idempotent).",
    )

    vector_db_url: str = Field(
        default="",
        validation_alias=AliasChoices("VECTOR_DB_URL", "vector_db_url"),
        description=(
            "Sync URL for the desktop vector store (psycopg). "
            "Falls back to sync form of DATABASE_URL if empty."
        ),
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
        default=DEFAULT_JWT_SECRET,
        min_length=16,
    )
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(
        default=120,
        ge=5,
        le=10080,
        description="Access JWT lifetime in minutes (default 2h). Max 7 days.",
    )
    refresh_token_expire_days: int = Field(default=30, ge=1, le=365)
    bcrypt_rounds: int = Field(default=12, ge=4, le=31)

    # Transactional email (Resend) — password reset
    resend_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("RESEND_API_KEY", "resend_api_key"),
        description="Resend API key for transactional email (password reset).",
    )
    email_from: str = Field(
        default="SurvyAI <noreply@survyai.app>",
        validation_alias=AliasChoices("EMAIL_FROM", "email_from"),
        description="Verified Resend From address, e.g. SurvyAI <noreply@yourdomain.com>.",
    )
    password_reset_ttl_minutes: int = Field(default=30, ge=5, le=1440)
    password_reset_code_length: int = Field(default=8, ge=6, le=16)
    rate_limit_auth_login_per_window: int = Field(
        default=20,
        ge=0,
        le=10_000,
        description="Max POST /v1/auth/login per IP per 15 min window; 0 disables.",
    )
    rate_limit_auth_forgot_per_window: int = Field(
        default=5,
        ge=0,
        le=10_000,
        description="Max POST /v1/auth/forgot-password per IP per hour; 0 disables.",
    )
    rate_limit_auth_reset_per_window: int = Field(
        default=10,
        ge=0,
        le=10_000,
        description="Max POST /v1/auth/reset-password per IP per hour; 0 disables.",
    )
    rate_limit_auth_change_password_per_window: int = Field(
        default=10,
        ge=0,
        le=10_000,
        description="Max POST /v1/auth/change-password per user+IP per window; 0 disables.",
    )

    # CORS (comma-separated origins; * for dev only)
    cors_origins: str = Field(default="*")

    # Support / ops: enable admin API when set
    admin_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("CLOUD_ADMIN_API_KEY", "admin_api_key"),
        description=(
            "If set, enables /v1/admin/* endpoints protected by header X-SurvyAI-Admin-Key."
        ),
    )

    # Paystack (Nigeria-first)
    paystack_secret_key: str = Field(default="", description="sk_live_* or sk_test_*")
    paystack_public_key: str = Field(default="", description="pk_live_* or pk_test_* (optional for backend)")
    paystack_plan_code_pro_daily: str = Field(
        default="",
        description="Plan code PLN_* for Pro daily subscriptions.",
    )
    paystack_plan_code_pro_weekly: str = Field(
        default="",
        description="Plan code PLN_* for Pro weekly subscriptions.",
    )
    paystack_plan_code_pro_monthly: str = Field(
        default="",
        description="Plan code PLN_* for Pro monthly subscriptions.",
    )
    paystack_plan_code_pro_annual: str = Field(
        default="",
        description="Plan code PLN_* for Pro annual subscriptions (from Paystack Dashboard).",
    )
    paystack_pro_daily_amount_ngn: int = Field(
        default=1000,
        ge=0,
        description="Display/label only; actual charge is set on the Paystack plan (default ₦1,000/day).",
    )
    paystack_pro_weekly_amount_ngn: int = Field(
        default=5000,
        ge=0,
        description="Display/label only; actual charge is set on the Paystack plan (default ₦5,000/week).",
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
        default="gpt-5.4-nano",
        description="OpenAI model for trivial tasks (desktop tiered selection).",
    )
    platform_openai_model_mini: str = Field(
        default="gpt-5.4-mini",
        description="OpenAI model for normal tasks (desktop tiered selection).",
    )
    platform_openai_model_complex: str = Field(
        default="gpt-5.4",
        description="OpenAI model for very complex tasks (desktop tiered selection).",
    )
    platform_enable_tiered_models: bool = Field(
        default=True,
        description="If True, desktop can use nano/mini/complex model routing like local .env.",
    )
    platform_gemini_model: str = Field(default="gemini-2.0-flash")
    platform_claude_model: str = Field(default="claude-3-5-sonnet-20241022")
    platform_deepseek_base_url: str = Field(default="https://api.deepseek.com/v1")
    platform_llm_proxy_path: str = Field(
        default="/v1/llm/chat",
        description="Relative API path used by desktop builds for hosted LLM proxy calls.",
    )
    platform_agent_config_path: str = Field(
        default="./agent/agent_config.json",
        description=(
            "Optional JSON file served through /v1/bootstrap for runtime agent config. "
            "Use this for production prompt/model changes without rebuilding the desktop app."
        ),
    )

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
        default=2.0,
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

    @field_validator("database_url", "async_database_url", "vector_db_url")
    @classmethod
    def _normalize_db_url(cls, v: str) -> str:
        return (v or "").strip()

    def resolved_database_urls(self) -> ResolvedDatabaseUrls:
        """Sync (Alembic/vector) vs async (FastAPI) URLs from env."""
        return resolve_database_urls(
            database_url=self.database_url,
            async_database_url=self.async_database_url,
            vector_db_url=self.vector_db_url,
        )

    def sqlalchemy_async_url(self) -> str:
        """URL for ``create_async_engine`` in the cloud API."""
        return self.resolved_database_urls().async_

    def sqlalchemy_sync_url(self) -> str:
        """URL for Alembic and other blocking SQLAlchemy/psycopg tools."""
        return self.resolved_database_urls().sync

    def vector_store_sync_url(self) -> str:
        """URL for the desktop agent PostgreSQL vector store."""
        return self.resolved_database_urls().vector_sync

    def cors_origin_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    def is_production(self) -> bool:
        return self.deployment_env == "production"

    def startup_warnings(self) -> list[str]:
        warnings: list[str] = []
        if self.is_production() and not self.redis_url.strip():
            warnings.append(
                "REDIS_URL is not configured; rate limits will be process-local only."
            )
        if self.is_production() and not self.admin_api_key.strip():
            warnings.append(
                "CLOUD_ADMIN_API_KEY is not configured; support override endpoints remain disabled."
            )
        return warnings

    def startup_validation_errors(self) -> list[str]:
        if not self.is_production():
            return []

        errors: list[str] = []
        if self.debug:
            errors.append("DEBUG must be false in production.")
        jwt = self.jwt_secret.strip()
        if jwt == DEFAULT_JWT_SECRET or len(jwt) < 32:
            errors.append("JWT_SECRET must be replaced with a strong production secret (32+ chars).")
        if "*" in self.cors_origin_list():
            errors.append("CORS_ORIGINS cannot be '*' in production; set explicit trusted origins.")
        async_url = self.sqlalchemy_async_url().lower()
        if async_url.startswith("sqlite"):
            errors.append("Production requires PostgreSQL; SQLite is not allowed for DATABASE_URL/ASYNC_DATABASE_URL.")
        if self.access_token_expire_minutes > 1440:
            errors.append("ACCESS_TOKEN_EXPIRE_MINUTES must be 1440 minutes (24h) or less in production.")
        if self.platform_primary_llm == "openai" and not self.platform_openai_api_key.strip():
            errors.append("PLATFORM_OPENAI_API_KEY is required when PLATFORM_PRIMARY_LLM=openai.")
        if self.platform_primary_llm == "claude" and not self.platform_anthropic_api_key.strip():
            errors.append("PLATFORM_ANTHROPIC_API_KEY is required when PLATFORM_PRIMARY_LLM=claude.")
        if self.platform_primary_llm == "gemini" and not self.platform_google_api_key.strip():
            errors.append("PLATFORM_GOOGLE_API_KEY is required when PLATFORM_PRIMARY_LLM=gemini.")
        if self.platform_primary_llm == "deepseek" and not self.platform_deepseek_api_key.strip():
            errors.append("PLATFORM_DEEPSEEK_API_KEY is required when PLATFORM_PRIMARY_LLM=deepseek.")
        callback = self.paystack_callback_url.strip()
        if callback and _is_local_callback_url(callback):
            errors.append("PAYSTACK_CALLBACK_URL must not point to localhost/127.0.0.1 in production.")
        return errors

    def assert_safe_startup(self) -> None:
        errors = self.startup_validation_errors()
        if errors and self.enforce_safe_production_startup:
            joined = "\n- ".join(["Unsafe production cloud configuration detected:"] + errors)
            raise RuntimeError(joined)


def _is_local_callback_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").strip().lower()
    return host in {"127.0.0.1", "localhost"}


@lru_cache
def get_cloud_settings() -> CloudSettings:
    return CloudSettings()


def reset_cloud_settings_cache() -> None:
    get_cloud_settings.cache_clear()

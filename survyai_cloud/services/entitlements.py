from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Tuple

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.config import CloudSettings, get_cloud_settings
from survyai_cloud.models import SubscriptionStatus, User
from survyai_cloud.schemas import BootstrapOut, EntitlementsOut


ACTIVE_LLM_STATUSES = frozenset(
    {
        SubscriptionStatus.trialing,
        SubscriptionStatus.active,
        SubscriptionStatus.non_renewing,
    }
)


def _month_start_utc(dt: datetime) -> datetime:
    """
    First instant of the calendar month in UTC, timezone-aware.

    SQLite often returns **naive** datetimes; ``datetime.now(timezone.utc)`` is
    aware. Comparing naive vs aware raises ``TypeError`` (HTTP 500 on /v1/me).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _pro_monthly_credit_budget_usd(settings: CloudSettings) -> float:
    """Convert the Pro monthly NGN price to a USD credit balance."""
    return round(settings.paystack_pro_monthly_amount_ngn * settings.ngn_to_usd_rate, 4)


def _pro_annual_credit_budget_usd(settings: CloudSettings) -> float:
    """Convert the Pro annual NGN price to a USD credit balance (full period pool)."""
    return round(settings.paystack_pro_annual_amount_ngn * settings.ngn_to_usd_rate, 4)


def credit_budget_and_interval_from_paystack_payload(
    data: dict[str, Any], settings: CloudSettings
) -> Tuple[Optional[float], str]:
    """
    Derive the subscriber's USD credit pool and billing interval from a Paystack transaction.

    ``amount`` is in kobo (NGN×100). USD uses the server's ``ngn_to_usd_rate`` (same as catalog).
    This is the **subscription budget** (not API-cost×markup); consumption still applies markup server-side.
    """
    settings = settings or get_cloud_settings()
    plan_code = ""
    pl = data.get("plan")
    if isinstance(pl, dict):
        plan_code = str(pl.get("plan_code") or pl.get("code") or "").strip()
    annual_code = (settings.paystack_plan_code_pro_annual or "").strip()
    monthly_code = (settings.paystack_plan_code_pro_monthly or "").strip()
    if annual_code and plan_code == annual_code:
        interval = "annual"
    elif monthly_code and plan_code == monthly_code:
        interval = "monthly"
    else:
        interval = "monthly"

    amount_raw = data.get("amount")
    credit_usd: Optional[float] = None
    try:
        if isinstance(amount_raw, (int, float)) and float(amount_raw) > 0:
            ngn = float(amount_raw) / 100.0
            credit_usd = round(ngn * settings.ngn_to_usd_rate, 4)
    except Exception:
        credit_usd = None
    return credit_usd, interval


async def ensure_usage_month_rolled(user: User, db: AsyncSession) -> None:
    """Reset monthly counters when UTC calendar month changes."""
    now = datetime.now(timezone.utc)
    anchor = user.usage_period_anchor
    if anchor is None:
        user.usage_period_anchor = _month_start_utc(now)
        db.add(user)
        return
    if _month_start_utc(anchor) < _month_start_utc(now):
        user.monthly_agent_runs_used = 0
        user.monthly_credits_used_usd = 0.0
        user.usage_period_anchor = _month_start_utc(now)
        db.add(user)


def can_use_platform_llm(user: User, settings: CloudSettings | None = None) -> bool:
    settings = settings or get_cloud_settings()
    if user.plan_slug != settings.pro_plan_slug:
        return False
    return user.subscription_status in ACTIVE_LLM_STATUSES


def entitlements_for_user(user: User, settings: CloudSettings | None = None) -> EntitlementsOut:
    settings = settings or get_cloud_settings()
    return EntitlementsOut(
        plan_slug=user.plan_slug,
        subscription_status=user.subscription_status.value,
        max_devices=user.max_devices,
        monthly_agent_runs_quota=user.monthly_agent_runs_quota,
        monthly_agent_runs_used=user.monthly_agent_runs_used,
        monthly_credits_usd=user.monthly_credits_usd,
        monthly_credits_used_usd=user.monthly_credits_used_usd,
        credit_markup_multiplier=settings.credit_markup_multiplier,
        credits_billing_interval=str(user.credits_billing_interval or "monthly"),
        can_use_platform_llm=can_use_platform_llm(user, settings),
        primary_llm=settings.platform_primary_llm,
    )


def build_bootstrap_payload(user: User, settings: CloudSettings | None = None) -> BootstrapOut:
    settings = settings or get_cloud_settings()
    if not can_use_platform_llm(user, settings):
        raise HTTPException(status_code=403, detail="Active Pro subscription required for platform keys")

    primary = settings.platform_primary_llm
    out = BootstrapOut(
        primary_llm=primary,
        openai_model=settings.platform_openai_model,
        openai_model_nano=settings.platform_openai_model_nano,
        openai_model_mini=settings.platform_openai_model_mini,
        openai_model_complex=settings.platform_openai_model_complex,
        enable_tiered_models=settings.platform_enable_tiered_models,
        gemini_model=settings.platform_gemini_model,
        claude_model=settings.platform_claude_model,
        deepseek_base_url=settings.platform_deepseek_base_url,
    )
    if settings.platform_openai_api_key:
        out.openai_api_key = settings.platform_openai_api_key
    if settings.platform_anthropic_api_key:
        out.anthropic_api_key = settings.platform_anthropic_api_key
    if settings.platform_google_api_key:
        out.google_api_key = settings.platform_google_api_key
    if settings.platform_deepseek_api_key:
        out.deepseek_api_key = settings.platform_deepseek_api_key

    if primary == "openai" and not out.openai_api_key:
        raise HTTPException(status_code=503, detail="Server missing platform OpenAI configuration")
    if primary == "claude" and not out.anthropic_api_key:
        raise HTTPException(status_code=503, detail="Server missing platform Anthropic configuration")
    if primary == "gemini" and not out.google_api_key:
        raise HTTPException(status_code=503, detail="Server missing platform Google configuration")
    if primary == "deepseek" and not out.deepseek_api_key:
        raise HTTPException(status_code=503, detail="Server missing platform DeepSeek configuration")

    return out


def apply_free_defaults(user: User, settings: CloudSettings | None = None) -> None:
    settings = settings or get_cloud_settings()
    user.plan_slug = "free"
    user.subscription_status = SubscriptionStatus.none
    user.paystack_subscription_code = None
    user.paystack_email_token = None
    user.subscription_current_period_end = None
    user.max_devices = settings.default_max_devices_free
    user.monthly_agent_runs_quota = settings.free_monthly_agent_runs
    user.monthly_credits_usd = 0.0
    user.monthly_credits_used_usd = 0.0
    user.credits_billing_interval = "monthly"


def apply_pro_defaults(
    user: User,
    settings: CloudSettings | None = None,
    *,
    credit_budget_usd: Optional[float] = None,
    credits_billing_interval: Optional[str] = None,
) -> None:
    """
    Set Pro plan quotas. Credit **budget** is the subscriber's USD pool (from payment NGN×rate or catalog).

    API usage is charged at raw provider cost × ``credit_markup_multiplier`` against that pool.
    """
    settings = settings or get_cloud_settings()
    user.plan_slug = settings.pro_plan_slug
    user.max_devices = settings.default_max_devices_pro
    user.monthly_agent_runs_quota = settings.pro_monthly_agent_runs

    raw_interval = (credits_billing_interval or getattr(user, "credits_billing_interval", None) or "monthly")
    raw_interval = str(raw_interval).strip().lower()
    if raw_interval not in ("monthly", "annual"):
        raw_interval = "monthly"
    user.credits_billing_interval = raw_interval

    if credit_budget_usd is not None and float(credit_budget_usd) >= 0:
        user.monthly_credits_usd = round(float(credit_budget_usd), 4)
    elif user.credits_billing_interval == "annual":
        user.monthly_credits_usd = _pro_annual_credit_budget_usd(settings)
    else:
        user.monthly_credits_usd = _pro_monthly_credit_budget_usd(settings)

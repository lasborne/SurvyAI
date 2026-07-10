from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from agent.runtime_config import resolve_agent_runtime_config
from survyai_cloud.config import CloudSettings, get_cloud_settings
from survyai_cloud.models import SubscriptionStatus, User
from survyai_cloud.schemas import AgentConfigOut, BootstrapOut, EntitlementsOut


ACTIVE_LLM_STATUSES = frozenset(
    {
        SubscriptionStatus.trialing,
        SubscriptionStatus.active,
        SubscriptionStatus.non_renewing,
    }
)


def _pro_monthly_credit_budget_usd(settings: CloudSettings) -> float:
    """Convert the Pro monthly NGN price to a USD credit balance."""
    return round(settings.paystack_pro_monthly_amount_ngn * settings.ngn_to_usd_rate, 4)


def _pro_weekly_credit_budget_usd(settings: CloudSettings) -> float:
    """Pro weekly credit pool from the weekly NGN catalog price."""
    return round(settings.paystack_pro_weekly_amount_ngn * settings.ngn_to_usd_rate, 4)


def _pro_daily_credit_budget_usd(settings: CloudSettings) -> float:
    """Pro daily credit pool from the daily NGN catalog price."""
    return round(settings.paystack_pro_daily_amount_ngn * settings.ngn_to_usd_rate, 4)


def _aware_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _subscription_period_still_open(user: User, now: datetime | None = None) -> bool:
    """True when subscription_current_period_end is missing (legacy) or still in the future."""
    now = now or datetime.now(timezone.utc)
    period_end = _aware_utc(getattr(user, "subscription_current_period_end", None))
    if period_end is None:
        return True
    return period_end > now


def _is_active_early_renewal(user: User, settings: CloudSettings, now: datetime | None = None) -> bool:
    """Active/trialing Pro with an unexpired paid window — remaining credits should carry forward."""
    now = now or datetime.now(timezone.utc)
    if user.plan_slug != settings.pro_plan_slug:
        return False
    if user.subscription_status not in ACTIVE_LLM_STATUSES:
        return False
    return _subscription_period_still_open(user, now)


def _pro_annual_credit_budget_usd(settings: CloudSettings) -> float:
    """Convert the Pro annual NGN price to a USD credit balance (full period pool)."""
    return round(settings.paystack_pro_annual_amount_ngn * settings.ngn_to_usd_rate, 4)


def billing_interval_period_days(interval: str) -> int:
    """Rolling billing window length for Paystack Pro intervals."""
    key = str(interval or "monthly").strip().lower()
    return {"daily": 1, "weekly": 7, "monthly": 30, "annual": 365}.get(key, 30)


def subscription_period_end_from_anchor(anchor: datetime, interval: str) -> datetime:
    """Next period end from payment anchor and billing interval."""
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    else:
        anchor = anchor.astimezone(timezone.utc)
    return anchor + timedelta(days=billing_interval_period_days(interval))


def manual_payment_period_anchor(user: User, paid_at: datetime | None = None) -> datetime:
    """Anchor a manual renewal at current expiry when the account is still active."""
    now = datetime.now(timezone.utc)
    anchor = paid_at or now
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    else:
        anchor = anchor.astimezone(timezone.utc)
    current_end = getattr(user, "subscription_current_period_end", None)
    if isinstance(current_end, datetime):
        if current_end.tzinfo is None:
            current_end = current_end.replace(tzinfo=timezone.utc)
        else:
            current_end = current_end.astimezone(timezone.utc)
        if current_end > anchor and user.subscription_status in ACTIVE_LLM_STATUSES:
            return current_end
    return anchor


def interval_from_paystack_plan_code(plan_code: str, settings: CloudSettings) -> str:
    """Map a Paystack plan code to a SurvyAI billing interval slug."""
    pc = (plan_code or "").strip()
    daily_code = (settings.paystack_plan_code_pro_daily or "").strip()
    weekly_code = (settings.paystack_plan_code_pro_weekly or "").strip()
    monthly_code = (settings.paystack_plan_code_pro_monthly or "").strip()
    annual_code = (settings.paystack_plan_code_pro_annual or "").strip()
    if annual_code and pc == annual_code:
        return "annual"
    if monthly_code and pc == monthly_code:
        return "monthly"
    if weekly_code and pc == weekly_code:
        return "weekly"
    if daily_code and pc == daily_code:
        return "daily"
    return "monthly"


def pro_agent_runs_quota_for_interval(settings: CloudSettings, interval: str) -> int:
    """Scale the monthly agent-run cap for shorter billing intervals."""
    base = int(settings.pro_monthly_agent_runs or 0)
    if base <= 0:
        return 0
    key = str(interval or "monthly").strip().lower()
    if key == "daily":
        return max(1, base // 30)
    if key == "weekly":
        return max(1, base // 4)
    return base


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
    elif isinstance(pl, str):
        plan_code = pl.strip()
    if not plan_code:
        meta = data.get("metadata")
        if isinstance(meta, dict):
            plan_code = str(meta.get("survyai_selected_plan_code") or meta.get("plan_code") or "").strip()
    interval = interval_from_paystack_plan_code(plan_code, settings)

    amount_raw = data.get("amount")
    paid_usd: Optional[float] = None
    try:
        if isinstance(amount_raw, (int, float)) and float(amount_raw) > 0:
            ngn = float(amount_raw) / 100.0
            paid_usd = round(ngn * settings.ngn_to_usd_rate, 4)
    except Exception:
        paid_usd = None

    if interval == "daily":
        return (paid_usd if paid_usd is not None else _pro_daily_credit_budget_usd(settings)), interval
    if interval == "weekly":
        return (paid_usd if paid_usd is not None else _pro_weekly_credit_budget_usd(settings)), interval
    if interval == "annual":
        return (paid_usd if paid_usd is not None else _pro_annual_credit_budget_usd(settings)), interval
    if interval == "monthly":
        return (paid_usd if paid_usd is not None else _pro_monthly_credit_budget_usd(settings)), interval

    return paid_usd, interval


async def ensure_usage_month_rolled(user: User, db: AsyncSession) -> None:
    """
    Keep ``usage_period_anchor`` aligned with the paid subscription window.

    Credit counters are **not** auto-reset just because calendar time elapsed —
    that would grant a free pool without payment. Fresh / carried budgets are
    applied on verified payment via ``apply_pro_defaults``.
    """
    now = datetime.now(timezone.utc)
    interval = str(user.credits_billing_interval or "monthly").strip().lower()
    period_days = billing_interval_period_days(interval)
    period_end = _aware_utc(getattr(user, "subscription_current_period_end", None))
    anchor = _aware_utc(user.usage_period_anchor)

    if anchor is None:
        if period_end is not None:
            # Back-derive the paid-window start from the known period end.
            user.usage_period_anchor = period_end - timedelta(days=period_days)
        else:
            user.usage_period_anchor = now
        db.add(user)
        return

    # While the paid window is still open (including early-renewal extensions),
    # preserve used credits and the original anchor so remaining balance carries.
    if period_end is not None and now < period_end:
        return

    # Expired window without a new payment: leave counters as-is so the UI still
    # reflects consumption in the last paid period. Hosted LLM access is blocked
    # separately by ``subscription_allows_platform_llm``.


def can_use_platform_llm(user: User, settings: CloudSettings | None = None) -> bool:
    return subscription_allows_platform_llm(user, settings) and has_platform_credit_remaining(
        user, settings
    )


def subscription_allows_platform_llm(
    user: User, settings: CloudSettings | None = None
) -> bool:
    settings = settings or get_cloud_settings()
    if user.plan_slug != settings.pro_plan_slug:
        return False
    if user.subscription_status not in ACTIVE_LLM_STATUSES:
        return False
    # Paid hosted access requires an unexpired subscription window when set.
    return _subscription_period_still_open(user)


def has_platform_credit_remaining(
    user: User, settings: CloudSettings | None = None
) -> bool:
    _ = settings or get_cloud_settings()
    budget = float(getattr(user, "monthly_credits_usd", 0.0) or 0.0)
    used = float(getattr(user, "monthly_credits_used_usd", 0.0) or 0.0)
    if budget <= 0:
        return False
    return used + 1e-6 < budget


def entitlements_for_user(user: User, settings: CloudSettings | None = None) -> EntitlementsOut:
    settings = settings or get_cloud_settings()
    primary = resolve_platform_llm_provider(settings)
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
        usage_period_anchor=getattr(user, "usage_period_anchor", None),
        subscription_current_period_end=getattr(user, "subscription_current_period_end", None),
        can_use_platform_llm=can_use_platform_llm(user, settings),
        primary_llm=primary,
    )


def resolve_platform_llm_provider(settings: CloudSettings) -> str:
    """Return the configured hosted provider that should be advertised to desktop.

    Production safety: if PLATFORM_PRIMARY_LLM points to a provider whose key is
    missing, fall back to the first configured provider. This prevents paid users
    from being blocked by a stale env var such as PLATFORM_PRIMARY_LLM=gemini
    when only PLATFORM_OPENAI_API_KEY is configured.
    """
    preferred = str(settings.platform_primary_llm or "openai").strip().lower()
    configured = {
        "openai": bool(settings.platform_openai_api_key.strip()),
        "claude": bool(settings.platform_anthropic_api_key.strip()),
        "gemini": bool(settings.platform_google_api_key.strip()),
        "deepseek": bool(settings.platform_deepseek_api_key.strip()),
    }
    if configured.get(preferred):
        return preferred
    for provider in ("openai", "claude", "gemini", "deepseek"):
        if configured.get(provider):
            return provider
    return preferred


def build_bootstrap_payload(user: User, settings: CloudSettings | None = None) -> BootstrapOut:
    settings = settings or get_cloud_settings()
    if not can_use_platform_llm(user, settings):
        raise HTTPException(status_code=403, detail="Active Pro subscription required for platform keys")

    primary = resolve_platform_llm_provider(settings)
    agent_cfg = resolve_agent_runtime_config(
        local_config_path=str(settings.platform_agent_config_path or ""),
        cloud_config_json=json.dumps(
            {
                "primary_llm": primary,
                "openai_model": settings.platform_openai_model,
                "openai_model_nano": settings.platform_openai_model_nano,
                "openai_model_mini": settings.platform_openai_model_mini,
                "openai_model_complex": settings.platform_openai_model_complex,
                "enable_tiered_models": settings.platform_enable_tiered_models,
                "gemini_model": settings.platform_gemini_model,
                "claude_model": settings.platform_claude_model,
                "deepseek_base_url": settings.platform_deepseek_base_url,
            }
        ),
    )
    out = BootstrapOut(
        llm_proxy_enabled=True,
        llm_proxy_path=str(settings.platform_llm_proxy_path or "/v1/llm/chat").strip()
        or "/v1/llm/chat",
        primary_llm=primary,
        openai_model=settings.platform_openai_model,
        openai_model_nano=settings.platform_openai_model_nano,
        openai_model_mini=settings.platform_openai_model_mini,
        openai_model_complex=settings.platform_openai_model_complex,
        enable_tiered_models=settings.platform_enable_tiered_models,
        gemini_model=settings.platform_gemini_model,
        claude_model=settings.platform_claude_model,
        deepseek_base_url=settings.platform_deepseek_base_url,
        agent_config=AgentConfigOut(**agent_cfg.to_payload_dict()),
    )
    if primary == "openai" and not settings.platform_openai_api_key.strip():
        raise HTTPException(status_code=503, detail="Server missing platform OpenAI configuration")
    if primary == "claude" and not settings.platform_anthropic_api_key.strip():
        raise HTTPException(status_code=503, detail="Server missing platform Anthropic configuration")
    if primary == "gemini" and not settings.platform_google_api_key.strip():
        raise HTTPException(status_code=503, detail="Server missing platform Google configuration")
    if primary == "deepseek" and not settings.platform_deepseek_api_key.strip():
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
    paid_at: Optional[datetime] = None,
) -> None:
    """
    Set Pro plan quotas. Credit **budget** is the subscriber's USD pool (from payment NGN×rate or catalog).

    API usage is charged at raw provider cost × ``credit_markup_multiplier`` against that pool.

    Early renewal (still inside an active paid window): add the new plan USD to the
    existing pool, preserve ``monthly_credits_used_usd``, and keep the original
    ``usage_period_anchor`` so remaining credits carry forward.

    Purchase after expiry: budget = new plan USD, used = 0, anchor = paid time.
    """
    settings = settings or get_cloud_settings()
    now = datetime.now(timezone.utc)
    paid_anchor = _aware_utc(paid_at) or now
    early_renewal = _is_active_early_renewal(user, settings, now)

    user.plan_slug = settings.pro_plan_slug
    user.max_devices = settings.default_max_devices_pro

    raw_interval = (credits_billing_interval or getattr(user, "credits_billing_interval", None) or "monthly")
    raw_interval = str(raw_interval).strip().lower()
    if raw_interval not in ("daily", "weekly", "monthly", "annual"):
        raw_interval = "monthly"
    user.credits_billing_interval = raw_interval
    user.monthly_agent_runs_quota = pro_agent_runs_quota_for_interval(settings, raw_interval)

    if credit_budget_usd is not None and float(credit_budget_usd) >= 0:
        new_budget = round(float(credit_budget_usd), 4)
        if early_renewal:
            existing = float(getattr(user, "monthly_credits_usd", 0.0) or 0.0)
            user.monthly_credits_usd = round(existing + new_budget, 4)
            # Preserve used credits and the original paid-window anchor.
            if user.usage_period_anchor is None:
                user.usage_period_anchor = paid_anchor
        else:
            user.monthly_credits_usd = new_budget
            user.monthly_agent_runs_used = 0
            user.monthly_credits_used_usd = 0.0
            user.usage_period_anchor = paid_anchor
    elif user.credits_billing_interval == "annual":
        user.monthly_credits_usd = _pro_annual_credit_budget_usd(settings)
    elif user.credits_billing_interval == "weekly":
        user.monthly_credits_usd = _pro_weekly_credit_budget_usd(settings)
    elif user.credits_billing_interval == "daily":
        user.monthly_credits_usd = _pro_daily_credit_budget_usd(settings)
    else:
        user.monthly_credits_usd = _pro_monthly_credit_budget_usd(settings)

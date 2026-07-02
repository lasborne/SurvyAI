from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.config import get_cloud_settings
from survyai_cloud.db import get_db
from survyai_cloud.deps import device_from_header
from survyai_cloud.models import Device, UsageEvent, User
from survyai_cloud.rate_limiting import rate_limit_user_dependency
from survyai_cloud.schemas import UsageBatchIn, UsageBatchOut
from survyai_cloud.services.entitlements import (
    can_use_platform_llm,
    ensure_usage_month_rolled,
    subscription_allows_platform_llm,
)
from utils.cost_estimator import estimate_token_cost_usd

router = APIRouter(prefix="/usage", tags=["usage"])


@router.post("/events", response_model=UsageBatchOut)
async def ingest_usage(
    body: UsageBatchIn,
    user: Annotated[
        User,
        Depends(rate_limit_user_dependency("usage_events", "rate_limit_usage_events_per_window")),
    ],
    db: Annotated[AsyncSession, Depends(get_db)],
    device: Annotated[Device | None, Depends(device_from_header)],
) -> UsageBatchOut:
    settings = get_cloud_settings()
    if len(body.events) > settings.usage_batch_max_events:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At most {settings.usage_batch_max_events} events per batch",
        )

    await ensure_usage_month_rolled(user, db)

    agent_run_delta = sum(e.quantity for e in body.events if e.kind == "agent_run")
    hosted_llm_allowed = subscription_allows_platform_llm(user, settings)
    computed_events: list[tuple[object, float, dict]] = []
    batch_cost_usd = 0.0

    for ev in body.events:
        meta = dict(ev.meta or {})
        computed_cost_usd = float(ev.cost_usd or 0.0)
        if ev.kind == "agent_run" and ev.model_name:
            computed_cost_usd = round(
                estimate_token_cost_usd(
                    ev.model_name,
                    ev.input_tokens,
                    ev.output_tokens,
                    cached_input_tokens=ev.cached_input_tokens,
                ),
                6,
            )
            meta["billing_basis"] = "server_computed_tokens"
            meta["model_name"] = ev.model_name
            meta["input_tokens"] = int(ev.input_tokens)
            meta["output_tokens"] = int(ev.output_tokens)
            meta["cached_input_tokens"] = int(ev.cached_input_tokens)
            if float(ev.cost_usd or 0.0) > 0:
                meta["client_cost_usd"] = round(float(ev.cost_usd), 6)
        else:
            meta["billing_basis"] = "client_cost_usd"
        batch_cost_usd += computed_cost_usd
        computed_events.append((ev, computed_cost_usd, meta))

    if agent_run_delta and hosted_llm_allowed:
        if device is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Pro usage events for agent runs must include a registered PC: "
                    "set header X-SurvyAI-Device-Id to the id returned by POST /v1/devices."
                ),
            )
        cap = user.monthly_agent_runs_quota
        if cap > 0 and user.monthly_agent_runs_used + agent_run_delta > cap:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Monthly agent run quota exceeded",
            )

    if batch_cost_usd > 0 and not can_use_platform_llm(user, settings):
        raise HTTPException(
            status_code=402,
            detail="Subscription API credit balance exhausted for this period",
        )

    dev_id = device.id if device is not None else None
    for ev, computed_cost_usd, meta in computed_events:
        db.add(
            UsageEvent(
                user_id=user.id,
                kind=ev.kind,
                quantity=ev.quantity,
                cost_usd=computed_cost_usd,
                meta=meta,
                device_id=dev_id,
            )
        )

    if agent_run_delta and hosted_llm_allowed:
        user.monthly_agent_runs_used += agent_run_delta
        db.add(user)

    if batch_cost_usd > 0:
        marked_up = batch_cost_usd * settings.credit_markup_multiplier
        if (
            hosted_llm_allowed
            and user.monthly_credits_usd > 0
            and user.monthly_credits_used_usd + marked_up > user.monthly_credits_usd + 1e-6
        ):
            raise HTTPException(
                status_code=402,
                detail="Subscription API credit balance exhausted for this period",
            )
        user.monthly_credits_used_usd = round(user.monthly_credits_used_usd + marked_up, 6)
        db.add(user)

    return UsageBatchOut(
        accepted=len(body.events),
        monthly_agent_runs_used=user.monthly_agent_runs_used,
        monthly_agent_runs_quota=user.monthly_agent_runs_quota,
        monthly_credits_used_usd=user.monthly_credits_used_usd,
        monthly_credits_usd=user.monthly_credits_usd,
    )

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.db import get_db
from survyai_cloud.deps import get_current_user, get_optional_user
from survyai_cloud.models import User, UserFeedback
from survyai_cloud.schemas import FeedbackIn, FeedbackOut, MeOut
from survyai_cloud.services.entitlements import ensure_usage_month_rolled, entitlements_for_user

router = APIRouter(tags=["me"])


@router.get("/me", response_model=MeOut)
async def get_me(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MeOut:
    await ensure_usage_month_rolled(user, db)
    ent = entitlements_for_user(user)
    return MeOut(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        plan_slug=ent.plan_slug,
        subscription_status=ent.subscription_status,
        subscription_current_period_end=user.subscription_current_period_end,
        usage_period_anchor=user.usage_period_anchor,
        can_manage_paystack_subscription=bool(user.paystack_subscription_code),
        max_devices=ent.max_devices,
        monthly_agent_runs_quota=ent.monthly_agent_runs_quota,
        monthly_agent_runs_used=ent.monthly_agent_runs_used,
        monthly_credits_usd=ent.monthly_credits_usd,
        monthly_credits_used_usd=ent.monthly_credits_used_usd,
        credit_markup_multiplier=ent.credit_markup_multiplier,
        credits_billing_interval=ent.credits_billing_interval,
        can_use_platform_llm=ent.can_use_platform_llm,
        primary_llm=ent.primary_llm or "",
    )


@router.post("/feedback", response_model=FeedbackOut)
async def submit_feedback(
    body: FeedbackIn,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[Optional[User], Depends(get_optional_user)],
) -> UserFeedback:
    email = (user.email or "").strip() if user is not None else (body.email or "").strip()
    row = UserFeedback(
        user_id=user.id if user is not None else None,
        email=email or None,
        name=body.name,
        trying_to_do=body.trying_to_do,
        trying_other=body.trying_other or None,
        what_happened=body.what_happened,
        expected=body.expected,
        rating=body.rating,
        context=body.context or None,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row

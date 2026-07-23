from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.config import get_cloud_settings
from survyai_cloud.db import get_db
from survyai_cloud.deps import get_current_user
from survyai_cloud.models import SubscriptionStatus, User
from survyai_cloud.schemas import (
    BillingPlanOption,
    BillingPlansOut,
    PaystackInitializeIn,
    PaystackInitializeOut,
    PaystackManageLinkOut,
    PaystackVerifyIn,
    PaystackVerifyOut,
)
from survyai_cloud.services.entitlements import (
    apply_pro_defaults,
    credit_budget_and_interval_from_paystack_payload,
    manual_payment_period_anchor,
    reconcile_pro_access,
    subscription_period_end_from_anchor,
)
from survyai_cloud.services.paystack import (
    PaystackError,
    disable_subscription,
    fetch_subscription_manage_link,
    initialize_transaction,
    verify_transaction,
)

router = APIRouter(prefix="/billing", tags=["billing"])


def _fmt_ngn(amount: int) -> str:
    return f"₦{amount:,}"


def _metadata_user_id(data: dict[str, Any]) -> Optional[str]:
    meta = data.get("metadata")
    if isinstance(meta, dict):
        uid = meta.get("survyai_user_id")
        if uid:
            return str(uid)
    return None


@router.get("/plans", response_model=BillingPlansOut)
async def list_paystack_plans(
    _user: Annotated[User, Depends(get_current_user)],
) -> BillingPlansOut:
    settings = get_cloud_settings()
    plans: list[BillingPlanOption] = []
    daily = settings.paystack_plan_code_pro_daily.strip()
    if daily:
        plans.append(
            BillingPlanOption(
                slug="pro_daily",
                label=(
                    f"SurvyAI Pro — {_fmt_ngn(settings.paystack_pro_daily_amount_ngn)} / day"
                ),
                plan_code=daily,
            )
        )
    weekly = settings.paystack_plan_code_pro_weekly.strip()
    if weekly:
        plans.append(
            BillingPlanOption(
                slug="pro_weekly",
                label=(
                    f"SurvyAI Pro — {_fmt_ngn(settings.paystack_pro_weekly_amount_ngn)} / week"
                ),
                plan_code=weekly,
            )
        )
    monthly = settings.paystack_plan_code_pro_monthly.strip()
    if monthly:
        plans.append(
            BillingPlanOption(
                slug="pro_monthly",
                label=(
                    f"SurvyAI Pro — {_fmt_ngn(settings.paystack_pro_monthly_amount_ngn)} / month"
                ),
                plan_code=monthly,
            )
        )
    annual = settings.paystack_plan_code_pro_annual.strip()
    if annual:
        plans.append(
            BillingPlanOption(
                slug="pro_annual",
                label=(
                    f"SurvyAI Pro — {_fmt_ngn(settings.paystack_pro_annual_amount_ngn)} / year"
                ),
                plan_code=annual,
            )
        )
    if not plans:
        raise HTTPException(
            status_code=501,
            detail=(
                "No Paystack plans configured. Set PAYSTACK_PLAN_CODE_PRO_DAILY, "
                "PAYSTACK_PLAN_CODE_PRO_WEEKLY, PAYSTACK_PLAN_CODE_PRO_MONTHLY, and/or "
                "PAYSTACK_PLAN_CODE_PRO_ANNUAL in .env.cloud (plan codes from Paystack Dashboard)."
            ),
        )
    return BillingPlansOut(plans=plans)


@router.post("/initialize", response_model=PaystackInitializeOut)
async def paystack_initialize(
    body: PaystackInitializeIn,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PaystackInitializeOut:
    settings = get_cloud_settings()
    if not settings.paystack_secret_key.strip():
        raise HTTPException(status_code=501, detail="Paystack is not configured on this server")

    plan = (body.plan_code or settings.paystack_plan_code_pro_monthly).strip()
    if not plan:
        raise HTTPException(status_code=400, detail="plan_code required (or set PAYSTACK_PLAN_CODE_PRO_MONTHLY)")

    try:
        res = initialize_transaction(
            email=user.email,
            plan_code=plan,
            callback_url=settings.paystack_callback_url,
            metadata={
                "survyai_user_id": str(user.id),
                "plan_slug": settings.pro_plan_slug,
                "requested_plan_code": plan,
            },
        )
    except PaystackError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    user.last_payment_reference = res.reference
    db.add(user)
    await db.flush()

    return PaystackInitializeOut(
        authorization_url=res.authorization_url,
        access_code=res.access_code,
        reference=res.reference,
    )


@router.post("/verify", response_model=PaystackVerifyOut)
async def paystack_verify(
    body: PaystackVerifyIn,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PaystackVerifyOut:
    """
    Confirm a completed Paystack transaction (fallback when webhooks are delayed or in dev).
    """
    settings = get_cloud_settings()
    if not settings.paystack_secret_key.strip():
        raise HTTPException(status_code=501, detail="Paystack is not configured on this server")

    ref = body.reference.strip()
    try:
        data = verify_transaction(reference=ref)
    except PaystackError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if str(data.get("status") or "").lower() != "success":
        return PaystackVerifyOut(
            ok=False,
            detail=f"Transaction not successful (status={data.get('status')!r})",
        )

    meta_uid = _metadata_user_id(data)
    if meta_uid and meta_uid != str(user.id):
        raise HTTPException(status_code=403, detail="This payment reference belongs to a different account")

    if meta_uid is None:
        cust = data.get("customer")
        email_ok = False
        if isinstance(cust, dict):
            email = str(cust.get("email") or "").lower().strip()
            if email:
                email_ok = email == user.email.lower().strip()
        if not email_ok and ref.strip() != (user.last_payment_reference or "").strip():
            raise HTTPException(
                status_code=403,
                detail="Cannot verify ownership of this transaction (missing metadata; reference must match your last checkout)",
            )

    budget_usd, bill_interval = credit_budget_and_interval_from_paystack_payload(data, settings)
    paid_at = data.get("paid_at")
    paid_anchor: datetime | None = None
    if isinstance(paid_at, str):
        try:
            dt = paid_at[:-1] + "+00:00" if paid_at.endswith("Z") else paid_at
            paid_anchor = datetime.fromisoformat(dt)
        except Exception:
            paid_anchor = None
    reconcile_pro_access(user, settings)
    anchor = manual_payment_period_anchor(user, paid_anchor, settings)
    apply_pro_defaults(
        user,
        settings,
        credit_budget_usd=budget_usd,
        credits_billing_interval=bill_interval,
        paid_at=paid_anchor,
    )
    user.subscription_status = SubscriptionStatus.active
    user.subscription_current_period_end = subscription_period_end_from_anchor(anchor, bill_interval)

    if isinstance(data.get("customer"), dict):
        cust = data["customer"]
        cc = cust.get("customer_code")
        if cc:
            user.paystack_customer_code = str(cc)
    sub_code = data.get("subscription_code")
    email_token = data.get("email_token")
    if sub_code and email_token:
        try:
            disable_subscription(subscription_code=str(sub_code), email_token=str(email_token), settings=settings)
            user.paystack_subscription_code = None
            user.paystack_email_token = None
            user.subscription_status = SubscriptionStatus.non_renewing
        except PaystackError:
            user.paystack_subscription_code = str(sub_code)
            user.paystack_email_token = str(email_token)
    elif sub_code:
        user.paystack_subscription_code = str(sub_code)
    elif email_token:
        user.paystack_email_token = str(email_token)
    user.last_payment_reference = ref

    db.add(user)
    await db.flush()

    return PaystackVerifyOut(
        ok=True,
        plan_slug=user.plan_slug,
        subscription_status=user.subscription_status.value,
    )


@router.get("/subscription/manage-link", response_model=PaystackManageLinkOut)
async def paystack_subscription_manage_link(
    user: Annotated[User, Depends(get_current_user)],
) -> PaystackManageLinkOut:
    settings = get_cloud_settings()
    if not settings.paystack_secret_key.strip():
        raise HTTPException(status_code=501, detail="Paystack is not configured on this server")
    code = (user.paystack_subscription_code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="No Paystack subscription on file for this account")
    try:
        url = fetch_subscription_manage_link(subscription_code=code)
    except PaystackError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return PaystackManageLinkOut(url=url)

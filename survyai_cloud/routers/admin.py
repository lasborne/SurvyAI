"""Support and ops overrides + read APIs (Phase 5/8)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.db import get_db
from survyai_cloud.deps import require_admin
from survyai_cloud.models import Device, DiagnosticsBundle, SubscriptionStatus, UsageEvent, User
from survyai_cloud.schemas import (
    AdminDiagnosticsOut,
    AdminUsageEventOut,
    AdminUserBillingPatch,
    AdminUserSnapshot,
    DeviceOut,
)
from survyai_cloud.services.entitlements import apply_free_defaults, apply_pro_defaults

router = APIRouter(prefix="/admin", tags=["admin"])


def _snapshot(user: User, device_count: int) -> AdminUserSnapshot:
    return AdminUserSnapshot(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        plan_slug=user.plan_slug,
        subscription_status=user.subscription_status.value
        if hasattr(user.subscription_status, "value")
        else str(user.subscription_status),
        subscription_current_period_end=user.subscription_current_period_end,
        grace_period_ends_at=user.grace_period_ends_at,
        last_reactivation_at=user.last_reactivation_at,
        max_devices=user.max_devices,
        monthly_agent_runs_quota=user.monthly_agent_runs_quota,
        monthly_agent_runs_used=user.monthly_agent_runs_used,
        monthly_credits_usd=float(user.monthly_credits_usd or 0.0),
        monthly_credits_used_usd=float(user.monthly_credits_used_usd or 0.0),
        credits_billing_interval=user.credits_billing_interval or "monthly",
        usage_period_anchor=user.usage_period_anchor,
        paystack_customer_code=user.paystack_customer_code,
        paystack_subscription_code=user.paystack_subscription_code,
        last_payment_reference=user.last_payment_reference,
        device_count=device_count,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


async def _device_count(db: AsyncSession, user_id: uuid.UUID) -> int:
    res = await db.execute(select(func.count()).select_from(Device).where(Device.user_id == user_id))
    return int(res.scalar_one() or 0)


async def _get_user_or_404(db: AsyncSession, user_id: uuid.UUID) -> User:
    res = await db.execute(select(User).where(User.id == user_id))
    user = res.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/users", response_model=AdminUserSnapshot)
async def admin_lookup_user_by_email(
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    email: Annotated[str, Query(min_length=3, max_length=320)],
) -> AdminUserSnapshot:
    """Lookup a user support snapshot by email (case-insensitive)."""
    needle = email.strip().lower()
    res = await db.execute(select(User).where(func.lower(User.email) == needle))
    user = res.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _snapshot(user, await _device_count(db, user.id))


@router.get("/users/{user_id}", response_model=AdminUserSnapshot)
async def admin_get_user(
    user_id: uuid.UUID,
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AdminUserSnapshot:
    user = await _get_user_or_404(db, user_id)
    return _snapshot(user, await _device_count(db, user.id))


@router.get("/users/{user_id}/devices", response_model=list[DeviceOut])
async def admin_list_user_devices(
    user_id: uuid.UUID,
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[Device]:
    await _get_user_or_404(db, user_id)
    res = await db.execute(
        select(Device).where(Device.user_id == user_id).order_by(Device.created_at.desc())
    )
    return list(res.scalars().all())


@router.get("/users/{user_id}/usage", response_model=list[AdminUsageEventOut])
async def admin_list_user_usage(
    user_id: uuid.UUID,
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[UsageEvent]:
    await _get_user_or_404(db, user_id)
    res = await db.execute(
        select(UsageEvent)
        .where(UsageEvent.user_id == user_id)
        .order_by(UsageEvent.created_at.desc())
        .limit(limit)
    )
    return list(res.scalars().all())


@router.get("/users/{user_id}/diagnostics", response_model=list[AdminDiagnosticsOut])
async def admin_list_user_diagnostics(
    user_id: uuid.UUID,
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[DiagnosticsBundle]:
    await _get_user_or_404(db, user_id)
    res = await db.execute(
        select(DiagnosticsBundle)
        .where(DiagnosticsBundle.user_id == user_id)
        .order_by(DiagnosticsBundle.created_at.desc())
        .limit(limit)
    )
    return list(res.scalars().all())


@router.patch("/users/{user_id}/billing", status_code=status.HTTP_200_OK)
async def admin_patch_user_billing(
    user_id: uuid.UUID,
    body: AdminUserBillingPatch,
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, object]:
    user = await _get_user_or_404(db, user_id)

    if body.apply_free_defaults:
        apply_free_defaults(user)
    if body.apply_pro_defaults:
        apply_pro_defaults(user)

    if body.plan_slug is not None:
        user.plan_slug = body.plan_slug.strip()

    if body.subscription_status is not None:
        raw = body.subscription_status.strip().lower()
        try:
            user.subscription_status = SubscriptionStatus(raw)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid subscription_status: " + raw)

    if body.clear_grace:
        user.grace_period_ends_at = None
    elif body.grace_period_ends_at is not None:
        user.grace_period_ends_at = body.grace_period_ends_at

    if body.touch_reactivated_now:
        user.last_reactivation_at = datetime.now(timezone.utc)
    elif body.last_reactivation_at is not None:
        user.last_reactivation_at = body.last_reactivation_at

    if body.reset_credits_used:
        user.monthly_credits_used_usd = 0.0
    elif body.monthly_credits_used_usd is not None:
        user.monthly_credits_used_usd = float(body.monthly_credits_used_usd)

    if body.monthly_credits_usd is not None:
        user.monthly_credits_usd = float(body.monthly_credits_usd)

    if body.max_devices is not None:
        user.max_devices = int(body.max_devices)

    if body.clear_period_end:
        user.subscription_current_period_end = None
    elif body.subscription_current_period_end is not None:
        user.subscription_current_period_end = body.subscription_current_period_end

    db.add(user)
    await db.flush()
    await db.refresh(user)
    return {
        "ok": True,
        "user_id": str(user.id),
        "plan_slug": user.plan_slug,
        "subscription_status": user.subscription_status.value,
        "grace_period_ends_at": user.grace_period_ends_at.isoformat()
        if user.grace_period_ends_at
        else None,
        "last_reactivation_at": user.last_reactivation_at.isoformat()
        if user.last_reactivation_at
        else None,
        "monthly_credits_usd": float(user.monthly_credits_usd or 0.0),
        "monthly_credits_used_usd": float(user.monthly_credits_used_usd or 0.0),
        "max_devices": user.max_devices,
        "subscription_current_period_end": user.subscription_current_period_end.isoformat()
        if user.subscription_current_period_end
        else None,
    }


@router.delete("/users/{user_id}/devices/{device_id}", status_code=status.HTTP_204_NO_CONTENT)
async def admin_delete_user_device(
    user_id: uuid.UUID,
    device_id: uuid.UUID,
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    res = await db.execute(select(Device).where(Device.id == device_id, Device.user_id == user_id))
    if res.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Device not found")
    await db.execute(delete(Device).where(Device.id == device_id, Device.user_id == user_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)

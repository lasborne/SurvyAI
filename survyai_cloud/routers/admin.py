"""Support and ops overrides + read APIs (Phase 5/8)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Optional

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
        admin_privilege_active=bool(getattr(user, "admin_privilege_active", False)),
        admin_privilege_note=getattr(user, "admin_privilege_note", None),
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


def _looks_like_legacy_admin_grant(user: User) -> bool:
    """Pre-migration admin Pro grants: Pro + credits, no Paystack subscription code."""
    plan = str(user.plan_slug or "").strip().lower()
    if plan != "pro":
        return False
    if float(user.monthly_credits_usd or 0.0) <= 0:
        return False
    sub = str(user.paystack_subscription_code or "").strip()
    return not sub


def _mark_admin_privilege(user: User, *, note: Optional[str] = None) -> None:
    user.admin_privilege_active = True
    if note is not None:
        cleaned = str(note).strip()
        user.admin_privilege_note = cleaned[:500] if cleaned else None


def _clear_admin_privilege(user: User) -> None:
    user.admin_privilege_active = False
    user.admin_privilege_note = None


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


@router.get("/privileges", response_model=list[AdminUserSnapshot])
async def admin_list_privileges(
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> list[AdminUserSnapshot]:
    """
    List users with admin-managed privileges (grant / extend / credit overrides).

    Includes explicit ``admin_privilege_active`` rows and legacy Pro grants that
    have credits but no Paystack subscription code (pre-flag admin overrides).
    """
    res = await db.execute(select(User).order_by(User.updated_at.desc()).limit(2000))
    users = list(res.scalars().all())
    out: list[AdminUserSnapshot] = []
    backfilled = False
    for user in users:
        active = bool(getattr(user, "admin_privilege_active", False))
        legacy = (not active) and _looks_like_legacy_admin_grant(user)
        if not active and not legacy:
            continue
        if legacy:
            # Soft backfill so the dashboard stays consistent after first list.
            user.admin_privilege_active = True
            db.add(user)
            backfilled = True
        out.append(_snapshot(user, await _device_count(db, user.id)))
        if len(out) >= limit:
            break
    if backfilled:
        await db.flush()
    return out


@router.delete("/privileges/{user_id}", status_code=status.HTTP_200_OK)
async def admin_revoke_privilege(
    user_id: uuid.UUID,
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, object]:
    """Revoke admin-managed privilege and restore Free defaults."""
    user = await _get_user_or_404(db, user_id)
    apply_free_defaults(user)
    _clear_admin_privilege(user)
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return {
        "ok": True,
        "user_id": str(user.id),
        "email": user.email,
        "plan_slug": user.plan_slug,
        "admin_privilege_active": False,
    }


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

    if body.revoke_privilege or body.apply_free_defaults:
        apply_free_defaults(user)
        _clear_admin_privilege(user)

    if body.apply_pro_defaults and not body.revoke_privilege and not body.apply_free_defaults:
        apply_pro_defaults(user)
        _mark_admin_privilege(user)

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

    if body.clear_privilege_note:
        user.admin_privilege_note = None
    elif body.admin_privilege_note is not None:
        note = str(body.admin_privilege_note).strip()
        user.admin_privilege_note = note[:500] if note else None

    # Any non-revoke privilege adjustment marks the user as admin-managed.
    privilege_touch = any(
        (
            body.apply_pro_defaults,
            body.monthly_credits_usd is not None,
            body.monthly_credits_used_usd is not None,
            body.reset_credits_used,
            body.subscription_current_period_end is not None,
            body.plan_slug is not None
            and str(body.plan_slug).strip().lower() in {"pro", "trialing"},
            body.subscription_status is not None
            and str(body.subscription_status).strip().lower()
            in {"trialing", "active", "non_renewing"},
        )
    )
    if privilege_touch and not body.revoke_privilege and not body.apply_free_defaults:
        _mark_admin_privilege(user, note=body.admin_privilege_note)

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
        "admin_privilege_active": bool(getattr(user, "admin_privilege_active", False)),
        "admin_privilege_note": getattr(user, "admin_privilege_note", None),
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

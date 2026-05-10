"""Support and ops overrides (Phase 5)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.db import get_db
from survyai_cloud.deps import require_admin
from survyai_cloud.models import Device, SubscriptionStatus, User
from survyai_cloud.schemas import AdminUserBillingPatch
from survyai_cloud.services.entitlements import apply_free_defaults, apply_pro_defaults

router = APIRouter(prefix="/admin", tags=["admin"])


@router.patch("/users/{user_id}/billing", status_code=status.HTTP_200_OK)
async def admin_patch_user_billing(
    user_id: uuid.UUID,
    body: AdminUserBillingPatch,
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, object]:
    res = await db.execute(select(User).where(User.id == user_id))
    user = res.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

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


from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.config import get_cloud_settings
from survyai_cloud.db import get_db
from survyai_cloud.deps import device_from_header, get_current_user
from survyai_cloud.models import Device, User
from survyai_cloud.rate_limiting import rate_limit_user_dependency
from survyai_cloud.schemas import BootstrapOut, EntitlementsOut
from survyai_cloud.services.entitlements import (
    build_bootstrap_payload,
    can_use_platform_llm,
    entitlements_for_user,
    ensure_usage_month_rolled,
)

router = APIRouter(tags=["entitlements"])


@router.get("/entitlements", response_model=EntitlementsOut)
async def get_entitlements(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> EntitlementsOut:
    await ensure_usage_month_rolled(user, db)
    return entitlements_for_user(user)


@router.get("/bootstrap", response_model=BootstrapOut)
async def bootstrap_keys(
    user: Annotated[
        User,
        Depends(rate_limit_user_dependency("bootstrap", "rate_limit_bootstrap_per_window")),
    ],
    db: Annotated[AsyncSession, Depends(get_db)],
    device: Annotated[Device | None, Depends(device_from_header)],
) -> BootstrapOut:
    """
    Platform LLM keys are only issued for an active Pro subscription and only for a
    registered PC (``X-SurvyAI-Device-Id`` from ``POST /v1/devices``). Pro is capped at
    ``user.max_devices`` (default 2) distinct fingerprints.
    """
    await ensure_usage_month_rolled(user, db)
    settings = get_cloud_settings()
    if can_use_platform_llm(user, settings):
        if device is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "This PC must be registered before hosted model keys are issued. "
                    "Call POST /v1/devices with a stable machine fingerprint, then retry with header "
                    "X-SurvyAI-Device-Id set to the returned device id."
                ),
            )
        device.last_seen_at = datetime.now(timezone.utc)
        db.add(device)
        await db.flush()
    return build_bootstrap_payload(user)

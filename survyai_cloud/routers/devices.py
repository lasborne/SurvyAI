from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.db import get_db
from survyai_cloud.deps import get_current_user
from survyai_cloud.models import Device, User
from survyai_cloud.schemas import DeviceOut, DeviceRegisterIn

router = APIRouter(prefix="/devices", tags=["devices"])


@router.get("", response_model=list[DeviceOut])
async def list_devices(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[Device]:
    res = await db.execute(select(Device).where(Device.user_id == user.id))
    return list(res.scalars().all())


@router.post("", response_model=DeviceOut)
async def register_device(
    body: DeviceRegisterIn,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Device:
    # Known PC first: at the Pro cap (e.g. 2), this machine must still be able to refresh
    # its registration without being blocked by the global count check.
    existing = await db.execute(
        select(Device).where(Device.user_id == user.id, Device.fingerprint == body.fingerprint)
    )
    hit = existing.scalar_one_or_none()
    if hit is not None:
        hit.label = body.label or hit.label
        hit.last_seen_at = datetime.now(timezone.utc)
        db.add(hit)
        await db.flush()
        await db.refresh(hit)
        return hit

    res = await db.execute(select(func.count()).select_from(Device).where(Device.user_id == user.id))
    n = int(res.scalar_one())
    if n >= user.max_devices:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Device limit reached ({user.max_devices} active PCs for your plan). "
                "Remove another device from your account (GET /v1/devices, DELETE /v1/devices/{{id}}) "
                "or use SurvyAI on a PC already registered."
            ),
        )

    dev = Device(
        user_id=user.id,
        fingerprint=body.fingerprint,
        label=body.label,
        last_seen_at=datetime.now(timezone.utc),
    )
    db.add(dev)
    await db.flush()
    await db.refresh(dev)
    return dev


@router.delete("/{device_id}")
async def delete_device(
    device_id: uuid.UUID,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    res = await db.execute(
        select(Device).where(Device.id == device_id, Device.user_id == user.id)
    )
    dev = res.scalar_one_or_none()
    if dev is None:
        raise HTTPException(status_code=404, detail="Device not found")
    await db.execute(delete(Device).where(Device.id == device_id, Device.user_id == user.id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)

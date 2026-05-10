from __future__ import annotations

import uuid
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.db import get_db
from survyai_cloud.models import Device, User
from survyai_cloud.security import safe_decode_access_token

security = HTTPBearer(auto_error=False)


async def get_current_user(
    db: Annotated[AsyncSession, Depends(get_db)],
    creds: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
) -> User:
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = safe_decode_access_token(creds.credentials)
    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject")
    try:
        uid = uuid.UUID(str(sub))
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user id")
    res = await db.execute(select(User).where(User.id == uid))
    user = res.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


async def device_from_header(
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    x_survyai_device_id: Annotated[Optional[str], Header(alias="X-SurvyAI-Device-Id")] = None,
) -> Optional[Device]:
    if not x_survyai_device_id:
        return None
    try:
        did = uuid.UUID(x_survyai_device_id)
    except ValueError:
        return None
    res = await db.execute(
        select(Device).where(Device.id == did, Device.user_id == user.id)
    )
    return res.scalar_one_or_none()

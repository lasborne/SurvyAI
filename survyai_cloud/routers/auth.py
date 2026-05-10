from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.config import get_cloud_settings
from survyai_cloud.db import get_db
from survyai_cloud.models import RefreshToken, User
from survyai_cloud.schemas import LoginIn, RefreshIn, TokenPair, UserCreate, UserOut
from survyai_cloud.security import (
    create_access_token,
    hash_password,
    hash_refresh_token,
    new_refresh_token_value,
    refresh_token_expiry,
    verify_password,
)
from survyai_cloud.services.entitlements import apply_free_defaults

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut)
async def register(body: UserCreate, db: Annotated[AsyncSession, Depends(get_db)]) -> User:
    settings = get_cloud_settings()
    user = User(
        email=body.email.lower().strip(),
        password_hash=hash_password(body.password),
        display_name=body.display_name,
    )
    apply_free_defaults(user, settings)
    db.add(user)
    try:
        await db.flush()
    except IntegrityError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    await db.refresh(user)
    return user


@router.post("/login", response_model=TokenPair)
async def login(body: LoginIn, db: Annotated[AsyncSession, Depends(get_db)]) -> TokenPair:
    settings = get_cloud_settings()
    res = await db.execute(select(User).where(User.email == body.email.lower().strip()))
    user = res.scalar_one_or_none()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    access = create_access_token(subject=str(user.id))
    raw_refresh = new_refresh_token_value()
    rt = RefreshToken(
        user_id=user.id,
        token_hash=hash_refresh_token(raw_refresh),
        expires_at=refresh_token_expiry(),
    )
    db.add(rt)
    await db.flush()
    return TokenPair(
        access_token=access,
        refresh_token=raw_refresh,
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/refresh", response_model=TokenPair)
async def refresh_token(body: RefreshIn, db: Annotated[AsyncSession, Depends(get_db)]) -> TokenPair:
    settings = get_cloud_settings()
    th = hash_refresh_token(body.refresh_token)
    res = await db.execute(
        select(RefreshToken).where(
            RefreshToken.token_hash == th,
            RefreshToken.revoked.is_(False),
        )
    )
    row = res.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    from datetime import datetime, timezone

    if row.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token expired")

    row.revoked = True
    db.add(row)

    res_user = await db.execute(select(User).where(User.id == row.user_id))
    user = res_user.scalar_one()
    access = create_access_token(subject=str(user.id))
    raw_refresh = new_refresh_token_value()
    new_rt = RefreshToken(
        user_id=user.id,
        token_hash=hash_refresh_token(raw_refresh),
        expires_at=refresh_token_expiry(),
    )
    db.add(new_rt)
    return TokenPair(
        access_token=access,
        refresh_token=raw_refresh,
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/logout")
async def logout(body: RefreshIn, db: Annotated[AsyncSession, Depends(get_db)]) -> Response:
    th = hash_refresh_token(body.refresh_token)
    res = await db.execute(select(RefreshToken).where(RefreshToken.token_hash == th))
    row = res.scalar_one_or_none()
    if row:
        row.revoked = True
        db.add(row)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.config import get_cloud_settings
from survyai_cloud.db import get_db
from survyai_cloud.models import PasswordResetToken, RefreshToken, User
from survyai_cloud.rate_limiting import rate_limit_ip_dependency, rate_limit_user_dependency
from survyai_cloud.schemas import (
    ChangePasswordIn,
    ForgotPasswordIn,
    ForgotPasswordOut,
    LoginIn,
    RefreshIn,
    ResetPasswordIn,
    TokenPair,
    UserCreate,
    UserOut,
)
from survyai_cloud.security import (
    create_access_token,
    hash_password,
    hash_password_reset_code,
    hash_refresh_token,
    new_password_reset_code,
    new_refresh_token_value,
    refresh_token_expiry,
    validate_password_strength,
    verify_password,
)
from survyai_cloud.services.email import send_password_reset_email
from survyai_cloud.services.entitlements import apply_free_defaults

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_FORGOT_OK = ForgotPasswordOut()


def _client_ip(request: Request) -> Optional[str]:
    if request.client and request.client.host:
        return str(request.client.host)[:64]
    return None


async def _revoke_all_refresh_tokens(db: AsyncSession, user_id) -> None:
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked.is_(False))
        .values(revoked=True)
    )


async def _issue_token_pair(db: AsyncSession, user: User) -> TokenPair:
    settings = get_cloud_settings()
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


@router.post("/register", response_model=UserOut)
async def register(body: UserCreate, db: Annotated[AsyncSession, Depends(get_db)]) -> User:
    settings = get_cloud_settings()
    email = body.email.lower().strip()
    err = validate_password_strength(body.password, email=email)
    if err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=err)

    now = datetime.now(timezone.utc)
    user = User(
        email=email,
        password_hash=hash_password(body.password),
        display_name=body.display_name,
        password_changed_at=now,
    )
    apply_free_defaults(user, settings)
    db.add(user)
    try:
        await db.flush()
    except IntegrityError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    await db.refresh(user)
    return user


@router.post(
    "/login",
    response_model=TokenPair,
    dependencies=[
        Depends(
            rate_limit_ip_dependency(
                "auth_login",
                "rate_limit_auth_login_per_window",
                window_seconds=900,
            )
        )
    ],
)
async def login(body: LoginIn, db: Annotated[AsyncSession, Depends(get_db)]) -> TokenPair:
    res = await db.execute(select(User).where(User.email == body.email.lower().strip()))
    user = res.scalar_one_or_none()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    return await _issue_token_pair(db, user)


@router.post("/refresh", response_model=TokenPair)
async def refresh_token(body: RefreshIn, db: Annotated[AsyncSession, Depends(get_db)]) -> TokenPair:
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

    expires_at = row.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token expired")

    row.revoked = True
    db.add(row)

    res_user = await db.execute(select(User).where(User.id == row.user_id))
    user = res_user.scalar_one()
    return await _issue_token_pair(db, user)


@router.post("/logout")
async def logout(body: RefreshIn, db: Annotated[AsyncSession, Depends(get_db)]) -> Response:
    th = hash_refresh_token(body.refresh_token)
    res = await db.execute(select(RefreshToken).where(RefreshToken.token_hash == th))
    row = res.scalar_one_or_none()
    if row:
        row.revoked = True
        db.add(row)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/forgot-password",
    response_model=ForgotPasswordOut,
    dependencies=[
        Depends(
            rate_limit_ip_dependency(
                "auth_forgot",
                "rate_limit_auth_forgot_per_window",
                window_seconds=3600,
            )
        )
    ],
)
async def forgot_password(
    body: ForgotPasswordIn,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ForgotPasswordOut:
    """Always return a generic success message (anti-enumeration)."""
    settings = get_cloud_settings()
    email = body.email.lower().strip()
    res = await db.execute(select(User).where(User.email == email))
    user = res.scalar_one_or_none()
    if user is None:
        return _FORGOT_OK

    now = datetime.now(timezone.utc)
    await db.execute(
        update(PasswordResetToken)
        .where(
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.used_at.is_(None),
        )
        .values(used_at=now)
    )

    raw_code = new_password_reset_code(settings.password_reset_code_length)
    token = PasswordResetToken(
        user_id=user.id,
        token_hash=hash_password_reset_code(raw_code),
        expires_at=now + timedelta(minutes=settings.password_reset_ttl_minutes),
        request_ip=_client_ip(request),
    )
    db.add(token)
    await db.flush()

    try:
        send_password_reset_email(
            to_email=email,
            code=raw_code,
            expires_minutes=settings.password_reset_ttl_minutes,
            settings=settings,
        )
    except Exception:
        logger.exception("Password reset email unexpected failure for user_id=%s", user.id)

    return _FORGOT_OK


@router.post(
    "/reset-password",
    dependencies=[
        Depends(
            rate_limit_ip_dependency(
                "auth_reset",
                "rate_limit_auth_reset_per_window",
                window_seconds=3600,
            )
        )
    ],
)
async def reset_password(
    body: ResetPasswordIn,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, str]:
    settings = get_cloud_settings()
    email = body.email.lower().strip()
    err = validate_password_strength(body.new_password, email=email)
    if err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=err)

    res = await db.execute(select(User).where(User.email == email))
    user = res.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset code",
        )

    code_hash = hash_password_reset_code(body.code)
    now = datetime.now(timezone.utc)
    tres = await db.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.token_hash == code_hash,
            PasswordResetToken.used_at.is_(None),
        )
    )
    token = tres.scalar_one_or_none()
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset code",
        )
    expires_at = token.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < now:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset code",
        )

    user.password_hash = hash_password(body.new_password, settings)
    user.password_changed_at = now
    token.used_at = now
    db.add(user)
    db.add(token)

    await db.execute(
        update(PasswordResetToken)
        .where(
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.used_at.is_(None),
            PasswordResetToken.id != token.id,
        )
        .values(used_at=now)
    )
    await _revoke_all_refresh_tokens(db, user.id)
    await db.flush()
    return {"detail": "Password updated. Sign in with your new password."}


@router.post(
    "/change-password",
    response_model=TokenPair,
)
async def change_password(
    body: ChangePasswordIn,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[
        User,
        Depends(
            rate_limit_user_dependency(
                "auth_change_password",
                "rate_limit_auth_change_password_per_window",
            )
        ),
    ],
) -> TokenPair:
    settings = get_cloud_settings()
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )
    err = validate_password_strength(body.new_password, email=user.email)
    if err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=err)
    if verify_password(body.new_password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from the current password",
        )

    now = datetime.now(timezone.utc)
    user.password_hash = hash_password(body.new_password, settings)
    user.password_changed_at = now
    db.add(user)
    await _revoke_all_refresh_tokens(db, user.id)
    await db.flush()
    return await _issue_token_pair(db, user)

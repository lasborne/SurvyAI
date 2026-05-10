from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import bcrypt
from jose import JWTError, jwt

from survyai_cloud.config import CloudSettings, get_cloud_settings


def hash_password(plain: str, settings: Optional[CloudSettings] = None) -> str:
    settings = settings or get_cloud_settings()
    rounds = max(4, min(settings.bcrypt_rounds, 31))
    salt = bcrypt.gensalt(rounds=rounds)
    return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain: str, password_hash: str, settings: Optional[CloudSettings] = None) -> bool:
    _ = settings
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(
    *,
    subject: str,
    settings: Optional[CloudSettings] = None,
    extra_claims: Optional[dict[str, Any]] = None,
) -> str:
    settings = settings or get_cloud_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {"sub": subject, "exp": expire, "type": "access"}
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str, settings: Optional[CloudSettings] = None) -> dict[str, Any]:
    settings = settings or get_cloud_settings()
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


def safe_decode_access_token(token: str, settings: Optional[CloudSettings] = None) -> Optional[dict[str, Any]]:
    try:
        return decode_access_token(token, settings=settings)
    except JWTError:
        return None


def new_refresh_token_value() -> str:
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def refresh_token_expiry(settings: Optional[CloudSettings] = None) -> datetime:
    settings = settings or get_cloud_settings()
    return datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)

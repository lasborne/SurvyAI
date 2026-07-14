from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import bcrypt
from jose import JWTError, jwt

from survyai_cloud.config import CloudSettings, get_cloud_settings

PASSWORD_MIN_LENGTH = 10
PASSWORD_MAX_LENGTH = 128
PASSWORD_SPECIAL_CHARS = r"!@#$%^&*()_+\/-=\[\]{}|;:,.<>?"
_PASSWORD_SPECIAL_RE = re.compile(rf"[{PASSWORD_SPECIAL_CHARS}]")
_COMMON_PASSWORDS = frozenset(
    {
        "password",
        "password1",
        "password1!",
        "password123",
        "password123!",
        "passw0rd",
        "passw0rd!",
        "1234567890",
        "1234567890!",
        "qwerty1234",
        "qwerty1234!",
        "welcome123",
        "welcome123!",
        "letmein123",
        "letmein123!",
        "admin12345",
        "admin12345!",
        "survyai123",
        "survyai123!",
        "changeme12",
        "changeme12!",
        "iloveyou12",
        "iloveyou12!",
        "password12",
        "Password1!",
        "Password12",
        "Password12!",
        "P@ssw0rd",
        "P@ssw0rd1",
        "P@ssword1",
    }
)

# Unambiguous charset for emailed reset codes (no 0/O, 1/I/l).
_RESET_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def validate_password_strength(password: str, *, email: str | None = None) -> Optional[str]:
    """
    Return an error message if ``password`` fails policy, else ``None``.

    Policy: 10–128 chars; upper + lower + digit + special; not email local-part;
    not on a small common-password denylist.
    """
    plain = password or ""
    if len(plain) < PASSWORD_MIN_LENGTH:
        return f"Password must be at least {PASSWORD_MIN_LENGTH} characters."
    if len(plain) > PASSWORD_MAX_LENGTH:
        return f"Password must be at most {PASSWORD_MAX_LENGTH} characters."
    if not re.search(r"[a-z]", plain):
        return "Password must include at least one lowercase letter."
    if not re.search(r"[A-Z]", plain):
        return "Password must include at least one uppercase letter."
    if not re.search(r"[0-9]", plain):
        return "Password must include at least one digit."
    if not _PASSWORD_SPECIAL_RE.search(plain):
        return (
            "Password must include at least one special character "
            f"({PASSWORD_SPECIAL_CHARS})."
        )
    if plain.lower() in _COMMON_PASSWORDS or plain in _COMMON_PASSWORDS:
        return "That password is too common. Please choose a stronger password."
    if email:
        local = str(email).split("@", 1)[0].strip().lower()
        if local and len(local) >= 3 and local in plain.lower():
            return "Password must not contain your email username."
    return None


def new_password_reset_code(length: int = 8) -> str:
    """Generate a short one-time reset code suitable for typing in the desktop app."""
    n = max(6, min(int(length or 8), 16))
    return "".join(secrets.choice(_RESET_CODE_ALPHABET) for _ in range(n))


def hash_password_reset_code(code: str) -> str:
    """SHA-256 hex of the normalized reset code (never store plaintext)."""
    normalized = re.sub(r"\s+", "", (code or "").strip().upper())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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

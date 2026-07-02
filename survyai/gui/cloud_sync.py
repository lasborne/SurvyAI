"""
Network-only cloud account sync (no Qt).

Used from background threads so the desktop UI stays responsive during
Refresh cloud account, sign-in follow-up, and credits sync.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from survyai.cloud_api import (
    CloudApiError,
    CloudTokenPair,
    access_token_expires_at_iso,
    get_bootstrap,
    get_entitlements,
    get_me,
    refresh_tokens,
    register_device,
)
from survyai.cloud_user_message import user_facing_cloud_message
from survyai.device_identity import compute_machine_fingerprint


def _entitlements_allow_platform_llm(me: dict, ent: dict) -> bool:
    if isinstance(ent, dict) and ent.get("can_use_platform_llm") is True:
        return True
    if isinstance(me, dict) and me.get("can_use_platform_llm") is True:
        return True
    return False


def _token_needs_refresh(expires_at_iso: str, *, now: Optional[datetime] = None) -> bool:
    exp_raw = (expires_at_iso or "").strip()
    if not exp_raw:
        return True
    now = now or datetime.now(timezone.utc)
    try:
        exp = datetime.fromisoformat(exp_raw.replace("Z", "+00:00"))
        return now >= exp - timedelta(minutes=2)
    except Exception:
        return True


@dataclass(frozen=True)
class CloudAccountSyncPayload:
    base_url: str
    access_token: str
    refresh_token: str
    access_token_expires_at: str
    device_id: str = ""
    device_fingerprint: str = ""
    machine_label: Optional[str] = None


@dataclass
class CloudAccountSyncResult:
    ok: bool = False
    session_expired: bool = False
    error_message: str = ""
    me: dict[str, Any] = field(default_factory=dict)
    ent: dict[str, Any] = field(default_factory=dict)
    bootstrap: dict[str, Any] = field(default_factory=dict)
    access_token: str = ""
    refresh_token: str = ""
    access_token_expires_at: str = ""
    device_id: str = ""
    device_fingerprint: str = ""
    registered: bool = False
    pro_keys: bool = False
    bootstrap_status: str = "ok"


@dataclass(frozen=True)
class CloudCreditsSyncPayload:
    base_url: str
    access_token: str
    refresh_token: str
    access_token_expires_at: str


@dataclass
class CloudCreditsSyncResult:
    ok: bool = False
    session_expired: bool = False
    error_message: str = ""
    ent: dict[str, Any] = field(default_factory=dict)
    access_token: str = ""
    refresh_token: str = ""
    access_token_expires_at: str = ""


def _refresh_access_token(
    payload: CloudAccountSyncPayload,
) -> tuple[str, str, str, Optional[CloudTokenPair]]:
    """Return (access_token, refresh_token, expires_at, optional new token pair)."""
    base = payload.base_url.strip()
    access = payload.access_token.strip()
    refresh = payload.refresh_token.strip()
    expires_at = payload.access_token_expires_at.strip()

    if not _token_needs_refresh(expires_at):
        return access, refresh, expires_at, None

    if not refresh:
        raise CloudApiError("Cloud session expired (no refresh token).")

    tokens = refresh_tokens(base_url=base, refresh_token=refresh)
    return (
        tokens.access_token,
        tokens.refresh_token,
        access_token_expires_at_iso(expires_in_seconds=tokens.expires_in),
        tokens,
    )


def _refresh_access_for_credits(
    payload: CloudCreditsSyncPayload,
) -> tuple[str, str, str, Optional[CloudTokenPair]]:
    p = CloudAccountSyncPayload(
        base_url=payload.base_url,
        access_token=payload.access_token,
        refresh_token=payload.refresh_token,
        access_token_expires_at=payload.access_token_expires_at,
    )
    return _refresh_access_token(p)


def sync_cloud_account(payload: CloudAccountSyncPayload) -> CloudAccountSyncResult:
    """
    Pull /v1/me, entitlements, device registration, and bootstrap off the UI thread.
    """
    result = CloudAccountSyncResult()
    base = payload.base_url.strip()
    if not base:
        result.error_message = "Cloud API base URL is not configured."
        return result

    try:
        access, refresh, expires_at, _tokens = _refresh_access_token(payload)
    except CloudApiError as exc:
        result.session_expired = True
        result.error_message = user_facing_cloud_message(exc)
        return result
    except Exception as exc:
        result.error_message = user_facing_cloud_message(exc)
        return result

    result.access_token = access
    result.refresh_token = refresh or payload.refresh_token.strip()
    result.access_token_expires_at = expires_at

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_me = pool.submit(get_me, base_url=base, access_token=access)
            f_ent = pool.submit(get_entitlements, base_url=base, access_token=access)
            me = f_me.result()
            ent = f_ent.result()
    except CloudApiError as exc:
        result.error_message = user_facing_cloud_message(exc)
        return result
    except Exception as exc:
        result.error_message = user_facing_cloud_message(exc)
        return result

    me_d = me if isinstance(me, dict) else {}
    ent_d = ent if isinstance(ent, dict) else {}
    result.me = me_d
    result.ent = ent_d
    result.pro_keys = _entitlements_allow_platform_llm(me_d, ent_d)

    fp = compute_machine_fingerprint()
    device_id = payload.device_id.strip()
    if payload.device_fingerprint.strip() and payload.device_fingerprint.strip() != fp:
        device_id = ""
    result.device_fingerprint = fp

    label = payload.machine_label
    try:
        dev = register_device(
            base_url=base,
            access_token=access,
            fingerprint=fp,
            label=label,
        )
        did = str(dev.get("id") or "").strip()
        if did:
            result.device_id = did
            result.registered = True
        else:
            result.device_id = device_id
            result.registered = False
    except CloudApiError:
        result.device_id = device_id
        result.registered = False

    if not result.registered and result.pro_keys:
        result.bootstrap = {}
        result.bootstrap_status = "skipped_no_device"
        result.ok = True
        return result

    did = result.device_id.strip()
    try:
        if result.pro_keys:
            if did:
                result.bootstrap = get_bootstrap(
                    base_url=base,
                    access_token=access,
                    device_id=did,
                )
            else:
                result.bootstrap = {}
                result.bootstrap_status = "skipped_no_device"
        else:
            result.bootstrap = get_bootstrap(base_url=base, access_token=access)
        if isinstance(result.bootstrap, dict):
            result.bootstrap_status = "ok"
        else:
            result.bootstrap = {}
            result.bootstrap_status = "ok"
    except CloudApiError:
        result.bootstrap = {}
        result.bootstrap_status = (
            "failed_pro" if result.pro_keys else "failed_free"
        )

    result.ok = True
    return result


def sync_cloud_credits(payload: CloudCreditsSyncPayload) -> CloudCreditsSyncResult:
    """Fetch entitlements only (Credits page refresh)."""
    result = CloudCreditsSyncResult()
    base = payload.base_url.strip()
    if not base:
        result.error_message = "Cloud API base URL is not configured."
        return result

    try:
        access, refresh, expires_at, _ = _refresh_access_for_credits(payload)
    except CloudApiError as exc:
        result.session_expired = True
        result.error_message = user_facing_cloud_message(exc)
        return result
    except Exception as exc:
        result.error_message = user_facing_cloud_message(exc)
        return result

    result.access_token = access
    result.refresh_token = refresh or payload.refresh_token.strip()
    result.access_token_expires_at = expires_at

    try:
        ent = get_entitlements(base_url=base, access_token=access)
    except CloudApiError as exc:
        result.error_message = user_facing_cloud_message(exc)
        return result
    except Exception as exc:
        result.error_message = user_facing_cloud_message(exc)
        return result

    result.ent = ent if isinstance(ent, dict) else {}
    result.ok = True
    return result

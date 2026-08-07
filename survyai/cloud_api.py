from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests


class CloudApiError(RuntimeError):
    pass


def _cloud_get(url: str, **kwargs: Any) -> requests.Response:
    try:
        return requests.get(url, **kwargs)
    except requests.RequestException as exc:
        raise CloudApiError(
            f"Network error ({type(exc).__name__}): unable to reach the cloud server."
        ) from exc


def _cloud_post(url: str, **kwargs: Any) -> requests.Response:
    try:
        return requests.post(url, **kwargs)
    except requests.RequestException as exc:
        raise CloudApiError(
            f"Network error ({type(exc).__name__}): unable to reach the cloud server."
        ) from exc


def _cloud_delete(url: str, **kwargs: Any) -> requests.Response:
    try:
        return requests.delete(url, **kwargs)
    except requests.RequestException as exc:
        raise CloudApiError(
            f"Network error ({type(exc).__name__}): unable to reach the cloud server."
        ) from exc


@dataclass(frozen=True)
class CloudTokenPair:
    access_token: str
    refresh_token: str
    expires_in: int


def _norm_base(url: str) -> str:
    return (url or "").strip().rstrip("/")


def _api_error_detail(body: Any, *, status_code: int) -> str:
    """Extract a human-readable message from FastAPI JSON error bodies."""
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        if isinstance(detail, list):
            parts: list[str] = []
            for item in detail:
                if isinstance(item, dict):
                    msg = item.get("msg")
                    parts.append(str(msg if msg is not None else item))
                else:
                    parts.append(str(item))
            if parts:
                return "; ".join(parts)
        if detail is not None:
            return str(detail)
        return str(body)
    if body is None or body == "":
        return f"HTTP {status_code}"
    return str(body)


def _parse_json(resp: requests.Response, *, what: str) -> Any:
    """
    Parse JSON from an API response; raise CloudApiError with context on empty or non-JSON bodies.

    Raw ``json.loads`` failures (e.g. empty body, HTML error page) become clear messages instead of
    ``Expecting value: line 1 column 1``.
    """
    try:
        return resp.json()
    except ValueError as exc:
        text = (resp.text or "").strip()
        if not text:
            raise CloudApiError(
                f"{what}: empty response (HTTP {resp.status_code}). "
                "Is the SurvyAI cloud server running? Check the base URL "
                "(e.g. https://survyai-api.onrender.com or http://127.0.0.1:8088 with no trailing path)."
            ) from exc
        preview = text[:500].replace("\n", " ")
        # Render/nginx often returns plain "Internal Server Error" for worker crashes.
        if resp.status_code >= 500 and what.startswith("POST /v1/llm/chat"):
            raise CloudApiError(
                f"{what}: hosted LLM proxy returned HTTP {resp.status_code} "
                f"(non-JSON: {preview}). "
                "Usually the SurvyAI cloud LLM worker failed upstream (OpenAI key/quota, "
                "timeout, or oversized tool payload). Try: Sign in again (Account), "
                "confirm Pro + credits remain, retry the prompt; if it persists, "
                "switch Primary LLM to Ollama for offline work or contact support."
            ) from exc
        raise CloudApiError(
            f"{what}: expected JSON but got HTTP {resp.status_code}: {preview}"
        ) from exc


def register(
    *,
    base_url: str,
    email: str,
    password: str,
    display_name: Optional[str] = None,
    timeout_s: int = 20,
) -> dict[str, Any]:
    """POST /v1/auth/register — create account; caller should call ``login`` next for tokens."""
    base = _norm_base(base_url)
    if not base:
        raise CloudApiError("Missing cloud API base URL")
    payload: dict[str, Any] = {"email": email.strip(), "password": password}
    if display_name and str(display_name).strip():
        payload["display_name"] = str(display_name).strip()
    resp = _cloud_post(
        f"{base}/v1/auth/register",
        json=payload,
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="Register")
    if not resp.ok:
        raise CloudApiError(_api_error_detail(body, status_code=resp.status_code))
    return body if isinstance(body, dict) else {}


def login(*, base_url: str, email: str, password: str, timeout_s: int = 20) -> CloudTokenPair:
    base = _norm_base(base_url)
    if not base:
        raise CloudApiError("Missing cloud API base URL")
    resp = _cloud_post(
        f"{base}/v1/auth/login",
        json={"email": email.strip(), "password": password},
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="Login")
    if not resp.ok:
        raise CloudApiError(_api_error_detail(body, status_code=resp.status_code))
    return CloudTokenPair(
        access_token=str(body.get("access_token") or ""),
        refresh_token=str(body.get("refresh_token") or ""),
        expires_in=int(body.get("expires_in") or 0),
    )


def refresh_tokens(*, base_url: str, refresh_token: str, timeout_s: int = 20) -> CloudTokenPair:
    base = _norm_base(base_url)
    if not base:
        raise CloudApiError("Missing cloud API base URL")
    resp = _cloud_post(
        f"{base}/v1/auth/refresh",
        json={"refresh_token": refresh_token},
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="Token refresh")
    if not resp.ok:
        raise CloudApiError(str(body.get("detail") or body))
    return CloudTokenPair(
        access_token=str(body.get("access_token") or ""),
        refresh_token=str(body.get("refresh_token") or ""),
        expires_in=int(body.get("expires_in") or 0),
    )


# Mirrors survyai_cloud.security password policy for desktop UX (no cloud import).
_PASSWORD_MIN_LENGTH = 10
_PASSWORD_MAX_LENGTH = 128
# Keep '-' at the end of the class so it is literal (not a range like /-=).
_PASSWORD_SPECIAL_CHARS = r"!@#$%^&*()_+/=[]{}|;:,.<>?-"
_PASSWORD_SPECIAL_RE = re.compile(r"[!@#$%^&*()_+/=\[\]{}|;:,.<>?-]")
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


def validate_password_strength(password: str, *, email: str | None = None) -> Optional[str]:
    """Return an error message if password fails policy, else None."""
    plain = password or ""
    if len(plain) < _PASSWORD_MIN_LENGTH:
        return f"Password must be at least {_PASSWORD_MIN_LENGTH} characters."
    if len(plain) > _PASSWORD_MAX_LENGTH:
        return f"Password must be at most {_PASSWORD_MAX_LENGTH} characters."
    if not re.search(r"[a-z]", plain):
        return "Password must include at least one lowercase letter."
    if not re.search(r"[A-Z]", plain):
        return "Password must include at least one uppercase letter."
    if not re.search(r"[0-9]", plain):
        return "Password must include at least one digit."
    if not _PASSWORD_SPECIAL_RE.search(plain):
        return (
            "Password must include at least one special character "
            f"({_PASSWORD_SPECIAL_CHARS})."
        )
    if plain.lower() in _COMMON_PASSWORDS or plain in _COMMON_PASSWORDS:
        return "That password is too common. Please choose a stronger password."
    if email:
        local = str(email).split("@", 1)[0].strip().lower()
        if local and len(local) >= 3 and local in plain.lower():
            return "Password must not contain your email username."
    return None

def password_policy_hint() -> str:
    return (
        f"At least {_PASSWORD_MIN_LENGTH} characters, with upper and lower case, "
        "a digit, and a special character."
    )


def logout(*, base_url: str, refresh_token: str, timeout_s: int = 20) -> None:
    """POST /v1/auth/logout — revoke the refresh token (best-effort)."""
    base = _norm_base(base_url)
    if not base or not (refresh_token or "").strip():
        return
    resp = _cloud_post(
        f"{base}/v1/auth/logout",
        json={"refresh_token": refresh_token.strip()},
        timeout=timeout_s,
    )
    if resp.status_code not in (200, 204) and resp.status_code >= 400:
        body = _parse_json(resp, what="Logout") if resp.content else {}
        raise CloudApiError(_api_error_detail(body, status_code=resp.status_code))


def forgot_password(*, base_url: str, email: str, timeout_s: int = 30) -> dict[str, Any]:
    base = _norm_base(base_url)
    if not base:
        raise CloudApiError("Missing cloud API base URL")
    resp = _cloud_post(
        f"{base}/v1/auth/forgot-password",
        json={"email": email.strip()},
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="Forgot password")
    if not resp.ok:
        raise CloudApiError(_api_error_detail(body, status_code=resp.status_code))
    return body if isinstance(body, dict) else {}


def reset_password(
    *,
    base_url: str,
    email: str,
    code: str,
    new_password: str,
    timeout_s: int = 30,
) -> dict[str, Any]:
    base = _norm_base(base_url)
    if not base:
        raise CloudApiError("Missing cloud API base URL")
    resp = _cloud_post(
        f"{base}/v1/auth/reset-password",
        json={
            "email": email.strip(),
            "code": code.strip(),
            "new_password": new_password,
        },
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="Reset password")
    if not resp.ok:
        raise CloudApiError(_api_error_detail(body, status_code=resp.status_code))
    return body if isinstance(body, dict) else {}


def change_password(
    *,
    base_url: str,
    access_token: str,
    current_password: str,
    new_password: str,
    timeout_s: int = 30,
) -> CloudTokenPair:
    base = _norm_base(base_url)
    if not base:
        raise CloudApiError("Missing cloud API base URL")
    resp = _cloud_post(
        f"{base}/v1/auth/change-password",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "current_password": current_password,
            "new_password": new_password,
        },
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="Change password")
    if not resp.ok:
        raise CloudApiError(_api_error_detail(body, status_code=resp.status_code))
    return CloudTokenPair(
        access_token=str(body.get("access_token") or ""),
        refresh_token=str(body.get("refresh_token") or ""),
        expires_in=int(body.get("expires_in") or 0),
    )


def access_token_expires_at_iso(*, expires_in_seconds: int) -> str:
    if expires_in_seconds <= 0:
        return ""
    return (datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)).isoformat()


def get_me(*, base_url: str, access_token: str, timeout_s: int = 20) -> dict[str, Any]:
    base = _norm_base(base_url)
    resp = _cloud_get(
        f"{base}/v1/me",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="GET /v1/me")
    if not resp.ok:
        raise CloudApiError(str(body.get("detail") or body))
    return body if isinstance(body, dict) else {}


def get_bootstrap(
    *,
    base_url: str,
    access_token: str,
    device_id: Optional[str] = None,
    timeout_s: int = 20,
) -> dict[str, Any]:
    base = _norm_base(base_url)
    headers: dict[str, str] = {"Authorization": f"Bearer {access_token}"}
    did = (device_id or "").strip()
    if did:
        headers["X-SurvyAI-Device-Id"] = did
    resp = _cloud_get(
        f"{base}/v1/bootstrap",
        headers=headers,
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="GET /v1/bootstrap")
    if resp.status_code == 401:
        raise CloudApiError("Unauthorized (access token expired or invalid)")
    if not resp.ok:
        raise CloudApiError(str(body.get("detail") or body))
    return body if isinstance(body, dict) else {}


def register_device(
    *,
    base_url: str,
    access_token: str,
    fingerprint: str,
    label: Optional[str] = None,
    timeout_s: int = 20,
) -> dict[str, Any]:
    """POST /v1/devices — register or refresh this PC; returns ``id`` for X-SurvyAI-Device-Id."""
    base = _norm_base(base_url)
    if not base:
        raise CloudApiError("Missing cloud API base URL")
    payload: dict[str, Any] = {"fingerprint": fingerprint.strip()}
    if label and str(label).strip():
        payload["label"] = str(label).strip()[:200]
    resp = _cloud_post(
        f"{base}/v1/devices",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="POST /v1/devices")
    if resp.status_code == 401:
        raise CloudApiError("Unauthorized (access token expired or invalid)")
    if not resp.ok:
        raise CloudApiError(str(body.get("detail") or body))
    return body if isinstance(body, dict) else {}


def list_devices(*, base_url: str, access_token: str, timeout_s: int = 20) -> list[dict[str, Any]]:
    """GET /v1/devices — registered PCs for the signed-in account."""
    base = _norm_base(base_url)
    resp = _cloud_get(
        f"{base}/v1/devices",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="GET /v1/devices")
    if resp.status_code == 401:
        raise CloudApiError("Unauthorized (access token expired or invalid)")
    if not resp.ok:
        raise CloudApiError(_api_error_detail(body, status_code=resp.status_code))
    if not isinstance(body, list):
        return []
    return [x for x in body if isinstance(x, dict)]


def delete_cloud_device(
    *,
    base_url: str,
    access_token: str,
    device_id: str,
    timeout_s: int = 20,
) -> None:
    """DELETE /v1/devices/{id} — free a slot so another PC can be registered."""
    base = _norm_base(base_url)
    did = (device_id or "").strip()
    if not did:
        raise CloudApiError("Missing device id")
    resp = _cloud_delete(
        f"{base}/v1/devices/{did}",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=timeout_s,
    )
    if resp.status_code == 401:
        raise CloudApiError("Unauthorized (access token expired or invalid)")
    if resp.status_code == 404:
        raise CloudApiError("Device not found (it may have been removed already).")
    if resp.status_code not in (200, 204):
        try:
            body = resp.json()
        except ValueError:
            body = {}
        detail = body.get("detail") if isinstance(body, dict) else None
        raise CloudApiError(str(detail or body or f"HTTP {resp.status_code}"))


def cloud_health(*, base_url: str, timeout_s: int = 8) -> dict[str, Any]:
    """GET /health — reachability and Paystack configuration flags (no auth)."""
    base = _norm_base(base_url)
    if not base:
        raise CloudApiError("Missing cloud API base URL")
    resp = _cloud_get(f"{base}/health", timeout=timeout_s)
    body = _parse_json(resp, what="GET /health")
    if not resp.ok:
        raise CloudApiError(_api_error_detail(body, status_code=resp.status_code))
    return body if isinstance(body, dict) else {}


def get_entitlements(*, base_url: str, access_token: str, timeout_s: int = 20) -> dict[str, Any]:
    base = _norm_base(base_url)
    resp = _cloud_get(
        f"{base}/v1/entitlements",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="GET /v1/entitlements")
    if resp.status_code == 401:
        raise CloudApiError("Unauthorized (access token expired or invalid)")
    if not resp.ok:
        raise CloudApiError(_api_error_detail(body, status_code=resp.status_code))
    return body if isinstance(body, dict) else {}


def report_usage_batch(
    *,
    base_url: str,
    access_token: str,
    events: list[dict[str, Any]],
    device_id: Optional[str] = None,
    timeout_s: int = 20,
) -> dict[str, Any]:
    base = _norm_base(base_url)
    headers: dict[str, str] = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    did = (device_id or "").strip()
    if did:
        headers["X-SurvyAI-Device-Id"] = did
    resp = _cloud_post(
        f"{base}/v1/usage/events",
        headers=headers,
        json={"events": events},
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="POST /v1/usage/events")
    if resp.status_code == 401:
        raise CloudApiError("Unauthorized (access token expired or invalid)")
    if not resp.ok:
        raise CloudApiError(_api_error_detail(body, status_code=resp.status_code))
    return body if isinstance(body, dict) else {}


def get_billing_plans(*, base_url: str, access_token: str, timeout_s: int = 20) -> dict[str, Any]:
    """GET /v1/billing/plans — Paystack plan codes configured on the server."""
    base = _norm_base(base_url)
    resp = _cloud_get(
        f"{base}/v1/billing/plans",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="GET /v1/billing/plans")
    if resp.status_code == 401:
        raise CloudApiError("Unauthorized (access token expired or invalid)")
    if not resp.ok:
        raise CloudApiError(_api_error_detail(body, status_code=resp.status_code))
    return body if isinstance(body, dict) else {}


def paystack_initialize(
    *,
    base_url: str,
    access_token: str,
    plan_code: Optional[str] = None,
    timeout_s: int = 30,
) -> dict[str, Any]:
    """POST /v1/billing/initialize — returns authorization_url and reference."""
    base = _norm_base(base_url)
    payload: dict[str, Any] = {}
    if plan_code and plan_code.strip():
        payload["plan_code"] = plan_code.strip()
    resp = _cloud_post(
        f"{base}/v1/billing/initialize",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="Paystack initialize")
    if not resp.ok:
        raise CloudApiError(_api_error_detail(body, status_code=resp.status_code))
    return body if isinstance(body, dict) else {}


def paystack_verify(
    *,
    base_url: str,
    access_token: str,
    reference: str,
    timeout_s: int = 30,
) -> dict[str, Any]:
    """POST /v1/billing/verify — confirm transaction after checkout (webhook fallback)."""
    base = _norm_base(base_url)
    resp = _cloud_post(
        f"{base}/v1/billing/verify",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json={"reference": reference.strip()},
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="Paystack verify")
    if not resp.ok:
        raise CloudApiError(_api_error_detail(body, status_code=resp.status_code))
    return body if isinstance(body, dict) else {}


def paystack_subscription_manage_url(
    *,
    base_url: str,
    access_token: str,
    timeout_s: int = 20,
) -> dict[str, Any]:
    """GET /v1/billing/subscription/manage-link — hosted Paystack subscription portal URL."""
    base = _norm_base(base_url)
    resp = _cloud_get(
        f"{base}/v1/billing/subscription/manage-link",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="GET subscription manage-link")
    if not resp.ok:
        raise CloudApiError(_api_error_detail(body, status_code=resp.status_code))
    return body if isinstance(body, dict) else {}


def get_update_manifest(
    *,
    base_url: str,
    current_version: str,
    channel: str = "stable",
    platform: str = "windows-x64",
    timeout_s: int = 20,
) -> dict[str, Any]:
    base = _norm_base(base_url)
    resp = _cloud_get(
        f"{base}/v1/updates/manifest",
        params={
            "current": current_version.strip(),
            "channel": channel.strip() or "stable",
            "platform": platform.strip() or "windows-x64",
        },
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="GET /v1/updates/manifest")
    if not resp.ok:
        raise CloudApiError(_api_error_detail(body, status_code=resp.status_code))
    return body if isinstance(body, dict) else {}


def proxy_llm_chat(
    *,
    base_url: str,
    access_token: str,
    payload: dict[str, Any],
    device_id: Optional[str] = None,
    proxy_path: str = "/v1/llm/chat",
    timeout_s: int = 120,
) -> dict[str, Any]:
    base = _norm_base(base_url)
    path = "/" + str(proxy_path or "/v1/llm/chat").lstrip("/")
    headers: dict[str, str] = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    did = (device_id or "").strip()
    if did:
        headers["X-SurvyAI-Device-Id"] = did
    resp = _cloud_post(
        f"{base}{path}",
        headers=headers,
        json=payload,
        timeout=timeout_s,
    )
    body = _parse_json(resp, what="POST /v1/llm/chat")
    if resp.status_code == 401:
        raise CloudApiError("Unauthorized (access token expired or invalid)")
    if resp.status_code == 402:
        raise CloudApiError(
            _api_error_detail(body, status_code=resp.status_code)
            or "Subscription API credit balance exhausted for this period"
        )
    if resp.status_code == 403:
        raise CloudApiError(
            _api_error_detail(body, status_code=resp.status_code)
            or "Hosted LLM access denied (Pro subscription / registered device required)"
        )
    if not resp.ok:
        detail = _api_error_detail(body, status_code=resp.status_code)
        if resp.status_code >= 500:
            raise CloudApiError(
                f"Hosted LLM proxy error (HTTP {resp.status_code}): {detail}. "
                "Try Sign in again, confirm Pro credits, then retry. "
                "Or switch Primary LLM to Ollama for local offline work."
            )
        raise CloudApiError(detail)
    return body if isinstance(body, dict) else {}


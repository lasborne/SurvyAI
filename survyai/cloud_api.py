from __future__ import annotations

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
                "(e.g. http://127.0.0.1:8088 with no trailing path)."
            ) from exc
        preview = text[:500].replace("\n", " ")
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
    if not resp.ok:
        raise CloudApiError(_api_error_detail(body, status_code=resp.status_code))
    return body if isinstance(body, dict) else {}


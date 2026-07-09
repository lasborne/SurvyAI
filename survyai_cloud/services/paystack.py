from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import requests

from survyai_cloud.config import CloudSettings, get_cloud_settings


PAYSTACK_BASE_URL = "https://api.paystack.co"


class PaystackError(RuntimeError):
    pass


@dataclass(frozen=True)
class PaystackInitializeResult:
    authorization_url: str
    access_code: Optional[str]
    reference: str


def _headers(settings: CloudSettings) -> dict[str, str]:
    if not settings.paystack_secret_key.strip():
        raise PaystackError("Paystack is not configured (missing PAYSTACK_SECRET_KEY)")
    return {
        "Authorization": f"Bearer {settings.paystack_secret_key.strip()}",
        "Content-Type": "application/json",
    }


def _reject_amount_confused_with_plan_code(plan_code: str) -> None:
    """Paystack plan codes look like ``PLN_…``; amounts in Naira are a common .env mistake."""
    pc = (plan_code or "").strip()
    if pc.isdigit() and len(pc) >= 4:
        raise PaystackError(
            "plan_code looks like a Naira amount, not a Paystack plan code. In Paystack "
            "Dashboard → Plans, copy each plan's Code (e.g. PLN_xxxxxxxxxx) into "
            "PAYSTACK_PLAN_CODE_PRO_DAILY, PAYSTACK_PLAN_CODE_PRO_WEEKLY, "
            "PAYSTACK_PLAN_CODE_PRO_MONTHLY, and PAYSTACK_PLAN_CODE_PRO_ANNUAL. "
            "Put the prices only in PAYSTACK_PRO_*_AMOUNT_NGN (or rely on defaults)."
        )


def _amount_kobo_for_plan(settings: CloudSettings, plan_code: str) -> int:
    """
    Paystack /transaction/initialize requires ``amount`` in the currency subunit.
    For NGN that is kobo (1 Naira = 100 kobo). The plan code is used only as
    SurvyAI's catalog selector; it must not be sent as Paystack's ``plan`` field
    because that creates a recurring subscription authorization.
    """
    pc = (plan_code or "").strip()
    daily = (settings.paystack_plan_code_pro_daily or "").strip()
    weekly = (settings.paystack_plan_code_pro_weekly or "").strip()
    monthly = (settings.paystack_plan_code_pro_monthly or "").strip()
    annual = (settings.paystack_plan_code_pro_annual or "").strip()
    if daily and pc == daily:
        ngn = int(settings.paystack_pro_daily_amount_ngn)
    elif weekly and pc == weekly:
        ngn = int(settings.paystack_pro_weekly_amount_ngn)
    elif monthly and pc == monthly:
        ngn = int(settings.paystack_pro_monthly_amount_ngn)
    elif annual and pc == annual:
        ngn = int(settings.paystack_pro_annual_amount_ngn)
    else:
        raise PaystackError(
            f"Unknown plan_code {plan_code!r}; it must match "
            "PAYSTACK_PLAN_CODE_PRO_DAILY, PAYSTACK_PLAN_CODE_PRO_WEEKLY, "
            "PAYSTACK_PLAN_CODE_PRO_MONTHLY, or PAYSTACK_PLAN_CODE_PRO_ANNUAL on the server."
        )
    if ngn <= 0:
        raise PaystackError(
            "Configured plan amount in NGN is invalid (set PAYSTACK_PRO_DAILY_AMOUNT_NGN / "
            "PAYSTACK_PRO_WEEKLY_AMOUNT_NGN / PAYSTACK_PRO_MONTHLY_AMOUNT_NGN / "
            "PAYSTACK_PRO_ANNUAL_AMOUNT_NGN to match your Paystack plan)."
        )
    return ngn * 100


def initialize_transaction(
    *,
    email: str,
    plan_code: str,
    callback_url: str,
    metadata: dict[str, Any],
    settings: Optional[CloudSettings] = None,
) -> PaystackInitializeResult:
    """Initialize a one-time Paystack transaction for manual Pro access purchase."""
    settings = settings or get_cloud_settings()
    _reject_amount_confused_with_plan_code(plan_code)
    url = f"{PAYSTACK_BASE_URL}/transaction/initialize"
    amount_kobo = _amount_kobo_for_plan(settings, plan_code)
    payload = {
        "email": email,
        "amount": amount_kobo,
        "currency": "NGN",
        "callback_url": callback_url,
        "metadata": {
            **metadata,
            "survyai_selected_plan_code": plan_code,
            "survyai_payment_mode": "manual_one_time",
        },
    }
    resp = requests.post(url, headers=_headers(settings), data=json.dumps(payload), timeout=25)
    try:
        body = resp.json()
    except Exception as exc:
        raise PaystackError(f"Paystack initialize failed (non-JSON): {resp.status_code}") from exc
    if not resp.ok or not body.get("status"):
        msg = body.get("message") or f"HTTP {resp.status_code}"
        raise PaystackError(f"Paystack initialize failed: {msg}")
    data = body.get("data") or {}
    return PaystackInitializeResult(
        authorization_url=str(data.get("authorization_url")),
        access_code=data.get("access_code"),
        reference=str(data.get("reference")),
    )


def verify_transaction(
    *,
    reference: str,
    settings: Optional[CloudSettings] = None,
) -> dict[str, Any]:
    settings = settings or get_cloud_settings()
    url = f"{PAYSTACK_BASE_URL}/transaction/verify/{reference}"
    resp = requests.get(url, headers=_headers(settings), timeout=25)
    body = resp.json()
    if not resp.ok or not body.get("status"):
        msg = body.get("message") or f"HTTP {resp.status_code}"
        raise PaystackError(f"Paystack verify failed: {msg}")
    return body.get("data") or {}


def disable_subscription(
    *,
    subscription_code: str,
    email_token: str,
    settings: Optional[CloudSettings] = None,
) -> bool:
    """Disable a Paystack subscription if one was accidentally created.

    This is a guardrail for older checkouts that used Paystack's recurring
    ``plan`` field. New SurvyAI checkouts are one-time transactions and should
    not create subscription codes at all.
    """
    settings = settings or get_cloud_settings()
    code = (subscription_code or "").strip()
    token = (email_token or "").strip()
    if not code or not token:
        return False
    url = f"{PAYSTACK_BASE_URL}/subscription/disable"
    resp = requests.post(
        url,
        headers=_headers(settings),
        data=json.dumps({"code": code, "token": token}),
        timeout=25,
    )
    try:
        body = resp.json()
    except Exception as exc:
        raise PaystackError(f"Paystack subscription disable failed (non-JSON): {resp.status_code}") from exc
    if not resp.ok or not body.get("status"):
        msg = body.get("message") or f"HTTP {resp.status_code}"
        raise PaystackError(f"Paystack subscription disable failed: {msg}")
    return True


def fetch_subscription_manage_link(
    *,
    subscription_code: str,
    settings: Optional[CloudSettings] = None,
) -> str:
    """
    Hosted subscription management URL (update card / cancel) from Paystack.
    See: https://paystack.com/docs/api/subscription/
    """
    settings = settings or get_cloud_settings()
    code = (subscription_code or "").strip()
    if not code:
        raise PaystackError("subscription_code required")
    url = f"{PAYSTACK_BASE_URL}/subscription/{code}/manage/link"
    resp = requests.get(url, headers=_headers(settings), timeout=25)
    try:
        body = resp.json()
    except Exception as exc:
        raise PaystackError(f"Paystack manage link failed (non-JSON): {resp.status_code}") from exc
    if not resp.ok or not body.get("status"):
        msg = body.get("message") or f"HTTP {resp.status_code}"
        raise PaystackError(f"Paystack manage link failed: {msg}")
    data = body.get("data") or {}
    link = data.get("link") or data.get("url")
    if not link:
        raise PaystackError("Paystack manage link response missing link")
    return str(link)


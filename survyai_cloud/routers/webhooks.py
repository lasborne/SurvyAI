from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.config import CloudSettings, get_cloud_settings
from survyai_cloud.db import get_db
from survyai_cloud.models import PaymentEventLog, SubscriptionStatus, User
from survyai_cloud.services.entitlements import (
    apply_free_defaults,
    apply_pro_defaults,
    credit_budget_and_interval_from_paystack_payload,
    manual_payment_period_anchor,
    subscription_period_end_from_anchor,
)
from survyai_cloud.services.paystack import PaystackError, disable_subscription

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def _verify_paystack_signature(payload: bytes, signature: str, secret: str) -> bool:
    mac = hmac.new(secret.encode("utf-8"), msg=payload, digestmod=hashlib.sha512).hexdigest()
    return hmac.compare_digest(mac, signature)


def _extract_user_id_from_metadata(event: dict[str, Any]) -> str | None:
    data = (event.get("data") or {}) if isinstance(event, dict) else {}
    if not isinstance(data, dict):
        return None
    meta = data.get("metadata") or {}
    if isinstance(meta, dict):
        uid = meta.get("survyai_user_id")
        if uid:
            return str(uid)
    return None


def _ts_parse(dt: str | None) -> datetime | None:
    if not dt:
        return None
    try:
        if dt.endswith("Z"):
            dt = dt[:-1] + "+00:00"
        return datetime.fromisoformat(dt)
    except Exception:
        return None


def _customer_dict(data: dict[str, Any]) -> dict[str, Any] | None:
    c = data.get("customer")
    if isinstance(c, dict):
        return c
    return None


def _paystack_invoice_indicates_paid(data: dict[str, Any]) -> bool:
    """Best-effort detection across Paystack invoice / charge payload shapes."""
    if data.get("paid") is True:
        return True
    st = str(data.get("status") or "").lower()
    if st in ("success", "paid", "paid_out", "complete"):
        return True
    if data.get("paid_at"):
        return True
    tx = data.get("transaction")
    if isinstance(tx, dict) and str(tx.get("status") or "").lower() == "success":
        return True
    try:
        ap = int(data.get("amount_paid") or 0)
        at = int(data.get("amount") or data.get("total") or 0)
        if at > 0 and ap >= at:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _first_ts(*candidates: Any) -> datetime | None:
    for c in candidates:
        if isinstance(c, str) and c.strip():
            t = _ts_parse(c)
            if t is not None:
                return t
    return None


def _renewal_hint_from_invoice_or_subscription(data: dict[str, Any]) -> datetime | None:
    """Prefer explicit period / next charge timestamps when Paystack includes them."""
    direct = _first_ts(
        data.get("period_end"),
        data.get("due_date"),
        data.get("next_payment_date"),
    )
    if direct is not None:
        return direct
    sub = data.get("subscription")
    if isinstance(sub, dict):
        return _first_ts(
            sub.get("next_payment_date"),
            sub.get("end"),
        )
    return None


async def _resolve_user_for_paystack_event(
    db: AsyncSession,
    event: dict[str, Any],
) -> User | None:
    """Map webhook payload to a local user (metadata, email, customer_code, subscription_code)."""
    uid = _extract_user_id_from_metadata(event)
    if uid:
        try:
            user_id = UUID(uid)
        except Exception:
            user_id = None
        if user_id:
            res = await db.execute(select(User).where(User.id == user_id))
            user = res.scalar_one_or_none()
            if user is not None:
                return user

    data = event.get("data") or {}
    if not isinstance(data, dict):
        return None

    sub_code = data.get("subscription_code")
    if isinstance(sub_code, str) and sub_code.strip():
        res = await db.execute(select(User).where(User.paystack_subscription_code == sub_code.strip()))
        user = res.scalar_one_or_none()
        if user is not None:
            return user

    cust = _customer_dict(data)
    if cust:
        cc = cust.get("customer_code")
        if cc:
            res = await db.execute(select(User).where(User.paystack_customer_code == str(cc)))
            user = res.scalar_one_or_none()
            if user is not None:
                return user
        email = str(cust.get("email") or "").lower().strip()
        if email:
            res = await db.execute(select(User).where(User.email == email))
            user = res.scalar_one_or_none()
            if user is not None:
                return user

    # Invoice payloads may nest customer differently
    inv_customer = data.get("customer")
    if isinstance(inv_customer, str) and "@" in inv_customer:
        email = inv_customer.lower().strip()
        res = await db.execute(select(User).where(User.email == email))
        return res.scalar_one_or_none()

    return None


def _apply_charge_success_entitlements(user: User, data: dict[str, Any], settings: CloudSettings) -> None:
    budget_usd, bill_interval = credit_budget_and_interval_from_paystack_payload(data, settings)
    paid_at = _ts_parse(data.get("paid_at") if isinstance(data, dict) else None)
    anchor = manual_payment_period_anchor(user, paid_at)
    apply_pro_defaults(
        user,
        settings,
        credit_budget_usd=budget_usd,
        credits_billing_interval=bill_interval,
        paid_at=paid_at,
    )
    user.subscription_status = SubscriptionStatus.active

    period_hint = _renewal_hint_from_invoice_or_subscription(data)
    user.subscription_current_period_end = period_hint or subscription_period_end_from_anchor(
        anchor, bill_interval
    )

    cust = _customer_dict(data)
    if cust:
        if cust.get("customer_code"):
            user.paystack_customer_code = str(cust.get("customer_code"))
    sub_code = data.get("subscription_code")
    email_token = data.get("email_token")
    if sub_code and email_token:
        try:
            disable_subscription(subscription_code=str(sub_code), email_token=str(email_token), settings=settings)
            user.paystack_subscription_code = None
            user.paystack_email_token = None
            user.subscription_status = SubscriptionStatus.non_renewing
        except PaystackError:
            user.paystack_subscription_code = str(sub_code)
            user.paystack_email_token = str(email_token)
    elif sub_code:
        user.paystack_subscription_code = str(sub_code)
    elif email_token:
        user.paystack_email_token = str(email_token)
    reference = data.get("reference")
    if reference:
        user.last_payment_reference = str(reference)


@router.post("/paystack", status_code=status.HTTP_200_OK)
async def paystack_webhook(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, str]:
    settings = get_cloud_settings()
    if not settings.paystack_webhook_enabled:
        return {"status": "disabled"}
    if not settings.paystack_secret_key.strip():
        raise HTTPException(status_code=501, detail="Paystack webhook not configured")

    payload = await request.body()
    sig = request.headers.get("x-paystack-signature") or ""
    if not sig:
        raise HTTPException(status_code=400, detail="Missing x-paystack-signature header")
    if not _verify_paystack_signature(payload, sig, settings.paystack_secret_key.strip()):
        raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        event = json.loads(payload.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    event_type = str(event.get("event") or "")
    data = event.get("data") or {}
    ref: Optional[str] = None
    if isinstance(data, dict):
        ref = data.get("reference") or data.get("id") or data.get("subscription_code")
    event_id = str(event.get("id") or ref or f"{event_type}:{hashlib.sha256(payload).hexdigest()[:16]}")

    existing = await db.execute(select(PaymentEventLog).where(PaymentEventLog.event_id == event_id))
    if existing.scalar_one_or_none() is not None:
        return {"status": "duplicate"}

    user = await _resolve_user_for_paystack_event(db, event)

    if user is not None and isinstance(data, dict):
        if event_type == "charge.success":
            _apply_charge_success_entitlements(user, data, settings)
            db.add(user)

        elif event_type == "subscription.disable":
            now = datetime.now(timezone.utc)
            current_end = user.subscription_current_period_end
            if isinstance(current_end, datetime):
                if current_end.tzinfo is None:
                    current_end = current_end.replace(tzinfo=timezone.utc)
                else:
                    current_end = current_end.astimezone(timezone.utc)
            if user.subscription_status in (SubscriptionStatus.active, SubscriptionStatus.non_renewing) and isinstance(current_end, datetime) and current_end > now:
                user.subscription_status = SubscriptionStatus.non_renewing
                user.paystack_subscription_code = None
                user.paystack_email_token = None
            else:
                apply_free_defaults(user, settings)
                user.subscription_status = SubscriptionStatus.canceled
            db.add(user)

        elif event_type == "subscription.create":
            # Paystack: subscription exists (often after authorization). New subs → trialing; do not downgrade active.
            if isinstance(data, dict):
                sub_code = data.get("subscription_code")
                if sub_code:
                    user.paystack_subscription_code = str(sub_code)
                email_token = data.get("email_token")
                if email_token:
                    user.paystack_email_token = str(email_token)
                if sub_code and email_token:
                    try:
                        disable_subscription(subscription_code=str(sub_code), email_token=str(email_token), settings=settings)
                        user.paystack_subscription_code = None
                        user.paystack_email_token = None
                    except PaystackError:
                        pass
                cust = _customer_dict(data)
                if cust and cust.get("customer_code"):
                    user.paystack_customer_code = str(cust.get("customer_code"))
            if user.subscription_status != SubscriptionStatus.active:
                apply_pro_defaults(user, settings)
                user.subscription_status = SubscriptionStatus.trialing
                hint = _renewal_hint_from_invoice_or_subscription(data)
                if hint is not None:
                    user.subscription_current_period_end = hint
            db.add(user)

        elif event_type == "invoice.create":
            # SurvyAI uses manual one-time renewals. Invoice events belong to
            # Paystack subscriptions and must not extend access or trigger renewals.
            db.add(user)

        elif event_type == "subscription.not_renew":
            user.subscription_status = SubscriptionStatus.non_renewing
            db.add(user)

        elif event_type == "invoice.payment_failed":
            user.subscription_status = SubscriptionStatus.past_due
            db.add(user)

        elif event_type == "invoice.update":
            # Successful recurring charge is usually invoice.update (Paystack has no separate invoice.paid event name).
            if _paystack_invoice_indicates_paid(data):
                paid_at = _ts_parse(data.get("paid_at") if isinstance(data.get("paid_at"), str) else None)
                now = datetime.now(timezone.utc)
                anchor = manual_payment_period_anchor(user, paid_at or now)
                budget_usd, bill_interval = credit_budget_and_interval_from_paystack_payload(data, settings)
                apply_pro_defaults(
                    user,
                    settings,
                    credit_budget_usd=budget_usd,
                    credits_billing_interval=bill_interval,
                    paid_at=paid_at or now,
                )
                user.subscription_status = SubscriptionStatus.active
                period_hint = _renewal_hint_from_invoice_or_subscription(data)
                user.subscription_current_period_end = period_hint or subscription_period_end_from_anchor(
                    anchor, bill_interval
                )
                cust = _customer_dict(data)
                if cust and cust.get("customer_code"):
                    user.paystack_customer_code = str(cust.get("customer_code"))
                sub_code = data.get("subscription_code")
                et = data.get("email_token")
                if sub_code and et:
                    try:
                        disable_subscription(subscription_code=str(sub_code), email_token=str(et), settings=settings)
                        user.paystack_subscription_code = None
                        user.paystack_email_token = None
                        user.subscription_status = SubscriptionStatus.non_renewing
                    except PaystackError:
                        user.paystack_subscription_code = str(sub_code)
                        user.paystack_email_token = str(et)
                elif sub_code:
                    user.paystack_subscription_code = str(sub_code)
                elif et:
                    user.paystack_email_token = str(et)
                db.add(user)

    db.add(PaymentEventLog(event_id=event_id, provider="paystack", type=event_type))
    return {"status": "ok"}

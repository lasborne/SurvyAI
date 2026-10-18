"""Support and ops overrides + read APIs (Phase 5/8)."""

from __future__ import annotations

import re
import secrets
import uuid
from datetime import datetime, time, timedelta, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import case, delete, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.config import get_cloud_settings
from survyai_cloud.db import get_db
from survyai_cloud.deps import require_admin
from survyai_cloud.models import (
    BetaSignup,
    Device,
    DiagnosticsBundle,
    SiteEvent,
    SubscriptionStatus,
    UsageEvent,
    User,
    UserFeedback,
)
from survyai_cloud.rate_limiting import rate_limit_ip_dependency
from survyai_cloud.schemas import (
    AdminDiagnosticsOut,
    AdminGrantProIn,
    AdminGrowthOut,
    AdminUsageEventOut,
    AdminUserBillingPatch,
    AdminUserSnapshot,
    BetaSignupOut,
    DeviceOut,
    GrowthNow,
    GrowthPage,
    GrowthPeriod,
    SiteCollectIn,
    SiteCollectOut,
    FeedbackOut,
)
from survyai_cloud.security import hash_password
from survyai_cloud.services.entitlements import (
    ACTIVE_LLM_STATUSES,
    UNCLAIMED_GRANT_PREFIX,
    apply_free_defaults,
    apply_pro_defaults,
    is_unclaimed_admin_grant,
)

router = APIRouter(prefix="/admin", tags=["admin"])
public_router = APIRouter(prefix="/analytics", tags=["analytics"])

_VISITOR_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_GROWTH_NOTE = (
    "Visits and download starts are counted from the day this tracking shipped. "
    "Each browser is counted once per page per day, and once per day when an installer link is opened. "
    "A click is not proof the file finished downloading. "
    "Registrations are people who chose a password in the app. "
    "Pro grants that are still waiting for that signup are listed separately."
)


def _snapshot(user: User, device_count: int) -> AdminUserSnapshot:
    return AdminUserSnapshot(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        plan_slug=user.plan_slug,
        subscription_status=user.subscription_status.value
        if hasattr(user.subscription_status, "value")
        else str(user.subscription_status),
        subscription_current_period_end=user.subscription_current_period_end,
        grace_period_ends_at=user.grace_period_ends_at,
        last_reactivation_at=user.last_reactivation_at,
        max_devices=user.max_devices,
        monthly_agent_runs_quota=user.monthly_agent_runs_quota,
        monthly_agent_runs_used=user.monthly_agent_runs_used,
        monthly_credits_usd=float(user.monthly_credits_usd or 0.0),
        monthly_credits_used_usd=float(user.monthly_credits_used_usd or 0.0),
        credits_billing_interval=user.credits_billing_interval or "monthly",
        usage_period_anchor=user.usage_period_anchor,
        paystack_customer_code=user.paystack_customer_code,
        paystack_subscription_code=user.paystack_subscription_code,
        last_payment_reference=user.last_payment_reference,
        admin_privilege_active=bool(getattr(user, "admin_privilege_active", False)),
        admin_privilege_note=getattr(user, "admin_privilege_note", None),
        awaiting_signup=is_unclaimed_admin_grant(user),
        device_count=device_count,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


async def _device_count(db: AsyncSession, user_id: uuid.UUID) -> int:
    res = await db.execute(select(func.count()).select_from(Device).where(Device.user_id == user_id))
    return int(res.scalar_one() or 0)


async def _get_user_or_404(db: AsyncSession, user_id: uuid.UUID) -> User:
    res = await db.execute(select(User).where(User.id == user_id))
    user = res.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def _looks_like_legacy_admin_grant(user: User) -> bool:
    """Pre-migration admin Pro grants: Pro + credits, no Paystack subscription code."""
    plan = str(user.plan_slug or "").strip().lower()
    if plan != "pro":
        return False
    if float(user.monthly_credits_usd or 0.0) <= 0:
        return False
    sub = str(user.paystack_subscription_code or "").strip()
    return not sub


def _mark_admin_privilege(user: User, *, note: Optional[str] = None) -> None:
    user.admin_privilege_active = True
    if note is not None:
        cleaned = str(note).strip()
        user.admin_privilege_note = cleaned[:500] if cleaned else None


def _clear_admin_privilege(user: User) -> None:
    user.admin_privilege_active = False
    user.admin_privilege_note = None


@router.get("/users", response_model=AdminUserSnapshot)
async def admin_lookup_user_by_email(
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    email: Annotated[str, Query(min_length=3, max_length=320)],
) -> AdminUserSnapshot:
    """Lookup a user support snapshot by email (case-insensitive)."""
    needle = email.strip().lower()
    res = await db.execute(select(User).where(func.lower(User.email) == needle))
    user = res.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _snapshot(user, await _device_count(db, user.id))


@router.get("/privileges", response_model=list[AdminUserSnapshot])
async def admin_list_privileges(
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> list[AdminUserSnapshot]:
    """
    List users with admin-managed privileges (grant / extend / credit overrides).

    Includes explicit ``admin_privilege_active`` rows and legacy Pro grants that
    have credits but no Paystack subscription code (pre-flag admin overrides).
    """
    res = await db.execute(select(User).order_by(User.updated_at.desc()).limit(2000))
    users = list(res.scalars().all())
    out: list[AdminUserSnapshot] = []
    backfilled = False
    for user in users:
        active = bool(getattr(user, "admin_privilege_active", False))
        legacy = (not active) and _looks_like_legacy_admin_grant(user)
        if not active and not legacy:
            continue
        if legacy:
            # Soft backfill so the dashboard stays consistent after first list.
            user.admin_privilege_active = True
            db.add(user)
            backfilled = True
        out.append(_snapshot(user, await _device_count(db, user.id)))
        if len(out) >= limit:
            break
    if backfilled:
        await db.flush()
    return out


@router.delete("/privileges/{user_id}", status_code=status.HTTP_200_OK)
async def admin_revoke_privilege(
    user_id: uuid.UUID,
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, object]:
    """Revoke admin-managed privilege and restore Free defaults."""
    user = await _get_user_or_404(db, user_id)
    apply_free_defaults(user)
    _clear_admin_privilege(user)
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return {
        "ok": True,
        "user_id": str(user.id),
        "email": user.email,
        "plan_slug": user.plan_slug,
        "admin_privilege_active": False,
    }


@router.get("/users/{user_id}", response_model=AdminUserSnapshot)
async def admin_get_user(
    user_id: uuid.UUID,
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AdminUserSnapshot:
    user = await _get_user_or_404(db, user_id)
    return _snapshot(user, await _device_count(db, user.id))


@router.get("/users/{user_id}/devices", response_model=list[DeviceOut])
async def admin_list_user_devices(
    user_id: uuid.UUID,
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[Device]:
    await _get_user_or_404(db, user_id)
    res = await db.execute(
        select(Device).where(Device.user_id == user_id).order_by(Device.created_at.desc())
    )
    return list(res.scalars().all())


@router.get("/users/{user_id}/usage", response_model=list[AdminUsageEventOut])
async def admin_list_user_usage(
    user_id: uuid.UUID,
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[UsageEvent]:
    await _get_user_or_404(db, user_id)
    res = await db.execute(
        select(UsageEvent)
        .where(UsageEvent.user_id == user_id)
        .order_by(UsageEvent.created_at.desc())
        .limit(limit)
    )
    return list(res.scalars().all())


@router.get("/users/{user_id}/diagnostics", response_model=list[AdminDiagnosticsOut])
async def admin_list_user_diagnostics(
    user_id: uuid.UUID,
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[DiagnosticsBundle]:
    await _get_user_or_404(db, user_id)
    res = await db.execute(
        select(DiagnosticsBundle)
        .where(DiagnosticsBundle.user_id == user_id)
        .order_by(DiagnosticsBundle.created_at.desc())
        .limit(limit)
    )
    return list(res.scalars().all())


@router.patch("/users/{user_id}/billing", status_code=status.HTTP_200_OK)
async def admin_patch_user_billing(
    user_id: uuid.UUID,
    body: AdminUserBillingPatch,
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, object]:
    user = await _get_user_or_404(db, user_id)

    if body.revoke_privilege or body.apply_free_defaults:
        apply_free_defaults(user)
        _clear_admin_privilege(user)

    if body.apply_pro_defaults and not body.revoke_privilege and not body.apply_free_defaults:
        apply_pro_defaults(user)
        _mark_admin_privilege(user)

    if body.plan_slug is not None:
        user.plan_slug = body.plan_slug.strip()

    if body.subscription_status is not None:
        raw = body.subscription_status.strip().lower()
        try:
            user.subscription_status = SubscriptionStatus(raw)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid subscription_status: " + raw)

    if body.clear_grace:
        user.grace_period_ends_at = None
    elif body.grace_period_ends_at is not None:
        user.grace_period_ends_at = body.grace_period_ends_at

    if body.touch_reactivated_now:
        user.last_reactivation_at = datetime.now(timezone.utc)
    elif body.last_reactivation_at is not None:
        user.last_reactivation_at = body.last_reactivation_at

    if body.reset_credits_used:
        user.monthly_credits_used_usd = 0.0
    elif body.monthly_credits_used_usd is not None:
        user.monthly_credits_used_usd = float(body.monthly_credits_used_usd)

    if body.monthly_credits_usd is not None:
        user.monthly_credits_usd = float(body.monthly_credits_usd)

    if body.max_devices is not None:
        user.max_devices = int(body.max_devices)

    if body.clear_period_end:
        user.subscription_current_period_end = None
    elif body.subscription_current_period_end is not None:
        user.subscription_current_period_end = body.subscription_current_period_end

    if body.clear_privilege_note:
        user.admin_privilege_note = None
    elif body.admin_privilege_note is not None:
        note = str(body.admin_privilege_note).strip()
        user.admin_privilege_note = note[:500] if note else None

    # Any non-revoke privilege adjustment marks the user as admin-managed.
    privilege_touch = any(
        (
            body.apply_pro_defaults,
            body.monthly_credits_usd is not None,
            body.monthly_credits_used_usd is not None,
            body.reset_credits_used,
            body.subscription_current_period_end is not None,
            body.plan_slug is not None
            and str(body.plan_slug).strip().lower() in {"pro", "trialing"},
            body.subscription_status is not None
            and str(body.subscription_status).strip().lower()
            in {"trialing", "active", "non_renewing"},
        )
    )
    if privilege_touch and not body.revoke_privilege and not body.apply_free_defaults:
        _mark_admin_privilege(user, note=body.admin_privilege_note)

    db.add(user)
    await db.flush()
    await db.refresh(user)
    return {
        "ok": True,
        "user_id": str(user.id),
        "plan_slug": user.plan_slug,
        "subscription_status": user.subscription_status.value,
        "grace_period_ends_at": user.grace_period_ends_at.isoformat()
        if user.grace_period_ends_at
        else None,
        "last_reactivation_at": user.last_reactivation_at.isoformat()
        if user.last_reactivation_at
        else None,
        "monthly_credits_usd": float(user.monthly_credits_usd or 0.0),
        "monthly_credits_used_usd": float(user.monthly_credits_used_usd or 0.0),
        "max_devices": user.max_devices,
        "subscription_current_period_end": user.subscription_current_period_end.isoformat()
        if user.subscription_current_period_end
        else None,
        "admin_privilege_active": bool(getattr(user, "admin_privilege_active", False)),
        "admin_privilege_note": getattr(user, "admin_privilege_note", None),
    }


_GRANT_PRO_DAYS = 30


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _pending_signup_note(extra: Optional[str]) -> str:
    text = (extra or "").strip()
    if text.startswith(UNCLAIMED_GRANT_PREFIX):
        return text[:500]
    detail = text or "Not signed up yet"
    return f"{UNCLAIMED_GRANT_PREFIX}: {detail}"[:500]


@router.post("/grant-pro", response_model=AdminUserSnapshot)
async def admin_grant_pro(
    body: AdminGrantProIn,
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AdminUserSnapshot:
    """
    Grant Pro for 30 days to any email.

    If the email has no SurvyAI account yet, create one and keep the grant
    until that person registers with the same email.
    """
    email = str(body.email).strip().lower()
    settings = get_cloud_settings()
    now = datetime.now(timezone.utc)
    res = await db.execute(select(User).where(func.lower(User.email) == email))
    user = res.scalar_one_or_none()
    created = user is None
    if user is None:
        signup = await db.execute(select(BetaSignup).where(func.lower(BetaSignup.email) == email))
        beta = signup.scalar_one_or_none()
        user = User(
            email=email,
            password_hash=hash_password(secrets.token_urlsafe(32)),
            display_name=(beta.name if beta and beta.name else None),
        )
        db.add(user)

    end = _aware(user.subscription_current_period_end)
    already_pro = (
        user.plan_slug == settings.pro_plan_slug
        and user.subscription_status in ACTIVE_LLM_STATUSES
        and (end is None or end > now)
    )
    if not already_pro:
        apply_pro_defaults(user, settings)
        if user.subscription_status not in (
            SubscriptionStatus.active,
            SubscriptionStatus.non_renewing,
        ):
            user.subscription_status = SubscriptionStatus.trialing
    horizon = now + timedelta(days=_GRANT_PRO_DAYS)
    end = _aware(user.subscription_current_period_end)
    if end is None or end < horizon:
        # Do not put an expiry on a Pro grant that was left open-ended.
        if not (already_pro and end is None):
            user.subscription_current_period_end = horizon

    if created or is_unclaimed_admin_grant(user):
        _mark_admin_privilege(user, note=_pending_signup_note(body.note))
    else:
        _mark_admin_privilege(user, note=body.note)

    db.add(user)
    await db.flush()
    await db.refresh(user)
    return _snapshot(user, await _device_count(db, user.id))


@router.get("/beta-signups", response_model=list[BetaSignupOut])
async def admin_list_beta_signups(
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> list[BetaSignup]:
    """First-user form rows, newest first. Used to grant Pro and read demand."""
    res = await db.execute(select(BetaSignup).order_by(BetaSignup.created_at.desc()).limit(limit))
    return list(res.scalars().all())


@router.delete("/users/{user_id}/devices/{device_id}", status_code=status.HTTP_204_NO_CONTENT)
async def admin_delete_user_device(
    user_id: uuid.UUID,
    device_id: uuid.UUID,
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    res = await db.execute(select(Device).where(Device.id == device_id, Device.user_id == user_id))
    if res.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Device not found")
    await db.execute(delete(Device).where(Device.id == device_id, Device.user_id == user_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _event_path(kind: str, path: str) -> Optional[str]:
    raw = (path or "").split("?", 1)[0].split("#", 1)[0].strip()
    if kind == "download":
        if raw in {"/get/latest", "/updates/latest", "/downloads/latest"}:
            return "/get/latest"
        if raw.startswith("/updates/") and raw.lower().endswith(".exe"):
            return "/get/latest"
        return None
    if raw in {"", "/"}:
        return "/"
    if raw.rstrip("/") == "/beta-user":
        return "/beta-user"
    return None


@public_router.post(
    "/collect",
    response_model=SiteCollectOut,
    dependencies=[
        Depends(
            rate_limit_ip_dependency(
                "site_collect",
                "rate_limit_site_events_per_window",
                window_seconds=3600,
            )
        )
    ],
)
async def collect_site_event(
    body: SiteCollectIn,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SiteCollectOut:
    """Record one anonymous visit or installer click. Invalid calls are ignored."""
    visitor_id = (body.visitor_id or "").strip().lower()
    kind = (body.kind or "").strip().lower()
    path = _event_path(kind, body.path)
    if kind not in {"pageview", "download"} or path is None or not _VISITOR_ID.match(visitor_id):
        return SiteCollectOut(stored=False)

    now = datetime.now(timezone.utc)
    row = SiteEvent(
        visitor_id=visitor_id,
        kind=kind,
        path=path,
        event_day=now.date(),
        created_at=now,
    )
    db.add(row)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        return SiteCollectOut(stored=False)
    return SiteCollectOut(stored=True)


def _per_100(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(100.0 * numerator / denominator, 1)


async def _event_counts(db: AsyncSession, start: Optional[datetime]) -> tuple[int, int, int]:
    stmt = select(
        func.count(func.distinct(case((SiteEvent.kind == "pageview", SiteEvent.visitor_id)))),
        func.count(case((SiteEvent.kind == "pageview", 1))),
        func.count(case((SiteEvent.kind == "download", 1))),
    ).select_from(SiteEvent)
    if start is not None:
        stmt = stmt.where(SiteEvent.created_at >= start)
    visitors, views, downloads = (await db.execute(stmt)).one()
    return int(visitors or 0), int(views or 0), int(downloads or 0)


async def _growth_period(db: AsyncSession, start: Optional[datetime]) -> GrowthPeriod:
    visitors, views, downloads = await _event_counts(db, start)
    beta = select(func.count()).select_from(BetaSignup)
    registered = select(func.count()).select_from(User).where(User.password_changed_at.isnot(None))
    activated = select(func.count(func.distinct(Device.user_id))).select_from(Device)
    if start is not None:
        beta = beta.where(BetaSignup.created_at >= start)
        registered = registered.where(User.password_changed_at >= start)
        activated = activated.where(Device.created_at >= start)
    return GrowthPeriod(
        unique_visitors=visitors,
        pageviews=views,
        download_starts=downloads,
        beta_applications=int((await db.execute(beta)).scalar_one() or 0),
        registered_accounts=int((await db.execute(registered)).scalar_one() or 0),
        activated_accounts=int((await db.execute(activated)).scalar_one() or 0),
    )


@router.get("/growth", response_model=AdminGrowthOut)
async def admin_growth(
    _: Annotated[None, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AdminGrowthOut:
    """Early-growth figures: site reach, installer clicks, applications, and accounts."""
    now = datetime.now(timezone.utc)
    start_today = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
    week = await _growth_period(db, now - timedelta(days=7))
    pages = await db.execute(
        select(
            SiteEvent.path,
            func.count(func.distinct(SiteEvent.visitor_id)),
            func.count(),
        )
        .where(SiteEvent.kind == "pageview", SiteEvent.created_at >= now - timedelta(days=7))
        .group_by(SiteEvent.path)
        .order_by(func.count(func.distinct(SiteEvent.visitor_id)).desc(), SiteEvent.path.asc())
    )
    pro = await db.execute(
        select(func.count())
        .select_from(User)
        .where(
            User.plan_slug == "pro",
            User.subscription_status.in_(tuple(ACTIVE_LLM_STATUSES)),
            or_(
                User.subscription_current_period_end.is_(None),
                User.subscription_current_period_end > now,
            ),
        )
    )
    awaiting = await db.execute(
        select(func.count())
        .select_from(User)
        .where(
            User.admin_privilege_note.like(f"{UNCLAIMED_GRANT_PREFIX}%"),
            User.password_changed_at.is_(None),
        )
    )
    current = await _growth_period(db, None)
    return AdminGrowthOut(
        generated_at=now,
        note=_GROWTH_NOTE,
        now=GrowthNow(
            registered_accounts=current.registered_accounts,
            awaiting_signup=int(awaiting.scalar_one() or 0),
            pro_accounts=int(pro.scalar_one() or 0),
            activated_accounts=current.activated_accounts,
            beta_applications=current.beta_applications,
        ),
        today=await _growth_period(db, start_today),
        days_7=week,
        days_30=await _growth_period(db, now - timedelta(days=30)),
        all_time=current,
        pages_7d=[
            GrowthPage(path=path, visitors=int(visitors or 0), views=int(views or 0))
            for path, visitors, views in pages.all()
        ],
        downloads_per_100_visitors_7d=_per_100(week.download_starts, week.unique_visitors),
        applications_per_100_visitors_7d=_per_100(week.beta_applications, week.unique_visitors),
        registrations_per_100_downloads_7d=_per_100(week.registered_accounts, week.download_starts),
    )


@router.get("/feedback", response_model=list[FeedbackOut])
async def list_feedback(
    db: Annotated[AsyncSession, Depends(get_db)],
    _: Annotated[None, Depends(require_admin)],
    limit: int = Query(default=200, ge=1, le=500),
) -> list[UserFeedback]:
    rows = await db.execute(
        select(UserFeedback).order_by(UserFeedback.created_at.desc()).limit(limit)
    )
    return list(rows.scalars().all())

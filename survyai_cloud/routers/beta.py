"""Public first-user signup from the marketing site."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.db import get_db
from survyai_cloud.models import BetaSignup
from survyai_cloud.rate_limiting import rate_limit_ip_dependency
from survyai_cloud.schemas import BetaJoinIn, BetaJoinOut

router = APIRouter(prefix="/beta", tags=["beta"])


def _opt(value: str) -> Optional[str]:
    text = (value or "").strip()
    if not text or text == "Not selected":
        return None
    return text


@router.post(
    "/join",
    response_model=BetaJoinOut,
    dependencies=[
        Depends(
            rate_limit_ip_dependency(
                "beta_join",
                "rate_limit_beta_join_per_window",
                window_seconds=3600,
            )
        )
    ],
)
async def join_beta(
    body: BetaJoinIn,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BetaJoinOut:
    """Store or refresh a first-user row. Honeypot hits look successful and are dropped."""
    if body.company_website.strip():
        return BetaJoinOut()

    email = str(body.email).strip().lower()
    res = await db.execute(select(BetaSignup).where(func.lower(BetaSignup.email) == email))
    row = res.scalar_one_or_none()
    now = datetime.now(timezone.utc)
    if row is None:
        row = BetaSignup(name=body.name, email=email, created_at=now, updated_at=now)
        db.add(row)
    else:
        row.name = body.name
        row.updated_at = now
    row.profession = _opt(body.profession)
    row.profession_other = _opt(body.profession_other)
    row.use_for = _opt(body.use_for)
    row.use_for_other = _opt(body.use_for_other)
    row.windows_specs = _opt(body.windows_specs)
    await db.flush()
    return BetaJoinOut()

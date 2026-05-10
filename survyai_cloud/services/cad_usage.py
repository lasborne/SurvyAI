from __future__ import annotations

"""CAD usage accounting helpers (dormant scaffolding)."""

from datetime import datetime
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.models import UsageEvent


CAD_SUCCESS_KIND = "cad_work_success"


async def count_successful_cad_jobs(
    *,
    db: AsyncSession,
    user_id,
    since_utc: datetime,
    until_utc: Optional[datetime] = None,
) -> int:
    q = select(func.count()).select_from(UsageEvent).where(
        UsageEvent.user_id == user_id,
        UsageEvent.kind == CAD_SUCCESS_KIND,
        UsageEvent.created_at >= since_utc,
    )
    if until_utc is not None:
        q = q.where(UsageEvent.created_at < until_utc)
    res = await db.execute(q)
    return int(res.scalar_one() or 0)


__all__ = ["CAD_SUCCESS_KIND", "count_successful_cad_jobs"]

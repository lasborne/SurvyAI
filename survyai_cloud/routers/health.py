from __future__ import annotations

from fastapi import APIRouter

from survyai_cloud.config import get_cloud_settings
from survyai_cloud.db import is_database_available

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, object]:
    settings = get_cloud_settings()
    monthly = settings.paystack_plan_code_pro_monthly.strip()
    annual = settings.paystack_plan_code_pro_annual.strip()
    db_ok, db_detail = is_database_available()
    resolved = settings.resolved_database_urls()
    return {
        "status": "ok" if db_ok else "degraded",
        "database_ok": db_ok,
        "database_detail": db_detail if not db_ok else "",
        "database_async_url_scheme": resolved.async_.split("://", 1)[0] if resolved.async_ else "",
        "database_sync_url_scheme": resolved.sync.split("://", 1)[0] if resolved.sync else "",
        "paystack_secret_configured": bool(settings.paystack_secret_key.strip()),
        "paystack_plans_configured": bool(monthly or annual),
    }

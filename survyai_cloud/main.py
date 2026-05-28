"""
FastAPI entrypoint for the SurvyAI commercial backend.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware

from survyai.database_urls import mask_database_url
from survyai_cloud.config import get_cloud_settings
from survyai_cloud.db import init_db, is_database_available
from survyai_cloud.routers import (
    admin,
    auth,
    billing,
    diagnostics,
    devices,
    entitlements,
    health,
    me,
    updates,
    usage,
    webhooks,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    db_ok, _db_detail = is_database_available()
    settings = get_cloud_settings()
    resolved = settings.resolved_database_urls()
    logger.info(
        "Database URLs — async (API): %s | sync (migrations): %s",
        mask_database_url(resolved.async_),
        mask_database_url(resolved.sync),
    )
    if db_ok:
        logger.info("INFO:     Application startup complete.")
    if not settings.paystack_secret_key.strip():
        logger.warning(
            "PAYSTACK_SECRET_KEY is not set — desktop billing (Subscribe to Pro) is disabled until configured."
        )
    if not (
        settings.paystack_plan_code_pro_monthly.strip()
        or settings.paystack_plan_code_pro_annual.strip()
    ):
        logger.warning(
            "No PAYSTACK_PLAN_CODE_PRO_* set — /v1/billing/plans will return 501 until plan codes are configured."
        )
    yield


def create_app() -> FastAPI:
    settings = get_cloud_settings()
    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
        debug=settings.debug,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(SQLAlchemyError)
    async def database_error_handler(_request: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.exception("Database error on request: %s", exc)
        detail = str(exc).strip() or type(exc).__name__
        if "does not exist" in detail.lower() or "undefinedcolumn" in detail.lower():
            detail = (
                f"{detail} — run: python -m alembic upgrade head, then restart python -m survyai_cloud"
            )
        return JSONResponse(status_code=500, content={"detail": detail})

    @app.get("/", tags=["health"])
    async def root() -> dict[str, object]:
        """
        Root URL: browsers often open http://host:port/ with no path.
        API routes live under /v1/...; use /docs for the full list.
        """
        return {
            "service": "survyai_cloud",
            "status": "ok",
            "message": "SurvyAI Cloud API is running. API routes are under /v1 (this path is not an API endpoint).",
            "links": {
                "health": "/health",
                "docs": "/docs",
                "openapi": "/openapi.json",
            },
        }

    app.include_router(health.router)
    app.include_router(auth.router, prefix="/v1")
    app.include_router(me.router, prefix="/v1")
    app.include_router(admin.router, prefix="/v1")
    app.include_router(devices.router, prefix="/v1")
    app.include_router(billing.router, prefix="/v1")
    app.include_router(webhooks.router, prefix="/v1")
    app.include_router(entitlements.router, prefix="/v1")
    app.include_router(usage.router, prefix="/v1")
    app.include_router(updates.router, prefix="/v1")
    app.include_router(diagnostics.router, prefix="/v1")

    return app


app = create_app()

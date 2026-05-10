"""
FastAPI entrypoint for the SurvyAI commercial backend.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from survyai_cloud.config import get_cloud_settings
from survyai_cloud.db import init_db
from survyai_cloud.routers import (
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
    app.include_router(devices.router, prefix="/v1")
    app.include_router(billing.router, prefix="/v1")
    app.include_router(webhooks.router, prefix="/v1")
    app.include_router(entitlements.router, prefix="/v1")
    app.include_router(usage.router, prefix="/v1")
    app.include_router(updates.router, prefix="/v1")
    app.include_router(diagnostics.router, prefix="/v1")

    return app


app = create_app()

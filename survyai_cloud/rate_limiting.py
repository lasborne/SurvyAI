"""
Per-user (+ client IP) fixed-window rate limits for hot authenticated routes.

Uses Redis when ``redis_url`` is set; otherwise an in-process fallback (single
worker only — configure Redis in production).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import Depends, HTTPException, Request, status

from survyai_cloud.config import get_cloud_settings
from survyai_cloud.deps import get_current_user
from survyai_cloud.models import User

logger = logging.getLogger(__name__)

_redis: Any = None
_mem_lock = asyncio.Lock()
_mem: dict[str, tuple[int, int]] = {}


async def _redis_client():
    global _redis
    settings = get_cloud_settings()
    url = settings.redis_url.strip()
    if not url:
        return None
    if _redis is None:
        import redis.asyncio as redis

        _redis = redis.from_url(url, decode_responses=True)
    return _redis


def _bucket(window_sec: int) -> int:
    return int(time.time()) // max(window_sec, 1)


def _client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


async def _check_memory(key: str, limit: int, window_sec: int) -> None:
    b = _bucket(window_sec)
    async with _mem_lock:
        cur = _mem.get(key)
        if cur is None or cur[1] != b:
            _mem[key] = (1, b)
            n = 1
        else:
            n = cur[0] + 1
            _mem[key] = (n, b)
    if n > limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again shortly.",
        )


async def _check_redis(key: str, limit: int, window_sec: int) -> None:
    client = await _redis_client()
    if client is None:
        return
    try:
        n = await client.incr(key)
        if n == 1:
            await client.expire(key, window_sec + 5)
        if int(n) > limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again shortly.",
            )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Redis rate limit error; falling back to in-process limiter")
        await _check_memory(key, limit, window_sec)


async def enforce_user_route_limit(
    *,
    request: Request,
    user_id: uuid.UUID,
    route_key: str,
    limit: int,
    window_sec: int,
) -> None:
    if limit <= 0:
        return
    ip = _client_ip(request)
    b = _bucket(window_sec)
    key = f"survyai:rl:v1:{route_key}:{user_id}:{ip}:{b}"
    if await _redis_client() is not None:
        await _check_redis(key, limit, window_sec)
    else:
        await _check_memory(key, limit, window_sec)


async def enforce_ip_route_limit(
    *,
    request: Request,
    route_key: str,
    limit: int,
    window_sec: int,
) -> None:
    """Unauthenticated fixed-window limit keyed by client IP."""
    if limit <= 0:
        return
    ip = _client_ip(request)
    b = _bucket(window_sec)
    key = f"survyai:rl:v1:{route_key}:ip:{ip}:{b}"
    if await _redis_client() is not None:
        await _check_redis(key, limit, window_sec)
    else:
        await _check_memory(key, limit, window_sec)


def rate_limit_user_dependency(route_key: str, settings_attr: str) -> Callable[..., Any]:
    """
    Return a FastAPI dependency (plain async function) so OpenAPI generation works.

    Callable-class ``__call__(self, request, user=Depends(...))`` breaks schema
    generation (Request mistaken for a body/query field).
    """

    async def _rate_limited_user(
        request: Request,
        user: User = Depends(get_current_user),
    ) -> User:
        settings = get_cloud_settings()
        limit = int(getattr(settings, settings_attr, 0) or 0)
        window = int(settings.rate_limit_window_seconds)
        await enforce_user_route_limit(
            request=request,
            user_id=user.id,
            route_key=route_key,
            limit=limit,
            window_sec=window,
        )
        return user

    return _rate_limited_user


def rate_limit_ip_dependency(
    route_key: str,
    settings_attr: str,
    *,
    window_seconds: int | None = None,
) -> Callable[..., Any]:
    """Rate-limit a public route by client IP using a CloudSettings limit field."""

    async def _rate_limited_ip(request: Request) -> None:
        settings = get_cloud_settings()
        limit = int(getattr(settings, settings_attr, 0) or 0)
        window = int(window_seconds if window_seconds is not None else settings.rate_limit_window_seconds)
        await enforce_ip_route_limit(
            request=request,
            route_key=route_key,
            limit=limit,
            window_sec=window,
        )

    return _rate_limited_ip

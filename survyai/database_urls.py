"""
Resolve sync vs async PostgreSQL (and SQLite) URLs for SurvyAI.

Use this module anywhere the project talks to Postgres so drivers match the workload:

| Workload                         | URL kind | Typical form                          |
|----------------------------------|----------|---------------------------------------|
| Alembic migrations               | **Sync** | ``postgresql+psycopg://…`` or ``postgresql://…`` |
| FastAPI / SQLAlchemy async       | **Async**| ``postgresql+asyncpg://…``            |
| Desktop vector store (psycopg)   | **Sync** | ``postgresql://…`` or ``postgresql+psycopg://…`` |

Environment variables (``.env`` / ``.env.cloud``):

- ``DATABASE_URL`` — primary URL. May be sync or async; other URLs are derived when needed.
- ``ASYNC_DATABASE_URL`` — optional; FastAPI uses this when set (recommended in production).
- ``VECTOR_DB_URL`` — optional sync URL for the desktop agent vector store; defaults to sync form of ``DATABASE_URL``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ResolvedDatabaseUrls:
    """Canonical URLs after normalizing env input."""

    sync: str
    """Alembic, admin scripts, desktop vector store (blocking driver)."""

    async_: str
    """FastAPI cloud API (asyncpg / aiosqlite)."""

    vector_sync: str
    """Desktop ``VectorStore`` — always sync psycopg-compatible."""


def _strip(url: str) -> str:
    return (url or "").strip()


def is_async_sqlalchemy_url(url: str) -> bool:
    u = _strip(url).lower()
    return "+asyncpg" in u or "+aiosqlite" in u


def is_sqlite_url(url: str) -> bool:
    return _strip(url).lower().startswith("sqlite")


def to_async_database_url(url: str) -> str:
    """
    Normalize to an async SQLAlchemy URL (``postgresql+asyncpg`` or ``sqlite+aiosqlite``).
    """
    u = _strip(url)
    if not u:
        return u
    if is_sqlite_url(u):
        if "+aiosqlite" in u:
            return u
        if u.startswith("sqlite://"):
            return "sqlite+aiosqlite://" + u[len("sqlite://") :]
        return u
    if "+asyncpg" in u:
        return u
    for prefix in ("postgresql+psycopg://", "postgresql://", "postgres://"):
        if u.startswith(prefix):
            rest = u[len(prefix) :]
            return "postgresql+asyncpg://" + rest
    return u


def to_sync_database_url(url: str) -> str:
    """
    Normalize to a sync SQLAlchemy / psycopg URL (``postgresql+psycopg`` or ``sqlite://``).
    """
    u = _strip(url)
    if not u:
        return u
    if is_sqlite_url(u):
        if "+aiosqlite" in u:
            return u.replace("sqlite+aiosqlite://", "sqlite://", 1)
        return u
    if "+asyncpg" in u:
        return "postgresql+psycopg://" + u.split("+asyncpg://", 1)[1]
    if u.startswith("postgresql+psycopg://"):
        return u
    if u.startswith("postgresql://"):
        return "postgresql+psycopg://" + u[len("postgresql://") :]
    if u.startswith("postgres://"):
        return "postgresql+psycopg://" + u[len("postgres://") :]
    return u


def to_vector_store_url(url: str) -> str:
    """
    Desktop vector store uses psycopg directly — prefer plain ``postgresql://`` DSN.
    """
    u = to_sync_database_url(url)
    if u.startswith("postgresql+psycopg://"):
        return "postgresql://" + u[len("postgresql+psycopg://") :]
    return u


def resolve_database_urls(
    *,
    database_url: str = "",
    async_database_url: str = "",
    vector_db_url: str = "",
) -> ResolvedDatabaseUrls:
    """
    Build sync, async, and vector URLs from ``.env`` values.

    Precedence:
    - Async API: ``ASYNC_DATABASE_URL`` if set, else async form of ``DATABASE_URL``.
    - Migrations: sync form of ``DATABASE_URL`` if it looks sync; else sync form of async URL.
    - Vector store: ``VECTOR_DB_URL`` if set, else sync form of ``DATABASE_URL``.
    """
    primary = _strip(database_url)
    async_explicit = _strip(async_database_url)
    vector_explicit = _strip(vector_db_url)

    if async_explicit:
        async_url = to_async_database_url(async_explicit)
    elif primary:
        async_url = to_async_database_url(primary)
    else:
        async_url = ""

    if primary and not is_async_sqlalchemy_url(primary) and not is_sqlite_url(primary):
        sync_url = to_sync_database_url(primary)
    elif primary and is_sqlite_url(primary):
        sync_url = primary if "+aiosqlite" not in primary else primary.replace(
            "sqlite+aiosqlite://", "sqlite://"
        )
    elif async_url:
        sync_url = to_sync_database_url(async_url)
    else:
        sync_url = ""

    if vector_explicit:
        vector_sync = to_vector_store_url(vector_explicit)
    elif sync_url:
        vector_sync = to_vector_store_url(sync_url)
    elif async_url:
        vector_sync = to_vector_store_url(async_url)
    else:
        vector_sync = ""

    return ResolvedDatabaseUrls(sync=sync_url, async_=async_url, vector_sync=vector_sync)


def mask_database_url(url: str) -> str:
    """Hide password in logs/UI."""
    return re.sub(r":([^:@/]+)@", ":****@", _strip(url))

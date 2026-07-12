from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text

from survyai_cloud.config import get_cloud_settings

logger = logging.getLogger(__name__)


def _ensure_sqlite_directory(url: str) -> None:
    if not url.startswith("sqlite"):
        return
    # sqlite+aiosqlite:///./relative/path.db
    rest = url.split("///", 1)
    if len(rest) != 2:
        return
    path_part = rest[1]
    if path_part.startswith(":memory:"):
        return
    p = Path(path_part)
    if not p.is_absolute():
        p = Path.cwd() / p
    p.parent.mkdir(parents=True, exist_ok=True)


_settings = get_cloud_settings()
_db_url = _settings.sqlalchemy_async_url()
_ensure_sqlite_directory(_db_url)

def _engine_connect_args(url: str) -> dict:
    """Fail fast when Postgres is unreachable (avoids 20s+ hangs on every API call)."""
    if url.startswith("postgresql+asyncpg://") or url.startswith("postgresql://"):
        return {"timeout": 10, "command_timeout": 20}
    return {}


engine = create_async_engine(
    _db_url,
    echo=_settings.debug,
    future=True,
    pool_pre_ping=True,
    connect_args=_engine_connect_args(_db_url),
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Commit after each successful request.

    ``async with session`` only calls ``close()`` on exit — it does **not**
    commit. Without an explicit ``commit()``, INSERT/UPDATE from auth and
    other routers are rolled back when the session closes, so e.g. register
    appears to succeed while login immediately fails with "Invalid email or password".

    If the database was unreachable at startup, this immediately raises a 503
    so the request fails fast instead of hanging for 10+ seconds.
    """
    if not _database_available:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=(
                f"Database unavailable: {_database_detail or 'unknown error'}. "
                "Check DATABASE_URL in .env and restart python -m survyai_cloud. "
                "Open http://127.0.0.1:8088/health for details."
            ),
        )
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def _alembic_upgrade_head_sync() -> None:
    """Run Alembic in a worker thread (avoids nested asyncio loops in lifespan)."""
    from alembic import command
    from alembic.config import Config

    root = Path(__file__).resolve().parents[1]
    ini = root / "alembic.ini"
    if not ini.is_file():
        raise FileNotFoundError(f"Alembic config missing: {ini}")
    cfg = Config(str(ini))
    cfg.set_main_option("script_location", str(root / "alembic"))
    command.upgrade(cfg, "head")


async def _ensure_pg_extensions(conn) -> None:
    """
    Idempotently create the PostgreSQL extensions required by SurvyAI.

    - vector   : pgvector for ANN/semantic search
    - postgis  : geospatial types and functions for survey coordinates
    - pg_trgm  : trigram similarity used in hybrid keyword search

    These are no-ops if the extensions are already installed, so it is safe
    to call on every startup.  Requires the database user to have CREATE
    privilege on the extensions (typically the database owner or a superuser).
    If the privilege is missing the error is logged but not re-raised so the
    application can still start in a degraded mode.
    """
    for ext in ("vector", "postgis", "pg_trgm"):
        try:
            await conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS {ext}"))
            logger.debug(f"PostgreSQL extension '{ext}' ready.")
        except Exception as exc:
            logger.warning(
                f"Could not create extension '{ext}': {exc}  "
                f"(The vector store may not function correctly – "
                f"ensure the database user has CREATE EXTENSION privilege.)"
            )


_database_available: bool = True
_database_detail: str = ""


def is_database_available() -> tuple[bool, str]:
    """Check the last-known DB status (set during startup and by /health probes)."""
    return _database_available, _database_detail


async def check_database(timeout_s: float = 5.0) -> tuple[bool, str]:
    """
    Lightweight connectivity probe for /health and startup logs.
    Returns (ok, detail_message).  Also updates the module-level flag so
    ``get_db()`` can fast-fail instead of hanging.
    """
    global _database_available, _database_detail

    settings = get_cloud_settings()
    async_url = settings.sqlalchemy_async_url()
    if async_url.startswith("sqlite"):
        try:
            async with asyncio.timeout(timeout_s):
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
            _database_available, _database_detail = True, "sqlite ok"
            return True, "sqlite ok"
        except TimeoutError:
            msg = f"Timed out after {timeout_s:.0f}s opening sqlite database."
            _database_available, _database_detail = False, msg
            return False, msg
        except Exception as exc:
            msg = str(exc).strip() or f"{type(exc).__name__}"
            _database_available, _database_detail = False, msg
            return False, msg

    if not async_url.startswith("postgresql"):
        msg = f"Unsupported database URL scheme: {async_url.split(':', 1)[0]}"
        _database_available, _database_detail = False, msg
        return False, msg

    try:
        async with asyncio.timeout(timeout_s):
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        _database_available, _database_detail = True, "postgresql ok"
        return True, "postgresql ok"
    except TimeoutError:
        msg = (
            f"Timed out after {timeout_s:.0f}s connecting to Postgres "
            f"(check DATABASE_URL, VPN, and that Supabase/local Docker is running)."
        )
        _database_available, _database_detail = False, msg
        return False, msg
    except Exception as exc:
        msg = str(exc).strip() or f"{type(exc).__name__}"
        _database_available, _database_detail = False, msg
        return False, msg


async def init_db() -> None:
    """
    Initialise the database layer.

    If the database is unreachable the server still starts in *degraded mode*
    so that ``/health`` can report the problem and the desktop app shows a
    clear error instead of a timeout.
    """
    global _database_available, _database_detail

    settings = get_cloud_settings()

    if settings.sqlalchemy_async_url().startswith("postgresql"):
        ok, detail = await check_database(timeout_s=12.0)
        if ok:
            logger.info("Database connection OK (%s).", detail)
            try:
                async with engine.begin() as conn:
                    await _ensure_pg_extensions(conn)
            except Exception as exc:
                logger.warning("Could not ensure PG extensions: %s", exc)
            if settings.run_migrations_on_startup:
                try:
                    await asyncio.to_thread(_alembic_upgrade_head_sync)
                except Exception as exc:
                    logger.error("Alembic migration failed: %s", exc)
        else:
            _database_available = False
            _database_detail = detail
            logger.error(
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║  DATABASE UNREACHABLE — server starting in degraded mode   ║\n"
                "╚══════════════════════════════════════════════════════════════╝\n"
                "  Error: %s\n"
                "  \n"
                "  Sign-in, billing, devices, and all authenticated routes\n"
                "  will return 503 until the database is reachable.\n"
                "  \n"
                "  ► For local dev:  docker compose up -d\n"
                "    DATABASE_URL=postgresql://survyai:survyai@localhost:5432/survyai\n"
                "    ASYNC_DATABASE_URL=postgresql+asyncpg://survyai:survyai@localhost:5432/survyai\n"
                "  ► For Supabase:   confirm the project is active (not paused) at\n"
                "    https://supabase.com/dashboard and copy the correct connection string.\n"
                "  \n"
                "  Restart python -m survyai_cloud after fixing .env.",
                detail,
            )
        return

    from survyai_cloud.models import Base

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await _sqlite_migrate(conn)
    except Exception as exc:
        _database_available = False
        _database_detail = str(exc).strip() or type(exc).__name__
        logger.error("SQLite init failed — degraded mode: %s", exc)


async def _sqlite_migrate(conn) -> None:
    """
    Minimal SQLite-only migrations for local dev.

    Production should use Alembic + Postgres; this keeps the dev DB working
    when models add new nullable columns.
    """
    settings = get_cloud_settings()
    if not settings.sqlalchemy_async_url().startswith("sqlite"):
        return

    # Users table: add columns if missing.
    res = await conn.execute(text("PRAGMA table_info(users)"))
    cols = {row[1] for row in res.fetchall()}  # row[1] == name

    alter = []
    if "paystack_customer_code" not in cols:
        alter.append("ALTER TABLE users ADD COLUMN paystack_customer_code VARCHAR(64)")
    if "paystack_subscription_code" not in cols:
        alter.append("ALTER TABLE users ADD COLUMN paystack_subscription_code VARCHAR(64)")
    if "paystack_email_token" not in cols:
        alter.append("ALTER TABLE users ADD COLUMN paystack_email_token VARCHAR(128)")
    if "last_payment_reference" not in cols:
        alter.append("ALTER TABLE users ADD COLUMN last_payment_reference VARCHAR(128)")
    if "usage_period_anchor" not in cols:
        alter.append("ALTER TABLE users ADD COLUMN usage_period_anchor DATETIME")
    if "monthly_credits_usd" not in cols:
        alter.append("ALTER TABLE users ADD COLUMN monthly_credits_usd FLOAT NOT NULL DEFAULT 0.0")
    if "monthly_credits_used_usd" not in cols:
        alter.append("ALTER TABLE users ADD COLUMN monthly_credits_used_usd FLOAT NOT NULL DEFAULT 0.0")
    if "credits_billing_interval" not in cols:
        alter.append(
            "ALTER TABLE users ADD COLUMN credits_billing_interval VARCHAR(16) NOT NULL DEFAULT 'monthly'"
        )
    if "password_changed_at" not in cols:
        alter.append("ALTER TABLE users ADD COLUMN password_changed_at DATETIME")

    for stmt in alter:
        await conn.execute(text(stmt))

    # Password reset tokens (local SQLite / create_all may already have created this).
    await conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
              id CHAR(32) NOT NULL PRIMARY KEY,
              user_id CHAR(32) NOT NULL,
              token_hash VARCHAR(64) NOT NULL UNIQUE,
              expires_at DATETIME NOT NULL,
              used_at DATETIME,
              request_ip VARCHAR(64),
              created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
              FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )
    )
    await conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_password_reset_tokens_token_hash "
            "ON password_reset_tokens (token_hash)"
        )
    )
    await conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_password_reset_tokens_user_id "
            "ON password_reset_tokens (user_id)"
        )
    )

    # Usage events table: add cost column if missing.
    res = await conn.execute(text("PRAGMA table_info(usage_events)"))
    ev_info = res.fetchall()
    ev_cols = {row[1] for row in ev_info}
    if "cost_usd" not in ev_cols:
        await conn.execute(text("ALTER TABLE usage_events ADD COLUMN cost_usd FLOAT NOT NULL DEFAULT 0.0"))

    # SQLite autoincrement fix: older dev DBs may have been created with BIGINT PK
    # which does not auto-populate, causing IntegrityError: NOT NULL constraint failed.
    # If the id column is not INTEGER, recreate the table with a correct PK.
    id_row = None
    for row in ev_info:
        # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
        if row[1] == "id":
            id_row = row
            break
    if id_row is not None:
        id_type = str(id_row[2] or "").upper()
        if id_type and id_type != "INTEGER":
            # Best-effort migration for local dev only.
            await conn.execute(text("PRAGMA foreign_keys=OFF"))
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS usage_events_new (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id CHAR(32) NOT NULL,
                      kind VARCHAR(64) NOT NULL,
                      quantity INTEGER NOT NULL DEFAULT 1,
                      cost_usd FLOAT NOT NULL DEFAULT 0.0,
                      meta JSON,
                      device_id CHAR(32),
                      created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP)
                    )
                    """
                )
            )
            # Copy what we can; ids will be regenerated.
            await conn.execute(
                text(
                    """
                    INSERT INTO usage_events_new (user_id, kind, quantity, cost_usd, meta, device_id, created_at)
                    SELECT user_id, kind, quantity,
                           COALESCE(cost_usd, 0.0),
                           meta,
                           device_id,
                           COALESCE(created_at, CURRENT_TIMESTAMP)
                    FROM usage_events
                    """
                )
            )
            await conn.execute(text("DROP TABLE usage_events"))
            await conn.execute(text("ALTER TABLE usage_events_new RENAME TO usage_events"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_usage_user_created ON usage_events (user_id, created_at)"))
            await conn.execute(text("PRAGMA foreign_keys=ON"))

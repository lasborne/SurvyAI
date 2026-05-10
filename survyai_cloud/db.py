from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text

from survyai_cloud.config import get_cloud_settings


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
_ensure_sqlite_directory(_settings.database_url)

engine = create_async_engine(
    _settings.database_url,
    echo=_settings.debug,
    future=True,
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
    """
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


async def init_db() -> None:
    settings = get_cloud_settings()
    if settings.database_url.startswith("postgresql"):
        if settings.run_migrations_on_startup:
            await asyncio.to_thread(_alembic_upgrade_head_sync)
        return

    from survyai_cloud.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _sqlite_migrate(conn)


async def _sqlite_migrate(conn) -> None:
    """
    Minimal SQLite-only migrations for local dev.

    Production should use Alembic + Postgres; this keeps the dev DB working
    when models add new nullable columns.
    """
    settings = get_cloud_settings()
    if not settings.database_url.startswith("sqlite"):
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

    for stmt in alter:
        await conn.execute(text(stmt))

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

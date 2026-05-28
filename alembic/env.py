"""
Alembic environment for SurvyAI Cloud.

Uses the **sync** database URL (``DATABASE_URL`` or sync form derived from
``ASYNC_DATABASE_URL``).  The live FastAPI app uses the async URL separately —
see ``survyai/database_urls.py`` and ``survyai_cloud.config.CloudSettings``.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool
from sqlalchemy.engine import Connection

# Repo root on sys.path (alembic/ -> parent)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from survyai_cloud.config import get_cloud_settings  # noqa: E402
from survyai_cloud.models import Base  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL without connecting (URL from settings)."""
    settings = get_cloud_settings()
    context.configure(
        url=settings.sqlalchemy_sync_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations with a synchronous engine (psycopg / sqlite)."""
    settings = get_cloud_settings()
    url = settings.sqlalchemy_sync_url()
    connectable = create_engine(url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        do_run_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

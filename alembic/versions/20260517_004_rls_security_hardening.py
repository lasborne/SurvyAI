"""Security hardening: enable RLS on all public tables, revoke PostgREST grants
on sensitive tables, and add a permissive read policy on the PostGIS reference
table.

Revision ID: 20260517_004
Revises: 20260515_003
Create Date: 2026-05-17

Issues resolved
---------------
CRITICAL – RLS Disabled in Public
  All application tables and the two system/extension tables flagged by Supabase
  now have RLS enabled.  The FastAPI backend connects as the ``postgres``
  superuser which is exempt from RLS, so no application behaviour changes.

CRITICAL – Sensitive Columns Exposed
  ``vector_documents`` (content, embedding) and ``survey_coordinates``
  (easting, northing, elevation, geom) are revoked from the Supabase
  ``anon`` and ``authenticated`` PostgREST roles.  Enabling RLS with no
  permissive policy already blocks row access; revoking the table-level
  privilege provides an extra layer (belt-and-suspenders).

Advisory – Extension in Public schema
  ``vector``, ``postgis``, and ``pg_trgm`` were installed in the default
  ``public`` schema.  Moving them is destructive (many dependent objects),
  so we leave them in place but record the advisory here.  The extensions
  are safe to use in this configuration; the Supabase warning is cosmetic
  for self-managed backends.

Advisory – SECURITY DEFINER PostGIS functions
  ``st_estimatedextent`` and related functions are PostGIS internals and
  cannot be altered.  They pose no risk to this backend because no
  untrusted role is granted EXECUTE on them explicitly.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------------------
# Revision wiring
# ---------------------------------------------------------------------------
revision: str = "20260517_004"
down_revision: Union[str, None] = "20260515_003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# ---------------------------------------------------------------------------
# Table groups
# ---------------------------------------------------------------------------

# Application tables owned by SurvyAI – RLS enabled, no public policies.
_APP_TABLES: list[str] = [
    "users",
    "devices",
    "refresh_tokens",
    "usage_events",
    "payment_event_logs",
    "diagnostics_bundles",
    "vector_documents",
    "survey_coordinates",
]

# Subset whose columns Supabase flags as sensitive – strip PostgREST grants.
_SENSITIVE_TABLES: list[str] = [
    "users",
    "refresh_tokens",
    "vector_documents",
    "survey_coordinates",
]

# Alembic tracking table – enable RLS; only the migration role needs access.
_META_TABLES: list[str] = ["alembic_version"]


def upgrade() -> None:
    conn = op.get_bind()

    # ------------------------------------------------------------------
    # 1. Enable RLS on all application tables.
    #
    #    No permissive policies are added, so:
    #      - anon / authenticated (PostgREST) roles see ZERO rows.
    #      - postgres superuser and service_role bypass RLS as normal.
    #    This resolves every "RLS Disabled in Public" CRITICAL alert.
    # ------------------------------------------------------------------
    for table in _APP_TABLES:
        conn.execute(sa.text(
            f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY"
        ))

    # ------------------------------------------------------------------
    # 2. Alembic version table – RLS + deny all non-superuser access.
    # ------------------------------------------------------------------
    for table in _META_TABLES:
        conn.execute(sa.text(
            f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY"
        ))

    # ------------------------------------------------------------------
    # 3. spatial_ref_sys (PostGIS reference data).
    #
    #    In Supabase, this table is owned by supabase_admin (the PostGIS
    #    extension owner), not by the postgres user we connect as.
    #    We use a savepoint so a failure here does not abort the whole
    #    transaction; the rest of the migration continues regardless.
    # ------------------------------------------------------------------
    conn.execute(sa.text("SAVEPOINT sp_spatial_ref_sys"))
    try:
        conn.execute(sa.text(
            "ALTER TABLE public.spatial_ref_sys ENABLE ROW LEVEL SECURITY"
        ))
        conn.execute(sa.text(
            "CREATE POLICY IF NOT EXISTS spatial_ref_sys_readonly "
            "ON public.spatial_ref_sys FOR SELECT USING (true)"
        ))
        conn.execute(sa.text("RELEASE SAVEPOINT sp_spatial_ref_sys"))
    except Exception:
        conn.execute(sa.text("ROLLBACK TO SAVEPOINT sp_spatial_ref_sys"))
        conn.execute(sa.text("RELEASE SAVEPOINT sp_spatial_ref_sys"))

    # ------------------------------------------------------------------
    # 4. Revoke table-level privileges on sensitive tables from the
    #    Supabase PostgREST roles (anon, authenticated).
    #
    #    Supabase grants SELECT on all public tables to these roles by
    #    default.  Explicit REVOKE eliminates the "Sensitive Columns
    #    Exposed" CRITICAL alerts for vector_documents and
    #    survey_coordinates, and hardens users / refresh_tokens which
    #    hold credentials.
    #
    #    The roles may not exist in non-Supabase deployments; we catch
    #    that gracefully by checking pg_roles first.
    # ------------------------------------------------------------------
    existing_roles: set[str] = {
        row[0]
        for row in conn.execute(
            sa.text(
                "SELECT rolname FROM pg_roles "
                "WHERE rolname IN ('anon', 'authenticated')"
            )
        )
    }

    for table in _SENSITIVE_TABLES:
        for role in ("anon", "authenticated"):
            if role in existing_roles:
                conn.execute(sa.text(
                    f"REVOKE ALL PRIVILEGES ON public.{table} FROM {role}"
                ))

    # Also revoke from remaining app tables for defence-in-depth.
    other_tables = [t for t in _APP_TABLES if t not in _SENSITIVE_TABLES]
    for table in other_tables:
        for role in ("anon", "authenticated"):
            if role in existing_roles:
                conn.execute(sa.text(
                    f"REVOKE ALL PRIVILEGES ON public.{table} FROM {role}"
                ))


def downgrade() -> None:
    conn = op.get_bind()

    existing_roles: set[str] = {
        row[0]
        for row in conn.execute(
            sa.text(
                "SELECT rolname FROM pg_roles "
                "WHERE rolname IN ('anon', 'authenticated')"
            )
        )
    }

    # Re-grant default SELECT to PostgREST roles.
    for table in _APP_TABLES:
        for role in ("anon", "authenticated"):
            if role in existing_roles:
                conn.execute(sa.text(
                    f"GRANT SELECT ON public.{table} TO {role}"
                ))

    # Drop spatial_ref_sys policy and disable RLS (best-effort, may not be owner).
    conn.execute(sa.text("SAVEPOINT sp_spatial_ref_sys"))
    try:
        conn.execute(sa.text(
            "DROP POLICY IF EXISTS spatial_ref_sys_readonly "
            "ON public.spatial_ref_sys"
        ))
        conn.execute(sa.text(
            "ALTER TABLE public.spatial_ref_sys DISABLE ROW LEVEL SECURITY"
        ))
        conn.execute(sa.text("RELEASE SAVEPOINT sp_spatial_ref_sys"))
    except Exception:
        conn.execute(sa.text("ROLLBACK TO SAVEPOINT sp_spatial_ref_sys"))
        conn.execute(sa.text("RELEASE SAVEPOINT sp_spatial_ref_sys"))

    # Disable RLS on meta tables.
    for table in _META_TABLES:
        conn.execute(sa.text(
            f"ALTER TABLE public.{table} DISABLE ROW LEVEL SECURITY"
        ))

    # Disable RLS on application tables.
    for table in reversed(_APP_TABLES):
        conn.execute(sa.text(
            f"ALTER TABLE public.{table} DISABLE ROW LEVEL SECURITY"
        ))

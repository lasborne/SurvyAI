"""Add pgvector + PostGIS: vector_documents and survey_coordinates tables.

Revision ID: 20260515_003
Revises: 20260503_002
Create Date: 2026-05-15

Design rationale
----------------
* Replaces the ChromaDB-on-disk store with a proper relational + vector store
  living inside the SAME PostgreSQL instance as the cloud API.  This gives us:
  - ACID for every INSERT/UPDATE, no "partial write" corruption under concurrency.
  - Rich JOINs: usage_events ↔ vector_documents, coordinates ↔ users, etc.
  - A single backup target (pg_dump covers everything).
  - HNSW index for sub-millisecond ANN queries at scale (pgvector ≥ 0.5).
  - Full-text search (tsvector/tsquery) for keyword recall, combined with
    cosine-similarity using Reciprocal Rank Fusion (hybrid retrieval).
  - PostGIS for professional geospatial queries on survey coordinates.

Extensions
----------
We create three extensions idempotently:
  vector    – pgvector ANN search
  postgis   – geospatial geometry types and functions
  pg_trgm   – trigram similarity (used for fuzzy keyword matching)

Tables
------
vector_documents
  Replaces all four ChromaDB collections (documents, drawings, coordinates,
  conversations).  ``collection`` column is the discriminator; indexed.
  ``embedding`` stores the dense vector (dimension from VECTOR_EMBEDDING_DIM
  env var, default 1536).  ``content_tsv`` is a generated column used for BM25
  / ts_rank full-text ranking.  ``metadata`` uses JSONB so we can index
  specific JSON keys and filter with @> operators.

survey_coordinates
  First-class geospatial store.  Each surveyed point stores:
  - raw easting/northing in the project CRS (doubles)
  - a PostGIS Point in WGS84 (for haversine proximity, leaflet display, etc.)
  - CRS metadata and the originating session_id for scoping.

Indexes
-------
  hnsw on embedding – cosine ops, m=16 ef_construction=64 (good default for
  medium corpora; tune m/ef_construction via env for very large deployments).
  gin on content_tsv – inverted index for full-text recall.
  gin on metadata   – for JSONB containment queries like metadata @> '{"type":"report"}'.
  btree on (collection, created_at) – efficient time-ordered scans per collection.
  btree on session_id              – fast conversation history retrieval.
  gist on geom                    – PostGIS spatial index.
"""

from __future__ import annotations

import os
from typing import Sequence, Union

import sqlalchemy as sa
import sqlalchemy.dialects.postgresql  # noqa: F401 – required for JSONB in create_table
from alembic import op

# ---------------------------------------------------------------------------
# Revision wiring
# ---------------------------------------------------------------------------
revision: str = "20260515_003"
down_revision: Union[str, None] = "20260503_002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# ---------------------------------------------------------------------------
# Embedding dimension – matches the provider used at runtime.
#   384  → all-MiniLM-L6-v2  (local, default for air-gapped deployments)
#   1536 → text-embedding-3-small / ada-002  (OpenAI, default cloud)
# Change via env: VECTOR_EMBEDDING_DIM=384 alembic upgrade head
# ---------------------------------------------------------------------------
_DIM: int = int(os.environ.get("VECTOR_EMBEDDING_DIM", "1536"))

# HNSW tuning (can be overridden via env vars for large-scale deployments)
_HNSW_M: int = int(os.environ.get("VECTOR_HNSW_M", "16"))
_HNSW_EF: int = int(os.environ.get("VECTOR_HNSW_EF_CONSTRUCTION", "64"))


def upgrade() -> None:
    conn = op.get_bind()

    # ------------------------------------------------------------------
    # 1. Create extensions idempotently (requires pg_extension_owner or
    #    superuser; harmless if extensions already installed).
    # ------------------------------------------------------------------
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS postgis"))
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

    # ------------------------------------------------------------------
    # 2. vector_documents – replaces the four ChromaDB collections
    # ------------------------------------------------------------------
    op.create_table(
        "vector_documents",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("collection", sa.String(64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        # vector(n) is the pgvector type – n MUST match the embedding model
        sa.Column("embedding", sa.Text(), nullable=True),   # placeholder; ALTER'ed below
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'{}'")),
        sa.Column("session_id", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
    )

    # Replace placeholder TEXT column with a proper vector(n) column.
    # We cannot do this in create_table because SQLAlchemy's DDL layer
    # does not know the pgvector type natively.
    conn.execute(sa.text(f"ALTER TABLE vector_documents DROP COLUMN embedding"))
    conn.execute(sa.text(
        f"ALTER TABLE vector_documents "
        f"ADD COLUMN embedding vector({_DIM})"
    ))

    # Generated tsvector column for full-text search (Postgres 12+)
    conn.execute(sa.text(
        "ALTER TABLE vector_documents "
        "ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    ))

    # ------------------------------------------------------------------
    # 3. Indexes on vector_documents
    # ------------------------------------------------------------------

    # HNSW index for cosine-similarity ANN (fast at query time, slower build)
    conn.execute(sa.text(
        f"CREATE INDEX ix_vd_embedding_hnsw "
        f"ON vector_documents "
        f"USING hnsw (embedding vector_cosine_ops) "
        f"WITH (m = {_HNSW_M}, ef_construction = {_HNSW_EF})"
    ))

    # GIN inverted index for full-text keyword recall
    conn.execute(sa.text(
        "CREATE INDEX ix_vd_content_tsv "
        "ON vector_documents USING gin (content_tsv)"
    ))

    # JSONB containment index (e.g. WHERE metadata @> '{\"type\": \"report\"}')
    conn.execute(sa.text(
        "CREATE INDEX ix_vd_metadata "
        "ON vector_documents USING gin (metadata)"
    ))

    # Efficient per-collection time-ordered scans
    conn.execute(sa.text(
        "CREATE INDEX ix_vd_collection_created "
        "ON vector_documents (collection, created_at DESC)"
    ))

    # Fast session-scoped conversation lookups
    conn.execute(sa.text(
        "CREATE INDEX ix_vd_session_id "
        "ON vector_documents (session_id) "
        "WHERE session_id IS NOT NULL"
    ))

    # ------------------------------------------------------------------
    # 4. survey_coordinates – PostGIS-backed coordinate store
    # ------------------------------------------------------------------
    op.create_table(
        "survey_coordinates",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("label", sa.String(256), nullable=True),
        sa.Column("easting", sa.Double(), nullable=True),
        sa.Column("northing", sa.Double(), nullable=True),
        sa.Column("elevation", sa.Double(), nullable=True),
        sa.Column("crs_name", sa.String(128), nullable=True),
        sa.Column("epsg_code", sa.Integer(), nullable=True),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'{}'")),
        sa.Column("session_id", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
    )

    # PostGIS geometry column: Point in WGS84 (lon, lat for global coverage)
    # Stored as GEOMETRY(Point, 4326) so PostGIS spatial functions work directly.
    conn.execute(sa.text(
        "SELECT AddGeometryColumn('survey_coordinates', 'geom', 4326, 'POINT', 2)"
    ))

    # GiST spatial index (enables ST_DWithin, ST_Within, bounding-box queries)
    conn.execute(sa.text(
        "CREATE INDEX ix_sc_geom "
        "ON survey_coordinates USING gist (geom)"
    ))
    conn.execute(sa.text(
        "CREATE INDEX ix_sc_session "
        "ON survey_coordinates (session_id) "
        "WHERE session_id IS NOT NULL"
    ))
    conn.execute(sa.text(
        "CREATE INDEX ix_sc_epsg "
        "ON survey_coordinates (epsg_code) "
        "WHERE epsg_code IS NOT NULL"
    ))

    # ------------------------------------------------------------------
    # 5. Expose embedding dimension as a DB comment for diagnostics
    # ------------------------------------------------------------------
    conn.execute(sa.text(
        f"COMMENT ON COLUMN vector_documents.embedding IS "
        f"'Dense vector, dimension={_DIM}. "
        f"Do not change VECTOR_EMBEDDING_DIM without re-running migrations.'"
    ))


def downgrade() -> None:
    op.drop_table("survey_coordinates")
    op.drop_table("vector_documents")
    # We intentionally do NOT drop the extensions – other databases/schemas
    # on the same PostgreSQL cluster might depend on them.

"""
================================================================================
SurvyAI Vector Store  –  PostgreSQL / pgVector / PostGIS backend
================================================================================

Replaces the previous ChromaDB-based implementation with a production-grade
store built on PostgreSQL + pgvector + PostGIS.

WHY THE CHANGE
--------------
ChromaDB is an excellent prototyping tool but has known limitations for
medium-to-heavy production usage:
  • Single-machine file-based HNSW index that does not survive concurrent
    writers without a running server process.
  • No ACID guarantees: a crash mid-write can leave the index inconsistent.
  • Limited WHERE filtering (dict equality only; no range queries, no JSONB
    containment).
  • No hybrid search (vector + keyword) out of the box.
  • Zero geospatial capability — critical for a surveying AI agent.
  • Separate backup/restore story from the application database.

PostgreSQL solves every one of those problems:
  • pgvector stores embeddings as first-class column types with HNSW or IVFFlat
    ANN indexes, queried with ``<=>`` (cosine), ``<->`` (L2), or ``<#>`` (inner
    product).
  • PostGIS gives us professional-grade geometry storage, spatial indexes, and
    all of `ST_DWithin`, `ST_Buffer`, `ST_Transform`, etc.
  • Full-text search (tsvector/tsquery / ts_rank) can be combined with vector
    similarity using Reciprocal Rank Fusion for hybrid retrieval — significantly
    better recall than pure semantic search.
  • One ``pg_dump`` covers the entire application state.

PUBLIC API
----------
The class intentionally preserves the exact same method signatures as the old
ChromaDB-based ``VectorStore`` so that ``agent/agent.py`` needs zero changes:

  add_documents(documents, collection)
  add_text(text, metadata, collection, doc_id)
  search(query, collection, top_k, where, include_embeddings)
  search_all_collections(query, top_k, where)
  add_conversation(role, content, session_id, metadata)
  get_recent_conversations(session_id, limit, role)
  get_stats()
  clear_collection(collection)
  delete_document(doc_id, collection)
  get_document(doc_id, collection)
  reset()
  add_autocad_entities(entities, drawing_name, session_id)
  add_coordinates(coordinates, crs_name, session_id)

NEW CAPABILITIES
----------------
  hybrid_search(query, collection, top_k, where, alpha)
    Combines cosine-similarity ANN with BM25/ts_rank full-text search using
    Reciprocal Rank Fusion (RRF).  ``alpha`` controls the balance:
      0.0  → pure keyword  |  1.0 → pure semantic  |  0.5 → equal weight.

  find_nearby_coordinates(lat, lon, radius_m, limit)
    PostGIS ST_DWithin query: returns all survey_coordinates within
    ``radius_m`` metres of the given WGS84 point.

THREADING / ASYNC
-----------------
``VectorStore`` is synchronous (for use in the agent's sync LangChain tool
wrappers).  It uses ``psycopg_pool.ConnectionPool`` which is thread-safe and
safe to share across worker threads.

The ``survyai_cloud`` FastAPI backend uses its own async SQLAlchemy + asyncpg
engine (unchanged).  They share the same PostgreSQL database/schema.

Author: SurvyAI Team
License: MIT
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# psycopg v3 (sync, thread-safe connection pool)
# ---------------------------------------------------------------------------
try:
    import psycopg
    from psycopg import Connection, sql as pgsql
    from psycopg_pool import ConnectionPool
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    psycopg = None  # type: ignore
    ConnectionPool = None  # type: ignore

# pgvector adapter for psycopg3
try:
    from pgvector.psycopg import register_vector as _register_vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    _register_vector = None  # type: ignore

# numpy – needed by pgvector for typed arrays
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

# Sentence Transformers – local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore

# OpenAI – cloud embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collection names (unchanged from the ChromaDB version)
# ---------------------------------------------------------------------------
COLLECTION_DOCUMENTS = "documents"
COLLECTION_DRAWINGS = "drawings"
COLLECTION_COORDINATES = "coordinates"
COLLECTION_CONVERSATIONS = "conversations"

ALL_COLLECTIONS = (
    COLLECTION_DOCUMENTS,
    COLLECTION_DRAWINGS,
    COLLECTION_COORDINATES,
    COLLECTION_CONVERSATIONS,
)

DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


# ==============================================================================
# EMBEDDING PROVIDERS  (unchanged API from previous implementation)
# ==============================================================================

class LocalEmbeddingProvider:
    """
    Sentence Transformers embedding provider.
    Runs fully offline after the initial model download.  Dimension is
    model-dependent (all-MiniLM-L6-v2 → 384, all-mpnet-base-v2 → 768).
    """

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL) -> None:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "pip install sentence-transformers"
            )
        self.model_name = model_name
        logger.info(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._dimension: int = self.model.get_sentence_embedding_dimension()
        logger.info(f"✓ Local embedding model loaded (dim={self._dimension})")

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbeddingProvider:
    """
    OpenAI embeddings (text-embedding-3-small / text-embedding-3-large).
    Requires OPENAI_API_KEY in environment or passed directly.
    """

    _DIMS: Dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_OPENAI_MODEL,
    ) -> None:
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required: pip install openai")
        if not api_key or not api_key.strip():
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self._dimension = self._DIMS.get(model_name, 1536)
        logger.info(f"✓ OpenAI embedding provider ready (model={model_name}, dim={self._dimension})")

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        batch = 100
        for i in range(0, len(texts), batch):
            resp = self.client.embeddings.create(
                model=self.model_name, input=texts[i : i + batch]
            )
            out.extend(item.embedding for item in resp.data)
        return out

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


# ==============================================================================
# VECTOR STORE
# ==============================================================================

class VectorStore:
    """
    Production vector store backed by PostgreSQL + pgvector + PostGIS.

    Drop-in replacement for the old ChromaDB-based implementation.
    The entire public API is preserved; new methods are additive.

    Connection
    ----------
    Pass ``db_url`` (sync libpq DSN or ``postgresql://...`` URL) or set the
    ``VECTOR_DB_URL`` environment variable.  The pool opens lazily on first use
    and is kept alive for the process lifetime.

    Embedding dimension
    -------------------
    The embedding column in ``vector_documents`` is created by Alembic migration
    ``20260515_003`` with a fixed dimension.  Ensure the provider you choose here
    produces vectors of that dimension (default 1536).  Set ``VECTOR_EMBEDDING_DIM``
    to change it and re-run Alembic.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        db_url: Optional[str] = None,
        embedding_provider: Literal["local", "openai"] = "local",
        openai_api_key: Optional[str] = None,
        local_model_name: str = DEFAULT_LOCAL_MODEL,
        openai_model_name: str = DEFAULT_OPENAI_MODEL,
        # Legacy ChromaDB argument – accepted and ignored for back-compat
        persist_directory: Optional[str] = None,
        # Pool tuning
        pool_min_size: int = 1,
        pool_max_size: int = 10,
    ) -> None:
        if not PSYCOPG_AVAILABLE:
            raise ImportError(
                "psycopg[binary,pool] is required for PostgreSQL vector storage.\n"
                "pip install 'psycopg[binary,pool]' pgvector"
            )

        # Resolve DSN
        self._db_url: str = (
            db_url
            or os.environ.get("VECTOR_DB_URL", "")
            or os.environ.get("DATABASE_URL", "")
        )
        if not self._db_url:
            raise ValueError(
                "No PostgreSQL URL provided.  Set VECTOR_DB_URL (or DATABASE_URL) "
                "in your .env file or pass db_url= to VectorStore()."
            )

        from survyai.database_urls import to_vector_store_url

        self._db_url = to_vector_store_url(self._db_url)

        self._pool_min = pool_min_size
        self._pool_max = pool_max_size
        self._pool: Optional[ConnectionPool] = None  # lazy init

        # Embedding provider
        self.embedding_provider = self._make_provider(
            embedding_provider, openai_api_key, local_model_name, openai_model_name
        )
        self.provider_type: str = embedding_provider
        self._expected_dim: int = self.embedding_provider.dimension

        logger.info(
            f"VectorStore initialised: backend=postgresql "
            f"embedding={embedding_provider} dim={self._expected_dim}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_provider(
        provider_type: str,
        openai_api_key: Optional[str],
        local_model_name: str,
        openai_model_name: str,
    ):
        if provider_type == "openai":
            key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
            if not key:
                logger.warning("OPENAI_API_KEY missing – falling back to local embeddings.")
                provider_type = "local"
            else:
                try:
                    return OpenAIEmbeddingProvider(key, openai_model_name)
                except Exception as exc:
                    logger.warning(f"OpenAI embedding init failed ({exc}) – falling back to local.")
                    provider_type = "local"
        return LocalEmbeddingProvider(local_model_name)

    def _get_pool(self) -> ConnectionPool:
        """Lazy-create the connection pool on first use."""
        if self._pool is None:
            self._pool = ConnectionPool(
                conninfo=self._db_url,
                min_size=self._pool_min,
                max_size=self._pool_max,
                kwargs={"autocommit": False},
            )
            # Validate connectivity + register pgvector adapter
            with self._pool.connection() as conn:
                if PGVECTOR_AVAILABLE:
                    _register_vector(conn)
            logger.info("✓ PostgreSQL connection pool ready")
        return self._pool

    @contextmanager
    def _conn(self) -> Iterator[psycopg.Connection]:
        """Yield a checked-out pooled connection with auto-commit on success."""
        pool = self._get_pool()
        with pool.connection() as conn:
            if PGVECTOR_AVAILABLE:
                _register_vector(conn)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    def _gen_id(content: str, metadata: Optional[Dict] = None) -> str:
        """Return a deterministic UUID from content + metadata (SHA-256 based)."""
        h = hashlib.sha256(content.encode("utf-8"))
        if metadata:
            h.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
        # Take first 32 hex chars (128 bits) and format as UUID
        return str(uuid.UUID(hex=h.hexdigest()[:32]))

    def _to_vec(self, embedding: List[float]):
        """Convert a Python list to a pgvector-compatible numpy array."""
        if NUMPY_AVAILABLE:
            return np.array(embedding, dtype=np.float32)
        return embedding

    # ------------------------------------------------------------------
    # DOCUMENT OPERATIONS  (API-compatible with ChromaDB VectorStore)
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        collection: str = COLLECTION_DOCUMENTS,
    ) -> List[str]:
        """
        Add documents to the specified collection.

        Each dict may contain:
          ``content``  – required text
          ``metadata`` – optional dict
          ``id``       – optional stable ID (SHA-256 of content if omitted)
        """
        if not documents:
            return []
        if collection not in ALL_COLLECTIONS:
            raise ValueError(
                f"Unknown collection '{collection}'. "
                f"Valid: {ALL_COLLECTIONS}"
            )

        rows: List[Tuple] = []
        now = datetime.utcnow().isoformat()
        for doc in documents:
            content = (doc.get("content") or "").strip()
            if not content:
                continue
            meta: Dict[str, Any] = dict(doc.get("metadata") or {})
            meta.update({"added_at": now, "collection": collection})
            doc_id = doc.get("id") or self._gen_id(content, meta)
            rows.append((doc_id, collection, content, json.dumps(meta),
                         doc.get("session_id") or meta.get("session_id")))

        if not rows:
            return []

        contents = [r[2] for r in rows]
        logger.info(f"Embedding {len(contents)} documents for '{collection}'…")
        embeddings = self.embedding_provider.embed(contents)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO vector_documents
                        (id, collection, content, embedding, metadata, session_id)
                    VALUES (%s, %s, %s, %s::vector, %s::jsonb, %s)
                    ON CONFLICT (id) DO UPDATE
                        SET content    = EXCLUDED.content,
                            embedding  = EXCLUDED.embedding,
                            metadata   = EXCLUDED.metadata,
                            session_id = EXCLUDED.session_id
                    """,
                    [
                        (r[0], r[1], r[2],
                         self._to_vec(embeddings[i]),
                         r[3], r[4])
                        for i, r in enumerate(rows)
                    ],
                )
        ids = [r[0] for r in rows]
        logger.info(f"✓ Upserted {len(ids)} documents into '{collection}'")
        return ids

    def add_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        collection: str = COLLECTION_DOCUMENTS,
        doc_id: Optional[str] = None,
    ) -> str:
        """Convenience wrapper: add a single text document."""
        ids = self.add_documents(
            [{"content": text, "metadata": metadata or {}, "id": doc_id}],
            collection=collection,
        )
        return ids[0] if ids else ""

    # ------------------------------------------------------------------
    # SEARCH OPERATIONS
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        collection: str = COLLECTION_DOCUMENTS,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Pure semantic (cosine ANN) search via pgvector.

        Args:
            query:             Natural-language query string.
            collection:        Which collection to search.
            top_k:             Maximum number of results.
            where:             Optional metadata filter expressed as a dict;
                               each key/value is added as ``metadata @> '{"k":"v"}'``.
                               E.g. ``where={"session_id": "abc"}`` →
                               ``WHERE metadata @> '{"session_id": "abc"}'``.
                               Special key ``session_id`` also matches the
                               dedicated session_id column for efficiency.
            include_embeddings: Not returned (overhead not worth it in prod).

        Returns:
            List of dicts with keys: id, content, metadata, score.
            ``score`` is cosine similarity (0–1, higher = more similar).
        """
        if not query:
            return []
        if collection not in ALL_COLLECTIONS:
            raise ValueError(f"Unknown collection '{collection}'")

        q_emb = self._to_vec(self.embedding_provider.embed_query(query))
        where_sql, where_params = self._build_where(collection, where)

        sql = f"""
            SELECT id::text, content, metadata, session_id,
                   1 - (embedding <=> %s::vector) AS score
            FROM vector_documents
            {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = (q_emb, q_emb, top_k)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, where_params + params)
                rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "content": row[1],
                "metadata": row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}"),
                "score": float(row[4]),
            }
            for row in rows
            if float(row[4]) >= 0.0
        ]

    def hybrid_search(
        self,
        query: str,
        collection: str = COLLECTION_DOCUMENTS,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        alpha: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: cosine ANN  +  BM25/ts_rank, fused with RRF.

        This delivers significantly better recall than pure semantic search:
        - Semantic catches paraphrase / conceptual matches.
        - Keyword catches exact terminology (parcel numbers, plan codes, names).
        - RRF combines rank positions rather than raw scores, making the
          fusion robust regardless of score distribution differences.

        Args:
            alpha: Weight of semantic component in RRF fusion.
                   0.0 = keyword only, 1.0 = semantic only, 0.6 = default.

        Returns:
            Same structure as ``search()`` with an additional ``hybrid_score`` key.
        """
        if not query:
            return []
        if collection not in ALL_COLLECTIONS:
            raise ValueError(f"Unknown collection '{collection}'")

        q_emb = self._to_vec(self.embedding_provider.embed_query(query))
        where_sql, where_params = self._build_where(collection, where)

        # For the kw CTE: extend the WHERE clause with the full-text filter
        # (we can't add a second WHERE keyword; we append AND instead)
        kw_where_sql = where_sql + " AND content_tsv @@ plainto_tsquery('english', %s)"

        # Candidates pool: 4× top_k so the final merge has room
        candidates = top_k * 4

        # RRF constant – 60 is the standard value from the literature
        rrf_k = 60

        sql = f"""
        WITH sem AS (
            SELECT id::text                                        AS id,
                   content,
                   metadata,
                   1 - (embedding <=> %s::vector)                  AS sem_score,
                   ROW_NUMBER() OVER (
                       ORDER BY embedding <=> %s::vector
                   )                                               AS rk_sem
            FROM vector_documents
            {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        ),
        kw AS (
            SELECT id::text                                        AS id,
                   content,
                   metadata,
                   ts_rank(content_tsv,
                       plainto_tsquery('english', %s))             AS kw_score,
                   ROW_NUMBER() OVER (
                       ORDER BY ts_rank(content_tsv,
                           plainto_tsquery('english', %s)) DESC
                   )                                               AS rk_kw
            FROM vector_documents
            {kw_where_sql}
            LIMIT %s
        ),
        merged AS (
            SELECT
                COALESCE(s.id,      k.id)       AS id,
                COALESCE(s.content, k.content)  AS content,
                COALESCE(s.metadata,k.metadata) AS metadata,
                COALESCE(s.sem_score, 0.0)      AS sem_score,
                COALESCE(k.kw_score,  0.0)      AS kw_score,
                {alpha}  * (1.0 / ({rrf_k} + COALESCE(s.rk_sem, 100000)))
              + {1-alpha} * (1.0 / ({rrf_k} + COALESCE(k.rk_kw,  100000)))
                                                AS rrf_score
            FROM sem s
            FULL OUTER JOIN kw k ON s.id = k.id
        )
        SELECT id, content, metadata, sem_score, kw_score, rrf_score
        FROM merged
        ORDER BY rrf_score DESC
        LIMIT %s
        """

        # Parameter order (must match %s placeholders above):
        # sem CTE:  q_emb(cos1) q_emb(cos2) *where_params q_emb(cos3) candidates
        # kw  CTE:  query(rank) query(order) *where_params query(tsquery) candidates
        # outer:    top_k
        params = (
            q_emb, q_emb,                         # sem cosine distance ×2
            *where_params,                         # sem WHERE conditions
            q_emb, candidates,                     # sem ORDER BY + LIMIT
            query, query,                          # kw ts_rank ×2
            *where_params,                         # kw WHERE conditions
            query, candidates,                     # kw tsquery filter + LIMIT
            top_k,                                 # final LIMIT
        )

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "content": row[1],
                "metadata": row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}"),
                "score": float(row[5]),           # rrf_score is the primary rank
                "sem_score": float(row[3]),
                "kw_score": float(row[4]),
            }
            for row in rows
        ]

    def search_all_collections(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search all collections and return results grouped by collection."""
        return {
            col: self.search(query, collection=col, top_k=top_k, where=where)
            for col in ALL_COLLECTIONS
        }

    # ------------------------------------------------------------------
    # CONVERSATION OPERATIONS
    # ------------------------------------------------------------------

    def add_conversation(
        self,
        role: str,
        content: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a single conversation turn in the conversations collection."""
        meta = dict(metadata or {})
        meta.update(
            {
                "role": role,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        return self.add_text(
            text=content,
            metadata=meta,
            collection=COLLECTION_CONVERSATIONS,
        )

    def get_recent_conversations(
        self,
        session_id: str,
        limit: int = 10,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent conversation turns for a session.
        Ordered oldest-first so they can be appended to a prompt directly.
        """
        where_clauses = ["collection = %s", "session_id = %s"]
        params: List[Any] = [COLLECTION_CONVERSATIONS, session_id]

        if role:
            where_clauses.append("metadata->>'role' = %s")
            params.append(role)

        sql = (
            "SELECT id::text, content, metadata, created_at "
            "FROM vector_documents "
            f"WHERE {' AND '.join(where_clauses)} "
            "ORDER BY created_at DESC "
            "LIMIT %s"
        )
        params.append(limit)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        # Return oldest-first for prompt assembly
        results = [
            {
                "id": r[0],
                "content": r[1],
                "metadata": r[2] if isinstance(r[2], dict) else json.loads(r[2] or "{}"),
                "created_at": r[3].isoformat() if hasattr(r[3], "isoformat") else str(r[3]),
            }
            for r in reversed(rows)
        ]
        return results

    # ------------------------------------------------------------------
    # GEOSPATIAL COORDINATE OPERATIONS  (NEW – PostGIS-backed)
    # ------------------------------------------------------------------

    def add_coordinates(
        self,
        coordinates: List[Dict[str, Any]],
        crs_name: str = "UTM",
        session_id: Optional[str] = None,
    ) -> List[str]:
        """
        Store survey coordinate points in the PostGIS-backed table.

        Each coordinate dict must have ``easting`` and ``northing`` keys.
        Optional keys: ``label``, ``elevation``, ``epsg_code``,
        ``lat`` / ``lon`` (WGS84 for the geometry column).

        Example:
            store.add_coordinates([
                {"label": "SC/Q 573", "easting": 292286.622, "northing": 536678.345,
                 "lat": 4.8500, "lon": 7.0100, "epsg_code": 32632}
            ], crs_name="UTM Zone 32N", session_id=session_id)
        """
        if not coordinates:
            return []

        now = datetime.utcnow().isoformat()
        rows = []
        for pt in coordinates:
            e = pt.get("easting") or pt.get("e") or pt.get("x")
            n = pt.get("northing") or pt.get("n") or pt.get("y")
            if e is None or n is None:
                continue
            cid = str(uuid.uuid4())
            lat = pt.get("lat")
            lon = pt.get("lon")
            geom_wkt = (
                f"ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326)"
                if lat is not None and lon is not None
                else "NULL"
            )
            meta = {
                "added_at": now,
                **{k: v for k, v in pt.items()
                   if k not in ("easting", "northing", "elevation", "lat", "lon", "e", "n", "x", "y")}
            }
            rows.append(
                (cid, pt.get("label"), float(e), float(n),
                 float(pt["elevation"]) if pt.get("elevation") is not None else None,
                 crs_name,
                 int(pt["epsg_code"]) if pt.get("epsg_code") is not None else None,
                 json.dumps(meta),
                 session_id or pt.get("session_id"),
                 lat, lon)
            )

        if not rows:
            return []

        with self._conn() as conn:
            with conn.cursor() as cur:
                for r in rows:
                    (cid, label, e, n, elev, crs, epsg,
                     meta_json, sid, lat, lon) = r
                    if lat is not None and lon is not None:
                        cur.execute(
                            """
                            INSERT INTO survey_coordinates
                              (id, label, easting, northing, elevation,
                               crs_name, epsg_code, metadata, session_id, geom)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s,
                                    ST_SetSRID(ST_MakePoint(%s, %s), 4326))
                            """,
                            (cid, label, e, n, elev, crs, epsg,
                             meta_json, sid, lon, lat),
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO survey_coordinates
                              (id, label, easting, northing, elevation,
                               crs_name, epsg_code, metadata, session_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                            """,
                            (cid, label, e, n, elev, crs, epsg, meta_json, sid),
                        )

        logger.info(f"✓ Stored {len(rows)} survey coordinates (CRS: {crs_name})")

        # Also store in vector collection for semantic recall
        text_docs = [
            {
                "content": (
                    f"Survey point {r[1] or r[0]}: E={r[2]:.3f} N={r[3]:.3f} "
                    f"CRS={crs_name}"
                ),
                "metadata": {
                    "type": "coordinate",
                    "label": r[1],
                    "easting": r[2],
                    "northing": r[3],
                    "crs_name": crs_name,
                    "epsg_code": r[6],
                    "session_id": r[8],
                },
                "id": r[0],
                "session_id": r[8],
            }
            for r in rows
        ]
        self.add_documents(text_docs, collection=COLLECTION_COORDINATES)
        return [r[0] for r in rows]

    def find_nearby_coordinates(
        self,
        lat: float,
        lon: float,
        radius_m: float = 1000.0,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        PostGIS proximity search: return survey points within ``radius_m``
        metres of the given WGS84 latitude/longitude.

        Requires that coordinates were inserted with lat/lon values.
        """
        sql = """
            SELECT id::text, label, easting, northing, elevation,
                   crs_name, epsg_code, metadata,
                   ST_Distance(
                       geom::geography,
                       ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
                   ) AS dist_m
            FROM survey_coordinates
            WHERE ST_DWithin(
                geom::geography,
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                %s
            )
            ORDER BY dist_m
            LIMIT %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (lon, lat, lon, lat, radius_m, limit))
                rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "label": r[1],
                "easting": r[2],
                "northing": r[3],
                "elevation": r[4],
                "crs_name": r[5],
                "epsg_code": r[6],
                "metadata": r[7] if isinstance(r[7], dict) else json.loads(r[7] or "{}"),
                "distance_m": float(r[8]),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # AUTOCAD ENTITIES  (unchanged high-level wrapper)
    # ------------------------------------------------------------------

    def add_autocad_entities(
        self,
        entities: List[Dict[str, Any]],
        drawing_name: str = "",
        session_id: Optional[str] = None,
    ) -> List[str]:
        """
        Store AutoCAD entity metadata in the drawings collection.
        Each entity dict should have at least ``type`` and ``layer`` keys.
        """
        if not entities:
            return []
        docs = []
        for ent in entities:
            parts = [
                f"AutoCAD entity type={ent.get('type', 'unknown')}",
                f"layer={ent.get('layer', 'unknown')}",
            ]
            if ent.get("text_content"):
                parts.append(f"text={ent['text_content']}")
            if ent.get("color"):
                parts.append(f"color={ent['color']}")
            content = " ".join(parts)
            meta = {
                "type": "autocad_entity",
                "drawing_name": drawing_name,
                "session_id": session_id,
                **{k: v for k, v in ent.items()
                   if k not in ("text_content",) and isinstance(v, (str, int, float, bool, type(None)))},
            }
            docs.append(
                {"content": content, "metadata": meta, "session_id": session_id}
            )
        return self.add_documents(docs, collection=COLLECTION_DRAWINGS)

    # ------------------------------------------------------------------
    # MANAGEMENT OPERATIONS
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return per-collection counts and total document count."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT collection, COUNT(*) FROM vector_documents GROUP BY collection"
                )
                counts = {r[0]: int(r[1]) for r in cur.fetchall()}
                cur.execute("SELECT COUNT(*) FROM survey_coordinates")
                geo_count = int(cur.fetchone()[0])

        total = sum(counts.values())
        return {
            "total_documents": total,
            "geospatial_points": geo_count,
            "collections": {c: counts.get(c, 0) for c in ALL_COLLECTIONS},
            "provider_type": self.provider_type,
            "embedding_dimension": self._expected_dim,
            "backend": "postgresql+pgvector",
        }

    def clear_collection(self, collection: str) -> None:
        """Delete all documents in a collection."""
        if collection not in ALL_COLLECTIONS:
            raise ValueError(f"Unknown collection '{collection}'")
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM vector_documents WHERE collection = %s",
                    (collection,),
                )
        logger.info(f"Cleared collection '{collection}'")

    def delete_document(self, doc_id: str, collection: str = COLLECTION_DOCUMENTS) -> bool:
        """Delete a single document by ID. Returns True if a row was deleted."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM vector_documents WHERE id = %s AND collection = %s",
                    (doc_id, collection),
                )
                deleted = cur.rowcount > 0
        return deleted

    def get_document(
        self, doc_id: str, collection: str = COLLECTION_DOCUMENTS
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a single document by ID."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id::text, content, metadata "
                    "FROM vector_documents WHERE id = %s AND collection = %s",
                    (doc_id, collection),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "content": row[1],
            "metadata": row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}"),
        }

    def reset(self) -> None:
        """Delete ALL documents from ALL collections. Irreversible."""
        logger.warning("VectorStore.reset() – deleting all vector_documents and survey_coordinates.")
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM vector_documents")
                cur.execute("DELETE FROM survey_coordinates")
        logger.info("✓ Vector store reset complete.")

    def close(self) -> None:
        """Return all connections to the pool and close it."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_where(
        collection: str,
        where: Optional[Dict[str, Any]],
    ) -> Tuple[str, tuple]:
        """
        Build a WHERE clause for collection + optional metadata filters.
        Returns (sql_fragment, params_tuple).

        ``where`` keys are matched against the ``metadata`` JSONB column
        using the containment operator @>.  The special key ``session_id``
        is also routed to the dedicated index column for performance.
        """
        clauses = ["collection = %s"]
        params: List[Any] = [collection]

        if where:
            for k, v in where.items():
                if k == "session_id":
                    clauses.append("session_id = %s")
                    params.append(str(v))
                else:
                    # JSONB containment: metadata @> '{"key": "value"}'
                    clauses.append("metadata @> %s::jsonb")
                    params.append(json.dumps({k: v}))

        return ("WHERE " + " AND ".join(clauses)), tuple(params)


# ==============================================================================
# FACTORY FUNCTION  (preserved for backward compat)
# ==============================================================================

def create_vector_store(
    db_url: Optional[str] = None,
    embedding_provider: str = "local",
    openai_api_key: Optional[str] = None,
    local_model_name: str = DEFAULT_LOCAL_MODEL,
    openai_model_name: str = DEFAULT_OPENAI_MODEL,
    **kwargs: Any,
) -> VectorStore:
    """
    Factory that creates a ``VectorStore`` connected to PostgreSQL.

    ``persist_directory`` from the old ChromaDB signature is accepted as
    a kwarg and silently ignored to avoid breaking call-sites.
    """
    kwargs.pop("persist_directory", None)
    return VectorStore(
        db_url=db_url,
        embedding_provider=embedding_provider,  # type: ignore[arg-type]
        openai_api_key=openai_api_key,
        local_model_name=local_model_name,
        openai_model_name=openai_model_name,
    )


__all__ = [
    "VectorStore",
    "create_vector_store",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "COLLECTION_DOCUMENTS",
    "COLLECTION_DRAWINGS",
    "COLLECTION_COORDINATES",
    "COLLECTION_CONVERSATIONS",
    "ALL_COLLECTIONS",
]

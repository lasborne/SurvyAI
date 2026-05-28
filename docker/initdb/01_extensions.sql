-- SurvyAI DB initialisation: install required PostgreSQL extensions.
-- This script runs ONCE on first container startup (via docker-entrypoint-initdb.d).
-- Extensions are created in the 'survyai' database by the superuser.

\connect survyai

CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector ANN search
CREATE EXTENSION IF NOT EXISTS postgis;     -- geospatial types + functions
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- trigram similarity for keyword search
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; -- gen_random_uuid() fallback

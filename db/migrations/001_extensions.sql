-- Enables pgvector's VECTOR column type and <=> distance operators.
-- IDs are generated as uuid4() in Python at write time (ingestion / agent code),
-- not DB-side, so we don't need pgcrypto or uuid-ossp -- keeps ID generation
-- visible in the Python code you're meant to be reading, not hidden in SQL.
CREATE EXTENSION IF NOT EXISTS vector;

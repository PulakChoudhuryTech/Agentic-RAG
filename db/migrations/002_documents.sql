-- One row per source document (a whole markdown file in data/raw_docs/**).
CREATE TABLE documents (
    id              UUID PRIMARY KEY,
    source_path     TEXT NOT NULL,
    title           TEXT NOT NULL,
    category        TEXT NOT NULL CHECK (category IN ('hr', 'it', 'travel')),
    department      TEXT,
    country         TEXT,              -- e.g. 'IN' when the doc has a country-specific section
    effective_date  DATE,
    raw_text        TEXT NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

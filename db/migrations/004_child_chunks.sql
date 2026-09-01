-- Child chunks are the "search" unit: small (~250-400 chars) so each embedding
-- represents one focused idea, which makes vector similarity much sharper than
-- embedding a whole parent chunk would. Both vector search and keyword (FTS)
-- search run against this table; results are expanded back to their parent
-- chunk (see backend/app/rag/parent_child.py) before being sent to the LLM.
CREATE TABLE child_chunks (
    id                UUID PRIMARY KEY,
    parent_chunk_id   UUID NOT NULL REFERENCES parent_chunks(id) ON DELETE CASCADE,
    document_id       UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,  -- denormalized so we can filter by category/country without a join
    chunk_index       INT NOT NULL,
    content           TEXT NOT NULL,
    char_start        INT NOT NULL,
    char_end          INT NOT NULL,
    token_count       INT NOT NULL,
    embedding         VECTOR(768),          -- dimension must match EMBEDDING_DIMS in config; Gemini gemini-embedding-001 requested at output_dimensionality=768 (see rag/embeddings.py)
    embedding_model    TEXT NOT NULL,        -- e.g. 'gemini:models/gemini-embedding-001' -- recorded so you can tell which rows came from which model if you ever swap providers
    content_tsv       TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    metadata          JSONB NOT NULL DEFAULT '{}',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

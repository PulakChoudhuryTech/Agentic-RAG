-- Parent chunks are the "context" unit: large enough (~1500-2000 chars) to give
-- the LLM a coherent section of the document. We never embed or search these
-- directly -- see child_chunks. Parent/child retrieval = search small, return big.
CREATE TABLE parent_chunks (
    id              UUID PRIMARY KEY,
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INT NOT NULL,
    content         TEXT NOT NULL,
    char_start      INT NOT NULL,
    char_end        INT NOT NULL,
    token_count     INT NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

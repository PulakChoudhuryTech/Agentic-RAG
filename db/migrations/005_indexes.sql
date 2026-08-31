-- HNSW: an approximate-nearest-neighbor graph index over the embedding column.
-- vector_cosine_ops means the index is built to accelerate `<=>` (cosine distance).
-- m = max connections per graph node, ef_construction = candidate list size while
-- building the graph. Both are the classic HNSW trade-off knobs: higher = better
-- recall, slower build, more memory. See docs/concepts/hnsw.md for a full writeup,
-- including the honest caveat that at this project's corpus size (a handful of
-- documents) HNSW behaves almost identically to an exact scan -- it's here so you
-- can see the mechanism, not because it's needed at this scale.
CREATE INDEX child_chunks_hnsw_idx ON child_chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- GIN index over the generated tsvector column, used by PostgreSQL Full Text
-- Search (see backend/app/rag/keyword_search.py). This is NOT a BM25 index --
-- see docs/concepts/fts_vs_bm25.md for exactly what's different.
CREATE INDEX child_chunks_fts_idx ON child_chunks USING GIN (content_tsv);

CREATE INDEX child_chunks_document_id_idx ON child_chunks (document_id);
CREATE INDEX parent_chunks_document_id_idx ON parent_chunks (document_id);
CREATE INDEX documents_category_idx ON documents (category);

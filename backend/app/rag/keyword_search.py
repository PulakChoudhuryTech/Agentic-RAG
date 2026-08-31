"""
PostgreSQL Full Text Search (FTS) keyword search.

*** This is NOT BM25. *** It is close enough in spirit ("rank documents by
keyword match quality") that it's a reasonable stand-in for a learning
project, and it's what lets us stay on PostgreSQL instead of adding
Elasticsearch/OpenSearch. But concretely, `ts_rank_cd` differs from BM25 in
three ways that matter:

  1. No inverse document frequency (IDF) term -- a rare word and a common
     word in the corpus are NOT weighted differently.
  2. No document-length normalization -- BM25 penalizes matches in long
     documents relative to short ones; ts_rank_cd does not.
  3. No term-frequency saturation curve -- BM25's score flattens out as a
     term repeats many times; ts_rank_cd's coverage-based score behaves
     differently.

Full write-up: docs/concepts/fts_vs_bm25.md. `keyword_search()` below is the
one function `rag/hybrid_search.py` depends on -- swapping this file for a
real BM25 engine (e.g. an Elasticsearch/OpenSearch or a Python bm25
library) later would not require changing any other file, as long as the
replacement returns the same `KeywordHit` shape.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from backend.app.db import get_connection
from backend.app.rag.vector_store import build_filter_clause


@dataclass
class KeywordHit:
    child_chunk_id: uuid.UUID
    parent_chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    ts_rank: float
    metadata: dict[str, Any] = field(default_factory=dict)


def keyword_search(
    query: str,
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[KeywordHit]:
    filter_sql, filter_params = build_filter_clause(filters)

    # plainto_tsquery: turns the raw query string into an AND-of-terms
    # tsquery (safe against user input, unlike websearch/raw tsquery syntax).
    # ts_rank_cd ("cover density") additionally rewards matched terms
    # appearing close together, which plain ts_rank ignores.
    sql = f"""
        SELECT
            c.id AS child_chunk_id,
            c.parent_chunk_id,
            c.document_id,
            c.content,
            ts_rank_cd(c.content_tsv, plainto_tsquery('english', %s)) AS ts_rank,
            c.metadata
        FROM child_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.content_tsv @@ plainto_tsquery('english', %s){filter_sql}
        ORDER BY ts_rank DESC
        LIMIT %s
    """
    params = [query, query, *filter_params, top_k]

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [
        KeywordHit(
            child_chunk_id=row[0],
            parent_chunk_id=row[1],
            document_id=row[2],
            content=row[3],
            ts_rank=float(row[4]),
            metadata=row[5] or {},
        )
        for row in rows
    ]

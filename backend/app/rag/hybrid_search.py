"""
Hybrid search: combine vector search results and keyword (FTS) search results
using Reciprocal Rank Fusion (RRF).

Why RRF instead of combining raw scores: cosine similarity (roughly 0..1) and
ts_rank_cd (an unbounded, corpus-dependent float) live on completely
different scales, so averaging or summing them directly would be meaningless
-- whichever score happens to have bigger numbers would dominate. RRF sidesteps
this by throwing away the raw scores and using only each result's RANK
(1st, 2nd, 3rd, ...) within its own list:

    RRF_score(doc) = sum over each ranked list L containing doc of  1 / (k + rank_in_L)

`k` (default 60, the value used in the original RRF paper) softens the
influence of rank 1 vs rank 2 -- without it, a document ranked #1 in one list
and absent from the other could swamp a document ranked #2 in BOTH lists.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from backend.app.rag.keyword_search import KeywordHit
from backend.app.rag.vector_store import VectorHit


@dataclass
class FusedHit:
    child_chunk_id: uuid.UUID
    parent_chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    rrf_score: float
    vector_rank: int | None  # 1-based rank in the vector list, or None if absent
    keyword_rank: int | None
    vector_similarity: float | None
    keyword_ts_rank: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


def reciprocal_rank_fusion(
    vector_hits: list[VectorHit],
    keyword_hits: list[KeywordHit],
    k: int = 60,
) -> list[FusedHit]:
    scores: dict[uuid.UUID, float] = {}
    vector_rank_by_id: dict[uuid.UUID, int] = {}
    keyword_rank_by_id: dict[uuid.UUID, int] = {}
    vector_sim_by_id: dict[uuid.UUID, float] = {}
    keyword_rank_score_by_id: dict[uuid.UUID, float] = {}
    hit_lookup: dict[uuid.UUID, VectorHit | KeywordHit] = {}

    for rank, hit in enumerate(vector_hits, start=1):
        scores[hit.child_chunk_id] = scores.get(hit.child_chunk_id, 0.0) + 1.0 / (k + rank)
        vector_rank_by_id[hit.child_chunk_id] = rank
        vector_sim_by_id[hit.child_chunk_id] = hit.cosine_similarity
        hit_lookup[hit.child_chunk_id] = hit

    for rank, hit in enumerate(keyword_hits, start=1):
        scores[hit.child_chunk_id] = scores.get(hit.child_chunk_id, 0.0) + 1.0 / (k + rank)
        keyword_rank_by_id[hit.child_chunk_id] = rank
        keyword_rank_score_by_id[hit.child_chunk_id] = hit.ts_rank
        hit_lookup.setdefault(hit.child_chunk_id, hit)

    fused = [
        FusedHit(
            child_chunk_id=chunk_id,
            parent_chunk_id=hit_lookup[chunk_id].parent_chunk_id,
            document_id=hit_lookup[chunk_id].document_id,
            content=hit_lookup[chunk_id].content,
            rrf_score=score,
            vector_rank=vector_rank_by_id.get(chunk_id),
            keyword_rank=keyword_rank_by_id.get(chunk_id),
            vector_similarity=vector_sim_by_id.get(chunk_id),
            keyword_ts_rank=keyword_rank_score_by_id.get(chunk_id),
            metadata=hit_lookup[chunk_id].metadata,
        )
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda h: h.rrf_score, reverse=True)
    return fused

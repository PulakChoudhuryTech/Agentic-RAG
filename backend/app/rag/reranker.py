"""
Cross-encoder reranking.

Vector/keyword/RRF all score a query against a chunk WITHOUT letting the two
texts attend to each other -- an embedding is computed once per chunk,
independent of the query. A cross-encoder instead takes the (query, chunk)
PAIR as a single input and outputs one relevance score directly, so it can
notice things bi-encoder similarity misses (e.g. negation, "not eligible for
X" vs "eligible for X" can embed as very similar vectors but a cross-encoder
scores them very differently). This is why reranking is a separate, slower,
"do less but do it more carefully" pass applied only to the RRF-fused
top candidates, not the whole corpus.

Default model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80MB) is small for a
fast first run. Swap to a full BGE reranker (`BAAI/bge-reranker-base`,
~1.1GB) via RERANKER_MODEL_NAME in .env to compare quality/latency.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sentence_transformers import CrossEncoder

from backend.app.rag.hybrid_search import FusedHit


@dataclass
class RerankedHit:
    child_chunk_id: uuid.UUID
    parent_chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    rerank_score: float
    pre_rerank_rank: int  # rank in the input list (post-RRF), 1-based
    metadata: dict[str, Any] = field(default_factory=dict)


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[FusedHit], top_n: int) -> list[RerankedHit]:
        if not candidates:
            return []

        pairs = [(query, c.content) for c in candidates]
        raw_scores = self._model.predict(pairs)

        reranked = [
            RerankedHit(
                child_chunk_id=c.child_chunk_id,
                parent_chunk_id=c.parent_chunk_id,
                document_id=c.document_id,
                content=c.content,
                rerank_score=float(score),
                pre_rerank_rank=i + 1,
                metadata=c.metadata,
            )
            for i, (c, score) in enumerate(zip(candidates, raw_scores))
        ]
        reranked.sort(key=lambda h: h.rerank_score, reverse=True)
        return reranked[:top_n]


_reranker_instance: CrossEncoderReranker | None = None


def get_reranker(model_name: str) -> CrossEncoderReranker:
    """Cached singleton -- loading the cross-encoder model is the slow part
    (first call downloads it), so we don't want to reload it per request."""
    global _reranker_instance
    if _reranker_instance is None or _reranker_instance.model_name != model_name:
        _reranker_instance = CrossEncoderReranker(model_name)
    return _reranker_instance

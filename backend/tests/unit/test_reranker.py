"""
Unit-tests the reranking/sorting logic in CrossEncoderReranker.rerank()
WITHOUT loading a real cross-encoder model (that would download ~80MB+ on
first run and hit no network in a unit test) -- we monkeypatch the
underlying `_model.predict` call with a fake scoring function instead.
"""

import uuid

from backend.app.rag.hybrid_search import FusedHit
from backend.app.rag.reranker import CrossEncoderReranker


def _fused_hit(content, rrf_score=0.5):
    return FusedHit(
        child_chunk_id=uuid.uuid4(),
        parent_chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=content,
        rrf_score=rrf_score,
        vector_rank=1,
        keyword_rank=1,
        vector_similarity=0.5,
        keyword_ts_rank=0.5,
    )


class _FakeCrossEncoder:
    def __init__(self, score_by_content: dict[str, float]):
        self._scores = score_by_content

    def predict(self, pairs):
        return [self._scores[content] for _query, content in pairs]


def _build_reranker(score_by_content: dict[str, float]) -> CrossEncoderReranker:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)  # skip __init__ (no real model load)
    reranker.model_name = "fake-model"
    reranker._model = _FakeCrossEncoder(score_by_content)
    return reranker


def test_rerank_reorders_by_cross_encoder_score():
    candidates = [_fused_hit("low relevance chunk"), _fused_hit("high relevance chunk")]
    reranker = _build_reranker({"low relevance chunk": 0.1, "high relevance chunk": 0.9})

    reranked = reranker.rerank("some query", candidates, top_n=2)

    assert [r.content for r in reranked] == ["high relevance chunk", "low relevance chunk"]
    assert reranked[0].rerank_score == 0.9
    assert reranked[0].pre_rerank_rank == 2  # it was second in the input order


def test_rerank_respects_top_n():
    candidates = [_fused_hit(f"chunk {i}") for i in range(5)]
    reranker = _build_reranker({f"chunk {i}": float(i) for i in range(5)})

    reranked = reranker.rerank("query", candidates, top_n=2)

    assert len(reranked) == 2
    assert reranked[0].content == "chunk 4"
    assert reranked[1].content == "chunk 3"


def test_rerank_empty_candidates_returns_empty():
    reranker = _build_reranker({})
    assert reranker.rerank("query", [], top_n=5) == []

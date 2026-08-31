import uuid

from backend.app.rag.hybrid_search import reciprocal_rank_fusion
from backend.app.rag.keyword_search import KeywordHit
from backend.app.rag.vector_store import VectorHit


def _vector_hit(chunk_id, similarity):
    return VectorHit(
        child_chunk_id=chunk_id,
        parent_chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="vector content",
        cosine_similarity=similarity,
    )


def _keyword_hit(chunk_id, ts_rank):
    return KeywordHit(
        child_chunk_id=chunk_id,
        parent_chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="keyword content",
        ts_rank=ts_rank,
    )


def test_rrf_score_matches_formula_for_a_result_in_both_lists():
    chunk_id = uuid.uuid4()
    vector_hits = [_vector_hit(chunk_id, 0.9)]
    keyword_hits = [_keyword_hit(chunk_id, 0.5)]

    fused = reciprocal_rank_fusion(vector_hits, keyword_hits, k=60)

    assert len(fused) == 1
    expected_score = 1 / (60 + 1) + 1 / (60 + 1)  # rank 1 in both lists
    assert abs(fused[0].rrf_score - expected_score) < 1e-9
    assert fused[0].vector_rank == 1
    assert fused[0].keyword_rank == 1


def test_rrf_result_present_in_both_lists_outranks_result_in_only_one():
    in_both = uuid.uuid4()
    only_vector = uuid.uuid4()

    vector_hits = [_vector_hit(only_vector, 0.99), _vector_hit(in_both, 0.5)]
    keyword_hits = [_keyword_hit(in_both, 0.8)]

    fused = reciprocal_rank_fusion(vector_hits, keyword_hits, k=60)
    fused_by_id = {f.child_chunk_id: f for f in fused}

    # only_vector is rank 1 in vector list (higher raw similarity) but absent
    # from keyword list; in_both is rank 2 in vector but rank 1 in keyword.
    # RRF should still rank in_both higher because it appears in both lists.
    assert fused_by_id[in_both].rrf_score > fused_by_id[only_vector].rrf_score
    assert fused[0].child_chunk_id == in_both


def test_rrf_preserves_all_unique_results_from_both_lists():
    ids = [uuid.uuid4() for _ in range(4)]
    vector_hits = [_vector_hit(ids[0], 0.9), _vector_hit(ids[1], 0.8)]
    keyword_hits = [_keyword_hit(ids[2], 0.7), _keyword_hit(ids[3], 0.6)]

    fused = reciprocal_rank_fusion(vector_hits, keyword_hits, k=60)

    assert {f.child_chunk_id for f in fused} == set(ids)


def test_rrf_empty_inputs_returns_empty():
    assert reciprocal_rank_fusion([], [], k=60) == []

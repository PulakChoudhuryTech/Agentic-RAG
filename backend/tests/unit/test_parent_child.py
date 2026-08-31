"""
Unit-tests expand_to_parents()'s dedup/ordering logic without a real
database -- monkeypatches get_connection() and get_parent_chunk() with an
in-memory fake.
"""

import uuid
from contextlib import contextmanager

import backend.app.rag.parent_child as parent_child_module
from backend.app.rag.reranker import RerankedHit


def _reranked_hit(child_id, parent_id, rerank_score):
    return RerankedHit(
        child_chunk_id=child_id,
        parent_chunk_id=parent_id,
        document_id=uuid.uuid4(),
        content="child content",
        rerank_score=rerank_score,
        pre_rerank_rank=1,
    )


def test_expand_to_parents_dedupes_children_from_same_parent(monkeypatch):
    parent_a = uuid.uuid4()
    parent_b = uuid.uuid4()
    child_1, child_2, child_3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    fake_parents = {
        parent_a: {
            "parent_chunk_id": parent_a,
            "document_id": uuid.uuid4(),
            "content": "Parent A content",
            "metadata": {},
            "document_title": "Doc A",
            "source_path": "hr/doc_a.md",
            "category": "hr",
        },
        parent_b: {
            "parent_chunk_id": parent_b,
            "document_id": uuid.uuid4(),
            "content": "Parent B content",
            "metadata": {},
            "document_title": "Doc B",
            "source_path": "hr/doc_b.md",
            "category": "hr",
        },
    }

    @contextmanager
    def fake_get_connection():
        yield None  # conn is unused by our fake get_parent_chunk

    monkeypatch.setattr(parent_child_module, "get_connection", fake_get_connection)
    monkeypatch.setattr(parent_child_module, "get_parent_chunk", lambda conn, pid: fake_parents.get(pid))

    # child_1 and child_2 both belong to parent_a; child_3 belongs to parent_b
    reranked_hits = [
        _reranked_hit(child_1, parent_a, rerank_score=0.9),
        _reranked_hit(child_3, parent_b, rerank_score=0.8),
        _reranked_hit(child_2, parent_a, rerank_score=0.7),
    ]

    contexts = parent_child_module.expand_to_parents(reranked_hits)

    assert len(contexts) == 2  # deduped to one context per unique parent
    assert contexts[0].parent_chunk_id == parent_a  # first-seen order preserved
    assert contexts[0].matched_child_chunk_ids == [child_1, child_2]
    assert contexts[0].best_rerank_score == 0.9  # max across its matched children
    assert contexts[1].parent_chunk_id == parent_b


def test_expand_to_parents_skips_missing_parent_rows(monkeypatch):
    @contextmanager
    def fake_get_connection():
        yield None

    monkeypatch.setattr(parent_child_module, "get_connection", fake_get_connection)
    monkeypatch.setattr(parent_child_module, "get_parent_chunk", lambda conn, pid: None)

    contexts = parent_child_module.expand_to_parents([_reranked_hit(uuid.uuid4(), uuid.uuid4(), 0.5)])
    assert contexts == []

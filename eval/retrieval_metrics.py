"""
Retrieval-ranking metrics, implemented directly (no RAGAS/library) because
these are simple, well-known formulas worth seeing plainly rather than
importing. All four take a ranked list of retrieved document identifiers
and a set of identifiers considered relevant (binary relevance -- a document
either matches the golden answer's expected source or it doesn't).

RAGAS is used instead for `faithfulness`/`answer_relevancy` (eval/ragas_eval.py),
which need an LLM judge and aren't simple closed-form formulas.
"""

from __future__ import annotations

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """MRR is the mean of this value across queries -- this function
    computes it for a single query's ranked list."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary relevance NDCG: DCG using relevance in {0, 1}, normalized by
    the ideal DCG (all relevant docs ranked first, up to k)."""
    top_k = retrieved[:k]
    dcg = sum(
        (1.0 if doc_id in relevant else 0.0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg

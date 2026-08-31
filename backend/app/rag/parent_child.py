"""
Parent/child expansion: given the final ranked list of CHILD chunks (the
small, precise units we searched over), fetch each one's PARENT chunk (the
larger unit with more surrounding context) to actually send to the LLM.

This is the second half of the parent/child retrieval strategy described in
chunking.py: search small for precision, return big for context. Multiple
child chunks from the same parent are deduplicated so the LLM doesn't see
the same parent content twice.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from backend.app.db import get_connection
from backend.app.rag.reranker import RerankedHit
from backend.app.rag.vector_store import get_parent_chunk


@dataclass
class ParentContext:
    parent_chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    document_title: str
    source_path: str
    category: str
    matched_child_chunk_ids: list[uuid.UUID] = field(default_factory=list)
    best_rerank_score: float = 0.0


def expand_to_parents(reranked_hits: list[RerankedHit]) -> list[ParentContext]:
    contexts: dict[uuid.UUID, ParentContext] = {}

    with get_connection() as conn:
        for hit in reranked_hits:
            if hit.parent_chunk_id in contexts:
                ctx = contexts[hit.parent_chunk_id]
                ctx.matched_child_chunk_ids.append(hit.child_chunk_id)
                ctx.best_rerank_score = max(ctx.best_rerank_score, hit.rerank_score)
                continue

            parent = get_parent_chunk(conn, hit.parent_chunk_id)
            if parent is None:
                continue  # parent row missing (shouldn't happen with FK constraints, but stay defensive)

            contexts[hit.parent_chunk_id] = ParentContext(
                parent_chunk_id=parent["parent_chunk_id"],
                document_id=parent["document_id"],
                content=parent["content"],
                document_title=parent["document_title"],
                source_path=parent["source_path"],
                category=parent["category"],
                matched_child_chunk_ids=[hit.child_chunk_id],
                best_rerank_score=hit.rerank_score,
            )

    # preserve the original ranking order (best reranked child first)
    ordered_parent_ids: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for hit in reranked_hits:
        if hit.parent_chunk_id not in seen:
            ordered_parent_ids.append(hit.parent_chunk_id)
            seen.add(hit.parent_chunk_id)

    return [contexts[pid] for pid in ordered_parent_ids if pid in contexts]

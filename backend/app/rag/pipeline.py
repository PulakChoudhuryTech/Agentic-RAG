"""
run_rag_pipeline(): the single orchestrator for the whole RAG pipeline.

This is the file to read first to understand the system. It's a straight
top-to-bottom function with explicit `if settings.enable_x:` branches for
every optional feature -- no strategy pattern, no plugin registry, no
decorator stack. Every stage appends to `trace` so the caller (a FastAPI
route or a LangGraph node) can hand the whole thing back to the user.

Pipeline stages, matching the diagram in the project README:

    query
      -> metadata filter (rule-based, rag/metadata_router.py)
      -> query rewrite (optional, rag/query_rewrite.py)
      -> embedding (rag/embeddings.py) [possibly of a HyDE passage instead of the query]
      -> query expansion (optional, rag/query_expansion.py) -> multiple vector searches, merged
      -> vector search (rag/vector_store.py, pgvector + HNSW)
      -> keyword search (optional if hybrid disabled, rag/keyword_search.py, PostgreSQL FTS)
      -> RRF fusion (rag/hybrid_search.py) [skipped if hybrid search disabled]
      -> cross-encoder reranking (optional, rag/reranker.py)
      -> parent/child expansion (optional, rag/parent_child.py)
      -> context assembly (rag/context_assembly.py)
      -> Gemini answer generation (llm/gemini_client.py)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from backend.app.config import Settings
from backend.app.llm.gemini_client import generate_text
from backend.app.llm.prompts import RAG_ANSWER_PROMPT
from backend.app.rag.chunking import estimate_token_count
from backend.app.rag.context_assembly import Citation, assemble_context
from backend.app.rag.embeddings import get_embedding_provider
from backend.app.rag.hybrid_search import FusedHit, reciprocal_rank_fusion
from backend.app.rag.hyde import generate_hypothetical_document
from backend.app.rag.keyword_search import keyword_search
from backend.app.rag.metadata_router import detect_metadata_filters
from backend.app.rag.parent_child import ParentContext, expand_to_parents
from backend.app.rag.query_expansion import expand_query
from backend.app.rag.query_rewrite import rewrite_query
from backend.app.rag.reranker import RerankedHit, get_reranker
from backend.app.rag.vector_store import VectorHit, get_document_info, vector_search
from backend.app.db import get_connection
from backend.app.trace.trace import Trace


@dataclass
class RagResult:
    answer: str
    citations: list[Citation]
    context: str
    trace: list[dict]


def _dedupe_vector_hits(hits: list[VectorHit]) -> list[VectorHit]:
    """When query expansion runs multiple vector searches, keep the
    highest-similarity occurrence of each child chunk."""
    best: dict[uuid.UUID, VectorHit] = {}
    for hit in hits:
        existing = best.get(hit.child_chunk_id)
        if existing is None or hit.cosine_similarity > existing.cosine_similarity:
            best[hit.child_chunk_id] = hit
    return sorted(best.values(), key=lambda h: h.cosine_similarity, reverse=True)


def _vector_hits_to_fused(hits: list[VectorHit]) -> list[FusedHit]:
    """Adapter used when hybrid search is disabled: wrap vector-only results
    in the same FusedHit shape the reranker expects, so downstream code
    doesn't need an "if hybrid search was used" branch of its own."""
    return [
        FusedHit(
            child_chunk_id=h.child_chunk_id,
            parent_chunk_id=h.parent_chunk_id,
            document_id=h.document_id,
            content=h.content,
            rrf_score=h.cosine_similarity,
            vector_rank=rank,
            keyword_rank=None,
            vector_similarity=h.cosine_similarity,
            keyword_ts_rank=None,
            metadata=h.metadata,
        )
        for rank, h in enumerate(hits, start=1)
    ]


def _fused_to_reranked(hits: list[FusedHit], top_n: int) -> list[RerankedHit]:
    """Adapter used when reranking is disabled: take the top N fused/vector
    results as-is, in their existing order."""
    return [
        RerankedHit(
            child_chunk_id=h.child_chunk_id,
            parent_chunk_id=h.parent_chunk_id,
            document_id=h.document_id,
            content=h.content,
            rerank_score=h.rrf_score,
            pre_rerank_rank=rank,
            metadata=h.metadata,
        )
        for rank, h in enumerate(hits[:top_n], start=1)
    ]


def run_rag_pipeline(
    query: str,
    settings: Settings,
    trace: Trace,
    filters: dict[str, str] | None = None,
) -> RagResult:
    trace.step("pipeline", "start", {"query": query})

    # ---- 1. metadata routing ----
    detected_filters = detect_metadata_filters(query) if settings.enable_metadata_routing else {}
    # Explicit filters passed in by the caller (e.g. the agent graph, which
    # may already know the employee's country) take precedence over the
    # rule-based guess.
    effective_filters = {**detected_filters, **(filters or {})}
    trace.step(
        "metadata_router",
        "filters_applied" if effective_filters else "no_filters_matched",
        {"detected": detected_filters, "caller_provided": filters or {}, "effective": effective_filters},
    )

    # ---- 2. query rewrite ----
    search_query = query
    if settings.enable_query_rewrite:
        search_query = rewrite_query(query, settings)
        trace.step("query_rewrite", "rewritten", {"original": query, "rewritten": search_query})

    embedding_provider = get_embedding_provider(settings)

    # ---- 3. embedding target: HyDE (optional) or the (possibly rewritten) query itself ----
    if settings.enable_hyde:
        hypothetical_doc = generate_hypothetical_document(search_query, settings)
        trace.step("hyde", "hypothetical_document_generated", {"passage": hypothetical_doc})
        embed_target = hypothetical_doc
    else:
        embed_target = search_query

    query_embedding = embedding_provider.embed_query(embed_target)
    trace.step(
        "embeddings",
        "query_embedded",
        {"model": embedding_provider.model_name, "dims": len(query_embedding)},
    )

    # ---- 4. query expansion (optional): search with multiple phrasings, merge ----
    search_queries = [search_query]
    if settings.enable_query_expansion:
        variants = expand_query(search_query, settings)
        trace.step("query_expansion", "variants_generated", {"variants": variants})
        search_queries = list({search_query, *variants})

    # ---- 5. vector search (one call per search query, merged) ----
    # query_embedding (computed above, possibly via HyDE) is reused for the
    # main search_query; expansion variants are embedded directly (HyDE is
    # not re-applied per variant, to keep this at one LLM call regardless of
    # how many expansion variants are generated).
    all_vector_hits: list[VectorHit] = []
    for sq in search_queries:
        sq_embedding = query_embedding if sq == search_query else embedding_provider.embed_query(sq)
        hits = vector_search(sq_embedding, top_k=settings.vector_top_k, filters=effective_filters)
        all_vector_hits.extend(hits)
    vector_hits = _dedupe_vector_hits(all_vector_hits)
    trace.step(
        "vector_search",
        "results",
        {
            "top_k": settings.vector_top_k,
            "count": len(vector_hits),
            "top_results": [
                {"content": h.content[:120], "cosine_similarity": round(h.cosine_similarity, 4)}
                for h in vector_hits[:5]
            ],
        },
    )

    # ---- 6. keyword search + RRF fusion (or a pass-through adapter) ----
    if settings.enable_hybrid_search:
        keyword_hits = keyword_search(search_query, top_k=settings.keyword_top_k, filters=effective_filters)
        trace.step(
            "keyword_search",
            "results",
            {
                "top_k": settings.keyword_top_k,
                "count": len(keyword_hits),
                "top_results": [
                    {"content": h.content[:120], "ts_rank": round(h.ts_rank, 4)} for h in keyword_hits[:5]
                ],
            },
        )
        fused_hits = reciprocal_rank_fusion(vector_hits, keyword_hits, k=settings.rrf_k)
        trace.step(
            "hybrid_search",
            "rrf_fused",
            {
                "rrf_k": settings.rrf_k,
                "count": len(fused_hits),
                "top_results": [
                    {
                        "content": h.content[:120],
                        "rrf_score": round(h.rrf_score, 5),
                        "vector_rank": h.vector_rank,
                        "keyword_rank": h.keyword_rank,
                    }
                    for h in fused_hits[:5]
                ],
            },
        )
    else:
        fused_hits = _vector_hits_to_fused(vector_hits)
        trace.step("hybrid_search", "skipped_hybrid_disabled", {"count": len(fused_hits)})

    # ---- 7. reranking ----
    rerank_candidate_pool = fused_hits[: max(settings.vector_top_k, settings.keyword_top_k)]
    if settings.enable_reranking:
        reranker = get_reranker(settings.reranker_model_name)
        reranked_hits = reranker.rerank(query, rerank_candidate_pool, top_n=settings.rerank_top_n)
        trace.step(
            "reranker",
            "reranked",
            {
                "model": settings.reranker_model_name,
                "candidates_considered": len(rerank_candidate_pool),
                "results": [
                    {
                        "content": h.content[:120],
                        "rerank_score": round(h.rerank_score, 4),
                        "pre_rerank_rank": h.pre_rerank_rank,
                    }
                    for h in reranked_hits
                ],
            },
        )
    else:
        reranked_hits = _fused_to_reranked(rerank_candidate_pool, top_n=settings.rerank_top_n)
        trace.step("reranker", "skipped_reranking_disabled", {"count": len(reranked_hits)})

    # ---- 8. parent/child expansion ----
    if settings.enable_parent_child:
        parent_contexts = expand_to_parents(reranked_hits)
        trace.step(
            "parent_child",
            "expanded_to_parents",
            {
                "child_count": len(reranked_hits),
                "parent_count": len(parent_contexts),
                "parents": [{"title": p.document_title, "chars": len(p.content)} for p in parent_contexts],
            },
        )
    else:
        doc_info_cache: dict[uuid.UUID, dict] = {}
        with get_connection() as conn:
            for h in reranked_hits:
                if h.document_id not in doc_info_cache:
                    doc_info_cache[h.document_id] = get_document_info(conn, h.document_id) or {
                        "title": "Unknown document",
                        "source_path": "",
                        "category": "",
                    }
        parent_contexts = [
            ParentContext(
                parent_chunk_id=h.child_chunk_id,  # no real parent lookup; child chunk stands alone
                document_id=h.document_id,
                content=h.content,
                document_title=doc_info_cache[h.document_id]["title"],
                source_path=doc_info_cache[h.document_id]["source_path"],
                category=doc_info_cache[h.document_id]["category"],
                matched_child_chunk_ids=[h.child_chunk_id],
                best_rerank_score=h.rerank_score,
            )
            for h in reranked_hits
        ]
        trace.step("parent_child", "skipped_parent_child_disabled", {"count": len(parent_contexts)})

    # ---- 9. context assembly ----
    context_str, citations = assemble_context(parent_contexts, max_tokens=settings.context_max_tokens)
    trace.step(
        "context_assembly",
        "assembled",
        {"sources_used": len(citations), "estimated_tokens": estimate_token_count(context_str)},
    )

    # ---- 10. Gemini answer generation ----
    prompt = RAG_ANSWER_PROMPT.format(context=context_str or "(no relevant context found)", query=query)
    answer = generate_text(prompt, settings)
    trace.step("llm", "answer_generated", {"answer": answer})

    return RagResult(answer=answer, citations=citations, context=context_str, trace=trace.as_list())

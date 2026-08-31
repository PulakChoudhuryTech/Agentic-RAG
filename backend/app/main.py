"""FastAPI application entrypoint."""

from __future__ import annotations

import os

from fastapi import FastAPI

from backend.app.api.routes_chat import router as chat_router
from backend.app.api.routes_rag import router as rag_router
from backend.app.config import get_settings
from backend.app.logging_conf import configure_logging

settings = get_settings()
configure_logging(settings)

# Optional LangSmith tracing: only touches process env vars (which LangChain
# reads automatically) when explicitly enabled -- otherwise nothing related
# to LangSmith is configured or contacted at all.
if settings.langsmith_tracing:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key

app = FastAPI(
    title="Enterprise Employee AI Assistant",
    description="Local learning project: RAG + agentic AI over HR/IT/Travel docs and mock Workday/ServiceNow APIs.",
)

app.include_router(rag_router)
app.include_router(chat_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "debug_mode": settings.debug_mode}


@app.get("/config")
def get_config() -> dict:
    """Read-only view of the current feature flags/tuning, for the
    Streamlit sidebar. Flags are set via .env and take effect on backend
    restart -- there is no live toggle endpoint, to keep "what's currently
    running" unambiguous while you're comparing configurations."""
    return {
        "embedding_provider": settings.embedding_provider,
        "gemini_model": settings.gemini_model,
        "enable_query_rewrite": settings.enable_query_rewrite,
        "enable_query_expansion": settings.enable_query_expansion,
        "enable_hyde": settings.enable_hyde,
        "enable_hybrid_search": settings.enable_hybrid_search,
        "enable_reranking": settings.enable_reranking,
        "enable_parent_child": settings.enable_parent_child,
        "enable_metadata_routing": settings.enable_metadata_routing,
        "enable_embedding_intent_classifier": settings.enable_embedding_intent_classifier,
        "vector_top_k": settings.vector_top_k,
        "keyword_top_k": settings.keyword_top_k,
        "rrf_k": settings.rrf_k,
        "rerank_top_n": settings.rerank_top_n,
        "reranker_model_name": settings.reranker_model_name,
        "debug_mode": settings.debug_mode,
    }

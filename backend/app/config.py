"""
Central configuration + feature flags.

Every "advanced retrieval feature" the user might want to toggle on/off and
compare (query rewriting, query expansion, HyDE, hybrid search, reranking,
parent/child expansion, metadata routing, embedding-based intent
classification) is a plain boolean here, read once at startup. There is no
dynamic feature-flag service, no per-request override layer -- just a
Settings object you can read top to bottom to see exactly what's on.

(`?debug=true` on API requests can still force trace collection on for a
single request even when DEBUG_MODE=false in .env -- see api/routes_rag.py
and api/routes_chat.py.)
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- Gemini ----
    gemini_api_key: str
    # gemini-2.5-flash-lite, not gemini-2.5-flash: the full flash model's
    # free tier is easy to exhaust for this project (a multi-agent request
    # can chain 3-4+ generateContent calls -- supervisor routing, RAG
    # answer, tool-calling, combine); flash-lite has a separate, more
    # workable free-tier quota bucket and is plenty capable for this
    # project's routing/synthesis tasks.
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_embedding_model: str = "models/gemini-embedding-001"

    # ---- Embeddings ----
    embedding_provider: Literal["gemini", "local"] = "gemini"
    embedding_dims: int = 768

    # ---- Postgres ----
    database_url: str

    # ---- Mock enterprise APIs ----
    workday_api_url: str = "http://localhost:8001"
    servicenow_api_url: str = "http://localhost:8002"

    # ---- RAG feature flags ----
    enable_query_rewrite: bool = False
    enable_query_expansion: bool = False
    enable_hyde: bool = False
    enable_hybrid_search: bool = True
    enable_reranking: bool = True
    enable_parent_child: bool = True
    enable_metadata_routing: bool = True
    enable_embedding_intent_classifier: bool = True

    # ---- Retrieval tuning ----
    vector_top_k: int = 20
    keyword_top_k: int = 20
    rrf_k: int = 60
    rerank_top_n: int = 5
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    context_max_tokens: int = 3000

    # Chunking (also used by the ingestion CLI)
    parent_chunk_max_chars: int = 1800
    parent_chunk_overlap: int = 200
    child_chunk_max_chars: int = 350
    child_chunk_overlap: int = 50

    # ---- Debug / learning mode ----
    debug_mode: bool = True

    # ---- Optional LangSmith tracing ----
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "agentic-rag-learning"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Cached singleton so we parse .env once per process, not per request."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

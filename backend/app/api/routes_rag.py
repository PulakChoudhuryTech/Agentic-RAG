"""
POST /rag/query -- runs ONLY the RAG pipeline (no agent routing, no
Workday/ServiceNow tools). This is the endpoint to hit when you want to
inspect retrieval behavior in isolation: toggle flags in .env, send the
same query, and diff the trace. /chat (routes_chat.py) is the full agentic
endpoint that decides whether to use this pipeline at all.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.api.schemas import CitationModel, RagQueryRequest, RagQueryResponse, TraceStepModel
from backend.app.config import get_settings
from backend.app.rag.pipeline import run_rag_pipeline
from backend.app.trace.trace import Trace

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RagQueryResponse)
def query_rag(request: RagQueryRequest, debug: bool = Query(default=None)) -> RagQueryResponse:
    settings = get_settings()
    trace_enabled = settings.debug_mode if debug is None else debug
    trace = Trace(enabled=trace_enabled)

    filters = {k: v for k, v in {"category": request.category, "country": request.country}.items() if v}

    result = run_rag_pipeline(request.query, settings, trace, filters=filters or None)

    return RagQueryResponse(
        answer=result.answer,
        citations=[CitationModel(**c.__dict__) for c in result.citations],
        trace=[TraceStepModel(**s) for s in result.trace] if trace_enabled else None,
    )

"""
RAG agent: runs the full RAG pipeline (rag/pipeline.py) scoped to the
document category implied by the supervisor's route, and folds the
pipeline's own trace (query rewrite, vector/keyword search, RRF, reranking,
parent/child, context assembly, LLM answer) into the agent trace so it shows
up alongside the routing/tool-call steps in one combined view.
"""

from __future__ import annotations

from backend.app.agents.state import AgentState
from backend.app.config import Settings
from backend.app.rag.pipeline import run_rag_pipeline
from backend.app.trace.trace import Trace, new_trace_steps

# Which document category to scope retrieval to, per supervisor route.
# `combined` searches HR docs because every combined-route example in this
# project (e.g. parental leave eligibility) is an HR eligibility question --
# see docs/example_queries.md.
ROUTE_TO_CATEGORY = {
    "rag_hr": "hr",
    "rag_it": "it",
    "rag_travel": "travel",
    "rag_personal": "personal",
    "combined": "hr",
}


def rag_agent_node(state: AgentState, settings: Settings) -> dict:
    intent = state.get("intent") or "rag_hr"
    category = ROUTE_TO_CATEGORY.get(intent)

    filters = dict(state.get("metadata_filters") or {})
    if category:
        filters["category"] = category  # route-implied category overrides/confirms the rule-based guess

    trace = Trace(enabled=True)
    result = run_rag_pipeline(state["user_query"], settings, trace, filters=filters)

    combined_trace = new_trace_steps("rag_agent", "started", {"category": category, "filters": filters})
    combined_trace.extend(result.trace)

    return {
        "rag_result": {
            "answer": result.answer,
            "citations": [c.__dict__ for c in result.citations],
        },
        "citations": [c.__dict__ for c in result.citations],
        "trace": combined_trace,
    }

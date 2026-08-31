"""
POST /chat -- the full agentic endpoint: supervisor routing, tool calling,
conditional IT troubleshoot/ticket workflow, all via the LangGraph graph in
agents/graph.py. `session_id` doubles as the LangGraph thread_id, so
multi-turn state (conversation history, the IT troubleshoot/confirm
workflow) persists across calls with the same session_id.

GET /chat/{session_id}/history -- replays chat_messages from Postgres so
past sessions (and their per-turn debug traces) stay inspectable from the UI
even after the in-memory LangGraph checkpointer is gone (e.g. after a
backend restart).
"""

from __future__ import annotations

import json

from fastapi import APIRouter
from langchain_core.messages import HumanMessage

from backend.app.agents.graph import build_graph
from backend.app.api.schemas import (
    ChatHistoryMessage,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    CitationModel,
    TraceStepModel,
)
from backend.app.config import get_settings
from backend.app.db import get_connection

router = APIRouter(prefix="/chat", tags=["chat"])

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(get_settings())
    return _graph


def _save_turn(session_id: str, role: str, content: str, trace: list[dict] | None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, trace) VALUES (%s, %s, %s, %s)",
            (session_id, role, content, json.dumps(trace) if trace is not None else None),
        )
        conn.commit()


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()
    graph = _get_graph()

    _save_turn(request.session_id, "user", request.message, trace=None)

    config = {"configurable": {"thread_id": request.session_id}}
    initial_state = {
        "session_id": request.session_id,
        "messages": [HumanMessage(content=request.message)],
        "user_query": request.message,
        "employee_id": request.employee_id,
        "citations": [],
    }
    final_state = graph.invoke(initial_state, config=config)

    answer = final_state.get("final_answer") or "I don't have an answer for that."
    citations = final_state.get("citations") or []
    trace = final_state.get("trace") or []

    _save_turn(request.session_id, "assistant", answer, trace=trace if settings.debug_mode else None)

    return ChatResponse(
        session_id=request.session_id,
        answer=answer,
        citations=[CitationModel(**c) for c in citations],
        intent=final_state.get("intent"),
        trace=[TraceStepModel(**s) for s in trace] if settings.debug_mode else None,
    )


@router.get("/{session_id}/history", response_model=ChatHistoryResponse)
def get_history(session_id: str) -> ChatHistoryResponse:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content, trace FROM chat_messages WHERE session_id = %s ORDER BY created_at",
            (session_id,),
        ).fetchall()

    messages = [
        ChatHistoryMessage(
            role=row[0],
            content=row[1],
            trace=[TraceStepModel(**s) for s in row[2]] if row[2] else None,
        )
        for row in rows
    ]
    return ChatHistoryResponse(session_id=session_id, messages=messages)

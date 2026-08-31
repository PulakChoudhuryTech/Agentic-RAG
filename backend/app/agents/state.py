"""
Explicit LangGraph state schema.

Every field a node can read or write lives here, typed, in one place --
there is no implicit "just stuff whatever you want into a dict" state. Two
reducer fields use LangGraph's `Annotated[..., reducer]` mechanism:

  - `messages`: the standard `add_messages` reducer (append new messages to
    the running conversation, matching by message id for updates). This is
    the one LangGraph-provided helper this project leans on, because message
    history bookkeeping really is pure plumbing, not something worth
    hand-rolling.
  - `trace`: `operator.add` (list concatenation) -- each node returns a
    single-item list from `trace.new_trace_steps(...)` and LangGraph
    concatenates them into one ordered history automatically. See
    backend/app/trace/trace.py for why trace entries are plain dicts.

Every other field is plain "last write wins" (LangGraph's default), which is
what we want: only one node should ever set `intent`, for example.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]

    user_query: str
    employee_id: str | None  # who is asking -- used by the employee/IT agents to call Workday/ServiceNow

    # supervisor / routing
    intent: str | None  # final routing decision: rag_hr | rag_it | rag_travel | workday | servicenow | combined
    intent_confidence: float | None
    intent_scores: dict[str, float] | None  # embedding-classifier hint, shown alongside the LLM's final `intent`
    metadata_filters: dict[str, str] | None

    # per-agent results
    rag_result: dict[str, Any] | None
    workday_result: dict[str, Any] | None
    servicenow_ticket: dict[str, Any] | None
    troubleshooting_resolved: bool | None

    # output
    final_answer: str | None
    citations: list[dict[str, Any]]

    trace: Annotated[list[dict[str, Any]], operator.add]

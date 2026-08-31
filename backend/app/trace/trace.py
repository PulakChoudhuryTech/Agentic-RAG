"""
Debug trace mechanism.

This is the single mechanism used everywhere the user wants to "see exactly
what is happening": both the RAG pipeline (rag/pipeline.py) and the LangGraph
agent state (agents/state.py) accumulate the SAME shape of trace step:

    {"node": str, "event": str, "data": dict, "ts": "<iso8601>"}

- In the RAG pipeline, a `Trace` object is passed down explicitly through
  every function call and each stage calls `trace.step(...)`.
- In the LangGraph agent graph, nodes return trace steps under the `trace`
  key of AgentState, which uses an `operator.add` reducer to concatenate
  them automatically -- see `new_trace_steps()` below, which produces a
  plain list a node can return directly.

Keeping trace entries as plain dicts (not custom classes) means the whole
thing stays JSON-serializable for free: it can go straight into an API
response, a `chat_messages.trace JSONB` column, or LangGraph's checkpointer
without any custom serialization code.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("agentic_rag.trace")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Trace:
    """Accumulates ordered debug steps for a single RAG pipeline run."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.steps: list[dict[str, Any]] = []

    def step(self, node: str, event: str, data: dict[str, Any] | None = None) -> None:
        entry = {"node": node, "event": event, "data": data or {}, "ts": _now()}
        if self.enabled:
            self.steps.append(entry)
        logger.debug("[%s] %s: %s", node, event, data or {})

    def as_list(self) -> list[dict[str, Any]]:
        return self.steps


def new_trace_steps(node: str, event: str, data: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """
    Build a single-item trace list, for LangGraph nodes to return under the
    `trace` state key (see agents/state.py's `operator.add` reducer, which
    concatenates the lists returned by each node into one running history).
    """
    entry = {"node": node, "event": event, "data": data or {}, "ts": _now()}
    logger.debug("[%s] %s: %s", node, event, data or {})
    return [entry]

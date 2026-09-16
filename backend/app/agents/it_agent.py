"""
IT agent: implements the conditional "troubleshoot, then create a ticket
only if still unresolved" workflow (example query 5 in the project spec).

Conceptually this is a small subgraph (troubleshoot -> check resolved? ->
create ticket if not), but it's wired as two plain nodes plus one routing
function in the main graph (see agents/graph.py) rather than a nested
LangGraph StateGraph -- for two nodes, a real subgraph would add ceremony
without adding clarity.

This spans TWO conversation turns, using `troubleshooting_resolved` as the
piece of conversational state that carries the workflow across them:

  Turn 1 (troubleshooting_resolved is None -> "no attempt yet"):
    route_it_entry() sends the graph to it_troubleshoot_node, which runs
    the RAG pipeline against IT docs, answers with the troubleshooting
    steps, asks the employee to confirm whether that fixed it, and sets
    troubleshooting_resolved = False ("given, but not yet confirmed").

  Turn 2 (troubleshooting_resolved is already False -> "awaiting confirmation"):
    route_it_entry() instead sends the graph to it_check_resolution_node,
    which asks the LLM whether the employee's new message indicates the
    issue is RESOLVED or still UNRESOLVED. If unresolved, it calls the
    create_ticket tool (via the same tool-calling loop as employee_agent.py)
    with a summary of the issue and the steps already tried.

This relies on LangGraph's checkpointer (MemorySaver, keyed by
session_id -- see agents/graph.py) persisting `rag_result` and
`troubleshooting_resolved` across turns for the same session.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.agents.state import AgentState
from backend.app.agents.tool_calling import run_tool_agent
from backend.app.agents.tools_servicenow import build_servicenow_tools
from backend.app.config import Settings
from backend.app.llm.gemini_client import generate_text, get_chat_model
from backend.app.llm.prompts import TROUBLESHOOTING_RESOLVED_PROMPT
from backend.app.rag.pipeline import run_rag_pipeline
from backend.app.skills.registry import get_skill
from backend.app.trace.trace import Trace, new_trace_steps

CREATE_TICKET_SYSTEM_PROMPT = """You are the IT Support assistant. The \
employee's issue could not be resolved with self-service troubleshooting. \
Create an IT support ticket summarizing the issue and the troubleshooting \
steps already attempted, choosing an appropriate category and urgency."""


def route_it_entry(state: AgentState) -> Literal["it_troubleshoot", "it_check_resolution"]:
    return "it_check_resolution" if state.get("troubleshooting_resolved") is False else "it_troubleshoot"


def it_troubleshoot_node(state: AgentState, settings: Settings) -> dict:
    filters = dict(state.get("metadata_filters") or {})
    filters["category"] = "it"

    # Same rag_it skill guidance rag_agent_node uses for the standalone
    # rag_it route -- this node runs the identical IT-scoped RAG pipeline,
    # just as the first turn of the troubleshoot/ticket workflow.
    skill = get_skill("rag_it")
    skill_guidance = skill.body if skill else ""

    trace = Trace(enabled=True)
    result = run_rag_pipeline(state["user_query"], settings, trace, filters=filters, skill_guidance=skill_guidance)

    answer = (
        f"{result.answer}\n\n"
        "Did that resolve the issue? Let me know and I'll open an IT support ticket if not."
    )

    combined_trace = new_trace_steps("it_troubleshoot", "troubleshoot_steps_given", {"issue": state["user_query"]})
    combined_trace.extend(result.trace)

    return {
        "rag_result": {"answer": result.answer, "citations": [c.__dict__ for c in result.citations]},
        "citations": [c.__dict__ for c in result.citations],
        "troubleshooting_resolved": False,  # steps given, awaiting the employee's confirmation next turn
        "final_answer": answer,
        "trace": combined_trace,
    }


def it_check_resolution_node(state: AgentState, settings: Settings) -> dict:
    prior_steps = (state.get("rag_result") or {}).get("answer", "")
    trace = new_trace_steps(
        "it_check_resolution",
        "checking_resolution",
        {"prior_steps": prior_steps[:200], "latest_message": state["user_query"]},
    )

    verdict_prompt = TROUBLESHOOTING_RESOLVED_PROMPT.format(
        issue=state["user_query"], steps=prior_steps, latest_message=state["user_query"]
    )
    verdict = generate_text(verdict_prompt, settings).strip().upper()
    resolved = verdict.startswith("RESOLVED")
    trace.extend(
        new_trace_steps("it_check_resolution", "resolution_verdict", {"verdict": verdict, "resolved": resolved})
    )

    if resolved:
        trace.extend(
            new_trace_steps("it_check_resolution", "next_node", {"next": "final_answer_node (resolved, no ticket)"})
        )
        return {
            "troubleshooting_resolved": True,
            "final_answer": "Glad that resolved it! Let me know if anything else comes up.",
            "trace": trace,
        }

    employee_id = state.get("employee_id") or "unknown"
    tools = build_servicenow_tools(employee_id, settings)
    llm = get_chat_model(settings)
    messages = [
        SystemMessage(content=CREATE_TICKET_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Issue: {state['user_query']}\nTroubleshooting steps already tried:\n{prior_steps}"
        ),
    ]
    ticket_answer, tool_trace = run_tool_agent(llm, tools, messages, "it_check_resolution")
    trace.extend(tool_trace)
    trace.extend(
        new_trace_steps("it_check_resolution", "next_node", {"next": "final_answer_node (ticket created)"})
    )

    return {
        "troubleshooting_resolved": False,
        "servicenow_ticket": {"answer": ticket_answer},
        "final_answer": ticket_answer,
        "trace": trace,
    }

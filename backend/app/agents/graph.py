"""
build_graph(): wires all the nodes above into the LangGraph StateGraph.

Read this file top to bottom alongside docs/architecture.md's diagram --
every edge below corresponds to an arrow in that diagram. Routing decisions
are all plain Python functions returning a node-name string (LangGraph's
"conditional edge" mechanism); none of the branching logic lives inside the
node functions themselves except where a node's OWN output determines what
happens next (it_troubleshoot/it_check_resolution, which is intrinsic to
that workflow -- see agents/it_agent.py's module docstring).

Node reuse: `rag_agent` and `employee_agent` are each used by TWO routes
(pure rag_x / combined, and workday / combined respectively) -- the
conditional edges AFTER those nodes check `state["intent"]` (unchanged
since the supervisor set it) to decide whether to continue to combine_node
or go straight to final_answer_node.
"""

from __future__ import annotations

from functools import partial
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.app.agents.employee_agent import employee_agent_node
from backend.app.agents.it_agent import it_check_resolution_node, it_troubleshoot_node, route_it_entry
from backend.app.agents.rag_agent import rag_agent_node
from backend.app.agents.state import AgentState
from backend.app.agents.supervisor import supervisor_node
from backend.app.config import Settings
from backend.app.llm.gemini_client import generate_text
from backend.app.llm.prompts import COMBINE_PROMPT
from backend.app.trace.trace import new_trace_steps


def combine_node(state: AgentState, settings: Settings) -> dict:
    workday_data = (state.get("workday_result") or {}).get("answer", "(no employee data retrieved)")
    policy_context = (state.get("rag_result") or {}).get("answer", "(no policy context retrieved)")

    prompt = COMBINE_PROMPT.format(query=state["user_query"], workday_data=workday_data, policy_context=policy_context)
    answer = generate_text(prompt, settings)

    trace = new_trace_steps(
        "combine",
        "combined_answer_generated",
        {"workday_data": workday_data, "policy_context": policy_context[:300], "answer": answer},
    )
    return {"final_answer": answer, "trace": trace}


def final_answer_node(state: AgentState) -> dict:
    """Terminal node for every route. Most routes (it_agent, combine) already
    set `final_answer` themselves; this node fills it in for the two routes
    that don't (pure RAG, pure workday) and always emits the closing trace
    step so every path ends with the same visible "final answer" marker."""
    final_answer = state.get("final_answer")
    citations = state.get("citations") or []

    if final_answer is None:
        if state.get("intent") == "workday":
            final_answer = (state.get("workday_result") or {}).get("answer", "I couldn't retrieve that information.")
        else:
            final_answer = (state.get("rag_result") or {}).get("answer", "I don't have an answer for that.")

    trace = new_trace_steps("final_answer", "answer_returned", {"answer": final_answer})
    return {"final_answer": final_answer, "citations": citations, "trace": trace}


def route_after_supervisor(
    state: AgentState,
) -> Literal["rag_agent", "employee_agent", "it_troubleshoot", "it_check_resolution"]:
    intent = state.get("intent")
    if intent in ("rag_hr", "rag_it", "rag_travel", "rag_personal"):
        return "rag_agent"
    if intent in ("workday", "combined"):
        return "employee_agent"
    if intent == "servicenow_troubleshoot":
        return route_it_entry(state)
    return "rag_agent"  # safe fallback


def route_after_employee_agent(state: AgentState) -> Literal["rag_agent", "final_answer"]:
    return "rag_agent" if state.get("intent") == "combined" else "final_answer"


def route_after_rag_agent(state: AgentState) -> Literal["combine", "final_answer"]:
    return "combine" if state.get("intent") == "combined" else "final_answer"


def build_graph(settings: Settings):
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", partial(supervisor_node, settings=settings))
    graph.add_node("rag_agent", partial(rag_agent_node, settings=settings))
    graph.add_node("employee_agent", partial(employee_agent_node, settings=settings))
    graph.add_node("combine", partial(combine_node, settings=settings))
    graph.add_node("it_troubleshoot", partial(it_troubleshoot_node, settings=settings))
    graph.add_node("it_check_resolution", partial(it_check_resolution_node, settings=settings))
    graph.add_node("final_answer", final_answer_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "rag_agent": "rag_agent",
            "employee_agent": "employee_agent",
            "it_troubleshoot": "it_troubleshoot",
            "it_check_resolution": "it_check_resolution",
        },
    )
    graph.add_conditional_edges(
        "employee_agent", route_after_employee_agent, {"rag_agent": "rag_agent", "final_answer": "final_answer"}
    )
    graph.add_conditional_edges(
        "rag_agent", route_after_rag_agent, {"combine": "combine", "final_answer": "final_answer"}
    )
    graph.add_edge("combine", "final_answer")
    graph.add_edge("it_troubleshoot", "final_answer")
    graph.add_edge("it_check_resolution", "final_answer")
    graph.add_edge("final_answer", END)

    # MemorySaver: in-process, in-memory checkpointer keyed by thread_id
    # (we use session_id as thread_id -- see routes_chat.py). This is what
    # gives the it_agent's troubleshoot/confirm workflow its "conversational
    # state" across turns, and is the ONE other LangGraph-provided helper
    # this project relies on besides `add_messages` (see agents/state.py).
    # It's process-memory only: state is lost on restart. A real deployment
    # would swap this for `SqliteSaver`/`PostgresSaver` -- the trace/state
    # design (plain dicts, JSON-serializable) was chosen specifically so
    # that swap wouldn't require touching any node code.
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

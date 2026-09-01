"""
Employee agent: answers questions about the current employee's own Workday
data (leave balance, benefits, profile) via genuine LLM tool-calling -- the
model decides which of get_employee/get_leave_balance/get_benefits to call,
not an if/else dispatcher matching keywords. See agents/tools_workday.py for
why employee_id is bound via closure rather than an LLM-supplied argument.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.agents.state import AgentState
from backend.app.agents.tool_calling import run_tool_agent
from backend.app.agents.tools_workday import build_workday_tools
from backend.app.config import Settings
from backend.app.llm.gemini_client import get_chat_model
from backend.app.trace.trace import new_trace_steps

SYSTEM_PROMPT = """You are the Employee Data assistant. Use the available \
tools to look up the current employee's own Workday data and answer their \
question concisely and factually. Only call the tools you actually need.

Some questions (e.g. "am I eligible for X", "do I qualify for Y") depend on \
BOTH the employee's Workday data AND a company policy document you don't \
have access to here -- a separate step will combine your Workday findings \
with the relevant policy afterwards. For these, still call the tool(s) that \
would supply the relevant profile facts (e.g. get_employee for country, \
tenure, and employment type) and report those facts plainly. Do NOT decline \
to call a tool just because you can't determine the final policy outcome \
yourself -- gathering the facts is your job here, not judging eligibility."""


def _extract_employee_country(tool_trace: list[dict]) -> str | None:
    """If get_employee was called, pull the employee's country out of its
    result. Used below to enrich metadata_filters for the "combined" route,
    so rag_agent's search (which runs AFTER this node -- see
    route_after_employee_agent in agents/graph.py) can be scoped to the
    employee's own country even when the user's question never mentions it
    by name (e.g. "am I eligible for parental leave" vs. "...in India") --
    without this, the rule-based metadata_router (which only reads the raw
    user query) has nothing to detect a country from."""
    for step in tool_trace:
        if step["event"] == "tool_result" and step["data"]["tool"] == "get_employee":
            return step["data"]["result"].get("country")
    return None


def employee_agent_node(state: AgentState, settings: Settings) -> dict:
    employee_id = state.get("employee_id")
    trace = new_trace_steps("employee_agent", "started", {"employee_id": employee_id})

    if not employee_id:
        trace.extend(new_trace_steps("employee_agent", "error", {"reason": "no employee_id in session"}))
        return {
            "workday_result": {"error": "no employee_id set for this session"},
            "trace": trace,
        }

    tools = build_workday_tools(employee_id, settings)
    llm = get_chat_model(settings)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=state["user_query"])]

    answer, tool_trace = run_tool_agent(llm, tools, messages, "employee_agent")
    trace.extend(tool_trace)

    result: dict = {"workday_result": {"answer": answer}, "trace": trace}

    country = _extract_employee_country(tool_trace)
    if country:
        merged_filters = {**(state.get("metadata_filters") or {}), "country": country}
        result["metadata_filters"] = merged_filters
        trace.extend(
            new_trace_steps(
                "employee_agent", "metadata_filters_enriched", {"country": country, "filters": merged_filters}
            )
        )

    trace.extend(new_trace_steps("employee_agent", "finished", {"answer": answer}))
    return result

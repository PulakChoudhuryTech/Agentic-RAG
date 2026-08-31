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
question concisely and factually. Only call the tools you actually need."""


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
    trace.extend(new_trace_steps("employee_agent", "finished", {"answer": answer}))

    return {"workday_result": {"answer": answer}, "trace": trace}

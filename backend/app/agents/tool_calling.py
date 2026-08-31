"""
Shared LLM tool-calling loop, used by both employee_agent.py (Workday
tools) and it_agent.py (the create_ticket tool). Pulled out as one small
shared function specifically because both agents need the exact same loop
(bind tools -> invoke -> execute any tool calls -> feed results back ->
repeat until the model stops calling tools) and duplicating it would be the
kind of copy-paste this project wants to avoid -- not a speculative
abstraction, a concrete second caller.
"""

from __future__ import annotations

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import StructuredTool

from backend.app.trace.trace import new_trace_steps

MAX_TOOL_ITERATIONS = 4


def run_tool_agent(
    llm: BaseChatModel,
    tools: list[StructuredTool],
    messages: list[BaseMessage],
    node_name: str,
) -> tuple[str, list[dict]]:
    """Runs the bind-tools/invoke/execute loop until the model responds
    without requesting any more tool calls (or MAX_TOOL_ITERATIONS is hit).
    Returns (final_answer_text, trace_steps)."""
    trace_steps: list[dict] = []
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}
    current_messages = list(messages)

    for _ in range(MAX_TOOL_ITERATIONS):
        response = llm_with_tools.invoke(current_messages)
        current_messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return response.content, trace_steps

        for call in tool_calls:
            trace_steps.extend(
                new_trace_steps(node_name, "tool_call", {"tool": call["name"], "args": call["args"]})
            )
            tool = tools_by_name.get(call["name"])
            try:
                result = tool.invoke(call["args"]) if tool else {"error": f"unknown tool {call['name']}"}
            except Exception as exc:  # noqa: BLE001 - surface any tool failure back to the LLM as a tool result
                result = {"error": str(exc)}

            trace_steps.extend(new_trace_steps(node_name, "tool_result", {"tool": call["name"], "result": result}))
            current_messages.append(
                ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"])
            )

    # Model kept calling tools past the iteration cap -- return whatever
    # text (if any) came with its last response rather than looping forever.
    last = current_messages[-1]
    return getattr(last, "content", "") or "(reached tool-call limit without a final answer)", trace_steps

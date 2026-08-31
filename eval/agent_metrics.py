"""
Scripted (non-LLM-judged) agent evaluation: routing correctness, tool
selection, tool execution success, task completion, and (for the
conditional IT flow) whether the graph actually visited the nodes we
expect. Deliberately not RAGAS/LLM-judge here -- these are checks against
concrete, deterministic facts (which route/tool/node was used), not
subjective answer quality, so a simple assertion is both simpler and more
trustworthy than an LLM judge would be.

Runs the LangGraph graph directly in-process (not over HTTP) -- requires
Postgres + the Workday/ServiceNow mocks to be running (`make db-up`), but
not the FastAPI backend itself.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from backend.app.agents.graph import build_graph
from backend.app.config import get_settings
from eval.golden_dataset import by_type, load_golden_dataset


@dataclass
class AgentCaseResult:
    case_id: str
    routing_correct: bool
    tool_selection_correct: bool | None  # None when no tool was expected
    tool_execution_success: bool | None
    task_completed: bool
    node_sequence: list[str] = field(default_factory=list)
    notes: str = ""


def _node_sequence(trace: list[dict]) -> list[str]:
    sequence: list[str] = []
    for step in trace:
        if not sequence or sequence[-1] != step["node"]:
            sequence.append(step["node"])
    return sequence


def _tool_calls(trace: list[dict]) -> list[dict]:
    return [step for step in trace if step["event"] == "tool_call"]


def _tool_results(trace: list[dict]) -> list[dict]:
    return [step for step in trace if step["event"] == "tool_result"]


def _run_turn(graph, thread_id: str, message: str, employee_id: str | None) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    state = {
        "session_id": thread_id,
        "messages": [HumanMessage(content=message)],
        "user_query": message,
        "employee_id": employee_id,
        "citations": [],
    }
    return graph.invoke(state, config=config)


def evaluate_agent_case(graph, case: dict) -> AgentCaseResult:
    thread_id = str(uuid.uuid4())
    final_state = _run_turn(graph, thread_id, case["query"], case.get("employee_id"))
    trace = final_state.get("trace") or []

    routing_correct = final_state.get("intent") == case["expected_intent"]

    tool_selection_correct = None
    tool_execution_success = None
    if case.get("expected_tool"):
        calls = _tool_calls(trace)
        tool_selection_correct = any(c["data"]["tool"] == case["expected_tool"] for c in calls)
        results = _tool_results(trace)
        tool_execution_success = any(
            r["data"]["tool"] == case["expected_tool"] and "error" not in r["data"].get("result", {})
            for r in results
        )

    answer = (final_state.get("final_answer") or "").lower()
    keywords = case.get("expected_answer_keywords", [])
    task_completed = all(kw.lower() in answer for kw in keywords) if keywords else True

    return AgentCaseResult(
        case_id=case["id"],
        routing_correct=routing_correct,
        tool_selection_correct=tool_selection_correct,
        tool_execution_success=tool_execution_success,
        task_completed=task_completed,
        node_sequence=_node_sequence(trace),
    )


def evaluate_agent_conditional_case(graph, case: dict) -> list[AgentCaseResult]:
    thread_id = str(uuid.uuid4())
    results: list[AgentCaseResult] = []

    for i, turn in enumerate(case["turns"]):
        final_state = _run_turn(graph, thread_id, turn["query"], case.get("employee_id"))
        trace = final_state.get("trace") or []
        node_sequence = _node_sequence(trace)

        routing_correct = final_state.get("intent") == turn["expected_intent"]
        # a loose subsequence check: every expected node appears, in order (extra nodes in between are fine)
        expected_nodes = turn.get("expected_node_sequence", [])
        it = iter(node_sequence)
        sequence_correct = all(any(node == expected for node in it) for expected in expected_nodes)

        tool_selection_correct = None
        if turn.get("expected_tool"):
            calls = _tool_calls(trace)
            tool_selection_correct = any(c["data"]["tool"] == turn["expected_tool"] for c in calls)

        task_completed = sequence_correct
        if "expected_resolved" in turn:
            task_completed = task_completed and (final_state.get("troubleshooting_resolved") == turn["expected_resolved"])

        results.append(
            AgentCaseResult(
                case_id=f"{case['id']}-turn{i + 1}",
                routing_correct=routing_correct,
                tool_selection_correct=tool_selection_correct,
                tool_execution_success=tool_selection_correct,  # tool call presence stands in for execution success here
                task_completed=task_completed,
                node_sequence=node_sequence,
            )
        )

    return results


def run_agent_evaluation() -> list[AgentCaseResult]:
    settings = get_settings()
    graph = build_graph(settings)
    dataset = load_golden_dataset()

    all_results: list[AgentCaseResult] = []
    for case in by_type(dataset, "agent"):
        all_results.append(evaluate_agent_case(graph, case))
    for case in by_type(dataset, "agent_conditional"):
        all_results.extend(evaluate_agent_conditional_case(graph, case))

    return all_results

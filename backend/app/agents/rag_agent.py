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
from backend.app.skills.registry import get_skill, load_skills
from backend.app.trace.trace import Trace, new_trace_steps

# Which document category to scope retrieval to, per supervisor route --
# sourced from each skill's `category` field (skills/registry.py) rather than
# hardcoded here. `combined` maps to "hr" because every combined-route
# example in this project (e.g. parental leave eligibility) is an HR
# eligibility question -- see docs/example_queries.md and
# skills/combined/SKILL.md.
ROUTE_TO_CATEGORY = {skill.name: skill.category for skill in load_skills() if skill.category}


def rag_agent_node(state: AgentState, settings: Settings) -> dict:
    intent = state.get("intent") or "rag_hr"
    category = ROUTE_TO_CATEGORY.get(intent)

    filters = dict(state.get("metadata_filters") or {})
    if category:
        filters["category"] = category  # route-implied category overrides/confirms the rule-based guess

    # The triggered skill's body is domain-specific answer guidance (tone,
    # caveats) folded into the final prompt -- see RAG_ANSWER_PROMPT in
    # llm/prompts.py and docs/concepts/skills_pattern.md. Skills with no body
    # (e.g. workday, servicenow_troubleshoot) contribute nothing here.
    skill = get_skill(intent)
    skill_guidance = skill.body if skill else ""

    trace = Trace(enabled=True)
    result = run_rag_pipeline(state["user_query"], settings, trace, filters=filters, skill_guidance=skill_guidance)

    combined_trace = new_trace_steps(
        "rag_agent", "started", {"category": category, "filters": filters, "skill": intent}
    )
    combined_trace.extend(result.trace)

    return {
        "rag_result": {
            "answer": result.answer,
            "citations": [c.__dict__ for c in result.citations],
        },
        "citations": [c.__dict__ for c in result.citations],
        "trace": combined_trace,
    }

"""
Supervisor node: the LangGraph entry point that decides which agent handles
the query.

Two signals feed the routing decision, and BOTH are recorded in the trace so
you can compare them:

  1. (optional, ENABLE_EMBEDDING_INTENT_CLASSIFIER) An embedding-similarity
     classifier (rag/intent_classifier.py) -- cheap, fast, no LLM call,
     computed first and passed into the LLM prompt as a hint.
  2. The LLM itself, which makes the FINAL routing decision. It can agree or
     disagree with the embedding hint -- the hint is advisory, never
     authoritative, since a novel phrasing can fool nearest-neighbor
     similarity in ways an LLM reading the actual sentence won't be fooled by.

Also runs rule-based metadata routing (rag/metadata_router.py) so
`metadata_filters` is available to whichever agent runs next.

The route list itself (both the descriptions shown to the LLM below and
VALID_ROUTES) comes from the skill registry (skills/registry.py), not a
hardcoded prompt string -- see docs/concepts/skills_pattern.md. Adding a new
route means adding a `backend/app/skills/<name>/SKILL.md`, not editing this
file.
"""

from __future__ import annotations

from backend.app.agents.state import AgentState
from backend.app.config import Settings
from backend.app.llm.gemini_client import generate_text
from backend.app.llm.prompts import SUPERVISOR_ROUTING_PROMPT
from backend.app.rag.embeddings import get_embedding_provider
from backend.app.rag.intent_classifier import get_intent_classifier
from backend.app.rag.metadata_router import detect_metadata_filters
from backend.app.skills.registry import load_skills
from backend.app.trace.trace import new_trace_steps

VALID_ROUTES = {skill.name for skill in load_skills()}


def supervisor_node(state: AgentState, settings: Settings) -> dict:
    query = state["user_query"]
    trace: list[dict] = []

    intent_scores: dict[str, float] | None = None
    intent_hint_text = ""
    if settings.enable_embedding_intent_classifier:
        embedding_provider = get_embedding_provider(settings)
        classifier = get_intent_classifier(embedding_provider)
        embedding_intent, intent_scores = classifier.classify(query)
        trace.extend(
            new_trace_steps(
                "supervisor",
                "embedding_intent_classified",
                {"top_intent": embedding_intent, "scores": {k: round(v, 4) for k, v in intent_scores.items()}},
            )
        )
        intent_hint_text = (
            f"(A quick embedding-similarity check suggests '{embedding_intent}', "
            "but use your own judgment based on the full message.)\n"
        )

    route_descriptions = "\n".join(f"- {skill.name}: {skill.description}" for skill in load_skills())
    prompt = SUPERVISOR_ROUTING_PROMPT.format(
        route_descriptions=route_descriptions, intent_hint=intent_hint_text, query=query
    )
    raw_route = generate_text(prompt, settings).strip().lower()
    route = raw_route if raw_route in VALID_ROUTES else "rag_hr"  # safe default if the LLM returns something unexpected

    metadata_filters = detect_metadata_filters(query) if settings.enable_metadata_routing else {}

    trace.extend(
        new_trace_steps(
            "supervisor",
            "route_decided",
            {"llm_raw_output": raw_route, "final_route": route, "metadata_filters": metadata_filters},
        )
    )

    return {
        "intent": route,
        "intent_scores": intent_scores,
        "metadata_filters": metadata_filters,
        "trace": trace,
    }

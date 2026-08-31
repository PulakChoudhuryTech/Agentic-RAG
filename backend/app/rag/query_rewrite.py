"""
Query rewriting: ask the LLM to clean up the user's raw query (expand
pronouns/abbreviations, fix ambiguity) before it's used for search. Behind
ENABLE_QUERY_REWRITE because it costs one extra LLM call per query -- worth
toggling on/off to see if it actually changes retrieved results for a given
question.
"""

from __future__ import annotations

from backend.app.config import Settings
from backend.app.llm.gemini_client import generate_text
from backend.app.llm.prompts import QUERY_REWRITE_PROMPT


def rewrite_query(query: str, settings: Settings) -> str:
    prompt = QUERY_REWRITE_PROMPT.format(query=query)
    rewritten = generate_text(prompt, settings).strip()
    return rewritten or query

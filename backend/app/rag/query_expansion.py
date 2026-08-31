"""
Query expansion: generate several alternative phrasings of the query and
search with all of them, unioning the vector-search candidates before RRF.
This helps when the user's exact wording doesn't closely match the
document's wording, at the cost of N extra embedding calls + vector
searches per query. Behind ENABLE_QUERY_EXPANSION.
"""

from __future__ import annotations

from backend.app.config import Settings
from backend.app.llm.gemini_client import generate_text
from backend.app.llm.prompts import QUERY_EXPANSION_PROMPT

DEFAULT_NUM_VARIANTS = 3


def expand_query(query: str, settings: Settings, n: int = DEFAULT_NUM_VARIANTS) -> list[str]:
    prompt = QUERY_EXPANSION_PROMPT.format(query=query, n=n)
    raw = generate_text(prompt, settings).strip()
    variants = [line.strip("-•* ").strip() for line in raw.splitlines() if line.strip()]
    return variants[:n] or [query]

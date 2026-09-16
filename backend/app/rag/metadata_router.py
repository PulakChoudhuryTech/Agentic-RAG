"""
Metadata-based routing: a small, explicit keyword/regex rule table that maps
words in the query to structural filters (documents.category,
documents.country) applied to vector_search/keyword_search's WHERE clause.

This is deliberately a plain rule table, not a classifier -- it's meant to
be simple enough to read in ten seconds and predict exactly what it will do.
The known limitation is that it only catches queries using these specific
words; a paraphrase like "leave when I have a baby" (no "parental" or
"leave policy") won't trigger the hr/country rule. Upgrade path: replace the
body of `detect_metadata_filters()` with a call to an LLM or embedding
classifier -- the function signature (str -> dict) would stay the same, so
nothing else in the pipeline would need to change.
"""

from __future__ import annotations

import re

from backend.app.skills.registry import load_skills


def _build_category_keywords() -> dict[str, list[str]]:
    """CATEGORY_KEYWORDS is sourced from each skill's `keywords` (see
    skills/registry.py) rather than hardcoded here, so adding a category
    means adding a SKILL.md, not editing this file. Skills are sorted by
    `match_priority` (lower = checked first) -- rag_personal sets
    match_priority: 0 because its vocabulary (account/policy/booking terms)
    rarely collides with the company-policy categories, and a personal
    document is usually the more specific, more relevant match when it
    does (e.g. "insurance" alone vs. HR's more specific "health insurance"
    phrase)."""
    skills = sorted(
        (s for s in load_skills() if s.category and s.keywords),
        key=lambda s: s.match_priority,
    )
    return {s.category: s.keywords for s in skills}


CATEGORY_KEYWORDS: dict[str, list[str]] = _build_category_keywords()

# ISO country codes we know how to detect by name/adjective in the query.
# Not sourced from the skill registry: country detection is orthogonal to
# any one domain (hr, travel, and personal docs can all be country-specific).
COUNTRY_KEYWORDS: dict[str, str] = {
    "india": "IN",
    "indian": "IN",
    "united states": "US",
    "u.s.": "US",
    "usa": "US",
    "united kingdom": "GB",
    "uk": "GB",
    "germany": "DE",
    "german": "DE",
}


def detect_metadata_filters(query: str) -> dict[str, str]:
    lowered = query.lower()
    filters: dict[str, str] = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(kw)}\b", lowered) for kw in keywords):
            filters["category"] = category
            break  # first matching category wins; queries rarely span categories

    for phrase, code in COUNTRY_KEYWORDS.items():
        if phrase in lowered:
            filters["country"] = code
            break

    return filters

"""
HyDE (Hypothetical Document Embeddings): instead of embedding the user's raw
question, ask the LLM to write a plausible ANSWER passage first, then embed
*that* and use it for vector search. The intuition: policy documents read
like answers/statements, not like questions, so a hypothetical
answer-shaped passage can be closer in embedding space to the real document
chunks than the question itself is. This is a research technique with mixed
results in practice -- exactly the kind of thing this project exists to let
you toggle on/off and compare on real queries. Behind ENABLE_HYDE.
"""

from __future__ import annotations

from backend.app.config import Settings
from backend.app.llm.gemini_client import generate_text
from backend.app.llm.prompts import HYDE_PROMPT


def generate_hypothetical_document(query: str, settings: Settings) -> str:
    prompt = HYDE_PROMPT.format(query=query)
    return generate_text(prompt, settings).strip()

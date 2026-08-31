"""
Assemble the final LLM context from ranked parent chunks, and build the
citation list returned alongside the answer.

Token budget is enforced with the same word-count estimate used elsewhere in
this project (see chunking.estimate_token_count) -- approximate, not a real
tokenizer, but consistent and easy to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.rag.chunking import estimate_token_count
from backend.app.rag.parent_child import ParentContext


@dataclass
class Citation:
    document_title: str
    source_path: str
    category: str
    parent_chunk_id: str
    excerpt: str


def assemble_context(
    parent_contexts: list[ParentContext],
    max_tokens: int,
) -> tuple[str, list[Citation]]:
    sections: list[str] = []
    citations: list[Citation] = []
    used_tokens = 0

    for i, ctx in enumerate(parent_contexts, start=1):
        chunk_tokens = estimate_token_count(ctx.content)
        if used_tokens + chunk_tokens > max_tokens and sections:
            # keep at least one section even if it alone exceeds the budget
            break

        sections.append(f"[Source {i}: {ctx.document_title}]\n{ctx.content}")
        citations.append(
            Citation(
                document_title=ctx.document_title,
                source_path=ctx.source_path,
                category=ctx.category,
                parent_chunk_id=str(ctx.parent_chunk_id),
                excerpt=ctx.content[:200],
            )
        )
        used_tokens += chunk_tokens

    context_str = "\n\n---\n\n".join(sections)
    return context_str, citations

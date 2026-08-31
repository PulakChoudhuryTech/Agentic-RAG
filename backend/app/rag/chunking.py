"""
Chunking: turn a raw document into overlapping "parent" and "child" chunks.

This is the parent/child chunking strategy:
  - Parent chunks (~1800 chars) are big enough to give the LLM real context.
  - Child chunks (~350 chars) are small enough that a single embedding
    represents one focused idea, which makes vector similarity search much
    sharper than embedding a whole parent chunk would.

We search over child chunks, then expand results back to their parent chunk
before building the LLM context (see rag/parent_child.py). This file only
does the splitting -- it doesn't touch the database or embeddings.

The splitting strategy is intentionally simple (paragraph-aware sliding
window over plain text) rather than a "smart" recursive/semantic splitter,
so the whole algorithm fits in one screen and its behavior is predictable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParentChunk:
    chunk_index: int
    content: str
    char_start: int
    char_end: int
    token_count: int


@dataclass
class ChildChunk:
    chunk_index: int
    parent_chunk_index: int
    content: str
    char_start: int  # offset within the PARENT chunk's content, not the full document
    char_end: int
    token_count: int


def estimate_token_count(text: str) -> int:
    """Rough token estimate (word count). Good enough for trace/debug display
    and context-budget checks -- not meant to match a real tokenizer exactly."""
    return len(text.split())


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_parent(text: str, max_chars: int = 1800, overlap: int = 200) -> list[ParentChunk]:
    """
    Group paragraphs into parent chunks of up to `max_chars` characters.

    Algorithm: walk paragraphs in order, appending each to the current chunk
    until adding the next paragraph would exceed max_chars. Then close the
    chunk and start a new one that begins with the last `overlap` characters
    of the chunk just closed, so context isn't lost at a chunk boundary.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[ParentChunk] = []
    current = ""
    current_start = 0
    cursor = 0  # position in `text` we've consumed up to (approximate, paragraph granularity)

    for para in paragraphs:
        para_start = text.find(para, cursor)
        if para_start == -1:
            para_start = cursor
        cursor = para_start + len(para)

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > max_chars and current:
            # close out the current chunk
            char_end = current_start + len(current)
            chunks.append(
                ParentChunk(
                    chunk_index=len(chunks),
                    content=current,
                    char_start=current_start,
                    char_end=char_end,
                    token_count=estimate_token_count(current),
                )
            )
            # start next chunk with overlap carried over from the tail of this one
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
            current_start = char_end - len(tail)
        else:
            current = candidate

    if current:
        chunks.append(
            ParentChunk(
                chunk_index=len(chunks),
                content=current,
                char_start=current_start,
                char_end=current_start + len(current),
                token_count=estimate_token_count(current),
            )
        )

    return chunks


def chunk_child(parent_text: str, max_chars: int = 350, overlap: int = 50) -> list[ChildChunk]:
    """
    Split a parent chunk's text into small child chunks using a plain
    character sliding window (fixed step = max_chars - overlap).

    Unlike chunk_parent, this does not try to respect paragraph boundaries --
    child chunks exist purely to give the embedding model a small, focused
    span of text to search over.
    """
    if not parent_text:
        return []

    step = max(max_chars - overlap, 1)
    chunks: list[ChildChunk] = []
    start = 0
    index = 0

    while start < len(parent_text):
        end = min(start + max_chars, len(parent_text))
        content = parent_text[start:end].strip()
        if content:
            chunks.append(
                ChildChunk(
                    chunk_index=index,
                    parent_chunk_index=-1,  # filled in by the caller, which knows the parent
                    content=content,
                    char_start=start,
                    char_end=end,
                    token_count=estimate_token_count(content),
                )
            )
            index += 1
        if end == len(parent_text):
            break
        start += step

    return chunks

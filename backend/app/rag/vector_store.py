"""
pgvector storage + vector similarity search. Plain SQL via psycopg -- no
LangChain VectorStore wrapper -- so the actual query pgvector runs is right
here, not inside a third-party library.

Cosine similarity via pgvector: the `<=>` operator computes COSINE DISTANCE
(1 - cosine_similarity) between two vectors. `ORDER BY embedding <=> %s`
therefore sorts by ascending distance = descending similarity, which is
exactly nearest-neighbor search. We convert back to a similarity score
(`1 - distance`) in the SELECT list purely so the trace/UI can show a more
intuitive "0 to 1, higher is better" number.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from pgvector import Vector

from backend.app.db import get_connection


@dataclass
class VectorHit:
    child_chunk_id: uuid.UUID
    parent_chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    cosine_similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)


def insert_document(
    conn,
    *,
    doc_id: uuid.UUID,
    source_path: str,
    title: str,
    category: str,
    department: str | None,
    country: str | None,
    effective_date: str | None,
    raw_text: str,
    metadata: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO documents
            (id, source_path, title, category, department, country, effective_date, raw_text, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (doc_id, source_path, title, category, department, country, effective_date, raw_text, json.dumps(metadata)),
    )


def insert_parent_chunk(
    conn,
    *,
    chunk_id: uuid.UUID,
    document_id: uuid.UUID,
    chunk_index: int,
    content: str,
    char_start: int,
    char_end: int,
    token_count: int,
    metadata: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO parent_chunks
            (id, document_id, chunk_index, content, char_start, char_end, token_count, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (chunk_id, document_id, chunk_index, content, char_start, char_end, token_count, json.dumps(metadata)),
    )


def insert_child_chunk(
    conn,
    *,
    chunk_id: uuid.UUID,
    parent_chunk_id: uuid.UUID,
    document_id: uuid.UUID,
    chunk_index: int,
    content: str,
    char_start: int,
    char_end: int,
    token_count: int,
    embedding: list[float],
    embedding_model: str,
    metadata: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO child_chunks
            (id, parent_chunk_id, document_id, chunk_index, content, char_start, char_end,
             token_count, embedding, embedding_model, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            chunk_id,
            parent_chunk_id,
            document_id,
            chunk_index,
            content,
            char_start,
            char_end,
            token_count,
            embedding,
            embedding_model,
            json.dumps(metadata),
        ),
    )


def build_filter_clause(filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
    """Translate a simple {"category": "hr", "country": "IN"} dict into a SQL
    WHERE fragment. Deliberately supports only equality on documents.category
    and country -- the exact two fields rag/metadata_router.py produces --
    rather than a generic filter DSL.

    Country is matched against EITHER documents.country (a whole document
    that's country-specific) OR child_chunks.metadata->>'country' (a section
    within a multi-country document, tagged per-chunk at ingest time -- see
    ingestion/loader.py's tag_country()). This is what lets a query like
    "parental leave in India" find the India subsection of a document that
    also covers the US/UK/Germany.
    """
    if not filters:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    if filters.get("category"):
        clauses.append("d.category = %s")
        params.append(filters["category"])
    if filters.get("country"):
        clauses.append("(d.country = %s OR c.metadata->>'country' = %s)")
        params.append(filters["country"])
        params.append(filters["country"])
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def vector_search(
    query_embedding: list[float],
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[VectorHit]:
    filter_sql, filter_params = build_filter_clause(filters)

    sql = f"""
        SELECT
            c.id AS child_chunk_id,
            c.parent_chunk_id,
            c.document_id,
            c.content,
            1 - (c.embedding <=> %s) AS cosine_similarity,
            c.metadata
        FROM child_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL{filter_sql}
        ORDER BY c.embedding <=> %s
        LIMIT %s
    """
    # Wrap explicitly as pgvector's Vector type: psycopg can infer "this
    # parameter is a vector" for an INSERT (the target column's type tells
    # it), but for a bare expression like `embedding <=> %s` in a SELECT/
    # ORDER BY there's no column context to infer from, so a plain Python
    # list would otherwise be sent as a generic float array and pgvector's
    # `<=>` operator would reject it with "operator does not exist".
    query_vector = Vector(query_embedding)
    params = [query_vector, *filter_params, query_vector, top_k]

    print(f"  [vector_search] filters={filters or {}} top_k={top_k}")
    print(f"  [vector_search] ORDER BY embedding <=> query_vector (pgvector cosine distance, ascending)")

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    hits = [
        VectorHit(
            child_chunk_id=row[0],
            parent_chunk_id=row[1],
            document_id=row[2],
            content=row[3],
            cosine_similarity=float(row[4]),
            metadata=row[5] or {},
        )
        for row in rows
    ]

    print(f"  [vector_search] {len(hits)} result(s):")
    for rank, hit in enumerate(hits, start=1):
        print(
            f"    #{rank} cosine_similarity={hit.cosine_similarity:.4f} "
            f"chunk_id={hit.child_chunk_id} content={hit.content[:80]!r}..."
        )

    return hits


def get_document_info(conn, document_id: uuid.UUID) -> dict[str, Any] | None:
    """Used when parent/child expansion is disabled (rag/pipeline.py falls
    back to treating each child chunk as its own context) but we still want
    to show a real document title/source/category rather than raw IDs."""
    row = conn.execute(
        "SELECT title, source_path, category FROM documents WHERE id = %s",
        (document_id,),
    ).fetchone()
    if row is None:
        return None
    return {"title": row[0], "source_path": row[1], "category": row[2]}


def get_parent_chunk(conn, parent_chunk_id: uuid.UUID) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT pc.id, pc.document_id, pc.content, pc.metadata, d.title, d.source_path, d.category
        FROM parent_chunks pc
        JOIN documents d ON d.id = pc.document_id
        WHERE pc.id = %s
        """,
        (parent_chunk_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "parent_chunk_id": row[0],
        "document_id": row[1],
        "content": row[2],
        "metadata": row[3] or {},
        "document_title": row[4],
        "source_path": row[5],
        "category": row[6],
    }

"""
Ingestion CLI: load raw documents -> chunk -> embed -> write to Postgres.

Run with: `python -m backend.app.ingestion.ingest` (or `make ingest`).
Safe to re-run: it wipes and reloads all three tables (documents,
parent_chunks, child_chunks) each time, so ingestion is idempotent and you
can freely edit data/raw_docs/** and re-ingest.
"""

from __future__ import annotations

import uuid

from backend.app.config import get_settings
from backend.app.db import get_connection
from backend.app.ingestion.loader import load_documents, tag_country
from backend.app.rag.chunking import chunk_child, chunk_parent
from backend.app.rag.embeddings import get_embedding_provider
from backend.app.rag.vector_store import insert_child_chunk, insert_document, insert_parent_chunk


def reset_tables(conn) -> None:
    conn.execute("TRUNCATE documents, parent_chunks, child_chunks CASCADE")


def ingest() -> None:
    settings = get_settings()
    embedding_provider = get_embedding_provider(settings)
    embedding_model_label = f"{settings.embedding_provider}:{embedding_provider.model_name}"

    documents = load_documents()
    print(f"found {len(documents)} raw documents in data/raw_docs/")
    print(f"embedding model: {embedding_model_label} (dims={embedding_provider.dims})")

    with get_connection() as conn:
        reset_tables(conn)

        total_parent_chunks = 0
        total_child_chunks = 0

        for doc in documents:
            doc_id = uuid.uuid4()
            print(f"\n{'=' * 70}")
            print(f"DOCUMENT: {doc.source_path}")
            print(f"  title={doc.title!r} category={doc.category} effective_date={doc.effective_date}")
            print(f"  raw_text: {len(doc.raw_text)} chars")

            # OKF-style fields (see docs/concepts/okf.md) land in the existing
            # metadata JSONB column -- no schema migration needed for this
            # additive layer. Documents without frontmatter (doc.tags/related
            # empty, doc.doc_type/stale_after None) get metadata={}, exactly
            # as before.
            document_metadata = {}
            if doc.doc_type:
                document_metadata["type"] = doc.doc_type
            if doc.tags:
                document_metadata["tags"] = doc.tags
            if doc.related:
                document_metadata["related"] = doc.related
            if doc.stale_after:
                document_metadata["stale_after"] = doc.stale_after
            if document_metadata:
                print(f"  OKF frontmatter: {document_metadata}")

            insert_document(
                conn,
                doc_id=doc_id,
                source_path=doc.source_path,
                title=doc.title,
                category=doc.category,
                department=None,
                country=None,  # document-level country left null; see tag_country() for per-chunk tagging
                effective_date=doc.effective_date,
                raw_text=doc.raw_text,
                metadata=document_metadata,
            )

            parent_chunks = chunk_parent(
                doc.raw_text,
                max_chars=settings.parent_chunk_max_chars,
                overlap=settings.parent_chunk_overlap,
            )
            print(
                f"  CHUNK_PARENT: {len(parent_chunks)} parent chunk(s) "
                f"(max_chars={settings.parent_chunk_max_chars}, overlap={settings.parent_chunk_overlap})"
            )

            for parent in parent_chunks:
                parent_id = uuid.uuid4()
                parent_country = tag_country(parent.content)
                parent_metadata = {"country": parent_country} if parent_country else {}

                print(
                    f"    parent[{parent.chunk_index}] chars={parent.char_start}-{parent.char_end} "
                    f"({parent.token_count} tokens est.)"
                    + (f" country={parent_country}" if parent_country else "")
                )

                insert_parent_chunk(
                    conn,
                    chunk_id=parent_id,
                    document_id=doc_id,
                    chunk_index=parent.chunk_index,
                    content=parent.content,
                    char_start=parent.char_start,
                    char_end=parent.char_end,
                    token_count=parent.token_count,
                    metadata=parent_metadata,
                )
                total_parent_chunks += 1

                children = chunk_child(
                    parent.content,
                    max_chars=settings.child_chunk_max_chars,
                    overlap=settings.child_chunk_overlap,
                )
                if not children:
                    continue
                print(
                    f"      CHUNK_CHILD: {len(children)} child chunk(s) of parent[{parent.chunk_index}] "
                    f"(max_chars={settings.child_chunk_max_chars}, overlap={settings.child_chunk_overlap})"
                )

                # Batch-embed all children of this parent in one call per
                # document section, rather than one API call per chunk.
                child_texts = [c.content for c in children]
                print(f"      EMBED: sending {len(child_texts)} text(s) to {embedding_model_label} in one batch call")
                child_embeddings = embedding_provider.embed_documents(child_texts)
                print(f"      EMBED: received {len(child_embeddings)} vector(s), {embedding_provider.dims} dims each")

                for child, embedding in zip(children, child_embeddings):
                    vector_preview = [round(v, 4) for v in embedding[:5]]
                    print(
                        f"        child[{child.chunk_index}] chars={child.char_start}-{child.char_end} "
                        f"({child.token_count} tokens est.) content={child.content[:70]!r}..."
                    )
                    print(f"          embedding preview (first 5 of {len(embedding)} dims): {vector_preview}")

                    insert_child_chunk(
                        conn,
                        chunk_id=uuid.uuid4(),
                        parent_chunk_id=parent_id,
                        document_id=doc_id,
                        chunk_index=child.chunk_index,
                        content=child.content,
                        char_start=child.char_start,
                        char_end=child.char_end,
                        token_count=child.token_count,
                        embedding=embedding,
                        embedding_model=embedding_model_label,
                        metadata=parent_metadata,  # child inherits the parent's country tag
                    )
                    total_child_chunks += 1

        conn.commit()

    print(
        f"done: {len(documents)} documents, {total_parent_chunks} parent chunks, "
        f"{total_child_chunks} child chunks (embedding dims={embedding_provider.dims})"
    )


if __name__ == "__main__":
    ingest()

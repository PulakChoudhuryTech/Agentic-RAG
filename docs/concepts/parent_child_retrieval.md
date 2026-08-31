# Parent/Child Retrieval

**Where it's implemented:** [`backend/app/rag/chunking.py`](../../backend/app/rag/chunking.py) (splitting) and [`backend/app/rag/parent_child.py`](../../backend/app/rag/parent_child.py) (expansion)

## The tension this solves

Chunk size for retrieval and chunk size for LLM context want opposite
things:

- **Small chunks are better for search.** A ~300-character chunk embeds one
  focused idea; a query about "VPN reconnection after a password reset"
  will match it sharply. Embed a whole 2000-character document section
  instead, and the vector becomes a blurry average of everything in that
  section -- much less discriminating.
- **Large chunks are better for the LLM.** A 300-character snippet, on its
  own, often lacks the surrounding context (what section is this from? what
  qualifies this statement?) the LLM needs to answer well and avoid
  misreading an isolated sentence.

Parent/child retrieval resolves the tension by using different chunk sizes
for each job: **search small, return big.**

## How it works in this project

1. At ingestion time (`ingestion/ingest.py`), each document is split into
   **parent chunks** (~1800 characters, `rag/chunking.chunk_parent`), and
   each parent chunk is further split into **child chunks** (~350
   characters, `rag/chunking.chunk_child`). Every child chunk stores a
   `parent_chunk_id` foreign key.
2. Only child chunks are embedded and searched (vector search, keyword
   search, RRF, reranking all operate on child chunks -- see
   `db/migrations/004_child_chunks.sql`).
3. After reranking narrows the results to the best few child chunks,
   `parent_child.expand_to_parents()` looks up each one's **parent** chunk
   and returns that instead -- deduplicating when multiple child chunks
   from the same parent both scored well (which is common and a good sign:
   it means multiple angles converged on the same section of the document).
4. The LLM only ever sees parent-chunk-sized context, assembled by
   `rag/context_assembly.py`.

## Try it yourself

Toggle `ENABLE_PARENT_CHILD=false` in `.env`. With it off, the pipeline
sends the raw ~350-character child chunks straight to the LLM instead of
expanding them -- for some queries this barely matters, but for anything
needing surrounding context (e.g. "what's the eligibility requirement" when
the requirement sentence is a few lines below the retrieved snippet) you
should see answers get noticeably thinner or start hedging. The debug
trace's `parent_child` step shows exactly how many child chunks got
deduplicated into how many parent chunks.

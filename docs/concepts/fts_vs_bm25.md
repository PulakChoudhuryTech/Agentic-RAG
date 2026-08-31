# PostgreSQL Full Text Search vs. BM25

**Where it's implemented:** [`backend/app/rag/keyword_search.py`](../../backend/app/rag/keyword_search.py)

This project uses PostgreSQL Full Text Search (FTS) for the "keyword"
side of hybrid search, instead of Elasticsearch/OpenSearch (which implement
real BM25). **This document exists specifically to be honest that these are
not the same thing.**

## What we actually use

```sql
SELECT ..., ts_rank_cd(c.content_tsv, plainto_tsquery('english', %s)) AS ts_rank
FROM child_chunks c
WHERE c.content_tsv @@ plainto_tsquery('english', %s)
ORDER BY ts_rank DESC
```

`content_tsv` is a generated column (`to_tsvector('english', content)`,
see `db/migrations/004_child_chunks.sql`), and a GIN index on it makes the
`@@` match fast. `ts_rank_cd` ("cover density") scores matches favoring
terms that appear close together.

## What's different from real BM25

| | PostgreSQL FTS (`ts_rank_cd`) | BM25 |
|---|---|---|
| Term frequency | Counted, but not saturating | Saturates via the `k1` parameter -- the 10th occurrence of a term matters much less than the 2nd |
| Inverse document frequency (IDF) | **Not applied** -- a rare, distinctive word and a common word are weighted the same | Rare terms across the corpus are weighted much higher |
| Document length normalization | **Not applied** | Via the `b` parameter -- a match in a long document is normalized against a match in a short one |
| Proximity/cover density | Applied (this is what "cd" means) | Not part of classic BM25 |

In practice, the biggest gap for this project is the **missing IDF term**:
BM25 would naturally rank a chunk matching a rare, specific word (e.g.
"GlobalProtect") much higher than a chunk matching a common word that
happens to appear everywhere (e.g. "employee"). PostgreSQL FTS treats them
the same, so on its own it's a weaker "specificity" signal than true BM25.

## Why this is still a reasonable choice for this project

1. It keeps the whole system on one database (Postgres + pgvector already
   have to be there for the vector side) instead of adding
   Elasticsearch/OpenSearch as a second piece of infrastructure to run,
   learn, and operate.
2. Combined with vector search via RRF (see `docs/concepts/rrf.md`), the
   keyword signal mostly needs to answer "does this chunk contain the
   literal words the user typed" -- which FTS does reasonably well -- while
   semantic matching is handled by the vector side.
3. `keyword_search()`'s function signature (`query, top_k, filters ->
   list[KeywordHit]`) is the only thing `hybrid_search.py` depends on. If
   you outgrow this project's scale and want real BM25, you can swap this
   file's implementation for a call to an actual search engine or a Python
   BM25 library (e.g. `bm25s`, `rank_bm25`) without touching
   `hybrid_search.py`, `pipeline.py`, or anything upstream.

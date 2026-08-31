# HNSW (Hierarchical Navigable Small World)

**Where it's implemented:** [`db/migrations/005_indexes.sql`](../../db/migrations/005_indexes.sql)

## What it is

HNSW is an approximate-nearest-neighbor (ANN) index structure. Instead of
comparing a query vector against every single row (an "exact" or "flat"
scan, which is `O(n)`), HNSW builds a multi-layer graph at insert time where
each vector is connected to a handful of its nearest neighbors. Searching
means starting at the top (sparse) layer and greedily walking toward the
query vector, layer by layer, until you reach the dense bottom layer --
much faster than scanning everything, at the cost of occasionally missing
the true nearest neighbor (hence "approximate").

## The index in this project

```sql
CREATE INDEX child_chunks_hnsw_idx ON child_chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```

- **`vector_cosine_ops`**: tells pgvector to build the graph optimized for
  cosine distance (matching the `<=>` operator used in
  `rag/vector_store.py`'s `ORDER BY embedding <=> %s`). pgvector also
  supports L2 and inner-product operator classes for other distance
  metrics.
- **`m = 16`**: the maximum number of graph connections per node. Higher
  `m` = better recall, but a bigger index and slower builds.
- **`ef_construction = 64`**: the size of the candidate list considered
  while building each node's connections. Higher = a better-quality graph,
  slower to build.
- There's also a *query-time* knob, `hnsw.ef_search` (not set explicitly in
  this project, so it uses pgvector's default), which controls how large a
  candidate list is explored *during search*. Higher = better recall,
  slower queries. You can set it per-session with `SET hnsw.ef_search = 40;`
  if you want to experiment.

## The honest caveat

This project's document corpus is a handful of files, chunked into a few
hundred child chunks. At that scale, HNSW's approximate search behaves
**almost identically to an exact linear scan** -- there simply isn't enough
data for the "graph shortcut vs. scan everything" trade-off to matter, and
you won't observe recall loss or a meaningful speed difference either way.

The index is here so you can see the mechanism (the SQL, the operator
class, the tuning knobs) and read about how it works at real scale --
not because this project's dataset needs it. At real-world scale (millions
of vectors), HNSW is often the difference between a sub-100ms query and one
that takes seconds, at a typically small (low single-digit %) recall cost
compared to exact search.

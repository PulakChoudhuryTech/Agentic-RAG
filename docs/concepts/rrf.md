# Reciprocal Rank Fusion (RRF)

**Where it's implemented:** [`backend/app/rag/hybrid_search.py`](../../backend/app/rag/hybrid_search.py)

## The problem it solves

Vector search returns cosine similarity scores (roughly 0 to 1). Keyword
search (PostgreSQL FTS) returns `ts_rank_cd` scores, which are unbounded and
depend on term frequency and the size of the corpus. These two scores are
not on the same scale, so you can't just add or average them -- whichever
happens to produce bigger numbers would dominate the combined ranking,
regardless of which result is actually more relevant.

## The formula

RRF sidesteps the scale problem entirely by throwing away the raw scores
and using only each result's **rank** (1st, 2nd, 3rd, ...) within its own
list:

```
RRF_score(document) = sum over every ranked list L containing the document of  1 / (k + rank_in_L)
```

A document that appears in both the vector list (rank 3) and the keyword
list (rank 1) gets:

```
1/(k+3) + 1/(k+1)
```

A document that appears only in the vector list at rank 1 gets just
`1/(k+1)` -- roughly half the score of a document that shows up decently in
both lists. This is the key property RRF is chosen for: **agreement between
retrieval methods is rewarded**, which tends to surface documents that are
robustly relevant rather than a fluke of one particular scoring method.

## Why `k = 60`

`k` softens the impact of small rank differences, especially at the top of
the list. Without it, the gap between rank 1 (`1/1`) and rank 2 (`1/2`)
would be huge (a 2x difference), letting one list's opinion at rank 1
completely swamp another list's opinion at rank 2. With `k = 60` (the value
used in the original RRF paper, and this project's default via
`RRF_K` in `.env`), the same gap becomes `1/61` vs `1/62` -- a much gentler
~1.6% difference. Larger `k` values flatten the influence of rank further;
smaller `k` values make rank 1 matter more decisively.

## Try it yourself

Toggle `ENABLE_HYBRID_SEARCH=false` in `.env` and compare answers/citations
for a query where the exact keyword match matters (e.g. an acronym or a
specific dollar figure) versus a query that's more about meaning/paraphrase.
The debug trace's `hybrid_search` step shows the RRF score, vector rank, and
keyword rank for each result, so you can see exactly why the fused ranking
came out the way it did.

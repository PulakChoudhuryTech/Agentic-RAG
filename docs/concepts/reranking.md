# Cross-Encoder Reranking

**Where it's implemented:** [`backend/app/rag/reranker.py`](../../backend/app/rag/reranker.py)

## Bi-encoder vs. cross-encoder

Everything before reranking in this pipeline (embeddings, vector search)
uses a **bi-encoder**: the query and each document chunk are embedded
*independently*, and similarity is computed afterward (cosine distance).
This is fast and scales to millions of documents (you can precompute every
document's embedding once), but the model never actually looks at the query
and the document *together*.

A **cross-encoder** takes the `(query, document)` pair as a single input
and outputs one relevance score directly. Because the model attends across
both texts at once, it can pick up on things a bi-encoder's independent
embeddings tend to miss -- most notably negation and fine-grained semantic
relationships ("eligible for X" vs. "not eligible for X" can produce very
similar embeddings, but a cross-encoder reading both spans together scores
them very differently).

The trade-off is cost: a cross-encoder can't be precomputed, so it has to
run at query time, once per candidate. That's why reranking only runs on
the small set of candidates that already survived vector + keyword search
and RRF fusion (`RERANK_TOP_N` results, default top 5 of the fused
candidate pool) -- not the whole corpus.

## The model

Default: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB, fast to download
and run on CPU). This project's `RERANKER_MODEL_NAME` setting also supports
swapping in a BGE cross-encoder (e.g. `BAAI/bge-reranker-base`, ~1.1GB, or
`BAAI/bge-reranker-large`) for a quality/latency comparison -- both are
loaded the same way via `sentence_transformers.CrossEncoder`.

## Try it yourself

Toggle `ENABLE_RERANKING=false` in `.env` and compare the same query's
results before/after. The debug trace's `reranker` step shows each
candidate's score before reranking (its RRF/vector rank) and after (its
cross-encoder score), so you can see exactly which results moved and by how
much.

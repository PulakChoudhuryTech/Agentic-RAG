"""
Embedding providers.

`EmbeddingProvider` is the swap point the user asked for ("configurable
embedding model"): anything implementing `embed_documents`/`embed_query`/
`.dims` can be dropped in via EMBEDDING_PROVIDER in .env, without touching
vector_store.py or pipeline.py.

Default: Gemini's `gemini-embedding-001`, called through the `google-genai`
SDK -- the same API key as the LLM, no extra setup. This model natively
outputs 3072-dim vectors (via Matryoshka Representation Learning, MRL), but
we request `output_dimensionality=768` explicitly so the vectors match this
project's `VECTOR(768)` column (see db/migrations/004_child_chunks.sql) --
768 was chosen to keep parity with the older text-embedding-004 model this
project originally targeted before Google deprecated it, not because 768 is
special; bump EMBEDDING_DIMS in .env (and the column type) if you want the
full 3072 dims instead.

`LocalSentenceTransformerEmbedding` is a working alternative (BGE-small,
runs fully offline after the first download) kept in the same file so the
two implementations are easy to compare side by side. It is NOT wired up by
default because BGE-small outputs 384-dim vectors, not 768 -- switching
providers means re-ingesting (the `embedding_model` column exists precisely
so you can tell, per row, which model produced which vector).
"""

from __future__ import annotations

import time
from typing import Protocol

from google import genai
from google.genai.errors import ClientError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


class EmbeddingProvider(Protocol):
    dims: int
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _is_rate_limit_error(exc: BaseException) -> bool:
    return isinstance(exc, ClientError) and getattr(exc, "code", None) == 429


class GeminiEmbeddingProvider:
    """
    Calls Gemini's embedding model in batches (Gemini's `embed_content`
    accepts a list of texts and returns one embedding per item in a SINGLE
    HTTP request) rather than one request per chunk. This matters
    concretely: the free tier's `embed_content_free_tier_requests` quota is
    easy to blow through with a one-request-per-chunk approach on even this
    project's small sample corpus (~150+ child chunks). Batching in groups
    of `BATCH_SIZE` cuts that down to a handful of requests, and a minimum
    gap enforced between requests (`MIN_SECONDS_BETWEEN_REQUESTS`) further
    spreads them out instead of bursting them.

    Every call to `_embed_batch` -- whether it's one of several batches
    within a single `embed_documents()` call, or a totally separate call
    for the next parent chunk (see ingestion/ingest.py, which calls
    `embed_documents()` once per parent chunk, not once per whole
    document) -- is throttled against a single `_last_request_time`
    tracked on the instance, so requests stay spread out across the
    instance's *entire* lifetime, not just within one method call.

    Each batch call is also retried with patient exponential backoff
    specifically on HTTP 429 (rate limit) responses -- other errors (bad
    API key, model not found) are NOT retried, since retrying those would
    just burn time before failing the same way anyway. The free tier's
    exact reset window isn't documented precisely, so the backoff here is
    deliberately generous (up to a few minutes total) rather than tuned to
    a specific number -- this only runs during `make ingest`, a one-time
    CLI operation, so a few extra minutes of patience costs nothing.
    """

    BATCH_SIZE = 10
    MIN_SECONDS_BETWEEN_REQUESTS = 2.0

    def __init__(self, api_key: str, model_name: str, dims: int = 768) -> None:
        self.model_name = model_name
        self.dims = dims
        self._client = genai.Client(api_key=api_key)
        self._last_request_time = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        remaining = self.MIN_SECONDS_BETWEEN_REQUESTS - elapsed
        if remaining > 0:
            print(f"        [embed throttle] sleeping {remaining:.1f}s (min gap={self.MIN_SECONDS_BETWEEN_REQUESTS}s)")
            time.sleep(remaining)
        self._last_request_time = time.monotonic()

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=3, min=5, max=45),
        retry=retry_if_exception(_is_rate_limit_error),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        print(
            f"        [embed_content] model={self.model_name} task_type={task_type} "
            f"texts={len(texts)} output_dimensionality={self.dims}"
        )
        response = self._client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config={"task_type": task_type, "output_dimensionality": self.dims},
        )
        vectors = [list(e.values) for e in response.embeddings]
        print(f"        [embed_content] <- {len(vectors)} vector(s) back from Gemini")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # task_type="RETRIEVAL_DOCUMENT" tells the model these vectors will be
        # searched *against* -- Gemini's embedding model uses different
        # internal weighting for documents vs. queries, so getting this right
        # measurably improves retrieval quality.
        #
        # Throttled (unlike embed_query below): this is the bulk-ingestion
        # path, called once per parent chunk by ingestion/ingest.py, and
        # it's the one that actually risks bursting past the free tier's
        # rate limit. A single interactive RAG query's embed_query() call
        # doesn't need this -- it's naturally spaced out by the rest of the
        # pipeline and the user's own pace.
        results: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            self._throttle()
            batch = texts[i : i + self.BATCH_SIZE]
            results.extend(self._embed_batch(batch, "RETRIEVAL_DOCUMENT"))
        return results

    def embed_query(self, text: str) -> list[float]:
        print(f"    [embed_query] embedding query text: {text[:80]!r}")
        vector = self._embed_batch([text], "RETRIEVAL_QUERY")[0]
        print(f"    [embed_query] -> vector: {vector[:5]} ... ({len(vector)} dims total)")
        return vector


class LocalSentenceTransformerEmbedding:
    """Fully offline alternative via sentence-transformers. Not the default
    (see module docstring) but implements the same interface so it's a
    one-line swap in get_embedding_provider() below."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dims = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        # BGE models expect a query instruction prefix for asymmetric search.
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        return self._model.encode([prefixed], normalize_embeddings=True)[0].tolist()


def get_embedding_provider(settings) -> EmbeddingProvider:  # noqa: ANN001 - avoids circular import on Settings
    if settings.embedding_provider == "gemini":
        return GeminiEmbeddingProvider(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_embedding_model,
            dims=settings.embedding_dims,
        )
    if settings.embedding_provider == "local":
        return LocalSentenceTransformerEmbedding()
    raise ValueError(f"unknown embedding_provider: {settings.embedding_provider}")

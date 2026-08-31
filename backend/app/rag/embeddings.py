"""
Embedding providers.

`EmbeddingProvider` is the swap point the user asked for ("configurable
embedding model"): anything implementing `embed_documents`/`embed_query`/
`.dims` can be dropped in via EMBEDDING_PROVIDER in .env, without touching
vector_store.py or pipeline.py.

Default: Gemini's `text-embedding-004` (768 dims), called through the
`google-genai` SDK -- the same API key as the LLM, no extra setup.

`LocalSentenceTransformerEmbedding` is a working alternative (BGE-small,
runs fully offline after the first download) kept in the same file so the
two implementations are easy to compare side by side. It is NOT wired up by
default because BGE-small outputs 384-dim vectors, not 768 -- switching
providers means re-ingesting (the `embedding_model` column exists precisely
so you can tell, per row, which model produced which vector).
"""

from __future__ import annotations

from typing import Protocol

from google import genai


class EmbeddingProvider(Protocol):
    dims: int
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbeddingProvider:
    """Calls Gemini's embedding model one text at a time. For this project's
    corpus size (a handful of documents, a few hundred chunks) that's simple
    and fast enough; a production system would batch more aggressively."""

    def __init__(self, api_key: str, model_name: str, dims: int = 768) -> None:
        self.model_name = model_name
        self.dims = dims
        self._client = genai.Client(api_key=api_key)

    def _embed_one(self, text: str, task_type: str) -> list[float]:
        response = self._client.models.embed_content(
            model=self.model_name,
            contents=text,
            config={"task_type": task_type},
        )
        return list(response.embeddings[0].values)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # task_type="RETRIEVAL_DOCUMENT" tells the model these vectors will be
        # searched *against* -- Gemini's embedding model uses different
        # internal weighting for documents vs. queries, so getting this right
        # measurably improves retrieval quality.
        return [self._embed_one(t, "RETRIEVAL_DOCUMENT") for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text, "RETRIEVAL_QUERY")


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

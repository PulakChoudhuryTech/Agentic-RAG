"""
Embedding-based intent classification.

This is a SEPARATE, simpler signal from the LangGraph supervisor's LLM-based
routing decision (agents/supervisor.py). It works by embedding a small,
hand-written set of example utterances per intent once, then classifying a
new query by finding which example it's most similar to (cosine similarity).

Relationship to the LLM supervisor (see agents/supervisor.py and
docs/architecture.md): when ENABLE_EMBEDDING_INTENT_CLASSIFIER is on, this
classifier's result is computed FIRST and passed into the supervisor's
prompt as a hint, and both are recorded in the trace side by side
(`intent_scores` here vs. the LLM's final `intent` decision) -- so toggling
the flag lets you compare "cheap embedding similarity" against "LLM
judgment" directly, on the same query. The LLM supervisor always makes the
final call; this is a hint, not an override.
"""

from __future__ import annotations

import numpy as np

from backend.app.rag.embeddings import EmbeddingProvider

# One or more example utterances per intent. Kept small and hand-written
# (not learned) so the whole classifier fits in this one file.
INTENT_EXAMPLES: dict[str, list[str]] = {
    "rag_hr": [
        "What is the parental leave policy?",
        "How many vacation days do employees get per year?",
        "What health insurance plans are available?",
        "Tell me about the remote work policy.",
        "What benefits does the company offer?",
    ],
    "rag_it": [
        "My VPN isn't working, how do I fix it?",
        "How do I reset my corporate password?",
        "How do I request a new laptop?",
        "My account is locked, what do I do?",
    ],
    "rag_travel": [
        "What is the travel expense policy?",
        "How much can I spend on hotels for business travel?",
        "Do I need a visa for my business trip?",
        "What is the per diem rate for international travel?",
    ],
    "rag_personal": [
        "What is my current phone/internet bill amount?",
        "When is my insurance premium due?",
        "What is the PNR for my flight ticket?",
        "What is my insurance policy number?",
        "How much do I owe on my last bill?",
    ],
    "workday": [
        "How many vacation days do I have left?",
        "What is my current leave balance?",
        "What benefits am I enrolled in?",
        "Can you look up my employee profile?",
    ],
    "servicenow_troubleshoot": [
        "Please create an IT ticket for my broken laptop.",
        "Open a support ticket for my VPN issue.",
        "What is the status of my IT ticket?",
        "Try troubleshooting my VPN and create a ticket if it still doesn't work.",
    ],
    "combined": [
        "Am I eligible for parental leave based on my profile?",
        "Given my tenure, how much vacation can I take?",
        "Based on my country and role, what benefits do I qualify for?",
    ],
}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class IntentClassifier:
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._provider = embedding_provider
        self._labels: list[str] = []
        self._examples: list[str] = []
        for intent, examples in INTENT_EXAMPLES.items():
            for example in examples:
                self._labels.append(intent)
                self._examples.append(example)
        # Computed lazily on first classify() call, not at import time, so
        # importing this module never triggers an API call by itself.
        self._example_embeddings: np.ndarray | None = None

    def _ensure_embeddings(self) -> None:
        if self._example_embeddings is None:
            vectors = self._provider.embed_documents(self._examples)
            self._example_embeddings = np.array(vectors)

    def classify(self, query: str) -> tuple[str, dict[str, float]]:
        """Returns (top_intent, {intent: max_similarity_to_any_example})."""
        self._ensure_embeddings()
        query_vec = np.array(self._provider.embed_query(query))

        best_per_intent: dict[str, float] = {intent: -1.0 for intent in INTENT_EXAMPLES}
        for label, example_vec in zip(self._labels, self._example_embeddings):
            sim = _cosine_similarity(query_vec, example_vec)
            if sim > best_per_intent[label]:
                best_per_intent[label] = sim

        top_intent = max(best_per_intent, key=best_per_intent.get)
        return top_intent, best_per_intent


_classifier_instance: IntentClassifier | None = None


def get_intent_classifier(embedding_provider: EmbeddingProvider) -> IntentClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier(embedding_provider)
    return _classifier_instance

"""
Unit-tests IntentClassifier's similarity math with a fake embedding provider
(no real Gemini calls) that returns deterministic, hand-picked vectors so we
can predict exactly which intent should win.
"""

from backend.app.rag.intent_classifier import INTENT_EXAMPLES, IntentClassifier


class _FakeEmbeddingProvider:
    """Maps specific known strings to specific vectors; anything else gets a
    default "neutral" vector. This lets us control similarity outcomes
    exactly instead of depending on a real embedding model's behavior."""

    model_name = "fake"
    dims = 3

    def __init__(self, vector_by_text: dict[str, list[float]], default=(0.0, 0.0, 1.0)):
        self._vectors = vector_by_text
        self._default = list(default)

    def embed_documents(self, texts):
        return [self._vectors.get(t, self._default) for t in texts]

    def embed_query(self, text):
        return self._vectors.get(text, self._default)


def test_classify_picks_the_most_similar_example_intent():
    # Pick one real example utterance from each of two intents and give them
    # clearly separated vectors; a query vector identical to one of them
    # must classify as that intent.
    hr_example = INTENT_EXAMPLES["rag_hr"][0]
    it_example = INTENT_EXAMPLES["rag_it"][0]

    provider = _FakeEmbeddingProvider(
        {
            hr_example: [1.0, 0.0, 0.0],
            it_example: [0.0, 1.0, 0.0],
            "some IT sounding question": [0.0, 0.9, 0.1],
        }
    )
    classifier = IntentClassifier(provider)

    top_intent, scores = classifier.classify("some IT sounding question")

    assert top_intent == "rag_it"
    assert scores["rag_it"] > scores["rag_hr"]


def test_classify_returns_a_score_for_every_intent():
    provider = _FakeEmbeddingProvider({})
    classifier = IntentClassifier(provider)

    _, scores = classifier.classify("anything")

    assert set(scores.keys()) == set(INTENT_EXAMPLES.keys())


def test_embeddings_are_computed_lazily_not_at_construction():
    provider = _FakeEmbeddingProvider({})
    classifier = IntentClassifier(provider)
    assert classifier._example_embeddings is None

    classifier.classify("anything")
    assert classifier._example_embeddings is not None

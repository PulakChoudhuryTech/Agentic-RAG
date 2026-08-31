"""
Requires the full stack: Postgres migrated + ingested, and a valid
GEMINI_API_KEY (this endpoint makes real Gemini calls for embeddings and
the answer). See backend/tests/conftest.py for the skip-if-unavailable
behavior.
"""


def test_rag_query_returns_answer_and_citations(client):
    resp = client.post("/rag/query", json={"query": "What is the parental leave policy in India?"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["answer"]
    assert len(data["citations"]) > 0
    assert any("parental_leave" in c["source_path"] for c in data["citations"])


def test_rag_query_trace_shows_every_pipeline_stage_when_debug_true(client):
    resp = client.post("/rag/query?debug=true", json={"query": "How much vacation do I accrue after 5 years?"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["trace"] is not None
    nodes_seen = {step["node"] for step in data["trace"]}
    # the full pipeline trace described in the README/architecture doc
    for expected_node in ["metadata_router", "vector_search", "reranker", "context_assembly", "llm"]:
        assert expected_node in nodes_seen


def test_rag_query_trace_omitted_when_debug_false(client):
    resp = client.post("/rag/query?debug=false", json={"query": "What is the travel expense policy?"})
    assert resp.status_code == 200
    assert resp.json()["trace"] is None


def test_manual_category_filter_scopes_results(client):
    resp = client.post("/rag/query", json={"query": "What are the requirements?", "category": "travel"})
    assert resp.status_code == 200
    data = resp.json()
    for citation in data["citations"]:
        assert citation["category"] == "travel"

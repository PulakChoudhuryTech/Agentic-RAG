"""Example query 1: "What is the parental leave policy in India?" -> RAG only."""

import uuid


def test_parental_leave_india_query(client):
    resp = client.post(
        "/chat",
        json={"session_id": str(uuid.uuid4()), "message": "What is the parental leave policy in India?"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["intent"] == "rag_hr"
    assert "26" in data["answer"] or "maternity" in data["answer"].lower()
    assert any("parental_leave" in c["source_path"] for c in data["citations"])

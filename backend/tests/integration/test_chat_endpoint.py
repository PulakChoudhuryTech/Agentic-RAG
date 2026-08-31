"""Requires the full stack (see test_rag_endpoint.py's docstring) plus the
Workday/ServiceNow mocks for the workday/combined/servicenow_troubleshoot
routes."""

import uuid


def _new_session() -> str:
    return str(uuid.uuid4())


def test_chat_persists_and_replays_history(client):
    session_id = _new_session()
    resp = client.post(
        "/chat", json={"session_id": session_id, "message": "What is the PTO carryover policy?"}
    )
    assert resp.status_code == 200

    history_resp = client.get(f"/chat/{session_id}/history")
    assert history_resp.status_code == 200
    messages = history_resp.json()["messages"]

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_chat_routes_to_workday_for_personal_data_question(client):
    session_id = _new_session()
    resp = client.post(
        "/chat",
        json={"session_id": session_id, "message": "How many vacation days do I have?", "employee_id": "E1002"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "workday"
    assert "15" in data["answer"]

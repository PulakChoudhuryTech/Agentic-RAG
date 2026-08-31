"""Example query 3: "Am I eligible for parental leave based on my profile?"
-> Workday API + RAG combined."""

import uuid


def test_parental_leave_eligibility_combines_workday_and_rag(client):
    # E1001: India, tenure_months=41 (>= 6 months required) -> should read as eligible
    resp = client.post(
        "/chat",
        json={
            "session_id": str(uuid.uuid4()),
            "message": "Am I eligible for parental leave based on my profile?",
            "employee_id": "E1001",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "combined"
    assert "india" in data["answer"].lower()


def test_parental_leave_eligibility_flags_short_tenure(client):
    # E1003: India, tenure_months=3 (< 6 months required) -> should read as not (yet) eligible
    resp = client.post(
        "/chat",
        json={
            "session_id": str(uuid.uuid4()),
            "message": "Given my tenure, do I qualify for paid parental leave?",
            "employee_id": "E1003",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "combined"
    assert "6 month" in data["answer"].lower() or "not eligible" in data["answer"].lower()

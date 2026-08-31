"""Example query 2: "How many vacation days do I have?" -> Workday API only."""

import uuid


def test_vacation_balance_query(client):
    resp = client.post(
        "/chat",
        json={
            "session_id": str(uuid.uuid4()),
            "message": "How many vacation days do I have?",
            "employee_id": "E1004",
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["intent"] == "workday"
    assert "7" in data["answer"]  # E1004's vacation_days_available in data/seed/employees.json

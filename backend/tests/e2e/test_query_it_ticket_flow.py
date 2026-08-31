"""
Example query 5: "My VPN isn't working. Try troubleshooting and create a
ServiceNow ticket if it still doesn't work." -> RAG + tool calling +
conditional LangGraph workflow, across two conversation turns:

  Turn 1: supervisor routes to servicenow_troubleshoot -> it_troubleshoot
          gives RAG-sourced troubleshooting steps and asks for confirmation.
  Turn 2: the employee says it's still broken -> it_check_resolution decides
          UNRESOLVED and calls the create_ticket tool.
"""

import uuid


def test_troubleshoot_then_create_ticket_across_two_turns(client):
    session_id = str(uuid.uuid4())

    turn1 = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "message": "My VPN isn't working. Try troubleshooting and create a ServiceNow ticket if it still doesn't work.",
            "employee_id": "E1005",
        },
    )
    assert turn1.status_code == 200
    data1 = turn1.json()
    assert data1["intent"] == "servicenow_troubleshoot"
    assert "resolve" in data1["answer"].lower() or "vpn" in data1["answer"].lower()

    turn2 = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "message": "I tried restarting the client and my machine, it still won't connect.",
            "employee_id": "E1005",
        },
    )
    assert turn2.status_code == 200
    data2 = turn2.json()
    assert data2["intent"] == "servicenow_troubleshoot"
    assert "INC" in data2["answer"]  # the mock ServiceNow ticket ID format

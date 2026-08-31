"""Example query 4: "My VPN isn't working. What should I do?" -> IT RAG."""

import uuid


def test_vpn_troubleshooting_query(client):
    resp = client.post(
        "/chat",
        json={"session_id": str(uuid.uuid4()), "message": "My VPN isn't working. What should I do?"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["intent"] == "rag_it"
    assert "globalprotect" in data["answer"].lower()
    assert any("vpn_troubleshooting" in c["source_path"] for c in data["citations"])

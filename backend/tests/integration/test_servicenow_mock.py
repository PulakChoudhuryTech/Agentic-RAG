import httpx

from backend.app.config import get_settings


def test_create_and_fetch_ticket(services_available):
    settings = get_settings()

    create_resp = httpx.post(
        f"{settings.servicenow_api_url}/tickets",
        json={
            "short_description": "VPN will not connect",
            "description": "Tried restarting client and machine, still fails.",
            "category": "Network > VPN",
            "urgency": "High",
            "requested_by": "E1005",
        },
    )
    create_resp.raise_for_status()
    ticket = create_resp.json()

    assert ticket["id"].startswith("INC")
    assert ticket["status"] == "open"
    assert ticket["requested_by"] == "E1005"

    fetch_resp = httpx.get(f"{settings.servicenow_api_url}/tickets/{ticket['id']}")
    fetch_resp.raise_for_status()
    assert fetch_resp.json()["short_description"] == "VPN will not connect"


def test_unknown_ticket_returns_404(services_available):
    settings = get_settings()
    resp = httpx.get(f"{settings.servicenow_api_url}/tickets/INC9999999")
    assert resp.status_code == 404

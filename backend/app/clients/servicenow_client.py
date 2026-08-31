"""Thin httpx client for the mock ServiceNow API."""

from __future__ import annotations

import httpx

from backend.app.config import Settings

TIMEOUT_SECONDS = 5.0


def create_ticket(
    *,
    short_description: str,
    description: str,
    category: str,
    urgency: str,
    requested_by: str,
    settings: Settings,
) -> dict:
    resp = httpx.post(
        f"{settings.servicenow_api_url}/tickets",
        json={
            "short_description": short_description,
            "description": description,
            "category": category,
            "urgency": urgency,
            "requested_by": requested_by,
        },
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


def get_ticket(ticket_id: str, settings: Settings) -> dict:
    resp = httpx.get(f"{settings.servicenow_api_url}/tickets/{ticket_id}", timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()

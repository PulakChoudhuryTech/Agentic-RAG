"""Pydantic request/response models for the mock ServiceNow API."""

from __future__ import annotations

from pydantic import BaseModel


class TicketCreateRequest(BaseModel):
    short_description: str
    description: str
    category: str = "General"
    urgency: str = "Medium"
    requested_by: str  # employee_id


class Ticket(BaseModel):
    id: str
    short_description: str
    description: str
    category: str
    urgency: str
    status: str
    requested_by: str
    created_at: str
    updated_at: str

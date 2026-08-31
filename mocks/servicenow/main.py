"""
Mock ServiceNow API.

Stands in for a real IT ticketing system. Separate FastAPI process (port
8002, see docker-compose.yml) for the same reason as the Workday mock: the
agent's IT tool-calling code should go through real HTTP + retries.

Endpoints:
  POST /tickets              create a ticket
  GET  /tickets/{ticket_id}  look up a ticket's status
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from models import Ticket, TicketCreateRequest
from store import create_ticket, get_ticket, init_db

app = FastAPI(title="Mock ServiceNow API")

init_db()


@app.post("/tickets", response_model=Ticket, status_code=201)
def post_ticket(request: TicketCreateRequest) -> Ticket:
    ticket = create_ticket(
        short_description=request.short_description,
        description=request.description,
        category=request.category,
        urgency=request.urgency,
        requested_by=request.requested_by,
    )
    return Ticket(**ticket)


@app.get("/tickets/{ticket_id}", response_model=Ticket)
def get_ticket_by_id(ticket_id: str) -> Ticket:
    ticket = get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"ticket {ticket_id} not found")
    return Ticket(**ticket)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

"""
SQLite-backed ticket store for the mock ServiceNow API.

Unlike the Workday mock (in-memory, read-only reference data), ServiceNow
tickets are created at runtime by the agent, so we persist them to a real
SQLite file (mocks/servicenow/data/tickets.db, or mocks/servicenow/tickets.db
locally) so tickets survive a process restart during a dev session -- useful
when you're iterating on the agent graph and want GET /tickets/{id} to keep
working across `uvicorn --reload`.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CANDIDATE_DB_DIRS = [
    Path("/app/mocks/servicenow/data"),
    Path(__file__).resolve().parent,
]


def _db_path() -> Path:
    for directory in _CANDIDATE_DB_DIRS:
        if directory.exists():
            return directory / "tickets.db"
    raise FileNotFoundError("no writable directory found for tickets.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id                TEXT PRIMARY KEY,
                short_description TEXT NOT NULL,
                description       TEXT NOT NULL,
                category          TEXT,
                urgency           TEXT,
                status            TEXT NOT NULL DEFAULT 'open',
                requested_by      TEXT NOT NULL,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL
            )
            """
        )


def _next_ticket_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()
    sequence = 10001 + row[0]
    return f"INC{sequence:07d}"


def create_ticket(
    *, short_description: str, description: str, category: str, urgency: str, requested_by: str
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        ticket_id = _next_ticket_id(conn)
        conn.execute(
            """
            INSERT INTO tickets
                (id, short_description, description, category, urgency, status, requested_by, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?)
            """,
            (ticket_id, short_description, description, category, urgency, requested_by, now, now),
        )
        conn.commit()
    return get_ticket(ticket_id)


def get_ticket(ticket_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    return dict(row) if row else None

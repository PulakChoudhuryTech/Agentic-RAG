"""Loads data/seed/employees.json into an in-memory dict at startup. No
database -- this mock exists to be called over HTTP like a real external
system (see mocks/workday/main.py), not to demonstrate persistence."""

from __future__ import annotations

import json
from pathlib import Path

# When run via Docker (see mocks/workday/Dockerfile) the working dir is
# /app/mocks/workday and data/seed is copied to /app/data/seed. When run
# locally via `make run-workday` (cd mocks/workday && uvicorn ...) the
# repo-relative path is the same two levels up. Try both so this file works
# in either dev workflow described in the README.
_CANDIDATE_PATHS = [
    Path(__file__).resolve().parent.parent.parent / "data" / "seed" / "employees.json",
    Path("/app/data/seed/employees.json"),
]


def load_employees() -> dict[str, dict]:
    for path in _CANDIDATE_PATHS:
        if path.exists():
            employees = json.loads(path.read_text())
            return {e["employee_id"]: e for e in employees}
    raise FileNotFoundError(f"employees.json not found in any of: {_CANDIDATE_PATHS}")

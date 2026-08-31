"""
Shared pytest fixtures.

Unit tests (backend/tests/unit/) never touch the network or a real
database -- they test pure functions and use monkeypatching for the few
places that would otherwise need one (e.g. parent/child DB lookups).

Integration and e2e tests need the real stack running: Postgres with the
schema migrated and documents ingested (`make db-up && make migrate &&
make ingest`), the mock Workday/ServiceNow APIs (`make db-up` starts these
too), and a valid GEMINI_API_KEY in .env. `services_available` pings all
three and calls `pytest.skip(...)` if they're not reachable, so `pytest`
still passes cleanly on a machine that only has the unit-test dependencies
set up, while giving you real answers once the stack is running.
"""

from __future__ import annotations

import httpx
import pytest

from backend.app.config import get_settings


@pytest.fixture(scope="session")
def settings():
    try:
        return get_settings()
    except Exception as exc:  # noqa: BLE001 - pydantic-settings validation error when .env is missing/incomplete
        pytest.skip(f"settings could not be loaded (check .env): {exc}")


@pytest.fixture(scope="session")
def services_available(settings):
    try:
        with httpx.Client(timeout=2.0) as client:
            client.get(f"{settings.workday_api_url}/health").raise_for_status()
            client.get(f"{settings.servicenow_api_url}/health").raise_for_status()
    except httpx.HTTPError:
        pytest.skip("Workday/ServiceNow mocks not reachable -- run `make db-up` first")

    try:
        from backend.app.db import get_connection

        with get_connection() as conn:
            conn.execute("SELECT 1")
    except Exception:
        pytest.skip("Postgres not reachable -- run `make db-up && make migrate && make ingest` first")

    return True


@pytest.fixture(scope="session")
def client(services_available):
    """FastAPI TestClient, imported lazily so a missing/incomplete .env
    (which would raise at import time via Settings()) turns into a clean
    skip rather than a collection error."""
    try:
        from fastapi.testclient import TestClient

        from backend.app.main import app
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not import the FastAPI app (check .env): {exc}")

    with TestClient(app) as test_client:
        yield test_client

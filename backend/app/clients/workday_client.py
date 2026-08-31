"""Thin httpx client for the mock Workday API. Plain functions, no class --
each one is a single HTTP call, kept simple so agents/tools_workday.py can
wrap them as LangChain tools with minimal indirection in between."""

from __future__ import annotations

import httpx

from backend.app.config import Settings

TIMEOUT_SECONDS = 5.0


def get_employee(employee_id: str, settings: Settings) -> dict:
    resp = httpx.get(f"{settings.workday_api_url}/employees/{employee_id}", timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def get_leave_balance(employee_id: str, settings: Settings) -> dict:
    resp = httpx.get(
        f"{settings.workday_api_url}/employees/{employee_id}/leave-balance", timeout=TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    return resp.json()


def get_benefits(employee_id: str, settings: Settings) -> dict:
    resp = httpx.get(f"{settings.workday_api_url}/employees/{employee_id}/benefits", timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()

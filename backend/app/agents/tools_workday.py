"""
LangChain tools wrapping the Workday client.

Each tool is built per-request via `build_workday_tools(employee_id,
settings)` rather than as free-standing @tool functions, so the LLM never
has to supply (and can't hallucinate) an employee_id argument -- the current
employee is a fact of who's asking, established once by the caller (see
routes_chat.py / the Streamlit "acting as" selector), not something the
model should be guessing. The tools take no arguments; the LLM just decides
WHICH tool to call.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from backend.app.agents.retry import with_retry
from backend.app.clients import workday_client
from backend.app.config import Settings


def build_workday_tools(employee_id: str, settings: Settings) -> list[StructuredTool]:
    def _get_employee() -> dict:
        """Look up the current employee's profile: name, country, department, tenure, employment type."""
        return with_retry(workday_client.get_employee, employee_id, settings)

    def _get_leave_balance() -> dict:
        """Look up the current employee's vacation/leave balance: total, used, and available days."""
        return with_retry(workday_client.get_leave_balance, employee_id, settings)

    def _get_benefits() -> dict:
        """Look up the current employee's enrolled benefits: health plan tier and covered dependents."""
        return with_retry(workday_client.get_benefits, employee_id, settings)

    return [
        StructuredTool.from_function(_get_employee, name="get_employee"),
        StructuredTool.from_function(_get_leave_balance, name="get_leave_balance"),
        StructuredTool.from_function(_get_benefits, name="get_benefits"),
    ]

"""
LangChain tools wrapping the ServiceNow client.

Unlike the Workday tools, `create_ticket` DOES take LLM-supplied arguments
(short_description, description, category, urgency) -- the model needs to
compose the ticket content from the conversation. `requested_by` is still
bound via closure to the current employee_id for the same reason as
tools_workday.py: it shouldn't be guessable/model-supplied.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from backend.app.agents.retry import with_retry
from backend.app.clients import servicenow_client
from backend.app.config import Settings


class CreateTicketArgs(BaseModel):
    short_description: str = Field(description="A one-line summary of the issue, e.g. 'VPN will not connect'")
    description: str = Field(
        description="Full details: what the employee reported, and which troubleshooting steps were already tried"
    )
    category: str = Field(default="General", description="e.g. 'Network > VPN', 'Access > Account Lockout'")
    urgency: str = Field(default="Medium", description="Low, Medium, or High")


def build_servicenow_tools(employee_id: str, settings: Settings) -> list[StructuredTool]:
    def _create_ticket(short_description: str, description: str, category: str = "General", urgency: str = "Medium") -> dict:
        """Create an IT support ticket in ServiceNow for an issue that could not be resolved via self-service troubleshooting."""
        return with_retry(
            servicenow_client.create_ticket,
            short_description=short_description,
            description=description,
            category=category,
            urgency=urgency,
            requested_by=employee_id,
            settings=settings,
        )

    def _get_ticket_status(ticket_id: str) -> dict:
        """Look up the current status of a previously created IT ticket by its ID (e.g. 'INC0010023')."""
        return with_retry(servicenow_client.get_ticket, ticket_id, settings)

    return [
        StructuredTool.from_function(_create_ticket, name="create_ticket", args_schema=CreateTicketArgs),
        StructuredTool.from_function(_get_ticket_status, name="get_ticket_status"),
    ]

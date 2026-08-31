"""Pydantic request/response models for the API. Kept in one file since
there are few enough to scan at a glance."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class TraceStepModel(BaseModel):
    node: str
    event: str
    data: dict[str, Any]
    ts: str


class CitationModel(BaseModel):
    document_title: str
    source_path: str
    category: str
    parent_chunk_id: str
    excerpt: str


class RagQueryRequest(BaseModel):
    query: str
    category: str | None = None  # optional manual override of metadata routing
    country: str | None = None


class RagQueryResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    trace: list[TraceStepModel] | None = None


class ChatRequest(BaseModel):
    session_id: str
    message: str
    employee_id: str | None = None  # who is "logged in" for this session -- see Streamlit's "acting as" selector


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[CitationModel]
    intent: str | None = None
    trace: list[TraceStepModel] | None = None


class ChatHistoryMessage(BaseModel):
    role: str
    content: str
    trace: list[TraceStepModel] | None = None


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatHistoryMessage]

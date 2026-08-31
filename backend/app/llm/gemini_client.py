"""
Thin wrapper around the Gemini chat model.

We use LangChain's `ChatGoogleGenerativeAI` here (not a raw google-genai
call) because two things downstream genuinely need LangChain's abstraction
rather than being hidden by it: `.bind_tools()` for agent tool-calling
(agents/*.py) and LangGraph's message-based state. For simple one-shot
prompt -> text calls (query rewriting, RAG answer generation, etc.) we still
go through the same client via `generate_text()` so there's exactly one
LLM entry point in the whole codebase.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.app.config import Settings

_chat_model: ChatGoogleGenerativeAI | None = None


def get_chat_model(settings: Settings) -> ChatGoogleGenerativeAI:
    """Cached singleton -- avoids reconstructing the client on every call.
    Fine because get_settings() itself is a process-wide singleton (config.py),
    so there's only ever one Settings configuration to build a client from."""
    global _chat_model
    if _chat_model is None:
        _chat_model = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.2,
        )
    return _chat_model


def generate_text(prompt: str, settings: Settings) -> str:
    model = get_chat_model(settings)
    response = model.invoke([HumanMessage(content=prompt)])
    return response.content

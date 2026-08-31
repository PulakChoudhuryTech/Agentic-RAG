"""
Streamlit UI: a thin HTTP client over the FastAPI backend.

Deliberately talks to the backend over HTTP (httpx), not via direct Python
import, to mirror a real deployment boundary (UI process vs. API process)
and avoid two processes accidentally sharing mutable state.

Two tabs:
  - Chat: the full agentic assistant (POST /chat), with an "acting as"
    employee selector and an expandable debug trace per turn.
  - RAG Inspector: hits POST /rag/query directly (bypassing agent routing)
    so you can study retrieval behavior in isolation, stage by stage.
"""

from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8000")

DEMO_EMPLOYEES = {
    "E1001 - Priya Sharma (India, Engineering)": "E1001",
    "E1002 - James Whitfield (US, Sales)": "E1002",
    "E1003 - Wei Chen (India, Engineering, 3mo tenure)": "E1003",
    "E1004 - Fatima Al-Sayed (UK, Marketing)": "E1004",
    "E1005 - Diego Alvarez (US, IT, part-time)": "E1005",
}

st.set_page_config(page_title="Enterprise Employee AI Assistant", layout="wide")
st.title("Enterprise Employee AI Assistant")
st.caption("Local learning project: RAG + agentic AI. Not for production use.")


def render_trace(trace: list[dict] | None) -> None:
    if not trace:
        st.caption("No trace available (DEBUG_MODE is off, or nothing was recorded).")
        return
    for step in trace:
        with st.expander(f"[{step['node']}] {step['event']}", expanded=False):
            st.json(step["data"])


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown("**Sources:**")
    for c in citations:
        st.markdown(f"- *{c['document_title']}* (`{c['source_path']}`) — {c['excerpt']}...")


def sidebar_config() -> None:
    st.sidebar.header("Current configuration")
    st.sidebar.caption("Toggled via .env + backend restart, shown here read-only.")
    try:
        config = httpx.get(f"{BACKEND_URL}/config", timeout=5.0).json()
        for key, value in config.items():
            st.sidebar.write(f"**{key}**: `{value}`")
    except httpx.HTTPError as exc:
        st.sidebar.error(f"Could not reach backend at {BACKEND_URL}: {exc}")


tab_chat, tab_rag = st.tabs(["Chat (agentic assistant)", "RAG Inspector"])

with tab_chat:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of (role, content, citations, trace)

    col1, col2 = st.columns([3, 1])
    with col2:
        employee_label = st.selectbox("Acting as", list(DEMO_EMPLOYEES.keys()))
        employee_id = DEMO_EMPLOYEES[employee_label]
        st.caption(f"session_id: `{st.session_state.session_id[:8]}...`")
        if st.button("New session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.rerun()

    with col1:
        for role, content, citations, trace in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(content)
                if role == "assistant":
                    render_citations(citations)
                    render_trace(trace)

        user_message = st.chat_input("Ask about HR/IT/travel policy, your Workday data, or report an issue...")
        if user_message:
            st.session_state.chat_history.append(("user", user_message, [], None))
            with st.chat_message("user"):
                st.markdown(user_message)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        resp = httpx.post(
                            f"{BACKEND_URL}/chat",
                            json={
                                "session_id": st.session_state.session_id,
                                "message": user_message,
                                "employee_id": employee_id,
                            },
                            timeout=60.0,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        answer = data["answer"]
                        citations = data.get("citations") or []
                        trace = data.get("trace")
                        st.markdown(answer)
                        if data.get("intent"):
                            st.caption(f"Routed to: `{data['intent']}`")
                        render_citations(citations)
                        render_trace(trace)
                        st.session_state.chat_history.append(("assistant", answer, citations, trace))
                    except httpx.HTTPError as exc:
                        st.error(f"Request failed: {exc}")

with tab_rag:
    st.subheader("RAG pipeline inspector")
    st.caption("Bypasses agent routing -- calls the RAG pipeline directly so you can study retrieval in isolation.")

    query = st.text_input("Query", placeholder="What is the parental leave policy in India?")
    col_a, col_b = st.columns(2)
    with col_a:
        category_override = st.selectbox("Category filter (optional)", ["(auto)", "hr", "it", "travel"])
    with col_b:
        country_override = st.text_input("Country filter (optional, e.g. IN, US, GB, DE)")

    if st.button("Run RAG query") and query:
        payload = {"query": query}
        if category_override != "(auto)":
            payload["category"] = category_override
        if country_override:
            payload["country"] = country_override

        with st.spinner("Running pipeline..."):
            try:
                resp = httpx.post(f"{BACKEND_URL}/rag/query", json=payload, timeout=60.0)
                resp.raise_for_status()
                data = resp.json()
                st.markdown("### Answer")
                st.markdown(data["answer"])
                render_citations(data.get("citations") or [])
                st.markdown("### Debug trace (every pipeline stage)")
                render_trace(data.get("trace"))
            except httpx.HTTPError as exc:
                st.error(f"Request failed: {exc}")

sidebar_config()

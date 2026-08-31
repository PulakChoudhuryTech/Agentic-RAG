-- Stores multi-turn chat history plus the full debug trace for each assistant
-- turn, so past sessions stay inspectable from the Streamlit UI's history view
-- even after the process restarts (LangGraph's MemorySaver alone is in-memory only).
CREATE TABLE chat_messages (
    id           BIGSERIAL PRIMARY KEY,
    session_id   TEXT NOT NULL,
    role         TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content      TEXT NOT NULL,
    trace        JSONB,               -- full debug trace for this turn (null for role='user')
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX chat_messages_session_idx ON chat_messages (session_id, created_at);

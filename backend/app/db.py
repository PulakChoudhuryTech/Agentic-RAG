"""
Plain psycopg connection pool. Deliberately not an ORM (no SQLAlchemy models) --
the point of this project is to see the actual SQL in rag/vector_store.py and
rag/keyword_search.py, not have it generated for you.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from backend.app.config import get_settings

_pool: ConnectionPool | None = None


def _configure_connection(conn: psycopg.Connection) -> None:
    register_vector(conn)


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=5,
            configure=_configure_connection,
            open=True,
        )
        # Explicitly close on normal interpreter exit -- without this, the
        # pool's background threads get torn down implicitly during
        # interpreter finalization (relevant for short-lived scripts like
        # db/migrate.py, ingestion/ingest.py, and the eval CLI, not the
        # long-running uvicorn server), which some Python versions report
        # as a noisy but harmless "cannot join thread at interpreter
        # shutdown" error.
        atexit.register(_pool.close)
    return _pool


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    with get_pool().connection() as conn:
        yield conn

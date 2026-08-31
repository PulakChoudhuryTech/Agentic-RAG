"""
Tiny hand-written migration runner.

We deliberately don't use Alembic here: for ~6 SQL files, Alembic's revision
graph / autogenerate machinery is its own abstraction layer to learn, and it
would hide the plain SQL this project is meant to teach. This script does the
same essential job in ~40 lines:

1. Ensure a `schema_migrations` table exists.
2. Look at db/migrations/*.sql, sorted by filename (hence the NNN_ prefixes).
3. Apply any file whose name isn't already recorded in schema_migrations,
   each inside its own transaction.

Run directly: `python db/migrate.py`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (check your .env file)")
    return url


def ensure_migrations_table(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version     TEXT PRIMARY KEY,
            applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    conn.commit()


def already_applied(conn: psycopg.Connection) -> set[str]:
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {row[0] for row in rows}


def apply_migration(conn: psycopg.Connection, path: Path) -> None:
    sql = path.read_text()
    with conn.transaction():
        conn.execute(sql)
        conn.execute(
            "INSERT INTO schema_migrations (version) VALUES (%s)", (path.name,)
        )
    print(f"applied {path.name}")


def run_migrations() -> None:
    database_url = get_database_url()
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not migration_files:
        print(f"no migration files found in {MIGRATIONS_DIR}")
        return

    with psycopg.connect(database_url, autocommit=False) as conn:
        ensure_migrations_table(conn)
        applied = already_applied(conn)

        pending = [f for f in migration_files if f.name not in applied]
        if not pending:
            print("database is up to date, nothing to apply")
            return

        for path in pending:
            apply_migration(conn, path)

    print(f"done: applied {len(pending)} migration(s)")


if __name__ == "__main__":
    try:
        run_migrations()
    except Exception as exc:  # noqa: BLE001 - top-level CLI error reporting
        print(f"migration failed: {exc}", file=sys.stderr)
        sys.exit(1)

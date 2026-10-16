"""Checkpointer factory for persisting conversation thread state."""

from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from legalrag.core.config import AppConfig


def build_checkpointer(app_config: AppConfig) -> Any:
    """Build checkpointer based on configuration backend."""
    backend = app_config.graph.checkpointer.backend

    if backend == "memory":
        return MemorySaver()

    elif backend == "sqlite":
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver
            db_path = app_config.graph.checkpointer.sqlite_path
            # Open persistent connection
            conn = sqlite3.connect(db_path, check_same_thread=False)
            return SqliteSaver(conn)
        except Exception:
            return MemorySaver()

    elif backend == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            pg_url = app_config.graph.checkpointer.postgres_url
            if pg_url:
                saver = PostgresSaver.from_conn_string(pg_url)
                saver.setup()
                return saver
        except Exception:
            return MemorySaver()

    return MemorySaver()

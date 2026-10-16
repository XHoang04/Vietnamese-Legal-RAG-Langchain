"""SQLite-backed response cache for LLM invocations."""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LLMCache:
    """Thread-safe SQLite persistent cache for LLM responses."""

    def __init__(self, db_path: str | Path = "data/cache/llm_cache.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=30.0)

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    role TEXT,
                    model TEXT,
                    created_at TIMESTAMP NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def compute_key(
        role: str,
        provider: str,
        model: str,
        params: dict[str, Any],
        messages: list[Any],
    ) -> str:
        """Deterministic sha256 cache key computed from call arguments."""
        normalized_messages = []
        for m in messages:
            if hasattr(m, "content"):
                normalized_messages.append({"role": getattr(m, "type", "user"), "content": m.content})
            elif isinstance(m, dict):
                normalized_messages.append(m)
            else:
                normalized_messages.append(str(m))

        payload = {
            "role": role,
            "provider": provider,
            "model": model,
            "params": params,
            "messages": normalized_messages,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> str | None:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT response FROM llm_cache WHERE cache_key = ?", (cache_key,))
            row = cursor.fetchone()
            return row[0] if row else None

    def set(self, cache_key: str, response: str, role: str = "", model: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache (cache_key, response, role, model, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cache_key, response, role, model, now),
            )
            conn.commit()

"""SQLite LLM response cache (SDD 3.8.3). Never invalidated automatically."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


def cache_key(provider, model, digest, system, messages, schema, params) -> str:
    blob = json.dumps([provider, model, digest, system, messages, schema, params], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class LLMCache:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS llm (key TEXT PRIMARY KEY, value TEXT)")

    def get(self, key: str) -> dict | None:
        with self._lock, sqlite3.connect(self.path) as c:
            row = c.execute("SELECT value FROM llm WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._lock, sqlite3.connect(self.path) as c:
            c.execute("INSERT OR REPLACE INTO llm VALUES (?, ?)", (key, json.dumps(value, default=str)))

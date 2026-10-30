"""Embedding cache keyed by (model_id, revision, sha256(text)) in SQLite (SDD 3.3.3)."""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import numpy as np


class EmbeddingCache:
    def __init__(self, path: Path | None):
        self.path = path
        self._mem: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._conn() as c:
                c.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)")

    def _conn(self):
        return sqlite3.connect(self.path)

    @staticmethod
    def key(model_id: str, revision: str | None, text: str) -> str:
        return hashlib.sha256(f"{model_id}|{revision}|{hashlib.sha256(text.encode()).hexdigest()}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out = {k: self._mem[k] for k in keys if k in self._mem}
        missing = [k for k in keys if k not in out]
        if missing and self.path:
            with self._lock, self._conn() as c:
                for i in range(0, len(missing), 500):
                    part = missing[i:i + 500]
                    q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})"
                    for k, blob in c.execute(q, part):
                        out[k] = self._mem[k] = np.frombuffer(blob, dtype=np.float32)
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        for k, v in items.items():
            self._mem[k] = v.astype(np.float32)
        if self.path and items:
            with self._lock, self._conn() as c:
                c.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?)",
                              [(k, len(v), v.astype(np.float32).tobytes()) for k, v in items.items()])

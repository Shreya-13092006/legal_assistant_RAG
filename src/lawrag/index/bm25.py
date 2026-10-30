"""In-process Okapi BM25 index over chunk embed_text (SDD 3.4.4).

A small, dependency-free implementation is used instead of `bm25s`: the corpus is < 20k chunks,
scores are deterministic, and the index is keyed by corpus hash + tokenizer version.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from lawrag.index.tokenizer import TOKENIZER_VERSION, LegalTokenizer


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer: LegalTokenizer | None = None):
        self.k1, self.b = k1, b
        self.tok = tokenizer or LegalTokenizer()
        self.ids: list[str] = []
        self.tfs: list[dict[str, int]] = []
        self.lens: list[int] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0
        self.key: str | None = None

    def build(self, docs: list[tuple[str, str]], key: str | None = None) -> "BM25Index":
        self.ids, self.tfs, self.lens, self.df = [], [], [], Counter()
        for doc_id, text in sorted(docs):
            toks = self.tok(text)
            tf = Counter(toks)
            self.ids.append(doc_id)
            self.tfs.append(dict(tf))
            self.lens.append(len(toks))
            self.df.update(tf.keys())
        self.avgdl = sum(self.lens) / len(self.lens) if self.lens else 0.0
        self.key = key
        return self

    def idf(self, term: str) -> float:
        n, df = len(self.ids), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        q = self.tok(query)
        if not q or not self.ids:
            return []
        scores = []
        for i, tf in enumerate(self.tfs):
            if allowed is not None and self.ids[i] not in allowed:
                continue
            s = 0.0
            dl = self.lens[i]
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                s += self.idf(term) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            if s > 0:
                scores.append((self.ids[i], s))
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:k]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "bm25.json").write_text(json.dumps({
            "key": self.key, "tokenizer": TOKENIZER_VERSION, "k1": self.k1, "b": self.b, "ids": self.ids,
            "tfs": self.tfs, "lens": self.lens, "df": dict(self.df), "avgdl": self.avgdl}), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "BM25Index | None":
        p = directory / "bm25.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("tokenizer") != TOKENIZER_VERSION:
            return None
        idx = cls(d["k1"], d["b"])
        idx.ids, idx.tfs, idx.lens = d["ids"], d["tfs"], d["lens"]
        idx.df, idx.avgdl, idx.key = Counter(d["df"]), d["avgdl"], d["key"]
        return idx

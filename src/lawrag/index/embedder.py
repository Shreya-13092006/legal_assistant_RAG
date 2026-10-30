"""Embedders (SDD 3.3.3). `BAAI/bge-m3` via sentence-transformers is the initial candidate.

`hashing-<dim>` is a deterministic, dependency-free lexical embedder (signed feature hashing of
legal tokens and character trigrams). It exists so the full pipeline and tests run on a machine
without torch; it is NOT a semantic model and must not be used for reported E1 results.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from lawrag.config import EmbeddingCfg
from lawrag.errors import ConfigError
from lawrag.index.embed_cache import EmbeddingCache
from lawrag.index.tokenizer import LegalTokenizer


class Embedder(Protocol):
    model_id: str
    revision: str | None
    dim: int

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.model_id = f"hashing-{dim}"
        self.revision = "1"
        self.dim = dim
        self.tok = LegalTokenizer(stem=True)

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        feats = self.tok(text)
        low = re.sub(r"\s+", " ", text.lower())
        feats += [low[i:i + 4] for i in range(0, max(len(low) - 3, 0), 2)]
        for f in feats:
            h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        return _l2(np.stack([self._vec(t) for t in texts])) if texts else np.zeros((0, self.dim), np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, cfg: EmbeddingCfg):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigError(
                f"embedding model {cfg.model} needs the 'embed' extra: pip install torch "
                "--index-url https://download.pytorch.org/whl/cpu && pip install sentence-transformers") from exc
        torch.set_num_threads(cfg.threads)
        t0 = time.perf_counter()
        self.model = SentenceTransformer(cfg.model, revision=cfg.revision, device="cpu")
        self.load_seconds = time.perf_counter() - t0
        self.model_id = cfg.model
        self.revision = cfg.revision
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.batch_size = cfg.batch_size
        self.normalize = cfg.normalize

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        # bge-m3 dense retrieval uses no query instruction prefix (VERIFY A-12)
        return np.asarray(self.model.encode(texts, batch_size=self.batch_size,
                                            normalize_embeddings=self.normalize,
                                            show_progress_bar=False), dtype=np.float32)


class CachedEmbedder:
    """Wraps an embedder with the persistent cache; only changed texts are recomputed."""

    def __init__(self, inner: Embedder, cache: EmbeddingCache):
        self.inner, self.cache = inner, cache
        self.model_id, self.revision, self.dim = inner.model_id, inner.revision, inner.dim
        self.hits = self.misses = 0

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        keys = [EmbeddingCache.key(self.model_id, self.revision, t) for t in texts]
        got = self.cache.get_many(keys)
        todo = [(i, t) for i, (k, t) in enumerate(zip(keys, texts)) if k not in got]
        self.hits += len(texts) - len(todo)
        self.misses += len(todo)
        if todo:
            vecs = self.inner.encode([t for _, t in todo], is_query=is_query)
            new = {keys[i]: vecs[j] for j, (i, _) in enumerate(todo)}
            self.cache.put_many(new)
            got.update(new)
        return np.stack([got[k] for k in keys]) if keys else np.zeros((0, self.dim), np.float32)


def make_embedder(cfg: EmbeddingCfg, cache_dir: Path | None = None) -> CachedEmbedder:
    m = re.fullmatch(r"hashing-(\d+)", cfg.model)
    inner: Embedder = HashingEmbedder(int(m.group(1))) if m else SentenceTransformerEmbedder(cfg)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", cfg.model)
    cache = EmbeddingCache(cache_dir / f"{safe}.sqlite" if cache_dir else None)
    return CachedEmbedder(inner, cache)

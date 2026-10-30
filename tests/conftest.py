from __future__ import annotations

import pytest

from lawrag.config import load_config
from lawrag.graph.loader import build_from_seed
from lawrag.graph.store import MemoryGraphStore
from lawrag.index.embedder import make_embedder
from lawrag.llm.adapters.fake import FakeAdapter
from lawrag.services.container import build_deps


@pytest.fixture(scope="session")
def cfg():
    return load_config(["providers/fake.yaml"])


@pytest.fixture(scope="session")
def seed_data(cfg):
    return build_from_seed(cfg, make_embedder(cfg.embedding))


@pytest.fixture(scope="session")
def store(seed_data):
    return MemoryGraphStore(seed_data)


@pytest.fixture()
def make_deps(cfg, seed_data):
    """Deps over the seed graph; pass `script` to control the FakeLLM per role."""
    def _make(script=None, overrides: dict | None = None, overlays: list[str] | None = None):
        c = load_config(["providers/fake.yaml"] + (overlays or []), overrides=overrides) if (overrides or overlays) else cfg
        return build_deps(c, store=MemoryGraphStore(seed_data), llm_adapters={"fake": FakeAdapter(script)})
    return _make


@pytest.fixture()
def deps(make_deps):
    return make_deps()

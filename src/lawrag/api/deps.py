"""Process-wide singletons for the API (embedder, BM25, driver loaded once at startup)."""
from __future__ import annotations

from dataclasses import dataclass

from lawrag.services.container import Deps
from lawrag.services.law_service import LawService
from lawrag.services.qa_service import QAService


@dataclass
class AppState:
    deps: Deps
    qa: QAService
    law: LawService


_state: AppState | None = None


def set_state(deps: Deps) -> AppState:
    global _state
    _state = AppState(deps=deps, qa=QAService(deps), law=LawService(deps))
    return _state


def get_state() -> AppState:
    if _state is None:
        raise RuntimeError("API dependencies not initialised")
    return _state

"""Budget guard for hosted providers (SDD 3.8.3). Local providers cost 0 USD."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from lawrag.errors import BudgetExceeded

LOCAL_PROVIDERS = {"ollama", "fake"}


class BudgetGuard:
    def __init__(self, limit_usd: float, prices: dict[str, dict[str, float]], state_path: Path | None = None):
        self.limit = float(limit_usd)
        self.prices = prices
        self.state_path = state_path
        self._lock = threading.Lock()
        self.spent = 0.0
        if state_path and state_path.exists():
            self.spent = float(json.loads(state_path.read_text()).get("spent_usd", 0.0))

    def remaining(self) -> float:
        return max(self.limit - self.spent, 0.0)

    def check(self, provider: str) -> None:
        if provider not in LOCAL_PROVIDERS and self.spent >= self.limit:
            raise BudgetExceeded(f"hosted LLM budget of {self.limit:.2f} USD reached "
                                 f"(spent {self.spent:.4f}); raise LAWRAG_BUDGET_USD to continue")

    def price(self, provider: str, model: str, tin: int, tout: int) -> float:
        if provider in LOCAL_PROVIDERS:
            return 0.0
        p = self.prices.get(f"{provider}/{model}")
        if p is None:
            raise BudgetExceeded(f"no price configured for {provider}/{model}; add it to llm.prices_per_mtok")
        return tin / 1e6 * p["input"] + tout / 1e6 * p["output"]

    def estimate(self, provider: str, model: str, tin: int, tout: int, calls: int) -> float:
        return self.price(provider, model, tin, tout) * calls

    def record(self, provider: str, model: str, tin: int, tout: int) -> float:
        cost = self.price(provider, model, tin, tout)
        if cost:
            with self._lock:
                self.spent += cost
                if self.state_path:
                    self.state_path.parent.mkdir(parents=True, exist_ok=True)
                    self.state_path.write_text(json.dumps({"spent_usd": self.spent}))
        return cost

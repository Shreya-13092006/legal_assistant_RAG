"""Provider-agnostic LLM client (SDD 3.8.1): roles, response cache, budget guard, structured output
with exactly one repair attempt."""
from __future__ import annotations

import contextvars
import json
import re
import time
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from lawrag.config import LLMCfg, RoleCfg
from lawrag.errors import LLMOutputError
from lawrag.llm.budget import BudgetGuard
from lawrag.llm.cache import LLMCache, cache_key


class Message(BaseModel):
    role: str
    content: str


class LLMResult(BaseModel):
    text: str
    parsed: Any = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    latency_ms: float = 0.0
    model_digest: str | None = None


class Adapter(Protocol):
    provider: str
    model: str

    def raw_complete(self, system: str, messages: list[Message], *, json_schema: dict | None,
                     temperature: float, max_tokens: int, seed: int | None) -> LLMResult: ...


_PROMPT_DIR = Path(__file__).parent / "prompts"

# per-request usage sink: the workflow collects (role, tokens, cost, cached) for its trace
_usage_sink: contextvars.ContextVar[list | None] = contextvars.ContextVar("llm_usage_sink", default=None)


def start_usage_sink() -> list:
    sink: list = []
    _usage_sink.set(sink)
    return sink


def _record_usage(role: str, res: "LLMResult") -> None:
    sink = _usage_sink.get()
    if sink is not None:
        sink.append({"role": role, "in": res.input_tokens, "out": res.output_tokens, "cost": res.cost_usd,
                     "cached": res.cached, "ms": res.latency_ms})


def load_prompt(name: str) -> str:
    return (_PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


def extract_json(text: str) -> Any:
    """Parse JSON from model output, tolerating code fences and leading prose."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
        raise


class LLMClient:
    """One role-bound client: adapter + cache + budget guard."""

    def __init__(self, role: str, adapter: Adapter, cache: LLMCache | None, budget: BudgetGuard,
                 cfg: RoleCfg):
        self.role, self.adapter, self.cache, self.budget, self.cfg = role, adapter, cache, budget, cfg
        self.provider, self.model = adapter.provider, adapter.model

    def complete(self, system: str, messages: list[Message], *, schema: type[BaseModel] | None = None,
                 temperature: float | None = None, max_tokens: int | None = None,
                 seed: int | None = None) -> LLMResult:
        temperature = self.cfg.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.cfg.max_tokens
        json_schema = schema.model_json_schema() if schema else None
        key = cache_key(self.provider, self.model, getattr(self.adapter, "digest", None), system,
                        [m.model_dump() for m in messages], json_schema,
                        {"temperature": temperature, "max_tokens": max_tokens, "seed": seed})
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                res = LLMResult(**hit, cached=True)
                res.cost_usd = 0.0
                if schema:
                    res.parsed = schema.model_validate(res.parsed)
                _record_usage(self.role, res)
                return res
        self.budget.check(self.provider)
        t0 = time.perf_counter()
        res = self.adapter.raw_complete(system, messages, json_schema=json_schema, temperature=temperature,
                                        max_tokens=max_tokens, seed=seed)
        res.latency_ms = (time.perf_counter() - t0) * 1000
        res.cost_usd = self.budget.record(self.provider, self.model, res.input_tokens, res.output_tokens)
        if schema:
            res = self._parse(res, schema, system, messages, temperature, max_tokens, seed)
        if self.cache:
            dump = res.model_dump(exclude={"cached"})
            dump["parsed"] = res.parsed.model_dump() if isinstance(res.parsed, BaseModel) else res.parsed
            self.cache.put(key, dump)
        _record_usage(self.role, res)
        return res

    def _parse(self, res: LLMResult, schema, system, messages, temperature, max_tokens, seed) -> LLMResult:
        try:
            res.parsed = schema.model_validate(extract_json(res.text))
            return res
        except (json.JSONDecodeError, ValidationError, TypeError) as first:
            repair = messages + [Message(role="assistant", content=res.text), Message(
                role="user", content=f"Your output was not valid JSON for the schema ({str(first)[:300]}). "
                                     "Reply with ONLY the corrected JSON object.")]
            self.budget.check(self.provider)
            res2 = self.adapter.raw_complete(system, repair, json_schema=schema.model_json_schema(),
                                             temperature=temperature, max_tokens=max_tokens, seed=seed)
            res2.cost_usd = res.cost_usd + self.budget.record(self.provider, self.model, res2.input_tokens,
                                                              res2.output_tokens)
            try:
                res2.parsed = schema.model_validate(extract_json(res2.text))
                return res2
            except (json.JSONDecodeError, ValidationError, TypeError) as second:
                raise LLMOutputError(f"{self.role}: invalid structured output after one repair: {second}") from second


class LLMRegistry:
    """Builds one LLMClient per role from config; shares the cache and budget guard."""

    def __init__(self, cfg: LLMCfg, root: Path, adapters: dict[str, Adapter] | None = None):
        self.cfg = cfg
        self.cache = LLMCache(root / cfg.cache.path) if cfg.cache.enabled else None
        self.budget = BudgetGuard(cfg.budget_usd, cfg.prices_per_mtok, root / "cache" / "budget.json")
        self._override = adapters or {}
        self._clients: dict[str, LLMClient] = {}

    def _adapter(self, rc: RoleCfg) -> Adapter:
        from lawrag.llm.adapters import make_adapter
        if rc.provider in self._override:
            return self._override[rc.provider]
        return make_adapter(rc, self.cfg)

    def get(self, role: str) -> LLMClient:
        if role not in self._clients:
            rc = self.cfg.role(role)
            self._clients[role] = LLMClient(role, self._adapter(rc), self.cache, self.budget, rc)
        return self._clients[role]

    def describe(self) -> dict:
        return {r: {"provider": c.provider, "model": c.model} for r, c in
                ((r, self.cfg.role(r)) for r in self.cfg.roles)}

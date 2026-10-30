"""Local Ollama adapter over plain HTTP (no extra dependency)."""
from __future__ import annotations

import httpx

from lawrag.errors import LLMError, LLMTimeout


class OllamaAdapter:
    provider = "ollama"

    def __init__(self, model: str, host: str, timeout_s: float):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout_s
        self.digest: str | None = None

    def _digest(self) -> str | None:
        if self.digest is None:
            try:
                r = httpx.post(f"{self.host}/api/show", json={"model": self.model}, timeout=10)
                self.digest = (r.json().get("details") or {}).get("parent_model") or r.json().get("modified_at")
            except Exception:
                self.digest = "unknown"
        return self.digest

    def ping(self) -> bool:
        try:
            return httpx.get(f"{self.host}/api/tags", timeout=5).status_code == 200
        except httpx.HTTPError:
            return False

    def raw_complete(self, system, messages, *, json_schema, temperature, max_tokens, seed):
        from lawrag.llm.client import LLMResult
        body = {
            "model": self.model, "stream": False,
            "messages": [{"role": "system", "content": system}] + [m.model_dump() for m in messages],
            "options": {"temperature": temperature, "num_predict": max_tokens,
                        **({"seed": seed} if seed is not None else {})},
        }
        if json_schema:
            body["format"] = json_schema
        try:
            r = httpx.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"ollama {self.model} timed out after {self.timeout}s") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc
        d = r.json()
        return LLMResult(text=d.get("message", {}).get("content", ""), input_tokens=d.get("prompt_eval_count", 0),
                         output_tokens=d.get("eval_count", 0), model_digest=self._digest())

"""OpenAI-compatible chat-completions adapter (local servers such as llama.cpp / LM Studio / vLLM)."""
from __future__ import annotations

import os

import httpx

from lawrag.errors import LLMError, LLMTimeout


class OpenAICompatAdapter:
    provider = "openai_compat"

    def __init__(self, model: str, base_url: str, timeout_s: float):
        self.base_url, self.timeout = base_url.rstrip("/"), timeout_s
        self._auto = not model or model == "auto"
        self.model = self._detect_model() if self._auto else model
        self.digest = self.model

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if "ngrok" in self.base_url.lower():
            headers["ngrok-skip-browser-warning"] = "true"
        if os.environ.get("OPENAI_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
        return headers

    def _detect_model(self) -> str:
        try:
            r = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=min(self.timeout, 10.0))
            r.raise_for_status()
            d = r.json()
            for item in d.get("data") or []:
                if item.get("id"):
                    return item["id"]
            for item in d.get("models") or []:
                name = item.get("model") or item.get("name")
                if name:
                    return name
        except httpx.HTTPError:
            pass
        return "auto"

    def ping(self) -> bool:
        try:
            return httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=5).status_code == 200
        except httpx.HTTPError:
            return False

    def raw_complete(self, system, messages, *, json_schema, temperature, max_tokens, seed):
        from lawrag.llm.client import LLMResult
        if self._auto and self.model == "auto":
            self.model = self._detect_model()
            self.digest = self.model
        body = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}] + [m.model_dump() for m in messages]}
        if seed is not None:
            body["seed"] = seed
        if json_schema:
            body["response_format"] = {"type": "json_object"}
        try:
            r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=self._headers(), timeout=self.timeout)
            r.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMTimeout(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"openai-compatible request failed: {exc}") from exc
        d = r.json()
        usage = d.get("usage") or {}
        return LLMResult(text=d["choices"][0]["message"]["content"], input_tokens=usage.get("prompt_tokens", 0),
                         output_tokens=usage.get("completion_tokens", 0))

"""OpenAI-compatible chat-completions adapter (local servers such as llama.cpp / LM Studio / vLLM)."""
from __future__ import annotations

import os

import httpx

from lawrag.errors import LLMError, LLMTimeout


class OpenAICompatAdapter:
    provider = "openai_compat"

    def __init__(self, model: str, base_url: str, timeout_s: float):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout_s
        self.digest = model

    def raw_complete(self, system, messages, *, json_schema, temperature, max_tokens, seed):
        from lawrag.llm.client import LLMResult
        body = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}] + [m.model_dump() for m in messages]}
        if seed is not None:
            body["seed"] = seed
        if json_schema:
            body["response_format"] = {"type": "json_object"}
        headers = {}
        if os.environ.get("OPENAI_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
        try:
            r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
            r.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMTimeout(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"openai-compatible request failed: {exc}") from exc
        d = r.json()
        usage = d.get("usage") or {}
        return LLMResult(text=d["choices"][0]["message"]["content"], input_tokens=usage.get("prompt_tokens", 0),
                         output_tokens=usage.get("completion_tokens", 0))

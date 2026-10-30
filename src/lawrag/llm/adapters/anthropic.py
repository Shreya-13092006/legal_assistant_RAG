"""Hosted Claude adapter via the official `anthropic` SDK (optional; used only under the budget cap).

Current Claude models reject sampling parameters, so `temperature`/`seed` are not sent; determinism
for experiments comes from the response cache.
"""
from __future__ import annotations

from lawrag.errors import ConfigError, LLMError, LLMTimeout


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(self, model: str, timeout_s: float):
        try:
            import anthropic
        except ImportError as exc:
            raise ConfigError("the hosted provider needs `pip install anthropic`") from exc
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=timeout_s)  # credentials from the environment
        self.model = model
        self.digest = model

    def raw_complete(self, system, messages, *, json_schema, temperature, max_tokens, seed):
        from lawrag.llm.client import LLMResult
        a = self._anthropic
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        kwargs = {"model": self.model, "max_tokens": max(max_tokens, 4096), "system": system, "messages": msgs}
        if json_schema:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": _strict(json_schema)}}
        try:
            resp = self.client.messages.create(**kwargs)
        except a.APITimeoutError as exc:
            raise LLMTimeout(str(exc)) from exc
        except a.RateLimitError as exc:
            raise LLMError(f"rate limited: {exc}") from exc
        except a.APIStatusError as exc:
            raise LLMError(f"anthropic API error {exc.status_code}: {exc.message}") from exc
        except a.APIConnectionError as exc:
            raise LLMError(f"anthropic connection error: {exc}") from exc
        if resp.stop_reason == "refusal":
            raise LLMError("the model declined the request (refusal)")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResult(text=text, input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
                         model_digest=resp.model)


def _strict(schema: dict) -> dict:
    """Structured outputs need additionalProperties: false on every object."""
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        return node
    import copy
    return walk(copy.deepcopy(schema))

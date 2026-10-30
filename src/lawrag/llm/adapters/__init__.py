from __future__ import annotations

from lawrag.config import LLMCfg, RoleCfg
from lawrag.errors import ConfigError


def make_adapter(rc: RoleCfg, cfg: LLMCfg):
    if rc.provider == "ollama":
        from lawrag.llm.adapters.ollama import OllamaAdapter
        return OllamaAdapter(rc.model, rc.base_url or cfg.ollama_host, cfg.timeout_s)
    if rc.provider == "anthropic":
        from lawrag.llm.adapters.anthropic import AnthropicAdapter
        return AnthropicAdapter(rc.model, cfg.timeout_s)
    if rc.provider == "openai_compat":
        from lawrag.llm.adapters.openai_compat import OpenAICompatAdapter
        if not rc.base_url:
            raise ConfigError("openai_compat provider needs base_url")
        return OpenAICompatAdapter(rc.model, rc.base_url, cfg.timeout_s)
    if rc.provider == "fake":
        from lawrag.llm.adapters.fake import FakeAdapter
        return FakeAdapter()
    raise ConfigError(f"unknown LLM provider {rc.provider}")

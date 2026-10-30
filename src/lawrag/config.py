"""Validated run configuration (SDD 3.8.2).

`load_config(overlays)` deep-merges configs/base.yaml with overlay files, expands ${ENV:default}
references, and validates the result. Unknown keys are an error. `config_hash` is the sha256 of the
canonical JSON of the resolved config, excluding secrets and paths.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lawrag.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectCfg(_Strict):
    name: str = "lawrag"
    cutoff: dict[str, str] = Field(default_factory=dict)
    bns_commencement: str = "2024-07-01"


class GraphBackendCfg(_Strict):
    backend: Literal["memory", "neo4j"] = "memory"
    snapshot_path: str = "data/processed/graph.json"
    seed_dir: str = "data/seed"


class Neo4jCfg(_Strict):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    database: str = "neo4j"


class CorpusCfg(_Strict):
    variant: Literal["text", "scan"] = "text"
    include_secondary: bool = False


class ChunkCfg(_Strict):
    max_tokens: int = Field(400, ge=50, le=4000)
    include_illustrations: bool = True
    section_summaries: bool = True


class EmbeddingCfg(_Strict):
    model: str = "hashing-512"
    revision: str | None = None
    batch_size: int = Field(16, ge=1)
    threads: int = Field(10, ge=1)
    normalize: bool = True
    cache_path: str = "cache/embeddings"


class TopK(_Strict):
    enabled: bool = True
    k: int = Field(30, ge=1, le=500)


class GraphCfg(_Strict):
    enabled: bool = True
    seeds: int = Field(5, ge=0)
    max_hops: int = Field(2, ge=0, le=4)
    max_fanout: int = Field(10, ge=1)
    max_expanded: int = Field(40, ge=1)
    hop_penalty: int = Field(5, ge=0)


class RetrievalCfg(_Strict):
    id_lookup: bool = True
    vector: TopK = TopK()
    bm25: TopK = TopK()
    graph: GraphCfg = GraphCfg()
    act_filter: bool = True


class FusionCfg(_Strict):
    method: Literal["rrf", "none"] = "rrf"
    k_rrf: int = Field(60, ge=1)
    top_n: int = Field(8, ge=1)
    weights: dict[str, float] = Field(default_factory=dict)
    illustration_weight: float = Field(0.7, ge=0, le=1)


class RerankCfg(_Strict):
    enabled: bool = False
    kind: Literal["llm", "crossencoder"] = "llm"
    top_in: int = 20
    top_out: int = 8


class ContextCfg(_Strict):
    budget_tokens: int = Field(3500, ge=200)


class GateCfg(_Strict):
    min_retriever_rank: int = Field(10, ge=1)


class WorkflowCfg(_Strict):
    max_retries: int = Field(1, ge=0, le=1)  # DECIDED: one bounded retry
    engine: Literal["langgraph", "python"] = "langgraph"
    gate: GateCfg = GateCfg()


class VerificationCfg(_Strict):
    deterministic: bool = True
    claim_judge: bool = True
    enforce: bool = True  # False = V0 baseline (generate -> finalize)
    tau_partial: float = Field(0.6, ge=0, le=1)


class AnalyzerCfg(_Strict):
    llm_fallback: bool = False


class RoleCfg(_Strict):
    provider: Literal["ollama", "anthropic", "openai_compat", "fake"] = "ollama"
    model: str = "llama3.2"
    temperature: float = 0.0
    max_tokens: int = 1024
    base_url: str | None = None


class LLMCacheCfg(_Strict):
    enabled: bool = True
    path: str = "cache/llm.sqlite"


class LLMCfg(_Strict):
    roles: dict[str, RoleCfg] = Field(default_factory=dict)
    cache: LLMCacheCfg = LLMCacheCfg()
    budget_usd: float = 5.0
    ollama_host: str = "http://localhost:11434"
    timeout_s: float = 180
    prices_per_mtok: dict[str, dict[str, float]] = Field(default_factory=dict)

    def role(self, name: str) -> RoleCfg:
        return self.roles.get(name) or self.roles.get("generator") or RoleCfg()


class ApiCfg(_Strict):
    timeout_s: float = 120


class LoggingCfg(_Strict):
    level: str = "INFO"
    json_format: bool = Field(True, alias="json")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class RunConfig(_Strict):
    project: ProjectCfg = ProjectCfg()
    acts: list[str] = Field(default_factory=lambda: ["IPC", "BNS"])
    act_aliases: dict[str, list[str]] = Field(default_factory=dict)
    graph: GraphBackendCfg = GraphBackendCfg()
    neo4j: Neo4jCfg = Neo4jCfg()
    corpus: CorpusCfg = CorpusCfg()
    chunking: ChunkCfg = ChunkCfg()
    embedding: EmbeddingCfg = EmbeddingCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    fusion: FusionCfg = FusionCfg()
    rerank: RerankCfg = RerankCfg()
    context: ContextCfg = ContextCfg()
    workflow: WorkflowCfg = WorkflowCfg()
    verification: VerificationCfg = VerificationCfg()
    analyzer: AnalyzerCfg = AnalyzerCfg()
    llm: LLMCfg = LLMCfg()
    api: ApiCfg = ApiCfg()
    logging: LoggingCfg = LoggingCfg()
    seed: int = 13

    @model_validator(mode="after")
    def _cross_field(self) -> "RunConfig":
        enabled = self.enabled_retrievers()
        if not enabled:
            raise ValueError("at least one retriever must be enabled")
        if self.fusion.method == "rrf" and len(enabled) < 2:
            raise ValueError("fusion.method = rrf requires >= 2 retrievers; use fusion.method = none")
        if self.fusion.method == "none" and len(enabled) > 1:
            raise ValueError("fusion.method = none allows exactly one retriever")
        if self.graph.backend == "neo4j" and not self.neo4j.uri:
            raise ValueError("neo4j.uri is required for the neo4j backend")
        return self

    def enabled_retrievers(self) -> list[str]:
        r = self.retrieval
        out = []
        if r.id_lookup:
            out.append("id_lookup")
        if r.vector.enabled:
            out.append("vector")
        if r.bm25.enabled:
            out.append("bm25")
        if r.graph.enabled:
            out.append("graph")
        return out

    def resolved(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    def config_hash(self) -> str:
        return config_hash(self)

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p


_SECRET_KEYS = re.compile(r"(password|api_key|secret|token)$", re.I)
_PATH_KEYS = {"snapshot_path", "seed_dir", "cache_path", "path", "uri", "ollama_host", "base_url"}


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in sorted(obj.items())
                if not _SECRET_KEYS.search(k) and k not in _PATH_KEYS}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def config_hash(cfg: RunConfig) -> str:
    canon = json.dumps(_strip(cfg.resolved()), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        whole = _ENV_RE.fullmatch(obj)
        if whole:
            val = os.environ.get(whole.group(1), whole.group(2) or "")
            return yaml.safe_load(val) if val and re.fullmatch(r"-?\d+(\.\d+)?", val) else val
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), obj)
    return obj


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_env_file(path: Path | None = None) -> None:
    """Minimal .env loader: sets variables that are not already set."""
    path = path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def load_config(overlays: list[str | Path] | None = None, overrides: dict | None = None,
                base: str | Path | None = None) -> RunConfig:
    load_env_file()
    data = _read_yaml(Path(base) if base else CONFIG_DIR / "base.yaml")
    for ov in overlays or []:
        p = Path(ov)
        if not p.is_absolute() and not p.exists():
            p = CONFIG_DIR / p
        data = deep_merge(data, _read_yaml(p))
    if overrides:
        data = deep_merge(data, overrides)
    data = _expand_env(data)
    try:
        return RunConfig.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError
        raise ConfigError(f"invalid configuration: {exc}") from exc

"""Run manifests for reproducibility (SDD 3.8.5)."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lawrag.config import PROJECT_ROOT, RunConfig


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_info() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=10)
        if sha.returncode != 0:
            return None, None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True,
                               text=True, timeout=10)
        return sha.stdout.strip(), bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def packages_hash() -> str:
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
                             timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    return hashlib.sha256(out.encode()).hexdigest()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def build_manifest(run_id: str, cfg: RunConfig, *, corpus_hash: str | None, graph_snapshot_hash: str | None,
                   audit_status: str | None, dataset: dict[str, Any] | None, models: dict[str, Any],
                   repeats: int = 1, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    sha, dirty = git_info()
    return {
        "run_id": run_id,
        "git_sha": sha,
        "git_dirty": dirty,
        "config_hash": cfg.config_hash(),
        "config_resolved": cfg.resolved(),
        "corpus_hash": corpus_hash,
        "graph_snapshot_hash": graph_snapshot_hash,
        "audit_status": audit_status,
        "dataset": dataset,
        "models": models,
        "environment": {"python": platform.python_version(), "os": platform.platform(),
                        "packages_sha256": packages_hash()},
        "seed": cfg.seed,
        "cache": "on" if cfg.llm.cache.enabled else "off",
        "repeats": repeats,
        "started": datetime.now(timezone.utc).isoformat(),
        "finished": None,
        "cost_usd": 0.0,
        **(extra or {}),
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

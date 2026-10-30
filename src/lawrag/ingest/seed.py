"""Load the hand-prepared M1 seed corpus (data/seed/*.yaml) into ActTrees, mappings and sources."""
from __future__ import annotations

from pathlib import Path

import yaml

from lawrag.ingest.tree_builder import build_tree
from lawrag.models.legal import ActTree, SourceRecord
from lawrag.models.mapping import MappingRecord


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_seed_trees(seed_dir: Path) -> list[ActTree]:
    trees = []
    for name in ("ipc.yaml", "bns.yaml"):
        p = seed_dir / name
        if p.exists():
            d = _yaml(p)
            trees.append(build_tree(d["act"], d.get("chapters", []), d["source_id"]))
    return trees


def load_seed_sources(seed_dir: Path) -> list[SourceRecord]:
    p = seed_dir / "sources.yaml"
    return [SourceRecord(**s) for s in _yaml(p).get("sources", [])] if p.exists() else []


def load_seed_mappings(seed_dir: Path) -> list[MappingRecord]:
    p = seed_dir / "mappings.yaml"
    return [MappingRecord.model_validate(m) for m in _yaml(p).get("mappings", [])] if p.exists() else []


def load_seed_secondary(seed_dir: Path) -> list[dict]:
    p = seed_dir / "secondary.yaml"
    return _yaml(p).get("secondary", []) if p.exists() else []

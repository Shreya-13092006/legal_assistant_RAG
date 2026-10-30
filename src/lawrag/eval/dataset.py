"""Gold dataset loading, validation, split locking and leakage checks (SDD 7.2)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lawrag.errors import ValidationError
from lawrag.graph.store import GraphStore
from lawrag.legal import uid as U
from lawrag.models.eval import GoldItem


def load(path: Path) -> list[GoldItem]:
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                items.append(GoldItem.model_validate_json(line))
            except Exception as exc:
                raise ValidationError(f"{path.name} line {n}: {exc}") from exc
    return items


def validate(items: list[GoldItem], store: GraphStore, require_verified: bool = False) -> list[str]:
    errors = []
    ids = [i.id for i in items]
    for d in {x for x in ids if ids.count(x) > 1}:
        errors.append(f"duplicate id {d}")
    for it in items:
        for g in it.gold_sections:
            if not store.exists(g.uid):
                errors.append(f"{it.id}: gold uid {g.uid} not in graph")
        for gm in it.gold_mappings:
            m = store.get_mapping(gm.mapping_id)
            if m is None:
                errors.append(f"{it.id}: mapping {gm.mapping_id} not in the reviewed mapping data")
                continue
            if m.status != gm.status or m.type != gm.type:
                errors.append(f"{it.id}: {gm.mapping_id} is {m.type}/{m.status}, gold says {gm.type}/{gm.status}")
            if sorted(map(U.section_of, gm.from_)) != sorted(map(U.section_of, m.from_uids)) or \
                    sorted(map(U.section_of, gm.to)) != sorted(map(U.section_of, m.to_uids)):
                errors.append(f"{it.id}: {gm.mapping_id} endpoints differ from the graph")
        for e in it.evidence:
            if store.get_source(e.source_id) is None:
                errors.append(f"{it.id}: evidence source {e.source_id} not registered")
        if require_verified and it.review_status != "verified":
            errors.append(f"{it.id}: review_status is {it.review_status}, not verified")
    llm_share = sum(1 for i in items if i.drafting_aid == "llm_draft") / max(len(items), 1)
    if llm_share > 0.2:
        errors.append(f"LLM-drafted share {llm_share:.0%} exceeds the 20% cap")
    return errors


def split_hash(items: list[GoldItem], split: str = "test") -> str:
    rows = sorted(i.model_dump_json(by_alias=True) for i in items if i.split == split)
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def lock_split(items: list[GoldItem], lock_path: Path, split: str = "test") -> str:
    h = split_hash(items, split)
    lock_path.write_text(json.dumps({"split": split, "sha256": h, "n": sum(i.split == split for i in items),
                                     "locked": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    return h


def check_lock(items: list[GoldItem], lock_path: Path, split: str = "test") -> str:
    if not lock_path.exists():
        raise ValidationError(f"{split} split is not locked ({lock_path} missing); run `lawrag eval lock` first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    h = split_hash(items, split)
    if lock["sha256"] != h:
        raise ValidationError(f"{split} split changed after locking (lock {lock['sha256'][:12]} vs {h[:12]})")
    return h


def near_duplicates(items: list[GoldItem], embedder, threshold: float = 0.9) -> list[tuple[str, str, float]]:
    dev = [i for i in items if i.split == "dev"]
    test = [i for i in items if i.split == "test"]
    if not dev or not test:
        return []
    a = embedder.encode([i.question for i in dev])
    b = embedder.encode([i.question for i in test])
    sims = a @ b.T / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :] + 1e-12)
    return [(dev[i].id, test[j].id, float(sims[i, j])) for i, j in zip(*np.where(sims >= threshold))]

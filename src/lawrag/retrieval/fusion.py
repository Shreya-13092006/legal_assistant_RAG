"""Reciprocal Rank Fusion with pinning, illustration down-weight and dedup (SDD 3.4.6, DECIDED)."""
from __future__ import annotations

from lawrag.models.qa import Candidate


def _merge(into: Candidate, c: Candidate) -> None:
    into.via = list(dict.fromkeys(into.via + c.via))
    into.chunk_ids = sorted(set(into.chunk_ids + c.chunk_ids))
    for k, v in c.ranks.items():
        into.ranks[k] = min(v, into.ranks.get(k, v))
    for k, v in c.scores.items():
        into.scores[k] = max(v, into.scores.get(k, v))
    into.pinned = into.pinned or c.pinned
    into.hops = min(into.hops, c.hops) if into.via else c.hops


def rrf(lists: dict[str, list[Candidate]], k_rrf: int = 60, weights: dict[str, float] | None = None,
        pinned: list[str] = (), top_n: int | None = None, illustration_weight: float = 1.0,
        intent: str = "lookup") -> list[Candidate]:
    """score(d) = sum_r w_r / (k_rrf + rank_r(d)). Pinned uids come first in mention order."""
    weights = weights or {}
    merged: dict[str, Candidate] = {}
    for name, cands in lists.items():
        for c in cands:
            c = c.model_copy(deep=True)
            if c.uid in merged:
                _merge(merged[c.uid], c)
            else:
                merged[c.uid] = c
    for c in merged.values():
        s = 0.0
        for name, rank in c.ranks.items():
            if name in lists or name in ("graph",):
                s += weights.get(name, 1.0) / (k_rrf + rank)
        if c.kind == "illustration" and intent != "scenario":
            s *= illustration_weight
        c.fused_score = round(s, 10)
    pinned_order = {u: i for i, u in enumerate(pinned)}
    pins = sorted((c for c in merged.values() if c.pinned or c.uid in pinned_order),
                  key=lambda c: (pinned_order.get(c.uid, len(pinned_order)), c.ranks.get("id_lookup", 10**6), c.uid))
    rest = sorted((c for c in merged.values() if not (c.pinned or c.uid in pinned_order)),
                  key=lambda c: (-c.fused_score, c.uid))
    if top_n is not None:
        rest = rest[:top_n]
    return pins + rest


def single(cands: list[Candidate], top_n: int) -> list[Candidate]:
    """fusion.method = none: one retriever's list, ordered by its own rank."""
    for c in cands:
        c.fused_score = 1.0 / (60 + min(c.ranks.values() or [10**6]))
    return sorted(cands, key=lambda c: (min(c.ranks.values() or [10**6]), c.uid))[:top_n]


def section_ranking(fused: list[Candidate]) -> list[str]:
    """Section-level roll-up used only for evaluation metrics."""
    return list(dict.fromkeys(c.section_uid for c in fused if not c.secondary))

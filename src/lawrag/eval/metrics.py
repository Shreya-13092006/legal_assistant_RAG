"""Retrieval and end-to-end metrics (SDD 7.5). All per-item values are computed first, then
aggregated, so paired statistics can use them."""
from __future__ import annotations

import math
import re

from lawrag.legal import uid as U
from lawrag.models.eval import GoldItem

DENIAL = re.compile(r"does not exist|no such section|is not a section|there is no section", re.I)
CAVEAT = re.compile(r"disput|differ|unresolved|not established|no verified mapping", re.I)


def _primary(item: GoldItem) -> list[str]:
    return [U.section_of(g.uid) for g in item.gold_sections if g.relevance == 2]


def _rel(item: GoldItem) -> dict[str, int]:
    out: dict[str, int] = {}
    for g in item.gold_sections:
        s = U.section_of(g.uid)
        out[s] = max(out.get(s, 0), g.relevance)
    return out


def recall_at_k(ranking: list[str], item: GoldItem, k: int) -> float | None:
    gold = set(_primary(item))
    return len(gold & set(ranking[:k])) / len(gold) if gold else None


def precision_at_k(ranking: list[str], item: GoldItem, k: int) -> float | None:
    rel = _rel(item)
    if not rel:
        return None
    return len(set(rel) & set(ranking[:k])) / k


def mrr(ranking: list[str], item: GoldItem, cutoff: int = 20) -> float | None:
    gold = set(_primary(item))
    if not gold:
        return None
    for i, s in enumerate(ranking[:cutoff], 1):
        if s in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking: list[str], item: GoldItem, k: int = 8) -> float | None:
    rel = _rel(item)
    if not rel:
        return None
    dcg = sum(rel.get(s, 0) / math.log2(i + 1) for i, s in enumerate(ranking[:k], 1))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def mapping_hit(context_mapping_ids: list[str], item: GoldItem) -> float | None:
    gold = {m.mapping_id for m in item.gold_mappings}
    return float(gold <= set(context_mapping_ids)) if gold else None


def section_ranking(pred: dict) -> list[str]:
    return list(dict.fromkeys(U.section_of(u) for u in pred.get("retrieved_ranking", [])
                              if not U.MAPPING_ID_RE.match(u) and "#" not in u and U.is_valid(u)))


def retrieval_item_metrics(pred: dict, item: GoldItem) -> dict:
    r = section_ranking(pred)
    return {"recall@5": recall_at_k(r, item, 5), "recall@8": recall_at_k(r, item, 8),
            "recall@20": recall_at_k(r, item, 20), "precision@8": precision_at_k(r, item, 8),
            "mrr": mrr(r, item), "ndcg@8": ndcg_at_k(r, item, 8),
            "mapping_hit": mapping_hit([m["mapping_id"] for m in pred.get("mapping_facts", [])], item),
            "context_tokens": pred.get("context_tokens"), "has_explicit_id": bool(item.gold_sections) and bool(
                re.search(r"\d", item.question)), "retrieval_ms": pred.get("retrieval_ms")}


def _checks(pred: dict) -> dict[str, dict]:
    return {c["check"]: c for c in pred.get("verification", {}).get("deterministic", {}).get("checks", [])}


def acceptable_outcome(pred: dict, item: GoldItem, correctness: int | None = None) -> bool:
    out = pred["outcome"]
    eb = item.expected_behavior
    text = pred.get("answer_markdown", "")
    checks = _checks(pred)
    if eb == "abstain":
        return out == "abstained"
    if eb == "correct_premise":
        return (out in ("answered", "partial") and DENIAL.search(text) is not None) or \
            (out == "abstained" and pred.get("abstention_kind") == "premise")
    if eb in ("state_dispute", "state_unresolved"):
        d4_ok = checks.get("D4", {}).get("passed", True)
        return out in ("answered", "partial") and bool(CAVEAT.search(text)) and d4_ok
    ok = out in ("answered", "partial") if eb == "answer" else out in ("answered", "partial")
    if correctness is not None:
        ok = ok and correctness >= 1
    return ok


def e2e_item_metrics(pred: dict, item: GoldItem, correctness: int | None = None) -> dict:
    checks = _checks(pred)
    out = pred["outcome"]
    answered = out in ("answered", "partial")
    claims = pred.get("verification", {}).get("claims", [])
    verdicts = [c.get("verdict") for c in claims]
    n = len(verdicts)
    det_hall = any(not checks.get(c, {"passed": True})["passed"] for c in ("D1", "D2", "D3", "D4", "D7"))
    cited = {c for cl in pred.get("claims", []) for c in cl.get("citations", []) if not c.startswith("MAP-")}
    gold_all = {U.section_of(g.uid) for g in item.gold_sections}
    gold_primary = set(_primary(item))
    cited_secs = {U.section_of(c) for c in cited if U.is_valid(c)}
    return {
        "outcome": out, "expected": item.expected_behavior, "answered": answered,
        "citation_valid": (checks.get("D1", {}).get("passed", True) and checks.get("D2", {}).get("passed", True))
        if answered else None,
        "citation_precision": (len(cited_secs & gold_all) / len(cited_secs)) if answered and cited_secs and gold_all else None,
        "citation_recall": (len(cited_secs & gold_primary) / len(gold_primary)) if answered and gold_primary else None,
        "claim_support": (verdicts.count("SUPPORTED") / n) if n else None,
        "claim_contradicted": (verdicts.count("CONTRADICTED") / n) if n else None,
        "claim_unsupported": ((verdicts.count("CONTRADICTED") + verdicts.count("NOT_ENOUGH_INFO")) / n) if n else None,
        "det_hallucination": det_hall if answered else False,
        "acceptable": acceptable_outcome(pred, item, correctness),
        "abstained": out == "abstained",
        "should_abstain": item.expected_behavior in ("abstain", "correct_premise"),
        "unsupported_answer": answered and any(v in ("CONTRADICTED", "NOT_ENOUGH_INFO") for v in verdicts),
        "correctness": correctness,
        "latency_ms": pred.get("timings_ms", {}).get("total"), "cost_usd": pred.get("cost_estimate_usd", 0.0),
        "retried": bool(pred.get("retry_actions")),
    }


def mean(xs) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(rows: list[dict]) -> dict:
    keys = {k for r in rows for k, v in r.items() if isinstance(v, (int, float, bool)) and not isinstance(v, str)}
    out = {k: mean(r.get(k) for r in rows) for k in sorted(keys)}
    out["n"] = len(rows)
    if rows and "expected" in rows[0]:
        ans = [r for r in rows if r["expected"] == "answer"]
        should = [r for r in rows if r["should_abstain"]]
        abst = [r for r in rows if r["abstained"]]
        rec = mean(r["abstained"] or r["acceptable"] for r in should) if should else None
        prec = mean(r["should_abstain"] for r in abst) if abst else None
        out["appropriate_abstention_recall"] = rec
        out["appropriate_abstention_precision"] = prec
        out["appropriate_abstention_f1"] = (2 * rec * prec / (rec + prec)) if rec and prec else None
        out["false_abstention"] = mean(r["abstained"] for r in ans) if ans else None
        out["coverage"] = mean(r["answered"] for r in rows)
        lat = sorted(r["latency_ms"] for r in rows if r.get("latency_ms") is not None)
        if lat:
            out["latency_p50"] = lat[len(lat) // 2]
            out["latency_p95"] = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        out["cost_usd_total"] = sum(r.get("cost_usd") or 0 for r in rows)
    return out

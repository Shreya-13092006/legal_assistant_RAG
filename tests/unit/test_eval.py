"""Metrics (hand-computed), statistics (vs scipy), dataset validation and split lock (SDD 5.1 #25)."""
import math

import pytest

from lawrag.errors import ValidationError
from lawrag.eval import dataset as ds
from lawrag.eval import metrics as M
from lawrag.eval import stats as S
from lawrag.models.eval import GoldItem


def item(**kw):
    base = {"id": "X-1", "split": "dev", "category": "lookup", "question": "q", "expected_behavior": "answer",
            "evidence": [{"source_id": "SRC-BNS-GAZETTE-2023"}],
            "gold_sections": [{"uid": "BNS:85", "relevance": 2}, {"uid": "BNS:86", "relevance": 2},
                              {"uid": "IPC:498A", "relevance": 1}]}
    base.update(kw)
    return GoldItem.model_validate(base)


def test_retrieval_metrics_hand_computed():
    it = item()
    ranking = ["BNS:85", "BNS:100", "IPC:498A", "BNS:86"]
    assert M.recall_at_k(ranking, it, 2) == 0.5
    assert M.recall_at_k(ranking, it, 8) == 1.0
    assert M.precision_at_k(ranking, it, 4) == 0.75
    assert M.mrr(ranking, it) == 1.0
    assert M.mrr(["BNS:1", "BNS:86"], it) == 0.5
    dcg = 2 / math.log2(2) + 1 / math.log2(4) + 2 / math.log2(5)
    idcg = 2 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert M.ndcg_at_k(ranking, it, 8) == pytest.approx(dcg / idcg)


def test_mcnemar_and_wilcoxon_vs_scipy():
    a = [True] * 10 + [False] * 10 + [True] * 2 + [False] * 8
    b = [True] * 10 + [False] * 10 + [False] * 2 + [True] * 8
    r = S.mcnemar_exact(a, b)
    from scipy.stats import binomtest
    assert r["p"] == pytest.approx(binomtest(2, 10, 0.5).pvalue)
    x = [0.1, 0.4, 0.5, 0.2, 0.9, 0.3, 0.6, 0.7]
    y = [0.3, 0.5, 0.4, 0.6, 1.0, 0.8, 0.9, 0.9]
    from scipy.stats import wilcoxon
    assert S.wilcoxon(x, y)["p"] == pytest.approx(wilcoxon([b - a for a, b in zip(x, y)]).pvalue)


def test_bootstrap_holm_kappa():
    mu, lo, hi = S.bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1], n=2000)
    assert lo <= mu <= hi and mu == 0.75
    h = S.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"]["p_holm"] == pytest.approx(0.03) and h["b"]["p_holm"] == pytest.approx(0.06)
    assert S.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert S.cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    assert S.weighted_kappa([0, 1, 2], [0, 1, 2]) == 1.0


def test_acceptable_outcomes():
    it = item(expected_behavior="abstain", abstain_reason="out_of_scope", gold_sections=[], evidence=[])
    assert M.acceptable_outcome({"outcome": "abstained"}, it)
    it = item(expected_behavior="state_unresolved")
    ok = {"outcome": "answered", "answer_markdown": "the correspondence is unresolved",
          "verification": {"deterministic": {"checks": [{"check": "D4", "passed": True}]}}}
    assert M.acceptable_outcome(ok, it)
    bad = {**ok, "answer_markdown": "IPC 377 has no equivalent"}
    assert not M.acceptable_outcome(bad, it)


def test_gold_schema_rules():
    with pytest.raises(ValueError):
        item(expected_behavior="abstain")  # needs abstain_reason
    with pytest.raises(ValueError):
        item(expected_behavior="correct_premise")  # needs trap_type
    with pytest.raises(ValueError):
        item(evidence=[])  # answer items need evidence


def test_validate_and_lock(store, tmp_path):
    good = item(gold_mappings=[{"mapping_id": "MAP-0004", "from": ["IPC:498A"], "to": ["BNS:85", "BNS:86"],
                                "type": "SPLIT", "status": "verified"}])
    assert ds.validate([good], store) == []
    wrong = item(id="X-2", gold_mappings=[{"mapping_id": "MAP-0004", "from": ["IPC:498A"], "to": ["BNS:85"],
                                           "type": "SPLIT", "status": "verified"}],
                 gold_sections=[{"uid": "BNS:999"}])
    errs = ds.validate([wrong], store)
    assert any("BNS:999" in e for e in errs) and any("endpoints" in e for e in errs)
    t = [item(id="T-1", split="test"), item(id="T-2", split="test")]
    lock = tmp_path / "splits.lock"
    h = ds.lock_split(t, lock)
    assert ds.check_lock(t, lock) == h
    t[0] = item(id="T-1", split="test", question="changed")
    with pytest.raises(ValidationError):
        ds.check_lock(t, lock)

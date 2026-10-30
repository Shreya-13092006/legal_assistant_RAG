"""Config validation/hashing, LLM cache/budget/repair, judge post-check, UI helpers, OCR metrics (SDD 5.1 #28)."""
import json

import pytest
from pydantic import BaseModel

from lawrag.config import LLMCfg, RoleCfg, load_config
from lawrag.errors import BudgetExceeded, ConfigError, LLMOutputError
from lawrag.ingest.ocr import boundary_f1, cer, section_number_accuracy, wer
from lawrag.llm.adapters.fake import FakeAdapter
from lawrag.llm.budget import BudgetGuard
from lawrag.llm.cache import LLMCache
from lawrag.llm.client import LLMClient, Message
from lawrag.models.qa import Claim, ContextBlock, ContextPack
from lawrag.ui.formatting import anchor, check_rows, diff_html, link_citations
from lawrag.verify.judge import judge_claims


def test_config_hash_stable_and_sensitive():
    a, b = load_config(), load_config()
    assert a.config_hash() == b.config_hash()
    c = load_config(["retrieval/vector_only.yaml"])
    assert c.config_hash() != a.config_hash()


def test_config_hash_ignores_paths_and_secrets():
    a = load_config()
    b = load_config(overrides={"graph": {"snapshot_path": "elsewhere.json"}})
    assert a.config_hash() == b.config_hash()


@pytest.mark.parametrize("bad", [{"retrieval": {"nonsense": 1}}, {"workflow": {"max_retries": 2}},
                                 {"retrieval": {"bm25": {"enabled": False}, "graph": {"enabled": False},
                                                "id_lookup": False}}])
def test_config_validation(bad):
    with pytest.raises(ConfigError):
        load_config(overrides=bad)


class Out(BaseModel):
    answer: str


def client(tmp_path, script, provider="fake", limit=5.0, prices=None):
    a = FakeAdapter(script)
    a.provider = provider
    budget = BudgetGuard(limit, prices or {}, tmp_path / "b.json")
    return LLMClient("generator", a, LLMCache(tmp_path / "c.sqlite"), budget, RoleCfg(provider="fake")), a


def test_llm_cache_hit(tmp_path):
    c, a = client(tmp_path, lambda r, s, u: {"answer": "x"})
    r1 = c.complete("# ROLE: generator", [Message(role="user", content="q")], schema=Out)
    r2 = c.complete("# ROLE: generator", [Message(role="user", content="q")], schema=Out)
    assert not r1.cached and r2.cached and r2.parsed.answer == "x" and len(a.calls) == 1


def test_llm_one_repair_then_error(tmp_path):
    seq = iter(["not json", json.dumps({"answer": "fixed"})])
    c, a = client(tmp_path, lambda r, s, u: next(seq))
    assert c.complete("# ROLE: generator", [Message(role="user", content="q")], schema=Out).parsed.answer == "fixed"
    c2, _ = client(tmp_path / "2", lambda r, s, u: "never json")
    with pytest.raises(LLMOutputError):
        c2.complete("# ROLE: generator", [Message(role="user", content="q2")], schema=Out)


def test_budget_guard(tmp_path):
    prices = {"anthropic/m": {"input": 1000.0, "output": 1000.0}}
    c, _ = client(tmp_path, lambda r, s, u: {"answer": "x"}, provider="anthropic", limit=0.01, prices=prices)
    c.model = "m"
    c.adapter.model = "m"
    c.complete("# ROLE: generator", [Message(role="user", content="a long question " * 10)], schema=Out)
    with pytest.raises(BudgetExceeded):
        c.complete("# ROLE: generator", [Message(role="user", content="another")], schema=Out)
    assert BudgetGuard(1, {}).price("ollama", "x", 10**6, 10**6) == 0.0


def test_judge_span_postcheck_and_uncited():
    ctx = ContextPack(blocks=[ContextBlock(label="BNS:1:body", kind="provision", text="[BNS:1:body] Whoever does X.",
                                           uids=["BNS:1:body"], source_id="S", section_uid="BNS:1")])
    llm_ok = LLMClient("judge", FakeAdapter(lambda r, s, u: {"verdict": "SUPPORTED", "evidence_span": "Whoever does X"}),
                       None, BudgetGuard(1, {}), RoleCfg(provider="fake"))
    llm_bad = LLMClient("judge", FakeAdapter(lambda r, s, u: {"verdict": "SUPPORTED", "evidence_span": "invented"}),
                        None, BudgetGuard(1, {}), RoleCfg(provider="fake"))
    claims = [Claim(text="X is an offence", citations=["BNS:1:body"]), Claim(text="no cite", citations=[])]
    assert [c.verdict for c in judge_claims(claims, ctx, llm_ok)] == ["SUPPORTED", "NOT_ENOUGH_INFO"]
    assert judge_claims(claims, ctx, llm_bad)[0].verdict == "NOT_ENOUGH_INFO"


def test_ui_formatting():
    assert "(#cite-bns-103-1)" in link_citations("see [BNS:103(1)]")
    assert anchor("IPC:498A") == "cite-ipc-498a"
    rows = check_rows({"deterministic": {"checks": [{"check": "D1", "passed": False, "severity": "hard",
                                                      "details": ["x"]}]}})
    assert rows[0]["result"] == "FAIL (hard)"
    assert "background" in diff_html([{"op": "replace", "left": "two", "right": "five"}], "right")


def test_ocr_metrics():
    assert cer("abcd", "abcd") == 0 and cer("abcd", "abed") == 0.25
    assert wer("a b c d", "a b x d") == 0.25
    assert section_number_accuracy(["302", "304A"], ["302"]) == 0.5
    assert boundary_f1(["a", "b"], ["a", "b"]) == 1.0

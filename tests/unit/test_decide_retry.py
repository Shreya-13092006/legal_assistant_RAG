"""Decision table: every row R1-R7 (SDD 3.5.3) and retry planning (3.5.4)."""
import pytest

from lawrag.models.qa import (CheckResult, Claim, ContextBlock, ContextPack, DeterministicReport, Draft,
                              QueryAnalysis, VerificationReport)
from lawrag.qa.retry import plan_retry
from lawrag.verify.decide import decide

CTX = ContextPack(blocks=[ContextBlock(label="BNS:103(1)", kind="provision", text="[BNS:103(1)] x", uids=["BNS:103(1)"],
                                       source_id="S", section_uid="BNS:103")])


def rep(verdicts, cores=None, hard=(), soft=(), judge=True, cites=None):
    cores = cores or [True] * len(verdicts)
    claims = [Claim(text=f"claim {i}", citations=(cites or {}).get(i, ["BNS:103(1)"]), core=c, verdict=v)
              for i, (v, c) in enumerate(zip(verdicts, cores))]
    checks = [CheckResult(check=h, passed=False, severity="hard", details=[f"{h} failed"], claim_indexes=[0]) for h in hard]
    checks += [CheckResult(check=s, passed=False, severity="soft", details=[f"{s} soft"], claim_indexes=[len(verdicts) - 1])
               for s in soft]
    return VerificationReport(deterministic=DeterministicReport(checks=checks), claims=claims, judge_enabled=judge)


def run(r, attempt=0, mode="normal", analysis=None, max_retries=1):
    return decide(r, Draft(answer="a", claims=r.claims, mode=mode), CTX, analysis or QueryAnalysis(), attempt,
                  max_retries, 0.6)


def test_r1_premise_correction():
    assert run(rep(["SUPPORTED"]), mode="premise_correction").rule == "R1"


@pytest.mark.parametrize("attempt", [0, 1])
def test_r2_accept(attempt):
    d = run(rep(["SUPPORTED", "SUPPORTED"]), attempt)
    assert (d.decision, d.rule) == ("accept", "R2")


def test_r3_hard_failure():
    assert run(rep(["SUPPORTED"], hard=["D1"])).decision == "retry"
    assert run(rep(["SUPPORTED"], hard=["D1"]), attempt=1).decision == "abstain"
    d = run(rep(["SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"], cores=[False, True, True, True], hard=["D1"]),
            attempt=1)
    assert d.decision == "partial" and 0 in d.drop_claims


def test_r4_contradicted_core():
    assert run(rep(["CONTRADICTED", "SUPPORTED"])).decision == "retry"
    assert run(rep(["CONTRADICTED", "SUPPORTED"]), attempt=1).decision == "abstain"


@pytest.mark.parametrize("attempt", [0, 1])
def test_r5_partial(attempt):
    d = run(rep(["SUPPORTED", "SUPPORTED", "NOT_ENOUGH_INFO"], cores=[True, True, False]), attempt)
    assert d.decision == "partial" and d.drop_claims == [2]


def test_partial_never_keeps_contradicted():
    d = run(rep(["SUPPORTED", "SUPPORTED", "SUPPORTED", "CONTRADICTED"], cores=[True, True, False, False]))
    assert d.decision == "partial" and 3 in d.drop_claims


def test_r6_retrievable_gap():
    r = rep(["NOT_ENOUGH_INFO"], cites={0: ["BNS:318(4)"]})
    assert run(r).decision == "retry" and run(r).rule == "R6"
    assert run(r, attempt=1).decision == "abstain"


def test_r7_abstain():
    d = run(rep(["NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO"]))
    assert (d.decision, d.rule) == ("abstain", "R7")
    assert run(rep([])).decision == "abstain"


def test_retry_budget_zero():
    assert run(rep(["SUPPORTED"], hard=["D1"]), max_retries=0).decision == "abstain"


def test_soft_failure_downgrades_to_partial():
    d = run(rep(["SUPPORTED", "SUPPORTED", "SUPPORTED"], cores=[True, True, False], soft=["D3"], judge=False))
    assert d.decision == "partial" and d.drop_claims == [2]


def test_retry_plans(store, cfg):
    r = rep(["NOT_ENOUGH_INFO"], cites={0: ["BNS:318(4)"]})
    p = plan_retry(r, QueryAnalysis(), CTX, store, cfg)
    assert p.action == "targeted_lookup" and "BNS:318(4)" in p.add_ids
    p = plan_retry(rep(["CONTRADICTED"]), QueryAnalysis(), CTX, store, cfg)
    assert p.action == "regenerate" and p.feedback
    p = plan_retry(None, QueryAnalysis(), ContextPack(), store, cfg)
    assert p.action == "widen" and p.vector_k == cfg.retrieval.vector.k * 2

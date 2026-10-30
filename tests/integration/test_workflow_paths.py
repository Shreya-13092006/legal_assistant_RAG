"""Every workflow path with a scripted FakeLLM (SDD 5.1 #18; M1 acceptance)."""
import json

import pytest

from lawrag.qa.workflow import Workflow

MURDER = "Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine."


def gen(answer, claims, mode=None):
    return json.dumps({"answer": answer, "claims": claims, "caveats": []})


def scripted(gen_outputs: list, judge="default"):
    calls = {"generator": 0}

    def script(role, system, user):
        if role == "generator":
            i = min(calls["generator"], len(gen_outputs) - 1)
            calls["generator"] += 1
            return gen_outputs[i]
        if role == "judge" and judge != "default":
            return judge(user)
        return None  # default fake behaviour
    return script, calls


@pytest.mark.parametrize("engine", ["langgraph", "python"])
def test_accept(make_deps, engine):
    out = gen(f'BNS:103(1) provides: "{MURDER}" [BNS:103(1)]',
              [{"text": f'BNS:103(1) provides: "{MURDER}"', "citations": ["BNS:103(1)"], "core": True}])
    script, calls = scripted([out])
    deps = make_deps(script, overrides={"workflow": {"engine": engine}})
    r = Workflow(deps).ask("What is the punishment for murder under the BNS?")
    assert r.outcome == "answered" and r.verification.decision_rule == "R2"
    assert calls["generator"] == 1 and r.citations[0].uid == "BNS:103(1)"


def test_retry_then_accept(make_deps):
    bad = gen("Murder: see [BNS:999].", [{"text": "Murder is punishable under BNS 999.", "citations": ["BNS:999"],
                                          "core": True}])
    good = gen(f'"{MURDER}" [BNS:103(1)]', [{"text": f'BNS:103(1) provides: "{MURDER}"', "citations": ["BNS:103(1)"],
                                             "core": True}])
    script, calls = scripted([bad, good])
    r = Workflow(make_deps(script)).ask("What is the punishment for murder under the BNS?")
    assert r.outcome == "answered" and calls["generator"] == 2 and r.retry_actions


def test_retry_then_abstain(make_deps):
    bad = gen("See [BNS:999].", [{"text": "Murder is punishable under BNS 999.", "citations": ["BNS:999"], "core": True}])
    script, calls = scripted([bad])
    r = Workflow(make_deps(script)).ask("What is the punishment for murder under the BNS?")
    assert r.outcome == "abstained" and calls["generator"] == 2  # exactly one retry
    assert r.abstention_kind == "evidence" and "Retry attempted" in r.abstention_reason


def test_no_retry_when_budget_zero(make_deps):
    bad = gen("See [BNS:999].", [{"text": "x", "citations": ["BNS:999"], "core": True}])
    script, calls = scripted([bad])
    r = Workflow(make_deps(script, overlays=["workflow/retry0.yaml"])).ask("punishment for murder under BNS?")
    assert r.outcome == "abstained" and calls["generator"] == 1


def test_partial_answer(make_deps):
    out = gen("x", [{"text": f'BNS:103(1) provides: "{MURDER}"', "citations": ["BNS:103(1)"], "core": True},
                    {"text": f'BNS:103(1) says: "{MURDER}"', "citations": ["BNS:103(1)"], "core": True},
                    {"text": "Murder trials are fast.", "citations": ["BNS:103(1)"], "core": False}])
    script, _ = scripted([out])
    r = Workflow(make_deps(script)).ask("What is the punishment for murder under the BNS?")
    assert r.outcome == "partial"
    assert r.unverified_claims == ["Murder trials are fast."]
    assert all(c.verdict == "SUPPORTED" for c in r.claims)


def test_out_of_scope_abstains_without_generation(make_deps):
    script, calls = scripted(["{}"])
    r = Workflow(make_deps(script)).ask("Can I get anticipatory bail for cheating?")
    assert r.outcome == "abstained" and r.abstention_kind == "scope" and calls["generator"] == 0


def test_gate_abstain_with_no_evidence(make_deps):
    script, calls = scripted(["{}"])
    deps = make_deps(script, overrides={"workflow": {"gate": {"min_retriever_rank": 1}}},
                     overlays=["retrieval/bm25_only.yaml"])
    r = Workflow(deps).ask("zzqx qqzz")
    assert r.outcome == "abstained" and calls["generator"] == 0 and r.retry_actions  # gate -> retry -> abstain


def test_premise_correction_default_fake(make_deps):
    r = Workflow(make_deps()).ask("What is the punishment for adultery under BNS 497?")
    assert r.outcome in ("answered", "partial")
    assert "does not exist" in r.answer_markdown
    assert any(m.mapping_id == "MAP-0006" for m in r.mapping_facts)  # corrected with the OMITTED hub


def test_omitted_vs_unresolved(make_deps):
    wf = Workflow(make_deps())
    omitted = wf.ask("Is there a BNS equivalent of IPC 497?")
    assert omitted.outcome == "answered" and "no direct BNS equivalent" in omitted.answer_markdown
    unresolved = wf.ask("What is the BNS equivalent of section 377 IPC?")
    assert "no direct" not in unresolved.answer_markdown.lower()
    assert "unresolved" in unresolved.answer_markdown


def test_split_mapping_smallest_demo(make_deps):
    r = Workflow(make_deps()).ask("What replaced IPC 498A?")
    assert r.outcome == "answered"
    assert "BNS:85, BNS:86" in r.answer_markdown and "SPLIT" in r.answer_markdown
    assert [m.mapping_id for m in r.mapping_facts] == ["MAP-0004"]
    assert all(c.passed for c in r.verification.deterministic.checks)


def test_verification_off_is_v0(make_deps):
    bad = gen("See [BNS:999].", [{"text": "x", "citations": ["BNS:999"], "core": True}])
    script, _ = scripted([bad])
    r = Workflow(make_deps(script, overlays=["verification/off.yaml"])).ask("punishment for murder under BNS?")
    assert r.outcome == "answered" and r.verification.decision_rule == "V0"
    assert not r.verification.deterministic.get("D1").passed  # still recorded


def test_generator_output_error_never_crashes(make_deps):
    script, _ = scripted(["not json at all"])
    r = Workflow(make_deps(script)).ask("What is the punishment for murder under the BNS?")
    assert r.outcome == "abstained"

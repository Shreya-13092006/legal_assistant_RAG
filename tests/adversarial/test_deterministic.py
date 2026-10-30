"""Adversarial drafts for D1-D7 (SDD 5.1 #20). Invented-section and wrong-number cases must be caught
100% of the time."""
from datetime import date

import pytest

from lawrag.models.qa import Candidate, Claim, Draft, QueryAnalysis
from lawrag.retrieval.context_builder import build_context
from lawrag.retrieval.mapping_facts import facts_for
from lawrag.verify.deterministic import deterministic_checks


def ctx_for(store, uids, mapping_of=None):
    fused = [Candidate(uid=u, section_uid=u.split(":body")[0].split("(")[0].split(":Expl")[0], act=u[:3],
                       via=["id_lookup"], ranks={"id_lookup": i + 1}, pinned=True) for i, u in enumerate(uids)]
    facts = [f for s in (mapping_of or []) for f in facts_for(s, store)]
    return build_context(fused, QueryAnalysis(), store, 5000, facts)


def failed(store, draft, ctx, analysis=None):
    rep = deterministic_checks(draft, ctx, analysis or QueryAnalysis(), store, date(2024, 7, 1))
    return {c.check for c in rep.checks if not c.passed}, rep


def D(answer, *claims, mode="normal"):
    return Draft(answer=answer, claims=[Claim(text=t, citations=c, core=core) for t, c, core in claims], mode=mode)


GOOD_MURDER = ("Whoever commits murder shall be punished with death or imprisonment for life, and shall also be "
               "liable to fine.", ["BNS:103(1)"], True)

# --- invented / ungrounded citations (D1/D2): must be 100% caught --------------------------------
INVENTED = [
    D("Murder is punishable under [BNS:999].", ("Murder is punishable under BNS 999.", ["BNS:999"], True)),
    D("See [IPC:302A].", ("See IPC 302A.", ["IPC:302A"], True)),
    D("See [BNS:103(9)].", ("Murder.", ["BNS:103(9)"], True)),
    D("See [MAP-9999].", ("Mapping.", ["MAP-9999"], True)),
    D("See [BNS:318(4)].", ("Cheating is punished.", ["BNS:318(4)"], True)),  # exists but not in context -> D2
    D("Bad uid [bns:103].", ("Bad uid.", ["bns:103"], True)),
]


@pytest.mark.parametrize("draft", INVENTED)
def test_invented_or_ungrounded_citations_caught(store, draft):
    ctx = ctx_for(store, ["BNS:103(1)"])
    bad, _ = failed(store, draft, ctx)
    assert bad & {"D1", "D2"}


# --- wrong numbers (D3): must be 100% caught ---------------------------------------------------
WRONG_NUMBERS = [
    "Murder is punishable with imprisonment for ten years.",
    "Murder is punishable with imprisonment for 7 years and fine.",
    "Causing death by negligence is punishable with imprisonment up to seven years.",
    "Causing death by negligence carries rigorous imprisonment for 3 months.",
    "It carries a fine of Rs. 5000.",
    "The punishment is community service.",
]


@pytest.mark.parametrize("text", WRONG_NUMBERS)
def test_wrong_numbers_caught(store, text):
    ctx = ctx_for(store, ["BNS:103(1)", "BNS:106(1)"])
    draft = D(text, (text, ["BNS:103(1)", "BNS:106(1)"], True))
    bad, rep = failed(store, draft, ctx)
    assert "D3" in bad
    assert rep.get("D3").severity == "hard"


def test_number_words_normalised(store):
    ctx = ctx_for(store, ["BNS:106(1)"])
    t = "It is punishable with imprisonment for a term which may extend to 5 years, and fine."
    bad, _ = failed(store, D(t, (t, ["BNS:106(1)"], True)), ctx)
    assert "D3" not in bad


def test_correct_answer_passes(store):
    ctx = ctx_for(store, ["BNS:103(1)"])
    bad, _ = failed(store, D("Murder: death or life imprisonment [BNS:103(1)].", GOOD_MURDER), ctx)
    assert bad == set()


def test_noncore_wrong_number_is_soft(store):
    ctx = ctx_for(store, ["BNS:103(1)"])
    bad, rep = failed(store, D("x", GOOD_MURDER, ("Also 9 years.", ["BNS:103(1)"], False)), ctx)
    assert rep.get("D3").severity == "soft"


# --- mapping fidelity (D4) ------------------------------------------------------------------
@pytest.mark.parametrize("text,cites,mapping_of", [
    ("IPC 377 has no BNS equivalent.", ["IPC:377:body"], ["IPC:377"]),          # unresolved != omitted
    ("IPC 302 has no BNS equivalent.", ["IPC:302:body"], ["IPC:302"]),          # verified IDENTICAL exists
    ("Section 498A IPC was omitted from the BNS.", ["IPC:498A:body"], ["IPC:498A"]),
    ("IPC 124A corresponds to BNS 152.", ["MAP-0008"], ["IPC:124A"]),           # disputed without caveat
    ("IPC 302 corresponds to BNS 318.", ["MAP-0002"], ["IPC:302"]),             # mapping does not link 318
    ("IPC 420 corresponds to BNS 304.", [], ["IPC:420"]),                       # only a rejected hub links them
    ("IPC 304A is identical to BNS 106(1).", ["MAP-0003"], ["IPC:304A"]),       # type is MODIFIED
    ("IPC 498A was merged into BNS 85.", ["MAP-0004"], ["IPC:498A"]),          # type is SPLIT
])
def test_mapping_misstatements_caught(store, text, cites, mapping_of):
    uids = [u for u in ("IPC:377:body", "IPC:302:body", "IPC:498A:body", "IPC:124A:body", "IPC:420:body",
                        "IPC:304A:body") if any(u.startswith(m + ":") for m in mapping_of)]
    ctx = ctx_for(store, uids, mapping_of)
    bad, _ = failed(store, D(text, (text, cites, True)), ctx)
    assert "D4" in bad


@pytest.mark.parametrize("text,cites,mapping_of", [
    ("According to the table, IPC:497 has no direct BNS equivalent (mapping type OMITTED, verified).", ["MAP-0006"], ["IPC:497"]),
    ("BNS:304 is new in the BNS and has no IPC predecessor.", ["MAP-0007"], ["BNS:304"]),
    ("Sources differ on whether IPC:124A corresponds to BNS:152; this mapping is disputed.", ["MAP-0008"], ["IPC:124A"]),
    ("No verified mapping is recorded for IPC:377; the correspondence is unresolved.", ["MAP-0009"], ["IPC:377"]),
    ("IPC 498A corresponds to BNS 85 and BNS 86 (split).", ["MAP-0004"], ["IPC:498A"]),
])
def test_permitted_mapping_statements_pass(store, text, cites, mapping_of):
    ctx = ctx_for(store, [], mapping_of)
    bad, _ = failed(store, D(text, (text, cites, True)), ctx)
    assert "D4" not in bad and "D5" not in bad


def test_mapping_claim_without_mapping_block_fails_d5(store):
    ctx = ctx_for(store, ["IPC:498A:body"])
    t = "IPC 498A corresponds to BNS 85."
    bad, _ = failed(store, D(t, (t, ["IPC:498A:body"], True)), ctx)
    assert "D5" in bad


# --- temporal (D6) and false premise (D7) -----------------------------------------------------
def test_temporal_caveat_missing(store):
    ctx = ctx_for(store, ["BNS:318(4)"])
    a = QueryAnalysis(offence_date=date(2023, 3, 1))
    t = "Cheating is punishable under BNS 318(4)."
    bad, _ = failed(store, D(t, (t, ["BNS:318(4)"], True)), ctx, a)
    assert "D6" in bad
    t2 = "The offence predates the BNS, so the IPC applies; BNS 318(4) is the current provision."
    bad, _ = failed(store, D(t2, (t2, ["BNS:318(4)"], True)), ctx, a)
    assert "D6" not in bad


def test_not_in_force_must_be_said(store):
    ctx = ctx_for(store, ["BNS:106(2)"])
    t = "Hit and run is punishable with imprisonment up to 10 years."
    bad, _ = failed(store, D(t, (t, ["BNS:106(2)"], True)), ctx)
    assert "D6" in bad
    t2 = t + " This sub-section is not yet in force."
    bad, _ = failed(store, D(t2, (t2, ["BNS:106(2)"], True)), ctx)
    assert "D6" not in bad


def test_false_premise(store):
    a = QueryAnalysis(nonexistent_ids=["BNS:999"])
    ctx = ctx_for(store, ["BNS:103(1)"])
    bad, _ = failed(store, D("Murder is punished with death [BNS:103(1)].", GOOD_MURDER), ctx, a)
    assert "D7" in bad
    t = "Section 999 does not exist in the BNS as indexed."
    bad, _ = failed(store, D(t, (t, [], True), mode="premise_correction"), ctx, a)
    assert "D7" not in bad

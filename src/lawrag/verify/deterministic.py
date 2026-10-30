"""Deterministic checks D1-D7 (SDD 3.6.1). Pure functions of (draft, context, analysis, graph), reused
by the evaluator as deterministic hallucination metrics."""
from __future__ import annotations

import re
from datetime import date

from lawrag.graph.store import GraphStore
from lawrag.legal import uid as U
from lawrag.legal.aliases import canonicalize_acts, extract_section_refs, normalize_text
from lawrag.legal.numbers import punishment_terms
from lawrag.models.qa import CheckResult, Claim, ContextPack, DeterministicReport, Draft, QueryAnalysis

CITE_IN_TEXT = re.compile(r"\[((?:IPC|BNS|BNSS|BSA):[^\]\s|]+|MAP-\d{4,}|MAPPING MAP-\d{4,}|SRC-[^\]\s]+#sec#c\d+)"
                          r"(?:\s*\|[^\]]*)?\]")
UID_IN_TEXT = re.compile(r"\b(IPC|BNS|BNSS|BSA):(\d+[A-Z]*)((?:\([0-9A-Za-z]+\))*)")
NEG_EQUIV = re.compile(
    r"no (?:direct )?(?:bns |ipc )?(?:equivalent|counterpart|corresponding (?:provision|section))|"
    r"(?:has|have) been (?:omitted|deleted|dropped|removed)|(?:was|were) (?:omitted|deleted|dropped|removed)|"
    r"not (?:been )?(?:retained|carried (?:over|forward))|does not have an? (?:equivalent|counterpart)|"
    r"no ipc predecessor|is new in the bns|has no predecessor", re.I)
POS_EQUIV = re.compile(r"correspond|replac|equivalent|mapped|maps to|successor|carried over|re-?numbered|"
                       r"is now (?:section|s\.)|became", re.I)
CAVEAT = re.compile(r"disput|differ|conflict|unresolved|not established|not verified|uncertain|contested|"
                    r"no verified mapping", re.I)
DENIAL = re.compile(r"does not exist|doesn't exist|no such section|is not a section|there is no section|"
                    r"not a valid section|does not contain (?:a )?section|not found in", re.I)
NOT_IN_FORCE = re.compile(r"not (?:yet )?in force|not been brought into force|struck down|unconstitutional|"
                          r"repealed|no longer (?:in force|an offence|applies)|read down", re.I)
TYPE_WORDS = {"IDENTICAL": r"\b(identical|unchanged|verbatim|without (?:any )?change)\b",
              "SPLIT": r"\bsplit\b", "MERGED": r"\b(merged|consolidated|combined)\b"}


def norm_citation(c: str) -> str:
    c = c.strip().strip("[]")
    c = re.sub(r"^MAPPING\s+", "", c)
    return c.split("|")[0].strip()


def citations_of(draft: Draft) -> list[tuple[int | None, str]]:
    out = [(i, norm_citation(c)) for i, cl in enumerate(draft.claims) for c in cl.citations]
    out += [(None, norm_citation(m.group(1))) for m in CITE_IN_TEXT.finditer(draft.answer)]
    return out


def mentioned_sections(text: str, codes: list[str]) -> set[str]:
    secs = {f"{m.group(1)}:{m.group(2)}" for m in UID_IN_TEXT.finditer(text)}
    canon = canonicalize_acts(normalize_text(UID_IN_TEXT.sub(" ", text)))
    for r in extract_section_refs(canon, codes):
        if r.act:
            secs.add(f"{r.act}:{r.number}")
    return secs


def _exists(c: str, store: GraphStore) -> bool:
    if U.MAPPING_ID_RE.match(c):
        m = store.get_mapping(c)
        return bool(m and m.status != "rejected")
    if "#sec#" in c:
        return store.get_chunk(c) is not None
    return U.is_valid(c) and store.exists(c)


def _grounded(c: str, ctx: ContextPack) -> bool:
    if U.MAPPING_ID_RE.match(c):
        return c in ctx.mapping_ids()
    uids = ctx.uids()
    if c in uids:
        return True
    if U.is_section_uid(c):
        return c in ctx.section_uids()
    try:
        return any(a in uids for a in U.ancestors(c))
    except ValueError:
        return False


def cited_text(claim: Claim, ctx: ContextPack) -> str:
    cites = {norm_citation(c) for c in claim.citations}
    parts = []
    for b in ctx.blocks:
        if any(c in b.uids or (U.is_section_uid(c) and b.section_uid == c) or
               (b.kind == "mapping_fact" and c in b.uids) for c in cites):
            parts.append(b.text)
    # qualifiers of a cited rule are evidence for that rule
    for b in ctx.blocks:
        if b.kind == "qualifier" and b.text not in parts and any(f"qualifies {c}" in b.text for c in cites):
            parts.append(b.text)
    return "\n".join(parts)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.;!?])\s+|\n+", text) if s.strip()]


def check_d1(draft, store) -> CheckResult:
    bad = [(i, c) for i, c in citations_of(draft) if not _exists(c, store)]
    return CheckResult(check="D1", passed=not bad, details=[f"cites nonexistent or malformed {c}" for _, c in bad],
                       claim_indexes=sorted({i for i, _ in bad if i is not None}))


def check_d2(draft, ctx) -> CheckResult:
    bad = [(i, c) for i, c in citations_of(draft) if not _grounded(c, ctx)]
    return CheckResult(check="D2", passed=not bad, details=[f"{c} is not in the context given to the generator"
                                                            for _, c in bad],
                       claim_indexes=sorted({i for i, _ in bad if i is not None}))


def check_d3(draft, ctx) -> CheckResult:
    details, idx, core_fail = [], [], False
    for i, cl in enumerate(draft.claims):
        terms = punishment_terms(cl.text)
        if not terms:
            continue
        evidence = punishment_terms(cited_text(cl, ctx))
        missing = sorted(t for t in terms if t not in evidence)
        if missing:
            details.append(f"claim {i}: {missing} not in the text of its cited provisions")
            idx.append(i)
            core_fail = core_fail or cl.core
    return CheckResult(check="D3", passed=not details, severity="hard" if core_fail else "soft",
                       details=details, claim_indexes=idx)


def check_d4(draft, ctx, analysis, store, codes) -> CheckResult:
    details, idx = [], set()
    units: list[tuple[int | None, str, list[str]]] = [(i, c.text, [norm_citation(x) for x in c.citations])
                                                      for i, c in enumerate(draft.claims)]
    claimed_text = " ".join(c.text for c in draft.claims)
    units += [(None, s, []) for s in _sentences(draft.answer) if s not in claimed_text]
    for i, text, cites in units:
        maps = [store.get_mapping(c) for c in cites if U.MAPPING_ID_RE.match(c)]
        maps = [m for m in maps if m]
        secs = mentioned_sections(text, codes)
        if NEG_EQUIV.search(text):
            subjects = secs or {U.section_of(u) for m in maps for u in m.from_uids + m.to_uids} or \
                {U.section_of(u) for u in analysis.section_ids}
            for s in sorted(subjects):
                ok = any(m.status == "verified" and m.evidence and (
                    (m.type == "OMITTED" and any(U.section_of(u) == s for u in m.from_uids)) or
                    (m.type == "NEW" and any(U.section_of(u) == s for u in m.to_uids)))
                         for m in store.mappings_for(s))
                if not ok:
                    details.append(f"'no equivalent' asserted for {s} without a verified OMITTED/NEW hub")
                    idx.add(i)
        for m in maps:
            ends = {U.section_of(u) for u in m.from_uids + m.to_uids}
            stray = {s for s in secs if s not in ends and U.act_of(s) in ("IPC", "BNS")}
            if stray and POS_EQUIV.search(text):
                details.append(f"{m.mapping_id} does not link {sorted(stray)}")
                idx.add(i)
            if m.status in ("disputed", "unresolved") and not CAVEAT.search(text):
                details.append(f"{m.mapping_id} is {m.status} but the claim states it without the caveat")
                idx.add(i)
            for typ, pat in TYPE_WORDS.items():
                if re.search(pat, text, re.I) and m.type != typ and not re.search(r"\bnot\b", text, re.I):
                    details.append(f"claim calls {m.mapping_id} {typ.lower()} but its type is {m.type}")
                    idx.add(i)
        if not maps and POS_EQUIV.search(text) and not NEG_EQUIV.search(text):
            ipc = {s for s in secs if U.act_of(s) == "IPC"}
            bns = {s for s in secs if U.act_of(s) == "BNS"}
            for a in sorted(ipc):
                for b in sorted(bns):
                    hubs = [m for m in store.mappings_for(a) if m.status != "rejected" and
                            any(U.section_of(u) == a for u in m.from_uids) and
                            any(U.section_of(u) == b for u in m.to_uids)]
                    if not hubs:
                        details.append(f"asserts {a} ~ {b} but no mapping hub links them")
                        idx.add(i)
                    elif not any(h.status == "verified" for h in hubs) and not CAVEAT.search(text):
                        details.append(f"asserts {a} ~ {b} as settled but the hub is {hubs[0].status}")
                        idx.add(i)
    return CheckResult(check="D4", passed=not details, details=details,
                       claim_indexes=sorted(x for x in idx if x is not None))


def check_d5(draft, ctx, store) -> CheckResult:
    details, idx, soft = [], [], True
    for i, c in enumerate(draft.claims):
        cites = [norm_citation(x) for x in c.citations]
        is_mapping = any(U.MAPPING_ID_RE.match(x) for x in cites) or bool(
            POS_EQUIV.search(c.text) and len(mentioned_sections(c.text, ["IPC", "BNS"])) >= 2)
        if is_mapping and not any(U.MAPPING_ID_RE.match(x) and x in ctx.mapping_ids() for x in cites):
            details.append(f"claim {i}: mapping claim does not cite a mapping fact block")
            idx.append(i)
            soft = False
        if cites and all("#sec#" in x for x in cites):
            if is_mapping:
                details.append(f"claim {i}: mapping claim supported only by a secondary source")
                idx.append(i)
                soft = False
            elif not re.search(r"secondary|judgment|court|commentary", c.text + " " + draft.answer, re.I):
                details.append(f"claim {i}: secondary-source claim not labelled as such")
                idx.append(i)
    return CheckResult(check="D5", passed=not details, severity="soft" if soft else "hard", details=details,
                       claim_indexes=idx)


def check_d6(draft, ctx, analysis, store, commencement: date) -> CheckResult:
    details, hard = [], False
    ans = draft.answer + " " + " ".join(c.text for c in draft.claims)
    if analysis.offence_date and analysis.offence_date < commencement:
        if not re.search(r"\bIPC\b|Indian Penal Code", ans):
            details.append(f"offence date {analysis.offence_date} is before BNS commencement; the answer does "
                           "not address IPC applicability")
        if re.search(r"\bBNS\b[^.]{0,40}\b(applies|will apply|governs|is applicable)\b", ans) and not re.search(
                r"\bIPC\b", ans):
            hard = True
    for _, c in citations_of(draft):
        if U.MAPPING_ID_RE.match(c) or "#sec#" in c or not _exists(c, store):
            continue
        p = store.get_provision(c)
        sec = store.get_section(U.section_of(c))
        inactive = (p is not None and not p.in_force) or (sec is not None and sec.status != "in_force")
        if inactive and not NOT_IN_FORCE.search(ans):
            details.append(f"{c} is not in force but the answer does not say so")
    return CheckResult(check="D6", passed=not details, severity="hard" if hard else "soft", details=details)


def check_d7(draft, analysis) -> CheckResult:
    if not analysis.nonexistent_ids:
        return CheckResult(check="D7", passed=True)
    details = []
    nums = {u.split(":", 1)[1] for u in analysis.nonexistent_ids}
    text = draft.answer + " " + " ".join(c.text for c in draft.claims)
    if not DENIAL.search(text) or not any(re.search(rf"\b{re.escape(n)}\b", text) for n in nums):
        details.append(f"question names nonexistent {analysis.nonexistent_ids}; the answer does not say so")
    cited = {c for _, c in citations_of(draft)}
    for u in analysis.nonexistent_ids:
        if u in cited:
            details.append(f"answer cites nonexistent {u} as if it existed")
    return CheckResult(check="D7", passed=not details, details=details)


def deterministic_checks(draft: Draft, ctx: ContextPack, analysis: QueryAnalysis, store: GraphStore,
                         commencement: date = date(2024, 7, 1), codes: list[str] | None = None) -> DeterministicReport:
    codes = codes or ["IPC", "BNS"]
    return DeterministicReport(checks=[
        check_d1(draft, store), check_d2(draft, ctx), check_d3(draft, ctx),
        check_d4(draft, ctx, analysis, store, codes), check_d5(draft, ctx, store),
        check_d6(draft, ctx, analysis, store, commencement), check_d7(draft, analysis)])


def premise_claim_indexes(draft: Draft, analysis: QueryAnalysis) -> list[int]:
    """Claims that only state a named section does not exist are verified deterministically (D7)."""
    if not analysis.nonexistent_ids:
        return []
    nums = {u.split(":", 1)[1] for u in analysis.nonexistent_ids}
    return [i for i, c in enumerate(draft.claims)
            if DENIAL.search(c.text) and any(re.search(rf"\b{re.escape(n)}\b", c.text) for n in nums)]

"""Query normalization and analysis, rules first (SDD 3.4.1)."""
from __future__ import annotations

import re
from datetime import date

from lawrag.config import RunConfig
from lawrag.legal.aliases import (DEFAULT_ALIASES, acts_mentioned, canonicalize_acts, extract_section_refs,
                                  normalize_text)
from lawrag.models.qa import QueryAnalysis

_MAPPING_CUES = r"\b(replac\w*|equivalent|correspond\w*|counterpart|new section for|old section|now under|" \
                r"map(?:s|ped|ping)? to|successor|predecessor|came from|which ipc section|which bns section|" \
                r"what is .* (?:in|under) (?:the )?(?:bns|ipc) now|renumber\w*|retained|dropped|omitted|deleted|" \
                r"what happened to|still (?:an offen[cs]e|exist\w*))\b"
_COMPARE_CUES = r"\b(differ\w*|compare\w*|comparison|changed?|changes|vs\.?|versus|contrast)\b"
_SCENARIO_CUES = r"\b(if [a-z]+ |suppose|scenario|\b[A-Z] (?:pushes|hits|kills|takes|steals|cheats|threatens)\b|" \
                 r"what (?:offen[cs]e|sections?) (?:is|are|may be|might be) (?:committed|relevant|attracted)|" \
                 r"which (?:offen[cs]e|sections?) (?:may|might|would|could) (?:apply|be relevant))"
_OUT_OF_SCOPE = [
    (r"\b(bail|anticipatory bail|remand|charge ?sheet|cognizable|non-cognizable|compoundable|bnss|crpc|"
     r"code of criminal procedure|bharatiya nagarik suraksha sanhita|arrest|trial procedure|f\.?i\.?r\b)",
     "criminal procedure (BNSS/CrPC) is out of scope"),
    (r"\b(bsa|evidence act|bharatiya sakshya|admissib\w+)\b", "evidence law (BSA) is out of scope"),
    (r"\b(will (?:i|he|she|they) (?:win|be convicted|get bail|go to jail)|chances of|predict\w*|outcome of my case|"
     r"should i (?:file|plead|sue)|my lawyer)\b", "case-outcome prediction and personal legal advice are out of scope"),
]
_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                        "september", "october", "november", "december"], 1)}


def _offence_date(text: str) -> date | None:
    t = text.lower()
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(_MONTHS) + r"),?\s+(\d{4})\b", t)
    if m:
        return date(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"\b(before|after|in|during)?\s*(" + "|".join(_MONTHS) + r")\s+(\d{4})\b", t)
    if m:
        y, mo = int(m.group(3)), _MONTHS[m.group(2)]
        if m.group(1) == "before":
            return date(y, mo, 1).fromordinal(date(y, mo, 1).toordinal() - 1)
        return date(y, mo, 1)
    m = re.search(r"\b(?:in|during|year)\s+((?:19|20)\d{2})\b", t)
    if m and not re.search(r"(act|code|sanhita),?\s*" + m.group(1), t):
        return date(int(m.group(1)), 1, 1)
    return None


class QueryAnalyzer:
    def __init__(self, cfg: RunConfig, section_ids: set[str], provision_ids: set[str] | None = None, llm=None):
        self.cfg = cfg
        self.codes = list(cfg.acts)
        self.aliases = cfg.act_aliases or DEFAULT_ALIASES
        self.section_ids = section_ids
        self.provision_ids = provision_ids or set()
        self.llm = llm

    def _resolve(self, act: str, number: str, path: list[str]) -> tuple[str | None, str]:
        sec = f"{act}:{number}"
        if path:
            prov = sec + "".join(f"({p})" for p in path)
            if prov in self.provision_ids:
                return prov, sec
        return (sec if sec in self.section_ids else None), sec

    def analyze(self, question: str) -> QueryAnalysis:
        norm = normalize_text(question)
        text = canonicalize_acts(norm, self.aliases)
        acts = acts_mentioned(text, self.codes)
        a = QueryAnalysis(rewritten_query=text, acts=acts)
        for pattern, reason in _OUT_OF_SCOPE:
            if re.search(pattern, text, re.I):
                a.out_of_scope, a.out_of_scope_reason = True, reason
                break
        for ref in extract_section_refs(text, self.codes):
            act = ref.act or (acts[0] if len(acts) == 1 else None)
            if act:
                found, sec = self._resolve(act, ref.number, ref.path)
                if found:
                    a.section_ids.append(found)
                else:
                    a.nonexistent_ids.append(sec)
                if not ref.act:
                    a.notes.append(f"act inferred for {ref.number}: {act}")
                continue
            hits = [c for c in ("IPC", "BNS") if f"{c}:{ref.number}" in self.section_ids]
            if len(hits) == 2:
                a.ambiguous_act = True
                a.section_ids += [self._resolve(c, ref.number, ref.path)[0] for c in hits]
            elif len(hits) == 1:
                a.section_ids.append(self._resolve(hits[0], ref.number, ref.path)[0])
                a.notes.append(f"act inferred for {ref.number}: {hits[0]} (only act containing it)")
            else:
                a.nonexistent_ids += [f"{c}:{ref.number}" for c in ("IPC", "BNS")]
                a.notes.append(f"section {ref.number} exists in neither act")
        a.section_ids = list(dict.fromkeys(u for u in a.section_ids if u))
        a.nonexistent_ids = list(dict.fromkeys(a.nonexistent_ids))
        a.acts = list(dict.fromkeys(acts + [u.split(":")[0] for u in a.section_ids]))
        low = text.lower()
        if re.search(_MAPPING_CUES, low):
            a.intent = "mapping"
        if re.search(_COMPARE_CUES, low):
            a.intent = "comparison"
        if a.intent == "unknown" and re.search(_SCENARIO_CUES, text):
            a.intent = "scenario"
        if a.intent == "unknown" and (a.section_ids or a.nonexistent_ids or re.search(
                r"\b(punish\w*|define\w*|definition|what (?:is|does)|which section|section deals|offen[cs]e of)\b", low)):
            a.intent = "lookup"
        ids_acts = {u.split(":")[0] for u in a.section_ids}
        if a.intent in ("mapping", "comparison"):
            if re.search(r"which ipc|ipc section did|came from|predecessor|old section", low) or ids_acts == {"BNS"}:
                a.direction = "bns_to_ipc"
            elif ids_acts == {"IPC"} or "IPC" in ids_acts:
                a.direction = "ipc_to_bns"
        a.offence_date = _offence_date(norm)
        if a.intent == "unknown" and self.cfg.analyzer.llm_fallback and self.llm is not None:
            a = self._llm_fallback(question, a)
        return a

    def _llm_fallback(self, question: str, a: QueryAnalysis) -> QueryAnalysis:
        from pydantic import BaseModel

        from lawrag.llm.client import Message, load_prompt

        class _Out(BaseModel):
            intent: str
            rewritten_query: str

        try:
            res = self.llm.complete(load_prompt("analyzer"), [Message(role="user", content=f"QUESTION: {question}")],
                                    schema=_Out)
            if res.parsed.intent in ("lookup", "mapping", "comparison", "scenario"):
                a.intent = res.parsed.intent  # IDs are never taken from the LLM
                a.rewritten_query = res.parsed.rewritten_query or a.rewritten_query
                a.analyzer = "rules+llm"
        except Exception as exc:  # invalid output -> keep 'unknown'
            a.notes.append(f"analyzer fallback failed: {exc}")
        return a

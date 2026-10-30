"""Act aliases and section-reference extraction from free text (SDD 3.4.1 step 2-3)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from lawrag.legal.uid import normalize_number

DEFAULT_ALIASES: dict[str, list[str]] = {
    "IPC": ["indian penal code", "i.p.c.", "i.p.c", "penal code", "ipc"],
    "BNS": ["bharatiya nyaya sanhita", "nyaya sanhita", "new criminal law", "b.n.s.", "bns"],
}


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                                   " ": " "}))
    return re.sub(r"\s+", " ", t).strip()


def canonicalize_acts(text: str, aliases: dict[str, list[str]] | None = None) -> str:
    """Replace act aliases with their canonical code (case-insensitive, longest alias first)."""
    aliases = aliases or DEFAULT_ALIASES
    pairs = sorted(((a, code) for code, al in aliases.items() for a in al), key=lambda x: -len(x[0]))
    out = text
    for alias, code in pairs:
        out = re.sub(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])", code, out, flags=re.I)
    # keep codes upper-case even when the user wrote "ipc"
    for code in aliases:
        out = re.sub(rf"(?<![A-Za-z]){code}(?![A-Za-z])", code, out, flags=re.I)
    return out


def acts_mentioned(text: str, codes: list[str]) -> list[str]:
    return [c for c in codes if re.search(rf"(?<![A-Za-z]){c}(?![A-Za-z])", text)]


@dataclass
class SectionRef:
    number: str            # normalized, e.g. "498A"
    path: list[str]        # ["2", "a"] for 376(2)(a)
    act: str | None        # explicit act if attached to the reference
    start: int
    end: int
    raw: str


_CUE = r"(?:sections?|secs?\.?|s\.|ss\.|u/s|u/ss|§|r/w|read with|and|to|,)"
_NUM = r"(?<![\w])(?<!\d\.)(\d{1,3})(?:-?\s?([A-Z]{1,2})|-?([a-z]{1,2}))?(?![A-Za-z0-9])"
_PATH = r"((?:\s*\(\s*(?:[0-9]+[A-Za-z]?|[a-z]{1,4})\s*\))*)"
_REF_RE = re.compile(_NUM + _PATH)


def extract_section_refs(text: str, codes: list[str]) -> list[SectionRef]:
    """Find section references. `text` should already be act-canonicalized.

    A number counts as a section reference when (a) a section cue precedes it, (b) an act code is
    attached before/after it, or (c) it carries a letter suffix (498A). Plain counts such as
    "seven years" or "5 persons" are ignored.
    """
    act_alt = "|".join(codes)
    refs: list[SectionRef] = []
    prev_ref: SectionRef | None = None
    for m in _REF_RE.finditer(text):
        suffix = m.group(2) or m.group(3) or ""
        if m.group(2) and text[m.start(2) - 1] == " " and not re.match(
                rf"\s*(?:$|[?.,;:)]|of\b|and\b|r/w|{act_alt})", text[m.end(2):]):
            suffix = ""  # "302 A pushes B": the A is a person, not a suffix
        number_raw = m.group(1) + suffix
        try:
            number = normalize_number(number_raw)
        except ValueError:
            continue
        before = text[max(0, m.start() - 40):m.start()]
        after = text[m.end():m.end() + 30]
        path = [p.strip() for p in re.findall(r"\(\s*([^)]+?)\s*\)", m.group(4) or "")]
        act_after = re.match(rf"^\s*(?:of\s+(?:the\s+)?)?({act_alt})\b", after)
        act_before = re.search(rf"\b({act_alt})\s*(?:{_CUE}\s*)?$", before, flags=re.I)
        cue_before = re.search(rf"(?:{_CUE})\s*$", before, flags=re.I)
        strong_cue = re.search(r"(?:sections?|secs?\.?|s\.|ss\.|u/s|u/ss|§|r/w|read with)\s*$", before, re.I)
        has_suffix = bool(suffix)
        unit_after = re.match(r"^\s*(?:years?|months?|days?|persons?|rupees|lakh|crore|hours?|%)", after, re.I)
        if unit_after:
            continue
        act = (act_after.group(1) if act_after else None) or (act_before.group(1).upper() if act_before else None)
        chained = bool(prev_ref and cue_before and m.start() - prev_ref.end <= 12)
        reads_with = bool(re.match(r"^\s*(?:r/w|read with)\s*(?:sections?|s\.)?\s*\d", after, re.I))
        if not (strong_cue or act or has_suffix or chained or reads_with):
            continue
        ref = SectionRef(number=number, path=path, act=act, start=m.start(), end=m.end(), raw=m.group(0))
        if chained and ref.act is None:
            ref.act = prev_ref.act
        refs.append(ref)
        prev_ref = ref
    # an act stated after a list ("sections 302 and 34 IPC") applies backwards to the list
    for i in range(len(refs) - 1, 0, -1):
        if refs[i - 1].act is None and refs[i].act and refs[i].start - refs[i - 1].end <= 12:
            refs[i - 1].act = refs[i].act
    return refs

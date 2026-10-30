"""Punishment-term extraction with number-word normalization (used by D3)."""
from __future__ import annotations

import re

_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30,
}
_WORD_RE = "|".join(sorted(_WORDS, key=len, reverse=True))


def words_to_digits(text: str) -> str:
    t = text.lower()
    t = re.sub(rf"\b({_WORD_RE})\b", lambda m: str(_WORDS[m.group(1)]), t)
    t = re.sub(r"(\d)[,](\d{2,3})", r"\1\2", t)  # 10,000 -> 10000
    t = re.sub(r"(\d)[,](\d{2,3})", r"\1\2", t)
    return t


_DURATION = re.compile(r"\b(\d+)\s*(years?|months?|days?)\b")
_RUPEES = re.compile(r"(?:rs\.?|rupees|₹)\s*(\d+)|(\d+)\s*rupees")


def punishment_terms(text: str) -> set[str]:
    """Normalized punishment terms: '7 year', 'life', 'death', 'fine', 'fine:5000', 'rigorous', ..."""
    t = words_to_digits(text)
    terms: set[str] = set()
    for n, unit in _DURATION.findall(t):
        terms.add(f"{int(n)} {unit.rstrip('s')}")
    if re.search(r"imprisonment for life|life imprisonment|imprisonment of life|remainder of (?:that )?person'?s? natural life", t):
        terms.add("life")
    if re.search(r"\bdeath\b", t) and re.search(r"punish|sentence|penalt", t):
        terms.add("death")
    if re.search(r"\bfine\b", t):
        terms.add("fine")
    for a, b in _RUPEES.findall(t):
        terms.add(f"fine:{int(a or b)}")
    if "rigorous" in t:
        terms.add("rigorous")
    if re.search(r"\bsimple imprisonment\b", t):
        terms.add("simple")
    if "community service" in t:
        terms.add("community service")
    return terms

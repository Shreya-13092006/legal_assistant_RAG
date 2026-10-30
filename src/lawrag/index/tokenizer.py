"""LegalTokenizer for BM25 (SDD 3.4.4)."""
from __future__ import annotations

import re

TOKENIZER_VERSION = "legal-1"

_STOP = set("""a an the of to in on for by with as at from or and is are was were be been being this that these
those it its he she his her they them their which who whom whose what when where why how any all such shall
may also than then there here into upon under about over if so do does did has have had will would can could
should i you we our your my me us""".split())
# legally significant words are never dropped
_KEEP = {"not", "without", "unless", "except", "no", "nor", "only"}


def _stem(tok: str) -> str:
    for suf in ("ings", "ing", "edly", "ed", "es", "s"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


class LegalTokenizer:
    def __init__(self, stem: bool = False):
        self.stem = stem

    def __call__(self, text: str) -> list[str]:
        t = text.lower()
        tokens: list[str] = []
        # section references: 376(2)(a) -> 376, 376_2, 376_2_a ; 120-b -> 120b
        for m in re.finditer(r"\b(\d{1,3})\s*-?\s*([a-z]{0,2})\b((?:\(\s*[0-9a-z]+\s*\))*)", t):
            base = m.group(1) + m.group(2)
            tokens.append(base)
            acc = base
            for part in re.findall(r"\(\s*([0-9a-z]+)\s*\)", m.group(3) or ""):
                acc = f"{acc}_{part}"
                tokens.append(acc)
        t = re.sub(r"\b\d{1,3}\s*-?\s*[a-z]{0,2}\b(?:\(\s*[0-9a-z]+\s*\))*", " ", t)
        for w in re.findall(r"[a-z]+", t):
            if w in _KEEP or (w not in _STOP and len(w) > 1):
                tokens.append(_stem(w) if self.stem and w not in _KEEP else w)
        return tokens

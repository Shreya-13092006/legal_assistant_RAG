"""Canonical identifiers (SDD 3.1.1). One shared implementation for parser, analyzer, verifier, gold data.

Formats
  Section    {ACT}:{number}            IPC:302, IPC:498A, IPC:120B
  Provision  {section_uid}{path}       BNS:103(1), BNS:2(1)(a), IPC:300:Exc1, IPC:300:Ill(a), IPC:302:body
  Chunk      {provision_uid}#{variant}#c{n}
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_ACTS = ("IPC", "BNS", "BNSS", "BSA")

_QUAL = r"(:(Exc|Expl|Prov)[0-9]*|:Ill\([a-z0-9]+\)|:body)"


def uid_regex(acts: tuple[str, ...] | list[str] = DEFAULT_ACTS) -> re.Pattern[str]:
    alt = "|".join(sorted(acts, key=len, reverse=True))
    return re.compile(
        rf"^({alt}):[0-9]+[A-Z]*(\([0-9]+[A-Z]*\))?(\([a-z]+\))*(\([ivxlc]+\))?{_QUAL}?$"
    )


UID_RE = uid_regex()
SECTION_RE = re.compile(r"^([A-Z]+):([0-9]+[A-Z]*)$")
MAPPING_ID_RE = re.compile(r"^MAP-\d{4,}$")
_PART_RE = re.compile(r"\(([^)]+)\)")


def normalize_number(raw: str) -> str:
    """'120-B' -> '120B', '498 a' -> '498A', ' 302 ' -> '302'."""
    s = re.sub(r"[\s\-–—.]", "", str(raw)).upper()
    if not re.fullmatch(r"[0-9]+[A-Z]*", s):
        raise ValueError(f"not a section number: {raw!r}")
    return s


def section_uid(act: str, number: str) -> str:
    return f"{act.upper()}:{normalize_number(number)}"


def sort_key(number: str) -> tuple[int, str]:
    m = re.fullmatch(r"([0-9]+)([A-Z]*)", number)
    if not m:
        return (10**9, number)
    return (int(m.group(1)), m.group(2))


@dataclass(frozen=True)
class UidParts:
    act: str
    number: str
    path: tuple[str, ...] = field(default_factory=tuple)  # ("1", "a", "ii")
    qualifier: str | None = None  # "Exc1", "Expl2", "Prov1", "Ill(a)", "body"

    @property
    def section(self) -> str:
        return f"{self.act}:{self.number}"


def is_valid(uid: str, acts: tuple[str, ...] | list[str] | None = None) -> bool:
    pattern = UID_RE if acts is None else uid_regex(acts)
    return bool(pattern.match(uid))


def parse_uid(uid: str) -> UidParts:
    if not is_valid(uid):
        raise ValueError(f"malformed uid: {uid!r}")
    act, rest = uid.split(":", 1)
    qualifier = None
    if ":" in rest:
        rest, qualifier = rest.split(":", 1)
    m = re.match(r"([0-9]+[A-Z]*)(.*)$", rest)
    assert m
    number, tail = m.group(1), m.group(2)
    path = tuple(_PART_RE.findall(tail))
    return UidParts(act=act, number=number, path=path, qualifier=qualifier)


def format_uid(parts: UidParts) -> str:
    s = parts.section + "".join(f"({p})" for p in parts.path)
    if parts.qualifier:
        s += f":{parts.qualifier}"
    return s


def section_of(uid: str) -> str:
    """Section uid of any provision/section uid (or mapping id passthrough)."""
    if MAPPING_ID_RE.match(uid):
        return uid
    m = re.match(r"^([A-Z]+:[0-9]+[A-Z]*)", uid)
    if not m:
        raise ValueError(f"malformed uid: {uid!r}")
    return m.group(1)


def act_of(uid: str) -> str:
    return uid.split(":", 1)[0]


def is_section_uid(uid: str) -> bool:
    return bool(SECTION_RE.match(uid))


def provision_uid(section: str, path: list[str] | tuple[str, ...] = (), qualifier: str | None = None) -> str:
    s = section + "".join(f"({p})" for p in path)
    return f"{s}:{qualifier}" if qualifier else s


def chunk_id(provision: str, variant: str = "text", part: int = 0) -> str:
    return f"{provision}#{variant}#c{part}"


def chunk_provision(cid: str) -> str:
    return cid.split("#", 1)[0]


def ancestors(uid: str) -> list[str]:
    """IPC:2(1)(a) -> [IPC:2(1), IPC:2]; IPC:300:Exc1 -> [IPC:300]."""
    p = parse_uid(uid)
    out = []
    if p.qualifier:
        base = format_uid(UidParts(p.act, p.number, p.path))
        if base != uid:
            out.append(base)
    path = list(p.path)
    while path:
        path.pop()
        out.append(format_uid(UidParts(p.act, p.number, tuple(path))))
    return [u for u in dict.fromkeys(out) if u != uid]

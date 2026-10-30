"""Structure-aware section and provision splitting (SDD 3.2.3).

Produces the nested chapter/section/provision description consumed by `tree_builder.build_tree`.
Header regexes are per act and must be checked against the actual print style (VERIFY A-10).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from lawrag.ingest.tree_builder import build_tree
from lawrag.models.legal import ActTree, Block

CHAPTER_RE = re.compile(r"^CHAPTER\s+([IVXLC]+(?:-?[A-Z])?)\b\.?\s*(.*)$")
SECTION_RE = re.compile(r"^(\d{1,3}[A-Z]{0,2})\.\s+(.+?)\.?\s*[—–-]{1,2}\s*(.*)$", re.S)
STATUS_RE = re.compile(r"^\[?\s*(omitted|repealed|rep\. by)\b", re.I)

# marker -> kind. A marker counts only at the start of the text or after "—", ":", ";", "." or a newline,
# so inline references such as "sub-section (2)" are not split.
_LEAD = (r"(?:(?<=^)|(?<=[—:;.]\s)|(?<=[—:;.])|(?<=\n)|(?<=; or )|(?<=; and )|(?<=, or )|(?<=, and )"
         r"|(?<=Illustrations )|(?<=Illustration )|(?<=Illustrations\. )|(?<=Illustration\. ))")
MARKERS = [
    ("subsection", re.compile(_LEAD + r"\s*\((\d+[A-Z]?)\)\s")),
    ("clause", re.compile(_LEAD + r"\s*\(([a-z]{1,2})\)\s")),
    ("explanation", re.compile(_LEAD + r"\s*Explanation(?:\s*(\d+))?\.?\s*[—–-]")),
    ("exception", re.compile(_LEAD + r"\s*Exception(?:\s*(\d+))?\.?\s*[—–-]")),
    ("illustration_hdr", re.compile(_LEAD + r"\s*Illustrations?\.?\s*")),
    ("proviso", re.compile(_LEAD + r"\s*(Provided(?: further)? that)")),
]
ROMAN = re.compile(r"^(?:i{1,3}|iv|v|vi{0,3}|ix|x{1,3})$")


@dataclass
class _Sec:
    number: str
    title: str
    page: int
    page_end: int
    text: str = ""
    footnotes: list[str] = field(default_factory=list)


def split_sections(blocks: list[Block]) -> list[dict]:
    """Group body blocks into chapters and sections (text still unsplit)."""
    chapters: list[dict] = []
    cur_ch: dict | None = None
    cur: _Sec | None = None
    expect_ch_title = False
    for b in blocks:
        if b.kind == "footnote":
            if cur:
                cur.footnotes.append(b.text)
            continue
        if b.kind != "text":
            continue
        text = b.text.strip()
        m = CHAPTER_RE.match(text)
        if m:
            cur_ch = {"number": m.group(1).replace("-", ""), "title": m.group(2).strip(), "sections": []}
            chapters.append(cur_ch)
            expect_ch_title = not m.group(2).strip()
            cur = None
            continue
        if expect_ch_title and cur_ch is not None:
            cur_ch["title"] = text
            expect_ch_title = False
            continue
        m = SECTION_RE.match(text)
        if m and (cur is None or _next_ok(cur.number, m.group(1))):
            if cur_ch is None:
                cur_ch = {"number": "I", "title": "", "sections": []}
                chapters.append(cur_ch)
            cur = _Sec(number=m.group(1), title=m.group(2).strip(), page=b.page, page_end=b.page, text=m.group(3))
            cur_ch["sections"].append(cur)
            continue
        if cur is not None:
            cur.text = f"{cur.text}\n{text}".strip()
            cur.page_end = b.page
    for ch in chapters:
        ch["sections"] = [_finish(s) for s in ch["sections"]]
    return chapters


def _next_ok(prev: str, nxt: str) -> bool:
    """Monotonic sequence check: a header must not go backwards (guards against inline 'N. ...' lines)."""
    pn = int(re.match(r"\d+", prev).group())
    nn = int(re.match(r"\d+", nxt).group())
    return nn > pn or (nn == pn and nxt > prev)


def split_provisions(text: str) -> list[dict]:
    """Split a section body into a provision list with nested sub-sections/clauses."""
    hits = []
    for kind, rx in MARKERS:
        for m in rx.finditer(text):
            hits.append((m.start(), m.end(), kind, (m.group(1) or "").strip() if m.groups() else ""))
    hits.sort(key=lambda h: (h[0], h[1]))
    dedup, last_end = [], -1
    for h in hits:
        if h[0] >= last_end:
            dedup.append(h)
            last_end = h[1]
    if not dedup:
        return [{"kind": "body", "text": text.strip()}] if text.strip() else []
    out: list[dict] = []
    lead = text[: dedup[0][0]].strip()
    if lead:
        out.append({"kind": "body", "text": lead})
    in_illus = False
    cur_sub: dict | None = None
    cur_clause: dict | None = None
    last_clause_letter = ""
    for i, (s, e, kind, label) in enumerate(dedup):
        body = text[e: dedup[i + 1][0] if i + 1 < len(dedup) else len(text)].strip()
        if kind == "proviso":
            body = f"{label} {body}".strip()
        if kind == "illustration_hdr":
            in_illus = True
            continue
        if kind == "clause" and in_illus:
            out.append({"kind": "illustration", "label": label, "text": body})
            continue
        if kind in ("explanation", "exception", "proviso"):
            in_illus = False
            item = {"kind": kind, "text": body}
            if label and label.isdigit():
                item["n"] = int(label)
            if kind == "proviso" and cur_sub is not None:
                item["target"] = f"({cur_sub['label']})"
            out.append(item)
            continue
        if kind == "subsection":
            in_illus = False
            cur_sub = {"kind": "subsection", "label": label, "text": body, "children": []}
            cur_clause = None
            last_clause_letter = ""
            out.append(cur_sub)
            continue
        if kind == "clause":
            is_roman = bool(ROMAN.match(label)) and cur_clause is not None and not (
                last_clause_letter and ord(label[0]) == ord(last_clause_letter[-1]) + 1 and len(label) == 1)
            if is_roman:
                cur_clause.setdefault("children", []).append({"kind": "subclause", "label": label, "text": body})
                continue
            cur_clause = {"kind": "clause", "label": label, "text": body}
            last_clause_letter = label
            parent = cur_sub["children"] if cur_sub is not None else out
            parent.append(cur_clause)
    for item in out:
        if item.get("children") == []:
            item.pop("children")
    return out


def _finish(s: _Sec) -> dict:
    d = {"number": s.number, "title": s.title, "page": s.page, "page_end": s.page_end,
         "amendment_notes": s.footnotes}
    if STATUS_RE.match(s.text.strip()) or not s.text.strip():
        d["status"] = "omitted_by_amendment" if "omit" in s.text.lower() else "repealed"
        d["status_note"] = s.text.strip()[:300]
        d["provisions"] = []
    else:
        d["provisions"] = split_provisions(s.text)
    return d


def apply_overrides(chapters: list[dict], overrides_path: Path) -> list[dict]:
    """review/overrides.yaml: [{uid: IPC:302, reason, page, by, section: {...replacement...}}]."""
    if not overrides_path.exists():
        return []
    ovs = yaml.safe_load(overrides_path.read_text(encoding="utf-8")) or []
    applied = []
    for ov in ovs:
        num = ov["uid"].split(":", 1)[1]
        for ch in chapters:
            for i, s in enumerate(ch["sections"]):
                if s["number"] == num:
                    ch["sections"][i] = {**s, **ov["section"]}
                    applied.append({k: ov.get(k) for k in ("uid", "reason", "page", "by")})
    return applied


def parse_act(blocks: list[Block], act: dict, source_id: str, overrides: Path | None = None) -> tuple[ActTree, list[dict]]:
    chapters = split_sections(blocks)
    applied = apply_overrides(chapters, overrides) if overrides else []
    act_code = act["code"]
    chapters = [c for c in chapters if c["sections"]]
    tree = build_tree(act, chapters, source_id)
    assert all(s.act == act_code for s in tree.sections)
    return tree, applied

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
SECTION_RE = re.compile(r"^(?:\d+\[)?(\d{1,3}-?[A-Z]{0,2})(?:\.|\s+(?=[\"\u201c]))\s*(.+?)\.?\s*(?:[—–]{1,2}|--)\s*(.*)$", re.S)
STUB_SEC_RE = re.compile(r"^(?:\d+\[|1(?=\d{3}[A-Z]{0,2}\.))?(\d{1,3}[A-Z]{0,2})\.\s*(\[?[^\n]*?\]\s*(?:Omitted|Rep\.|Repealed).*)$", re.I | re.S)
BARE_SEC_RE = re.compile(r"^(?:\d+\[)?(\d{1,3}[A-Z]{0,2})\.\s*(.*)$", re.S)
_FOOTNOTE_BODY_RE = re.compile(r"^(?:Ins\.|Subs\.|Rep\.|Omitted|Added|Substituted|The word|The words|See |For\s+[\"\u201c'\[])", re.I)
STATUS_RE = re.compile(r"^\[?\s*(omitted|repealed|rep\. by)\b", re.I)
_BOILERPLATE_RE = re.compile(
    r"^(?:THE GAZETTE OF INDIA|vlk/kkj\.k EXTRAORDINARY|Hkkx\s+II|izkf/kdkj ls|lañ\s+\d+|bl Hkkx esa|"
    r"xxxGID|jftLVªh lañ|MINISTRY OF LAW AND JUSTICE|\(Legislative Department\)|New Delhi, the|"
    r"The following Act of Parliament|CG-DL-E-|सी\.जी\.|DIWAKAR SINGH|UPLOADED BY THE MANAGER|"
    r"AND PUBLISHED BY THE CONTROLLER|MGIPMRND|[—–-]{4,}|\d+)$",
    re.I,
)

# marker -> kind. A marker counts only at the start of the text or after "—", ":", ";", "." or a newline,
# so inline references such as "sub-section (2)" are not split.
_LEAD = (r"(?:(?<=^)|(?<=[—:;.]\s)|(?<=[—:;.])|(?<=\n)|(?<=; or )|(?<=; and )|(?<=, or )|(?<=, and )"
         r"|(?<=Illustrations )|(?<=Illustration )|(?<=Illustrations\. )|(?<=Illustration\. ))")
MARKERS = [
    ("subsection", re.compile(_LEAD + r"\s*(?:\d+\[)?\((\d+[A-Z]?)\)\s")),
    ("clause", re.compile(_LEAD + r"\s*(?:\d+\[)?\(([a-z]{1,2})\)\s")),
    ("explanation", re.compile(_LEAD + r"\s*(?:\d+\[)?Explanation(?:\s*(\d+))?\.?\s*[—–-]")),
    ("exception", re.compile(_LEAD + r"\s*(?:\d+\[)?Exception(?:\s*(\d+))?\.?\s*[—–-]")),
    ("illustration_hdr", re.compile(_LEAD + r"\s*(?:\d+\[)?Illustrations?\.?\s*")),
    ("proviso", re.compile(_LEAD + r"\s*(?:\d+\[)?(Provided(?: further)? that)")),
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


def _match_section(text: str, page_margins: list[list], y0: float, gazette_mode: bool) -> tuple[str, str, str] | None:
    """Match a section start (IPC inline title, repealed stub, or Gazette bare number + marginal title)."""
    if not gazette_mode:
        m = SECTION_RE.match(text)
        if m:
            if _FOOTNOTE_BODY_RE.match(m.group(2).strip()):
                return None
            return m.group(1), m.group(2).strip(), m.group(3)
        m = STUB_SEC_RE.match(text)
        if m:
            return m.group(1), m.group(2).strip().rstrip("."), m.group(2)
        return None
    m = BARE_SEC_RE.match(text)
    if not m or _FOOTNOTE_BODY_RE.match(m.group(2).strip()):
        return None
    title = ""
    best = min((mg for mg in page_margins if not mg[2] and abs(mg[0] - y0) < 25), key=lambda mg: abs(mg[0] - y0), default=None)
    if best is not None:
        best[2] = True
        title = best[1].rstrip(".")
    return m.group(1), title, m.group(2)


def split_sections(blocks: list[Block]) -> list[dict]:
    """Group body blocks into chapters and sections (text still unsplit)."""
    margins_by_page: dict[int, list[list]] = {}
    gazette_mode = False
    for b in blocks:
        if b.kind == "marginal":
            gazette_mode = True
            margins_by_page.setdefault(b.page, []).append([b.bbox[1], b.text.strip(), False])

    chapters: list[dict] = []
    cur_ch: dict | None = None
    cur: _Sec | None = None
    last_sec: str | None = None
    expect_ch_title = False
    for b in blocks:
        if b.kind == "footnote":
            if cur:
                cur.footnotes.append(b.text)
            continue
        if b.kind != "text":
            continue
        text = b.text.strip()
        if _BOILERPLATE_RE.match(text):
            continue
        m = CHAPTER_RE.match(text)
        if m:
            cur_ch = {"number": m.group(1).replace("-", ""), "title": m.group(2).strip(), "sections": []}
            chapters.append(cur_ch)
            expect_ch_title = not m.group(2).strip()
            cur = None
            continue
        sec_hit = _match_section(text, margins_by_page.get(b.page, []), b.bbox[1], gazette_mode)
        if expect_ch_title and cur_ch is not None and not sec_hit:
            cur_ch["title"] = text
            expect_ch_title = False
            continue
        expect_ch_title = False
        if sec_hit and (last_sec is None or _next_ok(last_sec, sec_hit[0].replace("-", ""))):
            if cur_ch is None:
                cur_ch = {"number": "I", "title": "", "sections": []}
                chapters.append(cur_ch)
            cur = _Sec(number=sec_hit[0].replace("-", ""), title=sec_hit[1], page=b.page, page_end=b.page, text=sec_hit[2])
            last_sec = cur.number
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
    cur_qual: dict | None = None
    last_clause_letter = ""
    for i, (s, e, kind, label) in enumerate(dedup):
        body = text[e: dedup[i + 1][0] if i + 1 < len(dedup) else len(text)].strip()
        if kind == "proviso":
            body = f"{label} {body}".strip()
        if kind == "illustration_hdr":
            in_illus = True
            cur_qual = None
            continue
        if kind == "clause" and in_illus:
            if any(x.get("kind") == "illustration" and x.get("label") == label for x in out):
                out[-1]["text"] = f"{out[-1]['text']}\n({label}) {body}".strip()
            else:
                out.append({"kind": "illustration", "label": label, "text": body})
            continue
        if kind in ("explanation", "exception", "proviso"):
            in_illus = False
            item = {"kind": kind, "text": body}
            if label and label.isdigit():
                n_val = int(label)
                if not any(x.get("kind") == kind and x.get("n") == n_val for x in out):
                    item["n"] = n_val
            if kind == "proviso" and cur_sub is not None:
                item["target"] = f"({cur_sub['label']})"
            cur_qual = item
            out.append(item)
            continue
        if kind == "subsection":
            in_illus = False
            cur_qual = None
            if any(x.get("kind") == "subsection" and x.get("label") == label for x in out):
                out[-1]["text"] = f"{out[-1]['text']}\n({label}) {body}".strip()
                continue
            cur_sub = {"kind": "subsection", "label": label, "text": body, "children": []}
            cur_clause = None
            last_clause_letter = ""
            out.append(cur_sub)
            continue
        if kind == "clause":
            if cur_qual is not None:
                cur_qual["text"] = f"{cur_qual['text']}\n({label}) {body}".strip()
                continue
            is_roman = bool(ROMAN.match(label)) and cur_clause is not None and not (
                last_clause_letter and ord(label[0]) == ord(last_clause_letter[-1]) + 1 and len(label) == 1)
            if is_roman:
                subs = cur_clause.setdefault("children", [])
                if any(x.get("label") == label for x in subs):
                    subs[-1]["text"] = f"{subs[-1]['text']}\n({label}) {body}".strip()
                else:
                    subs.append({"kind": "subclause", "label": label, "text": body})
                continue
            parent = cur_sub["children"] if cur_sub is not None else out
            if any(x.get("kind") == "clause" and x.get("label") == label for x in parent):
                parent[-1]["text"] = f"{parent[-1]['text']}\n({label}) {body}".strip()
                continue
            cur_clause = {"kind": "clause", "label": label, "text": body}
            last_clause_letter = label
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

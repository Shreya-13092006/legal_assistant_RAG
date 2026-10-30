"""Native PDF extraction with PyMuPDF (SDD 3.2.2).

Pre-processing: repeating header/footer removal, de-hyphenation, NFKC normalization, Devanagari
blocks moved to a side list, small-font page-bottom blocks tagged as footnotes. Scanned pages are
detected and skipped (logged).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

from lawrag.errors import ParseError
from lawrag.logging_setup import get_logger
from lawrag.models.legal import Block

log = get_logger("ingest.pdf")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _fitz():
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ParseError("PyMuPDF is not installed (pip install lawrag[ingest])") from exc
    return fitz


def detect_mode(pdf: Path, min_chars: int = 200) -> dict[int, str]:
    """Per page: native | scanned | mixed, from text-layer character counts and image coverage."""
    fitz = _fitz()
    out = {}
    with fitz.open(pdf) as doc:
        for i, page in enumerate(doc, 1):
            chars = len(page.get_text("text").strip())
            has_img = bool(page.get_images())
            out[i] = "native" if chars >= min_chars else ("scanned" if has_img else ("mixed" if chars else "scanned"))
    return out


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "­": ""}))
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)          # de-hyphenate line breaks
    t = re.sub(r"[ \t]*\n[ \t]*", " ", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def _strip_repeating(blocks: list[Block], n_pages: int, threshold: float = 0.6) -> list[Block]:
    key = lambda b: (re.sub(r"\d+", "#", b.text.strip().lower())[:80], round(b.bbox[1] / 20))
    counts = Counter(key(b) for b in blocks)
    rep = {k for k, n in counts.items() if n >= max(3, threshold * n_pages)}
    out = []
    for b in blocks:
        if key(b) in rep:
            b.kind = "header"
        out.append(b)
    return out


def parse_native(pdf: Path, source_id: str, pages: list[int] | None = None) -> list[Block]:
    fitz = _fitz()
    try:
        doc = fitz.open(pdf)
    except Exception as exc:
        raise ParseError(f"cannot open {pdf}: {exc}") from exc
    if doc.needs_pass:
        raise ParseError(f"{pdf} is encrypted")
    modes = detect_mode(pdf)
    blocks: list[Block] = []
    with doc:
        for pno, page in enumerate(doc, 1):
            if pages and pno not in pages:
                continue
            if modes[pno] == "scanned":
                log.warning(f"page {pno} of {pdf.name} looks scanned; skipped in the main path")
                continue
            height = page.rect.height
            d = page.get_text("dict")
            sizes = [s["size"] for bl in d["blocks"] for ln in bl.get("lines", []) for s in ln["spans"]]
            body_size = Counter(round(s) for s in sizes).most_common(1)[0][0] if sizes else 10
            for bl in d["blocks"]:
                if bl.get("type") != 0:
                    continue
                spans = [s for ln in bl["lines"] for s in ln["spans"]]
                raw = "\n".join("".join(s["text"] for s in ln["spans"]) for ln in bl["lines"])
                if not raw.strip():
                    continue
                size = sum(s["size"] for s in spans) / len(spans)
                bold = any("bold" in s["font"].lower() or (s["flags"] & 16) for s in spans)
                kind = "text"
                if DEVANAGARI.search(raw) and len(DEVANAGARI.findall(raw)) > 0.3 * len(raw.strip()):
                    kind = "devanagari"
                elif size < body_size - 1.5 and bl["bbox"][1] > 0.75 * height:
                    kind = "footnote"
                blocks.append(Block(source_id=source_id, page=pno, bbox=tuple(bl["bbox"]), text=normalize(raw),
                                    font_size=round(size, 2), bold=bold, kind=kind))
    return _strip_repeating(blocks, len(modes))


def body_blocks(blocks: list[Block]) -> list[Block]:
    return [b for b in blocks if b.kind == "text"]


def write_jsonl(blocks: list[Block], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(b.model_dump_json() for b in blocks) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[Block]:
    return [Block.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

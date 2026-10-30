"""Structure-aware, provision-level chunking (SDD 3.3.1, DECIDED)."""
from __future__ import annotations

import hashlib
import re

from lawrag.config import ChunkCfg
from lawrag.legal import uid as U
from lawrag.models.legal import ActTree, Chunk, Provision, Section

_KIND_LABEL = {"body": "", "subsection": "", "clause": "", "subclause": ""}


def count_tokens(text: str) -> int:
    """Approximate token count (word pieces + punctuation). Deterministic and model-independent."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def _sentences(text: str) -> list[str]:
    # split at sentence ends, but never inside "(a)"-style markers or abbreviations like "s." / "No."
    parts = re.split(r"(?<=[.;:])\s+(?=[A-Z(\"'])(?<!\b[sS]\. )(?<!No\. )", text)
    return [p.strip() for p in parts if p.strip()]


def split_text(text: str, max_tokens: int) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    out, cur = [], ""
    for s in _sentences(text):
        cand = f"{cur} {s}".strip()
        if cur and count_tokens(cand) > max_tokens:
            out.append(cur)
            cur = s
        else:
            cur = cand
    if cur:
        out.append(cur)
    # a single over-long sentence is split on word boundaries as a last resort
    final = []
    for piece in out:
        words = piece.split()
        while count_tokens(" ".join(words)) > max_tokens and len(words) > 1:
            n = max(1, int(len(words) * max_tokens / max(count_tokens(" ".join(words)), 1)))
            final.append(" ".join(words[:n]))
            words = words[n:]
        if words:
            final.append(" ".join(words))
    return final


def header(section: Section, prov: Provision | None) -> str:
    h = f"[{section.act} s.{section.number} — {section.title}]"
    if prov is None:
        return h
    label = prov.label if prov.kind != "body" else ""
    return f"{h} {label} —".replace("  ", " ") if label else h


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def chunk_tree(tree: ActTree, cfg: ChunkCfg) -> list[Chunk]:
    sections = {s.uid: s for s in tree.sections}
    chunks: list[Chunk] = []
    for p in tree.provisions:
        if not p.text.strip():
            continue
        if p.kind == "illustration" and not cfg.include_illustrations:
            continue
        sec = sections[p.section_uid]
        parts = split_text(p.text, cfg.max_tokens)
        for i, part in enumerate(parts):
            embed = f"{header(sec, p)} {part}"
            chunks.append(Chunk(
                chunk_id=U.chunk_id(p.uid, "text", i), provision_uid=p.uid, section_uid=sec.uid,
                source_id=sec.source_id, act=sec.act, embed_text=embed, display_text=part, kind=p.kind,
                tokens=count_tokens(embed), corpus_variant="text", label="TextChunk",
                text_sha256=_sha(embed), part=i, n_parts=len(parts),
            ))
    if cfg.section_summaries:
        for sec in tree.sections:
            if not sec.text.strip():
                continue
            words = sec.text.split()
            summary = " ".join(words[:150])
            embed = f"{header(sec, None)} {summary}"
            chunks.append(Chunk(
                chunk_id=f"{sec.uid}#summary#c0", provision_uid=None, section_uid=sec.uid,
                source_id=sec.source_id, act=sec.act, embed_text=embed, display_text=summary,
                kind="section_summary", tokens=count_tokens(embed), corpus_variant="text",
                label="TextChunk", text_sha256=_sha(embed),
            ))
    return chunks


def chunk_secondary(items: list[dict], cfg: ChunkCfg) -> list[Chunk]:
    out = []
    for item in items:
        sid = item["source_id"]
        n = 0
        for para in item.get("paragraphs", []):
            for part in split_text(para["text"], cfg.max_tokens):
                embed = f"[SECONDARY {sid}] {part}"
                out.append(Chunk(
                    chunk_id=f"{sid}#sec#c{n}", provision_uid=None, section_uid=None, source_id=sid,
                    act=None, embed_text=embed, display_text=part, kind="secondary",
                    tokens=count_tokens(embed), corpus_variant="sec", label="SecondaryChunk",
                    text_sha256=_sha(embed), part=n, para_range=para.get("para_range"),
                    discusses=list(para.get("discusses", [])),
                ))
                n += 1
    return out

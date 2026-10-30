"""Source registration (SDD 3.2.1): sha256 + metadata into data/raw/sources.json and SOURCES.md.
Raw files are never edited."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from lawrag.errors import ValidationError
from lawrag.manifest import sha256_file
from lawrag.models.legal import SourceRecord


def _registry(raw_dir: Path) -> Path:
    return raw_dir / "sources.json"


def load_registry(raw_dir: Path) -> list[SourceRecord]:
    p = _registry(raw_dir)
    return [SourceRecord(**s) for s in json.loads(p.read_text(encoding="utf-8"))] if p.exists() else []


def register_source(pdf: Path, raw_dir: Path, *, kind: str, authority_rank: int, title: str,
                    source_id: str | None = None, url: str | None = None, publisher: str | None = None,
                    citation: str | None = None, license_note: str | None = None) -> SourceRecord:
    if not pdf.exists():
        raise ValidationError(f"file not found: {pdf}")
    sid = source_id or "SRC-" + re.sub(r"[^A-Z0-9]+", "-", pdf.stem.upper()).strip("-")
    rec = SourceRecord(source_id=sid, title=title, kind=kind, authority_rank=authority_rank,
                       sha256=sha256_file(pdf), retrieved_on=date.today().isoformat(), publisher=publisher,
                       url=url, local_path=pdf.name, citation=citation, license_note=license_note)
    recs = {r.source_id: r for r in load_registry(raw_dir)}
    if sid in recs and recs[sid].sha256 != rec.sha256:
        raise ValidationError(f"{sid} is already registered with a different file hash; use a new source id")
    recs[sid] = rec
    raw_dir.mkdir(parents=True, exist_ok=True)
    _registry(raw_dir).write_text(json.dumps([r.model_dump() for r in sorted(recs.values(), key=lambda r: r.source_id)],
                                             indent=2), encoding="utf-8")
    lines = ["# Registered sources", "", "| source_id | title | kind | rank | sha256 | retrieved | url |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(recs.values(), key=lambda r: r.source_id):
        lines.append(f"| {r.source_id} | {r.title} | {r.kind} | {r.authority_rank} | `{(r.sha256 or '')[:16]}…` | "
                     f"{r.retrieved_on} | {r.url or ''} |")
    (raw_dir / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rec


def verify_hash(rec: SourceRecord, raw_dir: Path) -> None:
    p = raw_dir / (rec.local_path or "")
    if not p.exists() or sha256_file(p) != rec.sha256:
        raise ValidationError(f"{rec.source_id}: file missing or hash mismatch with the registry")

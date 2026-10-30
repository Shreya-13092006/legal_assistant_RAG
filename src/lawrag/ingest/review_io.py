"""Human review gate I/O (SDD 3.2.6). `read_approved` validates review/approved_mappings.csv and
rejects rows with a missing reviewer/date/status, an unknown uid, or verified-without-evidence."""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from lawrag.errors import ValidationError
from lawrag.models.mapping import Endpoint, Evidence, MappingRecord

FINAL = {"verified", "disputed", "unresolved", "rejected"}


def parse_evidence(cell: str) -> list[Evidence]:
    out = []
    for part in [p for p in (cell or "").split(" || ") if p.strip()]:
        bits = part.split("|", 3)
        if len(bits) < 3:
            raise ValueError(f"bad evidence {part!r}; expected SOURCE|p<page>|r<row>|<raw row>")
        page = bits[1].lstrip("p")
        row = bits[2].lstrip("r")
        out.append(Evidence(source_id=bits[0].strip(), page=int(page) if page.isdigit() else None,
                            row=int(row) if row.isdigit() else None, raw_row=bits[3] if len(bits) > 3 else None,
                            extractor="pdfplumber"))
    return out


def _endpoints(cell: str) -> list[Endpoint]:
    out = []
    for u in [x.strip() for x in (cell or "").split(";") if x.strip()]:
        role = "partial" if u.endswith("~") else "primary"
        out.append(Endpoint(uid=u.rstrip("~"), role=role))
    return out


def read_approved(path: Path, known_uids: set[str], known_sources: set[str],
                  bulk_reviewer: str | None = None) -> list[MappingRecord]:
    errors, out, seen = [], [], set()
    with path.open(encoding="utf-8") as fh:
        for n, row in enumerate(csv.DictReader(fh), start=2):
            mid = (row.get("mapping_id") or "").strip()
            where = f"line {n} ({mid or '?'})"
            status = (row.get("status") or "").strip()
            if status.startswith("proposed"):
                continue  # not reviewed yet: stays out of the authoritative file
            if status not in FINAL:
                errors.append(f"{where}: status must be one of {sorted(FINAL)}")
                continue
            reviewer = (row.get("reviewed_by") or "").strip() or (bulk_reviewer or "")
            reviewed_on = (row.get("reviewed_on") or "").strip()
            if not reviewer or not reviewed_on:
                errors.append(f"{where}: reviewed_by and reviewed_on are required")
                continue
            try:
                date.fromisoformat(reviewed_on)
            except ValueError:
                errors.append(f"{where}: reviewed_on must be YYYY-MM-DD")
                continue
            if mid in seen:
                errors.append(f"{where}: duplicate mapping_id")
                continue
            seen.add(mid)
            try:
                ev = parse_evidence(row.get("evidence", ""))
            except ValueError as exc:
                errors.append(f"{where}: {exc}")
                continue
            frm, to = _endpoints(row.get("ipc_uids", "")), _endpoints(row.get("bns_uids", ""))
            unknown = [e.uid for e in frm + to if e.uid not in known_uids]
            if unknown:
                errors.append(f"{where}: unknown uid(s) {unknown}")
                continue
            if any(e.source_id not in known_sources for e in ev):
                errors.append(f"{where}: evidence cites an unregistered source")
                continue
            if status == "verified" and not ev:
                errors.append(f"{where}: verified without source evidence")
                continue
            sim = row.get("text_similarity")
            try:
                rec = MappingRecord.model_validate({
                    "mapping_id": mid, "type": (row.get("type") or row.get("proposed_type")).strip(), "status": status,
                    "granularity": row.get("granularity") or "section",
                    "sources_agree": str(row.get("sources_agree", "True")).lower() == "true",
                    "created_from": "concordance", "from": frm, "to": to, "evidence": ev,
                    "change_summary": row.get("change_summary") or None,
                    "text_similarity": float(sim) if sim not in (None, "", "None") else None,
                    "reviewed_by": reviewer, "reviewed_on": reviewed_on,
                    "review_note": row.get("review_note") or None, "dispute_note": row.get("dispute_note") or None})
            except Exception as exc:
                errors.append(f"{where}: {exc}")
                continue
            out.append(rec)
    if errors:
        raise ValidationError(f"{len(errors)} invalid review rows", errors=errors)
    return out

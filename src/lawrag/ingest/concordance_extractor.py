"""Concordance-table extraction with verbatim provenance (SDD 3.2.4)."""
from __future__ import annotations

import csv
import re
from pathlib import Path

from lawrag.errors import ParseError
from lawrag.legal.uid import normalize_number
from lawrag.models.mapping import RawMappingRow

MARKERS = {"new": "new", "deleted": "deleted", "omitted": "omitted", "—": "none_dash", "-": "none_dash",
           "nil": "none_dash", "": "none_dash"}
_NUM = r"\d{1,3}\s*-?\s*[A-Z]{0,2}"


def normalise_refs(cell: str, act: str, valid: set[str] | None = None) -> dict:
    """'Sec. 376(2)(a) & (b)' -> 2 uids; '299 to 304' -> expanded range (flagged); markers; parse errors kept."""
    raw = (cell or "").strip()
    low = raw.lower().strip(" .")
    out = {"uids": [], "marker": "none", "range_expanded": False, "parse_error": None}
    if low in ("new", "new section", "new offence"):
        out["marker"] = "new"
        return out
    if low in ("deleted", "omitted", "repealed", "omitted/deleted"):
        out["marker"] = "deleted" if low == "deleted" else "omitted"
        return out
    if low in ("—", "-", "–", "nil", "", "n/a"):
        return out
    t = re.sub(r"\b(?:sections?|secs?\.?|ss?\.)\s*", "", raw, flags=re.I)
    m = re.fullmatch(rf"\s*({_NUM})\s*(?:to|–|-)\s*({_NUM})\s*", t)
    if m:
        lo, hi = normalize_number(m.group(1)), normalize_number(m.group(2))
        lo_n, hi_n = int(re.match(r"\d+", lo).group()), int(re.match(r"\d+", hi).group())
        if hi_n - lo_n > 50 or hi_n < lo_n:
            out["parse_error"] = f"implausible range {raw!r}"
            return out
        nums = [str(n) for n in range(lo_n, hi_n + 1)]
        if valid is not None:
            nums = [n for n in sorted(valid, key=lambda x: (int(re.match(r'\d+', x).group()), x))
                    if lo_n <= int(re.match(r"\d+", n).group()) <= hi_n]
        out["uids"] = [f"{act}:{n}" for n in nums]
        out["range_expanded"] = True
        return out
    pieces = re.split(r"\s*(?:,|&|\band\b|;|/)\s*", t)
    base = None
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        mm = re.fullmatch(rf"({_NUM})((?:\(\s*[0-9A-Za-z]+\s*\))*)", p)
        if mm:
            base = normalize_number(mm.group(1))
            path = re.findall(r"\(\s*([0-9A-Za-z]+)\s*\)", mm.group(2))
            out["uids"].append(f"{act}:{base}" + "".join(f"({x})" for x in path))
            base_path = path
            continue
        mm = re.fullmatch(r"((?:\(\s*[0-9A-Za-z]+\s*\))+)", p)  # "& (b)" continues the previous reference
        if mm and base:
            tail = re.findall(r"\(\s*([0-9A-Za-z]+)\s*\)", mm.group(1))
            path = (base_path[: max(len(base_path) - len(tail), 0)] if base_path else []) + tail
            out["uids"].append(f"{act}:{base}" + "".join(f"({x})" for x in path))
            continue
        out["parse_error"] = f"cannot parse {p!r} in {raw!r}"
    if valid is not None:
        bad = [u for u in out["uids"] if u.split(":", 1)[1].split("(")[0] not in valid]
        if bad:
            out["parse_error"] = (out["parse_error"] or "") + f" numbers not valid for {act}: {bad}"
    return out


def extract_tables(pdf: Path, source_id: str, ipc_col: int = 0, bns_col: int = 1, remark_col: int | None = None,
                   pages: list[int] | None = None, valid_ipc: set[str] | None = None,
                   valid_bns: set[str] | None = None) -> tuple[list[RawMappingRow], list[int]]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ParseError("pdfplumber is not installed (pip install lawrag[ingest])") from exc
    rows: list[RawMappingRow] = []
    suspicious_pages: list[int] = []
    with pdfplumber.open(pdf) as doc:
        for pno, page in enumerate(doc.pages, 1):
            if pages and pno not in pages:
                continue
            page_rows = []
            for table in page.extract_tables():
                for ri, cells in enumerate(table):
                    cells = [(c or "").replace("\n", " ").strip() for c in cells]
                    if not any(cells) or re.search(r"\b(ipc|bns|section)\b", " ".join(cells).lower()) and not \
                            re.search(r"\d", " ".join(cells)):
                        continue  # header row
                    page_rows.append(build_row(source_id, pno, len(rows) + len(page_rows), cells, ipc_col, bns_col,
                                               remark_col, valid_ipc, valid_bns))
            errs = sum(1 for r in page_rows if r.parse_error and "not valid" in r.parse_error)
            if page_rows and errs > 0.3 * len(page_rows):
                suspicious_pages.append(pno)  # possible column swap
            rows += page_rows
    return rows, suspicious_pages


def build_row(source_id, page, row_idx, cells, ipc_col, bns_col, remark_col, valid_ipc=None, valid_bns=None):
    ipc = normalise_refs(cells[ipc_col] if ipc_col < len(cells) else "", "IPC", valid_ipc)
    bns = normalise_refs(cells[bns_col] if bns_col < len(cells) else "", "BNS", valid_bns)
    remark = cells[remark_col] if remark_col is not None and remark_col < len(cells) else ""
    marker = "new" if ipc["marker"] == "new" or bns["marker"] == "new" or re.search(r"\bnew\b", remark, re.I) else (
        "deleted" if "deleted" in (ipc["marker"], bns["marker"]) or re.search(r"\b(deleted|omitted)\b", remark, re.I)
        else "none")
    if not ipc["uids"] and bns["uids"] and marker == "none" and cells[ipc_col].strip() in ("", "—", "-", "–"):
        marker = "new"
    if ipc["uids"] and not bns["uids"] and marker == "none" and cells[bns_col].strip() in ("", "—", "-", "–"):
        marker = "deleted"
    errs = "; ".join(e for e in (ipc["parse_error"], bns["parse_error"]) if e) or None
    return RawMappingRow(source_id=source_id, page=page, row_idx=row_idx, raw_cells=cells, ipc_uids=ipc["uids"],
                         bns_uids=bns["uids"], marker=marker, remark=remark,
                         range_expanded=ipc["range_expanded"] or bns["range_expanded"], parse_error=errs)


def write_rows(rows: list[RawMappingRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source_id", "page", "row_idx", "raw_cells", "ipc_uids", "bns_uids", "marker", "remark",
                    "range_expanded", "parse_error"])
        for r in rows:
            w.writerow([r.source_id, r.page, r.row_idx, " | ".join(r.raw_cells), ";".join(r.ipc_uids),
                        ";".join(r.bns_uids), r.marker, r.remark, r.range_expanded, r.parse_error or ""])


def read_rows(path: Path) -> list[RawMappingRow]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for d in csv.DictReader(fh):
            out.append(RawMappingRow(source_id=d["source_id"], page=int(d["page"]), row_idx=int(d["row_idx"]),
                                     raw_cells=d["raw_cells"].split(" | "),
                                     ipc_uids=[x for x in d["ipc_uids"].split(";") if x],
                                     bns_uids=[x for x in d["bns_uids"].split(";") if x], marker=d["marker"],
                                     remark=d["remark"], range_expanded=d["range_expanded"] == "True",
                                     parse_error=d["parse_error"] or None))
    return out

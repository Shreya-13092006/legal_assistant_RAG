"""Graph integrity audit: rules G1-G12 and coverage report (SDD 3.1.5, 3.2.8)."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from lawrag.graph.data import GraphData, snapshot_hash
from lawrag.legal import uid as U
from lawrag.manifest import sha256_file

SHAPES = {  # type -> (min_in, max_in, min_out, max_out)
    "IDENTICAL": (1, 1, 1, 1), "MODIFIED": (1, 1, 1, 1), "SPLIT": (1, 1, 2, 999),
    "MERGED": (2, 999, 1, 1), "PARTIAL": (1, 999, 1, 999), "OMITTED": (1, 999, 0, 0),
    "NEW": (0, 0, 1, 999), "UNDETERMINED": (0, 999, 0, 999),
}


class RuleResult(BaseModel):
    rule: str
    passed: bool
    details: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AuditReport(BaseModel):
    status: str
    created: str
    snapshot_hash: str
    corpus_hash: str
    rules: list[RuleResult]
    coverage: dict
    unreviewed_sections: list[str]
    unresolved_xrefs: list[dict] = Field(default_factory=list)
    overrides: list[dict] = Field(default_factory=list)
    counts: dict = Field(default_factory=dict)


def audit(data: GraphData, *, expected: dict[str, list[str]] | None = None, raw_dir: Path | None = None,
          active_model: str | None = None, overrides: list[dict] | None = None) -> AuditReport:
    rules: list[RuleResult] = []
    secs = {s.uid: s for s in data.sections}
    provs = {p.uid: p for p in data.provisions}
    chapters = {c.uid for c in data.chapters}
    sources = {s.source_id: s for s in data.sources}
    act_codes = [a.code for a in data.acts] or list(U.DEFAULT_ACTS)

    # G1 section counts
    r = RuleResult(rule="G1", passed=True)
    if expected:
        for act, nums in expected.items():
            got = {s.number for s in data.sections if s.act == act}
            missing = [n for n in nums if n not in got]
            extra = sorted(got - set(nums), key=U.sort_key)
            if missing or extra:
                r.passed = False
                r.details.append(f"{act}: missing {missing[:20]}, unexpected {extra[:20]}")
    else:
        r.warnings.append("no expected section lists supplied (data/expected/<ACT>.txt); count check skipped")
    rules.append(r)

    # G2 every section has one chapter parent and one source
    r = RuleResult(rule="G2", passed=True)
    has_section = data.meta.get("has_section_edges")
    parents = Counter(dst for _, dst in has_section) if has_section is not None else None
    for s in data.sections:
        if s.chapter_uid not in chapters:
            r.passed = False
            r.details.append(f"{s.uid}: no parent chapter")
        if parents is not None and parents.get(s.uid, 0) != 1:
            r.passed = False
            r.details.append(f"{s.uid}: {parents.get(s.uid, 0)} HAS_SECTION parents")
        if s.source_id not in sources:
            r.passed = False
            r.details.append(f"{s.uid}: source {s.source_id} not registered")
    rules.append(r)

    # G3 provisions reachable from one section; canonical uids; no duplicates
    r = RuleResult(rule="G3", passed=True)
    dup = [u for u, n in Counter(p.uid for p in data.provisions).items() if n > 1]
    if dup:
        r.passed = False
        r.details.append(f"duplicate provision uids: {dup[:20]}")
    for p in data.provisions:
        if not U.is_valid(p.uid, act_codes):
            r.passed = False
            r.details.append(f"malformed uid {p.uid}")
        if p.section_uid not in secs:
            r.passed = False
            r.details.append(f"{p.uid}: section {p.section_uid} missing")
        if p.parent_uid not in secs and p.parent_uid not in provs:
            r.passed = False
            r.details.append(f"{p.uid}: parent {p.parent_uid} missing")
        for t in p.qualifies:
            if t not in secs and t not in provs:
                r.passed = False
                r.details.append(f"{p.uid}: QUALIFIES target {t} missing")
    rules.append(r)

    # G4 chunks
    r = RuleResult(rule="G4", passed=True)
    chunked = {c.provision_uid for c in data.chunks if c.provision_uid}
    for p in data.provisions:
        if p.text.strip() and p.uid not in chunked:
            r.passed = False
            r.details.append(f"{p.uid}: no chunk")
    for c in data.chunks:
        if c.provision_uid and c.provision_uid in provs and c.section_uid != provs[c.provision_uid].section_uid:
            r.passed = False
            r.details.append(f"{c.chunk_id}: crosses section boundary")
    rules.append(r)

    # G5 verified mappings: evidence + reviewer + date
    r = RuleResult(rule="G5", passed=True)
    for m in data.mappings:
        if m.status == "verified" and (not m.evidence or not m.reviewed_by or not m.reviewed_on):
            r.passed = False
            r.details.append(f"{m.mapping_id}: verified without evidence/reviewer/date")
        if m.reviewed_by == "SEED-DRAFT":
            r.warnings.append(f"{m.mapping_id}: reviewer is SEED-DRAFT (not human-reviewed yet)")
    rules.append(r)

    # G6 shape matches type
    r = RuleResult(rule="G6", passed=True)
    for m in data.mappings:
        lo_i, hi_i, lo_o, hi_o = SHAPES[m.type]
        ni, no = len(m.from_), len(m.to)
        if not (lo_i <= ni <= hi_i and lo_o <= no <= hi_o):
            r.passed = False
            r.details.append(f"{m.mapping_id}: type {m.type} with {ni} in / {no} out")
        if m.type == "UNDETERMINED" and m.status != "unresolved":
            r.passed = False
            r.details.append(f"{m.mapping_id}: UNDETERMINED is only valid with status unresolved")
    rules.append(r)

    # G7 / G8 SUCCEEDED_BY
    derived = data.succeeded_by()
    stored = data.meta.get("stored_succeeded_by")
    r7 = RuleResult(rule="G7", passed=True)
    if stored is not None:
        stored_t = sorted(tuple(x) for x in stored)
        if stored_t != derived:
            r7.passed = False
            r7.details.append(f"stored {len(stored_t)} vs derived {len(derived)} SUCCEEDED_BY edges")
    rules.append(r7)
    r8 = RuleResult(rule="G8", passed=True)
    rejected = {m.mapping_id for m in data.mappings if m.status == "rejected"}
    for a, b, mid, _ in (stored or derived):
        if a not in secs and a not in provs or b not in secs and b not in provs or mid in rejected:
            r8.passed = False
            r8.details.append(f"bad SUCCEEDED_BY {a}->{b} ({mid})")
    rules.append(r8)

    # G9 coverage (report only)
    covered = set()
    for m in data.mappings:
        for u in m.from_uids + m.to_uids:
            covered.add(U.section_of(u))
    unreviewed = sorted((s for s in secs if s not in covered), key=lambda u: (U.act_of(u), U.sort_key(u.split(":")[1])))
    rules.append(RuleResult(rule="G9", passed=True,
                            warnings=[f"{len(unreviewed)} sections have no mapping hub (unreviewed)"] if unreviewed else []))

    # G10 struck down / repealed need status evidence
    r = RuleResult(rule="G10", passed=True)
    for s in data.sections:
        if s.status in ("struck_down", "repealed") and not (s.status_source_id and s.status_source_id in sources):
            r.passed = False
            r.details.append(f"{s.uid}: status {s.status} without STATUS_EVIDENCED_BY source")
    rules.append(r)

    # G11 embeddings use the active model with a consistent dimension
    r = RuleResult(rule="G11", passed=True)
    dims = Counter(len(c.embedding) for c in data.chunks if c.embedding)
    if len(dims) > 1:
        r.passed = False
        r.details.append(f"mixed embedding dimensions {dict(dims)}")
    for c in data.chunks:
        if c.embedding and active_model and c.embedding_model != active_model:
            r.passed = False
            r.details.append(f"{c.chunk_id}: model {c.embedding_model} != active {active_model}")
            break
    rules.append(r)

    # G12 sources hashed and matching files
    r = RuleResult(rule="G12", passed=True)
    for s in data.sources:
        if not s.sha256:
            r.warnings.append(f"{s.source_id}: no sha256 recorded (source not yet registered)")
            continue
        path = Path(s.local_path) if s.local_path else None
        if path and not path.is_absolute() and raw_dir:
            path = raw_dir / path.name
        if path and path.exists():
            if sha256_file(path) != s.sha256:
                r.passed = False
                r.details.append(f"{s.source_id}: file hash mismatch")
        else:
            r.warnings.append(f"{s.source_id}: file not found for hash check")
    rules.append(r)

    coverage: dict[str, dict[str, int]] = {}
    for m in data.mappings:
        coverage.setdefault(m.type, {}).setdefault(m.status, 0)
        coverage[m.type][m.status] += 1
    return AuditReport(
        status="pass" if all(x.passed for x in rules) else "fail",
        created=datetime.now(timezone.utc).isoformat(),
        snapshot_hash=snapshot_hash(data), corpus_hash=data.corpus_hash(), rules=rules, coverage=coverage,
        unreviewed_sections=unreviewed, unresolved_xrefs=data.meta.get("unresolved_xrefs", []),
        overrides=overrides or [],
        counts={"acts": len(data.acts), "sections": len(data.sections), "provisions": len(data.provisions),
                "chunks": len(data.chunks), "mappings": len(data.mappings), "sources": len(data.sources),
                "xrefs": len(data.xrefs)},
    )


def write_report(report: AuditReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.created[:19].replace(":", "-")
    jp = out_dir / f"{stamp}.json"
    jp.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    lines = [f"# Graph audit — {report.status.upper()}", "", f"- created: {report.created}",
             f"- snapshot hash: `{report.snapshot_hash}`", f"- corpus hash: `{report.corpus_hash}`",
             f"- counts: {json.dumps(report.counts)}", "", "| Rule | Result | Details | Warnings |",
             "|---|---|---|---|"]
    for r in report.rules:
        lines.append(f"| {r.rule} | {'pass' if r.passed else 'FAIL'} | {'; '.join(r.details[:5])} | "
                     f"{'; '.join(r.warnings[:5])} |")
    lines += ["", "## Mapping coverage (type × status)", "", "| Type | Status | n |", "|---|---|---|"]
    for t, st in sorted(report.coverage.items()):
        for s, n in sorted(st.items()):
            lines.append(f"| {t} | {s} | {n} |")
    lines += ["", f"## Unreviewed sections ({len(report.unreviewed_sections)})", "",
              ", ".join(report.unreviewed_sections) or "none", "",
              f"## Unresolved cross-references ({len(report.unresolved_xrefs)})", ""]
    lines += [f"- {x}" for x in report.unresolved_xrefs[:50]]
    (out_dir / f"{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jp

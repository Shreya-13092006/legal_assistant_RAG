"""Reports built ONLY from runs/ (SDD 7.9). No LLM calls are made when (re)building a report."""
from __future__ import annotations

import json
from pathlib import Path

from lawrag.config import PROJECT_ROOT
from lawrag.eval import stats as S
from lawrag.manifest import sha256_file

PRIMARY = {  # experiment -> (baseline cell, treatment cell, metric, kind)
    "e1_retrieval": ("R-V", "R-H", "recall@8", "continuous"),
    "e2_verification": ("V0", "V-DJ", "det_hallucination", "binary"),
    "e3_retry_abstain": ("T0", "T1", "acceptable", "binary"),
}
SECONDARY = {
    "e1_retrieval": [("R-B", "R-H", "recall@8"), ("R-G", "R-H", "recall@8"), ("R-H-G", "R-H", "recall@8")],
    "e2_verification": [("V0", "V-DJ", "claim_unsupported"), ("V0", "V-D", "det_hallucination"),
                        ("V-D", "V-DJ", "det_hallucination")],
    "e3_retry_abstain": [("T0", "T1", "unsupported_answer"), ("T0", "T1", "false_abstention_item")],
}
TABLE_METRICS = {
    "retrieval": ["recall@5", "recall@8", "recall@20", "precision@8", "mrr", "ndcg@8", "mapping_hit",
                  "context_tokens", "retrieval_ms"],
    "end_to_end": ["acceptable", "citation_valid", "claim_support", "claim_unsupported", "det_hallucination",
                   "answered", "abstained", "correctness", "latency_ms", "cost_usd"],
}


def latest_runs(experiment: str, runs_dir: Path = PROJECT_ROOT / "runs") -> dict[str, Path]:
    found: dict[str, Path] = {}
    for d in sorted(runs_dir.glob(f"*_{experiment}_*")):
        if (d / "metrics.json").exists():
            found[d.name.split(f"_{experiment}_", 1)[1]] = d
    return found


def _per_item(d: Path) -> dict[str, dict]:
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    rows = {r["gold_id"]: r for r in m["per_item"]}
    for r in rows.values():
        if "expected" in r:
            r["false_abstention_item"] = (r["abstained"] if r["expected"] == "answer" else None)
    return rows


def _fmt(x) -> str:
    if x is None or x != x:
        return "—"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def build_report(experiment: str, runs: dict[str, Path] | None = None, out_dir: Path | None = None) -> Path:
    runs = runs or latest_runs(experiment)
    if not runs:
        raise FileNotFoundError(f"no completed runs for {experiment} in runs/")
    out_dir = out_dir or PROJECT_ROOT / "reports" / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    per = {cell: _per_item(d) for cell, d in runs.items()}
    mode = json.loads((next(iter(runs.values())) / "manifest.json").read_text(encoding="utf-8")).get("mode", "end_to_end")
    metrics = TABLE_METRICS.get(mode, TABLE_METRICS["end_to_end"])
    lines = [f"# Report — {experiment}", "", "Built from saved runs only (no model calls).", "",
             "| Run | Cell | manifest sha256 |", "|---|---|---|"]
    for cell, d in runs.items():
        lines.append(f"| {d.name} | {cell} | `{sha256_file(d / 'manifest.json')[:16]}` |")
    lines += ["", "## Metrics (mean [95% bootstrap CI])", "", "| Cell | n | " + " | ".join(metrics) + " |",
              "|---|---|" + "---|" * len(metrics)]
    for cell, rows in per.items():
        cells = []
        for m in metrics:
            vals = [r.get(m) for r in rows.values() if r.get(m) is not None]
            if not vals:
                cells.append("—")
                continue
            mu, lo, hi = S.bootstrap_ci([float(v) for v in vals])
            cells.append(f"{_fmt(mu)} [{_fmt(lo)}, {_fmt(hi)}]")
        lines.append(f"| {cell} | {len(rows)} | " + " | ".join(cells) + " |")

    def compare(base, treat, metric, kind):
        a, b = per.get(base), per.get(treat)
        if not a or not b:
            return None
        ids = sorted(set(a) & set(b))
        xa = [a[i].get(metric) for i in ids]
        xb = [b[i].get(metric) for i in ids]
        pairs = [(x, y) for x, y in zip(xa, xb) if x is not None and y is not None]
        if not pairs:
            return None
        xa, xb = [p[0] for p in pairs], [p[1] for p in pairs]
        diff = S.paired_bootstrap_diff([float(x) for x in xa], [float(y) for y in xb])
        test = S.mcnemar_exact([bool(x) for x in xa], [bool(y) for y in xb]) if kind == "binary" else \
            S.wilcoxon([float(x) for x in xa], [float(y) for y in xb])
        return {"n": len(pairs), "diff": diff, "test": test}

    lines += ["", "## Pre-registered comparisons", ""]
    if experiment in PRIMARY:
        base, treat, metric, kind = PRIMARY[experiment]
        r = compare(base, treat, metric, kind)
        if r:
            lines.append(f"**Primary:** {treat} − {base} on `{metric}`: {_fmt(r['diff'][0])} "
                         f"[{_fmt(r['diff'][1])}, {_fmt(r['diff'][2])}], n={r['n']}, p={_fmt(r['test']['p'])}")
        sec = {}
        for b2, t2, m2 in SECONDARY.get(experiment, []):
            k2 = "binary" if m2 in ("det_hallucination", "acceptable", "unsupported_answer", "false_abstention_item",
                                     "mapping_hit") else "continuous"
            r2 = compare(b2, t2, m2, k2)
            if r2:
                sec[f"{t2} vs {b2} on {m2}"] = (r2, r2["test"]["p"])
        if sec:
            adj = S.holm({k: v[1] for k, v in sec.items() if v[1] == v[1]})
            lines += ["", "| Secondary comparison | diff [95% CI] | n | p | p (Holm) |", "|---|---|---|---|---|"]
            for k, (r2, p) in sec.items():
                lines.append(f"| {k} | {_fmt(r2['diff'][0])} [{_fmt(r2['diff'][1])}, {_fmt(r2['diff'][2])}] | "
                             f"{r2['n']} | {_fmt(p)} | {_fmt(adj.get(k, {}).get('p_holm'))} |")
    lines += ["", "Results with n < 20 per subgroup are descriptive only. A non-significant difference means no "
                  "difference was detected at this sample size, not evidence of no effect."]
    md = out_dir / "report.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _figure(per, metrics, out_dir)
    return md


def _figure(per: dict, metrics: list[str], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    m = metrics[0]
    cells = list(per)
    means = []
    for c in cells:
        vals = [float(r[m]) for r in per[c].values() if r.get(m) is not None]
        means.append(sum(vals) / len(vals) if vals else 0.0)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(cells, means, color="#4c72b0")
    ax.set_ylabel(m)
    ax.set_ylim(0, max(1.0, max(means) * 1.1 if means else 1.0))
    ax.set_title(f"{m} by configuration")
    fig.tight_layout()
    fig.savefig(out_dir / f"{m.replace('@', '_at_')}.png", dpi=150)
    plt.close(fig)

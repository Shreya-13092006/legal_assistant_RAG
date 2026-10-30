"""lawrag CLI (typer): doctor, sources, ingest, review, load, audit, ask, serve, ui, eval, report, mcp."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer

from lawrag.config import PROJECT_ROOT, load_config

app = typer.Typer(add_completion=False, help="IPC <-> BNS Graph RAG", no_args_is_help=True)
sources_app = typer.Typer(help="Register official source documents")
ingest_app = typer.Typer(help="PDF ingestion (M2)")
review_app = typer.Typer(help="Mapping review gate")
eval_app = typer.Typer(help="Evaluation harness")
app.add_typer(sources_app, name="sources")
app.add_typer(ingest_app, name="ingest")
app.add_typer(review_app, name="review")
app.add_typer(eval_app, name="eval")

ConfigOpt = typer.Option([], "--config", "-c", help="Config overlay(s), e.g. providers/fake.yaml")


def _cfg(overlays: list[str]):
    from lawrag.logging_setup import setup_logging
    cfg = load_config(overlays)
    setup_logging(cfg.logging.level, json_format=False)
    return cfg


def _deps(overlays: list[str], rebuild: bool = False):
    from lawrag.services.container import build_deps
    return build_deps(_cfg(overlays), rebuild=rebuild)


@app.command()
def doctor(config: list[str] = ConfigOpt, probe_llm: bool = typer.Option(True, help="Send a JSON probe to the LLM")):
    """Environment health check (M0): Python, graph backend, embedder, LLM, with resource figures."""
    import platform

    from lawrag.index.embedder import make_embedder
    cfg = _cfg(config)
    ok = lambda b: typer.style("OK " if b else "FAIL", fg="green" if b else "red")
    typer.echo(f"{ok(sys.version_info[:2] == (3, 11))} Python {platform.python_version()} (3.11 expected)")
    if cfg.graph.backend == "neo4j":
        try:
            from lawrag.graph.neo4j_store import Neo4jGraphStore
            s = Neo4jGraphStore(cfg.neo4j.uri, cfg.neo4j.user, os.environ.get("NEO4J_PASSWORD", ""), cfg.neo4j.database)
            typer.echo(f"{ok(True)} Neo4j {cfg.neo4j.uri}: {s.health()}")
        except Exception as exc:
            typer.echo(f"{ok(False)} Neo4j {cfg.neo4j.uri}: {exc}")
    else:
        typer.echo(f"{ok(True)} graph backend: memory (snapshot {cfg.graph.snapshot_path})")
    t0 = time.perf_counter()
    try:
        emb = make_embedder(cfg.embedding)
        v = emb.encode([f"probe sentence {i}" for i in range(10)])
        dt = time.perf_counter() - t0
        rss = _rss_mb()
        typer.echo(f"{ok(True)} embedder {emb.model_id}: dim {emb.dim}, 10 sentences in {dt:.2f}s"
                   f"{f', process RSS {rss:.0f} MB' if rss else ''}")
        if cfg.embedding.model.startswith("hashing"):
            typer.echo("     note: hashing embedder is a lexical stand-in; use -c embedding/bge_m3.yaml for real runs")
    except Exception as exc:
        typer.echo(f"{ok(False)} embedder {cfg.embedding.model}: {exc}")
    if probe_llm:
        from pydantic import BaseModel

        from lawrag.llm.client import LLMRegistry, Message

        class Probe(BaseModel):
            answer: str
            ok: bool

        try:
            llm = LLMRegistry(cfg.llm, PROJECT_ROOT).get("generator")
            t0 = time.perf_counter()
            res = llm.complete("# ROLE: probe\nReply with JSON only.",
                               [Message(role="user", content='Return {"answer": "pong", "ok": true}')], schema=Probe)
            typer.echo(f"{ok(True)} LLM {llm.provider}/{llm.model}: valid JSON in {time.perf_counter() - t0:.1f}s")
        except Exception as exc:
            typer.echo(f"{ok(False)} LLM {cfg.llm.role('generator').provider}/{cfg.llm.role('generator').model}: {exc}")


def _rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return None


# -- sources / ingestion / review -----------------------------------------------------------------
@sources_app.command("register")
def sources_register(pdf: Path, kind: str = typer.Option(...), authority_rank: int = typer.Option(...),
                     title: str = typer.Option(...), source_id: Optional[str] = None, url: Optional[str] = None,
                     publisher: Optional[str] = None):
    from lawrag.ingest.sources import register_source
    rec = register_source(pdf, PROJECT_ROOT / "data" / "raw", kind=kind, authority_rank=authority_rank, title=title,
                          source_id=source_id, url=url, publisher=publisher)
    typer.echo(rec.model_dump_json(indent=2))


@ingest_app.command("act")
def ingest_act(pdf: Path, code: str = typer.Option(..., help="IPC or BNS"), name: str = typer.Option(...),
               year: int = typer.Option(...), text_as_of: str = typer.Option(...), source_id: str = typer.Option(...),
               expected: Optional[Path] = typer.Option(None, help="File with expected section numbers, one per line"),
               pages: Optional[str] = None):
    """Parse an act PDF -> data/processed/<act>.tree.json (fails loudly on validation errors)."""
    from lawrag.ingest.pdf_parser import body_blocks, parse_native, write_jsonl
    from lawrag.ingest.section_splitter import parse_act
    from lawrag.ingest.tree_builder import validate_tree
    pg = [int(x) for x in pages.split(",")] if pages else None
    blocks = parse_native(pdf, source_id, pg)
    write_jsonl(blocks, PROJECT_ROOT / "data" / "interim" / f"{source_id}.blocks.jsonl")
    tree, applied = parse_act([b for b in blocks if b.kind in ("text", "footnote")],
                              {"code": code, "name": name, "year": year, "text_as_of": text_as_of}, source_id,
                              PROJECT_ROOT / "review" / "overrides.yaml")
    warnings = validate_tree(tree, expected.read_text().split() if expected else None)
    out = PROJECT_ROOT / "data" / "processed" / f"{code.lower()}.tree.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tree.model_dump_json(indent=1), encoding="utf-8")
    typer.echo(f"{len(tree.sections)} sections, {len(tree.provisions)} provisions -> {out}")
    for w in warnings:
        typer.echo(f"warning: {w}")
    if applied:
        typer.echo(f"overrides applied: {applied}")
    _ = body_blocks


@ingest_app.command("concordance")
def ingest_concordance(pdf: Path, source_id: str = typer.Option(...), ipc_col: int = 0, bns_col: int = 1,
                       remark_col: Optional[int] = None, pages: Optional[str] = None):
    from lawrag.ingest.concordance_extractor import extract_tables, write_rows
    rows, sus = extract_tables(pdf, source_id, ipc_col, bns_col, remark_col,
                               [int(x) for x in pages.split(",")] if pages else None)
    out = PROJECT_ROOT / "data" / "interim" / f"concordance_{source_id}.csv"
    write_rows(rows, out)
    typer.echo(f"{len(rows)} rows ({sum(1 for r in rows if r.parse_error)} with parse errors) -> {out}")
    if sus:
        typer.echo(f"pages flagged for possible column swap: {sus}")


@review_app.command("queue")
def review_queue():
    """Reconcile all concordance CSVs against the act trees -> review/queue.csv."""
    from lawrag.ingest.concordance_extractor import read_rows
    from lawrag.ingest.mapping_reconciler import reconcile, write_queue
    from lawrag.models.legal import ActTree
    trees = [ActTree.model_validate_json(p.read_text(encoding="utf-8"))
             for p in sorted((PROJECT_ROOT / "data" / "processed").glob("*.tree.json"))]
    rows = [r for p in sorted((PROJECT_ROOT / "data" / "interim").glob("concordance_*.csv")) for r in read_rows(p)]
    cands = reconcile(rows, trees)
    write_queue(cands, trees, PROJECT_ROOT / "review" / "queue.csv")
    typer.echo(f"{len(cands)} candidates -> review/queue.csv (status 'proposed'; a human must review them)")


@review_app.command("validate")
def review_validate(path: Path = PROJECT_ROOT / "review" / "approved_mappings.csv"):
    import json as _j

    from lawrag.ingest.review_io import read_approved
    from lawrag.models.legal import ActTree
    trees = [ActTree.model_validate_json(p.read_text(encoding="utf-8"))
             for p in sorted((PROJECT_ROOT / "data" / "processed").glob("*.tree.json"))]
    known = {s.uid for t in trees for s in t.sections} | {p.uid for t in trees for p in t.provisions}
    srcs = {s["source_id"] for s in _j.loads((PROJECT_ROOT / "data" / "raw" / "sources.json").read_text())}
    recs = read_approved(path, known, srcs)
    typer.echo(f"{len(recs)} reviewed mappings are valid")


# -- graph ---------------------------------------------------------------------------------------
@app.command()
def load(config: list[str] = ConfigOpt, source: str = typer.Option("seed", help="seed | processed")):
    """Build GraphData (chunks, embeddings, xrefs, mappings) and write it to the configured backend."""
    from lawrag.graph.loader import build_from_processed, build_from_seed
    from lawrag.index.embedder import make_embedder
    cfg = _cfg(config)
    emb = make_embedder(cfg.embedding, cfg.path(cfg.embedding.cache_path))
    data = build_from_seed(cfg, emb) if source == "seed" else build_from_processed(
        cfg, emb, PROJECT_ROOT / "data" / "processed", PROJECT_ROOT / "review" / "approved_mappings.csv",
        PROJECT_ROOT / "data" / "raw" / "sources.json")
    data.save(cfg.path(cfg.graph.snapshot_path))
    if cfg.graph.backend == "neo4j":
        from lawrag.services.container import make_store
        make_store(cfg, emb).write(data)
    from lawrag.graph.data import snapshot_hash
    typer.echo(f"loaded {len(data.sections)} sections, {len(data.provisions)} provisions, {len(data.chunks)} chunks, "
               f"{len(data.mappings)} mappings into {cfg.graph.backend}; snapshot {snapshot_hash(data)[:16]}")


@app.command("audit")
def audit_cmd(config: list[str] = ConfigOpt):
    """Run graph integrity rules G1-G12; writes reports/audit/<timestamp>.{json,md}."""
    from lawrag.graph.audit import audit, write_report
    deps = _deps(config)
    expected = {}
    for act in deps.cfg.acts:
        p = PROJECT_ROOT / "data" / "expected" / f"{act}.txt"
        if p.exists():
            expected[act] = p.read_text().split()
    rep = audit(deps.store.export(), expected=expected or None, raw_dir=PROJECT_ROOT / "data" / "raw",
                active_model=deps.embedder.model_id)
    path = write_report(rep, PROJECT_ROOT / "reports" / "audit")
    typer.echo(f"audit {rep.status.upper()} -> {path}")
    for r in rep.rules:
        typer.echo(f"  {r.rule}: {'pass' if r.passed else 'FAIL'} {r.details[:2]} {r.warnings[:2]}")
    raise typer.Exit(0 if rep.status == "pass" else 1)


# -- QA ------------------------------------------------------------------------------------------
@app.command()
def ask(question: str, config: list[str] = ConfigOpt, debug: bool = False, as_json: bool = False):
    from lawrag.models.qa import AskOptions
    from lawrag.services.qa_service import QAService
    resp = QAService(_deps(config)).ask(question, AskOptions(debug=debug))
    if as_json:
        typer.echo(resp.model_dump_json(indent=2))
        return
    typer.echo(f"[{resp.outcome.upper()}] (decision rule {resp.verification.decision_rule})\n")
    typer.echo(resp.answer_markdown)
    if resp.unverified_claims and resp.outcome == "partial":
        typer.echo("\nCould not verify:\n" + "\n".join(f"  - {c}" for c in resp.unverified_claims))
    typer.echo("\nChecks: " + ", ".join(f"{c.check}={'ok' if c.passed else 'FAIL'}"
                                       for c in resp.verification.deterministic.checks))
    if resp.verification.claims:
        typer.echo("Claims: " + ", ".join(f"{c.verdict}" for c in resp.verification.claims))
    if resp.retry_actions:
        typer.echo("Retry: " + "; ".join(resp.retry_actions))
    typer.echo(f"\n{resp.disclaimer}")


@app.command()
def serve(config: list[str] = ConfigOpt, host: str = "127.0.0.1", port: int = 8000):
    """Start the FastAPI backend (single worker; models are shared in-process)."""
    import uvicorn
    os.environ["LAWRAG_CONFIG"] = ",".join(config)
    uvicorn.run("lawrag.api.app:create_app", factory=True, host=host, port=port, workers=1)


@app.command()
def ui(port: int = 8501, api: str = "http://localhost:8000"):
    """Start the Streamlit UI (expects `lawrag serve` running)."""
    env = {**os.environ, "LAWRAG_API": api}
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent / "ui" / "streamlit_app.py"),
                    "--server.port", str(port)], env=env, check=False)


@app.command()
def mcp(config: list[str] = ConfigOpt):
    """Optional MCP server (M5) over the service layer (stdio)."""
    from lawrag.mcp_server.lawgraph_server import run
    run(_deps(config))


# -- evaluation ----------------------------------------------------------------------------------
@eval_app.command("validate")
def eval_validate(path: Path = PROJECT_ROOT / "data" / "gold" / "qa_v1.jsonl", config: list[str] = ConfigOpt):
    from lawrag.eval import dataset as ds
    deps = _deps(config)
    items = ds.load(path)
    errs = ds.validate(items, deps.store)
    dups = ds.near_duplicates(items, deps.embedder)
    typer.echo(f"{len(items)} items; {len(errs)} errors; {len(dups)} dev/test near-duplicates")
    for e in errs:
        typer.echo(f"  error: {e}")
    for d in dups:
        typer.echo(f"  near-duplicate: {d}")
    raise typer.Exit(1 if errs else 0)


@eval_app.command("lock")
def eval_lock(path: Path = PROJECT_ROOT / "data" / "gold" / "qa_v1.jsonl"):
    from lawrag.eval import dataset as ds
    h = ds.lock_split(ds.load(path), path.parent / "splits.lock")
    typer.echo(f"test split locked: {h}")


@eval_app.command("run")
def eval_run(matrix: str, split: str = "dev", config: list[str] = ConfigOpt, dataset: Optional[str] = None,
             allow_dirty_graph: bool = False, resume: Optional[str] = None, limit: Optional[int] = None,
             yes: bool = False, grade: bool = False):
    from lawrag.eval.runner import run_matrix
    deps = _deps(config)
    dirs = run_matrix(matrix, deps, split, config, allow_dirty_graph, resume, limit, yes, grade, dataset)
    for d in dirs:
        m = json.loads((d / "metrics.json").read_text())["aggregate"]
        keys = [k for k in ("recall@8", "mrr", "mapping_hit", "acceptable", "det_hallucination", "citation_valid",
                            "coverage") if k in m]
        typer.echo(f"{d.name}: n={m['n']} " + " ".join(f"{k}={m[k]:.3f}" for k in keys if m[k] is not None))


@app.command()
def report(experiment: str):
    """Build reports/<experiment>/report.md from saved runs only."""
    from lawrag.eval.report import build_report
    typer.echo(f"report -> {build_report(experiment)}")


if __name__ == "__main__":
    app()

"""Evaluation runner (SDD 6.5 / 7): resolves matrix cells into RunConfigs, calls the service layer
in-process item by item, and writes resumable runs/<run_id>/{manifest,predictions,traces,metrics}."""
from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from lawrag.config import CONFIG_DIR, PROJECT_ROOT, load_config
from lawrag.errors import BudgetExceeded, ValidationError
from lawrag.eval import dataset as ds
from lawrag.eval import metrics as M
from lawrag.graph.audit import audit
from lawrag.logging_setup import get_logger, run_id_var
from lawrag.manifest import build_manifest, sha256_file, utc_stamp, write_json
from lawrag.qa.workflow import Workflow
from lawrag.services.container import Deps

log = get_logger("eval.runner")
RUNS = PROJECT_ROOT / "runs"
TEST_LOG = PROJECT_ROOT / "runs" / "test_split_access.log"


def load_matrix(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        p = CONFIG_DIR / "matrices" / Path(path).name
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _guard_test(items, matrix_name: str) -> dict:
    plan = PROJECT_ROOT / "docs" / "ANALYSIS_PLAN.md"
    if not plan.exists():
        raise ValidationError("the test split can only be run after docs/ANALYSIS_PLAN.md is frozen (SDD 7.2.5)")
    lock_hash = ds.check_lock(items, PROJECT_ROOT / "data" / "gold" / "splits.lock")
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TEST_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"time": utc_stamp(), "matrix": matrix_name, "lock": lock_hash,
                             "analysis_plan_sha256": sha256_file(plan)}) + "\n")
    return {"split_lock_sha256": lock_hash, "analysis_plan_sha256": sha256_file(plan)}


def _estimate_cost(deps: Deps, n_items: int, mode: str) -> float:
    if mode != "end_to_end":
        return 0.0
    total = 0.0
    for role, tin, tout, calls in (("generator", 3000, 600, 1.5), ("judge", 900, 120, 6)):
        rc = deps.cfg.llm.role(role)
        if rc.provider in ("ollama", "fake"):
            continue
        total += deps.llm.budget.estimate(rc.provider, rc.model, tin, tout, int(calls * n_items))
    return total


def run_matrix(matrix_path: str | Path, base_deps: Deps, split: str = "dev", base_overlays: list[str] | None = None,
               allow_dirty_graph: bool = False, resume: str | None = None, limit: int | None = None,
               yes: bool = False, grade: bool = False, dataset: str | None = None) -> list[Path]:
    mx = load_matrix(matrix_path)
    if dataset:
        mx["dataset"] = dataset
    data_path = PROJECT_ROOT / mx["dataset"]
    items = ds.load(data_path)
    errs = ds.validate(items, base_deps.store)
    if errs:
        raise ValidationError(f"dataset invalid: {errs[:5]}", errors=errs)
    extra_ds = {}
    if split == "test":
        extra_ds = _guard_test(items, mx["name"])
    items = [i for i in items if i.split == split][: limit or None]
    if not items:
        raise ValidationError(f"no items in split {split!r} of {data_path.name}")
    rep = audit(base_deps.store.export(), active_model=base_deps.embedder.model_id)
    if rep.status != "pass" and not allow_dirty_graph:
        raise ValidationError("graph audit failed; fix it or pass --allow-dirty-graph (recorded in the manifest)")
    out_dirs = []
    stamp = resume or utc_stamp()
    for cell, overlays in mx["cells"].items():
        cfg = load_config((base_overlays or []) + list(overlays))
        deps = base_deps.with_config(cfg)
        est = _estimate_cost(deps, len(items), mx["mode"])
        if est > deps.llm.budget.remaining() and not yes:
            raise BudgetExceeded(f"cell {cell}: estimated {est:.2f} USD exceeds the remaining budget "
                                 f"{deps.llm.budget.remaining():.2f}; pass --yes to proceed anyway")
        run_id = f"{stamp}_{mx['name']}_{cell}"
        run_id_var.set(run_id)
        d = RUNS / run_id
        d.mkdir(parents=True, exist_ok=True)
        pred_path, trace_path = d / "predictions.jsonl", d / "traces.jsonl"
        done = set()
        if pred_path.exists():
            done = {json.loads(l)["gold_id"] for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        manifest = build_manifest(
            run_id, cfg, corpus_hash=deps.corpus_hash, graph_snapshot_hash=deps.snapshot_hash,
            audit_status=rep.status if not allow_dirty_graph else f"{rep.status} (allow-dirty-graph)",
            dataset={"path": mx["dataset"], "sha256": sha256_file(data_path), "split": split, **extra_ds},
            models={"embedding": {"id": deps.embedder.model_id, "revision": deps.embedder.revision},
                    **{r: {"provider": c.provider, "model": c.model} for r, c in cfg.llm.roles.items()}},
            extra={"matrix": mx["name"], "cell": cell, "overlays": overlays, "mode": mx["mode"]})
        write_json(d / "manifest.json", manifest)
        wf = Workflow(deps)
        eval_judge = deps.llm.get("eval_judge") if grade else None
        with pred_path.open("a", encoding="utf-8") as pf, trace_path.open("a", encoding="utf-8") as tf:
            for it in items:
                if it.id in done:
                    continue
                t0 = time.perf_counter()
                if mx["mode"] == "retrieval":
                    st = wf.run_retrieval(it.question)
                    ctx = st["context"]
                    pred = {"gold_id": it.id, "retrieved_ranking": [c.uid for c in st["fused"]],
                            "mapping_facts": [f.model_dump() for f in ctx.mapping_facts],
                            "context_uids": sorted(ctx.uids()), "context_tokens": ctx.token_count,
                            "retrieval_ms": round((time.perf_counter() - t0) * 1000, 1)}
                    trace = st["trace"]
                else:
                    try:
                        st = wf.run(it.question, run_id=run_id, request_id=f"{run_id}:{it.id}")
                    except BudgetExceeded:
                        log.error("budget exceeded mid-run; resume later with --resume " + stamp)
                        raise
                    resp = st["final"]
                    pred = {"gold_id": it.id, **resp.model_dump(mode="json", exclude={"trace"})}
                    pred["context_tokens"] = st["context"].token_count if st.get("context") else 0
                    if eval_judge is not None:
                        v = __import__("lawrag.eval.judge", fromlist=["grade"]).grade(pred, it, eval_judge)
                        pred["eval_judge"] = v.model_dump()
                    trace = resp.trace or st.get("trace", [])
                pf.write(json.dumps(pred, default=str) + "\n")
                pf.flush()
                tf.write(json.dumps({"gold_id": it.id, "trace": [t.model_dump() for t in trace]}, default=str) + "\n")
        write_json(d / "metrics.json", compute_metrics(d, items))
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        manifest["finished"] = utc_stamp()
        preds = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        manifest["cost_usd"] = round(sum(p.get("cost_estimate_usd", 0) or 0 for p in preds), 6)
        write_json(d / "manifest.json", manifest)
        out_dirs.append(d)
        log.info(f"run complete {run_id}")
    return out_dirs


def load_predictions(run_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def compute_metrics(run_dir: Path, items) -> dict:
    by_id = {i.id: i for i in items}
    preds = load_predictions(run_dir)
    mode = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")).get("mode")
    rows = []
    for p in preds:
        it = by_id.get(p["gold_id"])
        if it is None:
            continue
        if mode == "retrieval":
            if it.expected_behavior == "abstain" and it.abstain_reason == "out_of_scope":
                continue
            rows.append({"gold_id": it.id, "category": it.category, **M.retrieval_item_metrics(p, it)})
        else:
            score = (p.get("eval_judge") or {}).get("score")
            rows.append({"gold_id": it.id, "category": it.category, **M.e2e_item_metrics(p, it, score)})
    return {"aggregate": M.aggregate(rows), "per_item": rows}

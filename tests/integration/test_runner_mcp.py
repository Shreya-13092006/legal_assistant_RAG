"""Evaluation runner (resume, manifest completeness), report building, MCP equivalence."""
import json

import pytest

from lawrag.eval import runner as R
from lawrag.eval.report import build_report
from lawrag.mcp_server.lawgraph_server import tool_functions
from lawrag.services.law_service import LawService


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RUNS", tmp_path)
    return tmp_path


def test_smoke_matrix_end_to_end_and_resume(deps, runs_dir):
    dirs = R.run_matrix("smoke.yaml", deps, split="smoke", base_overlays=["providers/fake.yaml"], resume="T")
    d = dirs[0]
    man = json.loads((d / "manifest.json").read_text())
    for key in ("run_id", "config_hash", "corpus_hash", "graph_snapshot_hash", "audit_status", "dataset", "models",
                "environment", "seed", "finished"):
        assert man[key] is not None, key
    preds = (d / "predictions.jsonl").read_text().splitlines()
    assert len(preds) == 14
    m = json.loads((d / "metrics.json").read_text())["aggregate"]
    assert m["n"] == 14 and m["acceptable"] >= 0.7
    # resume: nothing re-run
    R.run_matrix("smoke.yaml", deps, split="smoke", base_overlays=["providers/fake.yaml"], resume="T")
    assert len((d / "predictions.jsonl").read_text().splitlines()) == 14


def test_retrieval_matrix_and_report(deps, runs_dir, tmp_path):
    dirs = R.run_matrix("e1_retrieval.yaml", deps, split="smoke", base_overlays=["providers/fake.yaml"],
                        dataset="data/gold/smoke_v0.jsonl", resume="X")
    assert len(dirs) == 5
    agg = {d.name: json.loads((d / "metrics.json").read_text())["aggregate"] for d in dirs}
    hybrid = next(v for k, v in agg.items() if k.endswith("_R-H"))
    assert hybrid["recall@8"] >= 0.8 and hybrid["mapping_hit"] == 1.0
    report = build_report("e1_retrieval", {d.name.rsplit("_", 1)[1]: d for d in dirs}, tmp_path / "rep")
    text = report.read_text()
    assert "Primary:" in text and "R-H" in text


def test_test_split_guard(deps, runs_dir):
    from lawrag.errors import ValidationError
    with pytest.raises(ValidationError):
        R.run_matrix("smoke.yaml", deps, split="test", base_overlays=["providers/fake.yaml"])


def test_mcp_tools_equal_services(deps):
    tools = tool_functions(deps)
    law = LawService(deps)
    for uid in ["IPC:498A", "BNS:103", "IPC:497", "BNS:304", "IPC:377"]:
        assert json.loads(tools["get_section"](uid)) == law.get_section(uid).model_dump(mode="json")
        assert json.loads(tools["map_section"](uid)) == law.get_mappings(uid).model_dump(mode="json")
    assert json.loads(tools["get_section"]("BNS:999"))["error"]["code"] == "NOT_FOUND"
    assert json.loads(tools["compare_sections"]("IPC:304A"))["pairs"]

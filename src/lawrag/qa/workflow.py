"""QA workflow wiring (SDD 2.6 / 3.5). LangGraph StateGraph by default; an equivalent plain-Python
runner is used when `workflow.engine = python` or LangGraph is unavailable."""
from __future__ import annotations

import uuid
from functools import partial

from lawrag.llm.client import start_usage_sink
from lawrag.logging_setup import request_id_var
from lawrag.models.qa import AskOptions, QAResponse
from lawrag.qa import nodes as N
from lawrag.services.container import Deps

NODES = {
    "analyze_query": N.analyze_query, "retrieve": N.retrieve, "graph_expand": N.graph_expand, "fuse": N.fuse,
    "build_context": N.build_context, "generate": N.generate, "verify_deterministic": N.verify_deterministic,
    "verify_claims": N.verify_claims, "decide": N.decide, "plan_retry": N.plan_retry, "abstain": N.abstain,
    "finalize": N.finalize,
}


def _after_analyze(state, deps) -> str:
    return "abstain" if state["analysis"].out_of_scope else "retrieve"


ROUTERS = {
    "analyze_query": _after_analyze,
    "build_context": N.evidence_gate,
    "generate": N.after_generate,
    "decide": N.route_decision,
    "plan_retry": N.route_retry,
}
EDGES = {"retrieve": "graph_expand", "graph_expand": "fuse", "fuse": "build_context",
         "verify_deterministic": "verify_claims", "verify_claims": "decide"}
TERMINAL = {"abstain", "finalize"}


def build_langgraph(deps: Deps):
    from langgraph.graph import END, START, StateGraph

    from lawrag.qa.state import QAState

    g = StateGraph(QAState)
    for name, fn in NODES.items():
        g.add_node(name, partial(fn, deps=deps))
    g.add_edge(START, "analyze_query")
    for src, dst in EDGES.items():
        g.add_edge(src, dst)
    for src, router in ROUTERS.items():
        g.add_conditional_edges(src, partial(router, deps=deps))
    for t in TERMINAL:
        g.add_edge(t, END)
    return g.compile()


def run_python(state: dict, deps: Deps) -> dict:
    node = "analyze_query"
    steps = 0
    while True:
        steps += 1
        if steps > 40:
            raise RuntimeError("workflow did not terminate")
        state = {**state, **NODES[node](state, deps)}
        if node in TERMINAL:
            return state
        node = ROUTERS[node](state, deps) if node in ROUTERS else EDGES[node]


class Workflow:
    def __init__(self, deps: Deps):
        self.deps = deps
        self.graph = None
        if deps.cfg.workflow.engine == "langgraph":
            try:
                self.graph = build_langgraph(deps)
            except ImportError:
                self.graph = None

    def run(self, question: str, options: AskOptions | None = None, run_id: str | None = None,
            request_id: str | None = None) -> dict:
        rid = request_id or uuid.uuid4().hex[:12]
        request_id_var.set(rid)
        state = {"request_id": rid, "run_id": run_id, "config_hash": self.deps.cfg.config_hash(),
                 "question": question, "options": options or AskOptions(), "attempt": 0, "trace": [],
                 "retry_actions": [], "feedback": [], "usage": start_usage_sink()}
        if self.graph is not None:
            return self.graph.invoke(state, {"recursion_limit": 60})
        return run_python(state, self.deps)

    def ask(self, question: str, options: AskOptions | None = None, run_id: str | None = None) -> QAResponse:
        return self.run(question, options, run_id)["final"]

    def run_retrieval(self, question: str) -> dict:
        """E1 mode: analysis, retrieval, expansion, fusion and context only (no generation)."""
        state = {"request_id": uuid.uuid4().hex[:12], "question": question, "options": AskOptions(), "attempt": 0,
                 "trace": [], "usage": start_usage_sink()}
        for name in ("analyze_query", "retrieve", "graph_expand", "fuse", "build_context"):
            state = {**state, **NODES[name](state, self.deps)}
        return state

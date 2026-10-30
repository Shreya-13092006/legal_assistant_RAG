"""Optional MCP server `lawgraph` (SDD 3.7.3, Milestone 5). Each tool is a thin wrapper over a
service-layer function; nothing in retrieval, the workflow or evaluation imports this module.
The FastMCP API should be re-checked against the installed `mcp` package (VERIFY A-14)."""
from __future__ import annotations

import functools
import json

from lawrag.errors import LawRagError
from lawrag.services.container import Deps
from lawrag.services.law_service import LawService
from lawrag.services.qa_service import QAService

MAX_CHARS = 20_000


def _out(obj) -> str:
    text = json.dumps(obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj, default=str)
    if len(text) > MAX_CHARS:
        return json.dumps({"truncated": True, "data": text[:MAX_CHARS]})
    return text


def _safe(fn):
    @functools.wraps(fn)  # keeps the signature so the MCP SDK can build the tool schema
    def inner(*a, **k) -> str:
        try:
            return _out(fn(*a, **k))
        except LawRagError as exc:
            return json.dumps({"error": {"code": exc.code, "message": exc.message or str(exc)}})
    return inner


def tool_functions(deps: Deps) -> dict:
    law, qa = LawService(deps), QAService(deps)

    def get_section(uid: str) -> object:
        """Provision text, qualifiers, status and source for a section uid such as BNS:103 or IPC:498A."""
        return law.get_section(uid)

    def map_section(uid: str, direction: str | None = None) -> object:
        """Reviewed IPC<->BNS mapping hubs (type, status, evidence) for a section uid."""
        return law.get_mappings(uid, direction)

    def search_law(query: str, k: int = 10) -> object:
        """Hybrid retrieval over IPC/BNS provisions; returns ranked uids."""
        return [{"uid": c.uid, "score": c.fused_score, "via": c.via} for c in law.search(query, "hybrid", k)]

    def compare_sections(left_uid: str, right_uid: str | None = None) -> object:
        """Side-by-side comparison with a mechanical word diff."""
        return law.compare(left_uid, right_uid)

    def verify_citations(answer: str, context_uids: list[str]) -> object:
        """Run deterministic checks D1-D7 on an answer against the given context uids."""
        return qa.verify(answer, context_uids)

    def ask(question: str) -> object:
        """Full cited, verified QA workflow."""
        return qa.ask(question)

    return {f.__name__: _safe(f) for f in (get_section, map_section, search_law, compare_sections,
                                           verify_citations, ask)}


def run(deps: Deps) -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit("the optional MCP server needs `pip install mcp`") from exc
    server = FastMCP("lawgraph")
    for fn in tool_functions(deps).values():
        server.tool()(fn)
    server.run()

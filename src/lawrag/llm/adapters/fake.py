"""FakeAdapter: scripted responses for tests, or a deterministic extractive stand-in for every role.

The default (no script) behaviour is a rule-based "LLM" that only copies text from the prompt:
the generator quotes source blocks and mapping statements, the judge checks verbatim/lexical
support. It lets the whole pipeline and UI run offline. It is not a model and its outputs are not
experiment results.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from lawrag.legal.numbers import punishment_terms

Script = Callable[[str, str, str], "str | dict"]
_STOP = set("the a an of to in and or is be by with for any such as that this shall under which".split())


def _role(system: str) -> str:
    m = re.search(r"#\s*ROLE:\s*(\w+)", system)
    return m.group(1) if m else "generator"


def _field(text: str, name: str) -> str:
    m = re.search(rf"^{name}:\s*(.*)$", text, re.M)
    return m.group(1).strip() if m else ""


def _blocks(text: str) -> list[tuple[str, str]]:
    src = text.split("SOURCES:", 1)[1] if "SOURCES:" in text else text.split("CITED TEXT:", 1)[-1]
    out = []
    for chunk in re.split(r"\n(?=\[)", src.strip()):
        m = re.match(r"\[([^\]]+)\]\s*(.*)", chunk.strip(), re.S)
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


def _words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in _STOP}


def fake_generate(user: str) -> dict:
    mode = _field(user, "MODE")
    nonexistent = [x for x in _field(user, "NONEXISTENT SECTIONS").split(",") if x.strip() and x.strip() != "none"]
    claims, answer = [], []
    for uid in nonexistent:
        act, num = uid.strip().split(":", 1)
        t = f"Section {num} does not exist in the {act} as indexed in this system."
        claims.append({"text": t, "citations": [], "core": True})
        answer.append(t)
    blocks = _blocks(user)
    has_mapping = any(lbl.startswith("MAPPING") for lbl, _ in blocks)
    n_prov = 0
    for label, body in blocks:
        if label.startswith("MAPPING"):
            mid = label.split()[1]
            stmt = re.search(r"Statement:\s*(.+?)(?:\s+Change summary:|\s+Note:|$)", body, re.S)
            if stmt:
                t = stmt.group(1).strip()
                claims.append({"text": t, "citations": [mid], "core": True})
                answer.append(f"{t} [{mid}]")
        elif re.match(r"^(IPC|BNS):", label) and n_prov < (2 if has_mapping else 3):
            m = re.match(r"(.+?)\s*\|\s*((?:[^:(]|\([^)]*\))*):\s*(.*)$", body, re.S)
            text = (m.group(3) if m else body).strip()
            text = re.sub(r"\s*\[NOT IN FORCE[^\]]*\]\s*", " ", text).strip()
            t = f'{label} provides: "{text}"'
            core = n_prov == 0 and not has_mapping and mode != "premise_correction"
            claims.append({"text": t, "citations": [label], "core": core})
            answer.append(f"{t} [{label}]")
            if "NOT IN FORCE" in body:
                t2 = f"{label} is not in force."
                claims.append({"text": t2, "citations": [label], "core": False})
                answer.append(f"{t2} [{label}]")
            n_prov += 1
    if not claims:
        return {"answer": "The provided sources do not contain enough information to answer.", "claims": [],
                "caveats": []}
    return {"answer": "\n\n".join(answer), "claims": claims, "caveats": []}


def fake_judge(user: str) -> dict:
    claim = _field(user, "CLAIM")
    cited = "\n".join(body for _, body in _blocks(user))
    if not cited.strip():
        return {"verdict": "NOT_ENOUGH_INFO", "evidence_span": "", "rationale": "no cited text"}
    quoted = re.search(r'"(.+)"', claim, re.S)
    if quoted and quoted.group(1).strip() in cited:
        return {"verdict": "SUPPORTED", "evidence_span": quoted.group(1).strip(), "rationale": "verbatim quote"}
    extra = punishment_terms(claim) - punishment_terms(cited)
    if extra:
        return {"verdict": "CONTRADICTED", "evidence_span": "", "rationale": f"terms not in cited text: {sorted(extra)}"}
    sentences = re.split(r"(?<=[.;])\s+", cited)
    cw = _words(claim)
    best = max(sentences, key=lambda s: len(cw & _words(s)))
    overlap = len(cw & _words(cited)) / max(len(cw), 1)
    if overlap >= 0.75:
        return {"verdict": "SUPPORTED", "evidence_span": best.strip(), "rationale": f"lexical overlap {overlap:.2f}"}
    return {"verdict": "NOT_ENOUGH_INFO", "evidence_span": "", "rationale": f"lexical overlap {overlap:.2f}"}


def fake_decompose(user: str) -> dict:
    ans = user.split("ANSWER:", 1)[-1]
    claims = []
    for s in re.split(r"(?<=[.])\s+(?=[A-Z\"])", ans.strip()):
        cites = re.findall(r"\[((?:IPC|BNS|BNSS|BSA):[^\]]+|MAP-\d+)\]", s)
        text = re.sub(r"\s*\[[^\]]+\]", "", s).strip()
        if text:
            claims.append({"text": text, "citations": cites, "core": True})
    return {"claims": claims}


DEFAULT_BEHAVIOUR = {
    "generator": fake_generate, "judge": fake_judge, "decomposer": fake_decompose,
    "analyzer_fallback": lambda u: {"intent": "unknown", "rewritten_query": _field(u, "QUESTION") or u},
    "eval_judge": lambda u: {"score": 1, "rationale": "fake eval judge (no grading performed)"},
    "reranker": lambda u: {"order": re.findall(r"^- (\S+):", u, re.M)},
    "probe": lambda u: {"answer": "pong", "ok": True},
}


class FakeAdapter:
    provider = "fake"
    model = "fake"
    digest = "fake-1"

    def __init__(self, script: Script | None = None):
        self.script = script
        self.calls: list[tuple[str, str]] = []

    def raw_complete(self, system, messages, *, json_schema, temperature, max_tokens, seed):
        from lawrag.llm.client import LLMResult
        role = _role(system)
        user = messages[-1].content
        self.calls.append((role, user))
        out = self.script(role, system, user) if self.script else None
        if out is None:
            out = DEFAULT_BEHAVIOUR.get(role, lambda u: {})(user)
        text = out if isinstance(out, str) else json.dumps(out)
        return LLMResult(text=text, input_tokens=len(user) // 4, output_tokens=len(text) // 4, model_digest="fake-1")

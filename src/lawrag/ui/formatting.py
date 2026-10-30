"""Pure formatting helpers for the Streamlit UI (unit-testable without Streamlit)."""
from __future__ import annotations

import html
import re

OUTCOME_BADGE = {"answered": ("Answered", "#1a7f37"), "partial": ("Partial", "#9a6700"),
                 "abstained": ("Abstained", "#cf222e")}
STATUS_BADGE = {"verified": ("verified", "#1a7f37"), "disputed": ("disputed", "#9a6700"),
                "unresolved": ("unresolved", "#6e7781"), "proposed": ("proposed", "#6e7781"),
                "rejected": ("rejected", "#cf222e")}
VERDICT_ICON = {"SUPPORTED": "✅", "CONTRADICTED": "❌", "NOT_ENOUGH_INFO": "❔", None: "–"}


def badge(label: str, color: str) -> str:
    return (f"<span style='background:{color};color:white;padding:2px 8px;border-radius:10px;"
            f"font-size:0.85em'>{html.escape(label)}</span>")


def outcome_badge(outcome: str) -> str:
    return badge(*OUTCOME_BADGE.get(outcome, (outcome, "#6e7781")))


def status_badge(status: str) -> str:
    return badge(*STATUS_BADGE.get(status, (status, "#6e7781")))


def link_citations(md: str) -> str:
    """Turn [IPC:302] markers into anchor links to the citation cards."""
    return re.sub(r"\[((?:IPC|BNS):[^\]\s|]+|MAP-\d+)\]",
                  lambda m: f"[`{m.group(1)}`](#{anchor(m.group(1))})", md)


def anchor(uid: str) -> str:
    return "cite-" + re.sub(r"[^A-Za-z0-9]+", "-", uid).strip("-").lower()


def check_rows(verification: dict) -> list[dict]:
    rows = []
    for c in (verification.get("deterministic") or {}).get("checks", []):
        rows.append({"check": c["check"], "result": "pass" if c["passed"] else f"FAIL ({c['severity']})",
                     "details": "; ".join(c.get("details", []))})
    return rows


def claim_rows(verification: dict) -> list[dict]:
    return [{"verdict": f"{VERDICT_ICON.get(c.get('verdict'))} {c.get('verdict') or 'not judged'}",
             "core": "core" if c.get("core") else "", "claim": c["text"], "citations": ", ".join(c.get("citations", [])),
             "evidence": c.get("evidence_span") or ""} for c in verification.get("claims", [])]


def diff_html(segments: list[dict], side: str) -> str:
    out = []
    for s in segments:
        text = html.escape(s[side])
        if not text:
            continue
        if s["op"] == "equal":
            out.append(text)
        else:
            color = "#ffebe9" if side == "left" else "#dafbe1"
            out.append(f"<span style='background:{color}'>{text}</span>")
    return " ".join(out)

"""Pure formatting helpers for the Streamlit UI (unit-testable without Streamlit)."""
from __future__ import annotations

import html
import re

OUTCOME_BADGE = {
    "answered": ("Verified Answer", "#047857", "#ECFDF5", "#A7F3D0"),
    "partial": ("Partial Answer", "#B45309", "#FFFBEB", "#FDE68A"),
    "abstained": ("Abstained", "#B91C1C", "#FEF2F2", "#FECACA"),
}
STATUS_BADGE = {
    "verified": ("Verified", "#047857", "#ECFDF5", "#A7F3D0"),
    "disputed": ("Disputed", "#B45309", "#FFFBEB", "#FDE68A"),
    "unresolved": ("Unresolved", "#475569", "#F1F5F9", "#CBD5E1"),
    "proposed": ("Proposed", "#475569", "#F1F5F9", "#CBD5E1"),
    "rejected": ("Rejected", "#B91C1C", "#FEF2F2", "#FECACA"),
}
VERDICT_ICON = {"SUPPORTED": "[PASS]", "CONTRADICTED": "[FAIL]", "NOT_ENOUGH_INFO": "[GAP]", None: "—"}


def badge(label: str, fg: str, bg: str = "", border: str = "") -> str:
    """Render a compact pill badge with explicit accessible contrast."""
    bg_css = bg or fg
    fg_css = fg if bg else "#FFFFFF"
    border_css = f"border:1px solid {border};" if border else ""
    return (
        f"<span style='display:inline-flex;align-items:center;gap:4px;"
        f"background:{bg_css};color:{fg_css};{border_css}"
        f"padding:3px 10px;border-radius:999px;font-size:0.78rem;"
        f"font-weight:600;letter-spacing:0.01em'>{html.escape(label)}</span>"
    )


def outcome_badge(outcome: str) -> str:
    spec = OUTCOME_BADGE.get(outcome)
    return badge(*spec) if spec else badge(outcome, "#475569", "#F1F5F9", "#CBD5E1")


def status_badge(status: str) -> str:
    spec = STATUS_BADGE.get(status)
    return badge(*spec) if spec else badge(status, "#475569", "#F1F5F9", "#CBD5E1")


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
            bg = "#FEE2E2" if side == "left" else "#DCFCE7"
            fg = "#991B1B" if side == "left" else "#166534"
            out.append(
                f"<span style='background:{bg};color:{fg};padding:1px 4px;"
                f"border-radius:4px;font-weight:500'>{text}</span>"
            )
    return " ".join(out)

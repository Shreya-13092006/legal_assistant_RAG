"""Modern Streamlit UI (SDD 3.7.2): Ask, Section Mapper, Compare, System & Scope.

Run: streamlit run src/lawrag/ui/streamlit_app.py   (backend URL from LAWRAG_API, default :8000)
"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lawrag.ui.formatting import (  # noqa: E402
    anchor,
    check_rows,
    claim_rows,
    diff_html,
    link_citations,
    outcome_badge,
    status_badge,
)

API = os.environ.get("LAWRAG_API", "http://localhost:8000").rstrip("/")
EXAMPLES = [
    "What replaced IPC 498A?",
    "What is the punishment for murder under the BNS?",
    "How does BNS 106 differ from IPC 304A?",
    "Is there a BNS equivalent of IPC 497?",
    "What is the BNS equivalent of section 377 IPC?",
    "Which IPC section did BNS 304 come from?",
    "What happened to sedition, IPC 124A?",
    "What is the punishment under BNS section 999?",
]

st.set_page_config(
    page_title="NyayaGraph · IPC ↔ BNS Legal Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
  .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1180px;
  }
  [data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
  }
  .hero-card {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 100%);
    color: #F8FAFC;
    padding: 1.25rem 1.5rem;
    border-radius: 14px;
    margin-bottom: 1.15rem;
    border: 1px solid #1E293B;
  }
  .hero-title {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 0.25rem 0;
  }
  .hero-sub {
    font-size: 0.9rem;
    color: #CBD5E1;
    margin: 0;
  }
  .stat-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    border: 1px solid #CBD5E1;
    background: #F1F5F9;
    color: #334155;
    margin-right: 6px;
  }
  .stat-pill-ok {
    background: #ECFDF5;
    color: #047857;
    border-color: #A7F3D0;
  }
  .stat-pill-err {
    background: #FEF2F2;
    color: #B91C1C;
    border-color: #FECACA;
  }
  .section-uid-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.35rem;
  }
  .diff-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    line-height: 1.65;
    font-size: 0.94rem;
    color: #1E293B;
  }
  .disclaimer-bar {
    background: #F1F5F9;
    border-left: 3px solid #1D4ED8;
    padding: 0.65rem 0.9rem;
    border-radius: 6px;
    font-size: 0.82rem;
    color: #475569;
    margin-top: 0.75rem;
  }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def call(method: str, path: str, **kw):
    try:
        r = httpx.request(method, f"{API}{path}", timeout=kw.pop("timeout", 300), **kw)
    except httpx.TimeoutException:
        return None, "Request timed out. Local LLM verification can take up to 3 minutes; please retry."
    except httpx.HTTPError:
        return None, f"Backend unreachable at `{API}`. Start it with `lawrag serve`."
    if r.status_code >= 400:
        try:
            err = r.json()["error"]
            msg = err["message"]
            if err["code"] == "BUDGET_EXCEEDED":
                msg = "Hosted-model budget exhausted. " + msg
            return None, msg
        except Exception:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
    return r.json(), None


def _normalize_uid(act: str, num: str) -> str:
    clean = num.strip().upper().replace(" ", "").replace("-", "")
    if clean.startswith(("IPC:", "BNS:")):
        return clean
    return f"{act}:{clean}"


def render_mapping_card(m: dict) -> None:
    top_l, top_r = st.columns([4, 1])
    top_l.markdown(f"**{m['mapping_id']}** · `{m['type']}`")
    top_r.markdown(f"<div style='text-align:right'>{status_badge(m['status'])}</div>", unsafe_allow_html=True)

    from_s = ", ".join(m["from_uids"]) or "—"
    to_s = ", ".join(m["to_uids"]) or "—"
    st.markdown(f"`{from_s}`  **→**  `{to_s}`")

    if m.get("caveat"):
        st.warning(m["caveat"])
    if m.get("change_summary"):
        st.caption(f"**Change summary:** {m['change_summary']}")
    if m.get("dispute_note") or m.get("review_note"):
        st.caption(f"**Reviewer note:** {m.get('dispute_note') or m.get('review_note')}")
    for e in m.get("evidence", []):
        pg = f", p.{e['page']}" if e.get("page") is not None else ""
        rw = f", row {e['row']}" if e.get("row") is not None else ""
        raw = f" — “{e['raw_row']}”" if e.get("raw_row") else ""
        st.caption(f"Evidence: {e.get('title') or e['source_id']}{pg}{rw}{raw}")
    st.caption(f"Reviewed by `{m.get('reviewed_by')}` on `{m.get('reviewed_on')}`")


def render_section_card(sec: dict) -> None:
    st.markdown(
        f"<div class='section-uid-header'>{html.escape(sec['uid'])} — {html.escape(sec['title'])}</div>",
        unsafe_allow_html=True,
    )
    src = sec.get("source") or {}
    meta_parts = [f"Status: `{sec.get('status', 'unknown')}`"]
    if src.get("title"):
        pg = f", p.{src['page']}" if src.get("page") else ""
        meta_parts.append(f"Source: {src['title']}{pg}")
    st.caption(" · ".join(meta_parts))

    if sec.get("status_note"):
        st.info(sec["status_note"])
    for p in sec.get("provisions", []):
        lbl = p.get("label") or p.get("kind") or ""
        if p.get("kind") in ("explanation", "exception", "proviso", "illustration"):
            with st.expander(f"{lbl} ({p['uid']})", expanded=False):
                st.write(p["text"])
        else:
            prefix = f"**{lbl}** " if lbl and lbl.lower() != "body" else ""
            st.markdown(f"{prefix}{p['text']}")
    if sec.get("amendment_notes"):
        with st.expander("Amendment notes", expanded=False):
            for note in sec["amendment_notes"]:
                st.caption(f"• {note}")


# -- Sidebar & Health Check ------------------------------------------------------------------------
health, herr = call("GET", "/health", timeout=10)
graph_down = bool(herr) or (health is not None and health.get("status") != "ok")

with st.sidebar:
    st.markdown("## NyayaGraph")
    st.caption(f"Verified IPC 1860 ↔ BNS 2023 Statutory RAG · [Web Studio]({API}/)")

    if herr:
        st.markdown("<span class='stat-pill stat-pill-err'>Backend Offline</span>", unsafe_allow_html=True)
        st.error(herr)
        if st.button("Reconnect", width="stretch"):
            st.rerun()
    elif health:
        ok = health.get("status") == "ok"
        pill_cls = "stat-pill stat-pill-ok" if ok else "stat-pill stat-pill-err"
        st.markdown(f"<span class='{pill_cls}'>Backend {html.escape(str(health.get('status', '')).upper())}</span>",
                    unsafe_allow_html=True)
        st.caption(
            f"**Graph:** `{health.get('graph', {}).get('backend')}`  \n"
            f"**Embedder:** `{health.get('embedder')}`  \n"
            f"**LLM:** `{health.get('llm')}`"
        )

    st.divider()
    page = st.radio(
        "Workspace",
        ["Ask Assistant", "Section Mapper", "Compare Provisions", "System & Scope"],
        label_visibility="collapsed",
    )

    history: list[dict] = st.session_state.setdefault("history", [])
    if history and page == "Ask Assistant":
        st.divider()
        st.caption("Recent queries")
        for idx, item in enumerate(reversed(history[-6:])):
            q_label = item["question"][:44] + ("…" if len(item["question"]) > 44 else "")
            if st.button(q_label, key=f"hist_{idx}", width="stretch"):
                st.session_state["last"] = item["response"]
                st.session_state["ask_q"] = item["question"]
                st.rerun()

    st.divider()
    if health and health.get("cutoff"):
        c = health["cutoff"]
        st.caption(f"**Cutoff:** IPC ≤ `{c.get('IPC')}` · BNS ≥ `{c.get('BNS')}`")
    st.caption("Not legal advice. *Verified* means supported by cited statutory text or reviewed mapping evidence.")


# -- Top Hero Banner -------------------------------------------------------------------------------
st.markdown(
    """
    <div class='hero-card'>
      <div class='hero-title'>IPC ↔ BNS Statutory Research &amp; Concordance</div>
      <div class='hero-sub'>Citation-grounded answers across the Indian Penal Code (1860) and Bharatiya Nyaya Sanhita (2023) with deterministic &amp; claim-level verification.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -- Page 1: Ask Assistant -------------------------------------------------------------------------
if page == "Ask Assistant":
    if "ask_q" not in st.session_state:
        st.session_state["ask_q"] = ""

    picked = st.pills("Quick prompts", EXAMPLES, selection_mode="single", default=None)
    if picked and picked != st.session_state.get("_last_pill"):
        st.session_state["_last_pill"] = picked
        st.session_state["ask_q"] = picked

    with st.container(border=True):
        q = st.text_area(
            "Ask a legal research question (offence, punishment, IPC ↔ BNS mapping, temporal applicability)",
            key="ask_q",
            height=96,
            max_chars=1000,
            placeholder="e.g. What is the punishment for theft under BNS, and how does it differ from IPC 379?",
        )
        ctrl_l, ctrl_m, ctrl_r = st.columns([2, 3, 1.4])
        debug = ctrl_l.checkbox("Include retrieval trace", value=False)
        ctrl_m.caption("Answers run citation, punishment, mapping, provenance & NLI checks.")
        submitted = ctrl_r.button(
            "Run Query",
            disabled=graph_down or not q.strip(),
            type="primary",
            width="stretch",
        )

    if submitted:
        with st.status("Running retrieval, graph expansion, generation, and verification…", expanded=True) as status_box:
            st.write("Querying backend at", API)
            resp, err = call("POST", "/ask", json={"question": q.strip(), "options": {"debug": debug}})
            if err:
                status_box.update(label="Query failed", state="error")
                st.error(err)
            else:
                status_box.update(label=f"Completed ({resp['outcome'].upper()})", state="complete", expanded=False)
                st.session_state["last"] = resp
                hist = st.session_state.setdefault("history", [])
                if not hist or hist[-1]["question"] != q.strip():
                    hist.append({"question": q.strip(), "response": resp})

    resp = st.session_state.get("last")
    if resp:
        v = resp.get("verification") or {}
        timings = resp.get("timings_ms") or {}
        total_ms = sum(val for val in timings.values() if isinstance(val, (int, float)))

        # Answer Card
        with st.container(border=True):
            hdr_l, hdr_r = st.columns([3, 2])
            hdr_l.markdown(outcome_badge(resp["outcome"]), unsafe_allow_html=True)
            hdr_r.markdown(
                f"<div style='text-align:right;color:#64748B;font-size:0.84rem'>"
                f"Decision rule <b>{html.escape(str(v.get('decision_rule')))}</b></div>",
                unsafe_allow_html=True,
            )
            st.markdown(link_citations(resp["answer_markdown"]))

            if resp.get("unverified_claims") and resp["outcome"] != "abstained":
                with st.expander(f"Unverified claims flagged ({len(resp['unverified_claims'])})", expanded=True):
                    for c in resp["unverified_claims"]:
                        st.write(f"- {c}")

        # KPI Strip
        m1, m2, m3, m4 = st.columns(4)
        sr = v.get("support_ratio")
        m1.metric("Decision", str(v.get("decision") or "—").title(), f"Rule {v.get('decision_rule')}")
        m2.metric("Claim Support", f"{sr:.0%}" if isinstance(sr, (int, float)) else "—",
                  f"{len(v.get('claims') or [])} claim(s)")
        m3.metric("Citations", len(resp.get("citations") or []),
                  f"{len(resp.get('mapping_facts') or [])} mapping(s)")
        m4.metric("Latency", f"{total_ms / 1000:.1f}s" if total_ms else "—",
                  f"Retries: {', '.join(resp.get('retry_actions') or []) or 'none'}")

        # Evidence & Verification Tabs
        tab_names = ["Citations", "Mapping Facts", "Verification Audit"]
        if resp.get("trace"):
            tab_names.append("Execution Trace")
        tabs = st.tabs(tab_names)

        with tabs[0]:
            cites = resp.get("citations") or []
            if not cites:
                st.info("No statutory citations were attached to this response.")
            for c in cites:
                st.markdown(f"<a id='{anchor(c['uid'])}'></a>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(f"**{c['uid']}** — {c['section_title']}")
                    st.write(c["excerpt"])
                    pg = f", p.{c['page']}" if c.get("page") else ""
                    st.caption(f"Source: {c.get('source_title')}{pg} · status `{c.get('status')}`")

        with tabs[1]:
            mfacts = resp.get("mapping_facts") or []
            if not mfacts:
                st.info("No concordance mapping cards apply to this query.")
            for m in mfacts:
                with st.container(border=True):
                    render_mapping_card(m)

        with tabs[2]:
            st.dataframe(check_rows(v), width="stretch", hide_index=True)
            if v.get("claims"):
                st.markdown("**Claim-level NLI Judgements**")
                st.dataframe(claim_rows(v), width="stretch", hide_index=True)
            st.caption(f"Timings (ms): {timings} · Estimated cost: ${resp.get('cost_estimate_usd', 0.0):.4f}")

        if resp.get("trace") and len(tabs) > 3:
            with tabs[3]:
                st.json(resp["trace"])

        st.markdown(f"<div class='disclaimer-bar'>{html.escape(resp['disclaimer'])}</div>", unsafe_allow_html=True)


# -- Page 2: Section Mapper ------------------------------------------------------------------------
elif page == "Section Mapper":
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.2, 2.8, 1.2])
        act = c1.segmented_control("Act", ["IPC", "BNS"], default="IPC") or "IPC"
        num = c2.text_input("Section number", "498A", placeholder="e.g. 302, 498A, 103, 303")
        c3.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        lookup = c3.button("Inspect Section", disabled=graph_down or not num.strip(), type="primary", width="stretch")

    if lookup:
        uid = _normalize_uid(act, num)
        sec_data, sec_err = call("GET", f"/sections/{uid}")
        map_data, map_err = call("GET", f"/mappings/{uid}")
        st.session_state["mapper_result"] = {
            "uid": uid,
            "sec": sec_data,
            "sec_err": sec_err,
            "map": map_data,
            "map_err": map_err,
        }

    res = st.session_state.get("mapper_result")
    if res:
        col_sec, col_map = st.columns([1.15, 1])
        with col_sec:
            st.subheader("Statutory Text")
            with st.container(border=True):
                if res["sec_err"]:
                    st.error(res["sec_err"])
                else:
                    render_section_card(res["sec"])
        with col_map:
            st.subheader("IPC ↔ BNS Transition Mappings")
            if res["map_err"]:
                st.error(res["map_err"])
            elif not res["map"].get("mappings"):
                st.info(
                    f"No mapping row is recorded for `{res['uid']}` (coverage: `{res['map'].get('coverage')}`). "
                    "An unrecorded mapping means unreviewed—not that no equivalent exists."
                )
            else:
                for m in res["map"]["mappings"]:
                    with st.container(border=True):
                        render_mapping_card(m)


# -- Page 3: Compare Provisions --------------------------------------------------------------------
elif page == "Compare Provisions":
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 1.2])
        left = c1.text_input("Left UID", "IPC:304A", placeholder="e.g. IPC:304A")
        right = c2.text_input("Right UID (optional — uses mapped target if blank)", "", placeholder="e.g. BNS:106")
        c3.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        run_cmp = c3.button("Compare", disabled=graph_down or not left.strip(), type="primary", width="stretch")

    if run_cmp:
        data, err = call("POST", "/compare", json={"left_uid": left.strip(), "right_uid": right.strip() or None})
        if err:
            st.error(err)
        else:
            st.session_state["compare_result"] = data

    data = st.session_state.get("compare_result")
    if data:
        if data.get("note"):
            st.info(data["note"])
        for p in data.get("pairs", []):
            with st.container(border=True):
                if p.get("mapping"):
                    m = p["mapping"]
                    st.markdown(
                        f"**Mapping {m['mapping_id']}** · `{m['type']}` {status_badge(m['status'])}",
                        unsafe_allow_html=True,
                    )
                    if m.get("change_summary"):
                        st.caption(m["change_summary"])
                a, b = st.columns(2)
                for col, side, sec in ((a, "left", p["left"]), (b, "right", p["right"])):
                    col.markdown(f"#### {sec['uid']} — {sec['title']}")
                    col.markdown(f"<div class='diff-box'>{diff_html(p['diff_segments'], side)}</div>", unsafe_allow_html=True)
                    qualifiers = [q for q in sec.get("provisions", []) if q["kind"] in ("explanation", "exception", "proviso")]
                    if qualifiers:
                        with col.expander(f"Qualifiers ({len(qualifiers)})"):
                            for q in qualifiers:
                                st.caption(f"**{q['label']}:** {q['text']}")


# -- Page 4: System & Scope ------------------------------------------------------------------------
else:
    c_about, c_sys = st.columns([1.4, 1])
    with c_about:
        with st.container(border=True):
            st.subheader("Methodology & Guardrails")
            st.markdown("""
This assistant answers substantive criminal-law queries across the **Indian Penal Code, 1860** and the **Bharatiya Nyaya Sanhita, 2023**, grounded in official statutory PDFs and concordance tables.

**Verification Pipeline (D1–D7 + Claim Judge):**
* Every claim must resolve to real graph UIDs and source passages (`D1`, `D2`).
* Numeric punishments, fines, and years are cross-checked against verbatim text (`D3`).
* Transition claims (`replaced`, `split`, `merged`, `omitted`, `new`) are validated against recorded concordance edges (`D4`, `D5`).
* Temporal applicability honors the `2024-07-01` commencement boundary (`D6`).
* Missing or nonexistent section premises fail gracefully (`D7`).

**Scope Boundaries:**
* Covers substantive **IPC** and **BNS** offences only.
* Procedure (**BNSS / CrPC**), Evidence (**BSA / IEA**), and judicial case outcomes are outside scope.
* *“No direct equivalent”* is asserted only when backed by an explicit `OMITTED` / `NEW` mapping.
            """)
    with c_sys:
        with st.container(border=True):
            st.subheader("Live Deployment Status")
            if health:
                g = health.get("graph") or {}
                st.metric("Backend Status", str(health.get("status", "")).upper(), f"Graph: {g.get('backend')}")
                st.write({
                    "API endpoint": API,
                    "Embedder": health.get("embedder"),
                    "BM25 index": health.get("bm25"),
                    "LLM status": health.get("llm"),
                    "Cutoff dates": health.get("cutoff"),
                    "Graph snapshot": str(health.get("graph_snapshot", ""))[:16],
                    "Corpus hash": str(health.get("corpus_hash", ""))[:16],
                    "Hosted budget left (USD)": health.get("budget_remaining_usd"),
                })
            else:
                st.warning("Connect the FastAPI backend (`lawrag serve`) to view live runtime telemetry.")

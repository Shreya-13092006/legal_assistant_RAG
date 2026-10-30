"""Streamlit UI (SDD 3.7.2): Ask, Map, Compare, About. A thin client over the FastAPI backend.

Run: streamlit run src/lawrag/ui/streamlit_app.py   (backend URL from LAWRAG_API, default :8000)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lawrag.ui.formatting import (anchor, check_rows, claim_rows, diff_html, link_citations,  # noqa: E402
                                  outcome_badge, status_badge)

API = os.environ.get("LAWRAG_API", "http://localhost:8000")
EXAMPLES = ["What replaced IPC 498A?", "What is the punishment for murder under the BNS?",
            "How does BNS 106 differ from IPC 304A?", "Is there a BNS equivalent of IPC 497?",
            "What is the BNS equivalent of section 377 IPC?", "Which IPC section did BNS 304 come from?",
            "What is the punishment under BNS section 999?", "What happened to sedition, IPC 124A?"]

st.set_page_config(page_title="IPC ↔ BNS Graph RAG", page_icon="⚖️", layout="wide")


def call(method: str, path: str, **kw):
    try:
        r = httpx.request(method, f"{API}{path}", timeout=kw.pop("timeout", 180), **kw)
    except httpx.TimeoutException:
        return None, "The request timed out. Try a simpler question or retry; local CPU models can be slow."
    except httpx.HTTPError:
        return None, f"Backend unreachable at {API}. Start it with `lawrag serve`."
    if r.status_code >= 400:
        try:
            err = r.json()["error"]
            msg = err["message"]
            if err["code"] == "BUDGET_EXCEEDED":
                msg = "The hosted-model budget is exhausted. " + msg
            return None, msg
        except Exception:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
    return r.json(), None


health, herr = call("GET", "/health", timeout=10)
with st.sidebar:
    st.title("⚖️ IPC ↔ BNS")
    if herr:
        st.error(herr)
        if st.button("Retry"):
            st.rerun()
    elif health:
        ok = health["status"] == "ok"
        (st.success if ok else st.warning)(f"Backend {health['status']} · graph {health['graph'].get('backend')}")
        st.caption(f"Embedder: {health['embedder']} · LLM: {health['llm']}")
    st.caption("Not legal advice. “Verified” = supported by the cited text in this system.")
    page = st.radio("Page", ["Ask", "Map", "Compare", "About"])
graph_down = bool(herr) or (health and health["status"] != "ok")


def render_mapping_card(m: dict) -> None:
    st.markdown(f"**{m['mapping_id']}** · `{m['type']}` {status_badge(m['status'])}", unsafe_allow_html=True)
    st.write(f"{', '.join(m['from_uids']) or '—'} → {', '.join(m['to_uids']) or '—'}")
    if m.get("caveat"):
        st.warning(m["caveat"])
    if m.get("change_summary"):
        st.caption(f"Change summary: {m['change_summary']}")
    if m.get("dispute_note") or m.get("review_note"):
        st.caption(f"Reviewer note: {m.get('dispute_note') or m.get('review_note')}")
    for e in m.get("evidence", []):
        st.caption(f"Evidence: {e.get('title') or e['source_id']}"
                   f"{', p.' + str(e['page']) if e.get('page') is not None else ''}"
                   f"{', row ' + str(e['row']) if e.get('row') is not None else ''} — “{e.get('raw_row') or ''}”")
    st.caption(f"Reviewed by {m.get('reviewed_by')} on {m.get('reviewed_on')}")


if page == "Ask":
    st.header("Ask")
    ex = st.selectbox("Example questions", ["—"] + EXAMPLES)
    q = st.text_area("Question", value="" if ex == "—" else ex, max_chars=1000)
    debug = st.checkbox("Show trace")
    if st.button("Ask", disabled=graph_down or not q.strip(), type="primary"):
        with st.spinner("Retrieving, generating and verifying…"):
            resp, err = call("POST", "/ask", json={"question": q, "options": {"debug": debug}})
        if err:
            st.error(err)
        else:
            st.session_state["last"] = resp
    resp = st.session_state.get("last")
    if resp:
        st.markdown(outcome_badge(resp["outcome"]), unsafe_allow_html=True)
        st.markdown(link_citations(resp["answer_markdown"]))
        if resp.get("unverified_claims") and resp["outcome"] != "abstained":
            with st.expander(f"Could not verify ({len(resp['unverified_claims'])})", expanded=True):
                for c in resp["unverified_claims"]:
                    st.write(f"- {c}")
        if resp.get("mapping_facts"):
            st.subheader("Mapping facts")
            for m in resp["mapping_facts"]:
                with st.container(border=True):
                    render_mapping_card(m)
        if resp.get("citations"):
            st.subheader("Citations")
            for c in resp["citations"]:
                st.markdown(f"<a id='{anchor(c['uid'])}'></a>", unsafe_allow_html=True)
                with st.expander(f"{c['uid']} — {c['section_title']}"):
                    st.write(c["excerpt"])
                    st.caption(f"Source: {c.get('source_title')}"
                               f"{', p.' + str(c['page']) if c.get('page') else ''} · status {c.get('status')}")
        st.subheader("Verification report")
        v = resp["verification"]
        st.write(f"Decision: **{v.get('decision')}** (rule {v.get('decision_rule')}) · support ratio "
                 f"{v.get('support_ratio')} · retries: {', '.join(resp.get('retry_actions') or []) or 'none'}")
        st.table(check_rows(v))
        if v.get("claims"):
            st.dataframe(claim_rows(v), use_container_width=True)
        st.caption(f"Timings (ms): {resp['timings_ms']} · cost ≈ ${resp['cost_estimate_usd']}")
        st.caption(resp["disclaimer"])
        if resp.get("trace"):
            with st.expander("Trace"):
                st.json(resp["trace"])

elif page == "Map":
    st.header("Map a section")
    c1, c2 = st.columns([1, 3])
    act = c1.selectbox("Act", ["IPC", "BNS"])
    num = c2.text_input("Section number", "498A")
    if st.button("Look up", disabled=graph_down):
        uid = f"{act}:{num.strip().upper().replace(' ', '').replace('-', '')}"
        data, err = call("GET", f"/mappings/{uid}")
        if err:
            st.error(err)
        elif not data["mappings"]:
            st.info(f"No reviewed mapping is recorded for {uid} (coverage: unreviewed). This does NOT mean there "
                    "is no equivalent.")
        else:
            for m in data["mappings"]:
                with st.container(border=True):
                    render_mapping_card(m)

elif page == "Compare":
    st.header("Compare provisions")
    c1, c2 = st.columns(2)
    left = c1.text_input("Left uid", "IPC:304A")
    right = c2.text_input("Right uid (optional: uses verified mapping targets)", "")
    if st.button("Compare", disabled=graph_down):
        data, err = call("POST", "/compare", json={"left_uid": left.strip(), "right_uid": right.strip() or None})
        if err:
            st.error(err)
        else:
            if data.get("note"):
                st.caption(data["note"])
            for p in data["pairs"]:
                if p.get("mapping"):
                    m = p["mapping"]
                    st.markdown(f"Mapping **{m['mapping_id']}** · `{m['type']}` {status_badge(m['status'])}",
                                unsafe_allow_html=True)
                a, b = st.columns(2)
                for col, side, sec in ((a, "left", p["left"]), (b, "right", p["right"])):
                    col.markdown(f"#### {sec['uid']} — {sec['title']}")
                    col.markdown(diff_html(p["diff_segments"], side), unsafe_allow_html=True)
                    for q in sec["provisions"]:
                        if q["kind"] in ("explanation", "exception", "proviso"):
                            col.caption(f"{q['label']}: {q['text']}")

else:
    st.header("About")
    st.markdown("""
This system answers questions about the **Indian Penal Code, 1860** and the **Bharatiya Nyaya Sanhita, 2023**
with citations to statutory text and reviewed IPC↔BNS mapping evidence. Every answer passes deterministic checks
(citation existence and grounding, punishment fidelity, mapping fidelity, provenance, temporal applicability,
false premises) and, when enabled, claim-level judging. When the evidence is insufficient it abstains.

* **Not legal advice.** “Verified” means *supported by the cited text*, not that the legal interpretation is correct.
* “No direct equivalent” is stated only when a verified OMITTED/NEW mapping with evidence exists; a missing
  mapping means *unknown*, never *no equivalent*.
* Scope: IPC and BNS substantive offences. Procedure (BNSS/CrPC), evidence (BSA) and case outcomes are out of scope.
""")
    if health:
        st.write({"cutoff dates": health["cutoff"], "graph snapshot": health["graph_snapshot"][:16],
                  "corpus hash": health["corpus_hash"][:16]})

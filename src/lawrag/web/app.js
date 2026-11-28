/* NyayaGraph Frontend Application */
(() => {
  const STORAGE_KEY = "nyayagraph.history.v1";

  const state = {
    view: "research",
    act: "IPC",
    lastResponse: null,
    history: loadHistory(),
    health: null,
    timerId: null,
    loadingStart: 0,
  };

  // ==================== UTILITIES ====================
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function esc(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function normalizeUid(act, raw) {
    const clean = String(raw || "").trim().toUpperCase().replace(/\s+/g, "").replace(/-/g, "");
    if (/^(IPC|BNS):/.test(clean)) return clean;
    return `${act}:${clean}`;
  }

  function sectionBase(uid) {
    const m = String(uid || "").match(/^((?:IPC|BNS):[0-9]+[A-Z]*)/i);
    return m ? m[1].toUpperCase() : uid;
  }

  function formatDuration(ms) {
    const totalSec = Math.max(0, Math.floor(ms / 1000));
    const m = String(Math.floor(totalSec / 60)).padStart(2, "0");
    const s = String(totalSec % 60).padStart(2, "0");
    return `${m}:${s}`;
  }

  function loadHistory() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  function saveHistory() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state.history.slice(0, 10)));
    } catch {
      // Ignore quota errors
    }
  }

  async function api(method, path, body = null, timeoutMs = 300000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const opts = {
        method,
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
      };
      if (body !== null) opts.body = JSON.stringify(body);
      const res = await fetch(path, opts);
      const text = await res.text();
      let json = null;
      try {
        json = text ? JSON.parse(text) : null;
      } catch {
        json = null;
      }
      if (!res.ok) {
        const msg = json?.error?.message || `HTTP ${res.status}: ${text.slice(0, 240)}`;
        throw new Error(msg);
      }
      return json;
    } catch (err) {
      if (err.name === "AbortError") {
        throw new Error("Request timed out waiting for the local LLM/backend.");
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  // ==================== NAVIGATION ====================
  function switchView(name) {
    state.view = name;
    $$(".nav-tab").forEach((btn) => {
      const active = btn.dataset.view === name;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-pressed", String(active));
    });
    $$(".view").forEach((sec) => {
      sec.classList.toggle("is-active", sec.id === `view-${name}`);
    });
  }

  $$(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  });

  // ==================== HEALTH TELEMETRY ====================
  async function refreshHealth() {
    const pill = $("#health-pill");
    const label = $("#health-label");
    try {
      const data = await api("GET", "/health", null, 10000);
      state.health = data;
      const ok = data.status === "ok";
      pill.classList.toggle("is-ok", ok);
      pill.classList.toggle("is-err", !ok);
      label.textContent = ok ? `Online · ${data.embedder}` : "Degraded";
      renderTelemetry(data);
    } catch (err) {
      state.health = null;
      pill.classList.remove("is-ok");
      pill.classList.add("is-err");
      label.textContent = "Backend Offline";
      $("#system-telemetry").innerHTML = `<div class="alert-banner alert-danger">${esc(err.message)}</div>`;
    }
  }

  function renderTelemetry(h) {
    const host = $("#system-telemetry");
    if (!host || !h) return;
    const cutoff = h.cutoff ? `IPC <= ${h.cutoff.IPC || "—"} | BNS >= ${h.cutoff.BNS || "—"}` : "—";
    const items = [
      ["Overall Status", String(h.status || "—").toUpperCase()],
      ["Graph Backend", h.graph?.backend || "—"],
      ["Embedder Model", h.embedder || "—"],
      ["BM25 Index", h.bm25 || "—"],
      ["LLM Provider", h.llm || "—"],
      ["Cutoff Policy", cutoff],
      ["Graph Snapshot", String(h.graph_snapshot || "").slice(0, 16) || "—"],
      ["Corpus SHA-256", String(h.corpus_hash || "").slice(0, 16) || "—"],
      ["Hosted Budget Left", `$${Number(h.budget_remaining_usd ?? 0).toFixed(2)}`],
    ];
    host.innerHTML = `
      <div class="telemetry-grid">
        ${items.map(([k, v]) => `<div class="tele-item"><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`).join("")}
      </div>
    `;
  }

  $("#health-pill").addEventListener("click", () => {
    switchView("system");
    refreshHealth();
  });
  $("#refresh-health-btn").addEventListener("click", refreshHealth);

  // ==================== SVG RENDERING HELPERS ====================
  function svgLineageDiagram(m) {
    const froms = (m.from_uids && m.from_uids.length ? m.from_uids : ["—"]).slice(0, 3);
    const tos = (m.to_uids && m.to_uids.length ? m.to_uids : ["—"]).slice(0, 3);
    const rows = Math.max(froms.length, tos.length, 1);
    const height = Math.max(76, rows * 34 + 18);
    const midY = height / 2;

    const statusColor =
      m.status === "verified" ? "#065F46" :
      m.status === "disputed" ? "#92400E" :
      m.status === "rejected" ? "#991B1B" : "#475569";

    const leftNodes = froms.map((u, i) => {
      const y = ((i + 1) * height) / (froms.length + 1);
      return `
        <path d="M 102 ${y} C 130 ${y}, 130 ${midY}, 148 ${midY}" fill="none" stroke="#94A3B8" stroke-width="1.5"/>
        <rect x="6" y="${y - 13}" width="96" height="26" rx="6" fill="#FAF9F6" stroke="#CBD5E1"/>
        <text x="54" y="${y + 4}" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="10.5" font-weight="600" fill="#0F172A">${esc(u)}</text>
      `;
    }).join("");

    const rightNodes = tos.map((u, i) => {
      const y = ((i + 1) * height) / (tos.length + 1);
      return `
        <path d="M 252 ${midY} C 270 ${midY}, 270 ${y}, 298 ${y}" fill="none" stroke="#1E3A8A" stroke-width="1.5"/>
        <rect x="298" y="${y - 13}" width="96" height="26" rx="6" fill="#EFF6FF" stroke="#BFDBFE"/>
        <text x="346" y="${y + 4}" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="10.5" font-weight="600" fill="#1E3A8A">${esc(u)}</text>
      `;
    }).join("");

    const hub = `
      <rect x="148" y="${midY - 17}" width="104" height="34" rx="17" fill="#FFFFFF" stroke="${statusColor}" stroke-width="1.75"/>
      <text x="200" y="${midY + 4}" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="10" font-weight="700" fill="${statusColor}">${esc(m.type)}</text>
    `;

    return `<svg class="svg-lineage" viewBox="0 0 400 ${height}" role="img" aria-label="Mapping diagram from ${esc(froms.join(', '))} to ${esc(tos.join(', '))}">${leftNodes}${rightNodes}${hub}</svg>`;
  }

  function sealBadge(outcome) {
    if (outcome === "answered") {
      return `
        <span class="seal seal-answered">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><path d="m9 12 2 2 4-4"/></svg>
          Verified Answer
        </span>`;
    }
    if (outcome === "partial") {
      return `
        <span class="seal seal-partial">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>
          Partial Answer
        </span>`;
    }
    return `
      <span class="seal seal-abstained">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>
        Abstained
      </span>`;
  }

  function statusTag(status) {
    const cls = status === "verified" ? "seal-answered" : status === "disputed" ? "seal-partial" : "mono-tag";
    return `<span class="seal ${cls}" style="padding:2px 9px;font-size:0.72rem">${esc(status)}</span>`;
  }

  // ==================== SAFE MARKDOWN + CITATION CHIPS ====================
  function renderAnswerMarkdown(md) {
    const lines = String(md || "").split(/\r?\n/);
    const out = [];
    let inList = false;

    const inlineFormat = (text) => {
      let s = esc(text);
      s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
      s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
      s = s.replace(/\[((?:IPC|BNS):[^\]\s|]+|MAP-\d+)\]/g, (_, uid) => {
        return `<button type="button" class="cite-chip" data-cite-uid="${esc(uid)}" title="Inspect ${esc(uid)}">${esc(uid)}</button>`;
      });
      return s;
    };

    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line) {
        if (inList) { out.push("</ul>"); inList = false; }
        continue;
      }
      if (line.startsWith("> ")) {
        if (inList) { out.push("</ul>"); inList = false; }
        out.push(`<blockquote>${inlineFormat(line.slice(2))}</blockquote>`);
      } else if (/^[-*]\s+/.test(line)) {
        if (!inList) { out.push("<ul>"); inList = true; }
        out.push(`<li>${inlineFormat(line.replace(/^[-*]\s+/, ""))}</li>`);
      } else if (/^#{1,4}\s+/.test(line)) {
        if (inList) { out.push("</ul>"); inList = false; }
        const clean = line.replace(/^#{1,4}\s+/, "");
        out.push(`<p><strong>${inlineFormat(clean)}</strong></p>`);
      } else {
        if (inList) { out.push("</ul>"); inList = false; }
        out.push(`<p>${inlineFormat(line)}</p>`);
      }
    }
    if (inList) out.push("</ul>");
    return out.join("");
  }

  // ==================== MAPPING CARD HTML ====================
  function mappingCardHtml(m) {
    const evHtml = (m.evidence || []).map((e) => {
      const pg = e.page != null ? `, p.${e.page}` : "";
      const rw = e.row != null ? `, row ${e.row}` : "";
      const raw = e.raw_row ? ` — “${esc(e.raw_row)}”` : "";
      return `<div class="mapping-meta">Evidence: ${esc(e.title || e.source_id)}${pg}${rw}${raw}</div>`;
    }).join("");

    return `
      <div class="mapping-card">
        <div class="mapping-top">
          <span class="mono-tag">${esc(m.mapping_id)}</span>
          ${statusTag(m.status)}
        </div>
        ${svgLineageDiagram(m)}
        ${m.caveat ? `<div class="alert-banner alert-warn" style="margin:6px 0;padding:8px 10px;font-size:0.8rem">${esc(m.caveat)}</div>` : ""}
        ${m.change_summary ? `<div class="mapping-summary">${esc(m.change_summary)}</div>` : ""}
        ${evHtml}
        <div class="mapping-meta" style="margin-top:4px">Reviewed by <code>${esc(m.reviewed_by)}</code> on <code>${esc(m.reviewed_on)}</code></div>
      </div>
    `;
  }

  // ==================== RESEARCH DESK ====================
  const askInput = $("#ask-input");
  const askForm = $("#ask-form");

  askInput.addEventListener("input", () => {
    $("#ask-char-count").textContent = `${askInput.value.length} / 1000`;
  });

  $$("#prompt-chips .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      askInput.value = chip.dataset.q;
      $("#ask-char-count").textContent = `${askInput.value.length} / 1000`;
      askInput.focus();
    });
  });

  $$(".drawer-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const id = tab.dataset.tab;
      $$(".drawer-tab").forEach((t) => {
        const active = t.dataset.tab === id;
        t.classList.toggle("is-active", active);
        t.setAttribute("aria-selected", String(active));
      });
      $$(".drawer-pane").forEach((p) => {
        p.classList.toggle("is-active", p.id === `drawer-${id}`);
      });
    });
  });

  askForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = askInput.value.trim();
    if (!q) return;

    const debug = $("#ask-debug").checked;
    $("#ask-error").classList.add("hidden");
    $("#ask-empty").classList.add("hidden");
    $("#ask-loading").classList.remove("hidden");
    $("#ask-submit").disabled = true;

    state.loadingStart = Date.now();
    $("#loading-timer").textContent = "00:00";
    clearInterval(state.timerId);
    state.timerId = setInterval(() => {
      $("#loading-timer").textContent = formatDuration(Date.now() - state.loadingStart);
    }, 500);

    try {
      const resp = await api("POST", "/ask", { question: q, options: { debug } });
      renderQAResponse(resp);
      state.history = [{ question: q, ts: new Date().toLocaleTimeString(), response: resp },
                       ...state.history.filter((h) => h.question !== q)].slice(0, 10);
      saveHistory();
      renderHistory();
    } catch (err) {
      $("#ask-error").textContent = err.message;
      $("#ask-error").classList.remove("hidden");
    } finally {
      clearInterval(state.timerId);
      $("#ask-loading").classList.add("hidden");
      $("#ask-submit").disabled = false;
    }
  });

  function renderQAResponse(resp) {
    state.lastResponse = resp;
    $("#ask-empty").classList.add("hidden");
    $("#answer-panel").classList.remove("hidden");

    const v = resp.verification || {};
    const timings = resp.timings_ms || {};
    const totalMs = Object.values(timings).reduce((acc, n) => acc + (typeof n === "number" ? n : 0), 0);

    $("#outcome-seal").outerHTML = `<span id="outcome-seal">${sealBadge(resp.outcome)}</span>`;
    $("#decision-rule-tag").textContent = `Decision Rule ${v.decision_rule || "—"}`;

    $("#kpi-decision").textContent = v.decision ? String(v.decision).toUpperCase() : "—";
    $("#kpi-support").textContent = typeof v.support_ratio === "number" ? `${Math.round(v.support_ratio * 100)}%` : "—";
    $("#kpi-citations").textContent = `${(resp.citations || []).length} sources`;
    $("#kpi-latency").textContent = totalMs ? `${(totalMs / 1000).toFixed(1)}s` : "—";

    const bodyEl = $("#brief-body");
    bodyEl.innerHTML = renderAnswerMarkdown(resp.answer_markdown);
    $$(".cite-chip", bodyEl).forEach((chip) => {
      chip.addEventListener("click", () => focusCitation(chip.dataset.citeUid));
    });

    const unv = $("#unverified-box");
    if (resp.unverified_claims?.length && resp.outcome !== "abstained") {
      unv.innerHTML = `<strong>Unverified claims removed/flagged (${resp.unverified_claims.length}):</strong><ul>` +
        resp.unverified_claims.map((c) => `<li>${esc(c)}</li>`).join("") + "</ul>";
      unv.classList.remove("hidden");
    } else {
      unv.classList.add("hidden");
    }

    // Deterministic checks pane
    const checks = v.deterministic?.checks || [];
    $("#drawer-checks").innerHTML = checks.length
      ? `<div class="check-grid">${checks.map((c) => `
          <div class="check-pill ${c.passed ? "check-pass" : "check-fail"}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">${
              c.passed ? '<path d="m9 12 2 2 4-4"/><circle cx="12" cy="12" r="9"/>'
                       : '<circle cx="12" cy="12" r="9"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>'
            }</svg>
            <div>
              <strong>${esc(c.check)} · ${c.passed ? "PASS" : `FAIL (${esc(c.severity)})`}</strong>
              ${c.details?.length ? `<div>${esc(c.details.join("; "))}</div>` : ""}
            </div>
          </div>`).join("")}</div>`
      : `<p class="muted-empty">No deterministic checks recorded.</p>`;

    // Claims pane
    const claims = v.claims || [];
    $("#drawer-claims").innerHTML = claims.length
      ? `<table class="claims-table"><thead><tr><th>Verdict</th><th>Role</th><th>Claim</th><th>Citations</th></tr></thead><tbody>` +
        claims.map((c) => `<tr>
          <td><code>${esc(c.verdict || "UNJUDGED")}</code></td>
          <td>${c.core ? "Core" : "Minor"}</td>
          <td>${esc(c.text)}${c.evidence_span ? `<div class="citation-meta">Span: “${esc(c.evidence_span)}”</div>` : ""}</td>
          <td><code>${esc((c.citations || []).join(", ") || "—")}</code></td>
        </tr>`).join("") + `</tbody></table>`
      : `<p class="muted-empty">No claim-level NLI judgements returned.</p>`;

    // Trace pane
    const traceBtn = $("#trace-tab-btn");
    if (resp.trace) {
      traceBtn.classList.remove("hidden");
      $("#drawer-trace").innerHTML = `<pre class="trace-pre">${esc(JSON.stringify(resp.trace, null, 2))}</pre>`;
    } else {
      traceBtn.classList.add("hidden");
    }

    $("#brief-disclaimer").textContent = resp.disclaimer || "Not legal advice.";

    // Right rail: Mappings
    const maps = resp.mapping_facts || [];
    $("#mapping-count-tag").textContent = `${maps.length} mapping${maps.length === 1 ? "" : "s"}`;
    $("#rail-lineage").innerHTML = maps.length
      ? maps.map(mappingCardHtml).join("")
      : `<p class="muted-empty">No cross-statute mapping hub triggered for this query.</p>`;

    // Right rail: Citations
    const cites = resp.citations || [];
    $("#cite-count-tag").textContent = `${cites.length} source${cites.length === 1 ? "" : "s"}`;
    $("#rail-citations").innerHTML = cites.length
      ? cites.map((c) => `
          <div class="citation-item" data-uid="${esc(c.uid)}">
            <div class="citation-head">
              <span class="mono-tag">${esc(c.uid)}</span>
              <button type="button" class="link-btn inspect-sec-btn" data-uid="${esc(sectionBase(c.uid))}">Open section &rarr;</button>
            </div>
            <strong>${esc(c.section_title)}</strong>
            <div class="citation-excerpt">${esc(c.excerpt)}</div>
            <div class="citation-meta">${esc(c.source_title || "")}${c.page ? `, p.${c.page}` : ""} · ${esc(c.status || "")}</div>
          </div>`).join("")
      : `<p class="muted-empty">No statutory excerpts cited.</p>`;

    $$(".inspect-sec-btn", $("#rail-citations")).forEach((btn) => {
      btn.addEventListener("click", () => openStatuteByUid(btn.dataset.uid));
    });
  }

  function focusCitation(uid) {
    const cards = $$("#rail-citations .citation-item");
    let target = cards.find((el) => el.dataset.uid === uid) ||
                 cards.find((el) => sectionBase(el.dataset.uid) === sectionBase(uid));
    if (target) {
      cards.forEach((el) => el.classList.remove("is-focused"));
      target.classList.add("is-focused");
      target.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } else if (/^(IPC|BNS):/.test(uid)) {
      openStatuteByUid(sectionBase(uid));
    }
  }

  $("#copy-brief-btn").addEventListener("click", async () => {
    if (!state.lastResponse?.answer_markdown) return;
    try {
      await navigator.clipboard.writeText(state.lastResponse.answer_markdown);
      $("#copy-brief-btn span").textContent = "Copied";
      setTimeout(() => { $("#copy-brief-btn span").textContent = "Copy Brief"; }, 1600);
    } catch {
      // Ignore clipboard failure
    }
  });

  function renderHistory() {
    const host = $("#rail-history");
    if (!state.history.length) {
      host.innerHTML = `<p class="muted-empty">Previous briefs in this browser session are cached locally for instant recall.</p>`;
      return;
    }
    host.innerHTML = state.history.map((item, idx) => `
      <button type="button" class="history-btn" data-idx="${idx}">
        <span class="history-q">${esc(item.question)}</span>
        <span class="history-meta">${esc(item.response.outcome.toUpperCase())} · ${esc(item.ts)}</span>
      </button>
    `).join("");

    $$(".history-btn", host).forEach((btn) => {
      btn.addEventListener("click", () => {
        const entry = state.history[Number(btn.dataset.idx)];
        if (entry) {
          askInput.value = entry.question;
          $("#ask-char-count").textContent = `${askInput.value.length} / 1000`;
          renderQAResponse(entry.response);
        }
      });
    });
  }

  $("#clear-history-btn").addEventListener("click", () => {
    state.history = [];
    saveHistory();
    renderHistory();
  });

  // ==================== STATUTES & CONCORDANCE ====================
  $$(".act-switch .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.act = btn.dataset.act;
      $$(".act-switch .seg-btn").forEach((b) => b.classList.toggle("is-active", b === btn));
    });
  });

  async function inspectStatute(uid) {
    $("#search-results-drawer").classList.add("hidden");
    const textPane = $("#statute-text-pane");
    const mapPane = $("#statute-map-pane");
    textPane.innerHTML = `<p class="muted-empty">Loading ${esc(uid)}…</p>`;
    mapPane.innerHTML = `<p class="muted-empty">Loading concordance mappings…</p>`;

    const [secRes, mapRes] = await Promise.allSettled([
      api("GET", `/sections/${encodeURIComponent(uid)}`),
      api("GET", `/mappings/${encodeURIComponent(uid)}`),
    ]);

    if (secRes.status === "fulfilled") {
      const s = secRes.value;
      const src = s.source || {};
      textPane.innerHTML = `
        <div class="rail-header">
          <h2>${esc(s.uid)} — ${esc(s.title)}</h2>
          <button type="button" class="btn-ghost" id="jump-compare-btn">Compare &rarr;</button>
        </div>
        <div class="citation-meta" style="margin-bottom:12px">
          Status: <code>${esc(s.status)}</code>${src.title ? ` · Source: ${esc(src.title)}${src.page ? `, p.${src.page}` : ""}` : ""}
        </div>
        ${s.status_note ? `<div class="alert-banner alert-warn">${esc(s.status_note)}</div>` : ""}
        ${(s.provisions || []).map((p) => {
          const isQual = ["explanation", "exception", "proviso", "illustration"].includes(p.kind);
          if (isQual) {
            return `<div class="qualifier-card"><strong>${esc(p.label || p.kind)} (${esc(p.uid)}):</strong> ${esc(p.text)}</div>`;
          }
          const prefix = p.label && p.label.toLowerCase() !== "body" ? `<strong>${esc(p.label)}</strong> ` : "";
          return `<div class="provision-block">${prefix}${esc(p.text)}</div>`;
        }).join("")}
      `;
      $("#jump-compare-btn")?.addEventListener("click", () => {
        $("#cmp-left").value = s.uid;
        $("#cmp-right").value = "";
        switchView("compare");
        runCompare();
      });
    } else {
      textPane.innerHTML = `<div class="alert-banner alert-danger">${esc(secRes.reason.message)}</div>`;
    }

    if (mapRes.status === "fulfilled") {
      const mv = mapRes.value;
      const list = mv.mappings || [];
      mapPane.innerHTML = `
        <div class="rail-header">
          <h2>Transition Concordance</h2>
          <span class="mono-tag">${esc(mv.coverage || "—")}</span>
        </div>
        ${list.length
          ? list.map(mappingCardHtml).join("")
          : `<div class="alert-banner alert-warn">No mapping row recorded for <code>${esc(uid)}</code>. Missing mapping means unreviewed — not that no equivalent exists.</div>`}
      `;
    } else {
      mapPane.innerHTML = `<div class="alert-banner alert-danger">${esc(mapRes.reason.message)}</div>`;
    }
  }

  function openStatuteByUid(uid) {
    const clean = uid.toUpperCase();
    const act = clean.startsWith("BNS:") ? "BNS" : "IPC";
    state.act = act;
    $$(".act-switch .seg-btn").forEach((b) => b.classList.toggle("is-active", b.dataset.act === act));
    $("#statute-num").value = clean.replace(/^(IPC|BNS):/, "");
    switchView("statutes");
    inspectStatute(clean);
  }

  $("#statute-lookup-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const raw = $("#statute-num").value.trim();
    if (!raw) return;
    if (/^[0-9]+[a-zA-Z]*$/.test(raw) || /^(IPC|BNS):/i.test(raw)) {
      inspectStatute(normalizeUid(state.act, raw));
    } else {
      runHybridSearch(raw);
    }
  });

  $("#statute-search-btn").addEventListener("click", () => {
    const raw = $("#statute-num").value.trim();
    if (raw) runHybridSearch(raw);
  });

  async function runHybridSearch(query) {
    const drawer = $("#search-results-drawer");
    drawer.classList.remove("hidden");
    drawer.innerHTML = `<p class="muted-empty">Searching corpus via hybrid retrieval for “${esc(query)}”…</p>`;
    try {
      const data = await api("POST", "/search", { query, mode: "hybrid", k: 8 });
      const hits = data.candidates || [];
      if (!hits.length) {
        drawer.innerHTML = `<p class="muted-empty">No candidate sections matched “${esc(query)}”.</p>`;
        return;
      }
      drawer.innerHTML = `
        <div class="rail-header">
          <h2>Hybrid Search Candidates</h2>
          <span class="mono-tag">${hits.length} hits</span>
        </div>
        <div class="search-hit-grid">
          ${hits.map((h) => `
            <button type="button" class="hit-card" data-uid="${esc(sectionBase(h.uid))}">
              <div class="mono-tag" style="margin-bottom:4px">${esc(h.uid)}</div>
              <div class="citation-meta">Score: ${Number(h.score).toFixed(4)} · Via: ${esc((h.via || []).join(", "))}</div>
            </button>
          `).join("")}
        </div>
      `;
      $$(".hit-card", drawer).forEach((btn) => {
        btn.addEventListener("click", () => inspectStatute(btn.dataset.uid));
      });
    } catch (err) {
      drawer.innerHTML = `<div class="alert-banner alert-danger">${esc(err.message)}</div>`;
    }
  }

  $$(".ref-link").forEach((btn) => {
    btn.addEventListener("click", () => openStatuteByUid(btn.dataset.uid));
  });

  // ==================== COMPARE REDLINE DIFF ====================
  function renderDiffSegments(segments, side) {
    return (segments || []).map((seg) => {
      const text = esc(seg[side] || "");
      if (!text) return "";
      if (seg.op === "equal") return text;
      const cls = side === "left" ? "diff-del" : "diff-add";
      return `<span class="${cls}">${text}</span>`;
    }).filter(Boolean).join(" ");
  }

  async function runCompare() {
    const left = $("#cmp-left").value.trim().toUpperCase();
    const right = $("#cmp-right").value.trim().toUpperCase();
    const out = $("#compare-output");
    if (!left) return;

    out.innerHTML = `<p class="muted-empty">Generating side-by-side redline diff…</p>`;
    try {
      const data = await api("POST", "/compare", { left_uid: left, right_uid: right || null });
      const note = data.note ? `<div class="alert-banner alert-warn">${esc(data.note)}</div>` : "";
      const pairs = (data.pairs || []).map((p) => `
        <div class="compare-pair">
          ${p.mapping ? mappingCardHtml(p.mapping) : ""}
          <div class="split-two" style="margin-top:12px">
            ${[["left", p.left], ["right", p.right]].map(([side, sec]) => `
              <div>
                <h2 style="font-family:var(--font-serif);margin:0 0 8px">${esc(sec.uid)} — ${esc(sec.title)}</h2>
                <div class="diff-prose">${renderDiffSegments(p.diff_segments, side)}</div>
                ${(sec.provisions || []).filter((q) => ["explanation", "exception", "proviso"].includes(q.kind)).map((q) => `
                  <div class="qualifier-card"><strong>${esc(q.label)}:</strong> ${esc(q.text)}</div>
                `).join("")}
              </div>
            `).join("")}
          </div>
        </div>
      `).join("");
      out.innerHTML = note + (pairs || `<p class="muted-empty">No comparison pairs returned.</p>`);
    } catch (err) {
      out.innerHTML = `<div class="alert-banner alert-danger">${esc(err.message)}</div>`;
    }
  }

  $("#compare-form").addEventListener("submit", (e) => { e.preventDefault(); runCompare(); });
  $("#cmp-swap").addEventListener("click", () => {
    const l = $("#cmp-left").value;
    const r = $("#cmp-right").value;
    if (r) { $("#cmp-left").value = r; $("#cmp-right").value = l; }
  });
  $$(".cmp-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      $("#cmp-left").value = btn.dataset.left;
      $("#cmp-right").value = btn.dataset.right;
      runCompare();
    });
  });

  // ==================== VERIFY SANDBOX ====================
  $("#verify-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const out = $("#verify-output");
    const uids = $("#verify-uids").value.split(",").map((s) => s.trim()).filter(Boolean);
    const answer = $("#verify-answer").value.trim();
    out.innerHTML = `<p class="muted-empty">Evaluating D1–D7 checks…</p>`;
    try {
      const data = await api("POST", "/verify", { answer, context_uids: uids });
      const checks = data.deterministic?.checks || [];
      out.innerHTML = `<div class="check-grid">${checks.map((c) => `
        <div class="check-pill ${c.passed ? "check-pass" : "check-fail"}">
          <div>
            <strong>${esc(c.check)} · ${c.passed ? "PASS" : `FAIL (${esc(c.severity)})`}</strong>
            ${c.details?.length ? `<div>${esc(c.details.join("; "))}</div>` : ""}
          </div>
        </div>`).join("")}</div>`;
    } catch (err) {
      out.innerHTML = `<div class="alert-banner alert-danger">${esc(err.message)}</div>`;
    }
  });

  // ==================== QUICK JUMP MODAL (CTRL+K) ====================
  const modal = $("#jump-modal");
  const jumpInput = $("#jump-input");

  function openModal() {
    if (typeof modal.showModal === "function") modal.showModal();
    jumpInput.focus();
  }
  $("#quick-jump-btn").addEventListener("click", openModal);
  $("#jump-close").addEventListener("click", () => modal.close());

  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      openModal();
    }
  });

  $("#jump-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const val = jumpInput.value.trim();
    if (!val) return;
    if (/^(IPC|BNS):[0-9]+[a-zA-Z0-9()]*$/i.test(val)) {
      modal.close();
      openStatuteByUid(sectionBase(val));
      return;
    }
    const resBox = $("#jump-results");
    resBox.innerHTML = `<p class="muted-empty">Searching…</p>`;
    try {
      const data = await api("POST", "/search", { query: val, mode: "hybrid", k: 6 });
      const hits = data.candidates || [];
      resBox.innerHTML = hits.length
        ? hits.map((h) => `
            <button type="button" class="hit-card" data-uid="${esc(sectionBase(h.uid))}">
              <div class="mono-tag">${esc(h.uid)}</div>
              <div class="citation-meta">Score ${Number(h.score).toFixed(4)} · Via: ${esc((h.via || []).join(", "))}</div>
            </button>`).join("")
        : `<p class="muted-empty">No matches. Try a canonical UID like <code>IPC:302</code> or <code>BNS:103</code>.</p>`;
      $$(".hit-card", resBox).forEach((b) => {
        b.addEventListener("click", () => {
          modal.close();
          openStatuteByUid(b.dataset.uid);
        });
      });
    } catch (err) {
      resBox.innerHTML = `<div class="alert-banner alert-danger">${esc(err.message)}</div>`;
    }
  });

  // ==================== BOOT ====================
  renderHistory();
  refreshHealth();
})();

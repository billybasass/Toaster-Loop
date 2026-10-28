// panels/traces.js — Traces browser: filterable/searchable list with live
// refresh, daily-stats strip (hand-rolled SVG), and the detail view
// (#/traces/<id>): transcript review, metadata sidebar, feedback widget.
// All DOM via el()/textContent; transcript messages reuse ui.messageNode.

import { api } from "../api.js";
import { bus } from "../bus.js";
import { go } from "../router.js";
import { el, toast, badge, scoreBadge, table, empty, fmtTime, fmtNum,
  copyButton, messageNode, svgDailyChart } from "../ui.js";

const PAGE = 50;

// --------------------------------------------------------------- stats strip

async function statsStrip() {
  const host = el("div", { class: "card stats-strip" });
  const toggle = el("button", { class: "small", text: "buffer stats ▾" });
  const body = el("div", { class: "hidden" });
  host.append(toggle, body);

  let open = false;
  toggle.addEventListener("click", () => {
    open = !open;
    body.classList.toggle("hidden", !open);
    toggle.textContent = open ? "buffer stats ▴" : "buffer stats ▾";
    if (open && !body.dataset.loaded) load();
  });

  async function load() {
    try {
      const d = await api("/api/traces/stats?days=14");
      const series = d.series || [];
      body.dataset.loaded = "1";
      if (!series.length) {
        body.appendChild(el("p", { class: "hint", text: "No traces yet." }));
        return;
      }
      const total = series.reduce((a, s) => a + (s.traces || 0), 0);
      const fails = series.reduce(
        (a, s) => a + Math.round((s.fail_rate || 0) * (s.traces || 0)), 0);
      body.replaceChildren(
        el("div", { class: "loghead" },
          el("span", { class: "hint mono",
            text: `${total} traces · ${Math.round(100 * fails / Math.max(1, total))}% failing (14 d)` })),
        svgDailyChart(series.map((s) => ({ label: s.date, n: s.traces,
          score: s.avg_score, fail: s.fail_rate }))));
    } catch (e) {
      body.appendChild(el("p", { class: "err", text: e.message }));
    }
  }
  return host;
}

// ------------------------------------------------------------- failures card
// Pass 4 #33: failure remediation lifecycle. Status concept adapted from
// PIASystem (MIT) app/core/failure_taxonomy.py (RemediationStatus cycle).
// Each reason row in the failures histogram gets a compact status control:
// click cycles open -> acknowledged -> in_progress -> resolved -> open.
// acknowledge/in-progress ask for a one-line note via native prompt() (the
// #36 intent-prompt pattern); cycling back to open posts an empty note,
// which deletes the row server-side (back to implicit-open). Implicit-open
// and explicit-open render identically on purpose — an untracked reason is
// open, and there is no honest visual difference to draw.

const REMEDIATION_CYCLE = ["open", "acknowledged", "in_progress", "resolved"];
const NOTE_PROMPT_STATUSES = new Set(["acknowledged", "in_progress"]);

function statusBadgeClass(status) {
  return { acknowledged: "partial", in_progress: "unknown",
    resolved: "validated" }[status] || "failure";
}

function failuresCard() {
  const host = el("div", { class: "card failures-card" });
  host.replaceChildren(el("p", { class: "hint", text: "failures: loading…" }));

  async function load() {
    let ins;
    try {
      ins = await api("/api/insights?days=30");
    } catch (e) {
      host.replaceChildren(el("p", { class: "err", text: e.message }));
      return;
    }
    const reasons = (ins.failure_reasons && ins.failure_reasons.by_reason) || [];
    if (!reasons.length) {
      host.replaceChildren(
        el("h3", { text: "Failures (30 d)" }),
        el("p", { class: "hint", text: "no coded failures" }));
      return;
    }
    const rows = reasons.slice(0, 10).map((r) => {
      const status = r.status || "open";
      const btn = el("button", {
        class: "small", title: "click to cycle remediation status",
      }, badge(status, statusBadgeClass(status)));
      btn.addEventListener("click", async () => {
        const next = REMEDIATION_CYCLE[
          (REMEDIATION_CYCLE.indexOf(status) + 1) % REMEDIATION_CYCLE.length];
        let note = r.note || "";
        if (NOTE_PROMPT_STATUSES.has(next)) {
          const got = prompt(`Note for "${r.reason}" (${next}):`, note);
          if (got == null) return; // cancelled — no change
          note = got.trim();
        } else if (next === "open") {
          note = ""; // reset-to-open with empty note untracks the reason
        }
        try {
          await api("/api/failures/note", { method: "POST",
            body: { reason: r.reason, status: next, ...(note ? { note } : {}) } });
          toast(`${r.reason} → ${next}`, "ok");
          load();
        } catch (e) { toast(`${e.code}: ${e.message}`, "error"); }
      });
      return el("div", { class: "hint fail-row" },
        btn, " ",
        el("span", { class: "mono" }, `${r.reason} `),
        `(${r.count})`,
        r.note ? el("span", { class: "hint", title: r.note },
          ` — ${r.note.length > 60 ? r.note.slice(0, 60) + "…" : r.note}`)
          : null);
    });
    host.replaceChildren(el("h3", { text: "Failures (30 d)" }), ...rows);
  }
  load();
  return host;
}

// ---------------------------------------------------------------- list view

function listView(mount) {
  const unsubs = [];
  const filters = { status: "", profile: "", model_alias: "", curated: "",
    since: "", q: "", project: "" };
  let sortKey = "timestamp"; // client-side sort of the fetched page
  let sortDir = -1;
  let offset = 0;
  let rows = [];
  let total = null; // server-side filtered count (5d) — enables page X of Y
  let loading = false;

  const statusSel = el("select", { title: "status" },
    el("option", { value: "", text: "any status" }),
    ...["success", "failure", "partial", "unknown"].map((s) =>
      el("option", { value: s, text: s })));
  const profileSel = el("select", { title: "agent profile" },
    el("option", { value: "", text: "any profile" }));
  const modelSel = el("select", { title: "model alias" },
    el("option", { value: "", text: "any model" }));
  // Projects: filter dropdown populated from /api/projects; the selection
  // travels to the backend as ?project=<id> (also honored by curation's
  // source-trace listing, which shares this endpoint).
  const projectSel = el("select", { title: "project" },
    el("option", { value: "", text: "any project" }));
  api("/api/projects").then((d) => {
    for (const p of d.projects || [])
      projectSel.appendChild(el("option", { value: p.id, text: p.name }));
  }).catch(() => { /* projects optional; leave the single option */ });
  const curatedSel = el("select", { title: "curated state" },
    el("option", { value: "", text: "curated + uncurated" }),
    el("option", { value: "1", text: "curated only" }),
    el("option", { value: "0", text: "uncurated only" }));
  const sinceIn = el("input", { type: "date", title: "since (inclusive)" });
  const qIn = el("input", { type: "search", placeholder: "search task summaries…",
    class: "grow" });

  const tableHost = el("div");
  const pager = el("div", { class: "pager" });
  const hint = el("span", { class: "hint" });

  function paramUrl() {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(filters)) if (v) p.set(k, v);
    p.set("limit", String(PAGE));
    p.set("offset", String(offset));
    return "/api/traces?" + p.toString();
  }

  async function load() {
    if (loading) return;
    loading = true;
    hint.textContent = "loading…";
    try {
      const d = await api(paramUrl());
      rows = d.traces || [];
      total = typeof d.total === "number" ? d.total : null;
      // Populate filter dropdowns from data the first time we see it.
      addOptions(profileSel, rows.map((r) => r.agent_profile));
      addOptions(modelSel, rows.map((r) => r.model_alias));
      render();
      hint.textContent = total == null
        ? `${rows.length} of offset ${offset}`
        : `${total} matching`;
    } catch (e) {
      tableHost.replaceChildren(empty("failed to load: " + e.message));
      hint.textContent = "";
    } finally { loading = false; }
  }

  const seenOpts = new Map();
  function addOptions(sel, values) {
    let set = seenOpts.get(sel);
    if (!set) { set = new Set(); seenOpts.set(sel, set); }
    for (const v of values) {
      if (v && !set.has(v)) {
        set.add(v);
        sel.appendChild(el("option", { value: v, text: v }));
      }
    }
  }

  function sorted() {
    const r = [...rows];
    r.sort((a, b) => {
      let x = a[sortKey], y = b[sortKey];
      if (sortKey === "tokens") {
        x = (a.tokens_in || 0) + (a.tokens_out || 0);
        y = (b.tokens_in || 0) + (b.tokens_out || 0);
      }
      if (x == null) return 1;
      if (y == null) return -1;
      return (x < y ? -1 : x > y ? 1 : 0) * sortDir;
    });
    return r;
  }

  function sortable(label, key) {
    const mark = sortKey === key ? (sortDir > 0 ? " ▲" : " ▼") : "";
    return { label: label + mark, render: null, key,
      cls: "sortable-th",
      // header click handled after table build via th order
    };
  }

  function render() {
    const cols = [
      { ...sortable("time", "timestamp"),
        render: (r) => el("span", { class: "mono" }, fmtTime(r.timestamp, false)) },
      { label: "status", render: (r) => badge(r.status) },
      { label: "score", render: (r) => scoreBadge(r.outcome_score) },
      // WP12: plan-task chip next to the summary for runs started from a task.
      { label: "task", render: (r) => el("span", {}, r.task_summary || "–",
          r.plan_task ? el("span", { class: "hint mono",
            title: "started from plan task " + r.plan_task,
            text: " [" + r.plan_task + "]" }) : null) },
      { label: "profile", key: "agent_profile" },
      { label: "model", key: "model_alias" },
      { label: "tools", key: "num_tool_calls" },
      { label: "tok", render: (r) => r.tokens_in != null
          ? `${r.tokens_in}→${r.tokens_out ?? "?"}` : "–" },
      { label: "dur", render: (r) => r.duration_s != null
          ? fmtNum(r.duration_s, 1) + "s" : "–" },
      { label: "", render: (r) => r.curated
          ? el("a", { href: "#/datasets/" + encodeURIComponent(r.dataset_id || ""),
              title: "in dataset " + (r.dataset_id || "?") }, badge("curated", "validated"))
          : null },
    ];
    const t = table(cols, sorted(), {
      empty: "No traces match — adjust filters or run an agent task in Chat.",
      onRowClick: (r) => go("/traces/" + encodeURIComponent(r.trace_id)),
    });
    // Wire header sorting (columns align 1:1 with cols above).
    const ths = t.querySelectorAll("th");
    const keys = ["timestamp", null, "outcome_score", null, "agent_profile",
      "model_alias", "num_tool_calls", null, "duration_s", null];
    ths.forEach((th, i) => {
      if (!keys[i]) return;
      th.classList.add("sortable");
      th.tabIndex = 0;
      const act = () => {
        if (sortKey === keys[i]) sortDir = -sortDir;
        else { sortKey = keys[i]; sortDir = -1; }
        render();
      };
      th.addEventListener("click", act);
      th.addEventListener("keydown", (e) => { if (e.key === "Enter") act(); });
    });
    tableHost.replaceChildren(t);

    // 5d: with the server-side total we can show real page numbers and let
    // next jump past short pages; fall back to the old offset heuristic.
    const totalPages = total == null ? null : Math.max(1, Math.ceil(total / PAGE));
    const page = Math.floor(offset / PAGE) + 1;
    pager.replaceChildren(
      el("button", { class: "small", text: "← prev", disabled: offset <= 0,
        onclick: () => { offset = Math.max(0, offset - PAGE); load(); } }),
      el("span", { class: "hint mono",
        text: totalPages == null ? `offset ${offset}`
          : `page ${page} of ${totalPages} (${total})` }),
      el("button", { class: "small", text: "next →",
        disabled: totalPages != null ? page >= totalPages : rows.length < PAGE,
        onclick: () => { offset += PAGE; load(); } }));
  }

  function applyFilters() {
    filters.status = statusSel.value;
    filters.profile = profileSel.value;
    filters.model_alias = modelSel.value;
    filters.curated = curatedSel.value;
    filters.project = projectSel.value;
    filters.since = sinceIn.value ? sinceIn.value + "T00:00:00" : "";
    offset = 0;
    load();
  }

  let deb = null;
  qIn.addEventListener("input", () => {
    clearTimeout(deb);
    deb = setTimeout(() => { filters.q = qIn.value.trim(); applyFilters(); },
      300);
  });
  for (const s of [statusSel, profileSel, modelSel, curatedSel, projectSel])
    s.addEventListener("change", applyFilters);
  sinceIn.addEventListener("change", applyFilters);

  mount.replaceChildren(
    el("h2", { text: "Traces" }),
    el("div", { class: "toolbar" },
      statusSel, profileSel, modelSel, projectSel, curatedSel, sinceIn, qIn,
      hint),
    tableHost,
    el("div", { class: "btnrow" }, pager));

  // Failures remediation card (pass 4 #33) sits right under the stats strip.
  statsStrip().then((s) => {
    mount.insertBefore(s, tableHost);
    mount.insertBefore(failuresCard(), s.nextSibling);
  });
  load();

  // Live refresh: prepend new traces while sitting on page 1 (deep pages keep
  // their window — simple heuristic per spec).
  unsubs.push(bus.on("trace.recorded", async () => {
    if (offset !== 0 || loading) return;
    try {
      const d = await api(paramUrl());
      rows = d.traces || [];
      total = typeof d.total === "number" ? d.total : total;
      render();
    } catch { /* next event or manual refresh will retry */ }
  }));

  return { destroy() { clearTimeout(deb); for (const u of unsubs) u(); } };
}

// --------------------------------------------------------------- detail view

function feedbackBlock(traceId, existing, onSaved) {
  const host = el("div", { class: "feedback" });
  const histHost = el("div");

  function renderHistory(list) {
    histHost.replaceChildren(
      el("h4", { text: "Feedback history" }),
      ...(list.length ? list.map((f) => el("div", { class: "fb-row" },
        badge(f.source),
        f.score != null ? scoreBadge(f.score) : null,
        el("span", { class: "hint", text: fmtTime(f.timestamp) }),
        f.comment ? el("div", { class: "fb-note", text: f.comment }) : null))
        : [el("p", { class: "hint", text: "No feedback yet." })]));
  }
  renderHistory(existing || []);

  const slider = el("input", { type: "range", min: "-1", max: "1", step: "0.1",
    value: "0" });
  const sliderVal = el("span", { class: "mono hint", text: "0.0" });
  slider.addEventListener("input", () => {
    sliderVal.textContent = Number(slider.value).toFixed(1);
  });
  const note = el("textarea", { rows: 2,
    placeholder: "What went right/wrong? (note and/or score)" });

  async function post(body) {
    try {
      await api(`/api/traces/${encodeURIComponent(traceId)}/feedback`,
        { method: "POST", body });
      toast("feedback saved", "ok");
      const d = await api("/api/traces/" + encodeURIComponent(traceId));
      renderHistory(d.feedback || []);
      if (onSaved) onSaved(d);
    } catch (e) { toast(`${e.code}: ${e.message}`, "error"); }
  }

  const upBtn = el("button", { class: "small", text: "👍 good",
    title: "score +1" });
  const downBtn = el("button", { class: "small", text: "👎 bad",
    title: "score -1" });
  upBtn.addEventListener("click", () => post({ score: 1, note: note.value.trim() || undefined }));
  downBtn.addEventListener("click", () => post({ score: -1, note: note.value.trim() || undefined }));
  const sendBtn = el("button", { class: "primary small", text: "save" });
  sendBtn.addEventListener("click", () => {
    const body = { score: Number(slider.value) };
    if (note.value.trim()) body.note = note.value.trim();
    post(body);
  });

  host.append(
    histHost,
    el("h4", { text: "Leave feedback" }),
    el("div", { class: "btnrow" }, upBtn, downBtn),
    el("div", { class: "field" }, slider, sliderVal),
    note,
    el("div", { class: "btnrow" }, sendBtn));
  return host;
}

function transcriptHost(messages, opts = {}) {
  const host = el("div", { class: "msgs transcript" });
  for (const m of messages) host.appendChild(messageNode(m, opts));
  if (!messages.length)
    host.appendChild(empty("Transcript empty (file pruned?) or no match."));
  return host;
}

// ---------------------------------------------------------------- swimlane
// Timeline concept adapted from PIASystem (MIT) app/core/rollout_visualizer.py
// One horizontal strip above the transcript: segment width ∝ duration_ms,
// color by kind via existing design tokens. Click jumps to that message.

const TL_KINDS = { llm: 1, tool: 1, retry: 1, scrub: 1, other: 1 };

function fmtDur(ms) {
  if (!Number.isFinite(ms) || ms <= 0) return "0 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} s`;
  return `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`;
}

// Returns {node, totalMs}. Degenerate inputs (no segments / all-zero
// durations) render the 'no timing data' hint instead of an empty strip.
function timelineStrip(timeline, msgHost) {
  const segs = Array.isArray(timeline) ? timeline.filter(
    (s) => s && Number.isFinite(s.duration_ms) && s.duration_ms >= 0) : [];
  const total = segs.reduce((a, s) => a + (s.duration_ms || 0), 0);
  if (!segs.length || total <= 0) {
    return el("div", { class: "timeline-strip" },
      el("span", { class: "hint", text: "no timing data for this trace" }));
  }
  const strip = el("div", { class: "timeline-strip", role: "img",
    "aria-label": "trace timeline" });
  for (const s of segs) {
    const kind = TL_KINDS[s.kind] ? s.kind : "other";
    // Flex-grow by duration; a hairline min keeps zero-ms segments visible.
    const seg = el("div", {
      class: `tl-seg tl-${kind}`,
      style: `flex-grow:${(s.duration_ms || 0) + 1}`,
      title: `${kind} · ${fmtDur(s.duration_ms)} · ${s.label || ""}`,
    });
    seg.addEventListener("click", () => {
      const target = msgHost.children[s.message_index | 0];
      if (!target) return;
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      target.classList.remove("tl-flash");
      // restart the flash animation on repeat clicks of the same message
      void target.offsetWidth;
      target.classList.add("tl-flash");
    });
    strip.appendChild(seg);
  }
  const legend = el("span", { class: "hint mono" },
    segs.length + " segments · " + fmtDur(total));
  return el("div", { class: "timeline-wrap" }, strip, legend);
}

function metaSidebar(trace) {
  const rows = [
    ["status", null, badge(trace.status)],
    ["score", null, scoreBadge(trace.outcome_score)],
    ["model", trace.model_alias],
    ["profile", trace.agent_profile],
    ["endpoint", trace.endpoint_label || trace.model_version],
    ["tokens", trace.tokens_in != null
      ? `${trace.tokens_in}→${trace.tokens_out ?? "?"}` : null],
    ["duration", trace.duration_s != null ? fmtNum(trace.duration_s, 1) + "s" : null],
    ["tool calls", String(trace.num_tool_calls ?? 0)],
    ["session", trace.session_id],
    ["project", trace.project],
    // WP12: plan task this run was started from (task<->trace linkage).
    ["plan task", trace.plan_task],
  ];
  const kv = el("table", { class: "kv" }, el("tbody", {},
    ...rows.map(([k, v, node]) => el("tr", {},
      el("td", { class: "k", text: k }),
      el("td", {}, node || el("span", { text: v || "–" }))))));
  const curated = trace.curated && trace.dataset_id
    ? el("p", {}, badge("curated", "validated"), " ",
        el("a", { href: "#/datasets/" + encodeURIComponent(trace.dataset_id),
          class: "mono", text: trace.dataset_id }))
    : el("p", { class: "hint", text: "not curated" });
  return el("aside", { class: "trace-meta" },
    el("h4", { text: "Metadata" }), kv, curated,
    el("h4", { text: "Trace id" }),
    el("div", { class: "pathrow" },
      el("code", { class: "mono path", text: trace.trace_id }),
      copyButton(trace.trace_id)));
}

function detailView(mount, traceId) {
  const unsubs = [];
  const mainHost = el("div", { class: "trace-main" });
  mainHost.replaceChildren(el("p", { class: "hint", text: "loading…" }));

  let qOn = false;
  let lastData = null;

  async function load() {
    let d;
    try {
      const url = "/api/traces/" + encodeURIComponent(traceId) +
        (qOn && qText ? "?q=" + encodeURIComponent(qText) : "");
      d = await api(url);
    } catch (e) {
      mainHost.replaceChildren(el("p", { class: "err", text: e.message }),
        el("p", {}, el("a", { href: "#/traces", text: "← back to traces" })));
      return;
    }
    lastData = d;
    render(d);
  }

  let qText = "";
  const qIn = el("input", { type: "search", class: "grow",
    placeholder: "filter transcript by text…" });
  const qToggle = el("label", { class: "chk" }, el("input", { type: "checkbox" }),
    " filter");

  function render(d) {
    const t = d.trace;
    document.title = `trace ${t.trace_id.slice(0, 8)} — sable`;
    const backLink = el("a", { href: "#/traces", class: "hint", text: "← traces" });
    const head = el("div", { class: "loghead" },
      backLink,
      el("h3", { text: t.task_summary || t.trace_id }),
      badge(t.status), scoreBadge(t.outcome_score));
    const filterRow = el("div", { class: "toolbar" }, qIn, qToggle);
    const transcript = transcriptHost(d.messages || [],
      { highlight: qOn ? qText : "" });
    const body = el("div", { class: "with-drawer trace-split" },
      transcript,
      el("div", { class: "drawer-side" },
        metaSidebar(t),
        el("div", { class: "card" }, feedbackBlock(traceId, d.feedback, () => {}))));
    mainHost.replaceChildren(head,
      timelineStrip(d.timeline, transcript), filterRow, body);
  }

  let deb = null;
  function onQChange() {
    qOn = qToggle.querySelector("input").checked && !!qIn.value.trim();
    qText = qIn.value.trim();
    clearTimeout(deb);
    deb = setTimeout(load, 300);
  }
  qIn.addEventListener("input", onQChange);
  qToggle.querySelector("input").addEventListener("change", onQChange);

  const prevTitle = document.title;
  mount.replaceChildren(mainHost);
  load();
  unsubs.push(bus.on("trace.feedbacked", (p) => {
    if (p.trace_id === traceId) load();
  }));

  return { destroy() {
    document.title = prevTitle;
    clearTimeout(deb);
    for (const u of unsubs) u();
  } };
}

// -------------------------------------------------------------------- panel

export default {
  title: "Traces",
  render(mount, params) {
    if (params && params.length && params[0])
      return detailView(mount, decodeURIComponent(params[0]));
    return listView(mount);
  },
};

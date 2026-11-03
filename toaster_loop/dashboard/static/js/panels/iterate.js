// panels/iterate.js — full-loop pipeline runs (Phase 5d).
// Trigger form -> POST /api/iterate (202 {run_id}); active-run timeline is
// built live from iterate.step bus events and reconciled against the
// persisted run record (GET /api/iterate/runs) on load, so reopening the
// panel mid-run shows everything that already happened. Runs history links
// to each run's dataset, resulting model and train/export jobs.

import { api } from "../api.js";
import { bus } from "../bus.js";
import { el, toast, badge, table, empty, fmtTime } from "../ui.js";

const STAGES = ["curate", "register", "train", "export", "validate", "promote"];

/** Timeline element for one run record (+ live step overlay). */
function timelineEl(run) {
  const rows = el("div", { class: "timeline" });
  const byName = new Map();
  for (const s of run.steps || []) {
    // keep last state per step name (running -> done/failed transitions)
    byName.set(s.name, s);
  }
  const order = [];
  for (const s of run.steps || []) if (!order.includes(s.name)) order.push(s.name);

  for (const name of order) {
    const s = byName.get(name);
    rows.appendChild(el("div", { class: "step " + (s.state || "") },
      badge(s.state === "running" ? "running" : s.state),
      el("span", { class: "mono step-name", text: name }),
      el("span", { class: "hint step-detail", text: s.detail || "" })));
  }
  // upcoming stages not yet reported (only while the run is live)
  if (run.state === "running") {
    for (const name of STAGES)
      if (!byName.has(name))
        rows.appendChild(el("div", { class: "step pending" },
          badge("pending", "queued"), el("span", { class: "mono step-name", text: name })));
  }
  return rows;
}

function resultLinks(run) {
  const r = run.result || {};
  const bits = [];
  if (run.dataset || r.dataset_id) {
    const id = r.dataset_id || run.dataset;
    bits.push(el("a", { href: "#/datasets/" + encodeURIComponent(id),
      class: "mono", text: `dataset ${id}` }));
  }
  if (r.model_id)
    bits.push(el("a", { href: "#/models", class: "mono", text: `model ${r.model_id}` }));
  for (const [label, id] of [["train", r.train_job_id], ["export", r.export_job_id]])
    if (id) bits.push(el("a", { href: "#/jobs/" + encodeURIComponent(id),
      class: "mono", text: `${label} job` }));
  return bits.length ? el("div", { class: "field" }, ...bits) : null;
}

function runCard(run) {
  const links = resultLinks(run);
  return el("div", { class: "card run-card" },
    el("h3", {},
      el("span", { class: "mono", text: run.run_id }), " ",
      badge(run.state),
      el("span", { class: "hint mono" },
        ` ${run.name || ""} · ${run.kind}${run.dry_run ? " · dry-run" : ""}`)),
    el("p", { class: "hint" },
      `started ${fmtTime(run.started)}${run.finished ? " · finished " + fmtTime(run.finished) : ""}`),
    timelineEl(run),
    run.result && run.result.error
      ? el("p", { class: "err", text: String(run.result.error) }) : null,
    links);
}

export default {
  title: "Iterate",
  render(mount) {
    const activeHost = el("div");
    const historyHost = el("div");
    let runs = []; // persisted records, newest first
    let activeId = null; // run we're following live
    let activeRun = null; // merged record for the active run

    mount.replaceChildren(
      el("h2", { text: "Iterate" }),
      el("p", { class: "hint", text:
        "One command, whole loop: curate → register → train → export → " +
        "validate → promote. Watch the step timeline update live." }),
      triggerForm(), activeHost,
      el("h3", { text: "Runs" }), historyHost);

    // -------------------------------------------------------------- form
    function triggerForm() {
      const nameIn = el("input", { type: "text", placeholder: "run name (optional)" });
      const kindSel = el("select", {},
        el("option", { value: "sft", text: "sft" }),
        el("option", { value: "dpo", text: "dpo" }),
        el("option", { value: "grpo", text: "grpo" }));
      const dsSel = el("select", {}, el("option", { value: "", text: "auto-curate" }));
      const dryRun = el("input", { type: "checkbox" });
      const btn = el("button", { class: "primary", text: "Run pipeline" });

      api("/api/datasets").then((d) => {
        for (const ds of [...(d.datasets || [])].reverse())
          dsSel.appendChild(el("option", { value: ds.dataset_id, text: ds.dataset_id }));
      }).catch(() => {});

      btn.addEventListener("click", async () => {
        btn.disabled = true;
        try {
          const r = await api("/api/iterate", { method: "POST", body: {
            name: nameIn.value.trim() || undefined,
            kind: kindSel.value,
            dataset: dsSel.value || undefined,
            dry_run: dryRun.checked } });
          toast(`iterate run ${r.run_id} started`, "ok");
          startLive(r.run_id);
        } catch (e) { toast(`${e.code}: ${e.message}`, "error"); }
        finally { btn.disabled = false; }
      });

      return el("div", { class: "card" },
        el("h3", { text: "New run" }),
        el("div", { class: "field" },
          el("label", { text: "name" }), nameIn,
          el("label", { text: "kind" }), kindSel),
        el("div", { class: "field" },
          el("label", { text: "dataset" }), dsSel,
          el("label", { class: "chk" }, dryRun, " dry run")),
        el("div", { class: "field" }, btn));
    }

    // ------------------------------------------------------------- data
    function renderActive() {
      if (!activeRun) { activeHost.replaceChildren(); return; }
      activeHost.replaceChildren(el("h3", { text: "Active run" }), runCard(activeRun));
    }

    function renderHistory() {
      const rows = runs.filter((r) => r.run_id !== activeId);
      historyHost.replaceChildren(table([
        { label: "run", render: (r) => el("span", { class: "mono" }, r.run_id) },
        { label: "state", render: (r) => badge(r.state) },
        { label: "kind", key: "kind" },
        { label: "name", key: "name" },
        { label: "dataset", render: (r) => r.result?.dataset_id || r.dataset || "–" },
        { label: "model", render: (r) => el("span", { class: "mono" }, r.result?.model_id || "–") },
        { label: "started", render: (r) => fmtTime(r.started, false) },
      ], rows, { empty: "No iterate runs yet." }));
    }

    async function loadRuns() {
      const d = await api("/api/iterate/runs");
      runs = d.runs || [];
      // Resume a live timeline if the newest run is still executing.
      if (!activeId && runs.length && runs[0].state === "running")
        { activeId = runs[0].run_id; activeRun = runs[0]; }
      if (activeId) {
        const fresh = runs.find((r) => r.run_id === activeId);
        if (fresh) activeRun = mergeRun(activeRun, fresh);
      }
      renderActive();
      renderHistory();
    }

    // ------------------------------------------------------ live timeline
    function startLive(runId) {
      activeId = runId;
      activeRun = { run_id: runId, state: "running", started: new Date().toISOString(),
        name: "", kind: "", steps: [] };
      renderActive();
      loadRuns().catch(() => {});
    }

    // Prefer richer persisted data (steps with details) but keep live-only info.
    function mergeRun(live, persisted) {
      if (!live) return persisted;
      const steps = [...(persisted.steps || [])];
      for (const s of live.steps || [])
        if (!steps.some((p) => p.name === s.name && p.state === s.state))
          steps.push(s);
      return { ...persisted, ...live, steps };
    }

    const unsubs = [
      bus.on("iterate.step", (p) => {
        if (!activeId || p.run_id !== activeId) {
          // a run started elsewhere: pick it up
          if (p.step !== "finished") startLive(p.run_id);
          else loadRuns().catch(() => {});
          return;
        }
        if (p.step === "finished") {
          activeRun = { ...activeRun, state: p.state };
          renderActive();
          toast(`iterate ${p.run_id} → ${p.state}`,
            p.state === "done" ? "ok" : "error");
          loadRuns().catch(() => {});
          return;
        }
        const steps = [...(activeRun.steps || [])];
        const i = steps.findIndex((s) => s.name === p.step);
        if (i >= 0) steps[i] = { name: p.step, state: p.state, detail: p.detail };
        else steps.push({ name: p.step, state: p.state, detail: p.detail });
        activeRun = { ...activeRun, steps };
        renderActive();
      }),
      bus.on("overflow", () => loadRuns().catch(() => {})),
    ];

    loadRuns().catch((e) => historyHost.replaceChildren(empty(e.message)));

    return { destroy() { for (const u of unsubs) u(); } };
  },
};

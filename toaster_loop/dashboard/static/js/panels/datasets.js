// panels/datasets.js — curated datasets: list, manifest detail with SFT/DPO
// paged row previews (ShareGPT mini-transcripts, DPO chosen/rejected blocks),
// export-to-dir and the curate-trigger form. Source trace ids link back to
// #/traces/<id>. All DOM via el()/textContent — no innerHTML.

import { api } from "../api.js";
import { bus } from "../bus.js";
import { go } from "../router.js";
import { el, toast, badge, table, empty, fmtTime, copyButton, confirmDialog,
  messageNode } from "../ui.js";

const ROWS_PAGE = 10;

/**
 * POST /api/curate (5d: queued job, 202 {job_id}) and resolve when the job
 * finishes. Resolves {ok, dataset_id?, sft_count?, dpo_count?, claimed?,
 * warnings?, error?}. Rejects only on submit failure (ApiError).
 * WP3d: same helper drives /api/datasets/curate_source (folder/website jobs
 * share the 202+poll contract and the final {"event":"done", result} log line).
 * onProgress receives the newest {"event":"progress", message} log line while
 * the job runs, so long crawls/folder scans are never silent. There is
 * deliberately NO client-side timeout: big builds legitimately take a long
 * time, and the job's own error/cancel states are the only failure signals.
 */
async function curateJob(body, endpoint = "/api/curate", onProgress = null) {
  const r = await api(endpoint, { method: "POST", body });
  const jobId = r.job_id;
  if (!jobId) return Promise.reject(new Error("no job id in response"));
  toast(`curate queued (${jobId})`, "info");
  return new Promise((resolve) => {
    let timer = null;
    let settled = false;
    let lastProgressLine = 0;
    const finish = (res) => {
      if (settled) return;
      settled = true;
      unsub();
      if (timer) clearInterval(timer);
      resolve(res);
    };
    const check = async () => {
      let d;
      try {
        d = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
      } catch { return false; } // transient — retry on next event/timer
      const job = d.job;
      // Surface new progress lines to the caller (folder/crawl builds stream
      // {"event":"progress", message} into the job log).
      if (onProgress && Array.isArray(d.log.lines)) {
        const lines = d.log.lines;
        if (lines.length > lastProgressLine) {
          for (const line of lines.slice(lastProgressLine)) {
            try {
              const rec = JSON.parse(line);
              if (rec.event === "progress" && rec.message) onProgress(rec.message);
            } catch { /* non-JSON log lines are fine */ }
          }
          lastProgressLine = lines.length;
        }
      }
      if (job.state === "error" || job.state === "cancelled") {
        finish({ ok: false, error: job.error || job.state });
        return true;
      }
      if (job.state === "done") {
        // result rides the final log line written by the curate runner.
        for (const line of [...(d.log.lines || [])].reverse()) {
          try {
            const rec = JSON.parse(line);
            if (rec.event === "done" && rec.result) {
              finish({ ok: true, ...rec.result });
              return true;
            }
          } catch { /* non-JSON log lines are fine */ }
        }
        finish({ ok: true });
        return true;
      }
      return false;
    };
    const unsub = bus.on("job.state", (p) => {
      if (p.job_id === jobId && p.state !== "queued" && p.state !== "running")
        check();
    });
    check().then((done) => {
      if (!done && !settled)
        timer = setInterval(() => { check(); }, 3000); // fallback if SSE missed
    });
  });
}

// ShareGPT turn role -> OpenAI-ish role for the shared message renderer.
const SG_ROLE = { human: "user", gpt: "assistant", tool: "tool",
  system: "system" };

function traceLink(tid) {
  if (!tid) return el("span", { class: "hint", text: "–" });
  return el("a", { href: "#/traces/" + encodeURIComponent(tid), class: "mono",
    text: tid.length > 12 ? tid.slice(0, 10) + "…" : tid,
    title: tid });
}

// Pass 4 #42: trust badge + flagged-row summary from the gate sidecars
// (enriched server-side into every manifest as `trust` + `lint`). Old
// datasets without a flags.json sidecar report lint === null — shown
// honestly as "unlinted" with a hint to run `toaster_loop datasets lint`.
const TRUST_CLASS = { curated: "success", judge_scored: "partial",
  imported: "unknown" };

function trustBadge(d) {
  const t = d.trust || "imported";
  return badge(t, TRUST_CLASS[t] || "unknown");
}

function lintCell(d) {
  if (d.lint == null) {
    return el("span", { class: "hint", title:
      "No ingestion-gate sidecar yet — run `toaster_loop datasets lint " +
      (d.dataset_id || "") + "` to check this dataset.", text: "unlinted" });
  }
  const n = d.lint.flagged_rows || 0;
  if (!n) return el("span", { class: "hint", text: "clean" });
  const reasons = Object.entries(d.lint.reasons || {})
    .map(([k, v]) => `${k}: ${v}`).join("\n");
  return el("span", { title: reasons },
    badge(`⚑ ${n}`, "partial"));
}

// ------------------------------------------------------------- row previews

function sftRowNode(row) {
  const host = el("div", { class: "row-card" });
  const meta = el("div", { class: "loghead" },
    el("span", { class: "hint", text: "sft · sharegpt" }),
    row.trace_id ? el("span", { class: "hint" }, "trace ", traceLink(row.trace_id)) : null,
    row.model ? el("span", { class: "hint mono", text: row.model }) : null);
  host.appendChild(meta);
  const conv = Array.isArray(row.conversations) ? row.conversations : [];
  if (!conv.length) {
    host.appendChild(el("pre", { class: "log json",
      text: JSON.stringify(row, null, 2) }));
    return host;
  }
  for (const turn of conv) {
    const role = SG_ROLE[turn.from] || turn.from || "unknown";
    host.appendChild(messageNode({ role, content: turn.value ?? "" }));
  }
  return host;
}

function dpoRowNode(row) {
  const host = el("div", { class: "row-card" });
  const md = row.metadata || {};
  host.appendChild(el("div", { class: "loghead" },
    el("span", { class: "hint", text: "dpo · preference pair" }),
    el("span", { class: "hint" }, "chosen ", traceLink(md.chosen_trace_id)),
    el("span", { class: "hint" }, "rejected ", traceLink(md.rejected_trace_id))));
  const prompt = Array.isArray(row.prompt) ? row.prompt : [];
  if (prompt.length) {
    host.appendChild(el("h4", { text: "Prompt" }));
    for (const m of prompt) host.appendChild(messageNode(m));
  }
  const pair = el("div", { class: "dpo-pair" },
    el("div", { class: "dpo-chosen" },
      el("h4", {}, badge("chosen", "success")),
      el("pre", { class: "log", text: String(row.chosen ?? "") })),
    el("div", { class: "dpo-rejected" },
      el("h4", {}, badge("rejected", "failure")),
      el("pre", { class: "log", text: String(row.rejected ?? "") })));
  host.appendChild(pair);
  return host;
}

// -------------------------------------------------------------- detail view

async function detailView(mount, datasetId) {
  const unsubs = [];
  const host = el("div");
  host.replaceChildren(el("p", { class: "hint", text: "loading…" }));
  mount.replaceChildren(host);

  let manifest = null;

  async function loadManifest() {
    try {
      manifest = await api("/api/datasets/" + encodeURIComponent(datasetId));
    } catch (e) {
      host.replaceChildren(el("p", { class: "err", text: e.message }),
        el("p", {}, el("a", { href: "#/datasets", text: "← back to datasets" })));
      return false;
    }
    return true;
  }

  function manifestBlock() {
    const f = manifest.filters || {};
    const files = manifest.files || {};
    const warnings = manifest.warnings || [];
    const sftIds = files["sft.jsonl"]?.trace_ids || [];
    const dpoIds = files["dpo.jsonl"]?.trace_ids || [];

    const filterRows = Object.entries(f)
      .filter(([, v]) => v != null && v !== false && v !== "both" || v === true)
      .map(([k, v]) => el("tr", {},
        el("td", { class: "k", text: k }),
        el("td", { text: String(v) })));

    const block = el("div", { class: "card" },
      el("h3", { text: "Manifest" }),
      el("table", { class: "kv" }, el("tbody", {},
        el("tr", {}, el("td", { class: "k", text: "created" }),
          el("td", { text: fmtTime(manifest.created_at) })),
        el("tr", {}, el("td", { class: "k", text: "trust" }),
          el("td", {}, trustBadge(manifest))),
        el("tr", {}, el("td", { class: "k", text: "ingestion gate" }),
          el("td", {}, lintCell(manifest))),
        el("tr", {}, el("td", { class: "k", text: "sft rows" }),
          el("td", { text: String(files["sft.jsonl"]?.count ?? 0) })),
        el("tr", {}, el("td", { class: "k", text: "dpo rows" }),
          el("td", { text: String(files["dpo.jsonl"]?.count ?? 0) })),
        ...filterRows)),
      warnings.length ? el("div", { class: "inline-error" },
        el("h3", { text: "Warnings" }),
        ...warnings.map((w) => el("p", { text: w }))) : null,
      el("h4", { text: "Source traces" }),
      el("div", { class: "trace-links" },
        ...(sftIds.length + dpoIds.length
          ? [...new Set([...sftIds, ...dpoIds])].map((t) => traceLink(t))
          : [el("span", { class: "hint", text: "none recorded" })])));
    return block;
  }

  function exportBlock() {
    const pathIn = el("input", { type: "text", class: "grow",
      placeholder: "/path/to/export/parent-dir",
      title: "Parent directory; the dataset is copied into <dir>/" + datasetId +
        ". Destinations outside TOASTER_LOOP_HOME ask for confirmation first." });
    const out = el("div", { class: "result" });
    const btn = el("button", { text: "Export dataset",
      title: "Copy this dataset's files out of the toaster_loop state directory, " +
        "e.g. to share it or train elsewhere. Fails if the target already exists." });
    btn.addEventListener("click", async () => {
      const to = pathIn.value.trim();
      if (!to) return toast("enter a destination directory", "error");
      out.textContent = "exporting…";
      const send = (force) => api("/api/datasets/export", { method: "POST",
        body: force ? { dataset_id: datasetId, to, force: true }
                    : { dataset_id: datasetId, to } });
      try {
        let r;
        try {
          r = await send(false);
        } catch (e) {
          // 5d containment guard: confirm before exporting outside home.
          if (e.code === "outside_home") {
            const ok = await confirmDialog({
              title: "Export outside TOASTER_LOOP_HOME?",
              body: e.message + "\n\nThis writes outside the toaster_loop state " +
                "directory. Continue?",
              confirmLabel: "Export anyway", danger: true });
            if (!ok) { out.textContent = ""; return; }
            r = await send(true);
          } else throw e;
        }
        out.textContent = "exported → " + r.path;
        toast("dataset exported", "ok");
      } catch (e) {
        out.textContent = "";
        if (e.code === "target_exists")
          toast(`target already exists — ${e.message}`, "error");
        else toast(`${e.code}: ${e.message}`, "error");
      }
    });
    return el("div", { class: "card" },
      el("h3", { text: "Export" }),
      el("p", { class: "hint", text:
        "Copies the dataset directory to <dir>/" + datasetId +
        ". Fails if the target exists; destinations outside TOASTER_LOOP_HOME ask " +
        "for confirmation first." }),
      el("div", { class: "field" }, pathIn, btn), out);
  }

  function tabsBlock() {
    const host = el("div", { class: "card" });
    const bodyHost = el("div");
    const state = { file: "sft", offset: 0, total: 0 };
    const tabSft = el("button", { class: "seg",
      title: "Supervised fine-tuning rows: prompt → ideal completion pairs." },
      "sft.jsonl");
    const tabDpo = el("button", { class: "seg",
      title: "Preference rows: prompt with a chosen and a rejected answer, used " +
        "for DPO training." }, "dpo.jsonl");
    const info = el("span", { class: "hint mono" });

    async function loadRows() {
      bodyHost.replaceChildren(el("p", { class: "hint", text: "loading…" }));
      try {
        const d = await api(`/api/datasets/${encodeURIComponent(datasetId)}` +
          `/rows?file=${state.file}&offset=${state.offset}` +
          `&limit=${ROWS_PAGE}`);
        state.total = d.total;
        bodyHost.replaceChildren(
          ...(d.rows || []).map((r) => state.file === "sft"
            ? sftRowNode(r) : dpoRowNode(r)),
          d.truncated ? el("p", { class: "hint", text:
            "⚠ long strings truncated to 800 chars in this preview" }) : null,
          !d.rows.length && state.total === 0
            ? empty(`No ${state.file} rows in this dataset.`) : null);
        info.textContent = `${state.file}: rows ${state.offset}–` +
          `${Math.min(state.offset + ROWS_PAGE, d.total)} of ${d.total}`;
      } catch (e) {
        bodyHost.replaceChildren(el("p", { class: "err", text: e.message }));
      }
    }

    function setFile(f) {
      state.file = f;
      state.offset = 0;
      tabSft.classList.toggle("active", f === "sft");
      tabDpo.classList.toggle("active", f === "dpo");
      loadRows();
    }
    tabSft.addEventListener("click", () => setFile("sft"));
    tabDpo.addEventListener("click", () => setFile("dpo"));

    const pager = el("div", { class: "pager" },
      el("button", { class: "small", text: "← prev", onclick: () => {
        state.offset = Math.max(0, state.offset - ROWS_PAGE); loadRows(); } }),
      info,
      el("button", { class: "small", text: "next →", onclick: () => {
        if (state.offset + ROWS_PAGE < state.total) {
          state.offset += ROWS_PAGE; loadRows();
        } } }));
    host.append(el("div", { class: "toolbar" }, tabSft, tabDpo), bodyHost, pager);
    setFile("sft");
    return host;
  }

  if (!await loadManifest()) return { destroy() {} };

  // Source-derived datasets (hub/folder/website/import) carry no trace claims;
  // a trace-pipeline rebuild would erase their rows, so the server refuses —
  // hide the button and point at the source action instead.
  const prov = manifest.provenance || "curated";
  const isSourceBacked = prov !== "curated";
  const rebuildBtn = el("button", { text: "Rebuild (re-curate)",
    title: "Release this dataset's trace claims and regenerate it in place from " +
      "the current buffer with its original filters. Other datasets are untouched." });
  if (isSourceBacked) {
    rebuildBtn.disabled = true;
    rebuildBtn.title = `This dataset was curated from a ${prov} source — ` +
      "re-create it from that source instead of rebuilding from traces.";
  }
  rebuildBtn.addEventListener("click", async () => {
    const ok = await confirmDialog({
      title: `Rebuild ${datasetId}?`,
      body: "Releases this dataset's trace claims and regenerates it in place " +
        "from the current buffer. Other datasets keep their traces.",
      confirmLabel: "Rebuild", danger: true });
    if (!ok) return;
    rebuildBtn.disabled = true;
    try {
      // 5d: curate runs as a queued job; resolve when it finishes.
      const r = await curateJob({ rebuild: datasetId });
      if (r.ok) {
        toast(`rebuilt ${datasetId}: ${r.sft_count} sft, ${r.dpo_count} dpo`,
          "ok");
        go("/datasets/" + encodeURIComponent(datasetId));
      } else {
        toast(`rebuild failed: ${r.error}`, "error");
      }
    } catch (e) { toast(`${e.code}: ${e.message}`, "error"); }
    finally { rebuildBtn.disabled = false; }
  });

  // Delete for good: the dataset directory is removed and its trace claims
  // released (buffer traces become curatable again). Confirm first — there
  // is no undo; export first if the data matters.
  const deleteBtn = el("button", { text: "Delete dataset", class: "danger",
    title: "Permanently remove this dataset and release its traces back to the " +
      "buffer so they can be curated again. No undo — export first if it matters." });
  deleteBtn.addEventListener("click", async () => {
    const ok = await confirmDialog({
      title: `Delete ${datasetId}?`,
      body: "Permanently removes this dataset (sft.jsonl, dpo.jsonl, manifest, " +
        "sidecars) and releases its trace claims back to the buffer. This " +
        "cannot be undone — export it first if you may need it.",
      confirmLabel: "Delete forever", danger: true });
    if (!ok) return;
    deleteBtn.disabled = true;
    try {
      await api("/api/datasets/" + encodeURIComponent(datasetId),
        { method: "DELETE" });
      toast(`deleted ${datasetId}`, "ok");
      go("/datasets");
    } catch (e) {
      toast(`delete failed: ${e.message}`, "error");
      deleteBtn.disabled = false;
    }
  });

  host.replaceChildren(
    el("div", { class: "loghead" },
      el("a", { href: "#/datasets", class: "hint", text: "← datasets" }),
      el("h3", { text: datasetId }),
      copyButton(datasetId, "copy id")),
    el("div", { class: "btnrow" }, rebuildBtn, deleteBtn),
    manifestBlock(),
    tabsBlock(),
    exportBlock());

  return { destroy() { for (const u of unsubs) u(); } };
}

// ----------------------------------------------------------------- curate UI

async function curateForm(onDone) {
  const host = el("div", { class: "card" });
  let defaults = {};
  try {
    const c = await api("/api/config");
    defaults = (c.config && c.config.curation) || {};
  } catch { /* defaults stay blank */ }

  const nameIn = el("input", { type: "text", placeholder: "ui", class: "narrow",
    title: "Optional label appended to the new dataset's id, so you can recognize " +
      "it later in the table and when picking it for training." });
  const minScore = el("input", { type: "number", min: "-1", max: "1", step: "0.05",
    value: defaults.min_score != null ? String(defaults.min_score) : "0.5",
    class: "narrow",
    title: "Only claim traces whose judge/feedback score is at least this. " +
      "Scale −1…1; 0.5 keeps clearly-good traces. Lower = more data, more noise." });
  const sinceIn = el("input", { type: "date",
    title: "Ignore traces recorded before this date. Empty = consider all history." });
  const profileIn = el("input", { type: "text", placeholder: "any", class: "narrow",
    title: "Only claim traces produced by this agent profile (see Profiles page). " +
      "Empty = every profile." });
  const modelIn = el("input", { type: "text", placeholder: "any", class: "narrow",
    title: "Only claim traces served by this model alias. Empty = any model." });
  const limitIn = el("input", { type: "number", min: "1", placeholder: "no limit",
    class: "narrow",
    title: "Claim at most this many traces (newest first). Empty = no cap." });
  const formatsSel = el("select", {
    title: "Which training formats to emit. sft = single-turn pairs; dpo = " +
      "chosen/rejected preference pairs from feedback; both = each trace goes " +
      "into whichever file it qualifies for." },
    ...["both", "sft", "dpo"].map((f) => el("option", { value: f, text: f })));
  const failChk = el("input", { type: "checkbox",
    title: "Also turn failing traces (thumbs-down / errors) into SFT rows. " +
      "Off by default — usually you want DPO pairs for failures instead." });
  const out = el("div", { class: "result" });
  const btn = el("button", { class: "primary", text: "Curate dataset",
    title: "Queue a curation job with the filters above. Runs in the background; " +
      "progress streams here and in the Jobs panel." });

  btn.addEventListener("click", async () => {
    const body = {};
    if (nameIn.value.trim()) body.name = nameIn.value.trim();
    if (minScore.value !== "") body.min_score = Number(minScore.value);
    if (sinceIn.value) body.since = sinceIn.value + "T00:00:00";
    if (profileIn.value.trim()) body.profile = profileIn.value.trim();
    if (modelIn.value.trim()) body.model_alias = modelIn.value.trim();
    if (limitIn.value !== "") body.limit = Number(limitIn.value);
    if (formatsSel.value) body.formats = formatsSel.value;
    if (failChk.checked) body.include_failures = true;
    btn.disabled = true;
    out.textContent = "curating… (queued as a job — watch the Jobs panel)";
    try {
      // 5d: /api/curate queues a kind="curate" job; curateJob follows it to
      // completion via bus + polling and resolves with the result payload.
      const r = await curateJob(body);
      if (r.ok) {
        out.textContent = `created ${r.dataset_id} (${r.sft_count} sft, ` +
          `${r.dpo_count} dpo, ${r.claimed} claimed)`;
        toast(`dataset ${r.dataset_id} created`, "ok");
        if (r.warnings?.length) r.warnings.forEach((w) => toast(w, "info"));
        onDone(r);
      } else {
        out.textContent = "";
        toast(`curate failed: ${r.error}`, "error");
      }
    } catch (e) {
      out.textContent = "";
      toast(`${e.code}: ${e.message}`, "error");
    } finally { btn.disabled = false; }
  });

  host.append(
    el("h3", { text: "Curate" }),
    el("p", { class: "hint", text:
      "Selects currently-unclaimed traces and claims them into a new dataset." }),
    el("div", { class: "field" },
      el("label", {}, "name"), nameIn,
      el("label", {}, "min score"), minScore,
      el("label", {}, "since"), sinceIn),
    el("div", { class: "field" },
      el("label", {}, "profile"), profileIn,
      el("label", {}, "model alias"), modelIn,
      el("label", {}, "limit"), limitIn,
      el("label", {}, "formats"), formatsSel),
    el("div", { class: "field" },
      el("label", { class: "chk" }, failChk, " include failures as SFT"),
      btn),
    out);
  return host;
}

// ------------------------------------------------- running-curates banner
// The curateJob poller lives in the panel's closure, so navigating to Chat and
// back used to silently lose the progress display (the job itself keeps
// running server-side). This card re-attaches from the jobs API — the durable
// source of truth — so any queued/running curate or curate_source job shows its
// newest progress line again immediately, even after a full page reload. When
// the last one finishes it triggers a dataset-list refresh.
function runningCurationCard(onFinished) {
  const host = el("div", { class: "card hidden" });
  let stopped = false;
  let hadLive = false;
  async function tick() {
    if (stopped) return;
    let jobs = [];
    try { jobs = (await api("/api/jobs?tail=30")).jobs || []; }
    catch { return; } // transient — next tick retries
    const live = jobs.filter((j) =>
      (j.kind === "curate" || j.kind === "curate_source") &&
      (j.state === "queued" || j.state === "running"));
    if (!live.length) {
      host.classList.add("hidden");
      host.replaceChildren();
      if (hadLive) { hadLive = false; onFinished(); }
      return;
    }
    hadLive = true;
    const rows = live.map((j) => {
      let msg = j.state;
      for (const line of [...(j.log_tail || [])].reverse()) {
        try {
          const rec = JSON.parse(line);
          if (rec.event === "progress" && rec.message) { msg = rec.message; break; }
        } catch { /* non-JSON log lines are fine */ }
      }
      return el("div", { class: "field" },
        badge(j.state, j.state === "running" ? "done" : "queued"),
        el("span", { class: "mono hint", text: j.job_id + " " }),
        el("span", { class: "hint", text: msg }));
    });
    host.replaceChildren(
      el("h3", { text: "Running curation jobs" }),
      ...rows,
      el("p", { class: "hint", text:
        "Progress keeps streaming here even after you leave and return to this " +
        "page. The Jobs panel has the full log." }));
  }
  const timer = setInterval(tick, 2500);
  tick();
  return { node: host, destroy() { stopped = true; clearInterval(timer); } };
}

// ------------------------------------------------- WP3d: source curation card
// Local folder / website ingestion + hub-snapshot import. Folder and website
// run as queued kind="curate_source" jobs (202+poll, same contract as
// /api/curate); hub import is synchronous local work. All three materialize a
// normal dataset record — the table below needs no special cases beyond the
// provenance chip.

function sourcesCard(onDone) {
  const host = el("div", { class: "card" });
  const out = el("div", { class: "result" });
  const setProgress = (msg) => { out.textContent = msg; };

  // Synthesis server picker: "default" (the legacy serve endpoint) plus every
  // currently-running named instance. A dead/stale selection falls back to
  // default server-side, so the list only needs to be a helpful hint.
  const serverSel = el("select", { class: "narrow", title:
    "Server used to synthesize doc pairs (offline = deterministic heading fallback)" },
    el("option", { value: "", text: "default / auto" }));
  api("/api/servers").then((d) => {
    for (const s of (d.servers || [])) {
      if (!s.running || s.role === "embedding") continue;
      serverSel.appendChild(el("option", {
        value: s.id, text: s.name + (s.default ? " (default)" : "") }));
    }
  }).catch(() => { /* picker stays default-only; auto still works */ });

  // -- local folder
  const dirIn = el("input", { type: "text", class: "grow",
    placeholder: "/path/to/docs (recursive: .md .txt .rst .html .json .jsonl)",
    title: "Absolute path to a local docs folder. Every .md/.txt/.rst/.html/" +
      ".json/.jsonl file below it is read recursively and turned into ShareGPT pairs." });
  const dirName = el("input", { type: "text", placeholder: "name",
    class: "narrow",
    title: "Optional label for the resulting dataset id. Empty = derived from the folder name." });
  const dirBtn = el("button", { text: "Ingest folder",
    title: "Queue a job that reads the folder above and creates a dataset from it. " +
      "Large folders take a while; progress streams below." });
  dirBtn.addEventListener("click", async () => {
    if (!dirIn.value.trim()) return toast("folder path required", "error");
    dirBtn.disabled = true;
    setProgress("ingesting folder… (queued job; large folders take a while)");
    try {
      const body = { source: "local_folder", path: dirIn.value.trim() };
      if (dirName.value.trim()) body.name = dirName.value.trim();
      if (serverSel.value) body.server = serverSel.value;
      const r = await curateJob(body, "/api/datasets/curate_source",
        setProgress);
      if (r.ok) {
        out.textContent = `created ${r.dataset_id} (${r.rows} rows)`;
        toast(`dataset ${r.dataset_id} created from folder`, "ok");
        (r.warnings || []).slice(0, 3).forEach((w) => toast(w, "info"));
        onDone(r);
      } else out.textContent = `failed: ${r.error}`;
    } catch (e) { out.textContent = ""; toast(e.message, "error"); }
    finally { dirBtn.disabled = false; }
  });

  // -- website crawl
  // (placeholder deliberately avoids a scheme literal — the static-asset
  // hygiene test forbids remote URL strings in shipped JS.)
  const urlIn = el("input", { type: "text", class: "grow",
    placeholder: "entry page URL, e.g. docs.example.org/start.html",
    title: "First page of a documentation site. The crawler follows links from here, " +
      "but never leaves the same host (and by default stays in this page's folder)." });
  const pagesIn = el("input", { type: "number", min: "1", max: "30",
    placeholder: "30", class: "narrow",
    title: "Maximum number of pages to fetch (bounded crawl, 1–30). Each page costs " +
      "a synthesis call if a server is used." });
  // Language filter for localized docs trees (/zh/, /ja/…): empty = auto
  // (server detects from the system locale); "*" keeps every language.
  const langIn = el("input", { type: "text", class: "narrow",
    placeholder: "auto", title:
      "Only crawl pages in this language on /zh/ /ja/ … mirrored docs trees. " +
      "Empty = your system language; * = keep all languages." });
  // Crawl scope: by default the crawl stays inside the entry page's docs
  // subtree (e.g. /documentation/unreal-engine/) so it never wanders into the
  // host's forums/news/marketing pages. Empty = auto; "*" = whole same-origin.
  const scopeIn = el("input", { type: "text", class: "narrow",
    placeholder: "auto", title:
      "Path subtree to crawl, e.g. /documentation/unreal-engine/. " +
      "Empty = the entry page's folder (recommended); * = entire same-origin site." });
  const urlBtn = el("button", { text: "Crawl site",
    title: "Queue a same-origin bounded crawl of the site above and create a dataset " +
      "from the pages. Progress streams below and in the Jobs panel." });
  urlBtn.addEventListener("click", async () => {
    if (!urlIn.value.trim()) return toast("URL required", "error");
    urlBtn.disabled = true;
    setProgress("crawling… (queued job, same-origin bounded crawl)");
    try {
      const body = { source: "website", url: urlIn.value.trim() };
      if (pagesIn.value) body.max_pages = Number(pagesIn.value);
      if (langIn.value.trim()) body.lang = langIn.value.trim();
      if (scopeIn.value.trim())
        body.scope = scopeIn.value.trim() === "*" ? "" : scopeIn.value.trim();
      if (serverSel.value) body.server = serverSel.value;
      const r = await curateJob(body, "/api/datasets/curate_source",
        setProgress);
      if (r.ok) {
        out.textContent = `created ${r.dataset_id} (${r.rows} rows)`;
        toast(`dataset ${r.dataset_id} created from site`, "ok");
        (r.warnings || []).slice(0, 3).forEach((w) => toast(w, "info"));
        onDone(r);
      } else out.textContent = `failed: ${r.error}`;
    } catch (e) { out.textContent = ""; toast(e.message, "error"); }
    finally { urlBtn.disabled = false; }
  });

  // -- hub snapshot import (repo downloaded on the Hub page)
  const hubIn = el("input", { type: "text", class: "grow",
    placeholder: "hub dataset repo id (downloaded first, e.g. org/name)",
    title: "Repo id of a dataset you already downloaded on the Resource Hub page " +
      "(org/name). It is converted locally into the normal dataset format." });
  const hubBtn = el("button", { text: "Import hub dataset",
    title: "Convert a Hub-downloaded dataset into a local dataset record so it shows " +
      "up in the table and can train as-is. Runs immediately (no server needed)." });
  hubBtn.addEventListener("click", async () => {
    if (!hubIn.value.trim()) return toast("repo id required", "error");
    hubBtn.disabled = true;
    out.textContent = "importing hub snapshot…";
    try {
      const r = await api("/api/datasets/import_hub",
        { method: "POST", body: { repo_id: hubIn.value.trim() } });
      out.textContent = `created ${r.dataset_id} (${r.rows} rows)`;
      toast(`dataset ${r.dataset_id} imported from hub`, "ok");
      (r.warnings || []).slice(0, 3).forEach((w) => toast(w, "info"));
      onDone(r);
    } catch (e) { out.textContent = ""; toast(e.message, "error"); }
    finally { hubBtn.disabled = false; }
  });

  host.append(
    el("h3", { text: "Curation sources" }),
    el("p", { class: "hint", text:
      "Ingest a local docs folder, crawl a documentation site, or import a "
      + "dataset you downloaded on the Hub page. Doc files become ShareGPT "
      + "pairs via the configured local model (heading fallback offline). "
      + "Everything lands in the normal dataset list and trains as-is." }),
    el("div", { class: "field" }, el("label", {}, "folder"), dirIn, dirName,
      dirBtn),
    el("div", { class: "field" }, el("label", {}, "website"), urlIn,
      el("label", { title: "max pages (bounded crawl)" }, "pages"), pagesIn,
      el("label", { title: "crawl language (empty = system language, * = all)" },
        "lang"), langIn,
      el("label", { title: "path subtree to stay inside (empty = entry page's folder, * = whole site)" },
        "scope"), scopeIn,
      urlBtn),
    el("div", { class: "field" },
      el("label", { title: "serving instance used for pair synthesis" },
        "synth server"), serverSel),
    el("div", { class: "field" }, el("label", {}, "hub repo"), hubIn, hubBtn),
    out);
  return host;
}

// WP3b: `datasets` package status strip (training env). Install = one click,
// pip runs in the background against the training interpreter.
function datasetsEnvCard() {
  const host = el("div", { class: "card" });
  const reload = async () => {
    let st;
    try { st = await api("/api/datasets/env"); }
    catch { return; }
    if (st.datasets) {
      host.replaceChildren(el("p", { class: "hint", text:
        "datasets package: ok (" + st.python + ")" }));
      return;
    }
    const btn = el("button", { text: "Install datasets",
      title: "Run pip install against the training interpreter in the background. " +
        "The datasets package is required to start trainings." });
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      btn.textContent = "installing…";
      try {
        await api("/api/datasets/env/install", { method: "POST" });
        toast("pip install queued — poll back in a minute", "info");
        setTimeout(reload, 15000);
      } catch (e) { toast(e.message, "error"); btn.disabled = false;
                   btn.textContent = "Install datasets"; }
    });
    host.replaceChildren(
      el("p", { class: "hint", text: st.note || "datasets package missing" }),
      el("div", { class: "field" }, btn,
        el("code", { class: "mono hint",
          text: st.install_command || "pip install datasets" })));
  };
  reload();
  return host;
}

// ---------------------------------------------------------------- diff view

// Idea #6: pick two datasets -> summary of added/removed/unchanged trace ids,
// per-file hash verdicts and count deltas from GET /api/datasets/diff.
function diffCard(datasets) {
  const opts = (sel) => datasets.map((d) => el("option", {
    value: d.dataset_id || "", text: d.dataset_id || "?" , selected: sel === d.dataset_id }));
  const selA = el("select", { title: "Baseline dataset (side a)." }, ...opts());
  const selB = el("select", { title: "Dataset to compare against it (side b)." }, ...opts());
  if (datasets.length > 1) selB.selectedIndex = datasets.length - 1;
  const out = el("div", { class: "result" });
  const btn = el("button", { text: "Compare",
    title: "Show which trace ids were added/removed between the two datasets, " +
      "per-file content-hash verdicts, and row-count deltas." });

  const deltaCell = (v) => v == null ? "–"
    : (v > 0 ? `+${v}` : String(v));

  function verdictRow(label, equal) {
    return el("tr", {}, el("td", { class: "k", text: label }),
      el("td", {}, equal == null ? el("span", { class: "hint", text: "only one side" })
        : badge(equal ? "equal" : "differs", equal ? "success" : "partial")));
  }

  async function run() {
    const a = selA.value, b = selB.value;
    if (!a || !b) return toast("pick two datasets to compare", "error");
    out.textContent = "comparing…";
    let d;
    try {
      d = await api(`/api/datasets/diff?a=${encodeURIComponent(a)}` +
        `&b=${encodeURIComponent(b)}`);
    } catch (e) {
      out.replaceChildren(el("p", { class: "err", text: e.message }));
      return;
    }
    const tds = d.trace_diff || {}, fs = d.files || {};
    const rows = [];
    for (const [fname, t] of Object.entries(tds)) {
      rows.push(el("tr", {},
        el("td", { class: "k", text: `${fname} traces` }),
        el("td", { text: `+${t.added_count} / −${t.removed_count} / ` +
          `=${t.unchanged_count}` })));
      const sample = [...(t.added_sample || []), ...(t.removed_sample || [])];
      if (sample.length) {
        rows.push(el("tr", {}, el("td", {}),
          el("div", { class: "trace-links" }, ...sample.map((id) => traceLink(id)))));
      }
    }
    for (const [fname, f] of Object.entries(fs))
      rows.push(verdictRow(`${fname} hash`, f.hash_equal));
    rows.push(el("tr", {}, el("td", { class: "k", text: "sft count Δ" }),
      el("td", { text: `${d.counts.sft_a ?? 0} → ${d.counts.sft_b ?? 0} ` +
        `(${deltaCell(d.counts.sft_delta)})` })));
    rows.push(el("tr", {}, el("td", { class: "k", text: "dpo count Δ" }),
      el("td", { text: `${d.counts.dpo_a ?? 0} → ${d.counts.dpo_b ?? 0} ` +
        `(${deltaCell(d.counts.dpo_delta)})` })));
    rows.push(verdictRow("source", d.source_equal));
    out.replaceChildren(
      el("table", { class: "kv" }, el("tbody", {}, ...rows)),
      d.identical ? el("p", { class: "hint", text: "identical" }) : null);
  }
  btn.addEventListener("click", run);

  return el("div", { class: "card" },
    el("h3", { text: "Compare versions" }),
    el("p", { class: "hint", text:
      "Diff two datasets by trace-id sets and per-file content hashes " +
      "(manifest-derived; added/removed ids link back to the trace view)." }),
    el("div", { class: "field" },
      el("label", {}, "a"), selA, el("label", {}, "b"), selB, btn),
    out);
}

// ---------------------------------------------------------- WP9 funnel card

// WP9a: curation funnel history from GET /api/analytics/curation (persisted
// by Curater.curate() after each run). Purely additive: an empty/failed
// analytics history renders as a hint, never an error state.
function funnelCard() {
  const out = el("div");
  out.replaceChildren(el("p", { class: "hint", text: "loading funnel history…" }));

  api("/api/analytics/curation?limit=50").then((d) => {
    const hist = [...(d.history || [])].reverse(); // newest first
    if (!hist.length) {
      out.replaceChildren(el("p", { class: "hint", text:
        "No curation runs recorded yet — funnels appear after the next `toaster_loop curate`." }));
      return;
    }
    const maxScanned = Math.max(1, ...hist.map((h) => h.traces_scanned || 0));
    // Tiny inline funnel bar (reuses the .ctx-bar token styles): track width
    // ∝ scanned share of the biggest run, fill ∝ curated/scanned. The ratio
    // travels with its denominator in the title (metric-honesty rule).
    const funnelBar = (h) => {
      const scanned = h.traces_scanned || 0;
      const curated = h.curated || 0;
      const wPct = Math.max(8, Math.round((scanned / maxScanned) * 100));
      const cPct = scanned ? Math.round((curated / scanned) * 100) : 0;
      return el("div", { class: "ctx-bar", style: `width:${wPct}%`, title:
        `${curated} curated of ${scanned} scanned` },
        el("div", { class: "ctx-bar-fill", style: `width:${cPct}%` }));
    };
    out.replaceChildren(table([
      { label: "dataset", render: (h) => el("a", {
          href: "#/datasets/" + encodeURIComponent(h.dataset_id || ""),
          class: "mono", text: String(h.dataset_id || "?").slice(0, 18),
          onclick: (ev) => ev.stopPropagation() }) },
      { label: "run", render: (h) => fmtTime(h.created_at || h.ts, false) },
      { label: "funnel", render: funnelBar },
      { label: "scanned→curated", render: (h) =>
          `${h.traces_scanned ?? 0} → ${h.curated ?? 0}` },
      { label: "dedup", render: (h) => String(h.dedup_rejects ?? 0) },
      { label: "scored/total", render: (h) => {
          // Histogram denominator travels with the buckets (WP2c honesty).
          const sh = h.score_histogram || {};
          return el("span", { title: "scored / total traces in this run" },
            `${sh.scored ?? "?"}/${sh.total ?? "?"}`);
        } },
    ], hist, { empty: "no runs" }));
  }).catch((e) => {
    out.replaceChildren(el("p", { class: "hint", text: "funnel history unavailable: " + e.message }));
  });

  return el("div", { class: "card" },
    el("h3", { text: "Curation funnels" }),
    el("p", { class: "hint", text:
      "Per-run funnel: traces scanned → eligible → curated, with dedup rejects " +
      "and the score-histogram denominator (persisted history, newest first)." }),
    out);
}

// ---------------------------------------------------------------- list view

function listView(mount) {
  const unsubs = [];
  const tableHost = el("div");
  const diffHost = el("div");
  const funnelHost = el("div");
  let datasets = [];
  const runningCard = runningCurationCard(() => load());
  unsubs.push(() => runningCard.destroy());

  async function load() {
    try {
      datasets = (await api("/api/datasets")).datasets || [];
    } catch (e) {
      tableHost.replaceChildren(empty("failed to load: " + e.message));
      return;
    }
    if (datasets.length >= 2)
      diffHost.replaceChildren(diffCard(datasets));
    else
      diffHost.replaceChildren();
    const rows = [...datasets].reverse(); // server sorts oldest-first
    tableHost.replaceChildren(table([
      { label: "name", render: (d) => el("div", {},
          el("div", { text: d.title || d.dataset_id || "–" }),
          el("div", { class: "hint mono", text: d.dataset_id || "" })) },
      { label: "created", render: (d) => fmtTime(d.created_at, false) },
      // WP3c/d: provenance chip — curated datasets show nothing extra (the
      // common case stays quiet); hub/folder/website carry a source badge.
      { label: "source", render: (d) => {
          const p = d.provenance || "curated";
          if (p === "curated") return el("span", { class: "hint", text: "traces" });
          const what = p === "hub" ? (d.source_repo || "hub")
            : p === "local_folder" ? "folder" : p === "website" ? "web" : p;
          return badge(what, "info");
        } },
      { label: "trust", render: (d) => trustBadge(d) },
      { label: "gate", render: (d) => lintCell(d) },
      { label: "sft", render: (d) => d.files?.["sft.jsonl"]?.count ?? 0 },
      { label: "dpo", render: (d) => d.files?.["dpo.jsonl"]?.count ?? 0 },
      { label: "filters", render: (d) => {
          const f = d.filters || {};
          const bits = [];
          if (f.min_score != null) bits.push(`≥${f.min_score}`);
          if (f.include_failures) bits.push("+fail");
          if (f.agent_profile) bits.push(f.agent_profile);
          if (f.model_alias) bits.push(f.model_alias);
          if (d.warnings?.length) bits.push(`⚠${d.warnings.length}`);
          return bits.join(" · ") || "defaults";
        } },
    ], rows, {
      empty: "No datasets yet — curate buffered traces below.",
      onRowClick: (d) => go("/datasets/" + encodeURIComponent(d.dataset_id)),
    }));
  }

  const envHost = el("div");
  const sourcesHost = el("div");
  mount.replaceChildren(
    el("h2", { text: "Datasets" }), runningCard.node, tableHost, funnelHost,
    diffHost);
  funnelHost.replaceChildren(funnelCard());
  // WP3b/d: training-env package gate first (it blocks training, so it goes
  // above the work), then the extra curation sources next to Curate.
  envHost.replaceChildren(datasetsEnvCard());
  mount.insertBefore(envHost, funnelHost);
  sourcesHost.replaceChildren(sourcesCard(() => load()));
  curateForm(() => load()).then((f) => {
    mount.appendChild(f);
    mount.appendChild(sourcesHost);
  });
  load();
  return { destroy() { for (const u of unsubs) u(); } };
}

// -------------------------------------------------------------------- panel

export default {
  title: "Datasets",
  render(mount, params) {
    if (params && params.length && params[0])
      return detailView(mount, decodeURIComponent(params[0]));
    return listView(mount);
  },
};

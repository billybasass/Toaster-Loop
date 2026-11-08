// panels/train.js — Train wizard (Phase 5d).
// Readiness step over GET /api/train/check (unclaimed-traces + mean-score
// gauges vs thresholds, scheduler resource gate), dataset picker with a
// "curate new…" shortcut into the Datasets panel form, kind/dry-run selects,
// then POST /api/train. Success links to the created job and opens the shared
// LogPane inline; single-flight 409 surfaces the running job with a view link.

import { api } from "../api.js";
import { go } from "../router.js";
import { el, toast, badge, empty, svgGauge } from "../ui.js";
import { createLogPane } from "../logpane.js";

export default {
  title: "Train",
  render(mount) {
    const checkCard = el("div", { class: "card" },
      el("h3", { text: "1 · Readiness" }),
      el("p", { class: "hint", text: "checking buffer…" }));
    const formCard = el("div", { class: "card hidden" });
    const resultHost = el("div");
    let logpane = null;
    // GGUF→HF-base download poller (cleared on panel destroy)
    let pollTimer = null;

    mount.replaceChildren(
      el("h2", { text: "Train" }),
      el("p", { class: "hint", text:
        "Fine-tune from a curated dataset. One GPU, one job — training is " +
        "single-flight, so the wizard refuses to start while another job runs." }),
      checkCard, formCard, resultHost);

    // ---------------------------------------------------------- readiness
    function renderCheck(d) {
      const th = d.threshold || {};
      const unclaimed = d.unclaimed ?? 0;
      const mean = d.mean_score;
      const enoughTraces = !!d.enough_traces;
      const gate = d.gate || { ok: true, reasons: [] };

      const traceGauge = svgGauge(unclaimed, Math.max(th.min_traces || 1, unclaimed), {
        label: `unclaimed traces (need ${th.min_traces})`,
        threshold: th.min_traces, danger: !enoughTraces });
      const scoreOk = mean != null && mean >= (th.min_score ?? 0);
      const scoreGauge = svgGauge(mean == null ? 0 : Math.round(mean * 100) / 100,
        1, { label: `mean score (need ≥ ${th.min_score})`, unit: "",
          threshold: th.min_score, danger: !scoreOk });

      const missing = [];
      if (!enoughTraces)
        missing.push(`${Math.max(0, (th.min_traces || 0) - unclaimed)} more ` +
          `unclaimed traces (curate frees claimed ones, or record more sessions)`);
      if (mean == null) missing.push("no scored traces yet — rate some in Chat");
      else if (!scoreOk) missing.push(`mean score below ${th.min_score}`);

      const bits = [traceGauge, scoreGauge];
      const gateRow = el("div", { class: "field" },
        badge(gate.ok ? "gate ok" : "gate blocked", gate.ok ? "done" : "error"),
        el("span", { class: "hint" },
          gate.ok ? "resource gate would allow a training start"
                  : (gate.reasons || []).join("; ")));
      checkCard.replaceChildren(
        el("h3", {}, "1 · Readiness ",
          badge(enoughTraces ? "buffer ready" : "buffer not ready",
            enoughTraces ? "done" : "queued")),
        el("div", { class: "gauges" }, ...bits),
        gateRow,
        missing.length
          ? el("p", { class: "hint" }, "Missing: " + missing.join("; ") + ".")
          : el("p", { class: "hint", text: "Buffer thresholds met — pick a dataset below." }),
        el("div", { class: "field" },
          el("button", { class: "small", text: "re-check", onclick: loadCheck })));
    }

    async function loadCheck() {
      try { renderCheck(await api("/api/train/check")); }
      catch (e) {
        checkCard.replaceChildren(el("h3", { text: "1 · Readiness" }),
          el("p", { class: "err", text: e.message }));
      }
    }

    // -------------------------------------------------------------- form
    async function loadForm() {
      let datasets = [];
      try { datasets = (await api("/api/datasets")).datasets || []; }
      catch { /* picker stays empty; curate shortcut still works */ }

      // Discovered models from the base model folder. Training needs HF
      // checkpoints, but the dropdown lists GGUFs too: a GGUF is a serving
      // artifact, yet it carries its source model's name, so selecting one
      // opens the resolve panel (GET /api/model_base) — find or download the
      // real HF base instead of dead-ending at "serve-only". Training always
      // runs on safetensors; GGUF weights are never converted.
      let catalog = { exists: false, base_models: [], gguf: [], models_dir: null };
      try { catalog = await api("/api/model_catalog"); }
      catch { /* picker falls back to the configured default only */ }

      // Multi-select dataset list: checking several trains on a merged view
      // (the server concatenates their rows into one dataset on submit).
      const dsChecks = new Map();
      const dsBox = el("div", { class: "ds-checklist" },
        ...[...datasets].reverse().map((d) => {
          const c = el("input", { type: "checkbox" });
          dsChecks.set(d.dataset_id, c);
          return el("label", { class: "chk" }, c, " ",
            el("span", { class: "mono", text: d.dataset_id }),
            el("span", { class: "hint" },
              ` (${d.files?.["sft.jsonl"]?.count ?? 0} sft, ` +
              `${d.files?.["dpo.jsonl"]?.count ?? 0} dpo)`));
        }));
      if (!datasets.length) dsBox.appendChild(el("span", { class: "hint", text: "no datasets yet — curate one below" }));
      const pickedDatasets = () =>
        [...dsChecks.entries()].filter(([, c]) => c.checked).map(([id]) => id);

      // GGUF entries are selectable via a "gguf:<path>" sentinel: choosing one
      // opens the resolve panel below instead of submitting (the submit path
      // rejects sentinels with a friendly message).
      const baseSel = el("select", { title:
        "Base weights to fine-tune. Training always runs on HF safetensors, never " +
        "on a GGUF — picking a GGUF here just helps you find its HF base." },
        el("option", { value: "", text: "— configured default (training.base_model) —" }),
        ...(catalog.base_models || []).map((m) => el("option", {
          value: m.path, text: m.name })),
        ...(catalog.gguf || []).map((m) => el("option", {
          value: "gguf:" + m.path,
          text: `${m.name}${m.quant ? " [" + m.quant + "]" : ""} — GGUF (pick HF base…)` })));
      const baseHint = el("span", { class: "hint" },
        catalog.exists
          ? `${(catalog.base_models || []).length} HF checkpoint(s) in ${catalog.models_dir}` +
            ((catalog.gguf || []).length
              ? ` (+${(catalog.gguf || []).length} GGUF(s): select one to find its HF base)` : "")
          : "no model folder found (set models_dir or the HF cache)");
      const baseResolveHost = el("div", { class: "hidden" });

      // ------------------------------------------------- GGUF → HF base resolve
      async function renderBaseResolve(ggufPath) {
        baseResolveHost.classList.remove("hidden");
        baseResolveHost.replaceChildren(
          el("p", { class: "hint", text: "resolving the HF base for this GGUF…" }));
        let d;
        try {
          d = await api("/api/model_base?gguf=" + encodeURIComponent(ggufPath));
        } catch (e) {
          baseResolveHost.replaceChildren(
            el("p", { class: "err", text: "resolve failed: " + e.message }));
          return;
        }
        const bits = [];
        if (d.basename || d.arch) {
          bits.push(el("p", { class: "hint" },
            `GGUF metadata: ${d.basename || "?"}` +
            (d.arch ? ` · arch ${d.arch}` : "") +
            (d.source_repo ? ` · from hub repo ${d.source_repo}` : "")));
        }
        const useLocal = (path, label) => {
          baseSel.value = path;
          baseResolveHost.classList.add("hidden");
          toast(`base model set: ${label}`, "ok");
          loadEstimate();
        };
        for (const m of d.local || []) {
          bits.push(el("div", { class: "field" },
            el("button", { class: "small primary", text: "use this",
              onclick: () => useLocal(m.path, m.repo_id) }),
            " ", el("span", { class: "mono", text: m.repo_id }),
            el("span", { class: "hint" }, " — HF checkpoint already on disk")));
        }
        for (const c of d.remote || []) {
          const btn = el("button", { class: "small", text: "download" });
          btn.onclick = async () => {
            btn.disabled = true;
            try {
              const r = await api("/api/model_base/download",
                { method: "POST", body: { repo: c.repo_id } });
              btn.textContent = "downloading…";
              pollDownload(r.job_id, c.repo_id, btn);
            } catch (e) { toast(e.message, "error"); btn.disabled = false; }
          };
          bits.push(el("div", { class: "field" }, btn, " ",
            el("span", { class: "mono", text: c.repo_id }),
            el("span", { class: "hint" },
              ` — hub download${c.downloads ? ` (⬇ ${c.downloads})` : ""}` +
              `${c.source === "card" ? ", declared by the GGUF repo's card" : ""}`)));
        }
        const manualIn = el("input", { type: "text", class: "grow",
          placeholder: "org/model (HF repo id)",
          title: "Hugging Face repo id of the base checkpoint, if none of the " +
            "suggestions above are right. It is downloaded to the local models dir." });
        bits.push(el("div", { class: "field" }, manualIn,
          el("button", { class: "small", text: "download manual pick",
            title: "Download the typed HF repo id in the background; it appears in " +
              "the base-model list when finished.",
            onclick: async () => {
              const repo = manualIn.value.trim();
              if (!repo) return toast("enter an HF repo id first", "error");
              try {
                const r = await api("/api/model_base/download",
                  { method: "POST", body: { repo } });
                toast(`download ${r.job_id} started`, "ok");
              } catch (e) { toast(e.message, "error"); }
            } })));
        if (!d.local?.length && !d.remote?.length) {
          bits.unshift(el("p", { class: "hint" },
            d.reason || "no HF base found — enter the repo id manually below."));
        } else {
          bits.unshift(el("p", { class: "hint" },
            "Training runs on real HF weights (GGUFs are never converted). " +
            "Pick one of these, or enter the repo id manually:"));
        }
        baseResolveHost.replaceChildren(...bits);
      }
      function pollDownload(jobId, repo, btn) {
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(async () => {
          let j;
          try { j = await api("/api/hub/downloads/" + encodeURIComponent(jobId)); }
          catch { return; }
          if (j.state === "done") {
            clearInterval(pollTimer); pollTimer = null;
            toast(`${repo} downloaded — reselect it in the base list`, "ok");
            loadForm();  // refresh catalog so the new checkpoint appears
          } else if (j.state === "error" || j.state === "cancelled") {
            clearInterval(pollTimer); pollTimer = null;
            if (btn) { btn.disabled = false; btn.textContent = "retry download"; }
            toast(`download ${j.state}: ${j.error || ""}`, "error");
          }
        }, 2000);
      }
      baseSel.addEventListener("change", () => {
        const v = baseSel.value;
        if (v.startsWith("gguf:")) {
          renderBaseResolve(v.slice(5));
        } else {
          if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
          baseResolveHost.classList.add("hidden");
        }
      });
      const modelNameIn = el("input", { type: "text", class: "grow",
        placeholder: "name for the trained model (optional)", maxlength: "80",
        title: "Friendly label for the adapter produced by this run, shown on the " +
          "Models page. Empty = auto-generated from the dataset id." });
      const kindSel = el("select", { title:
        "sft = imitate the chosen answers; dpo = push toward chosen and away from " +
        "rejected pairs; grpo = reinforcement learning with rule-based rewards." },
        el("option", { value: "sft", text: "sft (supervised)" }),
        el("option", { value: "dpo", text: "dpo (preference pairs)" }),
        el("option", { value: "grpo", text: "grpo (RL, rule rewards)" }));
      const dryRun = el("input", { type: "checkbox",
        title: "Walk the whole pipeline — config validation, dataset loading, VRAM " +
          "estimate — but skip the actual GPU training. Use it to catch mistakes free." });
      const startBtn = el("button", { class: "primary", text: "Start training",
        title: "Queue a training job with the settings above. One job at a time; " +
          "if the GPU is busy the job waits and tells you why." });

      // VRAM estimate (#20): predicted footprint vs free VRAM before submit.
      const estRow = el("div", { class: "field hidden" });
      let estSeq = 0;
      async function loadEstimate() {
        const picked = pickedDatasets();
        if (!picked.length) { estRow.classList.add("hidden"); return; }
        // A GGUF sentinel is not a trainable path — no estimate until the
        // resolve panel sets a real HF checkpoint.
        if (baseSel.value.startsWith("gguf:")) { estRow.classList.add("hidden"); return; }
        const seq = ++estSeq;
        estRow.classList.remove("hidden");
        estRow.replaceChildren(el("span", { class: "hint", text: "estimating VRAM…" }));
        try {
          // Estimate off the first pick (footprint is config-driven, not
          // row-count driven); a merge cannot make it smaller.
          const qs = new URLSearchParams({ dataset: picked[0] });
          if (baseSel.value) qs.set("base_model", baseSel.value);
          const d = await api("/api/train/estimate?" + qs.toString());
          if (seq !== estSeq) return;  // stale response: a newer pick won
          if (!d.estimated) {
            estRow.replaceChildren(el("span", { class: "hint", text:
              "VRAM estimate unavailable (base model dims not found locally) — " +
              "the scheduler falls back to the static floor." }));
            return;
          }
          const e = d.estimated;
          const gib = (mib) => (mib / 1024).toFixed(1) + " GiB";
          const bits = [badge("est. " + gib(e.total_mib),
            d.fits === false ? "error" : "done"),
            el("span", { class: "hint" },
              `weights ${gib(e.weights_mib)} · optim ${gib(e.optimizer_mib)}` +
              ` · activations ${gib(e.activations_mib)} · overhead ${gib(e.overhead_mib)}` +
              ` (${e.load_mode})`)];
          if (d.free_mib != null) {
            bits.push(el("span", { class: "hint" },
              d.fits ? `free ${gib(d.free_mib)}, headroom ${gib(d.headroom_mib)}`
                     : `only ${gib(d.free_mib)} free — short ${gib(-d.headroom_mib)}; ` +
                       "the job will queue with this reason"));
          }
          estRow.replaceChildren(...bits);
        } catch {
          if (seq === estSeq) estRow.classList.add("hidden");
        }
      }
      for (const c of dsChecks.values()) c.addEventListener("change", loadEstimate);
      baseSel.addEventListener("change", loadEstimate);

      startBtn.addEventListener("click", async () => {
        const picked = pickedDatasets();
        if (!picked.length) return toast("pick at least one dataset first (or curate one)", "error");
        // GGUF entries are resolve-panel triggers, never training bases: the
        // real HF checkpoint must be chosen first (weights are not converted).
        if (baseSel.value.startsWith("gguf:")) {
          return toast("pick the HF base for that GGUF in the panel below it — " +
                       "training runs on safetensors, not GGUF", "error");
        }
        startBtn.disabled = true;
        try {
          const body = { datasets: picked, kind: kindSel.value,
            dry_run: dryRun.checked };
          if (baseSel.value) body.base_model = baseSel.value;
          if (modelNameIn.value.trim()) body.model_name = modelNameIn.value.trim();
          const r = await api("/api/train", { method: "POST", body });
          toast(`training job ${r.job_id} queued`, "ok");
          showJob(r.job_id);
        } catch (e) {
          if (e.code === "already_running") {
            const m = /job (\S+) is already running/.exec(e.message);
            toast(e.message, "error");
            showRunningLink(m ? m[1] : null);
          } else toast(`${e.code}: ${e.message}`, "error");
        } finally { startBtn.disabled = false; }
      });

      formCard.replaceChildren(
        el("h3", { text: "2 · Dataset & mode" }),
        el("div", { class: "field" }, el("label", { text: "datasets (pick one or more)" }), dsBox,
          el("button", { class: "small", text: "curate new…",
            title: "Open the Datasets page to build a fresh dataset from buffered traces.",
            onclick: () => go("/datasets") })),
        el("p", { class: "hint", text:
          "Multiple picks are merged into one training dataset at submit time " +
          "(rows concatenated; the originals stay untouched)." }),
        el("div", { class: "field" }, el("label", { text: "base model" }),
          baseSel, baseHint),
        baseResolveHost,
        el("div", { class: "field" }, el("label", { text: "model name" }), modelNameIn),
        el("div", { class: "field" }, el("label", { text: "kind" }), kindSel),
        el("div", { class: "field" },
          el("label", { class: "chk" }, dryRun,
            " dry run (validate config + dataset, no GPU)")),
        estRow,
        el("div", { class: "field" }, startBtn));
      formCard.classList.remove("hidden");
    }

    // ------------------------------------------------------------- result
    function showRunningLink(jobId) {
      resultHost.replaceChildren(el("div", { class: "card" },
        el("h3", { text: "Already running" }),
        el("p", {}, "A job holds the GPU right now. ",
          jobId ? el("a", { href: "#/jobs/" + encodeURIComponent(jobId),
            text: `view ${jobId} →` }) : el("a", { href: "#/jobs", text: "view jobs →" }))));
    }

    function showJob(jobId) {
      if (logpane) logpane.destroy();
      const link = el("a", { href: "#/jobs/" + encodeURIComponent(jobId),
        class: "mono", text: `#/jobs/${jobId}` });
      const paneHost = el("div");
      logpane = createLogPane(jobId, {});
      paneHost.appendChild(logpane.node);
      resultHost.replaceChildren(
        el("div", { class: "card" },
          el("h3", {}, "Job ", el("span", { class: "mono", text: jobId }), " ", link),
          el("p", { class: "hint", text:
            "Live log below (auto-tailing). Open the Jobs panel for queue context." }),
          paneHost));
    }

    loadCheck();
    loadForm();
    return { destroy() {
      if (logpane) logpane.destroy();
      if (pollTimer) clearInterval(pollTimer);
    } };
  },
};

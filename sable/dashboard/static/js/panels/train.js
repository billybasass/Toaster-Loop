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

      // Discovered base models (HF checkpoints) from the base model folder —
      // GGUFs are serve-only, so training lists HF-format snapshots only.
      let catalog = { exists: false, base_models: [], models_dir: null };
      try { catalog = await api("/api/model_catalog"); }
      catch { /* picker falls back to the configured default only */ }

      const dsSel = el("select", {},
        el("option", { value: "", text: "— select dataset —" }),
        ...[...datasets].reverse().map((d) => el("option", {
          value: d.dataset_id,
          text: `${d.dataset_id} (${d.sft_count ?? "?"} sft, ${d.dpo_count ?? "?"} dpo)` })));
      const baseSel = el("select", {},
        el("option", { value: "", text: "— configured default (training.base_model) —" }),
        ...(catalog.base_models || []).map((m) => el("option", {
          value: m.path, text: m.name })));
      const baseHint = el("span", { class: "hint" },
        catalog.exists
          ? `${(catalog.base_models || []).length} HF checkpoint(s) in ${catalog.models_dir}`
          : "no model folder found (set models_dir or the HF cache)");
      const kindSel = el("select", {},
        el("option", { value: "sft", text: "sft (supervised)" }),
        el("option", { value: "dpo", text: "dpo (preference pairs)" }),
        el("option", { value: "grpo", text: "grpo (RL, rule rewards)" }));
      const dryRun = el("input", { type: "checkbox" });
      const startBtn = el("button", { class: "primary", text: "Start training" });

      // VRAM estimate (#20): predicted footprint vs free VRAM before submit.
      const estRow = el("div", { class: "field hidden" });
      let estSeq = 0;
      async function loadEstimate() {
        if (!dsSel.value) { estRow.classList.add("hidden"); return; }
        const seq = ++estSeq;
        estRow.classList.remove("hidden");
        estRow.replaceChildren(el("span", { class: "hint", text: "estimating VRAM…" }));
        try {
          const qs = new URLSearchParams({ dataset: dsSel.value });
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
      dsSel.addEventListener("change", loadEstimate);
      baseSel.addEventListener("change", loadEstimate);

      startBtn.addEventListener("click", async () => {
        if (!dsSel.value) return toast("pick a dataset first (or curate one)", "error");
        startBtn.disabled = true;
        try {
          const body = { dataset: dsSel.value, kind: kindSel.value,
            dry_run: dryRun.checked };
          if (baseSel.value) body.base_model = baseSel.value;
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
        el("div", { class: "field" }, el("label", { text: "dataset" }), dsSel,
          el("button", { class: "small", text: "curate new…",
            onclick: () => go("/datasets") })),
        el("div", { class: "field" }, el("label", { text: "base model" }),
          baseSel, baseHint),
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
    return { destroy() { if (logpane) logpane.destroy(); } };
  },
};

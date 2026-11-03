// panels/hardware.js — live hardware panel (Phase 5d).
// Polls GET /api/hardware (~3 s while visible; server caches at 2 s TTL) and
// renders CPU, RAM, per-GPU (free/total bar + util/temp), disk free at
// TOASTER_LOOP_HOME and battery/AC. Sparkline history is client-side only (ring
// buffers). The AC-power indicator also shows the scheduler resource-gate
// verdict from GET /api/train/check so you see *why* training would wait.

import { api } from "../api.js";
import { bus } from "../bus.js";
import { el, badge, toast, confirmDialog, svgGauge, svgSparkline,
  empty } from "../ui.js";

const POLL_MS = 3000;
const RING = 120; // ~6 minutes at 3 s

class Ring {
  constructor(n = RING) { this.n = n; this.a = []; }
  push(v) { this.a.push(v); if (this.a.length > this.n) this.a.shift(); }
  get values() { return this.a; }
}

const fmtMib = (m) => m == null ? "?" : (m >= 1024 ? `${(m / 1024).toFixed(1)} GiB` : `${m} MiB`);

export default {
  title: "Hardware",
  render(mount) {
    const cards = el("div", { class: "cards" });
    const gateCard = el("div", { class: "card" },
      el("h3", { text: "Training gate" }),
      el("p", { class: "hint", text: "loading…" }));
    const runtimeCard = el("div", { class: "card" },
      el("h3", { text: "llama.cpp runtime" }),
      el("p", { class: "hint", text: "loading…" }));
    const enginesCard = el("div", { class: "card" },
      el("h3", { text: "Inference engines" }),
      el("p", { class: "hint", text: "loading…" }));
    const installCard = el("div", { class: "card" },
      el("h3", { text: "Install engines" }),
      el("p", { class: "hint", text: "loading…" }));
    mount.replaceChildren(
      el("h2", { text: "Hardware" }),
      el("p", { class: "hint", text:
        "cpu · memory · gpu · disk · power — sampled every 3 s while this " +
        "panel is open; sparklines cover the last ~6 minutes." }),
      cards, runtimeCard, enginesCard, installCard, gateCard);

    const hist = {
      cpu: new Ring(), memUsed: new Ring(), diskFree: new Ring(),
      gpus: [], // per-index Rings: {util, free}
    };
    let timer = null;

    function gpuHist(i) {
      while (hist.gpus.length <= i)
        hist.gpus.push({ util: new Ring(), free: new Ring() });
      return hist.gpus[i];
    }

    function card(title, nodes) {
      return el("div", { class: "card" }, el("h3", { text: title }), ...nodes);
    }

    function render(s) {
      const bits = [];

      // CPU
      if (s.cpu_pct != null) hist.cpu.push(s.cpu_pct);
      bits.push(card("CPU", [
        el("div", { class: "big mono" }, `${s.cpu_pct == null ? "–" : Math.round(s.cpu_pct)}%`),
        svgSparkline(hist.cpu.values, { label: "cpu %", min: 0, max: 100 }),
      ]));

      // RAM
      if (s.mem) {
        hist.memUsed.push(s.mem.used_mib);
        bits.push(card("Memory", [
          svgGauge(s.mem.used_mib, s.mem.total_mib, { label: "used", unit: " MiB" }),
          el("p", { class: "hint" }, `${fmtMib(s.mem.available_mib)} available`),
          svgSparkline(hist.memUsed.values, { label: "used MiB", min: 0, max: s.mem.total_mib }),
        ]));
      } else bits.push(card("Memory", [empty("unavailable")]));

      // GPUs
      const gpus = s.gpus || [];
      gpus.forEach((g, i) => {
        const h = gpuHist(i);
        if (g.util_pct != null) h.util.push(g.util_pct);
        if (g.free_mib != null && g.total_mib) h.free.push(g.total_mib - g.free_mib);
        const usedMib = g.total_mib != null && g.free_mib != null
          ? g.total_mib - g.free_mib : null;
        bits.push(card(`GPU ${i} · ${g.name || ""}`, [
          svgGauge(usedMib, g.total_mib, { label: "VRAM used", unit: " MiB" }),
          el("div", { class: "field" },
            el("span", { class: "hint mono" },
              `util ${g.util_pct ?? "?"}% · ${g.temperature_c ?? "?"}°C`),
            g.temperature_c != null && g.temperature_c >= 85
              ? badge("hot", "error") : null),
          svgSparkline(h.util.values, { label: "util %", min: 0, max: 100 }),
        ]));
      });
      if (!gpus.length) bits.push(card("GPU", [empty("no GPU detected (nvidia-smi unavailable)")]));

      // Disk at home
      if (s.disk) {
        hist.diskFree.push(s.disk.free_mib);
        bits.push(card("Disk (home)", [
          svgGauge(s.disk.total_mib - s.disk.free_mib, s.disk.total_mib,
            { label: "used", unit: " MiB" }),
          el("p", { class: "hint" }, `${fmtMib(s.disk.free_mib)} free`),
          svgSparkline(hist.diskFree.values, { label: "free MiB", min: 0 }),
        ]));
      } else bits.push(card("Disk", [empty("unavailable")]));

      // Power
      if (s.battery) {
        bits.push(card("Power", [
          el("div", { class: "big" }, s.battery.ac ? "⚡ AC power" : "🔋 battery"),
          el("p", { class: "hint", text:
            "training.require_ac_power blocks job starts on battery" }),
        ]));
      } else bits.push(card("Power", [empty("no battery reported (desktop?)")]));

      cards.replaceChildren(...bits);
    }

    async function poll() {
      if (!document.body.contains(cards)) return; // navigated away
      try { render(await api("/api/hardware")); } catch { /* transient */ }
    }

    async function pollGate() {
      try {
        const d = await api("/api/train/check");
        const g = d.gate || { ok: true, reasons: [] };
        gateCard.replaceChildren(
          el("h3", {}, "Training gate ", badge(g.ok ? "open" : "blocked", g.ok ? "done" : "error")),
          el("p", { class: "hint", text: g.ok
            ? "resource gate (VRAM headroom, AC power) would let jobs start now"
            : g.reasons.join("; ") }),
          el("a", { href: "#/train", text: "train wizard →" }));
      } catch (e) {
        gateCard.replaceChildren(el("h3", { text: "Training gate" }),
          el("p", { class: "err", text: e.message }));
      }
    }

    // llama.cpp runtime card: backend/GPU state, managed install provenance,
    // update check + one-click setup/update (repo/ref/backend overridable).
    // Each backend lives in its own install slot (llama.cpp, llama.cpp-vulkan,
    // …) and registers as a separate engine; the primary auto-setup slot is
    // represented by the built-in engine row.
    let runtimeBusy = false;
    let lastRuntime = null;
    let lastRuntimeJson = null; // last rendered payload — skips no-op re-renders
    let setupTimer = null;      // shared pollSetup timer (prevents stacked chains)
    // Install/setup logs go to the devtools console instead of the page. The
    // cursor map tracks how many lines of each run were already echoed; a
    // shorter log than the cursor means the server-side buffer reset (a new
    // run), so counting restarts.
    const logCursors = new Map();
    function echoLog(key, lines) {
      const all = lines || [];
      let n = logCursors.get(key) || 0;
      if (all.length < n) n = 0;
      for (let i = n; i < all.length; i++) console.log(`[${key}] ${all[i]}`);
      logCursors.set(key, Math.max(n, all.length));
    }
    // Which backend the update-check should follow (kept in sync with the
    // dropdown so the badge matches what Install/Update would act on).
    let runtimeCheckBackend = null;
    // Unapplied user edits to the backend/repo/ref form. The card re-renders
    // every 3 s from server state; without this the poll would snap the
    // dropdown back out from under the user mid-selection. Cleared once a
    // build finishes (the new state then comes from the server).
    let runtimePending = null;
    async function pollRuntime(check) {
      if (!document.body.contains(runtimeCard)) return;
      try {
        const qs = check ? "" : "?check=0";
        const be = runtimeCheckBackend
          ? (check ? "?backend=" + encodeURIComponent(runtimeCheckBackend)
                   : "&backend=" + encodeURIComponent(runtimeCheckBackend))
          : "";
        const d = await api("/api/runtime" + qs + be);
        // Skip the full re-render when nothing changed. Rebuilding the card
        // every 3 s destroyed and recreated the log <pre>, blanking it for a
        // frame each poll (the visible "log flicker").
        const json = JSON.stringify(d);
        if (json === lastRuntimeJson) return;
        lastRuntimeJson = json;
        renderRuntime(d);
      } catch (e) {
        runtimeCard.replaceChildren(el("h3", { text: "llama.cpp runtime" }),
          el("p", { class: "err", text: e.message }));
      }
    }

    // Poll the running/last llama.cpp build; on completion sync engine rows.
    // One shared timer keeps duplicate poll chains from stacking up when
    // several callers (click handler + render-time resume check) kick it off.
    function scheduleSetupPoll() {
      if (setupTimer) clearTimeout(setupTimer);
      setupTimer = setTimeout(pollSetup, 2000);
    }

    async function pollSetup() {
      if (!document.body.contains(runtimeCard)) return;
      try {
        const s = await api("/api/runtime/setup");
        echoLog("llama.cpp", s.log);
        if (s.state === "running") { runtimeBusy = true; scheduleSetupPoll(); return; }
        runtimeBusy = false;
        runtimePending = null; // build finished — server state is authoritative again
        if (s.state === "error") console.error(`[llama.cpp] error: ${s.error}`);
        // A finished build re-registers its engine row; make sure every
        // managed slot has an entry (idempotent, covers older installs too).
        try { await api("/api/runtime/sync", { method: "POST", body: {} }); }
        catch { /* transient */ }
        pollEngines();
        pollRuntime(true); // refresh provenance + update badge
      } catch { /* transient */ }
    }

    // Kick a llama.cpp build for one backend (new install or update). Shared
    // by the runtime-card button and the per-engine update buttons. A null/
    // undefined backend targets the primary slot (the built-in engine's
    // checkout).
    function startLlamaBuild(backend, update) {
      const what = update ? "Update" : "Install";
      const label = backend || "primary";
      if (!confirm(`${what} llama.cpp (${label}) on the server? Each backend ` +
        "gets its own install; existing builds are left untouched. This can " +
        "take many minutes.")) return;
      api("/api/runtime/setup", { method: "POST", body: {
        backend: backend || undefined, update,
      } }).then(() => {
        runtimeBusy = true;
        if (backend) runtimeCheckBackend = backend;
        if (lastRuntime) renderRuntime(lastRuntime);
        pollSetup();
      }).catch((e) => toast(e.message, "error"));
    }

    function renderRuntime(d) {
      lastRuntime = d;
      const installs = d.installs || [];
      const pend = runtimePending || {};
      const wantBackend = pend.backend != null ? pend.backend
        : (runtimeCheckBackend || d.configured.backend || d.backend);
      // Install for the backend currently selected in the dropdown (the
      // primary slot counts as its recorded backend).
      const inst = installs.find((i) => i.backend === wantBackend) || null;
      const upd = d.update || {};
      runtimeCheckBackend = wantBackend;
      const sel = el("select", { class: "input" });
      (d.backends || []).forEach((b) => {
        const o = el("option", { value: b, text: b });
        if (wantBackend === b) o.selected = true;
        sel.appendChild(o);
      });
      sel.addEventListener("change", () => {
        runtimePending = Object.assign({}, runtimePending, { backend: sel.value });
        runtimeCheckBackend = sel.value;
        renderRuntime(d); // re-label the button (Install vs Update <backend>)
      });
      const repoIn = el("input", { class: "input mono", type: "text",
        placeholder: inst?.repo || d.configured.repo || "",
        value: pend.repo != null ? pend.repo : (inst?.repo || "") });
      repoIn.addEventListener("input", () => {
        runtimePending = Object.assign({}, runtimePending, { repo: repoIn.value });
      });
      const refIn = el("input", { class: "input mono", type: "text",
        placeholder: inst?.ref || d.configured.ref || "master",
        value: pend.ref != null ? pend.ref : (inst?.ref || ""),
        style: "max-width:14em" });
      refIn.addEventListener("input", () => {
        runtimePending = Object.assign({}, runtimePending, { ref: refIn.value });
      });

      const btn = el("button", { class: "btn primary",
        text: runtimeBusy ? "running…"
          : (inst ? `Update ${wantBackend}` : `Install ${wantBackend}`) });
      if (runtimeBusy) btn.disabled = true;
      btn.addEventListener("click", async () => {
        if (!confirm(`Build llama.cpp (${sel.value}) on the server? Each ` +
          "backend gets its own install; existing builds are left untouched. " +
          "This can take many minutes.")) return;
        try {
          await api("/api/runtime/setup", { method: "POST", body: {
            backend: sel.value, repo: repoIn.value || undefined,
            ref: refIn.value || undefined, update: !!inst } });
          runtimeBusy = true; renderRuntime(d); pollSetup();
        } catch (e) { alert(e.message); }
      });

      // Resume polling for a build started in another tab or before this page
      // load; its log lines go to the devtools console, not the page.
      api("/api/runtime/setup").then((s) => {
        echoLog("llama.cpp", s.log);
        if (s.state === "running") { runtimeBusy = true; scheduleSetupPoll(); }
      }).catch(() => {});

      const status = !inst
        ? badge("not installed", "error")
        : upd.update_available ? badge("update available", "warn")
        : badge("up to date", "done");
      // NOTE: per-install rows used to live here too; they duplicated the
      // Inference-engines table (every managed slot registers an engine row
      // with its own update/delete), so provenance now shows only there.
      // NOTE: replaceChildren() stringifies non-Node args (a `null` renders as
      // visible "null" text), so filter the conditional lines out first.
      runtimeCard.replaceChildren(...[
        el("h3", {}, "llama.cpp runtime ", status),
        el("p", { class: "hint mono" },
          `${d.system}/${d.machine} · backend ${d.backend} · -ngl ${d.offload_ngl}` +
          ` · gpus ${(d.gpus || []).map((g) => g.name).join(", ") || "none"}`),
        installs.length
          ? el("p", { class: "hint" },
            `${installs.length} managed install(s); each backend builds into ` +
            "its own checkout and registers as a separate engine below.") : null,
        !inst && d.update?.reason ? el("p", { class: "hint" }, d.update.reason) : null,
        inst && upd.remote_commit ? el("p", { class: "hint mono" },
          `remote ${upd.remote_commit}${upd.update_available ? " — UPDATE AVAILABLE" : " (up to date)"}`) : null,
        el("div", { class: "field" }, sel, repoIn, refIn, btn),
      ].filter(Boolean));
    }

    // ----------------------------------------------- inference engines card
    // Registry of local OpenAI-compatible engines (vLLM, llama-swap, …). The
    // built-in llama.cpp engine is always present and immutable; registered
    // engines become selectable as the spawn engine for server instances on
    // the Serve page and as the default server's engine on Settings.
    async function pollEngines() {
      if (!document.body.contains(enginesCard)) return;
      try { renderEngines(await api("/api/engines")); }
      catch (e) {
        enginesCard.replaceChildren(el("h3", { text: "Inference engines" }),
          el("p", { class: "err", text: e.message }));
      }
    }

    function renderEngines(d) {
      const engines = d.engines || [];
      const defaultEngine = d.default_engine || "";
      // Catalog-managed engines (installed via the Install engines card) can
      // be updated in place; their registry id matches the recipe id. Toaster Loop-
      // managed llama.cpp backend builds (llama-cpp-<backend>) update through
      // the runtime setup endpoint instead.
      const catalogById = new Map(catalogRows.map((r) => [r.id, r]));
      const rows = engines.map((e) => {
        const actions = [];
        const llamaBe = !e.builtin && String(e.id).startsWith("llama-cpp-")
          ? String(e.id).slice("llama-cpp-".length) : null;
        const catRow = catalogById.get(e.id);
        if (e.builtin) {
          // The built-in engine is backed by the primary managed llama.cpp
          // checkout — offer the same git pull + rebuild the runtime card's
          // form performs (null backend => primary slot). Delete stays off:
          // the built-in row is immutable and always present.
          actions.push(el("button", {
            class: "btn small" + (runtimeBusy ? "" : " primary"),
            text: runtimeBusy ? "building…" : "update",
            title: "git pull + rebuild the primary llama.cpp checkout " +
              "(auto-setup install)",
            disabled: runtimeBusy || undefined,
            onclick: () => startLlamaBuild(null, true),
          }));
        } else if (llamaBe) {
          actions.push(el("button", {
            class: "btn small" + (runtimeBusy ? "" : " primary"),
            text: runtimeBusy ? "building…" : "update",
            title: `git pull + rebuild the ${llamaBe} llama.cpp checkout`,
            disabled: runtimeBusy || undefined,
            onclick: () => startLlamaBuild(llamaBe, true),
          }));
        } else if (catRow && !e.builtin) {
          const busy = installBusy.has(e.id);
          actions.push(el("button", {
            class: "btn small" + (busy ? "" : " primary"),
            text: busy ? "updating…" : "update",
            title: `Re-run the ${catRow.name} installer (git pull + rebuild / ` +
              "pip upgrade) and refresh this registration",
            disabled: busy || undefined,
            onclick: () => startInstall(e.id, true, catRow.compat || {}),
          }));
        }
        if (!e.builtin) {
          actions.push(el("button", { class: "small danger", text: "delete",
            onclick: async () => {
              const ok = await confirmDialog({
                title: `Delete engine “${e.name}”?`,
                body: llamaBe
                  ? `Removes the engine registration AND deletes the managed ` +
                    `llama.cpp-${llamaBe} checkout under TOASTER_LOOP_HOME. Running ` +
                    "servers keep running until stopped."
                  : "Removes the engine registration only — nothing on " +
                    "disk is deleted and running servers keep running.",
                confirmLabel: "Delete", danger: true });
              if (!ok) return;
              try {
                await api("/api/engines/" + encodeURIComponent(e.id),
                  { method: "DELETE" });
                toast(`engine ${e.name} removed`, "ok");
                pollEngines();
                if (llamaBe) pollRuntime(true); // its install row is gone too
              } catch (err) { toast(err.message, "error"); }
            } }));
        }
        return el("tr", {},
          el("td", {}, el("span", { class: "mono", text: e.name }),
            e.builtin ? badge("built-in", "done") : null),
          el("td", { class: "hint mono" }, e.kind || "openai"),
          el("td", {}, e.installed
            ? badge("installed", "done") : badge("not found", "warn")),
          el("td", { class: "hint mono path" },
            (e.probe_path || e.binary || "–") +
            (defaultEngine === e.id ? "  (default server)" : "")),
          el("td", { class: "hint" },
            e.builtin ? "auto-setup, full run_settings support"
              : ((e.in_use_by || []).length
                ? `in use by ${(e.in_use_by || []).join(", ")}`
                : (e.notes || "—"))),
          el("td", {}, el("div", { class: "btnrow" }, ...actions)));
      });

      // Registration form
      const nameIn = el("input", { class: "input", type: "text",
        placeholder: "vLLM", style: "max-width:14em" });
      const kindSel = el("select", { class: "input" },
        el("option", { value: "openai", text: "openai (spawnable)" }),
        el("option", { value: "diffusion", text: "diffusion (image server)" }),
        el("option", { value: "other", text: "other (inventory only)" }));
      const binIn = el("input", { class: "input mono", type: "text",
        placeholder: "vllm  (path or command on PATH)", style: "max-width:20em" });
      const argsIn = el("input", { class: "input mono", type: "text",
        placeholder: "--model {model} --host 127.0.0.1 --port {port}",
        style: "max-width:34em" });
      const notesIn = el("input", { class: "input", type: "text",
        placeholder: "notes (optional)", style: "max-width:20em" });
      const addBtn = el("button", { class: "btn primary",
        text: "Register engine" });
      addBtn.addEventListener("click", async () => {
        if (!nameIn.value.trim() || !binIn.value.trim()) {
          toast("name and binary are required", "error"); return;
        }
        addBtn.disabled = true;
        try {
          await api("/api/engines", { method: "POST", body: {
            name: nameIn.value.trim(), kind: kindSel.value,
            binary: binIn.value.trim(),
            args_template: argsIn.value.trim() || undefined,
            notes: notesIn.value.trim() || undefined } });
          toast("engine registered", "ok");
          pollEngines();
        } catch (e) { toast(e.message, "error"); }
        finally { addBtn.disabled = false; }
      });

      enginesCard.replaceChildren(
        el("h3", {}, "Inference engines ",
          badge(`${engines.filter((e) => e.installed).length}/${engines.length} installed`,
            engines.every((e) => e.installed) ? "done" : "warn")),
        el("p", { class: "hint", text:
          "Local engines toaster_loop can spawn server instances with " +
          "(placeholders {model} {host} {port}; loopback only). “openai” " +
          "engines speak the chat API; “diffusion” engines are image servers " +
          "(stable-diffusion.cpp's sd-server — health-checked on /, per-" +
          "instance steps/CFG/size instead of sampling). “other” engines are " +
          "recorded for inventory — toaster_loop never spawns them. The default " +
          "server's engine is set on the Settings page." }),
        engines.length ? el("table", {},
          el("thead", {}, el("tr", {},
            el("th", { text: "Engine" }), el("th", { text: "Kind" }),
            el("th", { text: "Status" }), el("th", { text: "Binary" }),
            el("th", { text: "" }), el("th", { text: "" }))),
          el("tbody", {}, ...rows))
          : empty("only the built-in engine is registered"),
        el("h4", { text: "Register an engine" }),
        el("div", { class: "field" }, nameIn, kindSel, binIn),
        el("div", { class: "field" }, argsIn, notesIn, addBtn));
    }

    // ------------------------------------------- installable engines card
    // Curated catalog (vLLM, AirLLM, Strata, stable-diffusion.cpp): toaster_loop
    // clones / pip-installs each engine into a private dir under
    // TOASTER_LOOP_HOME/engines/<id>/ and registers it in the engine registry.
    // Installs run in the background; progress streams via engine.install SSE
    // events + per-recipe polling (same pattern as the llama.cpp card).
    const installBusy = new Set();   // recipe ids with a running install
    let catalogRows = [];
    let hostInfo = null;             // {system, machine, backend, gpus}

    async function pollCatalog() {
      if (!document.body.contains(installCard)) return;
      try { renderCatalog(await api("/api/engine-catalog")); }
      catch (e) {
        installCard.replaceChildren(el("h3", { text: "Install engines" }),
          el("p", { class: "err", text: e.message }));
      }
    }

    async function pollInstall(rid) {
      if (!document.body.contains(installCard)) return;
      try {
        const s = await api("/api/engine-catalog/" + encodeURIComponent(rid) + "/install");
        // Install logs go to the devtools console, not the page.
        echoLog(rid, s.log || []);
        if (s.state === "running") {
          installBusy.add(rid);
          setTimeout(() => pollInstall(rid), 2500);
          return;
        }
        const wasBusy = installBusy.delete(rid);
        if (s.state === "error") console.error(`[${rid}] error: ${s.error}`);
        else if (wasBusy) {
          toast(`${rid} installed — see Inference engines below`, "ok");
          pollEngines();
        }
        if (wasBusy || s.state === "idle") pollCatalog();
      } catch { /* transient */ }
    }

    function startInstall(rid, update, compat) {
      const what = update ? "Update" : "Install";
      const blocked = compat && compat.ok === false;
      let msg = `${what} ${rid}? This downloads from the internet and can ` +
        "take many minutes. Everything stays under TOASTER_LOOP_HOME/engines.";
      if (blocked) msg = `⚠ ${compat.blocked}\n\nInstall anyway? The upstream ` +
        "project will likely fail to build or run on this host.";
      else if (compat?.warnings?.length)
        msg += `\n\nWarnings:\n• ${compat.warnings.join("\n• ")}`;
      if (!confirm(msg)) return;
      installBusy.add(rid);
      console.log(`[${rid}] ${what.toLowerCase()} started…`);
      renderCatalog();
      api("/api/engine-catalog/" + encodeURIComponent(rid) + "/install", {
        method: "POST", body: { update, force: blocked || undefined } })
        .then(() => pollInstall(rid))
        .catch((e) => { installBusy.delete(rid); toast(e.message, "error"); pollCatalog(); });
    }

    function renderCatalog(d) {
      if (d) { catalogRows = d.catalog || []; hostInfo = d.host || hostInfo; }
      const rows = catalogRows.map((r) => {
        const busy = installBusy.has(r.id);
        const compat = r.compat || {};
        // What toaster_loop picked for this host, e.g. "variant rocm" / "backend vulkan"
        const pick = [compat.variant ? `wheel ${compat.variant}` : null,
          compat.backend && compat.backend !== "cpu" ? `backend ${compat.backend}` : null]
          .filter(Boolean).join(" · ");
        const notes = [];
        if (compat.blocked) notes.push(el("div", { class: "err" }, "⛔ " + compat.blocked));
        (compat.warnings || []).forEach((w) =>
          notes.push(el("div", { class: "hint" }, "⚠ " + w)));
        const actions = [el("button", {
          class: "btn small " + (r.installed ? "" : "primary") +
            (compat.ok === false ? " danger" : ""),
          text: busy ? "installing…" : (r.installed ? "Update" : "Install"),
          disabled: busy || undefined,
          onclick: () => startInstall(r.id, !!r.installed, compat),
        })];
        return el("tr", {},
          el("td", {}, el("span", { class: "mono", text: r.name }),
            r.installed ? badge("installed", "done") : null,
            (r.kind || "") === "other" ? badge("inventory", "warn") : null,
            compat.blocked ? badge("blocked here", "error") : null),
          el("td", { class: "hint" },
            r.requires || "—",
            pick ? el("div", { class: "hint mono" }, "auto: " + pick) : null,
            ...notes),
          el("td", { class: "hint mono" },
            r.install_state?.binary || "–"),
          el("td", {}, el("div", { class: "btnrow" }, ...actions)));
      });

      const host = hostInfo ? `${hostInfo.system}/${hostInfo.machine} · ` +
        `backend ${hostInfo.backend} · ` +
        `${(hostInfo.gpus || []).map((g) => g.name).join(", ") || "no GPU"}` : "";
      const body = [
        el("h3", {}, "Install engines ",
          badge(`${catalogRows.filter((r) => r.installed).length}/${catalogRows.length} installed`,
            catalogRows.length && catalogRows.every((r) => r.installed) ? "done" : "warn")),
        el("p", { class: "hint", text:
          "One-click installs for known local engines (vLLM, AirLLM, Strata, " +
          "stable-diffusion.cpp). Each lands in a private directory under " +
          "TOASTER_LOOP_HOME/engines/<id>/ (own venv or git checkout — never the " +
          "toaster_loop environment) and is registered in the engine registry " +
          "below. OpenAI-compatible servers become spawnable; library/CLI " +
          "engines register as inventory-only. Wheel variant and build " +
          "backend are picked from this machine's OS + GPU." }),
        host ? el("p", { class: "hint mono" }, "host: " + host) : null,
        catalogRows.length ? el("table", {},
          el("thead", {}, el("tr", {},
            el("th", { text: "Engine" }), el("th", { text: "Requires / this host" }),
            el("th", { text: "Location" }), el("th", { text: "" }))),
          el("tbody", {}, ...rows))
          : empty("catalog unavailable"),
      ];
      // filter(Boolean): a null conditional line would render as visible
      // "null" text via replaceChildren()'s string coercion. Install logs are
      // echoed to the devtools console (echoLog), not rendered here.
      installCard.replaceChildren(...body.filter(Boolean));
    }

    // A finished/failed install anywhere (this tab or another) refreshes both
    // cards; per-recipe polling covers the live log tail (console only).
    const offInstall = bus.on("engine.install", (p) => {
      if (p?.recipe && p.state !== "running") pollInstall(p.recipe);
    });

    // bootstrap: any install still running from a previous visit resumes its
    // log polling without hammering the server up front.
    api("/api/engine-catalog").then((d) => {
      catalogRows = d.catalog || [];
      renderCatalog(d);
      pollEngines(); // update buttons need the catalog map to exist first
      catalogRows.forEach((r) => {
        if (r.installing) { installBusy.add(r.id); pollInstall(r.id); }
      });
    }).catch(() => {});

    // Idempotent: make sure every managed llama.cpp backend install has its
    // engine row (covers builds that predate per-backend registration).
    api("/api/runtime/sync", { method: "POST", body: {} })
      .then((r) => { if ((r.synced || []).length) pollEngines(); })
      .catch(() => {});

    poll(); pollGate(); pollRuntime(true); pollEngines(); pollCatalog();
    const offEngines = bus.on("engines.changed", () => pollEngines());
    timer = setInterval(() => { poll(); pollGate(); if (!runtimeBusy) pollRuntime(false); }, POLL_MS);
    return { destroy() {
      clearInterval(timer);
      if (setupTimer) clearTimeout(setupTimer);
      offEngines(); offInstall();
    } };
  },
};

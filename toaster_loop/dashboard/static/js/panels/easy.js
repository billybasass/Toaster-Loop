// panels/easy.js — Easy training (turnkey loop for one or more agent profiles).
// Pick profiles that are already bound to server instances, enable easy
// training, and the page handles the rest per agent: chat with a profile feeds
// its slice of the buffer (traces are stamped with the profile name), each
// agent's readiness gauge says when training is due, and "Train now" runs the
// full iterate loop — curate → register → train → export → validate → promote —
// scoped to that agent, with the optimal preset written into curation:/
// training: on enable. With auto-deploy on, each promoted model is hot-swapped
// onto its own profile's server. Agents share one config loop
// (easy_training.profile = primary, .profiles = full list); "add another
// agent" / "remove" patch just that list.
// GET /api/easy drives everything; live updates come from iterate.step and
// easy_training.* bus events.

import { api } from "../api.js";
import { bus } from "../bus.js";
import { el, toast, badge, empty, fmtTime, svgGauge, confirmDialog } from "../ui.js";

export default {
  title: "Easy",
  render(mount) {
    const setupCard = el("div", { class: "card" });
    const statusHost = el("div");          // one card per agent in the loop
    const addCard = el("div", { class: "card hidden" });
    const runHost = el("div");
    const historyHost = el("div");

    let data = null;          // last GET /api/easy payload
    let profiles = [];        // for the pickers
    let servers = [];         // id -> {name, model_name}
    let activeId = null;      // run we follow live (or the finished one shown)
    let activeRun = null;     // merged live record
    let editing = false;      // setup card open in "update" mode while enabled
    let extrasPending = null; // setup-card extra-agent picks (survive re-render)
    let primaryPending = null; // setup-card primary-profile pick (survives the
                            // re-render triggered by changing the select —
                            // otherwise it snaps back to the saved profile)

    mount.replaceChildren(
      el("h2", { text: "Easy training" }),
      el("p", { class: "hint", text:
        "Turn agent profiles into self-improving loops: chat with them, and " +
        "when enough good traces pile up per agent, train on them with optimal " +
        "settings — curate, train, validate and promote in one click per " +
        "agent. Each profile must already be bound to a server (Profiles " +
        "panel), which decides the base model and where that agent's improved " +
        "model goes back to. Only dashboard chat counts toward the loop — CLI " +
        "`toaster_loop run` sessions keep their backend stamp and are not scoped " +
        "in." }),
      setupCard, statusHost, addCard, runHost,
      el("h3", { text: "Runs" }), historyHost);

    // ------------------------------------------------------------------ data
    async function load() {
      try { data = await api("/api/easy"); }
      catch (e) {
        setupCard.replaceChildren(el("h3", { text: "Easy training" }),
          el("p", { class: "err", text: e.message }));
        return;
      }
      try {
        const p = await api("/api/profiles");
        profiles = p.profiles || [];
      } catch { profiles = []; }
      try {
        const s = await api("/api/servers");
        servers = s.servers || [];
      } catch { servers = []; }
      // Adopt a live run whenever we are not already following one — including
      // runs started by cron (train auto-check), which the panel used to miss.
      const live = agents().find((a) => a.active_run);
      if (live && (!activeId || !activeRun || activeRun.state !== "running")) {
        activeId = live.active_run.run_id;
        activeRun = live.active_run;
      }
      render();
    }

    // Per-agent views from the server; tolerate legacy payloads that only
    // carry the top-level single-profile fields.
    function agents() {
      if (data.agents && data.agents.length) return data.agents;
      if (!data.config?.profile) return [];
      return [{ profile: data.config.profile, bound: !!data.profile || !data.problem,
                server: data.server, problem: data.problem,
                readiness: data.readiness, active_run: data.active_run,
                runs: data.runs || [] }];
    }

    function allEasyRuns() {
      const seen = new Set();
      const out = [];
      for (const a of agents()) {
        for (const r of (a.runs || [])) {
          if (!seen.has(r.run_id)) { seen.add(r.run_id); out.push(r); }
        }
      }
      out.sort((x, y) => String(y.run_id).localeCompare(String(x.run_id)));
      return out;
    }

    function serverLabel(id) {
      const s = servers.find((x) => x.id === id);
      if (!s) return id || "—";
      return `${s.name}${s.model_name ? " · " + s.model_name : ""}` +
        (s.running ? "" : " (down)");
    }

    function render() {
      const cfg = data.config || {};
      if (cfg.enabled) {
        // "edit…" re-opens the setup card on top of the status cards so the
        // profile / auto-deploy knobs are reachable without disabling first.
        setupCard.classList.toggle("hidden", !editing);
        if (editing) renderSetup();
        renderStatus();
        renderAddCard();
      } else {
        editing = false;
        setupCard.classList.remove("hidden");
        statusHost.replaceChildren();
        addCard.classList.add("hidden");
        renderSetup();
      }
      renderActive();
      renderHistory();
    }

    // ------------------------------------------------------------- setup card
    function presetDiffNode(diff) {
      if (!diff || !diff.length) return null;
      return el("div", {},
        el("p", { class: "hint", text:
          "The optimal preset will change these settings:" }),
        el("ul", { class: "hint" }, ...diff.map((d) => el("li", {
          class: "mono",
          text: `${d.section}.${d.key}: ${JSON.stringify(d.from)} → ` +
            `${JSON.stringify(d.to)}` }))));
    }

    function renderSetup() {
      const cfg = data.config || {};
      // Primary picker (legacy `profile` key) + checkbox list of additional
      // agents. The primary can also be checked "again" — harmless, the
      // backend dedupes and keeps it first.
      const sel = el("select", {},
        el("option", { value: "", text: "— pick a profile —" }),
        ...profiles.map((p) => el("option", {
          value: p.name,
          text: `${p.name} — ${p.server_instance ? serverLabel(p.server_instance)
            : "no server bound (open Profiles)"}` })));
      sel.value = primaryPending != null ? primaryPending : (cfg.profile || "");
      // Pending primary + extras survive the re-render triggered by changing
      // the primary select; seeded once from the saved config.
      if (extrasPending === null) {
        const saved = (cfg.profiles && cfg.profiles.length)
          ? cfg.profiles : (cfg.profile ? [cfg.profile] : []);
        extrasPending = new Set(saved.filter((n) => n !== cfg.profile));
      }
      const extraChecks = new Map();
      const extraBox = el("div", { class: "field" },
        el("label", { text: "also train these agents" }),
        ...profiles
          .filter((p) => p.name !== sel.value)
          .map((p) => {
            const c = el("input", { type: "checkbox" });
            c.checked = extrasPending.has(p.name);
            c.addEventListener("change", () => {
              if (c.checked) extrasPending.add(p.name);
              else extrasPending.delete(p.name);
            });
            extraChecks.set(p.name, c);
            return el("label", { class: "chk" }, c, " " + p.name);
          }));
      sel.addEventListener("change", () => {
        primaryPending = sel.value;
        renderSetup();
      });
      const autoDeploy = el("input", { type: "checkbox" });
      autoDeploy.checked = !!cfg.auto_deploy;
      const applyPreset = el("input", { type: "checkbox" });
      applyPreset.checked = true;
      const btn = el("button", { class: "primary",
        text: cfg.enabled ? "Update easy training" : "Enable easy training" });
      if (profiles.length === 0) {
        setupCard.replaceChildren(
          el("h3", { text: "Get started" }),
          empty(el("span", {}, "No agent profiles yet — create one in the "),
            el("a", { href: "#/profiles", text: "Profiles panel" }),
            el("span", {}, " and bind it to a server first.")));
        return;
      }
      btn.addEventListener("click", async () => {
        if (!sel.value) return toast("pick a profile first", "error");
        const extras = [...extraChecks.entries()]
          .filter(([, c]) => c.checked)
          .map(([n]) => n)
          .filter((n) => n !== sel.value);
        btn.disabled = true;
        try {
          // Preview first: show exactly which curation/training knobs the
          // optimal preset overwrites before the user consents.
          let diff = [];
          if (applyPreset.checked) {
            const pv = await api("/api/easy/enable", { method: "POST", body: {
              profile: sel.value, preview: true } });
            diff = pv.diff || [];
            if (diff.length) {
              const ok = await confirmDialog({
                title: cfg.enabled ? "Update — re-apply optimal preset?"
                  : "Enable with optimal preset?",
                body: el("div", {},
                  el("p", { text:
                    "Enabling writes the optimal preset into your curation " +
                    "and training settings. Values you changed yourself get " +
                    "overwritten:" }),
                  presetDiffNode(diff)),
                confirmLabel: "Apply & enable", danger: false });
              if (!ok) { btn.disabled = false; return; }
            }
          }
          const r = await api("/api/easy/enable", { method: "POST", body: {
            profile: sel.value, profiles: extras,
            auto_deploy: autoDeploy.checked,
            apply_preset: applyPreset.checked } });
          toast(r.note || "easy training enabled", "ok");
          editing = false;
          extrasPending = null;  // re-seed from the saved config next time
          primaryPending = null;
          await load();
        } catch (e) { toast(e.message, "error"); btn.disabled = false; }
      });
      const kids = [
        el("h3", {}, cfg.enabled ? "Update settings " : "Enable ",
          badge("optimal preset", "done")),
        data.problem
          ? el("p", { class: "err", text: data.problem }) : null,
        el("p", { class: "hint", text:
          "Enabling writes the optimal preset (SFT + DPO curation with " +
          "failures as preference negatives, eval split, regression gate, " +
          "shadow-baseline comparison before promotion, golden mining from 👍 " +
          "chats) into your curation/training/validation/goldens_mine settings " +
          "— still editable in Settings." }),
        el("div", { class: "field" },
          el("label", { text: "primary profile" }), sel),
        extraBox,
        el("div", { class: "field" },
          el("label", { class: "chk" }, autoDeploy,
            " auto-deploy each promoted model onto its own profile's server"),
          el("span", { class: "hint", text:
            "(spawned instances are restarted on the new GGUF; attached " +
            "servers are never touched)" })),
        el("div", { class: "field" },
          el("label", { class: "chk" }, applyPreset,
            " apply the optimal preset (uncheck to only switch profile/" +
            "auto-deploy and leave your curation & training settings alone)")),
        el("div", { class: "btnrow" }, btn)];
      if (cfg.enabled) {
        kids[0] = el("div", { class: "btnrow" }, btn,
          el("button", { class: "small", text: "Cancel",
            onclick: () => {
              editing = false;
              extrasPending = null;  // discard unsaved picks
              primaryPending = null;
              render();
            } }));
      }
      // replaceChildren() stringifies non-Node args (a `null` conditional
      // line would render as visible "null" text) — filter them out.
      setupCard.replaceChildren(...kids.filter(Boolean));
    }

    // ------------------------------------------------------------ status card
    function agentStatusCard(a) {
      const cfg = data.config || {};
      const r = a.readiness || {};
      const th = r.threshold || {};
      const unclaimed = r.unclaimed ?? 0;
      const minTraces = th.min_traces ?? 50;
      const mean = r.mean_score;
      const scoreOk = mean != null && mean >= (th.min_score ?? 0);
      const gate = r.gate || { ok: true, reasons: [] };
      const ready = !!r.needs_training;

      const bits = [
        svgGauge(unclaimed, Math.max(minTraces, unclaimed), {
          label: `profile traces (need ${minTraces})`,
          threshold: minTraces, danger: !r.trigger }),
        svgGauge(mean == null ? 0 : Math.round(mean * 100) / 100, 1, {
          label: `mean score (need ≥ ${th.min_score})`, unit: "",
          threshold: th.min_score, danger: !scoreOk }),
      ];

      const nextSteps = [];
      if (!r.trigger) {
        if (unclaimed < minTraces) {
          nextSteps.push(`chat ${Math.max(1, minTraces - unclaimed)} more times ` +
            `with profile “${a.profile}” in Chat (each good run counts; CLI ` +
            "runs don't count toward this loop)");
        }
        if (mean == null) nextSteps.push("rate a few chats so scores exist");
        else if (!scoreOk) nextSteps.push(`raise the mean score above ${th.min_score}`);
      }
      for (const g of (gate.reasons || [])) nextSteps.push(g);

      const trainBtn = el("button", { class: "primary", text: "Train now" });
      trainBtn.addEventListener("click", async () => {
        trainBtn.disabled = true;
        try {
          const res = await api("/api/easy/run", { method: "POST",
            body: { profile: a.profile } });
          toast(`easy run ${res.run_id} started for “${a.profile}”`, "ok");
          activeId = res.run_id;
          activeRun = { run_id: res.run_id, state: "running",
            name: `easy:${a.profile}`, steps: [] };
          renderActive();
        } catch (e) { toast(e.message, "error"); }
        finally { trainBtn.disabled = false; load(); }
      });
      const removeBtn = el("button", { class: "small danger", text: "remove" });
      removeBtn.addEventListener("click", async () => {
        const ok = await confirmDialog({
          title: `Remove “${a.profile}” from the easy loop?`,
          body: "Its traces stay in the buffer (already-trained models too); " +
            "the agent just stops collecting toward training and auto-deploy.",
          confirmLabel: "Remove", danger: true });
        if (!ok) return;
        try {
          await api("/api/easy/remove", { method: "POST",
            body: { profile: a.profile } });
          toast(`“${a.profile}” removed from the easy loop`, "ok");
          await load();
        } catch (e) { toast(e.message, "error"); }
      });

      return el("div", { class: "card" },
        el("h3", {}, `Easy training — ${a.profile} `,
          a.profile === cfg.profile ? badge("primary", "queued") : null,
          badge(ready ? "training due" : "collecting traces",
            ready ? "done" : "queued")),
        a.problem
          ? el("p", { class: "err", text: a.problem }) : null,
        el("p", { class: "hint" },
          "server: ", a.server ? serverLabel(a.server.id) : "—",
          " · auto-deploy: ", cfg.auto_deploy ? "on" : "off"),
        el("div", { class: "gauges" }, ...bits),
        ready
          ? el("p", {}, el("strong", { text:
            `Ready — ${unclaimed} good unclaimed traces. Train now, or let ` +
            "the cron entrypoint (train auto-check --trigger) do it." }))
          : el("ul", { class: "hint" },
            ...nextSteps.map((s) => el("li", { text: s }))),
        el("div", { class: "btnrow" }, trainBtn, removeBtn));
    }

    function renderStatus() {
      // Traces stamped with a profile name that no longer exists (renamed
      // profile) — they silently stopped counting; tell the user once.
      const loopNames = new Set(agents().map((a) => a.profile));
      const stale = (data.stale_stamps || []).filter(
        (s) => !loopNames.has(s.profile));
      const disableBtn = el("button", { class: "small danger",
        text: "Disable" });
      disableBtn.addEventListener("click", async () => {
        const ok = await confirmDialog({
          title: "Disable easy training?",
          body: "The optimal preset stays in your settings (nothing is deleted); " +
            "traces stop being scoped to the profiles. You can re-enable anytime.",
          confirmLabel: "Disable", danger: true });
        if (!ok) return;
        try {
          await api("/api/easy/disable", { method: "POST", body: {} });
          toast("easy training disabled", "ok");
          editing = false;
          await load();
        } catch (e) { toast(e.message, "error"); }
      });
      statusHost.replaceChildren(
        ...stale.map((s) => el("p", { class: "err", text:
          `${s.traces} unclaimed trace(s) are stamped for profile ` +
          `“${s.profile}”, which no longer exists — if you renamed it, those ` +
          "old chats won't count toward this loop." })),
        el("div", { class: "btnrow" },
          el("button", { class: "small", text: "re-check",
            onclick: () => load() }),
          el("button", { class: "small", text: "edit…",
            onclick: () => { editing = true; render(); } }),
          disableBtn),
        ...agents().map(agentStatusCard));
    }

    // ------------------------------------------------------- add-an-agent card
    function renderAddCard() {
      const inLoop = new Set(agents().map((a) => a.profile));
      const candidates = profiles.filter((p) => !inLoop.has(p.name));
      if (!candidates.length) { addCard.classList.add("hidden"); return; }
      addCard.classList.remove("hidden");
      const sel = el("select", {},
        ...candidates.map((p) => el("option", {
          value: p.name,
          text: `${p.name} — ${p.server_instance ? serverLabel(p.server_instance)
            : "no server bound (open Profiles)"}` })));
      const btn = el("button", { class: "small", text: "Add agent" });
      btn.addEventListener("click", async () => {
        if (!sel.value) return;
        btn.disabled = true;
        try {
          const r = await api("/api/easy/add", { method: "POST",
            body: { profile: sel.value } });
          toast(r.note || `“${sel.value}” added`, "ok");
          await load();
        } catch (e) { toast(e.message, "error"); btn.disabled = false; }
      });
      addCard.replaceChildren(
        el("h3", { text: "Add another agent to the loop" }),
        el("p", { class: "hint", text:
          "Each agent keeps its own readiness gauge, its own scoped training " +
          "runs, and its own auto-deploy target. The optimal preset is shared " +
          "and stays as-is." }),
        el("div", { class: "field" }, sel),
        el("div", { class: "btnrow" }, btn));
    }

    // ----------------------------------------------------------- active run
    function renderActive() {
      if (!activeRun) { runHost.replaceChildren(); return; }
      const byName = new Map();
      for (const s of activeRun.steps || []) byName.set(s.name, s);
      const rows = el("div", { class: "timeline" });
      for (const [name, s] of byName) {
        rows.appendChild(el("div", { class: "step " + (s.state || "") },
          badge(s.state === "running" ? "running" : s.state),
          el("span", { class: "mono step-name", text: name }),
          el("span", { class: "hint step-detail", text: s.detail || "" })));
      }
      const res = activeRun.result || {};
      const bits = [];
      if (res.model_id) bits.push(el("a", { href: "#/models", class: "mono",
        text: `model ${res.model_id}` }));
      if (activeRun.dataset || res.dataset_id) bits.push(el("a", {
        href: "#/datasets/" + encodeURIComponent(res.dataset_id || activeRun.dataset),
        class: "mono", text: "dataset" }));
      const deploy = data.state?.last_run?.deploy;
      const showDeploy = activeRun.state !== "running" && deploy &&
        data.state?.last_run?.run_id === activeRun.run_id;
      runHost.replaceChildren(
        el("h3", {}, activeRun.state === "running" ? "Active run " : "Last run ",
          (activeRun.name || "").startsWith("easy:")
            ? el("span", { class: "mono hint", text: activeRun.name + " " }) : null,
          badge(activeRun.state)),
        rows,
        res.error ? el("p", { class: "err", text: String(res.error) }) : null,
        showDeploy
          ? el("p", { class: deploy.ok ? "hint" : "err", text:
            `auto-deploy: ${deploy.detail || ""}` })
          : null,
        bits.length ? el("div", { class: "field" }, ...bits) : null);
    }

    // -------------------------------------------------------------- history
    function renderHistory() {
      const runs = allEasyRuns().filter((x) => x.run_id !== activeId);
      if (!runs.length) {
        historyHost.replaceChildren(empty("No easy-training runs yet."));
        return;
      }
      const lastRun = data.state?.last_run || {};
      historyHost.replaceChildren(el("table", {},
        el("thead", {}, el("tr", {},
          el("th", { text: "Run" }), el("th", { text: "Agent" }),
          el("th", { text: "State" }),
          el("th", { text: "Model" }), el("th", { text: "Promoted" }),
          el("th", { text: "Deploy" }),
          el("th", { text: "Finished" }))),
        el("tbody", {}, ...runs.map((x) => {
          const dep = lastRun.run_id === x.run_id ? lastRun.deploy : null;
          const who = String(x.name || "").startsWith("easy:")
            ? x.name.slice("easy:".length) : "–";
          return el("tr", {},
            el("td", { class: "mono hint" }, x.run_id),
            el("td", { class: "mono hint" }, who),
            el("td", {}, badge(x.state)),
            el("td", { class: "mono hint" }, x.result?.model_id || "–"),
            el("td", {}, x.result?.promoted ? "★" : ""),
            el("td", { class: "hint", title: dep?.detail || "" },
              dep ? (dep.ok ? "✓" : "✗") : ""),
            el("td", { class: "hint" }, fmtTime(x.finished, false)));
        }))));
    }

    // --------------------------------------------------------- live updates
    const onStep = (p) => {
      if (!activeId || p.run_id !== activeId) return;
      activeRun = activeRun || { run_id: activeId, steps: [], state: "running" };
      if (p.step === "finished") {
        activeRun.state = p.state;
        load(); // refresh readiness/history/deploy info
      } else {
        (activeRun.steps = activeRun.steps || []).push(
          { name: p.step, state: p.state, detail: p.detail });
      }
      renderActive();
    };
    const onFinished = (p) => {
      if (p.state === "done") {
        toast(p.promoted
          ? `★ easy training promoted ${p.model_id || "a model"}` +
            (p.profile ? ` for “${p.profile}”` : "")
          : "easy training run finished", "ok");
      } else if (p.state) toast(`easy training run ${p.state}`, "error");
      load();
    };
    const un1 = bus.on("iterate.step", onStep);
    const un2 = bus.on("easy_training.run.finished", onFinished);

    load();
    // Always poll: keeps the readiness gauges fresh, picks up cron-started
    // runs, and recovers after a run finishes (used to stop when activeId
    // was set and never cleared).
    const timer = setInterval(() => load(), 30000);
    return { destroy() { clearInterval(timer); try { un1(); un2(); } catch {} } };
  },
};

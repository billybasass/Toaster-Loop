// panels/overview.js — status summary + deep-link quick actions (Phase 5e).
// No duplicated logic: Train/Iterate/Datasets actions navigate to their own
// panels; only Curate (a single queued job with global job.state toasts) runs
// inline here. Empty states point at the next step in the loop.

import { api } from "../api.js";
import { bus } from "../bus.js";
import { el, toast, fmtNum, badge } from "../ui.js";

export default {
  title: "Overview",
  render(mount) {
    const cards = el("div", { class: "cards" });
    const loopCard = el("div", { class: "card" });
    const insightsCard = el("div", { class: "card" });
    const ledgerCard = el("div", { class: "card" }); // pass4 #31 lineage
    const plansCard = el("div", { class: "card" });  // WP12 per-project plans
    const unsubs = [];

    function card(title, value, href = null) {
      const v = el("div", { class: "big" }, value);
      const c = el("div", { class: "card" + (href ? " clickable" : "") },
        el("h3", { text: title }), v);
      if (href) {
        c.tabIndex = 0;
        const go = () => { location.hash = href; };
        c.addEventListener("click", go);
        c.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
      }
      return c;
    }

    async function load() {
      const s = await api("/api/status");
      const srv = s.server || {};
      const st = srv.state || {};
      const total = s.buffer?.total_traces ?? 0;
      renderLedger(s.recent_ops || []);
      cards.replaceChildren(
        card("Server", srv.running
          ? el("span", {}, badge("up", "done"), " ",
              el("span", { class: "mono" }, st.base_url || ""),
              st.model_name ? " · " + st.model_name : "")
          : badge("down", "failed")),
        card("Active model", s.active_model || "none", "#/models"),
        // WP2c metric honesty: the avg is computed over SCORED traces only —
        // show the raw denominator so a mean over a few ratings can't read
        // as a whole-buffer quality signal.
        card("Buffer", total
          ? `${total} traces · avg ${fmtNum(s.buffer?.avg_outcome_score)} ` +
            `(${s.buffer?.scored_traces ?? 0}/${total} scored) · ` +
            `${s.buffer?.curated_traces ?? 0} curated`
          : el("span", { class: "hint" },
              el("a", { href: "#/chat", text: "No traces yet" }),
              " — start chatting to feed the loop"),
          total ? "#/traces" : "#/chat"),
        card("Jobs", `${s.jobs?.running ?? 0} running · ${s.jobs?.queued ?? 0} queued · ` +
          `${s.jobs?.total ?? 0} total`, "#/jobs"));

      // Latest iterate run (compact; full timeline lives in the Iterate panel).
      let lastIter = null;
      try {
        const runs = (await api("/api/iterate/runs")).runs || [];
        lastIter = runs[0] || null;
      } catch { /* fresh home */ }

      const loopBtns = el("div", { class: "btnrow" });

      // Curate stays inline: one queued job, global toasts report the result.
      const btnCurate = el("button", { text: "Curate dataset",
        title: "Queue curation of unclaimed traces as a job" });
      btnCurate.addEventListener("click", async () => {
        btnCurate.disabled = true;
        try {
          const r = await api("/api/curate", { method: "POST", body: { name: "ui" } });
          toast(`curate queued (${r.job_id}) — progress in Jobs`, "ok");
        } catch (e) { toast(`${e.code}: ${e.message}`, "error"); }
        finally { btnCurate.disabled = false; }
      });

      // Deep links — the panels own their forms and validation.
      const linkBtn = (text, href, cls = "", title = "") =>
        el("button", { class: cls, text, title, onclick: () => { location.hash = href; } });

      loopBtns.replaceChildren(
        btnCurate,
        linkBtn("Train…", "#/train", "", "choose a dataset and submit a job"),
        linkBtn("Iterate (full loop)…", "#/iterate", "primary",
          "curate → train → export → validate → promote"),
        linkBtn("Schedule…", "#/schedule", "", "auto-iterate thresholds + cron"));

      loopCard.replaceChildren(
        el("h3", { text: "Loop controls" }), loopBtns,
        lastIter
          ? el("p", { class: "hint" },
              "last iterate: ",
              el("a", { href: "#/iterate", class: "mono", text: lastIter.run_id }),
              ` ${lastIter.state} (${(lastIter.started || "").slice(0, 16)})`)
          : el("p", { class: "hint" },
              "no iterate runs yet — the Iterate panel closes the whole loop in one call."));
    }

    // Insights card (backlog #11): 30d score trend arrow, traces/day avg,
    // top failure reasons. Compact summary only — CLI/JSON carry the detail.
    async function loadInsights() {
      let ins;
      try {
        ins = await api("/api/insights?days=30");
      } catch {
        return; // fresh home / older server: leave the card out
      }
      const arrow = { up: "▲", down: "▼" }[ins.score_trend?.overall?.direction] || "→";
      const days = ins.activity?.traces_per_day || [];
      const perDay = days.length
        ? (days.reduce((a, d) => a + d.traces, 0) / days.length).toFixed(1)
        : "0";
      const reasons = (ins.failure_reasons?.by_reason || []).slice(0, 3);
      // Pass 4 #36: recorded intents vs observed delta (honest-empty — the
      // server never fabricates rows; empty means nobody recorded an intent).
      const ir = ins.intent_review || {};
      const irRows = (ir.rows || []).slice(0, 3);
      const intentKids = !ir.available
        ? []
        : irRows.length
          ? irRows.map((r) => el("div", { class: "hint" },
              el("span", { class: "mono" }, `${r.op} ${r.target || ""} `),
              `${fmtNum(r.delta)} — "${r.intent}"`))
          : [el("p", { class: "hint" },
              ir.hint || "no recorded intents")];
      // Pass 4 #40: confidence calibration from logprobs (same honest-empty
      // contract — the server never fabricates buckets; empty means no trace
      // has confidence data yet, e.g. `sampling.logprobs` was never on).
      const cal = ins.calibration || {};
      const calRows = (cal.rows || []).filter((r) => r.n > 0);
      const calKids = !cal.available
        ? []
        : calRows.length
          ? calRows.map((r) => el("div", { class: "hint" },
              el("span", { class: "mono" }, `${r.bucket_label} `),
              `n=${r.n} · conf ${fmtNum(r.mean_conf)} · score ${fmtNum(r.mean_score)}`))
              .concat(cal.confidently_wrong
                ? [el("p", { class: "hint" },
                    "⚠ confidently wrong: top-confidence bucket scores worse")]
                : [])
          : [el("p", { class: "hint" },
              cal.hint || "no traces with confidence data yet")];
      insightsCard.replaceChildren(
        el("h3", { text: `Insights (30d) ${arrow}` }),
        el("p", {},
          `${ins.overview?.traces ?? 0} traces · ` +
          `avg ${fmtNum(ins.overview?.avg_score)} · ${perDay}/day`),
        reasons.length
          ? el("p", { class: "hint" },
              "top failures: ",
              // Pass 4 #33: remediation status suffix when a note row exists
              // (implicit-open reasons render exactly as before).
              reasons.map((r) => `${r.reason} (${r.count})` +
                (r.explicit && r.status !== "open" ? `[${r.status}]` : "")
              ).join(", "))
          : el("p", { class: "hint" }, "no coded failures"),
        ...(intentKids.length
          ? [el("h3", { text: "Intents vs observed" }), ...intentKids]
          : []),
        ...(calKids.length
          ? [el("h3", { text: "Confidence calibration" }), ...calKids]
          : []));
    }

    // WP12 plans: small per-project card (tasks done/total + loop state).
    // Hidden entirely when config plans.enabled is off (API answers 404
    // plans_disabled) — same "off means invisible" contract everywhere else.
    async function loadPlans() {
      let projs = [];
      try { projs = (await api("/api/projects")).projects || []; }
      catch { /* fresh home */ }
      const keys = [...projs.map((p) => [p.id, p.name]),
                    ["default", "Unfiled"]];
      const rows = [];
      let disabled = false;
      for (const [id, name] of keys) {
        let d;
        try {
          d = await api("/api/plans/" + encodeURIComponent(id));
        } catch (e) {
          if (e.code === "plans_disabled") disabled = true;
          continue;
        }
        const c = d.counts || {};
        if (!c.total) continue;
        const looping = d.loop && d.loop.running;
        rows.push(el("div", { class: "hint" },
          el("a", { href: "#/chat", class: "mono", text: name }),
          ` ${c.done}/${c.total} done` +
          (c.blocked ? ` · ${c.blocked} blocked` : "") +
          (looping ? " · ↻ loop running" : "")));
      }
      if (disabled) { plansCard.classList.add("hidden"); return; }
      plansCard.classList.remove("hidden");
      plansCard.replaceChildren(
        el("h3", { text: "Project plans" }),
        ...(rows.length ? rows : [el("p", { class: "hint" },
          "no plan tasks yet — open a project's ▤ drawer on the Chat page " +
          "to add tasks or import a brief.")]));
    }

    // Operation ledger (pass4 #31): last ~10 hash-chained registry mutations.
    function renderLedger(ops) {
      const rows = ops.map((o) => el("div", { class: "hint" },
        el("span", { class: "mono" }, `${(o.ts || "").slice(5, 16)} `),
        `${o.actor} · ${o.op} `,
        el("span", { class: "mono" }, o.target_id || "")));
      ledgerCard.replaceChildren(
        el("h3", { text: "Recent operations" }),
        ...(rows.length ? rows : [el("p", { class: "hint" },
          "no registry operations yet — promotes, archives and job " +
          "transitions land here (hash-chained; see `toaster_loop history`).")]));
    }

    mount.replaceChildren(el("h2", { text: "Overview" }), cards, loopCard,
      plansCard, insightsCard, ledgerCard);

    load().catch((e) => toast(e.message, "error"));
    loadInsights();
    loadPlans();
    const timer = setInterval(() => load().catch(() => {}), 8000);
    unsubs.push(bus.on("job.state", () => load().catch(() => {})));
    unsubs.push(bus.on("serve.state", () => load().catch(() => {})));
    unsubs.push(bus.on("plan.updated", () => loadPlans()));
    unsubs.push(bus.on("plan.loop", () => loadPlans()));

    return { destroy() { clearInterval(timer); for (const u of unsubs) u(); } };
  },
};

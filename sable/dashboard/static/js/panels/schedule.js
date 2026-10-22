// panels/schedule.js — auto-iterate rules (Phase 5e).
// The threshold rule mirrors config `training.auto_*` (enable toggle +
// min_traces/min_score via section patch); readiness is computed live from
// GET /api/train/check (gate reasons are the source of truth — we never
// re-read thresholds client-side). Autonomous triggering itself runs from an
// external scheduler calling `sable train auto-check --trigger`; this panel
// shows the exact cron line and offers a trigger-now shortcut.

import { api } from "../api.js";
import { bus } from "../bus.js";
import { el, toast, badge, copyButton, fmtNum, fmtTime } from "../ui.js";

export default {
  title: "Schedule",
  render(mount) {
    const rulesCard = el("div", { class: "card" });
    const readyCard = el("div", { class: "card" });
    const cronCard = el("div", { class: "card" });
    const historyHost = el("div");

    // ------------------------------------------------------------ rules card
    async function loadRules() {
      let tr = {};
      try { tr = (await api("/api/config")).config?.training || {}; }
      catch (e) { rulesCard.replaceChildren(el("p", { class: "err", text: e.message })); return; }

      const enSel = el("select", {},
        el("option", { value: "true", text: "enabled" }),
        el("option", { value: "false", text: "disabled" }));
      enSel.value = String(!!tr.auto_enabled);
      const tracesIn = el("input", { type: "number", min: "1",
        value: tr.auto_min_traces ?? 50 });
      const scoreIn = el("input", { type: "number", step: "0.05", min: "-1", max: "1",
        value: tr.auto_min_score ?? 0.5 });
      const status = el("span", { class: "hint" });

      const saveBtn = el("button", { class: "small primary", text: "Apply" });
      saveBtn.addEventListener("click", async () => {
        saveBtn.disabled = true;
        try {
          await api("/api/config", { method: "PUT", body: { section: "training",
            patch: {
              auto_enabled: enSel.value === "true",
              auto_min_traces: Number(tracesIn.value),
              auto_min_score: Number(scoreIn.value),
            } } });
          toast("schedule rule saved", "ok");
          loadReadiness();
        } catch (e) {
          status.textContent = e.message;
          toast(`${e.code}: ${e.message}`, "error");
        } finally { saveBtn.disabled = false; }
      });

      const field = (label, input, hint) => el("div", { class: "field" },
        el("label", { text: label }), input,
        hint ? el("span", { class: "hint", text: hint }) : null);

      rulesCard.replaceChildren(
        el("h3", {}, "Threshold rule ", badge("threshold", "")),
        el("p", { class: "hint", text:
          "Retrain when the buffer holds at least min_traces unclaimed traces " +
          "whose mean outcome score reaches min_score. These are the same " +
          "training.auto_* keys Settings edits — one source of truth." }),
        field("enabled", enSel,
          "marks intent for the external scheduler; nothing fires inside the dashboard process"),
        field("min_traces", tracesIn),
        field("min_score", scoreIn),
        el("div", { class: "field" }, saveBtn, status));
    }

    // -------------------------------------------------------- readiness card
    const triggerBtn = el("button", { class: "primary", text: "Trigger now" });

    async function loadReadiness() {
      let c;
      try { c = await api("/api/train/check"); }
      catch (e) {
        readyCard.replaceChildren(el("h3", { text: "Readiness" }),
          el("p", { class: "err", text: e.message }));
        return;
      }
      const gateOk = c.gate?.ok !== false;
      const ready = !!c.enough_traces && gateOk;
      const reasons = [];
      if (!c.enough_traces) {
        reasons.push(`unclaimed traces ${c.unclaimed} < required ` +
          `${c.threshold?.min_traces}`);
        if (c.mean_score == null) reasons.push("no scored traces yet");
        else if (Number(c.mean_score) < Number(c.threshold?.min_score ?? 0))
          reasons.push(`mean score ${fmtNum(c.mean_score)} < ` +
            `${fmtNum(c.threshold?.min_score)}`);
      }
      for (const r of (c.gate?.reasons || [])) reasons.push(r);

      readyCard.replaceChildren(
        el("h3", {}, "Readiness ", badge(ready ? "ready" : "not ready",
          ready ? "done" : "failed")),
        el("p", { class: "hint mono", text:
          `unclaimed ${c.unclaimed} · mean score ${fmtNum(c.mean_score)} · ` +
          `thresholds ≥${c.threshold?.min_traces} traces, ≥${fmtNum(c.threshold?.min_score, 2)}` }),
        reasons.length
          ? el("ul", { class: "hint" }, ...reasons.map((r) => el("li", { text: r })))
          : el("p", { class: "hint", text: "All threshold and resource gates pass." }),
        el("div", { class: "btnrow" }, triggerBtn,
          el("span", { class: "hint", text:
            "checks the gate again, then starts a full iterate run (or sends " +
            "you to Train for a single dataset)" })));

      triggerBtn.onclick = async () => {
        triggerBtn.disabled = true;
        try {
          const chk = await api("/api/train/check");
          if (!chk.enough_traces || chk.gate?.ok === false) {
            toast("not ready — see the reasons above", "error");
            return;
          }
          const r = await api("/api/iterate", { method: "POST",
            body: { name: "auto", kind: "sft" } });
          toast(r.note || "iterate started", "ok");
          loadHistory();
        } catch (e) {
          if (e.code === "already_running") {
            toast("a job is already running — see Jobs", "error");
          } else toast(`${e.code}: ${e.message}`, "error");
        } finally { triggerBtn.disabled = false; }
      };
    }

    // ------------------------------------------------------------ cron card
    function renderCron() {
      const line = `15 3 * * * cd ~ && sable train auto-check --trigger >> $HOME/.sable/auto_iterate.log 2>&1`;
      cronCard.replaceChildren(
        el("h3", { text: "Autonomous runs (cron)" }),
        el("p", { class: "hint", text:
          "Honest labeling: the dashboard never fires rules on its own — no " +
          "ticker thread lives in this process. Schedule the check yourself; " +
          "`auto-check --trigger` exits 0 silently when thresholds or the " +
          "resource gate are not met, and runs the full iterate loop " +
          "(curate → train → export → validate → promote) when they are." }),
        el("pre", { class: "log", text: line }),
        copyButton(line, "copy cron line"));
    }

    // ----------------------------------------------------------- history
    async function loadHistory() {
      try {
        const d = await api("/api/iterate/runs");
        const runs = (d.runs || []).filter((r) => r.name === "auto").slice(0, 10);
        if (!runs.length) {
          historyHost.replaceChildren(el("h3", { text: "Auto-run history" }),
            el("p", { class: "hint", text:
              "No scheduled runs yet. Manual runs live in the Iterate panel." }));
          return;
        }
        historyHost.replaceChildren(
          el("h3", { text: "Auto-run history" }),
          el("table", {},
            el("thead", {}, el("tr", {},
              el("th", { text: "Run" }), el("th", { text: "Started" }),
              el("th", { text: "State" }), el("th", { text: "" }))),
            el("tbody", {}, ...runs.map((r) => el("tr", {},
              el("td", { class: "mono" }, r.run_id),
              el("td", {}, fmtTime(r.started)),
              el("td", {}, badge(r.state, r.state === "done" ? "done" :
                r.state === "running" ? "partial" : "failed")),
              el("td", {}, el("a", { href: "#/iterate", text: "open Iterate" })))))));
      } catch { /* history is best-effort */ }
    }

    mount.replaceChildren(
      el("h2", { text: "Schedule" }), rulesCard, readyCard, cronCard, historyHost);

    loadRules().catch((e) => toast(e.message, "error"));
    loadReadiness().catch(() => {});
    renderCron();
    loadHistory();

    const unsubs = [
      bus.on("iterate.step", () => { loadReadiness().catch(() => {}); loadHistory(); }),
      bus.on("job.state", () => loadReadiness().catch(() => {})),
    ];
    const timer = setInterval(() => loadReadiness().catch(() => {}), 20000);
    return { destroy() { clearInterval(timer); for (const u of unsubs) u(); } };
  },
};

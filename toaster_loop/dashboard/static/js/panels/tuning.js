// panels/tuning.js — setup & optimization panel: environment doctor (WP7),
// throughput bench + noise floor (WP1) and the calibration sweep (WP3).
//
// Everything here mirrors the CLI exactly (same modules under the hood):
//   toaster_loop doctor --env [--fix]   -> POST /api/doctor/run, poll GET /api/doctor/env
//   toaster_loop bench  --runs N …      -> POST /api/bench/run (+ SSE bench.run)
//   toaster_loop calibrate --quick      -> POST /api/calibrate/run (+ SSE calibrate.run)
// Nothing runs on page open: every measurement is an explicit user click.
// Long operations run as one server-side background task each (409 while
// busy); we poll the status endpoints and also subscribe to the bus events.

import { api } from "../api.js";
import { bus } from "../bus.js";
import { el, badge, table, empty, toast, confirmDialog } from "../ui.js";

const fmtTps = (v) => v == null ? "?" : `${Number(v).toFixed(2)} tok/s`;
const fmtPct = (v) => v == null ? "?" : `${Number(v).toFixed(2)}%`;

export default {
  title: "Tuning",
  render(mount) {
    // ---------------------------------------------------------------- state
    let doctorBusy = false;
    let benchBusy = false;
    let calBusy = false;
    let lastBench = null;   // latest /api/bench/run payload
    let lastCal = null;     // latest /api/calibrate/run payload

    // ---------------------------------------------------------- doctor card
    const doctorCard = el("div", { class: "card" },
      el("h3", { text: "Environment doctor" }),
      el("p", { class: "hint", text:
        "measures disk, RAM, VRAM telemetry, config, llama-server and " +
        "state — never assumes. Same checks as `toaster_loop doctor --env`. " +
        "Nothing runs until you click." }),
      el("div", { class: "field" },
        el("button", { class: "btn primary", text: "Run doctor",
          onclick: () => startDoctor(false) }),
        el("button", { class: "btn", text: "Run with --fix",
          title: "Also removes stale instance state files whose pid is " +
                 "verified gone (foreign pids are never touched)",
          onclick: () => confirmDialog({
            title: "Doctor --fix",
            body: "Removes only stale instance-state files whose pid is " +
                  "verified gone. Processes toaster_loop did not spawn are " +
                  "never touched.",
            confirmLabel: "Run with fix",
          }).then((yes) => yes && startDoctor(true)) })));

    const doctorBody = el("div", {});
    doctorCard.appendChild(doctorBody);

    async function startDoctor(fix) {
      try {
        await api("/api/doctor/run", { method: "POST", body: { fix } });
        doctorBusy = true;
        renderDoctor({ state: "running" });
        pollDoctor();
      } catch (e) { toast(e.message, "error"); }
    }

    let doctorPollTimer = null;
    function pollDoctor() {
      clearTimeout(doctorPollTimer);
      doctorPollTimer = setTimeout(async () => {
        if (!document.body.contains(doctorCard)) return;
        try {
          const d = await api("/api/doctor/env");
          if (d.state === "running") { renderDoctor(d); pollDoctor(); return; }
          doctorBusy = false;
          if (d.state === "error") toast("doctor failed: " + d.error, "error");
          renderDoctor(d);
        } catch { /* transient — keep polling */ pollDoctor(); }
      }, 1500);
    }

    function renderDoctor(d) {
      if (!d || d.state === "idle") { doctorBody.replaceChildren(""); return; }
      if (d.state === "running") {
        doctorBody.replaceChildren(
          el("p", { class: "hint" },
            badge("measuring…", "warn"),
            " vendor probes can take a while under GPU load — the panel " +
            "stays usable, click away and come back."));
        return;
      }
      if (d.state === "error") {
        doctorBody.replaceChildren(el("p", { class: "err", text: d.error }));
        return;
      }
      const rows = (d.checks || []).map((c) => ({
        status: badge(c.status === "ok" ? "OK" : c.status === "warn" ? "WARN" : "FAIL",
                      c.status === "ok" ? "done" : c.status === "warn" ? "warn" : "error"),
        name: el("span", { class: "mono", text: c.name }),
        detail: el("div", {},
          ...(c.lines || []).map((ln) => el("div", { class: "hint mono", text: ln })),
          ...(c.next_commands || []).length
            ? [el("div", { class: "hint",
                text: "next: " + c.next_commands.join(" · ") })] : []),
      }));
      const problems = (d.checks || []).filter((c) => c.status === "problem").length;
      const warns = (d.checks || []).filter((c) => c.status === "warn").length;
      doctorBody.replaceChildren(
        el("p", {}, badge(d.exit_code === 0
          ? (warns ? `ok, ${warns} warning(s)` : "all good")
          : `${problems} problem(s)`,
          d.exit_code === 0 ? (warns ? "warn" : "done") : "error"),
          d.finished ? el("span", { class: "hint mono", text: " " + d.finished }) : null),
        table([{ label: "", key: "status" }, { label: "check", key: "name" },
               { label: "detail", key: "detail" }], rows));
    }

    // ------------------------------------------------------------ bench card
    const benchCard = el("div", { class: "card" },
      el("h3", { text: "Bench (WP1)" }),
      el("p", { class: "hint", text: "loading…" }));

    const renderBenchResult = (s) => {
      if (!s || s.state === "idle") return "";
      if (s.state === "running")
        return el("p", { class: "hint mono",
          text: (s.lines || s.log || []).slice(-3).join("\n") || "running…" });
      if (s.state === "error")
        return el("p", { class: "err", text: "error: " + s.error });
      if (s.state === "diverged") {
        const p = el("p", { class: "warn" }, "tokens DIVERGED — speed numbers here are not trustworthy");
        return p;
      }
      const r = s.result || {};
      const gen = r.generation || {}, pp = r.prompt_processing || {};
      const ti = r.token_identity || {};
      return el("div", {},
        el("p", { class: "mono" },
          `gen median ${fmtTps(gen.median)} (spread ${fmtPct(gen.spread_pct)}) · ` +
          `prompt ${fmtTps(pp.median)} tok/s`),
        badge(ti.status === "ok" ? "token identity OK" : "tokens DIVERGED",
              ti.status === "ok" ? "done" : "error"),
        el("p", { class: "hint mono", text: r.path || "" }));
    };

    const pollBench = async () => {
      try {
        const s = await api("/api/bench/run");
        lastBench = s;
        if (s.state === "running") {
          benchBusy = true; renderBench();
          setTimeout(pollBench, 2000);
          return;
        }
        if (benchBusy) {
          benchBusy = false;
          if (s.state === "error") toast("bench failed: " + s.error, "error");
          else toast("bench complete", "ok");
        }
        renderBench();
      } catch { /* transient */ }
    };

    async function renderBench() {
      let list, targets;
      try {
        [list, targets] = await Promise.all([
          api("/api/bench"), api("/api/bench/targets")]);
      } catch (e) {
        benchCard.replaceChildren(el("h3", { text: "Bench (WP1)" }),
          el("p", { class: "err", text: String(e.message || e) }));
        return;
      }
      const nf = list.noise_floor;

      // Target selector: any RUNNING server instance (multi-agent friendly —
      // bench measures whichever endpoint you pick), or the config default.
      const targetSel = el("select", { class: "input", title:
        "Which server to measure: pick a running instance, or let the bench " +
        "spawn the default model (needs the GPU free)." });
      const running = (targets.instances || []).filter((i) => i.running);
      if (running.length) {
        for (const i of running) {
          targetSel.appendChild(el("option", { value: i.id },
            `${i.name || i.id}${i.model_name ? " — " + i.model_name : ""}`));
        }
      }
      targetSel.appendChild(el("option", { value: "" },
        targets.can_spawn_default
          ? "default (spawn from gguf_path)"
          : "default (no gguf_path configured)"));

      const runsIn = el("input", { class: "input", type: "number",
        min: "1", max: "50", value: "5", style: "width:5em",
        title: "Repetitions of the fixed prompt set; more runs tighten the measured spread." });
      const labelIn = el("input", { class: "input mono", type: "text",
        placeholder: "label (optional)", style: "max-width:14em",
        title: "Name this measurement so you can compare it with later runs in the table." });
      const nfChk = el("input", { type: "checkbox",
        title: "Record this run as the noise-floor baseline: later effects smaller " +
          "than its spread are treated as noise, not wins." });
      const btn = el("button", { class: "btn primary",
        title: "Measure temperature-0 throughput over a fixed prompt set against the chosen server.",
        text: benchBusy ? "measuring…" : "Run bench" });
      if (benchBusy) btn.disabled = true;
      btn.addEventListener("click", async () => {
        try {
          await api("/api/bench/run", { method: "POST", body: {
            runs: parseInt(runsIn.value, 10) || 5,
            label: labelIn.value || "",
            server_id: targetSel.value || undefined,
            noise_floor: nfChk.checked } });
          benchBusy = true; renderBench(); pollBench();
        } catch (e) { toast(e.message, "error"); }
      });

      const rows = (list.records || []).slice(0, 12).map((r) => ({
        ts: el("span", { class: "mono hint", text: String(r.timestamp || "") }),
        kind: badge(String(r.kind || "?"), r.kind === "noise_floor" ? "warn" : ""),
        label: el("span", { class: "mono", text: r.label || "-" }),
        gen: fmtTps(r.gen_median_tps),
        spread: fmtPct(r.gen_spread_pct),
        ident: r.token_identity === "ok" ? badge("tokens OK", "done")
                                         : badge(String(r.token_identity || "?"), "warn"),
      }));

      benchCard.replaceChildren(
        el("h3", {}, "Bench (WP1) ",
          nf ? badge(`noise floor ${fmtPct(nf.spread_pct)}`, "done")
             : badge("no noise-floor baseline", "warn")),
        el("p", { class: "hint", text:
          "temperature-0 throughput over a fixed prompt set with the " +
          "token-identity rule; effects below the recorded spread are noise, " +
          "not wins. Pick a running server to measure it, or let it spawn " +
          "the default (spawning needs the GPU free)." }),
        rows.length ? table([
            { label: "timestamp", key: "ts" }, { label: "kind", key: "kind" },
            { label: "label", key: "label" }, { label: "gen med", key: "gen" },
            { label: "spread", key: "spread" },
            { label: "identity", key: "ident" }], rows)
                   : empty("no persisted bench records yet"),
        el("div", { class: "field" },
          targetSel, runsIn, labelIn,
          el("label", {}, nfChk, " noise-floor baseline"),
          btn),
        renderBenchResult(lastBench) || "");
    }

    // ------------------------------------------------------- calibrate card
    const calCard = el("div", { class: "card" },
      el("h3", { text: "Calibration (WP3)" }),
      el("p", { class: "hint", text: "loading…" }));

    const renderCalResult = (s) => {
      if (!s || s.state === "idle") return "";
      if (s.state === "running")
        return el("pre", { class: "mono hint", style: "max-height:12em;overflow:auto",
          text: (s.lines || s.log || []).slice(-8).join("\n") || "running…" });
      if (s.state === "error")
        return el("p", { class: "err", text: "error: " + s.error });
      const r = s.result || {};
      const adopted = Object.entries(r.adopted_settings || {});
      return el("div", {},
        el("p", {}, adopted.length
          ? el("span", {}, "adopted: ",
              badge(adopted.map(([k, v]) => `${k}=${v}`).join(", "), "done"))
          : el("span", { class: "hint", text: "nothing adopted (all candidates within noise)" })),
        el("p", { class: "hint mono" },
          `threshold ${fmtPct(r.threshold_pct)} · noise floor ${fmtPct(r.noise_floor_pct)}`));
    };

    const pollCal = async () => {
      try {
        const s = await api("/api/calibrate/run");
        lastCal = s;
        if (s.state === "running") {
          calBusy = true; renderCal();
          setTimeout(pollCal, 3000);
          return;
        }
        if (calBusy) {
          calBusy = false;
          if (s.state === "error") toast("calibration failed: " + s.error, "error");
          else toast("calibration complete", "ok");
        }
        renderCal();
      } catch { /* transient */ }
    };

    async function setMode(mode) {
      try {
        await api("/api/config", { method: "PUT", body: {
          section: "serve", patch: { calibration: mode } } });
        toast(`serve.calibration: ${mode}`, "ok");
        renderCal();
      } catch (e) { toast(e.message, "error"); }
    }

    async function renderCal() {
      let d;
      try { d = await api("/api/calibrate"); }
      catch (e) {
        calCard.replaceChildren(el("h3", { text: "Calibration (WP3)" }),
          el("p", { class: "err", text: String(e.message || e) }));
        return;
      }
      const mode = d.calibration_mode || "off";
      const busyInsts = d.running_instances || [];

      const quickBtn = el("button", { class: "btn primary",
        title: "Short sweep of serving knobs (batch sizes, offload splits); adopts " +
          "only settings that beat the noise floor.",
        text: calBusy ? "sweeping…" : "Sweep (--quick)" });
      const fullBtn = el("button", { class: "btn", text: "Full sweep",
        title: "Exhaustive knob sweep — much slower than quick, same adoption rule." });
      if (calBusy) { quickBtn.disabled = true; fullBtn.disabled = true; }
      const startSweep = async (quick) => {
        try {
          await api("/api/calibrate/run", { method: "POST",
            body: { quick, runs: 3 } });
          calBusy = true; renderCal(); pollCal();
        } catch (e) { toast(e.message, "error"); }
      };
      const sweepWarning = () => busyInsts.length
        ? `WARNING: ${busyInsts.map((i) => i.name || i.id).join(", ")} ` +
          "running — the sweep spawns its OWN llama-server per candidate and " +
          "every measurement will be skewed (or OOM). Stop them on the " +
          "Serve panel first. "
        : "The sweep spawns its own llama-server per candidate — the GPU " +
          "must be free. ";
      quickBtn.addEventListener("click", () =>
        confirmDialog({
          title: "Calibration sweep",
          body: sweepWarning() +
                "Adoption requires > max(3%, noise floor) AND identical tokens.",
          confirmLabel: "Start quick sweep",
        }).then((yes) => yes && startSweep(true)));
      fullBtn.addEventListener("click", () =>
        confirmDialog({ title: "Full sweep",
          body: sweepWarning() + "Every knob, every candidate — slower than --quick.",
          confirmLabel: "Start full sweep" })
          .then((yes) => yes && startSweep(false)));

      const rows = (d.profiles || []).map((p) => ({
        ts: el("span", { class: "mono hint", text: String(p.timestamp || "") }),
        machine: p.active_machine ? badge("this machine", "done") : badge("other", ""),
        model: el("span", { class: "mono hint", text: p.model_hash || "?" }),
        adopted: Object.keys(p.adopted_settings || {}).length
          ? el("span", { class: "mono",
              text: Object.entries(p.adopted_settings)
                .map(([k, v]) => `${k}=${v}`).join(", ") })
          : el("span", { class: "hint", text: "—" }),
        active: p.is_active_profile ? badge("active", "warn") : "",
      }));

      calCard.replaceChildren(
        el("h3", {}, "Calibration (WP3) ",
          badge(mode === "auto" ? "serve.calibration: auto"
                                : "serve.calibration: off",
                mode === "auto" ? "done" : "warn")),
        el("p", { class: "hint", text: d.model
          ? `model: ${d.model}` : "no gguf_path configured — set it in Settings first." }),
        busyInsts.length && !calBusy
          ? el("p", { class: "warn" },
              `⚠ running servers hold the GPU: ${busyInsts.map((i) => i.name || i.id).join(", ")} — stop them before sweeping`)
          : null,
        el("div", { class: "field" },
          quickBtn, fullBtn,
          el("button", { class: "btn", text: mode === "auto" ? "Turn honoring OFF" : "Honor profile at serve",
            title: "auto = new servers launch with this machine's adopted sweep " +
              "profile; off = launches byte-identical to defaults.",
            onclick: () => setMode(mode === "auto" ? "off" : "auto") }),
          el("button", { class: "btn", text: "Clear profiles",
            title: "Delete all stored per-machine calibration profiles.",
            onclick: () => confirmDialog({
              title: "Clear calibration profiles",
              body: "Deletes all stored profiles under <TOASTER_LOOP_HOME>/calibrate/.",
              confirmLabel: "Clear",
            }).then(async (yes) => {
              if (!yes) return;
              try {
                const r = await api("/api/calibrate/clear", { method: "POST", body: {} });
                toast(`removed ${r.removed} profile(s)`, "ok");
                renderCal();
              } catch (e) { toast(e.message, "error"); }
            }) })),
        rows.length ? table([
            { label: "timestamp", key: "ts" }, { label: "machine", key: "machine" },
            { label: "model", key: "model" }, { label: "adopted", key: "adopted" },
            { label: "", key: "active" }], rows)
                    : empty("no calibration profiles stored yet"),
        renderCalResult(lastCal) || "");
    }

    // ---------------------------------------------------------------- mount
    mount.replaceChildren(
      el("h2", { text: "Tuning" }),
      el("p", { class: "hint", text:
        "measure, don't assume — environment doctor, throughput baseline " +
        "(noise floor) and the per-machine calibration sweep. Nothing runs " +
        "until you click; bench can measure any running server instance." }),
      doctorCard, benchCard, calCard);

    // Bootstrap WITHOUT running checks: show a finished report if one exists
    // (server-side state), otherwise leave the idle card with its buttons.
    api("/api/doctor/env").then((d) => {
      if (d.state === "running") { doctorBusy = true; renderDoctor(d); pollDoctor(); }
      else if (d.state === "done" || d.state === "error") renderDoctor(d);
    }).catch(() => {});
    renderBench();
    renderCal();
    // Bus nudges: another tab finishing a run refreshes ours immediately.
    const offBench = bus.on("bench.run", (p) => {
      if (p.state && p.state !== "running") pollBench();
    });
    const offCal = bus.on("calibrate.run", (p) => {
      if (p.state && p.state !== "running") pollCal();
    });
    const offDoctor = bus.on("doctor.run", (p) => {
      if (p.state && p.state !== "running") {
        api("/api/doctor/env").then((d) => { doctorBusy = false; renderDoctor(d); })
          .catch(() => {});
      }
    });
    return { destroy() { offBench(); offCal(); offDoctor(); clearTimeout(doctorPollTimer); } };
  },
};

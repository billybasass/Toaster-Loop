// panels/options.js — opt-in options panel: every master switch in the config
// in one place, with its EFFECTIVE state on this machine and the per-knob
// values the code actually reads.
//
// The Settings panel owns the core workflow sections (serve defaults, curation,
// training, validation, run_settings, sampling, agent healing, indexing,
// skills, playbooks, goldens, dashboard token) and the Raw YAML escape hatch.
// This panel owns the rest — the long tail of opt-in features that otherwise
// only existed as commented YAML:
//
//   serve:        calibration / WP4 watchdog / WP5 coexistence
//   agent:        recording, delegation, capabilities, compression, verify
//   tools:        secret blocking, spill, lint/LSP, web_fetch posture
//   pipeline:     evolution, rft, shadow eval
//   automation:   training auto-check, easy_training
//   storage:      janitor, recording retention
//
// Every card patches ONE config section through PUT /api/config {section,
// patch} (the same merge+validate+backup path as Settings). Blank inputs are
// never sent, so "unset" (code default) is preserved; the status line under
// each card shows what that means on THIS machine.

import { api } from "../api.js";
import { el, badge, toast, confirmDialog } from "../ui.js";

// Effective-value fallbacks, taken from the code defaults (core/config.py,
// core/agent_runner.py, training/scheduler.py, serving/*.py). Shown when the
// config leaves a key unset — the "what runs today" column.
const EFFECTIVE = {
  recording_enabled: true,
  recording_redact_secrets: true,
  delegation_enabled: true,
  heal_tool_calls: true,
  verify_on_stop: true,
  spill_enabled: true,
  spill_threshold_chars: 48000,
  turn_result_budget_chars: 120000,
  spill_retention_days: 7,
  lint_on_write: true,
  block_sensitive_files: true,
  run_command_argv_only: false,
  min_free_vram_mib: 0,
  require_ac_power: false,
  auto_enabled: false,
  auto_min_traces: 50,
  auto_min_score: 0.5,
  calibration: "off",
  watchdog_enabled: false,
  watchdog_window_seconds: 90,
  watchdog_sample_interval_seconds: 5,
  watchdog_cpu_threshold_ms: 10,
  watchdog_io_threshold_bytes: 0,
  watchdog_kill_grace_seconds: 8,
  watchdog_restart_backoff_base_seconds: 2,
  watchdog_max_restarts: 3,
  watchdog_restart_window_seconds: 600,
  idle_unload_seconds: 0,
  lazy_load: false,
  coexist_policy: "refuse",
  janitor_enabled: false,
  janitor_retention_days: 30,
  janitor_disk_retention_days: 90,
  evolution_enabled: false,
  evolution_max_call_budget: 60,
  rft_enabled: false,
  rft_sample_k: 4,
  rft_judge_min_score: 0.85,
  rft_max_prompts: 16,
  rft_temperature: 0.9,
  rft_max_tokens: 512,
  rft_max_prompt_tokens: 8192,
  rft_emit_dpo: true,
  rft_max_dpo_pairs_per_prompt: 3,
  shadow_enabled: false,
  max_shadow_regression_pct: 5.0,
  shadow_blocks_promotion: false,
  shadow_replay_n: 8,
  min_tier_auto_deploy: 2,
  easy_enabled: false,
  easy_auto_deploy: false,
};

// ------------------------------------------------------------------ widgets

function numInput(value, { placeholder = "", step = "any" } = {}) {
  return el("input", { type: "number", step, value: value == null ? "" : value,
    placeholder, style: "max-width:11em" });
}

function boolSelect(value, { allowNone = true } = {}) {
  const opts = [];
  if (allowNone) opts.push(el("option", { value: "", title:
    "Leave unset: the program's built-in default applies (shown in the status line)." },
    "— default —"));
  for (const v of [true, false]) opts.push(el("option", { value: String(v), text: String(v) }));
  const s = el("select", {}, ...opts);
  if (value === true || value === false) s.value = String(value);
  return s;
}

function choiceSelect(value, choices, { placeholder = "— default —" } = {}) {
  const opts = [el("option", { value: "", title:
    "Leave unset: the program's built-in default applies (shown in the status line)." },
    placeholder)];
  for (const c of choices) opts.push(el("option", { value: String(c), text: String(c) }));
  const s = el("select", {}, ...opts);
  if (value != null && choices.map(String).includes(String(value))) s.value = String(value);
  return s;
}

function textInput(value, placeholder = "") {
  return el("input", { type: "text", value: value == null ? "" : value,
    placeholder, style: "max-width:18em" });
}

function checkbox(checked, label, title = "") {
  const i = el("input", { type: "checkbox" });
  if (checked) i.checked = true;
  const node = el("label", { class: "chk", title }, i, " " + label);
  node.get = () => i.checked;
  return node;
}

// Collects [key, input, kind] pairs; blank inputs are dropped (patch merge
// keeps the previous/unset value server-side). kind "bool-optional" yields
// true/false only from an explicit select value.
function collect(pairs) {
  const patch = {};
  for (const [k, inp, kind] of pairs) {
    let v;
    if (kind === "chk") v = inp.get();
    else if (kind === "num" || kind === "bool" || kind === "bool-optional" || kind === "text")
      v = inp.value;
    else continue;
    if (v === "" || v == null) continue;
    if (kind === "num") patch[k] = Number(v);
    else if (kind === "bool" || kind === "bool-optional") patch[k] = v === true || v === "true";
    else patch[k] = v;
  }
  return patch;
}

function row(labelText, widget, hint) {
  // Hover tooltip: every row's inline hint doubles as the control's title so
  // newcomers discover what each knob does without reading the whole card.
  // (Skipped when the widget already carries its own, richer title.)
  if (hint && widget && !widget.getAttribute("title"))
    widget.setAttribute("title", String(hint));
  const kids = [el("label", { text: labelText }), widget];
  if (hint) kids.push(el("span", { class: "hint", text: hint }));
  return el("div", { class: "field" }, ...kids);
}

function card({ title, section, hint, status, build, applyLabel = "Apply" }) {
  const body = el("div");
  const statusEl = el("span", { class: "hint" });
  if (status) statusEl.textContent = status;
  const applyBtn = el("button", { class: "small primary", text: applyLabel,
    title: "Save only the fields you changed (blank fields stay untouched). " +
      "Writes go through the same validated, backed-up config path as Settings." });
  const patcher = build(body);
  applyBtn.addEventListener("click", async () => {
    const patch = patcher();
    if (!Object.keys(patch).length) {
      statusEl.textContent = "nothing to send";
      return;
    }
    applyBtn.disabled = true;
    statusEl.textContent = "applying…";
    try {
      await api("/api/config", { method: "PUT", body: { section, patch } });
      toast(`${title} saved`, "ok");
      location.hash = "#/options"; // re-render with fresh effective state
    } catch (e) {
      statusEl.textContent = e.message;
      toast(`${e.code || "error"}: ${e.message}`, "error");
    } finally { applyBtn.disabled = false; }
  });
  return el("div", { class: "card" },
    el("h3", {}, title, " ", badge(section, "none")),
    hint ? el("p", { class: "hint", text: hint }) : null,
    body,
    el("div", { class: "field" }, statusEl, "  ", applyBtn));
}

const onOff = (v) => badge(v ? "ON" : "off", v ? "done" : "dim");

// --------------------------------------------------------------------- panel

export default {
  title: "Options",
  render(mount) {
    let cfg = {};
    let hw = null;

    // ------------------------------------------------------------ load
    (async () => {
      const [c, h] = await Promise.allSettled([
        api("/api/config"), api("/api/hardware")]);
      if (c.status === "fulfilled") cfg = c.value.config || {};
      if (h.status === "fulfilled") hw = h.value;
      render();
    })().catch(() => render());

    function render() {
      const serve = cfg.serve || {};
      const agent = cfg.agent || {};
      const tools = cfg.tools || {};
      const jan = cfg.janitor || {};
      const evo = cfg.evolution || {};
      const rft = cfg.rft || {};
      const val = cfg.validation || {};
      const tr = cfg.training || {};
      const easy = cfg.easy_training || {};
      const caps = (agent.capabilities && typeof agent.capabilities === "object")
        ? agent.capabilities : {};
      const lsp = (tools.lsp && typeof tools.lsp === "object") ? tools.lsp : {};
      const wf = (tools.web_fetch && typeof tools.web_fetch === "object")
        ? tools.web_fetch : {};

      const ev = (k, explicit, fallback = EFFECTIVE[k]) =>
        explicit !== undefined && explicit !== null ? explicit : fallback;

      const cards = [];

      // ------------------------------------------------- serve: calibration
      cards.push(card({
        title: "Serving autotune",
        section: "serve",
        hint: "WP3: `toaster_loop calibrate` sweeps per-machine knobs (see the Tuning " +
          "panel for the sweep itself); this switch decides whether serving " +
          "APPLIES the stored profile at launch — underneath any explicit " +
          "run_settings values. Off = launches byte-identical to pre-WP3.",
        status: `effective serve.calibration: ${ev("calibration", serve.calibration)}`,
        build: (body) => {
          const modeSel = choiceSelect(serve.calibration, ["off", "auto"]);
          body.replaceChildren(
            row("calibration", modeSel,
              "auto = apply the adopted profile for this machine+model hash"),
            row("", el("a", { href: "#/tuning", text: "run the sweep → Tuning" })));
          return () => collect([["calibration", modeSel, "text"]]);
        },
      }));

      // ------------------------------------------------- serve: WP4 watchdog
      const wd = {
        enabled: serve.watchdog_enabled,
        window: serve.watchdog_window_seconds,
        sample: serve.watchdog_sample_interval_seconds,
        cpu: serve.watchdog_cpu_threshold_ms,
        io: serve.watchdog_io_threshold_bytes,
        grace: serve.watchdog_kill_grace_seconds,
        backoff: serve.watchdog_restart_backoff_base_seconds,
        maxRestarts: serve.watchdog_max_restarts,
        windowCap: serve.watchdog_restart_window_seconds,
      };
      cards.push(card({
        title: "Engine watchdog (WP4)",
        section: "serve",
        hint: "Detects a FROZEN llama-server child: a request in flight for the " +
          "whole window with ≈zero CPU and IO growth. A healthy-but-slow server " +
          "is never killed; an idle one never triggers anything. On detection: " +
          "SIGTERM → SIGKILL after grace → restart with exponential backoff " +
          "within a per-window cap. Linux-only (/proc); events on the bus as " +
          "serve.watchdog. Strictly opt-in — off by default.",
        status: `effective: ${ev("watchdog_enabled", serve.watchdog_enabled)} · ` +
          `window ${ev("watchdog_window_seconds", wd.window)} s · ` +
          `≤${ev("watchdog_max_restarts", wd.maxRestarts)} restarts / ` +
          `${ev("watchdog_restart_window_seconds", wd.windowCap)} s`,
        build: (body) => {
          const en = boolSelect(wd.enabled);
          const windowIn = numInput(wd.window);
          const sampleIn = numInput(wd.sample);
          const cpuIn = numInput(wd.cpu);
          const ioIn = numInput(wd.io);
          const graceIn = numInput(wd.grace);
          const backoffIn = numInput(wd.backoff);
          const maxIn = numInput(wd.maxRestarts);
          const capIn = numInput(wd.windowCap);
          body.replaceChildren(
            row("watchdog_enabled", en, "master switch — nothing runs when false"),
            row("watchdog_window_seconds", windowIn, "in-flight time before a freeze call"),
            row("watchdog_sample_interval_seconds", sampleIn, "/proc sampling cadence"),
            row("watchdog_cpu_threshold_ms", cpuIn, "max CPU growth per sample to count as ≈0"),
            row("watchdog_io_threshold_bytes", ioIn, "max disk-IO growth per sample to count as ≈0"),
            row("watchdog_kill_grace_seconds", graceIn, "SIGTERM → SIGKILL grace"),
            row("watchdog_restart_backoff_base_seconds", backoffIn, "base × 2^(attempt−1)"),
            row("watchdog_max_restarts", maxIn, "cap per window, then give up (restart_capped)"),
            row("watchdog_restart_window_seconds", capIn, "the window the cap applies to"));
          return () => collect([
            ["watchdog_enabled", en, "bool-optional"],
            ["watchdog_window_seconds", windowIn, "num"],
            ["watchdog_sample_interval_seconds", sampleIn, "num"],
            ["watchdog_cpu_threshold_ms", cpuIn, "num"],
            ["watchdog_io_threshold_bytes", ioIn, "num"],
            ["watchdog_kill_grace_seconds", graceIn, "num"],
            ["watchdog_restart_backoff_base_seconds", backoffIn, "num"],
            ["watchdog_max_restarts", maxIn, "num"],
            ["watchdog_restart_window_seconds", capIn, "num"]]);
        },
      }));

      // -------------------------------------------- serve: WP5 coexistence
      const armed = Number(serve.idle_unload_seconds || 0) > 0 &&
        Number(serve.min_free_vram_mib || 0) > 0;
      cards.push(card({
        title: "GPU coexistence (WP5)",
        section: "serve",
        hint: "Frees GPU memory for training without touching your models. " +
          "idle_unload_seconds > 0 starts the supervisor-side idle timer (never " +
          "unloads mid-request). min_free_vram_mib > 0 TOGETHER with idle " +
          "arming enables the train+serve handshake: a job that fails the VRAM " +
          "gate but would pass with instances unloaded triggers the unload " +
          "policy. refuse (default) only emits the coexist_refused event with " +
          "the headroom numbers; unload performs it. Manual load/unload: Serve " +
          "panel or `toaster_loop instances load|unload <id>`.",
        status: armed
          ? `armed — handshake active, policy=${ev("coexist_policy", serve.coexist_policy)}`
          : "not armed (both idle_unload_seconds and serve.min_free_vram_mib must be > 0)",
        build: (body) => {
          const idleIn = numInput(serve.idle_unload_seconds, { placeholder: "0 = off" });
          const lazy = boolSelect(serve.lazy_load);
          const vramIn = numInput(serve.min_free_vram_mib, { placeholder: "0 = off" });
          const policy = choiceSelect(serve.coexist_policy, ["refuse", "unload"]);
          body.replaceChildren(
            row("idle_unload_seconds", idleIn,
              "gracefully stop an idle instance after this long (0 = disabled)"),
            row("lazy_load", lazy,
              "true: a request to an unloaded instance transparently relaunches it"),
            row("min_free_vram_mib", vramIn, "handshake VRAM floor (0 = no handshake)"),
            row("coexist_policy", policy,
              "refuse = report headroom only · unload = actually unload"));
          return () => collect([
            ["idle_unload_seconds", idleIn, "num"],
            ["lazy_load", lazy, "bool-optional"],
            ["min_free_vram_mib", vramIn, "num"],
            ["coexist_policy", policy, "text"]]);
        },
      }));

      // ------------------------------------------------ agent: recording
      // The API returns the recording/delegation sections FLATTENED into
      // recording_enabled / delegation_* top-level fields, so read the
      // effective state from those; the patch still goes to the nested
      // section the loader reads from YAML.
      cards.push(card({
        title: "Trace recording & retention",
        section: "recording",
        hint: "Every run is appended to the experience buffer (the raw material " +
          "for curation and training). redact_secrets masks credentials in the " +
          "durable copy — disable only with a reason. retention_days is the " +
          "default window for `toaster_loop traces prune` (0 = prune always needs " +
          "--older-than; pruning never runs automatically).",
        status: `effective: enabled=${ev("recording_enabled",
          cfg.recording_enabled)} · ` +
          `redact=${ev("recording_redact_secrets",
            cfg.recording_redact_secrets)} · ` +
          `retention ${ev("recording_retention_days",
            cfg.recording_retention_days)} d`,
        build: (body) => {
          const en = boolSelect(cfg.recording_enabled);
          const red = boolSelect(cfg.recording_redact_secrets);
          const ret = numInput(cfg.recording_retention_days, { placeholder: "0 = no default" });
          body.replaceChildren(
            row("enabled", en, "off = runs still happen, nothing is recorded"),
            row("redact_secrets", red, "mask credentials in the durable trace copy"),
            row("retention_days", ret, "default age cutoff for `toaster_loop traces prune`"));
          return () => collect([
            ["enabled", en, "bool-optional"],
            ["redact_secrets", red, "bool-optional"],
            ["retention_days", ret, "num"]]);
        },
      }));

      // ------------------------------------------------ agent: delegation
      cards.push(card({
        title: "Delegation",
        section: "delegation",
        hint: "list_agents / delegate_task / check_delegation let one profile " +
          "hand a self-contained subtask to another profile on the same " +
          "project. Children have no timeout — they take as long as they " +
          "need; depth is capped at 1 — children cannot delegate further.",
        status: `effective: enabled=${ev("delegation_enabled",
          cfg.delegation_enabled)}`,
        build: (body) => {
          const en = boolSelect(cfg.delegation_enabled);
          body.replaceChildren(
            row("enabled", en, "off = the tools are not exposed at all"));
          return () => collect([
            ["enabled", en, "bool-optional"]]);
        },
      }));

      // ---------------------------------------- agent: runtime + capabilities
      const capKeys = ["internet", "run_commands", "vision", "screenshots"];
      const capChk = {};
      cards.push(card({
        title: "Agent runtime & capabilities",
        section: "agent",
        hint: "Global defaults for the agent runtime. context_token_budget is " +
          "where context compression kicks in (default = 75% of context_length; " +
          "0 disables); keep_recent_tokens keeps the tail verbatim during a " +
          "compression pass. Capabilities are the global DEFAULTS — a profile " +
          "may override each one (Profiles panel). run_commands executes with " +
          "YOUR privileges and is NOT workspace-contained: trusted profiles only.",
        status: `effective: budget ${ev("context_token_budget",
          agent.context_token_budget, "75% of ctx")} tok · ` +
          `heal=${ev("heal_tool_calls", agent.heal_tool_calls)} · ` +
          `verify_on_stop=${ev("verify_on_stop", agent.verify_on_stop)} · caps: ` +
          capKeys.map((k) => `${k}=${
            caps[k] === true ? "on" : caps[k] === false ? "off" : "unset"}`).join(" · "),
        build: (body) => {
          const budget = numInput(agent.context_token_budget, { placeholder: "75% of ctx" });
          const keep = numInput(agent.keep_recent_tokens, { placeholder: "2048" });
          const heal = boolSelect(agent.heal_tool_calls);
          const verify = boolSelect(agent.verify_on_stop);
          for (const k of capKeys) {
            capChk[k] = checkbox(!!caps[k],
              k === "internet" ? "internet (web_fetch, public http(s) only)" :
              k === "run_commands" ? "run_commands (shell — your privileges)" :
              k === "vision" ? "vision (view_image; needs mmproj + VL model)" :
              "screenshots (take_screenshot)");
          }
          const capsRow = el("div", { class: "field" },
            el("label", { text: "capabilities (global defaults — per-profile in Profiles)" }),
            ...capKeys.map((k) => capChk[k]));
          body.replaceChildren(
            row("context_token_budget", budget, "compression trigger (0 = off)"),
            row("keep_recent_tokens", keep, "tail kept verbatim during compression"),
            row("heal_tool_calls", heal, "rescue tool calls emitted as fenced text"),
            row("verify_on_stop", verify,
              "one nudge when a run edits code without a fresh check"),
            capsRow);
          return () => {
            const p = collect([
              ["context_token_budget", budget, "num"],
              ["keep_recent_tokens", keep, "num"],
              ["heal_tool_calls", heal, "bool-optional"],
              ["verify_on_stop", verify, "bool-optional"]]);
            // Capabilities merge into the existing dict; only explicit checks
            // are sent so an unchecked box means "unset" (inherit default),
            // not "false".
            const merged = {};
            for (const k of capKeys) {
              const prev = caps[k];
              if (capChk[k].get() && !prev) merged[k] = true;
              else if (capChk[k].get() && prev) merged[k] = true;
              else if (!capChk[k].get() && prev) merged[k] = false;
            }
            if (Object.keys(merged).length) p.capabilities = { ...caps, ...merged };
            return p;
          };
        },
      }));

      // ---------------------------------------------------- tools: hygiene
      const lspChk = checkbox(!!lsp.enabled, "LSP diagnostics (tier 2, needs a language server on PATH)");
      cards.push(card({
        title: "Tool hygiene & output limits",
        section: "tools",
        hint: "block_sensitive_files keeps read_file/search_files away from " +
          "credential paths (defense-in-depth before secrets can enter traces). " +
          "Spill: oversized tool results are persisted to a file and replaced " +
          "in-context by a small stub the agent can re-read. lint_on_write runs " +
          "a cheap syntax check after writes (never rejects); LSP tier 2 adds " +
          "real diagnostics when a server exists. run_command_argv_only strips " +
          "shell semantics from the run_commands capability.",
        status: `effective: block=${ev("block_sensitive_files", tools.block_sensitive_files)} · ` +
          `spill=${ev("spill_enabled", tools.spill_enabled)}@` +
          `${ev("spill_threshold_chars", tools.spill_threshold_chars)} chars · ` +
          `lint=${ev("lint_on_write", tools.lint_on_write)} · ` +
          `argv_only=${ev("run_command_argv_only", tools.run_command_argv_only)}`,
        build: (body) => {
          const block = boolSelect(tools.block_sensitive_files);
          const argvOnly = boolSelect(tools.run_command_argv_only);
          const spillEn = boolSelect(tools.spill_enabled);
          const spillChars = numInput(tools.spill_threshold_chars);
          const spillTok = numInput(tools.spill_threshold_tokens, { placeholder: "off" });
          const turnBudget = numInput(tools.turn_result_budget_chars);
          const spillRet = numInput(tools.spill_retention_days);
          const lint = boolSelect(tools.lint_on_write);
          const privateNet = boolSelect(wf.allow_private_networks);
          const lspStart = numInput(lsp.startup_timeout_seconds);
          const lspCheck = numInput(lsp.check_timeout_seconds);
          body.replaceChildren(
            row("block_sensitive_files", block, "refuse credential paths (.env*, keys, .aws, …)"),
            row("run_command_argv_only", argvOnly, "shell off: argv only, no pipes/redirection"),
            row("spill_enabled", spillEn, "persist oversized results as re-readable files"),
            row("spill_threshold_chars", spillChars, "per-result spill threshold (chars)"),
            row("spill_threshold_tokens", spillTok, "token-accurate gate; wins over chars when set"),
            row("turn_result_budget_chars", turnBudget, "per-turn aggregate budget (largest spill first)"),
            row("spill_retention_days", spillRet, "spillover file age before prune"),
            row("lint_on_write", lint, "tier-1 syntax check after write_file/patch"),
            row("web_fetch.allow_private_networks", privateNet,
              "SSRF posture: true only for deliberate local-API work"),
            row("lsp.enabled", lspChk),
            row("lsp.startup_timeout_seconds", lspStart, "hard cap on server startup"),
            row("lsp.check_timeout_seconds", lspCheck, "hard cap per diagnostics check"));
          const lspPairs = [["lsp.startup_timeout_seconds", lspStart, "num"],
                            ["lsp.check_timeout_seconds", lspCheck, "num"]];
          const wfPairs = [["web_fetch.allow_private_networks", privateNet, "bool-optional"]];
          return () => {
            const p = collect([
              ["block_sensitive_files", block, "bool-optional"],
              ["run_command_argv_only", argvOnly, "bool-optional"],
              ["spill_enabled", spillEn, "bool-optional"],
              ["spill_threshold_chars", spillChars, "num"],
              ["spill_threshold_tokens", spillTok, "num"],
              ["turn_result_budget_chars", turnBudget, "num"],
              ["spill_retention_days", spillRet, "num"],
              ["lint_on_write", lint, "bool-optional"]]);
            const l = collect(lspPairs);
            if (lspChk.get()) l.enabled = true;
            else if (lsp.enabled) l.enabled = false;
            if (Object.keys(l).length) p.lsp = { ...lsp, ...l };
            const w = collect(wfPairs);
            if (Object.keys(w).length) p.web_fetch = { ...wf, ...w };
            return p;
          };
        },
      }));

      // --------------------------------------------------- pipeline: RFT
      cards.push(card({
        title: "RFT (rejection sampling)",
        section: "rft",
        hint: "Turns serving into a data engine: for verifiable prompts, sample " +
          "K completions from the SERVED model, keep only verified-correct " +
          "ones, emit a ShareGPT SFT dataset the pipeline consumes unchanged. " +
          "Runs as job kind `rft` (in-process, VRAM-gated). Judge bar is " +
          "deliberately strict — it is not self-consistency.",
        status: `effective: ${ev("rft_enabled", rft.enabled)} · K=${ev("rft_sample_k", rft.sample_k)} · ` +
          `judge ≥ ${ev("rft_judge_min_score", rft.judge_min_score)} · ` +
          `≤${ev("rft_max_prompts", rft.max_prompts)} prompts/run`,
        build: (body) => {
          const en = boolSelect(rft.enabled);
          const k = numInput(rft.sample_k);
          const judge = numInput(rft.judge_min_score, { step: "0.01" });
          const maxP = numInput(rft.max_prompts);
          const temp = numInput(rft.temperature, { step: "0.05" });
          const maxTok = numInput(rft.max_tokens);
          const maxPt = numInput(rft.max_prompt_tokens);
          const dpo = boolSelect(rft.emit_dpo);
          const dpoCap = numInput(rft.max_dpo_pairs_per_prompt);
          body.replaceChildren(
            row("enabled", en, "master switch — job refuses while off"),
            row("sample_k", k, "candidate completions per prompt"),
            row("judge_min_score", judge, "STRICT pass bar for open-chat judge verification"),
            row("max_prompts", maxP, "prompts sampled per run (GPU-cost bound)"),
            row("temperature", temp, "elevated sampling for diversity"),
            row("max_tokens", maxTok, "completion budget per candidate"),
            row("max_prompt_tokens", maxPt, "exact-token budget; longer contexts skipped"),
            row("emit_dpo", dpo, "also emit kept-vs-failed preference pairs"),
            row("max_dpo_pairs_per_prompt", dpoCap, "cap per prompt (distinct failure reasons first)"));
          return () => collect([
            ["enabled", en, "bool-optional"],
            ["sample_k", k, "num"],
            ["judge_min_score", judge, "num"],
            ["max_prompts", maxP, "num"],
            ["temperature", temp, "num"],
            ["max_tokens", maxTok, "num"],
            ["max_prompt_tokens", maxPt, "num"],
            ["emit_dpo", dpo, "bool-optional"],
            ["max_dpo_pairs_per_prompt", dpoCap, "num"]]);
        },
      }));

      // ------------------------------------------- pipeline: evolution
      cards.push(card({
        title: "Prompt evolution",
        section: "evolution",
        hint: "GEPA-style search over a profile's system prompt using golden " +
          "tasks: deterministic golden checks FIRST (hard-fail eliminates " +
          "without judge calls), judge scoring for passers, top-2 per " +
          "generation. The output is ALWAYS a diff proposal — never " +
          "auto-applied; applying is an explicit gated action in the Profiles " +
          "panel or `toaster_loop evolve apply <job_id>`.",
        status: `effective: ${ev("evolution_enabled", evo.enabled)} · ` +
          `call budget ${ev("evolution_max_call_budget", evo.max_call_budget)}`,
        build: (body) => {
          const en = boolSelect(evo.enabled);
          const budget = numInput(evo.max_call_budget);
          body.replaceChildren(
            row("enabled", en, "master switch — evolve jobs refuse while off"),
            row("max_call_budget", budget, "hard cap on total LLM calls per job"));
          return () => collect([
            ["enabled", en, "bool-optional"],
            ["max_call_budget", budget, "num"]]);
        },
      }));

      // ------------------------------------------- pipeline: shadow eval
      cards.push(card({
        title: "Shadow-baseline (pre-promotion replay)",
        section: "validation",
        hint: "After training, before promotion: replay N recent high-feedback " +
          "traces against BOTH the active model and the candidate, scored by " +
          "the LLM judge (2×N calls). The verdict rides on the registry row " +
          "next to eval_loss. shadow_blocks_promotion=true turns a flagged " +
          "regression into a promotion veto (implement-but-off strict gate).",
        status: `effective: ${ev("shadow_enabled", val.shadow_enabled)} · ` +
          `N=${ev("shadow_replay_n", val.shadow_replay_n)} · ` +
          `regression flag > ${ev("max_shadow_regression_pct", val.max_shadow_regression_pct)}% · ` +
          `blocks promotion: ${ev("shadow_blocks_promotion", val.shadow_blocks_promotion)} · ` +
          `auto-deploy tier ≥ ${ev("min_tier_auto_deploy", val.min_tier_auto_deploy)}`,
        build: (body) => {
          const en = boolSelect(val.shadow_enabled);
          const n = numInput(val.shadow_replay_n);
          const reg = numInput(val.max_shadow_regression_pct, { step: "0.5" });
          const blocks = boolSelect(val.shadow_blocks_promotion);
          const tier = numInput(val.min_tier_auto_deploy, { step: "1" });
          body.replaceChildren(
            row("shadow_enabled", en, "iterate runs shadow eval only when true"),
            row("shadow_replay_n", n, "replay-set size (judge calls = 2×N)"),
            row("max_shadow_regression_pct", reg, "regression tolerance, % of the judge's 0..1 range"),
            row("shadow_blocks_promotion", blocks, "true = flagged regression vetoes promotion"),
            row("min_tier_auto_deploy", tier,
              "min derived tier for UNATTENDED auto-promotion (manual is never gated; 0 = off)"));
          return () => collect([
            ["shadow_enabled", en, "bool-optional"],
            ["shadow_replay_n", n, "num"],
            ["max_shadow_regression_pct", reg, "num"],
            ["shadow_blocks_promotion", blocks, "bool-optional"],
            ["min_tier_auto_deploy", tier, "num"]]);
        },
      }));

      // ------------------------------------------------- automation: auto
      cards.push(card({
        title: "Autonomous training (auto-check)",
        section: "training",
        hint: "Intent flag for the cron-driven loop: `toaster_loop train " +
          "auto-check --trigger` (the exact line is on the Schedule panel) " +
          "trains when the buffer holds ≥ auto_min_traces unclaimed traces " +
          "with mean score ≥ auto_min_score. toaster_loop never schedules itself — " +
          "the trigger stays an external-scheduler concern; this card only " +
          "declares intent and thresholds.",
        status: `effective: ${ev("auto_enabled", tr.auto_enabled)} · ` +
          `≥ ${ev("auto_min_traces", tr.auto_min_traces)} traces · ` +
          `mean ≥ ${ev("auto_min_score", tr.auto_min_score)} · ` +
          `VRAM floor ${ev("min_free_vram_mib", tr.min_free_vram_mib)} MiB · ` +
          `AC required: ${ev("require_ac_power", tr.require_ac_power)}`,
        build: (body) => {
          const en = boolSelect(tr.auto_enabled);
          const minT = numInput(tr.auto_min_traces);
          const minS = numInput(tr.auto_min_score, { step: "0.05" });
          const vram = numInput(tr.min_free_vram_mib, { placeholder: "0 = gate off" });
          const ac = boolSelect(tr.require_ac_power);
          body.replaceChildren(
            row("auto_enabled", en, "intent flag shown by the Schedule panel"),
            row("auto_min_traces", minT, "unclaimed traces required to trigger"),
            row("auto_min_score", minS, "mean outcome score required to trigger"),
            row("min_free_vram_mib", vram, "resource gate floor (0 = VRAM gate off)"),
            row("require_ac_power", ac, "refuse to train on battery (laptops)"),
            row("", el("a", { href: "#/schedule", text: "the cron line → Schedule" })));
          return () => collect([
            ["auto_enabled", en, "bool-optional"],
            ["auto_min_traces", minT, "num"],
            ["auto_min_score", minS, "num"],
            ["min_free_vram_mib", vram, "num"],
            ["require_ac_power", ac, "bool-optional"]]);
        },
      }));

      // ---------------------------------------------- automation: easy
      cards.push(card({
        title: "Easy training (turnkey loop)",
        section: "easy_training",
        hint: "One-profile turnkey loop: binding it to an agent profile stamps " +
          "that profile's chat traces, scopes curation/readiness to them, " +
          "writes the optimal preset (SFT+DPO with failures as preference " +
          "negatives, eval split, regression gate), and lets one click run the " +
          "full iterate loop. auto_deploy hot-swaps the profile's server to " +
          "the promoted GGUF when it clears the gate. The full flow lives on " +
          "the Easy panel.",
        status: `effective: ${ev("easy_enabled", easy.enabled)} · ` +
          `profile=${easy.profile || "—"} · auto_deploy=${ev("easy_auto_deploy", easy.auto_deploy)}`,
        build: (body) => {
          const en = boolSelect(easy.enabled);
          const prof = textInput(easy.profile, "agent profile name");
          const dep = boolSelect(easy.auto_deploy);
          const minT = numInput(easy.min_traces, { placeholder: "share Schedule" });
          const minS = numInput(easy.min_score, { placeholder: "share Schedule" });
          body.replaceChildren(
            row("enabled", en, "master switch — off means zero behavior change"),
            row("profile", prof, "agent profile whose traces feed the loop"),
            row("auto_deploy", dep, "hot-swap the profile's server on promotion"),
            row("min_traces", minT, "override training.auto_min_traces for this loop"),
            row("min_score", minS, "override training.auto_min_score for this loop"),
            row("", el("a", { href: "#/easy", text: "run the loop → Easy" })));
          return () => collect([
            ["enabled", en, "bool-optional"],
            ["profile", prof, "text"],
            ["auto_deploy", dep, "bool-optional"],
            ["min_traces", minT, "num"],
            ["min_score", minS, "num"]]);
        },
      }));

      // ------------------------------------------------- storage: janitor
      cards.push(card({
        title: "Disk janitor",
        section: "janitor",
        hint: "Recoverable archival of aged model artifacts: superseded/" +
          "failed/rolled-back models older than retention_days move to " +
          "<home>/archive/ (restorable — never deleted). Pinned and " +
          "active/candidate/validated models are never touched. Enabling it " +
          "also makes `train auto-check` (the cron entrypoint) archive on " +
          "every check. disk_retention_days separately ages operational files " +
          "(logs, uploads, spillover, checkpoints) — those ARE deleted after " +
          "the window. Preview: `toaster_loop models janitor --dry-run`.",
        status: `effective: auto=${ev("janitor_enabled", jan.enabled)} · ` +
          `model retention ${ev("janitor_retention_days", jan.retention_days)} d · ` +
          `disk retention ${ev("janitor_disk_retention_days", jan.disk_retention_days)} d`,
        build: (body) => {
          const en = boolSelect(jan.enabled);
          const ret = numInput(jan.retention_days);
          const disk = numInput(jan.disk_retention_days, { placeholder: "0 = off" });
          body.replaceChildren(
            row("enabled", en,
              "on = auto-check archives; manual `toaster_loop models janitor` always works"),
            row("retention_days", ret, "model-artifact age threshold (archived, restorable)"),
            row("disk_retention_days", disk,
              "operational-file age before deletion (0 = off; allow-listed dirs only)"));
          return () => collect([
            ["enabled", en, "bool-optional"],
            ["retention_days", ret, "num"],
            ["disk_retention_days", disk, "num"]]);
        },
      }));

      // ------------------------------------------------- header + cards
      const gpuNote = hw && hw.gpus && hw.gpus[0]
        ? `${hw.gpus[0].name} — free ${hw.gpus[0].free_mib ?? "?"} / ${hw.gpus[0].total_mib ?? "?"} MiB`
        : "no GPU detected";

      // Enable-all button: preview the preset diff first (POST with
      // preview:true writes nothing), show it per-key, apply only on
      // confirm. The server applies all-or-nothing through the same
      // validated/backed-up config path as the cards below — see
      // core/presets.py for what is covered and what is deliberately
      // skipped (easy_training loop, janitor, unmeasured GPU knobs).
      async function enableAll() {
        let preview;
        try {
          preview = await api("/api/features/enable_all",
            { method: "POST", body: { preview: true } });
        } catch (e) {
          toast(`${e.code || "error"}: ${e.message}`, "error");
          return;
        }
        const diff = preview.diff || [];
        if (!diff.length) {
          toast("Everything worth enabling is already on.", "ok");
          return;
        }
        const ok = await confirmDialog({
          title: "Enable all recommended features?",
          body: el("div", {},
            el("p", { text:
              "This turns on every opt-in feature judged safe-and-useful. " +
              "Each one stays individually editable below afterwards." }),
            el("p", { class: "hint", text:
              "Skipped on purpose: the easy-training loop (needs a bound " +
              "profile — use the Easy panel), the janitor's auto-archive, " +
              "and GPU knobs without a measured win. LSP/calibration are " +
              "only included when actually available on this machine." }),
            el("p", { class: "hint", text: "This will change:" }),
            el("ul", { class: "hint" }, ...diff.map((d) => el("li", {
              class: "mono",
              text: `${d.section}.${d.key}: ${JSON.stringify(d.from)} → ` +
                `${JSON.stringify(d.to)}` })))),
          confirmLabel: `Enable ${diff.length} setting${diff.length > 1 ? "s" : ""}`,
          danger: false });
        if (!ok) return;
        try {
          const r = await api("/api/features/enable_all",
            { method: "POST", body: {} });
          toast(r.note || "features enabled", "ok");
          location.hash = "#/options"; // re-render with fresh effective state
        } catch (e) { toast(`${e.code || "error"}: ${e.message}`, "error"); }
      }

      mount.replaceChildren(
        el("h2", { text: "Options" }),
        el("p", { class: "hint", text:
          "Every opt-in feature, one place, with what is actually in effect " +
          "on this machine. Blank fields mean “unset — code default” and are " +
          "never sent; each card writes one config section through the same " +
          "validated, backed-up path as Settings. Machine: " + gpuNote + ". " }),
        el("div", { class: "toolbar" },
          el("button", { class: "small primary", text: "★ Enable all recommended",
            title: "One click: skills, playbooks, indexing, RFT, evolution, " +
              "golden mining, shadow eval, routing, watchdog (+ LSP/calibration " +
              "when available). Preview first; easy-training & janitor stay yours.",
            onclick: enableAll }),
          el("a", { href: "#/settings", class: "small", text: "core sections & Raw YAML → Settings" }),
          el("a", { href: "#/tuning", class: "small", text: "measure & sweep → Tuning" }),
          el("a", { href: "#/serve", class: "small", text: "instances & manual load/unload → Serve" })),
        ...cards);
    }
  },
};

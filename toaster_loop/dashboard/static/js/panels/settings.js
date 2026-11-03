// panels/settings.js — config editor (Phase 5d).
// Two modes over GET/PUT /api/config:
//   * structured form — per-section patches (PUT {section, patch}); a hidden
//     "root" section patches top-level serve keys (gguf_path, context, port).
//   * raw YAML      — full-file textarea; server validates before writing.
// Secrets come back redacted as "***": inputs are blanked and only sent when
// the user types something new, so a saved "***" can never round-trip into
// the file. Every successful write creates a .bak backup next to the config.

import { api } from "../api.js";
import { skey, sget } from "../state.js";
import { el, toast, badge } from "../ui.js";
import { runSettingsEditor, samplingEditor } from "../runsettings.js";

const REDACTED = "***";

export default {
  title: "Settings",
  render(mount) {
    const infoCard = el("div", { class: "card" });
    const formHost = el("div");
    const rawHost = el("div");
    let cfg = null, rawText = "", path = "", tokenConfigured = false;

    mount.replaceChildren(
      el("h2", { text: "Settings" }),
      infoCard,
      tabs(), formHost, rawHost);

    function tabs() {
      const formBtn = el("button", { class: "small primary", text: "Structured" });
      const rawBtn = el("button", { class: "small", text: "Raw YAML" });
      const show = (raw) => {
        formHost.classList.toggle("hidden", raw);
        rawHost.classList.toggle("hidden", !raw);
        formBtn.className = "small" + (raw ? "" : " primary");
        rawBtn.className = "small" + (raw ? " primary" : "");
      };
      formBtn.onclick = () => show(false);
      rawBtn.onclick = () => show(true);
      return el("div", { class: "toolbar" }, formBtn, rawBtn);
    }

    // ------------------------------------------------------------- load
    async function load() {
      const d = await api("/api/config");
      cfg = d.config || {};
      rawText = d.raw || "";
      path = d.path || "";
      tokenConfigured = !!d.token_configured;
      infoCard.replaceChildren(
        el("h3", { text: "Config file" }),
        el("div", { class: "field" },
          el("span", { class: "mono hint", text: path || "(no file — defaults)" }),
          badge(tokenConfigured ? "token set" : "no token",
            tokenConfigured ? "done" : "unknown")),
        el("p", { class: "hint", text:
          "Secrets are redacted as “***” by the API; blank fields stay unchanged. " +
          "Every apply writes a .bak backup next to the file first." }));
      renderForm();
      renderRaw();
    }

    // ------------------------------------------------- structured form
    function field(labelText, input, hint) {
      return el("div", { class: "field" }, el("label", { text: labelText }),
        input, hint ? el("span", { class: "hint", text: hint }) : null);
    }

    function numInput(value, ph = "") {
      const i = el("input", { type: "number", step: "any", value: value ?? "", placeholder: ph });
      return i;
    }
    function textInput(value, ph = "") {
      return el("input", { type: "text", value: value === REDACTED ? "" : (value ?? ""), placeholder: ph });
    }
    function boolSelect(value) {
      const s = el("select", {},
        el("option", { value: "", text: "—" }),
        el("option", { value: "true", text: "true" }),
        el("option", { value: "false", text: "false" }));
      if (value === true || value === false) s.value = String(value);
      return s;
    }
    function collect(pairs) {
      // pairs: [key, input, kind]; skip blanks so unchanged secrets/optional
      // fields are never overwritten (patch merges server-side).
      const patch = {};
      for (const [k, inp, kind] of pairs) {
        const v = inp.value;
        if (v === "" || v == null) continue;
        if (kind === "num") patch[k] = Number(v);
        else if (kind === "bool") patch[k] = v === "true";
        else patch[k] = v;
      }
      return patch;
    }

    function goldenEditor(initial) {
      // list editor for validation.golden_tasks: rows of {prompt, expect_contains}
      const host = el("div");
      const rowsHost = el("div");
      let tasks = Array.isArray(initial) ? [...initial] : [];

      function row(t = {}) {
        const promptIn = el("input", { type: "text", value: t.prompt || "",
          placeholder: "prompt" });
        const expectIn = el("input", { type: "text",
          value: (t.expect_contains || []).join(", "),
          placeholder: "expect contains (comma-separated)" });
        promptIn.addEventListener("change", () => { t.prompt = promptIn.value; });
        expectIn.addEventListener("change", () => {
          t.expect_contains = expectIn.value.split(",").map((s) => s.trim()).filter(Boolean);
        });
        const del = el("button", { class: "small danger", text: "remove",
          onclick: () => { tasks = tasks.filter((x) => x !== t); render(); } });
        return el("div", { class: "field" }, promptIn, expectIn, del);
      }
      function render() {
        rowsHost.replaceChildren(...tasks.map(row));
      }
      host.appendChild(rowsHost);
      host.appendChild(el("div", { class: "field" },
        el("button", { class: "small", text: "+ add task",
          onclick: () => { tasks.push({}); render(); } })));
      render();
      return { node: host, get: () => tasks.filter((t) => t.prompt) };
    }

    function sectionCard(title, build) {
      const body = el("div");
      const status = el("span", { class: "hint" });
      const applyBtn = el("button", { class: "small", text: "Apply" });
      const card = el("div", { class: "card" },
        el("h3", {}, title.text, " ", status), body,
        el("div", { class: "field" }, applyBtn));
      const getPatch = build(body);
      applyBtn.addEventListener("click", async () => {
        const patch = getPatch();
        if (!Object.keys(patch).length) {
          status.textContent = "nothing to send"; return;
        }
        status.textContent = "applying…";
        try {
          const r = await api("/api/config", { method: "PUT",
            body: { section: title.section, patch } });
          status.textContent = r.backup ? `saved (backup: ${r.backup})` : "saved";
          toast(`${title.text} saved`, "ok");
          load().catch(() => {});
        } catch (e) {
          status.textContent = e.message;
          toast(`${e.code}: ${e.message}`, "error");
        }
      });
      return card;
    }

    function renderForm() {
      const root = cfg || {};
      const cur = (cfg.curation || {}), tr = (cfg.training || {}),
        va = (cfg.validation || {}), dash = (cfg.dashboard || {});

      // Serve defaults (top-level keys) — the DEFAULT server's config. Named
      // instances created on the Serve page carry their own settings and do
      // not inherit the Model run settings section below.
      const ggufSel = el("select", {},
        el("option", { value: "", text: "— none —" }));
      const ggufHint = el("span", { class: "hint" });
      // Same picker as the Serve page's "model from disk" dropdown
      // (GET /api/model_catalog); keep the saved path selected even when the
      // catalog no longer lists it.
      (async () => {
        try {
          const c = await api("/api/model_catalog");
          for (const m of c.gguf || []) {
            const size = m.size_bytes ? ` · ${(m.size_bytes / 1e9).toFixed(1)} GB` : "";
            ggufSel.appendChild(el("option", { value: m.path,
              text: `${m.name}${m.quant ? " [" + m.quant + "]" : ""}${size}` }));
          }
          if (root.gguf_path &&
              ![...ggufSel.options].some((o) => o.value === root.gguf_path)) {
            ggufSel.appendChild(el("option",
              { value: root.gguf_path, text: root.gguf_path }));
          }
          if (root.gguf_path) ggufSel.value = root.gguf_path;
          ggufHint.textContent = c.exists
            ? `${(c.gguf || []).length} GGUF(s) in ${c.models_dir}`
            : "no model folder configured (settings: models_dir)";
        } catch { /* picker stays minimal; saved value still selectable */ }
      })();
      const modelsDirIn = textInput(root.models_dir,
        "~/.cache/huggingface/hub");
      const ctxIn = numInput(root.context_length);
      const portIn = numInput(root.port, "0 = dynamic");
      const attachIn = textInput(root.attach_url, "http://127.0.0.1:8080/v1");
      const serveCard = sectionCard(Object.assign(
        { text: "Serve defaults (default server)" }, { section: "root" }), (body) => {
        body.replaceChildren(
          field("gguf_path", el("div", {}, ggufSel, ggufHint),
            "base GGUF for serving/training — same list as the Serve page"),
          field("models_dir", modelsDirIn,
            "base model folder scanned for the model pickers " +
            "(HF hub cache layout; empty = HF default cache)"),
          field("context_length", ctxIn),
          field("port", portIn, "dashboard+serve port (0 = auto)"),
          field("attach_url", attachIn, "skip spawning; use this OpenAI-compat URL"));
        return () => collect([
          ["gguf_path", ggufSel, "text"], ["models_dir", modelsDirIn, "text"],
          ["context_length", ctxIn, "num"],
          ["port", portIn, "num"], ["attach_url", attachIn, "text"]]);
      });

      // Serve defaults: inference engine (serve.default_engine). Built-in
      // llama.cpp by default — byte-identical to the pre-engine spawn path.
      // Registered engines come from the Hardware page; only OpenAI-kind
      // ones are spawnable. Patches the `serve` section, not `root`.
      const engineSel = el("select", {},
        el("option", { value: "llama-cpp", text: "llama.cpp (built-in)" }));
      (async () => {
        try {
          const d = await api("/api/engines");
          for (const e of (d.engines || [])) {
            if (e.builtin || e.kind !== "openai") continue;
            engineSel.appendChild(el("option", { value: e.id },
              e.name + (e.installed ? "" : " (not installed)")));
          }
          const cur = (cfg.serve || {}).default_engine;
          if (cur && [...engineSel.options].some((o) => o.value === cur))
            engineSel.value = cur;
        } catch { /* built-in-only select still works */ }
      })();
      const engineCard = sectionCard(Object.assign(
        { text: "Default inference engine" }, { section: "serve" }), (body) => {
        body.replaceChildren(
          field("default_engine", engineSel,
            "engine the default server spawns with — register more on the " +
            "Hardware page; named instances pick theirs on the Serve page"));
        return () => collect([["default_engine", engineSel, "text"]]);
      });

      // Curation section.
      const minScoreIn = numInput(cur.min_score);
      const minTurnsIn = numInput(cur.min_turns);
      const maxTurnsIn = numInput(cur.max_turns);
      const maxTokIn = numInput(cur.max_est_tokens);
      const incFailSel = boolSelect(cur.include_failures);
      const formatsSel = el("select", {}, el("option", { value: "", text: "—" }),
        ...["sft", "dpo", "both"].map((f) => el("option",
          { value: f, text: f, selected: cur.formats === f })));
      const curCard = sectionCard(Object.assign(
        { text: "Curation" }, { section: "curation" }), (body) => {
        body.replaceChildren(
          field("min_score", minScoreIn, "traces below this are dropped"),
          field("min_turns / max_turns", el("span", {}, minTurnsIn, maxTurnsIn)),
          field("max_est_tokens", maxTokIn),
          field("include_failures", incFailSel, "failures only as DPO negatives"),
          field("formats", formatsSel));
        return () => collect([
          ["min_score", minScoreIn, "num"], ["min_turns", minTurnsIn, "num"],
          ["max_turns", maxTurnsIn, "num"], ["max_est_tokens", maxTokIn, "num"],
          ["include_failures", incFailSel, "bool"], ["formats", formatsSel, "text"]]);
      });

      // Training section.
      const baseModelIn = textInput(tr.base_model, "HF id or local path");
      const pythonIn = textInput(tr.python, "/path/to/venv/bin/python");
      const vramIn = numInput(tr.min_free_vram_mib, "0 = gate off");
      const acSel = boolSelect(tr.require_ac_power);
      const autoSel = boolSelect(tr.auto_enabled);
      const autoTracesIn = numInput(tr.auto_min_traces);
      const autoScoreIn = numInput(tr.auto_min_score);
      const trCard = sectionCard(Object.assign(
        { text: "Training" }, { section: "training" }), (body) => {
        body.replaceChildren(
          field("base_model", baseModelIn),
          field("python", pythonIn, "interpreter for GPU workers"),
          field("min_free_vram_mib", vramIn),
          field("require_ac_power", acSel),
          field("auto_enabled", autoSel),
          field("auto_min_traces", autoTracesIn),
          field("auto_min_score", autoScoreIn));
        return () => collect([
          ["base_model", baseModelIn, "text"], ["python", pythonIn, "text"],
          ["min_free_vram_mib", vramIn, "num"], ["require_ac_power", acSel, "bool"],
          ["auto_enabled", autoSel, "bool"], ["auto_min_traces", autoTracesIn, "num"],
          ["auto_min_score", autoScoreIn, "num"]]);
      });

      // Validation section (golden tasks editor).
      const passRateIn = numInput(va.min_pass_rate);
      const golden = goldenEditor(va.golden_tasks);
      const vaCard = sectionCard(Object.assign(
        { text: "Validation" }, { section: "validation" }), (body) => {
        body.replaceChildren(
          field("min_pass_rate", passRateIn),
          el("p", { class: "hint", text:
            "Golden tasks replay after export; promotion only if the pass rate " +
            "clears the threshold. expect_contains is matched case-insensitively." }),
          golden.node);
        return () => {
          const p = collect([["min_pass_rate", passRateIn, "num"]]);
          const tasks = golden.get();
          if (tasks.length) p.golden_tasks = tasks;
          return p;
        };
      });

      // Model run settings (reference-style knobs -> llama-server flags)
      // for the DEFAULT server. Shared editor with the Serve page's per-
      // instance form; vision uses a load toggle + auto-detected/downloaded
      // mmproj next to the GGUF (no path input).
      const rs = cfg.run_settings || {};
      const rsEditor = runSettingsEditor(rs,
        { gguf: () => ggufSel.value || root.gguf_path || "" });
      const rsCard = sectionCard(Object.assign(
        { text: "Model run settings (default server)" },
        { section: "run_settings" }), (body) => {
        body.replaceChildren(
          el("p", { class: "hint", text:
            "Applied when toaster_loop spawns the DEFAULT llama-server (restart the " +
            "server to take effect). Blank/auto fields keep llama.cpp defaults. " +
            "Named instances on the Serve page have their own settings and do " +
            "not inherit these." }),
          rsEditor.node);
        rsEditor.refreshMmproj();
        ggufSel.addEventListener("change", () => rsEditor.refreshMmproj());
        return () => rsEditor.getPatch();
      });

      // Sampling + thinking defaults (injected into every agent chat
      // request). Shared editor with the Serve page's per-instance form so
      // the knobs stay identical everywhere (reference style): sampling
      // numbers plus enable_thinking / reasoning_effort / preserve_thinking.
      const smEditor = samplingEditor(cfg.sampling || {});
      const smCard = sectionCard(Object.assign(
        { text: "Sampling & thinking" }, { section: "sampling" }), (body) => {
        body.replaceChildren(
          el("p", { class: "hint", text:
            "Defaults sent with every agent/chat request (also for attached " +
            "servers). Leave blank to keep server defaults. Named instances " +
            "and profiles can override these on the Serve / Profiles pages." }),
          smEditor.node);
        return () => smEditor.getPatch();
      });

      // Agent: auto-healing tool calls toggle.
      const ag = cfg.agent || {};
      const healSel = boolSelect(ag.heal_tool_calls === undefined
        ? true : ag.heal_tool_calls);
      const agCard = sectionCard(Object.assign(
        { text: "Agent" }, { section: "agent" }), (body) => {
        body.replaceChildren(
          field("Auto-healing tool calls", healSel,
            "rescue tool calls emitted as fenced JSON/XML text (recommended on)"));
        return () => collect([["heal_tool_calls", healSel, "bool"]]);
      });

      // WP12 per-project plans (core/plans.py): master switch + prompt-
      // injection knobs + guarded-loop caps. The task graph itself is edited
      // on the Chat page (Tasks tab); toggling `enabled` here is what
      // un-hides it — while off the plans API answers 404 plans_disabled and
      // the whole Tasks tab + agent plan tools stay hidden ("off means
      // invisible" contract). New chat runs pick up the change; no restart.
      const pl = cfg.plans || {};
      const plEnSel = boolSelect(pl.enabled);
      const plSumSel = boolSelect(pl.inject_summary === undefined
        ? true : pl.inject_summary);
      const plCapIn = numInput(pl.max_inject_chars, "800");
      const plLoopReqSel = boolSelect(!pl.loop
        || pl.loop.require_verification === undefined
        ? true : pl.loop.require_verification);
      const plLoopMaxIn = numInput(pl.loop && pl.loop.max_tasks, "10");
      const plLoopFailIn = numInput(pl.loop && pl.loop.max_consecutive_failures, "1");
      const plCard = sectionCard(Object.assign(
        { text: "Project plans" }, { section: "plans" }), (body) => {
        body.replaceChildren(
          el("p", { class: "hint", text:
            "Per-project task graphs with dependencies (Tasks tab on the Chat " +
            "page; plan tools for agents; `toaster_loop plan …` CLI + MCP plan tools). " +
            "Off = zero behavior change. The loop fields guard the unattended " +
            "loop: `toaster_loop plan loop` / “run loop” on the Chat page." }),
          field("enabled", plEnSel,
            "master switch — on exposes plan tools, injects the plan summary, " +
            "and un-hides the Chat Tasks tab"),
          field("inject_summary", plSumSel,
            "append the bounded “## Project plan” section to run system prompts " +
            "(off keeps the tools but never injects)"),
          field("max_inject_chars", plCapIn,
            "hard cap on that section (lines omitted whole, never sliced)"),
          field("loop.require_verification", plLoopReqSel,
            "a run ending on the verification nudge blocks its task (never auto-done)"),
          field("loop.max_tasks", plLoopMaxIn,
            "tasks per unattended loop invocation"),
          field("loop.max_consecutive_failures", plLoopFailIn,
            "stop the loop after N non-done tasks in a row"));
        return () => {
          const p = collect([
            ["enabled", plEnSel, "bool"],
            ["inject_summary", plSumSel, "bool"],
            ["max_inject_chars", plCapIn, "num"]]);
          // The server replaces the whole `loop` dict on a section patch, so
          // seed it from the CURRENT config and overlay the edited fields —
          // unexposed loop keys survive an apply instead of being silently
          // dropped.
          const loop = Object.assign({}, pl.loop || {});
          const edits = collect([
            ["require_verification", plLoopReqSel, "bool"],
            ["max_tasks", plLoopMaxIn, "num"],
            ["max_consecutive_failures", plLoopFailIn, "num"]]);
          if (Object.keys(loop).length) p.loop = Object.assign(loop, edits);
          return p;
        };
      });

      // Codebase indexing / semantic search (Phase 5 rework):
      //   * embedding model = dropdown from GET /api/model_catalog, filtered
      //     to embedding-capable GGUFs (client-side name heuristic — the
      //     catalog exposes no capability flags; see EMBED_NAME_RE)
      //   * output_dim = MRL truncation select (768/512/256/128)
      //   * toaster_loop spawns/stops its own loopback llama-server --embedding,
      //     so the old embed endpoint url field is gone
      //   * top_k is an internal constant now; paths survive only as the
      //     advanced paths_override for the default (no-project) index —
      //     per-project indexes derive their root from workspace_dir.
      const ix = cfg.indexing || {};
      const EMBED_NAME_RE = /(embed|bge|gte|e5|mxbai|nomic|jina-embed|snowflake-arctic|bce-embedding)/i;
      const iEnSel = boolSelect(ix.enabled);
      const iModelSel = el("select", {},
        el("option", { value: "", text: "— none (lexical BM25 only) —" }));
      const catHint = el("span", { class: "hint" });
      // Populate like serve.js's "model from disk" picker; keep the saved
      // path selected even if the heuristic filtered it out.
      (async () => {
        try {
          const c = await api("/api/model_catalog");
          const all = c.gguf || [];
          const embeds = all.filter((m) => EMBED_NAME_RE.test(`${m.name} ${m.path}`));
          for (const m of embeds) {
            const size = m.size_bytes ? ` · ${(m.size_bytes / 1e9).toFixed(1)} GB` : "";
            iModelSel.appendChild(el("option", { value: m.path,
              text: `${m.name}${m.quant ? " [" + m.quant + "]" : ""}${size}` }));
          }
          if (ix.model && ![...iModelSel.options].some((o) => o.value === ix.model)) {
            iModelSel.appendChild(el("option", { value: ix.model, text: ix.model }));
          }
          if (ix.model) iModelSel.value = ix.model;
          catHint.textContent = c.exists
            ? `${embeds.length} embedding-capable GGUF(s) in ${c.models_dir}`
            : "no model folder configured (settings: models_dir)";
        } catch { /* picker stays empty; lexical-only still works */ }
      })();
      const iDimSel = el("select", {},
        el("option", { value: "", text: "— full model output —" }),
        ...[768, 512, 256, 128].map((d) => el("option",
          { value: String(d), text: `${d} dims`,
            selected: Number(ix.output_dim) === d })));
      if (ix.output_dim) iDimSel.value = String(ix.output_dim);
      const iPathsIn = textInput((ix.paths_override || []).join(", "),
        "advanced override, comma-separated (default index only)");
      const iModeSel = el("select", {}, el("option", { value: "", text: "—" }),
        ...["hybrid", "semantic", "lexical"].map((m) => el("option",
          { value: m, text: m, selected: ix.search_mode === m })));
      // Embed-server launch knobs: on a machine that already runs a chat
      // model (especially unified-memory APUs) the default full GPU offload
      // makes indexing starve the model. 0 GPU layers = CPU embeddings.
      const iGpuIn = numInput(ix.gpu_layers, "auto (full offload)");
      const iThreadsIn = numInput(ix.threads, "llama.cpp default");
      const iBatchIn = numInput(ix.batch_size, "2048");
      const ixStatus = el("span", { class: "hint" });
      const ixBuildBtn = el("button", { class: "small", text: "build index now" });
      ixBuildBtn.addEventListener("click", async () => {
        ixBuildBtn.disabled = true; ixStatus.textContent = "building…";
        try {
          // Build the ACTIVE project's index when one is open (same scoped
          // key the rest of the UI uses — pass 3 #25), else the default DB.
          const body = {};
          const activeProject = sget(skey("ui", "", "active-project"));
          if (activeProject) body.project = activeProject;
          const r = await api("/api/index/build", { method: "POST", body });
          ixStatus.textContent = `build started (${r.project || "default"}) — watch the console drawer`;
        } catch (e) { ixStatus.textContent = e.message; }
        finally { ixBuildBtn.disabled = false; }
      });
      async function refreshIndexStatus() {
        // Status of the ACTIVE project's index when one is open (the build
        // button targets it too), else the default DB. Previously this only
        // ever showed the default index while builds hit the project DB.
        const activeProject = sget(skey("ui", "", "active-project"));
        try {
          const s = await api("/api/index" +
            (activeProject ? "?project=" + encodeURIComponent(activeProject) : ""));
          const label = s.project_id
            ? ((s.roots && s.roots[0]) || s.project_id) : "default";
          ixStatus.textContent = `index (${label}): ${s.files} files · ${s.chunks} chunks · ` +
            `${s.embedded} embedded · embed=${s.embedding_mode}` +
            (s.built_dim ? ` · built@${s.built_dim}d` : "") +
            (s.pending ? ` · ${s.pending} pending` : "") +
            (s.building ? " · building…" : "") +
            (s.error ? ` · ⚠ ${s.error}` : "");
        } catch { ixStatus.textContent = ""; }
      }
      const ixCard = sectionCard(Object.assign(
        { text: "Codebase indexing" }, { section: "indexing" }), (body) => {
        body.replaceChildren(
          el("p", { class: "hint", text:
            "Semantic search of project codebases for the search_codebase tool. " +
            "Each project indexes its own workspace into a per-project database. " +
            "toaster_loop starts and stops its own local embedding server for the " +
            "selected model; without one a BM25 lexical index is used." }),
          field("enabled", iEnSel),
          field("embedding model", el("div", {}, iModelSel, catHint)),
          field("output embedding dimension", iDimSel,
            "Matryoshka truncation (MRL); changing this requires rebuilding indexes"),
          field("search mode", iModeSel),
          field("embedding GPU layers", iGpuIn,
            "-ngl for the embedding server. 0 = run embeddings on the CPU and " +
            "leave the GPU to your chat model (recommended while a model is " +
            "loaded on a shared-memory machine); blank = auto (full offload). " +
            "Applies the next time the embedding server starts — use “stop " +
            "embedding server” to force a restart."),
          field("embedding CPU threads", iThreadsIn,
            "-t for the embedding server; blank = llama.cpp default (all cores)"),
          field("embedding batch size", iBatchIn,
            "-b/-ub of the embedding server (default 2048); smaller uses less " +
            "memory at some throughput cost"),
          field("paths override", iPathsIn,
            "manual roots for the default/no-project index; projects index their workspace_dir"),
          el("div", { class: "field" }, ixBuildBtn, ixStatus));
        return () => {
          const p = collect([["enabled", iEnSel, "bool"],
            ["model", iModelSel, "text"],
            ["output_dim", iDimSel, "num"], ["search_mode", iModeSel, "text"],
            ["gpu_layers", iGpuIn, "num"], ["threads", iThreadsIn, "num"],
            ["batch_size", iBatchIn, "num"]]);
          if (iPathsIn.value.trim())
            p.paths_override = iPathsIn.value.split(",").map((s) => s.trim()).filter(Boolean);
          return p;
        };
      });
      refreshIndexStatus();

      // Persistent skills (core/skills.py): agent-authored procedures under
      // <TOASTER_LOOP_HOME>/skills/. Only name+description index lines are ever
      // injected; bodies are read on demand by the agent. Files stay
      // human-editable — this card toggles exposure + index cap AND is the
      // review surface: list / view / edit / delete via /api/skills.
      const sk = cfg.skills || {};
      const skEnSel = boolSelect(sk.enabled);
      const skCapIn = numInput(sk.index_total_cap_chars, "1500");
      const skList = el("div", { class: "hint" });

      async function loadSkills() {
        skList.textContent = "loading skills…";
        let d;
        try {
          d = await api("/api/skills");
        } catch (e) {
          skList.replaceChildren(
            el("span", { class: "hint", text: `skills API unavailable: ${e.message}` }));
          return;
        }
        const rows = (d.skills || []).map((s) => {
          const nameCell = el("span", { class: "mono", text: s.name });
          const meta = el("span", { class: "hint",
            text: ` ${s.description} · updated ${s.updated || "?"} · ${s.chars} chars` });
          const body = el("div", { class: "hidden" });
          const toggle = el("button", { class: "small", text: "view" });
          const delBtn = el("button", { class: "small", text: "delete" });
          toggle.onclick = async () => {
            if (!body.classList.contains("hidden")) {
              body.classList.add("hidden"); toggle.textContent = "view"; return;
            }
            body.textContent = "loading…"; body.classList.remove("hidden");
            toggle.textContent = "hide";
            try {
              const full = await api(`/api/skills/${encodeURIComponent(s.name)}`);
              renderSkillEditor(body, full, loadSkills);
            } catch (e2) {
              body.replaceChildren(el("span", { class: "hint", text: e2.message }));
            }
          };
          delBtn.onclick = async () => {
            if (!confirm(`Delete skill "${s.name}"? This removes its markdown file.`)) return;
            try {
              await api(`/api/skills/${encodeURIComponent(s.name)}`, { method: "DELETE" });
              toast(`skill ${s.name} deleted`, "ok");
              loadSkills();
            } catch (e2) { toast(e2.message, "err"); }
          };
          return el("div", {}, el("div", {}, nameCell, meta, " ", toggle, " ", delBtn), body);
        });
        skList.replaceChildren(
          ...(rows.length ? rows : [el("span", { class: "hint",
            text: "(no skills saved yet — agents create them with save_skill)" })]),
          el("div", { class: "hint mono", text: `dir: ${d.dir || ""}` }));
      }

      function renderSkillEditor(host, s, onDone) {
        const descIn = el("input", { type: "text", value: s.description || "" });
        const ta = el("textarea", { rows: 14, spellcheck: "false" });
        ta.value = s.content || "";
        const saveBtn = el("button", { class: "small primary", text: "save" });
        saveBtn.onclick = async () => {
          try {
            await api(`/api/skills/${encodeURIComponent(s.name)}`,
              { method: "PUT", body: { description: descIn.value, content: ta.value } });
            toast(`skill ${s.name} saved`, "ok");
            onDone();
          } catch (e2) { toast(e2.message, "err"); }
        };
        host.replaceChildren(
          el("div", { class: "field" }, el("label", { text: "description" }), descIn),
          ta, saveBtn);
      }

      const skCard = sectionCard(Object.assign(
        { text: "Skills (agent memory)" }, { section: "skills" }), (body) => {
        body.replaceChildren(
          el("p", { class: "hint", text:
            "Persistent procedures the agent writes for itself (save_skill) — " +
            "per-app recipes, machine facts, failure gotchas — stored as " +
            "editable markdown under <TOASTER_LOOP_HOME>/skills/. Only a one-line " +
            "index per skill enters prompts; full bodies are read on demand. " +
            "Complements Playbooks (auto-mined from successful traces)." }),
          field("enabled", skEnSel,
            "exposes save_skill/read_skill/list_skills and injects the index"),
          field("index_total_cap_chars", skCapIn,
            "hard cap on the injected skills index (most recently updated win)"),
          el("div", { class: "field" }, el("label", { text: "stored skills" }),
            skList));
        return () => collect([
          ["enabled", skEnSel, "bool"],
          ["index_total_cap_chars", skCapIn, "num"]]);
      });
      loadSkills();

      // Playbooks (core/playbooks.py): injection master switch + caps. The
      // recipe list and promote/retire live in the Profiles panel; this card
      // is what makes playbooks.enabled toggleable without editing YAML
      // (previously the Profiles hint sent users to the config file by hand).
      const pbs = cfg.playbooks || {};
      const pbEnSel = boolSelect(pbs.enabled);
      const pbTotalIn = numInput(pbs.total_cap_chars, "2000");
      const pbPerIn = numInput(pbs.per_recipe_cap_chars, "800");
      const pbCard = sectionCard(Object.assign(
        { text: "Playbooks (mined workflows)" }, { section: "playbooks" }), (body) => {
        body.replaceChildren(
          el("p", { class: "hint", text:
            "Markdown recipes distilled from recurring successful tool chains, " +
            "injected under a hard cap into profile system prompts. Manage " +
            "recipes (scan / promote / retire / assess) from the Profiles panel " +
            "or `toaster_loop playbooks …`." }),
          field("enabled", pbEnSel,
            "OFF = zero behavior change: no store access, prompts unchanged"),
          field("total_cap_chars", pbTotalIn,
            "hard cap on the whole ## Playbooks section (over-cap recipes are omitted whole)"),
          field("per_recipe_cap_chars", pbPerIn,
            "recipes longer than this never enter the prompt"));
        return () => collect([
          ["enabled", pbEnSel, "bool"],
          ["total_cap_chars", pbTotalIn, "num"],
          ["per_recipe_cap_chars", pbPerIn, "num"]]);
      });

      // Golden mining (curate/mine_goldens.py): config knobs + the missing
      // front door — a mine button (POST /api/mine_goldens queues the job)
      // and the mined-suite list with per-golden enable toggles, which is
      // what makes the per-line `enabled` flag usable without hand-editing
      // <home>/goldens/mined.jsonl.
      const gm = cfg.goldens_mine || {};
      const gmEnSel = boolSelect(gm.enabled);
      const gmMaxIn = numInput(gm.max_per_run, "8");
      const gmCandIn = numInput(gm.max_candidates, "64");
      const gmScoreIn = numInput(gm.min_feedback_score, "1.0");
      const gmJudgeSel = boolSelect(gm.use_judge);
      const gmMineBtn = el("button", { class: "small primary", text: "mine now" });
      const gmList = el("div", { class: "hint" });

      async function loadMined() {
        gmList.textContent = "loading mined goldens…";
        let d;
        try {
          d = await api("/api/goldens/mined");
        } catch (e) {
          gmList.replaceChildren(el("span", { class: "hint",
            text: `mined-goldens API unavailable: ${e.message}` }));
          return;
        }
        const rows = (d.goldens || []).map((g) => {
          const on = g.enabled !== false;
          const tgl = el("button", { class: "small", text: on ? "disable" : "enable" });
          const prompt = String(g.prompt || "").slice(0, 110);
          const src = (g._mine && g._mine.source_trace_id) || "?";
          tgl.onclick = async () => {
            try {
              await api("/api/goldens/mined/toggle",
                { method: "POST", body: { key: g.key, enabled: !on } });
              toast(`golden ${on ? "disabled" : "enabled"}`, "ok");
              loadMined();
            } catch (e2) { toast(e2.message, "err"); }
          };
          return el("div", {},
            el("span", { class: on ? "mono" : "mono hint", text: prompt }),
            el("span", { class: "hint",
              text: ` · from trace ${src} · checks: ${((g._mine && g._mine.verifiers) || []).join(", ") || "—"}` }),
            " ", tgl);
        });
        gmList.replaceChildren(
          ...(rows.length ? rows : [el("span", { class: "hint",
            text: "(nothing mined yet — 👍 a trace, then mine)" })]));
      }

      gmMineBtn.onclick = async () => {
        gmMineBtn.disabled = true;
        try {
          const r = await api("/api/mine_goldens", { method: "POST", body: {} });
          toast(`queued ${r.job_id || "mine_goldens job"} — watch the Jobs panel`, "ok");
          setTimeout(loadMined, 3000);
        } catch (e) { toast(e.message, "err"); }
        finally { gmMineBtn.disabled = false; }
      };

      const gmCard = sectionCard(Object.assign(
        { text: "Golden mining (👍 → regression suite)" },
        { section: "goldens_mine" }), (body) => {
        body.replaceChildren(
          el("p", { class: "hint", text:
            "Distills 👍-thumbed traces into golden validation tasks with " +
            "deterministic self-checked verifiers. Mined goldens join the " +
            "promotion gate and RFT prompt selection automatically." }),
          field("enabled", gmEnSel, "master switch — job refuses while off"),
          field("max_per_run", gmMaxIn),
          field("max_candidates", gmCandIn),
          field("min_feedback_score", gmScoreIn),
          field("use_judge", gmJudgeSel,
            "judge may propose extra substrings; proposals face the same self-check"),
          el("div", { class: "field" }, el("label", { text: "actions" }), gmMineBtn),
          el("div", { class: "field" }, el("label", { text: "mined suite" }), gmList));
        return () => collect([
          ["enabled", gmEnSel, "bool"],
          ["max_per_run", gmMaxIn, "num"],
          ["max_candidates", gmCandIn, "num"],
          ["min_feedback_score", gmScoreIn, "num"],
          ["use_judge", gmJudgeSel, "bool"]]);
      });
      loadMined();

      // Dashboard token (secret: blank = unchanged) + LAN exposure toggle.
      const tokIn = el("input", { type: "password", placeholder:
        tokenConfigured ? "•••••• (set — leave blank to keep)" : "new token" });
      const lanSel = boolSelect((cfg.dashboard || {}).expose_lan);
      const dashCard = sectionCard(Object.assign(
        { text: "Dashboard" }, { section: "dashboard" }), (body) => {
        body.replaceChildren(
          field("token", tokIn,
            "required for non-loopback-ish requests; never echoed back"),
          el("p", { class: "hint", text:
            "Set a value to replace the token; clearing is not offered from the " +
            "UI (edit the config file directly to remove it)." }),
          field("expose to local network", lanSel,
            "OFF = bind 127.0.0.1 only (this machine). ON = bind 0.0.0.0 so " +
            "other devices on your LAN can open the dashboard — restart the " +
            "dashboard to apply; the LAN URL prints at startup. Requires an " +
            "auth token (one is generated by default); boot refuses without " +
            "it. Anyone on the network who has the token gets full control, " +
            "so only enable this on networks you trust."));
        return () => collect([["token", tokIn, "text"],
          ["expose_lan", lanSel, "bool"]]);
      });

      formHost.replaceChildren(serveCard, engineCard, rsCard, smCard, agCard, curCard,
        trCard, vaCard, plCard, ixCard, skCard, pbCard, gmCard, dashCard);
    }

    // -------------------------------------------------------- raw YAML
    function renderRaw() {
      const ta = el("textarea", { rows: 24, spellcheck: "false" });
      ta.value = rawText;
      const status = el("span", { class: "hint" });
      const applyBtn = el("button", { class: "small primary", text: "Validate & apply" });
      const revertBtn = el("button", { class: "small", text: "revert" });
      applyBtn.addEventListener("click", async () => {
        applyBtn.disabled = true;
        status.textContent = "";
        try {
          const r = await api("/api/config", { method: "PUT", body: { yaml: ta.value } });
          status.textContent = `applied (backup: ${r.backup || "none yet"})`;
          toast("config applied", "ok");
          load().catch(() => {});
        } catch (e) {
          status.textContent = "";
          errPre.textContent = `${e.code}: ${e.message}`;
          errPre.classList.remove("hidden");
        } finally { applyBtn.disabled = false; }
      });
      revertBtn.addEventListener("click", () => { ta.value = rawText; status.textContent = "reverted"; });
      const errPre = el("pre", { class: "err hidden" });
      rawHost.replaceChildren(
        el("div", { class: "card" },
          el("h3", { text: "Raw YAML" }),
          el("p", { class: "hint", text:
            `Full config at ${path}. Unknown keys and YAML syntax errors are ` +
            "rejected before the file is touched; a .bak backup is written on apply." }),
          ta,
          el("div", { class: "field" }, applyBtn, revertBtn, status),
          errPre));
    }

    load().catch((e) => infoCard.replaceChildren(el("p", { class: "err", text: e.message })));
  },
};

// panels/profiles.js — agent profiles (Phase 5e).
// Profiles are named runner configurations {name, system_prompt, tools[],
// model_name?} stored in the config file (`profiles:` + `default_profile`),
// managed via GET/PUT/DELETE /api/profiles*. Chat (agent mode) picks one per
// run; the selected profile's system prompt and tool subset are applied by
// /api/chat/stream on both runner backends.

import { api } from "../api.js";
import { el, toast, badge, empty, confirmDialog, copyButton } from "../ui.js";

export default {
  title: "Profiles",
  render(mount) {
    let profiles = [];
    let defaultProfile = null;
    let knownTools = [];
    let servers = []; // Phase 6: managed server instances (server_instance binding)
    // Per-agent capabilities: {key: gated_tool_name} from the API, plus the
    // global agent.capabilities defaults (effective state when a profile is
    // silent on a toggle).
    let capabilityToggles = {
      internet: "web_fetch", run_commands: "run_command",
      vision: "view_image", screenshots: "take_screenshot",
    };
    let globalCaps = {};
    const CAP_LABELS = {
      internet: "internet (web_fetch)",
      run_commands: "run programs (run_command)",
      vision: "see images (view_image)",
      screenshots: "screen capture (take_screenshot)",
    };
    let editing = null; // name being edited, or "new"
    // Pass 4 #38: distilled playbooks (GET /api/playbooks). Compact section —
    // recipes live in the buffer DB; promote/retire are explicit POST actions.
    let playbooks = [];
    let pbChains = [];
    let pbEnabled = false;
    // Pass 4 #39: latest evolve proposal per profile ({profileName: proposal}).
    // Applying is a gated explicit action — never automatic.
    let proposals = {};

    const listHost = el("div");
    const editorHost = el("div");
    const playbookHost = el("div");
    const proposalHost = el("div");

    async function load() {
      const d = await api("/api/profiles");
      profiles = d.profiles || [];
      defaultProfile = d.default_profile || null;
      knownTools = d.known_tools || [];
      if (d.capability_toggles) capabilityToggles = d.capability_toggles;
      globalCaps = d.global_capabilities || {};
      try {
        const s = await api("/api/servers");
        servers = s.servers || [];
      } catch { servers = []; }
      renderList();
      loadPlaybooks().catch(() => {});
      loadProposals().catch(() => {});
    }

    // ------------------------------------------ prompt proposals (pass 4 #39)
    async function loadProposals() {
      const next = {};
      for (const p of profiles) {
        try {
          const d = await api("/api/evolve?profile=" + encodeURIComponent(p.name));
          if (d.proposal && !d.proposal.dismissed && !d.proposal.applied_at) {
            next[p.name] = d.proposal;
          }
        } catch { /* panel stays honest when the endpoint errors */ }
      }
      proposals = next;
      renderProposals();
    }

    function fmtFit(v) {
      return typeof v === "number" ? v.toFixed(3) : "–";
    }

    async function evolveAction(path, job, label) {
      try {
        await api(path, { method: "POST", body: { job_id: job } });
        toast(`${label} ${job}`, "ok");
        await load();
      } catch (e) { toast(e.message, "error"); }
    }

    function renderProposals() {
      const cards = [];
      for (const [name, pr] of Object.entries(proposals)) {
        if (!pr.improved) continue; // nothing beat the current prompt
        cards.push(el("div", { class: "card" },
          el("h3", { text: `Prompt proposal — ${name}` }),
          el("p", { class: "hint", text:
            `From evolve job ${pr.job_id} (${pr.ts || "?"}, variant ` +
            `${pr.best_variant_id || "?"}${pr.best_direction ? ", " +
            pr.best_direction : ""}); LLM calls ${pr.calls_used}/${pr.call_budget}` +
            (pr.partial ? " — PARTIAL: budget exhausted" : "") +
            ". Applying rewrites the profile's system prompt; the previous " +
            "prompt is kept in the profile entry. Never applied automatically." }),
          el("p", { class: "hint mono", text:
            `fitness current=${fmtFit(pr.fitness_current)} → ` +
            `proposed=${fmtFit(pr.fitness_proposed)} · golden ` +
            `${pr.golden_pass_current ? "pass" : "fail"} → ` +
            `${pr.golden_pass_proposed ? "pass" : "fail"}` }),
          el("pre", { class: "mono", style:
            "max-height:260px;overflow:auto;white-space:pre-wrap", text:
            pr.diff || "(no textual diff)" }),
          el("div", { class: "btnrow" },
            el("button", { class: "primary small", text: "Apply",
              onclick: async () => {
                const ok = await confirmDialog({
                  title: `Apply prompt proposal to “${name}”?`,
                  body: "The profile's system prompt is replaced by the " +
                    "evolved variant. The previous prompt is kept in the " +
                    "profile entry (system_prompt_prev) and a config backup " +
                    "is written. This never happens on its own.",
                  confirmLabel: "Apply", danger: false });
                if (!ok) return;
                evolveAction("/api/evolve/apply", pr.job_id, "applied");
              } }),
            el("button", { class: "small", text: "Dismiss",
              onclick: () => evolveAction("/api/evolve/dismiss", pr.job_id,
                "dismissed") }))));
      }
      proposalHost.replaceChildren(...cards);
    }

    // ------------------------------------------------ playbooks (pass 4 #38)
    async function loadPlaybooks() {
      const d = await api("/api/playbooks");
      playbooks = d.playbooks || [];
      pbChains = d.chains || [];
      pbEnabled = !!d.enabled;
      renderPlaybooks();
    }

    async function pbAction(path, id, label) {
      try {
        await api(path, { method: "POST", body: { id } });
        toast(`${label} ${id}`, "ok");
        loadPlaybooks().catch(() => {});
      } catch (e) { toast(e.message, "error"); }
    }

    async function pbAssess(id) {
      try {
        const d = await api("/api/playbooks/assess",
          { method: "POST", body: { id } });
        const r = (d.results || [])[0] || {};
        const f = (v) => typeof v === "number" ? v.toFixed(3) : "n/a";
        toast(r.demoted
          ? `${id}: regression (${f(r.mean_with)} vs ${f(r.mean_without)}) — retired`
          : `${id}: with ${f(r.mean_with)} / without ${f(r.mean_without)} — stays active`,
          r.demoted ? "warn" : "ok");
        loadPlaybooks().catch(() => {});
      } catch (e) { toast(e.message, "error"); }
    }

    function renderPlaybooks() {
      const rows = [];
      for (const p of playbooks) {
        const actions = [];
        if (p.status !== "active") {
          actions.push(el("button", { class: "small", text: "promote",
            title: "explicit confirmation — run `toaster_loop playbooks promote " +
              p.id + " --check` for the judge replay gate",
            onclick: () => pbAction("/api/playbooks/promote", p.id, "promoted") }));
        }
        if (p.status === "active") {
          actions.push(el("button", { class: "small", text: "assess",
            title: "correlational regression check: mean outcome score of " +
              "matching traces after vs before promotion; a clear regression " +
              "retires the playbook",
            onclick: () => pbAssess(p.id) }));
        }
        if (p.status !== "retired") {
          actions.push(el("button", { class: "small danger", text: "retire",
            onclick: () => pbAction("/api/playbooks/retire", p.id, "retired") }));
        }
        rows.push(el("tr", {},
          el("td", { class: "mono hint" }, p.id),
          el("td", {}, badge(p.status, p.status === "active" ? "done" :
            p.status === "candidate" ? "warn" : "idle")),
          el("td", { class: "hint mono" }, p.profile || "global"),
          el("td", { class: "mono hint", title: (p.text || "").slice(0, 400) },
            (p.chain || []).join(" → ")),
          el("td", {}, el("div", { class: "btnrow" }, ...actions))));
      }
      const children = [
        el("h3", { text: "Playbooks" }),
        el("p", { class: "hint", text:
          (pbEnabled
            ? "Recipes distilled from recurring successful tool chains, " +
              "injected into profile prompts under a cap."
            : "Distilled recipes exist, but injection is OFF — set " +
              "playbooks.enabled in the config to activate them.") }),
      ];
      if (rows.length) {
        children.push(el("table", {},
          el("thead", {}, el("tr", {},
            el("th", { text: "Id" }), el("th", { text: "Status" }),
            el("th", { text: "Profile" }), el("th", { text: "Chain" }),
            el("th", { text: "" }))),
          el("tbody", {}, ...rows)));
      } else if (pbChains.length) {
        children.push(empty(el("span", {},
          `No playbooks yet — ${pbChains.length} recurring tool chain(s) found; ` +
          "run \u201Csable playbooks scan --create\u201D to distill them.")));
      } else {
        children.push(empty(el("span", {},
          "No recurring tool chains yet — playbooks distill workflows that " +
          "succeed repeatedly across your traces.")));
      }
      playbookHost.replaceChildren(el("div", { class: "card" }, ...children));
    }

    // Effective capability state for a profile (profile override > global).
    function effCap(p, key) {
      const own = (p && p.capabilities) || {};
      return key in own ? !!own[key] : !!globalCaps[key];
    }

    function renderList() {
      if (!profiles.length) {
        listHost.replaceChildren(empty(
          el("span", {}, "No agent profiles yet — create one to give the agent ",
            "a persona (system prompt), a tool subset, or a model alias. ",
            "Chat in agent mode can then pick it per run.")));
        return;
      }
      listHost.replaceChildren(
        el("table", {},
          el("thead", {}, el("tr", {},
            el("th", { text: "Name" }), el("th", { text: "System prompt" }),
            el("th", { text: "Tools" }), el("th", { text: "Model" }),
            el("th", { text: "Server" }), el("th", { text: "Reach" }),
            el("th", { text: "" }))),
          el("tbody", {}, ...profiles.map((p) => {
            const tools = (p.tools || []);
            // Remote LAN agents live in the peer's config, not ours: no
            // editor, no delete — they appear here read-only (managed on
            // the Serve page's remote-peers card).
            if (p.remote) {
              return el("tr", {},
                el("td", {}, el("span", { class: "mono", text: p.name }),
                  " ", badge("remote", "partial")),
                el("td", { class: "hint" },
                  "profile on the peer machine — edit it there"),
                el("td", { class: "hint mono" }, "peer's tools"),
                el("td", { class: "hint mono" }, p.model_name || "–"),
                el("td", { class: "hint mono" }, "remote"),
                el("td", { class: "hint" }, "remote inference"),
                el("td", {}, ""));
            }
            const actions = [];
            if (p.name !== defaultProfile) {
              actions.push(el("button", { class: "small", text: "set default",
                title: "Use this profile when Chat does not pick one explicitly.",
                onclick: async () => {
                  try {
                    await api("/api/profiles/default", { method: "PUT",
                      body: { default: p.name } });
                    toast(`default profile → ${p.name}`, "ok");
                    load().catch(() => {});
                  } catch (e) { toast(e.message, "error"); }
                } }));
            }
            actions.push(el("button", { class: "small danger", text: "delete",
              title: "Remove this profile from the config. Existing traces keep their tags.",
              onclick: async () => {
                const ok = await confirmDialog({
                  title: `Delete profile “${p.name}”?`,
                  body: defaultProfile === p.name
                    ? "This is the default profile; the default will be cleared."
                    : "Existing traces keep their tags; nothing else changes.",
                  confirmLabel: "Delete", danger: true });
                if (!ok) return;
                try {
                  await api("/api/profiles/" + encodeURIComponent(p.name),
                    { method: "DELETE" });
                  toast(`deleted ${p.name}`, "info");
                  if (editing === p.name) editing = null;
                  load().catch(() => {});
                } catch (e) { toast(e.message, "error"); }
              } }));
            return el("tr", {},
              el("td", {}, el("span", { class: "mono", text: p.name }),
                p.name === defaultProfile ? badge("default", "done") : null),
              el("td", { class: "hint" },
                (p.system_prompt || "").slice(0, 80) +
                ((p.system_prompt || "").length > 80 ? "…" : "") || "–"),
              el("td", { class: "hint mono" }, tools.length ? tools.join(", ") : "all"),
              el("td", { class: "hint mono" }, p.model_name || "–"),
              el("td", { class: "hint mono" }, p.server_instance || "default"),
              el("td", { class: "hint mono" }, (() => {
                const on = Object.keys(capabilityToggles).filter(
                  (k) => effCap(p, k));
                return on.length ? on.map((k) => capabilityToggles[k]).join(", ")
                  : "local only";
              })()),
              el("td", {}, el("div", { class: "btnrow" },
                el("button", { class: "small", text: "edit",
                  title: "Change this profile's prompt, tools, model and server binding.",
                  onclick: () => { editing = p.name; renderEditor(); } }),
                ...actions)));
          }))));
    }

    // ---------------------------------------------------------------- editor
    function renderEditor() {
      const isNew = editing === "new";
      const p = isNew ? { name: "", system_prompt: "", tools: [], model_name: "" }
        : (profiles.find((x) => x.name === editing) ||
           { name: editing, system_prompt: "", tools: [], model_name: "" });

      const nameIn = el("input", { type: "text", value: p.name || "",
        placeholder: "coder", disabled: isNew ? null : "disabled",
        title: "Unique id for this profile — shown in the Chat profile picker " +
          "and stamped on every trace it produces." });
      const promptTa = el("textarea", { rows: 8, spellcheck: "false",
        placeholder: "You are a careful coding assistant… (system prompt)",
        title: "Instructions prepended to every run of this agent: persona, " +
          "style, rules. This is the main lever for what the agent does." });
      promptTa.value = p.system_prompt || "";
      const modelIn = el("input", { type: "text", value: p.model_name || "",
        placeholder: "(served model)",
        title: "Ask for a specific model alias by name; blank = whatever model " +
          "is currently served on the bound server." });

      // Phase 6: optional server-instance binding — runs for this profile
      // target that managed instance instead of the default serve endpoint.
      const curInstance = p.server_instance || "";
      const instSel = el("select", { title:
        "Send this profile's requests to a named server instance (Serve panel) " +
        "instead of the default one — lets agents run on different models at once." },
        el("option", { value: "", text: "default (whatever is served)" }));
      for (const s of servers) {
        if (s.id === "default") continue;
        const opt = el("option", { value: s.id,
          text: `${s.name} (${s.role || "custom"}${s.running ? "" : ", stopped"})` });
        if (s.id === curInstance) opt.selected = true;
        instSel.appendChild(opt);
      }
      if (curInstance && ![...instSel.options].some((o) => o.value === curInstance)) {
        const opt = el("option", { value: curInstance,
          text: `${curInstance} (unknown instance)` });
        opt.selected = true;
        instSel.appendChild(opt);
      }

      // Tool checkboxes: known set + any extra names already stored.
      const extras = (p.tools || []).filter((t) => !knownTools.includes(t));
      const toolNames = [...knownTools, ...extras];
      const checks = new Map();
      for (const t of toolNames) {
        const cb = el("input", { type: "checkbox" });
        cb.checked = (p.tools || []).includes(t);
        checks.set(t, cb);
      }
      const extraIn = el("input", { type: "text", placeholder: "extra tool name",
        title: "Add a tool by name (e.g. an MCP-provided tool) when it is not in the list." });
      const toolsGrid = el("div", { class: "btnrow" },
        ...[...checks.entries()].map(([t, cb]) =>
          el("label", { class: "chk", title: knownTools.includes(t) ? "" :
            "not a built-in tool name — will not be exposed to the agent" },
            cb, " " + t)));

      // Capability toggles (per-agent reach-out): separate from the tools grid
      // because enabling one GRANTS the gated tool regardless of checkboxes.
      // Checked state = effective value (profile override > global default);
      // saving writes an explicit per-profile map.
      const capChecks = new Map();
      for (const key of Object.keys(capabilityToggles)) {
        const cb = el("input", { type: "checkbox" });
        cb.checked = effCap(p, key);
        capChecks.set(key, cb);
      }
      const capsGrid = el("div", { class: "btnrow" },
        ...[...capChecks.entries()].map(([key, cb]) =>
          el("label", { class: "chk", title: ({
            internet:
              "fetch public web pages (SSRF-guarded http/https only)",
            run_commands:
              "execute local shell commands with your user privileges — " +
              "workspace containment does NOT apply; trusted agents only",
            vision:
              "look at image files you reference (requires a vision model " +
              "served with an mmproj projector — Settings ▸ run settings)",
            screenshots:
              "capture the screen and look at it (grim on Hyprland/wlroots)",
          })[key] || "" },
            cb, " " + (CAP_LABELS[key] || key))));

      // Sampling + thinking overrides moved to the Serve panel's per-instance
      // run-settings dialog (they belong to the server, not the agent). The
      // backend still honors profile.sampling for existing profiles.
      const samplingPassthrough = p.sampling || null;

      const status = el("span", { class: "hint" });
      const saveBtn = el("button", { class: "primary", text: isNew ? "Create" : "Save",
        title: "Write this profile into the config file. New chat runs pick it up immediately." });
      saveBtn.addEventListener("click", async () => {
        const name = nameIn.value.trim();
        if (!name) return toast("name required", "error");
        const body = {
          system_prompt: promptTa.value,
          tools: [...checks.entries()].filter(([, cb]) => cb.checked).map(([t]) => t),
          model_name: modelIn.value.trim() || null,
          server_instance: instSel.value || null,
          // Always send the full map so unchecking a globally-on capability
          // stores an explicit per-profile false.
          capabilities: Object.fromEntries(
            [...capChecks.entries()].map(([k, cb]) => [k, cb.checked])),
          // Preserve any previously stored overrides untouched (no UI here).
          sampling: samplingPassthrough,
        };
        saveBtn.disabled = true;
        try {
          await api("/api/profiles/" + encodeURIComponent(name),
            { method: "PUT", body });
          toast(`profile ${name} saved`, "ok");
          editing = null;
          await load();
          renderEditorHide();
        } catch (e) {
          status.textContent = e.message;
          toast(`${e.code}: ${e.message}`, "error");
        } finally { saveBtn.disabled = false; }
      });

      editorHost.replaceChildren(el("div", { class: "card" },
        el("h3", { text: isNew ? "New profile" : `Edit “${p.name}”` }),
        el("div", { class: "field" }, el("label", { text: "name" }), nameIn),
        el("div", { class: "field" }, el("label", { text: "system prompt" }), promptTa),
        el("div", { class: "field" }, el("label", { text: "tools" }), toolsGrid,
          el("div", { class: "btnrow", style: "margin-top:6px" }, extraIn,
            el("button", { class: "small", text: "+ add tool name",
              title: "Add a custom/MCP tool name to the checkbox list.",
              onclick: () => {
                const t = extraIn.value.trim();
                if (!t || checks.has(t)) return;
                const cb = el("input", { type: "checkbox" });
                cb.checked = true;
                checks.set(t, cb);
                toolsGrid.appendChild(el("label", { class: "chk",
                  title: "not a built-in tool name — will not be exposed to the agent" },
                  cb, " " + t));
                extraIn.value = "";
              } }))),
        el("div", { class: "field" }, el("label", { text: "model alias override" }), modelIn,
          el("span", { class: "hint", text:
            "leave blank to use whatever model is currently served" })),
        el("div", { class: "field" }, el("label", { text: "server instance" }), instSel,
          el("span", { class: "hint", text:
            "run this profile against a named server (Serve panel) — e.g. an " +
            "orchestrator model here while another profile codes on a second" })),
        el("div", { class: "field" }, el("label", { text: "capabilities" }), capsGrid,
          el("span", { class: "hint", text:
            "what this agent may reach beyond the workspace — off by default; " +
            "run programs executes with your own privileges and bypasses " +
            "workspace containment, so grant it only to agents you trust" })),
        el("div", { class: "field" }, saveBtn,
          el("button", { text: "cancel", title: "Discard changes and close the editor.",
            onclick: renderEditorHide }), status)));
      editorHost.scrollIntoView({ block: "nearest" });
    }

    function renderEditorHide() { editorHost.replaceChildren(); }

    mount.replaceChildren(
      el("h2", { text: "Profiles" }),
      el("p", { class: "hint", text:
        "Agent profiles are named runner configurations (system prompt, tool " +
        "subset, optional model alias) stored in the config file. Chat picks " +
        "one per run; the default is used when none is selected." }),
      el("div", { class: "toolbar" },
        el("button", { class: "primary", text: "＋ New profile",
          title: "Create an agent profile: a named system prompt + tool subset " +
            "+ optional model/server binding, selectable per chat run.",
          onclick: () => { editing = "new"; renderEditor(); } }),
        copyButton("curl -s -X PUT " + location.origin + "/api/profiles/coder " +
          "-H 'Content-Type: application/json' " +
          '-d \'{"system_prompt":"You are a careful coding assistant.",' +
          '"tools":["read_file","list_dir"]}\'',
          "copy curl example")),
      listHost, editorHost, proposalHost, playbookHost);

    load().catch((e) => listHost.replaceChildren(
      el("p", { class: "err", text: e.message })));
  },
};

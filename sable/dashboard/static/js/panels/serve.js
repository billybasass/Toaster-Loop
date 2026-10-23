// panels/serve.js — serving controls: multi-instance list (Phase 6), status
// card, spawn from GGUF / active model, deploy a registry model, attach an
// external endpoint, stop/detach. The legacy single-server cards drive the
// reserved "default" instance; the instances table manages N named servers.

import { api } from "../api.js";
import { bus } from "../bus.js";
import { el, toast, badge, confirmDialog, copyButton, fmtTime,
  fmtUptime, empty } from "../ui.js";
import { runSettingsEditor, samplingEditor, imageSettingsEditor }
  from "../runsettings.js";

export default {
  title: "Serve",
  render(mount) {
    const statusCard = el("div", { class: "card serve-status" });
    const errBox = el("div", { class: "inline-error hidden" });
    let models = [];
    let status = { running: false, mode: "none" };
    let servers = [];
    let catalog = { gguf: [], exists: false, models_dir: "" };
    let engines = [];   // registered inference engines (GET /api/engines)
    let unsubs = [];

    function showErr(e) {
      errBox.classList.remove("hidden");
      errBox.replaceChildren(
        el("strong", { text: String(e.code || "error") }),
        document.createTextNode(" — " + (e.message || String(e))));
    }
    function clearErr() { errBox.classList.add("hidden"); errBox.replaceChildren(); }

    // ---------------------------------------------------------- status card
    function renderStatus() {
      statusCard.replaceChildren(el("h3", { text: "Default server status" }));
      if (!status.running && status.mode === "none") {
        statusCard.appendChild(empty("No default server running. Start the " +
          "“default” instance in the table below (its model and engine come " +
          "from the Settings page), attach an endpoint via a named instance, " +
          "or connect a remote sable machine below."));
        return;
      }
      const modeBadge = badge(status.running ? status.mode : "down",
        status.running ? (status.mode === "attach" ? "partial" : "done")
          : "failed");
      const rows = [
        ["state", el("span", {}, modeBadge,
          status.running ? "" : " — server state file exists but the target " +
          "is not responding")],
        ["base_url", el("span", { class: "mono" },
          status.base_url || "–",
          status.base_url ? copyButton(status.base_url) : "")],
        ["model", status.model_name || "–"],
      ];
      if (status.mode === "spawn") {
        rows.push(["pid", String(status.pid ?? "–")]);
        rows.push(["gguf", el("span", { class: "mono path" },
          status.gguf_path || "–")]);
        if (status.model_id) rows.push(["model id", status.model_id]);
      } else if (status.mode === "attach") {
        rows.push(["attach url", el("span", { class: "mono path" },
          status.attach_url || "–")]);
      }
      if (status.started_at)
        rows.push(["uptime", `${fmtTime(status.started_at, false)} ` +
          `(${fmtUptime(status.started_at)})`]);
      const tbl = el("table", { class: "kv" }, el("tbody", {},
        ...rows.map(([k, v]) => el("tr", {},
          el("td", { class: "k", text: k }),
          el("td", {}, typeof v === "string" ? document.createTextNode(v) : v)))));
      statusCard.appendChild(tbl);

      const stopLabel = status.mode === "attach" ? "Detach" : "Stop server";
      const stopBtn = el("button", { class: "danger", text: stopLabel });
      stopBtn.addEventListener("click", async () => {
        const detach = status.mode === "attach";
        const ok = await confirmDialog({
          title: stopLabel,
          body: detach
            ? "Detach only forgets the attached endpoint — the external " +
              "server keeps running."
            : "Stop the llama-server process sable started.",
          confirmLabel: stopLabel, danger: true,
        });
        if (!ok) return;
        try {
          const r = await api("/api/serve/stop", { method: "POST", body: {} });
          toast(detach ? "detached"
            : (r.stopped ? "server stopped" : "nothing was running"), "ok");
          await refresh();
        } catch (e) { showErr(e); }
      });
      statusCard.appendChild(el("div", { class: "btnrow" }, stopBtn));
    }

    async function refresh() {
      try {
        status = await api("/api/serve/status");
        renderStatus();
      } catch (e) { showErr(e); }
    }

    // ------------------------------------------------------ instances table
    const instancesHost = el("div", { class: "card" });

    async function loadInstances() {
      try {
        const d = await api("/api/servers");
        servers = d.servers || [];
        renderInstances();
      } catch (e) { showErr(e); }
    }

    async function startServer(s) {
      try {
        const r = await api(`/api/servers/${encodeURIComponent(s.id)}/start`,
          { method: "POST", body: {} });
        if (r.warning) {
          // VRAM advisory: warn, don't block — offer force via confirm.
          const go = await confirmDialog({
            title: "VRAM warning",
            body: r.warning + "\n\nStart anyway?",
            confirmLabel: "Start anyway", danger: true });
          if (!go) return;
          const f = await api(`/api/servers/${encodeURIComponent(s.id)}/start`,
            { method: "POST", body: { force: true } });
          toast(`starting ${f.model_name || s.name} at ${f.base_url}`, "ok");
        } else {
          toast(`starting ${r.model_name || s.name} at ${r.base_url}`, "ok");
        }
      } catch (e) { showErr(e); }
      setTimeout(() => { loadInstances().catch(() => {}); }, 900);
    }

    async function stopServer(s) {
      const detach = s.mode === "attach";
      const ok = await confirmDialog({
        title: detach ? `Detach “${s.name}”?` : `Stop “${s.name}”?`,
        body: detach
          ? "Detach only forgets the attached endpoint — nothing is killed."
          : "Stops the llama-server process sable spawned for this instance.",
        confirmLabel: detach ? "Detach" : "Stop", danger: true });
      if (!ok) return;
      try {
        await api(`/api/servers/${encodeURIComponent(s.id)}/stop`,
          { method: "POST", body: {} });
        toast(detach ? "detached" : "stopped", "ok");
        await loadInstances();
      } catch (e) { showErr(e); }
    }

    function renderInstances() {
      const rows = servers.map((s) => {
        const stateBadge = badge(s.running ? (s.mode || "up") : "down",
          s.running ? (s.mode === "attach" ? "partial" : "done") : "failed");
        // Crash-loop budget (#28): refused until an explicit Start clears it.
        const crashBadge = s.crash_loop && s.crash_loop.refused
          ? badge(`crash loop (${s.crash_loop.failures})`, "failed")
          : null;
        // Circuit breaker (#28, runner/judge side): open/half-open state of
        // the backend's breaker keyed by base_url. Closed breakers are absent.
        const circ = Object.entries(s.circuits || {})[0];
        const circuitBadge = circ && (circ[1].state === "open" || circ[1].state === "half_open")
          ? badge(`circuit ${circ[1].state.replace("_", " ")} (${circ[1].failures})`,
              circ[1].state === "open" ? "failed" : "partial")
          : null;
        if (circuitBadge) circuitBadge.title =
          `backend circuit for ${circ[0]} — calls fail fast until the probe succeeds`;
        const actions = [
          s.running
            ? el("button", { class: "small danger",
              text: s.mode === "attach" ? "detach" : "stop",
              onclick: () => stopServer(s) })
            : el("button", { class: "small primary", text: "start",
              onclick: () => startServer(s) }),
        ];
        // The default instance's run settings live on the Settings page
        // (global section); named instances edit their own here.
        if (s.gguf && !s.default) {
          actions.push(el("button", { class: "small", text: "run settings",
            onclick: () => editRunSettings(s) }));
        }
        if (!s.default) {
          actions.push(el("button", { class: "small danger", text: "delete",
            onclick: async () => {
              const ok = await confirmDialog({
                title: `Delete “${s.name}”?`,
                body: "Stops the instance if running (owned processes only), " +
                  "then removes its definition.",
                confirmLabel: "Delete", danger: true });
              if (!ok) return;
              try {
                await api("/api/servers/" + encodeURIComponent(s.id),
                  { method: "DELETE" });
                toast(`deleted ${s.name}`, "info");
                await loadInstances();
              } catch (e) { showErr(e); }
            } }));
        }
        return el("tr", {},
          el("td", {}, el("span", { class: "mono", text: s.name }),
            s.default ? badge("default", "done") : null, " ",
            s.engine ? badge(s.engine, "partial") : null),
          el("td", { class: "hint mono" }, s.role || "custom"),
          el("td", {}, stateBadge, crashBadge ? " " : null, crashBadge,
            circuitBadge ? " " : null, circuitBadge),
          el("td", { class: "hint mono path" },
            s.base_url || s.attach_url || (s.gguf ? "spawn " + s.gguf : "–")),
          el("td", { class: "hint" }, s.model_name || "–"),
          el("td", {}, el("div", { class: "btnrow" }, ...actions)));
      });
      instancesHost.replaceChildren(
        el("h3", { text: "Server instances" }),
        servers.length ? el("table", {},
          el("thead", {}, el("tr", {},
            el("th", { text: "Name" }), el("th", { text: "Role" }),
            el("th", { text: "State" }), el("th", { text: "Endpoint" }),
            el("th", { text: "Model" }), el("th", { text: "" }))),
          el("tbody", {}, ...rows))
          : empty("No named instances yet — create one below."),
        el("p", { class: "hint", text:
          "The reserved “default” instance mirrors the legacy single-server " +
          "serve above. Named instances run concurrently; agent profiles can " +
          "bind to one. Sable never kills processes it did not spawn." }));
    }

    // ------------------------------------------------- new-instance form
    const nameIn = el("input", { type: "text", placeholder: "coding", size: 18 });
    const roleSel = el("select", {},
      ...["custom", "orchestrator", "coding", "embedding"].map((r) =>
        el("option", { value: r, text: r })));
    // Spawn engine: built-in llama.cpp by default; registered engines
    // (Hardware page) appear here once installed. Attach instances ignore it.
    const instEngineSel = el("select", {},
      el("option", { value: "", text: "llama.cpp (built-in)" }));
    function fillEngineSelect() {
      const cur = instEngineSel.value;
      instEngineSel.replaceChildren(
        el("option", { value: "", text: "llama.cpp (built-in)" }));
      for (const e of engines) {
        // Spawnable kinds: openai chat servers and diffusion image servers.
        // kind "other" stays inventory-only (Hardware page).
        if (e.builtin || !["openai", "diffusion"].includes(e.kind)) continue;
        const o = el("option", { value: e.id },
          e.name + (e.installed ? "" : " (not installed)"));
        instEngineSel.appendChild(o);
      }
      if ([...instEngineSel.options].some((o) => o.value === cur))
        instEngineSel.value = cur;
    }
    async function loadEngines() {
      try {
        const d = await api("/api/engines");
        engines = d.engines || [];
        fillEngineSelect();
        updateEngineForm();
      } catch { /* built-in-only form still works */ }
    }
    const instGgufSel = el("select", {},
      el("option", { value: "", text: "— pick a model from disk —" }));
    const instAttachIn = el("input", { type: "text",
      placeholder: "or attach URL (http://127.0.0.1:PORT/v1)", size: 30 });
    const instPortIn = el("input", { type: "number", min: "0", max: "65535",
      placeholder: "port (0=auto)", class: "narrow" });
    const instCtxIn = el("input", { type: "number", min: "512", step: "512",
      placeholder: "context (optional)", class: "narrow" });
    const instEmbedChk = el("input", { type: "checkbox" });
    const createBtn = el("button", { class: "primary", text: "Create instance" });

    // Per-instance model run settings — captured WITH the instance. New
    // instances never inherit the global Settings-page run_settings; these
    // belong to this server alone (the backend enforces the same rule).
    const instRs = runSettingsEditor({}, { gguf: () => instGgufSel.value });
    // Per-instance sampling + thinking defaults (request-time parameters sent
    // with every chat/agent request against this server, studio-style).
    const instSampling = samplingEditor({});
    // Per-instance diffusion generation defaults (sd-server: steps/CFG/size).
    // Shown INSTEAD of the chat run-settings/sampling sections when a
    // diffusion engine is selected — image models have no top_k.
    const instImage = imageSettingsEditor({});
    // model_kind "json" engines (strata) launch from their own run-config
    // file, not a GGUF; this input replaces the disk picker for them.
    const instCfgIn = el("input", { type: "text", class: "mono", size: 40,
      placeholder: "path to strata run-config.json (written by its setup)" });
    function selectedEngine() {
      if (!instEngineSel.value) return null;
      return engines.find((e) => e.id === instEngineSel.value) || null;
    }

    function ggufLabel(m) {
      const size = m.size_bytes ? ` · ${(m.size_bytes / 1e9).toFixed(1)} GB` : "";
      // Multi-part GGUFs arrive folded into one entry (first shard + summed
      // size); the badge tells the user all parts will be loaded.
      const shards = m.shard_count > 1 ? ` · ${m.shard_count} parts` : "";
      return `${m.name}${m.quant ? " [" + m.quant + "]" : ""}${size}${shards}`;
    }

    // Catalog hint shown under the instance model picker.
    const catHint = el("span", { class: "hint" });

    function fillInstCatalog() {
      instGgufSel.replaceChildren(
        el("option", { value: "", text: "— pick a model from disk —" }));
      for (const m of catalog.gguf || []) {
        instGgufSel.appendChild(el("option", { value: m.path, text: ggufLabel(m) }));
      }
    }
    instGgufSel.addEventListener("change", () => instRs.refreshMmproj());

    createBtn.addEventListener("click", async () => {
      clearErr();
      const body = { name: nameIn.value.trim(), role: roleSel.value };
      if (instEmbedChk.checked) body.role = "embedding";
      const eng = selectedEngine();
      const kind = eng ? eng.kind : "openai";
      const modelKind = eng ? (eng.model_kind || "gguf") : "gguf";
      if (modelKind === "json") {
        // strata-style engine: the "model" is its run-config.json path.
        if (!instCfgIn.value.trim()) {
          showErr({ code: "bad_request",
            message: "enter the path to this engine's run-config.json" });
          return;
        }
        body.gguf = instCfgIn.value.trim();
      } else if (instGgufSel.value) body.gguf = instGgufSel.value;
      else if (instAttachIn.value.trim()) body.attach_url = instAttachIn.value.trim();
      else { showErr({ code: "bad_request",
        message: "pick a GGUF from disk or enter an attach URL" }); return; }
      if (instEngineSel.value && body.gguf)
        body.engine = instEngineSel.value;
      if (instPortIn.value) body.port = parseInt(instPortIn.value, 10);
      if (instCtxIn.value) body.context_length = parseInt(instCtxIn.value, 10);
      if (kind === "diffusion") {
        // Image generation defaults instead of the chat knobs (which an
        // sd-server spawn ignores entirely).
        const imPatch = instImage.getPatch();
        if (Object.keys(imPatch).length) body.image_settings = imPatch;
      } else {
        const rsPatch = instRs.getPatch();
        if (Object.keys(rsPatch).length) body.run_settings = rsPatch;
        const spPatch = instSampling.getPatch();
        if (Object.keys(spPatch).length) body.sampling = spPatch;
      }
      createBtn.disabled = true;
      try {
        const r = await api("/api/servers", { method: "POST", body });
        toast(`instance ${r.server.name} created`, "ok");
        nameIn.value = ""; instAttachIn.value = "";
        await loadInstances();
      } catch (e) { showErr(e); }
      finally { createBtn.disabled = false; }
    });

    // Sections the engine dropdown toggles: chat engines get GGUF picker +
    // run settings + sampling; diffusion engines (sd-server) get image
    // generation defaults instead; model_kind "json" engines (strata) take a
    // run-config path instead of a GGUF.
    const ggufField = el("div", { class: "field" },
      el("label", { text: "model from disk" }), instGgufSel, catHint);
    const cfgField = el("div", { class: "field hidden" },
      el("label", { text: "run-config" }), instCfgIn,
      el("span", { class: "hint", text:
        "this engine launches from its own config file, not a GGUF — point it " +
        "at the run-config.json written by strata setup" }));
    const rsSection = el("div", {},
      el("h4", { text: "Model run settings (this instance only)" }),
      instRs.node);
    const spSection = el("div", {},
      el("h4", { text: "Sampling & thinking defaults (this instance only)" }),
      instSampling.node);
    const imSection = el("div", { class: "hidden" },
      el("h4", { text: "Image generation defaults (this instance only)" }),
      instImage.node);

    function updateEngineForm() {
      const e = selectedEngine();
      const kind = e ? e.kind : "openai";
      const mk = e ? (e.model_kind || "gguf") : "gguf";
      cfgField.classList.toggle("hidden", mk !== "json");
      ggufField.classList.toggle("hidden", mk === "json");
      imSection.classList.toggle("hidden", kind !== "diffusion");
      rsSection.classList.toggle("hidden", kind === "diffusion");
      spSection.classList.toggle("hidden", kind === "diffusion");
    }
    instEngineSel.addEventListener("change", updateEngineForm);

    const newInstanceCard = el("div", { class: "card" },
      el("h3", { text: "New server instance" }),
      el("div", { class: "field" }, el("label", { text: "name" }), nameIn,
        el("label", { text: "role" }), roleSel),
      el("div", { class: "field" }, el("label", { text: "engine" }),
        instEngineSel),
      ggufField, cfgField,
      el("div", { class: "field" }, el("label", { text: "or attach URL" }),
        instAttachIn,
        el("span", { class: "hint", text: "loopback only (127.0.0.1)" })),
      el("div", { class: "field" }, el("label", { text: "port" }), instPortIn,
        el("label", { text: "context" }), instCtxIn,
        el("label", { class: "chk" }, instEmbedChk, " --embedding server")),
      rsSection, spSection, imSection,
      el("div", { class: "btnrow" }, createBtn),
      el("p", { class: "hint", text:
        "Spawned servers bind 127.0.0.1 on dynamic ports; attach instances " +
        "point at an already-running endpoint and are never killed. These " +
        "settings belong to the instance — they are not inherited from the " +
        "Settings page defaults. Diffusion engines (stable-diffusion.cpp) " +
        "take image generation defaults; strata takes a run-config.json path." }));

    // ------------------------------------------------- edit-run-settings dialog
    async function editRunSettings(s) {
      // Diffusion instances (sd-server) edit image generation defaults;
      // everything else keeps the chat run-settings + sampling editors.
      const eng = engines.find((x) => x.id === s.engine);
      const diffusion = !!eng && eng.kind === "diffusion";
      const editor = diffusion
        ? imageSettingsEditor(s.image_settings || {})
        : runSettingsEditor(s.run_settings || {}, { gguf: () => s.gguf || "" });
      if (!diffusion) editor.refreshMmproj();
      // Sampling + thinking overrides live HERE (per server instance), not on
      // the Profiles page — an existing instance gets the same editors the
      // new-instance form shows.
      const sampEditor = diffusion ? null : samplingEditor(s.sampling || {});
      let done = false;
      await new Promise((resolve) => {
        const host = document.getElementById("modal-host");
        const close = (v) => { if (done) return; done = true; backdrop.remove(); resolve(v); };
        const save = async () => {
          saveBtn.disabled = true;
          try {
            const patch = diffusion
              ? { image_settings: editor.getPatch() }
              : { run_settings: editor.getPatch(),
                  sampling: sampEditor.getPatch() };
            await api("/api/servers/" + encodeURIComponent(s.id),
              { method: "PUT", body: patch });
            toast(`run settings saved for ${s.name}`, "ok");
            close(true);
          } catch (e) { showErr(e); close(false); }
        };
        const saveBtn = el("button", { class: "primary", text: "Save", onclick: save });
        const cancelBtn = el("button", { text: "Cancel", onclick: () => close(false) });
        const box = el("div", { class: "modal", role: "dialog", "aria-modal": "true" },
          el("h3", { text: `Run settings — ${s.name}` }), editor.node,
          diffusion ? null : el("h4",
            { text: "Sampling & thinking defaults (this instance only)" }),
          diffusion ? null : sampEditor.node,
          el("div", { class: "modal-actions" }, cancelBtn, saveBtn));
        const backdrop = el("div", { class: "backdrop",
          onclick: (ev) => { if (ev.target === backdrop) close(false); } }, box);
        host.appendChild(backdrop);
      });
      await loadInstances();
    }

    // ------------------------------------------------------------- controls
    // The legacy "Spawn default llama-server" card is gone — the reserved
    // "default" instance spawns via its Start button (model + engine
    // configured on the Settings page).

    // Populate the from-disk picker via GET /api/model_catalog (base model
    // folder: config models_dir / SABLE_MODELS_DIR / HF hub cache).
    async function loadCatalog() {
      try {
        catalog = await api("/api/model_catalog");
        catHint.textContent = catalog.exists
          ? `${(catalog.gguf || []).length} GGUF(s) in ${catalog.models_dir}`
          : "no model folder configured (settings: models_dir)";
        fillInstCatalog();
      } catch { /* picker stays empty; manual path still works */ }
    }

    // deploy -----------------------------------------------------------
    const deploySel = el("select", {}, el("option", { value: "", text: "— model —" }));
    const deployBtn = el("button", { text: "Deploy" });
    deployBtn.addEventListener("click", async () => {
      clearErr();
      if (!deploySel.value) { showErr({ code: "bad_request", message: "pick a model to deploy" }); return; }
      const ok = await confirmDialog({
        title: `Deploy ${deploySel.value}`,
        body: "This stops any running default server and serves the selected model's GGUF.",
        confirmLabel: "Deploy",
      });
      if (!ok) return;
      deployBtn.disabled = true;
      try {
        const r = await api("/api/serve/deploy", { method: "POST",
          body: { model_id: deploySel.value } });
        toast(`deployed ${r.model_name} → ${r.base_url}`, "ok");
        setTimeout(() => { refresh().catch(() => {}); loadInstances().catch(() => {}); }, 800);
      } catch (e) { showErr(e); }
      finally { deployBtn.disabled = false; }
    });
    const deployCard = el("div", { class: "card" },
      el("h3", { text: "Deploy registry model" }),
      el("div", { class: "field" }, deploySel, deployBtn),
      el("p", { class: "hint", text: "Hot-swap on the default instance: stops any owned server and starts the selected model." }));

    // remote sable peers (LAN) ----------------------------------------
    // Connect another computer running sable on the local network: probe its
    // dashboard and show what it serves, which agents it has active, and its
    // training + trace activity. The backend polls the peer read-only with the
    // peer's dashboard token (peers.json, 0600); nothing is pushed remotely.
    const peerUrlIn = el("input", { type: "text",
      placeholder: "192.168.1.42:8765  (remote dashboard URL)", size: 36 });
    const peerTokenIn = el("input", { type: "password",
      placeholder: "peer dashboard token", autocomplete: "off" });
    const peerOut = el("span", { class: "hint" });
    const connectBtn = el("button", { text: "Connect" });
    const peersHost = el("div", {});
    let peers = [];

    async function doConnect() {
      clearErr();
      const url = peerUrlIn.value.trim();
      if (!url) { showErr({ code: "bad_request", message: "enter the remote dashboard URL" }); return; }
      peerOut.textContent = "probing…";
      connectBtn.disabled = true;
      try {
        const body = { url };
        if (peerTokenIn.value) body.token = peerTokenIn.value;
        const r = await api("/api/remote/probe", { method: "POST", body });
        peerOut.textContent = "";
        toast(`connected → ${r.peer.name || url}`, "ok");
        peerUrlIn.value = ""; peerTokenIn.value = "";
        await loadPeers();
      } catch (e) {
        peerOut.textContent = "";
        showErr(e); // 400 carries the unreachable detail from the server
      } finally { connectBtn.disabled = false; }
    }
    connectBtn.addEventListener("click", doConnect);
    peerUrlIn.addEventListener("keydown", (e) => { if (e.key === "Enter") doConnect(); });

    async function loadPeers() {
      try { peers = (await api("/api/remote/peers")).peers || []; }
      catch (e) { peers = []; }
      renderPeers();
    }

    // Re-probe every remembered peer (live status, agents, jobs, traces).
    async function refreshPeers() {
      await Promise.allSettled(peers.map((p) =>
        api(`/api/remote/peers/${encodeURIComponent(p.id)}/live`)
          .catch(() => null)));
      await loadPeers();
    }

    function snapChips(snap) {
      if (!snap) return [];
      const out = [];
      const servers = (snap.servers || []).filter((s) => s.running);
      out.push(el("div", { class: "hint" },
        servers.length
          ? `serving: ${servers.map((s) => `${s.name}${s.model_name ? " → " + s.model_name : ""}`).join(", ")}`
          : "serving: nothing"));
      if ((snap.agents || []).length) {
        out.push(el("div", { class: "hint" },
          `agents: ${snap.agents.map((a) => a.name + (a.default ? " *" : "")).join(", ")}`));
      }
      const busy = (snap.jobs || []).filter(
        (j) => ["running", "queued", "gated"].includes(j.state));
      if (busy.length) {
        out.push(el("div", { class: "hint" },
          `training/jobs: ${busy.map((j) => `${j.kind} ${j.state}`).join(", ")}`));
      }
      const recent = (snap.traces || []).filter(
        (t) => String(t.created_at || "") >= cutoffIso());
      if (recent.length) {
        out.push(el("div", { class: "hint" },
          `traces (last 10 min): ${recent.map((t) =>
            `${String(t.trace_id || "").slice(0, 8)}${t.status === "failure" ? " ✗" : " ✓"}`).join(", ")}`));
      }
      return out;
    }
    function cutoffIso() {
      return new Date(Date.now() - 10 * 60 * 1000).toISOString();
    }

    function peerRow(p) {
      const snap = p.snapshot || null;
      const head = el("div", { class: "pathrow" },
        badge(p.healthy ? "online" : "offline", p.healthy ? "done" : "failed"),
        el("strong", { text: p.name || p.url }),
        el("span", { class: "mono hint" }, p.url),
        snap && snap.version ? el("span", { class: "hint mono" }, `v${snap.version}`) : null,
        el("span", { class: "hint" }, p.last_ok_at
          ? `last ok ${fmtTime(p.last_ok_at, true)}` : "never reached"),
        el("button", { class: "small", text: "re-check", style: "margin-left:auto",
          onclick: async () => {
            try { await api(`/api/remote/peers/${encodeURIComponent(p.id)}/live`); }
            catch (e) { /* offline shows as badge flip, no toast spam */ }
            await loadPeers();
          } }),
        el("button", { class: "small danger", text: "forget",
          onclick: async () => {
            const ok = await confirmDialog({ title: "Forget peer?",
              body: `“${p.name || p.url}” will be removed from this list. ` +
                "Nothing is signalled on the remote machine.",
              confirmLabel: "Forget", danger: true });
            if (!ok) return;
            try {
              await api("/api/remote/peers/" + encodeURIComponent(p.id),
                { method: "DELETE" });
              await loadPeers();
            } catch (e) { showErr(e); }
          } }));
      const body = p.healthy ? snapChips(snap)
        : [el("div", { class: "hint" },
            snap && snap.error ? `last error: ${snap.error}` : "no successful probe yet")];
      return el("div", { class: "card", style: "margin:6px 0;padding:8px 10px" }, head, ...body);
    }

    function renderPeers() {
      peersHost.replaceChildren(
        ...(peers.length ? peers.map(peerRow)
          : [empty("No remote sable machines connected yet.")]));
    }

    const peerCard = el("div", { class: "card" },
      el("h3", { text: "Remote sable on the local network" }),
      el("div", { class: "field" }, el("label", { text: "dashboard URL" }), peerUrlIn),
      el("div", { class: "field" }, el("label", { text: "token" }), peerTokenIn,
        connectBtn, peerOut),
      peersHost,
      el("p", { class: "hint", text:
        "Point at another machine's sable dashboard (its URL and token are " +
        "printed when it starts). sable remembers the machine and shows what " +
        "it serves, which agents it has active, plus its training jobs and " +
        "recent traces. This is read-only polling — nothing runs on or is " +
        "pushed to the remote box. To use a remote MODEL as if local, add a " +
        "named instance with its attach URL above (loopback endpoints only). "}));

    mount.replaceChildren(
      el("h2", { text: "Serve" }), statusCard, errBox,
      instancesHost, newInstanceCard,
      el("div", { class: "cols" }, deployCard, peerCard));

    // ------------------------------------------------------------- wiring
    async function loadModels() {
      try {
        const d = await api("/api/models");
        models = d.models || [];
        const cur = deploySel.value;
        deploySel.replaceChildren(el("option", { value: "", text: "— model —" }));
        for (const m of models) {
          if (!m.gguf_path) continue;
          deploySel.appendChild(el("option", { value: m.model_id,
            text: `${m.model_id} (${m.status})` }));
        }
        deploySel.value = cur;
      } catch { /* serve panel still usable without the registry */ }
    }

    refresh();
    loadModels();
    loadCatalog();
    loadInstances();
    loadEngines();

    unsubs.push(bus.on("serve.state", (p) => {
      const label = { started: "server started", attached: "endpoint attached",
        stopped: "server stopped", deployed: "model deployed" }[p.event] || p.event;
      toast(`${label}${p.model_name ? " — " + p.model_name : ""}`, "ok");
      refresh();
      loadInstances();
    }));
    unsubs.push(bus.on("servers.changed", () => { loadInstances(); }));
    unsubs.push(bus.on("model.promoted", loadModels));
    unsubs.push(bus.on("model.rolled_back", loadModels));
    loadPeers();
    const poll = setInterval(() => {
      refresh().catch(() => {});
      loadInstances().catch(() => {});
      // Re-probe remote peers on the same beat so their live state, active
      // agents, training and traces stay current (each probe is a handful of
      // cheap GETs against LAN machines).
      refreshPeers().catch(() => {});
    }, 10000);

    return { destroy() { clearInterval(poll); for (const u of unsubs) u(); } };
  },
};

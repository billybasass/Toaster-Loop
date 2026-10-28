// panels/chat.js — agent playground.
// Two modes:
//   Agent     -> POST /api/chat/stream (our event protocol; tool cards, footer)
//   Playground-> POST /api/proxy/chat  (raw OpenAI SSE passthrough)
// Threads persist via /api/chat/threads*. All model output is rendered through
// textContent / the sanitized markdown renderer — never innerHTML.

import { api, streamSSE, streamSSEGet } from "../api.js";
import { bus } from "../bus.js";
import { skey, sget, sset, sdel, janitorKeys } from "../state.js";
import { el, toast, empty, confirmDialog, renderMarkdown, fmtNum } from "../ui.js";

function newId() {
  return "t-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 7);
}

export default {
  title: "Chat",
  render(mount) {
    // ---------------------------------------------------------------- state
    let threads = [];            // summaries from the server
    let projects = [];           // project records from the server
    let threadId = null;         // current thread id (null until first save)
    let threadWasProjectless = true; // gate: only stamp project when unfiled
    let threadSummary = null;    // backlog #6: rolling compaction summary
    let messages = [];           // [{role, content, ...extra}] for current thread
    // Persistence keys are scope-declared (pass 3 #25): `uf:<scope>:<id?>:
    // <thing>`. Global panel state lives in the "ui" scope (empty id segment,
    // never janitored); per-project state carries the project id so deleting
    // a project can drop its keys. sget() transparently migrates the old
    // sable.* names on first read.
    let mode = sget(skey("ui", "", "mode"), "agent");
    // Projects: active-project context (null/"" = Unfiled). The id rides along
    // on new threads + chat runs.
    let activeProject = sget(skey("ui", "", "active-project")) || null;
    const collapsed = new Set(
      JSON.parse(sget(skey("ui", "", "collapsed-projects"), "[]")));
    let running = false;         // a stream is attached in THIS view
    let runId = null;            // active agent run (for cancel)
    let abort = null;            // AbortController for the response stream
    // Run lifecycle: leaving the chat panel (or switching threads) only stops
    // READING the stream — the run keeps executing on the server, which
    // buffers every frame and persists the transcript itself. Coming back
    // re-attaches to the live stream (GET /api/chat/runs/<id>/stream): the
    // viewer sees everything produced while away, then follows live tokens.
    // Different threads run independently: open one thread mid-run, switch
    // to another and start a second agent — both keep going server-side.
    let adoptedRun = null;       // {runId, thread} currently re-attached
    let adoptTimer = null;       // legacy poll handle (cleared defensively)
    // View generation: bumped on every thread switch / new thread so the
    // async tail of an aborted run can detect that the user moved on and
    // must not mutate the (now different) view's messages.
    let viewSession = 0;
    const unsubs = [];

    function setActiveProject(id) {
      const changed = activeProject !== (id || null);
      activeProject = id || null;
      if (activeProject) sset(skey("ui", "", "active-project"), activeProject);
      else sdel(skey("ui", "", "active-project"));
      renderThreadList();
      if (changed) explorerProjectChanged();
    }
    function saveCollapsed() {
      sset(skey("ui", "", "collapsed-projects"), JSON.stringify([...collapsed]));
    }

    // ----------------------------------------------------------------- DOM
    const threadList = el("nav", { class: "threads", "aria-label": "chat threads" });
    const msgHost = el("div", { class: "msgs" });
    const input = el("textarea", { rows: 3, placeholder: "Ask the agent a task… (Enter sends, Shift+Enter newline)" });
    const sendBtn = el("button", { class: "primary", text: "Send" });
    // Image support (agent mode): attach up to 4 images per message. They
    // are read as base64 data URLs and sent in `images`; the server saves
    // them under <home>/uploads and hands the agent attach markers that the
    // runner expands into native image parts when a vision model is served.
    const MAX_ATTACH = 4;
    // Per-image cap mirrors core/images.py MAX_IMAGE_BYTES (8 MB raw); the
    // server enforces it too — checking here gives a clear message instead of
    // an opaque 413 after upload.
    const MAX_IMAGE_BYTES = 8000000;
    let pendingImages = [];  // [{name, dataUrl}]
    const imgInput = el("input", { type: "file", accept: "image/*",
      multiple: "", class: "hidden" });
    const attachBtn = el("button", { class: "small", text: "🖼 attach",
      title: "attach reference image(s) for the agent to look at" });
    const attachRow = el("div", { class: "btnrow hidden" });
    function renderAttachRow() {
      attachRow.replaceChildren();
      attachRow.classList.toggle("hidden", pendingImages.length === 0);
      for (const [i, im] of pendingImages.entries()) {
        attachRow.appendChild(el("span", { class: "badge" },
          im.name || `image ${i + 1}`,
          el("button", { class: "small", text: "×", style: "margin-left:4px",
            onclick: () => { pendingImages.splice(i, 1); renderAttachRow(); } })));
      }
    }
    attachBtn.addEventListener("click", () => imgInput.click());
    imgInput.addEventListener("change", () => {
      for (const f of [...(imgInput.files || [])]) {
        if (pendingImages.length >= MAX_ATTACH) {
          toast(`max ${MAX_ATTACH} images per message`, "error"); break;
        }
        if (f.size > MAX_IMAGE_BYTES) {
          toast(`${f.name}: ${(f.size / 1e6).toFixed(1)} MB exceeds the ` +
            `${MAX_IMAGE_BYTES / 1e6} MB cap — downscale it first`, "error");
          continue;
        }
        const rd = new FileReader();
        rd.onload = () => { pendingImages.push({ name: f.name,
          dataUrl: String(rd.result) }); renderAttachRow(); };
        rd.readAsDataURL(f);
      }
      imgInput.value = "";
    });
    const stopBtn = el("button", { class: "danger hidden", text: "Stop" });
    const modeAgent = el("button", { class: "seg", text: "Agent" });
    const modePlay = el("button", { class: "seg", text: "Playground" });
    const hint = el("span", { class: "hint" });
    // 5e: agent profile selector (agent mode only; default = config default).
    const profileSel = el("select", { class: "profile-sel",
      title: "agent profile: system prompt + tool subset for this run" },
      el("option", { value: "", text: "profile: default" }));
    let defaultProfile = null;
    // The selected profile survives navigation (bug fix: returning from
    // Settings used to reset the dropdown to default mid-conversation).
    // `pendingProfile` is the restore target; it is applied whenever the
    // option list (re)builds, and overridden by a thread's stored profile
    // when that thread is opened.
    let pendingProfile = sget(skey("ui", "", "profile")) || "";

    function applyProfileSelection() {
      const names = [...profileSel.options].map((o) => o.value);
      profileSel.value = names.includes(pendingProfile) ? pendingProfile : "";
    }
    profileSel.addEventListener("change", () => {
      pendingProfile = profileSel.value;
      if (pendingProfile) sset(skey("ui", "", "profile"), pendingProfile);
      else sdel(skey("ui", "", "profile"));
    });

    async function loadProfiles() {
      try {
        const d = await api("/api/profiles");
        defaultProfile = d.default_profile || null;
        // Remote LAN agents (name "<agent>@<host>", p.remote) run their
        // inference on the peer machine; the agent loop stays local. The
        // label carries a suffix so it is obvious which box answers.
        profileSel.replaceChildren(
          el("option", { value: "", text: defaultProfile
            ? `default (${defaultProfile})` : "default (built-in prompt)" }),
          ...(d.profiles || []).map((p) =>
            el("option", { value: p.name,
              text: p.remote ? `${p.name} · remote` : p.name })));
        applyProfileSelection();
      } catch { /* profiles are optional; keep the default option */ }
    }

    // Select the profile a thread was last run with (null/absent = default).
    function selectThreadProfile(name) {
      pendingProfile = name || "";
      applyProfileSelection();
      if (pendingProfile) sset(skey("ui", "", "profile"), pendingProfile);
      else sdel(skey("ui", "", "profile"));
    }

    // auto-scroll with user-scroll-up guard
    let stick = true;
    msgHost.addEventListener("scroll", () => {
      stick = msgHost.scrollTop + msgHost.clientHeight >= msgHost.scrollHeight - 40;
    });
    const scrollDown = () => { if (stick) msgHost.scrollTop = msgHost.scrollHeight; };

    // ----------------------------------------------- projects + thread tree
    async function loadProjects() {
      try { projects = (await api("/api/projects")).projects || []; }
      catch (e) { projects = []; }
      // Persistence janitor (pass 3 #25): drop per-project keys whose project
      // no longer exists, and prune collapsed entries for dead ids — the
      // state-hygiene rule that every keyed family follows delete migrations.
      janitorKeys("project", projects.map((p) => p.id));
      let pruned = false;
      for (const id of [...collapsed]) {
        if (!projects.some((p) => p.id === id)) { collapsed.delete(id); pruned = true; }
      }
      if (pruned) saveCollapsed();
      // A deleted/unknown active project id falls back to Unfiled.
      if (activeProject && !projects.some((p) => p.id === activeProject))
        setActiveProject(null);
    }

    async function loadThreads() {
      try { threads = (await api("/api/chat/threads")).threads || []; }
      catch (e) { threads = []; }
      renderThreadList();
    }

    // ---------------------------------------------------- project dialogs
    // Shared create/edit form. Besides name + workspace dir it captures:
    //   * Description — what the project is about; injected into every
    //     agent run on the project (backend: project_context).
    //   * Agents — checkbox per profile, each with a "job on this project"
    //     role line. Every agent on the project sees the full roster with
    //     roles, so each one knows its own job and its teammates'.
    async function projectForm(p) {
      const nameIn = el("input", { type: "text", placeholder: "project name",
        value: p ? p.name || "" : "" });
      const dirIn = el("input", { type: "text",
        placeholder: "/absolute/path/to/workspace",
        value: p ? p.workspace_dir || "" : "" });
      const descIn = el("textarea", { rows: "3",
        placeholder: "what this project is about — shown to every agent "
          + "working on it" });
      descIn.value = (p && p.description) || "";
      const knownRoles = (p && p.agent_roles) || {};
      const chosen = new Set((p && p.agents) || []);
      let profList = [];
      try { profList = (await api("/api/profiles")).profiles || []; }
      catch { /* no profiles yet — roster below shows the hint instead */ }
      // Remote LAN agents ("<agent>@<host>", .remote) attach like any other
      // roster member: their inference runs on the peer, their tools and
      // files here run against this project's workspace.
      // An attached agent whose profile was since deleted must stay visible
      // (checked) so the user can edit or drop it deliberately.
      for (const a of chosen)
        if (!profList.some((x) => x.name === a)) profList.push({ name: a });
      const rows = [];
      const list = el("div", { class: "proj-agents" });
      for (const pr of profList) {
        const cb = el("input", { type: "checkbox" });
        cb.checked = chosen.has(pr.name);
        const roleIn = el("input", { type: "text",
          placeholder: "job on this project (optional)",
          title: `what ${pr.name} is responsible for — every agent on the `
            + "project sees the full roster with roles" });
        roleIn.value = knownRoles[pr.name] || "";
        roleIn.disabled = !cb.checked;
        cb.addEventListener("change", () => { roleIn.disabled = !cb.checked; });
        rows.push({ name: pr.name, cb, roleIn });
        list.appendChild(el("label", { class: "proj-agent-row" },
          cb, el("span", { class: "proj-agent-name mono", text: pr.name }),
          roleIn));
      }
      if (!rows.length)
        list.appendChild(el("div", { class: "thread-meta",
          text: "no agent profiles yet — add some on the Profiles page" }));
      const form = el("div", { class: "field" },
        el("label", { text: "Name" }), nameIn,
        el("label", { text: "Workspace directory" }), dirIn,
        el("label", { text: "Description" }), descIn,
        el("label", { text: "Agents on this project" }), list);
      return {
        form,
        read() {
          const agents = [], agent_roles = {};
          for (const r of rows) if (r.cb.checked) {
            agents.push(r.name);
            const role = r.roleIn.value.trim();
            if (role) agent_roles[r.name] = role;
          }
          return { name: nameIn.value.trim(),
            workspace_dir: dirIn.value.trim(),
            description: descIn.value.trim(), agents, agent_roles };
        },
      };
    }

    async function newProjectDialog() {
      const f = await projectForm(null);
      if (!await confirmDialog({ title: "New project", body: f.form,
        confirmLabel: "Create" })) return;
      try {
        const d = await api("/api/projects", { method: "POST",
          body: f.read() });
        setActiveProject(d.project.id);
        toast("project created", "ok");
        await loadProjects();
        renderThreadList();
      } catch (e) { toast(e.message, "error"); }
    }

    async function editProjectDialog(p) {
      const f = await projectForm(p);
      if (!await confirmDialog({ title: "Edit project", body: f.form,
        confirmLabel: "Save" })) return;
      try {
        await api("/api/projects/" + encodeURIComponent(p.id),
          { method: "PUT", body: f.read() });
        toast("project saved", "ok");
        await loadProjects();
        renderThreadList();
      } catch (e) { toast(e.message, "error"); }
    }

    async function deleteProjectDialog(p) {
      const ok = await confirmDialog({
        title: "Delete project?",
        body: `“${p.name}” is removed from sable (its threads move to `
          + "Unfiled) together with its codebase index. Nothing under "
          + `“${p.workspace_dir}” is deleted — the workspace and all files `
          + "the agents worked on stay exactly where they are.",
        confirmLabel: "Delete project", danger: true });
      if (!ok) return;
      try {
        await api("/api/projects/" + encodeURIComponent(p.id),
          { method: "DELETE" });
        toast("project deleted — workspace files kept", "ok");
        await loadProjects();
        renderThreadList();
      } catch (e) { toast(e.message, "error"); }
    }

    // -------------------------------------------------------- tree rendering
    function threadRow(t) {
      const active = t.id === threadId;
      // Phase 2 placeholder: delegation lineage chip ("A → B") when the
      // thread JSON carries delegations ({agent, ...} entries); absent = none.
      const lineage = (Array.isArray(t.delegations) && t.delegations.length)
        ? [el("span", { class: "thread-lineage mono", title: "delegation lineage",
            text: t.delegations.map((d) => d.agent || "?").join(" → ") })]
        : [];
      const item = el("div", { class: "thread-item" + (active ? " active" : ""),
        tabindex: 0, role: "button" },
        el("span", { class: "thread-title", text: t.title || t.id }),
        ...lineage,
        el("span", { class: "thread-meta",
          text: `${t.profile || "default"} · ${t.message_count ?? 0} msg` }),
        el("button", { class: "thread-x small", text: "✕", title: "delete",
          onclick: async (ev) => {
            ev.stopPropagation();
            const ok = await confirmDialog({ title: "Delete thread?",
              body: `“${t.title || t.id}” will be removed.`,
              confirmLabel: "Delete", danger: true });
            if (!ok) return;
            try {
              await api("/api/chat/threads/" + encodeURIComponent(t.id),
                { method: "DELETE" });
              if (t.id === threadId) newThread();
              loadThreads();
            } catch (e) { toast(e.message, "error"); }
          } }));
      const open = () => openThread(t.id);
      item.addEventListener("click", open);
      item.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
      // rename on double-click
      item.addEventListener("dblclick", async () => {
        const title = window.prompt("Thread title", t.title || t.id);
        if (!title) return;
        try {
          await api("/api/chat/threads/" + encodeURIComponent(t.id),
            { method: "PUT", body: { title } });
          loadThreads();
        } catch (e) { toast(e.message, "error"); }
      });
      return item;
    }

    function projectNode(p, itsThreads) {
      const isActive = p.id === activeProject;
      const isCollapsed = collapsed.has(p.id);
      const caret = el("span", { class: "proj-caret",
        text: isCollapsed ? "▸" : "▾" });
      const head = el("div", { class: "project-node" + (isActive ? " active" : ""),
        tabindex: 0, role: "button",
        title: ["click: set active project · caret: expand/collapse",
          p.description ? "— " + p.description : ""].filter(Boolean).join(" ") },
        caret,
        el("span", { class: "thread-title", text: p.name }),
        el("span", { class: "thread-meta", text: `${itsThreads.length}` }),
        planChip(p.id), planToggleBtn(p.id),
        el("button", { class: "thread-x proj-x small", text: "✎",
          title: "edit project (name, workspace, description, agents)",
          onclick: (ev) => {
            ev.stopPropagation();
            editProjectDialog(p);
          } }),
        el("button", { class: "thread-x proj-x small", text: "✕",
          title: "delete project (workspace files are kept)",
          onclick: (ev) => {
            ev.stopPropagation();
            deleteProjectDialog(p);
          } }));
      // caret toggles expansion; the node body selects the active project.
      caret.addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (collapsed.has(p.id)) collapsed.delete(p.id);
        else collapsed.add(p.id);
        saveCollapsed();
        renderThreadList();
      });
      const select = () => setActiveProject(isActive ? null : p.id);
      head.addEventListener("click", select);
      head.addEventListener("keydown", (e) => { if (e.key === "Enter") select(); });
      const kids = isCollapsed ? [] : itsThreads.map(threadRow);
      return [head, ...kids];
    }

    // ------------------------------------------------- WP12 plan chips/drawer
    // One persistent task graph per project (backend: /api/plans/<project>).
    // Project nodes get a done/total chip; the ▤ button opens a drawer with
    // task rows (start / in-progress / done / blocked / inline edit), an add
    // form, an import box (paste brief -> preview parsed tasks -> confirm)
    // and the guarded loop controls. When config plans.enabled is off the
    // API answers 404 plans_disabled and ALL of this hides itself — the same
    // "off means invisible" contract the agent tool family honors.
    // Colors come from .plan-dot-* classes (style.css, design tokens only —
    // the hygiene test forbids raw hex in JS).
    const planState = new Map();   // store key ("default"|pid) -> GET payload
    const planCtlCache = new Map(); // key -> persistent form controls (value
                                    // survives SSE re-renders; nodes are moved)
    let plansDisabled = false;
    let openPlanKey = sget(skey("ui", "", "plan-open")) || null;

    function planKey(pid) { return pid || "default"; }

    async function loadPlan(key, silent = true) {
      if (plansDisabled) return null;
      try {
        const d = await api("/api/plans/" + encodeURIComponent(key));
        planState.set(key, d);
        renderThreadList();
        return d;
      } catch (e) {
        if (e.code === "plans_disabled") {
          plansDisabled = true;
          openPlanKey = null;
          sdel(skey("ui", "", "plan-open"));
          renderThreadList();
        } else if (!silent) toast(e.message, "error");
        return planState.get(key) || null;
      }
    }

    async function loadAllPlans() {
      await loadPlan("default");     // first: detects plans_disabled once
      if (plansDisabled) return;
      for (const p of projects) loadPlan(p.id);
    }

    function planChip(pid) {
      const d = plansDisabled ? null : planState.get(planKey(pid));
      if (!d || !(d.counts && d.counts.total)) return el("span");
      const c = d.counts;
      const looping = d.loop && d.loop.running;
      return el("span", { class: "thread-meta " +
        (c.done === c.total ? "plan-chip-complete" : ""),
        title: `plan: ${c.done}/${c.total} done` +
          (looping ? " · loop running" : ""),
        text: `${c.done}/${c.total}` + (looping ? " ↻" : "") });
    }

    function planToggleBtn(pid) {
      const key = planKey(pid);
      const isOpen = openPlanKey === key;
      return el("button", { class: "thread-x proj-x small", text: "▤",
        title: isOpen ? "hide project plan" : "show project plan (WP12)",
        onclick: async (ev) => {
          ev.stopPropagation();
          if (isOpen) {
            openPlanKey = null;
            sdel(skey("ui", "", "plan-open"));
            renderThreadList();
            return;
          }
          openPlanKey = key;
          sset(skey("ui", "", "plan-open"), key);
          await loadPlan(key, false);
          renderThreadList();
        } });
    }

    // Deterministic mirror of the server's brief line format for PREVIEW
    // only ("title [deps=t1,t2] [priority=high]"); confirm re-sends the raw
    // text so the server (or the model, with use-ai) stays authoritative.
    function parsePlanBrief(text) {
      const out = [];
      for (const raw of String(text || "").split("\n")) {
        let s = raw.trim();
        if (!s || s.startsWith("#")) continue;
        const deps = []; let prio = "medium";
        s = s.replace(/\[(deps|dependencies|priority|prio)\s*=\s*([^\]]*)\]/gi,
          (_m, k, v) => {
            k = String(k).toLowerCase(); v = String(v).trim();
            if (k.startsWith("dep"))
              deps.push(...v.split(/[,\s]+/).filter(Boolean));
            else if (["high", "medium", "low"].includes(v.toLowerCase()))
              prio = v.toLowerCase();
            return "";
          }).trim().replace(/^[-*]\s*/, "");
        if (s) out.push({ title: s, deps, priority: prio });
      }
      return out;
    }

    function planCtl(key) {
      let c = planCtlCache.get(key);
      if (c) return c;
      const addTitle = el("input", { type: "text",
        placeholder: "new task title" });
      const addDesc = el("input", { type: "text",
        placeholder: "description (optional)" });
      const addPrio = el("select", {},
        ...["medium", "high", "low"].map((v) =>
          el("option", { value: v, text: v })));
      const addDeps = el("input", { type: "text",
        placeholder: "deps e.g. t1,t2" });
      const btnAdd = el("button", { class: "small", text: "＋ add task",
        onclick: async () => {
          const title = addTitle.value.trim();
          if (!title) { toast("title required", "error"); return; }
          try {
            const body = { op: "add", title,
              description: addDesc.value.trim(), priority: addPrio.value };
            const deps = addDeps.value.split(/[,\s]+/).filter(Boolean);
            if (deps.length) body.deps = deps;
            await api("/api/plans/" + encodeURIComponent(key),
              { method: "POST", body });
            addTitle.value = ""; addDesc.value = ""; addDeps.value = "";
            toast("task added", "ok");
            await loadPlan(key);
          } catch (e) { toast(e.message, "error"); }
        } });
      const impText = el("textarea", { rows: 4, class: "mono",
        placeholder: "paste a brief — one task per line:\n" +
          "Set up auth [priority=high]\nAdd login form [deps=t1]" });
      const impAi = el("input", { type: "checkbox" });
      const impPrev = el("div", { class: "hint" });
      const btnPreview = el("button", { class: "small", text: "preview",
        onclick: () => {
          const parsed = parsePlanBrief(impText.value);
          impPrev.replaceChildren(...(parsed.length
            ? parsed.map((t) => el("div", {},
                `• ${t.title}` + (t.deps.length
                  ? ` (deps: ${t.deps.join(", ")})` : "") +
                (t.priority !== "medium" ? ` [${t.priority}]` : "")))
            : [el("i", { text: "no task lines detected" })]));
        } });
      const btnImport = el("button", { class: "small primary",
        text: "import → confirm",
        onclick: async () => {
          const text = impText.value.trim();
          if (!text) { toast("paste a brief first", "error"); return; }
          if (impAi.checked && !await confirmDialog({
            title: "Import with AI?", body: el("p", {},
              "The served model structures the brief (one bounded call, " +
              "occupies the GPU briefly). Continue?"),
            confirmLabel: "Import" })) return;
          btnImport.disabled = true;
          try {
            const d = await api("/api/plans/" + encodeURIComponent(key),
              { method: "POST", body: { op: "import", text,
                use_ai: impAi.checked } });
            for (const w of (d.warnings || [])) toast(w, "error");
            toast(`imported ${d.added.length} task(s) (${d.source})`, "ok");
            impText.value = ""; impPrev.replaceChildren();
            await loadPlan(key);
          } catch (e) { toast(e.message, "error"); }
          finally { btnImport.disabled = false; }
        } });
      const loopMax = el("input", { type: "number", min: 1, max: 50, value: 5,
        style: "width:4.5em", title: "max tasks this loop run" });
      const btnLoop = el("button", { class: "small", text: "▶ run loop",
        title: "guarded autonomous loop: agents close tasks only via " +
          "plan_progress; stops on blocked/max-tasks/cancel" });
      const btnCancel = el("button", { class: "small danger hidden",
        text: "■ cancel loop" });
      btnLoop.addEventListener("click", async () => {
        if (!await confirmDialog({ title: "Start autonomous loop?", body:
          el("p", {}, "Runs agents unattended in the project workspace " +
            "under each task's profile grants, one unblocked task at a " +
            "time. Tasks close only on explicit completion."),
          confirmLabel: "Start loop" })) return;
        try {
          await api("/api/plans/" + encodeURIComponent(key),
            { method: "POST", body: { op: "loop-start",
              max_tasks: parseInt(loopMax.value, 10) || undefined } });
          toast("loop started", "ok");
          await loadPlan(key);
        } catch (e) { toast(e.message, "error"); }
      });
      btnCancel.addEventListener("click", async () => {
        try {
          await api("/api/plans/" + encodeURIComponent(key),
            { method: "POST", body: { op: "loop-cancel" } });
          toast("loop cancellation requested", "ok");
          await loadPlan(key);
        } catch (e) { toast(e.message, "error"); }
      });
      c = { addTitle, addDesc, addPrio, addDeps, btnAdd, impText, impAi,
        impPrev, btnPreview, btnImport, loopMax, btnLoop, btnCancel };
      planCtlCache.set(key, c);
      return c;
    }

    function planTaskRow(key, t, byId) {
      const blockers = (t.deps || []).filter((d) => {
        const dt = byId[d];
        return dt && !["done", "cancelled"].includes(dt.status);
      });
      const setStatus = async (status) => {
        try {
          await api("/api/plans/" + encodeURIComponent(key),
            { method: "POST", body: { op: "set-status", task_id: t.id,
              status } });
          await loadPlan(key);
        } catch (e) { toast(e.message, "error"); }
      };
      const startTask = async () => {
        try {
          const d = await api("/api/plans/" + encodeURIComponent(key),
            { method: "POST", body: { op: "start", task_id: t.id } });
          toast(`task ${t.id} started in a new thread`, "ok");
          await loadThreads();
          openThread(d.thread);
        } catch (e) { toast(e.message, "error"); }
      };
      const editTask = async () => {
        const titleIn = el("input", { type: "text", value: t.title });
        const descIn = el("textarea", { rows: 3 }); descIn.value = t.description || "";
        const detIn = el("textarea", { rows: 4 }); detIn.value = t.details || "";
        const testIn = el("textarea", { rows: 2 }); testIn.value = t.test_strategy || "";
        const form = el("div", {},
          el("label", { text: "Title" }), titleIn,
          el("label", { text: "Description" }), descIn,
          el("label", { text: "Details / notes" }), detIn,
          el("label", { text: "Test strategy" }), testIn);
        if (!await confirmDialog({ title: `Edit ${t.id}`, body: form,
          confirmLabel: "Save" })) return;
        try {
          await api("/api/plans/" + encodeURIComponent(key),
            { method: "POST", body: { op: "update", task_id: t.id,
              title: titleIn.value.trim(), description: descIn.value,
              details: detIn.value, test_strategy: testIn.value } });
          await loadPlan(key);
        } catch (e) { toast(e.message, "error"); }
      };
      const done = ["done", "cancelled"].includes(t.status);
      return el("div", { class: "plan-task" },
        el("span", { class: "plan-dot plan-dot-" + t.status,
          title: t.status, text: "●" }),
        el("span", { class: "mono hint", text: t.id }),
        el("span", { class: "plan-title" + (done ? " closed" : ""),
          text: t.title + (t.parent ? ` ↳${t.parent}` : "") }),
        t.priority !== "medium"
          ? el("span", { class: "plan-prio", text: t.priority })
          : el("span"),
        blockers.length
          ? el("span", { class: "plan-blockers",
            title: "waiting on unfinished dependencies",
            text: `blocked by ${blockers.join(", ")}` }) : el("span"),
        (t.labels || []).length
          ? el("span", { class: "hint",
            text: (t.labels || []).join(" ") }) : el("span"),
        !done ? el("button", { class: "small", title: "start task " +
          "(new thread seeded with the full task context)", text: "▶",
          onclick: startTask }) : el("span"),
        !done && t.status !== "in_progress"
          ? el("button", { class: "small", title: "mark in progress",
            text: "◐", onclick: () => setStatus("in_progress") })
          : el("span"),
        !done ? el("button", { class: "small", title: "mark done",
          text: "✔", onclick: () => setStatus("done") }) : el("span"),
        !done && t.status !== "blocked"
          ? el("button", { class: "small", title: "mark blocked",
            text: "⛔", onclick: () => setStatus("blocked") }) : el("span"),
        el("button", { class: "small", title: "edit task", text: "✎",
          onclick: editTask }));
    }

    function planDrawer(key) {
      const d = planState.get(key);
      if (!d) return null;
      const c = planCtl(key);
      const byId = {};
      for (const t of d.tasks) byId[t.id] = t;
      const rows = d.tasks.length
        ? d.tasks.map((t) => planTaskRow(key, t, byId))
        : [el("p", { class: "hint", text: "plan is empty — add a task or " +
            "import a brief" })];
      if (d.validate && !d.validate.ok)
        rows.unshift(el("p", { class: "plan-invalid",
          text: "⚠ graph invalid: " + JSON.stringify(d.validate) }));
      const looping = d.loop && d.loop.running;
      c.btnLoop.classList.toggle("hidden", !!looping);
      c.btnCancel.classList.toggle("hidden", !looping);
      const prog = (d.loop && d.loop.progress) || {};
      const loopLine = looping
        ? `loop running · iteration ${prog.iteration ?? "?"} · ` +
          `${(prog.tasks || []).filter((e) => e.outcome === "done").length} done`
        : prog.status
          ? `last loop: ${prog.status}` +
            (prog.stop_reason ? ` (${prog.stop_reason})` : "")
          : "";
      return el("div", { class: "plan-drawer" },
        el("div", { class: "btnrow" }, c.btnLoop, c.btnCancel),
        loopLine ? el("div", { class: "hint mono", text: loopLine }) : el("span"),
        ...rows,
        el("div", { class: "btnrow", style: "flex-wrap:wrap;gap:4px" },
          c.addTitle, c.addDesc, c.addPrio, c.addDeps, c.btnAdd),
        el("details", {}, el("summary", { class: "hint", text: "import brief" }),
          c.impText,
          el("label", { class: "hint" }, c.impAi, " use AI (served model) "),
          el("div", { class: "btnrow" }, c.btnPreview, c.btnImport),
          c.impPrev),
        el("div", { class: "btnrow" }, c.loopMax,
          el("span", { class: "hint", text: "max tasks per loop run" })));
    }

    function renderThreadList() {
      const nodes = [
        el("button", { class: "new-project", text: "＋ New project",
          onclick: newProjectDialog }),
        el("button", { class: "new-thread", text: "＋ New thread",
          onclick: newThread }),
      ];
      // Project nodes (sorted by name), each with its threads beneath.
      for (const p of [...projects].sort((a, b) =>
          String(a.name).localeCompare(String(b.name)))) {
        nodes.push(...projectNode(p, threads.filter(
          (t) => t.project === p.id)));
        if (openPlanKey === planKey(p.id)) {
          const drawer = planDrawer(planKey(p.id));
          if (drawer) nodes.push(drawer);
        }
      }
      // Unfiled: threads with null/absent project — including legacy chat
      // files written before projects existed — and threads pointing at a
      // deleted project id.
      const known = new Set(projects.map((p) => p.id));
      const unfiled = threads.filter((t) => !t.project || !known.has(t.project));
      const unfiledActive = activeProject === null;
      const uHead = el("div", { class: "project-node unfiled-node" +
        (unfiledActive && projects.length ? " active" : ""),
        tabindex: 0, role: "button",
        title: projects.length
          ? "click: clear active project (Unfiled)" : "no projects yet" },
        el("span", { class: "proj-caret", text: collapsed.has("__unfiled")
          ? "▸" : "▾" }),
        el("span", { class: "thread-title", text: "Unfiled" }),
        el("span", { class: "thread-meta", text: `${unfiled.length}` }),
        planChip(null), planToggleBtn(null));
      uHead.querySelector(".proj-caret").addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (collapsed.has("__unfiled")) collapsed.delete("__unfiled");
        else collapsed.add("__unfiled");
        saveCollapsed();
        renderThreadList();
      });
      uHead.addEventListener("click", () => {
        if (projects.length) setActiveProject(null);
      });
      nodes.push(uHead);
      if (!collapsed.has("__unfiled"))
        nodes.push(...unfiled.map(threadRow));
      if (openPlanKey === "default") {
        const drawer = planDrawer("default");
        if (drawer) nodes.push(drawer);
      }
      threadList.replaceChildren(...nodes);
    }

    // ------------------------------------------- workspace file explorer drawer
    // Right-side drawer on the chat page listing the ACTIVE project's workspace
    // via GET /api/fs/tree (one lazy level per expand, cached per panel
    // session). Strictly read-only: clicking a file inserts its relative path
    // into the composer at the caret. Hidden entirely with no active project;
    // width persisted under the scoped ui key (pass 3 #25).
    const EXP_WIDTH_KEY = skey("ui", "", "explorer-width");
    const explorerToggle = el("button", { class: "seg", text: "Files",
      title: "toggle workspace file explorer" });
    const explorerRefresh = el("button", {
      class: "small", text: "⟳", title: "reload the tree" });
    const explorerHiddenBtn = el("button", {
      class: "small", text: "···", title: "show/hide dotfiles" });
    const explorerTitle = el("span", { class: "explorer-title hint",
      text: "workspace" });
    const explorerTree = el("div", { class: "explorer-tree mono" });
    // Read-only file preview (GET /api/fs/read): opened from the eye button
    // on file rows; insert-at-caret stays the row click.
    const explorerPreview = el("div", { class: "explorer-preview hidden mono" });
    const explorerResize = el("div", { class: "explorer-resize",
      title: "drag to resize" });
    const explorerHost = el("aside", { class: "explorer hidden",
      "aria-label": "workspace files" },
      explorerResize,
      el("div", { class: "explorer-head" },
        explorerRefresh, explorerHiddenBtn, explorerTitle),
      explorerTree, explorerPreview);
    let explorerOpen = false;
    let explorerProjectId = null;     // project the cached tree belongs to
    let fsShowHidden = sget(skey("ui", "", "explorer-hidden"), "") === "1";
    const fsCache = new Map();  // `${pid}:${hidden}:${relPath}` -> {entries, truncated}

    function explorerWidth() {
      const w = parseInt(localStorage.getItem(EXP_WIDTH_KEY) || "", 10);
      return Number.isFinite(w) && w >= 180 && w <= 800 ? w : 260;
    }
    function applyExplorerWidth() {
      explorerHost.style.width = explorerWidth() + "px";
    }

    // No active project => drawer hidden, toggle disabled (clicking is a no-op).
    function syncExplorerAvailability() {
      const has = !!activeProject;
      explorerToggle.disabled = !has;
      explorerToggle.title = has
        ? "toggle workspace file explorer"
        : "select a project to browse its workspace";
      if (!has) {
        explorerOpen = false;
        explorerHost.classList.add("hidden");
      }
    }

    function explorerProjectChanged() {
      fsCache.clear();
      explorerProjectId = null;
      explorerTree.replaceChildren();
      explorerPreview.classList.add("hidden");
      explorerPreview.replaceChildren();
      delete explorerPreview.dataset.path;
      syncExplorerAvailability();
      if (explorerOpen && activeProject) loadExplorerRoot();
    }

    // Race-safe: the project is captured at call time and travels with the
    // request + cache key, so a project switch mid-fetch can never store the
    // OLD project's entries under the NEW project id. The hidden-flag lives
    // in the key too (toggling dotfiles invalidates naturally).
    async function fetchFsTree(relPath) {
      const pid = activeProject;
      if (!pid) throw new Error("no active project");
      const key = pid + ":" + (fsShowHidden ? "h" : "") + ":" + relPath;
      if (fsCache.has(key)) return fsCache.get(key);
      const d = await api("/api/fs/tree?project=" + encodeURIComponent(pid) +
        "&path=" + encodeURIComponent(relPath) +
        (fsShowHidden ? "&hidden=1" : ""));
      const res = { entries: d.entries || [], truncated: !!d.truncated };
      fsCache.set(key, res);
      return res;
    }

    // Read-only preview of one file via /api/fs/read (toggle on re-click).
    async function togglePreview(entry) {
      if (!explorerPreview.classList.contains("hidden") &&
          explorerPreview.dataset.path === entry.path) {
        explorerPreview.classList.add("hidden");
        explorerPreview.replaceChildren();
        return;
      }
      const pid = activeProject;
      explorerPreview.classList.remove("hidden");
      explorerPreview.replaceChildren(
        el("div", { class: "hint", text: "loading " + entry.path + "…" }));
      explorerPreview.dataset.path = entry.path;
      try {
        const d = await api("/api/fs/read?project=" + encodeURIComponent(pid) +
          "&path=" + encodeURIComponent(entry.path));
        if (explorerPreview.dataset.path !== entry.path) return; // switched away
        explorerPreview.replaceChildren(
          el("div", { class: "explorer-preview-head hint" },
            d.path + " · " + (d.binary ? "binary" : d.size + " bytes") +
            (d.truncated ? " (truncated)" : ""),
            el("button", { class: "small", text: "✕ close", onclick: () => {
              explorerPreview.classList.add("hidden");
              explorerPreview.replaceChildren();
            } })),
          d.binary
            ? el("div", { class: "hint", text: "(binary file — no preview)" })
            : el("pre", { class: "explorer-preview-body", text: d.content || "" }));
      } catch (e) {
        if (explorerPreview.dataset.path !== entry.path) return;
        explorerPreview.replaceChildren(el("div", { class: "hint",
          text: "⚠ " + e.message }));
      }
    }

    // Lazy tree node: dir rows fetch their children on first expand.
    function fsRow(entry) {
      const caret = entry.type === "dir"
        ? el("span", { class: "fs-caret", text: "▸" }) : null;
      const label = el("span", { class: "fs-name" +
        (entry.type === "dir" ? " dir" : ""), text: entry.name });
      const row = el("div", { class: "fs-row", tabindex: 0, role: "treeitem",
        title: entry.path }, caret, label);
      const wrap = el("div", { class: "fs-node" }, row);
      const kids = el("div", { class: "fs-kids hidden" });
      if (entry.type === "dir") {
        row.addEventListener("click", async () => {
          const open = !kids.classList.contains("hidden");
          kids.classList.toggle("hidden", open);
          if (caret) caret.textContent = open ? "▸" : "▾";
          if (open || kids.dataset.loaded) return;
          try {
            const res = await fetchFsTree(entry.path);
            kids.replaceChildren(...res.entries.map(fsRow));
            kids.dataset.loaded = "1";
            if (!res.entries.length)
              kids.appendChild(el("div", { class: "fs-empty hint", text: "(empty)" }));
            else if (res.truncated)
              kids.appendChild(el("div", { class: "fs-empty hint",
                text: "… too many entries to list" }));
          } catch (e) {
            kids.replaceChildren(el("div", { class: "fs-empty hint",
              text: "⚠ " + e.message }));
          }
        });
        wrap.appendChild(kids); // sibling of the row: children indent via .fs-kids
      } else {
        const insert = () => insertAtCaret(entry.path);
        const previewBtn = el("button", { class: "small fs-preview-btn",
          text: "👁", title: "preview file (read-only)" });
        previewBtn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          togglePreview(entry);
        });
        row.appendChild(previewBtn);
        row.addEventListener("click", insert);
        row.addEventListener("keydown", (e) => { if (e.key === "Enter") insert(); });
      }
      return wrap;
    }

    // Insert a file reference at the caret in the composer. Plain relative
    // path — matches the `file:line:` output convention of the agent's own
    // search_files tool without fabricating a line number.
    function insertAtCaret(text) {
      const start = input.selectionStart ?? input.value.length;
      const end = input.selectionEnd ?? input.value.length;
      input.value = input.value.slice(0, start) + text + input.value.slice(end);
      const pos = start + text.length;
      input.setSelectionRange(pos, pos);
      input.focus();
    }

    async function loadExplorerRoot() {
      if (!activeProject) return;
      explorerProjectId = activeProject;
      const proj = projects.find((p) => p.id === activeProject);
      explorerTitle.textContent = proj ? proj.name : "workspace";
      explorerTree.replaceChildren(el("div", { class: "hint", text: "loading…" }));
      try {
        const res = await fetchFsTree("");
        if (explorerProjectId !== activeProject) return; // project switched mid-fetch
        explorerTree.replaceChildren(...res.entries.map(fsRow));
        if (!res.entries.length)
          explorerTree.replaceChildren(empty("workspace is empty"));
        else if (res.truncated)
          explorerTree.appendChild(el("div", { class: "fs-empty hint",
            text: "… too many entries to list" }));
      } catch (e) {
        explorerTree.replaceChildren(el("div", { class: "hint",
          text: "⚠ " + e.message }));
      }
    }

    explorerToggle.addEventListener("click", () => {
      if (!activeProject) return; // disabled state: no-op
      explorerOpen = !explorerOpen;
      explorerHost.classList.toggle("hidden", !explorerOpen);
      if (explorerOpen && explorerProjectId !== activeProject)
        loadExplorerRoot();
    });
    // Full refresh: drop every cached level for this project (previously only
    // the root key was dropped, so expanded children stayed stale forever).
    explorerRefresh.addEventListener("click", () => {
      const prefix = activeProject + ":";
      for (const k of [...fsCache.keys()])
        if (k.startsWith(prefix)) fsCache.delete(k);
      loadExplorerRoot();
    });
    explorerHiddenBtn.addEventListener("click", () => {
      fsShowHidden = !fsShowHidden;
      sset(skey("ui", "", "explorer-hidden"), fsShowHidden ? "1" : "");
      fsCache.clear(); // hidden flag is part of every cache key
      if (activeProject && explorerOpen) loadExplorerRoot();
    });
    explorerHiddenBtn.classList.toggle("active", fsShowHidden);
    // drag-to-resize: mousedown on the left edge handle, clamp 180..800 px.
    explorerResize.addEventListener("mousedown", (ev) => {
      ev.preventDefault();
      const startX = ev.clientX;
      const startW = explorerHost.getBoundingClientRect().width;
      const onMove = (e) => {
        const w = Math.min(800, Math.max(180, startW + (startX - e.clientX)));
        explorerHost.style.width = w + "px";
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        sset(EXP_WIDTH_KEY,
          String(Math.round(explorerHost.getBoundingClientRect().width)));
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });

    function newThread() {
      // Switching away from a running thread is NOT stopping it: we drop the
      // local stream reader only — the run keeps executing server-side and
      // re-attaches when the thread is reopened. (The server enforces one
      // run per thread; different threads may run concurrently.)
      detachStream();
      viewSession++;
      threadId = null;
      sdel(skey("ui", "", "last-thread")); // don't reopen a discarded draft
      threadWasProjectless = true; // fresh thread adopts the active project
      messages = [];
      threadSummary = null; // backlog #6: rolling compaction summary
      renderMsgs();
      renderThreadList();
      input.focus();
    }

    // Drop the local reader for whatever stream this view is attached to.
    // The server-side run is untouched (its frames keep buffering and the
    // worker persists the transcript when it finishes).
    function detachStream() {
      if (abort) abort.abort();          // stops READING only
      if (adoptTimer) { clearInterval(adoptTimer); adoptTimer = null; }
      adoptedRun = null;
      runId = null;
      setRunning(false);
      hint.textContent = "";
    }

    async function fetchActiveRuns() {
      try { return (await api("/api/chat/runs")).runs || []; }
      catch { return []; }
    }

    function clearAdopt() {
      if (adoptTimer) { clearInterval(adoptTimer); adoptTimer = null; }
      adoptedRun = null;
      runId = null;
    }

    // Re-attach to a run that is still executing on the server (the user
    // navigated away mid-run and came back). Streams every frame buffered
    // so far — everything produced while nobody was watching — then follows
    // live until run.done/run.error. The Stop button points at the real
    // run_id. Leaving again just drops this reader; a 404 means the run
    // finished and was pruned between listing and attaching -> reload.
    async function attachRun(run, forThread) {
      if (abort) abort.abort();
      if (adoptTimer) { clearInterval(adoptTimer); adoptTimer = null; }
      adoptedRun = { runId: run.run_id, thread: forThread };
      runId = run.run_id;
      const session = viewSession;
      setRunning(true);
      hint.textContent = "run in progress — started before this view";

      // Replay the turn locally. The server persists the transcript only
      // when the run ends, so render the prompt (registry carries it) plus
      // a live streaming bubble fed by the frame replay.
      if (run.prompt && !messages.some((m) => m.role === "user"
          && m.content === run.prompt)) {
        messages.push({ role: "user", content: run.prompt });
        renderMsgs();
      }
      stick = true;
      const bubble = streamingBubble();
      msgHost.appendChild(bubble.node);
      const cards = new Map(); // tool call id -> card

      abort = new AbortController();
      let ended = false;
      try {
        await streamSSEGet(
          "/api/chat/runs/" + encodeURIComponent(run.run_id) + "/stream",
          (kind, d) => {
            if (kind === "run.started") {
              ctxWindow = d.context_window || 0; renderCtxChip();
            }
            else if (kind === "message.delta") bubble.append(d.text || "");
            else if (kind === "message.reset") bubble.reset(d.text || "");
            else if (kind === "reasoning.delta") bubble.appendReasoning(d.text || "");
            else if (kind === "context.usage") {
              ctxUsed = d.prompt_tokens || ctxUsed; renderCtxChip();
            }
            else if (kind === "tool.call") {
              const card = toolCard(d);
              cards.set(d.id, card);
              bubble.addCard(card.node);
              scrollDown();
            }
            else if (kind === "tool.result") {
              const entry = cards.get(d.id);
              if (entry) entry.result(d);
              scrollDown();
            }
            else if (kind === "run.done") {
              ended = true;
              if (!bubble.text() && !cards.size) bubble.node.remove();
              else bubble.finish(runFooter(d)); // stop reason shows here too
            }
            else if (kind === "run.error") {
              ended = true;
              bubble.node.classList.add("errored");
              if (!bubble.text()) bubble.append("⚠ " + (d.message || "error"));
            }
          }, abort.signal);
      } catch (e) {
        if (e.name === "AbortError") return;   // detached: the view moved on
        if (e.code !== "no_such_run") toast(e.message, "error");
        ended = true;  // run already gone: reload shows the final transcript
      } finally {
        if (session === viewSession) {
          clearAdopt();
          setRunning(false);
          hint.textContent = "";
        }
      }
      // The run finished while we watched: the server persisted the complete
      // transcript — reload so the canonical version replaces our replay.
      if (session === viewSession && ended) openThread(forThread);
    }

    async function openThread(id) {
      // Already viewing this thread with a live stream attached: no-op (a
      // re-fetch would wipe the streaming bubble mid-run). When NOT running,
      // always fall through — attachRun calls openThread after the run ends
      // to replace its replay with the persisted transcript.
      if (id === threadId && running) return;
      if (id !== threadId) {
        // Switching threads mid-run: drop this view's reader. The run keeps
        // going server-side and re-attaches when we come back to it.
        detachStream();
        viewSession++;
      }
      try {
        const d = await api("/api/chat/threads/" + encodeURIComponent(id));
        threadId = id;
        // Remember which thread was open so returning to the panel reopens
        // it (and adopts any run still going on it) instead of dropping the
        // user into an empty chat that looks like the agent stopped.
        sset(skey("ui", "", "last-thread"), id);
        // A thread that already belongs to a project keeps it on re-save;
        // a projectless one adopts the current active project context.
        threadWasProjectless = !d.thread.project;
        messages = (d.thread.messages || []).filter(
          (m) => m && typeof m.role === "string");
        threadSummary = typeof d.thread.summary === "string"
          ? d.thread.summary : null;
        // The dropdown follows the thread: reopen a "sable" conversation and
        // the selector shows sable, not default.
        selectThreadProfile(d.thread.profile || "");
        renderMsgs();
        renderThreadList();
        // Run lifecycle: if this thread has an in-flight run (orphaned by a
        // previous navigation), re-attach to its live stream.
        for (const run of await fetchActiveRuns()) {
          if (run.thread === id) { attachRun(run, id); break; }
        }
      } catch (e) { toast(e.message, "error"); }
    }

    async function saveThread(profile) {
      const id = threadId || newId();
      // NOTE: we deliberately do NOT send `title` here. The backend treats an
      // explicit client title as an authoritative USER rename — sending the
      // first-message snippet on every save would clobber both the stage-2
      // LLM title upgrade and any double-click rename. Omitting the key lets
      // the server derive a label once (stage 1) and preserve whatever
      // provenanced title already exists.
      try {
        const d = await api("/api/chat/threads/" + encodeURIComponent(id),
          { method: "PUT", body: { messages, summary: threadSummary,
            ...(profile !== undefined ? { profile } : {}),
            // New threads land in the active project (null = Unfiled);
            // re-saving an open thread keeps its stored project unless the
            // user explicitly moved it (backend keeps stored value when the
            // key is absent — here we only send it for fresh threads or when
            // the opened thread had none).
            ...(threadWasProjectless ? { project: activeProject } : {}) } });
        threadId = d.thread.id;
        // Project stamped once: later saves keep the stored project even if
        // the active-project context changed in the meantime.
        threadWasProjectless = false;
        loadThreads();
      } catch (e) { toast("thread save failed: " + e.message, "error"); }
    }

    // ------------------------------------------------------- message render
    // An assistant message carries its whole run when we have it:
    //   {role:"assistant", content, reasoning?, tools?:[{id,name,args,result}],
    //    footer?:{trace_id,tokens_in,...}}
    // Reopening a thread re-renders the same transcript it streamed with —
    // thinking pane, tool cards, answer, run footer — instead of only the
    // final text (the old behavior made thinking/tool history "disappear").
    function staticReasoning(raw) {
      const bodyEl = el("div", { class: "reasoning-body md" });
      renderMarkdown(bodyEl, raw);
      const toggle = el("button", { class: "reasoning-toggle", text: "◂ thinking" });
      const pane = el("div", { class: "reasoning" }, toggle, bodyEl);
      bodyEl.classList.add("hidden"); // collapsed by default on reload
      toggle.addEventListener("click", () => {
        const open = bodyEl.classList.toggle("hidden") === false;
        toggle.textContent = open ? "▾ thinking" : "◂ thinking";
      });
      return pane;
    }

    function staticToolCard(t) {
      const card = toolCard({ id: t.id, name: t.name, args: t.args });
      if (t.result) card.result(t.result);
      return card.node;
    }

    function msgBubble(m) {
      const b = el("div", { class: "msg " + (m.role || "") });
      if (m.role === "assistant") {
        if (m.reasoning) b.appendChild(staticReasoning(String(m.reasoning)));
        for (const t of (Array.isArray(m.tools) ? m.tools : []))
          b.appendChild(staticToolCard(t));
        const bodyEl = el("div", { class: "md" });
        renderMarkdown(bodyEl, String(m.content ?? ""));
        b.appendChild(bodyEl);
        if (m.footer && typeof m.footer === "object")
          b.appendChild(runFooter(m.footer));
      } else if (m.role === "user") {
        const bodyEl = el("div", { class: "md" });
        renderMarkdown(bodyEl, String(m.content ?? ""));
        b.appendChild(bodyEl);
      } else {
        b.appendChild(el("pre", { text: String(m.content ?? "") }));
      }
      return b;
    }

    function renderMsgs() {
      msgHost.replaceChildren();
      if (!messages.length) {
        msgHost.appendChild(empty(
          "No messages. Agent mode runs a task through the sable runner " +
          "(recorded as a trace); Playground talks raw to the served model."));
        return;
      }
      for (const m of messages) msgHost.appendChild(msgBubble(m));
    }

    // Streaming turn (Zoo-code style): ONE assistant block per run. A segment
    // host keeps everything in chronological order — thinking panes, text
    // segments and tool cards splice into the same column, so the run reads as
    // one narrative instead of loose siblings scattered between bubbles.
    // Content deltas extend the current md segment; a tool card or a new
    // thinking burst closes it so later text starts a fresh segment after it.
    //
    // Thinking segmentation (crash fix): each CONTIGUOUS burst of reasoning
    // deltas gets its own collapsible pane instead of one bubble that grows
    // for the whole task. Live deltas append to a single text node (O(1) DOM
    // work per token); markdown renders ONCE when the segment closes. A view
    // cap bounds DOM size on runaway streams — the full text is still
    // persisted with the message. The old design re-ran renderMarkdown over
    // the entire accumulated reasoning on every single delta: O(n²) parsing
    // plus one unbounded DOM node, which froze and eventually crashed the tab
    // on long thinking-heavy runs.
    function streamingBubble() {
      let raw = "";
      const reasoningParts = [];  // closed thinking segments, chronological
      let seg = null;             // current md segment element
      let segRaw = "";            // accumulated text of the current segment
      const ensureSeg = () => {
        if (seg) return seg;
        seg = el("div", { class: "md" });
        bubble.appendChild(seg);
        segRaw = "";
        return seg;
      };

      const THINK_VIEW_CAP = 20000;   // chars rendered live per pane
      let thinkPane = null, thinkToggle = null, thinkBody = null;
      let thinkText = null;           // live text node (appendData = O(1))
      let thinkRaw = "";
      let thinkCapped = false;

      function closeThink() {
        if (!thinkPane) return;
        reasoningParts.push(thinkRaw);
        // Final paint: markdown once for the finished segment (capped).
        thinkBody.replaceChildren();
        renderMarkdown(thinkBody,
          thinkRaw.length > THINK_VIEW_CAP
            ? thinkRaw.slice(0, THINK_VIEW_CAP) + "\n… [view truncated]"
            : thinkRaw);
        // Collapse: the reader follows the answer, not old thoughts.
        thinkBody.classList.add("hidden");
        thinkToggle.textContent = "◂ thinking";
        thinkPane = null; thinkToggle = null; thinkBody = null;
        thinkText = null; thinkRaw = ""; thinkCapped = false;
      }

      const bubble = el("div", { class: "msg assistant streaming" });
      return {
        node: bubble,
        appendReasoning(text) {
          if (!thinkPane) {
            thinkToggle = el("button", { class: "reasoning-toggle",
              text: "▾ thinking…" });
            thinkBody = el("div", { class: "reasoning-body md" });
            thinkPane = el("div", { class: "reasoning" },
              thinkToggle, thinkBody);
            // Capture THIS pane's nodes: closeThink() reassigns the outer
            // refs, and an already-closed pane's toggle must keep working.
            const tToggle = thinkToggle, tBody = thinkBody;
            tToggle.addEventListener("click", () => {
              const open = !tBody.classList.toggle("hidden");
              tToggle.textContent = open ? "▾ thinking" : "◂ thinking";
            });
            bubble.appendChild(thinkPane);
            seg = null;   // thinking interrupts the text segment
          }
          if (thinkCapped) { thinkRaw += text; return; }  // keep raw for save
          thinkRaw += text;
          if (thinkText) thinkText.appendData(text);
          else thinkBody.replaceChildren(
            thinkText = document.createTextNode(thinkRaw));
          if (thinkRaw.length > THINK_VIEW_CAP) {
            thinkCapped = true;
            thinkBody.replaceChildren(document.createTextNode(
              thinkRaw.slice(0, THINK_VIEW_CAP) +
              "\n… [streaming view truncated — full text saved with the message]"));
            thinkText = null;
          }
          scrollDown();
        },
        append(text) {
          if (thinkPane) closeThink();  // answer started: seal the thought
          // Acquire the segment BEFORE touching segRaw: ensureSeg() resets
          // segRaw when it opens a fresh segment (first delta after a tool
          // card), so appending first silently swallowed that chunk — every
          // text segment lost its opening words until the thread reloaded.
          const target = ensureSeg();
          raw += text;
          segRaw += text;
          renderMarkdown(target, segRaw);
          scrollDown();
        },
        // message.reset: the runner's stream→non-stream fallback replaced
        // already-streamed partial content — drop it and show the one-shot
        // authoritative text instead of leaving stale duplicates.
        reset(text) {
          raw = text || "";
          if (seg) seg.remove();
          seg = null;
          segRaw = "";
          // Re-open a segment and seed it with the authoritative text so
          // later message.delta appends continue from it, not from "".
          if (raw) { ensureSeg(); segRaw = raw; renderMarkdown(seg, segRaw); }
          scrollDown();
        },
        // Splice a tool card into the turn column (closes thought + segment:
        // the tool call was what the agent was thinking about).
        addCard(node) {
          closeThink();
          seg = null;
          bubble.appendChild(node);
        },
        // Footer / stop-reason go at the very end of the block.
        finish(node) { closeThink(); bubble.appendChild(node); },
        text: () => raw,
        reasoning: () => [...reasoningParts, thinkRaw]
          .filter(Boolean).join("\n\n"),
      };
    }

    // Phase 2 (delegation): nested card for delegate_task/check_delegation
    // results — child agent, status chip, result snippet, child-trace link.
    const DELEGATION_TOOLS = new Set(["delegate_task", "check_delegation"]);

    function parseDelegation(preview) {
      try {
        const d = JSON.parse(String(preview ?? ""));
        return (d && typeof d === "object" && d.status) ? d : null;
      } catch { return null; }  // non-JSON result: plain preview instead
    }

    function delegationCard(d, onStatus) {
      const chip = el("span", { class: "chip deleg-status " + (d.status || ""),
        text: d.status || "unknown" });
      const meta = el("div", { class: "hint mono" },
        "agent " + (d.agent || "?") +
        (d.delegation_id ? " · " + d.delegation_id : ""));
      const parts = [el("div", { class: "deleg-head" }, chip, meta)];
      if (d.error) parts.push(el("pre", { class: "log bad" }, String(d.error)));
      const snippet = d.result || d.partial_output;
      if (snippet) parts.push(el("pre", { class: "log" }, String(snippet).slice(0, 1200)));
      if (d.child_trace_id) {
        parts.push(el("a", { href: "#/traces/" + encodeURIComponent(d.child_trace_id),
          class: "mono", text: "child trace " + String(d.child_trace_id).slice(0, 10) }));
      }
      const node = el("div", { class: "msg deleg-card" }, ...parts);
      return {
        node,
        // Live update from a delegation.update bus event (same delegation id).
        status(rec) {
          chip.textContent = rec.status || chip.textContent;
          chip.className = "chip deleg-status " + (rec.status || "");
          if (rec.child_trace_id && !node.querySelector("a")) {
            parts.push(el("a", { href: "#/traces/" + encodeURIComponent(rec.child_trace_id),
              class: "mono", text: "child trace " + String(rec.child_trace_id).slice(0, 10) }));
            node.replaceChildren(...parts);
          }
          if (onStatus) onStatus(rec);
        },
      };
    }

    // Phase 4 (post-write checks): write_file/patch results can carry an
    // editor-style verdict appended by core/tools.py — "✗ SYNTAX ERROR in …"
    // (+ parser body, and/or a "⚠ NEW LSP DIAGNOSTICS" block) or a subtle
    // "✓ syntax ok". Render failures as a visible red block (mirroring the
    // Phase 2 delegation-card pattern: split the marker line from the error
    // body) so the agent's next turn and the user both see it; successes get
    // a dim ✓ chip in the card head.
    const SYNTAX_FAIL_MARK = "✗ SYNTAX ERROR";
    const SYNTAX_OK_MARK = "✓ syntax ok";
    const LSP_MARK = "⚠ NEW LSP DIAGNOSTICS";

    function syntaxCheck(preview) {
      const s = String(preview ?? "");
      const fi = s.indexOf(SYNTAX_FAIL_MARK);
      if (fi < 0 && !s.includes(LSP_MARK)) return null;
      const failIdx = fi >= 0 ? fi : s.indexOf(LSP_MARK);
      const head = s.slice(failIdx).split("\n", 1)[0];
      const body = s.slice(failIdx).slice(head.length).trim();
      return { head, body, ok: fi < 0 };  // LSP-only block = warning, not fail
    }

    // Tool call card: collapsible args + result preview.
    function toolCard(tc, onDelegation) {
      const body = el("div", { class: "tool-body hidden" });
      body.appendChild(el("pre", { class: "log json",
        text: JSON.stringify(tc.args ?? {}, null, 2) }));
      const head = el("button", { class: "tool-head" },
        el("span", { class: "tool-state", text: "●" }),
        el("strong", { text: tc.name || "tool" }),
        el("span", { class: "hint", text: "tool call" }));
      const card = el("div", { class: "msg tool-card" }, head, body);
      head.addEventListener("click", () => body.classList.toggle("hidden"));
      return {
        node: card,
        result(res) {
          const check = syntaxCheck(res.preview);
          // A failed post-write check flips the card state even though the
          // tool itself succeeded (the write happened; the content is broken).
          const bad = !res.ok || (check && !check.ok);
          head.querySelector(".tool-state").textContent = bad ? "✗" : "✓";
          head.querySelector(".tool-state").classList.add(bad ? "bad" : "ok");
          if (!bad && String(res.preview ?? "").includes(SYNTAX_OK_MARK)) {
            // subtle ✓ for clean syntax (success stays quiet by design)
            head.querySelector(".tool-state").title = "syntax ok";
          }
          const deleg = DELEGATION_TOOLS.has(tc.name)
            ? parseDelegation(res.preview) : null;
          if (deleg) {
            // Nested delegation card instead of a raw JSON blob.
            body.appendChild(el("div", { class: "hint", text: "delegation" }));
            const dc = delegationCard(deleg);
            body.appendChild(dc.node);
            if (onDelegation && deleg.delegation_id) onDelegation(deleg.delegation_id, dc);
            if (!deleg.status || deleg.status === "running") body.classList.remove("hidden");
            return;
          }
          const pre = el("pre", { class: "log" },
            String(res.preview ?? "") + (res.truncated ? "\n… [truncated]" : ""));
          if (check) {
            // Visible verdict block; auto-expand the card on failure so the
            // error body is seen without a click.
            const mark = el("div", {
              class: "syntax-mark " + (check.ok ? "warn" : "bad"),
              text: check.head });
            body.appendChild(mark);
            if (check.body) {
              body.appendChild(el("pre", {
                class: "log syntax-body " + (check.ok ? "" : "bad"),
                text: check.body }));
            }
            if (!check.ok) body.classList.remove("hidden");
          }
          body.appendChild(el("div", { class: "hint", text: "result" }));
          body.appendChild(pre);
        },
      };
    }

    // Context tracker (reference style): rolling view of how much of the
    // served context window the current run has used. `context.usage` events
    // arrive per model call; run.done carries the final numbers.
    let ctxWindow = 0;          // server context window (run.started)
    let ctxUsed = 0;            // latest prompt-token count
    const ctxChip = el("span", { class: "ctx-chip hint mono hidden" });
    function renderCtxChip() {
      if (!ctxWindow || !ctxUsed) {
        ctxChip.classList.add("hidden");
        return;
      }
      ctxChip.classList.remove("hidden");
      const pct = Math.min(100, Math.round((ctxUsed / ctxWindow) * 100));
      ctxChip.textContent = `context ${fmtNum(ctxUsed, 0)}/${fmtNum(ctxWindow, 0)} tok (${pct}%)`;
      ctxChip.title = "prompt tokens in the served context window";
    }

    // Human-readable explanation for runner-level failure_reason values so a
    // controlled stop never looks like the agent "just stopped". Unknown codes
    // fall through verbatim (future-proof against new runner reasons).
    const STOP_REASON_TEXT = {
      max_iterations:
        "stopped: hit the tool-call iteration limit — say “continue” or raise agent.max_iterations",
      guardrail_halt:
        "stopped by loop guardrail (repeated tool failures) — check the last tool card",
      cancelled: "stopped: cancelled by user",
    };
    function stopReasonText(reason) {
      if (!reason) return "";
      const base = String(reason).split(":")[0];
      return STOP_REASON_TEXT[base] || ("stopped: " + reason);
    }

    function runFooter(info) {
      const f = el("div", { class: "run-footer" });
      if (info.failure_reason) {
        f.appendChild(el("span", { class: "run-stop-reason",
          text: stopReasonText(info.failure_reason) }));
      }
      if (info.trace_id) {
        const a = el("a", { href: "#/traces/" + encodeURIComponent(info.trace_id),
          class: "mono", text: "trace " + String(info.trace_id).slice(0, 10) });
        f.appendChild(a);
      } else f.appendChild(el("span", { class: "hint", text: "no trace" }));
      const bits = [];
      if (info.tokens_in != null) bits.push(`${info.tokens_in}→${info.tokens_out ?? "?"} tok`);
      if (info.duration_s != null) bits.push(fmtNum(info.duration_s, 1) + "s");
      if (info.backend) bits.push(info.backend);
      f.appendChild(el("span", { class: "hint mono", text: bits.join(" · ") }));
      // Context bar: last window usage as a proportion of the server window.
      const win = info.context_window || ctxWindow;
      if (win && info.tokens_in != null) {
        const pct = Math.min(100, Math.round((info.tokens_in / win) * 100));
        f.appendChild(el("span", { class: "ctx-bar", title:
          `${info.tokens_in} of ${win} context tokens (${pct}%)` },
          el("span", { class: "ctx-bar-fill",
            style: `width:${pct}%` })));
      }
      return f;
    }

    // ------------------------------------------------------------ run flow
    function setRunning(on) {
      running = on;
      sendBtn.disabled = on;
      input.disabled = on;
      stopBtn.classList.toggle("hidden", !on);
      hint.textContent = on ? "running…" : "";
    }

    async function send() {
      let text = input.value.trim();
      const imgs = pendingImages.slice();
      if ((!text && !imgs.length) || running) return;
      input.value = "";
      pendingImages = [];
      renderAttachRow();
      stick = true;

      if (mode === "agent") await runAgent(text, imgs);
      else {
        // Playground is raw passthrough: no image pipeline there.
        if (imgs.length) toast("images are agent-mode only", "info");
        if (!text) return;
        await runProxy(text);
      }
    }

    // The server only ever needs role/content pairs: strip client-side
    // display metadata (image names, reasoning, tool cards, footer) before
    // sending history.
    function wireHistory(msgs) {
      return msgs.map(({ images, reasoning, tools, footer, ...rest }) => rest);
    }

    async function runAgent(promptText, imgs = []) {
      const session = viewSession;   // guards the tail against thread switches
      setRunning(true);
      // A brand-new thread gets its id BEFORE the run starts, and the thread
      // file is created immediately (fire-and-forget initial save). The old
      // flow assigned ids only when the finished run was saved, so a fresh
      // chat existed nowhere but in this page's memory mid-run: navigating
      // away left no `thread` on the run (the server persisted nothing for
      // thread-less runs) and no last-thread to reopen — the thread simply
      // vanished while the agent kept executing. With the id fixed up front:
      // the run carries it, the worker persists the transcript when done,
      // and returning to chat reopens the thread and re-attaches live.
      let freshThread = false;
      if (!threadId) {
        threadId = newId();
        freshThread = true;
        sset(skey("ui", "", "last-thread"), threadId);
        // Awaited: the file must exist before the run can persist into it,
        // and a late-resolving empty save could otherwise clobber a very
        // fast worker-persisted transcript.
        await saveThread(profileSel.value || null);  // <home>/chat/<id>.json
      }
      messages.push({ role: "user", content: promptText,
        ...(imgs.length ? { images: imgs.map((i) => i.name) } : {}) });
      renderMsgs();
      const bubble = streamingBubble();
      msgHost.appendChild(bubble.node);
      const cards = new Map(); // tool call id -> {card, rec}
      const delegCards = new Map(); // delegation id -> nested card
      const toolRecs = [];   // [{id,name,args,result}] in arrival order
      let footerInfo = {};
      let errored = false;
      let conflicted = false;   // 409 run_in_flight: the turn never ran
      let aborted = false;      // stream aborted (Stop / navigation away)

      // Phase 2 (delegation): live status flips for open nested cards.
      const offDeleg = bus.on("delegation.update", (rec) => {
        const dc = rec && delegCards.get(rec.delegation_id);
        if (dc) { dc.status(rec); scrollDown(); }
      });

      abort = new AbortController();
      try {
        await streamSSE("/api/chat/stream",
          { prompt: promptText,
            // Strip client-side display metadata from history — the server
            // only ever needs role/content.
            history: wireHistory(messages.slice(0, -1)),
            previous_summary: threadSummary ?? undefined,
            profile: profileSel.value || undefined,
            // Image support: base64 data URLs for this turn (server-side
            // saved + marker-injected; history keeps only the names).
            images: imgs.length ? imgs.map((i) => i.dataUrl) : undefined,
            // Projects: active-project context rides on the run so traces
            // record it and the runner can resolve the project workspace.
            project: activeProject || undefined,
            // Phase 2 (delegation): thread id lets the delegation tools write
            // lineage entries straight into this thread's JSON.
            thread: threadId || undefined },
          (kind, d) => {
            if (kind === "run.started") {
              runId = d.run_id;
              ctxWindow = d.context_window || 0;
              ctxUsed = 0; renderCtxChip();
            }
            else if (kind === "message.delta") bubble.append(d.text || "");
            // stream→non-stream fallback: swap the partially streamed text
            // for the authoritative one-shot reply (agent_runner #6).
            else if (kind === "message.reset") bubble.reset(d.text || "");
            else if (kind === "reasoning.delta") bubble.appendReasoning(d.text || "");
            else if (kind === "context.usage") {
              ctxUsed = d.prompt_tokens || ctxUsed; renderCtxChip();
            }
            else if (kind === "tool.call") {
              const card = toolCard(d, (did, dc) => delegCards.set(did, dc));
              const rec = { id: d.id, name: d.name, args: d.args ?? {},
                result: null };
              cards.set(d.id, { card, rec });
              toolRecs.push(rec);
              bubble.addCard(card.node); // chronological, inside the turn
              scrollDown();
            }
            else if (kind === "tool.result") {
              const entry = cards.get(d.id);
              if (entry) {
                entry.card.result(d);
                entry.rec.result = { ok: !!d.ok, preview: d.preview ?? "",
                  truncated: !!d.truncated };
              } else {
                const c = toolCard(d, (did, dc) => delegCards.set(did, dc));
                c.result(d);
                toolRecs.push({ id: d.id, name: d.name, args: d.args ?? {},
                  result: { ok: !!d.ok, preview: d.preview ?? "",
                    truncated: !!d.truncated } });
                bubble.addCard(c.node);
              }
              scrollDown();
            }
            else if (kind === "turn.done") { /* keep footer for run.done */ }
            else if (kind === "run.done") {
              footerInfo = d;
              // backlog #6: persist the rolling compaction summary with the
              // thread so the next run updates it instead of restarting cold.
              if (typeof d.summary === "string" && d.summary) {
                threadSummary = d.summary;
              }
              if (!bubble.text() && !toolRecs.length) bubble.node.remove();
              else bubble.finish(runFooter(d)); // stop reason shows here too
            }
            else if (kind === "run.error") {
              errored = true;
              toast("run error: " + (d.message || "unknown"), "error");
              bubble.node.classList.add("errored");
              if (!bubble.text()) bubble.append("⚠ " + (d.message || "error"));
            }
          }, abort.signal);
      } catch (e) {
        if (e.name === "AbortError") {
          aborted = true;
          hint.textContent = "stopped";
        }
        else if (e.code === "run_in_flight") {
          // Server-side single-flight won the race: a run is already in
          // flight for this thread (another panel/tab, or one we just lost
          // track of). Reload after a beat — openThread re-attaches to the
          // live run (Stop button comes back, tokens keep streaming).
          conflicted = true;
          const conflictThread = threadId;
          toast(e.message || "a run is already in flight for this thread",
            "info");
          setTimeout(() => {
            if (session === viewSession && conflictThread) {
              detachStream();  // drop this dead attempt's state first
              openThread(conflictThread);
            }
          }, 700);
        }
        else { errored = true; toast(String(e.message || e), "error"); }
      } finally {
        offDeleg();
        // Only clear view state if nobody switched threads mid-stream:
        // detachStream/attachRun already reset it (and may own the new
        // abort/runId — clobbering those would break a live re-attach).
        if (session === viewSession) {
          runId = null;
          abort = null;
          setRunning(false);
        }
      }

      // The user switched threads / navigated away while this stream was
      // aborted: the view now shows a different conversation. The run kept
      // executing server-side and its transcript is persisted there — do not
      // touch the new view's messages.
      if (session !== viewSession) return;

      // Persist the WHOLE turn — reasoning, tool cards, footer — so reopening
      // the thread re-renders everything instead of only the final text. The
      // compaction summary stays out of the per-message footer (it can be
      // large and lives at thread level anyway).
      const content = bubble.text();
      const reasoning = bubble.reasoning();
      if (content || toolRecs.length) {
        const { summary, ...foot } = footerInfo;
        messages.push({
          role: "assistant", content,
          ...(reasoning ? { reasoning } : {}),
          ...(toolRecs.length ? { tools: toolRecs } : {}),
          ...(Object.keys(foot).length ? { footer: foot } : {}),
        });
      }
      if (conflicted) {
        // Single-flight conflict: this turn never ran — drop the buffered
        // user message (the thread reload shows the real state).
        messages.pop();
        renderMsgs();
      } else if (aborted && threadId) {
        // Stream aborted mid-run (Stop or navigation away): the run keeps
        // executing on the server, which persists the COMPLETE turn when it
        // finishes (POST /api/chat/stream writes the thread file itself).
        // Saving our partial buffer here would race that write, and a late
        // arrival could clobber the finished transcript — so don't. Keep the
        // in-memory view as-is; returning to chat reloads from disk.
      } else if (errored && !content && !toolRecs.length) {
        messages.pop(); // dangling user msg
        if (freshThread) {
          // Nothing ever ran on the thread we just created for this turn —
          // remove the empty file so a dead "0 msg" thread does not linger
          // in the list, and fall back to a clean draft state.
          try {
            await api("/api/chat/threads/" + encodeURIComponent(threadId),
              { method: "DELETE" });
          } catch { /* best effort */ }
          threadId = null;
          sdel(skey("ui", "", "last-thread"));
          threadWasProjectless = true;
        }
      } else if (!errored) {
        saveThread(profileSel.value || null);
      } else {
        renderMsgs();
      }
    }

    async function runProxy(promptText) {
      setRunning(true);
      messages.push({ role: "user", content: promptText });
      renderMsgs();
      const bubble = streamingBubble();
      msgHost.appendChild(bubble.node);
      let raw = "";
      let errored = false;
      abort = new AbortController();
      try {
        await streamSSE("/api/proxy/chat",
          { messages: messages.slice(0, -1).map(
            ({ role, content }) => ({ role, content })), stream: true },
          (kind, d) => {
            if (kind === "proxy.error") {
              errored = true;
              toast("proxy error: " + (d.message || d.detail || d.status), "error");
              bubble.node.classList.add("errored");
              if (!raw) bubble.append("⚠ " + (d.message || d.detail ||
                String(d.status || "upstream error")));
              return;
            }
            if (d.raw === "[DONE]") return;
            const del = d?.choices?.[0]?.delta || {};
            const rc = del.reasoning_content || del.reasoning;
            if (typeof rc === "string" && rc) bubble.appendReasoning(rc);
            const delta = del.content;
            if (typeof delta === "string" && delta) { raw += delta; bubble.append(delta); }
          }, abort.signal);
      } catch (e) {
        if (e.name === "AbortError") hint.textContent = "stopped";
        else { errored = true; toast(String(e.message || e), "error"); }
      } finally { abort = null; setRunning(false); }

      // Same persistence rationale as runAgent: keep the thinking so a
      // reopened playground thread shows the full turn.
      const reasoning = bubble.reasoning();
      if (raw) messages.push({ role: "assistant", content: raw,
        ...(reasoning ? { reasoning } : {}) });
      if (!errored) saveThread();
      else renderMsgs();
    }

    async function stopRun() {
      // Server-side cancel: the runner polls the flag at its loop boundaries
      // (before each model call, between tool calls) and stops cleanly — an
      // in-flight request or tool call finishes first, then the run ends with
      // failure_reason "cancelled". Say so honestly.
      if (runId) {
        try {
          const r = await api("/api/chat/cancel", { method: "POST",
            body: { run_id: runId } });
          hint.textContent = r.cancelling
            ? "cancelling at next boundary…" : "already finished";
        } catch (e) { toast(e.message, "error"); }
      }
      // Abort the HTTP stream itself so the UI unlocks immediately.
      abort?.abort();
    }

    // -------------------------------------------------------------- wiring
    function setMode(m) {
      mode = m;
      sset(skey("ui", "", "mode"), m);
      modeAgent.classList.toggle("active", m === "agent");
      modePlay.classList.toggle("active", m === "playground");
      profileSel.classList.toggle("hidden", m !== "agent");
      input.placeholder = m === "agent"
        ? "Ask the agent a task… (recorded as a trace)"
        : "Chat with the served model (raw playground)…";
    }
    modeAgent.addEventListener("click", () => setMode("agent"));
    modePlay.addEventListener("click", () => setMode("playground"));
    sendBtn.addEventListener("click", send);
    stopBtn.addEventListener("click", stopRun);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
    });

    mount.replaceChildren(el("div", { class: "chat-layout" },
      threadList,
      el("div", { class: "chat-main" },
        el("div", { class: "toolbar" }, modeAgent, modePlay, profileSel,
          explorerToggle, ctxChip, hint),
        msgHost,
        el("div", { class: "composer" }, attachRow, input,
          el("div", { class: "btnrow" }, imgInput, attachBtn,
            stopBtn, sendBtn))),
      explorerHost));

    setMode(mode);
    renderMsgs();
    applyExplorerWidth();
    syncExplorerAvailability();
    loadProjects().then(() => { renderThreadList(); syncExplorerAvailability(); loadAllPlans(); });
    loadProfiles();
    // WP12: live plan updates from any surface (agent runs, CLI, the loop).
    unsubs.push(bus.on("plan.updated", (p) => {
      // Refresh just the changed project's drawer/chip (cheap single GET).
      loadPlan((p && p.project) || "default");
    }));
    unsubs.push(bus.on("plan.loop", () => {
      for (const key of planState.keys()) loadPlan(key);
      renderThreadList();
    }));
    // Reopen the thread that was open before navigating away (bug fix: the
    // panel used to start blank, hiding both the transcript and any run
    // still executing on the server — which read as "the agent stopped").
    // openThread also adopts an in-flight run for that thread.
    const lastThread = sget(skey("ui", "", "last-thread"));
    loadThreads().then(() => {
      if (lastThread && threads.some((t) => t.id === lastThread))
        openThread(lastThread);
    });
    input.focus();

    return { destroy() {
      // Navigation is NOT cancellation: the run keeps executing on its
      // worker thread (by design — leaving the page shouldn't kill the work).
      // detachStream only stops READING; the server buffers every frame, so
      // the next panel re-attaches via /api/chat/runs/<id>/stream and sees
      // everything produced while away (and can Stop the real run).
      detachStream();
      viewSession++;   // invalidate any in-flight async tail for this view
      for (const u of unsubs) u();
    } };
  },
};

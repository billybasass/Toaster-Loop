// app.js — dashboard entry point: shell wiring (nav, hash router), header
// status bar (serve pill, hardware mini-gauge, token + bus indicators),
// global SSE bus toasts and overflow-refetch. ES module; same-origin only.

import { api, getToken } from "./js/api.js";
import { bus } from "./js/bus.js";
import * as router from "./js/router.js";
import { el, toast } from "./js/ui.js";

import overview from "./js/panels/overview.js";
import chat from "./js/panels/chat.js";
import serve from "./js/panels/serve.js";
import models from "./js/panels/models.js";
import hub from "./js/panels/hub.js";
import jobs from "./js/panels/jobs.js";
import traces from "./js/panels/traces.js";
import datasets from "./js/panels/datasets.js";
import train from "./js/panels/train.js";
import easy from "./js/panels/easy.js";
import iterate from "./js/panels/iterate.js";
import schedule from "./js/panels/schedule.js";
import profiles from "./js/panels/profiles.js";
import hardware from "./js/panels/hardware.js";
import tuning from "./js/panels/tuning.js";
import settings from "./js/panels/settings.js";
import options from "./js/panels/options.js";

// ------------------------------------------------------------------ routing
// Grouped rail: five sections keep 14 panels readable at a glance.
const NAV_GROUPS = [
  ["Loop", [["overview", "Overview", overview], ["chat", "Chat", chat]]],
  ["Serving", [["serve", "Serve", serve], ["models", "Models", models],
    ["hub", "Resource Hub", hub]]],
  ["Data", [["traces", "Traces", traces], ["datasets", "Datasets", datasets]]],
  ["Training", [["easy", "Easy", easy], ["train", "Train", train],
    ["iterate", "Iterate", iterate],
    ["schedule", "Schedule", schedule], ["jobs", "Jobs", jobs]]],
  ["Setup", [["profiles", "Profiles", profiles], ["hardware", "Hardware", hardware],
    ["tuning", "Tuning", tuning], ["options", "Options", options],
    ["settings", "Settings", settings]]],
];
const NAV = NAV_GROUPS.flatMap(([, items]) => items);

const navHost = document.getElementById("nav");
const outlet = document.getElementById("outlet");

for (const [section, items] of NAV_GROUPS) {
  const h = el("div", { class: "nav-section", text: section });
  navHost.appendChild(h);
  for (const [name, label] of items) {
    const a = el("a", { href: "#/" + name, class: "nav-item", text: label,
      dataset: { route: name } });
    navHost.appendChild(a);
  }
}

router.start(outlet, {
  onNavigate(name) {
    for (const a of navHost.querySelectorAll(".nav-item"))
      a.classList.toggle("active", a.dataset.route === name);
  },
});
for (const [name, , panel] of NAV) router.register(name, panel);
router.register("_fallback", {
  title: "Not found",
  render(mount) {
    mount.replaceChildren(
      el("h2", { text: "Not found" }),
      el("div", { class: "card" },
        el("p", { text: "Unknown route — check the address bar." }),
        el("p", { class: "hint", text:
          "Expected #/" + NAV.map(([n]) => n).join(" | #/") })));
  },
});
if (!location.hash) location.replace("#/chat");

// -------------------------------------------------------------- status bar
// One pill per managed server instance (the reserved "default" first), plus a
// codebase-index pill. /api/servers carries live running state for every
// instance; /api/index reports the indexer + its embedding server.
const servePill = el("span", { class: "pill pill-down", title: "serving status" }, "● –");
const serverPillsHost = el("span", { class: "server-pills", title: "server instances" });
const indexPill = el("span", { class: "pill pill-dim", title: "codebase index" }, "index …");
const hwGauge = el("span", { class: "hw", title: "cpu · mem · gpu" });
const busPill = el("span", { class: "pill pill-dim", title: "event stream" }, "events…");
const tokPill = el("span", {
  class: "pill pill-dim hidden", title: "session token stored",
}, "🔒");
const closeBtn = el("button", {
  class: "small danger", text: "⏻ Close", title: "Close Toaster Loop (stops the " +
  "owned llama-server and exits the dashboard process)",
});
closeBtn.addEventListener("click", async () => {
  const ok = confirm("Close Toaster Loop?\n\nThe dashboard stops and the llama-server " +
    "it started is shut down. Training jobs that are running will be interrupted.");
  if (!ok) return;
  closeBtn.disabled = true;
  closeBtn.textContent = "closing…";
  try {
    await api("/api/shutdown", { method: "POST", body: {} });
    document.body.textContent = "";
    document.body.append(el("div", { class: "closed-note" },
      el("h2", { text: "Toaster Loop closed" }),
      el("p", { class: "hint", text: "You can close this browser tab. " +
        "Restart with `toaster-loop launch`." })));
  } catch (e) {
    closeBtn.disabled = false;
    closeBtn.textContent = "⏻ Close";
    toast(`shutdown failed: ${e.message || e}`, "error");
  }
});
const statusbar = document.getElementById("statusbar");
statusbar.append(servePill, serverPillsHost, indexPill, hwGauge, busPill, tokPill, closeBtn);

function renderTokenPill() { tokPill.classList.toggle("hidden", !getToken()); }
renderTokenPill();
window.addEventListener("toaster_loop:token", renderTokenPill);

window.addEventListener("toaster_loop:bus", (ev) => {
  const on = ev.detail?.online;
  busPill.textContent = on ? "events ●" : "events ○";
  busPill.className = "pill " + (on ? "pill-ok" : "pill-warn");
});
bus.connect();

function serverPill(s) {
  const cls = s.running
    ? (s.mode === "attach" ? "pill pill-warn" : "pill pill-ok")
    : "pill pill-down";
  const title = `${s.name} (${s.role || "custom"}) — ` +
    (s.running ? `${s.mode} · ${s.model_name || ""}\n${s.base_url || ""}`
      : "not running");
  return el("a", { href: "#/serve", class: cls, title },
    "● ", el("span", { class: "mono", text: s.name }));
}

async function refreshServePill() {
  try {
    const d = await api("/api/servers");
    const servers = d.servers || [];
    // The legacy default pill keeps showing the served model name; named
    // instances each get their own compact pill.
    const def = servers.find((s) => s.default);
    if (def && def.running) {
      servePill.className = "pill pill-ok";
      servePill.title = `${def.mode} · ${def.model_name || ""}\n${def.base_url || ""}`;
      servePill.replaceChildren("● ", el("span", { class: "mono",
        text: (def.model_name || "up") }));
    } else if (def) {
      servePill.className = "pill pill-down";
      servePill.title = "default server not running";
      servePill.replaceChildren(document.createTextNode("● default down"));
    }
    const named = servers.filter((s) => !s.default);
    const up = named.filter((s) => s.running).length;
    serverPillsHost.replaceChildren(
      ...named.map(serverPill),
      named.length && !up
        ? el("span", { class: "pill pill-dim", title: "no named instance running" },
            `${named.length} down`)
        : null);
  } catch {
    servePill.className = "pill pill-warn";
    servePill.replaceChildren(document.createTextNode("● ?"));
  }
}

async function refreshIndexPill() {
  try {
    const st = await api("/api/index");
    if (!st.enabled) {
      indexPill.className = "pill pill-dim hidden";
      return;
    }
    indexPill.classList.remove("hidden");
    const embedRunning = !!(st.embedding && st.embedding.running);
    if (st.building) {
      indexPill.className = "pill pill-warn";
      indexPill.textContent = "indexing…";
      indexPill.title = "codebase index build in progress";
    } else if ((st.pending || 0) > 0) {
      indexPill.className = "pill pill-warn";
      indexPill.textContent = `index +${st.pending}`;
      indexPill.title = `${st.pending} chunk(s) still to embed — more to index ` +
        `(embed server ${embedRunning ? "up" : "down"})`;
    } else if (st.chunks > 0) {
      indexPill.className = "pill pill-ok";
      indexPill.textContent = `indexed ✓`;
      indexPill.title = `${st.chunks} chunks indexed` +
        (embedRunning ? " · embedding server up" : "");
    } else {
      indexPill.className = "pill pill-dim";
      indexPill.textContent = "index empty";
      indexPill.title = "codebase indexing enabled but nothing indexed yet";
    }
  } catch {
    indexPill.className = "pill pill-warn";
    indexPill.textContent = "index ?";
  }
}

// Hardware mini-gauge (full panel arrives in 5d; component kept small & dumb).
async function refreshHw() {
  try {
    const h = await api("/api/hardware");
    const bits = [];
    if (h.cpu_pct != null) bits.push(`cpu ${Math.round(h.cpu_pct)}%`);
    if (h.mem) bits.push(`mem ${Math.round(100 * h.mem.used_mib / (h.mem.total_mib || 1))}%`);
    const g = h.gpus && h.gpus[0];
    if (g) {
      const used = g.total_mib ? Math.round(100 * (1 - (g.free_mib || 0) / g.total_mib)) : 0;
      bits.push(`gpu ${used}%`);
    }
    hwGauge.title = "cpu · memory · gpu (polling ~5 s)\n" +
      (g ? `${g.name} — free ${g.free_mib ?? "?"} MiB / ${g.total_mib ?? "?"} MiB` : "no GPU detected");
    hwGauge.textContent = bits.length ? bits.join(" · ") : "hw –";
  } catch { hwGauge.textContent = ""; }
}

refreshServePill(); refreshIndexPill(); refreshHw();
setInterval(refreshServePill, 5000);
setInterval(refreshIndexPill, 5000);
setInterval(refreshHw, 5000);
bus.on("serve.state", () => refreshServePill());
bus.on("servers.changed", () => refreshServePill());
bus.on("index.build", (p) => {
  // Live-build feedback without waiting for the next poll.
  if (p.state === "running") {
    indexPill.className = "pill pill-warn";
    indexPill.textContent = "indexing…";
  } else {
    refreshIndexPill();
  }
});

// ------------------------------------------------------------- update banner
// Self-update notice (core/updater.py): the dashboard checks GitHub once at
// boot (throttled cache) and again here on load; an `update.available` bus
// event from the startup check flips the banner without a poll. Applying is
// always an explicit click: git pull --ff-only + reinstall in the background,
// then a restart prompt. Never nags twice per session once dismissed.
let updateDismissed = false;
const updateBanner = el("div", { class: "card update-banner hidden" });

function renderUpdateBanner(st) {
  if (updateDismissed || !st || !st.available) {
    updateBanner.classList.add("hidden");
    return;
  }
  const rv = st.remote_version ? `v${st.remote_version}` : "a new version";
  const applying = st.apply && st.apply.state === "running";
  const applyBtn = el("button", {
    class: "small primary", text: applying ? "updating…" : "Update now",
    disabled: !!applying,
  });
  applyBtn.addEventListener("click", async () => {
    if (!confirm(`Update Toaster Loop to ${rv}?\n\n` +
      "The app pulls from GitHub and reinstalls itself. Running training " +
      "jobs will NOT be interrupted, but the dashboard must restart " +
      "afterwards.")) return;
    applyBtn.disabled = true;
    applyBtn.textContent = "updating…";
    try {
      await api("/api/update/apply", { method: "POST", body: {} });
      pollUpdateApply();
    } catch (e) {
      toast(`update failed: ${e.message || e}`, "error");
      applyBtn.disabled = false;
      applyBtn.textContent = "Update now";
    }
  });
  const dismissBtn = el("button", { class: "small", text: "Dismiss" });
  dismissBtn.addEventListener("click", () => {
    updateDismissed = true;
    updateBanner.classList.add("hidden");
  });
  updateBanner.replaceChildren(
    el("strong", { text: `Toaster Loop ${rv} is available` }),
    el("span", { class: "hint", text:
      ` (you have v${st.current_version || "?"}) — ` }),
    el("a", { href: st.url || "#", target: "_blank", rel: "noopener",
      text: "release notes" }),
    el("span", { style: "flex:1" }),
    applyBtn, dismissBtn);
  updateBanner.classList.remove("hidden");
}

async function pollUpdateApply() {
  for (let i = 0; i < 240; i++) {
    await new Promise((r) => setTimeout(r, 1500));
    let rec = null;
    try { rec = await api("/api/update/apply"); } catch { continue; }
    if (rec.state === "done") {
      toast("✓ Toaster Loop updated — restart to run the new version", "ok");
      updateBanner.replaceChildren(
        el("strong", { text: "Update applied" }),
        el("span", { class: "hint",
          text: " Close and relaunch Toaster Loop to finish." }));
      return;
    }
    if (rec.state === "error") {
      toast(`update failed: ${rec.error || "see log"}`, "error");
      renderUpdateBanner({ available: true, current_version: null });
      return;
    }
  }
}

async function refreshUpdateBanner() {
  try { renderUpdateBanner(await api("/api/update")); } catch { /* offline */ }
}
refreshUpdateBanner();
bus.on("update.available", () => refreshUpdateBanner());
document.querySelector(".app").insertBefore(updateBanner, outlet);

// ------------------------------------------------------------------- toasts
bus.on("job.state", (p) => {
  const icon = { done: "✓", failed: "✗", cancelled: "⊘" }[p.state] || "▶";
  toast(`${icon} job ${p.job_id} → ${p.state}`,
    p.state === "failed" ? "error" : "info");
});
bus.on("model.promoted", (p) => toast(`★ promoted ${p.model_id}`, "ok"));

// Easy training: when the turnkey loop is enabled and its buffer crosses the
// threshold, tell the user training is due (once per transition), and surface
// a compact rail pill while it waits.
const easyPill = el("a", { href: "#/easy", class: "pill pill-warn hidden",
  title: "easy training: your profile buffer is ready — train now" },
  "⛏ train due");
statusbar.insertBefore(easyPill, hwGauge);
let easyWasReady = false;
async function refreshEasyPill() {
  try {
    const d = await api("/api/easy");
    // Multi-agent loops report per-agent views in `agents`; fall back to the
    // legacy top-level readiness (which mirrors the primary agent) so older
    // payloads keep working.
    const agents = (d.agents && d.agents.length)
      ? d.agents
      : [{ profile: d.config?.profile, readiness: d.readiness,
           active_run: d.active_run }];
    const due = agents.filter((a) => a.readiness?.needs_training);
    const busy = agents.some((a) => !!a.active_run);
    const ready = !!(d.config?.enabled && due.length) && !busy;
    easyPill.classList.toggle("hidden", !ready);
    if (ready && !easyWasReady) {
      const who = due.length === 1
        ? `profile “${due[0].profile}”`
        : `${due.length} profiles`;
      toast(`⛏ easy training ready for ${who} — open the Easy panel to run it`,
        "ok");
    }
    easyWasReady = ready;
  } catch { easyPill.classList.add("hidden"); }
}
refreshEasyPill();
setInterval(refreshEasyPill, 60000);
bus.on("easy_training.run.finished", () => { easyWasReady = false; refreshEasyPill(); });
bus.on("model.rolled_back", (p) =>
  toast(p.model_id ? `↩ rolled back to ${p.model_id}` : "↩ active model cleared", "info"));
bus.on("serve.state", (p) => {
  if (p.event === "stopped") toast("■ server stopped", "info");
});
bus.on("trace.recorded", () => {}); // quiet by design; Traces panel (5c) surfaces these
bus.on("overflow", (p) => {
  toast(`event backlog overflowed (${p.dropped ?? "?"} dropped) — refetching`, "error");
  refreshServePill(); refreshHw();
});

// ------------------------------------------------------------- console drawer
// Bottom-of-UI toggle: slides up a tail of the program console (dashboard.log
// + llama-server.log) polled every 2 s while open. Pure DOM; no innerHTML.
const consoleHost = el("div", { class: "console-drawer hidden" });
let consoleTimer = null;
let consoleAutoScroll = true;
let consoleSource = "all";

const consolePre = el("pre", { class: "log" });
const consoleBody = el("div", { class: "logbody" }, consolePre);
const srcSel = el("select", { class: "small", title: "log source" },
  el("option", { value: "all", text: "all logs" }),
  el("option", { value: "dashboard", text: "dashboard" }));
const asChk = el("input", { type: "checkbox" });
asChk.checked = true;
asChk.addEventListener("change", () => { consoleAutoScroll = asChk.checked; });

// The server-source options (llama-server, embed-server, one per spawned
// instance "server-<id>") follow whatever /api/logs reports, so a newly
// started instance appears without reloading the page.
let knownSources = ["dashboard"];
function syncSources(d) {
  const names = Object.keys(d).filter((n) => d[n] && d[n].exists);
  if (names.length === knownSources.length &&
      names.every((n, i) => n === knownSources[i])) return;
  knownSources = names;
  const cur = srcSel.value;
  srcSel.replaceChildren(el("option", { value: "all", text: "all logs" }),
    ...names.map((n) => el("option", { value: n, text: n })));
  srcSel.value = [...srcSel.options].some((o) => o.value === cur) ? cur : "all";
  consoleSource = srcSel.value;
}

async function refreshConsole() {
  let d;
  try { d = await api("/api/logs?tail=600"); }
  catch { return; } // server gone (e.g. after shutdown) — keep last render
  syncSources(d);
  const wanted = consoleSource === "all" ? knownSources : [consoleSource];
  const lines = [];
  for (const name of wanted) {
    const src = d[name];
    if (!src || !src.exists) continue;
    lines.push(`───── ${name} ─────`);
    for (const l of src.lines) lines.push(l);
  }
  consolePre.replaceChildren(document.createTextNode(
    lines.length ? lines.join("\n") + "\n" : "(no log lines yet)"));
  if (consoleAutoScroll) consoleBody.scrollTop = consoleBody.scrollHeight;
}
srcSel.addEventListener("change", () => { consoleSource = srcSel.value; refreshConsole(); });

const consoleToggle = el("button", {
  class: "small", text: "▴ console", title: "Toggle the program console log",
  "aria-expanded": "false",
});
consoleToggle.addEventListener("click", () => {
  const open = consoleHost.classList.toggle("hidden");
  consoleToggle.textContent = open ? "▴ console" : "▾ console";
  consoleToggle.setAttribute("aria-expanded", String(!open));
  clearInterval(consoleTimer);
  if (!open) {
    refreshConsole();
    consoleTimer = setInterval(refreshConsole, 2000);
  }
});

const foot = document.querySelector(".foot");
if (foot) {
  const appDiv = document.querySelector(".app");
  // drawer sits above the footer, full width
  appDiv.insertBefore(consoleHost, foot);
  consoleHost.replaceChildren(el("div", { class: "loghead" },
    el("h3", { text: "console" }), srcSel,
    el("label", { class: "chk" }, asChk, " autoscroll"),
    el("button", { class: "small", text: "refresh", onclick: () => refreshConsole() })),
    consoleBody);
  const spacer = el("span", { text: " · " });
  foot.append(spacer, consoleToggle);
}

// footer meta
(async () => {
  try {
    const s = await api("/api/status");
    const f = document.getElementById("footmeta");
    if (f) f.textContent = `v${s.version} · ${s.home} · loopback-only · no telemetry`;
  } catch { /* footer stays default */ }
})();

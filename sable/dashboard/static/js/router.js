// router.js — tiny hash router. Panels register under a route prefix;
// routes look like #/chat, #/models/v2, #/jobs/jobid. Registration API:
//   register("chat", {render(mount, params), title, unsub?})
// Returning an object with destroy() from render() lets panels clean up
// intervals/listeners when navigated away from.

const routes = new Map(); // name -> panel
let current = null;       // {name, handle}
let outlet = null;
let onNavigate = null;    // callback(name) for nav highlighting

/**
 * Register a panel. panel: {render(mount, params) -> maybe {destroy()}, title}
 */
export function register(name, panel) { routes.set(name, panel); }

export function start(outletEl, opts = {}) {
  outlet = outletEl;
  onNavigate = opts.onNavigate || null;
  window.addEventListener("hashchange", () => resolve());
  resolve();
}

/** Programmatic navigation. */
export function go(path) {
  if (location.hash === "#" + path) resolve();
  else location.hash = path;
}

export function currentRoute() { return current ? current.name : null; }

function resolve() {
  const raw = (location.hash || "#/").slice(1); // "/chat/v2"
  const parts = raw.split("/").filter(Boolean);
  const name = parts[0] || "chat";
  const params = parts.slice(1);
  const panel = routes.get(name) || routes.get("_fallback");
  if (current && current.name !== name && current.handle?.destroy) {
    try { current.handle.destroy(); } catch (e) { console.error(e); }
  }
  current = { name, handle: null };
  outlet.replaceChildren();
  if (panel) {
    try {
      current.handle = panel.render(outlet, params) || null;
    } catch (e) {
      console.error("panel render failed:", name, e);
      const pre = document.createElement("pre");
      pre.className = "err";
      pre.textContent = "panel error: " + (e?.message || e);
      outlet.appendChild(pre);
    }
  }
  if (onNavigate) onNavigate(name);
}

// logpane.js — reusable job-log pane (factored out of panels/jobs.js in 5d).
// Renders one job's state + full log with live tailing: SSE `job.log`
// snapshots on transitions plus a 2 s poll while the job is live. Pure DOM
// via el()/textContent; no innerHTML.

import { api } from "./api.js";
import { bus } from "./bus.js";
import { el, badge } from "./ui.js";

const LIVE_STATES = new Set(["queued", "running"]);

/**
 * Build a log pane for one job. Returns { node, destroy() }.
 * opts.onClose: optional callback when the user closes the pane.
 */
export function createLogPane(jobId, opts = {}) {
  const pane = el("div", { class: "logpane" });
  let autoScroll = true;
  let destroyed = false;
  let liveTimer = null;

  const unsubLog = bus.on("job.log", (p) => {
    if (p.job_id !== jobId || !Array.isArray(p.lines)) return;
    const pre = pane.querySelector("pre.log");
    if (!pre) return;
    for (const l of p.lines) pre.appendChild(document.createTextNode(l + "\n"));
    if (autoScroll) {
      const body = pane.querySelector(".logbody");
      if (body) body.scrollTop = body.scrollHeight;
    }
  });

  function scheduleLiveAppend() {
    clearInterval(liveTimer);
    liveTimer = setInterval(async () => {
      if (destroyed || pane.classList.contains("hidden")) {
        clearInterval(liveTimer);
        return;
      }
      try {
        const d = await api(`/api/jobs/${encodeURIComponent(jobId)}?limit=5000`);
        const pre = pane.querySelector("pre.log");
        if (!pre) return;
        const have = pre.textContent.split("\n").length - 1;
        const all = d.log.lines || [];
        if (all.length > have) {
          pre.replaceChildren();
          for (const l of all) pre.appendChild(document.createTextNode(l + "\n"));
          if (autoScroll) {
            const body = pane.querySelector(".logbody");
            if (body) body.scrollTop = body.scrollHeight;
          }
        }
        if (!LIVE_STATES.has(d.job.state)) clearInterval(liveTimer);
      } catch { /* transient */ }
    }, 2000);
  }

  async function load() {
    pane.replaceChildren(el("p", { class: "hint", text: "loading log…" }));
    let d;
    try {
      d = await api(`/api/jobs/${encodeURIComponent(jobId)}?limit=2000`);
    } catch (e) {
      pane.replaceChildren(el("p", { class: "err", text: e.message }));
      return;
    }
    render(d.job, d.log.lines || []);
  }

  function render(job, lines) {
    const pre = el("pre", { class: "log" });
    for (const l of (lines.length ? lines : ["(no log lines yet)"]))
      pre.appendChild(document.createTextNode(l + "\n"));
    const closeBtn = el("button", { class: "small", text: "✕", onclick: () => {
      if (opts.onClose) opts.onClose();
      else pane.remove();
    } });
    const autoscrollChk = el("input", { type: "checkbox" });
    autoscrollChk.checked = autoScroll;
    autoscrollChk.addEventListener("change", () => {
      autoScroll = autoscrollChk.checked;
      if (autoScroll) {
        const body = pane.querySelector(".logbody");
        if (body) body.scrollTop = body.scrollHeight;
      }
    });
    const retryBtn = el("button", { class: "small", text: "reload",
      onclick: () => load() });
    const logBody = el("div", { class: "logbody" }, pre);
    pane.replaceChildren(
      el("div", { class: "loghead" },
        el("h3", { text: job.job_id }),
        badge(job.state),
        el("span", { class: "hint mono" },
          `${job.kind} · ${job.dataset_id || "–"}`),
        el("label", { class: "chk" }, autoscrollChk, " autoscroll"),
        retryBtn, closeBtn),
      logBody);
    if (LIVE_STATES.has(job.state)) scheduleLiveAppend();
  }

  load();

  return {
    node: pane,
    destroy() {
      destroyed = true;
      clearInterval(liveTimer);
      unsubLog();
    },
  };
}

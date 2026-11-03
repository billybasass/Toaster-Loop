// panels/jobs.js — training job queue + live log pane.
// Queue table refetches on job.state bus events; the log pane pages via
// GET /api/jobs/<id> and appends live from job.log events (filtered by id).

import { api } from "../api.js";
import { bus } from "../bus.js";
import { generation, coalesced } from "../state.js";
import { el, toast, badge, table, empty, fmtTime, confirmDialog } from "../ui.js";

const LIVE_STATES = new Set(["queued", "running"]);

export default {
  title: "Jobs",
  render(mount, params = []) {
    const tableHost = el("div");
    const logPane = el("div", { class: "logpane hidden" });
    let jobs = [];
    let selectedId = null;
    let autoScroll = true;
    let redrawSpark = null; // set by renderLog; called by the live tailer (#4)
    const unsubs = [];

    // ------------------------------------------------------------- log pane
    async function openJob(jobId) {
      selectedId = jobId;
      logPane.classList.remove("hidden");
      logPane.replaceChildren(el("p", { class: "hint", text: "loading log…" }));
      let d;
      try { d = await api(`/api/jobs/${encodeURIComponent(jobId)}?limit=2000`); }
      catch (e) {
        logPane.replaceChildren(el("p", { class: "err", text: e.message }));
        return;
      }
      renderLog(d.job, d.log.lines || []);
    }

    // Backlog #4 sparkline: one polyline through {"event":"progress","loss":x}
    // lines from the job log (eval events drawn as a second line). Pure view
    // code over the existing JSON protocol — no backend changes needed.
    function lossSparkline(lines) {
      const train = [], evs = [];
      for (const l of lines) {
        if (l.indexOf('"progress"') === -1 && l.indexOf('"eval"') === -1) continue;
        let r; try { r = JSON.parse(l); } catch { continue; }
        if (r.event === "progress" && typeof r.loss === "number")
          train.push([r.step || train.length, r.loss]);
        else if (r.event === "eval" && typeof r.eval_loss === "number")
          evs.push([r.step || evs.length, r.eval_loss]);
      }
      if (train.length < 2 && evs.length < 2) return null;
      const all = train.concat(evs);
      const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
      const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
      const y0 = Math.min(...ys), y1 = Math.max(...ys);
      const W = 240, H = 48, span = (y1 - y0) || 1;
      const pts = (arr) => arr.map(([x, y]) =>
        `${((x - x0) / ((x1 - x0) || 1)) * W},${H - ((y - y0) / span) * H}`).join(" ");
      const svgNS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(svgNS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      svg.setAttribute("class", "loss-spark");
      // Colors come from design tokens (pass 3 #26) — no literals in JS.
      svg.style.cssText = "width:240px;height:48px;background:var(--inset-bg);" +
        "border:1px solid var(--line)";
      const mk = (arr, token) => {
        if (arr.length < 2) return;
        const p = document.createElementNS(svgNS, "polyline");
        p.setAttribute("points", pts(arr));
        p.setAttribute("fill", "none");
        p.style.stroke = `var(${token})`;
        p.setAttribute("stroke-width", "1.5");
        svg.appendChild(p);
      };
      mk(train, "--spark-train");
      mk(evs, "--spark-eval");
      return el("span", { class: "hint", title: "train loss (green) / eval loss (blue)" }, svg);
    }

    function renderLog(job, lines) {
      const pre = el("pre", { class: "log" });
      const allLines = [...lines];
      const sparkHost = el("span");
      const drawSpark = () => {
        const s = lossSparkline(allLines);
        sparkHost.replaceChildren(...(s ? [s] : []));
      };
      // Live tailer replaces the line set and redraws the polyline (#4).
      redrawSpark = (fresh) => { allLines.length = 0; allLines.push(...fresh); drawSpark(); };
      const appendText = (arr) => {
        // allLines already seeded with the initial page in renderLog().
        for (const l of arr) pre.appendChild(document.createTextNode(l + "\n"));
        if (autoScroll) logBody.scrollTop = logBody.scrollHeight;
      };
      drawSpark();
      const closeBtn = el("button", { class: "small", text: "✕",
        onclick: () => { selectedId = null; logPane.classList.add("hidden"); } });
      const autoscrollChk = el("input", { type: "checkbox" });
      autoscrollChk.checked = autoScroll;
      autoscrollChk.addEventListener("change", () => {
        autoScroll = autoscrollChk.checked;
        if (autoScroll) logBody.scrollTop = logBody.scrollHeight;
      });
      const retryBtn = el("button", { class: "small", text: "reload" });
      retryBtn.addEventListener("click", () => openJob(job.job_id));

      appendText(lines.length ? lines : ["(no log lines yet)"]);

      const logBody = el("div", { class: "logbody" }, pre);
      logPane.replaceChildren(
        el("div", { class: "loghead" },
          el("h3", { text: job.job_id }),
          badge(job.state),
          el("span", { class: "hint mono" }, `${job.kind} · ${job.dataset_id || "–"}`),
          sparkHost,
          el("label", { class: "chk" }, autoscrollChk, " autoscroll"),
          retryBtn, closeBtn),
        logBody);

      if (LIVE_STATES.has(job.state)) scheduleLiveAppend(job.job_id);
    }

    // Live tail: while a job is live, poll its log window every 2 s in
    // addition to SSE job.log snapshots (which only fire on transitions).
    let liveTimer = null;
    function scheduleLiveAppend(jobId) {
      clearInterval(liveTimer);
      liveTimer = setInterval(async () => {
        if (selectedId !== jobId || logPane.classList.contains("hidden")) {
          clearInterval(liveTimer); return;
        }
        try {
          const d = await api(`/api/jobs/${encodeURIComponent(jobId)}?limit=5000`);
          const pre = logPane.querySelector("pre.log");
          if (!pre) return;
          const have = pre.textContent.split("\n").length - 1;
          const all = d.log.lines || [];
          if (all.length > have) {
            pre.replaceChildren();
            for (const l of all) pre.appendChild(document.createTextNode(l + "\n"));
            if (redrawSpark) redrawSpark(all); // keep the sparkline current (#4)
            if (autoScroll) {
              const body = logPane.querySelector(".logbody");
              body.scrollTop = body.scrollHeight;
            }
          }
          if (!LIVE_STATES.has(d.job.state)) clearInterval(liveTimer);
        } catch { /* transient */ }
      }, 2000);
    }

    // --------------------------------------------------------------- table
    // State authority (#24): the fetch carries a generation token, so a slow
    // response can never paint over a newer one; job.state events coalesce
    // EXCEPT terminal transitions (done/error/cancelled), which refresh now.
    const guard = generation("jobs");

    async function load() {
      const done = guard.begin();
      const d = await api("/api/jobs?tail=0");
      if (!done()) return; // superseded by a newer refresh — drop silently
      jobs = [...(d.jobs || [])].reverse(); // newest first
      renderTable();
    }

    async function cancelJob(j) {
      const ok = await confirmDialog({
        title: `Cancel ${j.job_id}?`,
        body: j.state === "running"
          ? "The running worker process will be terminated."
          : "The queued job will be removed from the queue.",
        confirmLabel: "Cancel job", danger: true,
      });
      if (!ok) return;
      try {
        await api("/api/jobs/cancel", { method: "POST", body: { job_id: j.job_id } });
        toast(`cancelled ${j.job_id}`, "ok");
        await load();
      } catch (e) { toast(`${e.code}: ${e.message}`, "error"); }
    }

    function renderTable() {
      // id -> job map for the deps column's waiting check (#43).
      const byId = new Map(jobs.map((j) => [j.job_id, j]));
      tableHost.replaceChildren(table([
        { label: "job", render: (j) => el("span", { class: "mono" }, j.job_id) },
        { label: "kind", key: "kind" },
        { label: "state", render: (j) => badge(j.state) },
        // Pass 4 #43: dependency edges. A queued job with unfinished deps is
        // marked waiting; the dep ids stay visible after they resolve so the
        // chain reads in the table (indented-tree idea, flattened to a cell).
        { label: "deps", render: (j) => {
            if (!Array.isArray(j.depends_on) || !j.depends_on.length) return "–";
            const waiting = j.state === "queued" && j.depends_on.some(
              (d) => { const p = byId.get(d); return !p || p.state !== "done"; });
            const s = el("span", { class: "mono small" },
              j.depends_on.join(", "));
            if (waiting) s.appendChild(el("span", { class: "hint", text: " ⏳" }));
            s.title = waiting ? "waiting for dependencies to finish"
                              : "dependency chain";
            return s;
          } },
        { label: "dataset", render: (j) => el("span", { class: "mono" }, j.dataset_id || "–") },
        { label: "model", render: (j) => el("span", { class: "mono" }, j.model_id || "–") },
        { label: "queued", render: (j) => fmtTime(j.queued_at, false) },
        { label: "finished", render: (j) => j.finished_at ? fmtTime(j.finished_at, false) : "–" },
        { label: "", render: (j) => {
            const cell = el("span");
            if (LIVE_STATES.has(j.state)) {
              const b = el("button", { class: "small danger", text: "cancel" });
              b.addEventListener("click", (ev) => { ev.stopPropagation(); cancelJob(j); });
              cell.appendChild(b);
            }
            return cell;
          } },
      ], jobs, {
        empty: "No training jobs yet. Submit one from the Train panel (5d).",
        rowClass: (j) => j.job_id === selectedId ? "active-row" : "",
        onRowClick: (j) => openJob(j.job_id),
      }));
    }

    mount.replaceChildren(
      el("h2", { text: "Jobs" }),
      el("div", { class: "toolbar" },
        el("button", { class: "small", text: "refresh", onclick: () =>
          load().catch((e) => toast(e.message, "error")) })),
      tableHost, logPane);

    load().then(() => {
      // 5d: deep link #/jobs/<id> (from Train/Iterate panels) opens the log.
      if (params.length && params[0]) openJob(decodeURIComponent(params[0]));
    }).catch((e) => tableHost.replaceChildren(empty("failed to load: " + e.message)));

    const refreshJobs = coalesced(() => load().catch(() => {}), {
      ms: 400,
      isTerminal: (p) => p && ["done", "error", "cancelled"].includes(p.state),
    });
    unsubs.push(bus.on("job.state", (p) => {
      refreshJobs(p);
      if (p.job_id === selectedId) openJob(p.job_id).catch(() => {});
    }));
    unsubs.push(bus.on("job.log", (p) => {
      // Terminal-transition snapshot: append if we're watching this job.
      if (p.job_id !== selectedId || !Array.isArray(p.lines)) return;
      const pre = logPane.querySelector("pre.log");
      if (!pre) return;
      for (const l of p.lines) pre.appendChild(document.createTextNode(l + "\n"));
      if (autoScroll) {
        const body = logPane.querySelector(".logbody");
        if (body) body.scrollTop = body.scrollHeight;
      }
    }));

    return { destroy() { clearInterval(liveTimer); for (const u of unsubs) u(); } };
  },
};

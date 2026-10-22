// panels/hub.js — Model Hub: discover page with a left rail of
// models (latest feed / search results with GGUF + capability + sort filters),
// right detail pane (capability chips, base model, quant picker with sizes,
// download with live progress, stat chips, README). "On Device" tab manages
// what's installed. Backed by /api/hub/* (core/modelhub.py).

import { api, getToken } from "../api.js";
import { bus } from "../bus.js";
import { el, toast, badge, empty, confirmDialog, fmtNum, renderMarkdown,
  copyButton } from "../ui.js";

const fmtBytes = (n) => {
  if (!n && n !== 0) return "–";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${fmtNum(v, v >= 100 || i === 0 ? 0 : 1)} ${u[i]}`;
};

const ago = (iso) => {
  if (!iso) return "";
  const days = Math.floor((Date.now() - new Date(iso).getTime()) / 864e5);
  if (days <= 0) return "today";
  if (days === 1) return "1d ago";
  if (days < 30) return `${days}d ago`;
  if (days < 365) return `${Math.floor(days / 30)}mo ago`;
  return `${Math.floor(days / 365)}y ago`;
};

const fmtCount = (n) => n >= 1e6 ? `${fmtNum(n / 1e6, 1)}M`
  : n >= 1e3 ? `${fmtNum(n / 1e3, 1)}K` : String(n || 0);

// Deterministic avatar tint from the author name (studio shows org logos; we
// stay offline + CSP-safe with initials chips instead).
// Tint palette lives in the design-token block (pass 3 #26); JS references the
// token names so a theme swap touches CSS only.
const AVATAR_TINTS = ["--tint-1", "--tint-2", "--tint-3", "--tint-4",
  "--tint-5", "--tint-6", "--tint-7", "--tint-8"];
function avatar(name) {
  let h = 0;
  for (const c of name || "?") h = (h * 31 + c.charCodeAt(0)) >>> 0;
  const bg = `var(${AVATAR_TINTS[h % AVATAR_TINTS.length]})`;
  const initials = (name || "?").replace(/[^A-Za-z0-9 ]/g, " ")
    .trim().split(/\s+/).slice(0, 2).map((w) => w[0]).join("").toUpperCase();
  return el("span", { class: "hub-avatar", style: `background:${bg};`,
    text: initials || "?" });
}

const CAPS = [
  ["", "All capabilities"],
  ["conversational", "Conversational"],
  ["vision", "Vision"],
  ["embeddings", "Embeddings"],
  ["audio", "Audio"],
];
const SORTS = [
  ["downloads", "Most downloads"],
  ["likes", "Most likes"],
  ["newest", "Newest"],
];

export default {
  title: "Model Hub",
  render(mount) {
    // ---------------------------------------------------------------- state
    let tab = "discover";            // discover | device
    let filters = { gguf: true, capability: "", sort: "downloads" };
    let list = [];                   // current left-rail rows
    let installedByRepo = new Map(); // repo_id -> installed entry
    let lastDevice = { installed: [] };
    let selectedRepo = null;
    let pollTimer = null;

    // ------------------------------------------------------------- elements
    const tabDiscover = el("button", { class: "hub-tab", text: "Discover" });
    const tabDevice = el("button", { class: "hub-tab", text: "On Device" });
    const query = el("input", { type: "search", class: "hub-search",
      placeholder: "Search all models" });
    const ggufSel = selectEl([["1", "GGUF"], ["", "All formats"]], "1");
    const capSel = selectEl(CAPS, "");
    const sortSel = selectEl(SORTS, "downloads");

    const listHost = el("div", { class: "hub-list" });
    const detailHost = el("div", { class: "hub-detail" });
    const downloadsHost = el("div", { class: "hub-downloads" });

    function selectEl(opts, value) {
      const s = el("select", { class: "hub-select" },
        ...opts.map(([v, label]) => el("option", {
          value: v, text: label, selected: v === value })));
      return s;
    }

    // ---------------------------------------------------------- data loads
    async function loadFeed() {
      setList(empty("loading the hub…"));
      try {
        const r = await api("/api/hub/feed?limit=25");
        list = r.models || [];
        renderList("Latest GGUF Models");
      } catch (e) { setList(el("p", { class: "err", text: e.message })); }
    }

    async function doSearch() {
      setList(empty("searching…"));
      try {
        const p = new URLSearchParams({
          q: query.value.trim(), limit: "40",
          gguf: ggufSel.value, capability: capSel.value, sort: sortSel.value,
        });
        const r = await api("/api/hub/search?" + p.toString());
        list = r.results || [];
        renderList(query.value.trim() ? `Results for “${query.value.trim()}”`
          : "Popular on the Hub");
      } catch (e) { setList(el("p", { class: "err", text: e.message })); }
    }

    function setList(...kids) { listHost.replaceChildren(...kids); }

    function renderList(titleText) {
      const rows = list.map((m) => {
        const installed = installedByRepo.has(m.repo_id);
        const row = el("div", { class: "hub-row" +
          (m.repo_id === selectedRepo ? " sel" : ""),
          onclick: () => openDetail(m.repo_id) },
          avatar(m.author),
          el("div", { class: "hub-row-main" },
            el("div", { class: "hub-row-name", text: m.name }),
            el("div", { class: "hub-row-sub" },
              el("span", { text: m.author }),
              installed ? badge("installed", "done") : null),
          ),
          el("div", { class: "hub-row-stats" },
            el("span", { title: "likes", text: `♥ ${fmtCount(m.likes)}` }),
            el("span", { title: "downloads", text: `⬇ ${fmtCount(m.downloads)}` }),
            el("span", { class: "hint", text: ago(m.last_modified) })),
        );
        return row;
      });
      setList(
        el("div", { class: "hub-list-head" },
          el("h3", { text: titleText }),
          el("button", { class: "small ghost", text: "⟳", title: "reload",
            onclick: (ev) => { ev.stopPropagation(); tab === "discover"
              && (query.value.trim() ? doSearch() : loadFeed()); } })),
        ...(rows.length ? rows : [empty("no results.")]));
    }

    // -------------------------------------------------------------- detail
    async function openDetail(repoId) {
      selectedRepo = repoId;
      refreshList();
      detailHost.replaceChildren(el("p", { class: "hint", text: "loading…" }));
      let d;
      try { d = await api("/api/hub/detail?repo=" + encodeURIComponent(repoId)); }
      catch (e) {
        detailHost.replaceChildren(el("p", { class: "err", text: e.message }));
        return;
      }
      renderDetail(d);
    }

    // Re-render whichever list the active tab owns (keeps selection state).
    function refreshList() {
      if (tab === "device") renderDevice(lastDevice);
      else renderList(currentTitle());
    }

    function currentTitle() {
      return query.value.trim() ? `Results for “${query.value.trim()}”`
        : "Latest GGUF Models";
    }

    // Card images load through our same-origin proxy (CSP default-src 'self'):
    // repo-relative sources become ?repo=&path=, allow-listed absolute https
    // URLs become ?url=. Anything else is dropped.
    function proxiedImage(repoId, src) {
      if (!src) return null;
      const s = String(src).trim();
      if (/^(data:|blob:)/i.test(s)) return null;
      if (/^[a-z][a-z0-9+.-]*:\/\//i.test(s) || s.startsWith("//")) {
        const ok = /(huggingface\.co|hf\.co|raw\.githubusercontent\.com)$/i
          .test(new URL(s, location.origin).hostname);
        return ok ? withToken("/api/hub/image?url=" + encodeURIComponent(
          s.startsWith("//") ? location.protocol + s : s)) : null;
      }
      if (s.startsWith("/")) {
        return withToken("/api/hub/image?repo=" + encodeURIComponent(repoId)
          + "&path=" + encodeURIComponent(s.slice(1)));
      }
      return withToken("/api/hub/image?repo=" + encodeURIComponent(repoId)
        + "&path=" + encodeURIComponent(s));
    }

    // <img> cannot set headers; when token auth is on, GET endpoints accept
    // ?token= (see dashboard/server.py _authorized).
    function withToken(u) {
      const t = getToken();
      return t ? u + "&token=" + encodeURIComponent(t) : u;
    }

    function renderDetail(d) {
      const banner = d.image
        ? el("div", { class: "hub-banner" }, el("img", {
            src: proxiedImage(d.repo_id, d.image) || "", alt: "",
            loading: "lazy", onerror: (ev) => ev.target.closest(".hub-banner")
              ?.remove() }))
        : null;
      const varSel = el("select", { class: "hub-select hub-quant" },
        ...(d.variants || []).map((v) => el("option", {
          value: v.variant,
          text: `${v.variant} · ${fmtBytes(v.size_bytes)}` +
            (v.shard_count ? ` (${v.shard_count} shards)` : ""),
          selected: v.variant === d.default_variant })));
      const dlBtn = el("button", { class: "primary hub-download" },
        "⬇  Download");
      dlBtn.addEventListener("click", () => startDownload(d.repo_id,
        varSel.value));

      const chips = [];
      if (d.pipeline_tag) chips.push(badge(prettyCap(d.pipeline_tag), "dim"));
      if ((d.tags || []).includes("vision")) chips.push(badge("Vision", "dim"));
      if ((d.tags || []).includes("text-generation") || d.pipeline_tag ===
          "text-generation") chips.push(badge("Conversational", "dim"));

      const stat = (label, value) => el("span", { class: "hub-stat",
        title: label }, value);
      const stats = el("div", { class: "hub-stats" },
        stat("updated", ago(d.last_modified)),
        d.context_length ? stat("context",
          `${fmtNum(d.context_length / 1024, d.context_length % 1024 ? 1 : 0)}K`) : null,
        d.architecture ? stat("architecture", d.architecture) : null,
        stat("downloads", fmtCount(d.downloads)),
        stat("likes", String(d.likes || 0)),
        d.license ? stat("license", d.license) : null,
        d.gated ? badge("gated", "failed") : null);

      const baseRow = (d.base_models || []).length
        ? el("div", { class: "hub-base" },
            el("span", { class: "hint", text: "Base:" }),
            ...d.base_models.map((b) => el("code", { class: "mono", text: b })))
        : null;

      detailHost.replaceChildren(
        banner,
        el("div", { class: "hub-detail-head" },
          avatar(d.author),
          el("div", {},
            el("h3", { class: "hub-title", text: d.name }),
            el("div", { class: "hint" }, d.author, " ",
              copyButton(d.repo_id)))),
        el("div", { class: "hub-chips-row" }, ...chips),
        baseRow,
        el("div", { class: "hub-dlrow" }, varSel, dlBtn),
        stats,
        d.gated ? el("p", { class: "hint", text:
          "Gated repo — set HF_TOKEN in the environment before downloading." })
          : null,
        readmeBlock(d.repo_id, d.readme));
    }

    function prettyCap(p) {
      return (p || "").split("-").map((w) => w[0]?.toUpperCase() + w.slice(1))
        .join(" ");
    }

    function readmeBlock(repoId, md) {
      const host = el("div", { class: "hub-readme" });
      if (!md) return host;
      // Strip YAML frontmatter (model cards are all front-matter + prose).
      const body = md.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, "");
      renderMarkdown(host, body.slice(0, 24000), {
        imageSrc: (src) => proxiedImage(repoId, src),
      });
      return host;
    }

    // ------------------------------------------------------------- download
    async function startDownload(repoId, variant) {
      try {
        await api("/api/hub/download", { method: "POST",
          body: { repo: repoId, variant } });
        toast(`downloading ${repoId} (${variant})`, "ok");
        ensurePolling();
      } catch (e) { toast(`${e.code}: ${e.message}`, "error"); }
    }

    function ensurePolling() {
      if (!pollTimer) pollTimer = setInterval(loadDownloads, 1200);
    }

    async function loadDownloads() {
      let jobs = [];
      try { jobs = (await api("/api/hub/downloads")).downloads || []; }
      catch { return; }
      renderDownloads(jobs);
      const inflight = jobs.some((j) =>
        j.state === "queued" || j.state === "running");
      if (!inflight && pollTimer) {
        clearInterval(pollTimer); pollTimer = null;
        loadInstalled();
      }
    }

    function renderDownloads(jobs) {
      const active = jobs.filter((j) => j.state !== "cancelled")
        .slice(-4).reverse();
      if (!active.length) { downloadsHost.replaceChildren(); return; }
      const rows = active.map((j) => {
        const pct = j.total_bytes
          ? Math.min(100, 100 * j.done_bytes / j.total_bytes) : 0;
        const bar = el("div", { class: "hub-progress" },
          el("div", { class: "hub-progress-fill",
            style: `width:${pct.toFixed(1)}%` }));
        const busy = j.state === "queued" || j.state === "running";
        return el("div", { class: "hub-dl-row" },
          badge(j.state, j.state === "done" ? "done"
            : j.state === "error" ? "failed" : ""),
          el("div", { style: "flex:1; min-width:0;" },
            el("div", { class: "mono hub-dl-name",
              text: `${j.repo_id} · ${j.variant || "default"}` }),
            bar,
            el("div", { class: "hint", style: "font-size:.8em;", text:
              j.error ? j.error :
              `${fmtBytes(j.done_bytes)} / ${fmtBytes(j.total_bytes)}` })),
          busy ? el("button", { class: "small danger", text: "✕",
            title: "cancel", onclick: async () => {
              try { await api("/api/hub/cancel", { method: "POST",
                body: { job_id: j.job_id } }); loadDownloads(); }
              catch (e) { toast(e.message, "error"); }
            } }) : null);
      });
      downloadsHost.replaceChildren(...rows);
    }

    // -------------------------------------------------------------- device
    async function loadInstalled() {
      try {
        const d = await api("/api/hub/installed");
        installedByRepo = new Map(
          (d.installed || []).map((m) => [m.repo_id, m]));
        lastDevice = d;
        if (tab === "device") renderDevice(d);
        else refreshList();  // installed badges on the discover list
      } catch (e) {
        setList(el("p", { class: "err", text: e.message }));
      }
    }

    function renderDevice(d) {
      const items = d.installed || [];
      const rows = items.length ? items.map((m) => el("div", {
        class: "hub-row" + (m.repo_id === selectedRepo ? " sel" : ""),
        onclick: () => openDetail(m.repo_id) },
        avatar(m.repo_id.split("/")[0]),
        el("div", { class: "hub-row-main" },
          el("div", { class: "hub-row-name", text: m.repo_id }),
          el("div", { class: "hub-row-sub hint", text:
            (m.quants && m.quants.length ? m.quants.join(", ")
              : "full checkpoint") })),
        el("div", { class: "hub-row-stats" },
          el("span", { class: "mono", text: fmtBytes(m.size_bytes) }),
          el("button", { class: "small danger", text: "delete",
            onclick: async (ev) => {
              ev.stopPropagation();
              const ok = await confirmDialog({
                title: `Delete ${m.repo_id}?`,
                body: "Removes the downloaded files from disk. Fine-tunes " +
                  "trained from it are unaffected.",
                confirmLabel: "Delete", danger: true });
              if (!ok) return;
              try {
                const r = await api("/api/hub/delete", { method: "POST",
                  body: { repo: m.repo_id } });
                toast(`freed ${fmtBytes(r.freed_bytes)}`, "ok");
                loadInstalled();
              } catch (e) { toast(`${e.code}: ${e.message}`, "error"); }
            } })),
      )) : [empty("nothing downloaded yet — pick something in Discover.")];
      setList(
        el("div", { class: "hub-list-head" },
          el("h3", { text: `On Device (${items.length})` }),
          el("span", { class: "hint", style: "font-size:.75em;",
            text: d.models_dir || "" })),
        ...rows);
    }

    // -------------------------------------------------------------- tabs
    function setTab(next) {
      tab = next;
      tabDiscover.classList.toggle("active", tab === "discover");
      tabDevice.classList.toggle("active", tab === "device");
      if (tab === "discover") {
        query.value.trim() ? doSearch() : loadFeed();
      } else {
        loadInstalled();
      }
    }

    // -------------------------------------------------------------- wiring
    tabDiscover.addEventListener("click", () => setTab("discover"));
    tabDevice.addEventListener("click", () => setTab("device"));
    query.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") doSearch();
      if (ev.key === "Escape") { query.value = ""; loadFeed(); }
    });
    for (const s of [ggufSel, capSel, sortSel])
      s.addEventListener("change", () => { if (tab === "discover") doSearch(); });

    mount.replaceChildren(
      el("div", { class: "hub-head" },
        el("h2", { text: "Model Hub" }),
        el("p", { class: "hint", text: "Discover and download models locally." })),
      el("div", { class: "hub-toolbar" },
        el("div", { class: "hub-tabs" }, tabDiscover, tabDevice),
        query, ggufSel, capSel, sortSel),
      el("div", { class: "hub-cols" }, listHost, detailHost),
      downloadsHost);

    setTab("discover");
    loadInstalled();
    loadDownloads();

    const unsubs = [bus.on("model.deleted", () => loadInstalled())];
    return { destroy() {
      for (const u of unsubs) u();
      if (pollTimer) clearInterval(pollTimer);
    } };
  },
};

// panels/hub.js — Resource Hub: discover page with a left rail of resources
// (latest feed / search results) across three kinds — Models, Datasets and
// MCP servers — plus capability/format/sort filters, right detail pane (card
// chips, base model, quant picker with sizes, download with live progress,
// stat chips, README). "On Device" tab manages what's installed. Backed by
// /api/hub/* (core/modelhub.py).

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

// Discover resource kinds. The value is the API ?kind=; models covers every
// serving backend (GGUF for llama.cpp, safetensors for vLLM, sd.cpp weights).
const KINDS = [
  ["models", "Models"],
  ["datasets", "Datasets"],
  ["spaces", "MCP Servers"],
];
const KIND_LABEL = Object.fromEntries(KINDS);

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

// README cards are markdown mixed with raw HTML blocks (divs, img tags in
// anchors, lists). The mini-markdown renderer escapes everything, so without
// a pass here the pane shows walls of tag soup and the images inside <img>
// tags never reach the proxy. Flatten HTML to markdown/text first: comments
// dropped, <img>/<a>/<li>/<hN>/<br> converted, remaining tags stripped.
function cleanReadme(md) {
  let body = String(md || "").replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, "");
  body = body.replace(/<!--[\s\S]*?(-->|$)/g, "");
  body = body.replace(/<br\s*\/?>/gi, "\n");
  // <img ...src="..."> (also nested inside <a>) -> ![alt](src)
  body = body.replace(
    /<img[^>]*?>/gi,
    (tag) => {
      const src = /\bsrc=["']([^"']+)["']/i.exec(tag)?.[1] || "";
      const alt = /\balt=["']([^"']*)["']/i.exec(tag)?.[1] || "";
      return src ? `\n![${alt}](${src})\n` : "";
    });
  // <a href="url">inner</a> -> [inner](url) (inner may hold the img above)
  body = body.replace(
    /<a[^>]*?\bhref=["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/gi,
    (_m, href, inner) => {
      const img = /!\[[^\]]*\]\(([^)\s]+)\)/.exec(inner);
      return img ? img[0] : `[${inner.replace(/<[^>]*>/g, "").trim()}](${href})`;
    });
  body = body.replace(
    /<h([1-6])[^>]*>([\s\S]*?)<\/\1>/gi,
    (_m, lvl, txt) => "#".repeat(Number(lvl)) + " " + txt.replace(/<[^>]*>/g, ""));
  body = body.replace(/<li[^>]*>([\s\S]*?)<\/li>/gi, "- $1");
  // Drop containers/tables wholesale of their tags; text survives.
  body = body.replace(/<\/?[a-zA-Z][a-zA-Z0-9-]*(\s[^>]*)?\/?>/g, "");
  // Collapse blank runs the unwrapping leaves behind.
  body = body.replace(/\n{3,}/g, "\n\n");
  return body;
}

export default {
  title: "Resource Hub",
  render(mount) {
    // ---------------------------------------------------------------- state
    let tab = "discover";            // discover | device
    let kind = "models";             // models | datasets | spaces
    let list = [];                   // current left-rail rows
    let installedByRepo = new Map(); // `${kind}:${repo_id}` -> installed entry
    let lastDevice = { installed: [] };
    let selectedRepo = null;
    let pollTimer = null;

    // ------------------------------------------------------------- elements
    const tabDiscover = el("button", { class: "hub-tab", text: "Discover" });
    const tabDevice = el("button", { class: "hub-tab", text: "On Device" });
    const kindTabs = el("div", { class: "hub-kindtabs" },
      ...KINDS.map(([k, label]) => el("button", {
        class: "hub-kindtab" + (k === kind ? " active" : ""),
        text: label, onclick: () => setKind(k) })));
    // MCP servers come from three catalogues: Hugging Face Spaces (download-
    // able repos), the official MCP Registry (registry.modelcontextprotocol.
    // io — external listings, no auth) and PulseMCP (external listings; needs
    // PULSEMCP_API_KEY). Only shown while the MCP Servers kind tab is active.
    const srcSel = selectEl([["hf", "Hugging Face"],
      ["registry", "MCP Registry"], ["pulsemcp", "PulseMCP"]], "hf");
    srcSel.title = "Which catalogue to browse for MCP servers. Only Hugging " +
      "Face repos download locally; the others link out.";
    const query = el("input", { type: "search", class: "hub-search",
      placeholder: "Search models",
      title: "Free-text search of the selected catalogue (models, datasets or MCP servers)." });
    // GGUF-only made sense when llama.cpp was the only backend; default to
    // all formats so vLLM / stable-diffusion.cpp weights surface too.
    const ggufSel = selectEl([["", "All formats"], ["1", "GGUF only"]], "");
    ggufSel.title = "Filter model weight formats. GGUF runs on llama.cpp; other " +
      "formats serve via vLLM or diffusion engines.";
    const capSel = selectEl(CAPS, "");
    capSel.title = "Filter by what the model can do (text generation, embeddings, vision…).";
    const sortSel = selectEl(SORTS, "downloads");
    sortSel.title = "Order of the result list.";

    const listHost = el("div", { class: "hub-list" });
    const detailHost = el("div", { class: "hub-detail" });
    const downloadsHost = el("div", { class: "hub-downloads" });

    function selectEl(opts, value) {
      const s = el("select", { class: "hub-select" },
        ...opts.map(([v, label]) => el("option", {
          value: v, text: label, selected: v === value })));
      return s;
    }

    // External MCP catalogue in use ("registry" | "pulsemcp") or null when
    // the HF Spaces source is selected / not on the MCP tab.
    const dirSource = () => (kind === "spaces" && srcSel.value !== "hf")
      ? srcSel.value : null;
    const DIR_LABEL = { registry: "MCP Registry", pulsemcp: "PulseMCP" };

    // Format/capability filters only mean something for models; the source
    // selector only for MCP servers (sort is HF-only too).
    function syncFilterVisibility() {
      const isModels = kind === "models";
      ggufSel.style.display = isModels ? "" : "none";
      capSel.style.display = isModels ? "" : "none";
      srcSel.style.display = kind === "spaces" ? "" : "none";
      sortSel.style.display = dirSource() ? "none" : "";
      query.placeholder = kind === "datasets"
        ? "Search datasets" : kind === "spaces"
          ? "Search MCP servers" : "Search models";
    }

    // ---------------------------------------------------------- data loads
    async function loadFeed() {
      setList(empty("loading the hub…"));
      try {
        const src = dirSource();
        if (src) {
          const r = await api(`/api/hub/mcp_dir?limit=50&source=${src}`);
          list = r.results || [];
          renderList(`${DIR_LABEL[src]} Servers`);
          return;
        }
        const r = await api(`/api/hub/feed?limit=25&kind=${kind}`);
        list = r.models || [];
        renderList(`Latest ${KIND_LABEL[kind]}`);
      } catch (e) { setList(el("p", { class: "err", text: e.message })); }
    }

    async function doSearch() {
      setList(empty("searching…"));
      try {
        let rows;
        const src = dirSource();
        if (src) {
          const p = new URLSearchParams(
            { q: query.value.trim(), limit: "50", source: src });
          rows = (await api("/api/hub/mcp_dir?" + p.toString())).results || [];
        } else {
          const p = new URLSearchParams({
            q: query.value.trim(), limit: "40", kind,
            gguf: ggufSel.value, capability: capSel.value,
            sort: sortSel.value,
          });
          rows = (await api("/api/hub/search?" + p.toString())).results || [];
        }
        list = rows;
        renderList(query.value.trim() ? `Results for “${query.value.trim()}”`
          : src ? `${DIR_LABEL[src]} servers`
            : `Popular ${KIND_LABEL[kind].toLowerCase()}`);
      } catch (e) { setList(el("p", { class: "err", text: e.message })); }
    }

    function setList(...kids) { listHost.replaceChildren(...kids); }

    function renderList(titleText) {
      const rows = list.map((m) => {
        if (m.kind === "mcp-dir" || m.kind === "mcp-registry") {
          // Registry listing: external server, links out instead of download.
          return el("div", { class: "hub-row" +
            (m.repo_id === selectedRepo ? " sel" : ""),
            onclick: () => openMcpDirDetail(m.repo_id) },
            avatar(m.author || m.name),
            el("div", { class: "hub-row-main" },
              el("div", { class: "hub-row-name", text: m.name }),
              el("div", { class: "hub-row-sub hint",
                text: m.description || m.author })),
            el("div", { class: "hub-row-stats" },
              m.downloads ? el("span", { title: "installs/downloads",
                text: `⬇ ${fmtCount(m.downloads)}` }) : null,
              m.github_stars ? el("span", { title: "github stars",
                text: `★ ${fmtCount(m.github_stars)}` }) : null),
          );
        }
        const installed = installedByRepo.has(`${kind}:${m.repo_id}`);
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
            kind === "spaces" || !m.downloads
              ? null : el("span", { title: "downloads",
                text: `⬇ ${fmtCount(m.downloads)}` }),
            el("span", { title: "likes", text: `♥ ${fmtCount(m.likes)}` }),
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
      try {
        d = await api("/api/hub/detail?kind=" + kind + "&repo="
          + encodeURIComponent(repoId));
      }
      catch (e) {
        detailHost.replaceChildren(el("p", { class: "err", text: e.message }));
        return;
      }
      renderDetail(d);
    }

    // Registry server card: external entry, no hub download — the pane shows
    // setup info and links back to the source catalogue / GitHub.
    async function openMcpDirDetail(serverId) {
      selectedRepo = serverId;
      refreshList();
      detailHost.replaceChildren(el("p", { class: "hint", text: "loading…" }));
      let d;
      try {
        d = await api("/api/hub/mcp_dir?source=" + (dirSource() || "registry")
          + "&server=" + encodeURIComponent(serverId));
      } catch (e) {
        detailHost.replaceChildren(el("p", { class: "err", text: e.message }));
        return;
      }
      const link = (href, label) => el("a", { href, target: "_blank",
        rel: "noopener noreferrer" }, label || href);
      const host = el("div", { class: "hub-readme" });
      if (d.readme) {
        renderMarkdown(host, cleanReadme(d.readme).slice(0, 24000));
      }
      detailHost.replaceChildren(...[
        el("div", { class: "hub-detail-head" },
          avatar(d.author || d.name),
          el("div", {},
            el("h3", { class: "hub-title", text: d.name }),
            el("div", { class: "hint" }, d.author || "PulseMCP"))),
        el("div", { class: "hub-chips-row" },
          badge("MCP", "dim"),
          badge(DIR_LABEL[dirSource()] || "Registry", "dim"),
          d.version ? badge("v" + d.version, "dim") : null,
          ...(d.transports || []).map((t) => badge(String(t), "dim")),
          ...(d.categories || []).map((c) => badge(c, "dim"))),
        el("div", { class: "hub-dlrow" },
          d.url ? link(d.url, `View on ${DIR_LABEL[dirSource()] || "registry"} ↗`)
            : null, " ",
          d.github_url ? link(d.github_url, "GitHub ↗") : null),
        el("p", { class: "hint", text:
          "External MCP directory entry — install via its own instructions "
          + "(npm / pip / docker) and add it to toaster_loop's MCP config." }),
        d.description && !d.readme ? el("p", { text: d.description }) : null,
        host].filter(Boolean));
    }

    // Re-render whichever list the active tab owns (keeps selection state).
    function refreshList() {
      if (tab === "device") renderDevice(lastDevice);
      else renderList(currentTitle());
    }

    function currentTitle() {
      return query.value.trim() ? `Results for “${query.value.trim()}”`
        : `Latest ${KIND_LABEL[kind]}`;
    }

    // Card images load through our same-origin proxy (CSP default-src 'self'):
    // repo-relative sources become ?repo=&path=, absolute https URLs become
    // ?url= (the backend screens for public hosts). Anything else is dropped.
    function proxiedImage(repoId, src) {
      if (!src) return null;
      const s = String(src).trim();
      if (/^(data:|blob:)/i.test(s)) return null;
      if (/^https:\/\//i.test(s) || s.startsWith("//")) {
        return withToken("/api/hub/image?kind=" + kind + "&url="
          + encodeURIComponent(s.startsWith("//") ? "https:" + s : s));
      }
      if (/^[a-z][a-z0-9+.-]*:\/\//i.test(s)) return null;  // non-http scheme
      const rel = s.startsWith("/") ? s.slice(1) : s;
      return withToken("/api/hub/image?kind=" + kind
        + "&repo=" + encodeURIComponent(repoId)
        + "&path=" + encodeURIComponent(rel));
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
      const variants = d.variants || [];
      // A single pseudo-variant ("full" — datasets, spaces, plain repos) does
      // not need a picker; just download the whole repo.
      const varSel = variants.length > 1
        ? el("select", { class: "hub-select hub-quant" },
            ...variants.map((v) => el("option", {
              value: v.variant,
              text: `${v.variant} · ${fmtBytes(v.size_bytes)}` +
                (v.shard_count ? ` (${v.shard_count} shards)` : ""),
              selected: v.variant === d.default_variant })))
        : null;
      const dlBtn = el("button", { class: "primary hub-download" },
        "⬇  Download");
      dlBtn.addEventListener("click", () => startDownload(d.repo_id,
        varSel ? varSel.value : ""));

      const chips = [];
      if (d.pipeline_tag) chips.push(badge(prettyCap(d.pipeline_tag), "dim"));
      if ((d.tags || []).includes("vision")) chips.push(badge("Vision", "dim"));
      if ((d.tags || []).includes("text-generation") || d.pipeline_tag ===
          "text-generation") chips.push(badge("Conversational", "dim"));
      if (d.kind === "spaces") {
        chips.push(badge("MCP", "dim"));
        if (d.sdk) chips.push(badge(String(d.sdk).toUpperCase(), "dim"));
      }

      const stat = (label, value) => el("span", { class: "hub-stat",
        title: label }, String(value));
      const stats = el("div", { class: "hub-stats" },
        ago(d.last_modified) ? stat("updated", ago(d.last_modified)) : null,
        d.context_length ? stat("context",
          `${fmtNum(d.context_length / 1024, d.context_length % 1024 ? 1 : 0)}K`) : null,
        d.architecture ? stat("architecture", d.architecture) : null,
        d.downloads ? stat("downloads", fmtCount(d.downloads)) : null,
        stat("likes", String(d.likes || 0)),
        d.license && d.license !== "null" ? stat("license", d.license) : null,
        d.gated ? badge("gated", "failed") : null);

      const baseRow = (d.base_models || []).length
        ? el("div", { class: "hub-base" },
            el("span", { class: "hint", text: "Base:" }),
            ...d.base_models.map((b) => el("code", { class: "mono", text: b })))
        : null;

      // NOTE: native replaceChildren() stringifies null args into a visible
      // "null" text node — filter optionals out of the list first.
      detailHost.replaceChildren(...[
        banner,
        el("div", { class: "hub-detail-head" },
          avatar(d.author),
          el("div", {},
            el("h3", { class: "hub-title", text: d.name }),
            el("div", { class: "hint" }, d.author || "", " ",
              copyButton(d.repo_id)))),
        el("div", { class: "hub-chips-row" }, ...chips),
        baseRow,
        el("div", { class: "hub-dlrow" }, varSel, dlBtn),
        stats,
        d.gated ? el("p", { class: "hint", text:
          "Gated repo — set HF_TOKEN in the environment before downloading." })
          : null,
        readmeBlock(d.repo_id, d.readme)].filter(Boolean));
    }

    function prettyCap(p) {
      return (p || "").split("-").map((w) => w[0]?.toUpperCase() + w.slice(1))
        .join(" ");
    }

    function readmeBlock(repoId, md) {
      const host = el("div", { class: "hub-readme" });
      if (!md) return host;
      // Flatten card HTML to markdown-ish text (see cleanReadme), bounded.
      renderMarkdown(host, cleanReadme(md).slice(0, 24000), {
        imageSrc: (src) => proxiedImage(repoId, src),
      });
      return host;
    }

    // ------------------------------------------------------------- download
    async function startDownload(repoId, variant) {
      try {
        await api("/api/hub/download", { method: "POST",
          body: { repo: repoId, variant, kind } });
        toast(`downloading ${repoId} (${variant || "full"})`, "ok");
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
        installedByRepo = new Map((d.installed || []).map(
          (m) => [`${m.kind || "models"}:${m.repo_id}`, m]));
        lastDevice = d;
        if (tab === "device") renderDevice(d);
        else refreshList();  // installed badges on the discover list
      } catch (e) {
        setList(el("p", { class: "err", text: e.message }));
      }
    }

    function renderDevice(d) {
      // The kind tabs filter the On Device list too, so Models/Datasets/MCP
      // each show only their own downloads.
      const items = (d.installed || []).filter(
        (m) => (m.kind || "models") === kind);
      const rows = items.length ? items.map((m) => el("div", {
        class: "hub-row" + (m.repo_id === selectedRepo ? " sel" : ""),
        onclick: () => openDetailKind(m.repo_id, m.kind || "models") },
        avatar(m.repo_id.split("/")[0]),
        el("div", { class: "hub-row-main" },
          el("div", { class: "hub-row-name", text: m.repo_id }),
          el("div", { class: "hub-row-sub hint" },
            ...(m.kind && m.kind !== "models"
              ? [badge(KIND_LABEL[m.kind] || m.kind, "dim"), " "] : []),
            (m.quants && m.quants.length ? m.quants.join(", ")
              : "full checkpoint"))),
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
                  body: { repo: m.repo_id, kind: m.kind || "models" } });
                toast(`freed ${fmtBytes(r.freed_bytes)}`, "ok");
                loadInstalled();
              } catch (e) { toast(`${e.code}: ${e.message}`, "error"); }
            } })),
      )) : [empty(`no ${KIND_LABEL[kind].toLowerCase()} downloaded yet — `
        + "pick something in Discover.")];
      setList(
        el("div", { class: "hub-list-head" },
          el("h3", { text: `On Device (${items.length})` }),
          el("span", { class: "hint", style: "font-size:.75em;",
            text: d.models_dir || "" })),
        ...rows);
    }

    // Open a detail pane for a repo of an arbitrary kind (On Device rows can
    // mix models, datasets and spaces).
    async function openDetailKind(repoId, k) {
      const prev = kind;
      kind = k;
      try { await openDetail(repoId); }
      finally { kind = prev; }
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

    function setKind(next) {
      kind = next;
      selectedRepo = null;
      for (const [i, [k]] of KINDS.entries())
        kindTabs.children[i].classList.toggle("active", k === kind);
      syncFilterVisibility();
      detailHost.replaceChildren(empty(
        `pick a ${KIND_LABEL[kind].toLowerCase().replace(/s$/, "")} to see details`));
      if (tab === "discover") {
        query.value.trim() ? doSearch() : loadFeed();
      } else {
        renderDevice(lastDevice);   // On Device list is kind-filtered too
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
    srcSel.addEventListener("change", () => {
      selectedRepo = null;
      syncFilterVisibility();
      if (tab === "discover") {
        query.value.trim() ? doSearch() : loadFeed();
      }
    });

    mount.replaceChildren(
      el("div", { class: "hub-head" },
        el("h2", { text: "Resource Hub" }),
        el("p", { class: "hint", text:
          "Discover and download models, datasets and MCP servers locally." })),
      el("div", { class: "hub-toolbar" },
        el("div", { class: "hub-tabs" }, tabDiscover, tabDevice),
        kindTabs, srcSel, query, ggufSel, capSel, sortSel),
      el("div", { class: "hub-cols" }, listHost, detailHost),
      downloadsHost);

    syncFilterVisibility();
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

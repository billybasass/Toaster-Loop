// ui.js — shared DOM helpers: element builder, toasts, modal confirm, table
// builder, drawer, formatters, and a *sanitized* mini-markdown renderer.
// SECURITY: nothing here ever assigns innerHTML from server/model data. The
// markdown renderer tokenizes text and builds DOM nodes via textContent only;
// links are restricted to http/https/relative URLs.

/** Build an element. attrs: {class, text, ...}; children: nodes|strings. */
export function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = String(v);
    else if (k === "dataset") Object.assign(e.dataset, v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, String(v));
  }
  for (const k of kids) {
    if (k == null || k === false) continue;
    if (typeof k === "string" || typeof k === "number")
      e.appendChild(document.createTextNode(String(k)));
    else e.appendChild(k);
  }
  return e;
}

// ------------------------------------------------------------------ toasts

const TOAST_MS = { info: 5000, ok: 4000, error: 8000 };

export function toast(msg, kind = "info") {
  const host = document.getElementById("toasts");
  if (!host) return;
  const t = el("div", { class: "toast toast-" + kind },
    el("span", { class: "toast-msg", text: String(msg) }),
    el("button", { class: "toast-x", "aria-label": "dismiss", text: "×",
      onclick: () => t.remove() }));
  host.appendChild(t);
  setTimeout(() => t.remove(), TOAST_MS[kind] || TOAST_MS.info);
}

// ------------------------------------------------------------- modal confirm

/** Modal confirm dialog. Resolves true on confirm, false on dismiss. */
export function confirmDialog({ title = "Confirm", body = "", confirmLabel =
  "Confirm", danger = false }) {
  return new Promise((resolve) => {
    const host = document.getElementById("modal-host");
    let done = false;
    const close = (v) => {
      if (done) return;
      done = true;
      backdrop.remove();
      document.removeEventListener("keydown", onKey);
      resolve(v);
    };
    const btnOk = el("button",
      { class: danger ? "danger" : "primary", text: confirmLabel,
        onclick: () => close(true) });
    const btnNo = el("button", { text: "Cancel", onclick: () => close(false) });
    const box = el("div", { class: "modal", role: "dialog", "aria-modal": "true" },
      el("h3", { text: title }),
      typeof body === "string" ? el("p", { text: body }) : body,
      el("div", { class: "modal-actions" }, btnNo, btnOk));
    const backdrop = el("div", { class: "backdrop",
      onclick: (ev) => { if (ev.target === backdrop) close(false); } }, box);
    function onKey(ev) {
      if (ev.key === "Escape") close(false);
      if (ev.key === "Enter") close(true);
    }
    document.addEventListener("keydown", onKey);
    host.appendChild(backdrop);
    btnOk.focus();
  });
}

// ------------------------------------------------------------- table builder

/**
 * Build a table. cols: [{label, render?(row)->Node|string, cls?}].
 * rows rendered via textContent-safe helpers only. Returns the <table>.
 */
export function table(cols, rows, opts = {}) {
  const thead = el("thead", {}, el("tr", {},
    ...cols.map((c) => el("th", { text: c.label, class: c.cls || "" }))));
  const tbody = el("tbody");
  for (const row of rows) {
    const tr = el("tr", { class: opts.rowClass ? opts.rowClass(row) : "" });
    for (const c of cols) {
      let v;
      try { v = c.render ? c.render(row) : row[c.key]; }
      catch { v = ""; }
      const td = el("td", { class: c.cls || "" });
      if (v == null || v === false) td.textContent = "–";
      else if (typeof v === "string" || typeof v === "number")
        td.textContent = String(v);
      else td.appendChild(v);
      tr.appendChild(td);
    }
    if (opts.onRowClick) {
      tr.classList.add("clickable");
      tr.tabIndex = 0;
      tr.addEventListener("click", () => opts.onRowClick(row));
      tr.addEventListener("keydown", (e) => {
        if (e.key === "Enter") opts.onRowClick(row);
      });
    }
    tbody.appendChild(tr);
  }
  const t = el("table", { class: opts.class || "" }, thead, tbody);
  if (!rows.length && opts.empty)
    return el("div", { class: "empty" }, opts.empty);
  return t;
}

/** Small state badge. */
export function badge(text, extraClass = "") {
  return el("span", { class: "badge " + (extraClass || String(text || "")),
    text: String(text ?? "–") });
}

/** Empty-state placeholder. */
export function empty(msg) {
  return el("div", { class: "empty" }, msg);
}

/** Score badge on the [-1, +1] scale: green >= .5, amber > -.5, red below. */
export function scoreBadge(score) {
  if (score == null) return badge("–", "unknown");
  const s = Number(score);
  const cls = s >= 0.5 ? "success" : s > -0.5 ? "partial" : "failure";
  return badge((s >= 0 ? "+" : "") + s.toFixed(2), cls);
}

// ------------------------------------------------------- transcript renderer
// Renders one OpenAI-format message (user/assistant/tool, assistant may carry
// tool_calls) as safe DOM: markdown via renderMarkdown for text, collapsible
// cards for tool calls/results. Reused by chat history, trace detail and the
// dataset row preview — never innerHTML.

function tryPrettyJson(v) {
  if (typeof v !== "string") return v;
  const t = v.trim();
  if (!t.startsWith("{") && !t.startsWith("[")) return v;
  try { return JSON.stringify(JSON.parse(t), null, 2); } catch { return v; }
}

/** One transcript message -> element. opts: {highlight?: string}. */
export function messageNode(m, opts = {}) {
  const role = m && m.role ? String(m.role) : "unknown";
  const wrap = el("div", { class: "msg " + role });
  const content = typeof m.content === "string" ? m.content
    : m.content == null ? "" : JSON.stringify(m.content, null, 2);

  if (role === "assistant" || role === "user" || role === "system") {
    const head = el("div", { class: "msg-role" }, role);
    wrap.appendChild(head);
    if (role === "system") {
      wrap.appendChild(el("pre", { text: content }));
    } else {
      const bodyEl = el("div", { class: "md" });
      renderMarkdown(bodyEl, content);
      wrap.appendChild(bodyEl);
    }
  } else if (role === "tool") {
    // Tool result: collapsible card keyed to its call id.
    const body = el("div", { class: "tool-body hidden" });
    body.appendChild(el("pre", { class: "log", text: String(tryPrettyJson(content)) }));
    const head = el("button", { class: "tool-head" },
      el("strong", { text: m.name || "tool result" }),
      el("span", { class: "hint mono", text: m.tool_call_id || "" }));
    head.addEventListener("click", () => body.classList.toggle("hidden"));
    wrap.classList.add("tool-card");
    wrap.append(head, body);
  } else {
    wrap.appendChild(el("pre", { text: content }));
  }

  for (const tc of (m && m.tool_calls) || []) {
    const fn = tc.function || {};
    let argsText;
    try { argsText = JSON.stringify(JSON.parse(fn.arguments || "{}"), null, 2); }
    catch { argsText = String(fn.arguments ?? ""); }
    const body = el("div", { class: "tool-body hidden" },
      el("pre", { class: "log json", text: argsText }));
    const head = el("button", { class: "tool-head" },
      el("span", { class: "tool-state", text: "▶" }),
      el("strong", { text: fn.name || "tool" }),
      el("span", { class: "hint", text: "tool call" }));
    head.addEventListener("click", () => body.classList.toggle("hidden"));
    const card = el("div", { class: "msg tool-card" }, head, body);
    wrap.appendChild(card);
  }

  if (opts.highlight && content.toLowerCase().includes(
      String(opts.highlight).toLowerCase()))
    wrap.classList.add("match");
  return wrap;
}

// ------------------------------------------------------------- SVG chart kit
// Hand-rolled inline SVG helpers (no libs). Elements are created with the SVG
// namespace; tooltips via <title> children. Used by the traces stats strip
// and reusable for hardware gauges / iterate timelines in 5d.

const SVG_NS = "http://www.w3.org/2000/svg";

export function svgEl(tag, attrs = {}, ...kids) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    e.setAttribute(k, String(v));
  }
  for (const k of kids) {
    if (k == null || k === false) continue;
    e.appendChild(typeof k === "string" ? document.createTextNode(k) : k);
  }
  return e;
}

/**
 * Daily bar chart: bars = counts, overlaid polyline = a -1..1 series (score),
 * red ticks mark failure rate. items: [{label, n, score?, fail?}]. Returns an
 * <svg> element sized w x h. Tooltips via title attrs (native browser tips).
 */
export function svgDailyChart(items, { w = 560, h = 90 } = {}) {
  const padL = 4, padR = 4, padT = 6, padB = 14;
  const iw = w - padL - padR, ih = h - padT - padB;
  const maxN = Math.max(1, ...items.map((d) => d.n || 0));
  const bw = items.length ? Math.max(3, Math.floor(iw / items.length) - 3) : 8;
  const svg = svgEl("svg", { class: "chart", viewBox: `0 0 ${w} ${h}`,
    width: w, height: h, role: "img",
    "aria-label": "daily traces chart" });
  // baseline
  svg.appendChild(svgEl("line", { x1: padL, y1: padT + ih, x2: padL + iw,
    y2: padT + ih, class: "chart-axis" }));
  const pts = [];
  items.forEach((d, i) => {
    const x = padL + Math.floor(i * iw / Math.max(1, items.length));
    const bh = Math.round((d.n || 0) / maxN * ih);
    const title = `${d.label}: ${d.n} traces` +
      (d.score != null ? ` · avg ${Number(d.score).toFixed(2)}` : "") +
      (d.fail != null ? ` · fail ${Math.round(d.fail * 100)}%` : "");
    const bar = svgEl("rect", { x, y: padT + ih - bh, width: bw, height: Math.max(bh, d.n ? 1 : 0),
      class: "chart-bar" }, svgEl("title", {}, title));
    svg.appendChild(bar);
    if (d.fail != null && d.fail > 0) {
      svg.appendChild(svgEl("rect", { x, y: padT + ih - Math.max(bh, 1),
        width: bw, height: Math.max(2, Math.round(d.fail * Math.max(bh, 4))),
        class: "chart-fail" }, svgEl("title", {}, title)));
    }
    if (d.score != null) {
      const cx = x + bw / 2;
      const cy = padT + ih / 2 - (Number(d.score) * ih) / 2;
      pts.push(`${cx},${cy}`);
    }
  });
  if (pts.length > 1)
    svg.appendChild(svgEl("polyline", { points: pts.join(" "), class: "chart-line",
      fill: "none" }));
  return svg;
}

/**
 * Horizontal capacity bar/gauge: value vs max with an optional threshold
 * tick. Used by the Train wizard (unclaimed traces vs threshold) and the
 * Hardware panel (VRAM free/total, RAM). Returns an <svg>.
 * opts: {w, h, label?, unit?, threshold? (0..max), danger? (bool)}.
 */
export function svgGauge(value, max, { w = 220, h = 34, label = "", unit = "",
  threshold = null, danger = false } = {}) {
  const v = Number(value) || 0;
  const m = Math.max(Number(max) || 0, v, 1e-9);
  const frac = Math.max(0, Math.min(1, v / m));
  const barH = 12, barY = h - barH - 10, padL = 2;
  const svg = svgEl("svg", { class: "chart gauge", viewBox: `0 0 ${w} ${h}`,
    width: w, height: h, role: "img", "aria-label": `${label}: ${v}${unit}` });
  if (label)
    svg.appendChild(svgEl("text", { x: padL, y: 11, class: "chart-label" }, label));
  svg.appendChild(svgEl("rect", { x: padL, y: barY, width: w - 2 * padL,
    height: barH, class: "chart-track", rx: 3 }));
  svg.appendChild(svgEl("rect", { x: padL, y: barY,
    width: Math.max(1, Math.round((w - 2 * padL) * frac)), height: barH,
    class: "chart-fill" + (danger ? " danger" : ""), rx: 3 },
    svgEl("title", {}, `${v}${unit} / ${max}${unit}`)));
  if (threshold != null && Number(threshold) > 0 && Number(threshold) <= m) {
    const tx = padL + Math.round((w - 2 * padL) * (Number(threshold) / m));
    svg.appendChild(svgEl("line", { x1: tx, y1: barY - 3, x2: tx,
      y2: barY + barH + 3, class: "chart-threshold" },
      svgEl("title", {}, `threshold ${threshold}${unit}`)));
  }
  svg.appendChild(svgEl("text", { x: w - 2, y: h - 1, class: "chart-value",
    "text-anchor": "end" }, `${v}${unit} / ${max}${unit}`));
  return svg;
}

/** Sparkline over a numeric history array (client ring buffer). */
export function svgSparkline(values, { w = 220, h = 42, min = null,
  max = null, label = "" } = {}) {
  const vals = (values || []).map(Number).filter((x) => !Number.isNaN(x));
  const svg = svgEl("svg", { class: "chart spark", viewBox: `0 0 ${w} ${h}`,
    width: w, height: h, role: "img", "aria-label": label || "sparkline" });
  if (vals.length < 2) {
    svg.appendChild(svgEl("text", { x: 4, y: h / 2 + 4, class: "chart-label" },
      vals.length ? "one sample…" : "collecting samples…"));
    return svg;
  }
  const lo = min != null ? Number(min) : Math.min(...vals);
  const hiRaw = max != null ? Number(max) : Math.max(...vals);
  const hi = hiRaw > lo ? hiRaw : lo + 1;
  const pad = 3;
  const pts = vals.map((v, i) => {
    const x = pad + (i * (w - 2 * pad)) / (vals.length - 1);
    const y = h - pad - ((v - lo) / (hi - lo)) * (h - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  svg.appendChild(svgEl("polyline", { points: pts.join(" "),
    class: "chart-line", fill: "none" }));
  return svg;
}

// ---------------------------------------------------------------- formatters

export function fmtTime(iso, withSeconds = true) {
  if (!iso) return "–";
  const s = String(iso).replace("T", " ");
  return withSeconds ? s.slice(0, 19) : s.slice(5, 16);
}

export function fmtUptime(startIso) {
  if (!startIso) return "";
  const t = Date.parse(startIso);
  if (Number.isNaN(t)) return "";
  let s = Math.max(0, Math.floor((Date.now() - t) / 1000));
  const h = Math.floor(s / 3600); s %= 3600;
  const m = Math.floor(s / 60);
  return h ? `${h}h ${m}m` : `${m}m`;
}

export function fmtNum(n, digits = 2) {
  return n == null ? "–" : Number(n).toFixed(digits);
}

// ------------------------------------------- sanitized mini-markdown renderer
// Supports: fenced code blocks (```), **bold**, *italic*, `inline code`,
// [text](http(s)-only links). Everything else renders as plain text. Output is
// built exclusively from createElement/textContent — no innerHTML, ever.

function safeUrl(u) {
  try {
    const parsed = new URL(u, location.origin);
    return ["http:", "https:"].includes(parsed.protocol) ? parsed.href : null;
  } catch { return null; }
}

function renderInline(target, text, opts = {}) {
  // Tokenize inline spans; unmatched markers stay literal. Images come first
  // so "![alt](url)" never matches as a link. opts.imageSrc(src) maps a card
  // image source to a displayable URL (the hub proxies through /api/hub/image
  // because our CSP is default-src 'self'); without it only http(s) srcs load.
  const re = /(!\[([^\]]*)\]\(([^)\s]+)\))|(\*\*([^*]+)\*\*)|(\*([^*]+)\*)|(`([^`]+)`)|(\[([^\]]+)\]\(([^)\s]+)\))/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) target.appendChild(document.createTextNode(text.slice(last, m.index)));
    if (m[1]) {
      const src = opts.imageSrc ? opts.imageSrc(m[3]) : safeUrl(m[3]);
      if (src) {
        target.appendChild(el("img", { class: "md-img", src, alt: m[2] || "",
          loading: "lazy" }));
      } else target.appendChild(document.createTextNode(m[1]));
    }
    else if (m[4]) target.appendChild(el("strong", {}, textNode(m[5])));
    else if (m[6]) target.appendChild(el("em", {}, textNode(m[7])));
    else if (m[8]) target.appendChild(el("code", { text: m[9] }));
    else if (m[10]) {
      const href = safeUrl(m[12]);
      if (href) {
        target.appendChild(el("a", { href, rel: "noopener noreferrer",
          target: "_blank", text: m[11] }));
      } else target.appendChild(document.createTextNode(m[11] + " (" + m[12] + ")"));
    }
    last = re.lastIndex;
  }
  if (last < text.length) target.appendChild(document.createTextNode(text.slice(last)));
}

function textNode(s) { return document.createTextNode(s); }

/** Render assistant markdown-ish text into a container of safe nodes.
 *  opts.imageSrc(src) optionally rewrites image URLs (see renderInline). */
export function renderMarkdown(container, text, opts = {}) {
  container.replaceChildren();
  const src = String(text ?? "");
  const lines = src.split("\n");
  let para = null;
  let inFence = false;
  let fenceLang = "";
  let fenceLines = [];

  const flushPara = () => {
    if (para) { container.appendChild(para); para = null; }
  };
  const ensurePara = () => {
    if (!para) { para = el("p", { class: "md-p" }); container.appendChild(para); }
    return para;
  };

  for (const line of lines) {
    const fence = line.match(/^\s*```(\w*)\s*$/);
    if (fence && !inFence) { inFence = true; fenceLang = fence[1] || ""; fenceLines = []; flushPara(); continue; }
    if (inFence) {
      if (line.match(/^\s*```\s*$/)) {
        inFence = false;
        const code = el("code", { text: fenceLines.join("\n") + "\n" });
        const pre = el("pre", { class: "md-pre" }, code);
        if (fenceLang) pre.dataset.lang = fenceLang;
        container.appendChild(pre);
      } else fenceLines.push(line);
      continue;
    }
    if (!line.trim()) { flushPara(); continue; }
    const head = line.match(/^(#{1,4})\s+(.*)$/);
    if (head) {
      flushPara();
      const h = el("h" + (head[1].length + 3), { class: "md-h" });
      renderInline(h, head[2], opts);
      container.appendChild(h);
      continue;
    }
    const li = line.match(/^\s*[-*]\s+(.*)$/);
    if (li) {
      flushPara();
      const item = el("li", { class: "md-li" });
      renderInline(item, li[1], opts);
      container.appendChild(item);
      continue;
    }
    const p = ensurePara();
    if (p.childNodes.length) p.appendChild(document.createTextNode(" "));
    renderInline(p, line, opts);
  }
  if (inFence && fenceLines.length) { // unterminated fence: show as code
    container.appendChild(el("pre", { class: "md-pre" },
      el("code", { text: fenceLines.join("\n") })));
  }
  flushPara();
  return container;
}

/** Copy text to clipboard with visual feedback on the button. */
export function copyButton(text, label = "copy") {
  const b = el("button", { class: "small", text: label });
  b.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      b.textContent = "copied ✓";
    } catch {
      // fallback: hidden textarea + execCommand (loopback http may block API)
      const ta = el("textarea"); ta.value = text;
      document.body.appendChild(ta); ta.select();
      try { document.execCommand("copy"); b.textContent = "copied ✓"; }
      catch { b.textContent = "copy failed"; }
      ta.remove();
    }
    setTimeout(() => { b.textContent = label; }, 1500);
  });
  return b;
}

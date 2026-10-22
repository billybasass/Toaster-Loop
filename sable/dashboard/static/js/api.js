// api.js — central fetch/SSE client. Same-origin only (CSP default-src 'self').
// Handles: token header + sessionStorage token, ?token= bootstrap, the
// {"error": {code, message}} envelope, and a line-based SSE reader for POST
// response streams (EventSource cannot do POST).

const TOKEN_KEY = "sable.token";

export function getToken() {
  try { return sessionStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; }
}

export function setToken(t) {
  try {
    if (t) sessionStorage.setItem(TOKEN_KEY, t);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch { /* private mode: token stays in-memory only */ }
  window.dispatchEvent(new CustomEvent("sable:token"));
}

// Bootstrap: ?token=... from the URL is moved into sessionStorage and stripped
// so it never lingers in history or logs.
(function bootstrapTokenFromUrl() {
  try {
    const u = new URLSearchParams(location.search);
    const t = u.get("token");
    if (t) {
      setToken(t);
      u.delete("token");
      const q = u.toString();
      history.replaceState(null, "",
        location.pathname + (q ? "?" + q : "") + location.hash);
    }
  } catch { /* non-browser context */ }
})();

export class ApiError extends Error {
  constructor(code, message, status) {
    super(message || code || "error");
    this.code = code || "error";
    this.status = status || 0;
  }
}

function headers(json = true) {
  const h = {};
  if (json) h["Content-Type"] = "application/json";
  const t = getToken();
  if (t) h["X-Sable-Token"] = t;
  return h;
}

// Prompt for a token on 401; returns true when one was stored.
async function promptToken() {
  const t = window.prompt(
    "sable dashboard requires the access token.\n" +
    "(printed by `sable dashboard` when started with --token)");
  if (t && t.trim()) { setToken(t.trim()); return true; }
  return false;
}

function parseEnvelope(text, status) {
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch { data = { raw: text }; }
  if (status >= 400) {
    const e = data && data.error;
    if (e && typeof e === "object") {
      const err = new ApiError(e.code, e.message, status);
      // Machine-readable extras ride on the envelope (e.g. run_id on a
      // 409 run_in_flight); callers can read them off the thrown error.
      Object.assign(err, e);
      throw err;
    }
    throw new ApiError("http_" + status,
      data && data.raw ? String(data.raw).slice(0, 300) : `${status} error`,
      status);
  }
  return data;
}

/** JSON fetch with token handling + error-envelope -> ApiError. */
export async function api(path, opts = {}) {
  const method = (opts.method || "GET").toUpperCase();
  const init = { method, headers: headers(opts.body != null) };
  if (opts.body != null) init.body = JSON.stringify(opts.body);
  let res = await fetch(path, init);
  if (res.status === 401 && (await promptToken())) res = await fetch(path, {
    ...init, headers: headers(),
  });
  const text = await res.text();
  return parseEnvelope(text, res.status);
}

/** URL for EventSource / links that must carry ?token= in token mode. */
export function sseUrl(path) {
  const t = getToken();
  if (!t) return path;
  return path + (path.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(t);
}

/**
 * POST a JSON body and consume the SSE response stream. Callback signature:
 * onEvent(kind, data). Handles both our own `event:`/`data:` protocol and raw
 * OpenAI-style passthrough (kind defaults to "message"; "[DONE]" arrives as
 * {raw:"[DONE]"}). Throws ApiError on non-200 JSON error envelopes.
 */
export async function streamSSE(path, body, onEvent, signal) {
  let res = await fetch(path, {
    method: "POST", headers: headers(), body: JSON.stringify(body), signal,
  });
  if (res.status === 401 && (await promptToken())) {
    res = await fetch(path, {
      method: "POST", headers: headers(), body: JSON.stringify(body), signal,
    });
  }
  if (!res.ok) {
    const text = await res.text();
    parseEnvelope(text, res.status); // throws ApiError
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let kind = null;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx).replace(/\r$/, "");
      buf = buf.slice(idx + 1);
      if (!line) { kind = null; continue; }        // frame boundary
      if (line.startsWith(":")) continue;          // SSE comment / heartbeat
      if (line.startsWith("event:")) { kind = line.slice(6).trim(); }
      else if (line.startsWith("data:")) {
        const raw = line.slice(5).trim();
        let data;
        try { data = JSON.parse(raw); } catch { data = { raw }; }
        onEvent(kind || "message", data);
        kind = null;
      }
    }
  }
}

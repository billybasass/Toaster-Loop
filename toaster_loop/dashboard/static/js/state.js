// state.js — state-authority rules for SSE/UI reconciliation (pass 3 #24)
// and scope-declared persistence keys with a load-time janitor (#25).
//
// State rules ("Server truth is cached, not
// owned"), adapted to our vanilla-JS stack:
//
//   * A panel OWNS its optimistic overlay (in-flight user intent); the server
//     response is MERGED into it, never replaces wholesale.
//   * Every refresh fetch carries a generation token; a response whose
//     generation was superseded while in flight is DROPPED — a late reply can
//     never overwrite newer intent or fresher data (out-of-order clobbering).
//   * Terminal SSE transitions bypass any coalescing throttle so a job
//     finishing is painted immediately, high-frequency cosmetic events coalesce.
//
// Persistence convention (#25): keys are `uf:<scope>:<id?>:<thing>`. Any key
// whose scope entity (project/profile) no longer exists is dropped by the
// janitor on load and by rename/delete mirrors — so a deleted project can
// never leave panel state pointing at dead ids.

// ------------------------------------------------ generation-guarded fetch

/** Per-resource generation counter: only the newest request may apply.
 *
 *  Usage in a panel:
 *     const guard = generation("jobs");
 *     async function load() {
 *       const done = guard.begin();
 *       const data = await api("/api/jobs");
 *       if (done()) render(data);      // stale responses stop here
 *     }
 */
const _gens = new Map();

export function generation(resource) {
  let n = _gens.get(resource) || 0;
  return {
    /** Claim the current generation; returns done() -> boolean. Call done()
     *  after each await before applying results; false means superseded. */
    begin() {
      const mine = ++n;
      _gens.set(resource, mine);
      return () => _gens.get(resource) === mine;
    },
    current() {
      return n;
    },
  };
}

// ------------------------------------------------------- merge-don't-clobber

/** Merge a freshly fetched server list over the live (local) one.
 *
 *  - rows present in both: server wins UNLESS an overlay marks them dirty
 *    (optimistic edit still in flight — the visible paint stays until the
 *    authoritative refresh that includes the new state arrives).
 *  - rows only in `live`: kept while their id is overlaid, dropped otherwise
 *    (server is authoritative about existence).
 *  - order follows the server list; unknown-but-live rows append.
 *
 *  Returns a NEW array; inputs untouched. `overlay` is an optional Set of ids
 *  with pending local intent.
 */
export function mergeById(fetched, live, keyFn, overlay = null) {
  const liveMap = new Map((live || []).map((r) => [keyFn(r), r]));
  const out = [];
  const seen = new Set();
  for (const row of fetched || []) {
    const id = keyFn(row);
    seen.add(id);
    const local = liveMap.get(id);
    if (local && overlay && overlay.has(id)) out.push(local); // intent wins mid-flight
    else out.push(row);                                        // server truth
  }
  for (const row of live || []) {
    const id = keyFn(row);
    if (!seen.has(id) && overlay && overlay.has(id)) out.push(row);
  }
  return out;
}

// ------------------------------------------------ SSE coalescing w/ bypass

/** Throttle high-frequency refresh triggers, but deliver `isTerminal` events
 *  immediately (cosmetic events coalesce, terminal transitions flush).
 *
 *     const kick = coalesced(() => load(), {
 *       ms: 400,
 *       isTerminal: (p) => p && ["done","error","cancelled"].includes(p.state),
 *     });
 *     bus.on("job.state", kick);
 */
export function coalesced(fn, { ms = 400, isTerminal = null } = {}) {
  let timer = null;
  const flush = () => {
    if (timer) { clearTimeout(timer); timer = null; }
    fn();
  };
  return (payload) => {
    if (isTerminal && isTerminal(payload)) { flush(); return; }
    if (timer) return; // coalesce: schedule one trailing call
    timer = setTimeout(() => { timer = null; fn(); }, ms);
  };
}

// --------------------------------------------- scoped persistence (#25)

const PREFIX = "uf:";
// Legacy un-prefixed keys still in the wild -> their canonical scoped form.
const LEGACY_MAP = {
  // ui scope = global panel state (empty id segment -> janitor never touches).
  // NOTE: the active-project POINTER lives in ui scope; only per-project
  // state ("uf:project:<id>:<thing>") is janitored when a project dies.
  "toaster_loop.chat.mode": PREFIX + "ui::mode",
  "toaster_loop.project.active": PREFIX + "ui::active-project",
  "toaster_loop.projects.collapsed": PREFIX + "ui::collapsed-projects",
  "toaster_loop.explorer.width": PREFIX + "ui::explorer-width",
};
// Reverse (new -> old) so sget() can transparently read a pre-convention key.
const LEGACY_OF = {};
for (const [oldK, newK] of Object.entries(LEGACY_MAP)) LEGACY_OF[newK] = oldK;

/** Build a scoped key. `scope` is the entity family ("project", "profile"),
 *  `id` the entity id ("" for global), `thing` the state name. */
export function skey(scope, id, thing) {
  return `${PREFIX}${scope}:${id || ""}:${thing}`;
}

export function sget(key, fallback = null) {
  try {
    let raw = localStorage.getItem(key);
    // One-time lazy migration: read the legacy key, move it under the new
    // name so the janitor/rename machinery sees only canonical keys.
    if (raw === null && LEGACY_OF[key]) {
      raw = localStorage.getItem(LEGACY_OF[key]);
      if (raw !== null) {
        localStorage.setItem(key, raw);
        localStorage.removeItem(LEGACY_OF[key]);
      }
    }
    return raw === null ? fallback : raw;
  } catch { return fallback; }
}

export function sset(key, value) {
  try { localStorage.setItem(key, String(value)); } catch { /* full/private */ }
}

export function sdel(key) {
  try { localStorage.removeItem(key); } catch { /* ignore */ }
}

/** Drop every key whose scoped id is not in `validIds` (janitor).
 *
 *  Called on app load with the live project ids, and again after a project
 *  delete. state-hygiene rule: EVERY keyed family must be covered by rename AND
 *  delete migrations — centralizing here keeps that enforceable.
 *
 *  Keys with an empty id segment (global state) are never touched. Returns
 *  the number of keys dropped (tests + debug).
 */
export function janitorKeys(scope, validIds) {
  const ok = new Set(validIds || []);
  let dropped = 0;
  const doomed = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (!k || !k.startsWith(`${PREFIX}${scope}:`)) continue;
      const rest = k.slice(`${PREFIX}${scope}:`.length);
      const id = rest.split(":")[0];
      if (id && !ok.has(id)) doomed.push(k);
    }
  } catch { return 0; }
  for (const k of doomed) { sdel(k); dropped++; }
  return dropped;
}

/** Mirror a rename: move every key under oldId to newId. */
export function renameScope(scope, oldId, newId) {
  const from = `${PREFIX}${scope}:${oldId}:`;
  const moves = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith(from)) moves.push([k, `${PREFIX}${scope}:${newId}:${k.slice(from.length)}`]);
    }
  } catch { return 0; }
  for (const [k, nk] of moves) {
    try { sset(nk, localStorage.getItem(k)); } catch { /* ignore */ }
    sdel(k);
  }
  return moves.length;
}

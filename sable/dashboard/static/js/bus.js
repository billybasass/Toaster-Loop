// bus.js — global SSE connection manager over GET /api/events.
// EventSource handles reconnect (server sets retry: 3000); we layer on:
//  - ?token= auth in token mode
//  - kind-based subscriptions with wildcard "*"
//  - synthetic `overflow` events -> refetch-all listeners
//  - online/offline status for the header pill

import { sseUrl } from "./api.js";

const KINDS = ["job.state", "job.log", "serve.state", "serve.watchdog",
  "model.promoted", "model.rolled_back", "trace.recorded",
  "trace.feedbacked", "iterate.step",
  "runtime.setup", "index.build", "app.state", "delegation.update",
  "bench.run", "calibrate.run", "doctor.run", "engines.changed",
  "engine.install"];

class EventBusClient {
  constructor() {
    this._subs = new Map(); // kind -> Set<fn>
    this._es = null;
    this.online = false;
    this._retryDelay = 3000;
  }

  /** Subscribe to an event kind (or "*" for all). Returns unsubscribe fn. */
  on(kind, fn) {
    if (!this._subs.has(kind)) this._subs.set(kind, new Set());
    this._subs.get(kind).add(fn);
    return () => this._subs.get(kind)?.delete(fn);
  }

  _emit(kind, event) {
    const fire = (set) => {
      if (!set) return;
      for (const fn of set) {
        try { fn(event.payload || {}, event); }
        catch (e) { console.error("bus listener error:", kind, e); }
      }
    };
    fire(this._subs.get(kind));
    fire(this._subs.get("*"));
  }

  connect() {
    if (this._es) { this._es.close(); this._es = null; }
    let es;
    try { es = new EventSource(sseUrl("/api/events")); }
    catch (e) {
      // e.g. blocked environment: degrade to offline, keep retrying slowly.
      this._setStatus(false);
      setTimeout(() => this.connect(), 10000);
      return;
    }
    this._es = es;

    es.onopen = () => this._setStatus(true);
    es.onerror = () => {
      // EventSource auto-reconnects (retry: 3000 from server). If it gave up
      // entirely (CONNECTING===closed), rebuild manually.
      if (es.readyState === EventSource.CLOSED) {
        this._setStatus(false);
        setTimeout(() => this.connect(), this._retryDelay);
      }
    };

    for (const kind of KINDS) {
      es.addEventListener(kind, (ev) => {
        let event;
        try { event = JSON.parse(ev.data); } catch { return; }
        this._emit(event.kind || kind, event);
      });
    }
    // Synthetic overflow: client missed events -> everyone should refetch.
    es.addEventListener("overflow", (ev) => {
      let event;
      try { event = JSON.parse(ev.data); } catch { event = { payload: {} }; }
      this._emit("overflow", event);
    });
  }

  _setStatus(on) {
    if (this.online === on) return;
    this.online = on;
    window.dispatchEvent(new CustomEvent("sable:bus",
      { detail: { online: on } }));
  }
}

export const bus = new EventBusClient();

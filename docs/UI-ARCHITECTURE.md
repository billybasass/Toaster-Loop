# UI-ARCHITECTURE — toaster-loop Phase 5 (design)

Status: **implemented** (Phases 5a–5e shipped; dashboard grew further in Phase 6 with
multi-instance serving and the Model Hub / Easy panels). This is the design document as
written; where the build diverged, the code is the source of truth. Notable deltas:
- `store.js` shipped as `js/state.js`; the panels are `overview chat serve models hub
  jobs traces datasets train easy iterate schedule profiles hardware settings` (15 —
  "agents"/"scheduler" became **Profiles**/**Schedule**, plus later **Hub** and **Easy**).
- Chat agent runs execute on dashboard daemon threads (in-process runner), not a worker
  subprocess; every long action follows the same thread + job-record pattern.
- Body caps are per-route: generic `/api/*` cap at 1 MB, but `POST /api/chat/stream`
  accepts ~43.5 MB for image attachments (up to 4 × 8 MB base64 data URLs).
- Token auth shipped as designed (`--token` / `dashboard.token` /
  `TOASTER_LOOP_DASHBOARD_TOKEN`, `X-Toaster Loop-Token` header or `?token=` for EventSource).
- Phase 6 added the `/api/servers*` instance endpoints on top of the legacy
  `/api/serve/*` default-instance path (README "Phase 6 API contract notes").

Supersedes the basic Phase 5 outline in [`ROADMAP.md`](ROADMAP.md). Companion doc:
[`UI-ROADMAP.md`](UI-ROADMAP.md) (implementation phases 5a–5e).

Design goal: one local web app that exposes **every user-facing capability of
toaster-loop's own backend** — chat/agent runs, serving, registry, training loop, buffer
analytics — over a dependency-free, offline-capable UI. The shipped API surface is
the route table in [`dashboard/app.py`](../toaster-loop/dashboard/app.py); contract notes
live in the README.

---

## 1. Tech stack decision

### 1.1 Backend HTTP layer — keep stdlib `ThreadingHTTPServer`

The existing [`dashboard/server.py`](../toaster-loop/dashboard/server.py) already
implements the right shape: loopback-only bind (`127.0.0.1`, not configurable),
transport-agnostic `DashboardApp.handle(method, path, query, body)` core
([`dashboard/app.py`](../toaster-loop/dashboard/app.py)), static file serving with a
path-traversal guard, CSP headers, and secret-free access logging. All 74+
existing tests call `DashboardApp.handle()` directly — zero-socket unit testability.

**Decision: grow the stdlib server; do NOT adopt FastAPI.**

| Criterion | stdlib (chosen) | FastAPI |
|---|---|---|
| Core deps | 0 added (pyyaml+httpx stay the only runtime deps, per D-1/D-4 minimalism) | +fastapi, +starlette, +pydantic, +uvicorn |
| SSE streaming | fine — handler writes `text/event-stream` chunks directly on its worker thread | better ergonomics, but we don't need async fanout (single local user) |
| Testability | already proven (`app.handle()` direct) | would force rewriting the API surface |
| Risk | routing boilerplate grows (mitigate with a small pattern-matcher, §3.2) | dependency-pinning fights the training extra's version pins |

The one gap worth borrowing from FastAPI is **route patterns with path params**.
Add a ~40-line matcher to `DashboardApp._route` (`("/api/traces/<id>", handler)`
table) instead of the current hand-chained `startswith` checks. Everything else
stays.

### 1.2 Frontend — dependency-free vanilla JS SPA, grown from `static/`

**Decision: keep vanilla JS (ES modules), no build step, no vendored framework.**

Rationale:
- The existing [`static/index.html`](../toaster-loop/dashboard/static/index.html)
  + `app.js` + `style.css` already enforce the hard constraints: CSP
  `default-src 'self'`, zero CDN references, offline-capable, and a hygiene test
  ([`tests/test_dashboard.py`](../tests/test_dashboard.py)) asserts no external
  URLs. A framework (even vendored Preact/Vue ESM) adds ~40–100 KB of third-party
  code we must vendor, audit, and keep license-clean — for a single-user local UI.
- Scale check: the full feature set (§4, ~12 panels) is comparable in DOM
  complexity to what `app.js` already handles ×5. Manageable with ES-module
  per-panel files (`static/js/panels/chat.js`, etc.), a tiny shared store, and
  a `<template>`-based render helper (~100 LOC). If a panel ever genuinely hurts,
  we can drop in vendored [Preact+htm](https://htm.dev) (MIT) later — that escape
  hatch stays open without paying for it now.
- No build step means `pip install toaster-loop` ships the UI as package data
  (`pyproject.toml` already includes `static/*`); extend to `static/js/*.js`.

Structure (multi-file, still zero-build; browser resolves module paths natively):

```
toaster-loop/dashboard/static/
  index.html          # app shell: nav rail + <main> outlet + dialog host
  style.css           # design tokens (CSS custom props) + layout + components
  js/
    api.js            # fetch/SSE client, error normalization, token header
    store.js          # app state + pub/sub (~60 LOC)
    router.js         # hash router (#/chat, #/traces/<id>, ...)
    ui.js             # el()/template helpers, toast, modal-confirm, table builder
    panels/
      overview.js  chat.js  serve.js  models.js  jobs.js  traces.js
      datasets.js  train.js  iterate.js  hardware.js  agents.js
      scheduler.js settings.js
```

### 1.3 Live updates — SSE one-way + polling hybrid

**Decision: Server-Sent Events for event notification; plain GET polling for
low-frequency state.** No WebSocket (adds an upgrade/handshake the stdlib server
would need hand-rolled; toaster-loop's needs are all server→client).

Two mechanisms, deliberately simple:

1. **`GET /api/events` (SSE)** — a per-connection event queue fed by an
   in-process `EventBus` singleton in `DashboardApp`. Events:
   `job.state`, `job.log` (tail lines), `serve.state`, `model.promoted`,
   `trace.recorded`, `iterate.step`. Reconnect on the client with
   `EventSource` (auto-reconnect built in). The bus is best-effort: a client
   that missed events refetches on reconnect (`Last-Event-ID` optional, v2).
2. **Poll fallback** — every panel also refreshes on tab focus + a slow timer
   (status 5 s, jobs 3 s while running). If SSE drops, the UI degrades to today's
   behavior rather than going stale.

Chat token streaming is separate from the bus: it is a *response stream*, not a
notification — see §5.

### 1.4 Serve model & auth

- Single command: `toaster-loop dashboard [--port N]` (exists; keep). Default port
  0 → dynamic; config `dashboard.port`; print URL once. Consider
  `--no-browser`/auto-open later (deferred — opening browsers is environment
  rude).
- Bind stays **hard-coded 127.0.0.1** (`BIND_HOST`), matching D-4's loopback
  posture. Never configurable to non-loopback in v1; remote access, if ever, is
  an explicit future feature with real auth (out of scope).
- **Optional token auth** for shared machines: `--token` flag or
  `TOASTER_LOOP_DASHBOARD_TOKEN` env. When set, every `/api/*` request must carry
  `X-Toaster Loop-Token`; static assets stay open (they contain no state); a login
  overlay in the SPA stores the token in `sessionStorage`. Default: off
  (loopback = single user). CSRF is handled by rejecting cross-origin requests:
  server checks `Origin`/`Sec-Fetch-Site` on mutating verbs regardless of token.
- CSP stays `default-src 'self'; script-src 'self'; style-src 'self'`; the
  hygiene test extends to `static/js/**`.

### 1.5 Asset guardrails (binding, per D-3)

- **All UI assets are authored in this repo**: CSS, icons, fonts, logos, copy
  strings — nothing is copied from third-party products.
- [`tests/test_hygiene.py`](../tests/test_hygiene.py) plus the
  dashboard tests enforce the shipped tree stays dependency-free: no external
  URLs, no vendored frameworks.
- Any runtime integration with other local programs is HTTP-only as an ordinary
  client (attach mode against `/v1`); toaster-loop never imports or embeds foreign
  UI code.

---

## 2. Relationship to the existing dashboard — grow, don't rewrite

**Recommendation: GROW.** The crash left behind ~870 LOC that are architecturally
correct for everything decided above:

| Existing asset | Verdict | Why |
|---|---|---|
| `app.py` (`DashboardApp`) | keep + extend | transport-agnostic, endpoint-per-method routing table already in place; 12 endpoints implemented and tested — exactly the pattern the new catalog extends |
| `server.py` (stdlib loopback server) | keep + minor edits | add SSE verb handling (long-lived response writes), token/Origin checks, route-pattern matcher |
| `static/index.html` + `app.js` | keep as shell seed; refactor into ES modules incrementally | tabs→panels mapping is direct; split per-panel code when each panel grows past ~150 LOC |
| `tests/test_dashboard.py` (332 lines) | keep; template for all new API tests | fixture pattern (`home`, `app`, `seed_trace`) reused by every 5a test |

What's missing is *breadth* (chat, serving controls, hardware, settings, agent
profiles, scheduler views, streaming), not a wrong foundation. A rewrite would
discard tested invariants (loopback bind, traversal guard, promote/rollback via
registry error mapping → 409, background-thread job execution with the
`_exec_lock` + registry single-flight backstop) and re-incur their bug surface.

---

## 3. App shell

### 3.1 Layout

```
┌───────────────────────────────────────────────────────────────┐
│ ⛭ toaster-loop        [status chips: server ● model v2 ● GPU ▾]   │ header
├──────────┬────────────────────────────────────────────────────┤
│ Chat     │                                                    │
│ Serve    │          panel outlet (hash-routed)                │
│ Models   │                                                    │
│ Train    │                                                    │
│ Iterate  │                                                    │
│ Traces   │                                                    │
│ Datasets │                                                    │
│ Hardware │                                                    │
│ Agents   │                                                    │
│ Schedule │                                                    │
│ Settings │                                                    │
└──────────┴────────────────────────────────────────────────────┘
 footer: home path · version · loopback-only · no telemetry
```

Left nav rail (collapsible), header status chips fed by `/api/status` + SSE,
`<main>` outlet per panel. Detail views (trace detail, model lineage, job logs)
open as split-panes or modal dialogs inside the owning panel — deep-linkable via
hash routes (`#/traces/abc123`, `#/models/v2`).

### 3.2 State management

`store.js`: one plain-object app store `{status, models, jobs, ...}` + pub/sub;
panels subscribe to slices. Server is the source of truth — client cache is
per-panel and refetched on SSE invalidation (`job.state` → refetch `/api/jobs`).
No persistence beyond `sessionStorage` (token, last tab, chat draft).

### 3.3 Component inventory → endpoints

| Component | Panel(s) | Backend endpoints |
|---|---|---|
| StatusChips | header | `GET /api/status`, SSE `serve.state` |
| ChatThread + MessageList + ToolCallCard | Chat | `POST /api/chat/stream`, `GET/DELETE /api/chat/threads*` |
| ServeControls (start/stop/attach/auto) | Serve | `POST /api/serve/{start,stop}`, `GET /api/serve/status` |
| ModelTable + LineageGraph + Promote/Rollback buttons | Models | `GET /api/models`, `POST /api/models/{promote,rollback}` |
| JobQueueTable + LogTail (xterm-less ANSI `<pre>`) | Jobs | `GET /api/jobs`, SSE `job.log`, `POST /api/jobs/cancel` |
| TraceTable + TranscriptView + FeedbackWidget | Traces | `GET /api/traces*`, `POST /api/feedback` |
| DatasetBrowser + ManifestView + RowPreview | Datasets | `GET /api/datasets*`, `GET /api/datasets/<id>/rows`, `POST /api/datasets/export` |
| TrainWizard (steps: dataset → kind → params → review) | Train | `POST /api/train`, `GET /api/config/training` |
| IteratePipeline (step timeline) | Iterate | `POST /api/iterate`, SSE `iterate.step` |
| HardwareMonitor (VRAM/util gauges, sparkline) | Hardware | `GET /api/hardware` (+ poll 2 s) |
| AgentProfileList + editor | Agents | `GET/PUT /api/config/profiles` |
| ScheduleView (auto-iterate rules + runs) | Schedule | `GET/POST /api/schedule/rules`, `GET /api/schedule/runs` |
| SettingsEditor (YAML view/edit + schema form) | Settings | `GET/PUT /api/config` |

---

## 4. Panel → backend capability map (what already exists vs new)

Backend capabilities reachable today from Python (verified against
[`cli.py`](../toaster-loop/cli.py), [`app.py`](../toaster-loop/dashboard/app.py)):

- **Already wrapped by dashboard API**: status, traces list/detail, feedback,
  datasets list/show, models + promote/rollback (+lineage edges), jobs list/cancel
  (+log tail), curate, train submit, iterate submit.
- **Exists in Python, needs endpoints**: ~~serve start/stop/attach
  (`ServingSupervisor`, `serve_state.json`)~~ — shipped (Phase 6:
  `/api/servers*`; legacy `/api/serve/start|attach` removed), deploy/hot-swap
  (`serving/deploy.py` — shipped as `/api/serve/deploy`), train auto-check (`cmd_train_auto_check`), dataset row
  preview (read `sft.jsonl`/`dpo.jsonl` slices), export dataset to dir
  (`cmd_datasets_export`), GPU stats (`training/scheduler.gpu_free_mib` +
  extended nvidia-smi query), config read/write (`core/config.py`, YAML round-trip).
- **Needs new backend work**: chat sessions (in-process runner or proxy-only,
  §5), SSE bus, hardware sampler loop, schedule rules engine (thin: config-driven
  timer honoring `train.auto-check` thresholds),
  agent profile records.

---

## 5. Chat & streaming design

Chat in toaster-loop is the **agent playground** — run tasks through the built-in agent
runner (`core/agent_runner.py`), capture traces, and let the user rate results
inline. Two modes:

1. **Raw playground (proxy)** — direct SSE passthrough to the active endpoint's
   OpenAI-compatible `/v1/chat/completions?stream=true`. The dashboard server
   proxies with `httpx.stream(...)`, forwarding chunks as they arrive
   (`text/event-stream`). Trivial, works in attach mode too.
2. **Agent run (recorded)** — `POST /api/chat/stream` runs the task through the
   toaster-loop agent runner (with tool calls), and streams *our own* event protocol:

```
event: message.delta    data: {"text": "..."}          # assistant text tokens
event: tool.call        data: {"id","name","args"}     # tool invocation began
event: tool.result      data: {"id","ok","preview"}    # truncated result (<8 KB)
event: run.done         data: {"trace_id","tokens_in","tokens_out","duration_s"}
event: error            data: {"message"}
```

Agent runs execute **in-process on a daemon thread** owned by the dashboard server:
the run emits events onto the in-memory event bus, which the SSE endpoint relays to
the browser while the same events are teed into the trace recorder. This keeps the
HTTP server responsive (the request handler never blocks on the model) and preserves
the "trace write failure cannot break a conversation" invariant (Phase 2). Runs are
cancellable (`POST /api/chat/cancel`), and in-flight runs survive page reloads —
reconnecting replays buffered events from the run registry.

**Tool-call rendering**: `ToolCallCard` shows name + pretty-printed args,
collapsible result (JSON tree / ANSI-safe `<pre>`), status color, duration.

Inline feedback: every recorded run ends with a FeedbackWidget (`👍/👎/score`,
note) hitting `POST /api/feedback` with the emitted `trace_id` — closing the
loop from chat → buffer in one click.

---

## 6. Security model (v1)

- Bind: `127.0.0.1` hard-coded; IPv6 loopback only if explicitly tested later.
- Headers: CSP `default-src 'self'`, `nosniff`, `no-referrer`, `no-store`;
  `X-Frame-Options: DENY` (blocks drive-by framing of the control UI).
- CSRF: mutating requests must have same-origin `Origin` (or absent, for curl)
  and `Content-Type: application/json`.
- Optional token (§1.4) when `TOASTER_LOOP_DASHBOARD_TOKEN`/`--token` is set.
- Static serving: traversal guard (exists); serve-time MIME allowlist (exists).
- Destructive actions (promote/rollback/delete dataset rows/cancel job/stop
  server) require a modal confirm in UI **and** are single-shot API calls with
  validation — never implicit (ROADMAP Phase 5 acceptance: "UI can never corrupt
  state beyond what the CLI could").
- Body cap 1 MB exists; chat uploads (image playground) deferred, would raise
  it narrowly under `/api/chat/attachments`.

## 7. Testing strategy (headless-first)

- **API**: every endpoint gets `DashboardApp.handle()` tests (the pattern in
  [`test_dashboard.py`](../tests/test_dashboard.py)); SSE endpoints tested via a
  generator-capture helper (`app.events()` iterator + bus emit assertions).
- **Streaming proxy**: reuse [`tests/fake_server.py`](../tests/fake_server.py) —
  it already fakes an OpenAI-compatible server; add a chunked SSE route.
- **Server hygiene**: loopback bind, CSP/no-external-URLs, token/Origin checks.
- **Manual UI checklist** per panel lives in [`UI-ROADMAP.md`](UI-ROADMAP.md);
  no browser automation (Playwright) in v1 — it would add the largest dev-dep in
  the project for marginal safety on a loopback tool.

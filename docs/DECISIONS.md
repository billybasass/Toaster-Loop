# DECISIONS — toaster-loop

Numbered Architecture Decision Records (ADRs). Status: accepted unless noted.

## D-1 — Standalone package, process boundaries to everything else
**Decision.** toaster-loop is one self-contained Python package. Anything it does not
implement itself (inference servers, external OpenAI-compatible endpoints) runs as a
separate process reached over HTTP or subprocess boundaries; nothing is vendored,
forked, or imported from other agent/trainer programs.
**Rationale.** A single importable dependency surface keeps the tool installable
anywhere and reviewable; external programs upgrade independently of toaster_loop.
**Consequence.** toaster-loop maintains its own thin seams: OpenAI HTTP (inference),
subprocess config/files (training/export), and a fully built-in agent runtime.

## D-2 — (superseded by D-12) External agent-loop reuse
**Status: SUPERSEDED.** An early design planned to reuse another program's agent loop
via import. See D-12 for what shipped instead; the reasoning against opaque-subprocess
reuse (in-process hooks for trace capture and hot-swap) still holds and shaped the
built-in runner's design.

## D-3 — Licensing posture: nothing external is imported
**Decision.** The dependency boundary is binding on design and enforced by CI
(`tests/test_hygiene.py`): toaster-loop imports only `pyyaml`, `httpx`, the stdlib,
and — lazily, inside the training worker only — the open ML stack
(torch/transformers/peft/trl/datasets). No agent framework, no trainer CLI library, no
external "studio"/hub packages. Interactions with other local programs are HTTP-only
(explicit attach targets) or subprocess-only (llama.cpp conversion scripts).
**Consequence.** The license-hygiene test is an import allowlist: any new third-party
import must be a deliberate, reviewed change to that test.

## D-4 — Primary serving path: supervisor-owned llama-server on GGUF
**Decision.** The default inference backend is a llama.cpp `llama-server` process owned
by toaster-loop, serving the active GGUF on a loopback dynamic port with an OpenAI-compatible
API. Any other OpenAI-compatible server is an optional, explicit attach target.
**Rationale.** Zero auth friction, trivial hot-swap (kill + respawn with new `-m`), and
immunity to confusion with servers other programs may run — toaster-loop never mutates or
depends on them, only kills processes it spawned itself.
**Trade-off.** Loses third-party model-manager conveniences; acceptable because
toaster-loop's registry replaces that role.

## D-5 — Training via generated YAML + subprocess worker
**Amended.** The original plan had two backends (an external CLI for SFT, a direct
worker for preference training). As built, *all* kinds (SFT/DPO/GRPO) run in the
toaster-loop-owned worker subprocess (`python -m toaster_loop.training.worker`) on
transformers/peft/trl. The YAML contract decision stands: the worker executes an
documented trainer config schema, mirrored as a file contract — nothing imported.
**Decision.** Scheduler emits a `train`-compatible YAML (model/data/training/lora/
logging) and runs training in a toaster-loop-owned worker subprocess importing the open ML
stack directly.
**Rationale.** Config-file contract is stable and `--dry-run` testable; progress via
trainer output files; preference data needs custom code anyway.
**Consequence.** One backend to maintain; job record/progress contract lives in
`train_runs/`.

## D-6 — Trace format = ShareGPT conversations + envelope metadata
**Decision.** Experience-buffer traces use ShareGPT-shaped `conversations`
(`from`/`value`) entries wrapped in an envelope `{trace_id, model_version, outcome,
tool_stats, tokens, ts}`. On disk: append-only JSONL + SQLite index. Tool calls render
as `< tool_call>` / `< tool_response>` blocks inside `gpt` turns.
**Rationale.** The ShareGPT shape is what mainstream trainers accept natively
(`format_type: sharegpt`) → near-zero conversion; HF `datasets` loads it directly.
**Alternative rejected.** OpenAI raw-HTTP-wire logging — noisier, template-coupled,
harder to feed trainers.

## D-7 — Adapter-first, GGUF-for-serving artifact flow
**Decision.** Train LoRA adapters (archival truth = PEFT dir); always produce a GGUF
(default q4_k_m) for serving via worker merge + llama.cpp convert/quantize; registry
stores both and the lineage chain. Full fine-tunes deferred.
**Rationale.** LoRA = fast local training, cheap rollback, composable lineage; GGUF is
what llama-server serves; merge+quant in one pipeline step.
**Trade-off.** Merge+quant per iteration costs time/disk; accepted for hot-swap
simplicity (no runtime adapter-merging support in the llama-server path).

## D-8 — Validation-gated promotion with golden tasks
**Decision.** New adapters never auto-serve to the user. A golden-task replay suite
(fixed prompts + rubric from prior high-score traces) runs against candidate vs active;
promotion requires non-regression; rollback is one command and automatic on swap failure.
**Rationale.** Weight updates are the risky edge of the loop; local models regress
subtly (tool-call syntax breakage especially).
**Consequence.** The replay harness ships before auto-training; golden-task curation
(`toaster-loop mine-goldens`) is its own mini-workstream.

## D-9 — Single-machine, single-GPU arbiter in toaster-loop-core
**Decision.** One scheduler owns GPU state: inference server + training jobs are
serialized unless headroom is proven; agent traffic preempts (training checkpoints,
server reloads after).
**Rationale.** The target machine is one Linux desktop/laptop; co-tenancy OOM is the
most likely failure mode of the whole concept.

## D-10 — Storage: SQLite + files under ~/.toaster-loop
**Decision.** No server-grade infra (no Postgres/object store). SQLite for index/registry,
JSONL for traces/datasets, directories for adapters/GGUFs. Deleting `~/.toaster-loop` is a
full reset.
**Rationale.** Single-user local tool; portability; debuggability (user can grep their
own experience buffer).

## D-11 — Delegation executes synchronously with a bounded timeout (not async enqueue)
**Status: AMENDED by D-17.** The synchronous wait is no longer the default:
`delegate_task` now dispatches asynchronously and delivers results as inbox turns.
The sync-with-timeout execution survives only behind `wait=true`. See D-17 for the
reversal and its rationale; the original text below is kept unedited for history.

**Decision.** `delegate_task` runs the child agent on a worker thread and waits for
it in-line, capped by `delegation.timeout_seconds` (default 300; per-call override).
On timeout it returns a structured payload (`status: timed_out`, partial output slot,
diagnostics) and abandons the child thread — never hangs the parent loop.
`check_delegation` polls the in-process delegation registry for terminal state.
**Rationale.** toaster-loop's tool loop is synchronous by construction
(`content = impl(**args)` inside `ToasterLoopAgentClient.run()`), so an async
enqueue/poll model would add a background scheduler, persistence and re-entry
machinery without changing what the parent can do while waiting — it is blocked
in the tool call either way. The sync-with-timeout shape keeps one execution
path (`create_runner` → guardrails/compaction/recording identical to normal
runs) and still yields a delegation id so lineage, bus events
(`delegation.update`) and `check_delegation` work.
**Consequences.** Parent trace sees only the delegation call + payload (child
intermediates live in the child's own trace). Parent linkage in the child
trace header is by `parent_run_id` (the parent's trace id does not exist until
its run finishes); `thread` lineage entries carry `delegation_id`, `agent`,
`status`, `created`. Depth is capped at 1 (children cannot delegate).

## D-12 — Built-in everything; adapt designs, import no frameworks
**Decision.** All core capabilities are implemented in-tree: the agent runtime
(`core/agent_runner.py`: chat loop, native tool calling, compaction, guardrails,
recording), the toolset (`core/tools.py`), curation, scheduling, registry, serving
supervision and the dashboard. Where prior art existed, designs were adapted in spirit
with the mechanism re-implemented for toaster-loop's constraints; no external agent or
trainer framework is imported at runtime.
**Rationale.** A minimal dependency surface (`pyyaml` + `httpx` + `Pillow` at runtime
— Pillow is the native screen-capture layer for `take_screenshot` on
Windows/macOS/X11, imported lazily so it never loads for text-only work) keeps the
loopback tool installable anywhere, removes version coupling to upstreams with their
own release cadences, and makes trace-capture/hot-swap hooks first-class instead of
adapter-shimmed. Hardening that mattered for a local-only endpoint (timeouts, infra
failure classification, context compression) was re-implemented where it pays, rather
than inherited wholesale with machinery toaster-loop never uses.
**Consequences.** `tests/test_hygiene.py` enforces the import allowlist.
Behavioral compatibility is defined by formats and payload shapes (ShareGPT trajectory,
OpenAI-compatible HTTP), not shared code. D-1's process-boundary rules survive:
external programs remain HTTP/subprocess-only.

## D-13 — The lora-state-dict name→tensor map is the frozen adapter contract
**Decision.** The streamed backend persists adapters as a plain
module-name → LoRA-tensor state dict (`lora_state.safetensors` + an
`adapter_config.json` carrying `"backend": "streamed"` and
`"format": "toaster-loop-lora-state-dict"`), and the GGUF-LoRA writer consumes ONLY
that contract shape. The name mapping is checkpoint naming
(`model.layers.<i>.<module>.lora_A|B`) → llama.cpp LoRA naming
(`blk.<i>.<role>.weight.lora_a|_b`), F32, complete pairs enforced.
**Rationale.** One frozen interchange between trainer, converter and hot-swap
means each side evolves independently; it reproduces exactly what the frozen
base checkpoint uses for names, so an arch adapter is a small role map rather
than framework coupling (D-12). Half pairs and unmapped modules are rejected
at conversion time instead of emitting adapters llama.cpp silently ignores.
**Consequences.** Once any streamed adapter exists on disk the key layout is
stone-age stable — changes require a format-version bump and a migration.
Verified here: writer output loads under a real llama.cpp `--lora` against a
tiny locally-built base GGUF (output measurably changes with the adapter);
real-model-size loads remain GPU/production-hardware acceptance. Registry
provenance records `backend: streamed` additively; peft-run provenance stays
byte-identical.

## D-14 — Streamed training pins one reference arch family first; new archs are explicit registry entries
**Decision.** WP10 ships the streamed backend against pinned llama-family
(llama/qwen3-shaped) checkpoints only: the HF glue and the GGUF-LoRA role map
are per-arch registry entries, and unmapped modules raise rather than guess.
**Rationale.** The spike identified in-tree arch glue as the dominant risk;
one reference arch keeps that risk bounded and testable hermetically (tiny
synthetic checkpoints written in-test, no downloads). An adapter that silently
maps a wrong tensor name produces a corrupt-but-loadable artifact — the worst
failure mode for a hot-swapped adapter.
**Consequences.** Supporting another arch = registering its module→role map
(`streamed/gguf_lora.KNOWN_ADAPTERS` + the shell builder's config reader) and
adding a synthetic-checkpoint test; until then `training.backend: streamed`
fails fast with an actionable error. Packed-expert (MoE) LoRA stays explicitly
out of scope (roadmap note; airllm's rationale: 512-expert adapters + Adam
states don't fit small VRAM).

## D-15 — Tiny-golden convention for the CPU golden-gate acceptance proof
**Decision.** WP10's golden-task gate acceptance on CPU uses a MINIMAL
DETERMINISTIC golden derived at runtime from the measured adapter effect on
greedy output: serve the tiny base with and without the adapter, take a
substring present in exactly ONE of the two temperature-0 continuations
(`expect_contains` when the adapted reply adds one, else `expect_not_contains`
on the base-only continuation), and require the positive run to pass the REAL
`validate_model` gate while the negative control (same golden, no adapter)
FAILS. The proof runs entirely through production code paths — streamed worker,
`run_export` gguf-lora branch, `ServingSupervisor` spawn (adapter via
`extra_llama_args ["--lora", …]`, no serving-code change), golden replay, and
registry register→set_validation→promote.
**Rationale.** A tiny random-init model cannot pass semantic goldens; a
hand-fixed expected string would be a tautology. Deriving the tripwire from
the observed adapter delta plus the mandatory negative control keeps the
assertion honest (it fails loudly if the adapter stops mattering) while still
exercising 100% of the gate code end-to-end — exactly what the acceptance
criterion asks; quality measurement stays where it belongs, on real models
under `bench --noise-floor` rules.
**Consequences.** The convention is valid for ACCEPTANCE plumbing proofs only;
it must never be cited as evidence of model quality, and mined/semantic
goldens remain the production gate content. Determinism relies on greedy
decoding on the same machine + llama.cpp build (the test pins discovery to an
existing binary and skips rather than building). Evidence:
`tests/test_training.py` (~5 s) and
`experiments/wp10_golden_gate/run_gate.py` (prints the full evidence JSON;
repeats verified identical).

## D-16 — Delegation to remote/peer agents: local loop, remote inference
**Decision.** `delegate_task` targeting a peer agent (`tim@192.168.1.78`) runs
the child agent loop **on this machine** — local workspace, local tools, local
guardrails, local trace — with inference pointed at the peer's served model
through its dashboard's `/remote/v1` passthrough, exactly like chat does today
(Option A). Option B (executing the subtask on the peer) is rejected.
Resolution falls back to `peers.resolve_remote_agent` when the name misses the
local profile list; before spawning the child, a healthcheck gate against
`{peer_url}/remote/v1` fails fast with a `peer_unreachable` error payload —
never an exception into the tool loop. The peer's persona travels: probe
snapshots now carry each agent's `system_prompt`, so both chat and delegated
children run the peer's actual persona instead of today's empty string. After
a remote-inference child settles, its transcript is pushed to the peer via
`/remote/v1/traces/import` — delegation gets parity with chat's
`_push_remote_trace`. A companion fix (also closing the pre-existing chat gap)
carries `resolve_remote_agent`'s `remote_instance` as an `X-Toaster Loop-Instance`
header so multi-instance peers stop landing everything on the first instance.
**Rationale.** Chat already established the semantics: "a remote agent is a
local agent whose completions cross the wire" (plans/remote-agents.md);
delegation diverging from chat would make `tim@host` mean one thing in the
dropdown and another in the roster. Option B means executing this machine's
tool calls (`run_command`, file writes) on *another* machine — remote code
execution as a feature — against a gateway surface that is deliberately
inference + trace-import only, and against the local-first capability gating
(D-12). It would also need machinery that exists nowhere: an authenticated
peer-side run API with lifecycle/status, reverse trace flow-back, and a
network cancel protocol. Meanwhile Option A keeps both just-landed fixes
intact for free: §3 trace stamping works because the child trace is recorded
locally (stamped `tim@host`, flowing back to the peer like any remote chat),
and §2 cooperative cancel works because the child loop is a local thread —
the `threading.Event` still stops it at the next loop boundary; only one
in-flight completion is ever in flight over the network, bounded by the
runner's httpx timeout. Effort collapses to wiring + resolution, reusing the
chat path's own pieces.
**Consequences.**
- `execute_delegation` resolves `<agent>@<host>` via a lazy `dashboard.peers`
  import (sanctioned: peers.json lives under TOASTER_LOOP_HOME and is pure state;
  guarded by `try/except ImportError` so standalone core keeps working).
  Unknown-with-`@` names get an error naming the host and pointing at the
  Serve page, distinct from the plain "unknown agent" message.
- Child wiring for remote profiles ignores the parent's `base_url`: it uses
  `peer_url + "/remote/v1"`, `api_key = peer_token`, `model =` the peer's
  served model — mirroring chat.py's remote branch, including its honest
  not-serving message. The healthcheck runs *before* the registry record is
  created, so an unreachable peer leaves no phantom lineage entry.
- Persona travel changes chat behavior too (intentional): `resolve_remote_agent`
  returns the snapshot prompt and chat picks it up via its existing
  `(prof or {}).get("system_prompt")`. Peer *tool grants* still do not
  travel — the child runs under this machine's tool list and guardrails, the
  same rule peers.py already documents for chat.
- Traces: local attribution stays `tim@host` (the §3 stamp path is unchanged;
  easy-training scoping only picks it up if that exact name is configured).
  Flow-back reuses the peer's existing `/remote/v1/traces/import`, mapping
  `tim@host` → `tim` server-side like chat already does — best-effort, a dead
  peer never breaks the local child.
- Cancel semantics: no new mechanism. Parent Stop flips the same cooperative
  event (§2); worst-case stop latency is one in-flight remote completion plus
  httpx read timeout. A peer dying mid-run surfaces as the runner's existing
  `AgentBackendError` → structured failed payload.
- Naive `?instance=` on the base URL is explicitly rejected: the runner builds
  `{base_url}/chat/completions`, so a query string would land mid-path. The
  instance selector rides a header instead (`X-Toaster Loop-Instance`), honored by
  `remote_gateway.resolve_upstream` ahead of model-name matching.
Implementation spec: `plans/delegation-remote.md` (appendix to this decision).

## D-17 — Async delegation inbox: dispatch-and-wake replaces the synchronous wait
**Amends D-11** (audit §4). **Decision.** `delegate_task` becomes an async
dispatch: it registers a durable delegation record and returns immediately with
`{status: "dispatched", delegation_id}`. Children execute on a bounded,
process-wide supervised dispatcher pool (one shared pool replaces the per-call
`ThreadPoolExecutor`). When a child settles, its result is delivered to the
parent's **thread inbox** — a durable per-thread queue persisted in the
delegation DB — and a dispatcher watcher **wakes the thread with a fresh run**:
the inbox turn is posted as a synthetic user-role message ("Delegation results
have arrived…") through the same chat-run machinery a real user message uses.
This is the *wake-on-arrival* model; the alternative (keep the parent loop alive
blocking on a queue) was rejected. The synchronous D-11 behavior survives behind
an explicit `wait: true` parameter on `delegate_task`, defaulting to the new
async semantics (`wait: false`). Completion deadlines are replaced by
**liveness**: a child must register and keep heartbeating (`last_activity_ts`)
within `delegation.liveness_timeout_seconds`; a stale child is cooperatively
cancelled and settled `wedged` (new terminal status) — a wedged-child detector,
not a wall-clock kill of healthy-but-slow work. The in-process registry becomes
a cache over a persisted store (`<TOASTER_LOOP_HOME>/delegations/delegations.db`,
SQLite, schema v11 of the shared `store/db.py` conventions), so delegation state
and undelivered inbox turns survive dashboard restarts; on startup, `running`
records owned by the dead process reconcile to `interrupted` (never requeued —
a child run cannot be resumed mid-loop).
**Rationale.** D-11's premise — "the parent is blocked in the tool call either
way, so async adds nothing" — stopped being true once delegation became a
multi-child, long-running feature: a synchronous parent holds the thread's
single-flight slot hostage for the whole child lifetime (a second user message
409s behind a 5-minute child), one wedged inference pins the parent for its
full `timeout_seconds`, and every D-16 remote-agent child inherits the worst
case. Wake-on-arrival composes with every piece of existing plumbing instead
of fighting it: each turn remains one `runner.run()` = one trace (the audit's
own caveat about "one giant trace" dissolves), the per-thread single-flight
rule keeps its exact meaning (a wake run and a user run still cannot overlap;
an arrival during an in-flight parent run simply queues — no new concurrency
state), cancel semantics are unchanged (Stop cancels the current run *and*
cooperatively cancels its dispatched children via the existing
`cancel_event` propagation), SSE/bus events need no new protocol (`run.started`
/ `delegation.update` already exist; the wake run is indistinguishable from a
user turn to the stream layer), and thread-history persistence already runs at
end-of-run in the chat worker — the inbox drain lands in that same seam. The
keep-alive alternative would require the parent loop to block across tool
iterations (new runner hook, cancellation during queue-wait, trace bloat, SSE
heartbeating through idle periods, and a redefinition of single-flight), i.e.
it changes more contracts for strictly less durability. Bounded dispatcher
(default 2 workers, FIFO queue) preserves the single-GPU arbiter posture of
D-9: delegations fan out but never oversubscribe inference beyond a small
configured window, and queue-full applies backpressure as a clean tool-error
payload rather than dropping work.
**Consequences.**
- Status vocabulary gains `dispatched` (queued, not yet started), `wedged`
  (liveness deadline missed; cooperative cancel attempted) and `interrupted`
  (process died with the child in flight). `running/completed/failed/
  timed_out/cancelling/cancelled` keep their D-11 meanings; `timed_out` only
  occurs on the legacy `wait=true` path.
- `check_delegation` becomes genuine polling against the persisted registry —
  it now answers across processes and restarts, and its payload gains
  `queued_ahead` (dispatcher position) so a model can decide whether to wait
  or continue working.
- A parent thread with pending dispatched delegations shows them in lineage
  (existing `thread.delegations` entries) and the wake run's user message is
  rendered as an ordinary turn; no new UI protocol is required for Phase 1.
- Restart recovery: `interrupted` records are terminal and *not* requeued
  (re-running a child blindly could repeat side effects); their inbox entry is
  written with the interruption so the next wake tells the parent honestly.
  Inbox entries survive; undelivered turns are delivered on the next wake
  opportunity, batched with any newer arrivals.
- `wait: true` keeps the whole D-11 code path (bounded poll loop around
  `future.result`) for deterministic callers — `plan_loop` iterates
  deterministically and its spec'd use stays sync; MCP/CLI direct callers get
  async by default like the model-facing tool.
- The dispatcher lives in the dashboard process (it owns chat-run startup).
  CLI `toaster-loop run` without a dashboard still dispatches (threadless inbox: the
  result is only reachable via `check_delegation`, and an explicit note says
  so) — same degradation rule as thread lineage today.
Implementation spec: `plans/delegation-inbox.md` (phases, state machines,
config keys, recovery table, test plan).

## D-18 — GGUF-as-pointer: resolve the HF base instead of converting weights
**Decision.** A GGUF is never a training input and is never converted back to
HF. When a user has only a GGUF, toaster_loop treats it as a *pointer* to its
canonical HF base: the GGUF KV metadata (`general.basename`,
`general.architecture`) plus hub provenance (the `models--org--name` layout /
download manifests / the GGUF repo card's `base_model` field) resolve to an HF
repo, which is either already on disk or downloaded through the normal
Resource Hub path. New modules: `core/gguf_meta.py` (stdlib GGUF KV reader —
header + KV only, microsecond cost on any quant size) and `core/gguf_base.py`
(local-checkpoint scan → card provenance → hub search, non-GGUF repos only).
API mirrors the mmproj pattern: `GET /api/model_base?gguf=` (status + local +
remote candidates) and `POST /api/model_base/download` (reuses
`ModelHub.start_download`; the checkpoint lands in the hub cache where
discovery picks it up). The Train picker's GGUF entries became actionable
("pick HF base…") instead of dead serve-only rows, and `submit_train` now
rejects a `.gguf` base_model with `gguf_not_trainable` pointing at the resolve
endpoint (defense in depth — previously workers failed deep in load).
**Rationale.** The Unsloth-Studio alternative (GGUF→HF dequantize → train →
re-export) is lossy, per-arch third-party territory, and turns every quant
variant into a distinct training identity — polluting the registry lineage
model (`base_model + dataset → vN`), where quant is a deployment choice. A
silently mis-mapped tensor yields a corrupt-but-loadable artifact, the exact
failure mode D-14 was written to prevent. Serving stays llama.cpp/GGUF (D-4):
training was never GGUF's job; running is where GGUF shines. The GGUF KV
metadata makes the mapping *declared by the file itself*, so resolution is a
lookup, not a guess — and an unresolvable GGUF degrades to manual repo-id
entry rather than a silent conversion.
**Consequences.**
- Training truth stays HF safetensors for every path (peft + streamed); the
  D-7 artifact flow (adapter → merge → convert/quantize at promotion) and the
  D-13 GGUF-LoRA bridge are untouched.
- `general.basename` is a display name, not a repo id: resolution folds names
  (strip quant tags/separators) and *ranks candidates*; it never auto-picks a
  download. The user confirms the base.
- The reader parses v1–v3 headers best-effort and keeps partially-parsed KV
  (`general.basename` sits early), so truncated/corrupt headers degrade to
  "no metadata" exactly like an unattested provenance row — never a crash.
- Known follow-up (not in this change): record the resolved `base_hf_repo` on
  the registry row's `provenance_json` at train time, so lineage names the HF
  source of any model trained from a GGUF-derived setup.

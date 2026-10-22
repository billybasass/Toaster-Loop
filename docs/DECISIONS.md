# DECISIONS — sable

Numbered Architecture Decision Records (ADRs). Status: accepted unless noted.

## D-1 — Standalone package, process boundaries to everything else
**Decision.** sable is one self-contained Python package. Anything it does not
implement itself (inference servers, external OpenAI-compatible endpoints) runs as a
separate process reached over HTTP or subprocess boundaries; nothing is vendored,
forked, or imported from other agent/trainer programs.
**Rationale.** A single importable dependency surface keeps the tool installable
anywhere and reviewable; external programs upgrade independently of sable.
**Consequence.** sable maintains its own thin seams: OpenAI HTTP (inference),
subprocess config/files (training/export), and a fully built-in agent runtime.

## D-2 — (superseded by D-12) External agent-loop reuse
**Status: SUPERSEDED.** An early design planned to reuse another program's agent loop
via import. See D-12 for what shipped instead; the reasoning against opaque-subprocess
reuse (in-process hooks for trace capture and hot-swap) still holds and shaped the
built-in runner's design.

## D-3 — Licensing posture: nothing external is imported
**Decision.** The dependency boundary is binding on design and enforced by CI
(`tests/test_license_hygiene.py`): sable imports only `pyyaml`, `httpx`, the stdlib,
and — lazily, inside the training worker only — the open ML stack
(torch/transformers/peft/trl/datasets). No agent framework, no trainer CLI library, no
external "studio"/hub packages. Interactions with other local programs are HTTP-only
(explicit attach targets) or subprocess-only (llama.cpp conversion scripts).
**Consequence.** The license-hygiene test is an import allowlist: any new third-party
import must be a deliberate, reviewed change to that test.

## D-4 — Primary serving path: supervisor-owned llama-server on GGUF
**Decision.** The default inference backend is a llama.cpp `llama-server` process owned
by sable, serving the active GGUF on a loopback dynamic port with an OpenAI-compatible
API. Any other OpenAI-compatible server is an optional, explicit attach target.
**Rationale.** Zero auth friction, trivial hot-swap (kill + respawn with new `-m`), and
immunity to confusion with servers other programs may run — sable never mutates or
depends on them, only kills processes it spawned itself.
**Trade-off.** Loses third-party model-manager conveniences; acceptable because
sable's registry replaces that role.

## D-5 — Training via generated YAML + subprocess worker
**Amended.** The original plan had two backends (an external CLI for SFT, a direct
worker for preference training). As built, *all* kinds (SFT/DPO/GRPO) run in the
sable-owned worker subprocess (`python -m sable.training.worker`) on
transformers/peft/trl. The YAML contract decision stands: the worker executes an
documented trainer config schema, mirrored as a file contract — nothing imported.
**Decision.** Scheduler emits a `train`-compatible YAML (model/data/training/lora/
logging) and runs training in a sable-owned worker subprocess importing the open ML
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
(`sable mine-goldens`) is its own mini-workstream.

## D-9 — Single-machine, single-GPU arbiter in sable-core
**Decision.** One scheduler owns GPU state: inference server + training jobs are
serialized unless headroom is proven; agent traffic preempts (training checkpoints,
server reloads after).
**Rationale.** The target machine is one Linux desktop/laptop; co-tenancy OOM is the
most likely failure mode of the whole concept.

## D-10 — Storage: SQLite + files under ~/.sable
**Decision.** No server-grade infra (no Postgres/object store). SQLite for index/registry,
JSONL for traces/datasets, directories for adapters/GGUFs. Deleting `~/.sable` is a
full reset.
**Rationale.** Single-user local tool; portability; debuggability (user can grep their
own experience buffer).

## D-11 — Delegation executes synchronously with a bounded timeout (not async enqueue)
**Decision.** `delegate_task` runs the child agent on a worker thread and waits for
it in-line, capped by `delegation.timeout_seconds` (default 300; per-call override).
On timeout it returns a structured payload (`status: timed_out`, partial output slot,
diagnostics) and abandons the child thread — never hangs the parent loop.
`check_delegation` polls the in-process delegation registry for terminal state.
**Rationale.** sable's tool loop is synchronous by construction
(`content = impl(**args)` inside `SableAgentClient.run()`), so an async
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
with the mechanism re-implemented for sable's constraints; no external agent or
trainer framework is imported at runtime.
**Rationale.** A minimal dependency surface (`pyyaml` + `httpx` + `Pillow` at runtime
— Pillow is the native screen-capture layer for `take_screenshot` on
Windows/macOS/X11, imported lazily so it never loads for text-only work) keeps the
loopback tool installable anywhere, removes version coupling to upstreams with their
own release cadences, and makes trace-capture/hot-swap hooks first-class instead of
adapter-shimmed. Hardening that mattered for a local-only endpoint (timeouts, infra
failure classification, context compression) was re-implemented where it pays, rather
than inherited wholesale with machinery sable never uses.
**Consequences.** `tests/test_license_hygiene.py` enforces the import allowlist.
Behavioral compatibility is defined by formats and payload shapes (ShareGPT trajectory,
OpenAI-compatible HTTP), not shared code. D-1's process-boundary rules survive:
external programs remain HTTP/subprocess-only.

## D-13 — The lora-state-dict name→tensor map is the frozen adapter contract
**Decision.** The streamed backend persists adapters as a plain
module-name → LoRA-tensor state dict (`lora_state.safetensors` + an
`adapter_config.json` carrying `"backend": "streamed"` and
`"format": "sable-lora-state-dict"`), and the GGUF-LoRA writer consumes ONLY
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
`tests/test_golden_gate_wp10.py` (~5 s) and
`experiments/wp10_golden_gate/run_gate.py` (prints the full evidence JSON;
repeats verified identical).

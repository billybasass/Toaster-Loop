# ARCHITECTURE — toaster-loop

**toaster-loop** is a local, self-improving agent system: a built-in agent runtime runs
against locally served models; its task traces flow into an experience buffer; a curater
turns high-value traces into SFT/DPO datasets; LoRA fine-tuning (transformers + peft +
TRL) trains on that data; the new weights are exported to GGUF and hot-swapped back into
the agent runtime. Everything runs on one Linux machine (Omarchy/Hyprland), no cloud,
zero API keys.

> **Status note.** This document describes the system as it exists today. Everything is
> built into toaster-loop: no external agent framework, trainer library, or model-hub SDK is
> imported at runtime. Where prior art existed, designs were adapted in spirit and
> re-implemented (see [`DECISIONS.md`](DECISIONS.md) D-12); the import allowlist is
> enforced by `tests/test_hygiene.py`.

## 1. Component diagram

```mermaid
flowchart TB
    subgraph toaster-loop [toaster-loop - one Python package]
        CLI[CLI<br/>toaster-loop/cli.py]
        RUN[Agent runtime<br/>core/agent_runner.py + core/tools.py]
        BUF[Experience Buffer<br/>SQLite index + JSONL traces, store/]
        CUR[Dataset Curater<br/>curate/ scoring, dedup, ShareGPT + DPO pairs]
        TSCHED[Training Scheduler<br/>training/scheduler.py FIFO single-GPU]
        REG[Model Registry<br/>registry/ lineage, tiers, ledger, janitor]
        SRV[Serving Supervisor<br/>serving/ supervisor + N instances + router]
        WD[Engine Watchdog<br/>serving/watchdog.py opt-in /proc freeze detection<br/>verified kill + backoff restart]
        COEX[Coexistence Engine<br/>serving/coexist.py opt-in idle-unload + lazy-load<br/>train+serve VRAM handshake]
        DASH[Dashboard<br/>dashboard/ stdlib HTTP + SSE + vanilla JS SPA<br/>GET /metrics Prometheus exposition - alias of /api/metrics]
        ANALYTICS[Analytics store<br/>store/analytics.py append-only JSONL<br/>curation funnels + validation telemetry]
    end

    subgraph external [External processes - process boundary only]
        LLAMA[llama-server GGUF<br/>OpenAI-compatible /v1 - MIT]
        USTUDIO[external OpenAI-compatible server<br/>optional attach target, HTTP only]
        TRAINW[Training Worker subprocess<br/>python -m toaster_loop.training.worker<br/>torch/transformers/peft/trl]
    end

    CLI --> RUN & BUF & CUR & TSCHED & REG & SRV & DASH
    DASH -- same core API objects --> BUF & CUR & TSCHED & REG & SRV
    RUN -- OpenAI tool_calls over HTTP --> LLAMA
    RUN -. optional attach .-> USTUDIO
    RUN -- append traces --> BUF
    BUF -- query high-value traces --> CUR
    CUR -- sft.jsonl / dpo.jsonl + train.yaml --> TSCHED
    CUR -- append funnel snapshot --> ANALYTICS
    TSCHED -- append validation-gate telemetry --> ANALYTICS
    DASH -- GET /api/analytics/* + /metrics --> ANALYTICS
    TSCHED -- job config --> TRAINW
    TRAINW -- adapter + GGUF --> REG
    REG -- active pointer / deploy --> SRV
    WD -. samples /proc, kills only owned pids .-> LLAMA
    SRV -. boot hook serve.watchdog_enabled .-> WD
    WD -- serve.watchdog events --> DASH
    COEX -. idle-unload + lazy reload, owned-stop paths only .-> LLAMA
    TSCHED -. VRAM gate rescue (coexist_policy) .-> COEX
    COEX -- serve.coexist events --> DASH
```

## 2. The train → deploy → iterate loop (data flow)

1. **Act.** The built-in runner (`core/agent_runner.py`) drives an OpenAI-compatible chat
   loop with native `tool_calls` against the active endpoint; `core/tool_calls.py`
   recovers text-emitted calls from GGUFs with flaky function calling. The toolset lives
   in `core/tools.py`: file tools (`read_file`, `list_dir`, `write_file`, `patch`,
   `search_files`), planning (`todo_write`/`todo_read`), retrieval (`search_codebase`,
   `search_traces`), delegation (`list_agents`, `delegate_task`, `check_delegation`),
   skills (opt-in), and capability-gated reach-out tools (`web_fetch`, `run_command`,
   `view_image`, `take_screenshot` — all off by default, grantable per profile). Long
   threads are auto-compacted (`core/compression.py`); oversized tool results spill to
   disk behind stubs (`core/spill.py`).
2. **Record.** Every run is captured post-completion at one canonical point into the
   experience buffer: append-only JSONL (header/messages/footer, `buffer/YYYY/MM/`) plus a
   SQLite index row (`buffer/toaster_loop.db`, WAL), with model alias, endpoint label, token
   counts, duration, heuristic outcome score, and profile/project tags. Secrets are
   redacted before the durable copy (`core/redact.py`). Recording is best-effort: a trace
   failure can never break a conversation.
3. **Score & curate.** The curater (`curate/curater.py`) filters by effective score
   (user feedback > judge > heuristic), dedups prefix-overlapping REPL conversations, and
   emits ShareGPT `sft.jsonl` (tool-call-block trajectory rendering) plus TRL-schema
   `dpo.jsonl`, a `manifest.json` and a documented-contract `train.yaml`. Selected traces
   are claimed so each appears in at most one dataset. Optional LLM-as-judge scores
   borderline traces; external datasets import via `curate/importer.py`; the ingestion
   quality gate (`curate/ingest_gate.py`) flags duplicates/conflicts.
4. **Train.** The scheduler (`training/scheduler.py`) is a FIFO single-GPU arbiter: it
   resource-gates (free VRAM, disk), spawns `python -m toaster_loop.training.worker` per job
   (SFT via transformers `SFTTrainer` + peft LoRA; DPO and GRPO via TRL), tails the worker
   log, reaps crashed workers, and supports cancel (SIGTERM), resume-from-checkpoint, and
   job dependencies. `training/easy.py` layers a one-profile turnkey preset on top.
   The resolved train.yaml carries `training.backend` (`peft` default | `streamed`,
   WP10): the worker — not the scheduler — selects the SFT implementation; the
   streamed branch streams checkpoint layers from disk and emits a GGUF-LoRA
   artifact instead of a peft adapter. Spawn path identical either way.
5. **Export & register.** The worker merges the adapter (peft) and exports GGUF via
   llama.cpp `convert_hf_to_gguf.py` + `llama-quantize` (or AWQ/GPTQ HF checkpoints).
   The registry records lineage: `base_model + parent_adapter + dataset_id +
   manifest_hash → vN`, plus derived evidence tiers T0–T3 (`registry/tiers.py`) and a
   SHA-256 hash-chained operation ledger (`registry/ledger.py`).
6. **Swap.** The golden-task validation gate (substring checks + tool-call grammar
   checks, `training/validate.py`) replays before promotion; passing candidates are
   promoted (previous superseded), failures keep the previous model live, and
   `toaster-loop models rollback` reverts one step anytime. Serving stops/starts the
   toaster-loop-owned llama-server on the new GGUF (`serving/deploy.py`); post-train shadow
   comparison (`evaluation/shadow.py`) and cross-profile transfer probes
   (`evaluation/evolve.py`-adjacent) watch for regressions warn-only.

## 3. Process topology

| Process | Contents | Lifetime |
|---|---|---|
| **toaster-loop CLI / dashboard** (Python) | runner, buffer, curater, scheduler, registry, serving supervisor; the dashboard runs long actions on daemon threads. The opt-in engine watchdog (`serving/watchdog.py`, `serve.watchdog_enabled`) adds one daemon loop: samples `/proc/<pid>/stat` + `/proc/<pid>/io` of managed llama-server children, flags a freeze only when a request is in flight with zero CPU-time AND zero IO growth over the window, re-verifies pid ownership before SIGTERM/SIGKILL, and restarts with exponential backoff under a per-window cap (`serve.watchdog` events on the bus). Off by default; non-Linux: never kills. The opt-in coexistence engine (WP5, `serving/coexist.py`) adds a second loop when `serve.idle_unload_seconds > 0`: gracefully unloads instances idle past the window (never mid-request; owned-stop paths only) and records `unloaded` state; `serve.lazy_load: true` reloads transparently on the next request, and with `serve.min_free_vram_mib > 0` the scheduler's VRAM gate consults it for the train+serve handshake (`serve.coexist` events). All keys off by default — no loop, no behavior change | per invocation; dashboard long-running |
| **llama-server** (per instance) | GGUF inference, OpenAI-compatible `/v1` on `127.0.0.1:<port>`; N named instances concurrently (Phase 6), optional tier-aware routing (`serving/router.py`) | supervised; only processes toaster-loop spawned are ever terminated (watchdog kills and WP5 idle-unloads go through the same ownership-verified paths); WP5 may stop an idle instance to free VRAM for training and re-spawn it on demand or on explicit `toaster-loop instances load <id>` |
| **training-worker** (Python subprocess) | torch/transformers/peft/trl; SFT/DPO/GRPO; merge + GGUF export; writes adapter + loss metrics. Backend selection (`training.backend`): `peft` (default, established path) or `streamed` (layer-streaming LoRA trainer + inline GGUF-LoRA writer, torch-free converter) | spawned per training job by scheduler |
| **embedding llama-server** | `llama-server --embedding` for codebase indexing (`core/indexer.py`) | spawned/stopped on demand by the indexer |
| **external OpenAI-compatible server** *(optional)* | attach target only, over HTTP `/v1` — never probed automatically, never depended on | user-managed |

Key rules:
- **One GPU arbiter** (the scheduler): training dispatches only when the resource gate
  passes; multi-GPU runs parallelize by pinning workers to device ids.
- Everything binds to loopback only. The only network uses are explicit Hugging Face
  downloads (`toaster-loop hub`, training base models) and explicitly granted `web_fetch`.

## 4. Storage layout

Root = `TOASTER_LOOP_HOME` env > config `home:` > `~/.toaster-loop` (config file:
`--config`/`TOASTER_LOOP_CONFIG`, else `<home>/toaster_loop.yaml` — the file the dashboard writes):

```
~/.toaster-loop/
  toaster_loop.yaml              # config (dashboard Settings writes here)
  buffer/
    toaster_loop.db              # SQLite: traces + feedback + registry + jobs (WAL)
    YYYY/MM/<trace_id>.jsonl# append-only trace (header/message*/footer, schema v1)
  datasets/<dataset_id>/    # sft.jsonl dpo.jsonl manifest.json train.yaml
  analytics/                # WP9 append-only JSONL: curation_funnels.jsonl,
                            # validation_runs.jsonl (one record per run)
  train_runs/               # <job_id>.log worker logs; <model_id>/ artifacts
  models/gguf/              # exported serving GGUFs
  archive/<model_id>/       # archived (recoverable) model artifacts
  index/<project_id>.db     # per-project codebase index (+ default.db)
  projects/<id>.json        # project records (workspace_dir, agents)
  chat/                     # dashboard chat threads (JSON per thread)
  uploads/                  # chat image attachments (janitor-pruned)
  batches/<run-name>.json   # batch-run checkpoints (stats only)
  iterate_runs/iter-*.json  # iterate run records (step timelines)
  easy_training/state.json  # easy-training run state
  skills/<name>/            # SKILL.md + scripts dir (skills.enabled)
  logs/dashboard.log        # launched-dashboard log
  serving/instances/<id>.json + serving/state/<id>.json  # Phase 6 instances
  serve_state.json          # reserved "default" instance state
  janitor_state.json        # disk-janitor runs
  llama.cpp/                # managed llama.cpp build (toaster-loop setup)
```

## 5. Tech choices (as built)

- **Python ≥3.11,<3.14, uv-managed venv** — runtime deps are only `pyyaml` + `httpx`;
  the optional `[training]` extra adds torch/transformers/peft/trl/datasets/accelerate/
  bitsandbytes.
- **Agent runtime**: built in (`core/agent_runner.py`) — no external agent framework.
  No external agent framework is used or vendored.
- **Serving**: llama.cpp `llama-server` for GGUF, auto-built per-GPU by
  `toaster-loop setup` (CUDA/ROCm/Metal/Vulkan/CPU), supervised with health checks and
  dynamic ports; any other OpenAI-compatible server is an optional explicit attach target.
- **Training**: LoRA via transformers `SFTTrainer` + peft; DPO/GRPO via TRL; GGUF export
  via llama.cpp subprocesses. No trainer library is imported — training runs on the
  open ML stack directly (enforced by the import-allowlist hygiene test).
- **Buffer/registry**: single-writer SQLite + JSONL append-only traces; no server-grade
  infra. Deleting `~/.toaster-loop` fully resets toaster_loop.
- **UI (shipped)**: loopback-only stdlib `ThreadingHTTPServer` + transport-agnostic
  `DashboardApp` JSON API + zero-build vanilla-JS SPA, live updates over one SSE channel;
  no external assets, optional token auth. See
  [`UI-ARCHITECTURE.md`](UI-ARCHITECTURE.md) (design) and README's dashboard tour.

## 6. Failure & safety posture

- Registry promotes only after the golden-task validation gate passes; rollback is one
  command and instant. Shadow evaluation flags post-train regressions warn-only.
- Trace recording is best-effort and must never break an agent turn (fault-injection
  tested).
- Training crash → job reaped to `error`, adapter kept, serving untouched; cancel
  SIGTERMs the worker.
- toaster-loop only ever terminates processes it spawned (pid-ownership verified via state
  files); foreign llama-servers and attach targets are never killed.
- Destructive lifecycle is archive-don't-delete (`models archive/restore`, janitor);
  pinned models bypass all automation; disk pruning touches an explicit allow-list only.
- All state under `~/.toaster-loop/`; deleting it fully resets toaster-loop without touching
  anything else on the machine.

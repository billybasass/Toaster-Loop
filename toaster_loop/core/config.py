"""Configuration for toaster_loop: YAML file + environment-variable overrides.

Precedence (highest wins): CLI flags > env vars > YAML file > built-in defaults.

Env vars:
  TOASTER_LOOP_CONFIG        path to YAML config file (when unset, load_config falls
                        back to <TOASTER_LOOP_HOME>/toaster_loop.yaml, the file the
                        dashboard writes settings to)
  TOASTER_LOOP_HOME          state root (default ~/.toaster-loop) — experience buffer + datasets
  TOASTER_LOOP_GGUF          path to GGUF model file
  TOASTER_LOOP_MODELS_DIR    base model folder to scan for models (e.g.
                        ~/.cache/huggingface/hub); see core/model_discovery.py
  TOASTER_LOOP_PORT          llama-server port (0 = dynamic)
  TOASTER_LOOP_CONTEXT       context length passed to llama-server -c
  TOASTER_LOOP_ATTACH_URL    OpenAI-compatible URL to attach to instead of spawning
  TOASTER_LOOP_MODEL         model name reported to / used with the endpoint
  TOASTER_LOOP_RECORDING     "0"/"false" disables trace recording (default: enabled)
  TOASTER_LOOP_GPU_LAYERS    llama-server -ngl offload (unset = auto-detect by GPU)
  TOASTER_LOOP_LLAMA_AUTO_SETUP  "false" disables building llama.cpp automatically
  TOASTER_LOOP_NO_AUTO_SETUP     "1" opts out of llama.cpp auto-setup entirely
  TOASTER_LOOP_EXTERNAL_SERVER_URL
                        OPTIONAL external OpenAI-compatible server base URL
                        for attach-mode discovery. Opt-in only: toaster_loop is
                        standalone and does not probe any default port (never
                        depends on any other local program existing).
  TOASTER_LOOP_CONFIG_STRICT "1"/"true" opts into strict unknown-key validation:
                        typo'd top-level keys become errors instead of the
                        default warning. Mistyped values are ALWAYS errors
                        (consolidated ConfigError report), strict or not.

Validation (WP2a — "refuse to guess", pattern inspiration from kimi-k3-in-c's
config validation; re-implemented inline, no imports per D-12): a config with
problems raises a single ConfigError whose message enumerates EVERY problem
field at once (path, expected kind, found value) instead of failing on the
first one. Found values pass through core.redact before display. Unknown keys
stay warn-only by default so stray keys in existing configs keep loading
byte-identically; set TOASTER_LOOP_CONFIG_STRICT=1 (or load_config(strict_unknown=
True)) to promote them into the same consolidated report.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from difflib import get_close_matches
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("toaster_loop.config")

# Generous default context window for local serving.
DEFAULT_CONTEXT_LENGTH = 65536

# Project root = the directory containing the `toaster_loop/` package and `docs/`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Sentinel stored in `agent["workspace_root"]` when the YAML carries an
#: *explicit* null (backlog #9: unset => cwd default, explicit null =>
#: unrestricted). The generic section loader below drops plain None values,
#: so this sentinel is how load_config preserves that distinction.
#: Consumed by core/agent_runner.resolve_workspace_roots.
UNSET = object()


class ConfigError(ValueError):
    """Invalid configuration: ONE consolidated report of every problem field.

    WP2a ("refuse to guess" — pattern inspiration from kimi-k3-in-c's config
    validation; re-implemented inline, no imports per D-12). Subclasses
    ValueError so existing `except ValueError` call sites keep working. The
    message enumerates ALL unknown / missing / mistyped fields at once (path,
    expected kind, redacted found value) instead of failing on the first one.
    """


# Top-level fields grouped by expected YAML kind (WP2a validator). Kept as
# explicit sets derived from the dataclass annotations; test_config_strictness
# asserts every name here is a real ToasterLoopConfig field so this cannot drift.
_INT_FIELDS = frozenset({"port", "context_length", "recording_retention_days",
                         "gpu_layers", "delegation_max_concurrent",
                         "delegation_queue_max"})
_FLOAT_FIELDS = frozenset({"delegation_wake_retry_seconds"})
_BOOL_FIELDS = frozenset({"recording_enabled", "recording_redact_secrets",
                          "delegation_enabled", "auto_model",
                          "llama_auto_setup", "delegation_async_default"})
_STR_FIELDS = frozenset({
    "gguf_path", "models_dir", "home", "attach_url", "attach_api_key",
    "model_name", "llama_server_binary", "host", "llama_repo", "llama_ref",
    "llama_backend",
})
_DICT_FIELDS = frozenset({
    "curation", "training", "validation", "dashboard", "agent", "tools",
    "janitor", "run_settings", "sampling", "indexing", "evolution",
    "playbooks", "skills", "logging", "rft", "goldens_mine", "easy_training",
    "routing", "serve", "plans", "update",
})
_LIST_FIELDS = frozenset({"extra_llama_args", "profiles"})

#: Top-level keys that MUST be present. Empty today — every knob has a sane
#: default — but the consolidated report supports this category, and future
#: mandatory fields land here (tests exercise the machinery by patching it).
REQUIRED_FIELDS: tuple = ()

_TRUTHY_STRINGS = ("1", "true", "yes", "on")
_FALSY_STRINGS = ("0", "false", "no", "off")


def _fmt_found(v: Any) -> str:
    """Redacted, length-capped rendering of a found value for error text."""
    from .redact import redact_text  # local import: keep module import graph lean

    s = redact_text(repr(v))
    if len(s) > 120:
        s = s[:117] + "..."
    return s


def _kind_for(path: str) -> Optional[str]:
    """Expected YAML kind ("int"/"float"/"bool"/"str"/"dict"/"list") of a
    known top-level field, or None when the loader treats it generically."""
    if path in _INT_FIELDS:
        return "int"
    if path in _FLOAT_FIELDS:
        return "float"
    if path in _BOOL_FIELDS:
        return "bool"
    if path in _STR_FIELDS:
        return "str"
    if path in _DICT_FIELDS:
        return "dict"
    if path in _LIST_FIELDS:
        return "list"
    return None


def _type_problem(path: str, v: Any,
                  kind: Optional[str] = None) -> Optional[str]:
    """One mistype description for a known field value, or None when usable.

    Deliberately permissive where YAML quoting is a common benign variation
    (numeric strings for int fields, "yes"/"off" strings for bool fields):
    those load successfully today, and valid configs must keep loading
    byte-identically. Only genuinely unusable values are reported. `kind`
    overrides the top-level field lookup (nested section keys pass theirs).
    """
    if v is None:  # None => built-in default (dropped by the loader)
        return None
    if kind is None:
        kind = _kind_for(path)
    if kind == "int":
        if isinstance(v, bool):
            return f"expected an integer, found {_fmt_found(v)} (bool)"
        if isinstance(v, int):
            return None
        if isinstance(v, float):
            return None if v.is_integer() else \
                f"expected an integer, found {_fmt_found(v)} (non-integral float)"
        if isinstance(v, str):
            try:
                int(v.strip(), 10)
                return None
            except ValueError:
                return f"expected an integer, found {_fmt_found(v)} (str)"
        return f"expected an integer, found {_fmt_found(v)} " \
               f"({type(v).__name__})"
    if kind == "float":
        if isinstance(v, bool):
            return f"expected a number, found {_fmt_found(v)} (bool)"
        if isinstance(v, (int, float)):
            return None
        if isinstance(v, str):
            try:
                float(v.strip())
                return None
            except ValueError:
                return f"expected a number, found {_fmt_found(v)} (str)"
        return f"expected a number, found {_fmt_found(v)} " \
               f"({type(v).__name__})"
    if kind == "bool":
        if isinstance(v, bool):
            return None
        if isinstance(v, str) and v.strip().lower() in (
                _TRUTHY_STRINGS + _FALSY_STRINGS):
            return None  # "yes"/"off"/... env-style strings are usable
        if isinstance(v, (int, float)) and float(v) in (0.0, 1.0):
            return None
        return f"expected a boolean, found {_fmt_found(v)} " \
               f"({type(v).__name__})"
    if kind == "str":
        if isinstance(v, str):
            return None
        return f"expected a string, found {_fmt_found(v)} " \
               f"({type(v).__name__})"
    if kind == "dict":
        if isinstance(v, dict):
            return None
        return f"expected a mapping, found {_fmt_found(v)} " \
               f"({type(v).__name__})"
    if kind == "list":
        if isinstance(v, list):
            return None
        return f"expected a list, found {_fmt_found(v)} " \
               f"({type(v).__name__})"
    return None  # unknown kind for this key: loader stays generic


# Recognized sub-keys of the two sections that pop into flat fields before the
# generic filter runs (mirrors the handling in load_config).
_NESTED_SECTIONS = {
    "recording": {"enabled": "bool", "redact_secrets": "bool",
                  "retention_days": "int"},
    # D-18: `timeout_seconds`, `liveness_timeout_seconds` and
    # `liveness_check_interval_seconds` are gone — children have no deadlines.
    # Old YAML carrying them still loads (unknown sub-keys are warn-only).
    "delegation": {"enabled": "bool", "max_concurrent": "int",
                   "queue_max": "int", "wake_retry_seconds": "float",
                   "async_default": "bool"},
    # WP12 plans: sub-keys registered for the consolidated report; `loop` is
    # a dict of loop knobs validated leniently by core/plans.plans_config and
    # consumed by core/plan_loop.py.
    "plans": {"enabled": "bool", "inject_summary": "bool",
              "max_inject_chars": "int", "loop": "dict"},
}


def _validate_config(data: Dict[str, Any], *, strict_unknown: bool,
                     ) -> List[str]:
    """Collect EVERY config problem as one list of human-readable strings.

    Categories: missing required (REQUIRED_FIELDS — empty today), mistyped
    values (always reported; this IS the feature), and unknown keys (only in
    strict mode, so stray keys in existing configs keep loading with just the
    pre-WP2 warning)."""
    problems: List[str] = []
    known = set(ToasterLoopConfig.__dataclass_fields__) | set(_NESTED_SECTIONS)

    for key in REQUIRED_FIELDS:
        if data.get(key) is None:
            problems.append(f"{key}: missing required field")

    for k in sorted(data):
        if k not in known:
            if strict_unknown:
                close = get_close_matches(k, sorted(known), n=1)
                hint = f" (did you mean {close[0]!r}?)" if close else ""
                problems.append(
                    f"{k}: unknown key{hint} — strict mode "
                    "(TOASTER_LOOP_CONFIG_STRICT=1)")
            continue
        prob = _type_problem(k, data[k])
        if prob:
            problems.append(f"{k}: {prob}")

    for sec, subs in _NESTED_SECTIONS.items():
        raw = data.get(sec)
        if not isinstance(raw, dict):
            if raw is not None:
                problems.append(
                    f"{sec}: expected a mapping, found {_fmt_found(raw)} "
                    f"({type(raw).__name__})")
            continue
        for sub in sorted(raw):
            path = f"{sec}.{sub}"
            kind = subs.get(sub)
            if kind is None:
                if strict_unknown:
                    close = get_close_matches(sub, sorted(subs), n=1)
                    hint = f" (did you mean {sec}.{close[0]}?)" if close else ""
                    problems.append(
                        f"{path}: unknown key{hint} — strict mode "
                        "(TOASTER_LOOP_CONFIG_STRICT=1)")
                continue
            v = raw[sub]
            if v is None:
                continue
            prob = _type_problem(path, v, kind=kind)
            if prob:
                problems.append(f"{path}: {prob}")
    return problems


def _strict_unknown_default() -> bool:
    """TOASTER_LOOP_CONFIG_STRICT env gate (truthy strings enable strict mode)."""
    return str(os.environ.get("TOASTER_LOOP_CONFIG_STRICT", "")).strip().lower() \
        in _TRUTHY_STRINGS


@dataclass
class ToasterLoopConfig:
    """Resolved toaster_loop configuration (Phase 1 subset)."""

    gguf_path: Optional[str] = None
    # Base model folder scanned for the serve/train model pickers (HF hub cache
    # layout or any tree of GGUFs / HF checkpoints). Unset => fall back to
    # $TOASTER_LOOP_MODELS_DIR, then $HF_HOME/hub, then ~/.cache/huggingface/hub.
    # See core/model_discovery.py.
    models_dir: Optional[str] = None
    # State root for the experience buffer etc. None => ~/.toaster-loop
    # (env TOASTER_LOOP_HOME wins over this key; see store/paths.py).
    home: Optional[str] = None
    port: int = 0  # 0 => dynamic free port (loopback)
    context_length: int = DEFAULT_CONTEXT_LENGTH
    attach_url: Optional[str] = None  # skip spawning; use this OpenAI-compat URL
    # Optional bearer token for attach mode (e.g. an external server that
    # requires auth). Never required for supervisor-owned llama-server (no auth,
    # loopback). Env: TOASTER_LOOP_ATTACH_API_KEY
    attach_api_key: Optional[str] = None
    model_name: Optional[str] = None  # override served model id
    llama_server_binary: Optional[str] = None  # explicit llama-server path
    host: str = "127.0.0.1"  # loopback only — do not change (project constraint)
    extra_llama_args: list[str] = field(default_factory=list)
    # GPU offload layers passed to llama-server as -ngl. None (default) =>
    # auto-detect: full offload (999) when a CUDA/ROCm/Metal/Vulkan runtime is
    # present, 0 on CPU-only boxes. Set explicitly to pin behavior
    # (0 = force CPU). Env: TOASTER_LOOP_GPU_LAYERS.
    gpu_layers: Optional[int] = None
    # Auto-build llama.cpp into <TOASTER_LOOP_HOME>/llama.cpp when no llama-server
    # exists anywhere (clone + cmake for the detected backend). Env opt-out:
    # TOASTER_LOOP_NO_AUTO_SETUP=1.
    llama_auto_setup: bool = True
    # Source + build options for the managed llama.cpp install. Point `repo` at
    # a fork (e.g. https://github.com/Lychee-Technology/llama-cpp-for-strix-halo)
    # and `ref` at a branch/tag; `backend` forces cuda/rocm/metal/vulkan/cpu
    # instead of auto-detection. Env: TOASTER_LOOP_LLAMA_REPO / _REF / _BACKEND.
    llama_repo: Optional[str] = None
    llama_ref: Optional[str] = None
    llama_backend: Optional[str] = None
    # Phase 2: experience-buffer trace recording (on by default).
    recording_enabled: bool = True
    # Backlog #1: redact secrets from the durable trace copy before the JSONL
    # write (YAML `recording.redact_secrets`, default true). Disabling it lets
    # credentials flow into training data — the recorder logs a loud warning.
    recording_redact_secrets: bool = True
    # Buffer retention (production hardening): YAML `recording.retention_days`
    # — the default age cutoff for `toaster-loop traces prune` (days; 0 disables
    # the default, i.e. prune then requires an explicit --older-than). The
    # buffer is otherwise append-only forever; like the model janitor, pruning
    # stays a manual/cron action and never runs from a server thread.
    recording_retention_days: int = 0
    # Feature-wave Phase 2 (delegation): YAML `delegation.enabled` master
    # switch for list_agents/delegate_task/check_delegation (default true).
    # Env: TOASTER_LOOP_DELEGATION ("0"/"false" disables).
    delegation_enabled: bool = True
    # D-17 (audit §4) async delegation inbox — dispatcher knobs. There is no
    # wait timeout and no liveness watchdog: children run as long as they need.
    # YAML `delegation.max_concurrent`: bounded dispatcher pool size (the
    # accidental one-parent-at-a-time serialization of D-11 is gone, so this
    # cap keeps GPU contention in check; set 1 on a single-GPU box).
    # Env: TOASTER_LOOP_DELEGATION_MAX.
    delegation_max_concurrent: int = 2
    # YAML `delegation.queue_max`: queued-but-unstarted cap; beyond it
    # delegate_task returns an explicit dispatcher_saturated error payload.
    delegation_queue_max: int = 16
    # YAML `delegation.wake_retry_seconds`: Phase 2 watcher redelivery
    # interval for threads busy with an in-flight run (stored, unused until
    # the wake loop lands).
    delegation_wake_retry_seconds: float = 2.0
    # YAML `delegation.async_default`: value of delegate_task's `wait` when
    # the model omits it AND the caller didn't override; false restores D-11
    # synchronous behavior globally as an escape hatch.
    delegation_async_default: bool = True
    # Phase 3: raw `curation:` YAML section (see curate/curater.py
    # CurationConfig for the recognized keys; empty => defaults).
    curation: dict[str, Any] = field(default_factory=dict)
    # Phase 4: raw `training:` / `validation:` YAML sections (see
    # training/scheduler.py TrainingConfig and training/validate.py
    # ValidationConfig for recognized keys; empty => defaults).
    training: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    # Phase 4: `serve --auto` resolves the ACTIVE promoted GGUF from the model
    # registry instead of gguf_path.
    auto_model: bool = False
    # Phase 5: raw `dashboard:` YAML section (see dashboard/server.py for the
    # recognized keys; empty => defaults). Binds loopback unless
    # ``expose_lan: true`` widens it to 0.0.0.0 (token auth then mandatory).
    dashboard: dict[str, Any] = field(default_factory=dict)
    # Raw `update:` YAML section — Toaster Loop self-update system
    # (core/updater.py). Keys: enabled (bool, default true), repo
    # ("owner/name" or URL override for the upstream; default = the clone's
    # origin), interval_hours (cached-check TTL for dashboard/CLI notices,
    # default 24). Applying is always an explicit user action.
    update: dict[str, Any] = field(default_factory=dict)
    # Raw `agent:` YAML section. Recognized keys:
    #   context_token_budget: int  — compress REPL/chat history when the
    #     estimated prompt exceeds this many tokens (0 disables; default is
    #     75% of context_length). See core/compression.py.
    #   keep_recent_tokens: int    — tail size kept verbatim during
    #     compression (default 2048).
    #   guardrails: bool           — tool-loop guardrails + repetition
    #     truncation in the agent loop (default true; false makes the
    #     controller always allow and skips repetition truncation).
    #     See core/guardrails.py.
    #   workspace_root: str|null   — workspace containment root for the file
    #     tools (backlog item #9). Unset => current working directory at
    #     runner creation; explicit null => unrestricted (one startup
    #     warning). TOASTER_LOOP_HOME and the system temp dir are always
    #     allow-listed alongside it. Profiles may override per profile.
    #     See core/path_security.py + core/agent_runner.resolve_workspace_roots.
    #   capabilities: dict         — GLOBAL defaults for per-agent reach-out
    #     toggles: {internet: bool (web_fetch), run_commands: bool
    #     (run_command), vision: bool (view_image), screenshots: bool
    #     (take_screenshot)}. All off by default; agent profiles may override
    #     per profile via their own `capabilities:` dict. See
    #     core/tools.resolve_capabilities + CAPABILITY_TOOLS in core/tools.py.
    #   vision: "auto"|bool        — image input for the served model. auto
    #     probes the endpoint for an mmproj projector (llama-server /v1/models
    #     metadata); a set run_settings.mmproj_path short-circuits to true.
    #     Gates whether attach markers ([image attached: path]) are expanded
    #     into native multimodal parts per-request. See core/images.py.
    agent: dict[str, Any] = field(default_factory=dict)
    # Raw `tools:` YAML section (backlog #1). Recognized keys:
    #   block_sensitive_files: bool — read_file/search_files refuse sensitive
    #     credential paths (.env, SSH keys, .aws/.gnupg stores; default true).
    #     See core/tools.py.
    #   lint_on_write: bool — post-write editor-style syntax check on the new
    #     content of write_file/patch (tier 1; default true). Env:
    #     TOASTER_LOOP_LINT_ON_WRITE. See core/tools.py (_post_write_suffix).
    #   lsp: dict — optional tier-2 LSP diagnostics (default disabled). Keys:
    #     enabled: bool (env TOASTER_LOOP_LSP), servers: {ext: "cmd args"} override
    #     map, startup_timeout_seconds (default 15), check_timeout_seconds
    #     (default 20). Servers are only used when detected via shutil.which.
    #   spill_enabled: bool — persist oversized tool results to
    #     <TOASTER_LOOP_HOME>/cache/spillover/ and keep a stub in context
    #     (default true). See core/spill.py (backlog item #10).
    #   spill_threshold_chars: int — per-result spill threshold (default 48000).
    #   spill_threshold_tokens: int — optional token-accurate per-result spill
    #     threshold; when set it WINS over spill_threshold_chars (exact
    #     /tokenize counts drive the gate, degrading to the chars comparison
    #     while the tokenize endpoint is down). Unset = char gate only
    #     (default; existing configs keep today's behavior).
    #   turn_result_budget_chars: int — per-turn aggregate tool-output budget;
    #     largest results spill first when exceeded (default 120000).
    #   spill_retention_days: int — age past which spillover files are
    #     pruned by the once-per-process sweep (default 7).
    tools: dict[str, Any] = field(default_factory=dict)
    # Raw `janitor:` YAML section (backlog #16). Recognized keys:
    #   enabled: bool — run the disk janitor at `train auto-check` time
    #     (default false; manual runs via `toaster-loop models janitor` always work).
    #   retention_days: int — age past which terminal models are archived
    #     (default 30). See registry/janitor.py.
    janitor: dict[str, Any] = field(default_factory=dict)
    # Raw `run_settings:` YAML section (reference-style model run knobs,
    # applied when toaster_loop spawns llama-server). Recognized keys (all optional;
    # see serving/run_args.py for the exact llama-server mapping):
    #   kv_cache_type: f16|q8_0|q4_0|bf16  — KV cache dtype (-ctk/-ctv)
    #   flash_attn: bool                    — --flash-attn on (value-taking flag)
    #   parallel: int                       — -np (parallel slots)
    #   batch_size: int                     — -b
    #   ubatch_size: int                    — -ub
    #   tensor_parallel: bool               — --tensor-parallel
    #   no_mmap: bool                       — --no-mmap
    #   mlock: bool                         — --mlock
    #   mmproj_path: str                    — advanced manual vision projector
    #     path (--mmproj); normally unset — with `vision` on, toaster_loop
    #     auto-detects an mmproj*.gguf next to the model GGUF (core/mmproj.py)
    #   vision: bool                        — load the sibling vision projector
    #     when one exists (default true = auto; false never loads)
    #   speculative_type: auto|ngram|mtp|ngram+mtp|off  — llama-server
    #     --spec-type (reference-style modes; see serving/run_args.py)
    #   spec_draft_n_max: int               — draft length 1..16
    #     (--spec-draft-n-max)
    #   gpu_memory_pct: int                 — VRAM budget 1..100; scales -ngl
    #     when gpu_layers is unset (auto): offload only the fraction of layers
    #     that should fit in that budget.
    #   extra_args: str                     — free-form args appended verbatim
    #     (e.g. "--rope-scaling yarn --yarn-orig-ctx 32768")
    # NOTE: this section configures the DEFAULT server. Named instances carry
    # their own run_settings (serving/instances.py) and do NOT inherit it.
    run_settings: dict[str, Any] = field(default_factory=dict)
    # Raw `sampling:` YAML section — default sampling parameters injected into
    # every /chat/completions request the built-in agent makes (applies to
    # attached servers too). Recognized keys: temperature, top_p, top_k,
    # min_p, repeat_penalty, presence_penalty, frequency_penalty,
    # penalty_free_repeat, max_tokens. Omitted/empty => server defaults.
    # Pass 4 #40 adds `logprobs: bool` (default false): when true the runner
    # requests per-token content logprobs and each trace gains two cheap
    # confidence aggregates (conf_mean / conf_p90_negentropy). Record-only —
    # never affects retries or behavior; backends without support degrade to
    # unchanged responses. Judge calls never request logprobs.
    # Thinking / reasoning controls (reference style) share the section:
    #   enable_thinking: bool             — on/off gate for thinking models
    #   reasoning_effort: none|minimal|low|medium|high
    #   preserve_thinking: bool           — keep historical thinking blocks
    #                                     from past assistant turns (Qwen-style)
    # They reach llama.cpp through chat_format_template_controls and
    # vLLM-style servers read them top-level (serving/run_args.py).
    sampling: dict[str, Any] = field(default_factory=dict)
    # Raw `indexing:` YAML section — codebase semantic search (core/indexer.py).
    # Phase 5 schema (per-project indexes; toaster_loop owns the embed server):
    #   enabled: bool                 — master switch (default false)
    #   model: str                    — embedding GGUF path; toaster_loop spawns +
    #     stops its own loopback llama-server --embedding for it
    #   output_dim: 768|512|256|128   — MRL truncation dim (default full)
    #   batch_size: int               — embed server -b/-ub/-c (default 2048);
    #     raise it for embedding models with larger trained context
    #   search_mode: hybrid|semantic|lexical (default hybrid)
    #   paths_override: [str]         — advanced: manual roots for the
    #     default/no-project DB only (project DBs derive roots from the
    #     project's workspace_dir)
    # Legacy keys (embedding_url, embedding_model, top_k, paths) are absorbed
    # by load_config without crashing — see _normalize_indexing().
    indexing: dict[str, Any] = field(default_factory=dict)
    # Phase 5e: agent profiles — list of
    #   {name, system_prompt, tools: [str], model_name?: str}
    # managed through the Profiles panel / `profiles:` section; default_profile
    # names the entry chat runs use when none is selected.
    profiles: list[dict[str, Any]] = field(default_factory=list)
    default_profile: Optional[str] = None
    # Raw `evolution:` YAML section (pass 4 #39, evaluation/evolve.py). Keys:
    #   enabled: bool                — master switch for prompt-evolution jobs
    #     (default false; when off the `evolve` job kind refuses to run and the
    #     CLI errors before queueing — zero behavior change)
    #   max_call_budget: int         — hard cap on total LLM calls per evolve
    #     job (mutation + replay + judge); a job's params.call_budget is
    #     clamped to this. Default 60.
    evolution: dict[str, Any] = field(default_factory=dict)
    # Raw `playbooks:` YAML section (pass 4 #38, core/playbooks.py). Keys:
    #   enabled: bool               — master switch for prompt injection
    #     (default false; when off create_runner never opens the playbook
    #     store and the system prompt is byte-identical to pre-#38 behavior)
    #   total_cap_chars: int        — hard cap on the whole "## Playbooks"
    #     section appended to a run's system prompt (default 2000); recipes
    #     that do not fit whole are omitted whole, never sliced
    #   per_recipe_cap_chars: int   — per-recipe cap before injection
    #     (default 800; over-cap recipes are omitted entirely)
    playbooks: dict[str, Any] = field(default_factory=dict)
    # Raw `skills:` YAML section (core/skills.py). Keys:
    #   enabled: bool               — master switch: exposes the
    #     save_skill/read_skill/list_skills tools AND appends the capped
    #     "## Skills" index to run system prompts (default false; when off
    #     create_runner never touches the skills dir and prompts stay
    #     byte-identical to pre-feature behavior)
    #   index_total_cap_chars: int  — hard cap on the whole index section
    #     (default 1500; entries that do not fit are omitted whole)
    skills: dict[str, Any] = field(default_factory=dict)
    # Raw `plans:` YAML section (WP12, core/plans.py). Keys (read leniently —
    # invalid values fall back to the defaults below):
    #   enabled: bool               — master switch for per-project task
    #     graphs (default false; when off NO plan tools are registered, no
    #     prompt section is injected, MCP lists no plan tools and nothing in
    #     this module touches <home>/plans/ — byte-identical everything)
    #   inject_summary: bool        — append the bounded "## Project plan"
    #     section to run system prompts when enabled (default true; off keeps
    #     the tools but never injects)
    #   max_inject_chars: int       — hard cap on that section (default 800)
    #   loop: dict                  — guarded autonomous loop knobs
    #     (core/plan_loop.py): require_verification: bool (default true — a
    #     run ending on the verification nudge blocks the task), max_tasks:
    #     int (default 10), max_consecutive_failures: int (default 1 — stop
    #     after one non-done). Task runs have no timeout.
    plans: dict[str, Any] = field(default_factory=dict)
    # Raw `logging:` YAML section (consumed by cli._setup_logging). Keys:
    #   level: str              — root level for non-verbose runs
    #     (DEBUG|INFO|WARNING|ERROR; -v always wins)
    #   file: str               — rotating log file; relative names resolve
    #     under TOASTER_LOOP_HOME (e.g. logs/toaster_loop.log). Unset => stderr only.
    #   max_bytes: int          — rotation size (default 5 MB)
    #   backup_count: int       — rotated files kept (default 3)
    #   module_levels: dict     — per-logger overrides, e.g.
    #     {toaster_loop.serving: DEBUG}
    logging: dict[str, Any] = field(default_factory=dict)
    # Raw `rft:` YAML section (training/rejection.py) — rejection-sampling
    # fine-tuning (STaR): sample K completions per prompt from the served
    # model, keep only verifier-confirmed ones, emit a ShareGPT SFT dataset.
    # Recognized keys (see RFTConfig below; empty => defaults, OFF by default):
    #   enabled: bool             — master switch (default false; when off the
    #     `rft` job kind refuses to run and the CLI errors before queueing)
    #   sample_k: int             — candidates sampled per prompt (default 4)
    #   judge_min_score: float    — STRICT pass threshold for open-chat judge
    #     verification (default 0.85; golden tasks need no threshold — their
    #     checks are deterministic)
    #   max_prompts: int          — cap on prompts sampled per run (default 16)
    #   temperature: float        — elevated sampling temperature for candidate
    #     diversity (default 0.9; verification stays deterministic)
    #   max_tokens: int           — completion budget per candidate (default 512)
    #   max_prompt_tokens: int    — exact-token budget gate (TokenCounter);
    #     prompts whose context exceeds it are skipped (default 8192)
    #   emit_dpo: bool            — emit dpo.jsonl alongside sft.jsonl: kept
    #     samples paired against verifier-failed candidates of the same prompt
    #     (default true; RFT itself is opt-in via `enabled`, and a dataset with
    #     no pairs stays valid — empty dpo.jsonl, curater parity)
    #   max_dpo_pairs_per_prompt: int — cap on pairs per prompt when a prompt
    #     yields several failed candidates (default 3; diverse failure reasons
    #     preferred)
    #   agent_profile / project   — trace-selection filters (store query helpers)
    #   base_model: str           — train.yaml template input for the dataset
    rft: dict[str, Any] = field(default_factory=dict)
    # Raw `goldens_mine:` YAML section (curate/mine_goldens.py) — auto-mine
    # verifiable golden tasks from 👍 traces. Recognized keys (see
    # GoldensMineConfig below; empty => defaults, OFF by default):
    #   enabled: bool              — master switch (default false; when off the
    #     `mine_goldens` job kind refuses to run and the CLI errors before
    #     queueing — zero behavior change)
    #   max_per_run: int           — goldens mined per run (default 8)
    #   max_candidates: int        — 👍 traces examined per run (default 64)
    #   min_feedback_score: float  — 👍 threshold on user feedback (default 1.0)
    #   use_judge: bool            — optionally ask the served model for extra
    #     substring checks (default false; every proposal must still pass the
    #     deterministic self-check gate — pure-judge goldens stay impossible)
    goldens_mine: dict[str, Any] = field(default_factory=dict)
    # Raw `easy_training:` YAML section — turnkey training loop for one or
    # more agent profiles (dashboard "Easy training" page + `train auto-check`
    # scoping). Keys:
    #   enabled: bool        — master switch (default false)
    #   profile: str         — PRIMARY agent profile whose traces feed the
    #     loop; every profile must already be bound to a managed server
    #     instance (`server_instance`), which is what ties it to a model to
    #     serve/train
    #   profiles: [str]      — full multi-agent list (primary included);
    #     absent/empty => exactly the legacy one-profile behavior
    #   auto_deploy: bool    — after a promoted iterate run, hot-swap the
    #     serving model to the new GGUF automatically (default false)
    #   min_traces: int      — readiness threshold override (else
    #     training.auto_min_traces)
    #   min_score: float     — mean-score threshold override (else
    #     training.auto_min_score)
    # Chat runs with an enabled profile stamp the profile name into
    # traces.agent_profile (store/recorder.py), so curation and the readiness
    # check scope to that profile's sessions only.
    easy_training: dict[str, Any] = field(default_factory=dict)
    # Raw `routing:` YAML section (serving/router.py) — capability-tier-aware
    # auto-routing across running instances. Recognized keys (see
    # RoutingConfig in serving/router.py; empty => defaults, OFF by default):
    #   enabled: bool              — master switch (default false; when off
    #     endpoint resolution is byte-identical to pre-feature behavior)
    #   escalate_on_failure: bool  — retry a failed run at the next-higher
    #     tier (default true; only meaningful with enabled)
    #   prefer: str                — "lowest_tier" (default: small models
    #     first, the point of the cascade) or "highest_tier"
    #   high_context_tokens: int   — prompt-token threshold raising the tier
    #     requirement to >=2 (default 8192)
    #   tool_heavy_min_tools: int  — enabled-tool count considered tool-heavy
    #     (requirement >=1; default 6)
    #   profile_hints: map         — {profile_name: "low"|"high"|"auto"}
    #   refusal_markers: list      — response substrings treated as refusals
    #     that trigger escalation (default empty = never)
    routing: dict[str, Any] = field(default_factory=dict)
    # Raw `serve:` YAML section (serving-side launch behavior). Recognized keys:
    #   calibration: off|auto      — WP3 per-machine autotune opt-in. `off`
    #     (default, and any unknown value) keeps the launch path byte-identical
    #     to pre-WP3; `auto` adopts the stored profile from `toaster-loop calibrate`
    #     for THIS machine + model as DEFAULTS underneath explicit run_settings
    #     (manual values always win). See serving/calibrate.py.
    #   watchdog_enabled: bool     — WP4 engine watchdog master switch, OFF by
    #     default (a kill-loop must never run uninvited). When on, a managed
    #     llama-server counts as FROZEN when a request stays in flight for
    #     `watchdog_window_seconds` (90) while its /proc CPU-time delta stays
    #     under `watchdog_cpu_threshold_ms` (10) AND its disk-IO delta under
    #     `watchdog_io_threshold_bytes` (0) over each
    #     `watchdog_sample_interval_seconds` (5); the pid is then re-verified
    #     as ours before SIGTERM/SIGKILL and restarted with exponential backoff
    #     (`watchdog_restart_backoff_base_seconds` 2, capped at
    #     `watchdog_max_restarts` 3 per `watchdog_restart_window_seconds` 600).
    #     See serving/watchdog.py; keys are read leniently (invalid values
    #     fall back to the default with a warning).
    #   --- WP5 GPU coexistence (STRICTLY OPT-IN; all defaults off) ----------
    #   idle_unload_seconds: float — 0 (default, disabled). When > 0, a
    #     supervisor-side timer gracefully stops (SIGTERM via the owned-stop
    #     paths) any running instance with no tracked request for this many
    #     seconds and records it as `unloaded`; never while a request is in
    #     flight. See serving/coexist.py.
    #   lazy_load: bool            — false (default). When true, a request
    #     resolving to an unloaded instance transparently re-launches it
    #     (blocking; cold-start latency accepted) before forwarding. With it
    #     off, requests to an unloaded instance fail with a load-it message.
    #   min_free_vram_mib: int     — 0 (default = no handshake). Together with
    #     idle_unload_seconds > 0 it arms the train+serve handshake: a job
    #     that FAILS the training VRAM gate but WOULD pass assuming served
    #     instances are unloaded triggers unload-first, then a bounded poll of
    #     the free-VRAM probe before admitting the job.
    #   coexist_policy: refuse|unload — refuse (default, safe): nothing is
    #     unloaded, a `coexist_refused` event explains that unloading WOULD
    #     have admitted the job; `unload`: perform the graceful unload.
    #   default_engine: str        — inference-engine id (serving/engines.py)
    #     the DEFAULT server spawns with. Unset/blank/"llama-cpp" (default) =>
    #     the built-in llama.cpp path, byte-identically. Any other value must
    #     name a registered engine (Hardware page); unknown ids fail the spawn
    #     with a clear error, never silently fall back. Read leniently.
    #   Events land on the dashboard bus as kind `serve.coexist`:
    #   instance_unloaded, instance_loaded, coexist_unload_triggered,
    #   coexist_refused. Race policy: unload/load serialize per instance id;
    #   a request arriving mid-unload reloads transparently (lazy_load) or
    #   gets a clear retryable error. Keys read leniently like the rest.
    serve: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_attach_url(self) -> Optional[str]:
        """Normalized attach URL with /v1 suffix, if attach mode is on."""
        if not self.attach_url:
            return None
        url = self.attach_url.rstrip("/")
        if not url.endswith("/v1"):
            url = url + "/v1"
        return url


@dataclass
class RFTConfig:
    """`rft:` YAML section (training/rejection.py). OFF by default — the whole
    rejection-sampling pipeline is inert until `enabled: true`."""

    enabled: bool = False
    sample_k: int = 4                # K candidates per prompt
    judge_min_score: float = 0.85    # strict pass bar for open-chat judging
    max_prompts: int = 16            # prompts sampled per run (GPU-cost bound)
    temperature: float = 0.9         # elevated for candidate diversity
    max_tokens: int = 512            # per-candidate completion budget
    max_prompt_tokens: int = 8192    # skip prompts whose context exceeds this
    # DPO negatives from failed candidates (deferred item of the RFT build):
    # default ON *within* RFT — the whole pipeline stays gated by `enabled`,
    # and prompts without a kept row never emit orphan pairs.
    emit_dpo: bool = True
    max_dpo_pairs_per_prompt: int = 3
    agent_profile: Optional[str] = None
    project: Optional[str] = None
    base_model: Optional[str] = None  # train.yaml template input

    @staticmethod
    def from_yaml(data: Dict[str, Any]) -> "RFTConfig":
        sec = data.get("rft") if isinstance(data, dict) else None
        if not isinstance(sec, dict):
            return RFTConfig()
        known = {f for f in RFTConfig.__dataclass_fields__}
        unknown = sorted(k for k in sec if k not in known)
        if unknown:
            logger.warning(
                "rft config: ignoring unknown key(s) %s (known: %s)",
                ", ".join(unknown), ", ".join(sorted(known)))
        kwargs = {k: v for k, v in sec.items() if k in known and v is not None}
        return RFTConfig(**kwargs)


@dataclass
class GoldensMineConfig:
    """`goldens_mine:` YAML section (curate/mine_goldens.py). OFF by default —
    golden auto-mining from 👍 traces is inert until `enabled: true`."""

    enabled: bool = False
    max_per_run: int = 8             # goldens mined per pass (hygiene cap)
    max_candidates: int = 64         # 👍 traces examined per pass
    min_feedback_score: float = 1.0  # 👍 threshold on user feedback rows
    use_judge: bool = False          # optional LLM-proposed extra substrings
                                     # (still gated by the deterministic check)

    @staticmethod
    def from_yaml(data: Dict[str, Any]) -> "GoldensMineConfig":
        sec = data.get("goldens_mine") if isinstance(data, dict) else None
        if not isinstance(sec, dict):
            return GoldensMineConfig()
        known = {f for f in GoldensMineConfig.__dataclass_fields__}
        unknown = sorted(k for k in sec if k not in known)
        if unknown:
            logger.warning(
                "goldens_mine config: ignoring unknown key(s) %s (known: %s)",
                ", ".join(unknown), ", ".join(sorted(known)))
        kwargs = {k: v for k, v in sec.items() if k in known and v is not None}
        return GoldensMineConfig(**kwargs)


_ENV_MAP = {
    "gguf_path": "TOASTER_LOOP_GGUF",
    "models_dir": "TOASTER_LOOP_MODELS_DIR",
    "home": "TOASTER_LOOP_HOME",
    "port": "TOASTER_LOOP_PORT",
    "context_length": "TOASTER_LOOP_CONTEXT",
    "attach_url": "TOASTER_LOOP_ATTACH_URL",
    "model_name": "TOASTER_LOOP_MODEL",
    "llama_server_binary": "TOASTER_LOOP_LLAMA_SERVER",
    "attach_api_key": "TOASTER_LOOP_ATTACH_API_KEY",
    "recording_enabled": "TOASTER_LOOP_RECORDING",
    "delegation_enabled": "TOASTER_LOOP_DELEGATION",
    "delegation_max_concurrent": "TOASTER_LOOP_DELEGATION_MAX",
    "auto_model": "TOASTER_LOOP_AUTO_MODEL",
    "gpu_layers": "TOASTER_LOOP_GPU_LAYERS",
    "llama_auto_setup": "TOASTER_LOOP_LLAMA_AUTO_SETUP",
    "llama_repo": "TOASTER_LOOP_LLAMA_REPO",
    "llama_ref": "TOASTER_LOOP_LLAMA_REF",
    "llama_backend": "TOASTER_LOOP_LLAMA_BACKEND",
}


def _coerce(fieldname: str, value: Any) -> Any:
    if fieldname in ("port", "context_length", "gpu_layers",
                     "delegation_max_concurrent"):
        return int(value)
    if fieldname in ("recording_enabled", "auto_model", "llama_auto_setup",
                     "delegation_enabled"):
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    return str(value)


def resolve_config_path(config_path: Optional[str] = None) -> Optional[Path]:
    """Effective YAML config path (highest precedence first).

    ``config_path`` arg > ``TOASTER_LOOP_CONFIG`` env > ``<TOASTER_LOOP_HOME>/toaster_loop.yaml``
    (the file the dashboard writes settings to; see DashboardApp._config_path).
    The home fallback is returned only when that file exists; returns None when
    no config file applies, meaning built-in defaults should be used. Uses the
    same TOASTER_LOOP_HOME resolution as store/paths.py.
    """
    path = config_path or os.environ.get("TOASTER_LOOP_CONFIG", "").strip()
    if path:
        return Path(path).expanduser()
    from ..store.paths import resolve_toaster_loop_home  # local import: no cycle

    p = resolve_toaster_loop_home() / "toaster_loop.yaml"
    return p if p.is_file() else None


def _normalize_indexing(section: dict) -> dict:
    """Map a raw `indexing:` YAML section onto the Phase 5 schema.

    Backward-compatible loading of pre-Phase-5 keys (absorbed, never crash):
      * embedding_model -> model
      * paths           -> paths_override (manual roots now only apply to
                           the default/no-project DB; project DBs derive
                           their root from the project's workspace_dir)
      * embedding_url   -> dropped (toaster_loop owns the embed server now)
      * top_k           -> dropped (internal constant SEARCH_TOP_K)
    Unknown future keys pass through untouched."""
    out = dict(section)
    if "model" not in out and "embedding_model" in out:
        out["model"] = out["embedding_model"]
    out.pop("embedding_model", None)
    out.pop("embedding_url", None)   # attach mode removed (Phase 5)
    out.pop("top_k", None)           # internal constant now (Phase 5)
    # auto_retrieve_threshold was never implemented anywhere — dropped so a
    # stale key in an old config can't masquerade as a live knob. It stays
    # accepted by the dashboard's PUT validator (round-trip compat for files
    # that already carry it; it just has no effect after load).
    out.pop("auto_retrieve_threshold", None)
    if "paths_override" not in out and "paths" in out:
        out["paths_override"] = out["paths"]
    out.pop("paths", None)
    od = out.get("output_dim")
    if od is not None:
        try:
            out["output_dim"] = int(od)
        except (TypeError, ValueError):
            out.pop("output_dim", None)  # garbage value: keep full-dim default
    return out


def load_config(config_path: Optional[str] = None, *,
                strict_unknown: Optional[bool] = None) -> ToasterLoopConfig:
    """Build config from defaults, YAML file, and env overrides (in that order).

    WP2a validation: a YAML with mistyped values raises ONE consolidated
    ConfigError enumerating every problem field (path, expected kind, redacted
    found value) — "refuse to guess", kimi-k3-in-c pattern re-implemented
    inline. Unknown keys stay warn-only by default (byte-identical loading for
    configs with stray keys); pass ``strict_unknown=True`` or set
    ``TOASTER_LOOP_CONFIG_STRICT=1`` to fold them into the same report. Env
    overrides are applied after and are not validated here (they are coerced).
    """
    data: dict[str, Any] = {}

    # Settings-persistence fix: with no explicit path or TOASTER_LOOP_CONFIG env,
    # fall back to <TOASTER_LOOP_HOME>/toaster_loop.yaml so dashboard-saved settings
    # (profiles, run_settings, sampling, indexing, ...) survive restarts.
    path = config_path or os.environ.get("TOASTER_LOOP_CONFIG")
    if not path:
        fallback = resolve_config_path()
        path = str(fallback) if fallback else None
    if path:
        p = Path(path).expanduser()
        if p.exists():
            loaded = yaml.safe_load(p.read_text()) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file {p} must contain a YAML mapping")
            data.update(loaded)
        elif config_path:  # explicitly requested but missing => hard error
            raise FileNotFoundError(f"Config file not found: {p}")

    # WP2a: consolidated validation of the YAML layer (before section pops,
    # so `recording:`/`delegation:` sub-keys are checked in place too).
    if data:
        strict = (_strict_unknown_default() if strict_unknown is None
                  else bool(strict_unknown))
        problems = _validate_config(data, strict_unknown=strict)
        if problems:
            where = f" in {path}" if path else ""
            raise ConfigError(
                f"invalid configuration{where} — refusing to guess "
                f"({len(problems)} problem"
                f"{'s' if len(problems) != 1 else ''}):\n"
                + "\n".join(f"  - {p}" for p in problems)
                + ("" if strict else
                   "\n(unknown keys are warn-only unless TOASTER_LOOP_CONFIG_STRICT=1; "
                   "mistyped values above are always errors)")
            )

    # Nested YAML sections map onto flat dataclass fields.
    # A typo'd top-level knob (e.g. `trainig:` or `contex_length:`) used to be
    # dropped silently by the kwargs filter below; warn like TrainingConfig.
    # from_yaml does — recognized set derived from the dataclass itself, plus
    # the two section keys that pop into flat fields before the filter runs.
    known = set(ToasterLoopConfig.__dataclass_fields__) | {"recording", "delegation"}
    unknown = sorted(k for k in data if k not in known)
    if unknown:
        logger.warning(
            "config: ignoring unknown key(s) %s (known: %s)",
            ", ".join(unknown), ", ".join(sorted(known)))

    if isinstance(data.get("recording"), dict):
        rec = data.pop("recording")
        if "enabled" in rec and data.get("recording_enabled") is None:
            data["recording_enabled"] = bool(rec["enabled"])
        if "redact_secrets" in rec and data.get("recording_redact_secrets") is None:
            data["recording_redact_secrets"] = bool(rec["redact_secrets"])
        if "retention_days" in rec \
                and data.get("recording_retention_days") is None:
            try:
                data["recording_retention_days"] = int(rec["retention_days"])
            except (TypeError, ValueError):
                logger.warning("config: recording.retention_days %r is not an "
                               "int; keeping default 0", rec["retention_days"])
    if isinstance(data.get("delegation"), dict):
        d = data.pop("delegation")
        if "enabled" in d and data.get("delegation_enabled") is None:
            data["delegation_enabled"] = bool(d["enabled"])
        # D-17 async delegation inbox knobs (same pop-onto-flat-field rule;
        # types are already validated by _NESTED_SECTIONS above). D-18: the
        # old timeout_seconds / liveness_* keys are gone — children have no
        # deadlines; old YAML carrying them warns as unknown and is ignored.
        for sub, kind in (("max_concurrent", int), ("queue_max", int),
                          ("wake_retry_seconds", float),
                          ("async_default", bool)):
            flat = "delegation_" + sub
            if sub in d and data.get(flat) is None:
                try:
                    data[flat] = kind(d[sub])
                except (TypeError, ValueError):
                    logger.warning("config: delegation.%s %r is not a %s; "
                                   "keeping default", sub, d[sub],
                                   kind.__name__)
    # Feature-wave Phase 5: normalize `indexing:` onto the new schema while
    # absorbing legacy keys (embedding_url/embedding_model/top_k/paths).
    if isinstance(data.get("indexing"), dict):
        data["indexing"] = _normalize_indexing(data["indexing"])

    kwargs = {
        k: v
        for k, v in data.items()
        if k in ToasterLoopConfig.__dataclass_fields__ and v is not None
    }
    # Backlog #9: preserve an explicit `agent.workspace_root: null` (the
    # generic loader above would drop the None and lose unrestricted mode).
    agent_section = kwargs.get("agent")
    if isinstance(agent_section, dict) and "workspace_root" in agent_section \
            and agent_section["workspace_root"] is None:
        agent_section["workspace_root"] = UNSET
    cfg = ToasterLoopConfig(**kwargs)

    # Env overrides beat the file.
    for fieldname, envname in _ENV_MAP.items():
        raw = os.environ.get(envname)
        if raw is not None and raw != "":
            setattr(cfg, fieldname, _coerce(fieldname, raw))

    # Feature-wave Phase 4: env overrides for nested `tools:` sub-knobs (the
    # flat _ENV_MAP can't reach dicts). Same truthy-string coercion as the
    # boolean fields above; explicit YAML stays the fallback.
    raw = os.environ.get("TOASTER_LOOP_LINT_ON_WRITE")
    if raw is not None and raw != "":
        cfg.tools = {**(cfg.tools or {}),
                     "lint_on_write":
                         str(raw).strip().lower() in ("1", "true", "yes", "on")}
    raw = os.environ.get("TOASTER_LOOP_LSP")
    if raw is not None and raw != "":
        lsp = dict((cfg.tools or {}).get("lsp") or {})
        lsp["enabled"] = str(raw).strip().lower() in ("1", "true", "yes", "on")
        cfg.tools = {**(cfg.tools or {}), "lsp": lsp}

    # Loopback-only constraint: force host regardless of config.
    cfg.host = "127.0.0.1"
    return cfg


def discover_external_server_url() -> Optional[str]:
    """Probe an EXPLICITLY configured external OpenAI-compatible server.

    toaster_loop is standalone and never depends on any other local server
    existing. This probe runs ONLY when the user opts in by setting
    $TOASTER_LOOP_EXTERNAL_SERVER_URL (or config ``attach_url``); there is no
    implicit default-port scan. Returns the base URL when healthy, else None.
    Never starts or mutates anything — an external server may own the port.
    """
    import httpx

    url = os.environ.get("TOASTER_LOOP_EXTERNAL_SERVER_URL")
    if not url:
        return None  # opt-in only: no discovery without explicit configuration
    base = url.rstrip("/")
    try:
        r = httpx.get(f"{base}/api/health", timeout=2.0)
        if r.status_code == 200:
            return base
    except Exception:
        pass
    return None

"""Config read/write endpoints ({yaml} or {section, patch})."""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from ...core.config import ToasterLoopConfig, load_config
from ...curate.curater import CurationConfig
from .base import ApiContext, HttpError, _merge_patch, _redact


class ConfigRoutes(ApiContext):
    """Config read/write endpoints ({yaml} or {section, patch})."""

    _CONFIG_PATH_SECRETS = ("token", "attach_api_key")

    def _config_path(self) -> Path:
        env = os.environ.get("TOASTER_LOOP_CONFIG", "").strip()
        if env:
            return Path(env).expanduser()
        return self.home / "toaster_loop.yaml"

    def h_config_get(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        import dataclasses

        cfg_dict = dataclasses.asdict(self.cfg)
        raw = ""
        path = self._config_path()
        if path.is_file():
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                raw = ""
        return {
            "config": _redact(cfg_dict),
            "raw": raw,
            "path": str(path),
            "token_configured": bool(self.token),
        }

    def h_config_put(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ...training.scheduler import TrainingConfig as _TC
        from ...training.validate import ValidationConfig as _VC

        known_sections = {
            "training": set(_TC.__dataclass_fields__),
            "curation": set(CurationConfig.__dataclass_fields__),
            "validation": set(_VC.__dataclass_fields__),
            # dashboard: port/token are long-standing; max_sse caps concurrent
            # SSE streams (server.py); expose_lan binds 0.0.0.0 for LAN access
            # (token auth becomes mandatory — boot refuses without one).
            "dashboard": {"port", "token", "max_sse", "expose_lan"},
            # Golden mining (curate/mine_goldens.py).
            "goldens_mine": {"enabled", "max_per_run", "max_candidates",
                             "min_feedback_score", "use_judge"},
            # Playbooks (core/playbooks.py) — injection switch + caps. Without
            # this entry, section patches to playbooks: skip field validation
            # entirely (typos land silently while goldens_mine typos 400).
            "playbooks": {"enabled", "total_cap_chars",
                          "per_recipe_cap_chars"},
            # Capability-tier auto-routing (serving/router.py) — review idea #2.
            "routing": {"enabled", "escalate_on_failure", "prefer",
                        "high_context_tokens", "tool_heavy_min_tools",
                        "profile_hints", "refusal_markers"},
            # Built-in toolset policy (backlog #1/#10): raw `tools:` section.
            # `lsp.servers` (per-extension overrides) is raw-YAML territory —
            # the flat LSP keys below cover the common case.
            "tools": {"block_sensitive_files", "run_command_argv_only",
                      "web_fetch", "spill_enabled", "spill_threshold_chars",
                      "spill_threshold_tokens", "turn_result_budget_chars",
                      "spill_retention_days", "lint_on_write",
                      "lsp"},
            # Prompt evolution (pass 4 #39) master switch + call budget.
            "evolution": {"enabled", "max_call_budget"},
            # Delegation (feature-wave Phase 2): flat fields populated from
            # the `delegation:` section by the loader. D-18: no timeout knob —
            # children have no deadlines.
            "delegation": {"enabled"},
            # Agent runtime (core/agent_runner.py): raw `agent:` section.
            # Keys the runner reads; `capabilities`/`vision` are nested
            # dicts/lists handled opaquely here (Profiles owns per-profile).
            "agent": {"context_token_budget", "keep_recent_tokens",
                      "workspace_root", "heal_tool_calls", "verify_on_stop",
                      "max_iterations", "capabilities", "vision"},
            # Disk janitor (backlog #16): raw `janitor:` section.
            "janitor": {"enabled", "retention_days", "disk_retention_days"},
            # Rejection-sampling fine-tuning (training/rejection.py): raw
            # `rft:` section.
            "rft": {"enabled", "sample_k", "judge_min_score", "max_prompts",
                    "temperature", "max_tokens", "max_prompt_tokens", "emit_dpo",
                    "max_dpo_pairs_per_prompt", "agent_profile", "project",
                    "base_model"},
            # Logging: flat keys the logging handler accepts (core/config.py
            # carries the section raw).
            "logging": {"level", "file", "max_bytes", "backup_count",
                        "module_levels"},
            # redact_secrets was loadable but un-patchable through the
            # dashboard validator (latent gap); retention_days is the
            # `toaster-loop traces prune` default window.
            "recording": {"enabled", "redact_secrets", "retention_days"},
            # Persistent skills (core/skills.py) — agent-authored procedures;
            # this section toggles exposure + index cap. The skill list
            # itself lives under /api/skills (Settings panel Skills card).
            "skills": {"enabled", "index_total_cap_chars"},
            # WP12 per-project plans (core/plans.py) — master switch, prompt
            # injection knobs, and the guarded-loop dict (validated leniently
            # by core/plans.plans_config). Without this entry the Settings
            # "Project plans" card could not patch plans: through the
            # section-patch path.
            "plans": {"enabled", "inject_summary", "max_inject_chars",
                      "loop"},
            # Easy training (turnkey loop) — see training/easy.py.
            # `profiles` is the multi-agent list; `profile` stays the primary.
            "easy_training": {"enabled", "profile", "profiles", "auto_deploy",
                              "min_traces", "min_score"},
            "run_settings": {
                "kv_cache_type", "flash_attn", "parallel", "batch_size",
                "ubatch_size", "tensor_parallel", "no_mmap", "mlock",
                "mmproj_path", "vision", "speculative_type",
                "spec_draft_n_max", "mtp_path", "gpu_memory_pct",
                "extra_args",
            },
            "sampling": {
                "temperature", "top_p", "top_k", "min_p", "repeat_penalty",
                "presence_penalty", "frequency_penalty", "penalty_free_repeat",
                "max_tokens",
                # Thinking / reasoning controls (reference style): sent
                # with every request; llama.cpp applies them through
                # chat_format_template_controls (serving/run_args).
                "enable_thinking", "reasoning_effort", "preserve_thinking",
            },
            "indexing": {
                # Phase 5 schema
                "enabled", "model", "output_dim", "search_mode",
                "paths_override", "auto_retrieve_threshold",
                # embed-server launch knobs (core/indexer.py): batch_size =
                # -b/-ub; gpu_layers = -ngl (0 = CPU embeddings, leaves the
                # GPU to the chat model); threads = -t.
                "batch_size", "gpu_layers", "threads",
                # legacy keys stay writable so old configs round-trip;
                # load_config absorbs them (core/config._normalize_indexing)
                "paths", "embedding_model", "embedding_url", "top_k",
            },
            # Self-update system (core/updater.py): raw `update:` section.
            "update": {"enabled", "repo", "interval_hours"},
            # Serving-launch behavior (WP3 calibration / WP4 watchdog / WP5
            # coexistence). Without this entry the Tuning panel could not
            # toggle serve.calibration through a section patch.
            "serve": {
                "calibration", "watchdog_enabled", "watchdog_window_seconds",
                "watchdog_sample_interval_seconds", "watchdog_cpu_threshold_ms",
                "watchdog_io_threshold_bytes", "watchdog_kill_grace_seconds",
                "watchdog_restart_backoff_base_seconds", "watchdog_max_restarts",
                "watchdog_restart_window_seconds", "idle_unload_seconds",
                "lazy_load", "min_free_vram_mib", "coexist_policy",
                # Inference-engine registry: default server's engine binding
                # (serving/engines.py). Blank/unset = built-in llama.cpp.
                "default_engine",
            },
        }
        # Top-level keys the write path accepts. Superset of the loader's own
        # recognition (core/config.py _NESTED_SECTIONS): delegation/agent/tools
        # and the other sections below are all loadable config, so a section
        # patch must never 400 on a key the loader would happily read.
        known_top = set(ToasterLoopConfig.__dataclass_fields__) | {
            "recording", "delegation", "agent", "tools", "janitor",
            "training", "curation", "validation", "dashboard",
        }

        if "yaml" in b:
            text = b.get("yaml")
            if not isinstance(text, str):
                raise HttpError(400, "yaml must be a string")
        elif "section" in b:
            # section-patch mode: merge into current raw.
            # 5d: section "root" patches top-level keys (gguf_path, port, ...)
            # so the Settings form can edit serve defaults too.
            section = str(b["section"])
            patch = b.get("patch")
            if not isinstance(patch, dict):
                raise HttpError(400, "patch must be an object")
            path = self._config_path()
            current: Dict[str, Any] = {}
            if path.is_file():
                try:
                    loaded = yaml.safe_load(path.read_text()) or {}
                    current = loaded if isinstance(loaded, dict) else {}
                except yaml.YAMLError:
                    pass
            if section in ("root", "_top"):
                unknown = [k for k in patch if k not in known_top]
                if unknown:
                    raise HttpError(
                        400,
                        "unknown config keys: " + ", ".join(sorted(map(str, unknown))),
                        code="unknown_keys",
                    )
                _merge_patch(current, patch)
            else:
                sec = current.get(section)
                merged = dict(sec) if isinstance(sec, dict) else {}
                _merge_patch(merged, patch)
                current[section] = merged
            text = yaml.safe_dump(current, sort_keys=False)
        else:
            raise HttpError(400, "body must contain yaml or section+patch")

        # ---- validate before touching the file ------------------------------
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            msg = str(e).replace("\n", " ")
            raise HttpError(400, f"invalid YAML: {msg}") from e
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise HttpError(400, "config must be a YAML mapping")
        unknown_top = [k for k in data if k not in known_top]
        if unknown_top:
            raise HttpError(
                400,
                "unknown config keys: " + ", ".join(sorted(map(str, unknown_top))),
                code="unknown_keys",
            )
        for section, fields in known_sections.items():
            sec = data.get(section)
            if sec is None:
                continue
            if not isinstance(sec, dict):
                raise HttpError(400, f"config section {section!r} must be a mapping")
            unknown = [k for k in sec if k not in fields]
            if unknown:
                raise HttpError(
                    400,
                    f"unknown keys in {section}: " + ", ".join(map(str, sorted(unknown))),
                    code="unknown_keys",
                )

        # ---- write with backup ----------------------------------------------
        path = self._config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            import shutil

            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        # pid-suffixed tmp: two concurrent writers must not clobber each
        # other's shared "<file>.tmp" mid-write (lost-update race).
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

        # Reflect the new config in this app instance (env overrides still win).
        old_indexing = dict(getattr(self.cfg, "indexing", None) or {})
        try:
            self.cfg = load_config(str(path))
        except Exception:
            pass  # keep serving with the previous cfg; file is valid regardless

        # Applying indexing settings should START indexing (incremental build),
        # not wait for a program restart: previously a config PUT reloaded cfg
        # but never kicked a build, so "apply" left the index un-embedded and
        # only the restart-time startup_indexing hook did the work. Same
        # non-blocking thread + SSE events as startup; builds are incremental,
        # and a failed embed surfaces as an index.build error event.
        new_indexing = dict(getattr(self.cfg, "indexing", None) or {})
        if new_indexing != old_indexing and new_indexing.get("enabled"):
            try:
                self.startup_indexing()
            except Exception:
                pass
        return {"ok": True, "path": str(path), "backup": str(
            path.with_suffix(path.suffix + ".bak")) if (
            path.with_suffix(path.suffix + ".bak")).is_file() else None}

"""sable CLI — serve/run/traces/curate/datasets + train/models/deploy/iterate.

Loopback-only, zero API keys, no telemetry. See README.md for quickstart.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from . import __version__
from .core.agent_runner import AgentBackendError, create_runner
from .core.config import (ConfigError, SableConfig,
                          discover_external_server_url, load_config)
from .core.exitcodes import EXIT_CONFIG_INVALID, EXIT_CORRUPT_ARTIFACT
from .serving.supervisor import (
    ServeError,
    ServingSupervisor,
    healthcheck,
    install_signal_handlers,
)
from .store.db import TraceStore
from .store.paths import resolve_sable_home
from .store.recorder import make_recorder

logger = logging.getLogger("sable")


def _setup_logging(verbose: bool, config_path: Optional[str] = None) -> None:
    """Configure root logging: stderr console always, plus an optional
    rotating file handler and per-module levels from the `logging:` config
    section (level, file, max_bytes, backup_count, module_levels). A broken
    config must never break logging setup, so it degrades to the defaults."""
    level = logging.INFO if verbose else logging.WARNING
    sec: dict = {}
    cfg = None
    try:
        cfg = load_config(config_path)
        raw = getattr(cfg, "logging", None)
        if isinstance(raw, dict):
            sec = raw
    except Exception:  # noqa: BLE001 - bad config: keep default logging
        pass
    cfg_level = str(sec.get("level") or "").upper()
    if not verbose and cfg_level and hasattr(logging, cfg_level):
        level = getattr(logging, cfg_level)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    # Optional rotating file handler (relative names resolve under SABLE_HOME).
    log_file = sec.get("file")
    if isinstance(log_file, str) and log_file.strip():
        try:
            from logging.handlers import RotatingFileHandler

            p = Path(log_file).expanduser()
            if not p.is_absolute():
                p = resolve_sable_home(cfg) / p
            p.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                str(p),
                maxBytes=int(sec.get("max_bytes", 5_000_000)),
                backupCount=int(sec.get("backup_count", 3)),
                encoding="utf-8",
            )
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logging.getLogger().addHandler(handler)
        except Exception as e:  # noqa: BLE001 - file logging is best-effort
            print(f"warning: logging.file {log_file!r} unusable: {e}",
                  file=sys.stderr)
    # Per-module overrides, e.g. module_levels: {sable.serving: DEBUG}.
    mods = sec.get("module_levels")
    if isinstance(mods, dict):
        for name, lvl in mods.items():
            l = str(lvl).upper()
            if hasattr(logging, l):
                logging.getLogger(str(name)).setLevel(getattr(logging, l))
    # CLI status lines must appear immediately even when piped/redirected.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(line_buffering=True)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="Path to sable YAML config file")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sable", description=__doc__)
    parser.add_argument("--version", action="version", version=f"sable {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---------------------------------------------------------------- serve
    serve = sub.add_parser("serve", help="Spawn/serve a llama-server GGUF endpoint")
    _add_common(serve)
    serve.add_argument("--gguf", help="Path to a .gguf model file")
    serve.add_argument("--port", type=int, help="Port (0 = dynamic; default 0)")
    serve.add_argument("--context", type=int, help="Context length (-c); default 65536")
    serve.add_argument(
        "--attach",
        metavar="URL",
        help="Attach to a running OpenAI-compatible server instead of spawning "
        "(e.g. http://127.0.0.1:8888 for any OpenAI-compatible server)",
    )
    serve.add_argument(
        "--attach-api-key",
        metavar="KEY",
        help="Bearer token for --attach targets that require auth "
        "(env SABLE_ATTACH_API_KEY also works)",
    )
    serve.add_argument(
        "--auto",
        action="store_true",
        help="Serve the ACTIVE promoted model from the registry (resolves its "
        "GGUF artifact) instead of --gguf/config gguf_path",
    )
    serve.set_defaults(func=cmd_serve)

    # ------------------------------------------------------------------ run
    run = sub.add_parser("run", help="Run an agent task against the endpoint")
    _add_common(run)
    run.add_argument("--task", help="One-shot task prompt (omit for interactive REPL)")
    run.add_argument(
        "--attach",
        metavar="URL",
        help="Use a running OpenAI-compatible server instead of spawning llama-server",
    )
    run.add_argument(
        "--attach-api-key",
        metavar="KEY",
        help="Bearer token for --attach targets that require auth "
        "(env SABLE_ATTACH_API_KEY also works)",
    )
    run.add_argument("--gguf", help="Path to a .gguf model file (spawn mode)")
    run.add_argument("--port", type=int, help="Port for spawned llama-server (0 = dynamic)")
    # Backlog #8: batch trace generation with content-based resume.
    run.add_argument(
        "--batch",
        metavar="TASKS_JSONL",
        help="Run tasks sequentially from a JSONL file instead of --task/REPL. "
        "Each line is {\"task\": \"...\"} or a bare JSON string; blank lines "
        "tolerated (extra per-line keys ignored in v0). Traces are ordinary "
        "buffer records. Checkpoint: <home>/batches/<run-name>.json.",
    )
    run.add_argument(
        "--run-name",
        metavar="NAME",
        help="Batch checkpoint name (default: --batch file stem). "
        "Reused with --resume.",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="With --batch: skip tasks whose recorded trace content already "
        "matches (infra failures — connection/timeout/model_not_found — "
        "auto-retry; model failures count as done).",
    )
    run.add_argument(
        "--project",
        metavar="PROJECT_ID",
        help="Active project id (dashboard-created project record): its "
        "workspace_dir becomes the file-tool containment root and runs "
        "route to that project's codebase index — same behavior as "
        "dashboard chat runs. Unknown ids are an error.",
    )
    # WP12: bind a one-shot run to a persistent plan task (requires
    # plans.enabled). Seeds the first message with build_task_prompt and
    # stamps the trace's plan_task field. Named --plan-task because --task is
    # already the free-text prompt (see cmd_run comment).
    run.add_argument(
        "--plan-task",
        metavar="TASK_ID",
        help="WP12: start this run from plan task ID (e.g. t3): first message "
        "= build_task_prompt(task), trace stamped with plan_task. Requires "
        "plans.enabled and --project for non-default plans.",
    )
    run.set_defaults(func=cmd_run)

    # ------------------------------------------------------------------ plan
    # WP12: persistent per-project task graphs (core/plans.py). Every
    # subcommand takes an optional --project ID (default plan when omitted).
    plan = sub.add_parser(
        "plan",
        help="Persistent per-project task graphs (WP12)",
        description=(
            "Dependency-aware task graphs that survive runs and threads: "
            "add/list/show/next/depends/validate/set-status/export/import/"
            "expand, plus the guarded autonomous loop. State lives at "
            "<home>/plans/<project>.json. Requires config plans.enabled."),
    )
    psub = plan.add_subparsers(dest="plan_command", required=True)

    def _pcommon(p: argparse.ArgumentParser) -> None:
        _add_common(p)
        p.add_argument("--project", metavar="PROJECT_ID", default=None,
                       help="Project id whose plan to use (default plan "
                            "when omitted)")

    padd = psub.add_parser("add", help="Add a task")
    _pcommon(padd)
    padd.add_argument("title", help="Short imperative title")
    padd.add_argument("--description", default="", metavar="TEXT")
    padd.add_argument("--details", default="", metavar="TEXT")
    padd.add_argument("--test-strategy", default="", metavar="TEXT")
    padd.add_argument("--priority", choices=["high", "medium", "low"],
                      default="medium")
    padd.add_argument("--deps", default="", metavar="IDS",
                      help="Comma-separated task ids this depends on")
    padd.add_argument("--parent", default="", metavar="ID",
                      help="Parent task id (makes this a subtask)")
    padd.set_defaults(func=cmd_plan)

    plist = psub.add_parser("list", help="List tasks with status marks")
    _pcommon(plist)
    plist.set_defaults(func=cmd_plan)

    pshow = psub.add_parser("show", help="Show one task in full")
    _pcommon(pshow)
    pshow.add_argument("task_id", metavar="TASK_ID")
    pshow.set_defaults(func=cmd_plan)

    pnext = psub.add_parser("next", help="Next unblocked task (ranked)")
    _pcommon(pnext)
    pnext.set_defaults(func=cmd_plan)

    pdep = psub.add_parser(
        "depends", help="Wire/remove a dependency (refuses cycles/self-dep)")
    _pcommon(pdep)
    pdep.add_argument("task_id", metavar="TASK_ID")
    pdep.add_argument("dep_id", metavar="DEP_ID")
    pdep.add_argument("--remove", action="store_true",
                      help="Remove the dependency instead of adding it")
    pdep.set_defaults(func=cmd_plan)

    pval = psub.add_parser(
        "validate", help="Check for cycles + dangling refs (machine-readable)")
    _pcommon(pval)
    pval.set_defaults(func=cmd_plan)

    pss = psub.add_parser("set-status", help="Set a task's status")
    _pcommon(pss)
    pss.add_argument("task_id", metavar="TASK_ID")
    pss.add_argument("status", choices=["pending", "in_progress", "blocked",
                                        "done", "cancelled"])
    pss.add_argument("--note", default="", metavar="TEXT",
                     help="Timestamped line appended to task details")
    pss.set_defaults(func=cmd_plan)

    prm = psub.add_parser("remove", help="Remove a task (rewires deps)")
    _pcommon(prm)
    prm.add_argument("task_id", metavar="TASK_ID")
    prm.set_defaults(func=cmd_plan)

    pexp = psub.add_parser("export", help="Canonical JSON to stdout")
    _pcommon(pexp)
    pexp.set_defaults(func=cmd_plan)

    pimp = psub.add_parser(
        "import", help="Import a brief (text file, - for stdin, or --use-ai)")
    _pcommon(pimp)
    pimp.add_argument("file", metavar="FILE",
                      help="Brief file ('-' reads stdin); plain text lines "
                           "'title [deps=t1,t2] [priority=high]' or a JSON "
                           "task array. Binaries rejected.")
    pimp.add_argument("--use-ai", action="store_true",
                      help="Ask the served model to structure the brief "
                           "(one bounded call; occupies GPU briefly)")
    pimp.add_argument("--num-tasks", type=int, default=8, metavar="N",
                      help="Cap imported/generated tasks (default 8, max 32)")
    pimp.set_defaults(func=cmd_plan)

    pexp2 = psub.add_parser("expand", help="Decompose one task into subtasks")
    _pcommon(pexp2)
    pexp2.add_argument("task_id", metavar="TASK_ID")
    pexp2.add_argument("--num", type=int, default=3, metavar="N",
                       help="How many subtasks to generate (default 3)")
    pexp2.add_argument("--prompt", default="", metavar="TEXT",
                       help="Extra guidance for the model")
    pexp2.set_defaults(func=cmd_plan)

    ploop = psub.add_parser(
        "loop", help="Run tasks unattended (guarded; single-flight)")
    _pcommon(ploop)
    ploop.add_argument("--max-tasks", type=int, default=None, metavar="N",
                       help="Override plans.loop.max_tasks for this run")
    ploop.add_argument("--resume", action="store_true",
                       help="Continue from the project's .loop.json progress")
    ploop.set_defaults(func=cmd_plan)

    ploopst = psub.add_parser("loop-status", help="Show loop progress file")
    _pcommon(ploopst)
    ploopst.set_defaults(func=cmd_plan)

    # ---------------------------------------------------------------- traces
    traces = sub.add_parser("traces", help="Inspect the experience buffer")
    tsub = traces.add_subparsers(dest="traces_command", required=True)

    tlist = tsub.add_parser("list", help="List recorded traces (newest first)")
    _add_common(tlist)
    tlist.add_argument("--limit", type=int, default=20, help="Max rows (default 20)")
    tlist.add_argument(
        "--status",
        choices=["success", "failure", "partial", "unknown"],
        help="Filter by status",
    )
    tlist.add_argument(
        "--since",
        metavar="ISO8601",
        help="Only traces at/after this timestamp (e.g. 2026-09-01)",
    )
    tlist.set_defaults(func=cmd_traces_list)

    tshow = tsub.add_parser("show", help="Pretty-print a trace's JSONL record")
    _add_common(tshow)
    tshow.add_argument("trace_id", help="Trace id (from `sable traces list`)")
    tshow.set_defaults(func=cmd_traces_show)

    tstats = tsub.add_parser("stats", help="Buffer statistics")
    _add_common(tstats)
    tstats.set_defaults(func=cmd_traces_stats)

    tprune = tsub.add_parser(
        "prune",
        help="Delete old traces (index + transcripts) to bound buffer growth",
    )
    _add_common(tprune)
    tprune.add_argument(
        "--older-than", type=int, metavar="DAYS", default=None,
        help="Delete traces older than this many days (default: config "
             "recording.retention_days; one of the two is required)",
    )
    tprune.add_argument(
        "--include-curated", action="store_true",
        help="Also delete traces already claimed into a dataset "
             "(default: curated traces are kept — they are training data)",
    )
    tprune.add_argument(
        "--yes", action="store_true",
        help="Actually delete (without it this is a dry run listing counts)",
    )
    tprune.set_defaults(func=cmd_traces_prune)

    # ---------------------------------------------------------------- curate
    cur = sub.add_parser(
        "curate",
        help="Curate buffered traces into an SFT/DPO dataset",
        description=(
            "Select traces from the experience buffer (by score/time/profile), "
            "dedup prefix-overlapping conversations, and write a dataset "
            "directory (sft.jsonl, dpo.jsonl, manifest.json, train.yaml) under "
            "<home>/datasets/<dataset_id>/. Selected traces are claimed "
            "(curated=1) so later runs skip them; --rebuild <dataset_id> "
            "releases that dataset's claims and rewrites it."
        ),
    )
    _add_common(cur)
    cur.add_argument("--name", help="Human-friendly suffix for the dataset id")
    cur.add_argument("--min-score", type=float,
                     help="Minimum effective outcome score (default: config "
                          "curation.min_score, else 0.5)")
    cur.add_argument("--since", metavar="ISO8601",
                     help="Only traces at/after this timestamp (e.g. 2026-09-01)")
    cur.add_argument("--profile", metavar="PROFILE",
                     help="Filter by agent profile/backend (e.g. sable)")
    cur.add_argument("--model-alias", metavar="ALIAS",
                     help="Filter by model alias that served the traces")
    # WP12: project filter (same stamp `sable run --project` writes) — pairs
    # naturally with per-project plans: curate just this project's work.
    cur.add_argument("--project", metavar="ID",
                     help="Only traces stamped with project ID "
                          "(see `sable run --project`)")
    cur.add_argument("--include-failures", action="store_true",
                     help="Let failure traces into selection (DPO negatives; "
                          "they still must pass quality gates for SFT)")
    cur.add_argument("--limit", type=int, help="Max rows per output file")
    cur.add_argument("--format", dest="fmt", choices=["sft", "dpo", "both"],
                     default=None, help="Which files to emit (default both)")
    cur.add_argument("--rebuild", metavar="DATASET_ID",
                     help="Re-curate: release DATASET_ID's claims, rewrite it")
    # Backlog #12: opt-in LLM-as-judge scoring of borderline traces. Purely
    # flag-driven — no config key enables it; default makes zero LLM calls.
    cur.add_argument("--judge", type=int, metavar="N",
                     help="Score the top-N borderline traces (|score - "
                          "min_score| < curation.judge_band) with an "
                          "LLM-as-judge pass against the served model; "
                          "verdicts land as feedback rows source=judge "
                          "(precedence user > judge > heuristic). Costs GPU "
                          "time; the judged model grading its own traces is "
                          "BIASED — tiebreaker only, not ground truth.")
    cur.set_defaults(func=cmd_curate)

    # -------------------------------------------------------------- datasets
    ds = sub.add_parser("datasets", help="List/inspect/export curated datasets")
    dsub = ds.add_subparsers(dest="datasets_command", required=True)

    dlist = dsub.add_parser("list", help="List curated datasets (oldest first)")
    _add_common(dlist)
    dlist.set_defaults(func=cmd_datasets_list)

    dshow = dsub.add_parser("show", help="Print a dataset manifest summary")
    _add_common(dshow)
    dshow.add_argument("dataset_id", help="Dataset id (from `datasets list`)")
    dshow.set_defaults(func=cmd_datasets_show)

    # Review idea #6: version-to-version diff from manifest trace_ids + hashes.
    ddiff = dsub.add_parser(
        "diff", help="Diff two datasets (trace-id sets, file hashes, counts)")
    ddiff.add_argument("a", help="First dataset id (left side)")
    ddiff.add_argument("b", help="Second dataset id (right side)")
    ddiff.set_defaults(func=cmd_datasets_diff)

    dexport = dsub.add_parser("export", help="Copy a dataset out for external training")
    _add_common(dexport)
    dexport.add_argument("dataset_id", help="Dataset id to copy")
    dexport.add_argument("--to", required=True, metavar="DIR",
                         help="Destination directory (gets <dir>/<dataset_id>/)")
    dexport.set_defaults(func=cmd_datasets_export)

    # Backlog #14: import external datasets into the curate layout.
    dimport = dsub.add_parser(
        "import",
        help="Import an external dataset (alpaca/sharegpt/chatml, auto-detected)",
        description=(
            "Read a .jsonl (or .json array) file, auto-detect its format "
            "(conversations -> sharegpt, instruction+output -> alpaca, "
            "messages -> chatml), normalize rows to the internal ShareGPT "
            "shape and write a dataset dir (sft.jsonl + manifest with "
            "provenance: imported + train.yaml) that trains exactly like a "
            "curated one. Malformed rows are skipped and counted."
        ),
    )
    _add_common(dimport)
    dimport.add_argument("file", help="Path to .jsonl/.json dataset file")
    dimport.add_argument("--format", dest="fmt", default="auto",
                         choices=["auto", "alpaca", "sharegpt", "chatml"],
                         help="Force a format (default: auto-detect)")
    dimport.add_argument("--name", help="Human-friendly suffix for the dataset id")
    dimport.set_defaults(func=cmd_datasets_import)

    # Pass 4 #42: re-run the ingestion quality gate over existing datasets.
    dlint = dsub.add_parser(
        "lint",
        help="Re-check datasets for duplicates/conflicts (refreshes sidecars)",
        description=(
            "Re-runs the ingestion quality gate (pass 4 #42) over existing "
            "datasets: exact + normalized dedupe and trust-weighted conflict "
            "checks. Refreshes each dataset's hashes.json sidecar and "
            "rewrites flags.json. Flag-only — nothing is dropped or edited. "
            "Omits the name argument to lint every dataset."
        ),
    )
    _add_common(dlint)
    dlint.add_argument("name", nargs="?",
                       help="Dataset id (or substring) to lint; default: all")
    dlint.set_defaults(func=cmd_datasets_lint)

    # -------------------------------------------------------------- feedback
    fb = sub.add_parser(
        "feedback",
        help="Attach explicit feedback (-1..1) to a trace",
        description=(
            "Record feedback for a trace. --score takes -1..1; --fail and --ok "
            "are shorthands for -1 / +1. Target a trace by id or use --last."
        ),
    )
    _add_common(fb)
    fb.add_argument("trace_id", nargs="?", help="Trace id to rate")
    fb.add_argument("--last", action="store_true", help="Target the most recent trace")
    fb.add_argument("--score", type=float, help="Score in [-1, 1]")
    fb.add_argument("--fail", action="store_true", help="Shorthand for --score -1")
    fb.add_argument("--ok", action="store_true", help="Shorthand for --score +1")
    fb.add_argument("--note", help="Optional free-text comment")
    fb.set_defaults(func=cmd_feedback)

    # ------------------------------------------------------------- insights
    ins = sub.add_parser(
        "insights",
        help="Analytics over the buffer: score trend, failures, promotion deltas",
        description=(
            "Read-only insights engine (backlog #11): per-day score trend by "
            "profile/model, per-tool failure rates (bounded JSONL scan), "
            "failure-reason histogram (infra vs model), duration/token "
            "percentiles, activity patterns and before/after-promotion score "
            "deltas. Pure SQL/JSONL reads — no new deps, no cost estimation."
        ),
    )
    _add_common(ins)
    ins.add_argument("--days", type=int, default=30,
                     help="Look-back window in days (default 30)")
    ins.add_argument("--profile", metavar="PROFILE",
                     help="Filter by agent profile/backend (e.g. sable)")
    ins.add_argument("--json", action="store_true", dest="as_json",
                     help="Emit the raw report as JSON (machine use)")
    ins.set_defaults(func=cmd_insights)

    # ------------------------------------------------------- history (#31)
    hist = sub.add_parser(
        "history",
        help="Operation ledger: recent registry mutations (hash-chained)",
        description=(
            "Append-only operation ledger over registry mutations (pass 4 #31): "
            "promotes, rollbacks, archives, pins, loss writes and job "
            "transitions, each entry chained to its predecessor via SHA-256. "
            "--verify re-walks the chain from genesis."
        ),
    )
    _add_common(hist)
    hist.add_argument("--limit", type=int, default=20,
                      help="Max entries to show, newest first (default 20)")
    hist.add_argument("--verify", action="store_true",
                      help="Verify the hash chain and report the result")
    hist.add_argument("--intents", action="store_true",
                      help="Show recorded intents vs observed score deltas "
                           "(pass 4 #36) instead of the raw operation list")
    hist.set_defaults(func=cmd_history)

    # ---------------------------------------------------------------- models
    mo = sub.add_parser("models", help="Model registry: list/show/promote/rollback")
    mosub = mo.add_subparsers(dest="models_command", required=True)

    mlist = mosub.add_parser("list", help="List registered models (v1 first)")
    _add_common(mlist)
    mlist.set_defaults(func=cmd_models_list)

    mshow = mosub.add_parser("show", help="Show one model (lineage + validation)")
    _add_common(mshow)
    mshow.add_argument("model_id", help="Model id (e.g. v2)")
    mshow.set_defaults(func=cmd_models_show)

    mpromote = mosub.add_parser(
        "promote", help="Promote a validated model (becomes the active pointer)"
    )
    _add_common(mpromote)
    mpromote.add_argument("model_id", help="Model id to promote")
    mpromote.add_argument("--intent", metavar="ONE_LINER",
                          help="Why you expect this promotion to help "
                               "(pass 4 #36; recorded on the ledger, later "
                               "paired with the observed score delta by "
                               "`sable history --intents`)")
    mpromote.set_defaults(func=cmd_models_promote)

    mroll = mosub.add_parser(
        "rollback", help="Roll active model back to the previous promoted one"
    )
    _add_common(mroll)
    mroll.add_argument("--base-model", help="Only consider this base model")
    mroll.add_argument("--intent", metavar="ONE_LINER",
                       help="Why you are rolling back (pass 4 #36; ledgered "
                            "with the op, reviewable via `history --intents`)")
    mroll.set_defaults(func=cmd_models_rollback)

    # Backlog item #16: archive-don't-delete lifecycle + pin + janitor.
    marx = mosub.add_parser(
        "archive",
        help="Archive a terminal model (recoverable; moves artifacts under "
             "<home>/archive/)",
    )
    _add_common(marx)
    marx.add_argument("model_id", help="Model id to archive")
    marx.set_defaults(func=cmd_models_archive)

    mrst = mosub.add_parser(
        "restore", help="Restore an archived model (moves artifacts back)"
    )
    _add_common(mrst)
    mrst.add_argument("model_id", help="Model id to restore")
    mrst.set_defaults(func=cmd_models_restore)

    mpin = mosub.add_parser(
        "pin", help="Pin a model (janitor never archives it)"
    )
    _add_common(mpin)
    mpin.add_argument("model_id", help="Model id to pin")
    mpin.set_defaults(func=cmd_models_pin)

    munpin = mosub.add_parser("unpin", help="Unpin a model")
    _add_common(munpin)
    munpin.add_argument("model_id", help="Model id to unpin")
    munpin.set_defaults(func=cmd_models_unpin)

    mjan = mosub.add_parser(
        "janitor",
        help="Archive aged superseded/failed/rolled_back models (never pinned)",
        description=(
            "Disk janitor (backlog #16): moves terminal models older than "
            "--days into <home>/archive/ — never deletes, never touches "
            "pinned or non-terminal models. State in <home>/janitor_state.json."
        ),
    )
    _add_common(mjan)
    mjan.add_argument("--days", type=int, help="Retention age in days (default 30)")
    mjan.add_argument(
        "--disk-days", type=int, default=None,
        help="Delete operational files (worker/server logs, chat uploads, "
             "spillover, batch checkpoints) older than N days (default from "
             "config janitor.disk_retention_days, else 90; 0 disables)")
    mjan.add_argument("--dry-run", action="store_true",
                      help="List what WOULD be archived and bytes freed; touch nothing")
    mjan.set_defaults(func=cmd_models_janitor)

    # ------------------------------------------------------------------ hub
    hb = sub.add_parser(
        "hub",
        help="Resource hub: search / download / list / delete models, "
             "datasets and MCP servers (Hugging Face)",
        description=(
            "Search Hugging Face across models, datasets and MCP servers "
            "(Spaces), browse GGUF quant variants, download into the models "
            "folder or hub roots (HF hub-cache layout, resumable) and manage "
            "what's installed. Gated repos: set HF_TOKEN."
        ),
    )
    hbsub = hb.add_subparsers(dest="hub_command", required=True)

    def _kind_arg(p):
        p.add_argument("--kind", default="models",
                       choices=["models", "datasets", "spaces", "mcp"],
                       help="Resource kind (mcp is an alias for spaces)")
        return p

    hsearch = _kind_arg(hbsub.add_parser(
        "search", help="Search the hub by text"))
    _add_common(hsearch)
    hsearch.add_argument("query", nargs="*", help="Search terms (empty = popular)")
    hsearch.add_argument("--limit", type=int, default=20)
    hsearch.set_defaults(func=cmd_hub_search)

    hvar = _kind_arg(hbsub.add_parser(
        "variants", help="List quant variants for a repo"))
    _add_common(hvar)
    hvar.add_argument("repo", help="Repo id (e.g. testorg/Qwen3-8B-GGUF)")
    hvar.set_defaults(func=cmd_hub_variants)

    hdl = _kind_arg(hbsub.add_parser(
        "download",
        help="Download a resource (models: default quant variant)"))
    _add_common(hdl)
    hdl.add_argument("repo", help="Repo id")
    hdl.add_argument("--variant", default="",
                     help="Quant variant name (see `sable hub variants`)")
    hdl.set_defaults(func=cmd_hub_download)

    hls = _kind_arg(hbsub.add_parser(
        "list", help="List resources downloaded via the hub"))
    _add_common(hls)
    hls.set_defaults(func=cmd_hub_list)

    hrmdir = _kind_arg(hbsub.add_parser(
        "delete", help="Delete a downloaded hub resource"))
    _add_common(hrmdir)
    hrmdir.add_argument("repo", help="Repo id to remove from the hub folder")
    hrmdir.set_defaults(func=cmd_hub_delete)

    # ---------------------------------------------------------------- train
    tr = sub.add_parser(
        "train",
        help="Queue/inspect training jobs (single-GPU FIFO scheduler)",
        description=(
            "Submit SFT/DPO training for a curated dataset, or check/cancel "
            "jobs. One job runs at a time; submissions while one runs simply "
            "queue (FIFO). GPU work happens in a worker subprocess."
        ),
    )
    trsub = tr.add_subparsers(dest="train_command", required=True)

    tsubmit = trsub.add_parser("start", help="Queue a training job for a dataset")
    _add_common(tsubmit)
    tsubmit.add_argument("--dataset", required=True, help="Dataset id to train on")
    tsubmit.add_argument("--kind", choices=["sft", "dpo", "grpo"], default="sft")
    tsubmit.add_argument("--dry-run", action="store_true",
                         help="Validate config + dataset without GPU")
    tsubmit.add_argument("--wait", action="store_true",
                         help="If another job is running, block until this one "
                              "runs (drains the FIFO queue) instead of exiting")
    tsubmit.add_argument(
        "--depends-on", action="append", default=[], metavar="JOB_ID",
        dest="depends_on",
        help="Wait for another queued/running job to finish successfully "
             "before this one may start (pass 4 #43; repeatable). The "
             "dependency must already exist. If it ends error/cancelled, "
             "this job fails with 'dependency <id> <state>'.")
    tsubmit.add_argument(
        "--config-key", action="append", default=[], metavar="K=V",
        dest="config_key",
        help="Override a resolved train.yaml knob by dotted path, repeatable "
             "(backlog #13), e.g. --config-key training.lora_r=32 "
             "--config-key lora.use_dora=true. Values typed true/false, int, "
             "float, JSON list/object, else string. Applied to the dataset's "
             "train.yaml copy before the worker runs; logged for provenance.")
    tsubmit.set_defaults(func=cmd_train_start)

    tstat = trsub.add_parser("status", help="Job status (all, or one job id)")
    _add_common(tstat)
    tstat.add_argument("job_id", nargs="?", help="Job id (omit for all jobs)")
    tstat.add_argument("--tail", type=int, default=8,
                       help="Show last N worker log lines (single job)")
    tstat.set_defaults(func=cmd_train_status)

    tcan = trsub.add_parser("cancel", help="Cancel a queued/running job (SIGTERM)")
    _add_common(tcan)
    tcan.add_argument("job_id", help="Job id to cancel")
    tcan.set_defaults(func=cmd_train_cancel)

    tres = trsub.add_parser(
        "resume",
        help="Re-run an interrupted job from its newest checkpoint",
        description=(
            "Queue a fresh job for a cancelled/crashed/errored training job's "
            "dataset, resuming from the newest resumable checkpoint in that run's "
            "output_dir (adapter weights + unfinished trainer state). Nothing is "
            "deleted; this submits a NEW job id. A completed job has nothing to "
            "resume and starts clean."
        ),
    )
    _add_common(tres)
    tres.add_argument("job_id", help="Interrupted training job id to resume")
    tres.add_argument("--wait", action="store_true",
                      help="Block until the resumed job runs (drain the FIFO queue)")
    tres.set_defaults(func=cmd_train_resume)

    tchk = trsub.add_parser(
        "auto-check",
        help="--auto trigger check: enough good unclaimed traces?",
        description=(
            "Prints whether the buffer has >= training.auto_min_traces unclaimed "
            "traces with mean score >= training.auto_min_score. With --trigger, "
            "runs the full iterate loop (curate->train->export->validate->promote) "
            "when the check AND the resource gate pass — this is the command an "
            "external scheduler/cron should call for autonomous retraining."
        ),
    )
    _add_common(tchk)
    tchk.add_argument(
        "--trigger",
        action="store_true",
        help="When the check passes, run `sable iterate` now (cron entrypoint; "
             "exits 0 without acting when the gate does not pass)",
    )
    tchk.add_argument("--dry-run", action="store_true",
                      help="With --trigger: dry-run the iterate loop (no GPU)")
    tchk.set_defaults(func=cmd_train_auto_check)

    # ---------------------------------------------------------------- deploy
    dep = sub.add_parser(
        "deploy",
        help="Hot-swap the running llama-server to a registered model's GGUF",
        description=(
            "Stops the sable-owned llama-server (if any), starts one on the "
            "model's GGUF, and re-points an in-process runner via switch_model. "
            "Daemonless v0: restart `sable serve --auto` to pick up new models."
        ),
    )
    _add_common(dep)
    dep.add_argument("model_id", help="Model id to serve (must have a GGUF)")
    dep.set_defaults(func=cmd_deploy)

    # ------------------------------------------------------------ dashboard
    dash = sub.add_parser(
        "dashboard",
        help="Serve the loop dashboard (web UI) on 127.0.0.1",
        description=(
            "Local control-panel web UI: traces + feedback, datasets, model "
            "registry with promote/rollback, job queue with logs, and curate/"
            "train/iterate controls. Loopback-only (127.0.0.1); no external "
            "assets — works offline."
        ),
    )
    _add_common(dash)
    dash.add_argument("--port", type=int,
                      help="Port on 127.0.0.1 (0 = dynamic; default 0)")
    dash.add_argument(
        "--token",
        metavar="SECRET",
        help="Require this token on /api/* calls (X-Sable-Token header or "
             "?token= query). Default when unset: a generated secret persisted "
             "at <home>/dashboard.token. Also config dashboard.token / "
             "SABLE_DASHBOARD_TOKEN.",
    )
    dash.add_argument(
        "--no-token",
        action="store_true",
        help="Disable /api/* token auth (single-user trust). Loopback + "
             "Origin checks still apply.",
    )
    dash.set_defaults(func=cmd_dashboard)

    # -------------------------------------------------------------- iterate
    it = sub.add_parser(
        "iterate",
        help="Full loop: curate -> train -> export -> validate -> promote",
        description=(
            "One call closes the train->deploy->iterate loop. Curation claims "
            "unclaimed traces, a training job produces an adapter, the export "
            "job emits a GGUF, the validation gate replays golden tasks, and a "
            "passing candidate is promoted (previous model superseded; rollback "
            "stays one command away)."
        ),
    )
    _add_common(it)
    it.add_argument("--name", help="Name suffix for the curated dataset id")
    it.add_argument("--kind", choices=["sft", "dpo", "grpo"], default="sft")
    it.add_argument("--dataset", help="Skip curation; train on this dataset id")
    it.add_argument("--dry-run", action="store_true",
                    help="Validate config + dataset only (no GPU, no export)")
    it.add_argument(
        "--config-key", action="append", default=[], metavar="K=V",
        dest="config_key",
        help="Override a resolved train.yaml knob by dotted path, repeatable "
             "(backlog #13), e.g. --config-key training.num_epochs=1. Same "
             "typing/precedence rules as `train start --config-key`.")
    it.add_argument(
        "--intent", metavar="ONE_LINER",
        help="What this retrain run expects to improve (pass 4 #36); stored "
             "on the register/promote ledger entries and reviewable via "
             "`sable history --intents`.")
    it.set_defaults(func=cmd_iterate)

    # -------------------------------------------------------------- evaluate
    ev = sub.add_parser(
        "evaluate",
        help="Post-train evaluation jobs (shadow-baseline comparison, pass 4 "
             "#32; cross-profile transfer probe, pass 4 #37)",
    )
    esub = ev.add_subparsers(dest="evaluate_command", required=True)

    esh = esub.add_parser(
        "shadow",
        help="Queue a shadow-baseline replay of a candidate against the active "
             "model (judge-scored; warn-only regression flag)",
    )
    _add_common(esh)
    esh.add_argument("model_id", help="Candidate model id to evaluate (e.g. v2)")
    esh.add_argument("--profile", default=None,
                     help="Agent profile whose recent high-feedback traces "
                          "form the replay set")
    esh.add_argument("--n", type=int, default=8,
                     help="Replay-set size (default 8; judge calls are 2xN)")
    esh.add_argument("--judge-model", default=None,
                     help="Model name to attribute judge calls to "
                          "(default: the served model per side)")
    esh.add_argument("--wait", action="store_true",
                     help="Block until the job runs (drain the FIFO queue)")
    esh.set_defaults(func=cmd_evaluate_shadow)

    etr = esub.add_parser(
        "transfer",
        help="Probe a candidate adapter against ANOTHER profile's replay set "
             "before pointing that profile at it (pass 4 #37)",
    )
    _add_common(etr)
    etr.add_argument("model_id", help="Candidate model id to probe (e.g. v2)")
    etr.add_argument("--profile", required=True,
                     help="Target profile P whose replay set the candidate "
                          "runs against")
    etr.add_argument("--baseline", default=None, metavar="MODEL_ID",
                     help="Baseline model id (default: the registry's current "
                          "active model — what P runs today; when no active "
                          "model exists the probe runs candidate-only and "
                          "stores absolute scores with baseline_id null)")
    etr.add_argument("--n", type=int, default=6,
                     help="Replay-set size (default 6; judge calls are up to 2xN)")
    etr.add_argument("--judge-model", default=None,
                     help="Model name to attribute judge calls to "
                          "(default: the served model per side)")
    etr.add_argument("--wait", action="store_true",
                     help="Block until the job runs (drain the FIFO queue)")
    etr.set_defaults(func=cmd_evaluate_transfer)

    # --------------------------------------------------------------- evolve
    # Shape: `sable evolve <profile> [--goldens FILE] [--k 3 --generations 2]
    # [--wait]` runs a search; `sable evolve apply|dismiss <job_id>` are the
    # gated follow-ups. One parser with a positional target + manual dispatch
    # (argparse can't make a subcommand default AND take bare positionals).
    evo = sub.add_parser(
        "evolve",
        help="Prompt/profile evolution loop on golden tasks (pass 4 #39): "
             "search prompt variants, output is ALWAYS a gated diff proposal",
    )
    _add_common(evo)
    evo.add_argument("target", help="Profile name to evolve, or the "
                                    "apply/dismiss subcommand")
    evo.add_argument("job_id", nargs="?", default=None,
                     help="Job id (with apply/dismiss)")
    evo.add_argument("--goldens", default=None, metavar="FILE",
                     help="Golden task file (validate_tool_calls format; JSON "
                          "or YAML)")
    evo.add_argument("--k", type=int, default=3,
                     help="Mutations per parent per generation (default 3)")
    evo.add_argument("--generations", type=int, default=2,
                     help="Generations G (default 2)")
    evo.add_argument("--call-budget", type=int, default=None,
                     help="Hard LLM call budget (capped by "
                          "evolution.max_call_budget)")
    evo.add_argument("--wait", action="store_true",
                     help="Block until the job runs (drain the FIFO queue)")
    evo.set_defaults(func=cmd_evolve)

    # ---------------------------------------------------------------- rft
    # Rejection-sampling fine-tuning (STaR): K completions per prompt from the
    # SERVED model, verified-correct rows only, emitted as a ShareGPT SFT
    # dataset. Queued job (kind `rft`, in-process like shadow/evolve); needs a
    # healthy serving endpoint and rft.enabled: true.
    rft = sub.add_parser(
        "rft",
        help="Rejection-sampling fine-tuning data: sample K completions per "
             "prompt from the served model, keep only verified-correct ones, "
             "emit a ShareGPT SFT dataset (needs rft.enabled + a live server)",
    )
    _add_common(rft)
    rft.add_argument("--seeds", nargs="*", default=None, metavar="PROMPT",
                     help="Explicit seed prompts to sample (in addition to "
                          "buffer/golden selection)")
    rft.add_argument("--goldens", default=None, metavar="FILE",
                     help="Golden task file (validate format): each task's "
                          "prompt is sampled and deterministically verified")
    rft.add_argument("--k", type=int, default=None,
                     help="Candidates per prompt (default rft.sample_k = 4)")
    rft.add_argument("--max-prompts", type=int, default=None,
                     help="Prompts this run (default rft.max_prompts = 16)")
    rft.add_argument("--name", default=None,
                     help="Dataset name prefix (default 'rft')")
    rft.add_argument("--wait", action="store_true",
                     help="Block until the job runs (drain the FIFO queue)")
    rft.set_defaults(func=cmd_rft)

    # ---------------------------------------------------------- mine-goldens
    # Auto-mine verifiable golden tasks from 👍 traces (curate/mine_goldens.py).
    # Queued job (kind `mine_goldens`, in-process like rft/evolve); needs
    # goldens_mine.enabled: true. Mined goldens join validation.golden_tasks
    # (promotion gate) and RFT golden selection automatically.
    mg = sub.add_parser(
        "mine-goldens",
        help="Distill 👍-thumbed traces into golden validation tasks with "
             "self-checked deterministic verifiers (needs goldens_mine.enabled)",
    )
    _add_common(mg)
    mg.add_argument("--max", type=int, default=None, metavar="N",
                    help="Goldens mined this run (default "
                         "goldens_mine.max_per_run = 8)")
    mg.add_argument("--use-judge", action="store_true",
                    help="Also ask the served model for extra substring checks "
                         "(proposals still pass the deterministic self-check; "
                         "needs a healthy endpoint)")
    mg.add_argument("--wait", action="store_true",
                    help="Block until the job runs (drain the FIFO queue)")
    mg.set_defaults(func=cmd_mine_goldens)

    # ------------------------------------------------------------ playbooks
    pb = sub.add_parser(
        "playbooks",
        help="Trajectory → playbook distillation (pass 4 #38): scan traces, "
             "manage the candidate/active/retired lifecycle, export metadata",
    )
    pbsub = pb.add_subparsers(dest="playbooks_command", required=True)

    pbscan = pbsub.add_parser(
        "scan", help="List recurring tool chains above the threshold")
    _add_common(pbscan)
    pbscan.add_argument("--profile", default=None,
                        help="Only traces from this agent profile")
    pbscan.add_argument("--min-occurrences", type=int, default=3,
                        help="Minimum successful traces per chain (default 3)")
    pbscan.add_argument("--window-days", type=int, default=30,
                        help="Look-back window in days (default 30)")
    pbscan.add_argument("--create", action="store_true",
                        help="Also create candidate playbooks for the chains "
                             "(idempotent; distills the recipe from examples)")
    pbscan.set_defaults(func=cmd_playbooks_scan)

    pblist = pbsub.add_parser(
        "candidates", help="List candidate playbooks")
    _add_common(pblist)
    pblist.add_argument("--profile", default=None)
    pblist.set_defaults(func=lambda a: _cmd_playbooks_list(a, "candidate"))

    pbl = pbsub.add_parser("list", help="List all playbooks (any status)")
    _add_common(pbl)
    pbl.add_argument("--profile", default=None)
    pbl.set_defaults(func=lambda a: _cmd_playbooks_list(a, None))

    pbp = pbsub.add_parser(
        "promote", help="candidate → active (explicit confirmation; --check "
                        "runs the bounded replay gate first)")
    _add_common(pbp)
    pbp.add_argument("id", help="Playbook id (from scan --create / candidates)")
    pbp.add_argument("--check", action="store_true",
                     help="Run the judge replay gate (needs a served "
                          "endpoint); promotion only on mean >= 0.7")
    pbp.add_argument("--n", type=int, default=6,
                     help="Replay-set size for --check (default 6)")
    pbp.set_defaults(func=cmd_playbooks_promote)

    pbas = pbsub.add_parser(
        "assess", help="Correlational regression check for ACTIVE playbooks "
                       "(mean outcome score after vs before promotion; "
                       "regressions retire the playbook — honest evidence, "
                       "not proof)")
    _add_common(pbas)
    pbas.add_argument("id", nargs="?", default=None,
                      help="Assess one playbook (default: all active ones)")
    pbas.set_defaults(func=cmd_playbooks_assess)

    pbr = pbsub.add_parser("retire", help="active/candidate → retired (kept)")
    _add_common(pbr)
    pbr.add_argument("id")
    pbr.set_defaults(func=cmd_playbooks_retire)

    pbsh = pbsub.add_parser("show", help="Show one playbook (recipe + history)")
    _add_common(pbsh)
    pbsh.add_argument("id")
    pbsh.set_defaults(func=cmd_playbooks_show)

    pbx = pbsub.add_parser(
        "export", help="Write recipe + chain + source trace ids as METADATA "
                       "(never fabricated training rows)")
    _add_common(pbx)
    pbx.add_argument("id")
    pbx.add_argument("--out", default=None,
                     help="Output path (default <home>/datasets/_playbooks/"
                          "<id>.json)")
    pbx.set_defaults(func=cmd_playbooks_export)

    # --------------------------------------------------------------- doctor
    doc = sub.add_parser(
        "doctor",
        help="Diagnose the runtime: GPUs, llama.cpp backend, binary, build tools",
        description=(
            "Read-only environment check: detects the GPU runtime (CUDA/ROCm/"
            "Metal/Vulkan/CPU), reports which llama-server would be used and "
            "how it would be launched, and checks whether auto-setup of "
            "llama.cpp is possible. Safe to run anytime."
        ),
    )
    _add_common(doc)
    # WP7: --env runs the environment MEASUREMENT doctor (disk/RAM/VRAM
    # presets + next commands, sable/doctor.py — kimi-k3-doctor pattern,
    # re-implemented inline per D-12). Without the flag the runtime diagnosis
    # below stays byte-identical to pre-WP7.
    doc.add_argument("--env", action="store_true", dest="env_doctor",
                     help="Environment measurement report (disk/RAM/VRAM "
                          "presets, config, ports, stale state) instead of "
                          "the runtime GPU/backend diagnosis")
    doc.add_argument("--fix", action="store_true",
                     help="With --env: remove stale instance state files "
                          "whose pid is verified gone (ownership rule "
                          "respected; foreign pids are never touched)")
    doc.set_defaults(func=cmd_doctor)

    # ---------------------------------------------------------------- setup
    st = sub.add_parser(
        "setup",
        help="One-click install: build llama.cpp for this machine's GPU",
        description=(
            "Clone and build llama.cpp into <SABLE_HOME>/llama.cpp with the "
            "backend matching the detected hardware (CUDA/ROCm/Metal/Vulkan/"
            "CPU). Needs git, cmake and a C/C++ compiler. Existing system "
            "installs (PATH, /opt/llama.cpp) are preferred automatically — "
            "this is only needed when none exists."
        ),
    )
    _add_common(st)
    st.add_argument(
        "--backend",
        choices=["auto", "cuda", "rocm", "metal", "vulkan", "cpu"],
        default="auto",
        help="Force a llama.cpp backend instead of auto-detection (default auto)",
    )
    st.add_argument(
        "--repo", metavar="URL",
        help="llama.cpp git repo to build from (default upstream ggml-org; "
             "e.g. https://github.com/Lychee-Technology/llama-cpp-for-strix-halo "
             "for AMD Strix Halo builds)",
    )
    st.add_argument(
        "--ref", metavar="REF",
        help="Branch or tag to build (default master; forks often carry a "
             "dedicated branch — check the repo's Releases page)",
    )
    st.set_defaults(func=cmd_setup)

    # --------------------------------------------------------------- update
    upd = sub.add_parser(
        "update",
        help="Check/update the managed llama.cpp build (and show sable version)",
        description=(
            "Compares the managed llama.cpp install's recorded commit with its "
            "remote ref and rebuilds when behind (or when --repo/--ref/--backend "
            "switch the source). With --check only reports. This is sable's "
            "llama.cpp update system; the sable package itself updates via pip."
        ),
    )
    _add_common(upd)
    upd.add_argument("--check", action="store_true",
                     help="Only report whether an update is available")
    upd.add_argument(
        "--backend",
        choices=["auto", "cuda", "rocm", "metal", "vulkan", "cpu"],
        default=None,
        help="Rebuild with a different backend (switches the build in place)",
    )
    upd.add_argument("--repo", metavar="URL",
                     help="Switch to a different llama.cpp repo and rebuild")
    upd.add_argument("--ref", metavar="REF",
                     help="Switch branch/tag and rebuild")
    upd.set_defaults(func=cmd_update)

    # ---------------------------------------------------------------- launch
    lch = sub.add_parser(
        "launch",
        help="One-click start: dashboard control UI in the browser",
        description=(
            "Start the loopback dashboard as a background process and open it "
            "in the default browser. Idempotent: when a dashboard already runs "
            "on this home, just prints its URL. `sable launch --stop` stops "
            "the launched instance (never foreign processes)."
        ),
    )
    _add_common(lch)
    lch.add_argument("--port", type=int, help="Dashboard port (0 = dynamic)")
    lch.add_argument("--no-browser", action="store_true",
                     help="Don't open a browser (print the URL only)")
    lch.add_argument("--stop", action="store_true", help="Stop the launched dashboard")
    lch.set_defaults(func=cmd_launch)

    # ----------------------------------------------------------------- reset
    rst = sub.add_parser(
        "reset",
        help="Restore sable to a fresh state (wipe SABLE_HOME)",
        description=(
            "Delete everything sable created: experience buffer, datasets, "
            "model registry artifacts, logs, serve state, spillover cache and "
            "the managed llama.cpp build — back to a freshly installed program. "
            "Only ever touches SABLE_HOME (never foreign installs or system "
            "installs). Stops the sable-owned llama-server/dashboard first. "
            "Requires --yes; use --keep-models to spare model artifacts."
        ),
    )
    _add_common(rst)
    rst.add_argument("--yes", action="store_true",
                     help="Confirm: actually delete (without it, dry-run listing only)")
    rst.add_argument("--keep-models", action="store_true",
                     help="Keep model artifacts: models/, archive/ and the "
                          "registry DB (engine builds are always wiped)")
    rst.set_defaults(func=cmd_reset)

    # ----------------------------------------------------------------- bench
    bch = sub.add_parser(
        "bench",
        help="Measure prompt/generation tokens/sec (noise floor + token identity)",
        description=(
            "Bench harness (WP1): measures prompt-processing and generation "
            "tokens/sec against a supervised llama-server endpoint over N "
            "repeated temperature-0 runs, reporting per-run values plus "
            "median/min/max and run-to-run spread ((max-min)/median). "
            "--noise-floor persists a baseline record later work packages use "
            "as the threshold for real effects. Speed claims are only VALID "
            "when generated tokens are identical between the runs compared; "
            "divergence is reported explicitly. Results persist as JSON under "
            "<SABLE_HOME>/bench/ (env SABLE_BENCH_DIR overrides)."
        ),
    )
    _add_common(bch)
    bch.add_argument("--runs", type=int, default=None,
                     help="Repeated runs (default 5)")
    bch.add_argument("--max-tokens", type=int, default=None,
                     help="Max generated tokens per prompt (default 64)")
    bch.add_argument("--label", default="",
                     help="Label for the persisted record (comparison key)")
    bch.add_argument("--prompts-file", metavar="PATH",
                     help="Prompt set: one prompt per line (# comments ok); "
                     "default is a small built-in deterministic fixture")
    bch.add_argument("--noise-floor", action="store_true",
                     help="Record this run as the noise-floor baseline "
                     "(kind=noise_floor) instead of a regular bench")
    bch.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"),
                     help="Compare two persisted records by --label or JSON "
                     "path; prints the speedup and the token-identity verdict "
                     "(a speedup that changed tokens is INVALID)")
    bch.add_argument("--list", action="store_true",
                     help="List persisted bench records (no measurement)")
    bch.add_argument("--attach", metavar="URL",
                     help="Bench a running OpenAI-compatible server instead "
                     "of the configured/spawned one (loopback only)")
    bch.set_defaults(func=cmd_bench)

    # -------------------------------------------------------------- calibrate
    cal = sub.add_parser(
        "calibrate",
        help="Per-machine autotune sweep over run_settings knobs (WP3)",
        description=(
            "Calibrate (WP3): coordinate-wise sweep of llama-server knobs "
            "already mapped in run_settings (kv_cache_type, flash_attn, "
            "gpu_memory_pct -> -ngl, batch/ubatch, parallel slots, "
            "speculative mode) against this machine + model. A candidate is "
            "adopted only when it beats the current best by MORE THAN "
            "max(3%, measured noise floor) AND generates identical tokens at "
            "temperature 0 (token-identity rule from `sable bench`). The "
            "noise floor is read from a persisted bench record or measured + "
            "persisted on first use. Profiles save under <SABLE_HOME>/"
            "calibrate/ keyed by machine-id + model hash (env "
            "SABLE_CALIBRATE_DIR overrides). Serving honors a profile ONLY "
            "with `serve.calibration: auto`; explicit run_settings always win."
        ),
    )
    _add_common(cal)
    cal.add_argument("--quick", action="store_true",
                     help="Preset with fewer knobs/values (bounded first pass)")
    cal.add_argument("--knobs", metavar="K1,K2",
                     help="Comma-separated knob subset "
                     "(kv_cache_type,flash_attn,gpu_memory_pct,batch_size,"
                     "ubatch_size,parallel,speculative_type)")
    cal.add_argument("--runs", type=int, default=None,
                     help="Repeated bench runs per candidate (default 3)")
    cal.add_argument("--model", metavar="REF",
                     help="GGUF path to calibrate (default: configured gguf_path)")
    cal.add_argument("--attach", metavar="URL",
                     help="Measure against a running loopback server instead "
                     "of spawning per candidate. NOTE: knob values are NOT "
                     "applied to an attached server — the profile records "
                     "endpoint_mode=attach so adopted values stay honest "
                     "(harness/testing use; real calibration spawns).")
    cal.add_argument("--list", action="store_true",
                     help="List stored calibration profiles (no measurement)")
    cal.add_argument("--show", action="store_true",
                     help="Print the stored profile for this machine+model "
                     "(or the newest stored one)")
    cal.add_argument("--clear", action="store_true",
                     help="Delete all stored calibration profiles")
    cal.set_defaults(func=cmd_calibrate)

    # ------------------------------------------------------------- instances
    instp = sub.add_parser(
        "instances",
        help="Named serving instances: status / load / unload (WP5)",
        description=(
            "Explicit control over managed llama-server instances. `status` "
            "lists every instance with its WP5 load state (loaded | unloaded "
            "| unloading | stopped). `load` (re)launches one by id, `unload` "
            "gracefully stops it and records the unloaded state so a later "
            "load — or lazy_load when serve.lazy_load: true — restarts it. "
            "Ownership rule holds throughout: only processes sable spawned "
            "(or recorded as ours) are ever stopped."
        ),
    )
    isub = instp.add_subparsers(dest="instances_command", required=True)
    for name, fn in (("status", cmd_instances_status),
                     ("load", cmd_instances_load),
                     ("unload", cmd_instances_unload)):
        ip = isub.add_parser(name, help=f"{name} serving instances")
        _add_common(ip)
        if name != "status":
            ip.add_argument("instance_id",
                            help="Instance id (see `sable instances status`)")
        ip.set_defaults(func=fn)

    # ------------------------------------------------------------------- mcp
    mcp = sub.add_parser(
        "mcp",
        help="Serve the MCP control plane over stdio (JSON-RPC 2.0, WP6)",
        description=(
            "Model Context Protocol server exposing sable's existing "
            "operations (status/instances/models/logs + gated curate/train/"
            "validate/promote/bench) as typed tools. Transport is MCP's "
            "canonical stdio: newline-delimited JSON-RPC on stdin/stdout — "
            "NO socket is opened, so the loopback-only rule holds by "
            "construction. Mutating tools are plan-then-confirm: the first "
            "call returns a plan + single-use token; presenting the token "
            "executes. Stdlib-only (no MCP SDK). Client config example: "
            '{"command": "python", "args": ["-m", "sable", "mcp"]}.'
        ),
    )
    _add_common(mcp)
    mcp.set_defaults(func=cmd_mcp)

    return parser


def _apply_overrides(cfg: SableConfig, args: argparse.Namespace) -> None:
    if getattr(args, "auto", False):
        cfg.auto_model = True
    if getattr(args, "gguf", None):
        cfg.gguf_path = args.gguf
    if getattr(args, "port", None) is not None:
        cfg.port = args.port
    if getattr(args, "context", None):
        cfg.context_length = args.context
    if getattr(args, "attach", None):
        cfg.attach_url = args.attach
    if getattr(args, "attach_api_key", None):
        cfg.attach_api_key = args.attach_api_key


def _resolve_endpoint(cfg: SableConfig, spawn_if_needed: bool):
    """Return (base_url, model_name, supervisor|None). Spawns llama-server when needed."""
    supervisor: Optional[ServingSupervisor] = None

    attach = cfg.effective_attach_url
    if not attach and cfg.attach_url is None:
        # Optional external-server discovery: only probes when the user
        # explicitly sets $SABLE_EXTERNAL_SERVER_URL (opt-in). sable itself
        # is standalone — with nothing configured it simply uses spawn mode.
        # NOTE: an external server may require a bearer token — discovery only
        # checks /api/health, so also verify the OpenAI surface before use.
        if not cfg.gguf_path:
            ext = discover_external_server_url()
            if ext and healthcheck(
                ext.rstrip("/") + "/v1", api_key=cfg.attach_api_key
            ):
                logger.info("Using configured external server at %s", ext)
                attach = ext.rstrip("/") + "/v1"
            elif ext:
                print(
                    f"note: found a server at {ext} but GET /v1/models failed "
                    "(it may require auth — set SABLE_ATTACH_API_KEY or pass "
                    "--attach-api-key). Falling back to spawn mode.",
                    file=sys.stderr,
                )

    if attach:
        if not healthcheck(attach, api_key=cfg.attach_api_key):
            raise ServeError(
                f"Attach target {attach} is not healthy (GET /models failed). "
                + (
                    "If it requires auth, pass --attach-api-key or set "
                    "SABLE_ATTACH_API_KEY."
                    if not cfg.attach_api_key
                    else "Check the API key."
                )
            )
        import httpx

        model = cfg.model_name
        try:
            headers = (
                {"Authorization": f"Bearer {cfg.attach_api_key}"}
                if cfg.attach_api_key
                else {}
            )
            r = httpx.get(attach.rstrip("/") + "/models", timeout=10, headers=headers)
            data = r.json().get("data") or []
            if data and not model:
                model = data[0].get("id")
        except Exception:
            pass
        return attach, (model or "attached-model"), None

    if cfg.gguf_path and spawn_if_needed:
        supervisor = ServingSupervisor(cfg)
        handle = supervisor.start()
        return handle.base_url, handle.model_name, supervisor

    raise ServeError(
        "No endpoint available. Provide --gguf (to spawn llama-server) or "
        "--attach <url>. (sable is standalone — no external server needed; "
        "run `sable setup` to build llama.cpp, then `sable serve --gguf "
        "<model.gguf>`.)"
    )


# ------------------------------------------------------------------ commands

def _open_registry(cfg: SableConfig):
    from .registry import ModelRegistry

    home = resolve_sable_home(cfg)
    return ModelRegistry(home / "buffer" / "sable.db")


def _training_config(cfg: SableConfig, config_path: Optional[str]):
    from .core.config import resolve_config_path
    from .training.scheduler import TrainingConfig, load_yaml_section

    # resolve_config_path: explicit --config > SABLE_CONFIG > <home>/sable.yaml
    path = resolve_config_path(config_path)
    data = load_yaml_section(str(path) if path else None, "training")
    tc = TrainingConfig.from_yaml({"training": data})
    if not tc.base_model and cfg.training.get("base_model"):
        tc.base_model = cfg.training["base_model"]
    # merge any raw keys carried on the flat config too
    for k, v in (cfg.training or {}).items():
        if hasattr(tc, k) and getattr(tc, k) in (None, 0, False):
            setattr(tc, k, v)
    return tc


def _validation_config(cfg: SableConfig, config_path: Optional[str]):
    from .core.config import resolve_config_path
    from .curate.mine_goldens import merge_mined_into_tasks
    from .training.validate import ValidationConfig

    data = cfg.validation or {}
    if not data:
        from .training.scheduler import load_yaml_section

        # Same resolution as load_config: --config > SABLE_CONFIG > <home>/sable.yaml
        path = resolve_config_path(config_path)
        data = load_yaml_section(str(path) if path else None, "validation")
    vcfg = ValidationConfig.from_yaml({"validation": data})
    # Auto-mined goldens (curate/mine_goldens.py) join the promotion gate:
    # no <home>/goldens/mined.jsonl -> list unchanged (zero behavior change).
    vcfg.golden_tasks = merge_mined_into_tasks(
        vcfg.golden_tasks, resolve_sable_home(cfg))
    return vcfg


def _resolve_active_gguf(cfg: SableConfig) -> None:
    """serve --auto / run --auto: point gguf_path at the ACTIVE promoted GGUF."""
    reg = _open_registry(cfg)
    try:
        active = reg.active_model()
        if active is not None and active.gguf_path:
            # Idea adapted from PIASystem (MIT) app/core/adapter_manager.py —
            # serve start binds a registry model to the running server, so this
            # is an activation. Once per serve start, never per healthcheck.
            reg.record_activation(active.model_id)
    finally:
        reg.close()
    if active is None or not active.gguf_path:
        raise ServeError(
            "no ACTIVE model in the registry yet — run `sable iterate` "
            "(or pass --gguf explicitly)"
        )
    print(f"active model: {active.model_id} "
          f"(base={active.base_model}, quant={active.quant})")
    cfg.gguf_path = active.gguf_path
    if not cfg.model_name:
        cfg.model_name = active.model_id


def cmd_doctor(args: argparse.Namespace) -> int:
    """Read-only runtime diagnosis (GPU + llama.cpp readiness).

    With --env: the WP7 environment MEASUREMENT doctor instead (sable/
    doctor.py — disk/RAM/VRAM presets, config passthrough, ports, stale
    state; exit codes 0/1/3 per core/exitcodes.py)."""
    if getattr(args, "env_doctor", False):
        from .doctor import run_doctor

        return run_doctor(config_path=args.config,
                          fix=bool(getattr(args, "fix", False)))

    from .serving import runtime

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)

    info = runtime.detect_runtime()
    print(f"platform : {info.system}/{info.machine}")
    print(f"backend  : {info.backend}")
    for note in info.backend_notes:
        print(f"           {note}")
    if info.gpus:
        for g in info.gpus:
            vram = (
                f" free={g.free_mib}MiB total={g.total_mib}MiB"
                if g.total_mib is not None else ""
            )
            print(f"gpu      : [{g.vendor}] {g.name}{vram}")
    else:
        print("gpu      : none detected (CPU-only serving)")

    ngl = runtime.gpu_layers(cfg.gpu_layers, info)
    print(f"offload  : -ngl {ngl}" + (" (auto)" if cfg.gpu_layers is None else " (config)"))

    # Binary discovery (read-only: no auto-setup during doctor).
    try:
        binary = runtime_find_binary(cfg)
        print(f"server   : {binary}")
        if os.environ.get("SABLE_LLAMA_CPP"):
            print(
                "           ^ external llama.cpp selected via SABLE_LLAMA_CPP; "
                "sable only executes it as its own separate llama-server "
                "process (own dynamic port) and never attaches to or stops it."
            )
        if runtime.is_unified_memory(info):
            print(
                f"unified  : {runtime.UNIFIED_MEMORY_ENV}=1 set for llama-server "
                "(Strix Halo / Jetson-class unified-memory hardware; override "
                "with SABLE_UNIFIED_MEMORY=0)"
            )
        ext = discover_external_server_url()
        if ext:
            print(
                f"external : server running at {ext} (sable leaves it "
                "alone — spawn mode uses its own port, attach mode is read-"
                "only unless you pass --attach)"
            )
    except ServeError as e:
        print(f"server   : NOT FOUND — {e}")
        import shutil as _shutil

        have_git = bool(_shutil.which("git"))
        have_cmake = bool(_shutil.which("cmake"))
        can_setup = have_git and have_cmake
        status = "possible" if can_setup else "NOT possible"
        print(f"auto-setup: {status} (git={have_git}, cmake={have_cmake}; "
              "also needs a C/C++ compiler)")
        if not runtime.auto_setup_enabled(cfg):
            print("auto-setup: disabled by config/env")
        print("\nhint: run `sable setup` to build llama.cpp for this machine.")
        return 1
    return 0


def runtime_find_binary(cfg: SableConfig) -> str:
    from .serving.supervisor import find_llama_server

    return find_llama_server(cfg.llama_server_binary, auto_setup=False)


def cmd_setup(args: argparse.Namespace) -> int:
    """One-click llama.cpp install for this machine (backend auto-detected)."""
    from .serving import runtime

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)

    info = runtime.detect_runtime()
    backend = None if getattr(args, "backend", "auto") == "auto" else args.backend
    backend = backend or cfg.llama_backend or None
    repo = getattr(args, "repo", None) or cfg.llama_repo
    ref = getattr(args, "ref", None) or cfg.llama_ref
    print(f"platform : {info.system}/{info.machine}")
    print(
        f"backend  : {backend or info.backend}"
        + ("" if backend else " (auto-detected)")
    )
    if repo:
        print(f"repo     : {repo}" + (f" ref={ref}" if ref else ""))
    existing = None
    try:
        existing = runtime_find_binary(cfg)
    except ServeError:
        pass
    managed = runtime.read_install_state()
    if existing and not (managed and existing.startswith(str(runtime.sable_llama_root()))):
        print(f"note: using existing llama-server at {existing}")
        print("setup not needed; re-run with SABLE_LLAMA_CPP set to override "
              "discovery, or delete the old install to force a rebuild.")
        return 0

    try:
        build_bin = runtime.setup_llama_cpp(
            backend=backend, log_fn=print, repo=repo, ref=ref,
            update=bool(existing),
        )
    except runtime.SetupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"\nllama.cpp ready under {build_bin}")
    print("run `sable doctor` to confirm, then `sable serve --gguf <model.gguf>`.")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    """Check/rebuild the managed llama.cpp install (its update system)."""
    from .serving import runtime

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)

    print(f"sable  : v{__version__} "
          "(package updates: `pip install --upgrade sable`)")
    switching = bool(getattr(args, "repo", None) or getattr(args, "ref", None)
                     or getattr(args, "backend", None))

    status = runtime.check_llama_update(cfg)
    if not status.get("installed"):
        print(f"llama.cpp: {status.get('reason')}")
        print("run `sable setup` to install.")
        return 1
    print(
        f"llama.cpp: repo={status['repo']} ref={status['ref']} "
        f"backend={status['backend']}"
    )
    print(f"           built_at={status.get('built_at')} "
          f"commit={status.get('local_commit')}")
    if status.get("reason"):
        print(f"check    : {status['reason']}")
    else:
        print(
            f"remote   : {status.get('remote_commit')} "
            + ("— UPDATE AVAILABLE" if status.get("update_available") else "— up to date")
        )

    if args.check:
        return 0 if not status.get("update_available") else 10
    if not (status.get("update_available") or switching):
        print("nothing to do.")
        return 0

    backend = None if getattr(args, "backend", None) in (None, "auto") else args.backend
    try:
        runtime.setup_llama_cpp(
            backend=backend,
            log_fn=print,
            repo=getattr(args, "repo", None),
            ref=getattr(args, "ref", None),
            update=True,
        )
    except runtime.SetupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print("llama.cpp updated. Restart any running server (`sable serve`) to use it.")
    return 0


# ------------------------------------------------------------------ launch

LAUNCH_STATE_FILENAME = "launch_state.json"


def _launch_state_path(cfg: SableConfig):
    return resolve_sable_home(cfg) / LAUNCH_STATE_FILENAME


def _url_reachable(url: str, timeout: float = 2.0) -> bool:
    """True when something is accepting TCP connections on the URL's host:port."""
    import socket
    from urllib.parse import urlparse

    try:
        u = urlparse(url)
        port = u.port or (443 if u.scheme == "https" else 80)
        with socket.create_connection((u.hostname or "127.0.0.1", port),
                                      timeout=timeout):
            return True
    except OSError:
        return False


def cmd_launch(args: argparse.Namespace) -> int:
    """One-click program launcher: detached dashboard + browser, idempotent."""
    import json as _json
    import subprocess as _sp
    import webbrowser

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    home = resolve_sable_home(cfg)
    state_p = _launch_state_path(cfg)

    def _read_state():
        try:
            return _json.loads(state_p.read_text())
        except (OSError, ValueError):
            return None

    from .serving import runtime as _runtime

    if args.stop:
        stt = _read_state()
        if not stt:
            print("no launched dashboard (state file absent).")
            return 0
        pid = int(stt.get("pid", 0))
        alive = _runtime.pid_alive(pid)
        if alive is False:
            print("launched dashboard already exited.")
        elif alive is None:
            print(f"pid {pid} is not ours; leaving it alone.", file=sys.stderr)
            return 1
        elif _runtime.terminate_pid(pid):
            print(f"stopped launched dashboard (pid={pid}).")
        state_p.unlink(missing_ok=True)
        return 0

    # Idempotent: an alive launched dashboard => just surface its URL.
    stt = _read_state()
    if stt and stt.get("pid"):
        try:
            alive = _runtime.pid_alive(int(stt["pid"]))
        except ValueError:
            alive = None
        if alive:
            url = stt.get("url", "")
            # The pid being alive is not enough: a stale state file can carry a
            # URL from an earlier run (dynamic --port 0). Verify it answers.
            if not url or not _url_reachable(url):
                print(f"pid {stt['pid']} is alive but {url} refuses "
                      "connections; restarting the dashboard.")
                _runtime.terminate_pid(int(stt["pid"]))
            else:
                print(f"dashboard already running (pid={stt['pid']}): {url}")
                if not args.no_browser and url:
                    webbrowser.open(url)
                return 0
        state_p.unlink(missing_ok=True)

    (home / "logs").mkdir(parents=True, exist_ok=True)
    logf = open(home / "logs" / "dashboard.log", "ab")
    # Only scan lines THIS launch appends: dashboard.log is append-mode and a
    # previous run's "sable dashboard:" line would otherwise be picked up as
    # the new URL while the child is still booting (stale-port bug). NB: an
    # O_APPEND fd starts at offset 0, so seek to EOF explicitly.
    log_start = logf.seek(0, os.SEEK_END)
    cmd = [sys.executable, "-m", "sable.cli", "dashboard", "--port",
           str(args.port or 0)]
    proc = _sp.Popen(
        cmd, stdout=logf, stderr=_sp.STDOUT, stdin=_sp.DEVNULL,
        cwd=str(home), **_runtime.detached_popen_kwargs(),
    )
    # Wait briefly for the server to bind; dashboard.log carries the URL.
    url = None
    for _ in range(40):
        time.sleep(0.25)
        if proc.poll() is not None:
            print("error: dashboard exited during startup; see logs/dashboard.log",
                  file=sys.stderr)
            return 1
        try:
            with open(home / "logs" / "dashboard.log", "rb") as lf:
                lf.seek(log_start)
                txt = lf.read().decode(errors="ignore")
            # Prefer the tokenized URL (token auth is ON by default; opening
            # it seeds sessionStorage via api.js's ?token= bootstrap). The
            # plain line still works when auth was explicitly disabled.
            for line in reversed(txt.splitlines()):
                if line.startswith("sable dashboard authed: http"):
                    url = line.split(": ", 1)[1].strip()
                    break
                if line.startswith("sable dashboard: http"):
                    url = line.split(": ", 1)[1].strip()
        except OSError:
            pass
        if url:
            break
    state_p.write_text(_json.dumps({"pid": proc.pid, "url": url or "",
                                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                time.gmtime())}))
    print(f"sable dashboard: {url or '(starting — see logs/dashboard.log)'}")
    print(f"launcher state   : {state_p}  (stop with `sable launch --stop`)")
    if not args.no_browser and url:
        webbrowser.open(url)
    return 0


# ------------------------------------------------------------------- reset

def cmd_reset(args: argparse.Namespace) -> int:
    """Restore sable to a fresh state by wiping SABLE_HOME contents."""
    import shutil as _shutil
    import tempfile as _tempfile

    from .serving.deploy import stop_owned_server
    from .serving import runtime

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    home = resolve_sable_home(cfg)
    if not home.exists():
        print(f"nothing to reset: {home} does not exist.")
        return 0

    # --keep-models spares MODEL artifacts (exported GGUFs under models/,
    # the registry archive/, and the registry DB that tracks them). The
    # managed engine builds (llama.cpp / llama.cpp-<backend>) are NOT model
    # artifacts: a fresh state always rebuilds them via `sable setup`.
    keep_names = {"models", "archive"} if args.keep_models else set()
    entries = sorted(p.name for p in home.iterdir())
    protected = [e for e in entries if e in keep_names]
    print(f"SABLE_HOME: {home}")
    for e in entries:
        tag = "  (kept: --keep-models)" if e in protected else ""
        print(f"  rm -rf {e}{tag}")

    if not args.yes:
        print("\ndry run — nothing deleted. Re-run with --yes to confirm.")
        return 0

    # Stop things WE launched before deleting their state (never foreign pids).
    stop_owned_server(home)
    st_p = _launch_state_path(cfg)
    try:
        import json as _json

        pid = int(_json.loads(st_p.read_text()).get("pid", 0))
        if pid and runtime.pid_alive(pid):
            runtime.terminate_pid(pid)
    except Exception:
        pass
    # NB: launch_state.json itself is deliberately NOT unlinked here — it
    # lives under home, so the wipe loop below removes (and counts) it. An
    # early unlink made the "removed N entries" line disagree with the
    # listing printed above.

    # The registry DB lives inside buffer/ but tracks archive/; when models
    # are kept we stash the DB outside home and restore it after the wipe,
    # so keeping one without the other can never orphan the archive.
    stash = None
    if args.keep_models:
        db_files = [p for g in ("buffer/sable.db", "buffer/sable.db-wal",
                                "buffer/sable.db-shm")
                    if (p := home / g).exists()]
        if db_files:
            stash = Path(_tempfile.mkdtemp(prefix="sable-reset-"))
            for f in db_files:
                _shutil.move(str(f), str(stash / f.name))

    removed = []
    for name in entries:
        if name in protected:
            continue
        p = home / name
        try:
            _shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
            removed.append(name)
        except OSError as e:
            print(f"warning: could not remove {p}: {e}", file=sys.stderr)
    if stash is not None:
        buf = home / "buffer"
        buf.mkdir(parents=True, exist_ok=True)
        for f in stash.iterdir():
            _shutil.move(str(f), str(buf / f.name))
        stash.rmdir()
    print(f"removed {len(removed)} entries; sable is back to a fresh state.")
    if protected:
        kept = list(protected)
        if stash is not None:
            kept.append("buffer/sable.db (registry)")
        print(f"kept: {', '.join(kept)}")
    print("re-run `sable doctor` / `sable setup` to rebuild the runtime if needed.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)
    if cfg.auto_model and not cfg.effective_attach_url:
        try:
            _resolve_active_gguf(cfg)
        except ServeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    # `sable serve` is an explicit user action: clear any crash-loop
    # refusal the dashboard supervisor recorded for the default instance (#28).
    from .serving.supervisor import clear_crash_loop

    clear_crash_loop("default")

    supervisor = ServingSupervisor(cfg)
    try:
        install_signal_handlers(supervisor)
        if cfg.effective_attach_url:
            handle = supervisor.attach()
            print(f"Attached to {handle.base_url} (model: {handle.model_name})")
            print("This is an external server — sable will not stop it.")
        else:
            handle = supervisor.start()
            print(f"Serving {cfg.gguf_path}")
            print(f"OpenAI-compatible base_url: {handle.base_url}")
            print(f"model: {handle.model_name}")
            print("Press Ctrl-C to stop.")

        # Block until interrupted (SIGINT/SIGTERM handlers stop the supervisor,
        # which clears handle) or the owned child dies.
        while supervisor.handle is not None:
            proc = getattr(supervisor.handle, "process", None)
            if proc is not None and proc.poll() is not None:
                print("llama-server exited; shutting down.", file=sys.stderr)
                break
            time.sleep(0.5)
        return 0
    except (ServeError, AgentBackendError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        supervisor.stop()


def cmd_run(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)

    # Project parity with dashboard chat runs: resolve --project to its
    # record so the runner gets the same containment root (workspace_dir)
    # and per-project codebase index routing. Unknown ids fail fast.
    project_id = getattr(args, "project", None) or None
    project_workspace_dir = None
    project_context = None
    if project_id:
        from .core.delegation import load_project

        proj = load_project(cfg, project_id)
        if proj is None:
            print(f"error: no project {project_id} "
                  "(create it in the dashboard first)", file=sys.stderr)
            return 1
        project_workspace_dir = str(proj.get("workspace_dir") or "") or None
        project_context = proj  # '## Project' prompt section (dashboard parity)

    # Backlog #8: batch mode parses its input before touching the GPU so a
    # malformed task file fails fast with the line number.
    batch_tasks = None
    if getattr(args, "batch", None):
        from .core.batch import BatchInputError, load_tasks

        try:
            batch_tasks = load_tasks(args.batch)
        except BatchInputError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if not batch_tasks:
            print(f"error: no tasks found in {args.batch}", file=sys.stderr)
            return 1

    supervisor: Optional[ServingSupervisor] = None
    try:
        base_url, model, supervisor = _resolve_endpoint(cfg, spawn_if_needed=True)
        if supervisor is not None:
            install_signal_handlers(supervisor)
        print(f"Endpoint: {base_url}  model: {model}")

        # Stamps traces with the active project (dashboard-parity), so
        # curation/RFT filters and the trace panel scope CLI runs too.
        recorder = make_recorder(cfg, project=project_id)

        # WP12: --plan-task ID seeds the first message from the persistent plan
        # (build_task_prompt: title/description/details/test-strategy +
        # completed-dependency context) and stamps the trace's plan_task so it
        # links task<->trace both ways. The spec's shorthand was
        # `sable run --project X --task ID`, but `--task` already means "the
        # one-shot prompt text"; overloading it would make `--task "fix login"`
        # ambiguous, so the plan binding is a distinct flag (documented per
        # AGENTS.md). Needs plans.enabled and a known task id.
        plan_task_id = getattr(args, "plan_task", None) or None
        seed_prompt = ""
        if plan_task_id:
            from .core.plans import (build_task_prompt, get_task, load_plan,
                                     plans_config)

            if not plans_config(cfg).get("enabled", False):
                print("error: --plan-task requires plans.enabled: true",
                      file=sys.stderr)
                return 1
            pstore = load_plan(project_id)
            if get_task(pstore, plan_task_id) is None:
                print(f"error: no task {plan_task_id!r} in project "
                      f"{project_id or 'default'} plan", file=sys.stderr)
                return 1
            seed_prompt = build_task_prompt(pstore, plan_task_id)
            try:
                recorder.plan_task = plan_task_id
            except Exception:  # noqa: BLE001 - NullRecorder has no attr set
                pass

        # No profile_name here on purpose: `sable run` has no profile
        # concept (profiles are a dashboard-chat/delegation construct — their
        # system_prompt/tools/capabilities all stay unset for CLI runs). With
        # playbooks.enabled this correctly injects only GLOBAL recipes;
        # threading a bare name without the rest of the profile would be a
        # half-state, not consistency.
        runner = create_runner(
            cfg, base_url=base_url, model=model, api_key=cfg.attach_api_key,
            recorder=recorder,
            project_workspace_dir=project_workspace_dir,
            project_id=project_id,
            project_context=project_context,
        )
        print(f"Agent backend: {getattr(runner, 'name', 'unknown')}")

        if batch_tasks is not None:
            return _cmd_run_batch(cfg, args, runner, batch_tasks)

        # WP12: with --plan-task the first message is seeded from the graph
        # (build_task_prompt); any free --task text rides along as extra
        # instruction. Without it, behavior is exactly as before.
        first_message = args.task
        if plan_task_id:
            first_message = (f"{seed_prompt}\n\n{args.task}"
                             if args.task else seed_prompt)
        if first_message:
            result = runner.run(first_message)
            print("\n" + result.final_response)
            if getattr(result, "trace_id", None):
                print(f"\n[trace {result.trace_id} recorded to buffer]")
            return 0 if result.completed else 2

        # Interactive REPL (history kept in-process).
        history: list = []
        print("sable REPL — type a task, 'exit' to quit.")
        while True:
            try:
                line = input("sable> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in ("exit", "quit"):
                break
            try:
                result = runner.run(line, history=history or None)
                history = result.messages
                print(result.final_response)
                if getattr(result, "trace_id", None):
                    print(f"[trace {result.trace_id} recorded to buffer]")
            except AgentBackendError as e:
                print(f"error: {e}", file=sys.stderr)
        return 0
    except (ServeError, AgentBackendError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 130
    finally:
        if supervisor is not None:
            supervisor.stop()


def _cmd_run_batch(cfg: SableConfig, args: argparse.Namespace, runner,
                   tasks: list) -> int:
    """`sable run --batch ...` body (backlog item #8).

    Each task executes exactly like a one-shot `sable run`: same runner,
    same recorder, ordinary traces. All loop/checkpoint logic lives in
    core/batch.run_batch (testable without subprocesses); this adapter only
    maps the batch module's run_fn contract onto the live runner.
    """
    from .core.batch import run_batch

    def run_fn(task: str):
        try:
            result = runner.run(task)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # unexpected crash -> treat as infra-ish retry
            from .core import errors as error_taxonomy
            reason = "backend_error:" + error_taxonomy.classify_exception(e).reason
            return None, reason
        return getattr(result, "trace_id", None), result.failure_reason

    run_name = getattr(args, "run_name", None) or Path(args.batch).stem or "default"
    home = resolve_sable_home(cfg)
    # Pass 4 #29: when the live server was started from a registry model
    # (deploy / serve --auto), the batch explicitly targets that model — stamp
    # its activation once (read_serve_state carries model_id only for
    # registry-backed spawns; plain attach/gguf runs stay None).
    from .serving.deploy import read_serve_state

    st = read_serve_state(home) or {}
    return run_batch(
        tasks,
        home=home,
        run_name=run_name,
        resume=bool(getattr(args, "resume", False)),
        run_fn=run_fn,
        input_path=args.batch,
        model_id=st.get("model_id"),
    )


# ------------------------------------------------------------- traces/feedback

def _open_store(cfg: SableConfig) -> TraceStore:
    home = resolve_sable_home(cfg)
    db = home / "buffer" / "sable.db"
    if not db.exists():
        # Creating an empty store is fine (stats/list on a fresh install).
        pass
    return TraceStore(db)


def cmd_traces_list(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    store = _open_store(cfg)
    try:
        rows = store.list_traces(limit=args.limit, status=args.status, since=args.since)
    finally:
        store.close()
    if not rows:
        print("no traces recorded yet")
        return 0
    for r in rows:
        score = "" if r.outcome_score is None else f" score={r.outcome_score:+.2f}"
        cur = " [curated]" if r.curated else ""
        # WP12: plan-task chip — traces started from a plan task show the link.
        task = getattr(r, "plan_task", None)
        tchip = f" [{task}]" if task else ""
        print(
            f"{r.timestamp}  {r.trace_id}  {r.status:<8}{score}  "
            f"{r.agent_profile}/{r.model_alias}  tools={r.num_tool_calls}{cur}"
            f"{tchip}"
        )
    return 0


def cmd_traces_show(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    store = _open_store(cfg)
    try:
        row = store.get_trace(args.trace_id)
        feedback = store.get_feedback(args.trace_id)
    finally:
        store.close()
    if row is None:
        print(f"error: no trace with id {args.trace_id}", file=sys.stderr)
        return 1
    print(json.dumps(row.to_dict(), indent=2, ensure_ascii=False))
    if feedback:
        print("\nfeedback:")
        for f in feedback:
            print(
                f"  [{f['source']}] {f['timestamp']} "
                f"score={f['score']}"
                + (f" — {f['comment']}" if f.get("comment") else "")
            )
    print("\ntranscript:")
    try:
        with open(row.jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                kind = rec.get("kind")
                if kind == "message":
                    m = rec["message"]
                    content = m.get("content")
                    if isinstance(content, list):  # multimodal parts
                        content = json.dumps(content, ensure_ascii=False)
                    tc = m.get("tool_calls")
                    suffix = ""
                    if tc:
                        suffix = "\n  tool_calls: " + json.dumps(
                            [
                                {
                                    "name": (t.get("function") or {}).get("name"),
                                    "arguments": (t.get("function") or {}).get("arguments"),
                                }
                                for t in tc
                            ],
                            ensure_ascii=False,
                        )
                    name = f" ({m['name']})" if m.get("name") else ""
                    print(f"\n--- {m.get('role')}{name} ---")
                    print((content or "").strip() if isinstance(content, str) else content)
                    if suffix:
                        print(suffix)
                elif kind == "trace_footer":
                    print("\n--- end of trace ---")
                    print(json.dumps(rec, indent=2, ensure_ascii=False))
    except FileNotFoundError:
        print(f"(transcript file missing: {row.jsonl_path})", file=sys.stderr)
    return 0


def cmd_traces_stats(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    store = _open_store(cfg)
    try:
        s = store.stats()
    finally:
        store.close()
    print(json.dumps(s, indent=2, ensure_ascii=False))
    return 0


def cmd_traces_prune(args: argparse.Namespace) -> int:
    from datetime import datetime, timedelta, timezone

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    days = args.older_than
    if days is None:
        days = int(getattr(cfg, "recording_retention_days", 0) or 0)
    if days <= 0:
        print("error: no retention window — pass --older-than N (days) or set "
              "recording.retention_days in your config", file=sys.stderr)
        return 1
    cutoff = (datetime.now(timezone.utc)
              - timedelta(days=days)).isoformat(timespec="seconds")
    store = _open_store(cfg)
    try:
        summary = store.prune_traces(
            cutoff, keep_curated=not args.include_curated,
            dry_run=not args.yes)
    finally:
        store.close()
    mode = "dry run" if summary["dry_run"] else "pruned"
    print(f"traces prune ({mode}, older than {days}d / {cutoff}):")
    print(f"  traces:   {summary['deleted']}"
          + ("" if args.include_curated
             else f"  (kept {summary['kept_curated']} curated)"))
    print(f"  freed:    ~{summary['freed_bytes'] / 1_048_576:.1f} MiB")
    for err in summary["errors"]:
        print(f"  warning:  {err}", file=sys.stderr)
    if summary["dry_run"] and summary["deleted"]:
        print("  (re-run with --yes to delete)")
    return 0


# ------------------------------------------------------------------ plan
# WP12: persistent per-project task graphs (core/plans.py). All subcommands
# require config plans.enabled (the store is never touched otherwise); errors
# exit 1 like every other sable command — the exit-code taxonomy is unchanged.

def _plan_enabled_or_exit(cfg) -> Optional[int]:
    from .core.plans import plans_config

    if not plans_config(cfg).get("enabled", False):
        print("error: plans are disabled — set plans.enabled: true in your "
              "config (see config.example.yaml)", file=sys.stderr)
        return 1
    return None


def cmd_plan(args: argparse.Namespace) -> int:
    """`sable plan ...` dispatcher (add/list/show/next/depends/validate/
    set-status/remove/export/import/expand/loop/loop-status)."""
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    rc = _plan_enabled_or_exit(cfg)
    if rc is not None:
        return rc

    from .core import plans as P
    from .core.plans import PlanError

    project = getattr(args, "project", None) or None
    cmd = args.plan_command

    def _load():
        return P.load_plan(project)

    try:
        if cmd == "add":
            store = _load()
            fields = {"title": args.title, "description": args.description,
                      "details": args.details,
                      "test_strategy": args.test_strategy,
                      "priority": args.priority}
            deps = [d.strip() for d in str(args.deps or "").split(",")
                    if d.strip()]
            if deps:
                fields["deps"] = deps
            if str(args.parent or "").strip():
                fields["parent"] = args.parent.strip()
            t = P.add_task(store, fields)
            report = P.validate(store)
            if not report["ok"]:
                print("error: plan invalid after add "
                      f"({report}); task NOT saved", file=sys.stderr)
                return 1
            P.save_plan(store)
            print(f"added {t['id']}: {t['title']} [{t['priority']}]")
            return 0

        if cmd == "list":
            store = _load()
            if not store["tasks"]:
                print("(plan is empty)")
                return 0
            marks = {"pending": "[ ]", "in_progress": "[~]", "blocked": "[!]",
                     "done": "[x]", "cancelled": "[-]"}
            for t in store["tasks"]:
                dep = f" (deps: {', '.join(t['deps'])})" if t["deps"] else ""
                par = f" (parent: {t['parent']})" if t["parent"] else ""
                print(f"{marks.get(t['status'], '[?]')} {t['id']} "
                      f"[{t['priority']}] {t['title']}{dep}{par}")
            c = P.counts(store)
            print(f"-- {c['done']}/{c['total']} done, "
                  f"{len(P.ready(store))} ready")
            return 0

        if cmd == "show":
            store = _load()
            t = P.get_task(store, args.task_id)
            if t is None:
                print(f"error: no task {args.task_id!r}", file=sys.stderr)
                return 1
            print(json.dumps(t, indent=2, ensure_ascii=False))
            return 0

        if cmd == "next":
            store = _load()
            t = P.next_task(store)
            if t is None:
                print("(no unblocked tasks)")
                return 0
            bl = P.blockers(store, t["id"])
            line = f"{t['id']} [{t['priority']}] {t['title']}"
            if bl:
                line += " (blocked by " + ", ".join(b["id"] for b in bl) + ")"
            print(line)
            return 0

        if cmd == "depends":
            store = _load()
            if args.remove:
                t = P.remove_dependency(store, args.task_id, args.dep_id)
                P.save_plan(store)
                print(f"{t['id']} no longer depends on {args.dep_id}")
            else:
                t = P.add_dependency(store, args.task_id, args.dep_id)
                P.save_plan(store)
                print(f"{t['id']} now depends on {args.dep_id}")
            return 0

        if cmd == "validate":
            store = _load()
            report = P.validate(store)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if report["ok"] else 1

        if cmd == "set-status":
            store = _load()
            t = P.set_status(store, args.task_id, args.status,
                             note=str(args.note or "") or None)
            P.save_plan(store)
            print(f"{t['id']} -> {t['status']}")
            return 0

        if cmd == "remove":
            store = _load()
            removed, rewired = P.remove_task(store, args.task_id)
            P.save_plan(store)
            print(f"removed {args.task_id}")
            for r in rewired:
                print(f"  rewired deps of {r}")
            return 0

        if cmd == "export":
            store = _load()
            print(json.dumps(store, indent=2, ensure_ascii=False))
            return 0

        if cmd == "import":
            return _cmd_plan_import(cfg, args, project)

        if cmd == "expand":
            from .core.plan_llm import PlanImportError, plan_expand
            from .curate.judge import make_chat_client

            try:
                base_url, model, sup = _resolve_endpoint(
                    cfg, spawn_if_needed=True)
            except ServeError as e:
                print(f"error: no endpoint for expansion ({e})",
                      file=sys.stderr)
                return 1
            try:
                client = make_chat_client(base_url,
                                          api_key=cfg.attach_api_key or None,
                                          max_tokens=1200)
                rep = plan_expand(project, args.task_id, num=args.num,
                                  prompt=args.prompt or None,
                                  client_fn=client, model=model)
            except (PlanImportError, PlanError) as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            finally:
                if sup is not None:
                    sup.stop()
            for w in rep["warnings"]:
                print(f"warning: {w}", file=sys.stderr)
            if not rep["ok"]:
                print("error: no subtasks could be parsed", file=sys.stderr)
                return 1
            for t in rep["added"]:
                print(f"subtask {t['id']}: {t['title']}")
            return 0

        if cmd == "loop":
            from .core.plan_loop import LoopConflictError, start_plan_loop

            print("plan loop: running tasks unattended under each task's "
                  "profile grants (Ctrl-C to cancel)")
            try:
                loop = start_plan_loop(cfg, project,
                                       max_tasks=args.max_tasks,
                                       resume=bool(args.resume))
            except LoopConflictError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            try:
                while True:
                    prog = loop.wait(timeout=1.0)
                    cur = prog.get("current_task")
                    entries = prog.get("tasks") or []
                    if entries:
                        last = entries[-1]
                        print(f"  iter {last['iteration']} {last['id']}: "
                              f"{last['outcome']}"
                              + (f" ({last['reason']})" if last["reason"]
                                 else ""))
                    elif cur:
                        print(f"  iter {prog.get('iteration')} {cur}: running")
                    if loop.finished:
                        break
            except KeyboardInterrupt:
                loop.cancel()
                print("cancelling… (current task finishes or is left "
                      "in_progress in the progress file)")
                loop.wait()
            prog = loop.progress
            done = sum(1 for e in prog.get("tasks") or []
                       if e["outcome"] == "done")
            print(f"loop {prog.get('status')}: {done} task(s) done — "
                  f"{prog.get('stop_reason')}")
            return 0

        if cmd == "loop-status":
            from .core.plan_loop import read_loop_progress
            from .store.paths import resolve_sable_home

            prog = read_loop_progress(project, resolve_sable_home(cfg))
            if prog is None:
                print("(no loop progress file for this project)")
                return 0
            print(json.dumps(prog, indent=2, ensure_ascii=False))
            return 0

        print(f"error: unknown plan command {cmd!r}", file=sys.stderr)
        return 1
    except (PlanError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _cmd_plan_import(cfg, args, project) -> int:
    """`sable plan import FILE` — text/JSON brief; '-' reads stdin; binaries
    rejected with a clear error (core/plan_llm.clean_text_input)."""
    from pathlib import Path as _P

    path = args.file
    try:
        if path == "-":
            raw = sys.stdin.buffer.read()
        else:
            p = _P(path)
            if not p.exists():
                print(f"error: no such file {path}", file=sys.stderr)
                return 1
            raw = p.read_bytes()
    except OSError as e:
        print(f"error: cannot read {path}: {e}", file=sys.stderr)
        return 1

    client = model = None
    if args.use_ai:
        from .curate.judge import make_chat_client

        try:
            base_url, model, sup = _resolve_endpoint(cfg, spawn_if_needed=True)
        except ServeError as e:
            print(f"error: no endpoint for --use-ai ({e})", file=sys.stderr)
            return 1
        try:
            client = make_chat_client(base_url,
                                      api_key=cfg.attach_api_key or None,
                                      max_tokens=1200)
            return _do_plan_import(cfg, args, project, raw, client, model)
        finally:
            if sup is not None:
                sup.stop()
    return _do_plan_import(cfg, args, project, raw, None, None)


def _do_plan_import(cfg, args, project, raw, client, model) -> int:
    from .core.plan_llm import PlanImportError, plan_import
    from .core.plans import PlanError

    # A .json brief is passed through as text either way: the deterministic
    # parser rejects JSON arrays (no title lines), so sniff and hand JSON to
    # the salvage path directly when --use-ai is off.
    try:
        rep = plan_import(project, raw, use_ai=bool(args.use_ai),
                          num_tasks=args.num_tasks,
                          client_fn=client, model=model)
    except PlanImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except PlanError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for w in rep["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    if not rep["ok"]:
        print("error: no tasks imported (see warnings)", file=sys.stderr)
        return 1
    for t in rep["added"]:
        print(f"imported {t['id']}: {t['title']} [{t['priority']}]")
    print(f"-- {len(rep['added'])} task(s) added ({rep['source']} mode)")
    return 0


# ---------------------------------------------------------------- insights

def _fmt_score(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:+.3f}"


def _fmt_pct(v: Optional[float]) -> str:
    return "  —  " if v is None else f"{v * 100:5.1f}%"


def _print_insights(report: dict) -> None:
    """Human-readable aligned tables from stdlib string formatting (#11)."""
    ov = report["overview"]
    trend = report["score_trend"]
    arrow = {"up": "▲", "down": "▼"}.get(
        trend["overall"]["direction"], "→")

    print(f"Insights — last {report['days']} days"
          + (f" (profile: {report['profile_filter']})"
             if report.get("profile_filter") else "")
          + f" · generated {report['generated_at']}")
    print("=" * 64)

    # Overview / trend ----------------------------------------------------
    print(f"\nTraces: {ov['traces']}   avg score: {_fmt_score(ov['avg_score'])}"
          f"   failures: {ov['status_counts']['failure']}"
          f" ({_fmt_pct(ov['failure_rate'])})")
    d = trend["overall"]["half_mean_delta"]
    print(f"Score trend (per-day buckets, half-mean delta): "
          f"{arrow} {_fmt_score(d)}")

    for label, key in (("by profile", "by_profile"), ("by model", "by_model")):
        groups = trend[key]
        if not groups:
            continue
        print(f"\nScore by {label}:")
        print(f"  {'name':<28} {'traces':>6} {'avg':>8} {'delta':>8}  dir")
        for g in groups:
            mark = {"up": "▲", "down": "▼"}.get(g["direction"], "→")
            print(f"  {g['name']:<28} {g['traces']:>6} "
                  f"{_fmt_score(g['avg_score']):>8} "
                  f"{_fmt_score(g['half_mean_delta']):>8}  {mark}")

    # Failure reasons ------------------------------------------------------
    fr = report["failure_reasons"]
    print(f"\nFailure reasons ({fr['total']} coded — infra {fr['infra_total']}"
          f" · model {fr['model_total']}):")
    if not fr["by_reason"]:
        print("  (none recorded)")
    for row in fr["by_reason"][:10]:
        bar = "█" * min(row["count"], 30)
        # Pass 4 #33: remediation lifecycle annotation — reasons with no note
        # row render as implicit open (no bracket shown, same as before).
        remediation = ""
        if row.get("explicit") and row.get("status") != "open":
            remediation = f" [{row['status']}"
            if row.get("note"):
                remediation += f": {row['note']}"
            remediation += "]"
        print(f"  {row['kind']:<6} {row['reason']:<32} "
              f"{row['count']:>4} {bar}{remediation}")

    # Tool reliability ------------------------------------------------------
    tr = report["tool_reliability"]
    if tr["tools"]:
        print(f"\nTool reliability (from {tr['traces_scanned']} transcripts):")
        print(f"  {'tool':<28} {'calls':>6} {'fails':>6} {'rate':>7}")
        for t in tr["tools"][:10]:
            print(f"  {t['tool']:<28} {t['calls']:>6} "
                  f"{t['failures']:>6} {_fmt_pct(t['failure_rate']):>7}")
    else:
        print("\nTool reliability: no tool calls in scanned transcripts")

    # Latency / tokens ------------------------------------------------------
    lat = report["latency"]
    ds, tk = lat["duration_s"], lat["tokens_total"]
    print("\nPercentiles (p50/p90/max):")
    if ds["count"]:
        print(f"  duration s : {ds['p50']} / {ds['p90']} / {ds['max']}"
              f"   (n={ds['count']})")
    else:
        print("  duration s : — (column empty)")
    if tk["count"]:
        print(f"  tokens     : {tk['p50']} / {tk['p90']} / {tk['max']}"
              f"   (n={tk['count']})")
    else:
        print("  tokens     : — (column empty)")

    # Activity ---------------------------------------------------------------
    act = report["activity"]
    if act["traces_per_day"]:
        peak = max(p["traces"] for p in act["traces_per_day"]) or 1
        print("\nTraces per day:")
        for p in act["traces_per_day"]:
            bar = "█" * max(1, int(p["traces"] / peak * 24)) \
                if p["traces"] else ""
            print(f"  {p['date']}  {p['traces']:>4} {bar}")
    if act["busiest_hour"]:
        print(f"\nBusiest hour (UTC): {act['busiest_hour']}:00")
    if act["profiles"]:
        print("\nActive profiles:")
        print(f"  {'profile':<28} {'traces':>6} {'avg':>8} {'fail%':>7}")
        for p in act["profiles"]:
            print(f"  {p['profile']:<28} {p['traces']:>6} "
                  f"{_fmt_score(p['avg_score']):>8} "
                  f"{_fmt_pct(p['failure_rate']):>7}")

    # Intent review (pass 4 #36) ------------------------------------------
    ir = report.get("intent_review") or {}
    if ir.get("available"):
        rows = ir.get("rows") or []
        print("\nRecorded intents vs observed delta:")
        if not rows:
            # Honest-empty: no synthetic rows, just the hint.
            print(f"  {ir.get('hint') or 'no recorded intents'}")
        for r in rows:
            print(f"  #{r['seq']:<4} {r['ts']} {r['op']:<8} "
                  f"{(r['target'] or '-'):<6} {_fmt_score(r['delta'])} "
                  f"({_fmt_score(r['before_score'])} -> "
                  f"{_fmt_score(r['after_score'])})  \"{r['intent']}\"")

    # Confidence calibration (pass 4 #40) -----------------------------------
    cal = report.get("calibration") or {}
    if cal.get("available"):
        print("\nConfidence calibration (mean logprob vs quality score):")
        cal_rows = cal.get("rows") or []
        if not cal_rows:
            # Honest-empty: no synthetic rows, just the hint.
            print(f"  {cal.get('hint') or 'no traces with confidence data yet'}")
        for cr in cal_rows:
            print(f"  {cr['bucket_label']:<22} n={cr['n']:>3} "
                  f"conf {_fmt_score(cr['mean_conf'])} "
                  f"score {_fmt_score(cr['mean_score'])}")
        if cal.get("confidently_wrong"):
            print("  ⚠ confidently_wrong: the highest-confidence bucket scores "
                  "notably worse than the rest")

    # Promotion deltas --------------------------------------------------------
    promo = report["promotions"]
    print("\nBefore/after promotion:")
    if not promo["available"]:
        print(f"  degraded: {promo['reason']}")
    else:
        for e in promo["deltas"]:
            delta = _fmt_score(e["delta"])
            arrow_e = ""
            if e["delta"] is not None:
                arrow_e = " ▲" if e["delta"] > 0.02 else (
                    " ▼" if e["delta"] < -0.02 else " →")
            print(f"  {e['model_id']:<6} promoted {e['promoted_at']} "
                  f"(±{e['window_days']}d): "
                  f"{_fmt_score(e['before_avg_score'])} -> "
                  f"{_fmt_score(e['after_avg_score'])} = {delta}{arrow_e}"
                  f"  [{e['before_traces']}/{e['after_traces']} traces, "
                  f"match={e['match']}]" if e["window_days"] is not None
                  else f"  {e['model_id']:<6} promoted {e['promoted_at']} "
                  f"(no comparable traces around promotion)")

    if report["empty"]:
        print("\n(no traces in window — run `sable run` to record some)")


def cmd_insights(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    from .store.insights import generate_insights

    report = generate_insights(resolve_sable_home(cfg), days=args.days,
                               profile=args.profile)
    if args.as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        _print_insights(report)
    return 0


# --------------------------------------------------------------- playbooks
# Pass 4 #38 (core/playbooks.py): trajectory -> playbook distillation.

def _open_playbooks(cfg: SableConfig):
    from .core.playbooks import PlaybookStore

    return PlaybookStore(resolve_sable_home(cfg) / "buffer" / "sable.db")


def cmd_playbooks_scan(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    from .core.playbooks import distill_recipe, find_recurring_chains
    from .curate.curater import load_trace

    cfg = load_config(args.config)
    store = _open_store(cfg)
    pb_store = _open_playbooks(cfg)
    try:
        chains = find_recurring_chains(
            store, args.profile,
            min_occurrences=args.min_occurrences,
            window_days=args.window_days)
        if not chains:
            print("no recurring tool chains above the threshold "
                  f"(min {args.min_occurrences} successful traces, last "
                  f"{args.window_days} days)")
            return 0
        for c in chains:
            arrow = " → ".join(c["chain"])
            print(f"{c['count']:>4}x  {arrow}")
            if args.create:
                examples = []
                for tid in c["trace_ids"][:5]:
                    row = store.get_trace(tid)
                    loaded = load_trace(store, row) if row else None
                    if loaded is not None:
                        examples.append(loaded)
                text = distill_recipe(c["chain"], examples,
                                      occurrences=c["count"],
                                      profile=args.profile)
                pb_id, created = pb_store.create_candidate(
                    c["chain"], text, profile=args.profile,
                    source_trace_ids=c["trace_ids"])
                print(f"        {'created' if created else 'exists  '} "
                      f"candidate {pb_id}")
        return 0
    finally:
        pb_store.close()
        store.close()


def _cmd_playbooks_list(args: argparse.Namespace,
                        status: Optional[str]) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    pb_store = _open_playbooks(cfg)
    try:
        rows = pb_store.list(status=status, profile=args.profile or None)
    finally:
        pb_store.close()
    if not rows:
        print("no playbooks" + (f" with status {status!r}" if status else ""))
        return 0
    for p in rows:
        arrow = " → ".join(p.chain)
        print(f"{p.id}  {p.status:<9}  {p.profile or '(global)':<12}  "
              f"{arrow}")
    return 0


def cmd_playbooks_assess(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    from .core.playbooks import assess_playbook

    cfg = load_config(args.config)
    store = _open_store(cfg)
    pb_store = _open_playbooks(cfg)
    try:
        if args.id:
            targets = [pb_store.get(args.id)]
            if targets[0] is None:
                print(f"error: no playbook {args.id}", file=sys.stderr)
                return 1
        else:
            targets = pb_store.list(status="active")
        if not targets:
            print("no active playbooks to assess")
            return 0
        rc = 0
        for pb in targets:
            if pb is None:
                continue
            res = assess_playbook(store, pb_store, pb)
            if not res.get("ok"):
                print(f"{pb.id}: {res.get('error')}", file=sys.stderr)
                rc = 1
                continue
            def _f(v):
                return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"
            print(f"{pb.id} ({' → '.join(pb.chain)}): "
                  f"with={_f(res.get('mean_with'))} "
                  f"without={_f(res.get('mean_without'))} "
                  f"(n={res.get('n')}) -> "
                  + ("RETIRED (regression)" if res.get("demoted")
                     else "stays active"))
        return rc
    finally:
        pb_store.close()
        store.close()


def cmd_playbooks_promote(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    from .core.playbooks import (GateUnavailable, promote,
                                 run_promotion_gate)

    cfg = load_config(args.config)
    store = _open_store(cfg)
    pb_store = _open_playbooks(cfg)
    try:
        pb = pb_store.get(args.id)
        if pb is None:
            print(f"error: no playbook {args.id}", file=sys.stderr)
            return 1
        gate = None
        if args.check:
            # Gate against an ALREADY-available endpoint only (attach URL /
            # serve state); spawning a GPU server just to gate is left to
            # the user (`sable serve` first) — honest refusal otherwise.
            base_url = cfg.effective_attach_url
            model = cfg.model_name or "sable"
            if not base_url:
                print("note: --check needs a served endpoint (attach_url); "
                      "promoting without the gate is an explicit "
                      "confirmation, so re-run without --check only if that "
                      "is what you intend.", file=sys.stderr)
                return 1
            # Trust the server's own model id over config: gating a replay
            # against a model name the endpoint does not actually serve would
            # silently judge nothing (or error per call into skipped tasks).
            try:
                import httpx

                hdrs = ({"Authorization": f"Bearer {cfg.attach_api_key}"}
                        if cfg.attach_api_key else {})
                r = httpx.get(base_url.rstrip("/") + "/models", timeout=10.0,
                              headers=hdrs)
                if r.status_code == 200:
                    ids = [m.get("id") for m in
                           (r.json().get("data") or []) if m.get("id")]
                    if ids and ids[0] != model:
                        print(f"note: gating against served model "
                              f"{ids[0]!r} (config says {model!r})")
                        model = ids[0]
            except Exception as e:  # endpoint up but /models unreadable
                print(f"note: could not confirm served model ({e}); "
                      f"using {model!r}", file=sys.stderr)
            try:
                gate = run_promotion_gate(store, pb, base_url=base_url,
                                          model=model, n=args.n,
                                          api_key=cfg.attach_api_key)
            except GateUnavailable as e:
                print(f"gate unavailable: {e}", file=sys.stderr)
                return 1
            mean = gate.get("judge_mean")
            print("replay gate: mean="
                  + (f"{mean:.3f}" if isinstance(mean, (int, float))
                     else str(mean))
                  + f" over {gate.get('scored')}/{gate.get('n')} tasks "
                  f"(threshold {gate.get('threshold')})")
        res = promote(pb_store, args.id, check=args.check, gate=gate)
        if not res["ok"]:
            print(f"error: {res.get('reason') or res.get('error')}",
                  file=sys.stderr)
            return 1
        print(f"{args.id}: " + ("promoted → active"
                                if res.get("promoted")
                                else res.get("reason", "ok")))
        return 0
    finally:
        pb_store.close()
        store.close()


def cmd_playbooks_retire(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    pb_store = _open_playbooks(cfg)
    try:
        ok = pb_store.set_status(args.id, "retired", actor="cli")
    finally:
        pb_store.close()
    if not ok:
        print(f"error: no playbook {args.id}", file=sys.stderr)
        return 1
    print(f"{args.id}: retired (row kept for the archive)")
    return 0


def cmd_playbooks_show(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    pb_store = _open_playbooks(cfg)
    try:
        pb = pb_store.get(args.id)
    finally:
        pb_store.close()
    if pb is None:
        print(f"error: no playbook {args.id}", file=sys.stderr)
        return 1
    d = pb.to_dict()
    hist = d.pop("score_history")
    print(json.dumps(d, indent=2, ensure_ascii=False))
    if hist:
        print("\nscore history:")
        for h in hist:
            print("  " + json.dumps(h, ensure_ascii=False))
    return 0


def cmd_playbooks_export(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    from .core.playbooks import export_playbook

    cfg = load_config(args.config)
    pb_store = _open_playbooks(cfg)
    try:
        pb = pb_store.get(args.id)
    finally:
        pb_store.close()
    if pb is None:
        print(f"error: no playbook {args.id}", file=sys.stderr)
        return 1
    out = export_playbook(pb, home=resolve_sable_home(cfg),
                          out_path=Path(args.out) if args.out else None)
    print(f"wrote metadata to {out}")
    print("note: this is recipe + source trace metadata, NOT training rows "
          "(fabricating conversations is out of scope — see core/playbooks.py "
          "docstring).")
    return 0


# ---------------------------------------------------------------- curate/datasets

def _curation_config(cfg: SableConfig, args: argparse.Namespace):
    from .curate import curation_config_from_yaml

    cc = curation_config_from_yaml({"curation": getattr(cfg, "curation", {})})
    if getattr(args, "min_score", None) is not None:
        cc.min_score = args.min_score
    if getattr(args, "since", None):
        cc.since = args.since
    if getattr(args, "profile", None):
        cc.agent_profile = args.profile
    if getattr(args, "model_alias", None):
        cc.model_alias = args.model_alias
    # WP12: --project filters by the existing trace project stamp (the
    # Curater passes CurationConfig.project straight to query_for_curation).
    if getattr(args, "project", None):
        cc.project = args.project
    if getattr(args, "include_failures", False):
        cc.include_failures = True
    if getattr(args, "limit", None) is not None:
        cc.limit = args.limit
    if getattr(args, "fmt", None):
        cc.formats = args.fmt
    # train.yaml template inputs come from the config where available.
    # NOTE: deliberately NOT falling back to cfg.gguf_path — GGUFs are
    # serve-only and the training worker rejects them; train.yaml must carry
    # an HF id/path from training.base_model (or the REPLACE_* placeholder).
    return cc


def cmd_curate(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    ccfg = _curation_config(cfg, args)

    home = resolve_sable_home(cfg)
    store = _open_store(cfg)
    try:
        from .curate import Curater

        # Backlog #12: opt-in judge pass BEFORE selection so verdicts land as
        # feedback rows and the effective-score precedence (user > judge >
        # heuristic) can admit/deny borderline traces in this very run.
        if getattr(args, "judge", None):
            _run_judge_pass(cfg, ccfg, store, n=args.judge)

        curater = Curater(store, home)
        res = curater.curate(ccfg, name=args.name, rebuild_id=args.rebuild)
    finally:
        store.close()

    print(f"dataset {res.dataset_id}")
    print(f"  dir:     {res.dataset_dir}")
    print(f"  sft:     {res.sft_count} rows -> sft.jsonl")
    print(f"  dpo:     {res.dpo_count} pairs -> dpo.jsonl")
    print(f"  claimed: {res.claimed} traces (curated=1, dataset_id={res.dataset_id})")
    for w in res.warnings:
        print(f"  warning: {w}", file=sys.stderr)
    if not ccfg.base_model:
        print(
            "  note: train.yaml has no base model yet (REPLACE_* placeholder). "
            "Set training.base_model in your config to an HF id/path "
            "(e.g. \"Qwen/Qwen3-8B\") before `sable train start` — GGUFs are "
            "serve-only and cannot be trained on.",
            file=sys.stderr,
        )
    elif str(ccfg.base_model).lower().endswith(".gguf"):
        print(
            f"  warning: curation.base_model is a GGUF ({ccfg.base_model}) — "
            "GGUFs are serve-only; set training.base_model to an HF id/path "
            "for training.",
            file=sys.stderr,
        )
    if res.sft_count == 0 and res.dpo_count == 0:
        print("  (nothing selected — traces may already be claimed by another "
              "dataset; use --rebuild <dataset_id> to release them)")
    return 0


def _run_judge_pass(cfg: SableConfig, ccfg, store: TraceStore, n: int) -> None:
    """`curate --judge N`: score the top-N borderline traces with the served
    model and record verdicts as feedback rows source="judge" (backlog #12).

    Borderline = |heuristic/effective score − min_score| < curation.judge_band.
    Loud caveat printed once per run: the judged model grading its own traces
    is biased — treat verdicts as a tiebreaker, not ground truth. Transport
    failures and unparseable replies count as skipped; judging never aborts
    the curation that follows.
    """
    from datetime import datetime, timezone

    from .curate.curater import load_trace
    from .curate.judge import judge_trace, make_chat_client, select_borderline

    statuses = ["success", "partial", "unknown"]
    if ccfg.include_failures:
        statuses.append("failure")
    rows = store.query_for_curation(
        include_curated=False, statuses=statuses, since=ccfg.since,
        model_alias=ccfg.model_alias, agent_profile=ccfg.agent_profile,
    )
    loaded = [t for r in rows if (t := load_trace(store, r)) is not None]
    borderline = select_borderline(loaded, ccfg.min_score,
                                   ccfg.judge_band, n)
    if not borderline:
        print(f"judge: no borderline traces "
              f"(|score − min_score| < {ccfg.judge_band}); nothing to score")
        return

    supervisor = None
    try:
        base_url, model, supervisor = _resolve_endpoint(cfg, spawn_if_needed=True)
    except ServeError as e:
        print(f"judge skipped: no endpoint for judging ({e})", file=sys.stderr)
        return
    judge_model = ccfg.judge_model or model
    client = make_chat_client(base_url, api_key=cfg.attach_api_key)
    try:
        print(f"judge: scoring {len(borderline)} borderline trace(s) "
              f"via {base_url} (model={judge_model})")
        print("judge: NOTE — the judged model grades its OWN traces; verdicts "
              "are a biased tiebreaker, not ground truth.")
        scored = skipped = 0
        for t in borderline:
            verdict = judge_trace(t.messages, client, judge_model)
            if verdict is None:
                skipped += 1
                print(f"  {t.row.trace_id}: SKIPPED (no verdict)")
                continue
            store.add_feedback(
                t.row.trace_id, score=verdict["score"],
                comment="judge: " + verdict["reason"], source="judge",
                timestamp=datetime.now(timezone.utc).isoformat(
                    timespec="seconds"),
            )
            scored += 1
            eff = t.effective_score
            print(f"  {t.row.trace_id}: judge={verdict['score']:.2f} "
                  f"(heuristic={'—' if eff is None else f'{eff:+.2f}'}) "
                  f"— {verdict['reason']}")
        print(f"judge summary: {scored} scored, {skipped} skipped "
              f"of {len(borderline)} borderline candidates")
    finally:
        if supervisor is not None:
            supervisor.stop()


def cmd_datasets_list(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    from .curate import Curater

    curater = Curater(_open_store(cfg), resolve_sable_home(cfg))
    manifests = curater.list_datasets()
    if not manifests:
        print("no datasets curated yet")
        return 0
    for m in manifests:
        files = m.get("files", {})
        print(
            f"{m.get('created_at', '?')}  {m['dataset_id']}  "
            f"sft={files.get('sft.jsonl', {}).get('count', 0)}  "
            f"dpo={files.get('dpo.jsonl', {}).get('count', 0)}"
        )
    return 0


def cmd_datasets_show(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    from .curate import Curater

    curater = Curater(_open_store(cfg), resolve_sable_home(cfg))
    manifest = curater.load_manifest(args.dataset_id)
    if manifest is None:
        # Allow unique id-prefix / name-suffix matching like traces do.
        matches = [
            m for m in curater.list_datasets()
            if m.get("dataset_id", "").startswith(args.dataset_id)
            or m.get("dataset_id", "").endswith("--" + args.dataset_id)
        ]
        if len(matches) == 1:
            manifest = matches[0]
        elif len(matches) > 1:
            print(f"error: ambiguous dataset id {args.dataset_id!r} "
                  f"({len(matches)} matches)", file=sys.stderr)
            return 1
    if manifest is None:
        print(f"error: no dataset with id {args.dataset_id}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def cmd_datasets_diff(args: argparse.Namespace) -> int:
    """Diff two datasets (review idea #6): trace-id sets, file hashes, counts."""
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    from .curate.dataset_diff import DatasetDiffError, diff_datasets

    try:
        result = diff_datasets(resolve_sable_home(cfg) / "datasets",
                               args.a, args.b)
    except DatasetDiffError as e:
        print(f"error: {e.message}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_datasets_export(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    from .curate import Curater

    curater = Curater(_open_store(cfg), resolve_sable_home(cfg))
    try:
        target = curater.export_dataset(args.dataset_id, args.to)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except FileExistsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"exported dataset {args.dataset_id} to {target}")
    return 0


def cmd_datasets_import(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    from .curate.importer import ImportError_, import_dataset

    try:
        res = import_dataset(args.file, fmt=args.fmt, name=args.name,
                             home=resolve_sable_home(cfg))
    except ImportError_ as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"dataset {res.dataset_id}")
    print(f"  dir:      {res.dataset_dir}")
    print(f"  format:   {res.fmt}"
          + (" (auto-detected)" if args.fmt == "auto" else " (forced)"))
    print(f"  sft:      {res.rows} rows -> sft.jsonl")
    print(f"  skipped:  {res.skipped_rows} malformed/unusable rows")
    print(f"  total:    {res.total_rows} input rows")
    print("  provenance: imported (no trace claims; trains like a curated "
          "dataset)")
    # Pass 4 #42 gate summary — informational; the gate never blocks import.
    if res.gate is not None:
        g = res.gate.summary()
        by_trust = " · ".join(f"{t}={n}" for t, n in
                              sorted(g["by_trust"].items())) or "—"
        print(f"  gate:     n_rows={g['n_rows']} "
              f"exact_dupes={g['exact_dupes']} "
              f"near_dupes={g['near_dupes']} conflicts={g['conflicts']} "
              f"notes={g['notes']}")
        print(f"  trust:    {by_trust}")
        for d in res.gate.details:
            print(f"    flag: {d}")
        if res.gate.flags:
            print("  (flagged rows are KEPT — see flags.json beside sft.jsonl)")
    return 0


def cmd_datasets_lint(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    from .curate.ingest_gate import lint_datasets

    reports = lint_datasets(resolve_sable_home(cfg) / "datasets",
                            name=args.name)
    if not reports:
        print("no datasets to lint" + (f" matching {args.name!r}"
                                        if args.name else ""))
        return 0
    for r in reports:
        print(f"{r['dataset_id']}  trust={r['trust']}  "
              f"rows={r['n_rows']}  exact_dupes={r['exact_dupes']}  "
              f"near_dupes={r['near_dupes']}  conflicts={r['conflicts']}  "
              f"notes={r['notes']}")
        for d in r["details"]:
            print(f"    flag: {d}")
    total = sum(len(r["details"]) for r in reports)
    print(f"linted {len(reports)} dataset(s), {total} flag(s) "
          "(flagged rows are kept)")
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)

    score = args.score
    if args.fail and args.ok:
        print("error: --fail and --ok are mutually exclusive", file=sys.stderr)
        return 2
    if args.fail:
        score = -1.0
    if args.ok:
        score = 1.0
    if score is not None and not (-1.0 <= score <= 1.0):
        print("error: --score must be within [-1, 1]", file=sys.stderr)
        return 2
    if score is None and not args.note:
        print("error: nothing to record — pass --score/--fail/--ok and/or --note",
              file=sys.stderr)
        return 2

    store = _open_store(cfg)
    try:
        if args.last:
            row = store.latest_trace()
        elif args.trace_id:
            row = store.get_trace(args.trace_id)
            if row is None and len(args.trace_id) >= 8:
                # Allow unique id prefixes.
                cands = [r for r in store.list_traces(limit=1000)
                         if r.trace_id.startswith(args.trace_id)]
                if len(cands) == 1:
                    row = cands[0]
                elif len(cands) > 1:
                    print(f"error: ambiguous trace id prefix {args.trace_id!r} "
                          f"({len(cands)} matches)", file=sys.stderr)
                    return 1
        else:
            print("error: pass a trace_id or --last", file=sys.stderr)
            return 2
        if row is None:
            print("error: no trace found", file=sys.stderr)
            return 1
        store.add_feedback(
            row.trace_id,
            score=score,
            comment=args.note,
            source="user",
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    finally:
        store.close()
    shown = "" if score is None else f" (score {score:+.2f})"
    print(f"feedback recorded for trace {row.trace_id}{shown}")
    return 0


# ------------------------------------------------------------- Phase 4: models

def _hub_for(cfg):
    from .core.modelhub import ModelHub

    return ModelHub(models_dir=cfg.models_dir, home=resolve_sable_home(cfg))


def _fmt_bytes(n: Optional[int]) -> str:
    if not n:
        return "-"
    v = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024 or unit == "TB":
            return f"{v:.0f}{unit}" if unit == "B" else f"{v:.1f}{unit}"
        v /= 1024
    return f"{v:.1f}TB"


def cmd_hub_search(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    from .core.modelhub import HubError

    try:
        results = _hub_for(cfg).search(" ".join(args.query or []),
                                       limit=args.limit,
                                       kind=getattr(args, "kind", "models"))
    except HubError as e:
        print(f"hub error: {e}", file=sys.stderr)
        return 1
    if not results:
        print("no results")
        return 0
    for m in results:
        tag = "gguf" if m.get("gguf") else (
            m.get("sdk") or "hf")[:4].ljust(4)
        print(f"{tag}  {m['repo_id']:<55} "
              f"dl={m['downloads']:<9} likes={m['likes']}")
    print("\nnext: sable hub variants <repo>   then   sable hub download <repo>")
    return 0


def cmd_hub_variants(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    from .core.modelhub import HubError

    try:
        info = _hub_for(cfg).variants(args.repo,
                                      kind=getattr(args, "kind", "models"))
    except HubError as e:
        print(f"hub error: {e}", file=sys.stderr)
        return 1
    if not info["gguf"]:
        v = info["variants"][0] if info["variants"] else None
        size = _fmt_bytes(v["size_bytes"]) if v else "-"
        print(f"{args.repo} is not a GGUF repo — full checkpoint "
              f"({size}, {len(v['files']) if v else 0} files); "
              "usable as a training base model")
        return 0
    for v in info["variants"]:
        star = "*" if v["variant"] == info["default_variant"] else " "
        shards = f" ({v['shard_count']} shards)" if v["shard_count"] else ""
        print(f"{star} {v['variant']:<16} {_fmt_bytes(v['size_bytes']):>9}"
              f"{shards}")
    print("\n* = recommended default. download: "
          f"sable hub download {args.repo} [--variant NAME]")
    return 0


def cmd_hub_download(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    from .core.modelhub import HubError

    hub = _hub_for(cfg)
    kind = getattr(args, "kind", "models")
    print(f"downloading {args.repo}"
          + (f" [{args.variant}]" if args.variant else "")
          + f" -> {hub.kind_base(kind)}")
    try:
        job = hub.download_sync(args.repo, args.variant,
                                progress_cb=lambda d: None, kind=kind)
    except HubError as e:  # download_sync doesn't raise; belt & braces
        print(f"hub error: {e}", file=sys.stderr)
        return 1
    if job.state != "done":
        print(f"download {job.state}: {job.error or 'see above'}",
              file=sys.stderr)
        print("retry the same command to resume from partial bytes.",
              file=sys.stderr)
        return 1
    print(f"done: {_fmt_bytes(job.done_bytes)} -> {job.target_dir}")
    print("the model is now offered by serve/train pickers "
          "(sable models catalog / dashboard).")
    return 0


def cmd_hub_list(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    hub = _hub_for(cfg)
    kind = getattr(args, "kind", "models")
    items = hub.installed(kind)
    print(f"{kind} folder: {hub.kind_base(kind)}")
    if not items:
        print("nothing downloaded via the hub yet — try `sable hub search`")
        return 0
    for m in items:
        quants = ",".join((m.get("quants") or [])[:4]) or "full"
        print(f"{m['repo_id']:<55} {_fmt_bytes(m['size_bytes']):>9}  {quants}")
    return 0


def cmd_hub_delete(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    from .core.modelhub import HubError

    try:
        out = _hub_for(cfg).delete(args.repo,
                                   kind=getattr(args, "kind", "models"))
    except HubError as e:
        print(f"hub error: {e}", file=sys.stderr)
        return 1
    print(f"deleted {out['deleted']} (freed {_fmt_bytes(out['freed_bytes'])})")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    """Recent ledger entries; --verify re-walks the hash chain (pass 4 #31).

    --intents switches to the pass-4 #36 view: recorded intents paired with
    the observed before/after score delta (honest-empty — never synthetic).
    """
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    if getattr(args, "intents", False):
        from .store.insights import intent_review

        ir = intent_review(resolve_sable_home(cfg))
        if not ir["available"]:
            print(f"intent review unavailable: {ir['reason']}",
                  file=sys.stderr)
            return 1
        rows = ir["rows"]
        if not rows:
            print(ir.get("hint") or "no recorded intents")
            print("record one with e.g. `sable models promote v2 "
                  "--intent \"fix tool-call regressions\"`")
            return 0
        print(f"{'SEQ':>5}  {'TS':<20}  {'OP':<9} {'TARGET':<7} "
              f"{'BEFORE':>8} {'AFTER':>8} {'DELTA':>8}  INTENT")
        for r in rows:
            print(f"{r['seq']:>5}  {r['ts']:<20}  {r['op']:<9} "
                  f"{(r['target'] or '-'):<7} "
                  f"{_fmt_score(r['before_score']):>8} "
                  f"{_fmt_score(r['after_score']):>8} "
                  f"{_fmt_score(r['delta']):>8}  \"{r['intent']}\"")
        return 0
    reg = _open_registry(cfg)
    try:
        ops = reg.history(limit=max(1, int(args.limit)))
        verified = reg.verify_ledger_chain() if args.verify else None
    finally:
        reg.close()
    if not ops:
        print("no operations recorded yet")
    else:
        print(f"{'SEQ':>5}  {'TS':<20}  {'ACTOR':<10}  {'OP':<14} TARGET")
        for e in ops:
            print(f"{e['seq']:>5}  {e['ts']:<20}  {e['actor']:<10}  "
                  f"{e['op']:<14} {e['target_id'] or '-'}")
    if verified is not None:
        ok, bad_seq, total = verified
        if ok:
            print(f"chain OK ({total} entries verified)")
        else:
            print(f"chain BROKEN at seq {bad_seq} ({total} entries)",
                  file=sys.stderr)
            return 1
    return 0


def cmd_models_list(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    reg = _open_registry(cfg)
    store = _open_store(cfg)
    try:
        models = reg.list_models()
        active = reg.active_model()
    finally:
        reg.close()
    if not models:
        store.close()
        print("no models registered yet — run `sable iterate`")
        return 0
    for m in models:
        mark = "* ACTIVE" if (active and m.model_id == active.model_id) else " "
        line = (
            f"{mark:<9} {m.model_id:<6} {m.status:<12} base={m.base_model} "
            f"dataset={m.dataset_id or '-'} quant={m.quant or '-'}"
        )
        if m.parent_adapter:
            line += f" parent={m.parent_adapter}"
        if m.pinned:  # backlog #16
            line += " [pinned]"
        # Pass 4 #41: derived capability tier (registry/tiers.py) — computed
        # at read time from evidence, never stored.
        try:
            from .registry.tiers import derive_tier, served_stats_for, tier_name

            stats = served_stats_for(store, m)
            t = derive_tier(m, stats)
            line += f" T{t} {tier_name(t)}"
        except Exception:  # display context only — never break models list
            pass
        print(line)
    store.close()
    return 0


def cmd_models_show(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    reg = _open_registry(cfg)
    try:
        m = reg.get_model(args.model_id)
        active = reg.active_model()
    finally:
        reg.close()
    if m is None:
        print(f"error: no model {args.model_id}", file=sys.stderr)
        return 1
    d = m.to_dict()
    d["active"] = bool(active and active.model_id == m.model_id)
    print(json.dumps(d, indent=2, ensure_ascii=False))
    return 0


def cmd_models_promote(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    reg = _open_registry(cfg)
    try:
        try:
            m = reg.promote(args.model_id,
                            intent=getattr(args, "intent", None))
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    finally:
        reg.close()
    print(f"promoted {m.model_id} (base={m.base_model})")
    print("the running server is unchanged — `sable deploy "
          f"{m.model_id}` or restart `sable serve --auto` to go live")
    return 0


def cmd_models_rollback(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    reg = _open_registry(cfg)
    try:
        m = reg.rollback(base_model=args.base_model,
                         intent=getattr(args, "intent", None))
    finally:
        reg.close()
    if m is None:
        print("rolled back; no previous model remains (active pointer cleared)")
        return 0
    print(f"rolled back: active model is now {m.model_id} "
          f"(base={m.base_model}, gguf={m.gguf_path})")
    return 0


# ------------------------------------------- models: archive/pin/janitor (#16)

def cmd_models_archive(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    home = resolve_sable_home(cfg)
    reg = _open_registry(cfg)
    try:
        try:
            m = reg.archive_model(args.model_id, home=home)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    finally:
        reg.close()
    rec = m.pre_archive or {}
    print(f"archived {m.model_id} "
          f"(was {rec.get('previous_status', '?')}) -> {rec.get('archive_dir')}")
    print(f"restore with `sable models restore {m.model_id}`")
    return 0


def cmd_models_restore(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    reg = _open_registry(cfg)
    try:
        try:
            m = reg.restore_model(args.model_id)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    finally:
        reg.close()
    print(f"restored {m.model_id} -> status {m.status}")
    if m.gguf_path:
        print(f"artifacts back in place (gguf={m.gguf_path})")
    return 0


def _set_pin(args: argparse.Namespace, pinned: bool) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    reg = _open_registry(cfg)
    try:
        try:
            m = reg.set_pinned(args.model_id, pinned)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    finally:
        reg.close()
    verb = "pinned" if pinned else "unpinned"
    print(f"{verb} {m.model_id}"
          + (" — janitor will never archive it" if pinned else ""))
    return 0


def cmd_models_pin(args: argparse.Namespace) -> int:
    return _set_pin(args, pinned=True)


def cmd_models_unpin(args: argparse.Namespace) -> int:
    return _set_pin(args, pinned=False)


def cmd_models_janitor(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    home = resolve_sable_home(cfg)
    from .registry.janitor import (
        janitor_config_from_yaml,
        run_janitor,
    )

    jc = janitor_config_from_yaml(getattr(cfg, "janitor", {}))
    days = args.days if args.days is not None else jc.retention_days
    disk_days = (args.disk_days if getattr(args, "disk_days", None) is not None
                 else jc.disk_retention_days)
    summary = run_janitor(home, days=days, dry_run=bool(args.dry_run),
                          disk_days=disk_days)
    reg = _open_registry(cfg)
    try:
        rows = {m.model_id: m for m in reg.list_models()}
    finally:
        reg.close()
    label = "would archive" if args.dry_run else "archived"
    disk = summary.get("disk") or {}
    disk_label = "would remove" if args.dry_run else "removed"
    if not summary["archived"]:
        print(f"janitor: nothing to archive ({days}d retention)")
    else:
        for mid in summary["archived"]:
            m = rows.get(mid)
            status = (m.status if m else "?") + (" (dry run)" if args.dry_run else "")
            print(f"  {label}: {mid} [{status}]")
        mb = summary["freed_bytes"] / (1024 * 1024)
        print(f"janitor: {label} {len(summary['archived'])} model(s), "
              f"{mb:.1f} MiB {'reclaimable' if args.dry_run else 'freed'}")
    n_disk = len(disk.get("removed", []))
    if disk_days > 0:
        print(f"janitor: {disk_label} {n_disk} operational file(s) older than "
              f"{disk_days}d (logs/uploads/spillover/checkpoints)")
    for err in summary["errors"]:
        print(f"  warning: {err}", file=sys.stderr)
    return 1 if summary["errors"] else 0


# -------------------------------------------------------------- Phase 4: train

def cmd_train_start(args: argparse.Namespace) -> int:
    from .training.scheduler import SchedulerError

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    tc = _training_config(cfg, args.config)
    reg = _open_registry(cfg)
    sched = Scheduler_for(reg, tc)
    try:
        home = resolve_sable_home(cfg)
        # Backlog item #13: --config-key overrides go through the same
        # resolution point `iterate` uses (resolved train.yaml copy), so both
        # commands honor them identically. Bad overrides fail fast, pre-queue.
        overrides = list(getattr(args, "config_key", None) or [])
        dataset_dir = home / "datasets" / args.dataset
        applied: dict = {}
        train_yaml_path = dataset_dir / "train.yaml"
        # WP10 step 2: a configured training.backend must reach the worker
        # through the resolved train.yaml too (job spec carries backend);
        # unset stays on the untouched dataset template (default behavior).
        cfg_backend = getattr(tc, "backend", None)
        if overrides or cfg_backend:
            from .training.pipeline import _resolved_train_yaml

            try:
                train_yaml_path, applied = _resolved_train_yaml(
                    dataset_dir, home / "train_runs" / f"{args.dataset}-cli",
                    tc.base_model, config_overrides=overrides,
                    backend=cfg_backend,
                )
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2
        job = sched.submit(
            args.kind,
            dataset_id=args.dataset,
            log_dir=home / "train_runs",
            params={
                "train_yaml": str(train_yaml_path),
                "dataset_dir": str(dataset_dir),
                "base_model": tc.base_model or "",
                "dry_run": args.dry_run,
            },
            depends_on=list(getattr(args, "depends_on", None) or []) or None,
        )
        if applied:
            # Provenance: which keys were overridden to what (job log + stdout).
            from .training.pipeline import log_config_overrides

            log_config_overrides(job.log_path, job.job_id, applied)
            print("config overrides: " + ", ".join(
                f"{k}={v!r}" for k, v in applied.items()))
        print(f"queued {job.job_id} ({args.kind}, dataset={args.dataset})")
        try:
            finished = sched.run_job(job)
        except SchedulerError as e:
            # Another job holds the GPU. The submission stays queued (FIFO);
            # without --wait we exit cleanly instead of crashing with a raw
            # traceback over an orphaned queued row.
            if not getattr(args, "wait", False):
                print(f"note: {e}", file=sys.stderr)
                print(
                    f"{job.job_id} stays queued and starts when the GPU frees "
                    "up (pass --wait to block until it runs)."
                )
                return 0
            finished = _await_job(sched, reg, job, tc)
        _print_job_result(sched, finished)
        if finished.state != "done":
            return 1
        # WP2b: the worker exited 0 — a NaN in its recorded loss history is
        # silent corruption, not success (exit 4; core/exitcodes.py).
        from .training.integrity import find_nan_losses
        from .training.pipeline import parse_worker_losses

        nan = find_nan_losses(parse_worker_losses(finished.log_path))
        if nan:
            print(f"error: corrupt training result — NaN recorded in loss "
                  f"history ({', '.join(nan)})", file=sys.stderr)
            return EXIT_CORRUPT_ARTIFACT
        return 0
    finally:
        reg.close()


def _await_job(sched, reg, job, tc):
    """--wait: poll until our queued job finishes (running it ourselves when
    the GPU frees up, or following along while another process runs it)."""
    from .training.scheduler import SchedulerError

    poll = max(1.0, tc.poll_interval_s)
    while True:
        cur = reg.get_job(job.job_id)
        if cur is None:
            return job
        if cur.state not in ("queued", "running"):
            return cur  # done / error / cancelled — reported by the caller
        if cur.state == "queued":
            try:
                return sched.run_job(cur)
            except SchedulerError:
                pass  # someone else grabbed the GPU; keep polling
        print(f"waiting for GPU… ({cur.job_id} {cur.state})", flush=True)
        time.sleep(poll)


def cmd_train_status(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    reg = _open_registry(cfg)
    try:
        if args.job_id:
            job = reg.get_job(args.job_id)
            if job is None:
                print(f"error: no job {args.job_id}", file=sys.stderr)
                return 1
            jobs = [job]
        else:
            jobs = reg.list_jobs()
    finally:
        reg.close()
    if not jobs:
        print("no jobs yet")
        return 0
    for j in jobs:
        print(
            f"{j.job_id}  {j.kind:<5} {j.state:<9} dataset={j.dataset_id or '-'} "
            f"model={j.model_id or '-'} queued={j.queued_at}"
            + (f" depends_on={','.join(j.depends_on)}" if j.depends_on else "")
            + (f" error={j.error}" if j.error else "")
        )
    if args.job_id and getattr(args, "tail", 0):
        j = jobs[0]
        if j.log_path and Path(j.log_path).is_file():
            print(f"\n--- {j.log_path} (last {args.tail}) ---")
            for line in Path(j.log_path).read_text().splitlines()[-args.tail:]:
                print(line)
    return 0


def cmd_train_resume(args: argparse.Namespace) -> int:
    """Re-queue an interrupted training job from its newest checkpoint (#21).

    Submits a NEW job (the jobs table stays append-only history) whose
    resolved train.yaml points `training.output_dir` back at the interrupted
    run's directory — where the resumable checkpoints live — and sets
    `training.resume: auto`, so the worker picks up from the newest unfinished
    checkpoint instead of restarting the epoch. The original job row is left
    untouched (provenance), its id recorded in the new job's params.
    """
    from .training.scheduler import SchedulerError

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    tc = _training_config(cfg, args.config)
    reg = _open_registry(cfg)
    sched = Scheduler_for(reg, tc)
    try:
        home = resolve_sable_home(cfg)
        orig = reg.get_job(args.job_id)
        if orig is None:
            print(f"error: no job {args.job_id}", file=sys.stderr)
            return 1
        if orig.kind not in ("sft", "dpo", "grpo"):
            print(f"error: job {orig.job_id} is kind={orig.kind!r}; only "
                  "training jobs carry resumable checkpoints", file=sys.stderr)
            return 1
        if orig.state in ("queued", "running"):
            print(f"error: job {orig.job_id} is {orig.state}; cancel it first",
                  file=sys.stderr)
            return 1
        dataset_id = orig.dataset_id or ""
        dataset_dir = home / "datasets" / dataset_id
        if not dataset_dir.is_dir():
            print(f"error: dataset dir for {dataset_id} is gone", file=sys.stderr)
            return 1

        # The interrupted run's output_dir (where its checkpoints live) comes
        # from the resolved train.yaml the original worker was given.
        import yaml as _yaml

        orig_yaml = (orig.params or {}).get("train_yaml") or ""
        prior_out: Optional[str] = None
        if orig_yaml and Path(orig_yaml).is_file():
            try:
                data = _yaml.safe_load(Path(orig_yaml).read_text()) or {}
                prior_out = ((data.get("training") or {}).get("output_dir"))
            except (OSError, ValueError):
                prior_out = None
        if not prior_out:
            # Dataset-template fallback (matches the curater default).
            prior_out = str(dataset_dir / "outputs")

        from .training.worker import find_resumable_checkpoint

        ckpt = find_resumable_checkpoint(prior_out)
        if ckpt is None:
            print(f"note: no unfinished checkpoint under {prior_out} — "
                  f"resuming would start from scratch. Run `sable train "
                  f"start --dataset {dataset_id}` for a clean job.")
            return 1

        run_dir = home / "train_runs" / f"{dataset_id}-resume-{orig.job_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        data_src = None
        if orig_yaml and Path(orig_yaml).is_file():
            try:
                data_src = _yaml.safe_load(Path(orig_yaml).read_text()) or {}
            except (OSError, ValueError):
                data_src = None
        if data_src is None:  # fall back to the dataset template
            tpl = dataset_dir / "train.yaml"
            data_src = (_yaml.safe_load(tpl.read_text()) or {}) \
                if tpl.is_file() else {}
        data_src.setdefault("training", {})
        data_src["training"]["output_dir"] = prior_out  # keep the checkpoints
        data_src["training"]["resume"] = "auto"
        new_yaml = run_dir / "train.resumed.yaml"
        new_yaml.write_text(_yaml.safe_dump(data_src, sort_keys=False),
                            encoding="utf-8")

        job = sched.submit(
            orig.kind, dataset_id=dataset_id, log_dir=home / "train_runs",
            params={
                "train_yaml": str(new_yaml),
                "dataset_dir": str(dataset_dir),
                "base_model": (orig.params or {}).get("base_model")
                              or tc.base_model or "",
                "resumed_from": orig.job_id,
                "resume_checkpoint": ckpt,
            },
        )
        print(f"queued {job.job_id} ({orig.kind}, dataset={dataset_id}) "
              f"resuming from {ckpt}")
        try:
            finished = sched.run_job(job)
        except SchedulerError as e:
            if not getattr(args, "wait", False):
                print(f"note: {e}", file=sys.stderr)
                print(f"{job.job_id} stays queued (pass --wait to block).")
                return 0
            finished = _await_job(sched, reg, job, tc)
        _print_job_result(sched, finished)
        return 0 if finished.state == "done" else 1
    finally:
        reg.close()


def cmd_train_cancel(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    tc = _training_config(cfg, args.config)
    reg = _open_registry(cfg)
    sched = Scheduler_for(reg, tc)
    try:
        job = sched.cancel(args.job_id)
    finally:
        reg.close()
    if job is None:
        print(f"error: no job {args.job_id}", file=sys.stderr)
        return 1
    print(f"job {job.job_id} -> {job.state}")
    return 0


def _janitor_hook(cfg: SableConfig) -> None:
    """Run the disk janitor at cron time (backlog item #16).

    `train auto-check` is the existing external-scheduler entrypoint, so the
    janitor piggybacks on it — no new daemon (state-file invariant: state file +
    host scheduler, not an app-internal timer). Gated by config
    ``janitor.enabled`` (default false); failures never break auto-check.
    """
    from .registry.janitor import janitor_config_from_yaml, run_janitor

    jc = janitor_config_from_yaml(getattr(cfg, "janitor", {}))
    if not jc.enabled:
        return
    try:
        summary = run_janitor(resolve_sable_home(cfg), days=jc.retention_days,
                              actor="scheduler",
                              disk_days=jc.disk_retention_days)
        if summary["archived"]:
            print(f"janitor: archived {', '.join(summary['archived'])} "
                  f"({summary['freed_bytes']} bytes reclaimable)")
    except Exception as e:  # maintenance must never break the training cron
        logger.warning("janitor run failed: %s", e)


def _auto_check_multi(args: argparse.Namespace, cfg, tc, ec,
                      targets: list) -> int:
    """Multi-agent easy loop for `train auto-check --trigger` (cron).

    Checks every agent in the loop with its scoped thresholds, then runs one
    iterate pass per agent whose thresholds are met and the resource gate is
    open. Runs are strictly sequential — the cross-process loop lock inside
    cmd_iterate serializes them anyway, and each run claims only its own
    profile's stamped traces. The golden-mining pass runs once before the
    first training (it is global), not per agent."""
    from .training.easy import mine_goldens_pass

    min_traces = (ec.min_traces if ec.min_traces is not None
                  else tc.auto_min_traces)
    min_score = (ec.min_score if ec.min_score is not None
                 else tc.auto_min_score)
    store = _open_store(cfg)
    try:
        from .training.scheduler import resource_gate_ok, should_auto_train

        checks = []
        for p in targets:
            d = should_auto_train(store, min_traces=min_traces,
                                  min_score=min_score, agent_profile=p)
            d["profile"] = p
            checks.append(d)
        gate = resource_gate_ok(tc)
    finally:
        store.close()

    for d in checks:
        print(json.dumps(d, indent=2))

    if not gate["ok"]:
        print("gate blocked: " + "; ".join(gate["reasons"]))
        return 0 if args.trigger else 1

    due = [d["profile"] for d in checks if d["trigger"]]
    if not due:
        print("no trigger")
        return 0 if args.trigger else 1
    if not args.trigger:
        print("TRIGGER: " + ", ".join(due))
        return 0

    dry = bool(getattr(args, "dry_run", False))
    rc_final = 0
    for i, p in enumerate(due):
        print(f"easy training [{p}]: thresholds met and resource gate open — "
              "starting iterate loop…")
        if i == 0:
            # Same pre-train golden-mining pass the dashboard's easy run does
            # (global; honors goldens_mine.enabled; never raises).
            mine = mine_goldens_pass(cfg, resolve_sable_home(cfg))
            if mine.get("ok"):
                print(f"easy training: mined {mine.get('mined', 0)} golden(s) "
                      f"from 👍 traces ({mine.get('skipped_dedup', 0)} "
                      "deduped)")
        rc = cmd_iterate(argparse.Namespace(
            config=args.config, verbose=args.verbose,
            name=f"easy:{p}", kind="sft", dataset=None, dry_run=dry,
            auto=True, easy_profile=p,
        ))
        if rc != 0:
            print(f"easy training [{p}]: iterate exited with {rc}",
                  file=sys.stderr)
            rc_final = rc
    return rc_final


def cmd_train_auto_check(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    tc = _training_config(cfg, args.config)
    _janitor_hook(cfg)
    # Easy training (dashboard "Easy training" page): when the turnkey loop is
    # enabled for a profile, this cron entrypoint checks and trains THAT
    # profile's traces only, with its threshold overrides. With several agents
    # in the loop (easy_training.profiles), every agent gets its own scoped
    # check and iterate pass — see _auto_check_multi.
    from .training.easy import EasyTrainingConfig

    ec = EasyTrainingConfig.from_yaml(
        getattr(cfg, "easy_training", None) or {})
    if ec.enabled and len(ec.all_profiles) > 1:
        return _auto_check_multi(args, cfg, tc, ec, ec.all_profiles)
    easy_profile = ec.profile if (ec.enabled and ec.profile) else None
    min_traces = (ec.min_traces if easy_profile and ec.min_traces is not None
                  else tc.auto_min_traces)
    min_score = (ec.min_score if easy_profile and ec.min_score is not None
                 else tc.auto_min_score)
    store = _open_store(cfg)
    try:
        from .training.scheduler import resource_gate_ok, should_auto_train

        d = should_auto_train(
            store, min_traces=min_traces, min_score=min_score,
            agent_profile=easy_profile,
        )
        gate = resource_gate_ok(tc)
    finally:
        store.close()
    d["gate"] = gate
    print(json.dumps(d, indent=2))

    trigger = bool(args.trigger)
    if not d["trigger"]:
        print("no trigger")
        # With --trigger the check simply reports and exits cleanly (a cron
        # line that runs unattended must not spam failures when thresholds
        # are not met yet).
        return 0 if args.trigger else 1
    if not gate["ok"]:
        print("gate blocked: " + "; ".join(gate["reasons"]))
        return 0 if args.trigger else 1
    if not trigger:
        print("TRIGGER")
        return 0

    # --trigger + thresholds/gate pass: run the full iterate loop in-process.
    # auto=True (pass 4 #41): unattended path — automatic promotion is gated
    # by validation.min_tier_auto_deploy inside run_iterate.
    print("thresholds met and resource gate open — starting iterate loop…")
    if easy_profile:
        print(f"easy training: scoping curation to profile “{easy_profile}”")
        # Same pre-train golden-mining pass the dashboard's easy run does:
        # 👍-thumbed chats since the last run join the promotion gate. Honors
        # goldens_mine.enabled (skips silently when off) and never raises, so
        # a non-easy auto-check stays byte-identical.
        from .training.easy import mine_goldens_pass

        mine = mine_goldens_pass(cfg, resolve_sable_home(cfg))
        if mine.get("ok"):
            print(f"easy training: mined {mine.get('mined', 0)} golden(s) "
                  f"from 👍 traces ({mine.get('skipped_dedup', 0)} deduped)")
    dry = bool(getattr(args, "dry_run", False))
    return cmd_iterate(argparse.Namespace(
        config=args.config, verbose=args.verbose,
        name=("auto" if not easy_profile else f"easy:{easy_profile}"),
        kind="sft",
        dataset=None, dry_run=dry,
        auto=True, easy_profile=easy_profile,
    ))


# ------------------------------------------------------------- Phase 4: deploy

def cmd_deploy(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    reg = _open_registry(cfg)
    try:
        from .serving.deploy import deploy_model

        home = resolve_sable_home(cfg)
        try:
            result = deploy_model(home, reg, args.model_id, cfg)
        except ServeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    finally:
        reg.close()
    print(f"deployed {result['model_id']}: {result['base_url']} "
          f"(gguf={result['gguf_path']})")
    if result["stopped_previous"] and result["previous"]:
        print(f"stopped previous server (was {result['previous'].get('base_url')})")
    print("live agents: restart `sable run`/`serve`, or in daemon mode "
          "switch_model is called automatically.")
    return 0


# ------------------------------------------------------------ Phase 5: dashboard

def cmd_dashboard(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    from .dashboard.server import serve_dashboard

    token = getattr(args, "token", None)
    if getattr(args, "no_token", False):
        # Explicit opt-out: the empty string disables auth in the resolver.
        token = ""
    try:
        serve_dashboard(cfg, port=getattr(args, "port", None), token=token)
        return 0
    except OSError as e:
        print(f"error: cannot start dashboard: {e}", file=sys.stderr)
        return 1


# ------------------------------------------------------------ Phase 4: iterate

def cmd_iterate(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    home = resolve_sable_home(cfg)
    # Cross-process loop guard shared with the dashboard (submit_iterate):
    # cron (`train auto-check --trigger`) and a manual run — here or there —
    # must not both curate the shared buffer. Dry runs claim nothing, so they
    # skip it. Busy => exit 0: an unattended cron line should not spam
    # failures when another loop owns the buffer.
    from .training.easy import acquire_loop_lock, release_loop_lock

    loop_handle = None
    if not getattr(args, "dry_run", False):
        loop_handle = acquire_loop_lock(home)
        if loop_handle is None:
            print("loop busy: another training run (dashboard or cli) holds "
                  "it; skipping")
            return 0
    store = _open_store(cfg)
    reg = _open_registry(cfg)
    try:
        from .training.pipeline import run_iterate

        ccfg = _curation_config(cfg, argparse.Namespace(
            min_score=None, since=None,
            profile=getattr(args, "easy_profile", None),
            model_alias=None,
            # None stubs keep the YAML-loaded values (the easy preset already
            # sets curation.formats: both); forcing "both" here silently
            # overrode a user's formats: setting on plain `sable iterate`.
            include_failures=False, limit=None, fmt=None,
        ))
        tc = _training_config(cfg, args.config)
        vc = _validation_config(cfg, args.config)
        sched = Scheduler_for(reg, tc)

        def stage(name: str, msg: str) -> None:
            print(f"[{name}] {msg}")

        overrides = list(getattr(args, "config_key", None) or [])
        res = run_iterate(
            home=home, registry=reg, scheduler=sched, store=store,
            curation_cfg=ccfg, training_cfg=tc, validation_cfg=vc,
            name=args.name, kind=args.kind, dataset_id=args.dataset,
            dry_run=args.dry_run, config_overrides=overrides, on_stage=stage,
            intent=getattr(args, "intent", None),
            auto=bool(getattr(args, "auto", False)),
        )
    finally:
        store.close()
        reg.close()
        release_loop_lock(loop_handle)

    print()
    print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
    if res.ok:
        if res.promoted:
            print(f"\nOK: {res.model_id} promoted. "
                  f"`sable deploy {res.model_id}` to go live now.")
        elif args.dry_run:
            print("\nOK (dry run): config + dataset validated; nothing trained.")
        return 0
    # WP2b: a corrupt artifact after an otherwise-successful flow gets its
    # own exit code (4) so automation can distinguish "quietly garbage" from
    # "loud failure" (see core/exitcodes.py).
    if res.corrupt:
        print(f"\nCORRUPT at stage {res.stage}: {res.error}", file=sys.stderr)
        return EXIT_CORRUPT_ARTIFACT
    print(f"\nFAILED at stage {res.stage}: {res.error}", file=sys.stderr)
    return 1


# ---------------------------------------------------------- evaluate (#32)

def cmd_evaluate_shadow(args: argparse.Namespace) -> int:
    """Enqueue (and usually run) a shadow_eval job for one candidate model.

    Mirrors `train start`'s queued-or-runs contract: on a busy GPU the job
    stays queued unless --wait is passed. The verdict lands on the registry
    row (models.shadow_json); this command prints it when available.
    """
    from .training.scheduler import SchedulerError

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    tc = _training_config(cfg, args.config)
    reg = _open_registry(cfg)
    sched = Scheduler_for(reg, tc)
    try:
        home = resolve_sable_home(cfg)
        if reg.get_model(args.model_id) is None:
            print(f"error: no model {args.model_id} in registry",
                  file=sys.stderr)
            return 1
        job = sched.submit(
            "shadow_eval",
            model_id=args.model_id,
            log_dir=home / "train_runs",
            params={
                "profile": args.profile,
                "n": int(args.n),
                "judge_model": args.judge_model,
            },
            enqueue_actor="cli",
        )
        print(f"queued {job.job_id} (shadow_eval, model={args.model_id})")
        try:
            finished = sched.run_job(job)
        except SchedulerError as e:
            if not getattr(args, "wait", False):
                print(f"note: {e}", file=sys.stderr)
                print(f"{job.job_id} stays queued and starts when the GPU "
                      "frees up (pass --wait to block until it runs).")
                return 0
            finished = _await_job(sched, reg, job, tc)
        _print_job_result(sched, finished)
        if finished.state == "done":
            row = reg.get_model(args.model_id)
            shadow = row.shadow if row else None
            if shadow and not shadow.get("skipped"):
                imp = shadow.get("improvement")
                print(f"shadow verdict vs {shadow.get('baseline_id')}: "
                      f"improvement="
                      + (f"{imp:.3f}" if isinstance(imp, (int, float))
                         else str(imp))
                      + ("  REGRESSION (warn-only)"
                         if shadow.get("is_regression") else ""))
            elif shadow and shadow.get("skipped"):
                print(f"shadow verdict: skipped — {shadow.get('reason')}")
        return 0 if finished.state == "done" else 1
    finally:
        reg.close()


def cmd_evaluate_transfer(args: argparse.Namespace) -> int:
    """Enqueue (and usually run) a cross-profile transfer probe (#37).

    Idea adapted from PIASystem (MIT) app/core/transfer_bridge.py — shadow
    simulation on the destination domain's data before an adapter transfer is
    trusted. Same in-process shadow_eval job kind with store_as="transfer":
    replay profile P's replay set against the candidate; baseline defaults to
    the registry's current active model (what P runs today). With no active
    model and no --baseline the probe runs candidate-only and stores absolute
    scores (baseline_id null). Results land in models.transfers_json.
    """
    from .training.scheduler import SchedulerError

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    tc = _training_config(cfg, args.config)
    reg = _open_registry(cfg)
    sched = Scheduler_for(reg, tc)
    try:
        home = resolve_sable_home(cfg)
        if reg.get_model(args.model_id) is None:
            print(f"error: no model {args.model_id} in registry",
                  file=sys.stderr)
            return 1
        if args.baseline and reg.get_model(args.baseline) is None:
            print(f"error: no baseline model {args.baseline} in registry",
                  file=sys.stderr)
            return 1
        job = sched.submit(
            "shadow_eval",
            model_id=args.model_id,
            log_dir=home / "train_runs",
            params={
                "profile": args.profile,
                "n": int(args.n),
                "judge_model": args.judge_model,
                "store_as": "transfer",
                "baseline_id": args.baseline,
            },
            enqueue_actor="cli",
        )
        print(f"queued {job.job_id} (transfer probe, model={args.model_id}, "
              f"profile={args.profile})")
        try:
            finished = sched.run_job(job)
        except SchedulerError as e:
            if not getattr(args, "wait", False):
                print(f"note: {e}", file=sys.stderr)
                print(f"{job.job_id} stays queued and starts when the GPU "
                      "frees up (pass --wait to block until it runs).")
                return 0
            finished = _await_job(sched, reg, job, tc)
        _print_job_result(sched, finished)
        if finished.state == "done":
            row = reg.get_model(args.model_id)
            entry = None
            for t in (row.transfers if row else []):
                if t.get("profile") == args.profile:
                    entry = t
                    break
            if entry is not None:
                imp = entry.get("improvement")
                mean = entry.get("candidate_mean")
                print(f"transfer probe on {args.profile!r} vs "
                      + (str(entry.get("baseline_id"))
                         if entry.get("baseline_id") else "(candidate-only)")
                      + ": candidate_mean="
                      + (f"{mean:.3f}" if isinstance(mean, (int, float))
                         else str(mean))
                      + ", improvement="
                      + (f"{imp:.3f}" if isinstance(imp, (int, float))
                         else str(imp))
                      + ("  REGRESSION" if entry.get("is_regression") else ""))
        return 0 if finished.state == "done" else 1
    finally:
        reg.close()


# ---------------------------------------------------------------- evolve (#39)

def _print_evolve_result(job) -> None:
    """Compact fitness table + diff for a finished evolve job (CLI + tests)."""
    res = (job.params or {}).get("_result")
    if not isinstance(res, dict):
        return
    print(f"\nfitness (calls {res.get('calls_used')}/{res.get('call_budget')}"
          + (", PARTIAL — budget exhausted" if res.get("partial") else "")
          + "):")
    print(f"  {'variant':<8} {'parent':<7} {'golden':<7} "
          f"{'judge_mean':<10} {'n':<3} note/direction")
    for e in res.get("lineage", []):
        mean = e.get("judge_mean")
        print(f"  {e.get('variant_id', '-'):<8} "
              f"{e.get('parent_id') or '-':<7} "
              f"{str(e.get('golden_pass')):<7} "
              + (f"{mean:<10.3f}" if isinstance(mean, (int, float))
                 else "-         ")
              + f" {e.get('n_scored', 0):<3} "
              + (e.get("rejected_reason") or e.get("note")
                 or e.get("direction") or ""))
    best = res.get("best") or {}
    if res.get("improved") and best.get("text"):
        print(f"\nproposal: variant {best.get('variant_id')} "
              f"(direction {best.get('direction')!r}) beats the current "
              "prompt — diff vs current:")
        print(res.get("diff") or "(no textual diff)")
    else:
        print("\nno variant beat the current prompt — nothing to apply.")
    print(f"\nproposal stored on job {job.job_id} (params._result). Apply with:"
          f"\n  sable evolve apply {job.job_id}")


def cmd_evolve(args: argparse.Namespace) -> int:
    """`sable evolve <profile> …` — plus manual dispatch of the gated
    follow-ups `evolve apply <job_id>` / `evolve dismiss <job_id>`."""
    if args.target in ("apply", "dismiss"):
        if not args.job_id:
            print(f"error: evolve {args.target} needs a job id",
                  file=sys.stderr)
            return 2
        args.job_id = str(args.job_id)
        return (cmd_evolve_apply(args) if args.target == "apply"
                else cmd_evolve_dismiss(args))
    return _evolve_run(args)


def _evolve_run(args: argparse.Namespace) -> int:
    """Enqueue (and usually run) an evolve job for one profile's prompt.

    Mirrors `evaluate shadow`'s queued-or-runs contract. The result is a
    PROPOSAL only — applying needs the explicit `evolve apply` follow-up.
    """
    from .training.scheduler import SchedulerError
    from .evaluation.evolve import evolution_config

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    tc = _training_config(cfg, args.config)

    ecfg = evolution_config(args.config)
    if not bool(ecfg.get("enabled", False)):
        print("error: evolution.enabled is false — refusing to run (set the "
              "master switch in the config to enable prompt evolution)",
              file=sys.stderr)
        return 2

    reg = _open_registry(cfg)
    sched = Scheduler_for(reg, tc)
    try:
        home = resolve_sable_home(cfg)
        names = [p.get("name") for p in (cfg.profiles or [])
                 if isinstance(p, dict)]
        if args.target not in names:
            print(f"error: no profile {args.target!r} in config "
                  f"(known: {', '.join(map(str, names)) or 'none'})",
                  file=sys.stderr)
            return 1
        params = {
            "profile": args.target,
            "golden_file": args.goldens,
            "k": int(args.k),
            "generations": int(args.generations),
        }
        if args.call_budget is not None:
            params["call_budget"] = int(args.call_budget)
        job = sched.submit("evolve", log_dir=home / "train_runs",
                           params=params, enqueue_actor="cli")
        print(f"queued {job.job_id} (evolve, profile={args.target})")
        try:
            finished = sched.run_job(job)
        except SchedulerError as e:
            if not getattr(args, "wait", False):
                print(f"note: {e}", file=sys.stderr)
                print(f"{job.job_id} stays queued and starts when the GPU "
                      "frees up (pass --wait to block until it runs).")
                return 0
            finished = _await_job(sched, reg, job, tc)
        _print_job_result(sched, finished)
        if finished.state == "done":
            _print_evolve_result(finished)
        return 0 if finished.state == "done" else 1
    finally:
        reg.close()


def cmd_evolve_apply(args: argparse.Namespace) -> int:
    """Explicitly apply a finished evolve proposal to its profile config."""
    from .core.config import resolve_config_path
    from .evaluation.evolve import (APPLIED_KEY, RESULT_KEY, EvolveError,
                                    apply_prompt_to_config)

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    reg = _open_registry(cfg)
    try:
        job = reg.get_job(args.job_id)
        if job is None or job.kind != "evolve":
            print(f"error: no evolve job {args.job_id}", file=sys.stderr)
            return 1
        if job.state != "done":
            print(f"error: job {job.job_id} is {job.state}; only finished "
                  "proposals can be applied", file=sys.stderr)
            return 1
        res = (job.params or {}).get(RESULT_KEY)
        profile_name = (job.params or {}).get("profile")
        if not isinstance(res, dict):
            print(f"error: job {job.job_id} has no stored proposal",
                  file=sys.stderr)
            return 1
        if (job.params or {}).get(APPLIED_KEY):
            print(f"error: proposal {job.job_id} was already applied "
                  f"({(job.params or {}).get(APPLIED_KEY)})", file=sys.stderr)
            return 1
        best = res.get("best") or {}
        if not res.get("improved") or not best.get("text"):
            print(f"error: proposal {job.job_id} has no improving variant to "
                  "apply", file=sys.stderr)
            return 1
        path = resolve_config_path(args.config)
        if path is None:
            print("error: no config file to write profiles to (profiles live "
                  "in the config YAML)", file=sys.stderr)
            return 1
        try:
            out = apply_prompt_to_config(Path(path), str(profile_name),
                                         best["text"])
        except EvolveError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        reg.set_job_param(job.job_id, APPLIED_KEY, _utcnow_stamp())
        print(f"applied variant {best.get('variant_id')} to profile "
              f"{profile_name!r}; previous prompt kept in the profile entry "
              f"(system_prompt_prev) and config backup {out['backup']}")
        return 0
    finally:
        reg.close()


def cmd_evolve_dismiss(args: argparse.Namespace) -> int:
    """Flag a proposal dismissed on the job row — history is never deleted."""
    from .evaluation.evolve import DISMISSED_KEY, RESULT_KEY

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    reg = _open_registry(cfg)
    try:
        job = reg.get_job(args.job_id)
        if job is None or job.kind != "evolve":
            print(f"error: no evolve job {args.job_id}", file=sys.stderr)
            return 1
        if not isinstance((job.params or {}).get(RESULT_KEY), dict):
            print(f"error: job {job.job_id} has no stored proposal",
                  file=sys.stderr)
            return 1
        reg.set_job_param(job.job_id, DISMISSED_KEY, _utcnow_stamp())
        print(f"dismissed proposal {job.job_id} (flagged; history kept)")
        return 0
    finally:
        reg.close()


def _utcnow_stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cmd_rft(args: argparse.Namespace) -> int:
    """`sable rft` — rejection-sampling pass as a queued job.

    Mirrors `evolve`'s queued-or-runs contract. The master switch (rft.enabled)
    is checked here for a friendly early refusal AND enforced again by the job
    runner (a job queued before the switch flipped must still refuse at run
    time). With no --seeds/--goldens, prompts come from the uncurated buffer."""
    from .training.scheduler import SchedulerError
    from .training.rejection import rft_config

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)

    rcfg = rft_config(args.config)
    if not rcfg.enabled:
        print("error: rft.enabled is false — refusing to run (set the master "
              "switch in the config to enable rejection sampling)",
              file=sys.stderr)
        return 2
    if not cfg.effective_attach_url and not cfg.gguf_path:
        # The job samples from what SERVES; attach-only resolution inside the
        # runner never spawns. Fail early with actionable advice instead of a
        # queued job that errors on endpoint resolution.
        print("error: no serving endpoint configured — start `sable serve` "
              "/ set attach_url (or --gguf) before rft; rejection sampling "
              "mines the SERVED model and never spawns its own server",
              file=sys.stderr)
        return 2

    reg = _open_registry(cfg)
    tc = _training_config(cfg, args.config)
    sched = Scheduler_for(reg, tc)
    try:
        home = resolve_sable_home(cfg)
        params: dict = {}
        if args.seeds:
            params["seeds"] = list(args.seeds)
        if args.goldens:
            params["golden_file"] = args.goldens
        if args.k is not None:
            params["sample_k"] = int(args.k)
        if args.max_prompts is not None:
            params["max_prompts"] = int(args.max_prompts)
        if args.name:
            params["name"] = args.name
        job = sched.submit("rft", log_dir=home / "train_runs",
                           params=params, enqueue_actor="cli")
        print(f"queued {job.job_id} (rft)")
        try:
            finished = sched.run_job(job)
        except SchedulerError as e:
            if not getattr(args, "wait", False):
                print(f"note: {e}", file=sys.stderr)
                print(f"{job.job_id} stays queued and starts when the GPU "
                      "frees up (pass --wait to block until it runs).")
                return 0
            finished = _await_job(sched, reg, job, tc)
        _print_job_result(sched, finished)
        if finished.state == "done":
            res = (finished.params or {}).get("_result") or {}
            print(f"prompts sampled: {res.get('prompts_sampled', 0)}, "
                  f"verified rows kept: {res.get('rows_kept', 0)}")
            if res.get("dataset_id"):
                print(f"dataset: {res['dataset_dir']} (id {res['dataset_id']})")
            else:
                print("no dataset emitted (no candidate passed verification "
                      "or no prompt candidates) — see manifest-less run notes")
            for w in res.get("warnings", []):
                print(f"note: {w}")
        return 0 if finished.state == "done" else 1
    finally:
        reg.close()


def cmd_mine_goldens(args: argparse.Namespace) -> int:
    """`sable mine-goldens` — distill 👍 traces into golden tasks as a queued
    job. Mirrors `rft`'s queued-or-runs contract: master switch
    (goldens_mine.enabled) checked here for a friendly early refusal AND
    enforced again by the job runner. No endpoint needed unless --use-judge."""
    from .curate.mine_goldens import goldens_mine_config
    from .training.scheduler import SchedulerError

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)

    mcfg = goldens_mine_config(args.config)
    if not mcfg.enabled:
        print("error: goldens_mine.enabled is false — refusing to run (set the "
              "master switch in the config to enable golden mining)",
              file=sys.stderr)
        return 2
    if args.use_judge and not cfg.effective_attach_url and not cfg.gguf_path:
        print("error: --use-judge needs a serving endpoint — start "
              "`sable serve` / set attach_url first (mining itself does "
              "not need one)", file=sys.stderr)
        return 2

    reg = _open_registry(cfg)
    tc = _training_config(cfg, args.config)
    sched = Scheduler_for(reg, tc)
    try:
        home = resolve_sable_home(cfg)
        params: dict = {}
        if args.max is not None:
            params["max_per_run"] = int(args.max)
        if args.use_judge:
            params["use_judge"] = True
        job = sched.submit("mine_goldens", log_dir=home / "train_runs",
                           params=params, enqueue_actor="cli")
        print(f"queued {job.job_id} (mine_goldens)")
        try:
            finished = sched.run_job(job)
        except SchedulerError as e:
            if not getattr(args, "wait", False):
                print(f"note: {e}", file=sys.stderr)
                print(f"{job.job_id} stays queued and starts when the GPU "
                      "frees up (pass --wait to block until it runs).")
                return 0
            finished = _await_job(sched, reg, job, tc)
        _print_job_result(sched, finished)
        if finished.state == "done":
            res = (finished.params or {}).get("_result") or {}
            print(f"goldens mined: {res.get('mined', 0)} "
                  f"(dropped {len(res.get('dropped', []))}, "
                  f"dedup-skipped {res.get('skipped_dedup', 0)})")
            print(f"suite file: {home / 'goldens' / 'mined.jsonl'}")
            for w in res.get("warnings", []):
                print(f"note: {w}")
        return 0 if finished.state == "done" else 1
    finally:
        reg.close()


# ---------------------------------------------------------------- bench (WP1)

def _bench_resolve_ref(ref: str):
    """Resolve a --compare operand: a path to a persisted record JSON, or a
    --label to match against saved records (newest wins). Returns the record
    dict plus a display label, or (None, ref) when nothing matches."""
    from .serving.bench import load_results

    p = Path(ref).expanduser()
    if p.is_file():
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None, ref
        if isinstance(rec, dict) and rec.get("kind"):
            return rec, str(rec.get("label") or p.stem)
        return None, ref
    recs = load_results(label=ref)
    if recs:
        return recs[-1], ref
    return None, ref


def cmd_bench(args: argparse.Namespace) -> int:
    """`sable bench` — throughput measurement harness (WP1).

    Modes: measure (default), ``--noise-floor`` (persist a baseline),
    ``--compare A B`` (token-identity-gated speedup verdict), ``--list``.
    Endpoint resolution reuses the existing serve machinery: --attach target,
    configured/spawned llama-server via ServingSupervisor, or a clear error —
    sable never touches processes it does not own."""
    from .serving import bench as bench_mod

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)
    if getattr(args, "attach", None):
        cfg.attach_url = args.attach

    if args.list:
        recs = bench_mod.load_results()
        if not recs:
            print(f"no persisted bench records under {bench_mod.bench_dir()}")
            return 0
        print(f"{'timestamp':<21} {'kind':<12} {'label':<20} "
              f"{'gen med tok/s':>13}  path")
        for r in recs:
            gen = (r.get("generation") or {}).get("median")
            print(f"{str(r.get('timestamp', '')):<21} "
                  f"{str(r.get('kind', '?')):<12} "
                  f"{str(r.get('label') or '-'):<20} "
                  + (f"{gen:>13.2f}" if isinstance(gen, (int, float))
                     else f"{'-':>13}")
                  + f"  {r.get('_path')}")
        return 0

    if args.compare:
        base, base_lbl = _bench_resolve_ref(args.compare[0])
        cand, cand_lbl = _bench_resolve_ref(args.compare[1])
        if base is None or cand is None:
            missing = args.compare[0] if base is None else args.compare[1]
            print(f"error: no persisted bench record for {missing!r} "
                  "(pass a --label or a JSON path; see `bench --list`)",
                  file=sys.stderr)
            return 2
        cmp_res = bench_mod.compare_records(base, cand)
        print(bench_mod.format_comparison(cmp_res,
                                          baseline_label=base_lbl,
                                          candidate_label=cand_lbl))
        return 0 if cmp_res["valid"] else 1

    supervisor: Optional[ServingSupervisor] = None
    try:
        base_url, model, supervisor = _resolve_endpoint(cfg, spawn_if_needed=True)
        prompts = None
        if getattr(args, "prompts_file", None):
            prompts = bench_mod.load_prompts_file(Path(args.prompts_file))
        record = bench_mod.run_bench(
            base_url, model,
            prompts=prompts,
            runs=args.runs if args.runs is not None else bench_mod.DEFAULT_RUNS,
            max_tokens=(args.max_tokens if args.max_tokens is not None
                        else bench_mod.DEFAULT_MAX_TOKENS),
            label=args.label or "",
            kind=(bench_mod.KIND_NOISE_FLOOR if args.noise_floor
                  else bench_mod.KIND_BENCH),
        )
    except (ServeError, bench_mod.BenchError, AgentBackendError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        if supervisor is not None:
            supervisor.stop()

    saved = bench_mod.save_result(record)
    print(bench_mod.format_report(record, saved_path=saved))
    ti = record.get("token_identity") or {}
    if ti.get("status") != "ok":
        # Token streams drifted inside this measurement: numbers above are
        # informational only — speedup claims built on them are INVALID.
        return 1
    return 0


# ------------------------------------------------------------- calibrate (WP3)

def _calibrate_model_ref(cfg, ref: Optional[str]) -> Optional[str]:
    """Resolve the GGUF to calibrate: --model path (or name under the models
    dir) > configured gguf_path. Returns None when nothing usable is found."""
    from .core.model_discovery import resolve_gguf_path

    candidate = ref or cfg.gguf_path
    if not candidate:
        return None
    p = Path(str(candidate)).expanduser()
    if p.is_file():
        return str(p)
    # Name lookup under the models dir (same convention the serve pickers use).
    try:
        resolved = resolve_gguf_path(str(candidate))
    except Exception:
        resolved = ""
    return resolved if resolved and Path(resolved).is_file() else None


def cmd_calibrate(args: argparse.Namespace) -> int:
    """`sable calibrate` — per-machine autotune (WP3).

    Modes: sweep (default), ``--list``, ``--show``, ``--clear``. The sweep is
    coordinate-wise over run_settings knobs; adoption requires a gain above
    max(3%, noise floor) AND token identity (see serving/calibrate.py). Exit 1
    when nothing could be measured (no server reachable / model unreadable);
    exit 2 for usage mistakes (unknown --knobs, no model to calibrate)."""
    from .serving import calibrate as cal_mod

    _setup_logging(args.verbose, args.config)
    cfg = load_config(args.config)

    if args.clear:
        n = cal_mod.clear_profiles()
        print(f"removed {n} calibration profile(s) under "
              f"{cal_mod.calibrate_dir()}")
        return 0

    if args.list:
        profs = cal_mod.list_profiles()
        if not profs:
            print(f"no calibration profiles under {cal_mod.calibrate_dir()}")
            return 0
        cur_mid = cal_mod.machine_id()
        print(f"{'timestamp':<21} {'machine':<18} {'model hash':<14} "
              f"{'adopted':<8} active  path")
        for p in profs:
            adopted = p.get("adopted_settings") or {}
            active = (p.get("machine_id") == cur_mid)
            print(f"{str(p.get('timestamp', '')):<21} "
                  f"{str(p.get('machine_id', ''))[:17]:<18} "
                  f"{str(p.get('model_hash', ''))[:13]:<14} "
                  f"{len(adopted):<8} "
                  f"{'yes' if active else '-':<7} {p.get('_path')}")
        return 0

    gguf = _calibrate_model_ref(cfg, getattr(args, "model", None))
    if args.show:
        # Active profile for this machine+model when resolvable; else newest.
        prof = None
        if gguf:
            mhash = cal_mod.model_hash(gguf)
            if mhash:
                prof = cal_mod.load_profile(cal_mod.machine_id(), mhash)
        if prof is None:
            profs = cal_mod.list_profiles()
            prof = profs[-1] if profs else None
        if prof is None:
            print("no calibration profile stored "
                  f"under {cal_mod.calibrate_dir()}")
            return 1
        print(cal_mod.format_profile_show(prof))
        return 0

    if not gguf:
        print("error: no GGUF to calibrate — pass --model /path/to/model.gguf "
              "or configure gguf_path (SABLE_GGUF)", file=sys.stderr)
        return 2

    knobs = None
    if getattr(args, "knobs", None):
        knobs = [k.strip() for k in args.knobs.split(",") if k.strip()]
    runs = args.runs if args.runs is not None else cal_mod.DEFAULT_SWEEP_RUNS

    # Explicit type: the attach branch below rebinds this name with a nested
    # def, and Pylance infers the callable from it (a bare `= None` then reads
    # as a type clash). Same seam contract as cal_mod.run_sweep(measure_fn=).
    measure_fn: Optional[Callable[..., Dict[str, Any]]] = None
    attach = getattr(args, "attach", None)
    if attach:
        # Honest attach mode: knobs are NOT applied to a foreign server, so
        # every candidate measures the same endpoint. Kept for harness/testing;
        # the profile records this via endpoint_mode=attach below.
        from .serving.supervisor import require_loopback

        try:
            base_url = require_loopback(attach)
        except ServeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        from .serving.bench import run_bench as _bench_run

        attach_model = cfg.model_name or "attached-model"

        def _attach_measure(settings, n_runs, label):  # noqa: ARG001 (seam parity)
            return _bench_run(base_url, attach_model, runs=n_runs, label=label)

        measure_fn = _attach_measure

    try:
        profile = cal_mod.run_sweep(
            cfg, gguf_path=gguf, runs=runs, knobs=knobs, quick=args.quick,
            measure_fn=measure_fn)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (cal_mod.CalibrateError, ServeError, _bench_error_cls()) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if attach:
        # Honest record of what was measured: an attached foreign server never
        # received the candidate knobs (spawn mode is required for that).
        profile["endpoint_mode"] = "attach"
        cal_mod.save_profile(profile)

    print(cal_mod.format_summary(profile))
    return 0


def _bench_error_cls():
    """The WP1 bench error class (for the except clause above)."""
    from .serving.bench import BenchError

    return BenchError


def _instances_coexist(args: argparse.Namespace):
    """Coexistence engine for the `instances` command family (WP5). Returns
    (coex, cfg, home). The engine self-gates its background behavior on
    serve.* keys; explicit load/unload work with any config."""
    from .serving.coexist import Coexistence, coexist_config, set_active

    cfg = load_config(args.config)
    home = resolve_sable_home(cfg)
    coex = Coexistence(coexist_config(cfg), parent_cfg=cfg, home=home)
    set_active(coex)
    return coex, cfg, home


def cmd_instances_status(args: argparse.Namespace) -> int:
    """`sable instances status` — every named instance + WP5 load state."""
    from .serving import instances as srv_inst

    _setup_logging(args.verbose, args.config)
    coex, cfg, home = _instances_coexist(args)
    rows = srv_inst.load_instances(home, cfg=cfg)
    if not rows:
        print("no serving instances (the reserved default exists once a "
              "gguf_path/attach_url is configured)")
        return 0
    print(f"{'id':<20} {'load_state':<10} {'mode':<7} {'name':<24} base_url")
    for inst in rows:
        iid = str(inst.get("id"))
        state = coex.load_state_of(iid)
        st = srv_inst.read_state(home, iid)
        if iid == srv_inst.DEFAULT_ID and not st:
            from .serving.deploy import read_serve_state
            st = read_serve_state(home)
        mode = (st or {}).get("mode") or ("attach" if inst.get("attach_url")
                                          else "spawn")
        url = (st or {}).get("base_url") if state == "loaded" else None
        print(f"{iid:<20} {state:<10} {mode:<7} "
              f"{str(inst.get('name') or '')[:23]:<24} {url or '-'}")
    return 0


def cmd_instances_load(args: argparse.Namespace) -> int:
    """`sable instances load <id>` — explicit (re)launch (WP5)."""
    _setup_logging(args.verbose, args.config)
    coex, cfg, home = _instances_coexist(args)
    from .serving import instances as srv_inst

    if srv_inst.get_instance(home, args.instance_id, cfg=cfg) is None:
        print(f"error: no server instance {args.instance_id!r} "
              "(see `sable instances status`)", file=sys.stderr)
        return 1
    try:
        st = coex.load_instance(args.instance_id, reason="manual")
    except Exception as e:  # CoexistError and spawn failures alike
        print(f"error: load of {args.instance_id} failed: {e}",
              file=sys.stderr)
        return 1
    print(f"{args.instance_id}: loaded at {st.get('base_url') or '?'}")
    return 0


def cmd_instances_unload(args: argparse.Namespace) -> int:
    """`sable instances unload <id>` — graceful stop + unloaded marker (WP5)."""
    from .serving import instances as srv_inst
    from .serving.watchdog import inflight_count

    _setup_logging(args.verbose, args.config)
    coex, cfg, home = _instances_coexist(args)
    if srv_inst.get_instance(home, args.instance_id, cfg=cfg) is None:
        print(f"error: no server instance {args.instance_id!r} "
              "(see `sable instances status`)", file=sys.stderr)
        return 1
    st = coex._read_state(args.instance_id) or {}
    url = str(st.get("base_url") or "")
    if url and inflight_count(url) > 0:
        print(f"error: {args.instance_id} has a request in flight — retry "
              "the unload shortly", file=sys.stderr)
        return 1
    unloaded = coex.unload_instance(args.instance_id, reason="manual")
    if not unloaded:
        print(f"{args.instance_id}: nothing to unload "
              f"(state={coex.load_state_of(args.instance_id)})")
        return 0
    print(f"{args.instance_id}: unloaded (load it again with "
          f"`sable instances load {args.instance_id}`)")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    """`sable mcp` — MCP control plane over stdio (WP6).

    Serves JSON-RPC 2.0 line-delimited on stdin/stdout (MCP's canonical
    stdio transport): no socket is opened, so the loopback-only rule holds
    by construction. Opt-in command: nothing else changes when it is not
    invoked. See sable/mcp/ and README "sable mcp".
    """
    from .mcp.server import serve_stdio

    # Logging goes to stderr (basicConfig default); stdout is reserved for
    # JSON-RPC frames per the MCP stdio contract.
    _setup_logging(args.verbose, args.config)
    return serve_stdio(config_path=args.config)


def Scheduler_for(reg, tc):
    from .training.scheduler import Scheduler

    # WP5 train+serve handshake: the rescue object self-gates on serve.*
    # keys (all default off => rescue_gate returns False immediately and the
    # gate refusal path is byte-identical to pre-WP5). Construction never
    # raises; a broken config section simply means no rescue.
    coex = None
    try:
        from .serving.coexist import build as _coexist_build
        from .store.paths import resolve_sable_home as _home_of

        cfg = load_config(None)
        coex = _coexist_build(cfg, _home_of(cfg))
    except Exception:  # noqa: BLE001 — dispatch must survive config oddities
        coex = None
    return Scheduler(registry=reg, cfg=tc, coexist=coex)


def _print_job_result(sched, job) -> None:
    print(f"{job.job_id}: {job.state}" + (f" — {job.error}" if job.error else ""))
    for line in sched.tail_log(job, 12):
        print(f"  | {line}")


def main(argv: Optional[list] = None) -> int:
    """Entry point. Exit-code taxonomy in core/exitcodes.py + README
    "Exit codes": 0 ok, 1 generic error, 2 usage (argparse), 3 invalid
    config (consolidated report on stderr), 4 finished-but-corrupt
    artifact, 10 update available, 130 interrupted. argparse itself exits
    2 on bad command lines before this function returns anything."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        # WP2a/2b: config validation failures exit with the distinct code 3
        # and print the FULL consolidated problem listing (every unknown /
        # missing / mistyped field at once) to stderr.
        print(f"error: {e}", file=sys.stderr)
        return EXIT_CONFIG_INVALID
    except FileNotFoundError as e:
        # Only load_config's "Config file not found:" is a config-bucket
        # error (code 3); any other FileNotFoundError keeps its previous
        # behavior (uncaught traceback, exit 1).
        if str(e).startswith("Config file not found:"):
            print(f"error: {e}", file=sys.stderr)
            return EXIT_CONFIG_INVALID
        raise


if __name__ == "__main__":
    raise SystemExit(main())

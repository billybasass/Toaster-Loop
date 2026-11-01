"""Agent runner: boot an agent against the supervisor endpoint.

The runtime is sable's own: a compact OpenAI-compatible chat loop with
native tool calling over httpx, wired to the built-in toolset in
``sable.core.tools`` (read_file / list_dir / write_file / patch /
search_files). No external agent framework is imported — everything the
runtime needs ships with sable and its declared dependencies.

Contract:
    runner = create_runner(cfg, base_url=..., model=..., recorder=...)
    result = runner.run(prompt)          # -> AgentResult(final_response,
                                         #     messages, completed, backend, ...)
    runner.switch_model(base_url, model) # in-place hot-swap (Phase 4)

Trace capture happens post-run from the message list in ``run()`` — the same
canonical point every backend uses — so the experience buffer records every
conversation regardless of how the loop was driven.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import sys
import threading
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from . import errors as error_taxonomy
from .circuit import CircuitOpenError, get_breaker
from .compression import (
    compress_messages,
    summary_from_messages,
)
from .tokenizer import TokenCounter
from .failure_types import (
    FAILURE_BACKEND_CIRCUIT_OPEN,
    FAILURE_BACKEND_ERROR_PREFIX,
    FAILURE_CANCELLED,
    FAILURE_GUARDRAIL_HALT,
    FAILURE_MAX_ITERATIONS,
    FAILURE_VERIFICATION_NUDGE,
)
from .guardrails import (
    ToolLoopGuardrails,
    append_guardrail_guidance,
    synthetic_blocked_result,
    truncate_repetition,
)
from .images import detect_vision_support, expand_message_images, reset_vision_cache
from .path_security import contain
from .spill import (
    DEFAULT_SPILL_RETENTION_DAYS,
    DEFAULT_SPILL_THRESHOLD_CHARS,
    DEFAULT_TURN_RESULT_BUDGET_CHARS,
    enforce_turn_budget,
    is_spillover_read,
    maybe_spill_result,
)
from .redact import redact_text
from .recovery import detect_reasoning_loop, truncate_at_loop
from .think_scrub import StreamingThinkScrubber, strip_think_blocks
from .tool_calls import parse_text_tool_calls, strip_tool_call_markup
from .verification import VerificationLedger
from ..core.config import UNSET, SableConfig
from ..core.tools import (
    CAPABILITY_TOOLS,
    CAPABILITY_TOOL_SCHEMAS,
    TOOL_IMPLS,
    TOOL_SCHEMAS,
)

logger = logging.getLogger("sable.agent")


def _redacted_args(args: Any) -> str:
    """Tool arguments formatted for LOGGING ONLY: JSON-serialized and passed
    through the trace redactor (core/redact.py). Tool args carry file paths,
    URLs and run_command argv and may embed credentials; INFO logs must not
    leak shapes that trace persistence masks (#10). One regex pass per tool
    call is cheap next to executing the tool itself, and reusing the single
    redaction implementation keeps logs and traces consistent by construction.
    """
    try:
        blob = json.dumps(args, default=str)
    except (TypeError, ValueError):
        blob = str(args)
    return redact_text(blob)

# Backlog #9: tools whose args carry filesystem paths and therefore get the
# workspace-containment wrapper (see core/path_security.contain).
_CONTAINED_TOOLS = frozenset(
    {"read_file", "list_dir", "write_file", "patch", "search_files"})

# One-time-per-process startup warning for `agent.workspace_root: null`
# (unrestricted mode), per the backlog #9 spec.
_UNRESTRICTED_WARNED = False


# ---------------------------------------------------------------------------
# Confidence aggregates from logprobs (inspiration pass 4 #40, record-only)
# ---------------------------------------------------------------------------

def _token_logprobs(choice: dict) -> List[float]:
    """Per-token logprobs of one completion choice; [] when absent/malformed.

    Backends without logprobs support simply omit the field — reading is
    defensive so a missing/garbage `logprobs` never raises, matching today's
    behavior exactly."""
    lp = choice.get("logprobs") if isinstance(choice, dict) else None
    if not isinstance(lp, dict):
        return []
    content = lp.get("content")
    if not isinstance(content, list):
        return []
    vals: List[float] = []
    for tok in content:
        v = tok.get("logprob") if isinstance(tok, dict) else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = float(v)
            if math.isfinite(v):
                vals.append(v)
    return vals


def _confidence_aggregate(logprobs: List[float]) -> tuple:
    """(conf_mean, conf_p90_negentropy) for one run's token logprobs.

    Formulas (stdlib math only; documented at the definition site per spec):
      * ``conf_mean`` = arithmetic mean of the per-token logprobs collected
        across every assistant completion of the run. Values are negative;
        closer to 0 means the model was more confident per token.
      * ``conf_p90_negentropy`` = ``exp(logprob)`` of the 90th-percentile
        token (nearest-rank over the sorted logprobs, matching
        store/insights.percentile — mirrored below because core must not
        import store at module level; parity pinned by test_dashboard.py).
        It reads as "the probability even the
        model's uncertain tail tokens were near-certain": high = confident
        throughout, low = a hesitant tail. (Documented proxy for token-level
        entropy — one float instead of a distribution.)

    Aggregate only — per-token arrays are never stored. Returns (None, None)
    when there is no logprob data; exp() underflow clamps to 0 via the -745
    floor.
    """
    if not logprobs:
        return None, None
    mean = sum(logprobs) / len(logprobs)
    ordered = sorted(logprobs)
    # Mirror of store/insights.percentile (canonical): ceil nearest-rank —
    # the old round() used banker's rounding and disagreed at exact .5
    # boundaries. Kept local because core cannot import store at module
    # level; test_dashboard.py parity tests pin the two together.
    idx = max(0, min(len(ordered) - 1, math.ceil(0.90 * len(ordered)) - 1))
    p90_prob = math.exp(max(ordered[idx], -745.0))
    return round(mean, 6), round(p90_prob, 6)


@dataclass
class AgentResult:
    final_response: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    completed: bool = True
    backend: str = "unknown"
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    trace_id: Optional[str] = None  # experience-buffer id when recorded
    # Runner-level failure mode (backlog #2/#7): None on success, else
    # "max_iterations" | "backend_error:<reason>" | FAILURE_GUARDRAIL_HALT
    # | "cancelled" (a human pressed Stop — neither infra nor model fault).
    failure_reason: Optional[str] = None
    # Backlog #6: compaction summary produced during this run (None when no
    # compression fired). Callers that keep long-lived threads persist this
    # alongside history and pass it back as `run(previous_summary=...)` so
    # the next compaction updates the summary instead of restarting cold.
    summary: Optional[str] = None


class AgentBackendError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Built-in agent runtime: OpenAI-compatible chat loop with tool calling
# ---------------------------------------------------------------------------

#: WP8 retry prompt, emitted (once per run, as a synthetic user turn) when a
#: final answer was truncated at a reasoning loop. Same injection mechanism
#: as the verification nudge -- deliberately NOT a second retry path.
_LOOP_NUDGE_TEXT = (
    "[System: Your previous answer was cut short because it entered a "
    "repetition loop (the same thought repeated without progress).\n\n"
    "Continue from where you stopped. Do NOT repeat any text you already "
    "wrote. Either finish the answer directly in a few sentences, or emit "
    "the tool call that actually moves the task forward.]"
)

SYSTEM_PROMPT = (
    "You are sable, a local assistant running entirely on the user's machine. "
    "Use the provided tools to inspect and edit local files when helpful, then "
    "answer concisely."
)

#: Persistent-skill tools (core/skills.py): present in TOOL_SCHEMAS/TOOL_IMPLS
#: but stripped from every runner unless config `skills.enabled` is on — same
#: never-exposed discipline as the capability tools, so a disabled feature is
#: byte-identical to pre-feature prompts.
SKILL_TOOL_NAMES = frozenset(
    {"save_skill", "read_skill", "list_skills", "delete_skill"})


_TODO_BIND_PRUNED = False


def _bind_run_todos(tool_impls: Dict[str, Callable[..., str]]) -> None:
    """Bind a private per-run todos file into todo_write/todo_read impls.

    Files live under ``SABLE_HOME/todos/<run>.json`` (inside the standing
    containment allow-list, so writes keep working). One-time-per-process
    prune of files older than tools.TODO_RUN_RETENTION_DAYS keeps the dir
    bounded. Best-effort: on any failure the impls stay unbound and fall
    back to the historical global todos.json."""
    global _TODO_BIND_PRUNED
    names = [n for n in ("todo_write", "todo_read") if n in tool_impls]
    if not names:
        return
    try:
        import uuid

        from .tools import TODO_RUN_RETENTION_DAYS, _todo_path

        run_dir = _todo_path().parent / "todos"
        path = run_dir / f"run-{uuid.uuid4().hex[:12]}.json"
        for n in names:
            if not isinstance(tool_impls[n], functools.partial):
                tool_impls[n] = functools.partial(tool_impls[n])
            tool_impls[n] = functools.partial(
                tool_impls[n], _todos_path=str(path))
        if not _TODO_BIND_PRUNED and run_dir.is_dir():
            _TODO_BIND_PRUNED = True
            cutoff = time.time() - TODO_RUN_RETENTION_DAYS * 86400
            for f in run_dir.glob("run-*.json"):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                except OSError:
                    pass
    except Exception as e:  # noqa: BLE001 - never break runner construction
        logger.debug("per-run todo binding failed (global file): %s", e)


def _schema_arg_names(schemas: List[dict]) -> Dict[str, Optional[frozenset]]:
    """tool name -> allowed argument names, derived from the exposed schemas.

    Security chokepoint data (see SableAgentClient.__init__): the agent loop
    executes ``impl(**json.loads(arguments))`` with raw model output, and a
    model-supplied kwarg OVERRIDES a functools.partial keyword — so private
    knobs bound at construction (block_sensitive, allow_private_networks,
    lsp_config, _ctx, ...) could be forged through tool args. The loop refuses
    any argument not present in the tool's schema ``properties`` before the
    impl runs. Tools with no discoverable schema map to None and are executed
    unvalidated (custom-schema escape hatch preserved; every built-in ships a
    schema)."""
    out: Dict[str, Optional[frozenset]] = {}
    for s in schemas or []:
        fn = s.get("function") or {}
        name = fn.get("name")
        if not isinstance(name, str):
            continue
        params = fn.get("parameters")
        props = params.get("properties") if isinstance(params, dict) else None
        out[name] = frozenset(props) if isinstance(props, dict) else None
    return out


class SableAgentClient:
    """Built-in agent runtime: chat loop with native tool calling.

    Talks to any OpenAI-compatible /chat/completions endpoint (the supervisor's
    llama-server, or an --attach target). Tool schemas/implementations come
    from sable.core.tools; profiles may restrict them via
    enabled_tool_names.
    """

    name = "sable"

    def __init__(
        self,
        base_url: str,
        model: str,
        tools: Optional[List[dict]] = None,
        tool_impls: Optional[Dict[str, Callable[..., str]]] = None,
        max_iterations: Optional[int] = 12,
        timeout: float = 600.0,  # local GGUFs can be slow on long prompts
        api_key: Optional[str] = None,
        recorder: Any = None,
        system_prompt: Optional[str] = None,
        enabled_tool_names: Optional[List[str]] = None,
        compress_threshold_tokens: int = 0,
        keep_recent_tokens: int = 2048,
        guardrails_enabled: bool = True,
        block_sensitive_files: bool = True,
        lint_on_write: bool = True,
        lsp_config: Optional[Dict[str, Any]] = None,
        workspace_roots: Optional[List[Path]] = None,
        spill_enabled: bool = True,
        spill_threshold_chars: int = DEFAULT_SPILL_THRESHOLD_CHARS,
        spill_threshold_tokens: Optional[int] = None,
        turn_result_budget_chars: int = DEFAULT_TURN_RESULT_BUDGET_CHARS,
        spill_retention_days: int = DEFAULT_SPILL_RETENTION_DAYS,
        spill_home: Optional[Path] = None,
        sampling: Optional[Dict[str, Any]] = None,
        logprobs: bool = False,
        heal_tool_calls: bool = True,
        delegation_context: Optional[Dict[str, Any]] = None,
        search_project: Optional[Dict[str, Any]] = None,
        search_cfg: Any = None,
        capabilities: Optional[Dict[str, bool]] = None,
        web_fetch_config: Optional[Dict[str, Any]] = None,
        run_command_argv_only: bool = False,
        vision: str = "auto",
        verification_nudge: bool = True,
        deny_write_paths: Optional[List[Path]] = None,
        cancel_event: Optional[threading.Event] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        token_counter: Optional[TokenCounter] = None,
        skills_enabled: bool = False,
        plan_ctx: Optional[Dict[str, Any]] = None,
        plans_enabled: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.tools = tools if tools is not None else TOOL_SCHEMAS
        self.tool_impls = tool_impls if tool_impls is not None else dict(TOOL_IMPLS)
        # Security: private knobs are bound via functools.partial below, but
        # Python lets a caller-supplied kwarg OVERRIDE a partial keyword — so
        # raw model args must never reach an impl unchecked. The loop
        # validates every call against the exposed schema's properties
        # (see _tool_arg_names / _run_inner): unknown parameters are refused
        # with "error: bad arguments" and the tool does NOT run. That closes
        # the injection vectors where a model could pass block_sensitive=false,
        # allow_private_networks=true, lsp_config={...} or a forged _ctx.
        self._tool_arg_names = _schema_arg_names(self.tools)
        # Backlog #1: bind the sensitive-read gate into the file tools the
        # same way agent.guardrails was threaded (config `tools.
        # block_sensitive_files`, default true). The flag is bound here, not
        # passed by the model, so it cannot be toggled from the conversation.
        self.block_sensitive_files = bool(block_sensitive_files)
        for _name in ("read_file", "list_dir", "search_files"):
            if _name in self.tool_impls:
                self.tool_impls[_name] = functools.partial(
                    self.tool_impls[_name],
                    block_sensitive=self.block_sensitive_files)
        # Per-run todo files: concurrent dashboard runs (and delegated
        # children) used to clobber each other's plans through the single
        # shared todos.json. Each runner gets its own file under
        # SABLE_HOME/todos/ bound as a private kwarg (the schema gate
        # rejects a model-supplied _todos_path). Unbound callers — direct
        # tool use in tests, CLI one-offs — keep the global file.
        _bind_run_todos(self.tool_impls)
        # Feature-wave Phase 4: bind the post-write check knobs into
        # write_file/patch the same way (config `tools.lint_on_write` +
        # `tools.lsp:`; invisible to the model). Bound BEFORE workspace
        # containment wraps the impls, so containment still short-circuits
        # first and a contained path never reaches the checker.
        self.lint_on_write = bool(lint_on_write)
        self.lsp_config = dict(lsp_config or {})
        for _name in ("write_file", "patch"):
            if _name in self.tool_impls:
                self.tool_impls[_name] = functools.partial(
                    self.tool_impls[_name],
                    lint_on_write=self.lint_on_write,
                    lsp_config=self.lsp_config)
        # Backlog #9: workspace containment — THE single enforcement point.
        # Applied here (tool-impl construction time) so every runner built by
        # create_runner — CLI `sable run` and dashboard chat alike — shares
        # identical semantics, while direct SableAgentClient construction in
        # tests stays unrestricted unless roots are passed. The wrapper is
        # outermost: containment refusals short-circuit before the tool (and
        # before its sensitive-read check) runs.
        # deny_write_paths: paths that stay READABLE through the allow-list
        # but can never be written (the live sable.yaml — SABLE_HOME is
        # allow-listed for todo/trace state, and without this carve-out a
        # prompt-injected agent could rewrite profiles/capabilities/dashboard
        # auth to escalate itself). Read-only tools skip the deny check.
        self.workspace_roots = list(workspace_roots) if workspace_roots else None
        self._deny_write_paths = [Path(d) for d in (deny_write_paths or [])]
        if workspace_roots:
            _read_only_tools = _CONTAINED_TOOLS - {"write_file", "patch"}
            for _name in _CONTAINED_TOOLS:
                if _name in self.tool_impls:
                    self.tool_impls[_name] = contain(
                        workspace_roots,
                        deny_writes=self._deny_write_paths,
                        read_only=_name in _read_only_tools)(
                        self.tool_impls[_name])
        # Capability-gated reach-out tools (per-agent internet / program use):
        # web_fetch and run_command exist in TOOL_IMPLS but are NEVER exposed
        # unless the matching capability is on. The map arrives fully resolved
        # (create_runner merges global `agent.capabilities` defaults with the
        # per-profile `capabilities:` override via tools.resolve_capabilities).
        # Gating removes BOTH schema and impl, so a hallucinated call from an
        # older conversation degrades to "unknown tool" — the model cannot
        # forge a capability through tool args. Config knobs (e.g. allowing
        # private-network fetches) bind via partial before exposure.
        caps = dict(capabilities or {})
        wf_cfg = dict(web_fetch_config or {})
        enabled_cap_tools = {t for t, c in CAPABILITY_TOOLS.items()
                             if caps.get(c)}
        for _tool in CAPABILITY_TOOLS:
            if _tool not in enabled_cap_tools:
                # Disabled: strip schema + impl (never usable/exposed).
                self.tools = [s for s in self.tools
                              if s.get("function", {}).get("name") != _tool]
                self.tool_impls.pop(_tool, None)
            else:
                if _tool == "web_fetch" and _tool in self.tool_impls \
                        and not isinstance(self.tool_impls[_tool],
                                           functools.partial):
                    self.tool_impls[_tool] = functools.partial(
                        self.tool_impls[_tool],
                        allow_private_networks=bool(
                            wf_cfg.get("allow_private_networks", False)))
                if _tool == "run_command" and _tool in self.tool_impls \
                        and run_command_argv_only:
                    # Hardening (config tools.run_command_argv_only): bind the
                    # private argv-mode kwarg — invisible to the model, so it
                    # cannot opt back into a shell through tool args.
                    self.tool_impls[_tool] = functools.partial(
                        self.tool_impls[_tool], argv_mode=True)
                if all(s.get("function", {}).get("name") != _tool
                       for s in self.tools):
                    # Default-schema construction: graft the capability
                    # schema in (it deliberately lives outside TOOL_SCHEMAS).
                    self.tools = self.tools + [CAPABILITY_TOOL_SCHEMAS[_tool]]
        # Persistent skills (core/skills.py): stripped unless config
        # `skills.enabled` — same both-ways removal as capabilities, so a
        # hallucinated call with the knob off degrades to "unknown tool".
        if not skills_enabled:
            self.tools = [s for s in self.tools
                          if s.get("function", {}).get("name")
                          not in SKILL_TOOL_NAMES]
            for _sk in SKILL_TOOL_NAMES:
                self.tool_impls.pop(_sk, None)
        # WP12 plans (core/plans.py): the plan tool family is stripped unless
        # config `plans.enabled` — same both-ways removal as skills/capabilities,
        # so a hallucinated call with the knob off degrades to "unknown tool"
        # and default prompts stay byte-identical. When enabled, bind the
        # private `_plan_ctx` ({project, thread}) into each impl via partial
        # (the schema gate refuses a model-supplied _plan_ctx exactly like
        # _todos_path). plan_progress records this run under task.runs at
        # finalize using the same ctx.
        self._plan_ctx = plan_ctx
        if not plans_enabled:
            from .tools import PLAN_TOOL_NAMES as _PTN
            self.tools = [s for s in self.tools
                          if s.get("function", {}).get("name")
                          not in _PTN]
            for _pt in _PTN:
                self.tool_impls.pop(_pt, None)
        else:
            from .tools import PLAN_TOOL_NAMES as _PTN
            for _pt in _PTN:
                if _pt in self.tool_impls and \
                        not isinstance(self.tool_impls[_pt],
                                       functools.partial):
                    self.tool_impls[_pt] = functools.partial(
                        self.tool_impls[_pt], _plan_ctx=plan_ctx or {})
        # Phase 5e agent profiles: when a profile lists enabled tool names,
        # restrict the loop to that subset. An ENABLED capability implies its
        # tool (unioned into the subset) — the capability toggle alone grants
        # reach-out; listing web_fetch/run_command in `tools` without the
        # capability does nothing (they were stripped above). The filter also
        # applies to a caller-supplied custom `tools` list (#13): profile
        # semantics are "the profile gates tools", so custom schemas are
        # INTERSECTED with the enabled set rather than silently bypassing it,
        # and every dropped tool is logged at WARNING.
        if enabled_tool_names is not None:
            allowed = (set(enabled_tool_names) | enabled_cap_tools
                       | (SKILL_TOOL_NAMES if skills_enabled else set()))
            if tools is not None:
                dropped = sorted({s.get("function", {}).get("name")
                                  for s in self.tools} - allowed)
                if dropped:
                    logger.warning(
                        "profile enabled_tool_names drops %d tool(s) from the "
                        "custom tools list: %s", len(dropped), ", ".join(dropped))
            self.tools = [s for s in self.tools
                          if s.get("function", {}).get("name") in allowed]
            self.tool_impls = {k: v for k, v in self.tool_impls.items()
                               if k in allowed}
        # Security: recompute the schema gate AFTER capability grafting and
        # profile filtering. _tool_arg_names was built from the pre-modification
        # tools list above, where grafted capability schemas (web_fetch,
        # run_command, ...) were absent — they mapped to None and ran
        # UNVALIDATED, so a model-supplied argv_mode/allow_private_networks
        # kwarg could override the private partial bindings. Rebuilding here
        # makes every exposed tool's schema authoritative for its arg gate.
        self._tool_arg_names = _schema_arg_names(self.tools)
        # None (or 0) = no cap on tool-calling turns. The loop is otherwise
        # bounded only by context compression, user cancel, and backend errors.
        self.max_iterations = int(max_iterations) if max_iterations else None
        # Context compression (core/compression.py): when the estimated prompt
        # crosses this many tokens, older turns are replaced by a summary
        # produced by the served model itself. 0 disables it.
        self.compress_threshold_tokens = int(compress_threshold_tokens or 0)
        self.keep_recent_tokens = int(keep_recent_tokens)
        # Improvement #4 (exact tokens): when compression is on, token-gate
        # decisions use llama.cpp's POST /tokenize via TokenCounter instead of
        # the chars/4 estimate. Injected here for tests; otherwise constructed
        # lazily on first gate hit (zero cost while compression is disabled).
        # The counter shares this client's httpx.Client (auth headers + pool)
        # with its own short per-request timeout, and degrades to exactly the
        # old estimate_tokens math whenever the endpoint can't answer.
        self._token_counter = token_counter
        self._tc_injected = token_counter is not None
        # Extra request headers (remote peer agents): merged into every
        # request's header set — e.g. X-Sable-Instance routes a multi-
        # instance peer gateway to the snapshot's named instance. None/empty
        # keeps every existing call byte-identical. Kept on self so
        # switch_model rebuilds the client with them too.
        self._extra_headers = dict(extra_headers or {})
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        headers.update(self._extra_headers)
        self._client = httpx.Client(timeout=timeout, headers=headers)
        self._recorder = recorder
        # Image support (vision input): "auto" probes the endpoint for a
        # vision projector (llama-server mmproj), "true"/"false" force it.
        # When on, image attach markers in tool results are expanded into
        # native image_url parts ON THE OUTBOUND REQUEST only — history,
        # traces and text-only backends keep the lean text stubs (core/images).
        self._vision_mode = str(vision or "auto").strip().lower()
        # Backlog #3: tool-loop guardrails + repetition guard (core/guardrails.py).
        # One controller per run() call; this flag comes from agent.guardrails.
        # When disabled the controller always allows; repetition truncation is
        # gated under the same flag for simplicity (documented deviation: it is
        # lossless-ish safety, but one knob keeps behavior predictable).
        self.guardrails_enabled = bool(guardrails_enabled)
        # Backlog #10: tool-result spill (core/spill.py). Config gate
        # `tools.spill_enabled` (default true); when off, behavior is exactly
        # as before. `spill_home=None` resolves SABLE_HOME lazily per spill.
        self.spill_enabled = bool(spill_enabled)
        self.spill_threshold_chars = int(spill_threshold_chars)
        # Token-accurate spill gate (deferred from improvement #4): when set
        # (> 0) it wins over the chars threshold — exact /tokenize counts via
        # this runner's lazy counter drive the per-result decision. Unset or
        # non-positive keeps today's char comparison byte-for-byte.
        self.spill_threshold_tokens = (int(spill_threshold_tokens)
                                       if spill_threshold_tokens else None)
        self.turn_result_budget_chars = int(turn_result_budget_chars)
        self.spill_retention_days = int(spill_retention_days)
        self._spill_home = spill_home
        # Sampling defaults (config `sampling:` section, reference-style): merged
        # into every chat request so attached servers honor them too. Empty
        # dict => server defaults unchanged. Thinking controls
        # (enable_thinking / reasoning_effort / preserve_thinking) live in the
        # same map and reach llama.cpp via chat_format_template_controls
        # (serving/run_args.request_sampling_payload).
        # Pass 4 #40: `logprobs` is a request-shape gate, not an upstream
        # sampling parameter — popped here so it can never leak verbatim into
        # the payload builder even when callers pass it inside the map.
        self.sampling = dict(sampling or {})
        lp_gate = self.sampling.pop("logprobs", None)
        # Inspiration pass 4 #40 (RECORD-ONLY): when true, _chat asks the
        # backend for per-token content logprobs and each run's responses are
        # folded into two floats on the trace row (see _confidence_aggregate +
        # store/recorder). Backends without logprobs support ignore the flag
        # and answer unchanged. NEVER read for control flow: no retries, no
        # behavior changes based on logprob content ("min_confidence reconsider"
        # is explicitly out of scope). Config knob: `sampling.logprobs`.
        self.logprobs = bool(logprobs) or bool(lp_gate)
        # Per-run token-logprob buffer (fresh list per _run_inner call; only
        # the two-float aggregate is persisted, never this list).
        self._conf_lp: List[float] = []
        # Live event sink (reference-style token streaming): when set to
        # a callable(kind, data), _chat switches the request to stream=true and
        # forwards content/reasoning deltas as they arrive; the agent loop
        # emits tool.call / tool.result / context.usage around each iteration.
        # The dashboard's chat run wires this into its SSE queue.
        self.on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
        # D-17 §3.1 (delegation liveness heartbeat): an OPTIONAL zero-arg
        # hook set by the delegation worker after create_runner
        # (`runner._heartbeat = lambda: delegation.touch(id)`). When set, the
        # agent loop ticks it at every iteration boundary and after each tool
        # call so a long-but-healthy child never looks wedged to the
        # watchdog. Default None -> zero behavior change for every other
        # caller; exceptions from the hook are swallowed (liveness is
        # observability, never fatal).
        self._heartbeat: Optional[Callable[[], None]] = None
        # Set once a streamed chat completion has been consumed (the dashboard
        # uses it to decide whether deltas already went out live).
        self.streamed = False
        # Per-streaming-call flag: at least one message.delta went out for the
        # current stream. The stream->non-stream fallback consults it to emit
        # a single message.reset replacement instead of duplicating text (#6).
        self._stream_emitted_any = False
        # Auto-healing tool calls ("auto-repair" pass): the
        # text-fallback parser that rescues tool calls emitted as fenced
        # JSON/XML text instead of the native field. Default on.
        self.heal_tool_calls = bool(heal_tool_calls)
        # Phase 2 (delegation): run context for list_agents/delegate_task/
        # check_delegation. create_runner always builds one (cfg/base_url/
        # model + caller overrides); direct SableAgentClient construction
        # without it leaves the delegation tools degraded to clean error
        # payloads. Bound into the impls via partial so the model can never
        # forge the context through tool args.
        self._delegation_ctx = delegation_context
        if delegation_context is not None:
            from .tools import DELEGATION_TOOL_NAMES

            for _name in DELEGATION_TOOL_NAMES:
                if _name in self.tool_impls:
                    self.tool_impls[_name] = functools.partial(
                        self.tool_impls[_name], _ctx=delegation_context)
        # Feature-wave Phase 5: per-project codebase indexing. Bind the
        # active project id + workspace into search_codebase (same partial
        # pattern as delegation's _ctx — the model cannot forge it), so the
        # tool routes to <SABLE_HOME>/index/<project_id>.db and never
        # returns cross-project results. No project => default DB.
        # `_cfg` rides along for the same reason: CodebaseIndex reads its
        # `indexing:` section off the config, and a cfg-less call silently
        # reported "indexing is disabled" no matter the real setting.
        _search_bind: Dict[str, Any] = {}
        if search_cfg is not None:
            _search_bind["_cfg"] = search_cfg
        if search_project:
            _search_bind["_project_id"] = search_project.get("id")
            _search_bind["_workspace_dir"] = search_project.get("workspace_dir")
        if _search_bind and "search_codebase" in self.tool_impls:
            self.tool_impls["search_codebase"] = functools.partial(
                self.tool_impls["search_codebase"], **_search_bind)
        # Inspiration pass 3 #19: verification-stop nudge (core/verification.py).
        # A passive per-run ledger of code edits vs fresh evidence; when the
        # loop tries to finish with unverified edits it injects exactly ONE
        # bounded follow-up, and a second unverified finish records
        # failure_reason="verification_nudge" for curation to down-weight.
        self.verification_nudge = bool(verification_nudge)
        # WP8: at most ONE reasoning-loop retry per run (mirrors the
        # verification ledger's single-nudge budget; per-run instance
        # state, never shared across runs).
        self._loop_nudges_sent = 0
        # Cooperative cancel (chat run lifecycle): the dashboard passes a
        # threading.Event that h_chat_cancel sets. The loop polls it at its
        # boundaries (between iterations, before each model call, between
        # batched tool calls); an in-flight HTTP request or tool call runs to
        # completion — cancel stops the RUN, not the syscall. CLI runs (and
        # stubs) pass None and get exactly the old behavior.
        self._cancel_event = cancel_event

    def _cancelled(self) -> bool:
        if self._cancel_event is None:
            return False
        try:
            return bool(self._cancel_event.is_set())
        except Exception:
            return False

    def _emit(self, kind: str, data: Dict[str, Any]) -> None:
        """Forward a live agent-loop event (tool.call / tool.result / ...) to
        the dashboard sink when one is attached; no-op otherwise."""
        if self.on_event is not None:
            try:
                self.on_event(kind, data)
            except Exception:  # a dead sink must never kill the run
                logger.debug("on_event sink failed", exc_info=True)

    def _vision_on(self) -> bool:
        """Vision expansion decision for this client (see __init__)."""
        if self._vision_mode in ("true", "on", "1", "yes"):
            return True
        if self._vision_mode in ("false", "off", "0", "no"):
            return False
        return bool(detect_vision_support(self._client, self.base_url))

    def _chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
        # WP4 watchdog: count this chat call as an IN-FLIGHT request against
        # self.base_url for its whole duration (streaming included — the
        # context manager spans both paths). A cheap lock-guarded dict tally;
        # observable behavior is unchanged, and the engine watchdog only
        # consults it when serve.watchdog_enabled: true.
        from ..serving.watchdog import track_request

        with track_request(self.base_url):
            try:
                return self._chat_inner(messages, tools)
            except AgentBackendError as e:
                # WP5 lazy load: the endpoint may have been idle-unloaded
                # since the run started. Reload it ONCE (only when an active
                # coexistence engine with serve.lazy_load owns this URL in the
                # unloaded state — otherwise reload_for_url returns False and
                # the original error propagates byte-identically to pre-WP5)
                # and retry exactly once. The documented race policy: a request
                # that raced an unload either succeeds after this reload or
                # surfaces the honest backend error.
                if self._coexist_lazy_reload():
                    logger.info("lazy-reloaded unloaded server for %s; "
                                "retrying the request once", self.base_url)
                    _ = e
                    with track_request(self.base_url):
                        return self._chat_inner(messages, tools)
                raise

    def _coexist_lazy_reload(self) -> bool:
        """Ask the process-wide coexistence engine (if any, lazy_load on) to
        relaunch the unloaded instance behind self.base_url. Never raises."""
        try:
            from ..serving.coexist import active as _active_coexist

            coex = _active_coexist()
            return bool(coex and coex.reload_for_url(self.base_url))
        except Exception:  # noqa: BLE001 — serving hook never breaks a run
            return False

    def _chat_inner(self, messages: List[dict],
                    tools: Optional[List[dict]] = None) -> dict:
        # `tools` overrides self.tools when not None (empty list => omit the
        # tools argument entirely — used by the format_error strip-tools hint).
        # Image support: expand attach markers into native image parts on a
        # COPY of the messages (the caller's history and the recorded trace
        # keep the lean text stubs). Marker-free turns and text-only backends
        # are byte-identical to pre-vision behavior.
        if self._vision_on():
            expanded: List[dict] = []
            for m in messages:
                em, _n = expand_message_images(m)
                expanded.append(em)
            messages = expanded
        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        from ..serving.run_args import request_sampling_payload
        payload.update(request_sampling_payload(self.sampling))
        if self.logprobs:
            # Pass 4 #40: content logprobs for the confidence aggregates. A
            # backend without support ignores the flag and answers unchanged —
            # nothing downstream requires the field, so a run can never fail
            # because of it. Judge calls (curate/judge.py) build their own
            # payloads and deliberately do NOT request logprobs.
            payload["logprobs"] = True
        use_tools = self.tools if tools is None else tools
        if use_tools:
            payload["tools"] = use_tools
            payload["tool_choice"] = "auto"
        streamed_partial = False
        if self.on_event is not None and not getattr(self, "_no_stream", False):
            try:
                return self._chat_streaming(payload)
            except AgentBackendError:
                raise
            except Exception as e:  # streaming hiccup: fall back to one-shot
                # Keep on_event attached (loop events keep flowing); just stop
                # attempting stream=true for the rest of this client's life.
                logger.warning("streamed chat failed (%s); retrying non-stream", e)
                self._no_stream = True
                # A partially consumed stream already pushed message.delta
                # events to the dashboard; the one-shot answer below carries
                # the FULL content again, so blindly continuing would leave
                # the UI with duplicated text. Remember to emit a single
                # replacement event once the authoritative reply is in hand.
                streamed_partial = bool(self._stream_emitted_any)
        try:
            r = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
            raise AgentBackendError(
                f"Cannot reach inference server at {self.base_url}: {e}. "
                "Is `sable serve` running?"
            ) from e
        if r.status_code != 200:
            raise AgentBackendError(
                f"Chat completion failed: HTTP {r.status_code} — {r.text[:500]}"
            )
        data = r.json()
        if streamed_partial:
            # Replacement/reset event (option b of the stream-fallback
            # contract): consumers that understand it swap the bubble
            # content for the authoritative one-shot text; older consumers
            # ignore unknown kinds and never see a duplicate emission.
            content = ((data.get("choices") or [{}])[0]
                       .get("message") or {}).get("content")
            self._emit("message.reset",
                       {"text": content if isinstance(content, str) else ""})
        return data

    def _chat_streaming(self, payload: Dict[str, Any]) -> dict:
        """stream=true chat completion with live delta forwarding.

        Emits through self.on_event while reading the upstream SSE stream:
          message.delta   {"text"}                visible content delta
          reasoning.delta {"text"}                thinking/reasoning delta
                                                (llama.cpp `reasoning_content`)
          context.usage   {"prompt_tokens", "completion_tokens"}
        and returns a dict shaped exactly like a non-streaming
        /chat/completions response so the agent loop is stream-agnostic.
        Reasoning deltas are NOT part of the returned message — only the
        visible content and assembled tool_calls (matching what llama.cpp
        itself puts in `message`)."""
        body = dict(payload)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        emit = self.on_event or (lambda kind, data: None)
        # Think-scrub (inspiration pass 3 #17): per-stream stateful scrubber so
        # inline <thinking>/<reasoning> blocks from chat-template-mismatched
        # GGUFs never reach the live deltas or the assembled message. Fresh per
        # call: a mid-stream retry falling back to non-stream starts clean.
        scrubber = StreamingThinkScrubber()
        content_parts: List[str] = []
        lp_content: List[Dict[str, Any]] = []  # pass 4 #40 (record-only)
        tool_calls: Dict[int, Dict[str, Any]] = {}
        # Assembly cursor for servers that omit the per-delta `index` field
        # (#7): without one, every fragment of parallel calls would pile into
        # slot 0. The cursor advances when a fresh tool_call_id (or a fresh
        # name carrying a non-empty id) arrives while an accumulation is open;
        # index-bearing servers keep using the server-provided index verbatim.
        tc_cursor = 0
        tc_open = False  # current cursor slot has accumulated anything yet
        self._stream_emitted_any = False
        usage: Dict[str, Any] = {}
        finish_reason = None
        got_any = False  # any chunk at all (guards servers that ignore stream)
        try:
            with self._client.stream(
                    "POST", f"{self.base_url}/chat/completions",
                    json=body) as r:
                if r.status_code != 200:
                    detail = r.read().decode(errors="replace")[:500]
                    raise AgentBackendError(
                        f"Chat completion failed: HTTP {r.status_code} — {detail}"
                    )
                if "text/event-stream" not in str(
                        r.headers.get("content-type", "")):
                    # Server ignored stream=true and answered with plain JSON:
                    # consume it as a normal completion (no deltas).
                    return r.json()
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    got_any = True
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for ch in chunk.get("choices") or []:
                        delta = ch.get("delta") or {}
                        if delta.get("content"):
                            vis = scrubber.feed(delta["content"])
                            if vis:
                                content_parts.append(vis)
                                self._stream_emitted_any = True
                                emit("message.delta", {"text": vis})
                        rc = delta.get("reasoning_content") or \
                            delta.get("reasoning")
                        if rc:
                            emit("reasoning.delta", {"text": rc})
                        for tcd in delta.get("tool_calls") or []:
                            raw_idx = tcd.get("index")
                            fn = tcd.get("function") or {}
                            if raw_idx is not None:
                                idx = int(raw_idx)
                            else:
                                # Index-less server (#7): keep fragments in the
                                # cursor slot; open a NEW slot when a fresh id
                                # (or a fresh name carrying a non-empty id)
                                # arrives while an accumulation is already open.
                                new_id = tcd.get("id")
                                if tc_open and new_id and \
                                        new_id != tool_calls[tc_cursor]["id"]:
                                    tc_cursor += 1
                                idx = tc_cursor
                            slot = tool_calls.setdefault(idx, {
                                "id": "", "type": "function",
                                "function": {"name": "", "arguments": ""}})
                            if tcd.get("id"):
                                slot["id"] = tcd["id"]
                            if fn.get("name"):
                                slot["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                slot["function"]["arguments"] += \
                                    fn["arguments"]
                            tc_open = True
                        if ch.get("finish_reason"):
                            finish_reason = ch["finish_reason"]
                        # Pass 4 #40: llama.cpp streams per-chunk logprobs on
                        # the choice when requested; keep them so the assembled
                        # response looks exactly like a non-streaming one.
                        lp = ch.get("logprobs")
                        if isinstance(lp, dict):
                            for _tok in lp.get("content") or []:
                                if isinstance(_tok, dict):
                                    lp_content.append(_tok)
        except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
            raise AgentBackendError(
                f"Cannot reach inference server at {self.base_url}: {e}. "
                "Is `sable serve` running?"
            ) from e
        if not got_any:
            # Stream opened but carried nothing: let _chat retry non-stream.
            raise RuntimeError("empty SSE stream")
        # Release prose the scrubber held back at a partial-tag boundary that
        # never resolved into a tag (inside an unterminated block it discards).
        tail = scrubber.flush()
        if tail:
            content_parts.append(tail)
            emit("message.delta", {"text": tail})
        if usage:
            emit("context.usage", {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0)})
        self.streamed = True
        msg: Dict[str, Any] = {"role": "assistant",
                               "content": "".join(content_parts) or None}
        if tool_calls:
            msg["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        choice_out: Dict[str, Any] = {"index": 0, "message": msg,
                                      "finish_reason": finish_reason or "stop"}
        if lp_content:
            choice_out["logprobs"] = {"content": lp_content}
        return {
            "id": "stream", "object": "chat.completion",
            "model": payload.get("model"),
            "choices": [choice_out],
            "usage": usage,
        }

    def _summarize(self, prompt: str) -> Optional[str]:
        """Summarizer for compression: one tools-off chat call.

        Backlog #3: a degenerate summarizer reply (repetition loop) is
        truncated at the loop start before it can be stitched into history.
        """
        # Sampling must go through the same request-shaping router as every
        # other call (#3): raw thinking keys spread verbatim are rejected with
        # a 400 by strict OpenAI-compatible servers, which used to kill
        # compaction silently (this method swallows non-200s).
        from ..serving.run_args import request_sampling_payload

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        body.update(request_sampling_payload(self.sampling))
        data = self._client.post(
            f"{self.base_url}/chat/completions", json=body,
        )
        if data.status_code != 200:
            return None
        msg = ((data.json().get("choices") or [{}])[0]).get("message") or {}
        text = msg.get("content")
        if not (isinstance(text, str) and text.strip()):
            return None
        # The summary is stitched into history: scrub leaked reasoning before
        # it can take root there too (inspiration pass 3 #17).
        text = strip_think_blocks(text)
        if self.guardrails_enabled:
            text = truncate_repetition(text)
        return text

    # -- retry (backlog #2): capped, classified, exponential backoff ---------

    #: injectable sleep for tests (module default time.sleep)
    _sleep = staticmethod(time.sleep)
    RETRY_BACKOFF_S = (1.0, 2.0, 4.0)  # capped schedule between attempts

    def retry_chat(self, messages: List[dict], max_attempts: int = 3,
                   previous_summary: Optional[str] = None) -> dict:
        """`_chat()` with classified retry (backlog item #2).

        Retries only when core/errors.py classifies the failure as retryable.
        Recovery hints applied once each, then retried:
          * context_overflow / payload_too_large -> force compression
          * format_error (malformed tool args)   -> strip tools arg
          * model_not_found (hot-swap race)      -> re-list /v1/models
        Raises the last AgentBackendError when attempts are exhausted or the
        error is not retryable.

        `previous_summary` (backlog #6): threaded into the forced-compression
        path so an in-run emergency compaction updates the rolling summary
        instead of discarding it.
        """
        # Inspiration pass 4 #28: per-backend circuit breaker, keyed by
        # base_url so every run AND the judge share one view of this backend.
        breaker = get_breaker(self.base_url)
        override_tools: Optional[List[dict]] = None
        compressed_once = False
        stripped_once = False
        relisted_once = False
        last_exc: Optional[AgentBackendError] = None

        for attempt in range(max(1, max_attempts)):
            # Checked before EVERY attempt (including the first): while the
            # breaker is open the call fails fast with CircuitOpenError —
            # no backoff cycles burned on a wedged llama.cpp instance.
            if not breaker.is_allowed():
                snap = breaker.snapshot()
                raise CircuitOpenError(
                    f"backend circuit open for {self.base_url}: "
                    f"{snap['failures']} consecutive failures, probe again in "
                    f"{snap['seconds_until_probe']:.0f}s "
                    "(recovers automatically after the cooldown)")
            if attempt:
                self._sleep(self.RETRY_BACKOFF_S[min(attempt - 1,
                                                     len(self.RETRY_BACKOFF_S) - 1)])
            try:
                if override_tools is None:
                    data = self._chat(messages)
                else:
                    data = self._chat(messages, tools=override_tools)
            except AgentBackendError as e:
                # Breaker semantics (spec #28): every failed attempt counts as
                # one failure; the first success resets the streak. A wedged
                # backend therefore opens the circuit within ~two retry_chats
                # (5 consecutive transport failures), while a healthy backend
                # that merely hiccups once closes it again on the next reply —
                # exactly what makes *consecutive* counting meaningful here.
                breaker.record_failure()
                last_exc = e
                ce = error_taxonomy.classify_exception(e)
                logger.warning("chat attempt %d/%d failed (%s%s): %s",
                               attempt + 1, max_attempts, ce.reason,
                               ", retryable" if ce.retryable else "", e)
                if not ce.retryable:
                    raise

                # one-shot recovery hints (each consumes a retry attempt)
                if ce.should_compress and not compressed_once \
                        and self.compress_threshold_tokens > 0:
                    # force the compression path regardless of estimate
                    messages[:] = compress_messages(
                        messages, self._summarize, threshold_tokens=1,
                        keep_recent_tokens=self.keep_recent_tokens,
                        previous_summary=previous_summary,
                        counter=self._tokens(),
                    )
                    compressed_once = True
                    continue
                if ce.strip_tools and not stripped_once:
                    override_tools = []
                    stripped_once = True
                    continue
                if ce.reason == error_taxonomy.FailoverReason.MODEL_NOT_FOUND \
                        and not relisted_once:
                    self._relist_models()
                    relisted_once = True
                    continue
            else:
                # Success closes the streak (and any half-open probe).
                breaker.record_success()
                return data

        assert last_exc is not None
        raise last_exc

    def _relist_models(self) -> None:
        """Re-probe /v1/models after a model_not_found (deploy hot-swap race)."""
        try:
            r = self._client.get(f"{self.base_url}/models")
            if r.status_code == 200:
                ids = [m.get("id") for m in (r.json().get("data") or [])]
                if ids and self.model not in ids:
                    logger.info("model %r gone; switching to served %r",
                                self.model, ids[0])
                    self.model = ids[0]
        except Exception as e:  # best-effort: retry proceeds regardless
            logger.debug("model re-list failed: %s", e)

    def _tokens(self) -> TokenCounter:
        """TokenCounter for the active endpoint (lazy; improvement #4).

        Shares this client's httpx.Client so the tokenize probe inherits the
        Authorization header and connection pool; the counter keeps its own
        short timeout + circuit breaker. Never raises: a dead /tokenize just
        means chars/4 answers (see core/tokenizer.py).
        """
        if self._token_counter is None:
            self._token_counter = TokenCounter(
                self.base_url, client=self._client)
        return self._token_counter

    def _maybe_compress(
        self, messages: List[dict], previous_summary: Optional[str] = None
    ) -> tuple:
        """Compress the message list when it approaches the context window.

        Backlog #6: returns ``(messages, summary)`` — the rolling compaction
        summary (updated by this call when compression fired, otherwise the
        incoming ``previous_summary``, unchanged). Callers persist the summary
        alongside history and pass it back on the next turn so successive
        compactions iteratively update instead of re-summarizing cold.

        Improvement #4: the gate uses exact /tokenize counts (whole-message
        granularity — one batched count per decision point, never per delta).
        """
        if self.compress_threshold_tokens <= 0:
            return messages, previous_summary
        counter = self._tokens()
        if counter.count_messages(messages) <= self.compress_threshold_tokens:
            return messages, previous_summary
        before = len(messages)
        compressed = compress_messages(
            messages, self._summarize,
            threshold_tokens=self.compress_threshold_tokens,
            keep_recent_tokens=self.keep_recent_tokens,
            previous_summary=previous_summary,
            counter=counter,
        )
        if len(compressed) != before:
            logger.info("context compressed: %d -> %d messages (~%d tokens)",
                        before, len(compressed),
                        counter.count_messages(compressed))
        summary = summary_from_messages(compressed) or previous_summary
        return compressed, summary

    def run(self, task: str, history: Optional[List[Dict[str, Any]]] = None, *,
            previous_summary: Optional[str] = None) -> AgentResult:
        """Run one agent turn.

        `previous_summary` (backlog #6): rolling compaction summary from an
        earlier turn on this thread; it is threaded into every compression
        performed during this run and the (possibly updated) result is
        returned as ``AgentResult.summary``.
        """
        t0 = time.monotonic()
        # Phase 2 (delegation): expose this run in the in-process registry so
        # list_agents can flag profiles that are currently mid-run, and so
        # child traces can link back by parent_run_id. Registry key is the
        # context's run_id when provided (the dashboard's chat run id).
        from . import delegation as _delegation

        reg_key: Optional[str] = None
        if self._delegation_ctx is not None:
            reg_key = _delegation.register_run(
                profile=self._delegation_ctx.get("profile"),
                project=self._delegation_ctx.get("project"),
                run_id=self._delegation_ctx.get("run_id"))
        try:
            return self._run_inner(task, history=history,
                                   previous_summary=previous_summary, t0=t0)
        finally:
            if reg_key is not None:
                _delegation.unregister_run(reg_key)

    def _run_inner(self, task: str, history: Optional[List[Dict[str, Any]]] = None, *,
                   previous_summary: Optional[str] = None,
                   t0: float) -> AgentResult:
        """The actual agent loop (split out of run() for registry cleanup)."""
        tokens_in = 0
        tokens_out = 0
        # Pass 4 #40: fresh per-run logprob buffer (aggregate-only persistence
        # happens in _finish; this list never leaves the client).
        self._conf_lp = []
        # Backlog #3: fresh guardrail controller per run (per-run counters).
        guardrails = ToolLoopGuardrails(enabled=self.guardrails_enabled)
        # Pass 3 #19: passive edit-vs-evidence ledger for the verification nudge.
        ledger = VerificationLedger() if self.verification_nudge else None
        messages: List[dict] = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": task})
        messages, summary = self._maybe_compress(messages, previous_summary)

        # Backlog #10: spill happens AFTER tool exec and AFTER the after_call
        # guardrail evaluation (loop detection needs the REAL failure text, not
        # stubs), but BEFORE the result is appended to `messages` — so the
        # trace recorded from message content at recorder time keeps the stub
        # (datasets stay lean). One assistant turn = one iteration's tool
        # calls; the per-turn aggregate budget runs after the batch.
        def _spill(name: str, tc_id: str, content: str) -> str:
            if not self.spill_enabled:
                return content
            # Token gate only when configured: with the knob unset we never
            # touch _tokens(), so runs stay byte-identical to before (no
            # tokenize probes at all). One count per tool output, here at
            # decision time — never per streaming chunk.
            use_tokens = self.spill_threshold_tokens is not None
            stub = maybe_spill_result(
                name, tc_id, content, self._spill_home,
                threshold=self.spill_threshold_chars,
                retention_days=self.spill_retention_days,
                threshold_tokens=self.spill_threshold_tokens,
                counter=self._tokens() if use_tokens else None)
            return stub if stub is not None else content

        loop = (range(self.max_iterations) if self.max_iterations is not None
                else iter(range(sys.maxsize)))  # no cap (max_iterations=None)
        for _ in loop:
            # D-17 §3.1: delegation liveness tick (loop boundary). No-op when
            # no hook is set (every non-delegation caller).
            if self._heartbeat is not None:
                try:
                    self._heartbeat()
                except Exception:  # liveness never breaks the loop
                    pass
            # Cancel boundary 1: before the next model call — a stop pressed
            # during the previous tool batch is honored here, after all its
            # results are already in `messages` (the trace stays consistent).
            if self._cancelled():
                return self._finish(
                    task,
                    AgentResult(
                        final_response="(stopped: cancelled by user)",
                        messages=messages,
                        completed=False,
                        backend=self.name,
                        tokens_in=tokens_in or None,
                        tokens_out=tokens_out or None,
                        failure_reason=FAILURE_CANCELLED,
                    ),
                    t0,
                    summary=summary,
                )
            try:
                data = self.retry_chat(messages, previous_summary=summary)
            except CircuitOpenError as e:
                # Fail-fast refusal from the breaker (pass 4 #28): infra fault,
                # classified so curation down-weights it like a connection error.
                return self._finish(
                    task,
                    AgentResult(
                        final_response=f"(stopped: {e})",
                        messages=messages,
                        completed=False,
                        backend=self.name,
                        tokens_in=tokens_in or None,
                        tokens_out=tokens_out or None,
                        failure_reason=FAILURE_BACKEND_CIRCUIT_OPEN,
                    ),
                    t0,
                    summary=summary,
                )
            except AgentBackendError as e:
                # Final failure: distinguish the runner-level failure mode so
                # curation can exclude infra faults (backlog #2/#7). Raw
                # messages stay untouched; only footer metadata gains this.
                reason = error_taxonomy.classify_exception(e).reason
                return self._finish(
                    task,
                    AgentResult(
                        final_response=f"(stopped: backend error — {e})",
                        messages=messages,
                        completed=False,
                        backend=self.name,
                        tokens_in=tokens_in or None,
                        tokens_out=tokens_out or None,
                        failure_reason=FAILURE_BACKEND_ERROR_PREFIX + reason,
                    ),
                    t0,
                    summary=summary,
                )
            # Backlog #6: a forced compaction inside retry_chat rewrote the
            # message list in place — refresh the rolling summary so later
            # compactions (and the caller) see the updated version.
            summary = summary_from_messages(messages) or summary
            usage = data.get("usage") or {}
            tokens_in += int(usage.get("prompt_tokens") or 0)
            tokens_out += int(usage.get("completion_tokens") or 0)
            choice = (data.get("choices") or [{}])[0]
            if self.logprobs:
                # Pass 4 #40: accumulate for the two-float trace aggregate.
                # Record-only: no branch here depends on the values, and a
                # backend that returned nothing simply contributes nothing.
                try:
                    self._conf_lp.extend(_token_logprobs(choice))
                except Exception:  # defensive: recording never breaks a run
                    logger.debug("logprob accumulation failed (ignored)",
                                 exc_info=True)
            msg = choice.get("message") or {}
            tool_calls = msg.get("tool_calls") or []

            # Text tool-call fallback: weak local GGUFs often emit the call as
            # fenced JSON / XML-ish text instead of the native field.
            if (self.heal_tool_calls and not tool_calls
                    and msg.get("content")):
                tool_calls = parse_text_tool_calls(msg["content"])
                if tool_calls:
                    logger.info("parsed %d text tool call(s) from content",
                                len(tool_calls))
                    msg = {**msg,
                           "content": strip_tool_call_markup(msg["content"])}

            if not tool_calls:
                # Backlog #3: a final answer dominated by a repetition loop is
                # truncated at the loop start before it is returned/recorded.
                final = msg.get("content") or ""
                # Inspiration pass 3 #17: non-streaming backends (and the
                # stream fallback) never saw the scrubber — apply the whole-
                # string pass here, THE chokepoint before history append and
                # trace record, so recorded SFT material stays clean.
                final = strip_think_blocks(final)
                if self.guardrails_enabled:
                    final = truncate_repetition(final)
                # WP8: reasoning-loop guard. A cycling answer that produced
                # no tool call is truncated at the FIRST lap (everything
                # before the loop is real content and survives). The retry
                # below reuses the verification-nudge mechanism shape --
                # synthetic user turn, at most one per run -- it is not a
                # second retry path. Detection is conservative (see
                # core/recovery.py); lists/code/healthy long answers do
                # not trip it, and failures to detect fail OPEN (full
                # answer kept).
                loop_verdict = None
                if self.guardrails_enabled:
                    loop_verdict = detect_reasoning_loop(final)
                    if loop_verdict.loop:
                        final = truncate_at_loop(final, loop_verdict)
                        logger.info(
                            "reasoning loop detected (reason=%s, laps=%d); "
                            "truncated at first occurrence",
                            loop_verdict.reason, loop_verdict.repeats)
                messages.append({"role": "assistant", "content": final})

                # -- verification-stop guard (pass 3 #19) ---------------------
                # Finishing with code edited and no fresh evidence: inject the
                # ONE allowed nudge as a synthetic user turn and keep looping.
                # A second unverified finish is recorded, not nagged again.
                if ledger is not None:
                    nudge = ledger.build_nudge()
                    if nudge is not None:
                        logger.info(
                            "verification nudge: %d path(s) edited without "
                            "fresh evidence", len(ledger.pending_paths()))
                        self._emit("verification.nudge",
                                   {"paths": ledger.pending_paths()})
                        messages.append({"role": "user", "content": nudge})
                        continue
                    finish_failure = (FAILURE_VERIFICATION_NUDGE
                                      if ledger.pending_paths() else None)
                    if finish_failure:
                        logger.info("run ended unverified after nudge "
                                    "(failure_reason=%s)", finish_failure)
                        return self._finish(
                            task,
                            AgentResult(
                                final_response=final,
                                messages=messages,
                                completed=True,
                                backend=self.name,
                                tokens_in=tokens_in or None,
                                tokens_out=tokens_out or None,
                                failure_reason=finish_failure,
                            ),
                            t0,
                            summary=summary,
                        )
                # WP8 loop retry: same synthetic-user-turn mechanism as the
                # verification nudge, one per run; if that nudge owned this
                # turn (continued above) we never get here, so a turn gets
                # at most one retry of any kind.
                if (loop_verdict is not None and loop_verdict.loop
                        and self._loop_nudges_sent < 1):
                    self._loop_nudges_sent += 1
                    logger.info(
                        "loop nudge: retrying once after loop "
                        "(reason=%s)", loop_verdict.reason)
                    self._emit("agent.loop_nudge", {
                        "reason": loop_verdict.reason,
                        "repeats": loop_verdict.repeats,
                    })
                    messages.append({"role": "user",
                                     "content": _LOOP_NUDGE_TEXT})
                    continue
                return self._finish(
                    task,
                    AgentResult(
                        final_response=final,
                        messages=messages,
                        completed=True,
                        backend=self.name,
                        tokens_in=tokens_in or None,
                        tokens_out=tokens_out or None,
                    ),
                    t0,
                    summary=summary,
                )

            # Same chokepoint for intermediate assistant turns: inline thinking
            # ahead of a tool call must not enter history or the trace either.
            _tc_content = msg.get("content")
            if isinstance(_tc_content, str):
                _tc_content = strip_think_blocks(_tc_content)
            messages.append(
                {
                    "role": "assistant",
                    "content": _tc_content,
                    "tool_calls": tool_calls,
                }
            )
            # Backlog #10: tool messages produced by THIS assistant turn,
            # collected for the per-turn aggregate budget after the batch.
            turn_tool_msgs: List[dict] = []
            skip_spill: set = set()
            for tc in tool_calls:
                # Cancel boundary 2: between batched tool calls. In-flight
                # calls finish (cancel stops the run, not the syscall), then
                # we stop: pending calls get synthetic "skipped (run cancelled)"
                # results so the message list stays well-formed — every
                # assistant tool_call has exactly one tool result.
                if self._cancelled():
                    for rest in tool_calls[tool_calls.index(tc):]:
                        rfn = rest.get("function") or {}
                        skipped = {
                            "role": "tool",
                            "tool_call_id": rest.get("id", ""),
                            "name": rfn.get("name", ""),
                            "content": "error: skipped — run cancelled by user",
                        }
                        turn_tool_msgs.append(skipped)
                        messages.append(skipped)
                        self._emit("tool.result", {
                            "id": rest.get("id", ""), "name": rfn.get("name", ""),
                            "ok": False,
                            "preview": skipped["content"],
                            "truncated": False})
                    return self._finish(
                        task,
                        AgentResult(
                            final_response="(stopped: cancelled by user)",
                            messages=messages,
                            completed=False,
                            backend=self.name,
                            tokens_in=tokens_in or None,
                            tokens_out=tokens_out or None,
                            failure_reason=FAILURE_CANCELLED,
                        ),
                        t0,
                        summary=summary,
                    )
                fn = (tc.get("function") or {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                impl = self.tool_impls.get(name)
                if not isinstance(args, dict):
                    args = {}
                # Live event: the dashboard renders the tool card as soon as
                # the call is parsed (reference-style inline streaming).
                self._emit("tool.call", {"id": tc.get("id", ""),
                                         "name": name, "args": args})
                # -- guardrail gate before execution (backlog #3) -------------
                decision = guardrails.before_call(name, args)
                if not decision.allows_execution:
                    # block: skip execution; synthetic error tool result keeps
                    # the loop going (counted as an error in the trace).
                    logger.warning("guardrail %s for %s: %s",
                                   decision.code, name, decision.message)
                    turn_tool_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "name": name,
                            "content": synthetic_blocked_result(decision),
                        }
                    )
                    messages.append(turn_tool_msgs[-1])
                    blocked = turn_tool_msgs[-1]["content"]
                    self._emit("tool.result", {
                        "id": tc.get("id", ""), "name": name, "ok": False,
                        "preview": blocked[:8000],
                        "truncated": len(blocked) > 8000})
                    continue

                # -- schema-args gate (security): refuse model-supplied
                # parameters that are not in the tool's schema BEFORE the impl
                # runs. Raw kwargs would otherwise override the
                # functools.partial knobs bound at construction
                # (block_sensitive / allow_private_networks / lsp_config /
                # _ctx ...), letting a prompt-injected model disable its own
                # guardrails or forge a delegation context. Placed after
                # before_call so repeated invalid calls still feed the loop
                # guard, and never reach impls (hence never the ledger).
                allowed_args = self._tool_arg_names.get(name)
                if allowed_args is not None and impl is not None:
                    extra = sorted(set(args) - set(allowed_args))
                    if extra:
                        content = (
                            f"error: bad arguments for {name}: unknown "
                            f"parameter(s) {', '.join(extra)} — this runtime "
                            "rejects them (config knobs cannot be set from "
                            "tool args); retry with only the schema "
                            "parameters")
                        logger.warning(
                            "rejected tool call %s with unknown arg(s): %s",
                            name, ", ".join(extra))
                        turn_tool_msgs.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.get("id", ""),
                                "name": name,
                                "content": content,
                            }
                        )
                        messages.append(turn_tool_msgs[-1])
                        self._emit("tool.result", {
                            "id": tc.get("id", ""), "name": name, "ok": False,
                            "preview": content, "truncated": False})
                        continue

                if impl is None:
                    content = f"error: unknown tool {name!r}"
                else:
                    try:
                        content = impl(**args)
                    except TypeError as e:
                        content = f"error: bad arguments for {name}: {e}"
                    except Exception as e:  # tools must never crash the loop
                        content = f"error: {name} failed: {e}"
                # Tool args go through the trace redactor before INFO (#10):
                # they carry file paths, URLs and run_command argv that may
                # embed credentials.
                logger.info("tool %s(%s) -> %d chars",
                            name, _redacted_args(args), len(content))

                # Pass 3 #19: ledger observes REAL results (before spill can
                # stub them away — the ✓/✗ post-write marks must be visible).
                if ledger is not None:
                    ledger.observe(name, args, content)

                # -- guardrail observation after execution (backlog #3) -------
                decision = guardrails.after_call(name, args, content)
                if decision.action == "halt":
                    # halt: stop the loop with a model-ish failure reason so
                    # the curater keeps it as DPO negative material.
                    logger.warning("guardrail halt for %s: %s",
                                   name, decision.message)
                    content = append_guardrail_guidance(content, decision)
                    # Deliberately NOT spilled: this terminal result (with its
                    # guardrail guidance) is DPO-negative material and must
                    # stay inline in the trace.
                    turn_tool_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "name": name,
                            "content": content,
                        }
                    )
                    messages.append(turn_tool_msgs[-1])
                    return self._finish(
                        task,
                        AgentResult(
                            final_response=f"(stopped: guardrail halt — {decision.message})",
                            messages=messages,
                            completed=False,
                            backend=self.name,
                            tokens_in=tokens_in or None,
                            tokens_out=tokens_out or None,
                            failure_reason=(f"{FAILURE_GUARDRAIL_HALT}:{decision.code}"),
                        ),
                        t0,
                        summary=summary,
                    )
                # -- spill (backlog #10): AFTER after_call saw the real ----
                # content (loop detection needs actual failure text, not a
                # stub), BEFORE append (the trace records what is appended).
                # Reads of spillover files are exempt: re-spilling them would
                # turn the stub's "read_file the path" hint into a doom loop.
                if is_spillover_read(name, args, self._spill_home):
                    skip_spill.add(len(turn_tool_msgs))
                else:
                    content = _spill(name, tc.get("id", ""), content)
                if decision.action == "warn":
                    # warn: nudge appended to this tool result (same role the
                    # chat template already tolerates for tool output). After
                    # spill so it stays visible in context, not buried in the
                    # spilled file.
                    content = append_guardrail_guidance(content, decision)
                turn_tool_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "name": name,
                        "content": content,
                    }
                )
                messages.append(turn_tool_msgs[-1])
                self._emit("tool.result", {
                    "id": tc.get("id", ""), "name": name,
                    "ok": not content.startswith("error:"),
                    "preview": content[:8000],
                    "truncated": len(content) > 8000})
                # D-17 §3.1: liveness tick after each tool in the batch — a
                # slow-but-healthy tool call keeps the child looking alive.
                if self._heartbeat is not None:
                    try:
                        self._heartbeat()
                    except Exception:  # liveness never breaks the loop
                        pass

            # Backlog #10 level 3: per-turn aggregate budget over this turn's
            # tool results (dicts are shared with `messages`, so spilling here
            # rewrites the appended rows in place). Idempotent + stub-skipping.
            if self.spill_enabled and turn_tool_msgs:
                enforce_turn_budget(
                    turn_tool_msgs, self.turn_result_budget_chars,
                    self._spill_home, retention_days=self.spill_retention_days,
                    exempt=skip_spill)

        return self._finish(
            task,
            AgentResult(
                final_response="(stopped: max tool-calling iterations reached)",
                messages=messages,
                completed=False,
                backend=self.name,
                tokens_in=tokens_in or None,
                tokens_out=tokens_out or None,
                failure_reason=FAILURE_MAX_ITERATIONS,
            ),
            t0,
            summary=summary,
        )

    def switch_model(self, new_base_url: str, new_model: str,
                     api_key: Optional[str] = None) -> bool:
        """Re-point the client at a new endpoint (Phase 4 hot-swap)."""
        try:
            self._client.close()
        except Exception:
            pass
        self.base_url = new_base_url.rstrip("/")
        self.model = new_model
        # Tokenizer is an endpoint property: drop the lazily-built counter so
        # it re-binds to the new base_url (an injected test double stays).
        if not self._tc_injected:
            self._token_counter = None
        # Vision support is a property of the served model: forget cached
        # probe results so the new endpoint gets re-detected.
        reset_vision_cache()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        headers.update(self._extra_headers)
        self._client = httpx.Client(timeout=self._client.timeout, headers=headers)
        return True

    def _finish(self, task: str, result: AgentResult, t0: float, *,
                summary: Optional[str] = None) -> AgentResult:
        """Native trace recording point.

        `summary` (backlog #6): the rolling compaction summary for this run;
        attached to the AgentResult (not the trace — traces keep raw messages).
        """
        result.summary = summary
        # Pass 4 #40: fold this run's collected token logprobs into two floats
        # (None/None when the backend returned no logprobs at all).
        conf_mean = conf_p90_negentropy = None
        if self.logprobs and self._conf_lp:
            try:
                conf_mean, conf_p90_negentropy = _confidence_aggregate(
                    self._conf_lp)
            except Exception:  # never fail (or alter) a run over bookkeeping
                logger.debug("confidence aggregate failed (ignored)",
                             exc_info=True)
                conf_mean = conf_p90_negentropy = None
        if self._recorder is not None:
            rec = self._recorder.record(
                task=task,
                messages=result.messages,
                backend=self.name,
                model_alias=self.model,
                base_url=self.base_url,
                completed=result.completed,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                duration_s=round(time.monotonic() - t0, 3),
                failure_reason=result.failure_reason,
                conf_mean=conf_mean,
                conf_p90_negentropy=conf_p90_negentropy,
            )
            if rec is not None:
                result.trace_id = rec.trace_id
        # WP12: fill this run's trace id under every plan task the run touched
        # via plan_progress (the tool recorded thread/time mid-run; the trace
        # only exists now). Best-effort — a store error never alters a result.
        touched = list((self._plan_ctx or {}).get("_touched") or [])
        if touched and result.trace_id:
            try:
                from . import plans as _plans

                # Lock the read-modify-write so a concurrent plan edit (or a
                # second finishing run) can't be clobbered by this bookkeeping.
                with _plans.lock():
                    pstore = _plans.load_plan(
                        (self._plan_ctx or {}).get("project"))
                    for tid in touched:
                        _plans.record_run(pstore, tid,
                                          trace_id=result.trace_id)
                    _plans.save_plan(pstore)
            except Exception:  # noqa: BLE001 - bookkeeping only
                logger.debug("plan run linkage failed (ignored)", exc_info=True)
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def resolve_workspace_roots(cfg: SableConfig,
                            workspace_root_override: Optional[str] = None,
                            project_workspace_dir: Optional[str] = None,
                            ) -> Optional[List[Path]]:
    """Resolve the containment roots for a runner (backlog item #9).

    Returns None for unrestricted mode. Semantics:
      * `agent.workspace_root` unset  -> primary root = os.getcwd() captured
        here, at runner creation time.
      * explicit null in config       -> unrestricted (None), with a one-time
        startup logging.warning.
      * `workspace_root_override` (per-profile, from the profile dict's
        `workspace_root` key) replaces the primary root when provided —
        profiles may widen or narrow the global workspace.
      * `project_workspace_dir` (active project's workspace_dir) sits between
        those two in precedence: profile > project > config > cwd. When no
        project is active behavior is exactly as before.
    The standing allow-list is always appended alongside the primary root:
    SABLE_HOME (todo/trace state dir, so todo_write keeps working) and the
    system temp dir (so tmp_path-based flows keep working).
    """
    global _UNRESTRICTED_WARNED
    agent_cfg = getattr(cfg, "agent", None) or {}
    if workspace_root_override:
        raw: Any = workspace_root_override
    elif project_workspace_dir:
        raw = project_workspace_dir
    elif "workspace_root" in agent_cfg:
        raw = agent_cfg["workspace_root"]  # UNSET sentinel == explicit null
    else:
        raw = os.getcwd()
    if raw is None or raw is UNSET:
        if not _UNRESTRICTED_WARNED:
            _UNRESTRICTED_WARNED = True
            logger.warning(
                "agent.workspace_root is null: workspace containment is "
                "DISABLED — the agent may read/write anywhere it can. "
                "Set agent.workspace_root to a directory to enable it.")
        return None
    from ..store.paths import resolve_sable_home

    primary = Path(str(raw)).expanduser()
    roots = [primary, resolve_sable_home(cfg),
             Path(tempfile.gettempdir())]
    # de-dupe while preserving order (resolved compare inside contain)
    seen: List[Path] = []
    for r in roots:
        dup = False
        for s in seen:
            try:
                if r.samefile(s):
                    dup = True
                    break
            except OSError:
                if r == s:
                    dup = True
                    break
        if not dup:
            seen.append(r)
    return seen


def _build_delegation_ctx(
    cfg: SableConfig, base_url: str, model: str,
    api_key: Optional[str], system_prompt: Optional[str],
    enabled_tool_names: Optional[List[str]],
    project_workspace_dir: Optional[str],
    delegation_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge caller-supplied delegation context over the defaults derived
    from create_runner's own arguments (Phase 2). Callers that pass an
    explicit dict win field-by-field; `cfg` is always injected so tools can
    read profiles/projects/delegation config."""
    ctx: Dict[str, Any] = {
        "cfg": cfg, "base_url": base_url, "model": model,
        "profile": None, "project": None, "thread": None,
        "run_id": None, "depth": 0,
    }
    if project_workspace_dir:
        ctx["project_workspace_dir"] = project_workspace_dir
    if delegation_context:
        ctx.update(delegation_context)
    ctx.setdefault("cfg", cfg)
    ctx.setdefault("base_url", base_url)
    ctx.setdefault("model", model)
    return ctx


def _with_playbooks(cfg: SableConfig, system_prompt: Optional[str],
                    profile_name: Optional[str]) -> Optional[str]:
    """Append the capped '## Playbooks' section to a run's system prompt.

    Pass 4 #38. Disabled (the default) => returns ``system_prompt``
    untouched and performs ZERO store access, so prompts stay byte-identical
    to pre-#38 behavior. Enabled with no active playbooks (or any store/
    render failure) likewise returns the prompt unchanged — injection can
    never break or silently alter a run beyond appending its section.
    """
    from .playbooks import playbooks_config, prompt_section

    if not playbooks_config(cfg).get("enabled", False):
        return system_prompt
    base = system_prompt or SYSTEM_PROMPT
    try:
        section = prompt_section(cfg, profile_name)
    except Exception as e:  # defensive: never fail a run over playbooks
        logger.warning("playbook injection skipped: %s", e)
        return system_prompt
    if not section:
        return system_prompt
    return base.rstrip("\n") + "\n\n" + section


#: Hard cap on the injected "## Project" section — description and roles
#: are free user text; the block must never blow up the system prompt.
MAX_PROJECT_CONTEXT_CHARS = 4000


def _with_project_context(project: Optional[Dict[str, Any]],
                          system_prompt: Optional[str],
                          profile_name: Optional[str]) -> Optional[str]:
    """Append a '## Project' section: the project description plus the
    roster of agents working on the project and each one's job, so every
    agent knows its own role and its teammates' (multi-agent awareness).

    ``project`` is the active project record (name, description, agents,
    agent_roles); None/empty — or a record with neither description nor
    agents — returns ``system_prompt`` untouched (byte-identical prompt,
    the no-project default path). ``profile_name`` marks the roster line
    that belongs to the running agent ("that's you")."""
    if not isinstance(project, dict):
        return system_prompt
    name = str(project.get("name") or "").strip()
    desc = str(project.get("description") or "").strip()
    agents_raw = project.get("agents")
    agents = ([str(a).strip() for a in agents_raw if str(a).strip()]
              if isinstance(agents_raw, list) else [])
    roles = project.get("agent_roles")
    roles = roles if isinstance(roles, dict) else {}
    if not (desc or agents):
        return system_prompt
    lines = [f"## Project: {name}" if name else "## Project"]
    if desc:
        lines += ["", desc]
    if agents:
        lines += ["", "Agents working on this project:"]
        for a in agents:
            role = str(roles.get(a) or "").strip()
            you = " (that's you)" if profile_name and a == profile_name else ""
            lines.append(f"- {a}{you}" + (f": {role}" if role else ""))
    section = "\n".join(lines)
    if len(section) > MAX_PROJECT_CONTEXT_CHARS:
        section = section[:MAX_PROJECT_CONTEXT_CHARS].rstrip() + "\u2026"
    base = system_prompt or SYSTEM_PROMPT
    return base.rstrip("\n") + "\n\n" + section


def _with_skills_index(cfg: SableConfig,
                       system_prompt: Optional[str]) -> Optional[str]:
    """Append the capped '## Skills' index to a run's system prompt.

    Mirror of _with_playbooks for agent-authored skills (core/skills.py):
    disabled (the default) => zero filesystem access and a byte-identical
    prompt; any render failure likewise leaves the prompt untouched.
    """
    from .skills import index_section, skills_config

    if not skills_config(cfg).get("enabled", False):
        return system_prompt
    try:
        section = index_section(cfg)
    except Exception as e:  # defensive: never fail a run over skills
        logger.warning("skills index skipped: %s", e)
        return system_prompt
    if not section:
        return system_prompt
    base = system_prompt or SYSTEM_PROMPT
    return base.rstrip("\n") + "\n\n" + section


def _with_plan(cfg: SableConfig, system_prompt: Optional[str],
               project_id: Optional[str]) -> Optional[str]:
    """Append the capped '## Project plan' section (WP12, core/plans.py).

    Same never-fail bounded-injection discipline as _with_playbooks: with
    plans.enabled off (the default) this performs ZERO store access and the
    prompt is byte-identical. Enabled but inject_summary=false, an empty/
    missing plan, or any store/render failure likewise return the prompt
    unchanged. The section is hard-capped by plans.max_inject_chars.
    """
    from .plans import load_plan, plans_config, summary

    pcfg = plans_config(cfg)
    if not pcfg.get("enabled", False):
        return system_prompt
    if not pcfg.get("inject_summary", True):
        return system_prompt
    try:
        store = load_plan(project_id)
        if not store.get("tasks"):
            return system_prompt  # empty plan => byte-identical prompt
        max_chars = int(pcfg.get("max_inject_chars", 800))
        section = summary(store, max_chars=max(120, max_chars))
    except Exception as e:  # defensive: never fail a run over plans
        logger.warning("plan injection skipped: %s", e)
        return system_prompt
    if not section:
        return system_prompt
    base = system_prompt or SYSTEM_PROMPT
    return base.rstrip("\n") + "\n\n" + section


def create_runner(cfg: SableConfig, base_url: str, model: str,
                  api_key: Optional[str] = None, recorder: Any = None,
                  system_prompt: Optional[str] = None,
                  enabled_tool_names: Optional[List[str]] = None,
                  workspace_root_override: Optional[str] = None,
                  project_workspace_dir: Optional[str] = None,
                  delegation_context: Optional[Dict[str, Any]] = None,
                  project_id: Optional[str] = None,
                  project_context: Optional[Dict[str, Any]] = None,
                  sampling_override: Optional[Dict[str, Any]] = None,
                  capabilities: Optional[Dict[str, Any]] = None,
                  profile_name: Optional[str] = None,
                  cancel_event: Optional[Any] = None,
                  plan_store: Optional[Dict[str, Any]] = None,
                  extra_headers: Optional[Dict[str, str]] = None,
                  ) -> SableAgentClient:
    """Build the built-in agent runtime.

    `recorder` (Phase 2) receives every completed run; None => no recording.
    Callers should build one via store.make_recorder(cfg).

    `system_prompt` / `enabled_tool_names` (Phase 5e agent profiles): the
    profile's prompt replaces the default system message and its tool list
    restricts the schemas exposed to the model.

    `project_workspace_dir` (projects): active project's workspace directory;
    used as the containment root when no profile-level override exists (see
    resolve_workspace_roots).

    `delegation_context` (Phase 2 delegation): run metadata for the
    list_agents/delegate_task/check_delegation tools — {cfg, base_url, model,
    profile, project, thread, run_id, depth, on_update}. When omitted, a
    minimal context is built from the arguments here so CLI runs get working
    (thread-less) delegation; pass an explicit dict to override any field.

    `project_id` (Phase 5): active project id; bound into search_codebase so
    semantic search routes to that project's index DB (no cross-project
    results). None => the default/no-project DB.

    `project_context`: the active project record (name, description, agents,
    agent_roles). When it carries a description or an agent roster, a
    '## Project' section is appended to the system prompt so every agent on
    the project knows what it is about, what its own job is, and which other
    agents (and their jobs) share it. None/empty (the no-project default)
    leaves the prompt byte-identical.

    `sampling_override` (Phase 6 multi-instance serving): per-server-instance
    sampling parameters merged ON TOP of the global `sampling:` section — a
    profile bound to an instance gets that instance's sampling defaults.

    `capabilities` (per-agent reach-out): {internet: bool, run_commands:
    bool} from the calling profile. Merged over the global
    `agent.capabilities:` defaults via tools.resolve_capabilities; gates the
    web_fetch / run_command tools (both schema and impl) per runner. All off
    by default — a local-first runtime until a human opts an agent in.

    `profile_name` (pass 4 #38): name of the calling agent profile, when one
    is in use. With `playbooks.enabled` on it selects which distilled recipes
    get appended to the effective system prompt (profile rows + global rows);
    with the knob off (default) this argument changes nothing and the prompt
    is byte-identical to pre-#38 behavior — no playbook store access at all.

    `extra_headers` (remote peer agents): additional HTTP headers merged into
    every request the runner makes (e.g. ``X-Sable-Instance`` so a multi-
    instance peer gateway routes completions to the snapshot's named
    instance). None/empty keeps every existing call byte-identical.

    Context compression is on by default at 75% of the configured context
    window (`agent.context_token_budget` overrides; 0 disables) — long REPL
    threads get their older turns summarized instead of overflowing llama's
    -c limit.
    """
    agent_cfg = getattr(cfg, "agent", None) or {}
    tools_cfg = getattr(cfg, "tools", None) or {}
    from ..store.paths import resolve_sable_home
    from ..core.tools import resolve_capabilities

    resolved_caps = resolve_capabilities(agent_cfg, capabilities)
    workspace_roots = resolve_workspace_roots(
        cfg, workspace_root_override, project_workspace_dir)
    # Config-takeover carve-out (see SableAgentClient): the live YAML stays
    # readable through the SABLE_HOME allow-list but unwritable by tools.
    from ..core.config import resolve_config_path as _resolve_cfg_path

    deny_writes: List[Path] = [resolve_sable_home(cfg) / "sable.yaml"]
    _explicit_cfg = _resolve_cfg_path()
    if _explicit_cfg is not None:
        deny_writes.append(_explicit_cfg)
    # The experience buffer (trace DB + JSONL) and the codebase indexes are
    # readable through the SABLE_HOME allow-list (search_traces /
    # search_codebase read them via their own stores) but must never be
    # tampered with by file tools: rewriting sable.db would poison every
    # downstream consumer (curation, insights, playbooks).
    _home = resolve_sable_home(cfg)
    deny_writes.extend([_home / "buffer", _home / "index"])
    budget = int(agent_cfg.get(
        "context_token_budget", int(cfg.context_length * 0.75)) or 0)
    # Pass 4 #38: distilled playbooks appended under a hard cap. The knob
    # (playbooks.enabled, default false) short-circuits BEFORE any store
    # access, so the disabled path is byte-identical to pre-#38 prompts.
    system_prompt = _with_playbooks(cfg, system_prompt, profile_name)
    # Persistent skills (core/skills.py): same disabled-by-default discipline
    # — knob off means no filesystem access and an unchanged prompt.
    skills_enabled = bool((getattr(cfg, "skills", None) or {})
                          .get("enabled", False))
    system_prompt = _with_skills_index(cfg, system_prompt)
    # Projects: active project's description + agent roster (description +
    # agents), then the WP12 plan summary appended last — closest to the
    # conversation. No project / empty plan / knob off => prompt untouched.
    system_prompt = _with_project_context(project_context, system_prompt,
                                          profile_name)
    from .plans import plans_config as _plans_config

    plans_enabled = bool(_plans_config(cfg).get("enabled", False))
    system_prompt = _with_plan(cfg, system_prompt, project_id)
    # WP12: private plan context bound into the plan tools via partial (the
    # model cannot forge it — schema gate). `plan_store` (accepted for API
    # symmetry with the WP12 spec) may carry {"thread": ...}; otherwise the
    # delegation context's thread rides along.
    _plan_ctx: Dict[str, Any] = {
        "project": project_id,
        "thread": (plan_store or {}).get("thread")
        or (delegation_context or {}).get("thread"),
        # Shared mutable list: plan_progress appends touched task ids during
        # the run; _finish fills the trace id under them at finalize.
        "_touched": [],
    }
    return SableAgentClient(base_url=base_url, model=model, api_key=api_key,
                              extra_headers=extra_headers,
                              recorder=recorder, system_prompt=system_prompt,
                              enabled_tool_names=enabled_tool_names,
                              skills_enabled=skills_enabled,
                              compress_threshold_tokens=budget,
                              keep_recent_tokens=int(agent_cfg.get(
                                  "keep_recent_tokens", 2048)),
                              guardrails_enabled=bool(agent_cfg.get(
                                  "guardrails", True)),
                               # Tool-calling turns are UNCAPPED by default:
                               # capped defaults (12, then 40) kept cutting real
                               # file-reading sessions off mid-task. The loop
                               # is otherwise bounded by context compression,
                               # user cancel, guardrail halts, and backend
                               # errors. Set agent.max_iterations to a positive
                               # int to restore a cap (0 or absent = uncapped).
                               max_iterations=(
                                   int(agent_cfg["max_iterations"])
                                   if int(agent_cfg.get("max_iterations") or 0) > 0
                                   else None),
                               block_sensitive_files=bool(tools_cfg.get(
                                   "block_sensitive_files", True)),
                              # Feature-wave Phase 4: post-write editor-style
                              # checks (tools.lint_on_write + tools.lsp.*).
                              lint_on_write=bool(tools_cfg.get(
                                  "lint_on_write", True)),
                              lsp_config=dict(tools_cfg.get("lsp") or {}),
                              workspace_roots=workspace_roots,
                              deny_write_paths=deny_writes,
                              # Backlog #10: spill knobs (tools.spill_*).
                              spill_enabled=bool(tools_cfg.get(
                                  "spill_enabled", True)),
                              spill_threshold_chars=int(tools_cfg.get(
                                  "spill_threshold_chars",
                                  DEFAULT_SPILL_THRESHOLD_CHARS)),
                              # None/unset => char gate (today's behavior);
                              # when set the token gate wins over chars.
                              spill_threshold_tokens=(int(_sp_tok) if (
                                  _sp_tok := tools_cfg.get(
                                      "spill_threshold_tokens")) else None),
                              turn_result_budget_chars=int(tools_cfg.get(
                                  "turn_result_budget_chars",
                                  DEFAULT_TURN_RESULT_BUDGET_CHARS)),
                              spill_retention_days=int(tools_cfg.get(
                                  "spill_retention_days",
                                  DEFAULT_SPILL_RETENTION_DAYS)),
                              spill_home=resolve_sable_home(cfg),
                              # Sampling defaults + tool-call healing toggle.
                              # Phase 6: per-instance overrides win over global.
                              # Pass 4 #40: `logprobs` is a REQUEST-shape knob,
                              # not an upstream sampling parameter, so it is
                              # popped out of the merged map here (it must never
                              # reach request_sampling_payload verbatim) and
                              # passed as its own flag. Default false.
                              sampling=_sampling_without_logprobs(
                                  cfg, sampling_override),
                              logprobs=bool(
                                  {**(getattr(cfg, "sampling", None) or {}),
                                   **(sampling_override or {})}
                                  .get("logprobs", False)),
                              heal_tool_calls=bool(agent_cfg.get(
                                  "heal_tool_calls", True)),
                              delegation_context=_build_delegation_ctx(
                                  cfg, base_url, model, api_key,
                                  system_prompt, enabled_tool_names,
                                  project_workspace_dir, delegation_context),
                              # Phase 5: per-project codebase index routing.
                              search_project=({"id": project_id,
                                               "workspace_dir":
                                                   project_workspace_dir}
                                              if project_id else None),
                              # Active config for search_codebase (its
                              # indexing: section + enabled flag).
                              search_cfg=cfg,
                              # Per-agent capabilities (internet / run_command)
                              # + web_fetch knobs from tools.web_fetch.*.
                              capabilities=resolved_caps,
                              web_fetch_config=dict(
                                  tools_cfg.get("web_fetch") or {}),
                              # Hardening: tools.run_command_argv_only runs the
                              # capability without a shell (see core/tools).
                              run_command_argv_only=bool(tools_cfg.get(
                                  "run_command_argv_only", False)),
                              # Vision input: agent.vision ("auto"|true|false)
                              # wins; auto also treats a configured mmproj as
                              # a strong hint the served model sees images.
                              # Pass 3 #19: verification-stop nudge on by
                              # default (agent.verify_on_stop: false opts out).
                              verification_nudge=bool(agent_cfg.get(
                                  "verify_on_stop", True)),
                              vision=_resolve_vision_mode(agent_cfg, cfg),
                              # Improvement #4: exact token counting for the
                              # compression gate is lazily constructed inside
                              # the client against base_url (agent.tokenizer_
                              # exact config knob intentionally not added —
                              # graceful fallback makes it always-on-safe).
                              cancel_event=cancel_event,
                              # WP12: plan tool family + private ctx binding
                              # (stripped entirely when plans.enabled is off).
                              plan_ctx=_plan_ctx,
                              plans_enabled=plans_enabled)

def _sampling_without_logprobs(cfg: SableConfig,
                               sampling_override: Optional[Dict[str, Any]]
                               ) -> Dict[str, Any]:
    """Merged `sampling:` map with the pass-4 #40 `logprobs` gate removed.

    The remaining keys are forwarded verbatim into every /chat/completions
    request (request_sampling_payload); `logprobs` is handled by _chat itself,
    so leaving it in would send it twice and defeat the false case."""
    merged = {**(getattr(cfg, "sampling", None) or {}),
              **(sampling_override or {})}
    merged.pop("logprobs", None)
    return merged


def _resolve_vision_mode(agent_cfg: Dict[str, Any],
                         cfg: SableConfig) -> str:
    """Vision mode for create_runner: explicit agent.vision (bool or string),
    else "auto" — with run_settings.mmproj_path set, the endpoint is known to
    carry a projector, so report "true" without probing."""
    raw = agent_cfg.get("vision")
    if isinstance(raw, bool):
        return "true" if raw else "false"
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    rs = getattr(cfg, "run_settings", None) or {}
    if str(rs.get("mmproj_path") or "").strip():
        return "true"
    return "auto"

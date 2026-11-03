"""Validation gate — golden-task replay before model promotion (Phase 4).

Serves a candidate GGUF on a temporary loopback port via the Serving
Supervisor, replays a tiny deterministic task suite (config
`validation.golden_tasks:`), scores the pass rate and compares against
`validation.min_pass_rate`. Promotion happens only when the gate passes;
otherwise the model is marked failed and the caller rolls back.

Check semantics are intentionally dumb and user-editable. Two check kinds:

    {prompt: "...", expect_contains: ["a", "b"], expect_not_contains?: [...]}
    {prompt: "...", expect_tool_call: {name: read_file, arg_contains: {path: "x"}}}

A substring check passes when every expected substring appears
(case-insensitive) in the model's reply and no forbidden substring appears.
An ``expect_tool_call`` check (backlog item #5) sends the prompt *with* the
production ``TOOL_SCHEMAS`` and asserts a tool call comes back — either the
native ``message.tool_calls`` field or text-emitted calls recovered via
``core.tool_calls.parse_text_tool_calls`` (the same oracle the agent runner
uses). This catches tool-call *grammar* regressions (quantization/merge bugs)
that prose answers would otherwise slip past. Tool-call checks are opt-in per
task; substring-only tasks behave exactly as before, and a task may mix both
kinds. No LLM judge — the point is regression tripwires, not quality scoring.

``server_starter`` is injectable so tests can hand it a scripted fake server
instead of llama-server + GGUF (no GPU needed).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import time

import httpx

from ..core.errors import classify_exception, classify_http_status
from ..core.tool_calls import _as_args, parse_text_tool_calls
from ..core.tools import TOOL_SCHEMAS

logger = logging.getLogger("toaster_loop.validate")

# (gguf_path) -> context manager yielding base_url like http://127.0.0.1:PORT/v1
ServerStarter = Callable[[str], Any]


@dataclass
class ValidationConfig:
    """`validation:` YAML section."""

    min_pass_rate: float = 1.0
    golden_tasks: List[Dict[str, Any]] = field(default_factory=list)
    request_timeout_s: float = 120.0
    max_tokens: int = 256
    temperature: float = 0.0  # deterministic tripwires
    # Deep-review housekeeping — bounded-concurrency golden replay. llama.cpp
    # serves one slot by default (`-np 1`): requests queue server-side, so the
    # win from client parallelism is only pipelining of client-side waits
    # (connection setup, response transfer, per-task retry backoff) — real but
    # bounded. Conservative default 2 captures that overlap without adding
    # queue pressure on the single slot; raise it only with `-np > 1` servers.
    parallel: int = 2
    # Pass 4 #32 — shadow-baseline evaluation (evaluation/shadow.py). Opt-in
    # master switch (pass-2 #13 knob pattern): nothing about shadow eval runs
    # unless shadow_enabled is true, and even then the verdict is warn-only
    # unless shadow_blocks_promotion flips it into a hard gate.
    shadow_enabled: bool = False
    # Regression tolerance as a PERCENTAGE of the judge's 0..1 score range:
    # improvement < -(pct/100 * 1.0) flags is_regression. Default 5%.
    max_shadow_regression_pct: float = 5.0
    shadow_blocks_promotion: bool = False
    shadow_replay_n: int = 8
    shadow_judge_model: Optional[str] = None
    # Pass 4 #41 — capability tiers (registry/tiers.py). Minimum derived tier
    # required for UNATTENDED automatic promotion (train auto-check --trigger
    # path). Default 2 ("shadow-proven"); set 0 to disable the policy. Manual
    # iterate/promote are never gated by it.
    min_tier_auto_deploy: int = 2

    @staticmethod
    def from_yaml(data: Dict[str, Any]) -> "ValidationConfig":
        sec = data.get("validation") if isinstance(data, dict) else None
        if not isinstance(sec, dict):
            return ValidationConfig()
        known = {f for f in ValidationConfig.__dataclass_fields__}
        unknown = sorted(k for k in sec if k not in known)
        if unknown:  # typo'd knobs used to vanish silently (same as training:)
            logger.warning("validation config: ignoring unknown key(s) %s "
                           "(known: %s)", ", ".join(unknown),
                           ", ".join(sorted(known)))
        kwargs = {k: v for k, v in sec.items() if k in known and v is not None}
        return ValidationConfig(**kwargs)


def default_supervisor_starter_factory(model_name: Optional[str] = None):
    """Real starter: spawn llama-server for the GGUF via ServingSupervisor."""

    def starter(gguf_path: str):
        from ..core.config import ToasterLoopConfig
        from ..serving.supervisor import ServingSupervisor

        cfg = ToasterLoopConfig(gguf_path=str(gguf_path), port=0)
        if model_name:
            cfg.model_name = model_name
        sup = ServingSupervisor(cfg)
        sup.start()
        return sup  # caller stops it; also usable as context manager

    return starter


def _extract_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Native ``tool_calls`` if present, else text-recovered calls (backlog #5).

    Mirrors core/agent_runner.py exactly: trust the native field first, fall
    back to core.tool_calls.parse_text_tool_calls for models that emit calls
    as fenced-JSON/XML text. Reusing the runner's own parsers keeps the
    validation oracle identical to production recovery.
    """
    calls = message.get("tool_calls") or []
    if not calls and message.get("content"):
        calls = parse_text_tool_calls(message["content"])
    return [c for c in calls if isinstance(c, dict)]


def _spec_label(name: Optional[str], arg_contains: Dict[str, Any]) -> str:
    """Human-readable expect_tool_call spec for failure messages."""
    parts = []
    for k, v in (arg_contains or {}).items():
        parts.append(f"{k}~{v!r}")
    inner = f"({', '.join(parts)})" if parts else ""
    return f"tool call {name or 'any'}{inner}"


def _check_tool_call(spec: Dict[str, Any],
                     calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate one expect_tool_call check against recovered tool calls.

    Never raises: malformed argument JSON fails the check (via _as_args ->
    None), it does not crash the suite.
    """
    spec = spec if isinstance(spec, dict) else {}
    name = spec.get("name")
    arg_contains = spec.get("arg_contains") or {}
    label = _spec_label(name, arg_contains)
    result = {"kind": "tool_call", "expect": label, "pass": False}

    if not calls:
        result["reason"] = f"expected {_spec_label(name, arg_contains)}, got prose only"
        return result

    # (name, args|None) per call; args None => unparseable arguments JSON,
    # which fails the check rather than crashing the suite.
    parsed: List[tuple] = []
    for c in calls:
        fn = c.get("function")
        if not isinstance(fn, dict):
            fn = {}
        cname = fn.get("name") if isinstance(fn.get("name"), str) else None
        parsed.append((cname, _as_args(fn.get("arguments"))))

    result["got"] = [n for n, _ in parsed]
    matched = parsed if not name else [(n, a) for n, a in parsed if n == name]
    if not matched:
        result["reason"] = (f"expected tool call {name!r}, got "
                            f"{result['got'] or 'prose only'}")
        return result

    if not arg_contains:
        result["pass"] = True
        return result

    # At least one name-matching call must satisfy every listed arg key.
    first_problem = None
    for _, args in matched:
        if args is None:
            first_problem = first_problem or "malformed arguments JSON"
            continue
        problem = next(
            (f"{k} missing" if k not in args else
             f"{k}={str(args[k])!r} !~ {str(v)!r}"
             for k, v in arg_contains.items()
             if k not in args
             or str(v).lower() not in str(args[k]).lower()),
            None,
        )
        if problem is None:
            result["pass"] = True
            return result
        first_problem = first_problem or problem
    result["reason"] = (f"expected {_spec_label(name, arg_contains)}, "
                        f"got bad args: {first_problem}")
    return result


def _check_task(message: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    """Score one golden task reply. Report is backward compatible: the legacy
    top-level keys (missing/forbidden_hit/reply_preview) stay; a per-check
    ``checks`` list adds the kind field ("substring" | "tool_call") so
    dashboards can tell which tripwire failed (backlog item #5).
    """
    reply = message.get("content") or ""
    low = reply.lower()
    missing = [s for s in (task.get("expect_contains") or []) if str(s).lower() not in low]
    forbidden = [s for s in (task.get("expect_not_contains") or []) if str(s).lower() in low]

    checks: List[Dict[str, Any]] = []
    has_substring_spec = bool(task.get("expect_contains") or task.get("expect_not_contains"))
    tool_spec = task.get("expect_tool_call")

    if has_substring_spec or not tool_spec:
        # Legacy behavior preserved: tasks without any tool-call spec always
        # carry the substring check (trivially passing when specs are empty).
        checks.append({
            "kind": "substring",
            "pass": not missing and not forbidden,
            "missing": missing,
            "forbidden_hit": forbidden,
            **({} if not missing and not forbidden else {
                "reason": (f"substring check failed: missing={missing} "
                           f"forbidden_hit={forbidden}"),
            }),
        })
    if tool_spec is not None:
        calls = _extract_tool_calls(message)
        checks.append(_check_tool_call(tool_spec, calls))

    ok = all(c.get("pass") for c in checks)
    entry: Dict[str, Any] = {
        "prompt": task.get("prompt"),
        "pass": ok,
        "missing": missing,
        "forbidden_hit": forbidden,
        "checks": checks,
        "reply_preview": reply[:200],
    }
    if not ok:
        entry["error"] = "; ".join(
            f"{c['kind']}: {c.get('reason', 'failed')}" for c in checks
            if not c.get("pass")
        ) or None
    return entry


#: injectable sleep for tests (module default time.sleep)
_sleep = time.sleep


def _needs_tools(task: Dict[str, Any]) -> bool:
    """True when the task carries an expect_tool_call check (backlog item #5):
    only those requests ship tools=TOOL_SCHEMAS; substring-only tasks are
    byte-identical to pre-#5 traffic."""
    return task.get("expect_tool_call") is not None


def _post_task(base_url: str, model: str, task: Dict[str, Any], *,
               timeout: float, max_tokens: int, temperature: float,
               headers: Dict[str, str], transport_attempts: int = 3,
               backoff_s: tuple = (1.0, 2.0),
               system_prompt: Optional[str] = None):
    """POST one golden task with classified transport retry (backlog #2).

    Returns (assistant_message|None, error_str|None) where the message is the
    full choice message dict ({content, tool_calls?, ...}) so callers can run
    both substring and tool-call checks. Tasks with an expect_tool_call check
    are sent with ``tools=TOOL_SCHEMAS`` — the same schemas the agent uses in
    production (backlog item #5). A transient blip (connection reset / timeout
    / 5xx from a restarting server) gets up to ``transport_attempts`` total
    tries with capped exponential backoff, so one hiccup doesn't mark the
    whole model failed. Deterministic errors (400/404 or content checks) fail
    fast.

    ``system_prompt`` (pass 4 #39) prepends a system turn — prompt-evolution
    replays goldens WITH the candidate system prompt in place. Omitted =>
    byte-identical to pre-#39 traffic for validation runs.
    """
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": task.get("prompt", "")})
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if _needs_tools(task):
        body["tools"] = TOOL_SCHEMAS
    last_err = "no attempt made"
    for attempt in range(max(1, transport_attempts)):
        if attempt:
            _sleep(backoff_s[min(attempt - 1, len(backoff_s) - 1)])
        try:
            r = httpx.post(
                base_url.rstrip("/") + "/chat/completions",
                json=body,
                timeout=timeout,
                headers=headers,
            )
        except Exception as e:  # transport-level: classify for retryability
            ce = classify_exception(e)
            last_err = f"{type(e).__name__}: {e}"
            if ce.retryable and attempt + 1 < transport_attempts:
                logger.warning("golden task transport blip (%s), retrying: %s",
                               ce.reason, e)
                continue
            return None, last_err
        if r.status_code == 200:
            data = r.json()
            message = ((data.get("choices") or [{}])[0].get("message")) or {}
            return message, None
        ce = classify_http_status(r.status_code, r.text[:500])
        last_err = f"HTTP {r.status_code}"
        if ce.retryable and attempt + 1 < transport_attempts:
            logger.warning("golden task HTTP %d (%s), retrying",
                           r.status_code, ce.reason)
            continue
        return None, last_err
    return None, last_err


def run_golden_suite(
    base_url: str,
    model: str,
    tasks: List[Dict[str, Any]],
    *,
    timeout: float = 120.0,
    max_tokens: int = 256,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    parallel: int = 1,
) -> Dict[str, Any]:
    """POST each golden task to {base_url}/chat/completions; return results.

    ``parallel`` caps concurrent in-flight requests (ThreadPoolExecutor).
    Thread-safety: replay is pure client I/O — tasks carry no per-task tool
    execution (expect_tool_call only *parses* replies), and _post_task /
    _check_task hold no shared mutable state; results are written into a
    pre-sized list by index so persisted order matches task order regardless
    of completion order. Per-task timeout/retry semantics are unchanged.
    ``parallel=1`` keeps the original serial path byte-identical.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _run_one(idx: int, t: Dict[str, Any]) -> Dict[str, Any]:
        entry = {"prompt": t.get("prompt")}
        message, err = _post_task(base_url, model, t, timeout=timeout,
                                  max_tokens=max_tokens, temperature=temperature,
                                  headers=headers)
        if err is not None:
            entry.update({"pass": False, "error": err})
        else:
            entry.update(_check_task(message or {}, t))
        return entry

    total = len(tasks)
    results: List[Dict[str, Any]] = [None] * total  # type: ignore[list-item]
    workers = max(1, min(int(parallel), total))
    if workers <= 1:
        for i, t in enumerate(tasks):
            results[i] = _run_one(i, t)
    else:
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="golden") as pool:
            futures = [pool.submit(_run_one, i, t)
                       for i, t in enumerate(tasks)]
            for i, fut in enumerate(futures):
                results[i] = fut.result()
    passed = sum(1 for r in results if r.get("pass"))
    rate = (passed / total) if total else 0.0
    return {"total": total, "passed": passed, "pass_rate": rate, "results": results}


def validate_model(
    gguf_path: str,
    vcfg: ValidationConfig,
    *,
    server_starter: Optional[ServerStarter] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Serve ``gguf_path`` and replay golden tasks. Returns a report dict.

    Report: {ok, pass_rate, min_pass_rate, total, passed, results, error?,
             started_at}. Never raises for serving failures — they become
    ok=False with an error string so the registry stores them.
    """
    report: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_pass_rate": vcfg.min_pass_rate,
        "gguf_path": str(gguf_path),
    }
    if not vcfg.golden_tasks:
        report.update({
            "ok": False, "error":
                "validation.golden_tasks is empty — refusing to promote blindly "
                "(add at least one golden task to config)",
            "total": 0, "passed": 0, "pass_rate": 0.0, "results": [],
        })
        return report

    starter = server_starter or default_supervisor_starter_factory(model_name)
    handle = None
    try:
        handle = starter(gguf_path)
        # Starter returns either a ServerHandle-like (.base_url/.model_name)
        # or a ServingSupervisor (use .handle).
        srv = getattr(handle, "handle", None) or handle
        base_url = getattr(srv, "base_url", None)
        model = model_name or getattr(srv, "model_name", None) or "candidate"
        if not base_url:
            raise RuntimeError("server starter returned no base_url")
        suite = run_golden_suite(
            base_url, model, vcfg.golden_tasks,
            timeout=vcfg.request_timeout_s,
            max_tokens=vcfg.max_tokens,
            temperature=vcfg.temperature,
            parallel=vcfg.parallel,
        )
        report.update(suite)
        report["ok"] = suite["pass_rate"] >= vcfg.min_pass_rate
    except Exception as e:
        report.update({
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "total": 0, "passed": 0, "pass_rate": 0.0, "results": [],
        })
    finally:
        stop = getattr(handle, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                pass
    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return report

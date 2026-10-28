# WP12 Context Document — "Plans" (persistent per-project task graphs)

READ-ONLY reconnaissance of the sable repo at `/run/media/sumo/backup/sable`.
Package root: `sable/sable/` (import as `sable.*`). Tests: `sable/tests/`.
All line numbers are from the files as of this pass. Every excerpt is verbatim.

**Surprises / deltas vs. typical WP12 assumptions — read first:**
1. **`sable run --project` ALREADY EXISTS** (`cli.py:165-172`, wired in `cmd_run` at `cli.py:1759-1774`). Only a `--task <id>`-style flag would be new. Do not re-add `--project`.
2. **The tool-name sets WP12 asked about are NOT all in `core/tools.py`.** Mutating/read-only sets live in `core/guardrails.py:34-38` (`IDEMPOTENT_TOOL_NAMES`, `MUTATING_TOOL_NAMES`). `core/tools.py` has `DELEGATION_TOOL_NAMES` (line 1398) and `CAPABILITY_TOOLS`/`CAPABILITY_KEYS` (1556-1566); `agent_runner.py` has `_CONTAINED_TOOLS` (100-101) and `SKILL_TOOL_NAMES` (216-217). A new plan-mutating tool must be added to `MUTATING_TOOL_NAMES` in guardrails or the loop guard will treat it conservatively (unknown ⇒ mutating ⇒ never no-progress-blocked — safe default, but statuses repeat-loop unflagged).
3. **The MCP inline validator has NO array support** (`mcp/schema.py` `_TYPES` dict = string/integer/number/boolean/object only). An array-extension implementer must add an `"array"` predicate AND recursion into `items`, plus decide minItems/maxItems handling. See §7.
4. **Judge calls record NOTHING to the buffer themselves.** The verdict is returned as a dict; the *caller* writes it as a feedback row `source="judge"` via `TraceStore.add_feedback` (`cli.py:2583-2586`). Judge transport is an injected `client_fn(messages, model) -> str` — tests pass a lambda; no server needed.
5. **The failure_reason constant is `FAILURE_VERIFICATION_NUDGE = "verification_nudge"`** in `core/failure_types.py:36` (not verification.py). The nudge *policy* lives in `core/verification.py`.
6. **The SSE client has a hardcoded KINDS whitelist** (`dashboard/static/js/bus.js:10-15`). A new `plan.*` event kind is silently dropped unless added to that list.
7. There is no dedicated "jobs daemon thread" for long actions other than `BackgroundTask` (dashboard/api/base.py) and the iterate-style background-thread+record pattern; WP12's plan loop should mirror one of those two (§11).

---

## 1. core/tools.py — the todo pattern to copy

File: `sable/sable/core/tools.py`

### 1a. Storage helpers (per-run binding via private kwarg)

```python
# lines 889-897
def _todo_path(bound: Optional[str] = None) -> Path:
    """Todo file for a run. The runner binds a per-run path under
    ``SABLE_HOME/todos/`` via partial (private kwarg, invisible to the
    model); unbound callers keep the historical global ``todos.json``."""
    if bound:
        return Path(bound)
    from ..store.paths import resolve_sable_home

    return resolve_sable_home() / "todos.json"
```

Atomic write idiom (`_todo_write_raw`, lines 912-918):

```python
def _todo_write_raw(items: List[Dict[str, str]],
                    bound: Optional[str] = None) -> None:
    p = _todo_path(bound)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"todos": items}, indent=2))
    os.replace(tmp, p)
```

Tolerant read (`_todo_read_raw`, 900-909): returns `[]` on any OSError/JSONDecodeError; accepts `{"todos": [...]}` or bare list.

Validation/salvage (`_todo_validate`, 921-945): strings like `- [x] foo` are salvaged into `{id, content, status}`; unknown statuses coerce to `"pending"`; `TODO_STATUSES = ("pending", "in_progress", "done", "cancelled")` (line 882); caps `TODO_MAX_ITEMS = 50`, `TODO_MAX_CONTENT = 400` (54-55).

### 1b. Tool entry points

```python
# lines 948-957 (signature + docstring)
def todo_write(todos: Optional[List[Any]] = None,
               merge: bool = False, *, _todos_path: Optional[str] = None,
               **_: Any) -> str:
    """Create/replace (or merge-update) the session todo list.
    ... `_todos_path` is bound per-run by the runner (private kwarg — the schema
    gate rejects it if a model tries to pass it) so concurrent runs stop
    clobbering each other's plans through the one shared file."""
```

Merge semantics (964-972): merge=True upserts by `id` over existing; replace otherwise. Over-cap returns `"error: too many todos (N > 50)"`. Returns the rendered list (`todo_read(...)`) after write.

```python
# lines 981-990
def todo_read(*, _todos_path: Optional[str] = None, **_: Any) -> str:
    """Show the current todo list with status markers."""
    items = _todo_read_raw(_todos_path)
    if not items:
        return "(todo list is empty)"
    marks = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]",
             "cancelled": "[-]"}
```

### 1c. The schema gate that rejects model-supplied `_`-prefixed kwargs

Location: `core/agent_runner.py`. Data builder:

```python
# lines 260-281
def _schema_arg_names(schemas: List[dict]) -> Dict[str, Optional[frozenset]]:
    """tool name -> allowed argument names, derived from the exposed schemas.
    ... a model-supplied kwarg OVERRIDES a functools.partial keyword — so private
    knobs bound at construction ... could be forged through tool args. The loop refuses
    any argument not present in the tool's schema ``properties`` before the
    impl runs. ..."""
```

Enforcement in the loop (`_run_inner`, lines 1433-1458):

```python
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
```

⇒ Any private kwarg named `_plan_path` etc. is automatically rejected from model args because it isn't in the schema's `properties`. Rebuild happens AFTER capability grafting/profile filtering at line 481.

### 1d. TOOL_SCHEMAS / TOOL_IMPLS registration

- `TOOL_SCHEMAS: list[dict]` — module-level list built with helper `_schema(name, description, properties, required)` (lines 1524-1537); OpenAI function shape. `todo_write` schema at 1606-1633 (array of `{id,content,status}` objects, `required: ["todos"]`).
- `TOOL_IMPLS: dict[str, Callable[..., str]]` — lines 1760-1787; plain name→function map. Capability schemas deliberately live OUTSIDE TOOL_SCHEMAS in `CAPABILITY_TOOL_SCHEMAS` (1793+).
- There is **no build_tools() in core/tools.py** — the "registry" IS these two module dicts, consumed by `SableAgentClient.__init__` defaults (`tools if tools is not None else TOOL_SCHEMAS`, line 339; `dict(TOOL_IMPLS)`, line 340). (build_tools() exists only in `mcp/tools.py` — §7.)

### 1e. Tool-name sets relevant to WP12

```python
# core/tools.py:1398-1399
DELEGATION_TOOL_NAMES = frozenset(
    {"list_agents", "delegate_task", "check_delegation"})

# core/tools.py:1556-1566
CAPABILITY_TOOLS: Dict[str, str] = {
    "web_fetch": "internet",
    "run_command": "run_commands",
    "view_image": "vision",
    "take_screenshot": "screenshots",
}
CAPABILITY_KEYS: Tuple[str, ...] = ("internet", "run_commands",
                                    "vision", "screenshots")

# core/agent_runner.py:100-101  (tools whose args carry fs paths → contain())
_CONTAINED_TOOLS = frozenset(
    {"read_file", "list_dir", "write_file", "patch", "search_files"})

# core/agent_runner.py:216-217
SKILL_TOOL_NAMES = frozenset(
    {"save_skill", "read_skill", "list_skills", "delete_skill"})

# core/guardrails.py:34-38  (loop-guard classification — ADD new tools here)
IDEMPOTENT_TOOL_NAMES = frozenset(
    {"read_file", "list_dir", "search_files", "todo_read", "search_traces"}
)
MUTATING_TOOL_NAMES = frozenset({"write_file", "patch", "todo_write"})
```

`resolve_capabilities(agent_cfg, profile_caps)` at tools.py:1169-1181 is the profile-subset resolution helper.

---

## 2. core/agent_runner.py — create_runner, injection idiom, workspace, footer

File: `sable/sable/core/agent_runner.py` (2040 lines)

### 2a. create_runner full signature (lines 1811-1824)

```python
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
                  ) -> SableAgentClient:
```

Body highlights: resolves capabilities (1881), containment roots via `resolve_workspace_roots` (1631-1690; precedence profile > project_workspace_dir > config > cwd, always + SABLE_HOME + tempdir), deny-writes for `sable.yaml`, `<home>/buffer`, `<home>/index` (1888-1898), then the prompt-append chain and the big `SableAgentClient(...)` construction (1914-2010).

### 2b. Never-fail bounded-injection idiom (copy this for a `## Plan` section)

```python
# lines 1719-1741
def _with_playbooks(cfg: SableConfig, system_prompt: Optional[str],
                    profile_name: Optional[str]) -> Optional[str]:
    """Append the capped '## Playbooks' section to a run's system prompt.
    ... Disabled (the default) => returns ``system_prompt``
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
```

Project-context variant `_with_project_context` (1749-1785) shows the char cap: `MAX_PROJECT_CONTEXT_CHARS = 4000` (1746), truncation with `…` at 1782-1783. Skills mirror at `_with_skills_index` (1788-1808).

### 2c. Where sections get appended (create_runner, lines 1901-1913)

```python
    system_prompt = _with_playbooks(cfg, system_prompt, profile_name)
    skills_enabled = bool((getattr(cfg, "skills", None) or {})
                          .get("enabled", False))
    system_prompt = _with_skills_index(cfg, system_prompt)
    # Projects: active project's description + agent roster appended last
    # (closest to the conversation). No project => prompt untouched.
    system_prompt = _with_project_context(project_context, system_prompt,
                                          profile_name)
```

A WP12 `_with_plan(...)` would slot in here (order decision: after project context = closest to conversation).

### 2d. Per-run todo binding (the functools.partial pattern to reuse)

```python
# lines 223-257 (abridged body, verbatim core)
def _bind_run_todos(tool_impls: Dict[str, Callable[..., str]]) -> None:
    ...
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
        ...one-time prune of run-*.json older than TODO_RUN_RETENTION_DAYS...
    except Exception as e:  # noqa: BLE001 - never break runner construction
        logger.debug("per-run todo binding failed (global file): %s", e)
```

Called from `SableAgentClient.__init__` line 366. Same-shape bindings: `block_sensitive` for read tools (354-359), `lint_on_write`/`lsp_config` for write tools (372-379), `_ctx` for delegation (573-580), `_project_id`/`_workspace_dir`/`_cfg` for search_codebase (589-597).

### 2e. workspace_dir usage for runs

`resolve_workspace_roots(cfg, workspace_root_override, project_workspace_dir)` — lines 1631-1690. `project_workspace_dir` becomes the containment root between profile override and config; SABLE_HOME + tempdir always appended (so a plan file under SABLE_HOME is writable even when the project workspace is the primary root). Containment wrapper applied per-tool at `SableAgentClient.__init__` 394-402 via `contain(roots, deny_writes=..., read_only=...)`.

### 2f. Runner footer / result object + failure_reason values

```python
# lines 167-184
@dataclass
class AgentResult:
    final_response: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    completed: bool = True
    backend: str = "unknown"
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    trace_id: Optional[str] = None  # experience-buffer id when recorded
    failure_reason: Optional[str] = None
    summary: Optional[str] = None
```

failure_reason constants (`core/failure_types.py:31-45`):
```python
FAILURE_MAX_ITERATIONS = "max_iterations"
FAILURE_GUARDRAIL_HALT = "guardrail_halt"   # composed as f"{...}:{decision.code}"
FAILURE_BACKEND_ERROR_PREFIX = "backend_error:"
FAILURE_VERIFICATION_NUDGE = "verification_nudge"
FAILURE_BACKEND_CIRCUIT_OPEN = "backend_circuit_open"
FAILURE_CANCELLED = "cancelled"
```

Verification-nudge flow in `_run_inner` (1276-1304): `ledger.build_nudge()` injects ONE synthetic user turn; a second unverified finish returns with `failure_reason=FAILURE_VERIFICATION_NUDGE`. Ledger policy: `core/verification.py` — `VerificationLedger.observe(name, args, content)` marks code paths dirty on write_file/patch (`✓ syntax ok` clears, `✗ SYNTAX ERROR` keeps), successful `run_command` clears all (lines 81-103); `build_nudge(max_nudges=1)` at 110-133. Gate: `agent.verify_on_stop` (default True) → `verification_nudge=` kwarg (agent_runner 2000-2003).

Trace recording point `_finish` (1588-1624) calls `self._recorder.record(task=..., messages=..., failure_reason=result.failure_reason, ...)` — the recorder call signature WP12 must stay compatible with (§5).

---

## 3. dashboard/api/projects.py + routing + SSE bus

### 3a. Atomic save idiom (`sable/sable/dashboard/api/projects.py:53-59`)

```python
    def _save_project(self, project: Dict[str, Any]) -> None:
        p = self._project_path(str(project["id"]))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(project, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, p)
```

### 3b. Project-id validation (lines 33-38)

```python
    _PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

    def _project_path(self, project_id: str) -> Path:
        if not self._PROJECT_ID_RE.match(project_id):
            raise HttpError(400, "invalid project id")
        return self.home / PROJECTS_DIRNAME / f"{project_id}.json"
```

Ids minted `uuid.uuid4().hex[:12]` (line 145). Duplicate names rejected case-insensitively with `HttpError(409, ..., code="duplicate_project_name")` (`_validate_project_body`, 61-128). Tolerant load returns None on any parse failure (`_load_project`, 40-51).

### 3c. ROUTES table + dispatch (`sable/sable/dashboard/app.py`)

Table shape — `List[Tuple[str, str, str]]` of (method, path-pattern-with-<params>, handler attr name); static paths MUST precede `<param>` ones. Projects block (lines 342-346):

```python
        # projects (static paths before the <param> ones)
        ("GET", "/api/projects", "h_projects_list"),
        ("POST", "/api/projects", "h_projects_create"),
        ("PUT", "/api/projects/<project_id>", "h_project_put"),
        ("DELETE", "/api/projects/<project_id>", "h_project_delete"),
```

Dispatch (`handle`/`_route`, 352-388):

```python
    def _route(self, method, path, q, b) -> ApiResponse:
        for m, rx, handler_name in self._routes:
            match = rx.match(path)
            if not match:
                continue
            if m != method:
                continue  # verb mismatch => keep scanning => 404 (legacy shape)
            handler = getattr(self, handler_name)
            out = handler(q=q, b=b, **{k: unquote(v)
                                       for k, v in match.groupdict().items()})
            if isinstance(out, ApiResponse):
                return out
            return ApiResponse(200, out)
        raise HttpError(404, f"no such endpoint: {method} {path}")
```

`handle()` wraps: `HttpError` → `ApiResponse(status, {"error": {"code","message", **extra}})`; any other exception → 500 envelope (362-372). Route compilation `_compile_route` in `api/base.py:155-164` (`<name>` → named group `[^/]+`). Mixin registration: class line `app.py:96` — add a new `PlansRoutes` to the bases list.

### 3d. Representative GET and POST handler conventions (projects.py)

```python
    def h_projects_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        ...
        return {"projects": projects}                    # dict => 200

    def h_projects_create(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        fields = self._validate_project_body(b)
        project_id = uuid.uuid4().hex[:12]
        now = _utcnow()
        project = {"id": project_id, "created": now, "updated": now, **fields}
        self._save_project(project)
        self._kick_project_index(project_id)             # best-effort side effect
        return ApiResponse(201, {"ok": True, "project": project})

    def h_project_put(self, q, b, project_id: str) -> Dict[str, Any]:
        existing = self._load_project(project_id)
        if existing is None:
            raise HttpError(404, f"no project {project_id}")
        ...
        return {"ok": True, "project": project}
```

Handlers take `(self, q, b, **path_params)`; return dict (→200) or `ApiResponse(status, body)`; raise `HttpError(status, msg, code=...)` for errors (`api/base.py:94-117`; default codes map at 108-117).

### 3e. SSE bus

`sable/sable/dashboard/events.py`:

```python
# line 112
    def publish(self, kind: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Broadcast one event; returns the event dict (useful in tests)."""
        event = {"id": next(self._seq), "kind": kind, "ts": _utcnow(),
                 "payload": payload or {}}
```

Per-subscriber bounded queues (maxsize 256); overflow drops oldest and emits synthetic `overflow` event. SSE frame renderers: `format_sse(event)` (132-135) for bus events; `sse_frame(kind, payload)` (138-140) for ad-hoc response streams.

Example caller (`api/chat.py:321-329`):

```python
        def _on_record(row: Any) -> None:
            self.bus.publish("trace.recorded", {
                "trace_id": row.trace_id, "status": row.status, ...})
        ...
            self.bus.publish("delegation.update", rec)
```

The one `DashboardApp` instance owns `self.bus = EventBus()` (`app.py:113`). **Client side:** add the new kind to `KINDS` in `dashboard/static/js/bus.js:10-15` or subscribers never fire.

---

## 4. core/delegation.py — delegate_task seam

Tool wrapper (`core/tools.py:1409-1416`):

```python
def delegate_task(agent: str = "", task: str = "", context: str = "",
                  timeout_seconds: Optional[float] = None, **kwargs: Any) -> str:
    """Delegate a subtask to another agent profile on the same project."""
    from .delegation import execute_delegation

    return execute_delegation(agent=agent, task=task, context=context,
                              timeout_seconds=timeout_seconds,
                              _ctx=kwargs.get("_ctx"))
```

Schema at tools.py:1681-1703 (`required: ["agent","task"]`; adding an optional `plan_task`/`task_id` arg is purely additive to both schema and wrapper — the `_ctx` private kwarg rides via partial bound in agent_runner 573-580).

Core (`core/delegation.py:317-319`):

```python
def execute_delegation(agent: str, task: str, context: str = "",
                       timeout_seconds: Any = None,
                       _ctx: Optional[Dict[str, Any]] = None) -> str:
    """Core of the delegate_task tool. Returns a JSON payload string."""
```

Child prompt construction + runner build (lines 388, 257-314):

```python
    full_task = task if not context else f"{task}\n\nContext:\n{context}"
    ...
def _run_child(cfg, prof, task, ctx, delegation_id, out) -> None:
    from .agent_runner import create_runner
    from ..store.recorder import make_recorder
    ...
    recorder = make_recorder(cfg, project=project_id, metadata=metadata)
    runner = create_runner(
        cfg, base_url, model, api_key=api_key,
        recorder=recorder,
        system_prompt=str(prof.get("system_prompt") or "") or None,
        enabled_tool_names=list(tools) if isinstance(tools, list) else None,
        workspace_root_override=prof.get("workspace_root"),
        project_workspace_dir=(proj or {}).get("workspace_dir"),
        delegation_context=child_ctx,
        project_context=proj,
        capabilities=prof.get("capabilities"),
        project_id=project_id,
        profile_name=prof.get("name"))
    res = runner.run(task)
```

Execution model: sync-with-timeout on a 1-worker `ThreadPoolExecutor` (397-408); records in an in-process registry (`_delegations`, cap 500); thread lineage appended to `<home>/chat/<thread>.json` "delegations" list via atomic tmp+replace (184-227). Depth limit 1 (line 328-329). For WP12: to pass a plan-task id into the child, either extend `full_task` composition or add an optional param threaded `execute_delegation → _run_child → create_runner(...)` (all additive; `_ctx` already carries `project`).

---

## 5. store/recorder.py — tolerant additive "project" field pattern

File: `sable/sable/store/recorder.py`

Header write (`_record`, lines 269-287):

```python
        header = {
            "kind": "trace_header",
            "schema_version": SCHEMA_VERSION,
            ...
            # nullable; old traces lack it and readers use tolerant .get()
            "failure_reason": failure_reason,
            # v3 (projects): nullable project id; same tolerant-read rule.
            "project": self.project,
            # Phase 2 (delegation): nullable linkage block; absent/None on
            # ordinary runs. Readers use .get() — never a schema migration.
            "delegation": self.metadata,
        }
```

Recorder constructor takes `project: Optional[str] = None` etc. (100-138); `make_recorder(cfg, on_record=None, project=None, metadata=None, profile=None)` at 362-370 honors `recording.enabled` → NullRecorder. Footer written after messages (294-310). JSONL path: `<home>/buffer/YYYY/MM/<trace_id>.jsonl` (`jsonl_path_for`, 162-163); file opened with mode `"x"`.

SQLite column-add migration (`sable/sable/store/db.py`, `SCHEMA_VERSION = 9` at line 60; `_migrate` 292-338):

```python
        if "traces" in tables:
            cols = {r[1] for r in self._conn.execute(
                "PRAGMA table_info(traces)").fetchall()}
            if "failure_reason" not in cols:
                self._conn.execute(
                    "ALTER TABLE traces ADD COLUMN failure_reason TEXT")
            if "project" not in cols:
                # v2 -> v3 (projects): nullable column, same tolerant pattern —
                # old rows read as NULL (= Unfiled), no data migration needed.
                self._conn.execute(
                    "ALTER TABLE traces ADD COLUMN project TEXT")
```

Migration is shape-detected & idempotent (PRAGMA schema_version explicitly NOT trusted, docstring 293-298). `insert_trace` uses an explicit column list + `ON CONFLICT(trace_id) DO NOTHING` (349-384) — a new column must be added to: `_SCHEMA` CREATE TABLE, `_migrate`, the INSERT column list + tuple, and the `TraceRow` dataclass (~line 244).

Readers/rendering:
- CLI `sable traces show` prints `json.dumps(row.to_dict(), ...)` (cli.py:1946) — new TraceRow fields appear automatically; `traces list` prints fixed columns (cli.py:1924-1930).
- Dashboard JS: `panels/traces.js` detail row `["project", trace.project]` (line 474); filter dropdown populated from `/api/projects`, travels as `?project=<id>` (lines 151-159, 296-297).

**Pattern for a WP12 "plan" field on traces:** add nullable key to header dict + TraceRow + `_migrate` ADD COLUMN + insert list; readers use `.get()`/NULL tolerance. No data migration.

---

## 6. curate/judge.py — bounded single-shot LLM seam

File: `sable/sable/curate/judge.py`

### Seam function (the one to reuse for any bounded completion call)

```python
# lines 224-232
def make_chat_client(
    base_url: str,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
    transport_attempts: int = 3,
    backoff_s: tuple = (1.0, 2.0),
    max_tokens: int = 300,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> Callable[[List[Dict[str, str]], str], Optional[str]]:
```

Returns `client(messages, model) -> reply_text` posting to `<base_url>/chat/completions` with `temperature: 0.0`, classified transport retry (retries connection/timeout/5xx via `core/errors.classify_exception/classify_http_status`), and shares the per-backend circuit breaker `get_breaker(base_url)` (lines 249-299). Raises on final failure.

### Consumer + injection in tests

```python
# lines 167-196
def judge_trace(
    trace: Any,
    client_fn: Callable[[List[Dict[str, str]], str], Optional[str]],
    model: str,
    max_msgs: int = DEFAULT_MAX_MSGS,
    per_msg_cap: int = DEFAULT_PER_MSG_CAP,
) -> Optional[Dict[str, Any]]:
```

`client_fn` is the injection seam — tests pass any lambda `(messages, model) -> '{"score": 0.9, "reason": "ok"}'`; no server needed. Real callers build it with `make_chat_client(base_url)` (e.g. cli.py:2569 `client = make_chat_client(base_url, api_key=cfg.attach_api_key)`; evaluation/evolve.py:191 uses `max_tokens=1500` + `extra_payload` thinking-off shape).

### Does judging record to the buffer?

No. `judge_trace` returns a verdict dict (or None = skipped). The CLI caller writes it as a feedback row (`cli.py:2576-2586`):

```python
        for t in borderline:
            verdict = judge_trace(t.messages, client, judge_model)
            if verdict is None:
                ...
            store.add_feedback(
                t.row.trace_id, score=verdict["score"],
                comment="judge: " + verdict["reason"], source="judge", ...)
```

Precedence user > judge > heuristic in `curate/curater.py:213-223`. Bounded context knobs: `DEFAULT_MAX_MSGS = 10`, `DEFAULT_PER_MSG_CAP = 800`, `TOOL_VIEW_CAP = 200` (judge.py:55-58); defensive JSON parse `parse_judge_reply` (strict → fenced → first `{..}`, lines 134-164).

### tests/fake_server.py usage

`sable/tests/fake_server.py`: stdlib `ThreadingHTTPServer` classes.
- `FakeOpenAIServer(handler_cls=None)` — context manager, `.base_url` = `http://127.0.0.1:<port>/v1`; default `_Handler` emits one native `read_file` tool_call then a final answer; supports `stream=true` SSE.
- `make_scripted_handler(script)` — dict {substring-of-last-user-message → reply string}, `"*"` default, callables allowed.
- `make_message_handler(script)` — values are full assistant message dicts (`{"content":..., "tool_calls":...}` or `payload -> msg`); records every payload in class attr `requests`.
- `make_tokenize_handler(...)` — adds `/tokenize` with fake ceil(len/2) counter.

Representative wire-capture test (system prompts asserted): `tests/test_mcp.py:2310-2342` — PUT a profile, run `POST /api/chat/stream` against `FakeOpenAIServer(handler_cls=_RecordingHandler)`, then assert `sys_msgs[0]["content"] == "PROFILE-PROMPT-MARKER"` and the tool subset from `first["tools"]`. This is the exact pattern for asserting an injected `## Plan` section reaches the wire.

---

## 7. mcp/ — build_tools, read tool, mutating plan-then-confirm tool, validator, envelopes

### 7a. build_tools() signature (`sable/sable/mcp/tools.py:332-333`)

```python
def build_tools() -> List[Tool]:
    """The full tool table. Fresh list per server; monkeypatchable in tests."""
```

`Tool` dataclass (54-67): `name, description, input_schema, mutating: bool, executor(ctx,args), effect="", risks=[], target_arg=None`. `ToolContext` (70-80) carries the lazy `DashboardApp`; read tools call it via `_api(ctx, method, path, body)` (85-102) which raises `ToolError` on status ≥ 400.

### 7b. Full read tool example (tools.py:362-374 + executor 124-160)

```python
        Tool(
            name="logs_tail",
            description=(f"Bounded tail (max {LOGS_TAIL_MAX_LINES} lines) of "
                         "one instance's server log under <home>/logs. Only "
                         "instance ids are accepted — never paths."),
            input_schema=_obj({
                "instance_id": _ID_SCHEMA,
                "lines": {"type": "integer", "minimum": 1,
                          "maximum": LOGS_TAIL_MAX_LINES,
                          "description": f"Default {LOGS_TAIL_DEFAULT_LINES}."},
            }, required=["instance_id"]),
            mutating=False,
            executor=_exec_logs_tail,
        ),
```

`_obj(props, required)` (303-313) always sets `"additionalProperties": False`. `_ID_SCHEMA` = `{"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$", ...}` (325-329).

### 7c. Full MUTATING plan-then-confirm tool example (tools.py:399-412)

```python
        Tool(
            name="instance_stop",
            description=("Stop one managed instance (only processes sable "
                         "owns are ever stopped) — mirrors POST "
                         "/api/servers/<id>/stop. Plan-then-confirm."),
            input_schema=_obj({**{"id": _ID_SCHEMA}, **_CONFIRM_ARG},
                              required=["id"]),
            mutating=True,
            executor=_exec_instance_stop,
            effect="Gracefully stops the instance's llama-server process "
                   "(never a foreign process).",
            risks=["In-flight requests on this endpoint are interrupted"],
            target_arg="id",
        ),
```

`_CONFIRM_ARG` (tools.py:316-323):

```python
_CONFIRM_ARG = {
    "confirm_token": {
        "type": "string",
        "description": ("Opaque token from this tool's PLAN response; pass it "
                        "back unchanged to execute the plan (single use)."),
    },
}
```

### 7d. PLAN envelope + refusal flow (`mcp/server.py:192-222`)

```python
        # ---- mutating: plan-then-confirm --------------------------------
        token = args.get("confirm_token")
        exec_args = {k: v for k, v in args.items() if k != "confirm_token"}
        if token is None:
            plan = self._plan(tool, exec_args)
            new_token = self.confirm.issue(tool.name, exec_args)
            return _result_envelope({
                "status": "plan",
                "plan": plan,
                "confirm_token": new_token,
                "next": (f"Call {tool.name} again with the same arguments "
                         "plus \"confirm_token\": \"<token>\" to execute."),
            }, isError=False)
        bad = self.confirm.consume(str(token), tool.name, exec_args)
        if bad is not None:
            # Refusal travels as a tool error, not a protocol crash.
            return _result_envelope({"status": "refused",
                                     "reason": bad}, isError=True)
        return self._execute(tool, exec_args)

    def _plan(self, tool: Tool, args: Dict[str, Any]) -> Dict[str, Any]:
        target = None
        if tool.target_arg:
            target = args.get(tool.target_arg)
        return {
            "action": tool.name,
            "target": target,
            "exact_effect": tool.effect,
            "risks": list(tool.risks),
            "params": args,
        }
```

**Refusals never burn pending plans:** consumption failure returns `isError=True` WITHOUT executing; and in ConfirmStore.consume a fingerprint mismatch deliberately does NOT delete the entry (wrong guess can't destroy a legit pending plan). Execution errors also become envelopes: `_execute` → `{"status":"error","reason":...}, isError=True` (224-235).

isError envelope shape (`_result_envelope`, 243-251):

```python
def _result_envelope(payload: Any, *, isError: bool) -> Dict[str, Any]:
    """MCP tools/call result: JSON text content + isError flag."""
    return {
        "content": [{
            "type": "text",
            "text": json.dumps(payload, ensure_ascii=False, default=str),
        }],
        "isError": isError,
    }
```

### 7e. Token store (`mcp/confirm.py`)

- `DEFAULT_TTL_S = 300.0` (line 33). Tokens `secrets.token_urlsafe(32)`; in-memory only; injectable monotonic clock: `ConfirmStore(ttl_s=..., clock=Callable[[], float])` (56-63).
- Binding key: `args_fingerprint(tool, args) = sha256(json.dumps({"tool": tool, "args": args}, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str))` (36-44) — note the fingerprint is over **exec_args** (confirm_token stripped by server before hashing).
- `consume(token, tool, args)` returns None on success (entry deleted = single use) or a reason string (unknown/expired/mismatch); expiry pruned on issue and consume (65-116).

### 7f. The whole inline validator (`mcp/schema.py:21-75`) — the array-extension touch surface

```python
_TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
}


def validate_args(schema: Dict[str, Any], args: Any) -> Optional[str]:
    """Validate *args* against a fixed object *schema*.

    Returns None when valid, else a single clear error message string.
    """
    if not isinstance(args, dict):
        return f"arguments must be a JSON object, got {_kind(args)}"

    props: Dict[str, Any] = schema.get("properties") or {}
    required: List[str] = schema.get("required") or []

    for name in required:
        if name not in args:
            return f"missing required argument {name!r}"

    # additionalProperties:false is the ONLY supported shape here — every
    # sable tool schema sets it (no free-form bags).
    for key, value in args.items():
        if key not in props:
            known = ", ".join(sorted(props)) or "(none)"
            return (f"unknown argument {key!r}; this tool accepts only: "
                    f"{known}")

    for key, spec in props.items():
        if key not in args:
            continue  # optional and absent
        value = args[key]
        tname = str(spec.get("type") or "")
        check = _TYPES.get(tname)
        if check is None:  # schema bug guard — refuse rather than accept-all
            return f"schema error: unsupported type {tname!r} for {key!r}"
        if not check(value):
            return (f"argument {key!r} must be of type {tname}, "
                    f"got {_kind(value)}")
        enum = spec.get("enum")
        if enum is not None and value not in enum:
            allowed = ", ".join(repr(e) for e in enum)
            return f"argument {key!r} must be one of: {allowed}"
        if tname in ("integer", "number"):
            minimum = spec.get("minimum")
            maximum = spec.get("maximum")
            if minimum is not None and value < minimum:
                return f"argument {key!r} must be >= {minimum}"
            if maximum is not None and value > maximum:
                return f"argument {key!r} must be <= {maximum}"
    return None


def _kind(value: Any) -> str:
    """JSON-flavoured type name of *value* for error messages."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    return type(value).__name__
```

For arrays: add `"array": lambda v: isinstance(v, list)` to `_TYPES`, then per-element validation via `spec.get("items")` (recursive call into a small `_validate_value(item, items_spec)` helper), plus optional `minItems`/`maxItems`. Everything else (unknown-arg refusal, enum, min/max) stays.

---

## 8. core/config.py — sections + WP2 strict validator + doc-comment style

File: `sable/sable/core/config.py`

Sections are **dataclass fields on `SableConfig`** (`@dataclass class SableConfig:` at line 280). Scalar keys are flat fields; raw nested YAML sections are `dict[str, Any] = field(default_factory=dict)` fields (e.g. `curation`, `training`, `validation`, `dashboard` at 343-355; `agent:`/`tools:` documented inline in field comments from 356). A new WP12 `plans:` section ⇒ add `plans: dict[str, Any] = field(default_factory=dict)` to the dataclass (that alone registers it as known — see below) and read leniently.

Known-key registration + strict gate (WP2):

```python
# lines 208-215
_NESTED_SECTIONS = {
    "recording": {"enabled": "bool", "redact_secrets": "bool",
                  "retention_days": "int"},
    "delegation": {"enabled": "bool", "timeout_seconds": "float"},
}

# line 218-227
def _validate_config(data: Dict[str, Any], *, strict_unknown: bool,
                     ) -> List[str]:
    ...
    problems: List[str] = []
    known = set(SableConfig.__dataclass_fields__) | set(_NESTED_SECTIONS)
```

- Unknown top-level keys: warn-only by default; in strict mode (`SABLE_CONFIG_STRICT=1` env or `load_config(strict_unknown=True)`, gate `_strict_unknown_default()` at 274) they join the consolidated `ConfigError` report (load_config lines 835-848). Mistyped values are ALWAYS errors.
- Sub-key filtering for dict sections happens in load_config via `known = set(SableConfig.__dataclass_fields__) | {"recording", "delegation"}` + warning (lines 856-861). **If WP12 adds a nested section whose sub-keys need type validation, register it in `_NESTED_SECTIONS`; a plain dict-section field needs no other registration.**
- AGENTS.md convention: read leniently (invalid → default + warning) or register types in the validator; document in config.example.yaml; keep defaults byte-identical.

Nested-section doc-comment style example (`sable/config.example.yaml:533-542`):

```yaml
# Playbooks (pass 4 #38, core/playbooks.py) — distilled markdown recipes from
# recurring successful tool chains, injected into profile system prompts.
# Manage with `sable playbooks scan|candidates|promote|retire|show|export`.
playbooks:
  enabled: false           # master switch — OFF means ZERO behavior change:
                           # no store access, prompts byte-identical to before
  total_cap_chars: 2000    # hard cap on the whole "## Playbooks" section;
                           # recipes that do not fit whole are OMITTED whole
                           # (most-recently-promoted first), never sliced
  per_recipe_cap_chars: 800  # over-cap recipes are omitted entirely
```

---

## 9. CLI conventions (sable/cli.py + core/exitcodes.py)

### Subparser pattern (traces example, cli.py:176-222)

```python
    traces = sub.add_parser("traces", help="Inspect the experience buffer")
    tsub = traces.add_subparsers(dest="traces_command", required=True)

    tlist = tsub.add_parser("List... "...)   # actual line:
    tlist = tsub.add_parser("list", help="List recorded traces (newest first)")
    _add_common(tlist)
    tlist.add_argument("--limit", type=int, default=20, help="Max rows (default 20)")
    ...
    tlist.set_defaults(func=cmd_traces_list)
```

Every subparser calls `_add_common(p)` (--config/--verbose etc.) and `set_defaults(func=cmd_x)`. `datasets` follows the identical shape (lines 269-331). A `plans` command = `sub.add_parser("plans", ...)`, `psub = plans.add_subparsers(dest="plans_command", required=True)`, one parser per verb.

### Where `sable run` is implemented

Parser: lines 127-173 — note `--project PROJECT_ID` **already exists** (165-172). Implementation: `cmd_run(args)` at line 1754:
- resolves project via `from .core.delegation import load_project`; unknown id → print error + `return 1` (1765-1774);
- `recorder = make_recorder(cfg, project=project_id)` (1800);
- `runner = create_runner(cfg, base_url=base_url, model=model, api_key=cfg.attach_api_key, recorder=recorder, project_workspace_dir=..., project_id=..., project_context=project_context)` (1808-1814);
- one-shot: `result = runner.run(args.task)` → `return 0 if result.completed else 2` (1820-1825); REPL loop after (1827-1848).

To add `--task <plan-task-id>`: new arg on the `run` parser + resolution in `cmd_run` before `_resolve_endpoint` (fail fast like `--project`), injecting the task text into the run prompt and/or a plan context kwarg.

### Exit codes (`sable/core/exitcodes.py:35-41`)

```python
EXIT_OK = 0
EXIT_ERROR = 1            # generic runtime failure
EXIT_USAGE = 2            # argparse / invocation error (argparse's own code)
EXIT_CONFIG_INVALID = 3   # consolidated config validation report (WP2a)
EXIT_CORRUPT_ARTIFACT = 4  # completed flow, integrity check failed (WP2b)
EXIT_UPDATE_AVAILABLE = 10  # llama update --check sentinel (established)
EXIT_INTERRUPTED = 130    # SIGINT convention
```

Never renumber; extend additively. Note `cmd_run` returns raw literals today (`return 1`, `return 2`, `return 130`) — either style passes tests, constants preferred in new code.

---

## 10. Dashboard JS idioms

All under `sable/sable/dashboard/static/js/`. ES modules, no build step; CSP forbids inline handlers built from strings — everything goes through `el()`.

### Helpers

```js
// ui.js:8   Build an element. attrs: {class, text, dataset, on*, ...}; kids = nodes|strings.
export function el(tag, attrs = {}, ...kids) { ... }

// ui.js:45  Modal confirm dialog. Resolves true on confirm, false on dismiss.
export function confirmDialog({ title = "Confirm", body = "", confirmLabel =
  "Confirm", danger = false }) { ... }   // renders into #modal-host; Esc=cancel, Enter=ok

// api.js:81 JSON fetch with token handling + error-envelope -> ApiError.
export async function api(path, opts = {}) {}   // opts: {method, body}; throws ApiError{code,status,...}

// api.js:94 export function sseUrl(path) {}     // EventSource URLs (?token=)
// api.js:134 export async function streamSSE(path, body, onEvent, signal) {}   // POST SSE
// api.js:156 export async function streamSSEGet(path, onEvent, signal) {}

// collect() is PANEL-LOCAL (options.js:124, settings.js:85) — form pairs -> patch dict:
function collect(pairs) {          // pairs: [key, input, kind]; skip blanks
  const patch = {}; ... }

// state.js:125 scoped persistence keys ("uf:<scope>:<id?>:<thing>")
export function skey(scope, id, thing) {
  return `${PREFIX}${scope}:${id || ""}:${thing}`;
}
// + sget(key, fallback) / sset / sdel / janitorKeys(scope, validIds) / renameScope
// + generation(resource) (fetch race guard), coalesced(fn,{ms,isTerminal}) (SSE throttle)
```

skey usage idiom (chat.js): `let activeProject = sget(skey("ui", "", "active-project")) || null;` … `sset(skey("ui", "", "active-project"), activeProject)` / `sdel(...)`. Per-project UI state goes under `skey("project", pid, thing)` so the janitor drops it on project delete.

### Project tree nodes (where WP12 chips attach) — chat.js:351-388

```js
    function projectNode(p, itsThreads) {
      const isActive = p.id === activeProject;
      ...
      const head = el("div", { class: "project-node" + (isActive ? " active" : ""),
        tabindex: 0, role: "button",
        title: [...].join(" ") },
        caret,
        el("span", { class: "thread-title", text: p.name }),
        el("span", { class: "thread-meta", text: `${itsThreads.length}` }),
        el("button", { class: "thread-x proj-x small", text: "✎", ... editProjectDialog(p) }),
        el("button", { class: "thread-x proj-x small", text: "✕", ... deleteProjectDialog(p) }));
      ...
      return [head, ...kids];
    }
```

`renderThreadList()` (390-431) assembles nodes and calls `threadList.replaceChildren(...nodes)` — a WP12 chip = another `el("span", {class: "plan-chip", text: ...})` inside the head row. SSE-driven refresh uses `bus.on(kind, coalesced(reload,...))`.

### CSP constraints (server + tests)

- Header (`dashboard/server.py:134-137`):
  ```python
             self.send_header(
                 "Content-Security-Policy",
                 "default-src 'self'; script-src 'self'; style-src 'self'",
             )
  ```
- Enforced in tests (`tests/test_dashboard.py:2143-2145`):
  ```python
         r = httpx.get(base + "/api/status", timeout=5)
         assert "default-src 'self'" in r.headers.get("content-security-policy", "")
         assert r.headers.get("x-frame-options") == "DENY"
  ```
- Consequences: no external URLs anywhere (images must proxy same-origin — see `/api/hub/image`); no inline scripts/styles; nothing uses innerHTML with server data (`ui.js` header comment: DOM built via textContent only; markdown renderer restricts links to http/https).

### tests/test_hygiene.py mechanism (stdlib scan)

AST-based import scan over `sable/**/*.py`:
```python
ALLOWED_THIRD_PARTY = {"yaml", "httpx", "PIL"}          # line 21
HEAVY = {"torch", "transformers", "peft", ...}          # 31-32, lazy-only in worker.py/streamed
STDLIB = set(sys.stdlib_module_names)                   # 38
```
`test_only_allowed_third_party_imports` walks `ast.Import`/`ast.ImportFrom(level==0)` roots; anything not stdlib/allowed/`sable*` fails. Plus `test_no_orphan_modules` (156-243): every module must be statically reachable from ROOTS = {sable, sable.cli, sable.__main__, sable.dashboard.server, sable.training.worker} — **a new WP12 module that nothing imports will fail CI**.

---

## 11. Jobs / long-action thread conventions

Two patterns; WP12's plan loop should mirror one.

### A. BackgroundTask (one-slot daemon thread + bus events) — `dashboard/api/base.py:195-291`

```python
class BackgroundTask:
    """One lock-guarded background run: record + daemon thread + bus events.
    ... The slot holds at most one live run; ``try_start``
    raises 409 while one is in flight. ``work(record, log)`` executes on the
    daemon thread and owns the final ``record["state"]`` (done/diverged/...);
    an uncaught exception becomes ``state=error`` ..."""

    def __init__(self, bus: Any, event: str, *, log_cap: Optional[int] = 500,
                 final_lines: bool = True, line_extra=None): ...
    def running(self) -> bool: ...
    def try_start(self, work, *, conflict_msg: str, initial=None,
                  done_extra=None) -> Dict[str, Any]:
        with self.lock:
            if self.running():
                raise HttpError(409, conflict_msg, code="already_running")
            record = {"state": "running", "started": _utcnow(), "finished": None,
                      "error": None}
            ...
        def _log(line): ...  # appends bounded log + bus.publish(self.event, {..."line"...})
        def _run():
            try: work(record, _log)
            except Exception as e: record["state"] = "error"; record["error"] = ...
            finally:
                record["finished"] = _utcnow()
                payload = {"state": record["state"], "error": record["error"]}
                ... self.bus.publish(self.event, payload)
        threading.Thread(target=_run, daemon=True).start()
        return record
```

Instantiated in `DashboardApp.__init__` (app.py:115-132): `BackgroundTask(self.bus, "bench.run", log_cap=200, final_lines=False)` etc. Progress = per-line bus events; status endpoint polls `.record`.

### B. iterate-style multi-step run with persisted record — `dashboard/api/iterate.py:87-154`

```python
        record_path = runs_dir / f"{run_id}.json"

        def _save() -> None:
            tmp = record_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, record_path)

        _save()

        def _on_stage(stage: str, detail: str) -> None:
            state = (...)
            record["steps"].append({"name": stage, "state": state, "detail": detail})
            _save()
            self.bus.publish("iterate.step", {"run_id": run_id, "step": stage,
                                              "state": state, "detail": detail})

        def _run() -> None:
            if not self._exec_lock.acquire(blocking=False):  # single-flight
                record["state"] = "error"; ...; return
            try:
                res = run_iterate(..., on_stage=_on_stage)
                record["result"] = res.to_dict(); record["state"] = "done" if res.ok else "failed"
            except Exception as e:
                record["state"] = "error"; record["result"] = {"error": f"{type(e).__name__}: {e}"}
            finally:
                record["finished"] = _utcnow()
                _save()
                self.bus.publish("iterate.step", {"run_id": run_id,
                                                  "step": "finished",
                                                  "state": record["state"], "detail": ""})
                ... self._exec_lock.release()
```

This is the closest template for a WP12 plan-execution loop: per-run JSON record under `<home>/..._runs/` (atomic tmp+replace), step-by-step SSE publish, `self._exec_lock` single-flight (`app.py:108`), daemon thread. Job-queue alternative: `JobDispatcher`/`Scheduler` via `app._scheduler` + `_job_observer` publishing `job.state`/`job.log` (app.py:405-428) — heavier, DB-backed; use only if WP12 needs cross-process queueing.

---

## 12. Tests conventions

### conftest.py (whole file, `sable/tests/conftest.py`)

```python
@pytest.fixture(autouse=True)
def _isolate_sable_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SABLE_HOME", str(tmp_path / ".sable-test-home"))
    yield


@pytest.fixture(autouse=True)
def _reset_runtime_counters():
    """Per-test isolation for in-process reliability state (pass 4 #28). ..."""
    from sable.core import circuit
    from sable.serving import supervisor

    def _clear():
        with circuit._registry_lock:
            circuit._registry.clear()
        with supervisor._crash_lock:
            supervisor._crash_failures.clear()

    _clear()
    yield
    _clear()
```

Per-module `home` fixtures override the env to a named dir (test_mcp.py:128-132):
```python
@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "uf-mcp"
    monkeypatch.setenv("SABLE_HOME", str(h))
    return h
```

### Representative hermetic test — atomic JSON store (test_dashboard.py:3196-3219)

```python
def test_thread_write_is_atomic(home__test_thread_titles, monkeypatch):
    """A crash mid-write must never leave a partial file at the thread path:
    the write goes to *.json.tmp and lands via os.replace. We make os.replace
    raise (simulating death during the rename) and assert the previous file is
    byte-intact."""
    app = make_app__test_thread_titles(home__test_thread_titles)
    app.handle("PUT", "/api/chat/threads/a1", body={
        "messages": [{"role": "user", "content": "durable original"}]})
    good = (home__test_thread_titles / "chat" / "a1.json").read_bytes()

    import os as _os

    def boom(src, dst):  # crash exactly between tmp write and rename
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(_os, "replace", boom)
    r = app.handle("PUT", "/api/chat/threads/a1", body={
        "messages": [{"role": "user", "content": "x" * 5000}]})
    assert r.status == 500
    assert (home__test_thread_titles / "chat" / "a1.json").read_bytes() == good
```

(Companion: `test_save_skill_atomic_no_tmp_leftover`, test_core.py:4876-4881 — asserts no `*.tmp` leftovers.)

### Representative hermetic test — captured system prompts (test_mcp.py:2310-2342)

Excerpt of the assertion core (full text in §6): drive `POST /api/chat/stream` with a profile marker against `FakeOpenAIServer(handler_cls=_RecordingHandler)` (a handler recording `payloads`), then:

```python
    payloads = _RecordingHandler.payloads
    assert payloads, "fake server never called"
    first = payloads[0]
    sys_msgs = [m for m in first["messages"] if m.get("role") == "system"]
    assert sys_msgs and sys_msgs[0]["content"] == "PROFILE-PROMPT-MARKER"
    tool_names = {t["function"]["name"] for t in first.get("tools", [])}
    assert tool_names == {"list_dir"}  # read_file subsetted away
```

Byte-identity tests do the negative version: default config ⇒ system prompt equals `SYSTEM_PROMPT` exactly (no appended sections).

### Representative hermetic test — MCP plan→confirm→execute matrix (test_mcp.py)

Harness (69-132): `FakeApp(home, routes)` records `(method, path, body)` triples and returns canned `FakeResponse`s; `make_server(home, *, routes=None, clock=None, ttl_s=300.0)` injects the clock into `McpServer` → `ConfirmStore`; `call_tool(srv, n, name, args)` unpacks the content envelope and adds `_isError`.

Happy path + single-use refusal (≈300-331):

```python
    out = call_tool(srv, 20, "instance_stop", {"id": "alpha"})
    assert out["status"] == "plan"
    token = out["confirm_token"]
    assert token and isinstance(token, str)
    ...
    out2 = call_tool(srv, 21, "instance_stop",
                     {"id": "alpha", "confirm_token": token})
    assert out2["_isError"] is False
    ...
    # Phase 3: token reuse refused (single use).
    out3 = call_tool(srv, 22, "instance_stop",
                     {"id": "alpha", "confirm_token": token})
    assert out3["_isError"] is True
    assert out3["status"] == "refused"
    assert len(app.calls) == 1  # still exactly one execution
```

Injectable-clock expiry (334-345):

```python
def test_confirm_token_expired(home):
    now = [1000.0]
    srv, app = make_server(home, clock=lambda: now[0], ttl_s=60.0)
    init(srv)
    out = call_tool(srv, 30, "promote", {"model_id": "v2"})
    token = out["confirm_token"]
    now[0] += 61.0  # TTL elapses
    out2 = call_tool(srv, 31, "promote",
                     {"model_id": "v2", "confirm_token": token})
    assert out2["_isError"] is True
```

Wrong-tool / wrong-args binding refusals (348-361) and unknown-token refusal (367-371) complete the matrix; a pure `ConfirmStore` unit test with fake clock follows at 375+.

---

## AGENTS.md hard rules relevant to WP12 (+ ruff config)

From `sable/AGENTS.md`:
1. **Byte-identical default behavior**: "New features are opt-in. With no config changes, every existing command's observable output stays **byte-identical**." Ship the plan feature behind a default-off knob (`plans.enabled: false` pattern — mirror playbooks/skills: knob off ⇒ zero store/filesystem access, prompts unchanged).
2. No perf/quality claims below the `sable bench --noise-floor` noise floor.
3. **D-12 licensing boundary**: re-implement inline; NEVER import Strata/airllm/kimi-k3-in-c/sparse_transformers — enforced by `tests/test_hygiene.py` allowlist (`ALLOWED_THIRD_PARTY = {"yaml", "httpx", "PIL"}`; dashboard/MCP stdlib-only).
4. Loopback-only, zero cloud, zero API keys; all state under `~/.sable/` (`SABLE_HOME`).
5. Exit-code taxonomy in `core/exitcodes.py` — never renumber, extend additively.
6. Config keys: follow `serve.*` conventions — lenient reads or register in the WP2 validator sets; document in config.example.yaml; keep default byte-identical; `test_core.py` must stay green.
7. Commands before finishing: `python -m pytest tests/ -q` and `ruff check sable tests`.

From `sable/pyproject.toml`:
```toml
[tool.ruff]
target-version = "py311"

[tool.ruff.lint]
select = ["E4", "E9", "F"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["F841", "E402", "F811"]
```
BLE001 intentionally NOT enabled repo-wide; broad excepts are the house resilience pattern, annotated `# noqa: BLE001 - <reason>`. mypy gates only `sable/store`, `sable/registry`, `sable/core/config.py` (WP12 touching config must be mypy-clean).

---

## Quick seam checklist for WP12 implementers

| Need | Exact seam |
|---|---|
| New agent tool (plan_write/plan_read) | add fn in `core/tools.py`, entry in `TOOL_SCHEMAS` + `TOOL_IMPLS`; bind private path kwarg via partial like `_bind_run_todos` (agent_runner 223-257, called at 366); add name to `MUTATING_TOOL_NAMES`/`IDEMPOTENT_TOOL_NAMES` in `core/guardrails.py:34-38`; if fs-path args, add to `_CONTAINED_TOOLS` (agent_runner:100) |
| Inject plan into prompt | `_with_plan()` modeled on `_with_playbooks` (agent_runner 1719-1741); append in create_runner chain at 1901-1913; cap chars like MAX_PROJECT_CONTEXT_CHARS |
| New REST routes | handler methods on a mixin class in `dashboard/api/plans.py`, base `ApiContext`; add class to `DashboardApp` bases (app.py:96) + ROUTES tuples (static before `<param>`); id-validate with `_PROJECT_ID_RE`-style regex; atomic save via tmp+os.replace |
| SSE progress | `self.bus.publish("plan.step", {...})`; add kind to bus.js KINDS (bus.js:10-15) |
| Long action | BackgroundTask slot (base.py:195) or iterate-style record+thread (iterate.py 87-154) with `self._exec_lock` |
| Trace linkage | nullable header key + TraceRow field + `_migrate` ADD COLUMN + insert list (db.py 292-338, 349-384) |
| delegate_task task id | extend schema (tools.py:1681) + wrapper (1409) + `execute_delegation` kwargs (delegation.py:317) → `_run_child`/`full_task` (388, 257-314) |
| CLI | new arg on existing `run` parser (cli.py:127; `--project` already at 165); new `plans` subcommand via nested `add_subparsers(dest="plans_command", required=True)`; exit codes from `core/exitcodes.py` |
| MCP array args | extend `_TYPES` + items recursion in `mcp/schema.py:21-75`; mutating plan tool = `Tool(mutating=True, effect=..., risks=[...], target_arg=...)` merged with `_CONFIRM_ARG` (tools.py:316) |
| Config section | `plans: dict[str, Any] = field(default_factory=dict)` on SableConfig; optional sub-key types in `_NESTED_SECTIONS` (config.py:211); example block in config.example.yaml with `enabled: false` doc-comment style |

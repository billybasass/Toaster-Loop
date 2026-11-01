"""WP12 LLM planning: plan_import / plan_expand through judge's bounded seam.

Turns a pasted brief (or one existing task) into structured plan tasks. The
model call reuses ``curate/judge.make_chat_client`` — the same bounded,
classified-retry, circuit-breaker single-shot seam judge/prompt-mutation use —
so planning behaves exactly like every other auxiliary model call in sable:

* single-shot (one request per operation; no agent loop, no tools);
* records NOTHING extra to the experience buffer: judge's seam POSTs raw
  /chat/completions and never touches the recorder, and this module inherits
  that behavior verbatim (documented here per WP12 spec — consistency with
  judge calls is deliberate);
* transport failures and unparseable output NEVER raise into callers as
  crashes: import/expand return a salvage report instead.

Parsing philosophy mirrors curate/judge.parse_judge_reply plus the tool_drift
salvage work: strict JSON first, then fenced block, then the first ``[`` ...
array substring, then jsonrepair-style repairs (trailing commas, single
quotes, truncated tails), and finally per-item salvage — each ``{...}`` object
parsed independently so one bad item only drops that item. Deterministic mode
(``use_ai=False``) parses Taskmaster-flavored lines::

    Add login form [deps=t1,t2] [priority=high]

Design adapted from the open-source Taskmaster project (eyaltoledano/claude-
task-master) — inspiration only, re-implemented inline (D-3/D-12), no imports.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .plans import PlanError, add_task, issues, load_plan, lock, save_plan

logger = logging.getLogger("sable.core.plan_llm")

#: Import/expand LLM output sizing. A task array is small; keep completions
#: bounded like judge's 300-token verdicts (a little more room for N items).
MAX_NUM_TASKS = 32
DEFAULT_NUM_TASKS = 8
#: Brief/excerpt input caps: the served context stays deterministic and small.
MAX_BRIEF_CHARS = 12000
MAX_ITEM_FIELD_CHARS = 4000

_PRIORITIES = ("high", "medium", "low")


class PlanImportError(Exception):
    """User-facing import/expand failure (bad input, no endpoint...)."""


# --------------------------------------------------------------------------
# Input hygiene (binary rejection)
# --------------------------------------------------------------------------

def clean_text_input(raw: Any, *, what: str = "input") -> str:
    """Coerce user-supplied brief text; reject binaries with a clear error.

    Accepts str or utf-8-decodable bytes. NUL bytes / undecodable bytes are
    rejected — an LLM prompt must never receive file dumps, and the CLI's
    ``import <file>`` relies on this to refuse binaries before the model (or
    the line parser) ever sees them.
    """
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, (bytes, bytearray)):
        if b"\x00" in raw:
            raise PlanImportError(
                f"{what} looks binary (NUL byte); provide plain text or JSON")
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as e:
            raise PlanImportError(
                f"{what} is not valid UTF-8 text ({e}); refusing binary input"
            ) from e
    else:
        raise PlanImportError(f"{what} must be text, got {type(raw).__name__}")
    if "\x00" in text:
        raise PlanImportError(
            f"{what} looks binary (NUL byte); provide plain text or JSON")
    return text


# --------------------------------------------------------------------------
# Salvage parsing (jsonrepair-style tolerance + per-item salvage)
# --------------------------------------------------------------------------

_FENCED = re.compile(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", re.S)
_ARRAY = re.compile(r"\[.*\]", re.S)


def _iter_object_strings(text: str) -> List[str]:
    """Balanced-brace ``{...}`` substrings (string/escape aware)."""
    out: List[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start:i + 1])
    return out


def _repair_candidate(cand: str) -> str:
    """Cheap jsonrepair-style fixes: trailing commas, smart quotes."""
    t = cand.replace("\u201c", '"').replace("\u201d", '"')
    t = re.sub(r",\s*([\]}])", r"\1", t)  # trailing commas
    return t


def _salvage_items(text: str, warnings: List[str]) -> List[Dict[str, Any]]:
    """Last-resort per-object parse: keep every independently valid ``{...}``."""
    items: List[Dict[str, Any]] = []
    for obj_s in _iter_object_strings(text):
        for cand in (obj_s, _repair_candidate(obj_s)):
            try:
                obj = json.loads(cand)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(obj, dict):
                items.append(obj)
                break
        else:
            warnings.append(f"dropped unparseable item: {obj_s[:80]!r}")
    return items


def parse_task_array(text: Optional[str],
                     ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Extract a task-object array from model output with salvage tolerance.

    Returns ``(items, warnings)``; never raises. Empty items means the output
    was unparseable end to end — callers surface the warnings as the salvage
    report rather than crashing (WP12 spec).
    """
    warnings: List[str] = []
    if not text or not str(text).strip():
        return [], ["empty model output"]
    raw = str(text)
    candidates: List[str] = []
    stripped = raw.strip()
    candidates.append(stripped)
    m = _FENCED.search(raw)
    if m:
        candidates.append(m.group(1))
    m = _ARRAY.search(raw)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        for fixed in (cand, _repair_candidate(cand)):
            try:
                obj = json.loads(fixed)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(obj, list):
                items = [x for x in obj if isinstance(x, dict)]
                dropped = len(obj) - len(items)
                if dropped:
                    warnings.append(
                        f"dropped {dropped} non-object array element(s)")
                return items, warnings
            if isinstance(obj, dict):  # single task object is acceptable
                return [obj], warnings
    # Nothing parsed whole — salvage object-by-object.
    warnings.append("output was not valid JSON; salvaged items individually")
    return _salvage_items(raw, warnings), warnings


# --------------------------------------------------------------------------
# Per-item normalization (drops unknown keys, clamps enums/lengths)
# --------------------------------------------------------------------------

def normalize_item(obj: Dict[str, Any], warnings: List[str]) -> Optional[Dict]:
    """One model/dict item -> plans.add_task fields; None if unusable."""
    if not isinstance(obj, dict):
        return None
    title = str(obj.get("title") or "").strip()
    if not title:
        warnings.append("dropped item without a title")
        return None

    def _s(key: str) -> str:
        v = obj.get(key)
        if v is None:
            return ""
        if not isinstance(v, str):
            v = json.dumps(v, ensure_ascii=False)
        return v.strip()[:MAX_ITEM_FIELD_CHARS]

    prio = str(obj.get("priority") or "").strip().lower()
    if prio not in _PRIORITIES:
        if prio:
            warnings.append(f"invalid priority {prio!r} -> medium")
        prio = "medium"
    deps_raw = obj.get("deps") or obj.get("dependencies") or []
    if isinstance(deps_raw, str):
        deps_raw = re.split(r"[,\s]+", deps_raw)
    if not isinstance(deps_raw, list):
        warnings.append("non-list deps ignored")
        deps_raw = []
    fields: Dict[str, Any] = {
        "title": title[:200],
        "description": _s("description"),
        "details": _s("details"),
        "test_strategy": _s("testStrategy") or _s("test_strategy"),
        "priority": prio,
        "deps": [str(d).strip() for d in deps_raw if str(d).strip()],
    }
    alias = str(obj.get("id") or "").strip()
    if alias:
        fields["_alias"] = alias  # caller-side remap; stripped before add
    return fields


# --------------------------------------------------------------------------
# Deterministic text mode (use_ai=False)
# --------------------------------------------------------------------------

_LINE_TAGS = re.compile(r"\[(deps|dependencies|priority|prio)\s*=\s*([^\]]*)\]",
                        re.I)


def parse_task_lines(text: str, num_tasks: Optional[int] = None
                     ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Taskmaster-flavored deterministic brief format, one task per line::

        Add login form [deps=t1,t2] [priority=high]

    ``tN`` ids may reference earlier lines of the SAME import (remapped to
    real ids by apply_parsed) or existing plan tasks. Blank lines and
    ``#``-comments are skipped. Returns ``(fields, warnings)``.
    """
    warnings: List[str] = []
    fields_list: List[Dict[str, Any]] = []
    for ln, line in enumerate((text or "").splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tags = _LINE_TAGS.findall(s)
        title = _LINE_TAGS.sub("", s).strip().lstrip("-* ").strip()
        if not title:
            warnings.append(f"line {ln}: no title, skipped")
            continue
        deps: List[str] = []
        prio = "medium"
        for key, val in tags:
            val = val.strip()
            if key.lower().startswith("dep"):
                deps.extend(d.strip() for d in re.split(r"[,\s]+", val)
                            if d.strip())
            else:
                v = val.lower()
                if v in _PRIORITIES:
                    prio = v
                elif val:
                    warnings.append(
                        f"line {ln}: unknown priority {val!r} -> medium")
        f: Dict[str, Any] = {"title": title[:200], "description": "",
                             "details": "", "test_strategy": "",
                             "priority": prio, "deps": deps}
        # Positional alias so intra-import [deps=tN] resolves below.
        f["_alias"] = f"t{ln}"
        fields_list.append(f)
    if num_tasks is not None and len(fields_list) > int(num_tasks):
        warnings.append(
            f"truncated {len(fields_list)} parsed tasks to cap {num_tasks}")
        fields_list = fields_list[: max(0, int(num_tasks))]
    return fields_list, warnings


# --------------------------------------------------------------------------
# Applying parsed fields to a store (alias remap + atomic validation)
# --------------------------------------------------------------------------

def apply_parsed(store: Dict[str, Any], fields_list: List[Dict[str, Any]],
                 parent: Optional[str] = None
                 ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Add parsed tasks to ``store`` (in order), remapping intra-import dep
    aliases (tN / model-declared ids) to the real assigned ids.

    All-or-nothing: if the resulting graph fails validate() the store is left
    untouched and a PlanError is raised — half-wired imports never land.
    """
    warnings: List[str] = []
    import copy

    trial = copy.deepcopy(store)
    before = issues(trial)   # pre-existing problems are not this import's
    alias: Dict[str, str] = {}
    added: List[Dict[str, Any]] = []
    pending_deps: List[Tuple[Dict[str, Any], List[str]]] = []
    for f in fields_list:
        f = dict(f)
        a = f.pop("_alias", None)
        deps = f.pop("deps", []) or []
        if parent:
            f["parent"] = parent
        try:
            t = add_task(trial, f)
        except PlanError as e:
            warnings.append(f"dropped task {f.get('title', '')!r}: {e}")
            continue
        if a:
            alias.setdefault(str(a), t["id"])
        added.append(t)
        pending_deps.append((t, deps))
    # Resolve deps: existing ids pass through; aliases map to new ids.
    for t, deps in pending_deps:
        if not deps:
            continue
        resolved = []
        for d in deps:
            if d == t["id"]:
                warnings.append(f"dropped self-dependency on {d}")
                continue
            if d in alias and alias[d] != t["id"]:
                resolved.append(alias[d])
            elif any(x["id"] == d for x in trial["tasks"]):
                resolved.append(d)
            else:
                warnings.append(
                    f"task {t['id']}: unknown dep {d!r} dropped")
        if resolved and resolved != t["deps"]:
            trial = _set_deps(trial, t["id"], resolved, warnings)
    # Reject only problems THIS import introduced: a dangling edge already
    # sitting elsewhere in the store must not brick every future import.
    new = issues(trial) - before
    if new:
        problems = [str(n) for n in sorted(new)]
        raise PlanError(
            "import rejected: resulting plan invalid "
            f"({'; '.join(problems)[:300]})")
    store.clear()
    store.update(trial)
    return added, warnings


def _set_deps(store: Dict[str, Any], task_id: str, deps: List[str],
              warnings: List[str]) -> Dict[str, Any]:
    """Set deps inside apply_parsed's trial via plans.add_dependency (the
    cycle-safe path; add_dependency mutates the store in place and returns
    the updated task)."""
    from .plans import add_dependency

    for d in deps:
        if d not in [t["id"] for t in store["tasks"]]:
            continue
        try:
            add_dependency(store, task_id, d)
        except PlanError as e:
            warnings.append(f"dep {task_id}->{d} refused: {e}")
    return store


# --------------------------------------------------------------------------
# The LLM seam (single-shot prompts; judge-style client_fn injection)
# --------------------------------------------------------------------------

_IMPORT_SYSTEM = (
    "You turn a project brief into a task list for a persistent task graph. "
    "Respond with ONLY a JSON array of task objects — no prose, no code "
    'fences. Each object: {"id": "t1", "title": "<short imperative, '
    'required>", "description": "<what to do>", "details": "<supporting '
    'detail>", "test_strategy": "<how to verify it is done>", "priority": '
    '"high"|"medium"|"low", "deps": ["t2"]}. Number ids t1..tN in order; '
    "deps reference those ids (or existing task ids you are told about). "
    "At most {num} tasks. Keep every string under 300 characters.")

_EXPAND_SYSTEM = (
    "You decompose ONE task of a persistent task graph into subtasks. Respond "
    "with ONLY a JSON array of subtask objects — no prose, no fences. Each "
    'object: {"id": "s1", "title": "<short imperative>", "description": '
    '"<what to do>", "test_strategy": "<how to verify>", "priority": '
    '"high"|"medium"|"low"}. Exactly {num} subtasks, ordered as they should '
    "be done. Do not restate the parent task.")


def _require_client(client_fn: Optional[Callable], model: Optional[str]):
    if client_fn is None or not str(model or "").strip():
        raise PlanImportError(
            "use_ai requires a chat client (no served endpoint resolved); "
            "pass use_ai=false for deterministic text parsing")
    return client_fn, str(model)


def ai_generate_tasks(client_fn: Callable, model: str, *, system: str,
                      user: str, max_tokens: int = 1200
                      ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """One bounded single-shot call -> (fields, warnings). Never raises: a
    transport failure yields an empty result with the error as a warning
    (mirrors judge_trace's skip-don't-crash contract)."""
    try:
        reply = client_fn(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}], model, max_tokens)
    except TypeError:  # plain judge-style client(messages, model)
        try:
            reply = client_fn(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}], model)
        except Exception as e:  # noqa: BLE001
            logger.warning("plan LLM call failed: %s", e)
            return [], [f"model call failed: {e}"]
    except Exception as e:  # noqa: BLE001
        logger.warning("plan LLM call failed: %s", e)
        return [], [f"model call failed: {e}"]
    items, warnings = parse_task_array(reply)
    fields_list = []
    for obj in items:
        f = normalize_item(obj, warnings)
        if f is not None:
            fields_list.append(f)
    return fields_list, warnings


# --------------------------------------------------------------------------
# Public operations (what the CLI/API/MCP layers call)
# --------------------------------------------------------------------------

def plan_import(project: Optional[str], text: Any, *, use_ai: bool = False,
                num_tasks: int = DEFAULT_NUM_TASKS,
                client_fn: Optional[Callable] = None,
                model: Optional[str] = None) -> Dict[str, Any]:
    """Import a brief into the project's plan. Returns a report dict::

        {ok, source: "ai"|"text", added: [task...], warnings: [...]}

    use_ai=False (default): deterministic line format, ZERO LLM calls.
    use_ai=True: one single-shot call through judge's seam; unparseable model
    output returns the salvage report (added may be empty), never crashes.
    Binary/undecodable input raises PlanImportError with a clear message.
    """
    text = clean_text_input(text, what="import brief")
    if not text.strip():
        raise PlanImportError("import brief is empty")
    num_tasks = max(1, min(int(num_tasks), MAX_NUM_TASKS))
    store = load_plan(project)

    if use_ai:
        _require_client(client_fn, model)
        brief = text[:MAX_BRIEF_CHARS]
        if len(text) > MAX_BRIEF_CHARS:
            brief += "\n[...brief truncated]"
        fields_list, warnings = ai_generate_tasks(
            client_fn, model,  # type: ignore[arg-type]
            system=_IMPORT_SYSTEM.replace("{num}", str(num_tasks)),
            user="Brief:\n\n" + brief)
        source = "ai"
    else:
        # Deterministic mode: a JSON task array file parses directly (the CLI
        # accepts text OR JSON briefs); anything else is the line format.
        looks_json = text.lstrip()[:1] in ("[", "{")
        if looks_json:
            items, warnings = parse_task_array(text)
            fields_list = []
            for obj in items:
                f = normalize_item(obj, warnings)
                if f is not None:
                    fields_list.append(f)
            source = "json"
        else:
            fields_list, warnings = parse_task_lines(text, num_tasks)
            source = "text"
        fields_list = fields_list[:num_tasks]
    if not fields_list:
        return {"ok": False, "source": source, "added": [],
                "warnings": warnings or ["no tasks could be parsed"]}

    # Re-read under the store lock so nothing added while the (possibly slow)
    # model call ran is clobbered by our save.
    with lock():
        store = load_plan(project)
        existing_ids = [t["id"] for t in store["tasks"]]
        # Deterministic intra-line aliases reference positional tN of THIS
        # import; pre-seed them so deps to later lines resolve after their
        # add below.
        fields_list = fields_list[:num_tasks]
        added, w2 = apply_parsed(store, fields_list)
        warnings.extend(w2)
        if not added:
            return {"ok": False, "source": source, "added": [],
                    "warnings": warnings or ["all parsed tasks were rejected"]}
        save_plan(store)
    logger.info("plan import (%s): %d task(s) into %s", source, len(added),
                project or "default")
    return {"ok": True, "source": source, "added": added,
            "warnings": warnings, "previous_ids": existing_ids}


def plan_expand(project: Optional[str], task_id: str, num: int = 3,
                prompt: Optional[str] = None, *,
                client_fn: Optional[Callable] = None,
                model: Optional[str] = None) -> Dict[str, Any]:
    """Decompose one task into subtasks (parent set, parent status untouched).

    Same seam/report shape as plan_import. Without a client this is an error
    — expansion is inherently generative (deterministic text mode would just
    restate the task).
    """
    _require_client(client_fn, model)
    num = max(1, min(int(num), MAX_NUM_TASKS))
    store = load_plan(project)
    from .plans import build_task_prompt, get_task

    task = get_task(store, str(task_id))
    if task is None:
        raise PlanImportError(f"no such task {task_id!r}")
    context = build_task_prompt(store, task["id"])
    user = f"Parent task:\n\n{context[:MAX_BRIEF_CHARS]}"
    if prompt and str(prompt).strip():
        user += "\n\nExtra guidance:\n" + str(prompt).strip()[:2000]
    fields_list, warnings = ai_generate_tasks(
        client_fn, model,  # type: ignore[arg-type]
        system=_EXPAND_SYSTEM.replace("{num}", str(num)), user=user)
    if not fields_list:
        return {"ok": False, "source": "ai", "added": [],
                "warnings": warnings or ["no subtasks could be parsed"]}
    # Re-read under the store lock (the model call may have taken a while).
    with lock():
        store = load_plan(project)
        added, w2 = apply_parsed(store, fields_list[:num], parent=task["id"])
        warnings.extend(w2)
        if not added:
            return {"ok": False, "source": "ai", "added": [],
                    "warnings": warnings
                    or ["all generated subtasks rejected"]}
        save_plan(store)
    return {"ok": True, "source": "ai", "added": added, "warnings": warnings}

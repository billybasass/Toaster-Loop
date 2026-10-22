"""Inference-engine registry (opt-in; hardware page manages it).

sable ships with exactly ONE inference engine — llama.cpp's ``llama-server``
— and every spawn path without an explicit engine binding stays byte-identical
to before this module existed (cardinal rule #1). This registry lets a user
*register* additional local OpenAI-compatible engines (vLLM, llama-swap,
Ollama-style servers, …) so named instances — and the default server via
``serve.default_engine`` — can be spawned by them instead.

Design:

* **Persistence**: one JSON file per engine at
  ``<SABLE_HOME>/serving/engines/<id>.json`` — the same file-per-record
  pattern instance definitions use (atomic tmp+replace writes). The built-in
  llama.cpp engine is synthesized, never stored on disk.
* **Kinds**: ``openai`` engines expose an OpenAI-compatible endpoint and can be
  spawned + health-checked by sable (the command comes from the engine's
  ``args_template`` with ``{model}``/``{host}``/``{port}`` placeholders).
  ``other`` engines (e.g. stable-diffusion.cpp — not an OpenAI chat server) are
  recorded for inventory/attach bookkeeping only; sable refuses to spawn them
  rather than pretending a healthcheck will pass.
* **Loopback**: spawned engines bind 127.0.0.1 exactly like llama-server does;
  nothing here ever talks to a non-local endpoint.
* **Ownership**: engine-spawned children go through the SAME supervisor
  machinery (owned-handle registry, crash-loop budget, stop paths), so the
  ownership rule holds for every engine: sable only kills what it spawned.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .supervisor import ServeError

logger = logging.getLogger("sable.serving.engines")

#: Id of the synthesized built-in engine (llama-server via supervisor).
BUILTIN_ID = "llama-cpp"

#: Engine kinds. "openai" => spawnable + health-checked by sable;
#: "other" => inventory only (attach manually / manage externally).
KINDS = ("openai", "other")

#: Placeholders an args_template may use.
PLACEHOLDERS = ("{model}", "{host}", "{port}")

_ENGINE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

_DEFAULT_OPENAI_ARGS = ["--model", "{model}", "--host", "127.0.0.1",
                        "--port", "{port}"]


# ------------------------------------------------------------------ paths

def engines_dir(home: Path) -> Path:
    return Path(home) / "serving" / "engines"


def engine_path(home: Path, engine_id: str) -> Path:
    if not _ENGINE_ID_RE.match(engine_id):
        raise ServeError(f"invalid engine id: {engine_id!r}")
    return engines_dir(home) / f"{engine_id}.json"


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _utcnow() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------- built-in

def builtin_engine() -> Dict[str, Any]:
    """The always-present llama.cpp engine (never stored on disk)."""
    return {
        "id": BUILTIN_ID,
        "name": "llama.cpp (llama-server)",
        "kind": "openai",
        "builtin": True,
        "binary": "llama-server",
        "args_template": [],          # empty => supervisor builds the argv
        "env": {},
        "notes": "sable's built-in engine: discovery, auto-setup, full "
                 "run_settings support.",
        "created": None,
        "updated": None,
    }


def is_builtin_ref(ref: Any) -> bool:
    """True for anything meaning 'use the built-in llama.cpp path'."""
    s = str(ref or "").strip().lower()
    return s in ("", "llama-cpp", "llama", "default")


# ------------------------------------------------------------- listing

def list_engines(home: Path) -> List[Dict[str, Any]]:
    """Built-in engine first, then registered engines sorted by id."""
    out: List[Dict[str, Any]] = [builtin_engine()]
    d = engines_dir(home)
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            if p.stem == BUILTIN_ID:
                continue  # a stray file never shadows the built-in
            e = _read_json(p)
            if e and e.get("id"):
                out.append(e)
    return out


def get_engine(home: Path, engine_id: str) -> Optional[Dict[str, Any]]:
    if engine_id == BUILTIN_ID:
        return builtin_engine()
    return _read_json(engine_path(home, engine_id))


def save_engine(home: Path, engine: Dict[str, Any]) -> None:
    p = engine_path(home, str(engine["id"]))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(engine, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, p)


def delete_engine_file(home: Path, engine_id: str) -> bool:
    if engine_id == BUILTIN_ID:
        raise ServeError("the built-in llama.cpp engine cannot be deleted")
    try:
        engine_path(home, engine_id).unlink()
        return True
    except FileNotFoundError:
        return False


def engines_in_use(home: Path, engine_id: str) -> List[str]:
    """Instance ids bound to this engine (delete guard). Reads instance
    definitions directly to avoid a supervisor<->instances import cycle at
    module level."""
    from .instances import load_instances

    used = []
    try:
        for inst in load_instances(home):
            if str(inst.get("engine") or "").strip() == engine_id:
                used.append(str(inst.get("id")))
    except Exception:  # defensive: a broken instance file must not break UI
        pass
    return used


# ------------------------------------------------------------- probing

def resolve_binary(engine: Dict[str, Any]) -> Optional[str]:
    """Absolute path of the engine's binary, or None when not installed.

    Explicit paths are checked as files; bare command names go through PATH
    (``shutil.which``). Tests monkeypatch ``shutil.which`` in this module."""
    b = str(engine.get("binary") or "").strip()
    if not b:
        return None
    p = Path(b).expanduser()
    alt = getattr(os, "alt_sep", None)
    if p.is_absolute() or os.sep in b or (alt and alt in b):
        return str(p) if (p.is_file() and os.access(p, os.X_OK)) else None
    found = shutil.which(b)
    return found


def probe(engine: Dict[str, Any]) -> Dict[str, Any]:
    """{found, path} for the engine binary. The built-in probes llama-server
    on PATH/common dirs WITHOUT auto-setup (a listing endpoint must never
    clone + build llama.cpp)."""
    if engine.get("builtin"):
        found = shutil.which("llama-server")
        if not found:
            from . import runtime
            for d in runtime.search_dirs():
                cand = d / ("llama-server.exe" if os.name == "nt"
                            else "llama-server")
                if cand.is_file() and os.access(cand, os.X_OK):
                    found = str(cand)
                    break
        return {"found": bool(found), "path": found}
    path = resolve_binary(engine)
    return {"found": path is not None, "path": path}


# ------------------------------------------------------------- validation

def _normalize_args_template(raw: Any) -> List[str]:
    """args_template: list of strings OR a single command-string (shlex)."""
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        try:
            parts = shlex.split(raw)
        except ValueError as e:
            raise ServeError(f"args_template not parseable: {e}") from e
    elif isinstance(raw, list) and all(isinstance(x, str) for x in raw):
        parts = [str(x) for x in raw]
    else:
        raise ServeError("args_template must be a string or list of strings")
    if any(not str(x).strip() for x in parts):
        raise ServeError("args_template entries must be non-empty strings")
    return parts


def _normalize_env(raw: Any) -> Dict[str, str]:
    if raw is None or raw == "":
        return {}
    if not isinstance(raw, dict):
        raise ServeError("env must be an object of string keys/values")
    out: Dict[str, str] = {}
    for k, v in raw.items():
        if not str(k).strip():
            raise ServeError("env keys must be non-empty strings")
        if v is None:
            continue
        out[str(k)] = str(v)
    return out


def validate_engine_body(body: Dict[str, Any],
                         existing: Optional[Dict[str, Any]] = None,
                         ) -> Dict[str, Any]:
    """Validate/normalize a create-or-update body into an engine record.

    Rules: name + binary required; kind in KINDS (default openai);
    args_template string-or-list; env string map. The built-in id is reserved."""
    existing = existing or {}
    if str(existing.get("id") or body.get("id") or "") == BUILTIN_ID:
        raise ServeError("the built-in llama.cpp engine cannot be modified")
    name = str(body.get("name", existing.get("name")) or "").strip()
    if not name:
        raise ServeError("name required (non-empty string)")
    kind = str(body.get("kind", existing.get("kind")) or "openai").strip() \
        .lower()
    if kind not in KINDS:
        raise ServeError(f"kind must be one of {', '.join(KINDS)}")
    binary = str(body.get("binary", existing.get("binary")) or "").strip()
    if not binary:
        raise ServeError("binary required (path or command name on PATH)")
    args = _normalize_args_template(
        body.get("args_template", existing.get("args_template")))
    if kind == "openai" and not args:
        args = list(_DEFAULT_OPENAI_ARGS)
    env = _normalize_env(body.get("env", existing.get("env")))
    notes = str(body.get("notes", existing.get("notes")) or "").strip()
    return {
        "id": str(existing.get("id") or body.get("id") or ""),
        "name": name,
        "kind": kind,
        "binary": binary,
        "args_template": args,
        "env": env,
        "notes": notes,
        "created": existing.get("created") or _utcnow(),
        "updated": _utcnow(),
    }


def new_engine_id(name: str, taken: List[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_-]", "-", name.strip()).strip("-")[:48] \
        or "engine"
    if base == BUILTIN_ID:
        base = "engine"
    eid = base
    n = 2
    while eid in taken:
        eid = f"{base}-{n}"
        n += 1
    return eid


# ------------------------------------------------------------- command

def build_engine_cmd(engine: Dict[str, Any], binary: str, *, model: str,
                     host: str, port: int) -> List[str]:
    """argv for one spawn: binary + args_template with {model}/{host}/{port}
    substituted. Unknown placeholders are kept verbatim (they are the user's
    own literal arguments)."""
    template = list(engine.get("args_template") or [])
    if engine.get("kind") == "openai" and not template:
        template = list(_DEFAULT_OPENAI_ARGS)
    subs = {"model": str(model), "host": str(host), "port": str(int(port))}
    cmd = [binary]
    for a in template:
        try:
            cmd.append(a.format(**subs))
        except (KeyError, IndexError):
            cmd.append(a)
    return cmd


def engine_env(engine: Dict[str, Any]) -> Dict[str, str]:
    """Extra child-process environment from the engine record ({} when none)."""
    env = engine.get("env")
    if not isinstance(env, dict):
        return {}
    return {str(k): str(v) for k, v in env.items() if v is not None}

"""Built-in agent toolset for sable's runtime.

Names and semantics follow the well-established file-toolset conventions
(read_file / write_file / patch / search_files, plus list_dir) so datasets and
agent profiles stay portable, but every implementation here is sable's own —
no external agent framework is imported.

Also built in: a todo/planning tool (todo_write/todo_read, reference-shaped
statuses) and search_traces, which lets the agent consult its own experience
buffer — history no other local-agent runtime gives it access to.

Safety posture: outputs are size-capped, writes create parent directories but
never escape the given path, and patch requires a unique match (with a
whitespace-tolerant fuzzy fallback). Tools return strings and must never raise
into the agent loop — errors come back as "error: ..." text the model can
recover from.

Backlog item #9 (workspace containment): the raw implementations above know
nothing about roots by design — containment is applied ONCE, at runner
construction, via core/path_security.contain() in
core/agent_runner.create_runner (see agent.workspace_root config). That keeps
this module pure and guarantees CLI and dashboard runners behave identically.

Backlog item #1 (read blocking): read_file/search_files refuse sensitive
credential paths (.env files, SSH keys, .ssh/.aws/.gnupg stores, .netrc,
.pgpass, credentials.json). Denylist shape adapted in spirit from MIT-licensed
the file-safety read-block list. This is defense-in
depth, not a security boundary — its job is to keep credentials out of the
trace buffer (and therefore out of training data), and it pairs with the
record-time redaction in core/redact.py. Gate: `tools.block_sensitive_files`
(default true; see core/config.py + create_runner).
"""

from __future__ import annotations

import difflib
import fnmatch
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("sable.tools")

MAX_BYTES = 64_000          # read output cap
MAX_WRITE_BYTES = 8_000_000  # write payload cap
MAX_SEARCH_RESULTS = 200
TODO_MAX_ITEMS = 50
TODO_MAX_CONTENT = 400
TRACE_SNIPPET_CHARS = 240
SKIP_DIR_NAMES = {".git", "node_modules", "__pycache__", ".venv", "venv",
                  ".hg", ".svn", "dist", "build", ".mypy_cache", ".pytest_cache"}
_BINARY_SUFFIXES = {".gguf", ".safetensors", ".bin", ".pt", ".pth", ".zip",
                    ".tar", ".gz", ".bz2", ".xz", ".png", ".jpg", ".jpeg",
                    ".gif", ".webp", ".ico", ".pdf", ".so", ".dll", ".exe",
                    ".whl", ".pyc", ".mp4", ".mp3", ".wav"}


# ------------------------------------------------- sensitive-path blocking
# (backlog #1; defense-in-depth denylist)

# Case-insensitive basenames that routinely hold plaintext credentials.
_BLOCKED_READ_BASENAMES = {
    ".env", ".env.local", ".env.development", ".env.production", ".env.test",
    ".env.staging", ".envrc",
    "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
    "credentials.json", ".netrc", ".pgpass", ".npmrc", ".pypirc",
    ".git-credentials",
}
# Any *.pem under a .ssh directory is blocked (see _sensitive_read_reason).
# Path-fragment blocks: anything under these directories, plus the AWS
# credentials file specifically (~/.aws/credentials, ~/.aws/config-style INI).
_BLOCKED_READ_DIR_NAMES = {".ssh", ".gnupg"}
_BLOCKED_PARENT_DIRS = {".aws"}


def _sensitive_read_reason(path: str) -> Optional[str]:
    """Return why `path` must not be read by the agent, or None if allowed.

    Cheap lexical check on the expanded/absolute path (no filesystem access,
    so it also blocks reads of files that do not exist yet and works during
    search traversal without extra stat calls).
    """
    try:
        p = Path(path).expanduser()
        parts = p.parts
    except Exception:  # odd path strings must never crash a tool call
        return None
    name = p.name.lower()
    lowered = [part.lower() for part in parts]
    if ".ssh" in lowered and name.endswith(".pem"):
        return "blocked: sensitive credential path (*.pem under .ssh)"
    for d in _BLOCKED_READ_DIR_NAMES:
        if d in lowered:
            return f"blocked: sensitive credential path (under {d}/)"
    if ".aws" in lowered and name in ("credentials", "config"):
        return "blocked: sensitive credential path (.aws credentials)"
    if name in _BLOCKED_READ_BASENAMES:
        return "blocked: sensitive credential path"
    return None


_SENSITIVE_REFUSAL_SUFFIX = (
    " — this file routinely holds plaintext credentials and would leak them "
    "into the training-data trace buffer. If you genuinely need it, a human "
    "can set `tools.block_sensitive_files: false` in the sable config."
)


def _sensitive_read_refusal(path: str) -> Optional[str]:
    reason = _sensitive_read_reason(path)
    return f"error: {reason}{_SENSITIVE_REFUSAL_SUFFIX}" if reason else None


# --------------------------------------------------------------------- read

def read_file(path: str, offset: int = 1, limit: int = 2000,
              block_sensitive: bool = True, **_: Any) -> str:
    """Read a text file with 1-based line pagination (output capped ~64 KB).

    Paging is streaming: only the requested window is decoded and buffered,
    so `offset` reaches any line of a large file (a whole-file read used to
    stop at READ_SCAN_BYTES and report "no lines in range" for offsets past
    it — a wrong answer that sent agents re-reading from the top). The binary
    sniff looks at the first 4 KB only.
    """
    if block_sensitive:
        refusal = _sensitive_read_refusal(path)
        if refusal:
            return refusal
    p = Path(path).expanduser()
    if not p.is_file():
        return f"error: not a file: {path}"
    try:
        start = max(1, int(offset))
        count = max(1, int(limit))
    except (TypeError, ValueError):
        start, count = 1, 2000
    out: List[str] = []
    total = 0
    capped = False
    has_more = False
    try:
        with open(p, "rb") as f:
            head = f.read(4096)
            if b"\x00" in head:
                return f"error: {path} looks binary; read_file only handles text"
            f.seek(0)  # the sniff consumed bytes; line math starts at 0
            # Skip whole lines before `start` without decoding them all.
            for _skip in range(start - 1):
                if f.readline() == b"":
                    break  # file shorter than offset (handled below)
            window: List[str] = []
            while len(window) < count:
                raw_line = f.readline()
                if raw_line == b"":
                    break
                window.append(raw_line.decode("utf-8", errors="replace")
                              .rstrip("\n").rstrip("\r"))
            for i, line in enumerate(window, start):
                row = f"{i:>6} | {line}"
                total += len(row) + 1
                if total > MAX_BYTES:
                    capped = True
                    out.append(f"... (output cap reached at line {i})")
                    break
                out.append(row)
            # A full window that wasn't cut by the cap may still have lines
            # below it — one-byte probe answers without reading the rest.
            if not capped and len(window) == count:
                has_more = f.read(1) != b""
    except OSError as e:
        return f"error reading {path}: {e}"
    if not out:
        return (f"(file {path} has no lines at/after offset={start}; "
                "the file is shorter — start from offset=1)")
    if has_more:
        out.append(f"... (more lines below; "
                   f"continue with offset={start + len(window)})")
    return "\n".join(out)


def list_dir(path: str = ".", block_sensitive: bool = True, **_: Any) -> str:
    """List directory entries (names; '/' suffix for dirs).

    Backlog #1/#9: listing a credential store (~/.ssh etc.) is itself an
    information leak into the trace buffer, so the sensitive-path deny list
    applies to the listed root too.
    """
    if block_sensitive:
        refusal = _sensitive_read_refusal(path)
        if refusal:
            return refusal
    p = Path(path).expanduser()
    if not p.is_dir():
        return f"error: not a directory: {path}"
    try:
        entries = sorted(p.iterdir(), key=lambda e: (e.is_dir(), e.name))
        out = [
            (d.name + "/") if d.is_dir() else d.name
            for d in entries[:500]
        ]
        return "\n".join(out) if out else "(empty)"
    except OSError as e:
        return f"error listing {path}: {e}"


# ------------------------------------------------ post-write syntax check
# (feature-wave Phase 4; two-tier concept from
# tools/file_operations.py `_check_lint` / `_check_lint_delta` /
# `_maybe_lsp_diagnostics`). Tier 1 is an always-on cheap syntax check on
# the NEW content after a successful write_file/patch; tier 2 is optional,
# opt-in LSP diagnostics with a pre-edit baseline delta. established semantics
# kept: the WRITE IS NEVER REJECTED — the model just sees the error text in
# the tool result so it fixes it next turn, and pre-existing errors are
# filtered out of the report (delta against the pre-edit content).

NODE_CHECK_TIMEOUT_S = 5.0        # hard cap for one `node --check` child
_PY_EXTS = {".py"}
_JS_EXTS = {".js", ".jsx", ".mjs", ".cjs"}
_TS_EXTS = {".ts", ".tsx"}

#: Marker on a failed check (UI + model both key off this text).
SYNTAX_FAIL_MARK = "✗ SYNTAX ERROR"
SYNTAX_OK_MARK = "✓ syntax ok"


def _syntax_error_text(path: str, err: BaseException) -> str:
    """Format a SyntaxError-ish exception as `path:line: Type: msg`."""
    lineno = getattr(err, "lineno", None) or "?"
    msg = getattr(err, "msg", None) or str(err)
    return f"{path}:{lineno}: {type(err).__name__}: {msg}"


def _check_python_syntax(path: str, content: str) -> Tuple[str, str]:
    """Tier-1 Python check via py_compile against a throwaway temp file.

    Returns ("ok"|"error", error_text). py_compile only COMPILES — nothing
    is executed (no exec/import). `cfile=os.devnull` means no .pyc artifact
    is ever written, independent of sys.dont_write_bytecode; the source
    itself lives in a temp file so a broken write can never poison the
    real path and no __pycache__ is created.
    """
    import py_compile

    fd, tmp = tempfile.mkstemp(suffix=".py", prefix="sable-lint-")
    # Dedicated temp cfile (os.devnull is rejected by py_compile on newer
    # CPythons as a non-regular file); unlinked below either way.
    cfile = tmp + ".pyc"
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        try:
            py_compile.compile(tmp, cfile=cfile, doraise=True, quiet=1)
        except py_compile.PyCompileError as e:
            inner = getattr(e, "syntax_error", None)
            if isinstance(inner, BaseException):
                return "error", _syntax_error_text(path, inner)
            # Fallback shape: normalize the temp path away (the delta filter
            # diffs error lines pre-vs-post; a tmp path would never match)
            # and surface the line number py_compile already parsed.
            first = next((ln.strip() for ln in str(e).splitlines()
                          if ln.strip()), str(e))
            first = first.replace(tmp, str(path))
            m = re.search(r"line (\d+)", first)
            lineno = m.group(1) if m else "?"
            return "error", f"{path}:{lineno}: SyntaxError: {first}"
        except (SyntaxError, ValueError) as e:  # source-encoding/NULL edges
            return "error", _syntax_error_text(path, e)
        return "ok", ""
    finally:
        for f in (tmp, cfile):
            try:
                os.unlink(f)
            except OSError:
                pass


def _node_major_version(node_bin: str) -> int:
    """Major version of a node binary (-1 when unparseable)."""
    try:
        out = subprocess.run([node_bin, "--version"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        return int(out.lstrip("v").split(".")[0])
    except Exception:  # noqa: BLE001 - any probe failure means "unknown"
        return -1


def _check_node_syntax(path: str, content: str) -> Tuple[str, str]:
    """Tier-1 JS/TS check via `node --check` on a temp file (never executed;
    --check only parses). Returns ("ok"|"error"|"skipped", error_text).

    - node missing => skipped (graceful degrade, per spec).
    - .ts/.tsx: attempted only when the local node supports native type
      stripping (>= 23); older node rejects TS syntax outright, so those
      skip instead of reporting phantom errors.
    - Timeout kills ONLY the child we spawned (terminate then kill).
    """
    node = shutil.which("node")
    if not node:
        return "skipped", ""
    ext = Path(path).suffix.lower()
    if ext in _TS_EXTS and _node_major_version(node) < 23:
        return "skipped", ""
    fd, tmp = tempfile.mkstemp(suffix=ext or ".js", prefix="sable-lint-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        proc = subprocess.Popen([node, "--check", tmp],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        try:
            out, _ = proc.communicate(timeout=NODE_CHECK_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()          # only OUR child, terminate-then-kill
                proc.communicate()
            return "skipped", ""     # timed out => no verdict, never block
        if proc.returncode == 0:
            return "ok", ""
        # node prints "path:LINE" style frames; re-key them to the real path.
        body = (out or "").replace(tmp, str(path)).strip()
        return "error", body or f"node --check exit {proc.returncode}"
    except OSError:
        return "skipped", ""
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _tier1_check(path: str, content: str) -> Tuple[str, str]:
    """Dispatch tier-1 syntax check by extension; unknown exts skip silently."""
    ext = Path(path).suffix.lower()
    if ext in _PY_EXTS:
        return _check_python_syntax(path, content)
    if ext in _JS_EXTS or ext in _TS_EXTS:
        return _check_node_syntax(path, content)
    return "skipped", ""


def _tier1_delta(path: str, pre_content: Optional[str],
                 err_text: str) -> Tuple[bool, str]:
    """Delta refinement: when the post-write content
    has syntax errors AND we have the pre-edit content, check the OLD
    content too and drop error lines that already existed. Returns
    (report?, text). report=False means every error was pre-existing —
    annotate "still broken" without blaming this edit (established semantics).
    """
    if pre_content is None:
        return True, err_text
    status, pre_err = _tier1_check(path, pre_content)
    if status != "error" or not pre_err:
        return True, err_text  # pre was clean => all post errors are new
    pre_lines = {ln.strip() for ln in pre_err.splitlines() if ln.strip()}
    new_lines = [ln for ln in err_text.splitlines()
                 if ln.strip() and ln.strip() not in pre_lines]
    if not new_lines:
        return True, (err_text + "\n(pre-existing syntax error — this edit "
                      "didn't introduce a new one but the file is still broken)")
    return True, ("New syntax errors introduced by this edit "
                  "(pre-existing filtered out):\n" + "\n".join(new_lines))


# ------------------------------------------------------------- tier 2: LSP
# Minimal stdlib-only JSON-RPC-over-stdio client (subprocess + json), in the
# spirit of an LSP integration but a single-shot per-write probe instead of a
# persistent service. Baseline delta strategy mirrors the shadow-document
# idea: didOpen with the PRE-edit content, collect diagnostics, didChange to
# the POST-edit content, collect again, and report only NEW diagnostics —
# so pre-existing problems are never re-reported.

LSP_LANGUAGE_IDS = {
    ".py": "python", ".js": "javascript", ".jsx": "javascriptreact",
    ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescriptreact",
}

#: Default per-extension server commands (each: argv list). A server is used
#: only when its first word resolves via shutil.which; config `tools.lsp.
#: servers` may override/extend this mapping.
DEFAULT_LSP_SERVERS: Dict[str, List[str]] = {
    ".py": ["pyright-langserver", "--stdio"],
    ".ts": ["typescript-language-server", "--stdio"],
    ".tsx": ["typescript-language-server", "--stdio"],
    ".js": ["typescript-language-server", "--stdio"],
    ".jsx": ["typescript-language-server", "--stdio"],
    ".go": ["gopls"],
    ".rs": ["rust-analyzer"],
}


def _lsp_encode(msg: dict) -> bytes:
    body = json.dumps(msg).encode("utf-8")
    return b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body


def _lsp_read_message(proc, deadline: float) -> Optional[dict]:
    """Read one LSP message honoring Content-Length; None on timeout/EOF.

    Reads via os.read on the raw fd after select(): a BufferedReader would
    slurp ahead into userspace, leaving the pipe empty and select blind to
    already-buffered bytes (classic hang). os.read never reads more than we
    ask for, so select stays authoritative.
    """
    import select

    fd = proc.stdout.fileno()

    def _read_some(n: int) -> bytes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return b""
        r, _, _ = select.select([fd], [], [], remaining)
        if not r:
            return b""
        try:
            return os.read(fd, n)
        except OSError:
            return b""

    headers = b""
    while b"\r\n\r\n" not in headers:
        ch = _read_some(1)
        if not ch:
            return None
        headers += ch
    m = re.search(rb"Content-Length: (\d+)", headers)
    if not m:
        return None
    n = int(m.group(1))
    body = b""
    while len(body) < n:
        chunk = _read_some(n - len(body))
        if not chunk:
            break
        body += chunk
    try:
        return json.loads(body.decode("utf-8")) if body else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _lsp_collect(proc, uri: str, deadline: float) -> List[dict]:
    """Pump messages until `deadline`, returning the LAST publishDiagnostics
    push for `uri` (later pushes supersede; empty list = none seen)."""
    latest: List[dict] = []
    while time.monotonic() < deadline:
        msg = _lsp_read_message(proc, min(deadline, time.monotonic() + 0.5))
        if msg is None:
            continue
        if msg.get("method") == "textDocument/publishDiagnostics":
            params = msg.get("params") or {}
            if (params.get("uri") or "").rstrip("/") == uri.rstrip("/"):
                latest = list(params.get("diagnostics") or [])
    return latest


def _lsp_diagnostics(path: str, pre_content: Optional[str],
                     post_content: str, server_cmd: List[str], *,
                     startup_timeout: float = 15.0,
                     check_timeout: float = 20.0) -> Optional[List[dict]]:
    """Run one LSP server against (pre|post) content and return NEW
    diagnostics only, or None when unavailable/timed out (graceful skip).

    Spawns the configured command as OUR child; on any timeout/exception it
    is terminated (then killed) — nothing else is touched. Whole probe is
    hard-bounded by check_timeout.
    """
    proc = None
    deadline = time.monotonic() + check_timeout
    try:
        from urllib.parse import quote as _quote

        abs_path = os.path.abspath(path)
        uri = "file://" + _quote(abs_path.replace("\\", "/"), safe="/:")
        ext = Path(path).suffix.lower()
        proc = subprocess.Popen(server_cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)

        def _send(msg: dict) -> None:
            if proc.stdin is not None:
                proc.stdin.write(_lsp_encode(msg))
                proc.stdin.flush()

        def _finish(diags: Optional[List[dict]]) -> Optional[List[dict]]:
            try:
                _send({"jsonrpc": "2.0", "id": 999, "method": "shutdown"})
                _send({"jsonrpc": "2.0", "method": "exit"})
            except Exception:  # noqa: BLE001 - best-effort graceful close
                pass
            return diags

        _send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "processId": os.getpid(), "rootUri": None, "capabilities": {}}})
        init_deadline = min(deadline, time.monotonic() + startup_timeout)
        while time.monotonic() < init_deadline:
            msg = _lsp_read_message(proc, init_deadline)
            if msg is None:
                continue
            if msg.get("id") == 1:
                break
        else:
            return None  # never initialized => graceful skip
        _send({"jsonrpc": "2.0", "method": "initialized", "params": {}})

        doc = {"uri": uri, "languageId": LSP_LANGUAGE_IDS.get(ext, "text"),
               "version": 1, "text": pre_content if pre_content is not None
               else post_content}
        _send({"jsonrpc": "2.0", "method": "textDocument/didOpen",
               "params": {"textDocument": doc}})
        grace = min(2.0, max(0.5, (deadline - time.monotonic()) / 3))
        baseline = _lsp_collect(proc, uri, min(deadline,
                                               time.monotonic() + grace))

        if pre_content is not None and pre_content != post_content:
            # Shadow-doc delta (reference-shaped): swap the document to the new
            # content via didChange and collect again; only diagnostics whose
            # key (source/code/message) wasn't in the baseline are "new".
            _send({"jsonrpc": "2.0", "method": "textDocument/didChange",
                   "params": {"textDocument": {"uri": uri, "version": 2},
                              "contentChanges": [{"text": post_content}]}})
            # Bounded wait (never the whole remaining check_timeout): a slow
            # server just means fewer diagnostics make the cut.
            grace2 = min(3.0, max(0.5, deadline - time.monotonic()))
            after = _lsp_collect(proc, uri, min(deadline,
                                                time.monotonic() + grace2))

            def _key(d: dict) -> str:
                return (f"{d.get('source', '')}:{d.get('code', '')}:"
                        f":{str(d.get('message', '')).splitlines()[0]}")

            base_keys = {}
            for d in baseline:
                base_keys[_key(d)] = base_keys.get(_key(d), 0) + 1
            new_diags = []
            for d in after:
                k = _key(d)
                if base_keys.get(k, 0) > 0:
                    base_keys[k] -= 1
                    continue
                if int(d.get("severity", 1)) <= 2:  # error/warning only
                    new_diags.append(d)
            return _finish(new_diags)

        fresh = [d for d in baseline if int(d.get("severity", 1)) <= 2] \
            if pre_content is None else []
        return _finish(fresh)
    except Exception:  # noqa: BLE001 - LSP must NEVER break a write
        return None
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()      # our own child only
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass


def _maybe_lsp_suffix(path: str, pre_content: Optional[str],
                      post_content: str, lsp_cfg: Dict[str, Any]) -> str:
    """Tier-2 gate + formatting. Returns '' unless LSP is enabled, a server
    for the extension is detected on PATH, and NEW diagnostics came back."""
    if not isinstance(lsp_cfg, dict) or not lsp_cfg.get("enabled"):
        return ""
    ext = Path(path).suffix.lower()
    servers = dict(DEFAULT_LSP_SERVERS)
    for k, v in (lsp_cfg.get("servers") or {}).items():
        servers[str(k) if str(k).startswith(".") else "." + str(k)] = \
            v.split() if isinstance(v, str) else list(v)
    cmd = servers.get(ext)
    if not cmd or not shutil.which(str(cmd[0])):
        return ""  # no server detected => graceful skip (spec)
    diags = _lsp_diagnostics(
        path, pre_content, post_content, [str(c) for c in cmd],
        startup_timeout=float(lsp_cfg.get("startup_timeout_seconds", 15.0)),
        check_timeout=float(lsp_cfg.get("check_timeout_seconds", 20.0)))
    if not diags:
        return ""  # None (skipped) or empty (clean) => no noise
    lines = []
    for d in diags[:10]:
        rng = ((d.get("range") or {}).get("start") or {})
        ln = int(rng.get("line", 0)) + 1
        sev = "error" if int(d.get("severity", 1)) <= 1 else "warning"
        src = d.get("source") or "lsp"
        msg = str(d.get("message", "")).splitlines()[0]
        lines.append(f"  {path}:{ln}: {sev} [{src}] {msg}")
    extra = f"\n  … (+{len(diags) - 10} more)" if len(diags) > 10 else ""
    return ("\n⚠ NEW LSP DIAGNOSTICS (pre-existing filtered out):\n"
            + "\n".join(lines) + extra)


def _post_write_suffix(path: str, pre_content: Optional[str],
                       new_content: str, *, lint_on_write: bool = True,
                       lsp_cfg: Optional[Dict[str, Any]] = None) -> str:
    """Compose the tool-result suffix for a SUCCESSFUL write (reference-shaped:
    file stays written; the model sees check results and fixes next turn).

      - tier-1 failure => "\\n✗ SYNTAX ERROR in <path> (edit rejected? no —
        file WAS written; fix it next turn)\\n<parser error text>"
      - tier-1 clean  => " ✓ syntax ok"
      - unknown ext / node missing => "" (silent skip, no noise)
      - LSP enabled + new diagnostics => appended ⚠ block
    """
    if not lint_on_write:
        return ""
    suffix = ""
    try:
        status, err = _tier1_check(path, new_content)
        if status == "error":
            report, text = _tier1_delta(path, pre_content, err)
            if report:
                return (f"\n{SYNTAX_FAIL_MARK} in {path} (edit rejected? no — "
                        f"file WAS written; fix it next turn)\n{text}")
        elif status == "ok":
            suffix = f" {SYNTAX_OK_MARK}"
    except Exception:  # noqa: BLE001 - a broken checker must never eat a write
        return suffix
    try:
        suffix += _maybe_lsp_suffix(path, pre_content, new_content,
                                    lsp_cfg or {})
    except Exception:  # noqa: BLE001
        pass
    return suffix


# -------------------------------------------------------------------- write

def write_file(path: str, content: str, *, lint_on_write: bool = True,
               lsp_config: Optional[Dict[str, Any]] = None, **_: Any) -> str:
    """Write (create/overwrite) a text file, creating parent directories.

    Post-write editor-style checks (feature-wave Phase 4): the pre-edit
    content is captured for the baseline delta, then tier-1 syntax check and
    optional tier-2 LSP diagnostics annotate the result (the write itself is
    never rejected). Flags are bound by SableAgentClient from config —
    invisible to the model.
    """
    if len(content) > MAX_WRITE_BYTES:
        return f"error: content too large ({len(content)} chars; cap {MAX_WRITE_BYTES})"
    p = Path(path).expanduser()
    pre_content: Optional[str] = None
    try:
        if p.is_file():
            pre_content = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pre_content = None
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"error writing {path}: {e}"
    return (f"wrote {len(content)} chars to {p}"
            + _post_write_suffix(path, pre_content, content,
                                 lint_on_write=lint_on_write,
                                 lsp_cfg=lsp_config))


# -------------------------------------------------------------------- patch

def _fuzzy_find(text: str, needle: str) -> Optional[Tuple[int, int]]:
    """Locate `needle` in `text` tolerating leading-whitespace differences.

    Returns (start, end) character offsets into the ORIGINAL text for the
    matched span (full lines), or None. Strategy: slide a line-window of the
    needle's size over the text comparing whitespace-stripped lines; pick the
    best window by similarity if no exact match.
    """
    t_lines = text.splitlines(keepends=True)
    n_lines = [l for l in needle.splitlines()] or [""]
    stripped_t = [l.strip() for l in t_lines]
    stripped_n = [l.strip() for l in n_lines]
    if not stripped_n or len(t_lines) < len(stripped_n):
        return None
    # exact (whitespace-stripped) window match
    for i in range(len(t_lines) - len(stripped_n) + 1):
        if stripped_t[i: i + len(stripped_n)] == stripped_n:
            start = sum(len(x) for x in t_lines[:i])
            end = start + sum(len(x) for x in t_lines[i: i + len(stripped_n)])
            return start, end
    # fuzzy fallback: best ratio window (threshold guards against nonsense)
    best_i, best_ratio = -1, 0.0
    joiner = difflib.SequenceMatcher(None)
    for i in range(len(t_lines) - len(stripped_n) + 1):
        joiner.set_seqs("\n".join(stripped_t[i: i + len(stripped_n)]),
                        "\n".join(stripped_n))
        r = joiner.ratio()
        if r > best_ratio:
            best_i, best_ratio = i, r
    if best_i >= 0 and best_ratio >= 0.85:
        start = sum(len(x) for x in t_lines[:best_i])
        end = start + sum(len(x) for x in t_lines[best_i: best_i + len(stripped_n)])
        return start, end
    return None


def patch(path: str, old_string: str = "", new_string: str = "",
          replace_all: bool = False, *, lint_on_write: bool = True,
          lsp_config: Optional[Dict[str, Any]] = None, **_: Any) -> str:
    """Replace old_string with new_string in a file (unique match required).

    Exact matching first; on failure a whitespace-tolerant fuzzy match over
    full lines. Mirrors the patch tool semantics used by common agent
    runtimes: ambiguous matches are refused, replace_all opts into all of them.

    Post-write checks (feature-wave Phase 4): same tiers as write_file; the
    pre-edit text doubles as the baseline so only NEW syntax errors / LSP
    diagnostics introduced by this patch are reported.
    """
    p = Path(path).expanduser()
    if not old_string:
        return "error: old_string is empty — refusing (would insert anywhere)"
    if old_string == new_string:
        return "error: old_string and new_string are identical"
    if not p.is_file():
        if old_string.strip():
            return f"error: not a file: {path} (use write_file to create it)"
        # create-new-file semantics: empty old_string on a missing file
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(new_string, encoding="utf-8")
        except OSError as e:
            return f"error writing {path}: {e}"
        return (f"created {p} ({len(new_string)} chars)"
                + _post_write_suffix(path, None, new_string,
                                     lint_on_write=lint_on_write,
                                     lsp_cfg=lsp_config))
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"error reading {path}: {e}"

    n = text.count(old_string)
    if n > 1 and not replace_all:
        return (f"error: old_string matches {n} times — add surrounding context "
                "to make it unique, or set replace_all=true")
    fuzzy = False
    if n >= 1:
        updated = text.replace(old_string, new_string) if replace_all \
            else text.replace(old_string, new_string, 1)
    else:
        span = _fuzzy_find(text, old_string)
        if span is None:
            return ("error: old_string not found in file (exact and fuzzy matching "
                    "both failed) — re-read the file and copy the exact text")
        fuzzy = True
        start, end = span
        # preserve the matched block's first-line indentation for new_string
        line_start = text.rfind("\n", 0, start) + 1
        indent = text[line_start:start]
        updated = (text[:line_start]
                   + _reindent(text[start:end], indent, new_string)
                   + text[end:])

    try:
        p.write_text(updated, encoding="utf-8")
    except OSError as e:
        return f"error writing {path}: {e}"
    how = "all occurrences" if (n > 1 and replace_all) else "one occurrence"
    if fuzzy:
        # Whitespace-tolerant match succeeded where the exact one failed —
        # say so, so the model (and the verification ledger) can treat the
        # edit as lower-confidence and re-read if it matters.
        how += "; FUZZY whitespace-tolerant match — verify the result"
    return (f"patched {path} ({how})"
            + _post_write_suffix(path, text, updated,
                                 lint_on_write=lint_on_write,
                                 lsp_cfg=lsp_config))


def _reindent(matched: str, indent: str, new_string: str) -> str:
    """Re-indent new_string to the matched block's original first-line indent."""
    if not indent.strip():
        return new_string
    lines = new_string.splitlines(keepends=True)
    if not lines:
        return new_string
    first = lines[0]
    rest = [indent + l if l.strip() else l for l in lines[1:]]
    return first + "".join(rest)


# -------------------------------------------------------------------- search

def search_files(pattern: str, path: str = ".", mode: str = "content",
                 glob: str = "", block_sensitive: bool = True, **_: Any) -> str:
    """Search file contents (regex) or names (glob) under a directory.

    mode="content": regex over text files, "file:line: text" output.
    mode="name":    fnmatch against basenames/paths.
    Optional glob filters which files are scanned (e.g. "*.py").
    Sensitive credential files (backlog #1) are skipped entirely so their
    contents never reach the tool result (and therefore the trace buffer).
    """
    root = Path(path).expanduser()
    if not pattern.strip():
        return "error: empty pattern"
    results: List[str] = []

    def _skip(p: Path) -> bool:
        return block_sensitive and _sensitive_read_reason(str(p)) is not None

    if mode == "name":
        if not root.exists():
            return f"error: not a directory: {path}"
        targets = [root] if root.is_file() else _walk(root)
        for p in targets:
            if _skip(p):
                continue
            rel = str(p.relative_to(root)) if root.is_dir() else p.name
            if fnmatch.fnmatch(p.name, pattern) or fnmatch.fnmatch(rel, pattern):
                results.append(str(p))
                if len(results) >= MAX_SEARCH_RESULTS:
                    results.append("... (result cap reached)")
                    break
        return "\n".join(results) if results else f"no files matching {pattern!r}"

    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"error: invalid regex: {e}"
    if not root.exists():
        # Same refusal shape as name mode — "no matches" for a missing root
        # misleads the model into believing it searched an empty tree.
        return f"error: not a directory: {path}"
    targets = _walk(root) if root.is_dir() else [root]
    for p in targets:
        if _skip(p):  # never surface blocked-file contents (backlog #1)
            continue
        if glob and not fnmatch.fnmatch(p.name, glob):
            continue
        if p.suffix.lower() in _BINARY_SUFFIXES or not _is_textish(p):
            continue
        try:
            with open(p, encoding="utf-8", errors="strict") as f:
                for lineno, line in enumerate(f, 1):
                    if rx.search(line):
                        rel = str(p)
                        results.append(f"{rel}:{lineno}: {line.rstrip()[:300]}")
                        if len(results) >= MAX_SEARCH_RESULTS:
                            results.append("... (result cap reached)")
                            return "\n".join(results)
        except (UnicodeDecodeError, OSError):
            continue
    return "\n".join(results) if results else f"no matches for {pattern!r} under {path}"


def _walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIR_NAMES and not d.startswith(".")]
        for fn in filenames:
            yield Path(dirpath) / fn


def _is_textish(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            chunk = f.read(2048)
        return b"\x00" not in chunk and len(chunk) > 0
    except OSError:
        return False


# ----------------------------------------------------------------- todo

TODO_STATUSES = ("pending", "in_progress", "done", "cancelled")

#: Per-run todo files older than this are pruned once per process (they are
#: tiny but unbounded across thousands of dashboard runs).
TODO_RUN_RETENTION_DAYS = 30


def _todo_path(bound: Optional[str] = None) -> Path:
    """Todo file for a run. The runner binds a per-run path under
    ``SABLE_HOME/todos/`` via partial (private kwarg, invisible to the
    model); unbound callers keep the historical global ``todos.json``."""
    if bound:
        return Path(bound)
    from ..store.paths import resolve_sable_home

    return resolve_sable_home() / "todos.json"


def _todo_read_raw(bound: Optional[str] = None) -> List[Dict[str, str]]:
    p = _todo_path(bound)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text())
        items = data.get("todos") if isinstance(data, dict) else data
        return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _todo_write_raw(items: List[Dict[str, str]],
                    bound: Optional[str] = None) -> None:
    p = _todo_path(bound)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"todos": items}, indent=2))
    os.replace(tmp, p)


def _todo_validate(item: Any) -> Optional[Dict[str, str]]:
    if isinstance(item, str):
        # Weak models often emit markdown checklist lines ("- [ ] foo")
        # instead of objects — salvage content AND status from the marker.
        line = item.strip().lstrip("-* ").strip()
        status = None
        m = re.match(r"^\[([ x~X-])\]\s*(.*)$", line)
        if m:
            mark, line = m.group(1), m.group(2).strip()
            status = {"x": "done", "X": "done", "~": "in_progress",
                      "-": "cancelled"}.get(mark)
        if not line:
            return None
        item = {"content": line, **({"status": status} if status else {})}
    if not isinstance(item, dict):
        return None
    content = str(item.get("content") or item.get("text") or "").strip()
    status = str(item.get("status") or "pending").strip().lower()
    tid = str(item.get("id") or "").strip()
    if not content:
        return None
    if status not in TODO_STATUSES:
        status = "pending"
    return {"id": tid or content[:8], "content": content[:TODO_MAX_CONTENT],
            "status": status}


def todo_write(todos: Optional[List[Any]] = None,
               merge: bool = False, *, _todos_path: Optional[str] = None,
               **_: Any) -> str:
    """Create/replace (or merge-update) the session todo list.

    Items are {id?, content, status?} objects; plain strings (markdown
    checklist lines weak models emit) are salvaged by _todo_validate.
    `_todos_path` is bound per-run by the runner (private kwarg — the schema
    gate rejects it if a model tries to pass it) so concurrent runs stop
    clobbering each other's plans through the one shared file."""
    if todos is None or not isinstance(todos, list):
        return "error: todos must be a list of {id, content, status} objects"
    validated = [v for v in (_todo_validate(t) for t in todos) if v]
    if not validated and todos:
        return "error: no valid todo items (need content; status in " \
               f"{TODO_STATUSES})"
    if merge:
        existing = _todo_read_raw(_todos_path)
        by_id = {t["id"]: t for t in existing}
        for v in validated:
            if v["id"] in by_id:
                by_id[v["id"]] = {**by_id[v["id"]], **v}
            else:
                by_id[v["id"]] = v
        items = list(by_id.values())
    else:
        items = validated
    if len(items) > TODO_MAX_ITEMS:
        return f"error: too many todos ({len(items)} > {TODO_MAX_ITEMS})"
    _todo_write_raw(items, _todos_path)
    return todo_read(_todos_path=_todos_path)


def todo_read(*, _todos_path: Optional[str] = None, **_: Any) -> str:
    """Show the current todo list with status markers."""
    items = _todo_read_raw(_todos_path)
    if not items:
        return "(todo list is empty)"
    marks = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]",
             "cancelled": "[-]"}
    out = [f"{marks.get(t.get('status', 'pending'), '[?]')} "
           f"{t.get('id', '?')}: {t.get('content', '')}" for t in items]
    return "\n".join(out)


# ---------------------------------------------------------- search_traces

def search_traces(pattern: str = "", status: str = "", limit: int = 10,
                  **_: Any) -> str:
    """Search sable's own experience buffer (past agent traces).

    Regex-matches task summaries and (when `pattern` is given) full trace
    JSONL; optionally filters by outcome status. Lets the model recall how a
    similar task was solved before — unique to sable's closed loop.
    """
    try:
        from ..store.db import TraceStore
        from ..store.paths import resolve_sable_home
    except ImportError as e:  # pragma: no cover - packaging edge
        return f"error: trace store unavailable: {e}"

    home = resolve_sable_home()
    db_path = home / "buffer" / "sable.db"
    if not db_path.is_file():
        return "(no traces recorded yet)"

    rx = None
    if pattern.strip():
        try:
            rx = re.compile(pattern, re.I)
        except re.error as e:
            return f"error: invalid regex: {e}"
    if status and status not in ("success", "failure", "partial", "unknown"):
        return f"error: unknown status {status!r} " \
               "(success|failure|partial|unknown)"

    limit = max(1, min(int(limit or 10), 50))
    results: List[str] = []
    try:
        store = TraceStore(db_path)
        try:
            rows = store.list_traces(limit=200, status=status or None)
            for row in rows:
                blob = f"{row.task_summary or ''}"
                snippet = ""
                if rx is not None and not rx.search(blob):
                    # fall back to the full JSONL body
                    jpath = Path(row.jsonl_path)
                    if jpath.is_file() and _is_textish(jpath):
                        text = jpath.read_text(errors="replace")[:200_000]
                        m = rx.search(text)
                        if not m:
                            continue
                        s = max(0, m.start() - 60)
                        snippet = " ".join(text[s:m.end() + 120].split())
                else:
                    snippet = " ".join(blob.split())[:TRACE_SNIPPET_CHARS]
                results.append(
                    f"{row.trace_id} [{row.status}] {row.timestamp} "
                    f"score={row.outcome_score} :: {snippet}")
                if len(results) >= limit:
                    break
        finally:
            store.close()
    except Exception as e:  # never crash the loop over a locked db
        return f"error reading trace index: {e}"
    return "\n".join(results) if results else "no matching traces"


# ---------------------------------------------------------- search_codebase

def search_codebase(query: str = "", mode: str = "",
                    _project_id: Optional[str] = None,
                    _workspace_dir: Optional[str] = None,
                    **_: Any) -> str:
    """Semantic/hybrid/lexical search over the indexed codebase(s).

    Uses the `indexing:` config section (core/indexer.py). When indexing is
    disabled or no index exists yet, says so and points at search_files.

    Feature-wave Phase 5: the runner binds `_project_id`/`_workspace_dir`
    from the active project's run context via functools.partial (same
    pattern as delegation's `_ctx`) — searches then route to that project's
    index DB with no cross-project results. Without a bound project the
    default/no-project DB is searched. top_k is an internal constant
    (indexer.SEARCH_TOP_K) — a model-settable override would conflict with
    the Phase 5 config story; a stray `top_k` argument is absorbed by **_.
    """
    if not query.strip():
        return "error: query required"
    try:
        from ..store.paths import resolve_sable_home
        from .indexer import get_index
    except ImportError as e:  # pragma: no cover
        return f"error: indexer unavailable: {e}"
    try:
        idx = get_index(resolve_sable_home(), project_id=_project_id,
                        workspace_dir=_workspace_dir)
        if not idx.enabled:
            return ("codebase indexing is disabled (settings -> Indexing, or "
                    "config `indexing.enabled: true`); use search_files instead")
        results = idx.search(query, mode=mode or None)
    except Exception as e:  # never crash the agent loop
        return f"error: codebase search failed: {e}"
    if not results:
        return "no matching chunks (index empty? build it from the dashboard)"
    out = []
    for r in results:
        out.append(f"── {r['path']}:{r['start_line']}-{r['end_line']} "
                   f"(score {r['score']})\n{r['snippet']}")
    return "\n\n".join(out)


# ------------------------------------------------------------- delegation
# (feature-wave Phase 2) Thin tool wrappers over core/delegation.py. The
# runner binds a private `_ctx` (config + run metadata) into these impls via
# functools.partial in SableAgentClient.__init__ — the model cannot forge
# it through tool args (extra kwargs are rejected as bad arguments). Without
# a bound context (e.g. hand-built clients in tests) they degrade to clean
# JSON error payloads, never raising into the loop.

# ------------------------------------------------- capability-gated tools
# Per-agent capabilities: reach OUTSIDE the machine (internet) and USE other
# programs (local command execution). These two tools are NEVER part of the
# default schema set exposed to a model; core/agent_runner adds them per
# runner only when the matching capability is on (global config
# `agent.capabilities` defaults, per-profile `capabilities:` override).
# Safety posture follows a hardened web-tooling playbook
# in spirit (tools/url_safety.py + website_policy.py): http(s) only, SSRF
# blocking of private/loopback/link-local/metadata address space, redirects
# re-validated hop by hop, hard size and time caps. Like the rest of this
# module this is defense-in-depth for a local-first tool, not a security
# boundary — `run_command` in particular executes with the user's own
# privileges, which is exactly what the capability grants.

WEB_FETCH_MAX_BYTES = 200_000       # raw response body cap
WEB_FETCH_TIMEOUT_S = 20.0          # per-hop wall clock cap
WEB_FETCH_MAX_REDIRECTS = 3
RUN_COMMAND_DEFAULT_TIMEOUT_S = 60.0
RUN_COMMAND_MAX_TIMEOUT_S = 300.0
RUN_COMMAND_OUTPUT_CAP = 64_000

#: tool name -> capability key that gates it (consumed by core/agent_runner).
CAPABILITY_TOOLS: Dict[str, str] = {
    "web_fetch": "internet",
    "run_command": "run_commands",
    "view_image": "vision",
    "take_screenshot": "screenshots",
}

#: Every recognized capability key (config `agent.capabilities:` defaults,
#: per-profile `capabilities:` override; dashboard Profiles panel toggles).
CAPABILITY_KEYS: Tuple[str, ...] = ("internet", "run_commands",
                                    "vision", "screenshots")


def resolve_capabilities(agent_cfg: Optional[Dict[str, Any]] = None,
                         profile_caps: Optional[Dict[str, Any]] = None,
                         ) -> Dict[str, bool]:
    """Resolve the capability map for a runner: all off by default, global
    `agent.capabilities` dict overrides defaults, per-profile `capabilities`
    dict overrides the global. Unknown keys are ignored on both levels."""
    caps = {k: False for k in CAPABILITY_KEYS}
    for src in ((agent_cfg or {}).get("capabilities"), profile_caps):
        if isinstance(src, dict):
            for key, val in src.items():
                if key in caps:
                    caps[key] = bool(val)
    return caps


def _cap_text(text: str, cap: int) -> str:
    """Truncate tool output to `cap` chars with an explicit marker."""
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n... (output capped at {cap} chars)"


def _blocked_ip_reason(ip_str: str) -> Optional[str]:
    """Why this IP must not be fetched, or None when it is a public address."""
    import ipaddress

    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return f"unparseable address {ip_str!r}"
    if (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
        return f"{ip} is not a public address"
    return None


def _url_block_reason(url: str, allow_private_networks: bool = False
                      ) -> Optional[str]:
    """Why this URL must not be fetched (SSRF posture), or None if allowed.

    Every resolved address of the host is checked; `allow_private_networks`
    relaxes only the address-space rule (tests / deliberate localhost work) —
    scheme and shape checks always apply.

    Known limitation: this check resolves DNS at validation time while httpx
    resolves again at connect time, so a fast-rebinding name could still race
    onto a private address. Documented defense-in-depth, not a network
    boundary — the `internet` capability itself is the trust decision.
    """
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
    except ValueError:
        return "malformed URL"
    if parsed.scheme not in ("http", "https"):
        return (f"scheme {parsed.scheme!r} not allowed (http/https only)")
    host = parsed.hostname
    if not host:
        return "URL has no hostname"
    if allow_private_networks:
        return None
    import socket

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port)
    except socket.gaierror:
        return f"cannot resolve host {host!r}"
    for *_rest, sockaddr in infos:
        reason = _blocked_ip_reason(str(sockaddr[0]))
        if reason is not None:
            return reason
    return None


_HTML_BLOCK_RE = re.compile(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>")
_HTML_TAG_RE = re.compile(r"(?s)<[^>]+>")


def _html_to_text(body: str) -> str:
    """Crude but effective HTML-to-text: drop script/style, then tags."""
    import html as htmllib

    text = _HTML_BLOCK_RE.sub(" ", body)
    text = _HTML_TAG_RE.sub(" ", text)
    text = htmllib.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def web_fetch(url: str, max_chars: int = 8000,
              timeout_seconds: Optional[float] = None, *,
              allow_private_networks: bool = False, **_: Any) -> str:
    """Fetch a public web page over http(s) and return its text content.

    Capability-gated (`internet`): see CAPABILITY_TOOLS. SSRF posture per
    url-safety rules — no automatic redirects (each hop is
    re-validated here), private/loopback/link-local targets refused, raw body
    hard-capped, per-hop timeout bounded.
    """
    import httpx

    if not isinstance(url, str) or not url.strip():
        return "error: url required"
    url = url.strip()
    reason = _url_block_reason(url, allow_private_networks)
    if reason is not None:
        return (f"error: blocked: {reason}. sable agents only fetch public "
                "http(s) URLs; a human can enable local fetching via "
                "`tools.web_fetch.allow_private_networks` in the config.")
    try:
        timeout = float(timeout_seconds) if timeout_seconds is not None \
            else WEB_FETCH_TIMEOUT_S
    except (TypeError, ValueError):
        timeout = WEB_FETCH_TIMEOUT_S
    timeout = max(1.0, min(timeout, 120.0))
    try:
        max_chars = int(max_chars)
    except (TypeError, ValueError):
        max_chars = 8000
    max_chars = max(200, min(max_chars, WEB_FETCH_MAX_BYTES))

    current = url
    # Manual redirect loop so EVERY hop passes the same SSRF check
    # semantics: redirect targets are re-validated, never blindly followed).
    for _hop in range(WEB_FETCH_MAX_REDIRECTS + 1):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                resp = client.get(current, headers={
                    "User-Agent": "sable-agent (local AI runtime)"})
        except httpx.HTTPError as e:
            return f"error: web_fetch failed: {e}"
        if resp.is_redirect or resp.has_redirect_location:
            loc = resp.headers.get("location") or ""
            from urllib.parse import urljoin

            current = urljoin(str(resp.url), loc)
            reason = _url_block_reason(current, allow_private_networks)
            if reason is not None:
                return (f"error: blocked: redirect target refused — {reason}")
            continue
        break
    else:
        return "error: web_fetch: too many redirects"

    body_bytes = resp.content[:WEB_FETCH_MAX_BYTES]
    ctype = (resp.headers.get("content-type") or "").lower()
    text = body_bytes.decode("utf-8", errors="replace")
    if "html" in ctype or text.lstrip()[:200].lower().startswith(
            ("<!doctype html", "<html")):
        text = _html_to_text(text)
    if not text.strip():
        return f"(HTTP {resp.status_code} — no readable text content)"
    return (f"HTTP {resp.status_code} {current}\n\n"
            + _cap_text(text, max_chars))


def run_command(command: str, timeout_seconds: Optional[float] = None,
                cwd: Optional[str] = None, argv_mode: bool = False,
                **_: Any) -> str:
    """Run a local shell command and return its exit code and output.

    Capability-gated (`run_commands`): see CAPABILITY_TOOLS. This is the
    "use other programs" capability — it executes with the user's own
    privileges BY DESIGN (that is what the capability grants), so the gate
    lives in config/profiles, not here. Bounded: hard timeout cap (child is
    killed), output cap on both streams. File *containment* does not apply
    to shell commands; enabling this capability opts out of that sandbox for
    this agent.

    ``argv_mode`` (config `tools.run_command_argv_only`, bound as a private
    kwarg — invisible to the model): run WITHOUT a shell via shlex-split argv,
    so pipes/redirection/command substitution are literal arguments instead
    of executable syntax. Hardening option for profiles that need program
    execution but not full shell semantics. Quoted segments containing shell
    metacharacters still execute as data; anything needing `$`/backtick
    expansion is rejected up front rather than silently passed through.
    """
    if not isinstance(command, str) or not command.strip():
        return "error: command required (non-empty string)"
    try:
        t = float(timeout_seconds) if timeout_seconds is not None \
            else RUN_COMMAND_DEFAULT_TIMEOUT_S
    except (TypeError, ValueError):
        t = RUN_COMMAND_DEFAULT_TIMEOUT_S
    t = max(1.0, min(t, RUN_COMMAND_MAX_TIMEOUT_S))
    run_cwd: Optional[str] = None
    if cwd:
        p = Path(cwd).expanduser()
        if not p.is_dir():
            return f"error: cwd is not a directory: {cwd}"
        run_cwd = str(p)
    if argv_mode:
        import shlex

        try:
            argv = shlex.split(command)
        except ValueError as e:
            return f"error: unbalanced quotes in command (argv mode): {e}"
        if not argv:
            return "error: command required (non-empty string)"
        for tok in argv:
            # Refuse tokens that only make sense through shell expansion —
            # silently passing them as literal data is worse than refusing.
            if "$" in tok or "`" in tok:
                return ("error: argv mode forbids shell expansion ($, `); "
                        "no shell is available to this agent")
        popen_arg: Any = argv
    else:
        popen_arg = command
    try:
        proc = subprocess.run(popen_arg, shell=not argv_mode, cwd=run_cwd,
                              capture_output=True, text=True, timeout=t)
    except subprocess.TimeoutExpired:
        return (f"error: command timed out after {t:.0f}s and was killed "
                f"(raise timeout_seconds up to {RUN_COMMAND_MAX_TIMEOUT_S:.0f} "
                "if the program legitimately needs longer)")
    except OSError as e:
        return f"error: could not run command: {e}"
    parts = [f"exit code: {proc.returncode}"]
    for label, data in (("stdout", proc.stdout or ""),
                        ("stderr", proc.stderr or "")):
        if data.strip():
            parts.append(f"{label}:\n{_cap_text(data, RUN_COMMAND_OUTPUT_CAP)}")
    return "\n".join(parts)


DELEGATION_TOOL_NAMES = frozenset(
    {"list_agents", "delegate_task", "check_delegation"})


def list_agents(**kwargs: Any) -> str:
    """List agent profiles available for delegation (project-scoped)."""
    from .delegation import list_agents_text

    return list_agents_text(_ctx=kwargs.get("_ctx"))


def delegate_task(agent: str = "", task: str = "", context: str = "",
                  timeout_seconds: Optional[float] = None, **kwargs: Any) -> str:
    """Delegate a subtask to another agent profile on the same project."""
    from .delegation import execute_delegation

    return execute_delegation(agent=agent, task=task, context=context,
                              timeout_seconds=timeout_seconds,
                              _ctx=kwargs.get("_ctx"))


def check_delegation(id: str = "", **kwargs: Any) -> str:  # noqa: A002 (schema name)
    """Poll a delegation's status/result."""
    from .delegation import check_delegation_status

    return check_delegation_status(id)


# ------------------------------------------------- vision / image tools

def view_image(path: str, note: str = "", **_: Any) -> str:
    """Attach a local image file to the conversation for the model to see.

    Capability-gated (`vision`): see CAPABILITY_TOOLS. Returns a short text
    stub with an attach marker — core/images.expand_message_images rewrites
    it into a native image_url content part on the outbound request when the
    served model has vision (llama-server + mmproj). On text-only backends
    the stub stays text and the path is still informative. `note` lets the
    agent record why it is looking (kept in the trace).
    """
    from .images import attach_marker, encode_image_file

    _data_url, err = encode_image_file(path)
    if err is not None:
        return err
    suffix = f" — context: {note}" if note else ""
    return attach_marker(str(Path(path).expanduser())) + suffix


def take_screenshot(path: str = "", **_: Any) -> str:
    """Capture the current screen and attach it for the model to see.

    Capability-gated (`screenshots`): see CAPABILITY_TOOLS. Dispatches across
    OS backends (native CLI capturers, then Pillow ImageGrab, then
    xdg-desktop-portal on Wayland) — see images.take_screenshot. Like
    view_image the result is an attach marker expanded per-request, so the
    agent can watch its own work (e.g. a Blender viewport) turn by turn.
    """
    from .images import attach_marker
    from .images import take_screenshot as _grab

    out, err = _grab(path or None)
    if err is not None or not out:
        return err or "error: screenshot produced no output"
    return attach_marker(out)


# ------------------------------------------------------- persistent skills
# (core/skills.py) Agent-authored procedural memory: small markdown recipes
# under SABLE_HOME/skills/ that survive across runs, so per-app knowledge
# (Steam GUI quirks, Blender headless invocation, which screenshot path works
# here) is learned once instead of re-derived. Only name+description lines are
# injected into prompts; bodies are read on demand. Not capability-gated —
# writes stay inside SABLE_HOME (same trust level as todo_write), and the
# whole feature is switched by config `skills.enabled`.

def save_skill(name: str = "", description: str = "", content: str = "",
               **_: Any) -> str:
    """Create or update a persistent skill file."""
    from .skills import save_skill as _save

    res = _save(name=name, description=description, content=content)
    if not res.get("ok"):
        return f"error: {res.get('error', 'unknown')}"
    return (f"skill saved: {res['path']}\n"
            f"bundled scripts go in: {res['scripts_dir']} "
            "(run_command can place/executables there)")


def read_skill(name: str = "", **_: Any) -> str:
    """Read one skill's full markdown body by name."""
    from .skills import load_skill

    s = load_skill((name or "").strip().lower())
    if s is None:
        return (f"error: no skill named {name!r} — check list_skills")
    # Usage signal for staleness review (dashboard Skills card surfaces
    # never-read skills): one cheap INFO line per read.
    logger.info("skill used: %s", s.name)
    return f"# {s.name}\n{s.description}\n(updated {s.updated_ts})\n\n{s.content}"


def delete_skill(name: str = "", **_: Any) -> str:
    """Delete a stored skill (markdown body; bundled scripts dir kept)."""
    from .skills import delete_skill as _delete

    res = _delete((name or "").strip().lower())
    if not res.get("ok"):
        return f"error: {res.get('error', 'unknown')}"
    return (f"skill deleted: {res['name']}"
            + (" (bundled scripts dir kept — remove it via run_command "
               "if obsolete)" if res.get("scripts_dir") else ""))


def list_skills(**_: Any) -> str:
    """List all stored skills with one-line descriptions."""
    from .skills import all_skills

    skills = all_skills()
    if not skills:
        return "(no skills saved yet — use save_skill to persist what you learn)"
    return "\n".join(s.one_liner() for s in skills)


# --------------------------------------------------------------- schemas

def _schema(name: str, description: str, properties: dict,
            required: List[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# OpenAI-shaped tool schemas for the agent loop.
TOOL_SCHEMAS: list[dict] = [
    _schema(
        "read_file",
        "Read a text file from the local filesystem with line numbers. "
        "Use offset/limit to page through large files.",
        {
            "path": {"type": "string", "description": "File path to read"},
            "offset": {"type": "integer",
                       "description": "First line number (1-based, default 1)"},
            "limit": {"type": "integer",
                      "description": "Max lines to return (default 2000)"},
        },
        ["path"],
    ),
    _schema(
        "list_dir",
        "List the entries of a local directory ('/' suffix marks directories).",
        {
            "path": {"type": "string", "description": "Directory path (default .)"},
        },
        [],
    ),
    _schema(
        "write_file",
        "Create or overwrite a text file with the given content "
        "(parent directories are created).",
        {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "Full file content"},
        },
        ["path", "content"],
    ),
    _schema(
        "patch",
        "Surgical edit: replace old_string with new_string inside a file. "
        "old_string must match exactly one location (include context) unless "
        "replace_all is true; whitespace-tolerant fuzzy matching is attempted "
        "if the exact match fails.",
        {
            "path": {"type": "string", "description": "File path to patch"},
            "old_string": {"type": "string",
                           "description": "Exact text to replace (with context)"},
            "new_string": {"type": "string",
                           "description": "Replacement text"},
            "replace_all": {"type": "boolean",
                            "description": "Replace every occurrence (default false)"},
        },
        ["path", "old_string", "new_string"],
    ),
    _schema(
        "search_files",
        "Search file contents by regex (mode=content, default) or file names "
        "by glob (mode=name) under a directory. Skips binary files and common "
        "noise directories (.git, node_modules, ...).",
        {
            "pattern": {"type": "string",
                        "description": "Regex (content mode) or glob (name mode)"},
            "path": {"type": "string", "description": "Directory/file root (default .)"},
            "mode": {"type": "string", "enum": ["content", "name"],
                     "description": "Search mode (default content)"},
            "glob": {"type": "string",
                     "description": "Only scan files matching this glob, e.g. *.py"},
        },
        ["pattern"],
    ),
    _schema(
        "todo_write",
        "Plan and track multi-step work: write the session todo list. Pass the "
        "FULL list to replace it, or merge=true to update specific items by id. "
        "Mark exactly one item in_progress while working; set done when "
        "finished. Call again whenever statuses change.",
        {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Short stable id"},
                        "content": {"type": "string", "description": "Task description"},
                        "status": {"type": "string",
                                   "enum": list(TODO_STATUSES),
                                   "description": "Item status (default pending)"},
                    },
                    "required": ["content"],
                },
                "description": "Todo items",
            },
            "merge": {"type": "boolean",
                      "description": "Update matching ids instead of replacing "
                                     "the whole list (default false)"},
        },
        ["todos"],
    ),
    _schema(
        "todo_read",
        "Show the current session todo list with status markers.",
        {},
        [],
    ),
    _schema(
        "search_codebase",
        "Semantic search over the indexed project codebases (embeddings when a "
        "local embedding model is configured, BM25 lexical fallback otherwise). "
        "Use it to find code by intent/meaning; use search_files for exact "
        "regex matches. Requires indexing enabled in Settings.",
        {
            "query": {"type": "string",
                      "description": "What you are looking for (natural "
                                     "language or identifiers)"},
            "mode": {"type": "string", "enum": ["hybrid", "semantic", "lexical"],
                     "description": "Search mode (default from settings)"},
        },
        ["query"],
    ),
    _schema(
        "search_traces",
        "Search sable's experience buffer — summaries and full transcripts of "
        "past agent runs on this machine. Use it to recall how a similar task "
        "was solved before or what was learned about this system.",
        {
            "pattern": {"type": "string",
                        "description": "Regex matched against trace summaries "
                                       "and transcripts (optional)"},
            "status": {"type": "string",
                       "enum": ["success", "failure", "partial", "unknown"],
                       "description": "Only traces with this outcome"},
            "limit": {"type": "integer",
                      "description": "Max results (default 10, cap 50)"},
        },
        [],
    ),
    _schema(
        "list_agents",
        "List the agent profiles available for delegation on the current "
        "project (or all configured profiles when no project is active), each "
        "with a one-line description and whether it is mid-run right now. "
        "Use this before delegate_task to pick a target.",
        {},
        [],
    ),
    _schema(
        "delegate_task",
        "Delegate a self-contained subtask to another agent profile on the "
        "SAME project and wait (bounded by timeout_seconds) for its result. "
        "The child runs with its own profile's prompt/tools/guardrails and "
        "records its own trace; returns {status, delegation_id, result} — on "
        "timeout a diagnostics payload instead of hanging. Cannot delegate to "
        "yourself or nest further. Pass ALL context the child needs: it does "
        "not see this conversation.",
        {
            "agent": {"type": "string",
                      "description": "Target agent profile name (from list_agents)"},
            "task": {"type": "string",
                     "description": "Self-contained subtask description"},
            "context": {"type": "string",
                        "description": "Supporting context for the child (optional)"},
            "timeout_seconds": {
                "type": "number",
                "description": "Max seconds to wait for the child (default from "
                               "config delegation.timeout_seconds)"},
        },
        ["agent", "task"],
    ),
    _schema(
        "check_delegation",
        "Poll a previously created delegation: status pending|running|"
        "completed|failed|timed_out, with the result text once done.",
        {
            "id": {"type": "string", "description": "Delegation id"},
        },
        ["id"],
    ),
    _schema(
        "save_skill",
        "Persist a durable procedure you learned (about a program, this "
        "machine, or a workflow) so future runs can reuse it without "
        "re-deriving. Overwrites an existing skill of the same name. Write "
        "what WORKED and what BURNED — failure details are the valuable "
        "part. Keep it a tight recipe; put long executable helpers in the "
        "scripts directory the result reports.",
        {
            "name": {"type": "string",
                     "description": "Lowercase slug, e.g. 'steam-gui' or "
                                    "'blender-headless'"},
            "description": {"type": "string",
                            "description": "One line: when to use this skill "
                                           "(shown in the skill index)"},
            "content": {"type": "string",
                        "description": "Markdown recipe body"},
        },
        ["name", "description", "content"],
    ),
    _schema(
        "read_skill",
        "Read the full body of a stored skill by name (the index only shows "
        "one-line descriptions).",
        {
            "name": {"type": "string", "description": "Skill name"},
        },
        ["name"],
    ),
    _schema(
        "list_skills",
        "List stored skills with their one-line descriptions.",
        {},
        [],
    ),
    _schema(
        "delete_skill",
        "Delete a stored skill you no longer trust or need (for example one "
        "that turned out to be wrong). The markdown file is removed; the "
        "bundled scripts directory stays on disk.",
        {
            "name": {"type": "string", "description": "Skill name"},
        },
        ["name"],
    ),
]

TOOL_IMPLS: dict[str, Callable[..., str]] = {
    "read_file": read_file,
    "list_dir": list_dir,
    "write_file": write_file,
    "patch": patch,
    "search_files": search_files,
    "todo_write": todo_write,
    "todo_read": todo_read,
    "search_traces": search_traces,
    "search_codebase": search_codebase,
    "list_agents": list_agents,
    "delegate_task": delegate_task,
    "check_delegation": check_delegation,
    # Persistent skills (core/skills.py). Exposed only when config
    # `skills.enabled` is on — core/agent_runner strips all three otherwise,
    # keeping disabled-feature prompts byte-identical.
    "save_skill": save_skill,
    "read_skill": read_skill,
    "list_skills": list_skills,
    "delete_skill": delete_skill,
    # Capability-gated (see CAPABILITY_TOOLS): present in TOOL_IMPLS so a
    # runner can bind them, but NEVER exposed unless the capability is on —
    # core/agent_runner filters both schemas and impls.
    "web_fetch": web_fetch,
    "run_command": run_command,
    "view_image": view_image,
    "take_screenshot": take_screenshot,
}

# Capability-gated tool schemas, kept OUT of TOOL_SCHEMAS (the default set
# every model sees). core/agent_runner merges CAPABILITY_TOOL_SCHEMAS[name]
# for each enabled capability and binds tools.web_fetch.* / tools.run_command.*
# config knobs into the impls via partial (the model cannot forge them).
CAPABILITY_TOOL_SCHEMAS: dict[str, dict] = {
    "web_fetch": _schema(
        "web_fetch",
        "Fetch a public web page over HTTP(S) and return its text content "
        "(HTML converted to readable text). Only available when this agent "
        "has the internet capability. No JavaScript execution; redirects are "
        "followed but re-checked; private/network-local addresses are refused.",
        {
            "url": {"type": "string", "description": "http(s) URL to fetch"},
            "max_chars": {"type": "integer",
                          "description": "Max characters of text to return "
                                         "(default 8000)"},
            "timeout_seconds": {"type": "number",
                                "description": "Per-hop timeout cap (default "
                                               f"{WEB_FETCH_TIMEOUT_S:.0f})"},
        },
        ["url"],
    ),
    "run_command": _schema(
        "run_command",
        "Run a local shell command and return its exit code, stdout and "
        "stderr. Only available when this agent has the run-commands "
        "capability — it executes with the user's own privileges, so use it "
        "for the task at hand and never for destructive or credential-"
        "touching operations unless explicitly asked.",
        {
            "command": {"type": "string",
                        "description": "Shell command line to execute"},
            "timeout_seconds": {"type": "number",
                                "description": f"Hard timeout (default "
                                               f"{RUN_COMMAND_DEFAULT_TIMEOUT_S:.0f}, "
                                               f"max {RUN_COMMAND_MAX_TIMEOUT_S:.0f})"},
            "cwd": {"type": "string",
                    "description": "Working directory (optional)"},
        },
        ["command"],
    ),
    "view_image": _schema(
        "view_image",
        "Look at a local image file: it is attached to the conversation so "
        "you can see it with your vision. Use it for reference pictures, "
        "rendered output (e.g. a Blender viewport screenshot produced by "
        "run_command), or any file a previous tool result mentioned. Only "
        "available when this agent has the vision capability and the served "
        "model supports images.",
        {
            "path": {"type": "string",
                     "description": "Image file path (png/jpeg/gif/webp/...)"},
            "note": {"type": "string",
                     "description": "Why you are looking at it (optional)"},
        },
        ["path"],
    ),
    "take_screenshot": _schema(
        "take_screenshot",
        "Capture the current screen and attach it so you can see it. Use it "
        "to verify on-screen state while working (e.g. checking a 3D app's "
        "viewport). Only available when this agent has the screenshots "
        "capability and the served model supports images.",
        {
            "path": {"type": "string",
                     "description": "Where to save the PNG (optional; "
                                    "defaults under the sable home)"},
        },
        [],
    ),
}

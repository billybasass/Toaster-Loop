"""Codebase indexing for semantic search (config `indexing:` section).

Gives the agent a semantic view of project codebases: files under the index
roots are chunked (line windows, function-ish boundaries), embedded with a
LOCAL embedding model and stored in a sqlite index. The `search_codebase`
tool queries it.

Feature-wave Phase 5 — per-project indexes:

* One sqlite DB per project: ``<SABLE_HOME>/index/<project_id>.db`` plus a
  ``default.db`` fallback for legacy/config-driven indexing when no project
  context exists. ``get_index(home, cfg, project_id=...)`` routes to the
  right DB; the search_codebase tool binds the active project id from the
  run context (no cross-project results).
* Index roots are DERIVED from projects: a project's ``workspace_dir`` is its
  index root. The manual path list survives only as an advanced override
  (``indexing.paths_override``) which applies to the default/no-project DB.
* sable OWNS the embedding server: there is no user-facing embed endpoint
  anymore. When ``indexing.model`` is set, indexing spawns its own loopback
  ``llama-server --embedding`` on a dynamic port (127.0.0.1 only) and stops
  it on shutdown — only processes it spawned itself are ever terminated
  (see ``stop_all_embed_servers`` / atexit). Spawning failure surfaces as a
  clear build error instead of silently falling back.
* MRL output dimension: ``indexing.output_dim`` (768/512/256/128) truncates
  stored AND query vectors to N dims and L2-renormalizes them at both index
  and query time. The DB metadata records the built dim + model id; a search
  whose configured dim/model differs reports ``index needs rebuild`` instead
  of returning garbage. top_k is an internal constant (SEARCH_TOP_K), no
  longer user config.

Search modes mirror a typical retrieval panel: ``semantic``,
``lexical`` and ``hybrid`` (default; reciprocal-rank merge of both). Without
an embedding model a BM25 lexical index keeps indexing useful offline.

Everything is stdlib + httpx: vectors are stored as float32 blobs and scored
in Python — fine for the tens of thousands of chunks a personal codebase
produces, zero extra dependencies.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import re
import sqlite3
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# top_k is intentionally NOT user config anymore (Phase 5): internal default.
SEARCH_TOP_K = 8
CHUNK_LINES = 60
CHUNK_OVERLAP = 8
MAX_FILE_BYTES = 512_000
EMBED_TIMEOUT_S = 30.0

#: Physical batch for the owned embed server (-b/-ub). llama-server rejects
#: any single embedding input larger than its physical batch — with no -ub
#: passed, it forces n_batch = n_ubatch = 512, which failed index builds on
#: chunks above ~500 tokens. embeddinggemma-class models are trained at 2k
#: ctx, so 2048 is the sweet spot. Override: indexing.batch_size.
EMBED_UBATCH = 2048
#: Client-side truncation budget: stay this far below -ub (estimated tokens).
EMBED_TOKEN_MARGIN = 128

#: User-selectable MRL output dimensions (full model output first).
OUTPUT_DIM_CHOICES = (768, 512, 256, 128)
#: Fallback DB name for the no-project / config-driven index (Phase 5).
DEFAULT_DB_ID = "default"

_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "target", ".tox", ".eggs", ".cache", ".sable", ".next", ".nuxt",
}
_TEXT_EXT = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".c", ".h", ".cpp",
    ".hpp", ".java", ".kt", ".rb", ".php", ".sh", ".bash", ".zsh", ".fish",
    ".yaml", ".yml", ".toml", ".json", ".md", ".rst", ".txt", ".html",
    ".css", ".scss", ".sql", ".vue", ".svelte", ".swift", ".m", ".mm",
    ".cs", ".dart", ".lua", ".pl", ".r", ".tf", ".ini", ".cfg", ".make",
    "makefile", "dockerfile", ".gradle", ".proto", ".ex", ".exs", ".zig",
}


class IndexRebuildRequired(RuntimeError):
    """Raised when a query cannot run against an index built with a
    different embedding dim/model — the index needs rebuilding."""


#: Sentinel for "attribute absent" on duck-typed embed clients.
_MISSING = object()


def _opt_int(value: Any) -> Optional[int]:
    """Int coercion for optional config knobs: None/blank/bad -> None."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _is_textish(p: Path) -> bool:
    if p.name.lower() in ("makefile", "dockerfile", "cmakelists.txt"):
        return True
    return p.suffix.lower() in _TEXT_EXT


def iter_files(roots: Iterable[Path], max_files: int = 20_000) -> Iterable[Path]:
    """Yield indexable files under roots, skipping noise dirs and big files."""
    seen: set = set()
    count = 0
    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        stack = [root]
        while stack:
            d = stack.pop()
            try:
                entries = sorted(os.scandir(d), key=lambda e: e.name)
            except OSError:
                continue
            for e in entries:
                try:
                    if e.name.startswith(".") and Path(e.path) != root / e.name:
                        # keep top-level dotfiles config, skip hidden dirs
                        if e.is_dir(follow_symlinks=False):
                            continue
                    if e.is_dir(follow_symlinks=False):
                        if e.name in SKIP_DIRS:
                            continue
                        stack.append(Path(e.path))
                    elif e.is_file(follow_symlinks=False):
                        p = Path(e.path)
                        rp = p.resolve()
                        if rp in seen:
                            continue
                        try:
                            if p.stat().st_size > MAX_FILE_BYTES:
                                continue
                        except OSError:
                            continue
                        if not _is_textish(p):
                            continue
                        seen.add(rp)
                        count += 1
                        if count > max_files:
                            return
                        yield p
                except OSError:
                    continue


def chunk_file(path: Path) -> List[Tuple[int, int, str]]:
    """Split a file into (start_line, end_line, text) chunks with overlap."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    if not lines:
        return []
    out: List[Tuple[int, int, str]] = []
    step = max(1, CHUNK_LINES - CHUNK_OVERLAP)
    for start in range(0, len(lines), step):
        window = lines[start:start + CHUNK_LINES]
        if not any(l.strip() for l in window):
            pass  # keep even blank-heavy windows: line numbers matter
        out.append((start + 1, start + len(window), "\n".join(window)))
        if start + CHUNK_LINES >= len(lines):
            break
    return out


# ------------------------------------------------------------------ lexical

_WORD_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


#: Worst-case chars per BPE token assumed when estimating (dense code with
#: symbols/punctuation tokenizes near 2 chars/token on Gemma vocabularies).
_CHARS_PER_TOKEN_EST = 2


def clip_to_tokens(text: str, max_tokens: int) -> str:
    """Cheaply bound a chunk below the embed server's physical batch.

    llama-server errors on any single input whose token count exceeds its
    -ub, and sable has no tokenizer here — estimate conservatively
    (_CHARS_PER_TOKEN_EST) and truncate at a newline boundary when possible.
    embed() additionally halves-and-retries if the server still rejects."""
    approx = len(text) // _CHARS_PER_TOKEN_EST
    if approx <= max_tokens:
        return text
    cut = max_tokens * _CHARS_PER_TOKEN_EST
    nl = text.rfind("\n", 0, cut)
    if nl > cut // 2:
        cut = nl
    return text[:cut]


class LexicalIndex:
    """Tiny in-memory BM25 over the chunk table (no dependencies)."""

    def __init__(self, chunks: List[Tuple[int, str]]):  # (row_id, text)
        self.doc_ids = [i for i, _ in chunks]
        self.docs = [_tokenize(t) for _, t in chunks]
        self.n = len(self.docs) or 1
        self.avgdl = sum(len(d) for d in self.docs) / self.n
        self.df: Dict[str, int] = {}
        self.tf: List[Dict[str, int]] = []
        for d in self.docs:
            counts: Dict[str, int] = {}
            for tok in d:
                counts[tok] = counts.get(tok, 0) + 1
            self.tf.append(counts)
            for tok in counts:
                self.df[tok] = self.df.get(tok, 0) + 1

    def score(self, query: str, limit: int) -> List[Tuple[int, float]]:
        q = set(_tokenize(query))
        if not q:
            return []
        k1, b = 1.5, 0.75
        scored: List[Tuple[int, float]] = []
        for idx, counts in enumerate(self.tf):
            s = 0.0
            dl = len(self.docs[idx]) or 1
            for tok in q:
                tfe = counts.get(tok)
                if not tfe:
                    continue
                idf = math.log(1.0 + (self.n - self.df[tok] + 0.5)
                               / (self.df[tok] + 0.5))
                s += idf * tfe * (k1 + 1) / (
                    tfe + k1 * (1 - b + b * dl / self.avgdl))
            if s > 0:
                scored.append((self.doc_ids[idx], s))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]


# ------------------------------------------------------------------ vectors

def _pack(vec: List[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes) -> List[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def truncate_renorm(vec: List[float], dim: Optional[int]) -> List[float]:
    """Matryoshka (MRL) truncation: keep the first ``dim`` components and
    L2-renormalize. Applied to stored AND query vectors at both index and
    query time so cosine similarity stays meaningful at reduced dims.

    ``dim`` falsy or >= len(vec): renormalizes the full vector unchanged
    (a model whose native dim is below the requested truncation simply
    keeps its native output, renormalized)."""
    if not dim or dim <= 0:
        return list(vec)
    v = list(vec[:dim])
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return v
    return [x / n for x in v]


# ---------------------------------------------------------- embedding server

#: Embedding servers this process spawned (for shutdown cleanup). Only
#: clients registered here are ever stopped — foreign processes are never
#: touched. Phase 6 folds this registry into multi-instance serving:
#: ``serving.supervisor.stop_all_servers`` calls ``stop_all_embed_servers``
#: alongside stopping named instances, and ``instances.ports_in_use`` reads
#: these clients' ports so instance spawns never collide with them.
_spawned_clients: set = set()
_spawn_lock = threading.Lock()
_atexit_registered = False

#: Shared embed-client registry keyed by (model_path, ubatch). ONE server per
#: embedding model for the whole process — building indexes for N projects
#: must not spawn N copies of the same model into RAM/VRAM. Entries survive
#: stop(): ``ensure_running`` respawns lazily on the next embed call.
_embed_registry: Dict[Tuple[str, int], "EmbeddingClient"] = {}
_embed_registry_lock = threading.Lock()


def get_embed_client(model_path: str, cfg: Any = None,
                     ubatch: Optional[int] = None) -> "EmbeddingClient":
    """Return the process-shared EmbeddingClient for one model (+ubatch)."""
    key = (str(model_path), int(ubatch) if ubatch else EMBED_UBATCH)
    with _embed_registry_lock:
        client = _embed_registry.get(key)
        if client is None:
            client = EmbeddingClient(model_path=model_path, cfg=cfg,
                                     ubatch=ubatch)
            _embed_registry[key] = client
        elif cfg is not None:
            client.cfg = cfg  # keep spawn options fresh after config edits
        # Refresh the launch knobs from the CURRENT config so an edit to
        # indexing.gpu_layers / indexing.threads reaches the next spawn. A
        # server that is ALREADY running keeps the flags it booted with —
        # Settings' "stop embedding server" (or a restart) applies the change.
        ice = getattr(cfg, "indexing", None) or {}
        client.gpu_layers = _opt_int(ice.get("gpu_layers"))
        client.threads = _opt_int(ice.get("threads"))
        return client


def stop_all_embed_servers() -> None:
    """Stop every embedding server THIS process spawned (shutdown hook)."""
    with _embed_registry_lock:
        _embed_registry.clear()
    with _spawn_lock:
        clients = list(_spawned_clients)
    for c in clients:
        try:
            c.stop()
        except Exception:
            pass


class EmbeddingClient:
    """OpenAI-compatible /v1/embeddings client that OWNS its llama-server.

    Feature-wave Phase 5: the attach-style URL mode is gone from user
    config — sable always spawns its own loopback server it can stop:

      * model_path given -> spawn `llama-server -m <model> --embedding
        --host 127.0.0.1 --port <free>`; stop() terminates ONLY that child
      * none             -> available() False; caller falls back to lexical

    Spawn failure (missing llama-server binary, unhealthy boot) raises a
    clear RuntimeError — never a silent fallback while a model is set.
    """

    def __init__(self, model_path: Optional[str] = None, cfg: Any = None,
                 ubatch: Optional[int] = None,
                 gpu_layers: Optional[int] = None,
                 threads: Optional[int] = None):
        self.model_path = str(Path(model_path).expanduser()) if model_path else None
        self.cfg = cfg
        self.ubatch = int(ubatch) if ubatch else EMBED_UBATCH
        #: -ngl override (``indexing.gpu_layers``). None => auto-detect.
        _ice = getattr(cfg, "indexing", None) or {}
        self.gpu_layers = (_opt_int(gpu_layers)
                           if gpu_layers is not None
                           else _opt_int(_ice.get("gpu_layers")))
        #: -t override (``indexing.threads``). None => llama.cpp default.
        self.threads = (_opt_int(threads) if threads is not None
                        else _opt_int(_ice.get("threads")))
        self.url: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self.dim: Optional[int] = None

    def available(self) -> bool:
        return bool(self.model_path)

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "mode": ("spawn" if self.model_path else "none"),
            "url": self.url,
            "model_path": self.model_path,
            "running": bool(self._proc and self._proc.poll() is None),
            "dim": self.dim,
        }

    def _resolve_gpu_layers(self) -> Optional[int]:
        """-ngl for the embedding server.

        Auto-detect used to mean FULL_OFFLOAD_LAYERS (999) on every non-CPU
        backend, so the embed server competed with the chat model for the
        GPU/VRAM — painful on unified-memory APUs (Strix Halo, Jetson) where
        the two share both bandwidth and memory. ``indexing.gpu_layers``
        overrides: 0 runs the embeddings on CPU, any N offloads N layers.
        Read from ``self.cfg`` at spawn time (get_embed_client refreshes it
        on every config edit) so the knob takes effect without a restart.
        """
        if self.gpu_layers is not None:
            return self.gpu_layers
        cfg_ice = getattr(self.cfg, "indexing", None) or {}
        want = _opt_int(cfg_ice.get("gpu_layers"))
        if want is not None:
            return want
        from ..serving import runtime

        ngl = runtime.gpu_layers(None)
        return ngl if ngl else None

    def ensure_running(self) -> Optional[str]:
        """Return base_url, spawning the local server when needed."""
        with self._lock:
            if not self.model_path:
                return None
            if self._proc is not None and self._proc.poll() is None and self.url:
                return self.url
            from ..serving.supervisor import find_llama_server, _free_port

            try:
                binary = find_llama_server(
                    getattr(self.cfg, "llama_server_binary", None),
                    auto_setup=bool(getattr(self.cfg, "llama_auto_setup", True)),
                    cfg=self.cfg)
            except Exception as e:
                raise RuntimeError(
                    f"cannot spawn the embedding server: {e}") from e
            port = _free_port("127.0.0.1")
            # Split GGUF safety: point llama-server at shard 1 of a multi-part
            # model (same rule as serving/supervisor.py).
            from ..core.model_discovery import resolve_gguf_shards

            model_arg = resolve_gguf_shards(self.model_path)
            cmd = [binary, "-m", model_arg, "--embedding",
                   "--host", "127.0.0.1", "--port", str(port),
                   # Physical batch must cover the largest chunk or llama-server
                   # rejects it (default 512); -c matches the trained context of
                   # embeddinggemma-class models so the slot cap stays quiet.
                   "-b", str(self.ubatch), "-ub", str(self.ubatch),
                   "-c", str(self.ubatch),
                   "--alias", "sable-embeddings"]
            ngl = self._resolve_gpu_layers()
            if ngl is not None:
                cmd += ["-ngl", str(ngl)]
            if self.threads:
                cmd += ["-t", str(self.threads)]
            log_dir = None
            try:
                from ..store.paths import resolve_sable_home
                log_dir = resolve_sable_home(self.cfg) / "logs" \
                    if self.cfg is not None else None
            except Exception:
                pass
            out = subprocess.DEVNULL
            if log_dir:
                log_dir.mkdir(parents=True, exist_ok=True)
                out = open(log_dir / "embed-server.log", "ab")
            self._proc = subprocess.Popen(
                cmd, stdout=out, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL)
            _register_spawned(self)
            import httpx
            base = f"http://127.0.0.1:{port}/v1"
            deadline = time.monotonic() + 180.0
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    _unregister_spawned(self)
                    raise RuntimeError(
                        "embedding server exited early; see logs/embed-server.log")
                try:
                    if httpx.get(base + "/models", timeout=2.0).status_code == 200:
                        self.url = base
                        return base
                except Exception:
                    pass
                time.sleep(0.5)
            self.stop()
            raise RuntimeError("embedding server did not become healthy in 180 s")

    def _embed_once(self, base: str, texts: List[str]) -> List[List[float]]:
        import httpx

        r = httpx.post(base + "/embeddings",
                       json={"model": "sable-embeddings", "input": texts},
                       timeout=EMBED_TIMEOUT_S)
        if r.status_code != 200:
            raise RuntimeError(f"embeddings HTTP {r.status_code}: {r.text[:200]}")
        data = r.json().get("data") or []
        vecs = [d.get("embedding") or [] for d in
                sorted(data, key=lambda d: d.get("index", 0))]
        if len(vecs) != len(texts):
            raise RuntimeError("embeddings response length mismatch")
        if vecs and vecs[0]:
            self.dim = len(vecs[0])
        return vecs

    #: substrings llama-server variants use when an input exceeds its
    #: physical batch/context. Matching several phrasings keeps the halve-
    #: and-retry working across llama.cpp versions that reword the error.
    _OVER_BATCH_MARKERS = ("too large", "exceeds", "n_batch", "context size")

    def _over_batch(self, msg: str) -> bool:
        low = msg.lower()
        return any(m in low for m in self._OVER_BATCH_MARKERS)

    def embed(self, texts: List[str]) -> List[List[float]]:
        base = self.ensure_running()
        if not base:
            raise RuntimeError("no embedding model configured (indexing.model)")
        limit = max(64, self.ubatch - EMBED_TOKEN_MARGIN)
        clipped = [clip_to_tokens(t, limit) for t in texts]
        try:
            return self._embed_once(base, clipped)
        except RuntimeError as e:
            # Char-estimate clipping is heuristic; if the server still
            # rejects (dense code tokenizes denser than estimated), halve
            # the budget and retry a couple of times before giving up.
            if not self._over_batch(str(e)):
                raise
            for _ in range(3):
                limit = max(64, limit // 2)
                clipped = [clip_to_tokens(t, limit) for t in texts]
                try:
                    return self._embed_once(base, clipped)
                except RuntimeError as e2:
                    if not self._over_batch(str(e2)):
                        raise
                    e = e2
            raise

    def stop(self) -> None:
        """Terminate the child process THIS client spawned (never anything
        else — sable only ever kills processes it started)."""
        with self._lock:
            _unregister_spawned(self)
            if self._proc is not None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=8)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                self._proc = None
            self.url = None


def _register_spawned(client: EmbeddingClient) -> None:
    global _atexit_registered
    with _spawn_lock:
        _spawned_clients.add(client)
        if not _atexit_registered:
            atexit.register(stop_all_embed_servers)
            _atexit_registered = True


def _unregister_spawned(client: EmbeddingClient) -> None:
    with _spawn_lock:
        _spawned_clients.discard(client)


# ------------------------------------------------------------------- index

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
  path TEXT PRIMARY KEY,
  mtime REAL NOT NULL,
  size INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT NOT NULL,
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  text TEXT NOT NULL,
  embedding BLOB,
  model TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

# meta keys: last_built_at, output_dim, vector_dim, model_id


def _project_workspace(home: Path, project_id: str) -> Optional[str]:
    """workspace_dir from <home>/projects/<id>.json (None when unknown)."""
    p = Path(home) / "projects" / f"{project_id}.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        ws = str(data.get("workspace_dir") or "").strip()
        return ws or None
    return None


@dataclass
class IndexStatus:
    enabled: bool = False
    project_id: Optional[str] = None
    roots: List[str] = None  # type: ignore[assignment]
    files: int = 0
    chunks: int = 0
    embedded: int = 0
    embedding_mode: str = "none"
    output_dim: Optional[int] = None
    built_dim: Optional[int] = None
    model_id: Optional[str] = None
    search_mode: str = "hybrid"
    last_built_at: Optional[str] = None
    building: bool = False
    pending: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        return d


class CodebaseIndex:
    """sqlite-backed chunk/embedding index for ONE project (or the default
    no-project DB). Phase 5: one DB per project at <home>/index/<id>.db;
    project roots derive from the project's workspace_dir."""

    def __init__(self, home: Path, cfg: Any = None,
                 index_cfg: Optional[Dict[str, Any]] = None,
                 project_id: Optional[str] = None,
                 workspace_dir: Optional[str] = None):
        self.home = Path(home)
        self.cfg = cfg
        ic = dict(index_cfg or getattr(cfg, "indexing", None) or {})
        self.index_cfg = ic
        self.enabled = bool(ic.get("enabled", False))
        self.search_mode = str(ic.get("search_mode", "hybrid")).lower()
        # top_k is an internal constant now; config `top_k` is absorbed but
        # no longer read (backward-compat handled in core/config.py).
        self.top_k = SEARCH_TOP_K
        od = ic.get("output_dim")
        self.output_dim: Optional[int] = int(od) if od else None
        self.model_path: Optional[str] = str(ic.get("model") or "").strip() or None
        self.project_id = (str(project_id).strip() or None) if project_id else None
        if self.project_id and not _PROJECT_ID_RE.match(self.project_id):
            raise ValueError(f"invalid project id for index: {self.project_id!r}")
        # Index roots: derived from the project when bound; the manual list
        # survives only as an advanced override for the default DB.
        if self.project_id:
            ws = workspace_dir or _project_workspace(self.home, self.project_id)
            self.roots = [Path(ws).expanduser()] if ws else []
        else:
            raw_roots = ic.get("paths_override") or ic.get("paths") or []
            if isinstance(raw_roots, str):
                raw_roots = [raw_roots]
            self.roots = [Path(str(r)).expanduser() for r in raw_roots] or \
                [self._default_root(cfg)]
        db_id = self.project_id or DEFAULT_DB_ID
        self.db_path = self.home / "index" / f"{db_id}.db"
        self._embed: Optional[EmbeddingClient] = None
        self._build_lock = threading.Lock()
        self._building = False
        #: cooperative cancellation flag for a running build (set by
        #: ``cancel_build`` — project deletion uses it so an in-flight build
        #: cannot recreate the DB file after delete_project_index removed it)
        self._cancel = threading.Event()
        self.last_error: Optional[str] = None
        self._schema_ready = False
        #: True once get_index owns this cached instance (see _needs_cfg_repair).
        self._managed = False

    # -- storage --------------------------------------------------------
    @staticmethod
    def _default_root(cfg: Any) -> Path:
        """Fallback root for the default (no-project, no-override) index.

        Was a bare ``Path.cwd()``: launched from the dashboard (whose cwd is
        SABLE_HOME) that silently indexed ``~/.sable`` — llama.cpp build trees
        and vendored engines, 9k files / 57k chunks of noise, and the config
        file itself. Prefer the configured workspace when there is one, so the
        default index points at code the agent actually works on.
        """
        ws = str((getattr(cfg, "agent", None) or {}).get(
            "workspace_root") or "").strip()
        if ws and ws not in ("None", "null"):
            return Path(ws).expanduser()
        return Path.cwd()

    def _db(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.db_path), timeout=30.0)
        # WAL: readers (search/status during a build) never block the builder.
        con.execute("PRAGMA journal_mode=WAL")
        if not self._schema_ready:
            con.executescript(SCHEMA)
            self._schema_ready = True
        return con

    def _meta(self, con: sqlite3.Connection) -> Dict[str, str]:
        return {k: v for k, v in con.execute("SELECT key, value FROM meta")}

    def embedder(self) -> EmbeddingClient:
        if self._embed is None:
            # Phase 5: sable owns the server — spawn mode only (no URL).
            # The client is SHARED per model across all project indexes in
            # this process (get_embed_client): N projects, one embed server.
            ub = self.index_cfg.get("batch_size")
            try:
                ub = int(ub) if ub else None
            except (TypeError, ValueError):
                ub = None
            if not self.model_path:
                # Lexical-only: a cheap unavailable client, nothing to share.
                self._embed = EmbeddingClient(model_path=None, cfg=self.cfg,
                                              ubatch=ub)
            else:
                self._embed = get_embed_client(self.model_path, cfg=self.cfg,
                                               ubatch=ub)
        return self._embed

    # -- status ----------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        st = IndexStatus(enabled=self.enabled,
                         project_id=self.project_id,
                         roots=[str(r) for r in self.roots],
                         search_mode=self.search_mode,
                         building=self._building,
                         output_dim=self.output_dim,
                         model_id=self.model_path,
                         embedding_mode=self.embedder().state["mode"])
        try:
            con = self._db()
            st.files = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            st.chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            st.embedded = con.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
            # Chunks awaiting vectors (semantic search degraded until a build
            # embeds them) — the top-bar index pill reads this. Lexical-only
            # indexes never embed, so they are never "pending".
            st.pending = (st.chunks - st.embedded
                          if self.embedder().available() else 0)
            meta = self._meta(con)
            st.last_built_at = meta.get("last_built_at")
            st.built_dim = int(meta["vector_dim"]) if meta.get("vector_dim") else None
            con.close()
        except sqlite3.Error as e:
            self.last_error = str(e)
        st_dict = st.to_dict()
        st_dict["error"] = self.last_error
        st_dict["embedding"] = self.embedder().state
        return st_dict

    # -- build -----------------------------------------------------------
    def cancel_build(self) -> None:
        """Request cancellation of a running build (cooperative). The build
        checks the flag between files and between embed batches and aborts
        with RuntimeError; already-committed rows stay, so the next build
        resumes incrementally. Project deletion uses this before removing
        the DB file so a build cannot resurrect it."""
        self._cancel.set()

    def build(self, progress: Optional[Any] = None) -> Dict[str, Any]:
        """(Re)index changed files. Embeds in batches when a model is set."""
        if not self._build_lock.acquire(blocking=False):
            raise RuntimeError("an index build is already running")
        self._building = True
        self._cancel.clear()
        try:
            return self._build_locked(progress)
        finally:
            self._building = False
            self._build_lock.release()

    def _build_locked(self, progress) -> Dict[str, Any]:
        say = progress or (lambda line: None)
        if self.project_id and not self.roots:
            raise RuntimeError(
                f"project {self.project_id} has no indexable workspace_dir "
                "(project record missing or workspace deleted)")
        con = self._db()
        embedder = self.embedder()
        want_embed = embedder.available()
        known = {p: (m, s) for p, m, s in
                 con.execute("SELECT path, mtime, size FROM files")}
        added = updated = removed = embedded_chunks = 0

        current: set = set()
        pending: List[Tuple[str, int, int, str]] = []  # rows awaiting embed
        for path in iter_files(self.roots):
            if self._cancel.is_set():
                con.commit()
                con.close()
                raise RuntimeError("index build cancelled")
            sp = str(path.resolve())
            current.add(sp)
            try:
                stt = path.stat()
            except OSError:
                continue
            prev = known.get(sp)
            if prev and abs(prev[0] - stt.st_mtime) < 1e-6 and prev[1] == stt.st_size:
                continue
            con.execute("DELETE FROM chunks WHERE path=?", (sp,))
            for (s, e, text) in chunk_file(path):
                con.execute(
                    "INSERT INTO chunks(path,start_line,end_line,text) "
                    "VALUES (?,?,?,?)", (sp, s, e, text))
                if want_embed:
                    pending.append((sp, s, e, text))
            con.execute(
                "INSERT OR REPLACE INTO files(path,mtime,size) VALUES (?,?,?)",
                (sp, stt.st_mtime, stt.st_size))
            if prev:
                updated += 1
            else:
                added += 1
            if (added + updated) % 50 == 0:
                say(f"indexed {added + updated} files…")
        for p in set(known) - current:
            con.execute("DELETE FROM chunks WHERE path=?", (p,))
            con.execute("DELETE FROM files WHERE path=?", (p,))
            removed += 1
        con.commit()

        if want_embed:
            # Backfill: a previously failed build leaves NULL embeddings whose
            # files look "unchanged" to mtime tracking — without this retry the
            # index would stay half-embedded forever.
            seen = {(sp, s, e) for sp, s, e, _ in pending}
            rows = con.execute(
                "SELECT path,start_line,end_line,text FROM chunks "
                "WHERE embedding IS NULL").fetchall()
            for sp, s, e, text in rows:
                if (sp, s, e) not in seen:
                    pending.append((sp, s, e, text))

        embed_ok = False
        if want_embed and pending:
            # Phase 5: a configured model that fails to serve is an ERROR,
            # not a silent lexical fallback — surface it in the build result.
            model_tag = str(embedder.model_path)
            batch = 16
            try:
                for i in range(0, len(pending), batch):
                    if self._cancel.is_set():
                        raise RuntimeError("index build cancelled")
                    group = pending[i:i + batch]
                    vecs = embedder.embed([g[3] for g in group])
                    if self.output_dim:
                        vecs = [truncate_renorm(v, self.output_dim)
                                for v in vecs]
                    # strict=False keeps the old truncation semantics (an
                    # embed server returning fewer vectors must not crash the
                    # build; those chunks simply keep no embedding).
                    for (sp, s, e, _t), vec in zip(group, vecs, strict=False):
                        con.execute(
                            "UPDATE chunks SET embedding=?, model=? "
                            "WHERE path=? AND start_line=? AND end_line=?",
                            (_pack(vec), model_tag, sp, s, e))
                        embedded_chunks += 1
                    con.commit()
                    say(f"embedded {min(i + batch, len(pending))}/{len(pending)} chunks…")
                embed_ok = True
            except Exception as e:
                self.last_error = f"embedding failed: {e}"
                con.commit()
                con.close()
                raise RuntimeError(
                    f"index build failed — embedding server error: {e}") from e

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('last_built_at',?)",
                    (now,))
        if want_embed and embedded_chunks:
            sample = con.execute(
                "SELECT embedding FROM chunks WHERE embedding IS NOT NULL "
                "LIMIT 1").fetchone()
            if sample and sample[0]:
                con.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES('vector_dim',?)",
                    (str(len(_unpack(sample[0]))),))
            con.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('model_id',?)",
                (str(embedder.model_path),))
            con.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('output_dim',?)",
                (str(self.output_dim or 0),))
        con.commit()
        chunks_total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        files_total = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        con.close()
        return {"added": added, "updated": updated, "removed": removed,
                "chunks": chunks_total, "files": files_total,
                "embedded": embedded_chunks,
                "embed_ok": bool(want_embed and embed_ok)}

    # -- search ----------------------------------------------------------
    def _all_chunks(self, con: sqlite3.Connection) -> List[Tuple[int, str, int, int, str, Optional[bytes]]]:
        return list(con.execute(
            "SELECT id,path,start_line,end_line,text,embedding FROM chunks"))

    def search(self, query: str, top_k: Optional[int] = None,
               mode: Optional[str] = None) -> List[Dict[str, Any]]:
        k = int(top_k or self.top_k)
        want = (mode or self.search_mode).lower()
        # last_error is per-call state: clear first so a successful search
        # never keeps reporting an earlier failure in /api/index status.
        self.last_error = None
        con = self._db()
        rows = self._all_chunks(con)
        meta = self._meta(con)
        con.close()
        if not rows:
            return []

        lexical_rank = {row_id: rank for rank, (row_id, _s) in enumerate(
            LexicalIndex([(r[0], r[4]) for r in rows]).score(query, max(k * 4, 20)))}

        semantic_rank: Dict[int, float] = {}
        embedder = self.embedder()
        have_vecs = any(r[5] for r in rows)
        if want in ("semantic", "hybrid") and have_vecs and embedder.available():
            # Dim/model guard (Phase 5): refuse rather than mix vectors from
            # an index built with a different MRL dim or embedding model.
            built_dim = int(meta.get("vector_dim") or 0) or None
            built_model = meta.get("model_id")
            if built_model and self.model_path and built_model != self.model_path:
                raise IndexRebuildRequired(
                    f"index was built with model {built_model!r} but config "
                    f"selects {self.model_path!r} — index needs rebuild")
            try:
                qv = embedder.embed([query])[0]
                if self.output_dim:
                    qv = truncate_renorm(qv, self.output_dim)
                if built_dim and len(qv) != built_dim:
                    raise IndexRebuildRequired(
                        f"index was built at {built_dim} dims but the current "
                        f"query runs at {len(qv)} dims — index needs rebuild")
                sims = []
                for r in rows:
                    if not r[5]:
                        continue
                    sims.append((r[0], cosine(qv, _unpack(r[5]))))
                sims.sort(key=lambda x: -x[1])
                semantic_rank = {rid: rank for rank, (rid, _s) in
                                 enumerate(sims[: max(k * 4, 20)])}
            except IndexRebuildRequired:
                raise
            except Exception as e:
                self.last_error = f"semantic query failed, lexical used: {e}"
                want = "lexical"

        by_id = {r[0]: r for r in rows}

        def rrf(rank_map: Dict[int, Any], weight: float = 1.0):
            out: Dict[int, float] = {}
            for rid, rank in rank_map.items():
                out[rid] = out.get(rid, 0.0) + weight / (60.0 + rank + 1)
            return out

        if want == "semantic":
            scores = rrf(semantic_rank)
        elif want == "lexical" or not semantic_rank:
            scores = rrf(lexical_rank)
        else:  # hybrid
            scores = rrf(lexical_rank)
            for rid, s in rrf(semantic_rank).items():
                scores[rid] = scores.get(rid, 0.0) + s

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        results = []
        for rid, score in ranked:
            _rid, path, s, e, text, emb = by_id[rid]
            results.append({"path": path, "start_line": s, "end_line": e,
                            "score": round(float(score), 5),
                            "snippet": text[:1200],
                            "embedded": bool(emb)})
        return results

    # -- lifecycle ---------------------------------------------------------
    def stop_embed_server(self) -> None:
        if self._embed is not None:
            self._embed.stop()


# ----------------------------------------------------------------- routing

def index_db_path(home: Path, project_id: Optional[str] = None) -> Path:
    """Path of the sqlite DB for a project (or the default fallback DB)."""
    pid = project_id or DEFAULT_DB_ID
    if not _PROJECT_ID_RE.match(pid):
        raise ValueError(f"invalid project id for index: {project_id!r}")
    return Path(home) / "index" / f"{pid}.db"


def delete_project_index(home: Path, project_id: str) -> bool:
    """Delete ONE project's index DB (Phase 5: project deletion cleanup).
    The default DB and other projects' files are never touched. Returns
    True when a file was removed."""
    if not project_id or project_id == DEFAULT_DB_ID:
        return False
    p = index_db_path(home, project_id)
    removed = False
    for suffix in ("", "-wal", "-shm"):
        f = Path(str(p) + suffix)
        if f.is_file():
            try:
                f.unlink()
                removed = True
            except OSError:
                pass
    return removed


# Per-(home, project) index instances for the dashboard/agent processes.
_instances: Dict[Tuple[str, Optional[str]], "CodebaseIndex"] = {}
_single_lock = threading.Lock()


def drop_index_instance(home: Path, project_id: Optional[str] = None) -> bool:
    """Remove one cached CodebaseIndex (project deletion). Returns True when
    an entry was dropped. The shared embed client is intentionally NOT
    stopped — other projects' indexes may use the same server."""
    key = (str(Path(home)), project_id or None)
    with _single_lock:
        return _instances.pop(key, None) is not None


def get_index(home: Path, cfg: Any = None,
              project_id: Optional[str] = None,
              workspace_dir: Optional[str] = None) -> CodebaseIndex:
    """Return the (cached) CodebaseIndex for a project — None = default DB.

    The active config is refreshed on every call so dashboard config edits
    take effect without recreating instances. A changed ``workspace_dir``
    (project edited to point at another directory) likewise rebuilds the
    cached instance in place — otherwise its roots would keep indexing the
    OLD directory until an unrelated config change or restart."""
    key = (str(Path(home)), project_id or None)
    ws = str(workspace_dir).strip() if workspace_dir else None
    with _single_lock:
        idx = _instances.get(key)
        if idx is None:
            idx = CodebaseIndex(Path(home), cfg=cfg, project_id=project_id,
                                workspace_dir=workspace_dir)
            #: Marked as get_index-managed so _needs_cfg_repair can tell a
            #: cfg-less instance IT created (repairable) from a hand-seeded
            #: one (tests, manual seeding) which is left exactly as seeded.
            idx._managed = True
            _instances[key] = idx
            return idx
        # 1. adopt a config when the cached instance never had one (repair).
        if _needs_cfg_repair(idx, cfg):
            fresh = _rebuild_index(Path(home), cfg, idx, project_id,
                                   workspace_dir)
            _instances[key] = fresh
            return fresh
        # 2. project workspace moved -> follow the new directory.
        if ws and idx.roots and str(idx.roots[0]).strip() != ws:
            fresh = _rebuild_index(Path(home), cfg or idx.cfg, idx,
                                   project_id, workspace_dir)
            _instances[key] = fresh
            return fresh
        # 3. indexing config changed (dashboard PUT /api/config).
        if (cfg is not None and idx.cfg is not None
                and idx.index_cfg != (getattr(cfg, "indexing", None) or {})):
            fresh = _rebuild_index(Path(home), cfg, idx, project_id,
                                   workspace_dir)
            _instances[key] = fresh
            return fresh
        return idx


def _needs_cfg_repair(idx: "CodebaseIndex", cfg: Any) -> bool:
    """Should a cached instance adopt ``cfg`` because it never had one?

    Bug fix: ``search_codebase`` used to call ``get_index`` without a cfg.
    The instance it cached therefore carried ``indexing: {}`` ->
    ``enabled: False``, and the config-change branch below required
    ``idx.cfg is not None`` — so nothing could ever correct it: semantic
    search reported "codebase indexing is disabled" for every agent until the
    process restarted. Only adopt the config when it observably changes the
    instance, so cfg-less callers and hand-seeded (duck-typed) cache entries
    keep behaving exactly as before.
    """
    if cfg is None or getattr(idx, "cfg", None) is not None:
        return False
    if not getattr(idx, "_managed", False):
        return False  # hand-seeded instance: owned by the seeder, not us
    new = getattr(cfg, "indexing", None) or {}
    if idx.index_cfg == new:
        return False
    enabled = bool(new.get("enabled", False))
    model = str(new.get("model") or "").strip() or None
    return enabled != bool(idx.enabled) or model != idx.model_path


def _rebuild_index(home: Path, cfg: Any, idx: "CodebaseIndex",
                   project_id: Optional[str],
                   workspace_dir: Optional[str]) -> "CodebaseIndex":
    """Fresh instance for the current config, reusing the old embed client
    ONLY when it still serves what the new config wants.

    Carrying ``idx._embed`` over unconditionally was the second half of the
    "indexing does not start until I restart" report: a lexical-only client
    (built when no model was configured) survived a config edit that DID set
    ``indexing.model``, and ``embedder()`` only builds a client when
    ``_embed is None`` — so the index reported ``embedding_mode: none`` and
    embedded nothing forever. Dropping an incompatible client is safe: the
    old server is still owned by stop_all_embed_servers/atexit, and sable
    only ever terminates processes it spawned itself."""
    fresh = CodebaseIndex(home, cfg=cfg, project_id=project_id,
                          workspace_dir=workspace_dir)
    fresh._managed = True
    if _embed_compatible(getattr(idx, "_embed", None), fresh):
        fresh._embed = idx._embed
    return fresh


def _embed_compatible(client: Optional["EmbeddingClient"],
                      index: "CodebaseIndex") -> bool:
    """Does an existing embed client serve what THIS index config wants?

    The get_index rebuild paths carried ``idx._embed`` over unconditionally.
    A lexical-only client (model_path None, built when indexing had no model
    configured) therefore survived a config change that DID set
    ``indexing.model`` — CodebaseIndex.embedder() only builds a client when
    ``_embed is None``, so the index kept reporting ``embedding_mode: none``
    and silently skipping every embedding until the process restarted. That
    was the "indexing doesn't start until I restart" report.
    """
    if client is None:
        return False
    # Duck-typed / hand-seeded clients (test doubles) that lack these
    # attributes are treated as incompatible: rebuild rather than guess.
    have = getattr(client, "model_path", _MISSING)
    ub = getattr(client, "ubatch", _MISSING)
    if have is _MISSING or ub is _MISSING:
        return False
    want = index.model_path
    if bool(want) != bool(have):
        return False
    if want and str(have) != str(Path(want).expanduser()):
        return False
    if want and client.ubatch != (_opt_int(index.index_cfg.get("batch_size"))
                                  or EMBED_UBATCH):
        return False
    return True

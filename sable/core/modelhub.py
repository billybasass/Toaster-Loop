"""Model Hub: search, download and manage AI models from Hugging Face.

A model-hub service pattern, distilled to sable's
dependency budget — plain ``httpx`` against the public HF Hub REST API, no
``huggingface_hub`` import. Four capabilities mirror studio's hub service:

* **Search** — ``GET /api/models`` with a text query, sorted by downloads,
  annotated GGUF or not (tags/name heuristic like studio's catalog split).
* **Variants** — repo file listing grouped into quantization variants
  (Q4_K_M, UD-Q5_K_S, ...), shard-aware: ``X-00001-of-00002.gguf`` files fold
  into one variant with summed sizes. Non-GGUF repos (training base models)
  expose a single pseudo-variant of every weight file.
* **Download** — background worker threads writing the standard HF hub-cache
  layout ``<base>/models--<org>--<name>/snapshots/<sha>/<relpath>`` so files
  land where ``core/model_discovery.py`` already looks (serve + train pickers
  need zero refresh logic). Resumable: bytes stream into ``*.part`` siblings
  and continue with an HTTP ``Range`` request on retry — the same trick as
  studio's resumable_partials worker.
* **Manage** — installed-repo inventory (sizes dedupe blob symlinks) and
  guarded deletion confined to the base folder.

Target folder resolution reuses ``model_discovery.resolve_models_dir``:
explicit config/env ``models_dir``, then the HF cache defaults; when nothing
exists yet, downloads go to ``<SABLE_HOME>/models`` (created on demand).

Endpoint/token are env-driven like huggingface_hub: ``HF_ENDPOINT`` (default
https://huggingface.co) and ``HF_TOKEN`` for gated repos.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .model_discovery import resolve_models_dir

DEFAULT_ENDPOINT = "https://huggingface.co"
REQUEST_TIMEOUT = 30.0

# Repo ids: org/name with conservative charset, no traversal, no hidden files.
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*(/[A-Za-z0-9][A-Za-z0-9._\-]*)?$")

# Quant tag inside a GGUF filename/path: Q4_K_M, Q5_K_S, iq4_xs, UD-Q3_K_XL...
_QUANT_RE = re.compile(r"(?i)(?:^|[-._])(u?d-)?(iq\d+[a-z0-9_]*|q\d+(?:_[a-z0-9]+)*)(?=[-._]|$)")
# Sharded GGUF suffix: -00001-of-00002 (folded when grouping a variant).
_SHARD_RE = re.compile(r"[-._]\d{5}(-of-\d{5})?(?=\.gguf$)", re.IGNORECASE)

_WEIGHT_EXTS = (".safetensors", ".bin", ".gguf")

#: Companion artifacts (mmproj projectors / MTP drafters): an "mmproj"/"mtp"
#: token in the filename stem or a directory segment of that name
#: (hub repos nest drafters in ``MTP/``). Never standalone models.
_COMPANION_TOKEN_RE = re.compile(r"(^|[-._ ])(mmproj|mtp)([-._ ]|$)",
                                 re.IGNORECASE)


def _is_companion_rel(rel: str) -> bool:
    """True for mmproj/mtp companion files anywhere in a repo tree."""
    parts = rel.replace("\\", "/").split("/")
    stem = Path(parts[-1]).stem if parts else ""
    if _COMPANION_TOKEN_RE.search(stem):
        return True
    return any(_COMPANION_TOKEN_RE.search(seg) for seg in parts[:-1])


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class HubError(RuntimeError):
    """User-facing hub failure (bad repo id, HTTP error, unsafe path...)."""


# ---------------------------------------------------------------- images
#
# Model cards render images (sample grids, banners). The dashboard CSP is
# default-src 'self', so remote <img> tags cannot load directly — instead the
# UI rewrites image URLs to GET /api/hub/image, a same-origin proxy that only
# fetches image bytes from allow-listed hub hosts.

MAX_IMAGE_BYTES = 8 * 1024 * 1024
_IMAGE_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}
_ALLOWED_IMAGE_HOSTS = ("huggingface.co", "hf.co", "raw.githubusercontent.com")


def fetch_image(repo_id: str = "", path: str = "",
                url: str = "") -> Tuple[bytes, str]:
    """Fetch one card image; returns (bytes, content_type).

    Two shapes: ``repo_id`` + repo-relative ``path`` (proxied through the
    configured endpoint's resolve route), or an absolute https ``url`` whose
    host must be on the allow-list. SVG is rejected outright (scripts); size is
    capped at MAX_IMAGE_BYTES."""
    if url:
        from urllib.parse import urlparse

        u = urlparse(url)
        host = (u.hostname or "").lower()
        if u.scheme != "https" or not any(
                host == h or host.endswith("." + h)
                for h in _ALLOWED_IMAGE_HOSTS):
            raise HubError("image host not allowed")
        target = url
    else:
        repo_id = validate_repo_id(repo_id)
        p = str(path or "").replace("\\", "/").lstrip("/")
        if not p or ".." in Path(p).parts:
            raise HubError("invalid image path")
        if Path(p).suffix.lower() not in _IMAGE_TYPES:
            raise HubError("unsupported image type")
        target = f"{endpoint()}/{repo_id}/resolve/main/{p}"
    try:
        r = httpx.get(target, headers=_headers(), timeout=REQUEST_TIMEOUT,
                      follow_redirects=True)
    except httpx.HTTPError as e:
        raise HubError(f"image fetch failed: {e}") from e
    if r.status_code != 200:
        raise HubError(f"image not found (HTTP {r.status_code})")
    ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip()
    if not ctype.startswith("image/") or ctype == "image/svg+xml":
        raise HubError("not a displayable image")
    return r.content[:MAX_IMAGE_BYTES], ctype


def _first_card_image(readme: str) -> str:
    """First content image referenced by a model card README (markdown or raw
    HTML img tag). Badge/shield/logo-ish and svg sources are skipped so the
    detail banner shows real model art, not status badges."""
    if not readme:
        return ""
    srcs = re.findall(r"!\[[^\]]*\]\(([^)\s]+)", readme)
    srcs += re.findall(r"<img[^>]*?src=[\"']([^\"']+)", readme, re.IGNORECASE)
    for src in srcs:
        low = src.lower()
        if low.endswith(".svg") or "badge" in low or "shields.io" in low \
                or "discord" in low or "logo" in low:
            continue
        return src
    return ""


def validate_repo_id(repo_id: str) -> str:
    """Return a sanitized repo id or raise. Guards every path built from it."""
    repo_id = (repo_id or "").strip().strip("/")
    if not repo_id or not _REPO_ID_RE.match(repo_id):
        raise HubError(f"invalid model repo id: {repo_id!r}")
    # No traversal / hidden segments even though the charset makes them hard.
    for seg in repo_id.split("/"):
        if seg == ".." or seg.startswith("."):
            raise HubError(f"invalid model repo id: {repo_id!r}")
    return repo_id


def endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT", "").strip() or DEFAULT_ENDPOINT).rstrip("/")


def _token() -> Optional[str]:
    tok = os.environ.get("HF_TOKEN", "").strip()
    return tok or None


def _headers() -> Dict[str, str]:
    h = {"User-Agent": "sable-modelhub"}
    tok = _token()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _get(url: str, **kw: Any) -> httpx.Response:
    try:
        r = httpx.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT,
                      follow_redirects=True, **kw)
    except httpx.HTTPError as e:
        raise HubError(f"hub request failed: {e}") from e
    if r.status_code in (401, 403):
        raise HubError(
            f"repo requires authorization (HTTP {r.status_code}) — "
            "set HF_TOKEN for gated models")
    if r.status_code == 404:
        raise HubError("model repo not found on the hub")
    if r.status_code >= 400:
        raise HubError(f"hub error (HTTP {r.status_code})")
    return r


# ------------------------------------------------------------------ search

#: UI capability chips -> HF pipeline_tag filter values.
CAPABILITIES: Dict[str, str] = {
    "conversational": "text-generation",
    "vision": "image-text-to-text",
    "embeddings": "sentence-similarity",
    "audio": "automatic-speech-recognition",
    "rerank": "text-ranking",
}

#: UI sort chips -> HF API sort fields.
SORTS: Dict[str, str] = {
    "downloads": "downloads",
    "likes": "likes",
    "newest": "lastModified",
}


def _is_gguf(entry: Dict[str, Any]) -> bool:
    tags = entry.get("tags") or []
    rid = entry.get("id") or ""
    return "gguf" in tags or bool(entry.get("gguf")) or rid.upper().endswith("-GGUF")


def _summary(entry: Dict[str, Any]) -> Dict[str, Any]:
    """One search/feed row (Discover card fields)."""
    gguf_meta = entry.get("gguf") or {}
    st = entry.get("safetensors") or {}
    params_total = 0
    if isinstance(st, dict):
        # safetensors expand: {parameters: {F8:...}, total: <bytes-ish>}
        params_total = int(st.get("total") or 0)
    card = entry.get("cardData") or {}
    return {
        "repo_id": entry.get("id", ""),
        "name": (entry.get("id") or "").split("/")[-1],
        "author": entry.get("author") or (entry.get("id") or "").split("/")[0],
        "downloads": int(entry.get("downloads") or 0),
        "likes": int(entry.get("likes") or 0),
        "last_modified": entry.get("lastModified") or "",
        "gguf": _is_gguf(entry),
        "pipeline_tag": entry.get("pipeline_tag") or "",
        "architecture": gguf_meta.get("architecture") or "",
        "context_length": int(gguf_meta.get("context_length") or 0),
        # GGUF total weight bytes across all quants (hub-computed).
        "size_bytes": int(gguf_meta.get("total") or 0) or None,
        "parameters": params_total or None,
        "license": str(card.get("license") or "").split(",")[0].strip(),
    }


def search_models(query: str = "", limit: int = 30, gguf_only: bool = False,
                  capability: str = "", sort: str = "downloads",
                  ) -> List[Dict[str, Any]]:
    """Text search the Hub (the Discover tab).

    ``capability`` is a key of :data:`CAPABILITIES`; ``sort`` a key of
    :data:`SORTS`. Empty query returns the overall popular models.
    """
    params = [
        ("limit", str(max(1, min(int(limit), 100)))),
        ("sort", SORTS.get(sort, "downloads")),
        ("direction", "-1"),
        ("expand", "downloads"),
        ("expand", "likes"),
        ("expand", "lastModified"),
        ("expand", "pipeline_tag"),
        ("expand", "gguf"),
        ("expand", "safetensors"),
        ("expand", "cardData"),
    ]
    if gguf_only:
        params.append(("filter", "gguf"))
    pipe = CAPABILITIES.get(capability)
    if pipe:
        params.append(("filter", pipe))
    if query and query.strip():
        params.insert(0, ("search", query.strip()))
    r = _get(f"{endpoint()}/api/models", params=params)
    return [_summary(e) for e in r.json()]


def latest_feed(limit: int = 15) -> List[Dict[str, Any]]:
    """Most recently updated GGUF models on the Hub (the Discover rail).

    Sorted by last-modified with the GGUF filter applied — freshly quantized
    tool-call-capable models surface first, which is what this pipeline serves.
    """
    params = [
        ("filter", "gguf"),
        ("sort", "lastModified"),
        ("direction", "-1"),
        ("limit", str(max(1, min(int(limit), 50)))),
        ("expand", "downloads"),
        ("expand", "likes"),
        ("expand", "lastModified"),
        ("expand", "pipeline_tag"),
        ("expand", "gguf"),
    ]
    r = _get(f"{endpoint()}/api/models", params=params)
    return [_summary(e) for e in r.json()]


# ---------------------------------------------------------------- variants

def _quant_of_path(rel: str) -> Optional[str]:
    """Quant tag naming a GGUF file: basename decides, else nearest parent dir
    (repos like testorg/*-GGUF nest quants in directory names)."""
    parts = rel.replace("\\", "/").split("/")
    for cand in reversed(parts):
        m = _QUANT_RE.search(cand)
        if m:
            ud = (m.group(1) or "").upper()
            return (ud + "-" if ud else "") + m.group(2).upper()
    return None


def repo_files(repo_id: str) -> Tuple[str, List[Dict[str, Any]]]:
    """(commit_sha, siblings[{rfilename,size}]) for a model repo."""
    repo_id = validate_repo_id(repo_id)
    r = _get(f"{endpoint()}/api/models/{repo_id}", params={"blobs": "true"})
    data = r.json()
    sha = data.get("sha") or "main"
    sibs = [
        {"rfilename": s.get("rfilename", ""), "size": int(s.get("size") or 0)}
        for s in (data.get("siblings") or []) if s.get("rfilename")
    ]
    return sha, sibs


def list_variants(repo_id: str) -> Dict[str, Any]:
    """Quantization variants for a repo (GGUF), or one 'full' pseudo-variant.

    Returns {repo_id, gguf, default_variant, variants:[{variant, files,
    size_bytes, shard_count}]}. ``default_variant`` prefers Q4_K_M-style
    medium quants, else the smallest — studio picks similarly so first-click
    downloads stay sane.

    Companion GGUFs (mmproj vision projectors, MTP drafters) are never a
    variant of their own: they ride along with EVERY quant download so a
    served model always has its projector/drafter available locally (studio
    does the same — mmproj lands next to the quant, MTP/ alongside).
    """
    sha, sibs = repo_files(repo_id)

    ggufs = [s for s in sibs if s["rfilename"].lower().endswith(".gguf")
             and not _is_companion_rel(s["rfilename"])]
    if not ggufs:
        # Non-GGUF repo: single variant of everything (training base model).
        total = sum(s["size"] for s in sibs)
        return {
            "repo_id": repo_id, "gguf": False, "sha": sha,
            "default_variant": "full",
            "variants": [{"variant": "full",
                          "files": [s["rfilename"] for s in sibs],
                          "size_bytes": total, "shard_count": 0}],
        }

    groups: Dict[str, Dict[str, Any]] = {}
    for s in ggufs:
        rel = s["rfilename"]
        quant = _quant_of_path(rel) or "other"
        g = groups.setdefault(quant, {"variant": quant, "files": [],
                                      "size_bytes": 0, "shard_count": 0})
        g["files"].append(rel)
        g["size_bytes"] += s["size"]
        if _SHARD_RE.search(rel):
            g["shard_count"] += 1

    # Companion GGUFs (mmproj projectors, MTP drafters — including repos that
    # nest them in an MTP/ directory) ride along with EVERY quant variant: a
    # downloaded model must always have its projector/drafter on disk, or the
    # serve path silently loses vision/speculative support.
    companions = [s for s in sibs
                  if s["rfilename"].lower().endswith(".gguf")
                  and _is_companion_rel(s["rfilename"])]

    out = sorted(groups.values(), key=lambda g: g["size_bytes"])
    for g in out:
        g["files"].extend(c["rfilename"] for c in companions)
        g["size_bytes"] += sum(c["size"] for c in companions)

    # Default: a medium quant (Q4_K_M / Q4_K_S / UD-Q4 variants), else smallest.
    def _pref(v: Dict[str, Any]) -> Tuple[int, int]:
        tier = 0 if re.search(r"Q4_K_[MS]$|Q4_K_XL$", v["variant"]) else 1
        return (tier, v["size_bytes"])

    default = min(out, key=_pref)["variant"] if out else ""
    return {"repo_id": repo_id, "gguf": True, "sha": sha,
            "default_variant": default, "variants": out}


def model_detail(repo_id: str) -> Dict[str, Any]:
    """Everything the detail pane shows (studio's discover right-hand page):
    card metadata + quant variants + README markdown + local install state.
    """
    repo_id = validate_repo_id(repo_id)
    r = _get(f"{endpoint()}/api/models/{repo_id}", params={
        "blobs": "true",
        "expand": ["cardData", "downloads", "likes", "lastModified",
                   "pipeline_tag", "gguf", "safetensors", "gated"],
    })
    data = r.json()
    card = data.get("cardData") or {}
    gguf_meta = data.get("gguf") or {}
    base_models: List[str] = []
    bm = card.get("base_model")
    if isinstance(bm, str):
        base_models = [bm]
    elif isinstance(bm, list):
        base_models = [str(x) for x in bm]
    # README (studio renders it under the download box); bounded + tolerant.
    readme = ""
    try:
        rr = httpx.get(f"{endpoint()}/{repo_id}/raw/main/README.md",
                       headers=_headers(), timeout=REQUEST_TIMEOUT,
                       follow_redirects=True)
        if rr.status_code == 200:
            readme = rr.text[:60_000]
    except httpx.HTTPError:
        pass
    variants_info = list_variants(repo_id)
    return {
        "repo_id": repo_id,
        "name": repo_id.split("/")[-1],
        "author": (data.get("author") or repo_id.split("/")[0]),
        "downloads": int(data.get("downloads") or 0),
        "likes": int(data.get("likes") or 0),
        "last_modified": data.get("lastModified") or "",
        "gguf": _is_gguf(data),
        "pipeline_tag": data.get("pipeline_tag") or "",
        "architecture": gguf_meta.get("architecture") or "",
        "context_length": int(gguf_meta.get("context_length") or 0),
        "license": str(card.get("license") or "").split(",")[0].strip(),
        "base_models": base_models,
        "tags": [t for t in (data.get("tags") or []) if ":" not in t][:12],
        "gated": bool(data.get("gated")),
        "gemma": bool(card.get("gemma")),
        "variants": variants_info["variants"],
        "default_variant": variants_info["default_variant"],
        "readme": readme,
        # Card banner image (studio shows card art above the fold). Raw src:
        # may be repo-relative or absolute — the UI rewrites it through
        # /api/hub/image so the same-origin CSP can display it.
        "image": str(card.get("image") or "") or _first_card_image(readme),
    }


def _files_for_variant(repo_id: str, variant: str) -> Tuple[str, List[Dict[str, Any]]]:
    """(sha, sibling subset) selected by variant name ('' => default)."""
    info = list_variants(repo_id)
    variants = info["variants"]
    if not variant or variant == "default":
        variant = info["default_variant"]
    chosen = next((v for v in variants if v["variant"] == variant), None)
    if chosen is None:
        raise HubError(f"unknown variant {variant!r}; available: "
                       + ", ".join(v["variant"] for v in variants))
    want = set(chosen["files"])
    sibs = [s for s in repo_files(info["repo_id"])[1] if s["rfilename"] in want]
    return info["sha"], sibs


# --------------------------------------------------------------- downloads

@dataclass
class DownloadJob:
    job_id: str
    repo_id: str
    variant: str
    target_dir: str
    state: str = "queued"          # queued|running|done|error|cancelled
    total_bytes: int = 0
    done_bytes: int = 0
    error: Optional[str] = None
    created_at: str = field(default_factory=_utcnow)
    finished_at: Optional[str] = None
    files_done: List[str] = field(default_factory=list)
    _cancel: threading.Event = field(default_factory=threading.Event,
                                     repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id, "repo_id": self.repo_id,
            "variant": self.variant, "target_dir": self.target_dir,
            "state": self.state, "total_bytes": self.total_bytes,
            "done_bytes": self.done_bytes, "error": self.error,
            "created_at": self.created_at, "finished_at": self.finished_at,
            "files_done": list(self.files_done),
        }


class ModelHub:
    """Hub client + download manager bound to one base model folder.

    ``base_dir`` overrides resolution (tests); otherwise the resolved
    ``models_dir`` / HF cache, falling back to ``<home>/models``.
    """

    def __init__(self, models_dir: Optional[str] = None,
                 home: Optional[Path] = None):
        self._models_dir_arg = models_dir
        self._home = Path(home) if home else None
        self._lock = threading.Lock()
        self._jobs: Dict[str, DownloadJob] = {}

    # ------------------------------------------------------------- folders

    def base_dir(self) -> Path:
        """Where repos live; created on demand (download time only).

        An *explicit* models_dir wins even when it doesn't exist yet (we
        create it on first download) — resolve_models_dir deliberately returns
        None for a missing explicit dir to keep discovery honest, which would
        otherwise silently redirect downloads to the HF cache fallback.
        """
        if self._models_dir_arg:
            return Path(self._models_dir_arg).expanduser()
        resolved = resolve_models_dir(None)
        if resolved is not None:
            return resolved
        home = self._home
        if home is None:
            from ..store.paths import resolve_sable_home
            home = resolve_sable_home()
        return Path(home) / "models"

    def repo_cache_dir(self, repo_id: str) -> Path:
        repo_id = validate_repo_id(repo_id)
        return self.base_dir() / ("models--" + repo_id.replace("/", "--"))

    def snapshot_dir(self, repo_id: str, sha: str) -> Path:
        return self.repo_cache_dir(repo_id) / "snapshots" / sha

    # ------------------------------------------------------------- search

    def search(self, query: str = "", limit: int = 30, gguf_only: bool = False,
               capability: str = "", sort: str = "downloads",
               ) -> List[Dict[str, Any]]:
        return search_models(query, limit=limit, gguf_only=gguf_only,
                             capability=capability, sort=sort)

    def feed(self, limit: int = 15) -> List[Dict[str, Any]]:
        return latest_feed(limit)

    def variants(self, repo_id: str) -> Dict[str, Any]:
        return list_variants(repo_id)

    def detail(self, repo_id: str) -> Dict[str, Any]:
        return model_detail(repo_id)

    # ----------------------------------------------------------- installed

    def installed(self) -> List[Dict[str, Any]]:
        """Repos present under the base dir (hub layout + flat GGUF dirs)."""
        base = self.base_dir()
        out: List[Dict[str, Any]] = []
        if not base.is_dir():
            return out
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and entry.name.startswith("models--"):
                repo = entry.name[len("models--"):].replace("--", "/")
                snaps = entry / "snapshots"
                revisions, size = [], 0
                if snaps.is_dir():
                    for snap in sorted(snaps.iterdir()):
                        if snap.is_dir():
                            revisions.append(snap.name)
                            size += _dir_size_dedupe(snap)
                # Quant tags come from the path relative to the repo dir so
                # directory-named quants (Q4_K_M/model.gguf) are recognized.
                quants = sorted({q for q in (
                    _quant_of_path(str(p.relative_to(entry)))
                    for p in entry.rglob("*.gguf")) if q})
                out.append({"repo_id": repo, "path": str(entry),
                            "layout": "hub", "revisions": revisions,
                            "size_bytes": size, "quants": quants[:12],
                            "complete": bool(revisions)})
        return out

    def delete(self, repo_id: str) -> Dict[str, Any]:
        """Remove a downloaded repo — confined to the base dir (no traversal)."""
        target = self.repo_cache_dir(repo_id)  # validates + anchors under base
        base = self.base_dir().resolve()
        resolved = target.resolve()
        if not resolved.is_relative_to(base):
            raise HubError("refusing to delete outside the models folder")
        if not resolved.is_dir():
            raise HubError(f"{repo_id} is not downloaded under {base}")
        size = _dir_size_dedupe(resolved)
        shutil.rmtree(resolved)
        return {"deleted": validate_repo_id(repo_id), "freed_bytes": size}

    # ----------------------------------------------------------- downloads

    def start_download(self, repo_id: str, variant: str = "") -> str:
        """Queue a background download; returns the job id (studio's
        POST /models/download pattern: 202 + poll)."""
        repo_id = validate_repo_id(repo_id)
        with self._lock:
            # Single-flight per (repo, variant): reject duplicate in-flight.
            for job in self._jobs.values():
                if (job.repo_id == repo_id and job.variant == variant
                        and job.state in ("queued", "running")):
                    return job.job_id
            job = DownloadJob(job_id="dl-" + uuid.uuid4().hex[:8],
                              repo_id=repo_id, variant=variant or "",
                              target_dir="")
            self._jobs[job.job_id] = job
        t = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        t.start()
        return job.job_id

    def job(self, job_id: str) -> DownloadJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise HubError(f"unknown download job: {job_id}")
        return job

    def jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [j.to_dict() for j in
                    sorted(self._jobs.values(), key=lambda j: j.created_at)]

    def cancel(self, job_id: str) -> Dict[str, Any]:
        job = self.job(job_id)
        if job.state in ("done", "error"):
            return job.to_dict()
        job._cancel.set()
        if job.state == "queued":
            job.state = "cancelled"
            job.finished_at = _utcnow()
        return job.to_dict()

    def prune_finished(self, keep: int = 20) -> None:
        """Drop old finished jobs so long-lived dashboard processes stay small."""
        with self._lock:
            finished = [j for j in self._jobs.values()
                        if j.state in ("done", "error", "cancelled")]
            if len(finished) <= keep:
                return
            for j in sorted(finished, key=lambda j: j.created_at)[:-keep]:
                self._jobs.pop(j.job_id, None)

    # -------------------------------------------------------------- worker

    def _run_job(self, job: DownloadJob) -> None:
        try:
            sha, sibs = _files_for_variant(job.repo_id, job.variant)
            if job._cancel.is_set():
                raise _Cancelled()
            job.variant = job.variant or ""
            snap = self.snapshot_dir(job.repo_id, sha)
            snap.mkdir(parents=True, exist_ok=True)
            job.target_dir = str(snap)
            job.total_bytes = sum(s["size"] for s in sibs)
            job.state = "running"
            for sib in sibs:
                if job._cancel.is_set():
                    raise _Cancelled()
                self._download_file(job, sha, sib)
            self._write_manifest(job, sha, sibs)
            job.state = "done"
        except _Cancelled:
            job.state = "cancelled"
        except HubError as e:
            job.state = "error"
            job.error = str(e)
        except Exception as e:  # noqa: BLE001 — jobs must never die silently
            job.state = "error"
            job.error = f"{type(e).__name__}: {e}"
        finally:
            job.finished_at = _utcnow()

    def _download_file(self, job: DownloadJob, sha: str,
                       sib: Dict[str, Any]) -> None:
        rel = sib["rfilename"]
        dest = (self.snapshot_dir(job.repo_id, sha) / rel).resolve()
        if not dest.is_relative_to(self.base_dir().resolve()):
            raise HubError(f"unsafe path in repo file list: {rel!r}")
        part = dest.with_name(dest.name + ".part")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and (not sib["size"] or dest.stat().st_size == sib["size"]):
            job.done_bytes += sib["size"]
            job.files_done.append(rel)
            return

        headers = _headers()
        resume_from = part.stat().st_size if part.exists() else 0
        if resume_from and sib["size"] and resume_from >= sib["size"]:
            part.unlink(missing_ok=True)
            resume_from = 0
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"

        url = (f"{endpoint()}/{job.repo_id}/resolve/main/{rel}"
               .replace("resolve/main", f"resolve/{sha}", 1))
        mode = "ab" if resume_from else "wb"
        with httpx.stream("GET", url, headers=headers, timeout=REQUEST_TIMEOUT,
                          follow_redirects=True) as r:
            if r.status_code == 416 and part.exists():  # already complete
                part.unlink(missing_ok=True)
                job.files_done.append(rel)
                return
            if r.status_code in (401, 403):
                raise HubError(f"{rel}: authorization required — set HF_TOKEN")
            if r.status_code not in (200, 206):
                raise HubError(f"{rel}: download failed (HTTP {r.status_code})")
            if resume_from and r.status_code == 200:
                # Server ignored Range: restart the file from scratch.
                mode, resume_from = "wb", 0
            try:
                with open(part, mode) as fh:
                    # iter_raw() with NO chunk size: yields bytes as they
                    # arrive. Chunked variants buffer until a full size and
                    # discard the remainder on a mid-stream drop — which would
                    # defeat the .part resume below.
                    for chunk in r.iter_raw():
                        if job._cancel.is_set():
                            raise _Cancelled()
                        fh.write(chunk)
                        job.done_bytes += len(chunk)
            except httpx.HTTPError as e:
                raise HubError(
                    f"{rel}: connection dropped after "
                    f"{part.stat().st_size if part.exists() else 0} bytes — "
                    "retry the download to resume") from e
        # Size check (when the hub reported one) then promote .part -> final.
        if sib["size"] and part.stat().st_size != sib["size"]:
            raise HubError(
                f"{rel}: incomplete download "
                f"({part.stat().st_size}/{sib['size']} bytes) — retry resumes it")
        os.replace(part, dest)
        job.files_done.append(rel)

    def _write_manifest(self, job: DownloadJob, sha: str,
                        sibs: List[Dict[str, Any]]) -> None:
        # HF-style refs/<sha> marker + our own manifest for inventory honesty.
        refs = self.repo_cache_dir(job.repo_id) / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(sha + "\n")
        manifest = {
            "repo_id": job.repo_id, "sha": sha, "variant": job.variant,
            "files": [{"rfilename": s["rfilename"], "size": s["size"]}
                      for s in sibs],
            "downloaded_at": _utcnow(), "tool": "sable/modelhub",
        }
        (self.snapshot_dir(job.repo_id, sha) / "sable_download.json"
         ).write_text(json.dumps(manifest, indent=2))

    def download_sync(self, repo_id: str, variant: str = "",
                      progress_cb: Optional[Any] = None) -> DownloadJob:
        """Inline download for the CLI (no thread); same code path as jobs."""
        repo_id = validate_repo_id(repo_id)
        job = DownloadJob(job_id="dl-" + uuid.uuid4().hex[:8],
                          repo_id=repo_id, variant=variant or "", target_dir="")
        self._run_job(job)
        if progress_cb:
            progress_cb(job.to_dict())
        return job


class _Cancelled(Exception):
    pass


def _dir_size_dedupe(root: Path) -> int:
    """Sum file sizes under ``root``, counting blob symlinks once (HF cache)."""
    total, seen = 0, set()
    try:
        for p in root.rglob("*"):
            try:
                if not p.is_file():
                    continue
                rp = p.resolve()
                key = str(rp)
                if key in seen:
                    continue
                seen.add(key)
                total += rp.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total

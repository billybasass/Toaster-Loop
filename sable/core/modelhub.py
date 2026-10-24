"""Resource Hub: search, download and manage models, datasets and MCP
servers from Hugging Face.

One hub service for every resource kind sable can use, distilled to its
dependency budget — plain ``httpx`` against the public HF Hub REST API, no
``huggingface_hub`` import. Three kinds share one code path:

* ``models``   — GGUF quants (llama.cpp), safetensors (vLLM and friends),
  audio/vision checkpoints — anything on the Hub.
* ``datasets`` — training corpora (``/api/datasets``).
* ``spaces``   — repos tagged ``mcp``: Model Context Protocol servers
  (``/api/spaces?filter=mcp``).

Four capabilities:

* **Search** — ``GET /api/<kind>`` with a text query, sorted by downloads,
  annotated GGUF or not (tags/name heuristic like studio's catalog split).
* **Variants** — repo file listing grouped into quantization variants
  (Q4_K_M, UD-Q5_K_S, ...), shard-aware: ``X-00001-of-00002.gguf`` files fold
  into one variant with summed sizes. Non-GGUF repos (training base models,
  datasets, spaces) expose a single pseudo-variant of every file.
* **Download** — background worker threads writing the standard HF hub-cache
  layout ``<base>/models--<org>--<name>/snapshots/<sha>/<relpath>`` so files
  land where ``core/model_discovery.py`` already looks (serve + train pickers
  need zero refresh logic). Datasets and spaces use the same snapshot layout
  under separate roots (``datasets--``/``spaces--`` prefixes below
  ``<SABLE_HOME>/hub``) so they never pollute model discovery. Resumable:
  bytes stream into ``*.part`` siblings and continue with an HTTP ``Range``
  request on retry — the same trick as studio's resumable_partials worker.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from .model_discovery import resolve_models_dir

DEFAULT_ENDPOINT = "https://huggingface.co"
REQUEST_TIMEOUT = 30.0

#: Resource kinds the hub can browse/download. The plural doubles as the HF
#: REST path segment (``/api/models``, ``/api/datasets``, ``/api/spaces``);
#: content routes (resolve/raw) nest datasets/spaces one level deeper than
#: models (``/{kind}/{repo}/resolve/...``).
KINDS = ("models", "datasets", "spaces")

#: HF-hub-cache directory prefixes per kind (same convention as
#: huggingface_hub, so external tools see a familiar layout).
_CACHE_PREFIX = {"models": "models--", "datasets": "datasets--",
                 "spaces": "spaces--"}


class HubError(RuntimeError):
    """User-facing hub failure (bad repo id, HTTP error, unsafe path...)."""


def normalize_kind(kind: str) -> str:
    """Validate/alias a resource kind; unknown values raise HubError."""
    k = (kind or "models").strip().lower()
    aliases = {"model": "models", "dataset": "datasets", "space": "spaces",
               "mcp": "spaces", "mcp-server": "spaces", "": "models"}
    k = aliases.get(k, k)
    if k not in KINDS:
        raise HubError(f"unknown hub resource kind: {kind!r}")
    return k


def _url_seg(kind: str) -> str:
    """Path prefix for the *content* routes (resolve/raw): '' | 'datasets/'
    | 'spaces/'. The metadata API is always /api/<kind>/... so no prefix."""
    return "" if kind == "models" else kind + "/"


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


# ---------------------------------------------------------------- images
#
# Resource cards render images (sample grids, banners). The dashboard CSP is
# default-src 'self', so remote <img> tags cannot load directly — instead the
# UI rewrites image URLs to GET /api/hub/image, a same-origin proxy that only
# fetches image bytes from public hosts (allow-listed hub CDNs skip the DNS
# screen; everything else must resolve to a public address).

MAX_IMAGE_BYTES = 8 * 1024 * 1024
_IMAGE_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}
#: Always-trusted hosts (hub + GitHub raw). Any *other* public https host is
#: still allowed through the proxy after the private-address screen below —
#: model cards legitimately embed art on many CDNs (aliyuncs, gitbook,
#: cloudfront...), and blocking those renders banners/README images broken.
_ALLOWED_IMAGE_HOSTS = ("huggingface.co", "hf.co", "githubusercontent.com")


def _host_is_public(host: str) -> bool:
    """False for loopback/private/link-local hosts (SSRF screen). Names we
    cannot resolve are rejected too — the proxy never dials out blindly."""
    import ipaddress
    import socket

    try:
        infos = socket.getaddrinfo(host, None)
    except (OSError, UnicodeError):
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast):
            return False
    return True


def fetch_image(repo_id: str = "", path: str = "", url: str = "",
                kind: str = "models") -> Tuple[bytes, str]:
    """Fetch one card image; returns (bytes, content_type).

    Two shapes: ``repo_id`` + repo-relative ``path`` (proxied through the
    configured endpoint's resolve route), or an absolute https ``url``. SVG is
    rejected outright (scripts); size is capped at MAX_IMAGE_BYTES; foreign
    hosts must be public https (no network-internal targets)."""
    if url:
        from urllib.parse import urlparse

        u = urlparse(url)
        host = (u.hostname or "").lower()
        if u.scheme != "https" or not host:
            raise HubError("image host not allowed")
        trusted = any(host == h or host.endswith("." + h)
                      for h in _ALLOWED_IMAGE_HOSTS)
        if not (trusted or _host_is_public(host)):
            raise HubError("image host not allowed")
        target = url
    else:
        kind = normalize_kind(kind)
        repo_id = validate_repo_id(repo_id)
        p = str(path or "").replace("\\", "/").lstrip("/")
        if not p or ".." in Path(p).parts:
            raise HubError("invalid image path")
        if Path(p).suffix.lower() not in _IMAGE_TYPES:
            raise HubError("unsupported image type")
        target = (f"{endpoint()}/{_url_seg(kind)}{repo_id}"
                  f"/resolve/main/{p}")
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
    h = {"User-Agent": "sable-resourcehub"}
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
        raise HubError("repo not found on the hub")
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


def _summary(entry: Dict[str, Any], kind: str = "models") -> Dict[str, Any]:
    """One search/feed row (Discover card fields). Spaces have no download
    counter — likes/createdAt stand in so the rails stay informative."""
    gguf_meta = entry.get("gguf") or {}
    st = entry.get("safetensors") or {}
    params_total = 0
    if isinstance(st, dict):
        # safetensors expand: {parameters: {F8:...}, total: <bytes-ish>}
        params_total = int(st.get("total") or 0)
    card = entry.get("cardData") or {}
    return {
        "kind": kind,
        "repo_id": entry.get("id", ""),
        "name": (entry.get("id") or "").split("/")[-1],
        "author": entry.get("author") or (entry.get("id") or "").split("/")[0],
        "downloads": int(entry.get("downloads") or 0),
        "likes": int(entry.get("likes") or 0),
        "last_modified": (entry.get("lastModified")
                          or entry.get("createdAt") or ""),
        "gguf": _is_gguf(entry) if kind == "models" else False,
        "pipeline_tag": entry.get("pipeline_tag") or "",
        "sdk": entry.get("sdk") or "",
        "architecture": gguf_meta.get("architecture") or "",
        "context_length": int(gguf_meta.get("context_length") or 0),
        # GGUF total weight bytes across all quants (hub-computed).
        "size_bytes": int(gguf_meta.get("total") or 0) or None,
        "parameters": params_total or None,
        "license": str(card.get("license") or "").split(",")[0].strip(),
    }


def search_models(query: str = "", limit: int = 30, gguf_only: bool = False,
                  capability: str = "", sort: str = "downloads",
                  kind: str = "models") -> List[Dict[str, Any]]:
    """Text search the Hub (the Discover tab).

    ``kind`` selects models/datasets/spaces (spaces are pre-filtered to the
    ``mcp`` tag — MCP servers live in Spaces). ``capability`` is a key of
    :data:`CAPABILITIES` (models only); ``sort`` a key of :data:`SORTS`.
    Empty query returns the overall popular resources.
    """
    kind = normalize_kind(kind)
    # Spaces have no download counter; "most downloads" maps to likes there.
    sort_field = SORTS.get(sort, "downloads")
    if kind == "spaces" and sort_field == "downloads":
        sort_field = "likes"
    params = [
        ("limit", str(max(1, min(int(limit), 100)))),
        ("sort", sort_field),
        ("direction", "-1"),
    ]
    # The API rejects expand fields it doesn't know per kind (HTTP 400), so
    # each kind expands exactly the keys it supports.
    if kind == "models":
        params += [("expand", "downloads"), ("expand", "likes"),
                   ("expand", "lastModified"), ("expand", "pipeline_tag"),
                   ("expand", "cardData"), ("expand", "gguf"),
                   ("expand", "safetensors")]
        if gguf_only:
            params.append(("filter", "gguf"))
        pipe = CAPABILITIES.get(capability)
        if pipe:
            params.append(("filter", pipe))
    elif kind == "datasets":
        params += [("expand", "downloads"), ("expand", "likes"),
                   ("expand", "lastModified")]
    else:  # spaces
        # The MCP catalogue is Spaces tagged `mcp`; keep that filter on every
        # query so the tab never drifts into plain demos.
        params += [("filter", "mcp"), ("expand", "likes")]
    if query and query.strip():
        params.insert(0, ("search", query.strip()))
    r = _get(f"{endpoint()}/api/{kind}", params=params)
    return [_summary(e, kind) for e in r.json()]


def latest_feed(limit: int = 15, kind: str = "models") -> List[Dict[str, Any]]:
    """Most recently updated resources on the Hub (the Discover rail).

    Sorted by last-modified. For models this surfaces freshly quantized /
    freshly uploaded tool-call-capable models first regardless of format —
    GGUF is no longer privileged since vLLM and stable-diffusion.cpp run
    safetensors and other weights natively.
    """
    kind = normalize_kind(kind)
    params = [
        ("sort", "lastModified"),
        ("direction", "-1"),
        ("limit", str(max(1, min(int(limit), 50)))),
    ]
    if kind == "models":
        params += [("expand", "downloads"), ("expand", "likes"),
                   ("expand", "lastModified"), ("expand", "pipeline_tag"),
                   ("expand", "gguf")]
    elif kind == "datasets":
        params += [("expand", "downloads"), ("expand", "likes"),
                   ("expand", "lastModified")]
    else:  # spaces
        params += [("filter", "mcp"), ("expand", "likes")]
    r = _get(f"{endpoint()}/api/{kind}", params=params)
    return [_summary(e, kind) for e in r.json()]


# ------------------------------------------------- MCP registries (mcp dir)
#
# Second/third MCP catalogues besides HF Spaces:
#
# * **Official MCP Registry** — https://registry.modelcontextprotocol.io,
#   the vendor-neutral community registry (Generic Registry API). No auth.
# * **PulseMCP** — https://www.pulsemcp.com; its v0.1 API requires an API key
#   sent as X-API-Key (read from PULSEMCP_API_KEY; signup is now contact-based).
#
# Entries in both are *directory listings* of external servers (npm/PyPI/
# docker packages + hosted URLs), not HF repos, so they link out rather than
# download into the hub cache.

PULSEMCP_ENDPOINT = "https://api.pulsemcp.com"
MCP_REGISTRY_ENDPOINT = os.environ.get(
    "MCP_REGISTRY_URL", "https://registry.modelcontextprotocol.io").rstrip("/")


def _registry_get(path: str, params: Sequence[Tuple[str, str]] = ()
                  ) -> Dict[str, Any]:
    try:
        r = httpx.get(f"{MCP_REGISTRY_ENDPOINT}{path}", params=list(params),
                      headers={"User-Agent": "sable-resourcehub"},
                      timeout=REQUEST_TIMEOUT, follow_redirects=True)
    except httpx.HTTPError as e:
        raise HubError(f"mcp registry request failed: {e}") from e
    if r.status_code == 404:
        raise HubError("server not found in the MCP registry")
    if r.status_code >= 400:
        raise HubError(f"mcp registry error (HTTP {r.status_code})")
    return r.json()


def _registry_row(e: Dict[str, Any]) -> Dict[str, Any]:
    """One Generic Registry API server record -> Discover row."""
    s = e.get("server") or {}
    meta = (e.get("_meta") or {}).get(
        "io.modelcontextprotocol.registry/official") or {}
    name = str(s.get("name") or "")
    repo = (s.get("repository") or {}).get("url") or ""
    remotes = s.get("remotes") or []
    return {
        "kind": "mcp-registry",
        "repo_id": name,
        "name": str(s.get("title") or name.split("/")[-1] or name),
        "author": name.split("/")[0] if "/" in name else "",
        "description": str(s.get("description") or "")[:400],
        "version": str(s.get("version") or ""),
        "transports": sorted({str(rm.get("type")) for rm in remotes
                              if rm.get("type")}),
        "remote_url": str(remotes[0].get("url")) if remotes else "",
        "github_url": repo if repo.startswith("http") else "",
        "downloads": 0,
        # No per-server web page in the official registry; GitHub (when the
        # record has a repository) is the link out.
        "url": "",
        "published_at": str(meta.get("publishedAt") or ""),
        "last_modified": str(meta.get("updatedAt") or ""),
    }


def search_mcp_registry(query: str = "", limit: int = 40) -> List[Dict[str, Any]]:
    """Search the official MCP Registry (no auth). Keeps only latest-version
    rows when the registry flags them, so one server appears once."""
    params = [("limit", str(max(1, min(int(limit), 100))))]
    if query and query.strip():
        params.append(("search", query.strip()))
    data = _registry_get("/v0.1/servers", params)
    rows = []
    for e in data.get("servers") or []:
        meta = (e.get("_meta") or {}).get(
            "io.modelcontextprotocol.registry/official") or {}
        if meta and meta.get("isLatest") is False:
            continue  # older version of a server already listed
        rows.append(_registry_row(e))
    return rows


def mcp_registry_detail(server_name: str) -> Dict[str, Any]:
    """One official-registry server (latest version), with package/remote
    setup info rendered as a card."""
    from urllib.parse import quote

    name = str(server_name or "").strip()
    if not name or ".." in name or any(c.isspace() for c in name):
        raise HubError(f"invalid registry server name: {server_name!r}")
    data = _registry_get(
        f"/v0.1/servers/{quote(name, safe='')}/versions/latest")
    row = _registry_row(data)
    s = data.get("server") or {}
    lines = [f"# {s.get('title') or s.get('name', '')}", "",
             str(s.get("description") or "")]
    if s.get("version"):
        lines += ["", f"Version: `{s['version']}`"]
    for rm in s.get("remotes") or []:
        lines.append(f"- remote ({rm.get('type', '?')}): `{rm.get('url', '')}`")
    pkg = s.get("packageVersions") or []
    for p in pkg[:4]:
        lines.append(f"- package ({p.get('registryType', '?')}): "
                     f"`{p.get('identifier', '')}`")
    repo = (s.get("repository") or {}).get("url")
    if repo:
        lines.append(f"- repository: {repo}")
    row["readme"] = "\n".join(lines)[:24000]
    return row


def _pulsemcp_get(path: str, params: Sequence[Tuple[str, str]]
                  ) -> Dict[str, Any]:
    key = os.environ.get("PULSEMCP_API_KEY", "").strip()
    if not key:
        raise HubError(
            "PulseMCP requires a free API key — sign up at https://www."
            "pulsemcp.com/api and set PULSEMCP_API_KEY in the environment")
    headers = {"User-Agent": "sable-resourcehub", "X-API-Key": key}
    try:
        r = httpx.get(f"{PULSEMCP_ENDPOINT}{path}", params=list(params),
                      headers=headers, timeout=REQUEST_TIMEOUT,
                      follow_redirects=True)
    except httpx.HTTPError as e:
        raise HubError(f"pulsemcp request failed: {e}") from e
    if r.status_code in (401, 403):
        raise HubError("pulsemcp rejected the API key (HTTP "
                       f"{r.status_code}) — check PULSEMCP_API_KEY")
    if r.status_code == 404:
        raise HubError("server not found on PulseMCP")
    if r.status_code >= 400:
        raise HubError(f"pulsemcp error (HTTP {r.status_code})")
    return r.json()


def _mcp_dir_row(e: Dict[str, Any]) -> Dict[str, Any]:
    """One PulseMCP server record -> Discover row. Tolerant of the v0beta and
    v0.1 field spellings (name/tagline|description, downloads counters)."""
    sid = str(e.get("id") or e.get("slug") or "")
    name = str(e.get("name") or sid)
    desc = str(e.get("tagline") or e.get("description") or "").strip()
    cats = e.get("categories") or []
    if cats and isinstance(cats[0], dict):
        cats = [c.get("name", "") for c in cats]
    github = str(e.get("github_repo_url") or e.get("github_url") or "")
    return {
        "kind": "mcp-dir",
        "repo_id": sid,
        "name": name,
        "author": str(e.get("author") or e.get("author_name") or ""),
        "description": desc[:400],
        "categories": [str(c) for c in cats][:6],
        "downloads": int(e.get("downloads")
                         or e.get("docker_download_count") or 0),
        "github_stars": int(e.get("github_stars") or 0),
        "url": f"https://www.pulsemcp.com/servers/{sid}",
        "github_url": github,
        "last_modified": str(e.get("updated_at") or ""),
    }


def search_mcp_dir(query: str = "", limit: int = 40) -> List[Dict[str, Any]]:
    """Browse/search the PulseMCP server directory."""
    params = [("limit", str(max(1, min(int(limit), 100))))]
    if query and query.strip():
        params.append(("search_query", query.strip()))
    data = _pulsemcp_get("/v0.1/servers", params)
    rows = data.get("data") or data.get("servers") or []
    return [_mcp_dir_row(e) for e in rows]


def mcp_dir_detail(server_id: str) -> Dict[str, Any]:
    """One PulseMCP server with its full card (readme/setup when present)."""
    sid = str(server_id or "").strip().strip("/")
    if not sid or ".." in sid or any(c.isspace() for c in sid):
        raise HubError(f"invalid pulsemcp server id: {server_id!r}")
    data = _pulsemcp_get(f"/v0.1/servers/{sid}", [])
    e = data.get("data") or data
    row = _mcp_dir_row(e)
    row["readme"] = str(e.get("readme") or e.get("description") or "")[:24000]
    transports = e.get("transports") or {}
    if isinstance(transports, dict):
        row["transports"] = sorted(k for k, v in transports.items() if v)
    return row


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


def repo_files(repo_id: str, kind: str = "models") -> Tuple[str, List[Dict[str, Any]]]:
    """(commit_sha, siblings[{rfilename,size}]) for a repo of any kind."""
    kind = normalize_kind(kind)
    repo_id = validate_repo_id(repo_id)
    r = _get(f"{endpoint()}/api/{kind}/{repo_id}", params={"blobs": "true"})
    data = r.json()
    sha = data.get("sha") or "main"
    sibs = [
        {"rfilename": s.get("rfilename", ""), "size": int(s.get("size") or 0)}
        for s in (data.get("siblings") or []) if s.get("rfilename")
    ]
    return sha, sibs


def list_variants(repo_id: str, kind: str = "models") -> Dict[str, Any]:
    """Quantization variants for a repo (GGUF), or one 'full' pseudo-variant.

    Returns {repo_id, kind, gguf, default_variant, variants:[{variant, files,
    size_bytes, shard_count}]}. ``default_variant`` prefers Q4_K_M-style
    medium quants, else the smallest — studio picks similarly so first-click
    downloads stay sane. Datasets and spaces (and non-GGUF model repos) get a
    single 'full' variant of every file.

    Companion GGUFs (mmproj vision projectors, MTP drafters) are never a
    variant of their own: they ride along with EVERY quant download so a
    served model always has its projector/drafter available locally (studio
    does the same — mmproj lands next to the quant, MTP/ alongside).
    """
    kind = normalize_kind(kind)
    sha, sibs = repo_files(repo_id, kind)

    ggufs = ([s for s in sibs if s["rfilename"].lower().endswith(".gguf")
              and not _is_companion_rel(s["rfilename"])]
             if kind == "models" else [])
    if not ggufs:
        # Non-GGUF repo: single variant of everything (training base model,
        # dataset corpus, or MCP space).
        total = sum(s["size"] for s in sibs)
        return {
            "repo_id": repo_id, "kind": kind, "gguf": False, "sha": sha,
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
    return {"repo_id": repo_id, "kind": kind, "gguf": True, "sha": sha,
            "default_variant": default, "variants": out}


def model_detail(repo_id: str, kind: str = "models") -> Dict[str, Any]:
    """Everything the detail pane shows (studio's discover right-hand page):
    card metadata + variants + README markdown + local install state.
    """
    kind = normalize_kind(kind)
    repo_id = validate_repo_id(repo_id)
    # Kind-specific expand sets: the API 400s on keys a kind doesn't know.
    if kind == "models":
        expand = ["cardData", "downloads", "likes", "lastModified",
                  "pipeline_tag", "gated", "gguf", "safetensors"]
    elif kind == "datasets":
        expand = ["cardData", "downloads", "likes", "lastModified", "gated"]
    else:  # spaces — no `gated`/`downloads` expand (the API 400s on them)
        expand = ["cardData", "likes", "sdk"]
    r = _get(f"{endpoint()}/api/{kind}/{repo_id}", params={
        "blobs": "true", "expand": expand,
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
        rr = httpx.get(
            f"{endpoint()}/{_url_seg(kind)}{repo_id}/raw/main/README.md",
            headers=_headers(), timeout=REQUEST_TIMEOUT,
            follow_redirects=True)
        if rr.status_code == 200:
            readme = rr.text[:60_000]
    except httpx.HTTPError:
        pass
    variants_info = list_variants(repo_id, kind)
    return {
        "kind": kind,
        "repo_id": repo_id,
        "name": repo_id.split("/")[-1],
        "author": (data.get("author") or repo_id.split("/")[0]),
        "downloads": int(data.get("downloads") or 0),
        "likes": int(data.get("likes") or 0),
        "last_modified": (data.get("lastModified")
                          or data.get("createdAt") or ""),
        "gguf": _is_gguf(data) if kind == "models" else False,
        "pipeline_tag": data.get("pipeline_tag") or "",
        "sdk": data.get("sdk") or "",
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


def _files_for_variant(repo_id: str, variant: str, kind: str = "models",
                       ) -> Tuple[str, List[Dict[str, Any]]]:
    """(sha, sibling subset) selected by variant name ('' => default)."""
    info = list_variants(repo_id, kind)
    variants = info["variants"]
    if not variant or variant == "default":
        variant = info["default_variant"]
    chosen = next((v for v in variants if v["variant"] == variant), None)
    if chosen is None:
        raise HubError(f"unknown variant {variant!r}; available: "
                       + ", ".join(v["variant"] for v in variants))
    want = set(chosen["files"])
    sibs = [s for s in repo_files(info["repo_id"], kind)[1]
            if s["rfilename"] in want]
    return info["sha"], sibs


# --------------------------------------------------------------- downloads

@dataclass
class DownloadJob:
    job_id: str
    repo_id: str
    variant: str
    target_dir: str
    kind: str = "models"
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
            "kind": self.kind,
            "variant": self.variant, "target_dir": self.target_dir,
            "state": self.state, "total_bytes": self.total_bytes,
            "done_bytes": self.done_bytes, "error": self.error,
            "created_at": self.created_at, "finished_at": self.finished_at,
            "files_done": list(self.files_done),
        }


class ModelHub:
    """Hub client + download manager bound to one base model folder.

    ``base_dir`` overrides resolution (tests); otherwise the resolved
    ``models_dir`` / HF cache, falling back to ``<home>/models``. Datasets
    and spaces live in sibling roots (``<home>/hub/datasets``,
    ``<home>/hub/spaces``) so model discovery never sees them.
    """

    def __init__(self, models_dir: Optional[str] = None,
                 home: Optional[Path] = None):
        self._models_dir_arg = models_dir
        self._home = Path(home) if home else None
        self._lock = threading.Lock()
        self._jobs: Dict[str, DownloadJob] = {}

    # ------------------------------------------------------------- folders

    def _resolve_home(self) -> Path:
        if self._home is not None:
            return self._home
        from ..store.paths import resolve_sable_home
        return resolve_sable_home()

    def base_dir(self) -> Path:
        """Where model repos live; created on demand (download time only).

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
        return self._resolve_home() / "models"

    def kind_base(self, kind: str = "models") -> Path:
        """Root folder for a resource kind. Models keep the discovery-visible
        base dir; datasets/spaces get their own hub roots underneath SABLE_HOME."""
        kind = normalize_kind(kind)
        if kind == "models":
            return self.base_dir()
        return self._resolve_home() / "hub" / kind

    def repo_cache_dir(self, repo_id: str, kind: str = "models") -> Path:
        kind = normalize_kind(kind)
        repo_id = validate_repo_id(repo_id)
        prefix = _CACHE_PREFIX[kind]
        return self.kind_base(kind) / (prefix + repo_id.replace("/", "--"))

    def snapshot_dir(self, repo_id: str, sha: str,
                     kind: str = "models") -> Path:
        return self.repo_cache_dir(repo_id, kind) / "snapshots" / sha

    # ------------------------------------------------------------- search

    def search(self, query: str = "", limit: int = 30, gguf_only: bool = False,
               capability: str = "", sort: str = "downloads",
               kind: str = "models") -> List[Dict[str, Any]]:
        return search_models(query, limit=limit, gguf_only=gguf_only,
                             capability=capability, sort=sort, kind=kind)

    def feed(self, limit: int = 15, kind: str = "models") -> List[Dict[str, Any]]:
        return latest_feed(limit, kind)

    def variants(self, repo_id: str, kind: str = "models") -> Dict[str, Any]:
        return list_variants(repo_id, kind)

    def detail(self, repo_id: str, kind: str = "models") -> Dict[str, Any]:
        return model_detail(repo_id, kind)

    # ----------------------------------------------------------- installed

    def installed(self, kind: str = "models") -> List[Dict[str, Any]]:
        """Repos of one kind present under its base dir (hub layout)."""
        kind = normalize_kind(kind)
        base = self.kind_base(kind)
        out: List[Dict[str, Any]] = []
        if not base.is_dir():
            return out
        prefix = _CACHE_PREFIX[kind]
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and entry.name.startswith(prefix):
                repo = entry.name[len(prefix):].replace("--", "/")
                snaps = entry / "snapshots"
                revisions, size = [], 0
                if snaps.is_dir():
                    for snap in sorted(snaps.iterdir()):
                        if snap.is_dir():
                            revisions.append(snap.name)
                            size += _dir_size_dedupe(snap)
                row: Dict[str, Any] = {
                    "repo_id": repo, "kind": kind, "path": str(entry),
                    "layout": "hub", "revisions": revisions,
                    "size_bytes": size, "complete": bool(revisions)}
                if kind == "models":
                    # Quant tags come from the path relative to the repo dir
                    # so directory-named quants (Q4_K_M/model.gguf) parse.
                    row["quants"] = sorted({q for q in (
                        _quant_of_path(str(p.relative_to(entry)))
                        for p in entry.rglob("*.gguf")) if q})[:12]
                out.append(row)
        return out

    def installed_all(self) -> List[Dict[str, Any]]:
        """Installed repos across every kind (the On Device tab)."""
        out: List[Dict[str, Any]] = []
        for kind in KINDS:
            try:
                out.extend(self.installed(kind))
            except HubError:
                continue
        return out

    def delete(self, repo_id: str, kind: str = "models") -> Dict[str, Any]:
        """Remove a downloaded repo — confined to the kind's base dir."""
        kind = normalize_kind(kind)
        target = self.repo_cache_dir(repo_id, kind)  # validates + anchors
        base = self.kind_base(kind).resolve()
        resolved = target.resolve()
        if not resolved.is_relative_to(base):
            raise HubError("refusing to delete outside the hub folder")
        if not resolved.is_dir():
            raise HubError(f"{repo_id} is not downloaded under {base}")
        size = _dir_size_dedupe(resolved)
        shutil.rmtree(resolved)
        return {"deleted": validate_repo_id(repo_id), "kind": kind,
                "freed_bytes": size}

    # ----------------------------------------------------------- downloads

    def start_download(self, repo_id: str, variant: str = "",
                       kind: str = "models") -> str:
        """Queue a background download; returns the job id (studio's
        POST /models/download pattern: 202 + poll)."""
        kind = normalize_kind(kind)
        repo_id = validate_repo_id(repo_id)
        with self._lock:
            # Single-flight per (kind, repo, variant): reject duplicates.
            for job in self._jobs.values():
                if (job.repo_id == repo_id and job.variant == variant
                        and job.kind == kind
                        and job.state in ("queued", "running")):
                    return job.job_id
            job = DownloadJob(job_id="dl-" + uuid.uuid4().hex[:8],
                              repo_id=repo_id, variant=variant or "",
                              target_dir="", kind=kind)
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
            sha, sibs = _files_for_variant(job.repo_id, job.variant, job.kind)
            if job._cancel.is_set():
                raise _Cancelled()
            job.variant = job.variant or ""
            snap = self.snapshot_dir(job.repo_id, sha, job.kind)
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
        dest = (self.snapshot_dir(job.repo_id, sha, job.kind)
                / rel).resolve()
        if not dest.is_relative_to(self.kind_base(job.kind).resolve()):
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

        url = (f"{endpoint()}/{_url_seg(job.kind)}{job.repo_id}"
               f"/resolve/main/{rel}").replace("resolve/main",
                                               f"resolve/{sha}", 1)
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
        refs = self.repo_cache_dir(job.repo_id, job.kind) / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(sha + "\n")
        manifest = {
            "repo_id": job.repo_id, "kind": job.kind, "sha": sha,
            "variant": job.variant,
            "files": [{"rfilename": s["rfilename"], "size": s["size"]}
                      for s in sibs],
            "downloaded_at": _utcnow(), "tool": "sable/resourcehub",
        }
        (self.snapshot_dir(job.repo_id, sha, job.kind) / "sable_download.json"
         ).write_text(json.dumps(manifest, indent=2))

    def download_sync(self, repo_id: str, variant: str = "",
                      progress_cb: Optional[Any] = None,
                      kind: str = "models") -> DownloadJob:
        """Inline download for the CLI (no thread); same code path as jobs."""
        kind = normalize_kind(kind)
        repo_id = validate_repo_id(repo_id)
        job = DownloadJob(job_id="dl-" + uuid.uuid4().hex[:8],
                          repo_id=repo_id, variant=variant or "",
                          target_dir="", kind=kind)
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

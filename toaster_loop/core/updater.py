"""Toaster Loop self-update system (app updates, not llama.cpp — that is
serving/runtime.py's `check_llama_update`).

How users get the program today: a git clone + ./setup.sh, which pip-installs
the repo editable into ./.venv. So the natural update flow is `git pull` +
reinstall, and the natural "there is an update" signal is origin moving ahead
of HEAD — plus the GitHub Releases API for a human-readable version number
and release notes URL when releases are published.

Principles:
  * Notify, never surprise. Nothing here mutates the install without an
    explicit user action (`toaster-loop upgrade --yes` or the dashboard's
    Apply button). A tool that runs training jobs must not swap its own code
    out from under a live process.
  * Checks are cheap and throttled. Results cache in <home>/update_check.json
    with a configurable TTL (default 24 h), so dashboard startup / status
    polls never hammer the GitHub API. `git ls-remote` sends no data and
    mutates nothing; the Releases call is one GET.
  * Apply is safe or refuses. A dirty worktree, detached HEAD, or a non-fast-
    forward situation aborts with a clear message instead of merging or
    stashing the user's local edits.
  * Offline / private-repo friendly: every network step degrades to
    {available: False, error: ...} — an unreachable GitHub must never break
    the CLI or dashboard.

Repo resolution precedence: config ``update.repo`` ("owner/name" or a URL) >
the clone's ``origin`` remote URL > DEFAULT_REPO.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("toaster_loop.updater")

#: Fallback upstream when the install is not a git clone and no override set.
DEFAULT_REPO = "billybasass/Toaster-Loop"

CHECK_FILENAME = "update_check.json"
DEFAULT_INTERVAL_HOURS = 24.0
_GIT_TIMEOUT_S = 15.0
_HTTP_TIMEOUT_S = 6.0

# Cache read/write is process-local best-effort; one lock keeps two threads in
# the same dashboard from interleaving writes.
_cache_lock = threading.Lock()


class UpdateError(Exception):
    """Apply-path failure with a user-presentable message."""


# --------------------------------------------------------------------- config

def update_config(cfg: Any) -> Dict[str, Any]:
    """Normalized `update:` section (raw dict on the dataclass; lenient)."""
    sec = getattr(cfg, "update", None)
    if not isinstance(sec, dict):
        sec = {}
    out: Dict[str, Any] = {
        "enabled": bool(sec.get("enabled", True)),
        "repo": str(sec.get("repo") or "").strip() or None,
        "interval_hours": DEFAULT_INTERVAL_HOURS,
    }
    try:
        iv = float(sec.get("interval_hours", DEFAULT_INTERVAL_HOURS))
        if iv > 0:
            out["interval_hours"] = iv
    except (TypeError, ValueError):
        pass
    return out


# ------------------------------------------------------------- install types

def package_root() -> Path:
    """The toaster_loop package directory (…/toaster_loop)."""
    return Path(__file__).resolve().parent.parent


def git_install_root() -> Optional[Path]:
    """Repo root when running from a git checkout of Toaster Loop, else None.

    A pip/wheel install has the package under site-packages with no .git and
    no pyproject.toml beside it, so this cleanly separates the two installs."""
    root = package_root().parent
    if (root / ".git").exists() and (root / "pyproject.toml").is_file():
        return root
    return None


def _repo_from_url(url: str) -> Optional[str]:
    """owner/name from an https or ssh git remote URL."""
    url = (url or "").strip().removesuffix(".git")
    m = re.search(r"[:/]([^/:]+)/([^/:]+?)(?:\.git)?$", url)
    if m and "/" in url:
        return f"{m.group(1)}/{m.group(2)}"
    return None


def resolve_repo(cfg: Any, root: Optional[Path] = None) -> str:
    """Upstream "owner/name": config override > origin remote > default."""
    uc = update_config(cfg)
    if uc["repo"]:
        r = _repo_from_url(uc["repo"]) or uc["repo"]
        return r
    if root is None:
        root = git_install_root()
    if root is not None:
        url = _git(root, "remote", "get-url", "origin")
        if url:
            r = _repo_from_url(url)
            if r:
                return r
    return DEFAULT_REPO


def is_editable_install() -> bool:
    """True when the running package was pip-installed `-e` (setup.sh path).

    Reads direct_url.json from the distribution metadata; falls back to "yes"
    when running from inside a git checkout (the only way that happens today
    is an editable install, and reinstalling -e keeps users on the same
    workflow as setup.sh)."""
    try:
        from importlib.metadata import distribution

        txt = distribution("toaster-loop").read_text("direct_url.json")
        if txt:
            data = json.loads(txt)
            return bool((data.get("dir_info") or {}).get("editable"))
    except Exception:  # noqa: BLE001 - metadata missing => heuristic below
        pass
    return git_install_root() is not None


# ------------------------------------------------------------------ versions

def parse_version(s: str) -> Optional[Tuple[int, ...]]:
    """'v0.2.1-rc1' -> (0, 2, 1); None when unparseable."""
    m = re.match(r"^\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(s or ""))
    if not m:
        return None
    return tuple(int(g) if g else 0 for g in m.groups())


def current_version() -> str:
    from .. import __version__

    return __version__


def _is_newer(remote: Optional[str], local: str) -> bool:
    rv, lv = parse_version(remote or ""), parse_version(local)
    if rv is None or lv is None:
        return False
    return rv > lv


# ------------------------------------------------------------------ git bits

def _git(root: Path, *args: str) -> str:
    """Run a read-only-ish git command; '' on any failure (never raises)."""
    try:
        r = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S)
        if r.returncode != 0:
            return ""
        return (r.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _git_strict(root: Path, *args: str, timeout: float = 120.0) -> str:
    """Apply-path git command; raises UpdateError with stderr on failure."""
    try:
        r = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        raise UpdateError(f"git {' '.join(args)} failed to run: {e}") from e
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()[:800]
        raise UpdateError(f"git {' '.join(args)} failed: {err}")
    return (r.stdout or "").strip()


def git_state(root: Path) -> Dict[str, Any]:
    """branch / head sha / dirty flag for the checkout (empty branch when detached)."""
    return {
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "head": _git(root, "rev-parse", "HEAD"),
        "dirty": bool(_git(root, "status", "--porcelain")),
    }


# ------------------------------------------------------------------- checks

def _remote_release(repo: str) -> Dict[str, Any]:
    """Latest GitHub release tag + html URL; falls back to the newest tag."""
    import httpx

    base = f"https://api.github.com/repos/{repo}"
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "toaster-loop-updater"}
    try:
        r = httpx.get(f"{base}/releases/latest", headers=headers,
                      timeout=_HTTP_TIMEOUT_S, follow_redirects=True)
        if r.status_code == 200:
            d = r.json()
            return {"tag": str(d.get("tag_name") or ""),
                    "url": str(d.get("html_url") or f"https://github.com/{repo}/releases"),
                    "notes": str(d.get("body") or "")[:4000]}
        if r.status_code != 404:
            return {"error": f"releases/latest HTTP {r.status_code}"}
        # No releases published — newest tag is the next best thing; no tags
        # either (early repo) simply means "no release channel yet".
        r = httpx.get(f"{base}/tags?per_page=1", headers=headers,
                      timeout=_HTTP_TIMEOUT_S, follow_redirects=True)
        if r.status_code == 200:
            tags = r.json()
            tag = str(tags[0].get("name") or "") \
                if isinstance(tags, list) and tags else ""
            return {"tag": tag, "url": f"https://github.com/{repo}/tags",
                    "notes": ""}
        return {"error": f"tags HTTP {r.status_code}"}
    except httpx.HTTPError as e:
        return {"error": f"github request failed: {e}"}


def _run_check(cfg: Any) -> Dict[str, Any]:
    """One uncached check: git-aware for clones, Releases API otherwise."""
    from ..store.paths import resolve_toaster_loop_home

    home = resolve_toaster_loop_home(cfg)
    root = git_install_root()
    repo = resolve_repo(cfg, root=root)
    local = current_version()
    result: Dict[str, Any] = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "current_version": local,
        "repo": repo,
        "install": "git" if root else "package",
        "available": False,
        "remote_version": None,
        "url": f"https://github.com/{repo}",
        "error": None,
    }

    # 1) Release channel (both install kinds): version number + notes URL.
    rel = _remote_release(repo)
    if rel.get("error"):
        result["error"] = rel["error"]
    else:
        tag = rel.get("tag") or ""
        result["remote_version"] = tag.lstrip("vV") or None
        result["url"] = rel.get("url") or result["url"]
        result["release_notes"] = rel.get("notes", "")
        if _is_newer(tag, local):
            result["available"] = True

    # 2) Git channel (clones): origin ahead of HEAD is the ground truth even
    #    when no release/tag has been cut yet (the common dev-push case).
    if root is not None:
        st = git_state(root)
        result["git"] = {"branch": st["branch"], "head": st["head"][:12],
                         "dirty": st["dirty"]}
        branch = st["branch"] or "HEAD"
        remote_sha = _git(root, "ls-remote", "origin", f"refs/heads/{branch}")
        remote_sha = remote_sha.split("\t")[0].split()[0] if remote_sha else ""
        if remote_sha and st["head"] and remote_sha != st["head"]:
            # ls-remote can't count commits; a plain sha difference on the
            # tracked branch means origin moved (behind or diverged — apply
            # path enforces --ff-only either way).
            result["available"] = True
            result["git"]["remote_head"] = remote_sha[:12]
        elif not remote_sha and not result.get("error"):
            result["error"] = f"could not read origin/{branch}"

    try:
        home.mkdir(parents=True, exist_ok=True)
        with _cache_lock:
            _cache_path(home).write_text(
                json.dumps(result, indent=2), encoding="utf-8")
    except OSError as e:  # cache write is best-effort
        logger.debug("update check cache write failed: %s", e)
    return result


def _cache_path(home: Path) -> Path:
    return home / CHECK_FILENAME


def _read_cache(home: Path) -> Optional[Dict[str, Any]]:
    try:
        with _cache_lock:
            data = json.loads(_cache_path(home).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _cache_age_hours(rec: Dict[str, Any]) -> Optional[float]:
    try:
        ts = datetime.fromisoformat(str(rec.get("checked_at")))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
    except (TypeError, ValueError):
        return None


def check_for_update(cfg: Any, home: Optional[Path] = None, *,
                     force: bool = False) -> Dict[str, Any]:
    """Update status for this install, honoring the cached-check TTL.

    Returns a dict with at least {available, current_version, remote_version,
    checked_at, error}. Never raises: network/git failures land in `error`
    with available=False (stale cache beats nothing)."""
    from ..store.paths import resolve_toaster_loop_home

    uc = update_config(cfg)
    if home is None:
        home = resolve_toaster_loop_home(cfg)
    cached = _read_cache(home)
    age = _cache_age_hours(cached) if cached else None
    fresh = (cached is not None and age is not None
             and age < uc["interval_hours"])
    if not uc["enabled"]:
        # Disabled: report the last known state (if any) but never check.
        out = dict(cached) if cached else {
            "available": False, "current_version": current_version(),
            "checked_at": None, "error": None}
        out["enabled"] = False
        return out
    if fresh and not force:
        out = dict(cached or {})
        out["cached"] = True
        return out
    try:
        res = _run_check(cfg)
    except Exception as e:  # noqa: BLE001 - a failed check must never propagate
        logger.debug("update check failed", exc_info=True)
        if cached:
            out = dict(cached)
            out["error"] = f"check failed: {e}"
            out["cached"] = True
            return out
        return {"available": False, "current_version": current_version(),
                "checked_at": None, "error": f"check failed: {e}"}
    res["enabled"] = True
    return res


# -------------------------------------------------------------------- apply

def apply_update(cfg: Any, log_fn: Optional[Callable[[str], None]] = None
                 ) -> Dict[str, Any]:
    """Pull + reinstall the app from its git clone. Explicit user action only.

    Refuses (UpdateError) unless: git-checkout install, clean worktree, a real
    branch (not detached), and a strict fast-forward is possible. Reinstalls
    with pip in the same editable mode as before, so setup.sh workflows stay
    identical. Returns {from, to, restart_required: True} on success."""
    log = log_fn or (lambda s: None)
    root = git_install_root()
    if root is None:
        raise UpdateError(
            "not a git-checkout install — update this one with "
            "`pip install --upgrade toaster-loop` (or re-run setup.sh from a "
            "fresh clone)")

    st = git_state(root)
    if st["dirty"]:
        raise UpdateError(
            "working tree has uncommitted changes — commit or stash them "
            f"first ({root})")
    if not st["branch"] or st["branch"] == "HEAD":
        raise UpdateError("detached HEAD — checkout a branch first "
                          "(e.g. `git checkout main`)")

    log(f"fetching origin/{st['branch']} …")
    _git_strict(root, "fetch", "origin", st["branch"])
    local = st["head"]
    remote = _git_strict(root, "rev-parse", "FETCH_HEAD")
    if local == remote:
        return {"from": current_version(), "to": current_version(),
                "changed": False, "restart_required": False}

    log(f"fast-forwarding {local[:12]} -> {remote[:12]} …")
    _git_strict(root, "merge", "--ff-only", "FETCH_HEAD")

    editable = is_editable_install()
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if editable:
        cmd.append("-e")
    cmd.append(str(root))
    log(f"reinstalling package ({'editable' if editable else 'regular'}) …")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900.0)
    except (OSError, subprocess.SubprocessError) as e:
        raise UpdateError(f"reinstall failed to run: {e}") from e
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip()[-1200:]
        raise UpdateError(f"pip reinstall failed (exit {r.returncode}):\n{tail}")

    log("update applied.")
    return {"from": current_version(), "to": _pyproject_version(root),
            "changed": True, "restart_required": True}


def _pyproject_version(root: Path) -> str:
    """Version string from the checked-out pyproject.toml (post-pull truth).

    TOML would be the principled parse, but the version line is a stable
    one-liner in our own repo — a regex avoids tomllib in the apply path for
    a display string."""
    try:
        m = re.search(r'^\s*version\s*=\s*"([^"]+)"',
                      (root / "pyproject.toml").read_text(encoding="utf-8"),
                      re.MULTILINE)
        if m:
            return m.group(1)
    except OSError:
        pass
    return current_version()


def format_status(status: Dict[str, Any]) -> str:
    """Human-readable multi-line summary for the CLI."""
    lines = [f"toaster_loop  : v{status.get('current_version') or '?'} "
             f"(install: {status.get('install', 'package')}, "
             f"repo: {status.get('repo', '?')})"]
    if status.get("error"):
        lines.append(f"check       : could not verify ({status['error']})")
    if status.get("git"):
        g = status["git"]
        line = f"git         : branch={g.get('branch') or 'detached'} " \
               f"head={g.get('head')}"
        if g.get("dirty"):
            line += " (uncommitted changes)"
        lines.append(line)
    if status.get("available"):
        rv = status.get("remote_version")
        lines.append("update      : AVAILABLE"
                     + (f" -> v{rv}" if rv else "")
                     + f"  ({status.get('url')})")
        if status.get("cached"):
            lines.append("              (from cached check; "
                         "run with --check to re-verify)")
    elif not status.get("error"):
        lines.append("update      : up to date"
                     + (f" (v{status['remote_version']})"
                        if status.get("remote_version") else ""))
    return "\n".join(lines)

"""Workspace file explorer: read-only lazy tree + file preview,
contained to the project workspace_dir.
"""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import unquote


from .base import ApiContext, HttpError


class FsBrowseRoutes(ApiContext):
    """Workspace file explorer: read-only lazy tree + file preview,
contained to the project workspace_dir.
    """

    def h_fs_tree(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ...core.path_security import has_traversal_component, validate_within_dir
        from ...core.tools import _sensitive_read_reason

        project_id = str(q.get("project") or "").strip()
        if not project_id:
            raise HttpError(400, "project parameter required",
                            code="bad_request")
        proj = self._load_project(project_id)
        if proj is None:
            raise HttpError(404, f"no project {project_id}",
                            code="unknown_project")
        ws_raw = str(proj.get("workspace_dir") or "")
        try:
            ws_root = Path(ws_raw).expanduser().resolve()
        except OSError:
            ws_root = None
        if ws_root is None or not ws_root.is_dir():
            raise HttpError(404, f"project workspace missing: {ws_raw}",
                            code="unknown_project")

        rel = str(q.get("path") or "").strip()
        if rel in ("", ".", "./"):
            target = ws_root
        else:
            # Absolute paths and '..' components are rejected up front (same
            # cheap pre-check the contain() decorator runs), then resolve()-
            # based containment catches symlink escapes to existing targets.
            # %2e%2e-style encoded traversal: the HTTP layer's parse_qs already
            # decodes query values, but re-check the unquoted form here too so
            # direct handle() callers get the same 400.
            if (Path(rel).is_absolute() or rel.startswith("~")
                    or Path(unquote(rel)).is_absolute()):
                raise HttpError(400, "path must be relative to the workspace",
                                code="bad_request")
            if (has_traversal_component(rel)
                    or has_traversal_component(unquote(rel))):
                raise HttpError(400, "'..' path traversal not allowed",
                                code="bad_request")
            candidate = ws_root / rel
            reason = validate_within_dir(candidate, ws_root)
            if reason is not None:
                raise HttpError(400, f"path escapes workspace: {rel}",
                                code="bad_request")
            target = candidate

        # A sensitive directory (.ssh/.gnupg/... or .aws credentials) is
        # blocked as a listing root too — listing it is itself the leak.
        if _sensitive_read_reason(str(target)) is not None:
            raise HttpError(400, "path blocked (sensitive credential path)",
                            code="bad_request")
        if not target.is_dir():
            raise HttpError(404, f"no such directory: {rel}",
                            code="not_found")

        show_hidden = str(q.get("hidden") or "") in ("1", "true", "yes")
        entries: List[Dict[str, Any]] = []
        truncated = False
        try:
            children = list(target.iterdir())
        except OSError as e:
            raise HttpError(404, f"unreadable directory: {rel}",
                            code="not_found") from e
        # Entry cap: a node_modules-scale directory must not serialize every
        # child into one response. Alphabetical pre-sort keeps the visible
        # slice deterministic; `truncated` tells the client more exists.
        if len(children) > FS_TREE_MAX_ENTRIES:
            children = sorted(children, key=lambda c: c.name.lower()) \
                [:FS_TREE_MAX_ENTRIES]
            truncated = True
        for child in children:
            name = child.name
            if not show_hidden and name.startswith("."):
                continue
            try:
                # Deny-list first (lexical, like tools.py), then stat.
                if _sensitive_read_reason(str(child)) is not None:
                    continue
                is_dir = child.is_dir()  # follows symlinks
                st = child.stat()
                # Symlink escapes never appear in the listing at all: the
                # target lives outside the workspace, so expanding it would
                # be blocked anyway — omit rather than show a dead node.
                if validate_within_dir(child, ws_root) is not None:
                    continue
                child_rel = (str(rel) + "/" if rel and rel != "." else "") + name
                entry: Dict[str, Any] = {
                    "name": name,
                    "path": child_rel,
                    "type": "dir" if is_dir else "file",
                }
                if not is_dir:
                    entry["size"] = st.st_size
                entry["mtime"] = datetime.fromtimestamp(
                    st.st_mtime, timezone.utc).isoformat(timespec="seconds")
                entries.append(entry)
            except OSError:
                continue  # unreadable entries are silently omitted
        entries.sort(key=lambda e: (e["type"] != "dir",
                                    e["name"].lower()))
        return {"project": project_id, "path": rel or ".", "entries": entries,
                "truncated": truncated}

    # GET /api/fs/read?project=<id>&path=<rel> — read-only file preview for
    # the explorer: same containment + sensitive-path rules as /api/fs/tree,
    # plus a hard size cap and a binary sniff (binary files return metadata
    # only). Strictly text; never follows out of the workspace.
    def h_fs_read(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ...core.path_security import has_traversal_component, validate_within_dir
        from ...core.tools import _sensitive_read_reason

        project_id = str(q.get("project") or "").strip()
        if not project_id:
            raise HttpError(400, "project parameter required",
                            code="bad_request")
        proj = self._load_project(project_id)
        if proj is None:
            raise HttpError(404, f"no project {project_id}",
                            code="unknown_project")
        ws_raw = str(proj.get("workspace_dir") or "")
        try:
            ws_root = Path(ws_raw).expanduser().resolve()
        except OSError:
            ws_root = None
        if ws_root is None or not ws_root.is_dir():
            raise HttpError(404, f"project workspace missing: {ws_raw}",
                            code="unknown_project")

        rel = str(q.get("path") or "").strip()
        if (not rel or rel in (".", "./")
                or Path(rel).is_absolute() or rel.startswith("~")
                or Path(unquote(rel)).is_absolute()):
            raise HttpError(400, "path must be a relative file path",
                            code="bad_request")
        if (has_traversal_component(rel)
                or has_traversal_component(unquote(rel))):
            raise HttpError(400, "'..' path traversal not allowed",
                            code="bad_request")
        target = ws_root / rel
        if validate_within_dir(target, ws_root) is not None:
            raise HttpError(400, f"path escapes workspace: {rel}",
                            code="bad_request")
        if _sensitive_read_reason(str(target)) is not None:
            raise HttpError(400, "path blocked (sensitive credential path)",
                            code="bad_request")
        try:
            if not target.is_file():
                raise HttpError(404, f"no such file: {rel}", code="not_found")
            size = target.stat().st_size
        except OSError as e:
            raise HttpError(404, f"unreadable file: {rel}",
                            code="not_found") from e
        # Size gate before touching contents: big files get metadata only.
        if size > FS_READ_MAX_CHARS * 4:
            return {"project": project_id, "path": rel, "size": size,
                    "binary": False, "truncated": True, "content": "",
                    "note": f"file too large to preview (>{FS_READ_MAX_CHARS * 4} bytes)"}
        try:
            raw = target.read_bytes()
        except OSError as e:
            raise HttpError(404, f"unreadable file: {rel}",
                            code="not_found") from e
        if b"\x00" in raw[:8192]:  # cheap binary sniff on the head window
            return {"project": project_id, "path": rel, "size": size,
                    "binary": True, "truncated": False, "content": ""}
        text = raw.decode("utf-8", errors="replace")
        truncated = len(text) > FS_READ_MAX_CHARS
        if truncated:
            text = text[:FS_READ_MAX_CHARS]
        return {"project": project_id, "path": rel, "size": size,
                "binary": False, "truncated": truncated, "content": text}


# ------------------------------------------- module-level helpers

FS_TREE_MAX_ENTRIES = 2000

FS_READ_MAX_CHARS = 64_000

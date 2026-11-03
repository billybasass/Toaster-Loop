"""Resource Hub: models/datasets/MCP search, download, manage
(core/modelhub.py).
"""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

from typing import Any, Dict


from .base import ApiContext, ApiResponse, HttpError


class HubRoutes(ApiContext):
    """Resource Hub: models/datasets/MCP search, download, manage
(core/modelhub.py).
    """

    def _modelhub(self):
        from ...core.modelhub import ModelHub

        hub = getattr(self, "_hub", None)
        if hub is None:
            hub = ModelHub(models_dir=self.cfg.models_dir, home=self.home)
            self._hub = hub
        return hub

    @staticmethod
    def _hub_call(fn, *args, **kwargs):
        """Run a hub op, translating HubError -> HttpError for the envelope."""
        from ...core.modelhub import HubError

        try:
            return fn(*args, **kwargs)
        except HubError as e:
            msg = str(e)
            status = 404 if "not found" in msg else \
                401 if ("authorization" in msg or "HF_TOKEN" in msg) else 400
            raise HttpError(status, msg) from e

    def h_hub_search(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        limit = 30
        try:
            limit = int(q.get("limit", "30"))
        except ValueError:
            pass
        results = self._hub_call(
            self._modelhub().search, q.get("q", ""), limit,
            gguf_only=q.get("gguf", "") in ("1", "true", "yes"),
            capability=q.get("capability", ""),
            sort=q.get("sort", "downloads"), kind=q.get("kind", "models"))
        return {"query": q.get("q", ""), "results": results}

    def h_hub_feed(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        limit = 15
        try:
            limit = int(q.get("limit", "15"))
        except ValueError:
            pass
        return {"models": self._hub_call(
            self._modelhub().feed, limit, kind=q.get("kind", "models"))}

    def h_hub_mcp_dir(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """MCP registries — catalogues besides HF Spaces. ?source=registry
        (official MCP Registry, no auth; default) or pulsemcp (needs
        PULSEMCP_API_KEY). ?q= searches; ?server=<id> fetches one card."""
        from ...core.modelhub import (mcp_dir_detail, mcp_registry_detail,
                                     search_mcp_dir, search_mcp_registry)

        limit = 40
        try:
            limit = int(q.get("limit", "40"))
        except ValueError:
            pass
        server = q.get("server", "")
        if q.get("source", "registry") == "pulsemcp":
            if server:
                return self._hub_call(mcp_dir_detail, server)
            rows = self._hub_call(search_mcp_dir, q.get("q", ""), limit)
        else:
            if server:
                return self._hub_call(mcp_registry_detail, server)
            rows = self._hub_call(search_mcp_registry, q.get("q", ""), limit)
        return {"results": rows}

    def h_hub_detail(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        repo = q.get("repo", "")
        return self._hub_call(self._modelhub().detail, repo,
                              kind=q.get("kind", "models"))

    def h_hub_image(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        """Same-origin image proxy for resource cards (CSP is default-src 'self').
        ?repo=<id>&path=<repo-relative> or ?url=<public https url>;
        ?kind= routes repo-relative fetches to datasets/spaces."""
        from ...core.modelhub import fetch_image

        data, ctype = self._hub_call(
            fetch_image, repo_id=q.get("repo", ""), path=q.get("path", ""),
            url=q.get("url", ""), kind=q.get("kind", "models"))
        return ApiResponse(200, raw=data, content_type=ctype)

    def h_hub_variants(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        repo = q.get("repo", "")
        return self._hub_call(self._modelhub().variants, repo,
                              kind=q.get("kind", "models"))

    def h_hub_installed(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        hub = self._modelhub()
        hub.prune_finished()
        return {"models_dir": str(hub.base_dir()),
                "installed": self._hub_call(hub.installed_all),
                "downloads": hub.jobs()}

    def h_hub_download(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        # 202 + poll (studio contract): the worker thread writes the HF hub
        # cache layout; progress via GET /api/hub/downloads/<job_id>.
        repo = str(b.get("repo") or "")
        variant = str(b.get("variant") or "")
        kind = str(b.get("kind") or "models")
        job_id = self._hub_call(self._modelhub().start_download, repo,
                                variant, kind)
        return ApiResponse(202, {"ok": True, "job_id": job_id,
                                 "state": "queued"})

    def h_hub_downloads(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        return {"downloads": self._modelhub().jobs()}

    def h_hub_download_get(self, q: Dict[str, str], b: Dict[str, Any],
                           job_id: str) -> Dict[str, Any]:
        return self._hub_call(lambda: self._modelhub().job(job_id).to_dict())

    def h_hub_cancel(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        job_id = str(b.get("job_id") or "")
        return self._hub_call(self._modelhub().cancel, job_id)

    def h_hub_delete(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        repo = str(b.get("repo") or "")
        kind = str(b.get("kind") or "models")
        out = self._hub_call(self._modelhub().delete, repo, kind)
        self.bus.publish("model.deleted", out)
        return out

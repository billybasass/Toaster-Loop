"""Playbooks (list/promote/retire/assess), the auto-mined
golden suite and persistent skills (core/skills.py).
"""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

from typing import Any, Dict, List


from .base import ApiContext, HttpError


class PlaybooksRoutes(ApiContext):
    """Playbooks (list/promote/retire/assess), the auto-mined
golden suite and persistent skills (core/skills.py).
    """

    def _playbook_store(self):
        from ...core.playbooks import PlaybookStore

        return PlaybookStore(self.home / "buffer" / "toaster_loop.db")

    _CHAIN_SCAN_TTL_SECS = 120.0

    def _cached_chain_scan(self, profile):
        import time as _time

        now = _time.monotonic()
        cache = self.__dict__.setdefault("_chain_scan_cache", {})
        hit = cache.get(profile)
        if hit and now - hit[0] < self._CHAIN_SCAN_TTL_SECS:
            return hit[1]
        store = self._store()
        try:
            from ...core.playbooks import find_recurring_chains

            chains = find_recurring_chains(store, profile)[:10]
        finally:
            store.close()
        cache[profile] = (now, chains)
        return chains

    def h_playbooks_list(self, q: Dict[str, str],
                         b: Dict[str, Any]) -> Dict[str, Any]:
        from ...core.playbooks import playbooks_config

        profile = (str(q.get("profile")).strip() or None) \
            if q.get("profile") else None
        pb_store = self._playbook_store()
        try:
            rows = pb_store.list(profile=profile)
        finally:
            pb_store.close()
        chains = self._cached_chain_scan(profile)
        return {
            "enabled": bool(playbooks_config(self.cfg).get("enabled", False)),
            "playbooks": [p.to_dict() for p in rows],
            "chains": chains,
        }

    def h_playbook_assess(self, q: Dict[str, str],
                          b: Dict[str, Any]) -> Dict[str, Any]:
        """Correlational regression check (core/playbooks.assess_playbook):
        mean outcome score of matching-chain traces after vs before promotion;
        a clear regression retires the playbook. Explicit user action — never
        automatic from the panel."""
        from ...core.playbooks import Playbook, assess_playbook

        pb_id = str(b.get("id") or "").strip()
        pb_store = self._playbook_store()
        store = self._store()
        try:
            targets: List[Playbook] = []
            if pb_id:
                found = pb_store.get(pb_id)
                if found is None:
                    raise HttpError(404, f"no playbook {pb_id}")
                targets.append(found)
            else:
                targets.extend(pb_store.list(status="active"))
            results = []
            for pb in targets:
                res = assess_playbook(store, pb_store, pb)
                results.append({"id": pb.id, "chain": pb.chain, **res})
        finally:
            store.close()
            pb_store.close()
        return {"ok": True, "results": results}

    def h_goldens_mined_list(self, q: Dict[str, str],
                             b: Dict[str, Any]) -> Dict[str, Any]:
        """Every mined golden (enabled + disabled) with provenance — the
        operator view that makes the per-line enabled flag usable."""
        from ...curate.mine_goldens import list_mined_goldens

        return {"goldens": list_mined_goldens(self.home)}

    def h_goldens_mined_toggle(self, q: Dict[str, str],
                               b: Dict[str, Any]) -> Dict[str, Any]:
        from ...curate.mine_goldens import set_mined_enabled

        key = str(b.get("key") or "").strip()
        if not key:
            raise HttpError(400, "key required")
        enabled = bool(b.get("enabled"))
        if not set_mined_enabled(self.home, key, enabled):
            raise HttpError(404, f"no mined golden {key}")
        return {"ok": True, "key": key, "enabled": enabled}

    def h_skills_list(self, q: Dict[str, str],
                      b: Dict[str, Any]) -> Dict[str, Any]:
        from ...core.skills import all_skills, skills_config, skills_dir

        scfg = skills_config(self.cfg)
        return {
            "enabled": bool(scfg.get("enabled", False)),
            "dir": str(skills_dir()),
            "index_total_cap_chars": int(scfg.get("index_total_cap_chars", 1500)),
            "skills": [{"name": s.name, "description": s.description,
                        "updated": s.updated_ts, "path": s.path,
                        "chars": len(s.content)} for s in all_skills()],
        }

    def h_skill_get(self, q: Dict[str, str], b: Dict[str, Any], *,
                    skill_name: str) -> Dict[str, Any]:
        from ...core.skills import load_skill

        s = load_skill((skill_name or "").strip().lower())
        if s is None:
            raise HttpError(404, f"no such skill: {skill_name}")
        return {"name": s.name, "description": s.description,
                "updated": s.updated_ts, "path": s.path, "content": s.content}

    def h_skill_put(self, q: Dict[str, str], b: Dict[str, Any], *,
                    skill_name: str) -> Dict[str, Any]:
        from ...core.skills import save_skill

        name = (skill_name or "").strip().lower()
        description = b.get("description")
        content = b.get("content")
        if not isinstance(description, str) or not isinstance(content, str):
            raise HttpError(400, "description and content must be strings")
        # Same path the agent tool uses: validation + secret redaction +
        # atomic write, so a human edit can't land unredacted either.
        res = save_skill(name=name, description=description, content=content)
        if not res.get("ok"):
            raise HttpError(400, res.get("error") or "save failed")
        return {"ok": True, "name": name, "path": res["path"]}

    def h_skill_delete(self, q: Dict[str, str], b: Dict[str, Any], *,
                       skill_name: str) -> Dict[str, Any]:
        from ...core.skills import delete_skill

        res = delete_skill((skill_name or "").strip().lower())
        if not res.get("ok"):
            raise HttpError(404, res.get("error") or "delete failed")
        return res

    def h_playbook_promote(self, q: Dict[str, str],
                           b: Dict[str, Any]) -> Dict[str, Any]:
        from ...core.playbooks import promote

        pb_id = str(b.get("id") or "").strip()
        if not pb_id:
            raise HttpError(400, "id required")
        pb_store = self._playbook_store()
        try:
            # Dashboard promote is the EXPLICIT user confirmation path (no
            # judge gate here — --check lives in the CLI where an endpoint
            # can be resolved honestly).
            res = promote(pb_store, pb_id, check=False, actor="dashboard")
        finally:
            pb_store.close()
        if not res["ok"]:
            raise HttpError(404, res.get("error") or res.get("reason")
                            or "promote failed")
        return res

    def h_playbook_retire(self, q: Dict[str, str],
                          b: Dict[str, Any]) -> Dict[str, Any]:
        pb_id = str(b.get("id") or "").strip()
        if not pb_id:
            raise HttpError(400, "id required")
        pb_store = self._playbook_store()
        try:
            ok = pb_store.set_status(pb_id, "retired", actor="dashboard")
        finally:
            pb_store.close()
        if not ok:
            raise HttpError(404, f"no playbook {pb_id}")
        return {"ok": True, "id": pb_id, "status": "retired"}

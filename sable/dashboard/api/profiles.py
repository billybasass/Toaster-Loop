"""Agent profiles (Phase 5e) live in the config file: profiles:
is a list of {name, system_prompt, tools[], model_name?} and
default_profile names one. Dedicated endpoints wrap config
read/merge/atomic-write so the UI never patches raw YAML.
"""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


from .base import ApiContext, HttpError


class ProfilesRoutes(ApiContext):
    """Agent profiles (Phase 5e) live in the config file: profiles:
is a list of {name, system_prompt, tools[], model_name?} and
default_profile names one. Dedicated endpoints wrap config
read/merge/atomic-write so the UI never patches raw YAML.
    """

    KNOWN_TOOLS = ["read_file", "list_dir", "write_file", "patch",
                   "search_files", "todo_write", "todo_read",
                   "search_traces", "search_codebase",
                   "list_agents", "delegate_task", "check_delegation",
                   "save_skill", "read_skill", "list_skills", "delete_skill"]

    CAPABILITY_TOGGLES = {
        "internet": "web_fetch",
        "run_commands": "run_command",
        "vision": "view_image",
        "screenshots": "take_screenshot",
    }

    _PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

    def _profiles(self) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Current (profiles, default_profile) from the live config object."""
        profs = self.cfg.profiles if isinstance(self.cfg.profiles, list) else []
        default = self.cfg.default_profile
        return [p for p in profs if isinstance(p, dict)], (
            str(default) if default else None)

    def _write_profiles(self, profiles: List[Dict[str, Any]],
                        default: Optional[str]) -> Dict[str, Any]:
        patch: Dict[str, Any] = {"profiles": profiles}
        # `default_profile: null` deletes the key (unset semantics).
        patch["default_profile"] = default
        r = self.handle("PUT", "/api/config", body={"section": "root",
                                                    "patch": patch})
        if r.status >= 400:
            err = r.body.get("error", {}) if isinstance(r.body, dict) else {}
            raise HttpError(r.status, str(err.get("message", "config write failed")),
                            code=str(err.get("code", "internal_error")))
        return r.body if isinstance(r.body, dict) else {"ok": True}

    def _validate_profile(self, name: str, b: Dict[str, Any]) -> Dict[str, Any]:
        if not self._PROFILE_NAME_RE.match(name):
            raise HttpError(
                400, "profile name must match [A-Za-z0-9][A-Za-z0-9_-]{0,63}")
        prompt = b.get("system_prompt", "")
        if not isinstance(prompt, str):
            raise HttpError(400, "system_prompt must be a string")
        tools = b.get("tools", [])
        if not isinstance(tools, list) or any(not isinstance(t, str) for t in tools):
            raise HttpError(400, "tools must be a list of tool names")
        model_name = b.get("model_name")
        if model_name is not None and not isinstance(model_name, str):
            raise HttpError(400, "model_name must be a string or null")
        # Backlog #9: optional per-profile workspace containment override —
        # widens/narrows the global agent.workspace_root for chats using this
        # profile (see core/agent_runner.resolve_workspace_roots).
        workspace_root = b.get("workspace_root")
        if workspace_root is not None and not isinstance(workspace_root, str):
            raise HttpError(400, "workspace_root must be a string or null")
        # Phase 6: optional server-instance binding — chat runs for this
        # profile target that managed instance's endpoint instead of the
        # default serve target ("" | "default" => unset).
        server_instance = b.get("server_instance")
        if server_instance is not None and not isinstance(server_instance, str):
            raise HttpError(400, "server_instance must be a string or null")
        # Per-agent capabilities: {internet?: bool, run_commands?: bool}.
        # Only recognized keys are stored; absent => inherit global defaults.
        caps_in = b.get("capabilities")
        caps: Dict[str, bool] = {}
        if caps_in is not None:
            if not isinstance(caps_in, dict):
                raise HttpError(400, "capabilities must be an object or null")
            for key, val in caps_in.items():
                if key not in self.CAPABILITY_TOGGLES:
                    raise HttpError(
                        400, f"unknown capability {key!r} "
                        f"(known: {', '.join(self.CAPABILITY_TOGGLES)})")
                caps[str(key)] = bool(val)
        # Per-profile sampling + thinking overrides (reference style):
        # merged over the bound instance's / global `sampling:` sections for
        # chat runs using this profile (see h_chat_stream).
        samp_in = b.get("sampling")
        samp: Dict[str, Any] = {}
        if samp_in is not None:
            from ...serving.run_args import sanitize_sampling

            if not isinstance(samp_in, dict):
                raise HttpError(400, "sampling must be an object or null")
            try:
                samp = {k: v for k, v in sanitize_sampling(samp_in).items()
                        if v is not None}
            except ValueError as e:
                raise HttpError(400, f"sampling: {e}") from e
        prof: Dict[str, Any] = {
            "name": name,
            "system_prompt": prompt,
            "tools": sorted(set(tools)),
            "model_name": model_name or None,
        }
        if workspace_root:
            prof["workspace_root"] = workspace_root
        si = (server_instance or "").strip()
        if si and si != "default":
            prof["server_instance"] = si
        if caps:
            prof["capabilities"] = caps
        if samp:
            prof["sampling"] = samp
        return prof

    def h_profiles_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        profiles, default = self._profiles()
        # Global capability defaults (config `agent.capabilities:`) so the UI
        # can show each toggle's effective state when a profile is silent.
        agent_cfg = self.cfg.agent if isinstance(self.cfg.agent, dict) else {}
        global_caps = agent_cfg.get("capabilities")
        return {"profiles": profiles, "default_profile": default,
                "known_tools": list(self.KNOWN_TOOLS),
                "capability_toggles": dict(self.CAPABILITY_TOGGLES),
                "global_capabilities": {
                    k: bool((global_caps or {}).get(k, False))
                    for k in self.CAPABILITY_TOGGLES}}

    def h_profile_put(self, q: Dict[str, str], b: Dict[str, Any],
                      name: str) -> Dict[str, Any]:
        prof = self._validate_profile(name, b)
        profiles, default = self._profiles()
        existing = {p.get("name"): i for i, p in enumerate(profiles)}
        if name in existing:
            profiles[existing[name]] = prof
        else:
            profiles.append(prof)
        self._write_profiles(profiles, default)
        return {"ok": True, "profile": prof}

    def h_profile_delete(self, q: Dict[str, str], b: Dict[str, Any],
                         name: str) -> Dict[str, Any]:
        profiles, default = self._profiles()
        if not any(p.get("name") == name for p in profiles):
            raise HttpError(404, f"no profile {name}")
        profiles = [p for p in profiles if p.get("name") != name]
        if default == name:
            default = None
        self._write_profiles(profiles, default)
        return {"ok": True}

    def h_profiles_set_default(self, q: Dict[str, str],
                               b: Dict[str, Any]) -> Dict[str, Any]:
        default = b.get("default")
        profiles, _cur = self._profiles()
        if default is not None:
            default = str(default)
            if not any(p.get("name") == default for p in profiles):
                raise HttpError(404, f"no profile {default}")
        self._write_profiles(profiles, default)
        return {"ok": True, "default_profile": default}

    def _resolve_profile(self, name: Optional[str]) -> Optional[Dict[str, Any]]:
        """Look up a profile for /api/chat/stream. Empty/'default' => config
        default; unknown name => 404."""
        profiles, default = self._profiles()
        if not name or name in ("default", "none"):
            name = default
            if not name:
                return None
        for p in profiles:
            if p.get("name") == name:
                return p
        raise HttpError(404, f"no profile {name}", code="unknown_profile")

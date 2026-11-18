"""Easy training (turnkey loop, one or more agent profiles) plus
the one-shot feature preset enable-all.
"""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


from ...core.config import ToasterLoopConfig, load_config
from ...serving import instances as srv_inst
from ...training.easy import (
    OPTIMAL_CURATION,
    OPTIMAL_GOLDENS_MINE,
    OPTIMAL_TRAINING,
    OPTIMAL_VALIDATION,
    EasyTrainingConfig,
    evaluate_readiness,
    load_state as easy_load_state,
    mine_goldens_pass,
    optimal_config_patches,
    save_state as easy_save_state,
    validate_profile as easy_validate_profile,
)
from .base import (ApiContext, HttpError, ITERATE_RUNS_DIRNAME, ApiResponse,
                   logger, _utcnow)


class EasyRoutes(ApiContext):
    """Easy training (turnkey loop, one or more agent profiles) plus
the one-shot feature preset enable-all.
    """

    def _easy_cfg(self) -> EasyTrainingConfig:
        return EasyTrainingConfig.from_yaml(
            (self.cfg.easy_training if isinstance(self.cfg.easy_training, dict)
             else {}) or {})

    def _easy_readiness(self, ec: EasyTrainingConfig) -> Dict[str, Any]:
        from ...training.scheduler import resource_gate_ok

        tc = self._training_config()
        gate = resource_gate_ok(tc)
        store = self._store()
        try:
            return evaluate_readiness(store, ec, tc, gate)
        finally:
            store.close()

    def _easy_deploy(self, model_id: str, server_instance: str) -> Dict[str, Any]:
        """Hot-swap the easy profile's bound server to a new GGUF (best-effort).

        Spawn instances are restarted on the promoted model's GGUF (the
        instance definition is updated first so the swap survives restarts);
        attach instances are left alone — an external server is not ours to
        re-point. Never raises: a failed deploy is reported, not fatal, since
        the model is already promoted and the loop's data is intact.
        """
        out: Dict[str, Any] = {"attempted": True, "ok": False,
                               "server": server_instance, "model_id": model_id,
                               "detail": None}
        try:
            reg = self._registry()
            try:
                m = reg.get_model(model_id)
            finally:
                reg.close()
            if m is None or not m.gguf_path:
                out["detail"] = "promoted model has no GGUF on disk"
                return out
            inst = srv_inst.get_instance(self.home, server_instance,
                                         cfg=self.cfg)
            if inst is None:
                out["detail"] = f"server instance {server_instance} vanished"
                return out
            if inst.get("attach_url"):
                out["detail"] = ("attached server is external — deploy "
                                 f"{model_id} there manually")
                return out
            # Update the definition, then bounce the running process (or just
            # record the new gguf when it wasn't running).
            inst = dict(inst)
            inst["gguf"] = str(m.gguf_path)
            srv_inst.save_instance(self.home, inst)
            st = self._instance_state(server_instance)
            if srv_inst.state_alive(st):
                self._stop_instance(server_instance)
                start = self.handle("POST",
                                    f"/api/servers/{server_instance}/start",
                                    body={"force": True})
                # A failed restart must NOT report success: the definition is
                # updated (so the next manual start uses the new GGUF), but
                # the profile's server is down right now — say so.
                if getattr(start, "status", 500) >= 400:
                    err: Dict[str, Any] = {}
                    if isinstance(getattr(start, "body", None), dict):
                        err = start.body.get("error") or {}
                    out["detail"] = (
                        f"definition updated to {model_id} but restart failed"
                        + (f": {err.get('message')}" if err.get("message")
                           else " — start it from the Serve panel"))
                    return out
            out["ok"] = True
            out["detail"] = f"{server_instance} now serves {model_id}"
            self.bus.publish("easy_training.deployed",
                             {"model_id": model_id, "server": server_instance})
        except Exception as e:  # deploy is best-effort by design
            out["detail"] = f"{type(e).__name__}: {e}"
        return out

    def _easy_after_run(self, run_id: str, record: Dict[str, Any],
                        ec: EasyTrainingConfig,
                        profile: Dict[str, Any]) -> None:
        """Easy-loop completion hook: ledger the result and auto-deploy.

        Called from the iterate worker thread AFTER the run record is saved
        (wired through submit_iterate's on_done)."""
        try:
            result = record.get("result") or {}
            state = record.get("state")
            patch: Dict[str, Any] = {
                "last_run": {
                    "run_id": run_id, "state": state,
                    "finished": record.get("finished"),
                    "model_id": result.get("model_id"),
                    "dataset_id": result.get("dataset_id"),
                    "promoted": bool(result.get("promoted")),
                },
            }
            deploy_info: Optional[Dict[str, Any]] = None
            if (state == "done" and result.get("promoted")
                    and ec.auto_deploy and result.get("model_id")):
                si = str(profile.get("server_instance") or "").strip()
                if si:
                    deploy_info = self._easy_deploy(str(result["model_id"]), si)
                    patch["last_run"]["deploy"] = deploy_info
            easy_save_state(self.home, patch)
            self.bus.publish("easy_training.run.finished", {
                "run_id": run_id, "state": state,
                "profile": ec.profile,
                "model_id": result.get("model_id"),
                "promoted": bool(result.get("promoted")),
                "deploy": deploy_info,
            })
        except Exception:  # completion bookkeeping must never break the run
            logger.exception("easy training: completion hook failed for %s",
                             run_id)

    def _easy_agent_entry(self, ec: EasyTrainingConfig, name: str,
                          profiles: List[Dict[str, Any]],
                          servers: Dict[str, Any],
                          run_recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Per-agent view for one profile in the easy loop.

        Same shape as the legacy top-level fields (profile/server/problem/
        readiness/active_run/runs) so the panel renders each agent card with
        the code it already has; `name` lets it key them apart."""
        match = next((p for p in profiles if p.get("name") == name), None)
        problem = None
        server = None
        if match is None:
            problem = (f"configured profile “{name}” no longer exists — "
                       "remove it or pick one below")
        else:
            si = str(match.get("server_instance") or "").strip()
            server = servers.get(si)
            if not si:
                problem = ("the configured profile is not bound to a "
                           "server instance")
        server_state = (self._instance_state(str(server.get("id")))
                        if server else None)
        # Readiness scoped to THIS agent (single-profile projection).
        readiness = (self._easy_readiness(ec.for_profile(name))
                     if ec.enabled and match else None)
        runs = [r for r in run_recs if r.get("name") == f"easy:{name}"]
        active = next((r for r in runs if r.get("state") == "running"), None)
        return {
            "profile": name,
            "bound": match is not None,
            "server": ({"id": server.get("id"), "name": server.get("name"),
                        "model_name": (server_state or {}).get("model_name")
                        or server.get("model_name"),
                        "running": srv_inst.state_alive(server_state)}
                       if server else None),
            "problem": problem,
            "readiness": readiness,
            "active_run": active,
            "runs": runs[:10],
        }

    def h_easy_get(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        ec = self._easy_cfg()
        profiles, _default = self._profiles()
        servers = {str(i.get("id")): i
                   for i in srv_inst.load_instances(self.home, cfg=self.cfg)}
        # All iterate records once (newest first); per-agent views filter by
        # their `easy:<profile>` run name without re-reading the runs dir.
        run_recs: List[Dict[str, Any]] = []
        runs_dir = self.home / ITERATE_RUNS_DIRNAME
        if runs_dir.is_dir():
            for f in sorted(runs_dir.glob("iter-*.json"), reverse=True)[:40]:
                try:
                    run_recs.append(json.loads(f.read_text()))
                except (OSError, json.JSONDecodeError):
                    continue
        agents = [self._easy_agent_entry(ec, n, profiles, servers, run_recs)
                  for n in ec.all_profiles]
        # Legacy top-level fields mirror the PRIMARY agent (ec.profile, or the
        # first configured one) so existing readers keep working unchanged.
        primary_name = ec.profile or (ec.all_profiles[0]
                                      if ec.all_profiles else None)
        primary = next((a for a in agents
                        if a["profile"] == primary_name), None)
        prof = server = readiness = problem = None
        active = None
        runs: List[Dict[str, Any]] = []
        if primary:
            problem = primary["problem"]
            server = primary["server"]
            readiness = primary["readiness"]
            active = primary["active_run"]
            runs = primary["runs"]
            if primary["bound"]:
                prof = next((p for p in profiles
                             if p.get("name") == primary_name), None)
        # Stale stamps: unclaimed traces whose agent_profile matches no known
        # profile (typical after a profile rename — old chats keep the old
        # name and silently stop counting toward readiness). 'toaster_loop' is the
        # legacy backend stamp (pre-easy traces) and never stale. Reported so
        # the panel can nudge instead of showing an unexplained zero gauge.
        stale_stamps: List[Dict[str, Any]] = []
        if ec.enabled and agents:
            try:
                names = {str(p.get("name")) for p in profiles}
                store = self._store()
                try:
                    counts: Dict[str, int] = {}
                    for r in store.query_for_curation(include_curated=False):
                        ap = str(r.agent_profile or "").strip()
                        if ap and ap not in names and ap != "toaster_loop":
                            counts[ap] = counts.get(ap, 0) + 1
                    stale_stamps = [{"profile": k, "traces": v} for k, v
                                    in sorted(counts.items())]
                finally:
                    store.close()
            except Exception:  # purely informational — never break GET
                stale_stamps = []
        state = easy_load_state(self.home)
        return {
            "config": ec.as_dict(),
            "agents": agents,
            "profile": prof,
            "server": server,
            "problem": problem,
            "stale_stamps": stale_stamps,
            "readiness": readiness,
            "active_run": active,
            "runs": runs[:10],
            "state": state,
        }

    def _easy_preset_diff(self) -> List[Dict[str, Any]]:
        """What the optimal preset WOULD change in curation:/training:/
        validation:/goldens_mine:, as [{section, key, from, to}] — only keys
        whose value actually differs. The enable dialog shows this so a user
        who tuned those sections sees exactly which knobs get overwritten
        before consenting."""
        diff: List[Dict[str, Any]] = []
        for section, patch in (("curation", OPTIMAL_CURATION),
                               ("training", OPTIMAL_TRAINING),
                               ("validation", OPTIMAL_VALIDATION),
                               ("goldens_mine", OPTIMAL_GOLDENS_MINE)):
            cur = getattr(self.cfg, section, None)
            cur = cur if isinstance(cur, dict) else {}
            for key, to in patch.items():
                if cur.get(key) != to:
                    diff.append({"section": section, "key": key,
                                 "from": cur.get(key), "to": to})
        return diff

    def h_easy_enable(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        name = str(b.get("profile") or "").strip()
        profiles, _default = self._profiles()
        try:
            prof = easy_validate_profile(profiles, name)
        except ValueError as e:
            raise HttpError(400, str(e)) from e
        # Optional extra agents (multi-agent loop): every name must exist and
        # be server-bound, same bar as the primary. `profile` stays the primary.
        extras_raw = b.get("profiles")
        extras: List[str] = []
        if extras_raw is not None:
            if not isinstance(extras_raw, list):
                raise HttpError(400, "profiles must be a list of profile names")
            for extra in extras_raw:
                ename = str(extra or "").strip()
                if not ename or ename == name:
                    continue
                try:
                    easy_validate_profile(profiles, ename)
                except ValueError as e:
                    raise HttpError(400, str(e)) from e
                if ename not in extras:
                    extras.append(ename)
        # Preview mode: report the preset diff without writing anything.
        if b.get("preview"):
            return {"ok": True, "diff": self._easy_preset_diff()}
        auto_deploy = bool(b.get("auto_deploy"))
        apply_preset = bool(b.get("apply_preset", True))
        loop_all = [str(prof["name"])] + extras
        ec = EasyTrainingConfig(enabled=True, profile=str(prof["name"]),
                                profiles=loop_all, auto_deploy=auto_deploy)
        patches = optimal_config_patches(ec) if apply_preset else [
            {"section": "easy_training", "patch": ec.as_dict()}]
        # All-or-nothing: snapshot the config file and restore it if any
        # patch fails, so a partial enable (enabled=true with a half-applied
        # preset) can never persist.
        cfg_path = self._config_path()
        snapshot = (cfg_path.read_bytes() if cfg_path.is_file() else None)

        def _rollback() -> None:
            try:
                if snapshot is None:
                    cfg_path.unlink(missing_ok=True)
                    self.cfg = ToasterLoopConfig()
                else:
                    cfg_path.write_bytes(snapshot)
                    self.cfg = load_config(str(cfg_path))
            except Exception:  # last resort: keep serving, file may be mixed
                logger.exception("easy training: config rollback failed")

        for body in patches:
            r = self.handle("PUT", "/api/config", body=body)
            if r.status >= 400:
                _rollback()
                err = r.body.get("error", {}) if isinstance(r.body, dict) else {}
                raise HttpError(r.status,
                                str(err.get("message", "config write failed")),
                                code=str(err.get("code", "internal_error")))
        self.bus.publish("easy_training.enabled",
                         {"profile": ec.profile, "auto_deploy": auto_deploy})
        return {"ok": True, "config": ec.as_dict(),
                "diff": self._easy_preset_diff() if apply_preset else [],
                "note": ("Easy training on: chat with this profile feeds the "
                         "buffer; run it until the readiness gauge fills, "
                         "then Train now closes the loop.")}

    def h_easy_disable(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        # Section-patch (not root): keeps profile/auto_deploy keys intact so a
        # later re-enable remembers them.
        r = self.handle("PUT", "/api/config", body={
            "section": "easy_training",
            "patch": {"enabled": False}})
        if r.status >= 400:
            err = r.body.get("error", {}) if isinstance(r.body, dict) else {}
            raise HttpError(r.status,
                            str(err.get("message", "config write failed")),
                            code=str(err.get("code", "internal_error")))
        self.bus.publish("easy_training.disabled", {})
        return {"ok": True}

    def h_features_enable_all(self, q: Dict[str, str],
                              b: Dict[str, Any]) -> Dict[str, Any]:
        """Turn on every safe-and-useful opt-in feature (Options panel button).

        Same contract as h_easy_enable: `{"preview": true}` writes nothing and
        returns the diff; an apply is all-or-nothing (config file snapshot +
        restore on any failed patch, so a half-enabled preset can never
        persist). Each patch goes through the normal PUT /api/config path, so
        validation, .bak backup and per-section merge semantics are identical
        to flipping each card by hand. What is covered vs deliberately skipped
        lives in core/presets.py (easy_training loop, janitor, unmeasured GPU
        knobs, and conditional LSP/calibration)."""
        from ...core.presets import enable_all_diff, enable_all_patches

        if b.get("preview"):
            return {"ok": True,
                    "diff": enable_all_diff(self.cfg, home=str(self.home))}

        patches = enable_all_patches(self.cfg, home=str(self.home))
        cfg_path = self._config_path()
        snapshot = (cfg_path.read_bytes() if cfg_path.is_file() else None)

        def _rollback() -> None:
            # Snapshot restore AND in-memory reload must both happen even
            # when no config file existed before (fresh install): load_config
            # raises FileNotFoundError on a missing path, which used to leave
            # self.cfg stale-with-half-the-preset while the file was gone.
            try:
                if snapshot is None:
                    cfg_path.unlink(missing_ok=True)
                    self.cfg = ToasterLoopConfig()
                else:
                    cfg_path.write_bytes(snapshot)
                    self.cfg = load_config(str(cfg_path))
            except Exception:  # last resort: keep serving, file may be mixed
                logger.exception("enable-all: config rollback failed")

        diff = enable_all_diff(self.cfg, home=str(self.home))
        for body in patches:
            r = self.handle("PUT", "/api/config", body=body)
            if r.status >= 400:
                _rollback()
                err = r.body.get("error", {}) if isinstance(r.body, dict) else {}
                raise HttpError(r.status,
                                str(err.get("message", "config write failed")),
                                code=str(err.get("code", "internal_error")))
        self.bus.publish("features.enabled_all", {"changed": len(diff)})
        return {"ok": True, "changed": diff,
                "note": ("Features enabled: knowledge (skills/playbooks/"
                         "indexing), learning pipeline (RFT, evolution, "
                         "golden mining, shadow eval), routing, watchdog"
                         + (" — restart running servers to pick up "
                            "watchdog/routing changes." if diff else "."))}

    def _easy_write_loop(self, names: List[str], auto_deploy: Optional[bool]
                         = None) -> EasyTrainingConfig:
        """Persist a new agent list; returns the resulting config.

        Single-agent lists collapse back to the legacy shape (profile only, no
        profiles key) so removing everyone but one leaves the same config file
        an original single-profile enable would have written."""
        primary = names[0] if names else None
        patch: Dict[str, Any] = {"profile": primary}
        patch["profiles"] = list(names) if len(names) > 1 else None
        if auto_deploy is not None:
            patch["auto_deploy"] = bool(auto_deploy)
        r = self.handle("PUT", "/api/config", body={
            "section": "easy_training", "patch": patch})
        if r.status >= 400:
            err = r.body.get("error", {}) if isinstance(r.body, dict) else {}
            raise HttpError(r.status,
                            str(err.get("message", "config write failed")),
                            code=str(err.get("code", "internal_error")))
        return self._easy_cfg()

    def h_easy_add(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Add one agent profile to the easy loop (must exist + be server-bound).

        Works while the loop is enabled; when disabled it just grows the saved
        list so a later enable picks everyone up. No preset re-application —
        quality machinery is section-global and already armed by the first
        enable (enable it fresh if it was never on)."""
        name = str(b.get("profile") or "").strip()
        profiles, _default = self._profiles()
        try:
            easy_validate_profile(profiles, name)
        except ValueError as e:
            raise HttpError(400, str(e)) from e
        ec = self._easy_cfg()
        current = ec.all_profiles
        if name in current:
            return {"ok": True, "config": ec.as_dict(),
                    "note": f"“{name}” is already in the easy loop"}
        new_ec = self._easy_write_loop(current + [name])
        self.bus.publish("easy_training.agent_added",
                         {"profile": name, "profiles": new_ec.all_profiles})
        return {"ok": True, "config": new_ec.as_dict(),
                "note": (f"“{name}” joined the easy loop: chat with it to "
                         "feed its slice; Train now runs per-agent loops.")}

    def h_easy_remove(self, q: Dict[str, str], b: Dict[str, Any]
                      ) -> Dict[str, Any]:
        """Remove one agent from the easy loop (traces stay; only scoping ends)."""
        name = str(b.get("profile") or "").strip()
        ec = self._easy_cfg()
        current = ec.all_profiles
        if name not in current:
            raise HttpError(400, f"“{name}” is not in the easy loop")
        remaining = [p for p in current if p != name]
        new_ec = self._easy_write_loop(remaining)
        self.bus.publish("easy_training.agent_removed",
                         {"profile": name, "profiles": new_ec.all_profiles})
        return {"ok": True, "config": new_ec.as_dict()}

    def h_easy_run(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        ec = self._easy_cfg()
        if not ec.enabled or not ec.all_profiles:
            raise HttpError(400, "easy training is not enabled — pick a "
                                 "profile and enable it first",
                            code="easy_disabled")
        # Which agent to train: explicit body.profile (multi-agent panels run
        # one loop per agent) else the primary — legacy callers unchanged.
        target = str(b.get("profile") or "").strip() or ec.profile \
            or ec.all_profiles[0]
        if target not in ec.all_profiles:
            raise HttpError(
                400, f"“{target}” is not in the easy loop "
                f"({', '.join(ec.all_profiles)})", code="unknown_profile")
        profiles, _default = self._profiles()
        try:
            prof = easy_validate_profile(profiles, target)
        except ValueError as e:
            raise HttpError(400, str(e)) from e
        dry_run = bool(b.get("dry_run"))
        if not dry_run:
            readiness = self._easy_readiness(ec.for_profile(target))
            reasons = []
            if not readiness["trigger"]:
                th = {"min_traces": readiness["min_traces"],
                      "min_score": readiness["min_score"]}
                if readiness["unclaimed"] < th["min_traces"]:
                    reasons.append(
                        f"only {readiness['unclaimed']} of "
                        f"{th['min_traces']} traces — chat more with this "
                        "profile (or lower the threshold in Settings)")
                ms = readiness["mean_score"]
                if ms is None:
                    reasons.append("no scored traces yet — rate a few chats")
                elif ms < th["min_score"]:
                    reasons.append(
                        f"mean score {ms:.2f} below {th['min_score']}")
            if readiness["gate"].get("ok") is False:
                reasons.extend(readiness["gate"].get("reasons") or [])
            if reasons:
                raise HttpError(409, "; ".join(reasons), code="not_ready")
        # Reuse the iterate engine with easy-loop semantics: profile-scoped
        # curation, sft kind (the preset curates both formats; sft trains from
        # sft.jsonl and dpo pairs ride along in the dataset). On completion
        # _easy_after_run ledgers the result and hot-deploys when configured.
        # Before training: one best-effort golden-mining pass so 👍-thumbed
        # chats since the last run gate the model about to be promoted
        # (goldens_mine.enabled gates it; stdlib-only, never breaks the run —
        # failures are logged inside mine_goldens_pass and recorded here).
        mine = mine_goldens_pass(self.cfg, self.home)
        easy_save_state(self.home, {"last_mine": {
            "ts": _utcnow(), "profile": target, **mine}})
        # Single-agent projection so the completion hook (auto-deploy to THIS
        # agent's server, bus event naming) behaves exactly like the legacy
        # one-profile loop. The cross-process loop lock still serializes runs
        # globally — per-agent loops queue behind each other, never double-
        # claim the shared buffer.
        run = self.submit_iterate(
            {
                "name": f"easy:{target}",
                "kind": str(b.get("kind") or "sft"),
                "dry_run": dry_run,
                "profile": target,
                # Optional human name for the trained model (UI); forwarded to
                # the iterate pipeline as display_name.
                "model_name": str(b.get("model_name") or "").strip()[:80] or None,
            },
            on_done=lambda record: self._easy_after_run(
                record.get("run_id", ""), record, ec.for_profile(target), prof),
        )
        easy_save_state(self.home, {"last_start": {
            "run_id": run["run_id"], "ts": _utcnow(), "profile": target}})
        return ApiResponse(202, run)

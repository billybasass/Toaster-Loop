"""Hardware-page operations: inference-engine registry,
installable engine catalog, llama.cpp runtime setup,
doctor (WP7), bench (WP1) and calibrate (WP3).
"""

# Domain slice of the former monolithic dashboard/app.py: a pure move,
# no behavior change. Handlers resolve through DashboardApp attribute
# lookup (see ROUTES in app.py).

from __future__ import annotations

from typing import Any, Dict, Optional


from ...serving import instances as srv_inst
from .base import ApiContext, ApiResponse, BackgroundTask, HttpError


class HardwareOpsRoutes(ApiContext):
    """Hardware-page operations: inference-engine registry,
installable engine catalog, llama.cpp runtime setup,
doctor (WP7), bench (WP1) and calibrate (WP3).
    """

    _BACKENDS = ("cuda", "rocm", "metal", "vulkan", "cpu")

    def _engine_row(self, eng: Dict[str, Any]) -> Dict[str, Any]:
        from ...serving import engines as engines_mod

        row = dict(eng)
        row.setdefault("model_kind",
                       "gguf" if eng.get("builtin") else None)
        try:
            row["installed"] = bool(engines_mod.probe(eng)["found"])
        except Exception:  # defensive: probe must never break listing
            row["installed"] = False
        row["in_use_by"] = (engines_mod.engines_in_use(self.home, str(eng["id"]))
                            if not eng.get("builtin") else [])
        return row

    def h_engines_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        from ...serving import engine_installs as ei
        from ...serving import engines as engines_mod

        # Recipe upgrades (kind/binary/args changes) reach existing installs
        # here: idempotent, stat()-cheap, and a no-op once records are current.
        try:
            if ei.sync_installed_recipes(self.home):
                self.bus.publish("engines.changed", {"event": "synced"})
        except Exception:  # defensive: listing must never fail on a sync
            pass
        rows = [self._engine_row(e) for e in engines_mod.list_engines(self.home)]
        default_engine = str((self.cfg.serve or {}).get("default_engine")
                             or "").strip() or None
        return {"engines": rows, "default_engine": default_engine}

    def h_engines_create(self, q: Dict[str, str], b: Dict[str, Any]) -> ApiResponse:
        from ...serving import engines as engines_mod
        from ...serving.supervisor import ServeError as _ServeErr

        try:
            eng = engines_mod.validate_engine_body(b)
        except _ServeErr as e:
            raise HttpError(400, str(e)) from e
        taken = [str(e.get("id")) for e in engines_mod.list_engines(self.home)]
        eng["id"] = engines_mod.new_engine_id(eng["name"], taken)
        engines_mod.save_engine(self.home, eng)
        self.bus.publish("engines.changed", {"event": "created",
                                             "id": eng["id"]})
        return ApiResponse(201, {"ok": True, "engine": self._engine_row(eng)})

    def h_engine_put(self, q: Dict[str, str], b: Dict[str, Any],
                     engine_id: str) -> Dict[str, Any]:
        from ...serving import engines as engines_mod
        from ...serving.supervisor import ServeError as _ServeErr

        existing = engines_mod.get_engine(self.home, engine_id)
        if existing is None:
            raise HttpError(404, f"no engine {engine_id}", code="unknown_engine")
        if existing.get("builtin"):
            raise HttpError(400, "the built-in llama.cpp engine cannot be "
                                 "modified", code="reserved_engine")
        b = dict(b)
        b["id"] = engine_id
        try:
            eng = engines_mod.validate_engine_body(b, existing=existing)
        except _ServeErr as e:
            raise HttpError(400, str(e)) from e
        eng["id"] = engine_id
        engines_mod.save_engine(self.home, eng)
        self.bus.publish("engines.changed", {"event": "updated",
                                             "id": engine_id})
        return {"ok": True, "engine": self._engine_row(eng)}

    def h_engine_delete(self, q: Dict[str, str], b: Dict[str, Any],
                        engine_id: str) -> Dict[str, Any]:
        from ...serving import engines as engines_mod

        if engine_id == engines_mod.BUILTIN_ID:
            raise HttpError(400, "the built-in llama.cpp engine cannot be "
                                 "deleted", code="reserved_engine")
        if engines_mod.get_engine(self.home, engine_id) is None:
            raise HttpError(404, f"no engine {engine_id}", code="unknown_engine")
        used = engines_mod.engines_in_use(self.home, engine_id)
        if used:
            raise HttpError(409, f"engine is used by instance(s): "
                                 f"{', '.join(used)}", code="in_use")
        engines_mod.delete_engine_file(self.home, engine_id)
        removed_build = False
        # A sable-managed llama.cpp backend build (llama-cpp-<backend>):
        # deleting the engine also removes its dedicated checkout under
        # SABLE_HOME so the two views can never disagree.
        if engine_id.startswith("llama-cpp-"):
            try:
                from ...serving import runtime as _rt
                removed_build = bool(_rt.uninstall_llama_cpp(
                    backend=engine_id[len("llama-cpp-"):]))
            except Exception:  # defensive: registration already deleted
                removed_build = False
        self.bus.publish("engines.changed", {"event": "deleted",
                                             "id": engine_id})
        return {"ok": True, "removed_build": removed_build}

    def h_engine_catalog(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Installable-engine catalog with per-recipe install status + host
        compatibility verdicts. stat() probes only; the GPU probe behind
        detect_host() is process-cached, so this stays cheap to poll."""
        from ...serving import engine_installs as ei

        host = ei.detect_host()
        rows = []
        for row in ei.list_catalog(self.home):
            rid = str(row.get("id"))
            task = self._engine_install_tasks.get(rid)
            run = (task.record if task else None) or {}
            row["installing"] = run.get("state") == "running"
            try:
                row["compat"] = ei.recipe_compat(rid, host)
            except Exception as e:  # defensive: verdict is display-only
                row["compat"] = {"ok": True, "blocked": None,
                                 "warnings": [f"compat check failed: {e}"],
                                 "variant": None, "backend": None}
            rows.append(row)
        return {"catalog": rows, "host": host}

    def h_engine_install_start(self, q: Dict[str, str], b: Dict[str, Any],
                               recipe_id: str) -> Dict[str, Any]:
        """Start (or re-run/update) a catalog engine install in the background.

        Body: {ref?: str, update?: bool, register?: bool (default True)}.
        Progress streams as `engine.install` SSE events + GET the same path.
        On success the engine is registered in the engines registry too, so it
        shows up in the Inference-engines card / Serve page immediately.
        """
        from ...serving import engine_installs as ei

        try:
            recipe = ei.get_recipe(recipe_id)
        except KeyError:
            raise HttpError(404, f"unknown engine recipe {recipe_id!r}",
                            code="unknown_recipe") from None

        ref = (b.get("ref") or "").strip() or None
        do_update = bool(b.get("update"))
        do_register = b.get("register") is not False  # default True
        backend = (b.get("backend") or "").strip() or None
        force = bool(b.get("force"))
        name = str(recipe["name"])

        # Fail fast on a known-incompatible host (e.g. vLLM on Windows) so the
        # user sees the reason instead of a doomed pip/build in the log. The
        # install itself re-checks; force skips both gates.
        if not force:
            try:
                compat = ei.recipe_compat(recipe_id)
            except Exception:  # defensive: never block on a broken probe
                compat = {"ok": True}
            if not compat.get("ok"):
                raise HttpError(400, str(compat.get("blocked")
                                         or "incompatible host"),
                                code="incompatible_host")

        def _work(record: Dict[str, Any], log) -> None:
            st = ei.install_recipe(recipe_id, home=self.home, ref=ref,
                                   update=do_update, backend=backend,
                                   force=force, log_fn=log)
            record["state_file"] = st
            if do_register:
                self._register_installed_engine(recipe_id, st)
            record["state"] = "done"

        task = self._install_task(recipe_id)
        # try_start's 409 keeps its message generic; wrap it with the engine
        # name so the UI can attribute the conflict.
        try:
            task.try_start(
                _work, conflict_msg=f"an install of {name} is already running",
                initial={"recipe": recipe_id, "ref": ref, "state_file": None},
                done_extra=lambda r: {"recipe": recipe_id})
        except HttpError as e:
            if e.code == "already_running":
                raise HttpError(409, f"an install of {name} is already "
                                     "running",
                                code="already_running") from None
            raise
        return {"ok": True, "state": "running",
                "note": f"{name} install running in background; watch "
                        "engine.install events or poll the same URL"}

    def _install_task(self, recipe_id: str) -> BackgroundTask:
        """Per-recipe install run-slot (created on first use). Two different
        engines can install concurrently; one cannot double-run."""
        with self._engine_install_lock:
            task = self._engine_install_tasks.get(recipe_id)
            if task is None:
                task = BackgroundTask(
                    self.bus, "engine.install",
                    line_extra=lambda _r: {"recipe": recipe_id})
                self._engine_install_tasks[recipe_id] = task
            return task

    def _register_installed_engine(self, recipe_id: str,
                                   state: Dict[str, Any]) -> None:
        """Create-or-refresh the engines-registry record for an install. The
        registry entry keeps the same id as the recipe so instances can bind
        to it; user edits of other fields survive (only binary/args/kind/notes
        track the install)."""
        from ...serving import engine_installs as ei
        from ...serving import engines as engines_mod

        body = ei.engine_body_for(recipe_id, state, home=self.home)
        existing = engines_mod.get_engine(self.home, recipe_id)
        if existing is not None and not existing.get("builtin"):
            merged = dict(existing)
            user_notes = str(existing.get("notes") or "").strip()
            merged.update(body)
            if user_notes:  # re-install refreshes binary/args, keeps notes
                merged["notes"] = user_notes
            merged["id"] = recipe_id
            eng = engines_mod.validate_engine_body(merged, existing=existing)
            eng["id"] = recipe_id
        else:
            eng = engines_mod.validate_engine_body(body)
            taken = [str(e.get("id")) for e in engines_mod.list_engines(self.home)]
            # Stable id: the recipe id when free, else the usual dedup.
            eng["id"] = recipe_id if recipe_id not in taken \
                else engines_mod.new_engine_id(body["name"], taken)
        engines_mod.save_engine(self.home, eng)
        self.bus.publish("engines.changed", {"event": "installed",
                                             "id": eng["id"]})

    def h_engine_install_status(self, q: Dict[str, str], b: Dict[str, Any],
                                recipe_id: str) -> Dict[str, Any]:
        from ...serving import engine_installs as ei

        try:
            ei.get_recipe(recipe_id)
        except KeyError:
            raise HttpError(404, f"unknown engine recipe {recipe_id!r}",
                            code="unknown_recipe") from None
        task = self._engine_install_tasks.get(recipe_id)
        rec = task.record if task else None
        try:
            tail = max(1, min(int(q.get("tail") or 40), 500))
        except (TypeError, ValueError):
            tail = 40
        if not rec:
            return {"state": "idle",
                    "installed": ei.recipe_installed(self.home, recipe_id)}
        return {**rec, "log": rec["log"][-tail:],
                "installed": ei.recipe_installed(self.home, recipe_id)}

    def _validate_engine_ref(self, inst: Dict[str, Any]) -> None:
        """Reject instance bodies binding an unregistered engine id."""
        from ...serving import engines as engines_mod

        ref = str(inst.get("engine") or "").strip()
        if ref and not engines_mod.is_builtin_ref(ref) and \
                engines_mod.get_engine(self.home, ref) is None:
            raise HttpError(400, f"unknown inference engine {ref!r} — "
                                 "register it on the Hardware page first",
                            code="unknown_engine")

    def h_hardware(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        return self.hardware.snapshot()

    def h_runtime_get(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """GPU + llama.cpp install state for the Hardware panel runtime card.

        The remote-commit check is a network probe; skip with ?check=0 (the
        panel polls, so only the explicit user refresh pays for it).
        """
        from ...serving import runtime

        info = runtime.detect_runtime()
        out: Dict[str, Any] = {
            "system": info.system,
            "machine": info.machine,
            "backend": info.backend,
            "backend_notes": info.backend_notes,
            "gpus": [vars(g) for g in info.gpus],
            "offload_ngl": runtime.gpu_layers(self.cfg.gpu_layers, info),
            "configured": {
                "repo": self.cfg.llama_repo or runtime.LLAMA_CPP_REPO,
                "ref": self.cfg.llama_ref or runtime.LLAMA_CPP_REF,
                "backend": self.cfg.llama_backend,
                "auto_setup": runtime.auto_setup_enabled(self.cfg),
            },
            "install": runtime.read_install_state(),
            # every managed install (primary + per-backend slots like
            # llama.cpp-vulkan) so the panel can show/label them individually
            "installs": runtime.list_installs(),
            "backends": list(self._BACKENDS),
        }
        if q.get("check", "1") != "0":
            try:
                out["update"] = runtime.check_llama_update(
                    self.cfg, backend=q.get("backend") or None)
            except Exception as e:  # network probes must never break the card
                out["update"] = {"installed": False, "update_available": False,
                                 "reason": f"check failed: {e}"}
        return out

    def h_runtime_setup(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Start (or restart) the managed llama.cpp build in a background thread.

        Body: {backend?: cuda|rocm|metal|vulkan|cpu, repo?, ref?, update?}.
        Progress streams as `runtime.setup` SSE events + GET /api/runtime/setup.
        """
        from ...serving import runtime

        backend = b.get("backend") or None
        if backend == "auto":
            backend = None
        if backend is not None and backend not in self._BACKENDS:
            raise HttpError(400, f"unknown backend {backend!r}", code="bad_request")
        repo = (b.get("repo") or "").strip() or None
        ref = (b.get("ref") or "").strip() or None
        do_update = bool(b.get("update"))

        def _work(record: Dict[str, Any], log) -> None:
            runtime.setup_llama_cpp(backend=backend, log_fn=log,
                                    repo=repo, ref=ref, update=do_update)
            record["state"] = "done"
            if backend:
                # Each backend install is also a first-class engine entry
                # (llama-cpp-<backend>) so Vulkan/ROCm builds appear side by
                # side in the Inference-engines card with update/delete.
                try:
                    self._register_llama_backend_engine(backend)
                except Exception as e:  # registration must not fail a build
                    log(f"warning: engine registration failed: {e}")

        self._runtime_task.try_start(
            _work, conflict_msg="a llama.cpp setup is already running",
            initial={"backend": backend, "repo": repo, "ref": ref})
        return {"ok": True, "state": "running",
                "note": "llama.cpp build running in background; watch "
                        "runtime.setup events or poll GET /api/runtime/setup"}

    def h_runtime_setup_status(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        rec = self._runtime_task.record
        if not rec:
            return {"state": "idle"}
        tail = int(q.get("tail", "40"))
        return {**rec, "log": rec["log"][-tail:]}

    def _register_llama_backend_engine(self, backend: str) -> None:
        """Create-or-refresh the engines-registry record for one managed
        llama.cpp backend build (id ``llama-cpp-<backend>``). The binary is the
        built llama-server path; an empty args_template means sable builds the
        full argv like it does for the built-in engine. Refuses to touch the
        built-in slot when the build *is* the primary auto-setup install — that
        one stays represented by the immutable built-in engine row."""
        from ...serving import engines as engines_mod
        from ...serving import runtime

        root, state_be = runtime.install_root_for(backend)
        st = runtime.read_install_state(backend=state_be)
        if not st:
            return
        if state_be is None:
            return  # primary slot == the built-in engine row (immutable);
                    # a duplicate entry could delete the auto-setup install
        exe = runtime.llama_server_path(root)
        if exe is None:
            return  # nothing built yet — nothing to register
        eng_id = f"llama-cpp-{str(backend).lower()}"
        taken = [str(e.get("id")) for e in engines_mod.list_engines(self.home)]
        existing = engines_mod.get_engine(self.home, eng_id)
        body = {
            "name": f"llama.cpp ({str(backend).lower()})",
            "kind": "openai",
            "binary": str(exe),
            # default llama-server argv (--model/--host/--port); richer
            # run_settings translation stays a built-in-engine feature
            "args_template": [],
            "notes": (f"sable-managed llama.cpp {str(backend).lower()} build "
                      f"({st.get('repo', '?')} @ {st.get('ref', '?')} "
                      f"commit {st.get('commit', '?')})"),
        }
        if existing is not None:
            merged = dict(existing)
            merged.update(body)
            merged["id"] = eng_id
            eng = engines_mod.validate_engine_body(merged, existing=existing)
            eng["id"] = eng_id
        else:
            eng = engines_mod.validate_engine_body(body)
            eng["id"] = eng_id if eng_id not in taken \
                else engines_mod.new_engine_id(body["name"], taken)
        engines_mod.save_engine(self.home, eng)
        self.bus.publish("engines.changed", {"event": "installed",
                                             "id": eng["id"]})

    def h_runtime_sync(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Register every managed llama.cpp backend build as an engine entry.
        Idempotent; covers installs that predate per-backend registration."""
        from ...serving import runtime

        synced = []
        for rec in runtime.list_installs():
            be = str(rec.get("backend") or "").strip()
            if not be:
                continue
            try:
                self._register_llama_backend_engine(be)
                synced.append(be)
            except Exception:  # defensive: one bad slot must not break sync
                continue
        return {"ok": True, "synced": synced}

    def h_doctor_env(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Latest doctor report WITHOUT running anything (panel bootstrap).

        Returns {state: idle|running|done|error, ...}. The checks only run
        when the user clicks Run — never on page open — because vendor probes
        (rocm-smi under GPU load) can block in the kernel past their
        subprocess timeout and must not stall either the browser or a server
        worker thread."""
        rec = self._doctor_task.record
        if not rec:
            return {"state": "idle", "calibration_mode":
                    self._calibration_mode()}
        return {**rec, "calibration_mode": self._calibration_mode()}

    def _start_doctor(self, fix: bool) -> Dict[str, Any]:
        """Kick off collect_checks in a background thread (one at a time)."""
        def _work(record: Dict[str, Any], _log) -> None:
            from dataclasses import asdict

            from ...doctor import collect_checks

            results, rc = collect_checks(fix=fix, out=lambda _s: None)
            record["checks"] = [asdict(r) for r in results]
            record["exit_code"] = rc
            record["state"] = "done"

        self._doctor_task.try_start(
            _work, conflict_msg="a doctor run is already in progress",
            initial={"fix": fix, "checks": [], "exit_code": None},
            done_extra=lambda r: {"exit_code": r.get("exit_code")})
        return {"ok": True, "state": "running"}

    def h_doctor_run(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """POST /api/doctor/run — start the WP7 checks (user action only)."""
        return self._start_doctor(fix=bool(b.get("fix")))

    def h_doctor_env_fix(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """POST /api/doctor/env — legacy alias of /api/doctor/run with
        fix=True (removes stale instance state files whose pid is verified
        gone; foreign pids are never touched — ownership rule lives in the
        doctor/serving code)."""
        return self._start_doctor(fix=bool(b.get("fix", True)))

    def _calibration_mode(self) -> str:
        from ...serving.calibrate import calibration_mode

        try:
            return calibration_mode(self.cfg)
        except Exception:  # defensive: display-only
            return "off"

    def h_bench_list(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Persisted bench records (newest first) + the active noise floor."""
        from ...serving import bench as bench_mod

        recs = []
        for r in bench_mod.load_results():
            gen = (r.get("generation") or {}).get("median")
            recs.append({
                "timestamp": r.get("timestamp"),
                "kind": r.get("kind"),
                "label": r.get("label") or "",
                "model": r.get("model"),
                "gen_median_tps": gen,
                "gen_spread_pct": (r.get("generation") or {}).get("spread_pct"),
                "token_identity": (r.get("token_identity") or {}).get("status"),
                "path": r.get("_path"),
            })
        nf = bench_mod.load_noise_floor()
        return {
            "records": recs,
            "noise_floor": {
                "spread_pct": (nf or {}).get("generation", {}).get("spread_pct")
                if nf else None,
                "label": (nf or {}).get("label"),
            } if nf else None,
        }

    def h_bench_run(self, q: Dict[str, str], b: Dict[str, Any]) -> Dict[str, Any]:
        """Start a bench run in the background (WP1 semantics identical to
        `sable bench`: temperature 0, token-identity check, persisted).

        Body: {runs?: int, max_tokens?: int, label?: str, noise_floor?: bool}.
        Progress: SSE `bench.run` events + GET /api/bench/run. A running
        calibration refuses the GPU (409) and vice versa."""
        from ...serving import bench as bench_mod

        runs = b.get("runs") or bench_mod.DEFAULT_RUNS
        try:
            runs = max(1, min(int(runs), 50))
        except (TypeError, ValueError):
            raise HttpError(400, "runs must be an integer") from None
        max_tokens = b.get("max_tokens") or bench_mod.DEFAULT_MAX_TOKENS
        try:
            max_tokens = max(1, min(int(max_tokens), 4096))
        except (TypeError, ValueError):
            raise HttpError(400, "max_tokens must be an integer") from None
        label = str(b.get("label") or "")[:80]
        noise_floor = bool(b.get("noise_floor"))
        # Multi-instance aware: bench a RUNNING server instance by id (spawn
        # mode attaches to its endpoint — never spawns or stops anything it
        # does not own). Without server_id: legacy behavior (resolve from
        # config, spawn the default if needed).
        server_id = str(b.get("server_id") or "").strip() or None

        target = None
        if server_id:
            inst = next((i for i in srv_inst.load_instances(self.home,
                                                            cfg=self.cfg)
                         if str(i.get("id")) == server_id), None)
            if inst is None:
                raise HttpError(404, f"no such server instance: {server_id}",
                                code="not_found")
            row = self._instance_row(inst)
            if not row.get("running") or not row.get("base_url"):
                raise HttpError(
                    409, f"server instance {server_id} is not running — start "
                         "it on the Serve panel first", code="not_running")
            target = {"base_url": str(row["base_url"]),
                      "model": str(row.get("model_name") or "instance-model")}

        if self._calibrate_task.running():
            raise HttpError(409, "a calibration sweep is running",
                            code="already_running")

        def _work(record: Dict[str, Any], log) -> None:
            supervisor = None
            try:
                if target is not None:
                    base_url, model = (target["base_url"],
                                       target["model"])
                else:
                    from ...cli import _resolve_endpoint
                    from ...core.config import load_config as _load

                    cfg = _load(None)
                    base_url, model, supervisor = _resolve_endpoint(
                        cfg, spawn_if_needed=True)
                log(f"endpoint {base_url} model={model}")
                log(f"measuring {runs} run(s), max_tokens={max_tokens}…")
                res = bench_mod.run_bench(
                    base_url, model, runs=runs, max_tokens=max_tokens,
                    label=label,
                    kind=(bench_mod.KIND_NOISE_FLOOR if noise_floor
                          else bench_mod.KIND_BENCH))
                saved = bench_mod.save_result(res)
                record["result"] = {
                    "generation": res.get("generation"),
                    "prompt_processing": res.get("prompt_processing"),
                    "token_identity": res.get("token_identity"),
                    "model": res.get("model"),
                    "path": str(saved),
                }
                ti = (res.get("token_identity") or {}).get("status")
                record["state"] = "done" if ti == "ok" else "diverged"
            finally:
                if supervisor is not None:
                    try:
                        supervisor.stop()
                    except Exception:  # noqa: BLE001 - best-effort cleanup
                        pass

        self._bench_task.try_start(
            _work, conflict_msg="a bench run is already in progress",
            initial={"kind": "noise_floor" if noise_floor else "bench",
                     "label": label, "runs": runs, "server_id": server_id,
                     "result": None},
            done_extra=lambda r: {"result": r.get("result")})
        return {"ok": True, "state": "running"}

    def h_bench_targets(self, q: Dict[str, str],
                        b: Dict[str, Any]) -> Dict[str, Any]:
        """Benchable targets for the Tuning panel: running server instances
        (any agent may be assigned to them; bench just measures the endpoint)
        + whether a default spawn is possible (gguf_path configured)."""
        rows = []
        try:
            insts = srv_inst.load_instances(self.home, cfg=self.cfg)
        except Exception:  # defensive: panel must still render
            insts = []
        for inst in insts:
            row = self._instance_row(inst)
            if not isinstance(row, dict):
                continue
            rows.append({
                "id": row.get("id"), "name": row.get("name"),
                "running": bool(row.get("running")),
                "model_name": row.get("model_name"),
                "base_url": row.get("base_url") if row.get("running") else None,
            })
        return {"instances": rows,
                "can_spawn_default": bool(self.cfg.gguf_path)}

    def h_bench_run_status(self, q: Dict[str, str],
                           b: Dict[str, Any]) -> Dict[str, Any]:
        rec = self._bench_task.record
        if not rec:
            return {"state": "idle"}
        tail = int(q.get("tail", "40"))
        return {**rec, "log": rec["log"][-tail:]}

    def _calibrate_model_ref(self) -> Optional[str]:
        """GGUF to calibrate: configured gguf_path when readable (same rule
        as the CLI's _calibrate_model_ref without the --model override)."""
        from ...cli import _calibrate_model_ref as cli_ref

        try:
            return cli_ref(self.cfg, None)
        except Exception:  # noqa: BLE001 - display-only resolution
            return None

    def h_calibrate_show(self, q: Dict[str, str],
                         b: Dict[str, Any]) -> Dict[str, Any]:
        """Profiles for this machine (newest first) + the active one for the
        configured model + serve.calibration mode."""
        from ...serving import calibrate as cal_mod

        profs = cal_mod.list_profiles()
        cur_mid = cal_mod.machine_id()
        gguf = self._calibrate_model_ref()
        mhash = cal_mod.model_hash(gguf) if gguf else None
        active_path = None
        if mhash:
            active = cal_mod.load_profile(cur_mid, mhash)
            if active:
                active_path = str(active.get("_path"))
        out = []
        for p in profs:
            out.append({
                "timestamp": p.get("timestamp"),
                "machine_id": p.get("machine_id"),
                "active_machine": p.get("machine_id") == cur_mid,
                "model_hash": (p.get("model_hash") or "")[:12],
                "model_path": p.get("model_path"),
                "adopted_settings": p.get("adopted_settings") or {},
                "threshold_pct": p.get("threshold_pct"),
                "noise_floor_pct": (p.get("noise_floor") or {}).get(
                    "spread_pct"),
                "is_active_profile": str(p.get("_path")) == active_path,
                "path": p.get("_path"),
            })
        # The sweep spawns its OWN llama-server per candidate; anything else
        # holding the GPU skews every measurement. Surface them for the UI.
        busy = []
        try:
            for inst in srv_inst.load_instances(self.home, cfg=self.cfg):
                row = self._instance_row(inst)
                if isinstance(row, dict) and row.get("running"):
                    busy.append({"id": row.get("id"), "name": row.get("name")})
        except Exception:  # defensive: display-only
            pass
        return {"profiles": out, "calibration_mode": self._calibration_mode(),
                "model": gguf, "running_instances": busy}

    def h_calibrate_run(self, q: Dict[str, str],
                        b: Dict[str, Any]) -> Dict[str, Any]:
        """Start a WP3 autotune sweep in the background (identical adoption
        rules to `sable calibrate`: gain > max(3%, noise floor) AND token
        identity). Body: {quick?: bool, runs?: int, knobs?: "a,b"}.

        Progress: SSE `calibrate.run` events + GET /api/calibrate/run."""
        from ...serving import calibrate as cal_mod

        quick = bool(b.get("quick", True))
        runs = b.get("runs") or cal_mod.DEFAULT_SWEEP_RUNS
        try:
            runs = max(1, min(int(runs), 20))
        except (TypeError, ValueError):
            raise HttpError(400, "runs must be an integer") from None
        knobs = None
        if b.get("knobs"):
            knobs = [k.strip() for k in str(b["knobs"]).split(",") if k.strip()]

        gguf = self._calibrate_model_ref()
        if not gguf:
            raise HttpError(
                400, "no GGUF to calibrate — set gguf_path (Settings or "
                     "config) or download a model first", code="no_model")
        if self._bench_task.running():
            raise HttpError(409, "a bench run is in progress",
                            code="already_running")

        def _work(record: Dict[str, Any], log) -> None:
            profile = cal_mod.run_sweep(
                self.cfg, gguf_path=gguf, runs=runs, knobs=knobs,
                quick=quick, log=log)
            record["result"] = {
                "adopted_settings": profile.get("adopted_settings") or {},
                "threshold_pct": profile.get("threshold_pct"),
                "noise_floor_pct": (profile.get("noise_floor") or {}).get(
                    "spread_pct"),
                "knobs": profile.get("knobs"),
            }
            record["state"] = "done"

        self._calibrate_task.try_start(
            _work, conflict_msg="a calibration sweep is already running",
            initial={"quick": quick, "runs": runs, "model": gguf,
                     "result": None},
            done_extra=lambda r: {
                "adopted": (r.get("result") or {}).get("adopted_settings")})
        return {"ok": True, "state": "running", "model": gguf}

    def h_calibrate_run_status(self, q: Dict[str, str],
                               b: Dict[str, Any]) -> Dict[str, Any]:
        rec = self._calibrate_task.record
        if not rec:
            return {"state": "idle"}
        tail = int(q.get("tail", "40"))
        return {**rec, "log": rec["log"][-tail:]}

    def h_calibrate_clear(self, q: Dict[str, str],
                          b: Dict[str, Any]) -> Dict[str, Any]:
        """Remove stored calibration profiles (same as `calibrate --clear`)."""
        from ...serving import calibrate as cal_mod

        if self._calibrate_task.running():
            raise HttpError(409, "a calibration sweep is running",
                            code="already_running")
        n = cal_mod.clear_profiles()
        self.bus.publish("app.state", {"what": "calibrate.cleared",
                                       "removed": n})
        return {"ok": True, "removed": n}

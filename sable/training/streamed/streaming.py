"""(a) Meta-device shell + per-layer weight streaming from safetensors shards.

Inspiration (D-12: read as reference, re-implemented inline, never imported):
``airllm/airllm/air_llm/airllm/airllm_lora.py`` — ``_StreamedModule`` plus the
``_materialize`` / ``_evict`` / ``_prefetch_stream`` trio on AirLLMLoRA. WHY
each piece exists:

* A HF checkpoint that does not fit in VRAM is still *addressable* as a graph:
  build the module skeleton on the meta device (no storage allocated), then
  swap real tensors into each layer only for the microseconds it computes.
* Reading layer i+1 from disk while layer i computes hides most IO; a bounded
  read-ahead window keeps RAM at ~(1 + prefetch_depth) layers.
* Eviction must be explicit (drop refs + empty_cache): Python refcounts alone
  leave CUDA caching-allocator blocks reserved.

Design decisions for sable:
* ``LayerStreamer`` is PURE PYTHON bookkeeping with an injected loader, so the
  residency-window invariant is provable without torch or safetensors.
* The concrete ``SafetensorsLayerLoader`` lazy-imports both libs inside
  methods (training-extra seam; importing this module needs neither).
* Prefetch thread is OPT-IN (``prefetch_thread=True``); correctness never
  depends on it — with the thread off, ``prefetch()`` loads synchronously and
  the invariant is identical. Threaded mode uses ONE worker thread so shard
  reads are serialized: peak RAM stays deterministic at the window size.

Residency invariant (tracked by LayerStreamer, asserted by tests): at any
moment, materialized layers <= 1 (the layer in use) + ``prefetch_depth``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------- loader
class Loader:
    """Protocol for weight providers injected into :class:`LayerStreamer`.

    ``load(layer_idx)`` returns an opaque handle holding that layer's weights;
    ``unload(handle)`` must drop every reference (and empty caches if CUDA).
    Implementations must be callable from the prefetch worker thread.
    """

    def load(self, layer_idx: int) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def unload(self, handle: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError


# -------------------------------------------------------------------- streamer
@dataclass
class _Pending:
    """A prefetch in flight (threaded mode) or already loaded (sync mode)."""

    done: threading.Event = field(default_factory=threading.Event)
    future: Any = None      # concurrent.futures.Future | None
    handle: Any = None      # set once the load resolved (claimed or not)
    error: BaseException | None = None


class LayerStreamer:
    """Bounded-window layer streaming with optional read-ahead prefetch.

    Usage (per forward/backward pass over layers ``0..n-1``)::

        st.prefetch(0)                      # prime the window
        for i in range(n):
            w = st.acquire(i)               # waits for an in-flight prefetch
            y = compute(i, w)
            st.prefetch(i + 1)              # read-ahead (threaded or sync)
            st.release(i)                   # evict once truly done

    ``acquire`` may be called again for the same layer (backward recompute):
    if it was evicted meanwhile, it is simply loaded again — the streamer is a
    cache, not an owner of correctness. Thread-safe: all state transitions
    happen under one RLock; the prefetch worker never holds it across IO.
    """

    def __init__(self, loader: Loader, num_layers: int, *,
                 prefetch_depth: int = 1, prefetch_thread: bool = False):
        if prefetch_depth < 0:
            raise ValueError("prefetch_depth must be >= 0")
        self.loader = loader
        self.num_layers = int(num_layers)
        self.prefetch_depth = int(prefetch_depth)
        #: hard ceiling on simultaneously materialized layers.
        self.max_resident = 1 + self.prefetch_depth

        self._lock = threading.RLock()
        self._resident: Dict[int, Any] = {}      # idx -> live handle
        self._pending: Dict[int, _Pending] = {}  # idx -> prefetch record
        self._executor = None
        if prefetch_thread:
            from concurrent.futures import ThreadPoolExecutor

            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="sable-prefetch")

        # Bookkeeping for tests / telemetry.
        self.load_count = 0
        self.evict_count = 0
        self.max_resident_seen = 0
        self.closed = False

    # -- internals (lock discipline: never call loader under self._lock) ----
    def _note_state(self) -> None:
        # Called with the lock held. Residency == materialized handles.
        cur = len(self._resident)
        if cur > self.max_resident_seen:
            self.max_resident_seen = cur

    def _load_sync(self, layer_idx: int) -> Any:
        handle = self.loader.load(layer_idx)
        with self._lock:
            self.load_count += 1
            # Re-check: another thread may have resolved a prefetch meanwhile.
            if layer_idx in self._resident:
                self.loader.unload(handle)
                return self._resident[layer_idx]
            self._resident[layer_idx] = handle
            self._note_state()
        return handle

    def _worker_load(self, layer_idx: int, pend: _Pending) -> None:
        try:
            handle = self.loader.load(layer_idx)  # IO OUTSIDE the lock
        except BaseException as exc:              # surface it in acquire()
            pend.error = exc
            with self._lock:
                self._pending.pop(layer_idx, None)
            pend.done.set()
            return
        discard = False
        with self._lock:
            self.load_count += 1
            if (self._pending.get(layer_idx) is not pend
                    or layer_idx in self._resident):
                # Released (or raced a sync acquire) while loading: drop it.
                discard = True
            else:
                self._resident[layer_idx] = handle
            self._note_state()
        if discard:
            self.loader.unload(handle)
        else:
            pend.handle = handle
        pend.done.set()

    def _discard_when_done(self, pend: _Pending) -> None:
        """Let an unwanted in-flight load finish (interrupting disk IO is
        worse than a transient extra layer), then free it exactly once."""

        def _run() -> None:
            pend.done.wait()
            handle, pend.handle = pend.handle, None
            if handle is not None:
                self.loader.unload(handle)

        threading.Thread(target=_run, daemon=True,
                         name="sable-discard").start()

    # -- public API -----------------------------------------------------------
    def prefetch(self, layer_idx: int) -> None:
        """Start loading ``layer_idx`` if the read-ahead window has room.

        No-op past the last layer, for already-resident layers, or when the
        window (1 + prefetch_depth distinct layers) is already committed.
        """
        if self.closed or layer_idx < 0 or layer_idx >= self.num_layers:
            return
        with self._lock:
            if layer_idx in self._resident or layer_idx in self._pending:
                return
            committed = len(set(self._resident) | set(self._pending))
            if committed >= self.max_resident:
                return
            pend = _Pending()
            self._pending[layer_idx] = pend
        if self._executor is not None:
            pend.future = self._executor.submit(self._worker_load, layer_idx, pend)
        else:
            # Synchronous prefetch: load now, claim on acquire. Same window.
            # Load errors are DEFERRED to acquire() (same semantics as the
            # threaded path) so a failed read-ahead never explodes in the
            # middle of the compute loop.
            try:
                handle = self.loader.load(layer_idx)  # IO outside the lock
            except BaseException as exc:
                pend.error = exc
                pend.done.set()
                return
            with self._lock:
                self.load_count += 1
                self._resident[layer_idx] = handle
                self._note_state()
            pend.handle = handle
            pend.done.set()

    def acquire(self, layer_idx: int) -> Any:
        """Return the materialized weights for ``layer_idx``, loading on
        demand (waiting on an in-flight prefetch when one exists)."""
        if self.closed:
            raise RuntimeError("streamer is closed")
        with self._lock:
            if layer_idx in self._resident:
                return self._resident[layer_idx]
            pend = self._pending.get(layer_idx)
        if pend is not None:
            pend.done.wait()
            if pend.error is not None:
                # Clear the failed record so a later acquire retries cleanly.
                with self._lock:
                    if self._pending.get(layer_idx) is pend:
                        del self._pending[layer_idx]
                raise pend.error
            with self._lock:
                if layer_idx in self._resident:
                    return self._resident[layer_idx]
            # Worker discarded its result (shouldn't happen while pending was
            # claimed here) — fall through to a fresh synchronous load.
            return self._load_sync(layer_idx)
        return self._load_sync(layer_idx)

    def release(self, layer_idx: int) -> None:
        """Evict layer ``layer_idx`` (the loader frees its memory)."""
        with self._lock:
            handle = self._resident.pop(layer_idx, None)
            pend = self._pending.pop(layer_idx, None)
        if pend is not None and not pend.done.is_set():
            self._discard_when_done(pend)
        if handle is not None:
            self.loader.unload(handle)
            with self._lock:
                self.evict_count += 1

    def resident_layers(self) -> List[int]:
        with self._lock:
            return sorted(self._resident)

    def peak_residency(self) -> int:
        with self._lock:
            return self.max_resident_seen

    def close(self) -> None:
        """Evict everything and shut the prefetch thread down."""
        self.closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        with self._lock:
            handles = list(self._resident.values())
            self._resident.clear()
            self._pending.clear()
        for h in handles:
            self.loader.unload(h)
            self.evict_count += 1

    # Context-manager sugar ---------------------------------------------------
    def __enter__(self) -> "LayerStreamer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ------------------------------------------------- safetensors loader (torch)
class SafetensorsLayerLoader(Loader):
    """Concrete loader: per-layer tensors from one or more ``*.safetensors``
    shards via ``safe_open`` + ``get_tensor`` (both LAZILY imported — training
    extras; importing this class needs no torch).

    ``layer_keys[i]`` lists ``(shard_path, tensor_name)`` pairs belonging to
    decoder layer ``i``. Shard handles open on first use and are reused;
    tensors move to ``device`` after load. ``unload`` drops the refs and, when
    CUDA is active, calls ``torch.cuda.empty_cache()`` — mirrors airllm's
    evict path (inspiration ``airllm_lora.py::_evict``).
    """

    def __init__(self, layer_keys: Sequence[Sequence[Tuple[str, str]]], *,
                 device: str = "cpu", dtype: Optional[str] = None):
        self.layer_keys = [list(part) for part in layer_keys]
        self.device_name = device
        self.dtype_name = dtype
        self._handles: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def _handle(self, shard_path: str):
        with self._lock:
            h = self._handles.get(shard_path)
            if h is None:
                from safetensors import safe_open  # lazy: training extra

                h = safe_open(shard_path, framework="pt")
                self._handles[shard_path] = h
            return h

    def load(self, layer_idx: int) -> Dict[str, Any]:
        import torch  # lazy: training extra

        out: Dict[str, Any] = {}
        for shard_path, tensor_name in self.layer_keys[layer_idx]:
            f = self._handle(shard_path)
            t = f.get_tensor(tensor_name)
            if self.dtype_name:
                t = t.to(getattr(torch, self.dtype_name))
            out[tensor_name] = t.to(self.device_name)
        return out

    def unload(self, handle: Dict[str, Any]) -> None:
        handle.clear()  # drop this handle's tensor references
        import torch  # lazy

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        with self._lock:
            self._handles.clear()  # safetensors file handles close on GC


# --------------------------------------------------------- meta-device shell
def meta_shell(layer_specs: Sequence[Tuple[int, int]], *,
               device: str = "meta") -> Any:
    """Build a ``torch.nn.ModuleList`` skeleton of Linears with NO storage.

    With ``device="meta"`` (default) parameters carry shape/dtype but zero
    bytes — the shell costs nothing until layers are streamed in (the airllm
    trick that lets a 120B graph exist on a 24GB card). LAZY torch import:
    raises ImportError with a clear message when torch is absent; importing
    this module never needs torch.

    ``layer_specs``: sequence of ``(in_features, out_features)`` per layer.
    """
    import torch  # lazy: training extra
    from torch import nn

    shells = []
    with torch.device(device):
        for in_f, out_f in layer_specs:
            shells.append(nn.Linear(in_f, out_f, bias=False))
    return nn.ModuleList(shells)


def load_layer_into(shell_linear: Any, weights: Dict[str, Any]) -> None:
    """Swap streamed tensors into a meta-shell ``nn.Linear`` by replacing
    ``weight.data`` / ``bias.data`` IN PLACE — Parameter object identity is
    preserved so ``state_dict()`` keys keep matching the checkpoint names
    (required for the later GGUF-LoRA mapping; same reason airllm keeps the
    original Parameter objects)."""

    def _strip(name: str) -> str:
        return name.rsplit(".", 1)[-1] if "." in name else name

    import torch  # lazy: training extra

    def _assign(param: Any, tensor: Any) -> None:
        if param.device.type == "meta":
            # A meta Parameter cannot take .data from a real tensor (no
            # storage to overwrite); swap_tensors exchanges the two objects'
            # contents IN PLACE so `shell_linear.weight` KEEPS its identity.
            with torch.no_grad():
                new = torch.nn.Parameter(tensor, requires_grad=False)
            torch.utils.swap_tensors(param, new)
        else:
            with torch.no_grad():
                param.data = tensor

    for name, tensor in weights.items():
        leaf = _strip(name)
        if leaf == "weight":
            _assign(shell_linear.weight, tensor)
        elif leaf == "bias":
            if getattr(shell_linear, "bias", None) is not None:
                _assign(shell_linear.bias, tensor)

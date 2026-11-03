"""(c) CPU-held autograd graph with per-layer backward recompute.

Inspiration (D-12: read as reference, re-implemented inline, never imported):
``airllm/airllm/air_llm/airllm/airllm_lora.py`` — ``_StreamedModule``: a
torch.autograd.Function whose forward runs the module under ``no_grad``
saving ONLY its input tensor, and whose backward re-runs that one module WITH
grad (recompute) to rebuild just enough graph to backprop through it. WHY:
a normal autograd graph pins every layer's activations AND weights alive
until the whole backward completes — exactly what a streamed trainer cannot
afford. Saving inputs only keeps peak memory at ~one layer of graph, and the
optimizer state / grads live on CPU between steps.

LOUD WARNING — dropout MUST be zero in streamed mode:
backward RECOMPUTES the forward; any stochastic op (dropout with p > 0,
stochastic depth, ...) would replay a DIFFERENT mask than the original
forward and silently corrupt gradients. This module ENFORCES it at
construction (:func:`assert_no_stochastic_dropout`); do not work around it.

torch is imported LAZILY inside methods (training extra). The orchestration
class itself is plain Python, so importing this module never needs torch.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def assert_no_stochastic_dropout(module: Any) -> None:
    """Raise if ``module`` contains dropout with p > 0 (torch-free check).

    Duck-typed on purpose (class name + ``p`` attribute) so the guard runs
    without importing torch at module level. Streamed recompute requires a
    deterministic forward — see module docstring WARNING.
    """
    for name, child in module.named_modules():
        cls = type(child).__name__
        p = getattr(child, "p", None)
        # Duck-typed (torch-free): Dropout / Dropout1d / Dropout2d / ...
        if p and "dropout" in cls.lower():
            raise ValueError(
                f"streamed recompute requires dropout p=0, but "
                f"{name or '<module>'} is Dropout(p={p}); disable it — "
                f"recomputed forward would replay a different mask")


class StreamedRecomputeTrainer:
    """no_grad forward (save inputs) + reverse recompute-and-backprop pass.

    ``layers`` is an ordered list of modules ``l_0 .. l_{n-1}``. Each step:

      forward   h0 = x; for i: with no_grad: h_{i+1} = l_i(h_i)  — only the
                (leaf) inputs h_i are kept; activations carry NO graph.
      loss      ``loss_fn(h_n) -> (loss_scalar, dL/dh_n)`` (caller owns the
                head; chunked_ce provides the LM-head version).
      backward  for i = n-1 .. 0: re-run ``l_i`` on the saved input WITH
                grad, then ``autograd.backward(recomputed, grad_outputs=g)``;
                the input's ``.grad`` becomes the next g. Only layer i's
                private graph exists at any moment.

    Optional ``streamer`` (anything with ``acquire(i)`` / ``release(i)``, e.g.
    :class:`toaster_loop.training.streamed.streaming.LayerStreamer`) is invoked
    around each compute: weights for layer i are materialized before its
    forward/recompute and evicted right after — the same window discipline as
    inference streaming, extended over both passes.

    Autograd state stays on CPU between steps: grads accumulate into the
    (CPU) parameters' ``.grad``; nothing device-side survives a step once the
    layers are released. On CUDA this is exactly airllm's contract: the only
    per-layer graph lives for microseconds inside backward.
    """

    def __init__(self, layers: List[Any], params: List[Any], *,
                 streamer: Optional[Any] = None) -> None:
        for layer in layers:
            assert_no_stochastic_dropout(layer)
        self.layers = list(layers)
        self.params = list(params)
        self.streamer = streamer
        #: per-step counters (tests assert the streaming discipline).
        self.forward_layer_calls = 0
        self.backward_layer_calls = 0

    # -- streamer helpers -----------------------------------------------------
    def _acquire(self, i: int) -> None:
        if self.streamer is not None:
            self.streamer.acquire(i)

    def _release(self, i: int) -> None:
        if self.streamer is not None:
            self.streamer.release(i)

    # -- one full streamed step -------------------------------------------------
    def forward_inputs(self, x: Any) -> tuple:
        """no_grad forward over the stack; returns (final_hidden, inputs).

        ``inputs[i]`` is the leaf input of layer i (requires_grad so the
        recompute-backward can hand us dL/dh_i)."""
        import torch  # lazy: training extra

        inputs: List[Any] = []
        h = x
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                self._acquire(i)
                h_in = h.detach().requires_grad_(True)
                inputs.append(h_in)
                h = layer(h_in)
                self.forward_layer_calls += 1
                # Evict immediately: backward RE-ACQUIRES (reloads) layer i
                # when its recompute turn comes. This keeps the residency
                # window bounded across BOTH passes — never the whole stack.
                self._release(i)
        return h, inputs

    def backward_recompute(self, inputs: List[Any], grad_out: Any) -> None:
        """Reverse pass: per-layer recompute + backprop through one layer."""
        import torch  # lazy: training extra

        g = grad_out
        for i in reversed(range(len(self.layers))):
            h_in = inputs[i]
            self._acquire(i)          # may reload weights evicted after fwd
            with torch.enable_grad():
                recomputed = self.layers[i](h_in)
            torch.autograd.backward(recomputed, grad_tensors=g)
            g = h_in.grad
            h_in.grad = None          # free the input grad once forwarded
            self._release(i)          # layer i's graph is dead: evict weights
            self.backward_layer_calls += 1

    def step(self, x: Any, loss_fn: Callable[[Any], tuple]) -> float:
        """One streamed training step; returns the loss value (float).

        ``loss_fn(final_hidden)`` must return ``(loss, dL/d_final_hidden)``.
        Caller owns the optimizer (``optimizer.step()/zero_grad()`` around
        this) — keeps the seam torch-agnostic for tests and Step 2 wiring.
        """
        final_h, inputs = self.forward_inputs(x)
        loss, grad_out = loss_fn(final_h.detach())
        self.backward_recompute(inputs, grad_out)
        return float(loss)

    # -- CPU autograd-state helpers ----------------------------------------------
    def collect_grads_cpu(self) -> Dict[int, Any]:
        """Snapshot grads to plain CPU tensors keyed by ``id(param)``.

        Between steps the optimizer state and grads live on CPU; this is the
        explicit hand-off point (on CUDA you would call this right after
        backward and before evicting device copies)."""
        out: Dict[int, Any] = {}
        for p in self.params:
            if p.grad is not None:
                out[id(p)] = p.grad.detach().cpu().clone()
        return out

    def zero_grads(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad = None

"""WP10 step-1 spike: streamed-LoRA mechanisms (sable/training/streamed/).

Two tiers, matching the machine reality:

  TIER 1 (always runs, no torch): streaming bookkeeping through injected
  fake weight providers — residency-window invariant, prefetch hit rates,
  eviction accounting, threaded mode; plus the PURE-Python chunked-CE
  numerics against a list-based reference. The spike package must import
  cleanly without torch installed (asserted here too).

  TIER 2 (pytest.importorskip("torch")): real numeric proofs — LoRA output
  identity + frozen-base optimizer behavior, streamed recompute vs a dense
  autograd reference (gradients + loss decrease), chunked CE equality vs
  F.cross_entropy, and end-to-end layer streaming from REAL safetensors
  shards written in-test (tiny random weights; no network, no HF download).
"""

from __future__ import annotations

import math
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sable.training.streamed import chunked_ce  # noqa: E402
from sable.training.streamed.streaming import LayerStreamer, Loader  # noqa: E402


# ============================================================== fake loaders
class CountingLoader(Loader):
    """Tracks simultaneously-live handles: the residency oracle for tier 1."""

    def __init__(self, delay_s: float = 0.0, fail_on: int | None = None):
        self.delay_s = delay_s
        self.fail_on = fail_on
        self.live = 0
        self.max_live = 0
        self.loads: List[int] = []
        self.unloads: List[int] = []
        self._lock = threading.Lock()

    def load(self, layer_idx: int) -> Dict[str, Any]:
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail_on is not None and layer_idx == self.fail_on:
            raise RuntimeError(f"shard read failed for layer {layer_idx}")
        with self._lock:
            self.loads.append(layer_idx)
            self.live += 1
            self.max_live = max(self.max_live, self.live)
        return {"layer": layer_idx, "w": [float(layer_idx)]}

    def unload(self, handle: Dict[str, Any]) -> None:
        with self._lock:
            self.unloads.append(handle["layer"])
            self.live -= 1


# ======================================================= TIER 1: pure python
def test_package_imports_without_torch():
    """The spike module must be import-safe without torch (WP10 constraint).

    We can't uninstall torch here, so assert the *mechanism*: no heavy import
    appears at module level in any streamed module (AST-level guarantee that
    doubles as the license-hygiene contract), and a fresh subprocess import
    with torch blocked still succeeds.
    """
    import ast

    pkg = Path(__file__).resolve().parents[1] / "sable" / "training" / "streamed"
    for py in sorted(pkg.glob("*.py")):
        tree = ast.parse(py.read_text())
        for node in tree.body:  # module level only
            names: List[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert n.split(".")[0] not in {"torch", "transformers", "peft",
                                               "trl", "safetensors"}, \
                    f"{py.name}:{getattr(node, 'lineno', 0)}: module-level {n}"


def _run_pass(st: LayerStreamer, n_layers: int) -> None:
    st.prefetch(0)
    for i in range(n_layers):
        w = st.acquire(i)
        assert w["layer"] == i
        st.prefetch(i + 1)   # read-ahead while "computing"
        st.release(i)


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_streamer_residency_window_sync(depth):
    n = 7
    loader = CountingLoader()
    with LayerStreamer(loader, n, prefetch_depth=depth) as st:
        _run_pass(st, n)
        assert st.peak_residency() <= 1 + depth
    assert loader.max_live <= 1 + depth
    # Every layer loaded exactly once (prefetch hits), evicted exactly once.
    assert sorted(loader.loads) == list(range(n))
    assert sorted(loader.unloads) == list(range(n))


def test_streamer_threaded_prefetch_respects_window():
    n = 6
    loader = CountingLoader(delay_s=0.005)
    with LayerStreamer(loader, n, prefetch_depth=1, prefetch_thread=True) as st:
        _run_pass(st, n)
        assert st.peak_residency() <= 2
    # With read-ahead on, each layer is still loaded exactly once.
    assert sorted(loader.loads) == list(range(n))
    assert loader.max_live <= 2


def test_streamer_threaded_acquire_waits_for_prefetch():
    """acquire() on an in-flight prefetch blocks until the worker lands it."""
    loader = CountingLoader(delay_s=0.05)
    with LayerStreamer(loader, 3, prefetch_depth=1, prefetch_thread=True) as st:
        st.prefetch(0)
        t0 = time.monotonic()
        w = st.acquire(0)
        assert time.monotonic() - t0 >= 0.04
        assert w["layer"] == 0


def test_streamer_window_blocks_overfetch():
    """prefetch() beyond the window is a no-op: never more than 1+depth."""
    loader = CountingLoader()
    with LayerStreamer(loader, 10, prefetch_depth=1) as st:
        st.prefetch(0)
        st.prefetch(1)
        st.prefetch(2)          # window full -> dropped
        st.prefetch(3)          # still dropped
        assert sorted(st.resident_layers()) == [0, 1]
        assert st.peak_residency() == 2


def test_streamer_release_during_inflight_prefetch():
    """Releasing a layer whose threaded load is in flight must not leak."""
    loader = CountingLoader(delay_s=0.03)
    with LayerStreamer(loader, 4, prefetch_depth=1, prefetch_thread=True) as st:
        st.prefetch(1)               # in flight
        st.release(1)                # abandoned before landing
        time.sleep(0.08)             # worker lands + discards
        assert 1 not in st.resident_layers()
        assert loader.live == 0


def test_streamer_prefetch_error_surfaces_on_acquire():
    loader = CountingLoader(fail_on=2)
    with LayerStreamer(loader, 4, prefetch_depth=1) as st:
        st.prefetch(2)
        with pytest.raises(RuntimeError, match="layer 2"):
            st.acquire(2)


def test_streamer_reload_after_release():
    """The streamer is a cache: acquire after release re-loads transparently."""
    loader = CountingLoader()
    with LayerStreamer(loader, 2, prefetch_depth=0) as st:
        w1 = st.acquire(0)
        st.release(0)
        w2 = st.acquire(0)
        assert w1["layer"] == w2["layer"] == 0
        assert loader.loads.count(0) == 2


def test_streamer_close_evicts_all():
    loader = CountingLoader()
    st = LayerStreamer(loader, 3, prefetch_depth=1)
    st.prefetch(0)
    st.acquire(1)
    st.close()
    assert loader.live == 0
    with pytest.raises(RuntimeError):
        st.acquire(0)


# ------------------------------------------------- pure chunked-CE numerics
def _softmax_ce_ref(logits_rows, labels):
    """Independent stdlib reference: plain two-pass softmax CE per row."""
    total, count = 0.0, 0
    for row, lab in zip(logits_rows, labels):
        if lab == -100:
            continue
        m = max(row)
        z = sum(math.exp(v - m) for v in row)
        lse = m + math.log(z)
        total += lse - row[lab]
        count += 1
    return total / max(count, 1)


def test_chunked_ce_reference_matches_plain_softmax():
    rng = random.Random(7)
    n, v = 5, 64
    rows = [[rng.uniform(-4, 4) for _ in range(v)] for _ in range(n)]
    labels = [rng.randrange(v) for _ in range(n)]
    labels[3] = -100  # ignored row exercises the mask path
    loss, per_token = chunked_ce.chunked_ce_reference(
        rows, labels, vocab_chunk=7)
    assert abs(per_token[3]) < 1e-12
    assert math.isclose(loss, _softmax_ce_ref(rows, labels), rel_tol=1e-9)


def test_online_lse_merge_properties():
    assert chunked_ce.online_lse_merge(float("-inf"), 2.0) == 2.0
    assert chunked_ce.online_lse_merge(2.0, float("-inf")) == 2.0
    merged = chunked_ce.online_lse_merge(0.0, 0.0)
    assert math.isclose(merged, math.log(2.0), rel_tol=1e-12)
    # order independence within float noise
    a = chunked_ce.online_lse_merge(3.0, -1.0)
    b = chunked_ce.online_lse_merge(-1.0, 3.0)
    assert math.isclose(a, b, rel_tol=1e-12)
    # merging many singletons == full logsumexp
    vals = [1.0, 4.0, -2.0, 0.5]
    run = float("-inf")
    for x in vals:
        run = chunked_ce.online_lse_merge(run, x)
    m = max(vals)
    assert math.isclose(run, m + math.log(sum(math.exp(x - m) for x in vals)),
                        rel_tol=1e-12)


def test_chunked_ce_reference_chunk_invariance():
    """Chunk width must not change the answer (online merge is associative)."""
    rng = random.Random(11)
    rows = [[rng.uniform(-6, 6) for _ in range(48)] for _ in range(3)]
    labels = [5, 47, 0]
    l1, _ = chunked_ce.chunked_ce_reference(rows, labels, vocab_chunk=1)
    l7, _ = chunked_ce.chunked_ce_reference(rows, labels, vocab_chunk=7)
    lfull, _ = chunked_ce.chunked_ce_reference(rows, labels, vocab_chunk=48)
    assert math.isclose(l1, l7, rel_tol=1e-9)
    assert math.isclose(l7, lfull, rel_tol=1e-12)


def test_dropout_guard_rejects_p_gt_zero():
    """Recompute needs deterministic forwards — enforced without torch."""
    from sable.training.streamed.recompute import assert_no_stochastic_dropout

    class DropoutFake:  # duck-typed on purpose: name contains "dropout"
        p = 0.1

    class FakeInner:
        def __init__(self):
            self.drop = DropoutFake()

    class FakeModel:
        def __init__(self):
            self.block = FakeInner()

        def named_modules(self):
            yield "", self
            yield "block", self.block
            yield "block.drop", self.block.drop

    with pytest.raises(ValueError, match="dropout p=0"):
        assert_no_stochastic_dropout(FakeModel())


# ====================================================== TIER 2: real torch
def _torch():
    return pytest.importorskip("torch")


def test_lora_linear_numerics_and_frozen_base():
    torch = _torch()
    import torch.nn as nn

    from sable.training.streamed.lora_linear import LoRALinear

    torch.manual_seed(0)
    base = nn.Linear(12, 8, bias=True).double()
    lin = LoRALinear(base, r=4, alpha=8.0)

    # (1) adapter starts as an exact no-op (B = 0).
    x = torch.randn(3, 12, dtype=torch.float64)
    base_out = torch.nn.functional.linear(x, base.weight, base.bias)
    out = lin(x)
    assert torch.allclose(out, base_out, atol=1e-10)

    # (2) output identity: y = base(x) + scaling * B(A(x)) numerically.
    with torch.no_grad():
        lin.lora_A.normal_()
    expected = base_out + ((x @ lin.lora_A.detach().t())
                           @ lin.lora_B.detach().t()) * (8.0 / 4)
    assert torch.allclose(lin(x), expected, atol=1e-10)

    # (3) frozen base: requires_grad False and bit-identical after a step.
    assert not lin.weight.requires_grad
    assert not lin.bias.requires_grad
    w_before = lin.weight.detach().clone()
    b_before_bias = lin.bias.detach().clone()
    loss = lin(x).pow(2).mean()
    opt = torch.optim.SGD([lin.lora_A, lin.lora_B], lr=0.5)
    loss.backward()
    a_before = lin.lora_A.detach().clone()
    bmat_before = lin.lora_B.detach().clone()
    # Standard LoRA init consequence: B = 0 => dL/dA == 0 exactly on step 1;
    # only B receives gradient. Both are trainable; A moves from step 2 on.
    assert torch.allclose(lin.lora_A.grad, torch.zeros_like(lin.lora_A.grad))
    assert lin.lora_B.grad is not None and lin.lora_B.grad.abs().sum() > 0
    opt.step()
    assert torch.equal(lin.weight, w_before)       # base untouched
    assert torch.equal(lin.bias, b_before_bias)    # base bias untouched
    assert not torch.allclose(lin.lora_B, bmat_before)  # adapter moved
    # second step moves A as well (B is non-zero now)
    opt.zero_grad(set_to_none=True)
    lin(x).pow(2).mean().backward()
    opt.step()
    assert not torch.allclose(lin.lora_A, a_before)

    # (4) dropout > 0 is rejected loudly (recompute contract).
    with pytest.raises(ValueError, match="dropout must be 0"):
        LoRALinear(nn.Linear(4, 4), r=2, alpha=2, dropout=0.1)


def test_lora_state_dict_layout():
    _torch()
    import torch.nn as nn

    from sable.training.streamed.lora_linear import (inject_lora,
                                                       lora_state_dict)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(6, 6)
            self.unused = nn.Linear(6, 6)

    m = Block()
    n = inject_lora(m, ["q_proj"], r=2, alpha=4)
    assert n == 1
    keys = set(lora_state_dict(m))
    assert keys == {"q_proj.lora_A", "q_proj.lora_B"}
    sd_keys = set(m.state_dict())
    # Base checkpoint key survives wrapping (GGUF mapping requirement).
    assert "q_proj.weight" in sd_keys and "unused.weight" in sd_keys


def test_recompute_matches_dense_reference():
    """Streamed no_grad-forward/recompute-backward grads == dense autograd."""
    torch = _torch()
    import torch.nn as nn

    from sable.training.streamed.recompute import StreamedRecomputeTrainer

    torch.manual_seed(3)
    dims = [(8, 6), (6, 6), (6, 4)]

    def stack():
        mods = nn.ModuleList()
        for i_f, o_f in dims:
            lin = nn.Linear(i_f, o_f, bias=False).double()
            mods.append(nn.Sequential(lin))
        return mods

    dense = stack()
    streamed = [nn.Sequential(nn.Linear(i_f, o_f, bias=False).double())
                for i_f, o_f in dims]
    # Identical weights (seeded copy).
    torch.manual_seed(4)
    for a, b in zip(dense, streamed):
        b[0].weight.data.copy_(a[0].weight.data)

    x = torch.randn(5, 8, dtype=torch.float64)
    target = torch.randn(5, 4, dtype=torch.float64)

    def loss_fn(h):
        l = (h - target).pow(2).mean()
        return l, torch.autograd.grad(l, h, retain_graph=False)[0] \
            if h.requires_grad else 2.0 * (h - target) / (5 * 4)

    # Dense reference: ordinary autograd through the whole stack.
    for p in dense.parameters():
        p.grad = None
    hd = x
    for m in dense:
        hd = m(hd)
    ld = (hd - target).pow(2).mean()
    ld.backward()

    # Streamed: no_grad forward, per-layer recompute backward.
    trainer = StreamedRecomputeTrainer(streamed,
                                       [m[0].weight for m in streamed])
    final_h, inputs = trainer.forward_inputs(x)
    loss_val, grad_out = loss_fn(final_h)
    trainer.backward_recompute(inputs, grad_out)

    assert math.isclose(float(loss_val), float(ld.detach()), rel_tol=1e-9)
    for mdense, mstream in zip(dense, streamed):
        pd = mdense[0].weight.grad
        ps = mstream[0].weight.grad
        assert pd is not None and ps is not None
        assert torch.allclose(pd, ps, atol=1e-8), \
            f"grad mismatch: max diff {(pd - ps).abs().max()}"


def test_streamed_training_loss_decreases():
    """Several full streamed steps (SGD on CPU-held grads) reduce the loss."""
    torch = _torch()
    import torch.nn as nn

    from sable.training.streamed.recompute import StreamedRecomputeTrainer

    torch.manual_seed(5)
    dims = [(6, 6), (6, 6), (6, 3)]
    layers = [nn.Sequential(nn.Linear(i_f, o_f, bias=False).double())
              for i_f, o_f in dims]
    params = [m[0].weight for m in layers]
    trainer = StreamedRecomputeTrainer(layers, params)
    opt = torch.optim.SGD(params, lr=0.3)

    x = torch.randn(4, 6, dtype=torch.float64)
    target = torch.randn(4, 3, dtype=torch.float64)

    def loss_fn(h):
        h = h.detach().requires_grad_(False)
        l = (h - target).pow(2).mean()
        return float(l), 2.0 * (h - target) / (4 * 3)

    losses = []
    for _ in range(15):
        opt.zero_grad(set_to_none=True)
        loss = trainer.step(x, loss_fn)
        losses.append(loss)
        opt.step()
    assert losses[-1] < losses[0] * 0.5, f"loss did not fall: {losses}"
    # Streaming discipline: every layer ran once per pass, fwd and bwd.
    assert trainer.forward_layer_calls == 15 * 3
    assert trainer.backward_layer_calls == 15 * 3


def test_chunked_ce_matches_cross_entropy():
    torch = _torch()
    import torch.nn.functional as F

    from sable.training.streamed.chunked_ce import \
        chunked_linear_cross_entropy

    torch.manual_seed(9)
    hidden = torch.randn(2, 8, 1024, dtype=torch.float64)
    weight = torch.randn(64, 1024, dtype=torch.float64)
    labels = torch.randint(0, 64, (2, 8))
    labels[0, 3] = -100  # ignore path

    dense_logits = hidden.reshape(-1, 1024) @ weight.t()
    ref = F.cross_entropy(dense_logits, labels.reshape(-1),
                          ignore_index=-100, reduction="mean")

    hid = hidden.reshape(-1, 1024).clone().requires_grad_(True)
    w = weight.clone().requires_grad_(True)
    loss = chunked_linear_cross_entropy(hid, w, labels, chunk_size=5)
    assert torch.isclose(loss.double(), ref, rtol=1e-9), (loss, ref)

    # Gradients flow: d_hidden matches the dense autograd gradient. NOTE:
    # `weight` is stashed (not an apply() input), so it stays FROZEN by
    # design — exactly the streamed-LoRA contract where lm_head is never
    # trained; d_weight is intentionally not produced.
    loss.backward()
    hid2 = hidden.reshape(-1, 1024).clone().requires_grad_(True)
    w2 = weight.clone().requires_grad_(True)
    F.cross_entropy(hid2 @ w2.t(), labels.reshape(-1),
                    ignore_index=-100).backward()
    assert torch.allclose(hid.grad.double(), hid2.grad, atol=1e-7)
    assert w.grad is None  # frozen head contract


def test_layer_streaming_from_real_safetensors(tmp_path):
    """End-to-end (a): real safetensors shards -> LayerStreamer -> meta shell.

    Tiny random weights written in-test with the safetensors save API — no
    network, no HF download (WP10 constraint). Proves the loader/shell seam:
    keys stay ``model.layers.i.*``, tensors land on demand, shells fill via
    load_layer_into with Parameter identity preserved.
    """
    torch = _torch()
    from safetensors.torch import save_file

    from sable.training.streamed.streaming import (SafetensorsLayerLoader,
                                                     load_layer_into,
                                                     meta_shell)

    n_layers, dim = 3, 5
    shard_paths = []
    layer_keys = []
    for i in range(n_layers):
        p = tmp_path / f"model-{i:02d}.safetensors"
        key = f"model.layers.{i}.mlp.weight"
        save_file({key: torch.randn(dim, dim, dtype=torch.float32)}, str(p))
        shard_paths.append(p)
        layer_keys.append([(str(p), key)])

    loader = SafetensorsLayerLoader(layer_keys, device="cpu")
    shell = meta_shell([(dim, dim)] * n_layers)  # zero-storage skeleton
    for lin in shell:
        assert lin.weight.device.type == "meta"

    loaded_order = []

    class TracingLoader(SafetensorsLayerLoader):
        def load(self, layer_idx):
            loaded_order.append(layer_idx)
            return super().load(layer_idx)

    tl = TracingLoader(layer_keys, device="cpu")
    with LayerStreamer(tl, n_layers, prefetch_depth=1) as st:
        st.prefetch(0)
        for i in range(n_layers):
            weights = st.acquire(i)
            load_layer_into(shell[i], weights)
            assert shell[i].weight.device.type == "cpu"
            y = shell[i](torch.ones(dim))
            assert torch.isfinite(y).all()
            st.prefetch(i + 1)
            st.release(i)
        assert st.peak_residency() <= 2
    assert loaded_order == [0, 1, 2]
    tl.close()

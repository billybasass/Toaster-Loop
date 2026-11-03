"""(d) Chunked linear cross-entropy — never materialises ``[N, V]`` logits.

Inspiration (D-12: read as reference, re-implemented inline, never imported):
``airllm/airllm/air_llm/airllm/chunked_ce.py``. WHY it exists: a 70B-class
lm_head is ~[100k+, hidden] — the full bf16 logits for one 2048-token batch
can exceed the VRAM the layers were supposed to have. Walking the vocabulary
in chunks with an ONLINE log-sum-exp accumulation (merge rule
``LSE(a, b) = max + log(exp(a-max) + exp(b-max))``) keeps peak memory at
``[N, chunk]``; backward recomputes the same chunks to form d_hidden, so the
forward never has to keep logits around at all.

Split for testability:
* ``online_lse_merge`` / ``chunked_ce_reference`` — PURE PYTHON (math only),
  the numerics spec of the chunking math; unit-testable with no torch and
  no numpy on tiny matrices.
* ``chunked_linear_cross_entropy`` — the torch autograd.Function doing the
  real work on tensors (LAZY torch import; class built on first call via
  factory so importing this module never needs torch).

Semantics match ``F.cross_entropy(logits, labels, ignore_index=-100,
reduction=...)`` where ``logits = hidden @ weight.T``.
"""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple

IGNORE_INDEX_DEFAULT = -100


# ----------------------------------------------------------- pure-Python core
def online_lse_merge(running: float, chunk_lse: float) -> float:
    """Merge a running log-sum-exp with one chunk's log-sum-exp.

    Stable two-term LSE: ``log(exp(a) + exp(b))`` computed via the max trick.
    ``-inf`` is the identity (empty chunks / no rows seen yet).
    """
    if running == float("-inf"):
        return chunk_lse
    if chunk_lse == float("-inf"):
        return running
    m = max(running, chunk_lse)
    return m + math.log(math.exp(running - m) + math.exp(chunk_lse - m))


def _logsumexp_row(row: Sequence[float]) -> float:
    m = max(row)
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(sum(math.exp(v - m) for v in row))


def chunked_ce_reference(logits_rows: Sequence[Sequence[float]],
                         labels: Sequence[int], *, vocab_chunk: int,
                         ignore_index: int = IGNORE_INDEX_DEFAULT,
                         reduction: str = "mean") -> Tuple[float, List[float]]:
    """Reference implementation of the chunking math over explicit logits.

    Splits each row's vocabulary into ``vocab_chunk``-wide slices, computes a
    per-chunk logsumexp, merges them ONLINE (the exact accumulation order the
    torch kernel uses), and picks the target logit from whichever chunk owns
    the label. Returns ``(loss, per_token_losses)``.

    This is deliberately naive/list-based: it exists to be the oracle for the
    online-merge numerics without dragging in numpy/torch.
    """
    n = len(logits_rows)
    vocab = len(logits_rows[0]) if n else 0
    logz = [float("-inf")] * n
    target = [0.0] * n
    for start in range(0, vocab, vocab_chunk):
        end = min(start + vocab_chunk, vocab)
        for i, row in enumerate(logits_rows):
            chunk = row[start:end]
            logz[i] = online_lse_merge(logz[i], _logsumexp_row(chunk))
            lab = labels[i]
            if lab != ignore_index and start <= lab < end:
                target[i] = float(chunk[lab - start])
    per_token = [
        0.0 if labels[i] == ignore_index else logz[i] - target[i]
        for i in range(n)
    ]
    n_valid = max(1, sum(1 for l in labels if l != ignore_index))
    if reduction == "sum":
        loss = math.fsum(per_token)
    elif reduction == "none":
        loss = 0.0  # caller uses per_token
    else:
        loss = math.fsum(per_token) / n_valid
    return loss, per_token


# ------------------------------------------------------------- torch pathway
def _torch():
    import torch  # lazy: training extra

    return torch


_FUNC_SINGLETON: Any = None


def _build_chunked_ce_function() -> Any:
    """Build (once) the autograd.Function implementing chunked CE on tensors.

    Mirrors the SHAPE of airllm's design (inspiration ``chunked_ce.py``): a
    stateless autograd.Function with forward = vocab-chunked online LSE +
    target-logit gather under no_grad; backward = re-run the same chunks,
    form ``softmax - onehot`` and accumulate d_hidden chunk by chunk. The
    weight matrix may live on CPU (chunks move to the compute device).
    """
    global _FUNC_SINGLETON
    if _FUNC_SINGLETON is not None:
        return _FUNC_SINGLETON
    import torch

    class ChunkedLinearCE(torch.autograd.Function):
        """Tensor front door; config stashed per-apply via class attrs."""

        weight = None
        chunk_size = 4096
        ignore_index = IGNORE_INDEX_DEFAULT
        reduction = "mean"

        @staticmethod
        def forward(ctx, hidden: Any, labels: Any) -> Any:  # type: ignore[override]
            torch = _torch()
            weight = ChunkedLinearCE.weight
            if weight is None:
                raise RuntimeError("chunked CE weight was not set")
            chunk_size = int(ChunkedLinearCE.chunk_size)
            ignore_index = int(ChunkedLinearCE.ignore_index)
            reduction = str(ChunkedLinearCE.reduction)

            n, _h = hidden.shape
            vocab = weight.shape[0]
            if chunk_size < 1:
                chunk_size = vocab
            labels = labels.reshape(-1).to(device=hidden.device)
            if labels.numel() != n:
                raise ValueError(
                    f"labels has {labels.numel()} elements, hidden has {n} rows")

            # Accumulate at fp32 minimum; keep fp64 when the input is fp64 so
            # equality checks against F.cross_entropy are honest.
            acc_dtype = torch.float32 if hidden.dtype in (torch.float16,
                                                          torch.bfloat16) \
                else hidden.dtype

            valid = labels != ignore_index
            logz = torch.full((n,), float("-inf"),
                              device=hidden.device, dtype=acc_dtype)
            target_logit = torch.zeros(n, device=hidden.device,
                                       dtype=acc_dtype)

            with torch.no_grad():
                for start in range(0, vocab, chunk_size):
                    end = min(start + chunk_size, vocab)
                    w = weight[start:end].to(device=hidden.device,
                                             dtype=hidden.dtype)
                    logits = (hidden @ w.t()).to(acc_dtype)
                    # ONLINE merge — same recurrence as online_lse_merge.
                    logz = torch.logaddexp(logz, logits.logsumexp(dim=-1))
                    in_chunk = valid & (labels >= start) & (labels < end)
                    if in_chunk.any():
                        rows = in_chunk.nonzero(as_tuple=False).squeeze(-1)
                        target_logit[rows] = logits[rows, labels[rows] - start]

            n_valid = int(valid.sum().clamp(min=1).item())
            per_token = torch.where(valid, logz - target_logit,
                                    torch.zeros_like(logz))
            if reduction == "sum":
                loss = per_token.sum()
            elif reduction == "none":
                loss = per_token
            else:
                loss = per_token.sum() / max(n_valid, 1)

            ctx.save_for_backward(hidden, labels)
            ctx.weight = weight
            ctx.chunk_size = chunk_size
            ctx.n_valid = n_valid
            ctx.reduction = reduction
            ctx.logz = logz
            return loss

        @staticmethod
        def backward(ctx, grad_loss: Any) -> Tuple[Any, None]:  # type: ignore[override]
            torch = _torch()
            hidden, labels = ctx.saved_tensors
            weight = ctx.weight
            chunk_size = ctx.chunk_size
            vocab = weight.shape[0]
            acc_dtype = ctx.logz.dtype

            scale = grad_loss.to(device=hidden.device, dtype=acc_dtype)
            if ctx.reduction == "mean":
                scale = scale / max(ctx.n_valid, 1)
            elif ctx.reduction == "none":
                scale = scale.reshape(-1)

            d_hidden = torch.zeros(hidden.shape, device=hidden.device,
                                   dtype=acc_dtype)
            valid = labels != int(ChunkedLinearCE.ignore_index)
            for start in range(0, vocab, chunk_size):
                end = min(start + chunk_size, vocab)
                w = weight[start:end].to(device=hidden.device,
                                         dtype=hidden.dtype)
                logits = (hidden @ w.t()).to(acc_dtype)
                probs = torch.exp(logits - ctx.logz.unsqueeze(-1))
                in_chunk = valid & (labels >= start) & (labels < end)
                if in_chunk.any():
                    rows = in_chunk.nonzero(as_tuple=False).squeeze(-1)
                    probs[rows, labels[rows] - start] -= 1.0
                probs = probs.masked_fill(~valid.unsqueeze(-1), 0.0)
                w_acc = w.to(acc_dtype)
                if ctx.reduction == "none":
                    d_hidden += (probs * scale.unsqueeze(-1)) @ w_acc
                else:
                    d_hidden += (probs @ w_acc) * scale

            return d_hidden.to(dtype=hidden.dtype), None

    _FUNC_SINGLETON = ChunkedLinearCE
    return ChunkedLinearCE


def chunked_linear_cross_entropy(hidden: Any, weight: Any, labels: Any, *,
                                 chunk_size: int = 4096,
                                 ignore_index: int = IGNORE_INDEX_DEFAULT,
                                 reduction: str = "mean") -> Any:
    """Cross-entropy of ``hidden @ weight.T`` vs ``labels``, vocab-chunked.

    ``hidden``: [N, H] or [B, S, H]; ``weight``: [V, H]; ``labels`` matches
    hidden's token layout. Peak extra memory ~ ``[N, chunk_size]`` logits —
    never the full ``[N, V]``. Requires torch (lazy import at call time).
    """
    _torch()  # eager availability check with a clear ImportError if absent
    if hidden.dim() == 3:
        hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1)
    cls = _build_chunked_ce_function()
    cls.weight = weight
    cls.chunk_size = int(chunk_size)
    cls.ignore_index = int(ignore_index)
    cls.reduction = reduction
    try:
        return cls.apply(hidden, labels)
    finally:
        cls.weight = None

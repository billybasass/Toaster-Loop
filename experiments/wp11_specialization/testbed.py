"""Tiny controllable MoE testbed for WP11 (torch, lazily imported).

WHY hand-rolled instead of transformers: a 4-expert top-level MoE over a
32-char vocabulary is ~22k parameters, trains in ~2 s on CPU, and every
routing decision is directly inspectable — exactly what a specialization
spike needs. ``AutoModelForCausalLM.from_config`` would work offline (no
weight download) but adds a dependency and hides the router; not worth it at
this scale. No downloads occur (nothing here touches network).

Mechanism inspiration (D-12: cited, NEVER imported): the expert-pruning and
calibrated activation-sparsity ideas follow sparse_transformers-style
methodology — collect statistics on a calibration set, then prune/mask at
inference. Re-implemented inline below in plain torch.

Model (pure-tensor param dict; autograd on leaf tensors keeps this file
torch-import-lazy — no nn.Module needed):

    X      = concat(one-hot(slot1..slot4))          padded prefix  [B, 4V]
    r      = softmax(X @ Wr)                        router probs   [B, E]
    z_e    = relu(X @ W1[e] + b1[e])                expert hidden  [B, H]
    y_e    = z_e @ W2[e] + b2[e]                    expert logits  [B, V]
    logits = sum_e r[:, e] * y_e                    dense mixture (train+eval)

At inference, pruned experts contribute nothing and the router is
renormalized over survivors; activation-sparsity masks low expert-hidden
activations at calibrated per-unit thresholds. argmax over logits is the
greedy ("temperature 0") completion.

Task families ("domains") — two disjoint alphabets of 16 chars each, so a
LINEAR read of the prefix already identifies the domain (a shared alphabet
would make domain detection relational and unlearnable at this scale;
documented simplification in docs/SPECIALIZATION-WP11.md):

    domain A: alphabet "abcdefghijklmnop", next = (idx + 1) mod 16
    domain B: alphabet "qrstuvwxyz234567", next = (3*idx + 7) mod 16

A task extends a deterministic sequence by ONE char given its first L chars
(L in {2,3,4}; shorter prefixes zero-pad slots 3/4).

SPLIT SEMANTICS (deliberate, mirrors the PRODUCTION validation gate):
validate.py golden tasks are REGRESSION TRIPWIRES over trained behaviors,
not generalization tests — and novel-start generalization is
information-theoretically impossible here (one-hot char identities carry no
order structure, so unseen start chars are unlearnable; verified 0% and
documented as a dead end in the findings doc §testbed). Therefore:

    train  = every domain x start x prefix length {2,3,4} (full behavior
             distribution — production tripwire semantics)
    calib  = prefix-length-2 tasks only, skewed ~70% toward domain A (the
             simulated per-user buffer shape)
    eval   = prefix-length-4 tasks only; equal coverage of both domains

The WP11-mandated DISCIPLINE split (calibration vs eval) is enforced BY
CONSTRUCTION: calib prompts are 2 chars, eval prompts 4 chars, so no prompt
string ever crosses the boundary. Baseline saturation behavior is measured,
not assumed — the driver runs a "full" (converged) and a "limited-steps"
regime precisely because a saturated baseline puts the noise floor at zero
and would make any variant look like a win (cardinal rule 2).

All tensors float32 CPU; deterministic given the seed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

# Two disjoint 16-char alphabets (subset of RFC 4648 base32, lowercase).
ALPHABETS = {"A": "abcdefghijklmnop", "B": "qrstuvwxyz234567"}
VOCAB = ALPHABETS["A"] + ALPHABETS["B"]  # model-level vocab, 32 chars
V = len(VOCAB)
ALPH_SIZE = 16
SEQ_LEN = 4               # fixed input slots (shorter prompts zero-pad)
DIM = SEQ_LEN * V         # ordered one-hot concat width
PROMPT_LENGTHS = (2, 3, 4)  # task variety: extend after L observed chars
N_EXPERTS = 4
HIDDEN = 32
N_TRAIN_STARTS = 12       # starts [0,12) train/calib; [12,16) eval-only

DOMAINS = ("A", "B")


def _torch():
    import torch  # lazy: [training]-extra policy holds even in experiments/

    return torch


# --------------------------------------------------------------- task families

def rule(domain: str, c: int) -> int:
    """Next-char index WITHIN the domain alphabet (the ground-truth rule)."""
    if domain == "A":
        return (c + 1) % ALPH_SIZE
    if domain == "B":
        return (3 * c + 7) % ALPH_SIZE
    raise ValueError(f"unknown domain {domain!r}")


def make_sequence(domain: str, start: int, length: int = SEQ_LEN + 1) -> str:
    """Deterministic char sequence of one domain starting at ``start``."""
    alph = ALPHABETS[domain]
    idx = start % ALPH_SIZE
    chars = [alph[idx]]
    for _ in range(length - 1):
        idx = rule(domain, idx)
        chars.append(alph[idx])
    return "".join(chars)


def make_task(start: int, domain: str, plen: int = SEQ_LEN) -> Dict[str, Any]:
    """One golden-style task: first ``plen`` chars -> the NEXT char.

    ``expect_contains`` mirrors training/validate.py check semantics exactly:
    a case-insensitive substring check over the reply. The expected answer is
    one char and the reply is one char, so pass <=> greedy completion equals
    it; single base32 chars can never accidentally substring-collide.
    """
    if plen not in PROMPT_LENGTHS:
        raise ValueError(f"plen must be one of {PROMPT_LENGTHS}")
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}")
    seq = make_sequence(domain, start % ALPH_SIZE)
    return {
        "start": start % ALPH_SIZE,
        "domain": domain,
        "plen": plen,
        "prompt": seq[:plen],
        "expect_contains": [seq[plen]],  # the held-out next char
    }


def _tasks(starts: Sequence[int], plens: Sequence[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in DOMAINS:
        for s in starts:
            for plen in plens:
                out.append(make_task(s, d, plen))
    return out


def split_tasks() -> Dict[str, List[Dict[str, Any]]]:
    """Train / calibration / eval task sets (deterministic; semantics above).

    * train: every domain x start x prefix length {2,3,4} — the full
             behavior distribution (production-tripwire semantics).
    * calib: prefix-length-2 tasks only, skewed ~70% toward domain A — the
             simulated per-user buffer shape. Prompt-DISJOINT from eval.
    * eval:  prefix-length-4 tasks only, equal coverage of both domains so
             a specialization that hurts one domain cannot hide behind the
             calibration skew. Same behaviors as train (tripwires), prompts
             disjoint from calib.
    """
    all_starts = range(ALPH_SIZE)
    # Train covers ALL shapes incl. the eval shape (plen 4): golden tasks are
    # regression tripwires over TRAINED behaviors (production semantics), and
    # unseen-shape transfer measures zero on this toy (verified: plen-4-only
    # eval with plen-{2,3}-only training is 0% — documented in findings doc).
    train = _tasks(all_starts, PROMPT_LENGTHS)

    # calibration: deterministic pseudo-random pick with a 70/30 domain-A
    # skew (a user who mostly does domain A work), short prompts only.
    calib: List[Dict[str, Any]] = []
    m = 64
    for i in range(m):
        d = "A" if (i % 10) < 7 else "B"
        s = (i * 7 + 3 * i * i + 5) % ALPH_SIZE
        calib.append(make_task(s, d, 2))

    eval_tasks = _tasks(all_starts, (4,))

    # discipline assertion: no prompt string crosses the calib/eval boundary
    cal_prompts = {t["prompt"] for t in calib}
    assert all(t["prompt"] not in cal_prompts for t in eval_tasks), \
        "split leak: calibration and evaluation prompts must be disjoint"
    return {"train": train, "calib": calib, "eval": eval_tasks}


# --------------------------------------------------------------------- model

def make_params(seed: int) -> Dict[str, Any]:
    """Deterministic random-init parameter dict for one replicate."""
    torch = _torch()
    g = torch.Generator().manual_seed(seed)
    p: Dict[str, Any] = {
        "Wr": torch.randn(N_EXPERTS, DIM, generator=g) * DIM ** -0.5,
        # experts stacked: W1 [E,H,DIM], W2 [E,V,H]
        "W1": torch.randn(N_EXPERTS, HIDDEN, DIM, generator=g) * (HIDDEN * DIM) ** -0.5,
        "b1": torch.zeros(N_EXPERTS, HIDDEN),
        "W2": torch.randn(N_EXPERTS, V, HIDDEN, generator=g) * (V * HIDDEN) ** -0.5,
        "b2": torch.zeros(N_EXPERTS, V),
    }
    for t in p.values():
        t.requires_grad_(True)
    return p


def encode(tasks: Sequence[Dict[str, Any]]) -> Tuple[Any, Any]:
    """Tasks -> (X [B, 4V] ordered one-hot concat w/ zero padding, Y [B])."""
    torch = _torch()
    xs, ys = [], []
    for t in tasks:
        x = torch.zeros(DIM)
        for i, c in enumerate(t["prompt"][:SEQ_LEN]):
            x[i * V + VOCAB.index(c)] = 1.0
        xs.append(x)
        ys.append(VOCAB.index(t["expect_contains"][0]))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def forward(params: Dict[str, Any], X: Any, *,
            keep: Any = None, act_thresholds: Any = None) -> Any:
    """Dense-mixture forward with optional expert pruning + act masking.

    keep           bool tensor [E]: pruned-out experts contribute nothing and
                   the router is RENORMALIZED over survivors (the standard
                   renormalization rule from MoE pruning methodology).
    act_thresholds float tensor [E,H] or None: expert-hidden activations
                   below the calibrated threshold are zeroed (activation
                   sparsity — calibrated top-k-style masking at inference).
    """
    torch = _torch()
    r = torch.softmax(X @ params["Wr"].T, dim=-1)  # [B,E]
    if keep is not None:
        r = r * keep.to(r.dtype).unsqueeze(0)
        denom = r.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        r = r / denom  # renormalize over survivors
    logits = torch.zeros(X.shape[0], V)
    for e in range(N_EXPERTS):
        if keep is not None and not bool(keep[e]):
            continue
        z = torch.relu(X @ params["W1"][e].T + params["b1"][e])  # [B,H]
        if act_thresholds is not None:
            z = z * (z >= act_thresholds[e].unsqueeze(0)).to(z.dtype)
        y = z @ params["W2"][e].T + params["b2"][e]  # [B,V]
        logits = logits + r[:, e].unsqueeze(-1) * y
    return logits


def greedy_completion(params: Dict[str, Any], task: Dict[str, Any], *,
                      keep: Any = None, act_thresholds: Any = None) -> str:
    """Greedy one-char completion for one golden task (temperature-0 style)."""
    torch = _torch()
    X, _ = encode([task])
    with torch.no_grad():
        logits = forward(params, X, keep=keep, act_thresholds=act_thresholds)
        idx = int(logits.argmax(dim=-1)[0])
    return VOCAB[idx]


# ------------------------------------------------------------------- training

def train_model(params: Dict[str, Any], tasks: Sequence[Dict[str, Any]], *,
                steps: int = 600, lr: float = 0.05) -> Dict[str, Any]:
    """Full-batch Adam on the dense mixture; deterministic given (params).

    The TRAIN corpus is identical across replicates; only init differs — so
    replicate-to-replicate eval variance measures model-init nondeterminism
    at fixed config, which is exactly the noise source the floor captures.
    """
    torch = _torch()
    X, Y = encode(tasks)
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    loss = float("nan")
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = forward(params, X)
        obj = torch.nn.functional.cross_entropy(logits, Y)
        obj.backward()
        opt.step()
        loss = float(obj.detach())
    params["_final_loss"] = loss
    return params


def evaluate_accuracy(params: Dict[str, Any], tasks: Sequence[Dict[str, Any]],
                      **kw: Any) -> float:
    """Raw next-char accuracy (diagnostic ONLY — never the adoption metric;
    adoption is judged exclusively through evalgate/validate.py semantics)."""
    if not tasks:
        return 0.0
    hits = sum(1 for t in tasks
               if greedy_completion(params, t, **kw) == t["expect_contains"][0])
    return hits / len(tasks)

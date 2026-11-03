"""Streamed-LoRA SFT runner — the worker's ``training.backend: streamed`` path.

WP10 step 2. Heavy imports (torch / transformers / safetensors) happen
LAZILY inside functions only — this module lives on the training-extra seam
alongside worker.py (tests/test_hygiene.py folds the streamed
package into that seam; importing it from the daemon would violate D-12,
and nothing outside worker.py does). Top-level imports here are stdlib +
the torch-free siblings (masking, gguf_lora), so ``toaster_loop.training.worker``
can import this module statically without dragging GPU deps in.

Flow (mirrors docs/SPIKE-WP10.md's recommended seam):

  1. resolve the base model: an HF checkpoint DIRECTORY already on disk —
     streamed mode never downloads (zero-cloud rule; hub ids are refused
     with an actionable error).
  2. layer splitting (first run): re-shard the checkpoint by decoder layer
     into ``<output_dir>/layers/`` while streaming "splitting checkpoint:
     shard i/n" events into the job log. Disk cost ~1x model size while
     both trees exist; ``training.delete_original: true`` reclaims AFTER
     verification, and only for trees under the run dir — a user-owned
     checkpoint is never deleted.
  3. build the model shell on the meta device from config.json (pinned
     reference archs only), materialize embeddings/norm/lm_head on CPU,
     ``inject_lora`` into the decoder layers and stream layer weights from
     the split shards through ``LayerStreamer`` with a bounded window.
  4. dataset: ShareGPT sft.jsonl rows -> per-turn masked token ids via
     :mod:`toaster_loop.training.streamed.masking` (stricter than airllm:
     ALL non-assistant tokens are -100, not just the prefix before the
     last assistant turn).
  5. training loop: ``StreamedRecomputeTrainer.step`` with
     ``chunked_linear_cross_entropy(hidden, lm_head.weight, labels)`` as
     the loss head; per-step rows appended to ``train_metrics.csv`` with
     the exact schema ``step,loss,peak_vram_gb,elapsed_s`` (peak reads
     0.0 without CUDA — recorded honestly, no fake numbers).
  6. artifact: adapter saved as the FROZEN lora-state-dict contract
     (module-name -> tensor) via safetensors + adapter_config.json, then
     converted to GGUF-LoRA inline (:mod:`gguf_lora`) at
     ``<output_dir>/lora.gguf`` — the artifact the existing registry/
     export flow consumes (worker.run_export copies it into place).

HONESTY (cardinal rule 2): everything proven here ran on a CPU-only box
(torch 2.14.1, transformers 5.19.0 pinned). No speed or VRAM claims;
CUDA-specific paths code-reviewed only. The HF from_pretrained-free shell
glue IS executed by the hermetic e2e test with a tiny synthetic
llama-shaped checkpoint (CPU). Loading the produced GGUF-LoRA under a real
llama-server -lora is UNVERIFIED HERE (see docs/ROADMAP.md WP10).
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# gguf_lora / masking are torch-free and imported at module level; the heavy
# (torch/transformers/safetensors) seams stay lazily imported inside
# run_streamed_sft / build_streamed_model below.
from .masking import mask_conversation, sharegpt_to_messages

#: model_type values this runner can glue (spike GO condition 1: ONE
#: reference arch family first — llama-shaped dense decoders). qwen3 shares
#: the layer role map; its GGUF arch name differs.
SUPPORTED_MODEL_TYPES = {
    "llama": "llama",
    "qwen3": "qwen3",
}

#: LoRA targets shared by the supported archs (attention + MLP projections).
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj")

METRICS_CSV = "train_metrics.csv"
LAYER_SPLIT_DIRNAME = "layers"


# ------------------------------------------------------------- backend parse
def parse_backend(raw: Any, warn: Optional[Callable[[str], None]] = None) -> str:
    """Lenient ``training.backend`` parse (WP2 convention for enum keys).

    "streamed" -> streamed; anything else (missing/invalid/mistyped) ->
    "peft", the pre-WP10 path, with a warning for unknown non-default
    strings. The default value reproduces today's behavior byte-identically.
    """
    if raw is None:
        return "peft"
    token = str(raw).strip().lower()
    if token in ("", "peft", "standard"):
        return "peft"
    if token == "streamed":
        return "streamed"
    if warn is not None:
        warn(f"training.backend={raw!r} is not one of peft|streamed; "
             "falling back to default 'peft'")
    return "peft"


# --------------------------------------------------------------- progress
def _emit(log_path: Optional[str], event: str, **fields: Any) -> None:
    """Same append-only JSON protocol as worker._emit (duplicated so the
    streamed runner stays importable without importing worker.py)."""
    from datetime import datetime, timezone

    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pid": __import__("os").getpid(),
        "event": event,
        **fields,
    }
    line = json.dumps(rec, ensure_ascii=False)
    print(line, flush=True)
    if log_path:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


# --------------------------------------------------------------- checkpoint
def _layer_index(tensor_names: List[str]) -> Dict[int, List[str]]:
    """Group checkpoint tensor names by decoder-layer index."""
    out: Dict[int, List[str]] = {}
    pat = re.compile(r"^model\.layers\.(\d+)\.")
    for name in tensor_names:
        m = pat.match(name)
        if m:
            out.setdefault(int(m.group(1)), []).append(name)
    return out


def _shard_tensor_index(model_dir: Path) -> Tuple[Dict[str, Tuple[str, str]], int]:
    """tensor_name -> (shard_path, tensor_name), + num_hidden_layers."""
    from safetensors import safe_open  # lazy: training extra

    index_f = model_dir / "model.safetensors.index.json"
    mapping: Dict[str, str] = {}
    if index_f.is_file():
        idx = json.loads(index_f.read_text())
        for tname, shard in (idx.get("weight_map") or {}).items():
            mapping[tname] = str(model_dir / shard)
    else:
        for shard in sorted(model_dir.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt") as f:
                for tname in f.keys():
                    mapping[tname] = str(shard)
    if not mapping:
        raise FileNotFoundError(
            f"no safetensors weights found under {model_dir} — streamed "
            "mode needs an HF checkpoint directory already on disk")
    cfg = json.loads((model_dir / "config.json").read_text())
    n_layers = int(cfg.get("num_hidden_layers") or 0)
    if n_layers < 1:
        raise ValueError(f"config.json in {model_dir} has no usable "
                         "num_hidden_layers")
    return {t: (p, t) for t, p in mapping.items()}, n_layers


def split_checkpoint(model_dir: Path, split_dir: Path, *,
                     log: Optional[Callable[..., None]] = None,
                     delete_original: bool = False) -> Dict[str, Any]:
    """Re-shard the checkpoint by decoder layer under ``split_dir``.

    Writes ``layer-<i>.safetensors`` per decoder layer plus
    ``shell.safetensors`` for everything else (embeddings, final norm,
    lm_head), and a ``split_index.json`` consumed by the loader. Progress
    ("splitting checkpoint: shard i/n") goes to the job log so the
    dashboard Jobs pane shows live splitting state.

    Disk cost note: while both trees exist this costs ~1x the model size;
    ``delete_original`` reclaims afterwards — but ONLY for shard files
    inside a tree under ``split_dir`` (a copy we made). A user-owned
    checkpoint directory is never deleted, regardless of the flag.
    """
    from safetensors import safe_open           # lazy: training extra
    from safetensors.torch import save_file     # lazy

    log = log or (lambda **kw: None)
    resolved, n_layers = _shard_tensor_index(model_dir)
    by_layer = _layer_index(list(resolved))
    split_dir.mkdir(parents=True, exist_ok=True)

    handles: Dict[str, Any] = {}

    def _handle(path: str):
        h = handles.get(path)
        if h is None:
            h = safe_open(path, framework="pt")
            handles[path] = h
        return h

    def _grab(names: List[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for n in names:
            shard, tname = resolved[n]
            out[n] = _handle(shard).get_tensor(tname)
        return out

    total = n_layers + 1  # decoder layers + shell
    index_files: Dict[str, List[str]] = {}
    for i in range(n_layers):
        names = sorted(by_layer.get(i, []))
        tensors = _grab(names)
        fname = f"layer-{i}.safetensors"
        save_file(tensors, str(split_dir / fname),
                  metadata={"toaster_loop_layer": str(i)})
        index_files[fname] = names
        log(event="phase", phase="splitting_checkpoint", shard=i + 1,
            total=total,
            message=f"splitting checkpoint: shard {i + 1}/{total}")
    layer_keys = {n for ns in by_layer.values() for n in ns}
    shell_names = sorted(set(resolved) - layer_keys)
    save_file(_grab(shell_names), str(split_dir / "shell.safetensors"))
    index_files["shell.safetensors"] = shell_names
    log(event="phase", phase="splitting_checkpoint", shard=total,
        total=total, message=f"splitting checkpoint: shard {total}/{total}")

    (split_dir / "split_index.json").write_text(json.dumps(
        {"files": index_files, "num_layers": n_layers}, indent=1))

    # Verify before any reclaim: every split file opens and covers its keys.
    for fname, names in index_files.items():
        with safe_open(str(split_dir / fname), framework="pt") as f:
            have = set(f.keys())
        missing = set(names) - have
        if missing:
            raise RuntimeError(
                f"split verification failed: {fname} missing {sorted(missing)[:4]}")

    reclaimed = 0
    if delete_original:
        src = Path(model_dir).resolve()
        root = Path(split_dir).resolve().parent
        if src == root or str(src).startswith(str(root) + "/"):
            for shard in Path(model_dir).glob("*.safetensors"):
                reclaimed += shard.stat().st_size
                shard.unlink()
        else:
            log(event="warning",
                message=f"delete_original ignored: {model_dir} is not under "
                        f"{root} — toaster_loop never deletes a user-owned "
                        "checkpoint tree")
    return {"files": len(index_files), "num_layers": n_layers,
            "reclaimed_bytes": reclaimed, "dir": str(split_dir)}


# ---------------------------------------------------------- layer streaming
class ShellLayerLoader:
    """LayerStreamer loader: swap real tensors into a meta shell layer.

    ``load(i)`` reads layer i's checkpoint tensors and swaps them into the
    shell's Parameter objects via ``torch.utils.swap_tensors`` (Parameter
    identity preserved — required by the frozen lora_state_dict contract);
    ``unload`` swaps the previous (meta) parameters back so residency drops
    to the window size. LoRA parameters are never swapped out: they live on
    CPU permanently (tiny, trainable).
    """

    def __init__(self, layer_params: List[Dict[str, Any]], split_dir: Path,
                 *, base_prefix: str = "model.layers"):
        self.layer_params = layer_params
        self.split_dir = Path(split_dir)
        self.base_prefix = base_prefix
        self._handles_cache: Dict[str, Any] = {}
        idx = json.loads((self.split_dir / "split_index.json").read_text())
        self.file_of: Dict[str, str] = {}
        for fname, names in idx["files"].items():
            for n in names:
                self.file_of[n] = fname

    def _handle(self, fname: str):
        h = self._handles_cache.get(fname)
        if h is None:
            from safetensors import safe_open  # lazy: training extra
            h = safe_open(str(self.split_dir / fname), framework="pt")
            self._handles_cache[fname] = h
        return h

    def load(self, layer_idx: int) -> Dict[str, Any]:
        import torch  # lazy: training extra

        swapped: List[Tuple[Any, Any]] = []
        for pname, p in self.layer_params[layer_idx].items():
            if "lora_" in pname:
                continue  # adapter params live on CPU permanently
            flat = f"{self.base_prefix}.{layer_idx}.{pname}"
            fname = self.file_of.get(flat)
            if fname is None:
                continue  # not a checkpoint tensor (buffer etc.) — skip-safe
            t = self._handle(fname).get_tensor(flat)
            with torch.no_grad():
                real = torch.nn.Parameter(t.to(p.dtype), requires_grad=False)
            if p.device.type == "meta":
                torch.utils.swap_tensors(p, real)
            else:
                cpu_backup = torch.nn.Parameter(
                    p.detach().clone(), requires_grad=False)
                with torch.no_grad():
                    p.data = real.data
                real = cpu_backup  # swap-back restores the previous copy
            swapped.append((p, real))
        return {"layer": layer_idx, "swapped": swapped}

    def unload(self, handle: Dict[str, Any]) -> None:
        import torch  # lazy: training extra

        for obj, other in reversed(handle.get("swapped", [])):
            torch.utils.swap_tensors(obj, other)
        handle["swapped"] = []


# ------------------------------------------------------------- shell build
def _streamed_lora_settings(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    lora = cfg_dict.get("lora") or {}
    targets = lora.get("target_modules") or list(DEFAULT_TARGET_MODULES)
    if isinstance(targets, str):
        targets = [t.strip() for t in targets.split(",") if t.strip()]
    return {
        "r": int(lora.get("lora_r", 16)),
        "alpha": int(lora.get("lora_alpha", 32)),
        "target_modules": list(targets),
    }


def build_streamed_model(cfg_dict: Dict[str, Any], model_dir: Path,
                         split_dir: Path, *, log: Callable[..., None],
                         ):
    """Meta-device shell (pinned arch) + injected LoRA + CPU head tensors.

    Returns (model, layers, layer_params, embed_w, lm_head_w, rope_cfg).
    The from_pretrained-free glue (from_config on meta + explicit
    position_embeddings per layer) is pinned to SUPPORTED_MODEL_TYPES and
    IS executed by the hermetic e2e test on a tiny synthetic checkpoint.
    """
    import torch  # lazy: training extra
    from transformers import AutoConfig, AutoModelForCausalLM  # lazy

    from .lora_linear import inject_lora
    from .recompute import assert_no_stochastic_dropout

    cfg = AutoConfig.from_pretrained(str(model_dir))
    mtype = str(getattr(cfg, "model_type", "")).lower()
    if mtype not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"streamed backend supports model_type "
            f"{sorted(SUPPORTED_MODEL_TYPES)} (pinned reference archs — one "
            f"arch first is a spike GO condition, see docs/ROADMAP.md WP10), "
            f"got {mtype!r}")

    split_info = split_checkpoint(model_dir, split_dir, log=log)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    try:
        model.config.use_cache = False
    except Exception:
        pass
    model.eval()

    lset = _streamed_lora_settings(cfg_dict)
    layers_mod = model.model.layers
    n_injected = inject_lora(
        layers_mod, lset["target_modules"],
        r=lset["r"], alpha=lset["alpha"], dropout=0.0)
    if not n_injected:
        raise ValueError(
            f"inject_lora matched 0 modules — target_modules="
            f"{lset['target_modules']} do not exist in this arch")
    assert_no_stochastic_dropout(model)

    # Materialize the non-layer parameters (embeddings, final norm, lm_head)
    # on CPU once; they stay resident (small next to the layer stack).
    from safetensors import safe_open  # lazy: training extra
    with safe_open(str(split_dir / "shell.safetensors"), framework="pt") as f:
        shell_tensors = {k: f.get_tensor(k) for k in f.keys()}
    _fill_cpu_tensors(model, shell_tensors)

    layer_params = [dict(layer.named_parameters()) for layer in layers_mod]
    rope_cfg = {
        "head_dim": int(cfg.hidden_size) // int(cfg.num_attention_heads),
        "theta": float(getattr(cfg, "rope_theta", 10000.0)),
        "partial": float(getattr(cfg, "partial_rotary_factor", 1.0)),
    }
    return model, list(layers_mod), layer_params, \
        model.model.embed_tokens.weight, model.lm_head.weight, \
        rope_cfg, split_info


def _fill_cpu_tensors(model: Any, tensors: Dict[str, Any]) -> None:
    """Swap CPU tensors into meta parameters / set buffers (identity kept)."""
    import torch  # lazy: training extra

    params = dict(model.named_parameters())
    bufs = dict(model.named_buffers())
    for name, t in tensors.items():
        obj = params.get(name)
        if obj is not None:
            with torch.no_grad():
                new = torch.nn.Parameter(t.to(obj.dtype), requires_grad=False)
            if obj.device.type == "meta":
                torch.utils.swap_tensors(obj, new)
            else:
                with torch.no_grad():
                    obj.data = t.to(obj.dtype)
            continue
        if name in bufs:
            owner_path, leaf = name.rsplit(".", 1)
            owner = model.get_submodule(owner_path)
            setattr(owner, leaf, t.to(bufs[name].dtype))


class LayerStep:
    """Callable layer view injecting position_embeddings + causal mask.

    StreamedRecomputeTrainer calls ``layer(h)``; the pinned reference archs
    take ``(hidden_states, attention_mask=..., position_embeddings=(cos,
    sin), use_cache=False)`` — verified byte-equal to the full-model pass
    on transformers 5.19.0 (CPU). The rotary module itself stays on meta:
    cos/sin are computed here from config rope parameters (standard
    non-interpolated RoPE; exotic rope configs are rejected by the shell
    builder's arch pinning). Not an nn.Module child of anything — holds the
    real layer in a plain list so parameter discovery is unaffected.
    """

    def __init__(self, layer: Any):
        self._ref = [layer]
        self.position_embeddings: Optional[Tuple[Any, Any]] = None
        self.attention_mask: Optional[Any] = None

    def named_modules(self):  # duck-type for assert_no_stochastic_dropout
        yield "", self._ref[0]
        yield from self._ref[0].named_modules()

    def __call__(self, h: Any) -> Any:
        return self._ref[0](
            h,
            attention_mask=self.attention_mask,
            position_embeddings=self.position_embeddings,
            use_cache=False,
        )


def compute_rope(rope_cfg: Dict[str, Any], seq_len: int, batch: int):
    """(cos, sin) [B, S, rot_dim] fp32 for standard llama-style RoPE."""
    import torch  # lazy: training extra

    rot_dim = max(2, int(rope_cfg["head_dim"] * rope_cfg.get("partial", 1.0)))
    half = rot_dim // 2
    theta = float(rope_cfg["theta"])
    inv_freq = torch.tensor(
        [1.0 / (theta ** (2.0 * i / rot_dim)) for i in range(half)],
        dtype=torch.float32)
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = pos.view(-1, 1) * inv_freq.view(1, -1)      # [S, half]
    emb = torch.cat([freqs, freqs], dim=-1)             # [S, rot_dim]
    cos = emb.cos().view(1, seq_len, rot_dim).expand(batch, -1, -1)
    sin = emb.sin().view(1, seq_len, rot_dim).expand(batch, -1, -1)
    return cos.contiguous(), sin.contiguous()


def causal_mask(seq_len: int, dtype: Any):
    """Additive causal mask [1, 1, S, S] (0 / -inf upper triangle)."""
    import torch  # lazy: training extra

    m = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32)
    m = torch.triu(m, diagonal=1)
    return m.view(1, 1, seq_len, seq_len).to(dtype)


# ------------------------------------------------------------------- train
def _peak_vram_gb() -> float:
    """Peak CUDA memory in GB; 0.0 without CUDA — recorded honestly."""
    try:
        import torch  # lazy: training extra
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return 0.0


def _load_rows(files: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _state_for_gguf(state: Dict[str, Any]) -> Dict[str, Any]:
    """Rebase inject_lora's ModuleList-indexed keys onto checkpoint names.

    lora_state_dict keys look like ``0.self_attn.q_proj.lora_A`` (the layer
    index is the ModuleList position); the GGUF writer wants the frozen
    contract shape ``model.layers.<i>.<...>.lora_A``.
    """
    out: Dict[str, Any] = {}
    for k, v in state.items():
        m = re.match(r"^(\d+)\.(.+)$", k)
        out[f"model.layers.{m.group(1)}.{m.group(2)}" if m else k] = v
    return out


def run_streamed_sft(cfg: Dict[str, Any], log_path: Optional[str],
                     dry_run: bool) -> Dict[str, Any]:
    """Worker entry for backend=streamed (called from worker.run_sft)."""
    import torch  # lazy: training extra
    from transformers import AutoTokenizer  # lazy

    from .chunked_ce import chunked_linear_cross_entropy
    from .gguf_lora import write_gguf_lora
    from .lora_linear import lora_parameters, lora_state_dict
    from .recompute import StreamedRecomputeTrainer
    from .streaming import LayerStreamer

    def log(**kw: Any) -> None:
        _emit(log_path, **kw)

    data_sec = cfg.get("data") or {}
    files: List[str] = list(data_sec.get("local_dataset") or [])
    if not files:
        raise ValueError("data.local_dataset is empty — nothing to train on")
    for f in files:
        if not Path(f).is_file():
            raise FileNotFoundError(f"dataset file missing: {f}")
    tr = cfg.get("training") or {}
    out_dir = Path(tr.get("output_dir") or "outputs").expanduser()
    adapter_dir = out_dir / "adapter"

    model_ref = str(cfg.get("model") or "")
    model_dir = Path(model_ref).expanduser()
    if not model_dir.is_dir():
        raise ValueError(
            "streamed backend trains from an HF checkpoint DIRECTORY already "
            f"on disk (zero-cloud); training.base_model={model_ref!r} is not "
            "a local directory. Point it at the downloaded checkpoint.")

    rows = _load_rows(files)
    if not rows:
        raise ValueError("dataset(s) contain no rows")
    resolved = {"model": model_ref, "files": files, "rows": len(rows),
                "output_dir": str(out_dir), "adapter_dir": str(adapter_dir),
                "backend": "streamed"}
    if dry_run:
        log(event="dry_run_ok", config=resolved)
        return {"adapter_dir": str(adapter_dir), "dry_run": True, **resolved}

    max_seq = int(tr.get("max_seq_length") or 16384)
    lr = float(tr.get("learning_rate", 2e-4))
    epochs = max(1, int(tr.get("num_epochs", 1) or 1))
    batch_size = max(1, int(tr.get("batch_size", 1)))
    seed = int(tr.get("random_seed", 3407))
    torch.manual_seed(seed)

    split_dir = out_dir / LAYER_SPLIT_DIRNAME
    log(event="phase", phase="load_model_streamed", model=model_ref)
    (model, layers_raw, layer_params, embed_w, lm_head_w, rope_cfg,
     split_info) = build_streamed_model(cfg, model_dir, split_dir, log=log)
    log(event="split_done", num_layers=split_info["num_layers"],
        shards=split_info["files"], dir=split_info["dir"])

    # Tokenizer from the local checkpoint dir (already on disk — no network).
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    tokenized: List[Dict[str, List[int]]] = []
    for r in rows:
        msgs = sharegpt_to_messages(r.get("conversations") or [])
        enc = mask_conversation(tokenizer, msgs, max_seq)
        if enc is not None:
            tokenized.append(enc)
    if not tokenized:
        raise ValueError(
            "no usable rows after per-turn masking — every row failed exact "
            "masking (chat template does not compose over prefixes)")
    log(event="dataset_ready", rows=len(tokenized),
        skipped=len(rows) - len(tokenized))

    # Callable layer views + streamer + recompute trainer.
    steps = [LayerStep(l) for l in layers_raw]
    loader = ShellLayerLoader(layer_params, split_dir)
    streamer = LayerStreamer(loader, len(steps), prefetch_depth=1)
    params = lora_parameters(model)
    trainer = StreamedRecomputeTrainer(steps, params, streamer=streamer)

    H = int(embed_w.shape[1])
    opt = torch.optim.AdamW(params, lr=lr)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_f = out_dir / METRICS_CSV
    new_csv = not metrics_f.exists()
    mf = open(metrics_f, "a", newline="", encoding="utf-8")
    mw = csv.writer(mf)
    if new_csv:
        # Exact schema; peak_vram_gb is 0.0 without CUDA (honest absence).
        mw.writerow(["step", "loss", "peak_vram_gb", "elapsed_s"])

    def _batch(encs: List[Dict[str, List[int]]]):
        B = len(encs)
        S = max(len(e["input_ids"]) for e in encs)
        ids = torch.zeros((B, S), dtype=torch.long)
        labels = torch.full((B, S), -100, dtype=torch.long)
        for b, e in enumerate(encs):
            n = len(e["input_ids"])
            ids[b, :n] = torch.tensor(e["input_ids"])
            labels[b, :n] = torch.tensor(e["labels"])
        return ids, labels

    def _configure_steps(cos: Any, sin: Any, mask: Any) -> None:
        for st in steps:
            st.position_embeddings = (cos, sin)
            st.attention_mask = mask

    t0 = time.time()
    step = 0
    train_loss: Optional[float] = None
    try:
        for _ in range(epochs):
            for start in range(0, len(tokenized), batch_size):
                batch = tokenized[start:start + batch_size]
                ids, labels = _batch(batch)
                B, S = ids.shape
                cos, sin = compute_rope(rope_cfg, S, B)
                mask = causal_mask(S, embed_w.dtype)
                _configure_steps(cos, sin, mask)

                with torch.no_grad():
                    x = torch.nn.functional.embedding(ids, embed_w)

                def loss_fn(final_h: Any):
                    # Chunked CE on a leaf copy: the scalar AND dL/dh come
                    # from the same kernel (== F.cross_entropy per spike),
                    # handed to the recompute-backward pass.
                    hh = final_h.detach().reshape(-1, H).requires_grad_(True)
                    loss = chunked_linear_cross_entropy(
                        hh, lm_head_w, labels.reshape(-1))
                    loss.backward()
                    grad = hh.grad.reshape(final_h.shape).detach()
                    return loss.detach(), grad

                opt.zero_grad(set_to_none=True)
                loss = trainer.step(x, loss_fn)
                opt.step()

                step += 1
                train_loss = float(loss)
                mw.writerow([step, f"{train_loss:.6f}",
                             f"{_peak_vram_gb():.4f}",
                             f"{time.time() - t0:.3f}"])
                mf.flush()
                ev: Dict[str, Any] = {"event": "progress", "step": step,
                                      "loss": train_loss}
                log(**ev)
    finally:
        mf.close()
        streamer.close()

    log(event="phase", phase="save_adapter", adapter_dir=str(adapter_dir))
    adapter_dir.mkdir(parents=True, exist_ok=True)
    state = lora_state_dict(model)
    _save_lora_state(state, adapter_dir / "lora_state.safetensors")
    gguf_arch = SUPPORTED_MODEL_TYPES[str(
        getattr(model.config, "model_type", "")).lower()]
    lset = _streamed_lora_settings(cfg)
    (adapter_dir / "adapter_config.json").write_text(json.dumps({
        "backend": "streamed",
        "base_model": model_ref,
        "format": "toaster_loop-lora-state-dict",
        "lora_r": lset["r"],
        "lora_alpha": lset["alpha"],
        "target_modules": lset["target_modules"],
    }, indent=1))

    # Step 3 artifact: GGUF-LoRA via the frozen contract. Parseability is
    # proven by tests; a llama.cpp adapter load was exercised HERE against a
    # tiny synthetic checkpoint (docs/ROADMAP.md WP10) — real-model loads
    # remain hardware acceptance.
    gguf_path = out_dir / "lora.gguf"
    gguf_done = False
    try:
        summary = write_gguf_lora(
            _state_for_gguf(state), gguf_path,
            arch="llama", base_arch=gguf_arch, alpha=float(lset["alpha"]))
        gguf_done = True
        log(event="gguf_lora_done", path=str(gguf_path),
            tensors=summary["tensors"], arch=summary["arch"],
            verified=("parseable-here; llama.cpp -lora load proven here on a "
                      "tiny synthetic ckpt; real-model load untested"))
    except ValueError as e:
        log(event="warning", message=f"GGUF-LoRA conversion skipped: {e}")

    log(event="streamed_sft_done", adapter_dir=str(adapter_dir),
        train_loss=train_loss, eval_loss=None, backend="streamed",
        gguf_lora_path=str(gguf_path) if gguf_done else None)
    return {"adapter_dir": str(adapter_dir), "train_loss": train_loss,
            "eval_loss": None, "backend": "streamed",
            "gguf_lora_path": str(gguf_path) if gguf_done else None}


def _save_lora_state(state: Dict[str, Any], path: Path) -> None:
    from safetensors.torch import save_file  # lazy: training extra

    save_file({k: v.contiguous() for k, v in state.items()}, str(path))

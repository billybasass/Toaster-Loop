"""GGUF-LoRA writer + arch adapters for the streamed backend (WP10 step 3).

Stdlib-only and torch-free BY DESIGN: the input is the frozen adapter
contract — a plain ``{module-name -> tensor-like}`` mapping (exactly what
``toaster_loop.training.streamed.lora_linear.lora_state_dict`` emits; anything
duck-typed with ``shape`` + bytes support works, so unit tests need no
torch). Keeping this module heavy-import-free means the GGUF writer is
unit-testable on any machine and never needs the training extras (the
import allowlist in tests/test_hygiene.py stays as-is for it).

Format — verified by READING the local llama.cpp checkout this install
builds (``src/llama-adapter.cpp::llama_adapter_lora_init_impl``,
``src/llama-arch.cpp`` KV table, ``convert_lora_to_gguf.py``):

* GGUF v3 container: magic ``GGUF``, version u32=3, n_tensors u64,
  n_kv u64, KV pairs, tensor-info records, data aligned to
  ``general.alignment`` (default 32), then F32 tensor data.
* Metadata the loader REQUIRES: ``general.type`` = "adapter" (throws
  otherwise), ``general.architecture`` matching the BASE model's arch
  (throws on mismatch), ``adapter.type`` = "lora", ``adapter.lora.alpha``
  f32. Conventional extras: ``general.name``, ``adapter.name``.
* Tensor naming: llama.cpp pairs ``<base_key>.lora_a`` / ``<base_key>.lora_b``
  where ``<base_key>`` is the GGUF name of the adapted base-model tensor
  (e.g. ``blk.7.attn_q.weight``). The loader throws on any other suffix,
  and when a pair is missing one side.
* Orientation: HF A [r, in] / B [out, r] are written AS-IS under the .lora_a
  /.lora_b names (row-major -> ggml dims reversed), matching llama.cpp's
  own convert_lora_to_gguf.py; the loader validates
  ``base.ne[0]==a.ne[0] && base.ne[1]==b.ne[1] && a.ne[1]==b.ne[0]``.

VERIFICATION STATUS (honest, per cardinal rule 2): container layout,
metadata keys and tensor naming were verified against the local llama.cpp
SOURCE and by round-tripping this writer through a minimal GGUF reader in
tests. Loading a produced file under a real ``llama-server -lora`` against
its base GGUF is UNVERIFIED HERE — no base GGUF of a trainable HF arch
exists locally without a download, and making one is out of scope (zero
cloud). That acceptance step is recorded in docs/ROADMAP.md for GPU
hardware.

D-12: llama.cpp's convert_lora_to_gguf.py was READ as reference; this
writer is re-implemented inline from the C++ loader contract cited above,
importing nothing.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# GGUF value types (gguf.h).
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

#: ggml tensor type id for F32 (tensor-info records use GGML types, NOT the
#: GGUF value-type ids above — verified against llama.cpp's gguf reader:
#: writing 6 here is read as q5_0 and the file fails to load).
GGML_TYPE_F32 = 0


# --------------------------------------------------------------- arch adapter
@dataclass
class ArchAdapter:
    """One reference architecture's HF-module -> GGUF-tensor-name mapping.

    WP10 pins ONE reference arch first (spike GO condition 1): llama-shaped
    dense decoders (llama/llama2/qwen3 family), whose HF module names map
    onto GGUF tensor roles with the deterministic pattern below (as used by
    llama.cpp's tensor naming). Other archs register additional adapters;
    an unmappable module name is a hard error, never a silent skip — a
    half-converted adapter is worse than a failed one.
    """

    #: value of general.architecture in the GGUF (and in the base model).
    gguf_arch: str
    #: HF leaf module names -> llama.cpp tensor role id.
    role_map: Dict[str, str]
    #: decoder layers live under this HF prefix (e.g. "model.layers").
    layer_prefix: str = "model.layers"

    def base_key(self, module_name: str) -> Optional[str]:
        """Map an HF module name (lora_A/B suffix already stripped) to its
        GGUF tensor key; None when this adapter does not know the name."""
        if not module_name.startswith(self.layer_prefix + "."):
            return None
        rest = module_name[len(self.layer_prefix) + 1:]
        parts = rest.split(".", 1)
        if len(parts) != 2 or not parts[0].isdigit():
            return None
        idx, leaf_path = parts[0], parts[1]
        # leaf path is "<proj>" or "mlp.<proj>" — the final component names it.
        leaf = leaf_path.rsplit(".", 1)[-1]
        role = self.role_map.get(leaf)
        if role is None:
            return None
        return f"blk.{idx}.{role}.weight"


#: WP10 reference arch: llama / llama2 / qwen3 (llama-shaped attention).
LLAMA_ADAPTER = ArchAdapter(
    gguf_arch="llama",
    role_map={
        "q_proj": "attn_q",
        "k_proj": "attn_k",
        "v_proj": "attn_v",
        "o_proj": "attn_output",
        "gate_proj": "ffn_gate",
        "up_proj": "ffn_up",
        "down_proj": "ffn_down",
    },
)

KNOWN_ADAPTERS: Dict[str, ArchAdapter] = {
    "llama": LLAMA_ADAPTER,
}


def adapter_for(arch: str) -> ArchAdapter:
    key = str(arch or "").strip().lower()
    if key not in KNOWN_ADAPTERS:
        raise ValueError(
            f"no GGUF-LoRA arch adapter for {arch!r}; WP10 ships "
            f"{sorted(KNOWN_ADAPTERS)} (one reference arch first — see "
            "docs/ROADMAP.md WP10). Adding an arch = registering its "
            "module->tensor role map in KNOWN_ADAPTERS.")
    return KNOWN_ADAPTERS[key]


# ------------------------------------------------------- tensor duck protocol
def _seq_shape(x: Any) -> List[int]:
    """Row-major shape of nested python sequences (list-of-lists etc.)."""
    shape: List[int] = []
    cur = x
    while isinstance(cur, (list, tuple)):
        shape.append(len(cur))
        cur = cur[0] if len(cur) else None
    return shape or [1]


def _tensor_view(t: Any) -> Tuple[List[int], bytes]:
    """(row-major shape, F32 little-endian bytes) for a tensor-like.

    Accepts torch tensors (duck-typed via .detach()/.cpu()/.numpy() — never
    imported) and numpy arrays / nested lists. Everything normalizes to
    F32 C-contiguous bytes: LoRA adapters are tiny, the copy is cheap, and
    the GGUF loader contract stays simple (F32 tensors).
    """
    if hasattr(t, "detach"):  # torch.Tensor (duck-typed — never imported)
        t = t.detach().cpu()
        if str(getattr(t, "dtype", "")) != "torch.float32":
            t = t.float()
        arr = t.numpy()
    else:
        arr = t
    if hasattr(arr, "dtype") and str(getattr(arr, "dtype", "")) != "float32":
        try:
            arr = arr.astype("float32")
        except (TypeError, ValueError):
            pass
    shape = list(getattr(arr, "shape", None) or _seq_shape(arr))
    if hasattr(arr, "tobytes"):
        data = arr.tobytes()
    else:  # plain nested lists / sequences
        import array as _array

        flat: List[float] = []

        def _flat(x: Any) -> None:
            if isinstance(x, (list, tuple)):
                for y in x:
                    _flat(y)
            else:
                flat.append(float(x))

        _flat(arr)
        data = _array.array("f", flat).tobytes()
    return shape, bytes(data)


# ------------------------------------------------------------ GGUF primitives
def _w_str(out: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    out += struct.pack("<Q", len(b)) + b


def _w_kv(out: bytearray, key: str, vtype: int, value: Any) -> None:
    _w_str(out, key)
    out += struct.pack("<I", vtype)
    if vtype == GGUF_TYPE_STRING:
        _w_str(out, str(value))
    elif vtype == GGUF_TYPE_UINT32:
        out += struct.pack("<I", int(value))
    elif vtype == GGUF_TYPE_INT32:
        out += struct.pack("<i", int(value))
    elif vtype == GGUF_TYPE_FLOAT32:
        out += struct.pack("<f", float(value))
    elif vtype == GGUF_TYPE_BOOL:
        out += struct.pack("<B", 1 if value else 0)
    else:
        raise ValueError(f"unsupported KV type {vtype}")


def _align(n: int, alignment: int) -> int:
    return n + (-n % alignment)


# --------------------------------------------------------------- the writer
@dataclass
class LoraTensor:
    """One prepared GGUF-LoRA tensor record (ggml dims = reversed storage)."""

    name: str          # e.g. "blk.0.attn_q.weight.lora_a"
    shape: List[int]   # row-major HF shape; ggml dims written reversed
    data: bytes        # F32 little-endian, C-contiguous


def map_state_to_gguf(state: Dict[str, Any], *, arch: str = "llama",
                      ) -> Tuple[List[LoraTensor], Dict[str, int]]:
    """Frozen adapter contract -> GGUF-LoRA tensor records.

    ``state``: module-name -> lora tensor, exactly what
    ``streamed.lora_linear.lora_state_dict`` emits (keys like
    ``model.layers.0.self_attn.q_proj.lora_A``). Every key must map and
    every pair must be complete — unmappable or half pairs raise. Returns
    the prepared tensors plus {target_role: rank} for metadata.
    """
    ad = adapter_for(arch)
    pairs: Dict[str, Dict[str, Tuple[List[int], bytes]]] = {}
    ranks: Dict[str, int] = {}
    for key in sorted(state):
        tensor = state[key]
        if ".lora_A" in key:
            module = key.rsplit(".lora_A", 1)[0]
            part = "A"
        elif ".lora_B" in key:
            module = key.rsplit(".lora_B", 1)[0]
            part = "B"
        else:
            raise ValueError(
                f"adapter key {key!r} is not a lora_A/lora_B tensor — the "
                "GGUF-LoRA writer consumes ONLY the frozen lora-state-dict "
                "contract (module-name -> lora tensor)")
        base = ad.base_key(module)
        if base is None:
            raise ValueError(
                f"module {module!r} has no GGUF tensor mapping for arch "
                f"{ad.gguf_arch!r}; refusing to emit a partial adapter")
        shape, data = _tensor_view(tensor)
        pairs.setdefault(base, {})[part] = (shape, data)

    tensors: List[LoraTensor] = []
    for base in sorted(pairs):
        pair = pairs[base]
        if "A" not in pair or "B" not in pair:
            side = "a" if "A" in pair else "b"
            raise ValueError(
                f"GGUF tensor {base!r} is missing its lora_{side} partner; "
                "llama.cpp refuses half pairs")
        (a_shape, a_bytes), (b_shape, b_bytes) = pair["A"], pair["B"]
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise ValueError(f"lora tensors for {base!r} must be 2-D")
        r_a, in_f = a_shape       # HF A: [r, in]
        out_f, r_b = b_shape      # HF B: [out, r]
        if r_a != r_b:
            raise ValueError(
                f"rank mismatch for {base!r}: A{a_shape} vs B{b_shape}")
        # Stored row-major as-is (llama.cpp convention); ggml dims reverse.
        tensors.append(LoraTensor(f"{base}.lora_a", a_shape, a_bytes))
        tensors.append(LoraTensor(f"{base}.lora_b", b_shape, b_bytes))
        role = base.split(".", 2)[-1].rsplit(".weight", 1)[0]
        ranks[role] = int(r_a)
    if not tensors:
        raise ValueError("adapter state is empty — nothing to convert")
    return tensors, ranks


def write_gguf_lora(state: Dict[str, Any], out_path: Any, *,
                    arch: str = "llama", alpha: float = 16.0,
                    base_arch: Optional[str] = None,
                    adapter_name: str = "toaster_loop-streamed",
                    ) -> Dict[str, Any]:
    """Write the frozen lora-state-dict as a GGUF-LoRA file. Returns summary.

    ``base_arch`` overrides general.architecture — it MUST match the base
    GGUF's own value (llama.cpp throws on mismatch); defaults to ``arch``.
    Atomic write (tmp + rename): a crashed conversion never leaves a
    half-written GGUF that would pass the magic-byte integrity check.
    """
    tensors, ranks = map_state_to_gguf(state, arch=arch)
    gguf_arch = str(base_arch or arch)

    kv = bytearray()
    _w_kv(kv, "general.architecture", GGUF_TYPE_STRING, gguf_arch)
    _w_kv(kv, "general.type", GGUF_TYPE_STRING, "adapter")  # loader-mandated
    _w_kv(kv, "general.name", GGUF_TYPE_STRING, adapter_name)
    _w_kv(kv, "adapter.type", GGUF_TYPE_STRING, "lora")     # loader-mandated
    _w_kv(kv, "adapter.lora.alpha", GGUF_TYPE_FLOAT32, float(alpha))
    _w_kv(kv, "adapter.name", GGUF_TYPE_STRING, adapter_name)
    n_kv = 6

    # Tensor-info record: name, n_dims(u32), dims(n_dims x u64 — ggml order,
    # i.e. reversed from row-major storage), type(u32), offset(u64). The
    # first pass only sizes the header region so the aligned data start is
    # known; offsets are identical in shape between passes, so the size
    # computed here stays valid for the final record below.
    info = bytearray()
    for t in tensors:
        _w_str(info, t.name)
        info += struct.pack("<I", len(t.shape))
        for d in reversed(t.shape):                      # ggml dims reversed
            info += struct.pack("<Q", int(d))
        info += struct.pack("<I", GGML_TYPE_F32)
        info += struct.pack("<Q", 0)                     # offset placeholder

    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", n_kv)

    pre_len = len(header) + len(kv) + len(info)
    data_start = _align(pre_len, DEFAULT_ALIGNMENT)

    offsets: List[int] = []
    cur = 0
    for i, t in enumerate(tensors):
        if i:  # every tensor after the first starts on an aligned boundary
            cur = _align(cur, DEFAULT_ALIGNMENT)
        offsets.append(cur)
        cur += len(t.data)

    info_final = bytearray()
    for i, t in enumerate(tensors):
        _w_str(info_final, t.name)
        info_final += struct.pack("<I", len(t.shape))
        for d in reversed(t.shape):
            info_final += struct.pack("<Q", int(d))
        info_final += struct.pack("<I", GGML_TYPE_F32)
        info_final += struct.pack("<Q", offsets[i])

    blob = bytes(header) + bytes(kv) + bytes(info_final)
    assert len(blob) <= data_start, "info sizing diverged between passes"
    pad = b"\x00" * (data_start - len(blob))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(blob)
        f.write(pad)
        written = 0
        for i, t in enumerate(tensors):
            # llama.cpp's gguf reader strictly validates that every tensor
            # starts at an aligned offset -> pad BETWEEN tensors.
            want = offsets[i]
            if written < want:
                f.write(b"\x00" * (want - written))
                written = want
            f.write(t.data)
            written += len(t.data)
    tmp.replace(out_path)

    return {
        "path": str(out_path),
        "version": GGUF_VERSION,
        "arch": gguf_arch,
        "alpha": float(alpha),
        "tensors": len(tensors),
        "ranks": ranks,
        "names": [t.name for t in tensors],
        "alignment": DEFAULT_ALIGNMENT,
        "data_offset": data_start,
    }

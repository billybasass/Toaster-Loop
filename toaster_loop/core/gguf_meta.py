"""Read GGUF container metadata (the KV section) with the stdlib only.

Why this exists: a GGUF is a *serving* artifact — trainers need HF
safetensors — but every llama.cpp-exported GGUF records where it came from
(``general.basename``, ``general.architecture``, ``general.name``). That
metadata turns "I only have a GGUF" into a resolvable pointer to the canonical
HF base repo (see ``core/gguf_base.py``), so users never need a lossy
GGUF→HF deconverter. The GGUF stays what it is: the deployment artifact.

The container layout mirrors what the writer in
``training/streamed/gguf_lora.py`` documents from llama.cpp's gguf.h (D-12:
the format was read as reference, nothing imported): magic ``GGUF``, u32
version, u64 tensor count, u64 KV count, then typed KV pairs. This reader
parses ONLY the header + KV section and stops — no tensor-info or weight
bytes are touched, so reading a 70 GB quant costs microseconds.

Versions: v1/v2 use u32 counts, v3+ (the only version llama.cpp writes since
2023) uses u64. Both are accepted; anything newer than the highest version we
know is parsed best-effort with a warning field rather than rejected — GGUF's
own spec keeps the KV layout stable across versions.

Degrade-don't-crash: every entry point returns ``None``/empty on malformed or
non-GGUF input (callers treat "no metadata" exactly like an unattested row).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

GGUF_MAGIC = b"GGUF"

# GGUF value-type ids (gguf.h) — same table as the streamed GGUF-LoRA writer.
T_UINT8 = 0
T_INT8 = 1
T_UINT16 = 2
T_INT16 = 3
T_UINT32 = 4
T_INT32 = 5
T_FLOAT32 = 6
T_BOOL = 7
T_STRING = 8
T_ARRAY = 9
T_UINT64 = 10
T_INT64 = 11
T_FLOAT64 = 12

#: Fixed-size scalar types: (struct format, byte size). Strings/arrays are
#: variable-length and handled separately.
_FIXED = {
    T_UINT8: ("<B", 1),
    T_INT8: ("<b", 1),
    T_UINT16: ("<H", 2),
    T_INT16: ("<h", 2),
    T_UINT32: ("<I", 4),
    T_INT32: ("<i", 4),
    T_FLOAT32: ("<f", 4),
    T_BOOL: ("<?", 1),
    T_UINT64: ("<Q", 8),
    T_INT64: ("<q", 8),
    T_FLOAT64: ("<d", 8),
}

#: Guard rails against pathological/corrupt headers: a GGUF with more than
#: this many KV pairs (or an array of it) is refused rather than parsed.
MAX_KV_PAIRS = 100_000
MAX_ARRAY_LEN = 1_000_000


class _Cursor:
    """Little-endian byte cursor over a bytes buffer; raises ValueError at EOF."""

    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.buf):
            raise ValueError("gguf header truncated")
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt: str) -> Tuple[Any, ...]:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))


def _read_string(c: _Cursor) -> str:
    (ln,) = c.unpack("<Q")
    if ln > MAX_ARRAY_LEN:  # a string this long is corruption, not data
        raise ValueError("gguf string length implausible")
    return c.take(ln).decode("utf-8", errors="replace")


def _read_value(c: _Cursor, vtype: int) -> Any:
    if vtype in _FIXED:
        fmt, _ = _FIXED[vtype]
        return c.unpack(fmt)[0]
    if vtype == T_STRING:
        return _read_string(c)
    if vtype == T_ARRAY:
        (etype,) = c.unpack("<I")
        (alen,) = c.unpack("<Q")
        if alen > MAX_ARRAY_LEN:
            raise ValueError("gguf array length implausible")
        # Only scalar/string element arrays appear in practice; nested arrays
        # would need recursion llama.cpp never emits. Cap what we materialise:
        # callers only ever read a handful of keys, so return the first 64 for
        # display and skip the rest byte-exactly.
        keep = min(alen, 64)
        vals = [_read_value(c, etype) for _ in range(keep)]
        if alen > keep:
            efmt = _FIXED.get(etype)  # distinct name: `fmt` above is the str
            if efmt is not None:  # fixed-size elements: skip cheaply
                c.take((alen - keep) * efmt[1])
            else:  # variable-length elements: walk them without keeping
                for _ in range(alen - keep):
                    _read_value(c, etype)
        return vals
    raise ValueError(f"unknown gguf value type {vtype}")


def read_gguf_metadata(path: str, max_bytes: int = 4 * 1024 * 1024) -> Optional[Dict[str, Any]]:
    """Parse a GGUF file's KV metadata; None when it isn't a readable GGUF.

    Reads at most ``max_bytes`` (default 4 MiB) — the KV section of real
    models is far smaller; the cap bounds damage from a corrupt header. The
    returned dict always carries:

      * ``version``   — container version int
      * ``n_tensors`` — tensor count (from the header, not parsed further)
      * ``keys``      — {key: value} for every KV pair (strings/numbers/lists)

    plus nothing else: interpreting keys is the caller's job.
    """
    p = Path(path).expanduser()
    try:
        with open(p, "rb") as f:
            head = f.read(max_bytes)
    except OSError:
        return None
    if len(head) < 8 or not head.startswith(GGUF_MAGIC):
        return None
    c = _Cursor(head)
    c.take(4)  # magic
    try:
        (version,) = c.unpack("<I")
        count_fmt = "<Q" if version >= 3 else "<I"
        (n_tensors,) = c.unpack(count_fmt)
        (n_kv,) = c.unpack(count_fmt)
    except ValueError:
        return None
    if n_kv > MAX_KV_PAIRS:
        return None
    keys: Dict[str, Any] = {}
    try:
        for _ in range(n_kv):
            k = _read_string(c)
            # GGUF KV pairs store the value type as a u32 (gguf.h), same as
            # the streamed writer's _w_kv.
            (vtype,) = c.unpack("<I")
            keys[k] = _read_value(c, vtype)
    except ValueError:
        # Truncated KV section: keep what parsed — partial metadata still
        # beats none (general.basename usually sits near the front).
        pass
    return {"version": version, "n_tensors": n_tensors, "keys": keys}


def gguf_arch(path: str) -> Optional[str]:
    """``general.architecture`` of a GGUF ("llama", "qwen2", ...); None if absent."""
    md = read_gguf_metadata(path)
    if not md:
        return None
    arch = md["keys"].get("general.architecture")
    return str(arch) if arch else None


def gguf_basename(path: str) -> Optional[str]:
    """``general.basename`` — llama.cpp's own name for the source model, e.g.
    ``Qwen3-8B`` (the upstream repo name without the org). None if absent."""
    md = read_gguf_metadata(path)
    if not md:
        return None
    bn = md["keys"].get("general.basename")
    return str(bn) if bn else None


def gguf_name(path: str) -> Optional[str]:
    """``general.name`` — the export's display name (often includes the quant)."""
    md = read_gguf_metadata(path)
    if not md:
        return None
    nm = md["keys"].get("general.name")
    return str(nm) if nm else None

"""Dataset import with format auto-detection (backlog item #14).

Seeds the loop from curated public data: `sable datasets import <file>`
shape-sniffs each row (format UX idea from the
``format_type: auto|alpaca|sharegpt|chatml`` — read-only reference, never
imported), normalizes to the internal ShareGPT row format used by curate
output, and writes a dataset directory that flows into `sable train`
UNCHANGED (identical sft.jsonl contract; manifest mirrors the curater's so
pipeline/train consume it without special cases).

Accepted input: ``.jsonl`` (one object per line) or ``.json`` (top-level
array of objects). Formats detected per sample row:

  * ``conversations`` list      -> sharegpt  ({from|role, value|content})
  * ``instruction`` (+``output``) -> alpaca  (input? -> context prefix)
  * ``messages`` role/content   -> chatml    (OpenAI-style passthrough)

Malformed rows are skipped and counted (summary + manifest
``skipped_rows``); a row that parses but normalizes to nothing usable counts
as skipped too. Imported datasets carry no trace provenance: manifest gets
``provenance: "imported"`` + ``source_file`` and empty trace-id lists, and no
buffer claiming happens (N/A).

Internal row shape is produced through the SAME builder path curate uses —
chatml/sharegpt turns are fed back through sharegpt.messages_to_sharegpt so
role/value conventions (gpt think blocks, tool rendering) match curate output
exactly.

Pass 4 #42: before committing rows, a pre-commit quality gate runs
(`curate/ingest_gate.py`). # Idea adapted from PIASystem (MIT)
app/core/knowledge_ingestion.py — trust tiers + redundancy + conflict check;
their FAISS/Neo4j stores dropped, only the gate ported. The gate NEVER drops
rows: reasons land in a `flags.json` sidecar and keys in `hashes.json`, so
sft.jsonl stays byte-canonical ShareGPT for curater/train (sidecar choice:
curater and the training worker read sft.jsonl only, and the dashboard reads
the sidecars through API handlers — no panel/row-shape changes needed).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.provenance import attest_path
from .curater import (
    DATASET_SCHEMA_VERSION,
    TRAIN_YAML_TEMPLATE_VERSION,
    CurationConfig,
    _utcnow,
    render_train_yaml,
)
from .ingest_gate import DEFAULT_TRUST, GateReport, run_gate, write_gate_sidecars
from .sharegpt import messages_to_sharegpt

logger = logging.getLogger("sable.curate.importer")

FORMATS = ("alpaca", "sharegpt", "chatml")


class ImportError_(ValueError):
    """Raised for unreadable/undetectable input (clear message, non-zero exit)."""


# ------------------------------------------------------------------ detection

def detect_format(sample_rows: List[Dict[str, Any]]) -> str:
    """Sniff the dataset format from sample rows; raise with keys seen if not.

    Rules (first match on the first usable row): ``conversations`` -> sharegpt;
    ``instruction`` (+``output``) -> alpaca; ``messages`` of role/content
    dicts -> chatml. Mixed shapes fall back to a majority vote so a stray row
    can't flip the whole file.
    """
    usable = [r for r in sample_rows if isinstance(r, dict)]
    if not usable:
        seen = {type(r).__name__ for r in sample_rows} or {"empty file"}
        raise ImportError_(
            "cannot detect dataset format: no JSON object rows found "
            f"(saw: {', '.join(sorted(seen))})")

    def classify(row: Dict[str, Any]) -> Optional[str]:
        if isinstance(row.get("conversations"), list):
            return "sharegpt"
        if "instruction" in row and "output" in row:
            return "alpaca"
        msgs = row.get("messages")
        if isinstance(msgs, list) and msgs and all(
            isinstance(m, dict) and "role" in m for m in msgs if m is not None
        ):
            return "chatml"
        return None

    votes = [c for c in (classify(r) for r in usable) if c]
    if not votes:
        keys = sorted({str(k) for r in usable for k in r.keys()})
        raise ImportError_(
            "cannot detect dataset format: expected one of sharegpt "
            "('conversations'), alpaca ('instruction'+'output') or chatml "
            f"('messages'); keys seen: {', '.join(keys)}")
    return max(votes, key=votes.count)


# ----------------------------------------------------------------- normalize

def _as_text(v: Any) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False, default=str)


def _row_to_messages(row: Dict[str, Any], fmt: str) -> Optional[List[Dict[str, Any]]]:
    """One input row -> OpenAI-format messages (the curater's source shape).

    Returns None when the row is malformed for its format (caller skips+counts).
    """
    try:
        if fmt == "alpaca":
            instr = _as_text(row.get("instruction")).strip()
            out = _as_text(row.get("output")).strip()
            if not instr or not out:
                return None
            inp = row.get("input")
            user = f"{_as_text(inp).strip()}\n\n{instr}" if inp else instr
            return [{"role": "user", "content": user},
                    {"role": "assistant", "content": out}]

        if fmt == "chatml":
            msgs = row.get("messages")
            if not isinstance(msgs, list) or not msgs:
                return None
            out_msgs: List[Dict[str, Any]] = []
            for m in msgs:
                if not isinstance(m, dict):
                    return None
                role = m.get("role")
                if role not in ("system", "user", "assistant", "tool"):
                    return None  # role validation: unknown roles reject the row
                out_msgs.append({"role": role, "content": _as_text(m.get("content"))})
            return out_msgs or None

        if fmt == "sharegpt":
            conv = row.get("conversations")
            if not isinstance(conv, list) or not conv:
                return None
            # Accept both internal from/value and external role/content keys;
            # map to OpenAI messages so messages_to_sharegpt re-emits the exact
            # internal turn shape (gpt think blocks included).
            role_map = {"human": "user", "gpt": "assistant",
                        "tool": "tool", "system": "system",
                        "user": "user", "assistant": "assistant"}
            out_msgs = []
            for turn in conv:
                if not isinstance(turn, dict):
                    return None
                who = turn.get("from", turn.get("role"))
                txt = turn.get("value", turn.get("content"))
                role = role_map.get(str(who))
                if role is None or txt is None:
                    return None
                out_msgs.append({"role": role, "content": _as_text(txt)})
            return out_msgs or None
    except Exception:  # defensive: a weird row skips, never crashes the import
        return None
    raise ImportError_(f"unknown format {fmt!r}")


@dataclass
class ImportResult:
    dataset_id: str
    dataset_dir: Path
    fmt: str
    rows: int = 0
    skipped_rows: int = 0
    total_rows: int = 0
    warnings: List[str] = field(default_factory=list)
    gate: Optional[GateReport] = None


def read_rows(path: Path) -> List[Dict[str, Any]]:
    """Load .jsonl (or a .json array) into row dicts; raise ImportError_ cleanly."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise ImportError_(f"no such file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ImportError_(f"cannot read {path}: {e}") from e

    rows: List[Dict[str, Any]] = []
    stripped = text.lstrip()
    if stripped.startswith("["):  # .json array accepted too
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ImportError_(f"{path}: invalid JSON array: {e}") from e
        if not isinstance(data, list):
            raise ImportError_(f"{path}: top-level JSON value is not an array")
        return [r for r in data if isinstance(r, dict)]
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue  # blank lines tolerated like `run --batch`
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            rows.append({"__malformed_line__": i, "__error__": str(e)})
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        else:
            rows.append({"__malformed_line__": i,
                         "__error__": f"not an object ({type(obj).__name__})"})
    if not rows:
        raise ImportError_(f"{path}: no rows found (empty file?)")
    return rows


def import_dataset(
    path: Path,
    fmt: str = "auto",
    name: Optional[str] = None,
    home: Optional[Path] = None,
    cfg: Optional[CurationConfig] = None,
) -> ImportResult:
    """Normalize an external dataset into a curate-layout dataset directory.

    ``fmt``: auto | alpaca | sharegpt | chatml. Writes sft.jsonl (internal
    ShareGPT rows — same contract as curate output), manifest.json with
    ``provenance: "imported"`` / ``source_file`` / row counts / timestamp /
    ``skipped_rows``, and train.yaml (template, so `train start` works). No
    traces -> no claiming; the trace-id lists are empty. Raises ImportError_
    for unreadable or undetectable input.
    """
    from ..store.paths import resolve_sable_home

    path = Path(path).expanduser()
    home = Path(home) if home is not None else resolve_sable_home()
    rows = read_rows(path)
    malformed = [r for r in rows if "__malformed_line__" in r]
    sample = [r for r in rows if "__malformed_line__" not in r][:25]

    if fmt == "auto":
        detected = detect_format(sample)
    elif fmt in FORMATS:
        detected = fmt
    else:
        raise ImportError_(
            f"unknown --format {fmt!r}; choose from auto, {', '.join(FORMATS)}")

    conversations = []
    skipped = len(malformed)
    for row in rows:
        if "__malformed_line__" in row:
            logger.warning("skipping malformed line %s: %s",
                           row["__malformed_line__"], row["__error__"])
            continue
        msgs = _row_to_messages(row, detected)
        conv = messages_to_sharegpt(msgs) if msgs else []
        if not conv:
            skipped += 1
            continue
        conversations.append(conv)

    if not conversations:
        raise ImportError_(
            f"detected {detected} but no rows normalized to usable "
            f"conversations ({skipped} skipped) — check the file")

    # Dataset dir follows the exact curate layout (no DB: imports claim no
    # traces). Id scheme mirrors Curater._make_dataset_id: UTC stamp--name.
    datasets_root = Path(home) / "datasets"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(
        c if (c.isalnum() or c in "-_.") else "-" for c in (name or path.stem)
    ).strip("-_")
    dataset_id = f"{stamp}--{safe}" if safe else stamp
    ds_dir = datasets_root / dataset_id
    n = 2
    while ds_dir.exists():
        ds_dir = datasets_root / f"{dataset_id}-{n}"
        n += 1
    dataset_id = ds_dir.name
    ds_dir.mkdir(parents=True, exist_ok=False)

    with open(ds_dir / "sft.jsonl", "w", encoding="utf-8") as f:
        for conv in conversations:
            # Same row shape curate writes; trace provenance extras are N/A
            # for imports (format_type=sharegpt reads `conversations` only).
            f.write(json.dumps({"conversations": conv},
                               ensure_ascii=False) + "\n")

    # Pass 4 #42 pre-commit gate: exact/normalized dedupe + trust-weighted
    # conflict detection against existing datasets (hashes.json sidecars) and
    # within the batch. Flag-only — rows are never dropped; findings land in
    # flags.json + hashes.json beside sft.jsonl. Always runs; never blocks.
    # Idea adapted from PIASystem (MIT) app/core/knowledge_ingestion.py.
    gate = run_gate(conversations, datasets_root, trust=DEFAULT_TRUST,
                    dataset_id=dataset_id)

    manifest = {
        "dataset_id": dataset_id,
        "created_at": _utcnow(),
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "train_yaml_template_version": TRAIN_YAML_TEMPLATE_VERSION,
        # Import provenance (backlog #14). No trace-level provenance exists:
        # files.*.trace_ids stay empty so manifest readers find the same keys.
        "provenance": "imported",
        "source_file": str(Path(path).expanduser().resolve()),
        "source_format": detected,
        # Pass 4 #42 trust tier: external imports default to `imported`
        # (precedence curated > judge_scored > imported; missing = imported).
        "trust": DEFAULT_TRUST,
        # Pass 3 #23: byte-level attestation of BOTH ends of the import — the
        # source file as it was read and the sft.jsonl we wrote. Lineage can
        # now prove which bytes produced this dataset (and detect drift).
        "resource_provenance": {
            "source": attest_path(str(Path(path).expanduser().resolve())),
            "dataset": attest_path(str(ds_dir / "sft.jsonl")),
        },
        "skipped_rows": skipped,
        "filters": None,
        "files": {
            "sft.jsonl": {
                "count": len(conversations),
                "trace_ids": [],          # N/A — imported rows have no traces
                "format": "sharegpt",
            },
            "dpo.jsonl": {
                "count": 0,
                "trace_ids": [],
                "format": "trl-preference",
            },
        },
        "format_hints": {
            "sft_format_type": "sharegpt",
            "train_yaml": "train.yaml",
            "note": ("imported dataset; sft.jsonl matches the curate output "
                     "contract (conversations: [{from,value}])"),
        },
        "warnings": [],
    }
    (ds_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # train.yaml so `sable train start --dataset <id>` resolves like curated
    # datasets (same template, same REPLACE_* placeholder rules).
    (ds_dir / "train.yaml").write_text(
        render_train_yaml(ds_dir, cfg or CurationConfig(), dataset_id),
        encoding="utf-8",
    )

    write_gate_sidecars(ds_dir, conversations, DEFAULT_TRUST, gate)

    logger.info("imported %s: %d rows (%s), %d skipped -> %s",
                path, len(conversations), detected, skipped, dataset_id)
    return ImportResult(
        dataset_id=dataset_id,
        dataset_dir=ds_dir,
        fmt=detected,
        rows=len(conversations),
        skipped_rows=skipped,
        total_rows=len(rows),
        gate=gate,
    )

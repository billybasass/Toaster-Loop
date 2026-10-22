"""Dataset diff between two versions (review idea #6).

Compares two dataset directories under ``<home>/datasets/`` using what the
manifests already carry plus content hashes of the shipped files:

- trace-ID sets per shared file (``files.<name>.trace_ids``): added / removed
  / unchanged IDs (counts + capped, deterministically sorted samples).
- per-file hash equality for files present in both datasets (sha256 over the
  raw bytes; sft.jsonl/dpo.jsonl additionally get a row-level content diff —
  rows keyed by their trace_id when present, else by row hash).
- count deltas (sft/dpo), ``source`` field comparison (curate vs rft), and
  created_at delta in seconds.

Ordering is deterministic everywhere (sorted sets). Missing datasets ->
``DatasetDiffError(404)``; malformed manifests / bad params -> code 400. The
dashboard route and the CLI both call :func:`diff_datasets`, so semantics can
never diverge between surfaces.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Cap on ID samples embedded in the response (full sets stay derivable from
# the manifests; scrapes/UIs only need a glance).
ID_SAMPLE_CAP = 50


class DatasetDiffError(Exception):
    """Bad request (400) or missing dataset (404) with an honest message."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _load_manifest(ds_dir: Path, ds_id: str) -> Dict[str, Any]:
    """Manifest for one dataset id; 404 when absent, 400 when malformed.

    A directory without manifest.json is not a dataset (list_datasets skips
    it), so that is a 404 like any unknown id — matching h_dataset_get's
    honest-missing contract. Unparseable/non-object manifests are 400: the
    dataset exists but cannot be diffed.
    """
    mf = ds_dir / "manifest.json"
    if not mf.is_file():
        raise DatasetDiffError(404, f"no dataset with id {ds_id}")
    try:
        manifest = json.loads(mf.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise DatasetDiffError(
            400, f"dataset {ds_id}: manifest.json unreadable or malformed") \
        from None
    if not isinstance(manifest, dict):
        raise DatasetDiffError(400, f"dataset {ds_id}: manifest is not an object")
    return manifest


def _trace_ids(manifest: Dict[str, Any], fname: str) -> Optional[Set[str]]:
    """trace_ids of one manifest file entry; None when the entry is absent."""
    files = manifest.get("files")
    if not isinstance(files, dict):
        return None
    entry = files.get(fname)
    if not isinstance(entry, dict):
        return None
    ids = entry.get("trace_ids")
    if not isinstance(ids, list):
        return None
    return {str(t) for t in ids}


def _file_count(manifest: Dict[str, Any], fname: str) -> Optional[int]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        return None
    entry = files.get(fname)
    if not isinstance(entry, dict):
        return None
    c = entry.get("count")
    return int(c) if isinstance(c, (int, float)) else None


def _file_hash(path: Path) -> Optional[str]:
    """sha256 over the raw file bytes; None when the file does not exist."""
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_keys(path: Path) -> Optional[List[str]]:
    """Per-row content keys for a jsonl file.

    Key = trace_id when the row carries one (stable across re-serialization),
    else sha256 of the raw line. Unparseable lines fall back to the line hash.
    None when the file is missing.
    """
    if not path.is_file():
        return None
    keys: List[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                keys.append(hashlib.sha256(line.encode()).hexdigest())
                continue
            tid = row.get("trace_id") if isinstance(row, dict) else None
            if tid:
                keys.append(f"tid:{tid}")
            else:
                keys.append(hashlib.sha256(line.encode()).hexdigest())
    return keys


def _set_diff(a: Set[str], b: Set[str]) -> Dict[str, Any]:
    added = sorted(b - a)
    removed = sorted(a - b)
    unchanged = sorted(a & b)
    return {
        "added_count": len(added),
        "removed_count": len(removed),
        "unchanged_count": len(unchanged),
        "added_sample": added[:ID_SAMPLE_CAP],
        "removed_sample": removed[:ID_SAMPLE_CAP],
        "unchanged_sample": unchanged[:ID_SAMPLE_CAP],
    }


def _created_delta_s(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[float]:
    def parse(m: Dict[str, Any]) -> Optional[datetime]:
        try:
            dt = datetime.fromisoformat(str(m.get("created_at")))
        except (TypeError, ValueError):
            return None
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

    ta, tb = parse(a), parse(b)
    if ta is None or tb is None:
        return None
    return round((tb - ta).total_seconds(), 3)


def diff_datasets(datasets_root: Path, a_id: str, b_id: str) -> Dict[str, Any]:
    """Diff two datasets by id. Raises DatasetDiffError (400/404).

    Deterministic: every collection in the payload is sorted; comparing
    identical datasets yields all-unchanged with hash_equal true per file.
    """
    root = Path(datasets_root)
    if not a_id or not b_id:
        raise DatasetDiffError(400, "both dataset ids (a and b) are required")

    a_dir, b_dir = root / a_id, root / b_id
    ma = _load_manifest(a_dir, a_id)
    mb = _load_manifest(b_dir, b_id)

    # ---- trace-id set diffs over the union of known file names -----------
    file_names = ["sft.jsonl", "dpo.jsonl"]

    def _extra_names(m: Dict[str, Any]) -> List[str]:
        files = m.get("files")
        if not isinstance(files, dict):
            return []
        return sorted(n for n in files
                      if isinstance(files.get(n), dict) and n not in file_names)

    for extra in _extra_names(ma) + _extra_names(mb):
        if extra not in file_names:
            file_names.append(extra)

    trace_diffs: Dict[str, Any] = {}
    for fname in file_names:
        ta, tb = _trace_ids(ma, fname), _trace_ids(mb, fname)
        if ta is None and tb is None:
            continue  # file not in either manifest -> nothing to say
        entry = {
            "in_a": ta is not None,
            "in_b": tb is not None,
        }
        entry.update(_set_diff(ta or set(), tb or set()))
        trace_diffs[fname] = entry

    # ---- per-file content hash verdicts -----------------------------------
    files_out: Dict[str, Any] = {}
    for fname in file_names:
        ha = _file_hash(a_dir / fname)
        hb = _file_hash(b_dir / fname)
        if ha is None and hb is None:
            continue  # file absent on disk in both datasets
        entry: Dict[str, Any] = {
            "present_a": ha is not None,
            "present_b": hb is not None,
            "sha256_a": ha,
            "sha256_b": hb,
        }
        if ha is not None and hb is not None:
            entry["hash_equal"] = ha == hb
        else:
            entry["hash_equal"] = None  # only one side has the file
        files_out[fname] = entry

    # ---- row-level content diff for shared jsonl files --------------------
    rows_diff: Dict[str, Any] = {}
    for fname in file_names:
        ka, kb = _row_keys(a_dir / fname), _row_keys(b_dir / fname)
        if ka is None or kb is None:
            continue
        sa, sb = set(ka), set(kb)
        rows_diff[fname] = {
            "rows_a": len(ka),
            "rows_b": len(kb),
            "rows_added": len(sb - sa),
            "rows_removed": len(sa - sb),
            "rows_unchanged": len(sa & sb),
            # True when the multiset of row keys matches exactly (same rows,
            # possibly reordered lines still count as content-equal).
            "content_equal": _multiset(ka) == _multiset(kb),
        }

    # ---- metadata comparisons ---------------------------------------------
    src_a = ma.get("source") if isinstance(ma.get("source"), str) else None
    src_b = mb.get("source") if isinstance(mb.get("source"), str) else None
    sft_a, sft_b = _file_count(ma, "sft.jsonl"), _file_count(mb, "sft.jsonl")
    dpo_a, dpo_b = _file_count(ma, "dpo.jsonl"), _file_count(mb, "dpo.jsonl")

    # identical: every file present on either side exists on both with equal
    # hashes (no shared-file pairs at all -> honest False), and no trace id
    # was added or removed in any manifest file entry. Same-id compares are
    # trivially identical.
    files_equal = bool(files_out) and all(
        f["present_a"] and f["present_b"] and f["hash_equal"] is True
        for f in files_out.values())
    traces_equal = not any(d["added_count"] or d["removed_count"]
                           for d in trace_diffs.values())
    identical = (files_equal and traces_equal) or a_id == b_id

    return {
        "a": {"dataset_id": a_id, "created_at": ma.get("created_at"),
              "source": src_a},
        "b": {"dataset_id": b_id, "created_at": mb.get("created_at"),
              "source": src_b},
        "identical": identical,
        "trace_diff": trace_diffs,
        "files": files_out,
        "rows": rows_diff,
        "counts": {
            "sft_a": sft_a, "sft_b": sft_b,
            "sft_delta": (sft_b - sft_a)
            if sft_a is not None and sft_b is not None else None,
            "dpo_a": dpo_a, "dpo_b": dpo_b,
            "dpo_delta": (dpo_b - dpo_a)
            if dpo_a is not None and dpo_b is not None else None,
        },
        "source_equal": src_a == src_b,
        "created_at_delta_s": _created_delta_s(ma, mb),
    }


def _multiset(items: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out

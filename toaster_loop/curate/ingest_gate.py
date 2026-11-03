"""Dataset ingestion quality gate: trust tiers, dedupe, conflict check.

Idea adapted from PIASystem (MIT) app/core/knowledge_ingestion.py — the
pre-commit gate of their ingestion pipeline: source trust tiers, redundancy
check and conflict detection before rows land in a dataset. Their FAISS/Neo4j
stores are explicitly NOT taken (backlog #42); only the
*gate* survives, stdlib-only (hashlib, difflib, json).

The gate NEVER drops or mutates rows — flagged rows are kept and visible:

  * ``exact_duplicate:<ds>:<line>``  content hash seen before (existing
    datasets via hashes.json sidecars, or within the batch)
  * ``near_duplicate:<ds>:<line>``   normalized collision (lowercase,
    whitespace collapsed, punctuation stripped — comparison key only; true
    fuzzy MinHash is explicitly out of scope, future work)
  * ``conflicts_with:<ds>:<line>``   same normalized input, outputs differ
    beyond a small edit-distance ratio AND the other side has strictly higher
    trust
  * ``conflict_note:<ds>:<line>``    same shape, but the other side is
    equal/lower trust — informational only

Trust tiers on manifests: ``trust`` ∈ {curated, judge_scored, imported} with
precedence curated > judge_scored > imported. Missing trust on old manifests =
imported. Internal curate output writes ``curated``, or ``judge_scored`` when
a judge verdict (feedback source="judge") decided the effective score of any
selected trace — honest per-row flow: judge verdicts only enter selection via
load_trace's user > judge > heuristic precedence, so "any selected row had a
judge row" is exactly when judge scoring mattered.

Sidecars live next to the dataset file and keep sft.jsonl byte-canonical
ShareGPT for curater/train consumers (sidecar choice documented in importer
docstring; the dashboard reads them through /api/datasets handlers, so no
panel-side file access is needed):

  hashes.json  {"sidecar_version":1,"trust","row_count",
                "rows":[{"line","exact","norm","in_key","out"},...]} —
    per-row comparison keys. ``out`` is normalized + truncated to
    _MAX_STORED_OUTPUT chars; edit-distance runs on that stored view. Built
    at import time; ``toaster_loop datasets lint`` regenerates it; imports rebuild
    missing/stale sidecars of OTHER datasets on demand (cheap, O(rows) once
    per dataset — cross-dataset scans never re-read sft.jsonl per row).
  flags.json   {"sidecar_version":1,"flags":{"<line>":[reasons]}} — written
    even when empty: its presence marks the dataset as linted. Absence means
    "unlinted" (old datasets) and the panel says so honestly.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("toaster_loop.curate.ingest_gate")

TRUST_TIERS = {"imported": 1, "judge_scored": 2, "curated": 3}
DEFAULT_TRUST = "imported"
HASHES_SIDECAR_VERSION = 1
FLAGS_SIDECAR_VERSION = 1
#: same input key but output similarity below this counts as a conflict.
_CONFLICT_SIMILARITY = 0.9
#: normalized outputs are stored (and compared) capped, for cheap sidecars.
_MAX_STORED_OUTPUT = 4000

_PUNCT_TABLE = str.maketrans({c: None for c in string.punctuation})


def trust_rank(trust: Any) -> int:
    """Precedence curated > judge_scored > imported; unknown/missing = imported."""
    return TRUST_TIERS.get(str(trust), TRUST_TIERS[DEFAULT_TRUST])


def normalize_compare_key(text: str) -> str:
    """Comparison key only: lowercase, strip punctuation, collapse whitespace."""
    return " ".join(text.lower().translate(_PUNCT_TABLE).split())


def _turn_values(conv: List[Dict[str, Any]],
                 roles: Tuple[str, ...]) -> str:
    return "\n".join(str(t.get("value", "")) for t in conv
                     if isinstance(t, dict) and t.get("from") in roles)


def row_keys(conv: List[Dict[str, Any]]) -> Dict[str, str]:
    """Comparison keys for one canonical ShareGPT row.

    ``exact`` hashes the exact serialized row (same bytes curate/import
    write); ``norm`` is the normalized full-conversation sha; ``in_key``
    hashes the normalized system+human turns (empty string when a row has no
    input side — such rows never participate in conflict checks); ``out`` is
    the normalized gpt+tool text truncated for cheap storage.
    """
    exact = hashlib.sha256(
        json.dumps({"conversations": conv}, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    norm = hashlib.sha256(
        normalize_compare_key(_turn_values(conv, ("system", "human", "gpt",
                                                  "tool"))).encode("utf-8")
    ).hexdigest()
    in_text = normalize_compare_key(_turn_values(conv, ("system", "human")))
    out_norm = normalize_compare_key(_turn_values(conv, ("gpt", "tool")))
    return {
        "exact": exact,
        "norm": norm,
        "in_key": hashlib.sha256(in_text.encode("utf-8")).hexdigest()
        if in_text else "",
        "out": out_norm[:_MAX_STORED_OUTPUT],
    }


# ---------------------------------------------------------------- sidecars

def read_sft_conversations(ds_dir: Path) -> List[List[Dict[str, Any]]]:
    """Conversations from a dataset's sft.jsonl (unreadable rows -> [])."""
    path = ds_dir / "sft.jsonl"
    if not path.is_file():
        return []
    out: List[List[Dict[str, Any]]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            out.append([])
            continue
        conv = row.get("conversations") if isinstance(row, dict) else None
        out.append(conv if isinstance(conv, list) else [])
    return out


def dataset_trust(ds_dir: Path) -> str:
    """Manifest trust field; missing/old manifests default to `imported`."""
    try:
        mf = json.loads((ds_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_TRUST
    t = mf.get("trust") if isinstance(mf, dict) else None
    return str(t) if t in TRUST_TIERS else DEFAULT_TRUST


def build_hashes_sidecar(ds_dir: Path,
                         trust: Optional[str] = None) -> Dict[str, Any]:
    """(Re)build hashes.json for one dataset from its sft.jsonl. Writes to disk."""
    if trust is None:
        trust = dataset_trust(ds_dir)
    rows = []
    for i, conv in enumerate(read_sft_conversations(ds_dir)):
        rows.append({"line": i, **row_keys(conv)})
    sidecar = {
        "sidecar_version": HASHES_SIDECAR_VERSION,
        "trust": trust,
        "row_count": len(rows),
        "rows": rows,
    }
    (ds_dir / "hashes.json").write_text(
        json.dumps(sidecar, ensure_ascii=False) + "\n", encoding="utf-8")
    return sidecar


def load_hashes_sidecar(ds_dir: Path) -> Optional[Dict[str, Any]]:
    """Read hashes.json; None when absent/unreadable/stale.

    Staleness check: row_count must equal the sft.jsonl non-blank line count,
    so an edited dataset never gets checked against outdated keys.
    """
    path = ds_dir / "hashes.json"
    if not path.is_file():
        return None
    try:
        sc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(sc, dict) or not isinstance(sc.get("rows"), list):
        return None
    sft = ds_dir / "sft.jsonl"
    n_lines = -1
    if sft.is_file():
        try:
            n_lines = sum(1 for l in sft.read_text(encoding="utf-8")
                          .splitlines() if l.strip())
        except (OSError, UnicodeDecodeError):
            n_lines = -1
    if sc.get("row_count") != n_lines:
        return None  # stale sidecar -> caller rebuilds
    if "trust" not in sc:
        sc["trust"] = dataset_trust(ds_dir)
    return sc


def load_flags_sidecar(ds_dir: Path) -> Optional[Dict[str, Any]]:
    """Read flags.json; None when absent/unreadable (dataset unlinted)."""
    path = ds_dir / "flags.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_existing_sidecars(datasets_root: Path,
                            skip_dataset_id: Optional[str]) -> List[Dict[str, Any]]:
    """Sidecars of every dataset on disk (self-healing missing/stale ones).

    Cross-dataset scans use these sidecars only — no per-row full rescans of
    sft.jsonl files (cheapness guardrail). A dataset without a valid sidecar
    gets one built once, here.
    """
    out: List[Dict[str, Any]] = []
    if not datasets_root.is_dir():
        return out
    for d in sorted(datasets_root.iterdir()):
        if not d.is_dir() or d.name == skip_dataset_id:
            continue
        if not (d / "manifest.json").is_file():
            continue  # partial/foreign dirs are not datasets
        sc = load_hashes_sidecar(d)
        if sc is None:
            sc = build_hashes_sidecar(d)
        sc = dict(sc)
        # The manifest is the trust authority — a sidecar built earlier may
        # carry a stale tier (manifest edited/rebuilt since).
        sc["trust"] = dataset_trust(d)
        sc["dataset_id"] = d.name
        out.append(sc)
    return out


# ------------------------------------------------------------------- gate

@dataclass
class GateReport:
    """Outcome of one pre-commit gate pass (never blocks; flag-only)."""

    n_rows: int = 0
    exact_dupes: int = 0
    near_dupes: int = 0
    conflicts: int = 0
    notes: int = 0
    by_trust: Dict[str, int] = field(default_factory=dict)
    flags: Dict[int, List[str]] = field(default_factory=dict)
    details: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {"n_rows": self.n_rows, "exact_dupes": self.exact_dupes,
                "near_dupes": self.near_dupes, "conflicts": self.conflicts,
                "notes": self.notes, "by_trust": dict(self.by_trust)}


def run_gate(conversations: List[List[Dict[str, Any]]], datasets_root: Path,
             trust: str = DEFAULT_TRUST,
             dataset_id: Optional[str] = None) -> GateReport:
    """Pre-commit quality gate over canonical rows (pass 4 #42).

    Checks each row against every row already present in existing datasets
    (via hashes.json sidecars) and within the batch: exact content hash
    dedupe, normalized-key near-dedupe, and same-input/different-output
    conflicts resolved by trust precedence (curated > judge_scored >
    imported). Rows are NEVER dropped — reasons land in `flags` keyed by
    output line index. O(rows) hashing + dict lookups; cross-dataset cost is
    one sidecar read per dataset.
    """
    rep = GateReport(n_rows=len(conversations))
    my_rank = trust_rank(trust)
    exact_map: Dict[str, Tuple[str, int]] = {}
    norm_map: Dict[str, Tuple[str, int]] = {}
    input_map: Dict[str, List[Tuple[str, int, str]]] = {}

    for sc in _load_existing_sidecars(datasets_root, dataset_id):
        t_rank = trust_rank(sc.get("trust"))
        tier = str(sc.get("trust", DEFAULT_TRUST))
        rep.by_trust[tier] = rep.by_trust.get(tier, 0) + len(sc["rows"])
        for r in sc["rows"]:
            ref = f"{sc['dataset_id']}:{r.get('line', '?')}"
            if r.get("exact"):
                exact_map.setdefault(r["exact"], (ref, t_rank))
            if r.get("norm"):
                norm_map.setdefault(r["norm"], (ref, t_rank))
            if r.get("in_key"):
                input_map.setdefault(r["in_key"], []).append(
                    (ref, t_rank, r.get("out", "")))

    rep.by_trust[trust] = rep.by_trust.get(trust, 0) + len(conversations)

    for idx, conv in enumerate(conversations):
        k = row_keys(conv)
        row_flags: List[str] = []

        # 1. exact dedupe (existing sidecars first, then within-batch)
        hit = exact_map.get(k["exact"])
        if hit is None:
            exact_map[k["exact"]] = (f"batch:{idx}", my_rank)
        else:
            row_flags.append(f"exact_duplicate:{hit[0]}")
        if any(f.startswith("exact_duplicate:") for f in row_flags):
            rep.exact_dupes += 1

        # 2. normalized near-dedupe (skip when the exact key already hit —
        # same relationship, don't double-flag)
        if not row_flags:
            hitn = norm_map.get(k["norm"])
            if hitn is None:
                norm_map[k["norm"]] = (f"batch:{idx}", my_rank)
            else:
                row_flags.append(f"near_duplicate:{hitn[0]}")
                rep.near_dupes += 1

        # 3. conflict check: same normalized input, outputs differ beyond the
        # small edit-distance ratio. Strictly higher-trust other side ->
        # conflicts_with; equal/lower trust -> informational conflict_note.
        if k["in_key"]:
            for ref2, rank2, out2 in list(input_map.get(k["in_key"], [])):
                if difflib.SequenceMatcher(
                        None, k["out"], out2).ratio() >= _CONFLICT_SIMILARITY:
                    continue  # outputs agree (or near-agree) — not a conflict
                if rank2 > my_rank:
                    row_flags.append(f"conflicts_with:{ref2}")
                else:
                    row_flags.append(f"conflict_note:{ref2}")
            if any(f.startswith("conflicts_with:") for f in row_flags):
                rep.conflicts += 1
            elif any(f.startswith("conflict_note:") for f in row_flags):
                rep.notes += 1
            input_map.setdefault(k["in_key"], []).append(
                (f"batch:{idx}", my_rank, k["out"]))

        if row_flags:
            rep.flags[idx] = row_flags
            rep.details.extend(f"sft.jsonl:{idx} {f}" for f in row_flags)
    return rep


def write_gate_sidecars(ds_dir: Path,
                        conversations: List[List[Dict[str, Any]]],
                        trust: str, report: GateReport) -> None:
    """Persist hashes.json (fresh keys for this dataset) + flags.json.

    flags.json is written even when empty — its presence marks the dataset as
    linted; absence means "unlinted" (old datasets). sft.jsonl itself is
    never touched, so curater/train consumers stay byte-compatible.
    """
    build_hashes_sidecar(ds_dir, trust=trust)
    flags_out = {"sidecar_version": FLAGS_SIDECAR_VERSION,
                 "flags": {str(i): reasons for i, reasons in
                           sorted(report.flags.items())}}
    (ds_dir / "flags.json").write_text(
        json.dumps(flags_out, ensure_ascii=False) + "\n", encoding="utf-8")


def lint_datasets(datasets_root: Path,
                  name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Re-run the gate over existing datasets; refresh sidecars; report.

    Two deterministic passes so the result is idempotent regardless of
    iteration order: (1) rebuild every in-scope dataset's hashes.json from
    its own sft.jsonl + manifest trust; (2) per dataset, run the gate against
    the refreshed sidecars of all OTHER datasets (its own rows come straight
    from its file) and rewrite its flags.json. Nothing is dropped or
    reordered — lint is flag-only like the import-time gate.
    """
    if not datasets_root.is_dir():
        return []
    dirs = [d for d in sorted(datasets_root.iterdir())
            if d.is_dir() and (d / "manifest.json").is_file()]
    if name:
        dirs = [d for d in dirs if name == d.name or name in d.name]
    reports: List[Dict[str, Any]] = []
    for d in dirs:  # pass 1: refresh hash sidecars everywhere in scope
        build_hashes_sidecar(d)
    for d in dirs:  # pass 2: flags against everyone else's refreshed sidecars
        trust = dataset_trust(d)
        convs = read_sft_conversations(d)
        rep = run_gate(convs, datasets_root, trust=trust, dataset_id=d.name)
        write_gate_sidecars(d, convs, trust, rep)  # rewrites flags.json only
        entry: Dict[str, Any] = {"dataset_id": d.name, "trust": trust}
        entry.update(rep.summary())
        entry["details"] = rep.details
        reports.append(entry)
    return reports

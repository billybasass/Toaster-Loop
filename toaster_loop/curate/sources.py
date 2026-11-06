"""WP3d/WP3c: extra curation SOURCES -> curate-layout dataset dirs.

Three entry points, one shared materializer:

  * ``import_hub_snapshot``   — a downloaded hub dataset snapshot
    (<home>/hub/datasets--org--name/snapshots/<sha>/) run through the existing
    importer auto-detection (sharegpt/alpaca/chatml sniff + ingest gate);
  * ``ingest_folder``         — any local directory: .json/.jsonl go through
    importer.py unchanged, prose/doc files (.md/.txt/.rst/.html) are chunked
    and synthesized into ShareGPT rows by the configured local model;
  * ``crawl_website``         — same-origin bounded crawl (SSRF posture from
    core/tools.web_fetch: every URL re-validated, robots.txt respected), HTML
    -> text -> the same doc->ShareGPT synthesis.

All three land a normal <home>/datasets/<id>/ record (sft.jsonl + manifest
with provenance + train.yaml via render_train_yaml), so train/iterate consume
them with ZERO special cases. Synthesis is injectable (``synth_fn``) — the
dashboard passes the live serve endpoint's chat-completions; tests pass a stub;
without any model the fallback synthesizer derives heading-scoped extractive
QA pairs so ingestion never hard-fails offline (flagged in manifest warnings).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

from ..curate.importer import ImportError_, detect_format, read_rows
from .ingest_gate import run_gate, write_gate_sidecars
from .sharegpt import messages_to_sharegpt

logger = logging.getLogger("toaster_loop.curate.sources")

#: Doc extensions ingested as prose (vs data files through importer.py).
DOC_EXTS = {".md", ".txt", ".rst", ".html", ".htm"}
DATA_EXTS = {".json", ".jsonl", ".parquet"}

#: Directory names never descended into during folder ingest: vendored
#: dependencies and tool caches produce thousands of "cannot detect dataset
#: format" warnings (and zero training value) when someone points the folder
#: source at a repo root with a .venv inside.
SKIP_DIR_NAMES = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", "site-packages", "dist-info", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", "build", "dist", ".eggs",
}

#: Per-file / total read caps: a "curate my whole home dir" pick must not OOM
#: the dashboard or take minutes per file.
MAX_FILE_BYTES = 2_000_000
MAX_TOTAL_BYTES = 40_000_000

#: Synthesis chunking: ~4 KB of prose per model call keeps latency bounded
#: while giving the model enough context for a useful pair.
CHUNK_CHARS = 4000
OVERLAP_CHARS = 200

#: Manifest warning cap: link-heavy JS docs sites produce hundreds of junk
#: URLs (route manifests inside <script> JSON), and every failed fetch used to
#: land verbatim in manifest.warnings — an unreadable wall that hid the real
#: problems. Beyond the cap we keep the first MAX_WARNINGS plus a summary line.
MAX_WARNINGS = 50

#: Script/style blocks, stripped before link extraction (the crawl's href
#: regex otherwise matches "href":"..." inside framework JSON payloads and
#: queues hundreds of nonexistent routes as 404 warnings).
_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>")

ProgressFn = Callable[[str], None]
"""Progress reporter: one line per unit of work (file/page/chunk batch). The
dashboard writes these into the job log so long crawls aren't silent."""

ShouldCancelFn = Callable[[], bool]
"""Cooperative cancellation probe: checked between pages/files/chunks. When
it returns True the pipeline raises CurateCancelled instead of starting the
next unit of work — a cancelled job must stop in seconds, not finish its crawl."""


class CurateCancelled(RuntimeError):
    """Raised when should_cancel fires mid-build (job cancelled from the UI)."""


#: Path segments that mark a localized docs tree (/zh/, /ja/tutorials/, …).
#: Curated to real language codes so English routes like /id/ or /no/ are not
#: misread — only these exact segments count as a language prefix.
LANG_SEGMENTS = {
    "en", "ar", "cs", "da", "de", "el", "es", "fa", "fi", "fr", "he", "hi",
    "hu", "id", "it", "ja", "ko", "nl", "no", "pl", "pt", "ro", "ru", "sv",
    "th", "tr", "uk", "vi", "zh", "zh-cn", "zh-tw", "zh-hans", "zh-hant",
    "pt-br", "es-419",
}


def _path_lang(path: str) -> Optional[str]:
    """Language code of a URL path's first matching segment, else None."""
    for seg in path.lower().strip("/").split("/"):
        if seg in LANG_SEGMENTS:
            return seg
    return None


def _lang_ok(path: str, lang: Optional[str]) -> bool:
    """True when ``path`` may be crawled for the user's language.

    Pages without a recognized language segment always pass (most docs live
    un-prefixed). A page under /zh/ or /ja/... passes only when it matches the
    requested language, including prefix forms (lang "zh" accepts "zh-tw")."""
    if not lang or lang == "*":
        return True
    found = _path_lang(path)
    if found is None:
        return True
    want = lang.lower()
    return found == want or found.split("-")[0] == want \
        or want.split("-")[0] == found


def _looks_like_data(text: str) -> bool:
    """True when a text blob is machine data (JSON/JS/config) rather than prose.

    Docs sites ship content inside framework payloads; feeding those chunks to
    synthesis produced training rows of raw JSON — actively harmful for SFT.
    Structural punctuation density separates the two cleanly: prose almost
    never exceeds these ratios, minified/embedded JSON always does."""
    s = text[:4000]
    n = max(1, len(s))
    structural = sum(s.count(c) for c in "{}[]")
    quotes = s.count('"')
    colons = s.count(":")
    return (structural >= 8 and structural / n > 0.01) \
        or (quotes / n > 0.04 and colons / n > 0.01)


def _scrub_machine_lines(text: str) -> str:
    """Drop lines that are clearly machine data (JSON key/value streams).

    Crawl text from docs SPAs is prose and embedded payloads interleaved;
    chunk-level _looks_like_data misses mixed chunks, so scrub line by line
    first. A line with 6+ double quotes, or brace+quote pairs, is payload."""
    kept: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if s:
            q = s.count('"')
            br = s.count("{") + s.count("}")
            if q >= 6 or (br >= 2 and q >= 2):
                continue
        kept.append(ln)
    return "\n".join(kept)


@dataclass
class SourceResult:
    dataset_id: str
    dataset_dir: Path
    rows: int
    skipped_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_id(name: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-"
                   for c in name).strip("-_")
    return safe[:64] or "source"


def alloc_dataset_id(home: Path, name: Optional[str]) -> Tuple[Path, str]:
    """Dataset dir + id as ``<name>-<UTC stamp>`` — reserved (mkdir) on return.

    Name-first so the dataset list reads like "docs-20261103T192214Z" instead
    of a timestamp everyone has to decode; the trailing second-granularity UTC
    stamp keeps ids unique and sortable within a name. Collision (rapid
    successive runs) appends -2, -3… Shared by sources/importer/curater so all
    three producers write one scheme. The mkdir(exist_ok=False) doubles as the
    cross-process reservation lock the curater's claim-before-write relies on.
    """
    datasets_root = Path(home) / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = _safe_id(name) if name else ""
    base = f"{safe}-{stamp}" if safe else stamp
    candidate = base
    n = 2
    while True:
        ds_dir = datasets_root / candidate
        try:
            ds_dir.mkdir(parents=True, exist_ok=False)
            return ds_dir, ds_dir.name
        except FileExistsError:
            candidate = f"{base}-{n}"
            n += 1



def materialize(home: Path, conversations: List[List[Dict[str, Any]]], *,
                name: str, provenance: str, source: Dict[str, Any],
                warnings: Optional[List[str]] = None,
                title: Optional[str] = None) -> SourceResult:
    """Write a curate-layout dataset record from ShareGPT conversations.

    Same contract as importer.import_dataset's output (sft.jsonl rows of
    {"conversations": [...]}, gate sidecars, train.yaml template) — the ONLY
    differences are the provenance value and source metadata keys, both purely
    additive manifest fields. Raises ImportError_ when nothing usable arrived.
    """
    from .curater import CurationConfig, render_train_yaml

    if not conversations:
        raise ImportError_("no usable rows produced from this source")
    ds_dir, dataset_id = alloc_dataset_id(Path(home), name)
    with open(ds_dir / "sft.jsonl", "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps({"conversations": conv},
                               ensure_ascii=False) + "\n")
    warnings = list(warnings or [])
    if len(warnings) > MAX_WARNINGS:
        extra = len(warnings) - MAX_WARNINGS
        warnings = warnings[:MAX_WARNINGS] + [
            f"(+{extra} more warnings suppressed — see the job log)"]
    gate = run_gate(conversations, Path(home) / "datasets",
                    dataset_id=dataset_id)
    manifest: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "title": title or name,            # human label for the UI list
        "created_at": _utcnow(),
        "provenance": provenance,          # hub | local_folder | website
        **source,                          # source_repo / source_path / urls
        "skipped_rows": 0,
        "filters": None,
        "files": {
            "sft.jsonl": {"count": len(conversations), "trace_ids": [],
                          "format": "sharegpt"},
            "dpo.jsonl": {"count": 0, "trace_ids": [],
                          "format": "trl-preference"},
        },
        "format_hints": {"sft_format_type": "sharegpt",
                         "train_yaml": "train.yaml"},
        "warnings": list(warnings or []),
    }
    (ds_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (ds_dir / "train.yaml").write_text(
        render_train_yaml(ds_dir, CurationConfig(), dataset_id),
        encoding="utf-8")
    write_gate_sidecars(ds_dir, conversations, "imported", gate)
    return SourceResult(dataset_id=dataset_id, dataset_dir=ds_dir,
                        rows=len(conversations), warnings=list(warnings or []))


# ------------------------------------------------------------- hub snapshots

def find_hub_snapshot(home: Path, repo_id: str) -> Optional[Path]:
    """Latest snapshot dir for a downloaded hub DATASET repo, or None.

    modelhub lays datasets out under its per-kind root —
    <home>/hub/datasets/datasets--<org>--<name>/snapshots/<sha>/ (see
    HubClient.kind_base/repo_cache_dir). The older flat layout
    <home>/hub/datasets--<org>--<name>/ is checked too so snapshots downloaded
    before the fix still import. Newest snapshot wins by mtime; path-traversal-
    safe because repo_id is validated first."""
    from ..core.modelhub import _CACHE_PREFIX, validate_repo_id

    repo_id = validate_repo_id(repo_id)
    seg = _CACHE_PREFIX["datasets"] + repo_id.replace("/", "--")
    for snaps in (Path(home) / "hub" / "datasets" / seg / "snapshots",
                  Path(home) / "hub" / seg / "snapshots"):
        if not snaps.is_dir():
            continue
        cands = [d for d in snaps.iterdir() if d.is_dir()]
        if cands:
            return max(cands, key=lambda d: d.stat().st_mtime)
    return None


# ------------------------------------------------- generic table -> pairs

#: Column-name candidates for mapping a tabular row (e.g. a HF parquet) onto a
#: question/answer pair. Order = preference; first present combination wins.
_Q_COLS = ("question", "instruction", "prompt", "description", "caption",
           "query", "input", "text")
_A_COLS = ("answer", "output", "completion", "response", "code", "solution",
           "result")


def _pair_columns(cols: List[str]) -> Optional[Tuple[str, str]]:
    for q in _Q_COLS:
        for a in _A_COLS:
            if q != a and q in cols and a in cols:
                return (q, a)
    return None


def parquet_to_conversations(
        path: Path) -> Tuple[List[List[Dict[str, Any]]], List[str]]:
    """.parquet -> (ShareGPT conversations, warnings).

    HF datasets ship most public corpora as parquet; without this the hub
    import produced nothing for them. Binary columns (the HF image struct
    {bytes,path}) are dropped. When no recognized question/answer column pair
    exists we degrade to the doc pipeline over the concatenated text columns,
    so prose-only parquet still yields rows."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return [], [f"{path.name}: parquet support needs pyarrow "
                    "(pip install pyarrow); file skipped"]
    try:
        table = pq.read_table(path)
        records = table.to_pylist()
    except Exception as e:  # noqa: BLE001 — one bad file must not kill import
        return [], [f"{path.name}: unreadable parquet ({e})"]

    def _texty(v: Any) -> bool:
        if isinstance(v, str):
            return True
        if isinstance(v, dict):  # HF image struct or nested metadata
            return False
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    cols = [c for c in table.schema.names]
    pair = _pair_columns(cols)
    conversations: List[List[Dict[str, Any]]] = []
    warnings: List[str] = []
    if pair:
        qcol, acol = pair
        for rec in records:
            q, a = rec.get(qcol), rec.get(acol)
            if not _texty(q) or not _texty(a):
                continue
            q, a = str(q).strip(), str(a).strip()
            if not q or not a:
                continue
            if acol == "code":  # description->script pairs train better framed
                conversations.append([
                    {"from": "human",
                     "value": f"Write a Python script that satisfies this "
                              f"description:\n\n{q}"},
                    {"from": "gpt", "value": f"```python\n{a}\n```"}])
            else:
                conversations.append([
                    {"from": "human", "value": q},
                    {"from": "gpt", "value": a}])
        return conversations, warnings

    # No recognizable pair: fall back to doc synthesis over text columns.
    text_cols = [c for c in cols if records and _texty(records[0].get(c))]
    if not text_cols:
        return [], [f"{path.name}: no text columns recognized"]
    blob = "\n\n".join(
        "\n".join(str(r.get(c, "")) for c in text_cols) for r in records)
    convs, w = docs_to_rows([(path.stem, blob)], synth_fn=None)
    warnings.extend(f"{path.name}: {x}" for x in w)
    return convs, warnings


def import_hub_snapshot(home: Path, repo_id: str, *,
                        name: Optional[str] = None) -> SourceResult:
    """Materialize a downloaded hub dataset snapshot into <home>/datasets/.

    Data files (.jsonl/.json) go through importer.py's reader + format sniff
    unchanged; .parquet files map through the generic column-pair reader (HF's
    dominant format); if the snapshot carries only prose files they fall back
    to the doc pipeline WITHOUT synthesis (raw text rows are useless, so we
    synthesize with the configured model when one is live — callers pass
    synth_fn)."""
    snap = find_hub_snapshot(home, repo_id)
    if snap is None:
        raise ImportError_(
            f"no downloaded hub snapshot for {repo_id!r} — download it from "
            "the Hub page first")

    # modelhub's own download manifest is metadata, not dataset content.
    def _skip(p: Path) -> bool:
        return p.name == "toaster_loop_download.json"

    data_files = sorted(p for p in snap.rglob("*")
                        if p.is_file() and not _skip(p)
                        and p.suffix.lower() in DATA_EXTS - {".parquet"})
    parquet_files = sorted(p for p in snap.rglob("*")
                           if p.is_file() and not _skip(p)
                           and p.suffix.lower() == ".parquet")
    doc_files = sorted(p for p in snap.rglob("*")
                       if p.is_file() and not _skip(p)
                       and p.suffix.lower() in DOC_EXTS)
    warnings: List[str] = []
    conversations: List[List[Dict[str, Any]]] = []
    from ..curate.importer import _row_to_messages
    for f in data_files:
        try:
            rows = read_rows(f)
            fmt = detect_format(
                [r for r in rows if "__malformed_line__" not in r])
        except ImportError_ as e:
            warnings.append(f"{f.name}: {e}")
            continue
        for row in rows:
            if "__malformed_line__" in row:
                continue
            msgs = _row_to_messages(row, fmt)
            conv = messages_to_sharegpt(msgs) if msgs else []
            if conv:
                conversations.append(conv)
    for f in parquet_files:
        convs, w = parquet_to_conversations(f)
        conversations.extend(convs)
        warnings.extend(w)
    if not conversations and doc_files:
        conversations, w = docs_to_rows(
            [(p.name, p.read_text(encoding="utf-8", errors="replace"))
             for p in doc_files[:200]], synth_fn=None)
        warnings.extend(w)
    label = name or repo_id.split("/")[-1]
    res = materialize(home, conversations, name=label, provenance="hub",
                      source={"source_repo": repo_id,
                              "snapshot_dir": str(snap)},
                      warnings=warnings, title=repo_id)
    return res


# ---------------------------------------------------------- doc -> ShareGPT

def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS,
               overlap: int = OVERLAP_CHARS) -> List[str]:
    """Paragraph-greedy chunking with small overlap; deterministic."""
    text = re.sub(r"\r\n?", "\n", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    cur = ""
    for p in paras:
        while len(p) > chunk_chars:          # single huge paragraph
            head = p[:chunk_chars]
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(head)
            p = p[chunk_chars - overlap:]
        if len(cur) + len(p) + 2 > chunk_chars and cur:
            chunks.append(cur)
            cur = (cur[-overlap:] + "\n\n" + p) if overlap else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in (s.strip() for s in chunks) if c]


SynthFn = Callable[[str, str], List[Tuple[str, str]]]
"""(source_name, chunk) -> [(question, answer), ...] — may raise; the caller
degrades to the fallback synthesizer and records a warning."""


def fallback_synth(name: str, chunk: str) -> List[Tuple[str, str]]:
    """No-model synthesis: heading-scoped extractive pairs.

    For each markdown/rst heading in the chunk, pair 'Summarize what the
    "<heading>" section of <name> covers.' with the first ~600 chars under it.
    Deterministic and dependency-free — keeps offline ingestion working; the
    manifest warning tells the user quality improves with a live model."""
    pairs: List[Tuple[str, str]] = []
    secs = re.split(r"(?m)^(#{1,4}\s+.+|\S.*\n[=\-]{3,})$", chunk)
    for i in range(1, len(secs), 2):
        head = secs[i].strip().lstrip("#").strip()
        body = secs[i + 1].strip() if i + 1 < len(secs) else ""
        if not body or len(body) < 40:
            continue
        pairs.append((
            f"Summarize what the \"{head}\" section of {name} covers.",
            body[:600]))
    if not pairs and len(chunk.strip()) >= 80:
        pairs.append((f"What are the key points of this {name} excerpt?",
                      chunk[:600]))
    return pairs


def _http_synth(base_url: str, model: str) -> SynthFn:
    """Synthesis through the LIVE serve endpoint (plain httpx chat-completions).

    One call per chunk; the prompt demands strict Q:/A: blocks. Raises on any
    transport/parse failure so docs_to_rows can degrade to fallback_synth."""
    import httpx

    sys_prompt = (
        "You convert documentation excerpts into training pairs for a local "
        "assistant. From the excerpt, write 1-4 question/answer pairs that "
        "are fully answerable from the text alone. Output format EXACTLY:\n"
        "Q: <question>\nA: <answer>\nNo preamble, no markdown fences.")

    def _synth(name: str, chunk: str) -> List[Tuple[str, str]]:
        r = httpx.post(base_url.rstrip("/") + "/chat/completions",
                       json={"model": model, "temperature": 0.2,
                             "max_tokens": 700,
                             "messages": [
                                 {"role": "system", "content": sys_prompt},
                                 {"role": "user",
                                  "content": f"Source: {name}\n\n{chunk}"}]},
                       timeout=120.0)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"] or ""
        return _parse_qa(text)

    return _synth


def _parse_qa(text: str) -> List[Tuple[str, str]]:
    """Parse Q:/A: blocks (tolerating **Q:** markdown and blank lines)."""
    text = re.sub(r"\*\*([QA]):\*\*", r"\1:", text)
    pairs: List[Tuple[str, str]] = []
    q = a = None
    for line in text.splitlines():
        m = re.match(r"^\s*(Q|A)\s*[:.]\s*(.*)$", line)
        if m and m.group(1) == "Q":
            if q is not None and a:
                pairs.append((q.strip(), a.strip()))
            q, a = m.group(2), None
        elif m and m.group(1) == "A" and q is not None:
            a = m.group(2)
        elif a is not None:
            a += "\n" + line
        elif q is not None:
            q += "\n" + line
    if q is not None and a:
        pairs.append((q.strip(), a.strip()))
    return [(x, y) for x, y in pairs if x and y]


def _usable_pair(q: str, a: str) -> bool:
    """Training-value screen for one synthesized pair.

    Rejects empty/short answers and machine-data answers (JSON/JS blobs from
    SPA payloads). Rows like these were the "may hurt a model if trained on
    it" complaint — a garbage answer teaches the model to emit JSON noise."""
    q, a = q.strip(), a.strip()
    if not q or len(a) < 20:
        return False
    return not _looks_like_data(a)


def detect_lang() -> str:
    """Best-effort user language (BCP-47 base) from the environment.

    Checks LANGUAGE/LC_ALL/LC_MESSAGES/LANG (POSIX) and locale.getlocale();
    "en" for a US/EU desktop, "ja" on a Japanese one. "*" when undetectable —
    callers treat that as no filtering."""
    import os

    for var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        v = os.environ.get(var, "").strip()
        if v and not v.startswith(("C.", "C")):
            code = v.split(":")[0].split(".")[0].replace("@", "_") \
                   .lower().replace("_", "-")
            if code:
                return code
    try:
        import locale as _locale

        loc = _locale.getlocale()[0]
        if loc:
            return loc.lower().replace("_", "-")
    except Exception:  # noqa: BLE001 — locale is best-effort
        pass
    return "*"


def docs_to_rows(files: List[Tuple[str, str]], *,
                 synth_fn: Optional[SynthFn],
                 max_chunks_per_file: int = 20,
                 progress: Optional[ProgressFn] = None,
                 should_cancel: Optional[ShouldCancelFn] = None,
                 ) -> Tuple[List[List[Dict[str, Any]]], List[str]]:
    """[(name, text)] -> (ShareGPT conversations, warnings).

    Each chunk goes through synth_fn (live model) or fallback_synth; a failing
    model call degrades that file to the fallback with one warning. Machine
    data is screened at two levels: whole chunks that are JSON/JS payloads are
    skipped, and individual pairs whose answer looks like data are dropped —
    SPA docs sites embed framework state in the page text, and feeding it to
    synthesis produced harmful rows. Output rows are single-turn user/assistant
    ShareGPT conversations — identical shape to importer output, so the
    training mask path is unchanged."""
    warnings: List[str] = []
    conversations: List[List[Dict[str, Any]]] = []
    dropped_data = 0
    for fi, (name, text) in enumerate(files):
        if should_cancel is not None and should_cancel():
            raise CurateCancelled("curation cancelled before synthesis finished")
        scrubbed = _scrub_machine_lines(text)
        chunks = chunk_text(scrubbed)[:max_chunks_per_file]
        if not chunks:
            continue
        used_fallback = False
        for chunk in chunks:
            if should_cancel is not None and should_cancel():
                raise CurateCancelled(
                    "curation cancelled during synthesis")
            if _looks_like_data(chunk):
                dropped_data += 1
                continue
            pairs: List[Tuple[str, str]] = []
            if synth_fn is not None:
                try:
                    pairs = synth_fn(name, chunk)
                except Exception as e:  # noqa: BLE001 — degrade, never die
                    if not used_fallback:
                        warnings.append(
                            f"{name}: model synthesis failed ({e}); used "
                            "heading fallback (start a local model on the "
                            "Serve page for higher-quality pairs)")
                        used_fallback = True
            else:
                pairs = []
            if not pairs:
                pairs = fallback_synth(name, chunk)
            for q, a in pairs:
                if not _usable_pair(q, a):
                    dropped_data += 1
                    continue
                conversations.append([
                    {"from": "human", "value": q},
                    {"from": "gpt", "value": a}])
        if progress is not None:
            progress(f"{name}: {len(conversations)} rows so far "
                     f"({fi + 1}/{len(files)} files)")
    if dropped_data:
        warnings.append(
            f"dropped {dropped_data} machine-data chunk(s)/pair(s) "
            "(JSON/JS payloads are not usable training text)")
    return conversations, warnings


# ---------------------------------------------------------------- folder

def iter_doc_files(root: Path) -> Tuple[List[Path], List[Path]]:
    """(data_files, doc_files) under ``root``, recursive, caps applied.

    Vendored-dependency and tool-cache directories (SKIP_DIR_NAMES — .venv,
    node_modules, site-packages, …) are pruned before descent: pointing the
    folder source at a repo root should curate the project's docs, not
    thousands of dependency metadata files. Hidden directories are pruned too
    (.git and friends), except dot-prefixed *files* inside kept dirs still
    count (a README-style .md is fine)."""
    data: List[Path] = []
    docs: List[Path] = []
    total = 0

    def walk(d: Path):
        nonlocal total
        try:
            entries = sorted(d.iterdir())
        except OSError:
            return
        for p in entries:
            if p.is_symlink():
                continue
            if p.is_dir():
                if p.name in SKIP_DIR_NAMES or (
                        p.name.startswith(".") and p is not d):
                    continue
                walk(p)
                continue
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext not in DOC_EXTS and ext not in DATA_EXTS:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_BYTES or total + size > MAX_TOTAL_BYTES:
                continue
            total += size
            (data if ext in DATA_EXTS else docs).append(p)

    walk(root)
    return data, docs


def ingest_folder(home: Path, folder: str, *, name: Optional[str] = None,
                  synth_fn: Optional[SynthFn] = None,
                  progress: Optional[ProgressFn] = None,
                  should_cancel: Optional[ShouldCancelFn] = None
                  ) -> SourceResult:
    """Recursive local-folder curation (WP3d). .json/.jsonl through importer.py
    unchanged; .parquet through the column-pair reader; prose/doc files
    through the doc->ShareGPT synthesis pipeline."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ImportError_(f"not a directory: {folder}")
    data_files, doc_files = iter_doc_files(root)
    json_files = [p for p in data_files if p.suffix.lower() != ".parquet"]
    parquet_files = [p for p in data_files if p.suffix.lower() == ".parquet"]
    total_units = len(json_files) + len(parquet_files) + len(doc_files)
    logger.info("folder ingest %s: %d files (%d data, %d docs)",
                root, total_units, len(data_files), len(doc_files))
    if progress is not None:
        progress(f"scanned {root}: {total_units} files "
                 f"({len(data_files)} data, {len(doc_files)} docs)")
    warnings: List[str] = []
    conversations: List[List[Dict[str, Any]]] = []
    skipped: List[str] = []
    done_units = 0
    from ..curate.importer import _row_to_messages

    def _check_cancel() -> None:
        if should_cancel is not None and should_cancel():
            raise CurateCancelled("folder ingest cancelled")

    for f in json_files:
        _check_cancel()
        try:
            rows = read_rows(f)
        except ImportError_ as e:
            warnings.append(f"{f}: {e}")
            skipped.append(f.name)
            continue
        try:
            fmt = detect_format(
                [r for r in rows if "__malformed_line__" not in r])
        except ImportError_ as e:
            warnings.append(f"{f}: {e}")
            skipped.append(f.name)
            continue
        for row in rows:
            if "__malformed_line__" in row:
                continue
            msgs = _row_to_messages(row, fmt)
            conv = messages_to_sharegpt(msgs) if msgs else []
            if conv:
                conversations.append(conv)
        done_units += 1
        logger.info("folder ingest %s: %d/%d files, %d rows",
                    root.name, done_units, total_units, len(conversations))
        if progress is not None:
            progress(f"{f.name}: {len(conversations)} rows "
                     f"({done_units}/{total_units} files)")
    for f in parquet_files:
        _check_cancel()
        convs, w = parquet_to_conversations(f)
        conversations.extend(convs)
        warnings.extend(w)
        done_units += 1
        logger.info("folder ingest %s: %d/%d files, %d rows",
                    root.name, done_units, total_units, len(conversations))
        if progress is not None:
            progress(f"{f.name}: {len(conversations)} rows "
                     f"({done_units}/{total_units} files)")
    doc_payload = []
    for p in doc_files:
        try:
            doc_payload.append((str(p.relative_to(root)),
                                p.read_text(encoding="utf-8",
                                            errors="replace")))
        except OSError as e:
            warnings.append(f"{p}: unreadable ({e})")

    def _doc_progress(msg: str) -> None:
        if progress is not None:
            progress(msg)
        logger.info("folder ingest %s: %s", root.name, msg)

    conv2, w = docs_to_rows(doc_payload, synth_fn=synth_fn,
                            progress=_doc_progress if progress is not None
                            or logger.isEnabledFor(logging.INFO) else None,
                            should_cancel=should_cancel)
    warnings.extend(w)
    conversations.extend(conv2)
    label = name or root.name
    res = materialize(home, conversations,
                      name=label, provenance="local_folder",
                      source={"source_path": str(root.resolve()),
                              "files_data": len(data_files),
                              "files_docs": len(doc_files)},
                      warnings=warnings, title=label)
    res.skipped_files = skipped
    return res


# --------------------------------------------------------------- website

#: Crawl bounds: docs sites are deep; depth 2 from the entry page plus a hard
#: page cap keeps one crawl to seconds/minutes, not hours.
CRAWL_MAX_PAGES = 30
CRAWL_MAX_DEPTH = 2
CRAWL_TIMEOUT_S = 20.0


def _html_to_text(body: str) -> str:
    """HTML -> readable text (same posture as core/tools._html_to_text, kept
    local so tools.py stays the single fetch implementation)."""
    from ..core.tools import _html_to_text as _t

    return _t(body)


def _fetch_guarded(url: str, *, allow_private: bool = False
                   ) -> Tuple[Optional[str], Optional[str]]:
    """(html_or_None, error_or_None) with the web_fetch SSRF posture re-checked
    per hop (no automatic redirects)."""
    import httpx

    from ..core.tools import _url_block_reason

    current = url
    for _hop in range(4):
        reason = _url_block_reason(current, allow_private)
        if reason is not None:
            return None, f"blocked: {reason}"
        try:
            with httpx.Client(timeout=CRAWL_TIMEOUT_S,
                              follow_redirects=False) as client:
                resp = client.get(current, headers={
                    "User-Agent": "toaster_loop-dataset-curator (local)"})
        except httpx.HTTPError as e:
            return None, f"fetch failed: {e}"
        if resp.is_redirect or resp.has_redirect_location:
            current = urljoin(str(resp.url),
                              resp.headers.get("location") or "")
            continue
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        ctype = resp.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype:
            return None, f"not text/html ({ctype or 'unknown'})"
        return resp.text, None
    return None, "too many redirects"


def _robots_allows(base: str, path: str, *, allow_private: bool = False
                   ) -> bool:
    """Minimal robots.txt check (User-agent: * Disallow prefixes). Fetch
    failures ALLOW crawling — an unreachable robots.txt must not silently
    block a docs site; the bounded crawl is the real guard."""
    html, err = _fetch_guarded(urljoin(base, "/robots.txt"),
                               allow_private=allow_private)
    if err or html is None:
        return True
    disallow: List[str] = []
    applies = False
    for line in html.splitlines():
        line = line.strip()
        if line.lower().startswith("user-agent:"):
            applies = line.split(":", 1)[1].strip() == "*"
        elif applies and line.lower().startswith("disallow:"):
            v = line.split(":", 1)[1].strip()
            if v:
                disallow.append(v)
    return not any(path.startswith(d) for d in disallow)


#: Non-page assets never worth queueing during a crawl. Docs sites reference
#: fonts/bundles inline; fetching them produced "not text/html (font/woff2)"
#: style warnings that were pure noise.
_ASSET_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".css", ".js", ".mjs",
                   ".zip", ".pdf", ".svg", ".ico", ".woff", ".woff2",
                   ".ttf", ".otf", ".eot", ".mp4", ".webm", ".webp",
                   ".map", ".xml", ".txt.gz", ".avif")


def _scope_prefix(entry_path: str) -> str:
    """Docs-tree prefix of an entry page path: the directory containing it.

    ``/documentation/unreal-engine/unreal-engine-5-8-documentation`` scopes to
    ``/documentation/unreal-engine/`` — so a crawl started on a docs page never
    wanders into sibling sections of the same host (forums, news, store). An
    entry at the site root scopes to "/" (whole-site, unchanged behavior)."""
    p = entry_path or "/"
    if p.endswith("/"):
        return p
    return p.rsplit("/", 1)[0] + "/"


def crawl_website(home: Path, url: str, *, name: Optional[str] = None,
                  max_pages: int = CRAWL_MAX_PAGES,
                  max_depth: int = CRAWL_MAX_DEPTH,
                  synth_fn: Optional[SynthFn] = None,
                  allow_private_networks: bool = False,
                  sleep_s: float = 0.2,
                  progress: Optional[ProgressFn] = None,
                  should_cancel: Optional[ShouldCancelFn] = None,
                  lang: Optional[str] = None,
                  path_prefix: Optional[str] = None) -> SourceResult:
    """Bounded same-origin crawl -> HTML text -> doc->ShareGPT pipeline.

    Politeness + safety: robots.txt honored (crawl-delay-free fixed sleep),
    same-origin only, page/depth caps, per-page fetch re-validated through the
    web_fetch SSRF posture (private targets refused unless explicitly allowed).
    ``lang`` restricts the crawl to one language on localized docs trees:
    pages under a foreign language prefix (/zh/, /ja/…) are skipped without
    fetching; None = auto-detect from the environment, "*" disables filtering.
    ``path_prefix`` bounds the crawl to one subtree of the host (e.g.
    "/documentation/unreal-engine/"). Default is derived from the entry page's
    directory, so a docs-page entry never crawls the site's forums/news/store;
    pass "" to opt back into whole-same-origin crawling. Source URLs land in
    the manifest for provenance."""
    entry = url.strip()
    if not entry:
        raise ImportError_("url required")
    parsed = urlparse(entry)
    if parsed.scheme not in ("http", "https"):
        raise ImportError_("only http(s) URLs can be crawled")
    explicit_lang = lang is not None
    lang = detect_lang() if lang is None else str(lang).strip().lower() or "*"
    # An entry page that IS a language mirror (site/zh/...) crawls that
    # language unless the caller said otherwise — auto-detect must not fight
    # an intentional deep entry.
    if not explicit_lang and lang != "*":
        entry_lang = _path_lang(parsed.path)
        if entry_lang:
            lang = entry_lang
    origin = f"{parsed.scheme}://{parsed.netloc}"
    allowed = _robots_allows(origin, parsed.path or "/",
                             allow_private=allow_private_networks)
    if not allowed:
        raise ImportError_(
            "robots.txt disallows crawling this path — the crawl refuses")

    warnings: List[str] = []
    pages: List[Tuple[str, str]] = []      # (url, text)
    fetched: Dict[str, str] = {}           # url -> depth
    skipped_lang = 0
    skipped_scope = 0
    scope = (_scope_prefix(parsed.path) if path_prefix is None
             else str(path_prefix))
    queue: List[Tuple[str, int]] = [(entry, 0)]
    logger.info("crawl start %s (max_pages=%d max_depth=%d lang=%s "
                "scope=%s)", entry, max_pages, max_depth, lang, scope or "/")
    if progress is not None:
        progress(f"start crawl {entry} (up to {max_pages} pages, "
                 f"language {lang}, under {scope or '/'})")
    while queue and len(fetched) < max_pages:
        if should_cancel is not None and should_cancel():
            raise CurateCancelled("website crawl cancelled")
        u, depth = queue.pop(0)
        u = urldefrag(u)[0]
        if u in fetched:
            continue
        # Asset URLs are dropped without fetching — no warning noise.
        if urlparse(u).path.lower().endswith(_ASSET_SUFFIXES):
            continue
        # Foreign-language pages (docs mirrors under /zh/, /ja/…) are skipped
        # silently: they are real pages, just not in the user's language.
        if u != entry and not _lang_ok(urlparse(u).path, lang):
            fetched[u] = str(depth)
            skipped_lang += 1
            continue
        fetched[u] = str(depth)
        html, err = _fetch_guarded(u, allow_private=allow_private_networks)
        if progress is not None:
            progress(f"page {len(fetched)}/{max_pages}: {u}"
                     + (f" — {err}" if err else ""))
        logger.info("crawl page %d/%d %s%s", len(fetched), max_pages, u,
                    "" if not err else f" — {err}")
        if err is not None or html is None:
            warnings.append(f"{u}: {err or 'no body'}")
            continue
        # robots applies per-page (cheap prefix cache via fetched set).
        if not _robots_allows(origin, urlparse(u).path or "/",
                              allow_private=allow_private_networks):
            warnings.append(f"{u}: disallowed by robots.txt")
            continue
        text = _html_to_text(html)
        if len(text.strip()) >= 80:
            pages.append((u, text))
        if depth < max_depth:
            # Links come from real markup only: script/style bodies are
            # stripped first so framework JSON ("href":"...") inside the page
            # is never mistaken for navigable routes (hundreds of phantom 404s
            # on docs SPAs).
            for href in re.findall(
                    r"""href=["']([^"'#]+)["']""",
                    _SCRIPT_STYLE_RE.sub(" ", html)):
                nu = urljoin(u, href)
                if urlparse(nu).netloc != parsed.netloc or nu in fetched:
                    continue
                # Subtree scope: links outside the entry page's docs tree
                # (forums, news, marketing pages on the same host) are never
                # queued — the crawl follows the documentation, not the site.
                if scope and not urlparse(nu).path.startswith(scope):
                    skipped_scope += 1
                    continue
                if urlparse(nu).path.lower().endswith(_ASSET_SUFFIXES):
                    continue
                # Never queue a foreign-language mirror (never counted as
                # fetched either, so it stays cheap).
                if not _lang_ok(urlparse(nu).path, lang):
                    skipped_lang += 1
                    continue
                queue.append((nu, depth + 1))
        time.sleep(max(0.0, sleep_s))
    if not pages:
        raise ImportError_(f"crawl produced no readable pages from {entry}")
    if skipped_lang:
        warnings.append(
            f"skipped {skipped_lang} page(s) in other languages "
            f"(keeping '{lang}')")
    if skipped_scope:
        warnings.append(
            f"skipped {skipped_scope} link(s) outside the docs subtree "
            f"'{scope or '/'}' (pass a wider scope path to include them)")
    if progress is not None:
        progress(f"crawled {len(pages)} readable pages; synthesizing…")
    logger.info("crawl done %s: %d readable pages (%d other-language skipped),"
                " synthesizing", entry, len(pages), skipped_lang)

    def _synth_progress(msg: str) -> None:
        if progress is not None:
            progress(msg)
        logger.info("crawl synth %s: %s", parsed.netloc, msg)

    conversations, w = docs_to_rows(
        [(urlparse(u).path or "/", t) for u, t in pages],
        synth_fn=synth_fn,
        progress=_synth_progress if progress is not None
        or logger.isEnabledFor(logging.INFO) else None,
        should_cancel=should_cancel)
    warnings.extend(w)
    label = name or (parsed.netloc.split(".")[0] or "site")
    res = materialize(home, conversations,
                      name=label, provenance="website",
                      source={"source_url": entry,
                              "crawled_urls": [u for u, _ in pages]},
                      warnings=warnings, title=parsed.netloc or label)
    return res


# ------------------------------------------------------- synthesis resolver

def live_synth_fn(app: Any, server_id: Optional[str] = None
                  ) -> Optional[SynthFn]:
    """Synthesis via a serving instance's chat-completions endpoint.

    ``server_id`` selects a named multi-instance server (Phase 6); None or
    "default" uses the legacy serve_state view. The chosen instance must be
    alive (state_alive) — a dead state file falls back to the default so a
    stale selection never silently kills synthesis. Returns None when nothing
    is serving — callers then use fallback_synth and record the hint. Kept
    app-shaped (not importing chat.py) so curate/ stays dashboard-free."""
    home = Path(app.home)
    st: Optional[Dict[str, Any]] = None
    sid = str(server_id or "").strip()
    if sid and sid != "default":
        try:
            from ..serving import instances as srv_inst

            cand = srv_inst.read_state_merged(home, sid)
            if srv_inst.state_alive(cand):
                st = cand
            else:
                logger.warning(
                    "curation server %r is not running; falling back to the "
                    "default serve endpoint", sid)
        except Exception:  # noqa: BLE001 — unknown instance == no override
            logger.warning("unknown curation server %r; using default", sid)
    if st is None:
        try:
            from ..serving.deploy import read_serve_state

            st = read_serve_state(home)
        except Exception:  # noqa: BLE001 — no state file == nothing serving
            return None
    base = str((st or {}).get("base_url") or "").strip()
    model = str((st or {}).get("model_name") or "").strip()
    if not base or not model:
        return None
    return _http_synth(base, model)

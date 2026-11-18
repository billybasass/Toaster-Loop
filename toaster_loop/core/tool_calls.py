r"""Text tool-call fallback parsing for GGUFs with flaky function calling.

Local quantized models frequently emit tool calls as *text* instead of the
OpenAI ``tool_calls`` field -- usually a fenced JSON block naming a tool and
its arguments, or an XML-ish block. Without help those turns simply end and
the agent silently does nothing. This module recognizes the common shapes
and converts them into native tool-call dicts so the runner loop can execute
them::

    ```json
    {"name": "read_file", "arguments": {"path": "a.py"}}
    ```

    < tool_call>read_file{"path": "a.py"}</tool_call>   (spacing varies)

WP8 (tool-call / reasoning-loop recovery hardening) added three mechanisms
on top of the original fenced-JSON/XML recognizer:

  1. :func:`strip_stop_tokens` -- iterative tail stripping of repeated stop
     sequences (end-of-turn / end-of-document control-marker families,
     duplicated assistant-turn headers, acknowledged literal "stop" words)
     with mid-content preservation guarantees (see its docstring).
  2. A *repair pipeline* (:func:`recover` + :func:`_repair_json`) for
     invocations the legacy recognizer cannot load: single-quoted values,
     trailing commas, Python literals, truncated JSON, unclosed fences,
     XML-style wrappers and parameter-tag bodies, and bare JSON embedded in
     prose. Every mechanism applies its own attribution guard -- tags are
     shaped, or names are shaped and must pass a looks-like-a-tool-call
     test -- so plain data JSON pasted into a reply is never rewritten.
  3. Repair attribution: :class:`ParseReport` records *which* repairs each
     recovered call went through (``repairs_by_call_id``) without touching
     the wire shape of the call dicts, so every existing consumer sees the
     exact dict shape it always did.

Deliberate conservative posture (unchanged since day one): we only rewrite
text that unambiguously claims to be a tool call. A user pasting example
JSON into a prompt never triggers it because parsing runs on *assistant*
messages only. If the content leaks its own chat-template markup (the
pipe-delimited im_start / im_end / endoftext family): a model that does
that is broken, not merely weak at function calling -- rewriting would
paper over a real bug, so ALL passes, recovery included, refuse such text.

Equivalence contract (WP8, verified by the drift-corpus gate ladder in
``tests/test_core.py``): :func:`parse_legacy` is the verbatim
pre-WP8 recognizer; the new entry points run it FIRST and return its
result untouched whenever it yields anything. Recovery therefore only ever
adds calls for inputs that previously produced ``[]`` -- nothing that
parsed before can possibly parse differently.

Returns [] when nothing recognizable is present (the turn is then treated
as a final answer, exactly like before).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

LT = "<"   # assembled piecewise so this source never holds a whole control token
GT = ">"
BAR = "|"

# Fenced JSON object/array claiming to be a tool call.
_JSON_BLOCK = re.compile(
    r"```(?:json|tool_call)?\s*(\[.*?\]|\{.*?\})\s*```",
    re.S | re.I,
)
# XML-ish tool-call blocks: < tool_call>name{json}</ tool_call>
_XML_BLOCK = re.compile(
    LT + r"\s*(?:tool_call|tool_use)\s*>"
    r"([^<>{}]+?)\s*(\{.*?\})\s*"
    r"(?:" + LT + r"\s*/\s*(?:tool_call|tool_use)\s*>|$)",
    re.S,
)
# Leaked chat-template markup => never rewrite (see module docstring).
_SPECIAL_MARKUP = re.compile(
    LT + chr(92) + BAR + r"(?:im_start|im_end|endoftext)" + chr(92) + BAR + GT)


#: Synthetic marker tail used only to let the tail-band matcher reach
#: stop junk that sits INSIDE a candidate (between the closing brace and
#: the fence close). Built piecewise like every other marker shape here.
IM_END_TAIL = LT + BAR + "im_end" + BAR + GT


def _new_id() -> str:
    return f"call_text_{uuid.uuid4().hex[:12]}"


def _as_args(value: Any) -> Optional[Dict[str, Any]]:
    """Coerce an 'arguments' payload into a dict (accepts JSON strings)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _call_from_obj(obj: Any) -> Optional[Dict[str, Any]]:
    """One OpenAI-format tool_call dict from a parsed candidate object."""
    if not isinstance(obj, dict):
        return None
    # Already OpenAI-shaped (some models nest under "function").
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else None
    name = (fn or {}).get("name") or obj.get("name") or obj.get("tool") \
        or obj.get("tool_name")
    if not isinstance(name, str) or not name.strip():
        return None
    args: Dict[str, Any] = {}
    for key in ("arguments", "parameters", "args"):
        src = (fn or {}).get(key) if fn else obj.get(key)
        got = _as_args(src)
        if got is not None:
            args = got
            break
    return {
        "id": _new_id(),
        "type": "function",
        "function": {"name": name.strip(),
                     "arguments": json.dumps(args, ensure_ascii=False)},
    }


# ---------------------------------------------------------------------------
# Legacy recognizer (pre-WP8, verbatim) -- the drift-corpus baseline
# ---------------------------------------------------------------------------

def parse_legacy(content: Optional[str]) -> List[Dict[str, Any]]:
    """The PRE-WP8 recognizer, kept verbatim as the gate-ladder baseline.

    Behaviorally identical to the historical ``parse_text_tool_calls``. The
    new entry points run this first and only reach the recovery pipeline
    when it returns ``[]``, so the corpus baseline is exactly "what this
    function parses".
    """
    if not content or not isinstance(content, str):
        return []
    if _SPECIAL_MARKUP.search(content):
        return []

    found: List[Dict[str, Any]] = []
    spans: List[tuple] = []

    for m in _JSON_BLOCK.finditer(content):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        cands = obj if isinstance(obj, list) else [obj]
        calls = [_call_from_obj(c) for c in cands]
        if calls and all(calls):
            found.extend(calls)  # type: ignore[arg-type]
            spans.append(m.span())

    for m in _XML_BLOCK.finditer(content):
        if any(s[0] <= m.start() < s[1] for s in spans):
            continue  # already consumed by the JSON pass (fenced example)
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            continue
        if not isinstance(args, dict):
            continue
        found.append({
            "id": _new_id(),
            "type": "function",
            "function": {"name": m.group(1).strip(),
                         "arguments": json.dumps(args, ensure_ascii=False)},
        })
        spans.append(m.span())

    return found


# ---------------------------------------------------------------------------
# WP8 mechanism 1: repeated stop-token stripping
# ---------------------------------------------------------------------------

#: Box-drawing / separator junk models interleave with stop runs.
_BAND_CHARS = "\u02c1\u00a6"  # double vertical bar, broken bar

#: Names that mark a stop/end-of-turn sequence when wrapped in either
#: bracket shape (pipe-delimited or bare-angle). Bare "stop"/"end" words
#: WITHOUT brackets are only ever cut via the acknowledged-word pass
#: (separator-guarded, end-anchored) -- prose like "please stop." survives.
_STOP_NAMES = ("im_end", "endoftext", "eos_token", "eos", "eot_id", "eot",
               "eom", "finish", "complete", "end", "stop")


def _bracketed(name: str) -> str:
    """Regex for NAME in both the pipe-delimited and bare-angle shape.

    The name sits inside the brackets only -- prose like "end > 5" or a
    comparison chain can never match, which is what keeps mid-content
    text byte-identical. The PIPE form alternates FIRST (order matters
    for readability; both shapes match).
    """
    pipe = chr(92) + BAR  # escaped pipe for regex use
    return (LT + r"\s*(?:" + pipe + r"\s*" + re.escape(name) + r"\s*" + pipe
            + r"|" + re.escape(name) + r")\s*" + GT)


#: Acknowledged bracketed word markers -- band material wherever they
#: trail ("[DONE]", the angled STOP form).
_ACK_BRACKET_ALT = (LT + r"\s*STOP\s*" + GT + r"|\[DONE\]|\[done\]")

#: Bracketed atoms ONLY (no separator-character alternatives): stop-name
#: markers, closing role headers, acknowledged bracketed markers. Kept
#: separate from the separator junk because the leaked-markup residue scan
#: must never delete characters that are PART of a marker name (the "_" of
#: an im_start marker) -- stripping bare separators there would make a
#: structural leak invisible to the scan.
_BRACKETED_ATOMS = (
    "|".join("(?:" + _bracketed(n) + ")" for n in
             sorted(_STOP_NAMES, key=len, reverse=True))
    # NOTE: only CLOSING role markers ("</assistant>") are band material.
    # An OPENING "<assistant>" is live turn structure -- models that leak
    # one are broken (same posture as the leaked chat-template gate) and
    # stripping it would rewrite conversation structure, not stop junk.
    + "|" + LT + r"\s*/\s*(?:assistant|human|user|system)\s*>"
    + "|" + "(?:" + _ACK_BRACKET_ALT + ")"
)

#: One stop-band atom: bracketed stop names, turn-role headers, separator
#: junk, acknowledged bracketed markers. All shapes REQUIRE brackets --
#: no bare word can ever be a band atom.
_BAND_ATOMS = (
    "(" + "|".join(re.escape(c) for c in _BAND_CHARS) + ")"
    + r"|" + r"[*_~#=-]+"
    + "|" + _BRACKETED_ATOMS
)

#: Band atoms only (no whitespace): marks candidate band material.
_ATOM_RE = re.compile(r"(?:" + _BAND_ATOMS + r")", re.I)

#: Max strip passes (pathological-input guard; the loop is monotone so
#: this never bites real output -- it bounds adversarial input).
MAX_STRIP_ITERATIONS = 64

#: Bound on how much of the tail the band matcher examines per pass.
_BAND_SCAN_LIMIT = 4096

#: Bare acknowledged words: cut at the tail ONLY when separator-guarded
#: and end-anchored -- the word must start at line start or right after
#: a newline / band char / sentence punctuation, never after a plain
#: space. "the bus will not stop" keeps "stop"; "done. stop" sheds it.
_ACK_WORDS = ("end of turn", "End of Turn", "stop", "Stop", "STOP")
_ACK_WORD_RE = re.compile(
    r"(?:^|(?<=[\n\t\r*" + re.escape(_BAND_CHARS) + r"#!~>.!?])[ \t]*)"
    + "(?:" + "|".join(re.escape(w) for w in _ACK_WORDS) + r")[\s.!]*\Z")

#: A band strip can leave a bare stop word exposed at the new tail (the run
#: sat between prose and a trailing "STOP"). That follow-up cut must stay
#: END-ANCHORED like the standalone pass above -- an unanchored variant
#: would happily match "stop" in the middle of text and then delete
#: everything after it, which is exactly the mid-content damage the
#: preservation rules forbid.
_ACK_TAIL_RE = re.compile(
    r"(?:^|(?<=[\n\t\r*" + re.escape(_BAND_CHARS) + r"#!~>.!?])[ \t]*)"
    + "(?:" + "|".join(re.escape(w) for w in _ACK_WORDS) + r")[\s.!]*\Z")


#: Hard bounds on one band run (adversarial + false-positive guards): how
#: many marker atoms a tail may contain and how much NON-marker junk may
#: sit between/around them. A wall of markers, a text made of separator
#: characters, or a stop token followed by real prose all fail the scan and
#: are kept byte-for-byte; the caps also keep the walk linear.
MAX_BAND_ATOMS = 4096
MAX_BAND_JUNK_CHARS = 24


#: Bracketed-only atoms pattern (see _BRACKETED_ATOMS) for the residue scan.
_BRACKETED_ATOM_RE = re.compile(r"(?:" + _BRACKETED_ATOMS + r")", re.I)

#: Non-overlapping atom spans are the ONLY reliable primitive for the
#: right-to-left walk: ``Pattern.match`` ignores its positional start, and a
#: bounded slice lets an inner separator character of a marker (the ``_`` of
#: an end-of-turn marker) match as an atom on its own -- which makes one
#: marker look like many atoms and lets the walk eat real text sitting
#: between them. One linear pass over the tail window gives exact spans.
#: Legal whitespace padding between the last stop marker and end of text.
_TAIL_PAD_RE = re.compile(r"[ \t\r\n]*\Z")


def _atom_spans(text: str, lo: int,
                atoms: "re.Pattern[str]") -> List[Tuple[int, int]]:
    """Sorted non-overlapping (start, end) atom spans inside [lo, len)."""
    return [(m.start(), m.end()) for m in atoms.finditer(text, lo)]


def _is_sep_atom(text: str, start: int, end: int) -> bool:
    """True iff the atom span is pure separator junk (no bracket shape).

    Separator characters are band material ONLY as decoration around real
    markers; on their own they occur inside identifiers ("todo_read"), in
    markdown and in prose. A run that had to step over real content to
    reach a bare separator is therefore not a stop band any more, and the
    separator stays where it is. Bracketed atoms always contain '<' or '['.
    """
    seg = text[start:end]
    return "<" not in seg and "[" not in seg


def _trail_scan(text: str, lo: int, atoms: "re.Pattern[str]") -> int:
    """Earliest index >= lo whose whole suffix to EOF is band atoms +
    whitespace + short junk runs, or -1.

    Right-to-left walk over the tail window, driven by exact atom spans:
    an atom ending exactly at the walk position is consumed (the position
    jumps to its start); otherwise the walk steps back one character and
    counts it as junk if it is not whitespace. Junk is capped, so real
    content inside or before the run ends the scan and nothing is stripped
    -- a stop-shaped sequence in the MIDDLE of text can never be reached,
    because the band must be anchored at end of text.

    Cost: one linear atom pass over the window plus one step per character
    (both capped by ``_BAND_SCAN_LIMIT``), with hard caps of
    ``MAX_BAND_ATOMS`` atoms and ``MAX_BAND_JUNK_CHARS`` junk characters.
    """
    n = len(text)
    if n <= lo:
        return -1
    spans = _atom_spans(text, lo, atoms)
    i = len(spans) - 1
    p = n                     # walk position (exclusive upper edge)
    # A tail band is anchored at EOF: either an atom ends exactly at EOF,
    # or legal whitespace padding sits between the last atom and EOF.
    anchored = not spans or bool(_TAIL_PAD_RE.match(text, spans[-1][1]))
    start = -1                # leftmost atom start of the run so far
    atoms_seen = 0
    junk = 0
    stepped = False           # non-whitespace stepped since last consume
    while True:
        while i >= 0 and spans[i][1] > p:
            i -= 1            # spans that end past the walk position
        if i >= 0 and spans[i][1] == p:
            # Separator-only atoms (asterisk/dash/pipe junk) never extend a
            # run ACROSS real content: reaching one required stepping over
            # non-whitespace, so it is part of that content (the "_" inside
            # "todo_read" is a separator atom -- consuming it would cut the
            # identifier in half). Markers are still allowed within the junk
            # cap; separators only chain through whitespace.
            if stepped and _is_sep_atom(text, spans[i][0], spans[i][1]):
                break
            start = spans[i][0]
            p = start
            atoms_seen += 1
            i -= 1
            stepped = False
            if atoms_seen > MAX_BAND_ATOMS:
                return -1     # pathological marker wall: refuse to strip
            continue
        if p <= lo:
            break             # window exhausted
        if text[p - 1].strip():
            junk += 1
            stepped = True
            if junk > MAX_BAND_JUNK_CHARS:
                # Real content inside the run: stop extending left HERE.
                # Atoms already consumed sit right of this point, so what
                # remains (start..EOF) is still an EOF-anchored band with a
                # bounded amount of interleaved junk -- long prose BEFORE
                # the band never blocks stripping; content BETWEEN markers
                # caps how far back the run may reach.
                break
        p -= 1
    if start < 0:
        return -1             # no atom at all in the tail window
    if not anchored:
        # The walk ran into real text before it ever found an atom ending at
        # EOF (or after EOF, past legal whitespace padding): the markers sit
        # MID-text, so nothing here is a stop tail.
        return -1
    if p <= lo and lo > 0:
        # The run reached the left edge of the scan window: it would have to
        # eat past 4 KB of real content, which the mid-content preservation
        # rule forbids.
        return -1
    return start


def _band_match_suffix(text: str, extra: Tuple[str, ...]) -> int:
    """Length of the stop-band suffix of *text* (0 when the tail is clean).

    Bounded to the last ``_BAND_SCAN_LIMIT`` chars; the strip point steps
    back over whitespace preceding the first band atom.
    """
    if not text:
        return 0
    lo = max(0, len(text) - _BAND_SCAN_LIMIT)
    if extra:
        key = tuple(sorted(extra, key=len, reverse=True))
        atoms = _ATOM_EXTRA_CACHE.get(key)
        if atoms is None:
            atoms = re.compile(
                r"(?:" + _BAND_ATOMS + "|"
                + "|".join(re.escape(t) for t in key) + r")", re.I)
            _ATOM_EXTRA_CACHE[key] = atoms
    else:
        atoms = _ATOM_RE
    n = len(text)
    start = _trail_scan(text, lo, atoms)
    if start < 0:
        return 0
    while start > lo and text[start - 1] in " \t\r\n":
        start -= 1
    cut = n - start
    return cut if cut > 0 else 0


_ATOM_EXTRA_CACHE: Dict[Tuple[str, ...], "re.Pattern[str]"] = {}


def _ack_suffix_cut(text: str) -> int:
    """Suffix length of an acknowledged literal stop word at the tail.

    The cut may never cross a band marker: text like
    "see <im_end> inline in middle, keep me" ends in a shape the word regex
    could reach only by deleting real content that sits AFTER a mid-text
    marker, which is exactly what the mid-content preservation rule
    forbids. So a candidate whose window contains a band atom is refused --
    unless that atom run itself forms the whole tail (a stop band followed
    by a trailing stop word, which IS strippable junk).
    """
    m = _ACK_WORD_RE.search(text)
    if not m:
        return 0
    lo = max(0, m.start() - _BAND_SCAN_LIMIT)
    if _ATOM_RE.search(text, lo, m.start()):
        # Something marker-shaped precedes the word inside the tail window.
        # Only strip when everything from that marker run to EOF is junk:
        # band atoms + whitespace + the stop word itself, nothing else.
        start = _trail_scan(text, lo, _ATOM_RE)
        if start < 0 or start > m.start():
            return 0
        gap = text[start:m.start()] + text[m.end():]
        if gap.strip(" \t\r\n"):
            # non-whitespace between atoms / after the word that is not the
            # acknowledged word itself -> real content, keep everything
            residual = text[start:m.start()]
            if residual.strip(" \t\r\n"):
                return 0
        return len(text) - start
    return len(text) - m.start()


def strip_stop_tokens(text: str,
                      extra_tokens: Iterable[str] = ()) -> str:
    """Strip repeated trailing stop sequences until the text is stable.

    Handles what small GGUFs actually emit at the tail: repeated end-of-turn
    / end-of-document control markers of the pipe-delimited and bare-angle
    families, junk interleaved between them (box-drawing pipes, asterisks,
    dash runs), duplicated assistant-turn header lines, acknowledged
    bracketed word markers ("[DONE]", the angled stop form) and
    separator-guarded trailing stop words. Stripping repeats until a fixed
    point (capped at ``MAX_STRIP_ITERATIONS``) so "tok tok tok" and
    "tok * tok" runs collapse in one call.

    Preservation guarantees (what keeps already-handled inputs
    byte-identical):

      * only the *tail* is examined -- a stop-shaped sequence in the
        middle of prose or code is never touched;
      * a text whose tail is not stop-shaped comes back unchanged, byte
        for byte (bare "stop" anywhere in ordinary prose survives;
        "end > 5" comparisons survive; mid-text markers survive);
      * a trailing colon or comma is never eaten (lead-ins such as
        "arguments:" must survive for later passes);
      * ``extra_tokens`` are caller-opt-in shapes treated as band atoms
        at the tail only.
    """
    if not text or not isinstance(text, str):
        return text or ""
    extra = tuple(t for t in (extra_tokens or ()) if isinstance(t, str) and t)
    if (LT not in text and "[" not in text and not extra
            and not _ACK_WORD_RE.search(text)):
        return text  # fast path: no band bracket or ack-word shape
    out = text
    for _ in range(MAX_STRIP_ITERATIONS):
        prev = out
        n = _band_match_suffix(out, extra)
        if n and n < len(out):
            out = out[:len(out) - n].rstrip()
        elif n:
            out = ""  # whole text was one stop band
            break
        n2 = _ack_suffix_cut(out)
        if n2 and n2 < len(out):
            out = out[:len(out) - n2].rstrip()
        if out == prev:
            break
    # Punctuation a stop run left behind (":" and "," are preserved).
    if len(out) != len(text) and out and out[-1] in ".!?":
        out = out[:-1].rstrip()
    return out


# ---------------------------------------------------------------------------
# WP8 mechanism 2: JSON repair pipeline
# ---------------------------------------------------------------------------

#: Canonical repair tags. Every recovered call is attributable to the exact
#: set of repairs that made it loadable.
REPAIR_TAGS = (
    "stop_tokens",        # repeated trailing stop sequences stripped
    "unclosed_fence",     # ``` fence opened but never closed
    "bare_json_prose",    # un-fenced JSON embedded in prose
    "xml_tool_wrapper",   # tool>name{json}</tool> style wrapper
    "xml_parameters",     # parameter-tag argument bodies
    "py_literals",        # Python True/False/None -> true/false/null
    "single_quotes",      # single-quoted strings -> double-quoted
    "trailing_commas",    # comma before } or ] removed
    "truncated_close",    # unbalanced EOF: dangling key/value trimmed + closed
)

#: Candidate-start caps keep the recovery passes linear on adversarial input.
MAX_BARE_CANDIDATES = 16
MAX_CANDIDATE_SPAN = 40000


_OPEN_FENCE_RE = re.compile(
    r"```(?:json|tool_call)?\s*([\[{].*)\Z", re.S | re.I)

# Fenced call whose payload is followed by stop junk BEFORE the fence close
# ("}```"). The junk run must be entirely band atoms/whitespace, so this
# pass never fires on fenced prose or clean fences (junk group >= 1).
_FENCE_STOPJUNK_RE = re.compile(
    r"```(?:json|tool_call)?[ \t]*\r?\n?[ \t]*([\[{].*?[\]}])"
    r"((?:[ \t\r\n]|(?:" + _BAND_ATOMS + r"))+)" + "`{3}",
    re.S | re.I)

# <tool>name {json}</tool> and friends (tool/function wrappers -- the
# legacy recognizer only knows tool_call/tool_use with clean JSON).
_XML_TOOL_RE = re.compile(
    LT + r"\s*(?:tool|function)\s*>[^<>{}]*?"
    r"(?:name\s*=\s*)?([A-Za-z_][\w.\-]*)\s*(\{.*?\})\s*"
    r"(?:" + LT + r"\s*/\s*(?:tool|function)\s*>|$)",
    re.S,
)
# Attribute-name wrappers: <tool_call name="read_file" ...>body</...>,
# and the Gemini-style <function=name> form. Both end-anchored with the
# close tag OPTIONAL (models truncate them at EOS).
_XML_ATTR_RE = re.compile(
    LT + r"\s*(?:function|tool|tool_call|invoke)\s+(?:name|fn)\s*=\s*"
    r"[\"']?([A-Za-z_][\w.\-]*)[\"']?\s*" + GT + r"(.*?)"
    + r"(?:" + LT + r"\s*/\s*(?:function|tool|tool_call|invoke)\s*>|$)",
    re.S,
)
_XML_EQ_RE = re.compile(
    LT + r"\s*(?:function|tool)\s*=\s*([A-Za-z_][\w.\-]*)\s*" + GT
    + r"(.*?)"
    + r"(?:" + LT + r"\s*/\s*(?:function|tool)\s*>|$)",
    re.S,
)
_XML_PARAM_RE = re.compile(
    LT + r"\s*parameter\s+name\s*=\s*[\"']?([\w.\-]+)[\"']?\s*>?"
    r"\s*(.*?)\s*" + LT + r"\s*/\s*parameter\s*>",
    re.S,
)


def _scalar_or_str(raw: str) -> Any:
    """Coerce a parameter body: JSON scalars keep type, else plain string."""
    t = raw.strip()
    if not t:
        return ""
    try:
        v = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        return raw
    return v if isinstance(v, (int, float, bool, str)) else raw


def _name_key() -> Tuple[str, ...]:
    return ("name", "tool", "tool_name", "function")


def _obj_name(obj: Dict[str, Any]) -> Optional[str]:
    f = obj.get("function")
    fn = f if isinstance(f, dict) else None
    name = (fn or {}).get("name") or next(
        (v for k in ("name", "tool", "tool_name")
         if isinstance((v := obj.get(k)), str) and v.strip()), None)
    return name if isinstance(name, str) and name.strip() else None


def _looks_like_call(obj: Any) -> bool:
    """Attribution guard: dict that claims to be a tool invocation.

    Requires a non-empty name under name/tool/tool_name/function AND either
    an arguments-ish key -- so a plain data dict like {"path": "a.py"}
    pasted in prose is NEVER rewritten, matching the legacy posture.
    """
    if not isinstance(obj, dict):
        return False
    if not _obj_name(obj):
        return False
    f = obj.get("function")
    fn = f if isinstance(f, dict) else {}
    return any(k in obj or k in fn
               for k in ("arguments", "parameters", "args"))


def normalize_tool_name(name: str) -> str:
    """Canonical form for tool-name matching: lowercase, underscored.

    "Read File", "read-file", "ReadFile" (as-is, camel stays) all fold to
    "read_file" only when the FOLDED spellings are equal -- the match is
    EXACT on this normal form; there is deliberately no edit-distance or
    prefix stage (see _match_known: a typo like "serch_files" must fail
    closed, because a mis-routed tool call is worse than a refused one).
    """
    n = re.sub(r"[\s\-./]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", n).strip("_")


def _match_known(name: str,
                 known: Optional[Iterable[str]]
                 ) -> Tuple[Optional[str], bool]:
    """Exact-normalized match of *name* against *known*.

    Returns (canonical_name_or_None, ambiguous). Resolution order:
      1. exact case-folded equality (BEFORE normal-form ambiguity: a known
         name spelled exactly is that name, even if other known names fold
         to the same normal form);
      2. unique normal-form match;
      3. ambiguous normal form (two known names fold together) -> refusal.
    NO fuzzy/edit distance -- unique hit or refusal, never a guess (a
    mis-routed tool call executes the wrong thing; a refused one just
    re-prompts the model).
    """
    if known is None:
        return None, False
    names = list(known)
    lowered = name.strip().lower()
    exact = sorted({n for n in names if n.lower() == lowered})
    want = normalize_tool_name(name)
    uniq = sorted({n for n in names if normalize_tool_name(n) == want})
    # Ambiguity dominates: if two DIFFERENT known names fold to the same
    # normal form, a request matching that form cannot be routed without
    # guessing -- refuse even when one candidate also matches exactly
    # case-folded ("READ_FILE" against {"read_file", "Read File"}).
    if len(uniq) > 1:
        return None, True
    if len(exact) == 1:
        return exact[0], False
    if len(exact) > 1:
        return None, True
    if len(uniq) == 1:
        return uniq[0], False
    return None, False


def _call_from_named(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": _new_id(),
        "type": "function",
        "function": {"name": name,
                     "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _calls_from_obj(obj: Any,
                    known: Optional[Iterable[str]]
                    ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Tool calls from one parsed candidate object (+ name-normalization).

    Same acceptance rules as the legacy _call_from_obj (name required,
    args optional/default {}), plus exact-normalized name routing when a
    known-tool list is supplied: unknown or ambiguous names are REFUSED
    (no calls) rather than emitted mis-routed.
    """
    tags: List[str] = []
    if isinstance(obj, list):
        out: List[Dict[str, Any]] = []
        for item in obj:
            calls, t = _calls_from_obj(item, known)
            if not calls:
                return [], []  # all-or-nothing, like the legacy fence pass
            out.extend(calls)
            for tg in t:
                if tg not in tags:
                    tags.append(tg)
        return out, tags
    if not isinstance(obj, dict):
        return [], []
    call = _call_from_obj(obj)
    if call is None:
        return [], []
    name = call["function"]["name"]
    if known is not None:
        canonical, ambiguous = _match_known(name, known)
        if canonical is None:
            return [], []  # unknown/ambiguous => refuse, never mis-route
        if canonical != name:
            call["function"]["name"] = canonical
            if "name_normalized" not in tags:
                tags.append("name_normalized")
    return [call], tags


# ---------------------------------------------------------------------------
# JSON repair engine
# ---------------------------------------------------------------------------

_PY_LIT_RE = re.compile(r"(?<![\w.\"])'\b(True|False|None)\b'?"
                        r"|\b(True|False|None)\b(?![\w.\"])")


def _sub_py_literals(text: str) -> Tuple[str, bool]:
    """Replace Python True/False/None with JSON forms, OUTSIDE strings."""
    out: List[str] = []
    in_str: Optional[str] = None
    esc = False
    changed = False
    i = 0
    mapping = {"True": "true", "False": "false", "None": "null"}
    while i < len(text):
        c = text[i]
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == in_str:
                in_str = None
        elif c in "\"'":
            in_str = c
            out.append(c)
        else:
            m = re.match(r"(True|False|None)(?![\w])", text[i:])
            if m:
                out.append(mapping[m.group(1)])
                i += len(m.group(1))
                changed = True
                continue
            out.append(c)
        i += 1
    return "".join(out), changed


def _drop_trailing_commas(text: str) -> Tuple[str, bool]:
    """Remove commas that directly precede a closing } / ] (outside strings)."""
    out: List[str] = []
    in_str: Optional[str] = None
    esc = False
    changed = False
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == in_str:
                in_str = None
        elif c in "\"'":
            in_str = c
            out.append(c)
        elif c == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                changed = True
                i = j
                continue
            out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out), changed


def _requote_single(text: str) -> Tuple[str, bool]:
    """Single-quoted strings -> double-quoted (outside double-quoted runs)."""
    out: List[str] = []
    i = 0
    n = len(text)
    changed = False
    while i < n:
        c = text[i]
        if c == '"':
            out.append(c)
            i += 1
            while i < n:
                if text[i] == "\\":
                    out.append(text[i:i + 2])
                    i += 2
                    continue
                out.append(text[i])
                if text[i] == '"':
                    i += 1
                    break
                i += 1
        elif c == "'":
            j = i + 1
            buf: List[str] = []
            closed = False
            while j < n:
                if text[j] == "\\":
                    buf.append(text[j:j + 2])
                    j += 2
                    continue
                if text[j] == "'":
                    closed = True
                    j += 1
                    break
                buf.append(text[j])
                j += 1
            if not closed:
                # apostrophe mid-prose (or unterminated): leave it alone
                out.append(c)
                i += 1
                continue
            body = "".join(buf)
            # unescape \' and keep \" intact; escape raw double quotes
            body = body.replace("\\'", "'").replace('\\"', '"')
            body = body.replace('"', '\\"')
            out.append('"' + body + '"')
            i = j
            changed = True
        else:
            out.append(c)
            i += 1
    return "".join(out), changed


def _close_truncated(text: str) -> Tuple[str, bool]:
    """Close unbalanced braces/brackets (and unterminated strings) at EOF.

    Dangling ``"key":`` / ``"key": 12,`` fragments after the last complete
    value are trimmed so closing does not fabricate half a pair. Only
    touches text that is genuinely unbalanced.
    """
    stack: List[str] = []
    in_str: Optional[str] = None
    esc = False
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == in_str:
                in_str = None
        elif c in "\"'":
            in_str = c
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]":
            if stack:
                stack.pop()
        i += 1
    if not stack and not in_str:
        return text, False
    t = text
    if in_str:
        # terminate the open string first, then re-check the tail
        t = t + ("'" if in_str == "'" else '"')
    if in_str:
        # The final string was cut mid-value: its value is INCOMPLETE, so
        # drop the whole last key:value pair rather than fabricate a value
        # from the fragment (closing the quote then keeping it would turn
        # "hal" into the answer to "path").
        t = re.sub(r"(?<=[{,])\s*\"(?:[^\"\\]|\\.)*\"\s*:\s*"
                   r"(?:\"(?:[^\"\\]|\\.)*\"?|[^,{}\[\]]*)\Z", "", t)
    # dangling key (string + optional colon at the tail): drop it
    t = re.sub(r"(?<=[{,])\s*\"(?:[^\"\\]|\\.)*\"\s*:?\s*$",
               "", t)
    t = re.sub(r",\s*$", "", t.rstrip())
    closers = "".join(reversed(stack))
    return t + closers, True


def _repair_json(text: str) -> Tuple[Any, List[str]]:
    """Parse *text* as JSON, applying repairs in fixed order until it loads.

    Order (each step re-attempts the load, tags accumulate): python
    literals -> trailing commas -> single quotes -> truncated close. The
    cascade is a constant (four) number of linear passes -- never a fix-
    point loop -- so cost stays bounded on adversarial input.
    """
    try:
        return json.loads(text), []
    except (json.JSONDecodeError, ValueError):
        pass
    tags: List[str] = []
    t = text
    for fn, tag in ((_sub_py_literals, "py_literals"),
                    (_drop_trailing_commas, "trailing_commas"),
                    (_requote_single, "single_quotes"),
                    (_close_truncated, "truncated_close")):
        t, changed = fn(t)
        if not changed and tag != "truncated_close":
            continue
        if changed:
            tags.append(tag)
        try:
            return json.loads(t), tags
        except (json.JSONDecodeError, ValueError):
            continue
    return None, []


# ---------------------------------------------------------------------------
# WP8 mechanism 3: recovery pipeline + attributable parse report
# ---------------------------------------------------------------------------

@dataclass
class ParseReport:
    """Outcome of one (possibly repairing) parse.

    ``calls`` is exactly what the public parser returns (legacy wire shape,
    no extra keys). ``repairs_by_call_id`` maps each call's id -> the repair
    tags that were needed to obtain it (empty list = clean legacy parse);
    ids are unique per call, making this side-channel safe for consumers
    that only ever look at ``calls``. ``recovered`` is True when the legacy
    recognizer found nothing and a repair path produced these calls.
    """

    calls: List[Dict[str, Any]] = field(default_factory=list)
    repairs_by_call_id: Dict[str, List[str]] = field(default_factory=dict)
    recovered: bool = False

    def tags_for(self, call: Dict[str, Any]) -> List[str]:
        return list(self.repairs_by_call_id.get(call.get("id", ""), ()))


def _balanced_slice(text: str, start: int) -> Tuple[Optional[str], bool]:
    """String-aware balanced {...} slice from *start* ('{' position).

    Returns (slice, truncated). Linear in the span; the span itself is
    capped at MAX_CANDIDATE_SPAN so a pathological nesting-free wall of
    braces costs at most the cap per candidate.
    """
    limit = min(len(text), start + MAX_CANDIDATE_SPAN)
    if start >= len(text) or text[start] != "{":
        return None, True
    depth = 0
    in_str = False
    esc = False
    j = start
    while j < limit:
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return text[start:j + 1], False
        j += 1
    tail = text[start:limit]
    return (tail, True) if len(tail) > 1 else (None, True)


def _bare_json_candidates(text: str) -> List[Tuple[str, List[str]]]:
    """Walk un-fenced '{' starts in *text*, string-aware, bounded two ways
    (candidate count AND total scanned chars -- adversarial brace walls
    stay linear). The scan skips braces inside string literals so prose
    quotes/apostrophes cannot masquerade as JSON structure.
    """
    out: List[Tuple[str, List[str]]] = []
    i = 0
    in_str = False
    esc = False
    n = len(text)
    while i < n and len(out) < MAX_BARE_CANDIDATES:
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            i += 1
            continue
        if c == "{":
            slice_, trunc = _balanced_slice(text, i)
            if slice_ is not None:
                tags = ["bare_json_prose"]
                if trunc:
                    tags.append("truncated_candidate")
                out.append((slice_, tags))
            i += 1
            continue
        i += 1
    return out


def _xml_variants(text: str) -> List[Tuple[Optional[str], Optional[str],
                                           Optional[str], List[str]]]:
    """(args_raw, name, body, tags) for each XML-ish wrapper candidate."""
    out: List[Tuple[Optional[str], Optional[str], Optional[str],
                    List[str]]] = []
    for m in _XML_TOOL_RE.finditer(text):
        out.append((m.group(2), m.group(1), None, ["xml_tool_wrapper"]))
    for m in _XML_ATTR_RE.finditer(text):
        out.append((None, m.group(1), m.group(2), ["xml_tool_wrapper"]))
    for m in _XML_EQ_RE.finditer(text):
        out.append((None, m.group(1), m.group(2), ["xml_tool_wrapper"]))
    return out


def _params_to_args(body: str) -> Tuple[Dict[str, Any], bool]:
    """parameter-tag bodies -> args dict. (dict, found_any)."""
    args: Dict[str, Any] = {}
    found = False
    for m in _XML_PARAM_RE.finditer(body):
        args[m.group(1)] = _scalar_or_str(m.group(2) or "")
        found = True
    return args, found


def _markup_residue_clean(text: str) -> bool:
    """True iff every leaked chat-template marker in *text* sits inside a
    contiguous band-atom run (stop junk), i.e. nothing markup-shaped remains
    once all band atoms are removed.

    This refines the legacy blanket refusal for the WP8 recovery path only:
    a model that emits pipe-delimited end-of-turn markers as (repeated) STOP
    JUNK is weak at function calling -- exactly what recovery exists for --
    while a model leaking template STRUCTURE stays refused: im_start is not
    a band atom, so it survives the residue scan and keeps its turn headers
    intact. parse_legacy keeps the old blanket refusal unchanged; this gate
    only decides whether the RECOVERY passes may run on such text.
    """
    if _BRACKETED_ATOM_RE.search(text) is None:
        return False  # no atoms at all => residue == text => still dirty
    residue = _BRACKETED_ATOM_RE.sub("", text)
    return _SPECIAL_MARKUP.search(residue) is None


def recover(content: Optional[str],
            known_tool_names: Optional[Iterable[str]] = None,
            ) -> ParseReport:
    """Repair-first parse with full attribution. NEVER used when the legacy
    recognizer already produced calls -- the caller (parse_text_tool_calls)
    enforces the ordering, so recovery can only ADD calls for inputs that
    previously yielded []. Pass-order (first pass producing calls wins):

      stop_tokens  : strip trailing stop sequences, retry legacy verbatim
      fenced loads : legacy fence spans, JSON-repaired
      unclosed     : fence opened but never closed (repair + close)
      xml          : <tool>-style wrappers / parameter-tag bodies
      bare prose   : balanced (or truncated) JSON objects in prose

    Every emitted call carries its repair tags in the report. Name routing
    is exact-normalized only, and unknown/ambiguous names are refused when
    a known-tool list is supplied (documented conservatism: a typo like
    "serch_files" is NOT routed).
    """
    empty = ParseReport()
    if not content or not isinstance(content, str):
        return empty
    if _SPECIAL_MARKUP.search(content) and not _markup_residue_clean(content):
        return empty  # leaked template STRUCTURE => never rewrite (legacy)

    text = content
    carry: List[str] = []
    stripped = strip_stop_tokens(content)
    if stripped != content:
        calls = parse_legacy(stripped)
        if calls:
            return ParseReport(
                calls=calls,
                repairs_by_call_id={c["id"]: ["stop_tokens"] for c in calls},
                recovered=True)
        text = stripped
        carry = ["stop_tokens"]

    def _emit(calls: List[Dict[str, Any]],
              tags: List[str]) -> ParseReport:
        merged = list(carry) + [t for t in tags if t not in carry]
        return ParseReport(
            calls=calls,
            repairs_by_call_id={c["id"]: list(dict.fromkeys(merged))
                                for c in calls},
            recovered=True)

    def _try(raw: str, tags: List[str]
             ) -> Optional[ParseReport]:
        # second variant: candidate-internal trailing stop junk (between
        # the closing brace and the fence close, e.g. ``}````) --
        # attributed via the stop_tokens tag.
        variants: List[Tuple[str, Tuple[str, ...]]] = [(raw, ())]
        cleaned = strip_stop_tokens(raw)
        if cleaned != raw:
            variants.append((cleaned, ("stop_tokens",)))
        # fence/parameter payloads end in a quote or brace -- a stop shape
        # trailing THERE is payload-internal junk (models emit it between
        # the closing brace and the fence close). Prose ends in a word,
        # so this heuristic never touches prose candidates.
        if raw and raw[-1] in "\"'}]":
            cut = _band_match_suffix(raw, ())
            if not cut:
                # stop run followed by junk: probe with a synthetic marker
                # tail so the band matcher can reach across the junk.
                probe = IM_END_TAIL + IM_END_TAIL
                full = _band_match_suffix(raw + probe, ())
                cut = (full - len(probe)) if full > len(probe) else 0
            if 0 < cut < len(raw):
                c2 = raw[:len(raw) - cut].rstrip()
                if c2 and all(v[0] != c2 for v in variants):
                    variants.append((c2, ("stop_tokens",)))
        for variant, extra in variants:
            obj, rtags = _repair_json(variant)
            if obj is None:
                continue
            if not isinstance(obj, list) and not _looks_like_call(obj):
                continue
            calls, ntags = _calls_from_obj(obj, known_tool_names)
            if not calls:
                continue
            return _emit(calls, tags + list(extra) + rtags + ntags)
        return None

    # pass A: legacy fence spans whose payload merely failed to load.
    for m in _JSON_BLOCK.finditer(text):
        rep = _try(m.group(1), [])
        if rep is not None:
            return rep
    # pass A2: fenced call with stop junk between the payload and the
    # fence close ("}```"). The junk group is band-only by construction,
    # so this never fires on clean fences or fenced prose.
    for m in _FENCE_STOPJUNK_RE.finditer(text):
        rep = _try(m.group(1), ["stop_tokens"])
        if rep is not None:
            return rep
    # pass B: unclosed fence (``` opened, never closed).
    mo = _OPEN_FENCE_RE.search(text)  # distinct name: `m` above is Match[str]
    if mo is not None:
        rep = _try(mo.group(1), ["unclosed_fence"])
        if rep is not None:
            return rep
    # pass C: XML-ish wrappers.
    for args_raw, name, body, tags in _xml_variants(text):
        args: Optional[Dict[str, Any]] = None
        extra: List[str] = []
        if args_raw is not None:
            obj, rtags = _repair_json(args_raw)
            if isinstance(obj, dict):
                args = obj
                extra = rtags
        elif body:
            pargs, found = _params_to_args(body)
            if found:
                args = pargs
                extra = ["xml_parameters"]
        if args is None:
            continue
        calls, ntags = _calls_from_obj(
            {"name": name, "arguments": args}, known_tool_names)
        if calls:
            return _emit(calls, tags + extra + ntags)
    # pass D: bare JSON in prose.
    for slice_, tags in _bare_json_candidates(text):
        rep = _try(slice_, tags)
        if rep is not None:
            return rep
    # pass E: truncated bare JSON anywhere (balanced scan never failed but
    # the object may end at EOF inside a string -- try from each '{').
    if _looks_like_trailing_call(text):
        for slice_, tags in _bare_json_candidates(strip_stop_tokens(text)):
            rep = _try(slice_, tags)
            if rep is not None:
                return rep
    return empty


def _looks_like_trailing_call(text: str) -> bool:
    """Cheap gate for pass E: a name key near the end of the text."""
    tail = text[-400:]
    return ('"name"' in tail or '"tool"' in tail or '"arguments"' in tail)


def parse_text_tool_calls_with_repairs(
        content: Optional[str],
        known_tool_names: Optional[Iterable[str]] = None,
) -> ParseReport:
    """Parse assistant text, legacy-identical when legacy succeeds.

    ``known_tool_names`` (schemas' function names) additionally enables
    exact-normalized name routing for RECOVERED calls; when omitted,
    recovered names pass through unvalidated (legacy never validated).
    """
    base = parse_legacy(content)
    if base:
        return ParseReport(calls=base, repairs_by_call_id={},
                           recovered=False)
    if not content or not isinstance(content, str):
        return ParseReport()
    return recover(content, known_tool_names)


def parse_text_tool_calls(
        content: Optional[str],
        known_tool_names: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract tool calls from assistant text; [] when there are none.

    Byte-identical to the legacy parser on everything the legacy parser
    accepted (parse_legacy runs first and short-circuits); additionally
    recovers common corruptions (see recover()). Use
    parse_text_tool_calls_with_repairs when repair attribution matters.
    """
    return parse_text_tool_calls_with_repairs(
        content, known_tool_names).calls


def strip_tool_call_markup(content: str, extended: bool = False) -> str:
    """Remove parsed tool-call blocks from the visible text.

    Keeps any prose the model wrote around them (that part is real content);
    with no prose left returns "" so the loop can substitute a summary line.

    ``extended=False`` (default) is the legacy function verbatim -- byte
    identical on every input. ``extended=True`` additionally removes the
    XML wrapper shapes WP8 recovery recognizes (``<tool>`` / attribute-style
    headers and their parameter bodies); the runner enables it only for
    RECOVERED turns, so legacy turns keep the legacy strip semantics.
    """
    out = _JSON_BLOCK.sub("", content)
    out = _XML_BLOCK.sub("", out)
    if extended:
        out = _XML_TOOL_RE.sub("", out)
        out = _XML_ATTR_RE.sub("", out)
        out = _XML_EQ_RE.sub("", out)
        out = _OPEN_FENCE_RE.sub("", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()

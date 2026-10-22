"""WP8 tests: tool-call / reasoning-loop recovery hardening.

Ladder (kimi-style gate ladder, all layers in this file):

  LAYER 1  weightless text scan: every drift fixture loads, the manifest
           is valid JSON, keys and files agree exactly.
  LAYER 2  per-fixture assertions: recovery expectations (tool name, arg
           subset, repair attribution) AND the zero-regression contract
           -- anything parse_legacy() already parsed must parse to the
           identical name+arguments through the new API.
  LAYER 3  aggregate recovery-rate report (printed -- run with -s to see
           it live): baseline = what parse_legacy recovers, new = what
           parse_text_tool_calls recovers, per-class + total, improvement
           in percentage points. Strictly-more-recovered and
           zero-regression are asserted, not just printed.

Mechanism unit tests (stop-token stripping edge cases incl. mid-content
preservation, loop-detection TPs incl. interleaved think blocks and TNs
for lists/code, each repair class, exact-normalized name matching with
documented refusal policy) live above the ladder; runner-level nudge
integration tests live at the bottom. All hermetic: fixtures are plain
text, no network, no GPU.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeOpenAIServer, _Handler  # noqa: E402

from sable.core.agent_runner import SableAgentClient  # noqa: E402
from sable.core.recovery import (LOOP_MARKER, detect_reasoning_loop,
                                   truncate_at_loop)
from sable.core.tool_calls import (REPAIR_TAGS, ParseReport,
                                     normalize_tool_name, parse_legacy,
                                     parse_text_tool_calls,
                                     parse_text_tool_calls_with_repairs,
                                     strip_stop_tokens,
                                     strip_tool_call_markup)
from sable.core.tools import (CAPABILITY_TOOL_SCHEMAS, TOOL_SCHEMAS)

LT, GT, BAR = "<", ">", "|"
FENCE = "```"


def _ct(name: str) -> str:
    """Control marker of the pipe-delimited family, built piecewise."""
    return LT + BAR + name + BAR + GT


IM_END = _ct("im_end")
EOS = _ct("eos")

KNOWN = sorted({s["function"]["name"] for s in TOOL_SCHEMAS}
               | set(CAPABILITY_TOOL_SCHEMAS))

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tool_drift"


def _norm_calls(calls) -> List[tuple]:
    """(name, args-json) tuples -- ids are random per call by design."""
    return [(c["function"]["name"], c["function"]["arguments"])
            for c in calls]


def _args(calls, i=0) -> Dict[str, Any]:
    return json.loads(calls[i]["function"]["arguments"])


# ===========================================================================
# Mechanism 1: strip_stop_tokens
# ===========================================================================

@pytest.mark.parametrize("inp, want", [
    # repeated markers collapse in one call
    ("hello world" + IM_END * 3, "hello world"),
    # junk interleaved between markers (box pipes, asterisks)
    ("result *" + IM_END + " \u02c1 " + EOS + "**", "result"),
    ("done." + IM_END, "done"),
    ("line one\n" + IM_END + "\n" + IM_END + "\n", "line one"),
    # duplicated assistant-turn header shapes
    ("all done" + LT + "/assistant>", "all done"),
    # acknowledged bracketed / literal word markers
    ("Task complete" + LT + "STOP" + GT, "Task complete"),
    ("done [DONE]", "done"),
])
def test_strip_removes_trailing_stop_shapes(inp, want):
    assert strip_stop_tokens(inp) == want


@pytest.mark.parametrize("kept", [
    "The stop sign is ahead",            # bare word mid-sentence
    "the bus will not stop",             # bare word at tail, prose
    "arguments:",                        # colon lead-in survives
    "plain prose stays put.",            # untouched sentence
    "2 < 3 and 4 > 1 math is fine",      # comparisons are not brackets
    "see " + IM_END + " inline in middle, keep me",  # mid-content marker
    "code: x = 1  # " + IM_END + " comment stays",   # marker inside code
])
def test_strip_preserves_non_tail_content(kept):
    """Mid-content preservation: legacy-kept text comes back byte-identical."""
    assert strip_stop_tokens(kept) == kept


def test_strip_punctuation_only_after_real_strip():
    # sentence-final period on untouched text is kept...
    assert strip_stop_tokens("All done.") == "All done."
    # ...but punctuation a strip LEFT BEHIND goes (never colon/comma).
    assert strip_stop_tokens("All done." + IM_END) == "All done"
    assert strip_stop_tokens("args:" + IM_END) == "args:"


def test_strip_extra_tokens():
    custom = LT + "end-of-sentence" + GT
    assert strip_stop_tokens("text" + custom * 2) == "text" + custom * 2
    assert strip_stop_tokens("text" + custom * 2,
                             extra_tokens=(custom,)) == "text"


def test_strip_iterates_until_stable_and_is_idempotent():
    s = "keep" + IM_END + " junk" if False else "keep" + IM_END
    once = strip_stop_tokens(s)
    assert once == "keep"
    assert strip_stop_tokens(once) == once


def test_strip_empty_and_none_shapes():
    assert strip_stop_tokens("") == ""
    assert strip_stop_tokens(IM_END) == ""


# ===========================================================================
# Mechanism 2: repair classes + attribution
# ===========================================================================

def _report(text: str) -> ParseReport:
    return parse_text_tool_calls_with_repairs(text, known_tool_names=KNOWN)


def test_repair_single_quotes():
    calls = parse_text_tool_calls(
        FENCE + "json\n{'name': 'read_file', 'arguments': {'path': 'a'}}\n"
        + FENCE, known_tool_names=KNOWN)
    assert len(calls) == 1 and _args(calls) == {"path": "a"}


def test_repair_trailing_commas():
    rep = _report(FENCE + "json\n{\"name\": \"list_dir\", \"arguments\": "
                          "{\"path\": \"s\",},}\n" + FENCE)
    assert rep.calls and "trailing_commas" in rep.tags_for(rep.calls[0])


def test_repair_py_literals():
    rep = _report(FENCE + "json\n{\"name\": \"run_command\", \"arguments\": "
                          "{\"bg\": True, \"x\": None}}\n" + FENCE)
    assert rep.calls and _args(rep.calls)["bg"] is True
    assert _args(rep.calls)["x"] is None
    assert "py_literals" in rep.tags_for(rep.calls[0])


def test_repair_truncated_close():
    rep = _report("ok\n" + FENCE + "json\n{\"name\": \"search_files\", "
                           "\"arguments\": {\"query\": \"n\"}")
    assert rep.calls and _args(rep.calls) == {"query": "n"}
    tags = rep.tags_for(rep.calls[0])
    assert "unclosed_fence" in tags and "truncated_close" in tags


def test_repair_truncated_value_refuses_fabrication():
    # A value cut mid-string must NOT fabricate an answer from the
    # fragment: the pair is dropped, the call survives with no args.
    rep = _report(FENCE + "json\n{\"name\": \"search_files\", \"arguments\": "
                          "{\"query\": \"nee")
    assert rep.calls and _args(rep.calls) == {}


def test_repair_xml_wrappers():
    a = parse_text_tool_calls("use <tool>\nread_file {\"path\": \"x\"}\n"
                              "</tool>", known_tool_names=KNOWN)
    assert _norm_calls(a)[0][0] == "read_file"
    b = parse_text_tool_calls(
        "<tool_call name=\"list_dir\"><parameter name=\"path\">src" + LT
        + "/parameter>< /tool_call>", known_tool_names=KNOWN)
    assert _args(b) == {"path": "src"}
    c = parse_text_tool_calls(
        LT + " function=search_files>\n< parameter name=\"query\">n"
        "< /parameter>\n< /function>", known_tool_names=KNOWN)
    assert _args(c) == {"query": "n"}


def test_repair_bare_json_in_prose():
    rep = _report("Here you are: {\"name\": \"list_dir\", \"arguments\": "
                          "{\"path\": \"docs\"}} Hope it helps!")
    assert rep.calls and _args(rep.calls) == {"path": "docs"}
    assert "bare_json_prose" in rep.tags_for(rep.calls[0])


def test_bare_data_json_never_parsed():
    # legacy posture: data without a name/arguments shape is not a call
    assert parse_text_tool_calls("x = {\"retention\": 30} ok",
                                 known_tool_names=KNOWN) == []
    assert parse_text_tool_calls("x = {\"name\": \"list\"} ok",
                                 known_tool_names=KNOWN) == []


def test_clean_parse_has_empty_attribution():
    rep = _report("Sure!\n" + FENCE + "json\n{\"name\": \"list_dir\", "
                           "\"arguments\": {\"path\": \".\"}}\n" + FENCE)
    assert rep.calls and not rep.recovered
    assert rep.tags_for(rep.calls[0]) == []


def test_repair_tags_constant_is_superset_of_emitted():
    seen = set()
    for text in [
        "{'name': 'list_dir', 'arguments': {}}",
        FENCE + "json\n{'name': 'list_dir', 'arguments': {}}",
        "<tool>\nlist_dir {}\n</tool>",
        "here {\"name\": \"list_dir\", \"arguments\": {\"a\": 1,}} " + IM_END,
    ]:
        rep = _report(text)
        for c in rep.calls:
            seen.update(rep.tags_for(c))
    assert seen <= set(REPAIR_TAGS) | {"name_normalized",
                                       "truncated_candidate"}


# ===========================================================================
# Name routing: exact-normalized only (documented refusal)
# ===========================================================================

def test_name_exact_normalized_match_recovers_variant_spellings():
    # Policy: EXACT match on the normal form (case/underscore folding)
    # routes; nothing else does. "serch_files" is edit-distance 1 from
    # "search_files" and must NOT be routed -- a mis-routed call executes
    # the wrong thing, a refused one just re-prompts the model.
    rep = _report(FENCE + "\n{'name': 'Read-File', 'arguments': "
                          "{'path': 'z'}}")
    assert rep.calls and rep.calls[0]["function"]["name"] == "read_file"
    assert "name_normalized" in rep.tags_for(rep.calls[0])


def test_name_typo_not_routed():
    assert parse_text_tool_calls(
        FENCE + "\n{\"name\": \"serch_files\", \"arguments\": "
        "{\"query\": \"x\"}}", known_tool_names=KNOWN) == []


def test_name_ambiguous_normal_form_refused():
    known = ["read_file", "Read File", "search_files"]  # fold to 1 dup pair
    rep = parse_text_tool_calls_with_repairs(
        FENCE + "\n{'name': 'READ_FILE', 'arguments': {}}", known)
    assert rep.calls == []  # two known names fold the same => refuse


def test_no_known_list_passes_names_through():
    rep = parse_text_tool_calls_with_repairs(
        FENCE + "\n{'name': 'Whatever', 'arguments': {}}")
    assert rep.calls and rep.calls[0]["function"]["name"] == "Whatever"


def test_normalize_tool_name_shape():
    assert normalize_tool_name("Read File") == "read_file"
    assert normalize_tool_name(" read--file ") == "read_file"
    assert normalize_tool_name("search.files/2") == "search_files_2"


# ===========================================================================
# Mechanism 3: reasoning-loop detection
# ===========================================================================

LAP = ("I still haven't actually opened the file. Let me check the "
       "directory listing first to confirm the path exists before I "
       "attempt reading it again, since the earlier error may have been "
       "a transient permissions issue.")


def test_loop_exact_paragraph_repeats():
    text = "Hmm.\n\n" + "\n\n".join([LAP] * 4)
    v = detect_reasoning_loop(text)
    assert v.loop and v.reason == "paragraph_repeats" and v.repeats >= 3
    out = truncate_at_loop(text, v)
    assert out.startswith("Hmm.")
    assert out.endswith(LOOP_MARKER)
    assert LAP not in out  # truncated at FIRST occurrence


def test_loop_near_duplicate_laps():
    text = ("Working.\n\n" + LAP + "\n\n"
            + LAP.replace("directory listing first",
                          "listing of the dir first") + "\n\n"
            + LAP.replace("directory listing first to confirm",
                          "dir listing first to double-check") + "\n\n"
            + LAP)
    v = detect_reasoning_loop(text)
    assert v.loop and v.reason == "near_duplicates"


def test_loop_hedging_cycles():
    h = ["Wait, I need to check the config first.",
         "Actually, the file might not exist yet.",
         "No, the error said permission denied specifically.",
         "Hold on, maybe I should try a different path instead.",
         "Hmm, but wait, what about the symlink situation?"]
    v = detect_reasoning_loop("\n".join(h * 4))
    assert v.loop and v.reason == "hedging_cycles"


def test_loop_interleaved_think_blocks():
    text = ("<thinking>\nchecking\n</thinking>\n\n" + LAP + "\n\n"
            + "<thinking>\nmore\n</thinking>\n\n" + LAP + "\n\n" + LAP)
    assert detect_reasoning_loop(text).loop


def test_loop_closed_think_noise_scrubs_to_clean():
    # loops ENTIRELY inside closed think pairs are removed by the scrubber
    # before judging -- the visible answer is clean, no truncation needed.
    text = ("<thinking>\n" + LAP + "\n</thinking>\n\n") * 4
    assert not detect_reasoning_loop(text).loop


@pytest.mark.parametrize("healthy", [
    "\n\n".join("- item %d: %s" % (i, "detailed explanation of this entry "
                                   "covering its purpose and usage " * 2)
                for i in range(30)),
    "```python\n" + "\n\n".join(
        "def handler_%d(ctx):\n    # process ctx and return\n"
        "    result = ctx.value\n    return result" % i
        for i in range(12)) + "\n```\n",
    LAP + "\n\n" + LAP,  # exactly two laps: below threshold
    "Short answer: yes, the file is in src/.",
])
def test_loop_false_negatives_healthy_text(healthy):
    assert not detect_reasoning_loop(healthy).loop


def test_loop_truncation_fails_open():
    v = detect_reasoning_loop("totally different text than analysed")
    assert not v.loop
    assert truncate_at_loop("whatever", v) == "whatever"
    assert truncate_at_loop("no verdict here") == "no verdict here"


# ===========================================================================
# Zero-regression contract (legacy-identical on legacy inputs)
# ===========================================================================

@pytest.mark.parametrize("sample", [
    "Sure!\n" + FENCE + "json\n{\"name\": \"read_file\", \"arguments\": "
    "{\"path\": \"a.py\"}}\n" + FENCE + "\n",
    FENCE + "\n[{\"function\": {\"name\": \"list_dir\", \"arguments\": "
    "\"{\\\"path\\\": \\\".\\\"}\"}}]\n" + FENCE,
    "Let me check.\n" + LT + "tool_call>read_file" +
    "{\"path\": \"x.txt\"}" + LT + "/tool_call>",
    f"```json\n{json.dumps({'foo': 1})}\n```",
    "", "Just a normal answer.",
])
def test_legacy_samples_identical(sample):
    legacy = parse_legacy(sample)
    new = parse_text_tool_calls(sample, known_tool_names=KNOWN)
    assert _norm_calls(new) == _norm_calls(legacy)


def test_strip_markup_extended_flag_off_is_legacy():
    text = "hi\n" + FENCE + "json\n{\"name\": \"a\", \"arguments\": {}}\n" \
        + FENCE
    assert strip_tool_call_markup(text) == strip_tool_call_markup(
        text, extended=False)
    assert strip_tool_call_markup(
        "<tool>\nx {}\n</tool>", extended=True) == ""


# ===========================================================================
# E2E: runner-level loop recovery via the nudge mechanism (once per run)
# ===========================================================================

def _loop_handler(nudges_allowed: bool = True):
    class LoopHandler(_Handler):
        def _completion_payload(self, payload):  # noqa: N802
            messages = payload.get("messages", [])
            nudged = any("repetition loop" in str(m.get("content"))
                         for m in messages if m.get("role") == "user")
            if nudged and nudges_allowed:
                content = "Done! The file is at a.py and it parses fine."
            else:
                content = "Hmm.\n\n" + "\n\n".join([LAP] * 4)
            return {"id": "c", "object": "chat.completion",
                    "model": "fake",
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant",
                                             "content": content},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2}}
    return LoopHandler


def _run(client_kwargs):
    with FakeOpenAIServer(handler_cls=_loop_handler()) as fs:
        client = SableAgentClient(base_url=fs.base_url, model="fake",
                                    **client_kwargs)
        return client.run("do the thing")


def test_runner_truncates_loop_and_nudges_once():
    res = _run({})
    assert res.completed
    nudges = [m for m in res.messages
              if m.get("role") == "user"
              and "repetition loop" in str(m.get("content"))]
    assert len(nudges) == 1  # exactly one synthetic retry turn
    # the truncated turn (with the marker) is what entered history
    loop_turns = [m for m in res.messages
                  if LOOP_MARKER in str(m.get("content"))]
    assert len(loop_turns) == 1
    assert res.final_response.startswith("Done!")


def test_runner_loop_nudge_budget_is_one_per_run():
    # handler NEVER escapes the loop: run must still terminate with the
    # truncated answer and exactly one nudge (no retry spiral).
    def _handler_cls():
        class StuckHandler(_Handler):
            def _completion_payload(self, payload):  # noqa: N802
                content = "Hmm.\n\n" + "\n\n".join([LAP] * 4)
                return {"id": "c", "object": "chat.completion",
                        "model": "fake",
                        "choices": [{"index": 0,
                                     "message": {"role": "assistant",
                                                 "content": content},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1,
                                  "completion_tokens": 1,
                                  "total_tokens": 2}}
        return StuckHandler
    with FakeOpenAIServer(handler_cls=_handler_cls()) as fs:
        client = SableAgentClient(base_url=fs.base_url, model="fake")
        res = client.run("do the thing")
    nudges = [m for m in res.messages
              if m.get("role") == "user"
              and "repetition loop" in str(m.get("content"))]
    assert len(nudges) == 1
    assert res.completed


def test_runner_guardrails_off_disables_loop_recovery():
    res = _run({"guardrails_enabled": False})
    # No truncation, no nudge: legacy behavior preserved byte-for-byte
    # when guardrails are opted out.
    assert not any(LOOP_MARKER in str(m.get("content"))
                   for m in res.messages)
    assert not any("repetition loop" in str(m.get("content"))
                   for m in res.messages if m.get("role") == "user")


# ===========================================================================
# Performance sanity: adversarial inputs stay linear-ish
# ===========================================================================

def _build_pathological() -> str:
    # One fence per unit combining every repair class (trailing commas,
    # single quotes, python literals) so the FIRST fence repairs and wins;
    # the cost under test is the scan over thousands of corrupt candidates.
    unit = (FENCE + "json\n{'name': 'search_files', 'arguments': "
            "{\"query\": \"x\"}, \"b\": {'c': True,},}\n" + FENCE + "\n")
    # ~200KB of repeated corruption patterns + a stop-token storm tail
    return unit * 2600 + IM_END * 5000


def test_pathological_input_under_one_second():
    text = _build_pathological()
    assert len(text) > 200_000
    t0 = time.time()
    calls = parse_text_tool_calls(text, known_tool_names=KNOWN)
    strip_stop_tokens(text + IM_END * 100)
    detect_reasoning_loop(text)
    dt = time.time() - t0
    assert calls  # still parses (first fence wins)
    assert dt < 1.0, f"adversarial 200KB took {dt:.2f}s"


# ===========================================================================
# GATE LADDER over the drift corpus
# ===========================================================================

def _load_corpus():
    manifest_path = FIXTURES / "manifest.json"
    assert manifest_path.exists(), "drift corpus manifest missing"
    manifest: Dict[str, Dict[str, Any]] = json.loads(
        manifest_path.read_text(encoding="utf-8"))
    files = sorted(str(p.relative_to(FIXTURES))
                   for p in FIXTURES.rglob("*.txt"))
    return manifest, files


def _class_of(rel: str) -> str:
    return rel.split("/", 1)[0]


def test_layer1_weightless_text_scan():
    """LAYER 1: every fixture loads; manifest and files agree exactly."""
    manifest, files = _load_corpus()
    assert files, "fixture corpus is empty"
    assert set(files) == set(manifest), (
        f"manifest/files mismatch: {set(files) ^ set(manifest)}")
    for rel in files:
        text = (FIXTURES / rel).read_text(encoding="utf-8")
        assert text.strip(), f"{rel} is empty"
        assert _class_of(rel) in {"stop_token_repeat", "reasoning_loop",
                                  "malformed_json", "xml_wrapped", "mixed"}


def _ladder_rows():
    manifest, files = _load_corpus()
    rows = []
    regressions: List[str] = []
    for rel in files:
        meta = manifest[rel]
        text = (FIXTURES / rel).read_text(encoding="utf-8")
        legacy = parse_legacy(text)                      # baseline (live)
        rep = parse_text_tool_calls_with_repairs(
            text, known_tool_names=KNOWN)
        rows.append((rel, meta, text, legacy, rep))
        if legacy:
            # ZERO-REGRESSION: legacy-parsing inputs parse identically.
            if _norm_calls(rep.calls) != _norm_calls(legacy):
                regressions.append(rel)
    return rows, regressions


def test_layer2_per_fixture_expectations():
    """LAYER 2: recovery + zero-regression assertions per fixture."""
    rows, regressions = _ladder_rows()
    assert not regressions, f"REGRESSIONS: {regressions}"
    for rel, meta, text, legacy, rep in rows:
        calls = rep.calls
        assert bool(calls) == meta["expect_parsed"], (
            f"{rel}: parsed={bool(calls)} "
            f"want {meta['expect_parsed']} calls={calls}")
        if not calls:
            continue
        assert calls[0]["function"]["name"] == meta["expect_name"], rel
        want_args = meta.get("expect_args")
        if want_args is not None:
            got = json.loads(calls[0]["function"]["arguments"])
            for k, v in want_args.items():
                assert got.get(k) == v, f"{rel}: arg {k}"
        tags = rep.tags_for(calls[0])
        want_any = meta.get("expect_tags_any")
        if want_any:
            assert any(t in tags for t in want_any), (
                f"{rel}: tags {tags} want any of {want_any}")
        # attribution coherence: recovered => non-empty tags and vice versa
        assert bool(tags) == rep.recovered, f"{rel}: attribution"
        if _class_of(rel) == "reasoning_loop":
            v = detect_reasoning_loop(text)
            assert v.loop == meta["expect_loop"], rel
            if v.loop:
                cut = truncate_at_loop(text, v)
                assert cut.endswith(LOOP_MARKER), rel
                assert text[:v.start].rstrip() in cut, rel


def test_layer3_recovery_rate_report():
    """LAYER 3: aggregate recovery-rate table (printed) + strict gate."""
    rows, regressions = _ladder_rows()
    assert not regressions, f"REGRESSIONS: {regressions}"

    parse_fixtures = [r for r in rows if _class_of(r[0]) != "reasoning_loop"]
    classes = sorted({_class_of(rel) for rel, *_ in parse_fixtures})
    header = (f"{'class':<18} {'n':>3} {'baseline(pre-WP8)':>17} "
              f"{'new(WP8)':>9} {'recovered+':>10}")
    lines = ["", "WP8 drift corpus recovery-rate report "
                  "(baseline = parse_legacy, live):", header,
             "-" * len(header)]
    total_n = base_n = new_n = 0
    for cls in classes:
        sub = [r for r in parse_fixtures if _class_of(r[0]) == cls]
        n = len(sub)
        b = sum(1 for r in sub if r[3])
        w = sum(1 for r in sub if r[4].calls)
        total_n += n
        base_n += b
        new_n += w
        lines.append(f"{cls:<18} {n:>3} {b:>17} {w:>9} {w - b:>+10}")
    lines.append("-" * len(header))
    bp = 100.0 * base_n / total_n
    wp = 100.0 * new_n / total_n
    lines.append(f"{'TOTAL':<18} {total_n:>3} {base_n:>17} {new_n:>9} "
                 f"{new_n - base_n:>+10}")
    lines.append(f"recovery rate: baseline {bp:.1f}% -> new {wp:.1f}% "
                 f"(+{wp - bp:.1f} percentage points)")
    lines.append(f"zero regressions: {'YES' if not regressions else 'NO'}")
    report = "\n".join(lines)
    print(report)  # -s friendly; also captured in failure output

    assert new_n > base_n, (
        "WP8 must recover strictly more than the pre-WP8 baseline:\n"
        + report)
    # all parse-class fixtures are the target: full recovery of every
    # positive expectation (the corpus is curated so expect_parsed=True
    # fixtures are ALL recovered).
    expected = sum(1 for r in parse_fixtures if r[1]["expect_parsed"])
    assert new_n == expected, report

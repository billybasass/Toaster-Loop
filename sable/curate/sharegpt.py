"""ShareGPT rendering — byte-compatible with the ShareGPT trajectory format.

Verified against (read-only reference, MIT):
  runtime helpers :: convert_to_trajectory_format
  trajectory module :: save_trajectory / convert_scratchpad_to_think
  batch runner :: {"conversations": [...]} entry shape

Conventions replicated exactly (including the unusual literal space after the
``<`` in tag names — that is how the format writes them and what downstream
parsers/trackers expect):

  * turns are ``{"from": <role>, "value": <str>}`` with roles:
      system | human | gpt | tool
    (the format uses ``gpt`` for assistant turns containing tool calls rather than
    ShareGPT's ``function_call``, and ``tool`` rather than ``observation`` —
    we match the Phase 3 spec "MATCH the conventions of
    batch_runner.py uses").
  * every ``gpt`` turn carries a ``< think>\\n< /think>`` block (empty when no
    reasoning is present); ``<REASONING_SCRATCHPAD>`` tags convert to it.
  * tool calls render inside the ``gpt`` turn as
    ``< tool_call>\\n{"name": ..., "arguments": {...}}\\n< /tool_call>`` blocks,
    one per call, after optional reasoning/content text.
  * the consecutive ``role: tool`` messages following a tool-call assistant
    message collapse into ONE ``tool`` turn whose value is the newline-joined
    ``< tool_response>\\n{"tool_call_id":..., "name":..., "content":...}\\n< /tool_response>``
    blocks (JSON payload, parsed back from string content when it looks like
    JSON).

Deliberate divergence (documented): the reference format silently drops orphan ``tool``
messages that have no preceding assistant tool_calls turn; we keep them as a
plain ``tool`` turn with the raw text so nothing recorded is lost — the
buffer's JSONL is our source of truth and dropping data at export time would
be surprising.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def convert_scratchpad_to_think(content: str) -> str:
    """Standard trajectory conversion."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "< think>").replace(
        "</REASONING_SCRATCHPAD>", "< /think>"
    )


def _tool_call_payload(tool_call: Any) -> Dict[str, Any] | None:
    """Normalize one OpenAI tool_call into the {"name", "arguments"} shape."""
    if not isinstance(tool_call, dict):
        return None
    fn = tool_call.get("function") or {}
    raw_args = fn.get("arguments")
    if isinstance(raw_args, str):
        try:
            arguments = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            # Warn and fall back to {}; consistent behavior.
            arguments = {}
    elif raw_args is None:
        arguments = {}
    else:
        arguments = raw_args
    return {"name": fn.get("name", "unknown"), "arguments": arguments}


def _tool_response_block(tool_msg: Dict[str, Any], name: str) -> str:
    """One < tool_response> block."""
    content = tool_msg.get("content")
    if isinstance(content, str):
        try:
            if content.strip().startswith(("{", "[")):
                content = json.loads(content)
        except (json.JSONDecodeError, AttributeError):
            pass  # keep as string if not valid JSON
    elif content is None:
        content = ""
    block = "< tool_response>\n"
    block += json.dumps(
        {
            "tool_call_id": tool_msg.get("tool_call_id", ""),
            "name": name,
            "content": content,
        },
        ensure_ascii=False,
    )
    block += "\n< /tool_response>"
    return block


def messages_to_sharegpt(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert verbatim OpenAI-format messages to ShareGPT turns.

    ``messages`` are the recorded message dicts (system/user/assistant/tool,
    assistant may carry ``tool_calls``, optionally ``reasoning``). Returns a
    list of ``{"from", "value"}`` turns ready for a ``conversations`` row.
    """
    conv: List[Dict[str, str]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")

        if role == "system":
            conv.append({"from": "system", "value": msg.get("content") or ""})

        elif role == "user":
            content = msg.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            conv.append({"from": "human", "value": content})

        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                content = ""
                reasoning = msg.get("reasoning")
                if reasoning and str(reasoning).strip():
                    content = f"< think>\n{reasoning}\n< /think>\n"
                if msg.get("content") and str(msg["content"]).strip():
                    content += convert_scratchpad_to_think(msg["content"]) + "\n"

                names: List[str] = []
                for tc in tool_calls:
                    payload = _tool_call_payload(tc)
                    if payload is None:
                        continue
                    names.append(payload["name"])
                    content += (
                        "< tool_call>\n"
                        + json.dumps(payload, ensure_ascii=False)
                        + "\n< /tool_call>\n"
                    )

                # Ensure every gpt turn has a think block.
                if "< think>" not in content:
                    content = "< think>\n< /think>\n" + content
                conv.append({"from": "gpt", "value": content.rstrip()})

                # Collect all subsequent tool responses into one tool turn.
                blocks: List[str] = []
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    name = names[len(blocks)] if len(blocks) < len(names) else "unknown"
                    blocks.append(_tool_response_block(messages[j], name))
                    j += 1
                if blocks:
                    conv.append({"from": "tool", "value": "\n".join(blocks)})
                    i = j - 1
            else:
                content = ""
                reasoning = msg.get("reasoning")
                if reasoning and str(reasoning).strip():
                    content = f"< think>\n{reasoning}\n< /think>\n"
                content += convert_scratchpad_to_think(msg.get("content") or "")
                if "< think>" not in content:
                    content = "< think>\n< /think>\n" + content
                conv.append({"from": "gpt", "value": content.strip()})

        elif role == "tool":
            # Orphan tool result (no preceding assistant tool_calls): the format
            # drops these; we keep the raw text as a tool turn (see module
            # docstring).
            content = msg.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            conv.append({"from": "tool", "value": content})

        i += 1
    return conv


def final_assistant_text(messages: List[Dict[str, Any]]) -> str | None:
    """Text of the last assistant message that is a plain answer (no tool_calls).

    Used as the DPO chosen/rejected completion. Returns None when the trace
    never produced a final text answer.
    """
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        if m.get("tool_calls"):
            continue
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None

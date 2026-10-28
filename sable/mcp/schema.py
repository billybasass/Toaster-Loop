"""Strict, minimal JSON-Schema validator for MCP tool arguments (WP6).

A deliberately tiny subset of JSON Schema — just what the fixed tool schemas
in tools.py declare: object type, ``properties`` with scalar types
(string/integer/number/boolean), ``required``, ``additionalProperties: false``
and ``enum``. The server REFUSES unknown args and wrong types outright; there
is no coercion beyond an int satisfying a ``number`` schema (JSON numbers have
one numeric type family). Every rejection returns a human-readable message
that becomes a JSON-RPC -32602 error — arguments never reach an
implementation, let alone a shell.

STDLIB-ONLY (D-12): re-implemented inline; no jsonschema package.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

#: JSON type name -> python predicate. ``bool`` is deliberately excluded from
#: integer/number: bool subclasses int in Python but is NOT a number here.
_TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
}

#: WP12: the ONLY item types an ``array`` argument may declare. Scalars only —
#: nested objects/arrays as items are out of scope for tool args and refused
#: (reject-don't-accept-all). Used so plan deps can be real id lists.
_ITEM_TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
}

#: Hard ceiling on array length regardless of a schema's own ``maxItems``: a
#: deps list is short, and this bounds any pathological client input. A schema
#: asking for more than this is treated as a schema bug (capped here).
MAX_ARRAY_ITEMS = 64


def validate_args(schema: Dict[str, Any], args: Any) -> Optional[str]:
    """Validate *args* against a fixed object *schema*.

    Returns None when valid, else a single clear error message string.
    """
    if not isinstance(args, dict):
        return f"arguments must be a JSON object, got {_kind(args)}"

    props: Dict[str, Any] = schema.get("properties") or {}
    required: List[str] = schema.get("required") or []

    for name in required:
        if name not in args:
            return f"missing required argument {name!r}"

    # additionalProperties:false is the ONLY supported shape here — every
    # sable tool schema sets it (no free-form bags).
    for key, value in args.items():
        if key not in props:
            known = ", ".join(sorted(props)) or "(none)"
            return (f"unknown argument {key!r}; this tool accepts only: "
                    f"{known}")

    for key, spec in props.items():
        if key not in args:
            continue  # optional and absent
        value = args[key]
        tname = str(spec.get("type") or "")
        if tname == "array":
            err = _validate_array(key, spec, value)
            if err is not None:
                return err
            continue
        check = _TYPES.get(tname)
        if check is None:  # schema bug guard — refuse rather than accept-all
            return f"schema error: unsupported type {tname!r} for {key!r}"
        if not check(value):
            return (f"argument {key!r} must be of type {tname}, "
                    f"got {_kind(value)}")
        enum = spec.get("enum")
        if enum is not None and value not in enum:
            allowed = ", ".join(repr(e) for e in enum)
            return f"argument {key!r} must be one of: {allowed}"
        if tname in ("integer", "number"):
            minimum = spec.get("minimum")
            maximum = spec.get("maximum")
            if minimum is not None and value < minimum:
                return f"argument {key!r} must be >= {minimum}"
            if maximum is not None and value > maximum:
                return f"argument {key!r} must be <= {maximum}"
    return None


def _validate_array(key: str, spec: Dict[str, Any], value: Any) -> Optional[str]:
    """Validate an ``array`` argument (WP12; strict and closed by design).

    Rules — same refuse-don't-accept-all philosophy as scalar handling:
      * the value must be a JSON list;
      * ``items`` is REQUIRED and must declare a scalar item type
        (string/integer) — missing item schema, unknown item type, or nested
        object/array items are all SCHEMA ERRORS (refused at validate time);
      * optional per-item ``pattern`` (regex, applied to string items);
      * ``maxItems`` is enforced and hard-capped at MAX_ARRAY_ITEMS — a schema
        asking for more is a schema bug (capped), not an invitation.
    No coercion: wrong item type rejects the whole call.
    """
    if not isinstance(value, list):
        return f"argument {key!r} must be of type array, got {_kind(value)}"

    item_spec = spec.get("items")
    if not isinstance(item_spec, dict):
        return (f"schema error: array argument {key!r} requires an items "
                f"schema")
    itype = str(item_spec.get("type") or "")
    check = _ITEM_TYPES.get(itype)
    if check is None:
        return (f"schema error: unsupported item type {itype!r} for {key!r} "
                f"(only string/integer items are allowed)")

    max_items = spec.get("maxItems")
    if max_items is not None and (not isinstance(max_items, int)
                                  or isinstance(max_items, bool)):
        return f"schema error: maxItems for {key!r} must be an integer"
    cap = min(int(max_items), MAX_ARRAY_ITEMS) if max_items else \
        MAX_ARRAY_ITEMS
    if len(value) > cap:
        return (f"argument {key!r} has {len(value)} items; maximum is "
                f"{cap}")

    pattern = item_spec.get("pattern")
    rx = None
    if pattern is not None:
        if itype != "string" or not isinstance(pattern, str):
            return (f"schema error: pattern on {key!r} requires string items")
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"schema error: bad item pattern for {key!r}: {e}"

    for i, item in enumerate(value):
        if not check(item):
            return (f"argument {key!r}[{i}] must be of type {itype}, "
                    f"got {_kind(item)}")
        if rx is not None and not rx.search(item):
            return (f"argument {key!r}[{i}] does not match pattern "
                    f"{pattern!r}")
        enum = item_spec.get("enum")
        if enum is not None and item not in enum:
            allowed = ", ".join(repr(e) for e in enum)
            return f"argument {key!r}[{i}] must be one of: {allowed}"
    return None


def _kind(value: Any) -> str:
    """JSON-flavoured type name of *value* for error messages."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    return type(value).__name__

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

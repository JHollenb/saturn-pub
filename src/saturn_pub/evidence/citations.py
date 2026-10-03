"""Writer-side helper for emitting index-ready address citations.

Receipts historically mention state addresses in free text, which forces the
xref indexer (:mod:`saturn_pub.evidence.xref`) to guess which model (if any)
produced the cited activation. A citation dict removes the guesswork:
experiments call :func:`make_citation` and embed the returned dict anywhere in
their JSON receipt. The xref extractor recognizes the ``saturn_pub_citation``
marker and copies ``model_id`` / ``revision`` / ``input_sha256`` into the index
directly, taking precedence over best-effort structural attribution.

Addresses use the ``ar://`` scheme (the public autoregressive address space).
The scheme and its grammar are module constants here and are deliberately kept
in sync with the xref extractor by a test, so this module stays stdlib-only and
importable in isolation. A citation address must be concrete (no ``{template}``
holes) and must not end in punctuation that the extractor strips (``.,;:}/``) --
otherwise the cited string and the indexed string would disagree.

Honest claim boundary: a citation records that an experiment *cited* an address
for a quantity; it makes no claim that the citation is correct.

CLI exit codes: 0 = success, 2 = invalid input.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from typing import Any

#: The public autoregressive address scheme. Both the citation grammar below
#: and the xref extractor are built from this one constant.
ADDRESS_SCHEME = "ar"

CITATION_KEY = "saturn_pub_citation"
CITATION_VERSION = "v1"

# Concrete-address grammar: scheme + payload from the xref character class,
# minus the brace characters that only ever appear in template strings.
_CONCRETE_ADDRESS_RE = re.compile(rf"\A{ADDRESS_SCHEME}://[A-Za-z0-9_./+-]+\Z")
_TRAILING_PUNCTUATION = ".,;:}/"
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")

_OPTIONAL_STRING_FIELDS = ("model_id", "revision", "quantity", "note")


def validate_address(address: Any) -> str:
    """Return ``address`` unchanged if it is a concrete ``ar://`` address.

    Raises ``ValueError`` (fail-closed) for non-strings, template strings,
    grammar violations, and addresses ending in extractor-stripped punctuation.
    """

    if not isinstance(address, str):
        raise ValueError(f"citation address must be a string, got {type(address).__name__}")
    if not _CONCRETE_ADDRESS_RE.match(address):
        raise ValueError(
            f"citation address {address!r} does not match the concrete {ADDRESS_SCHEME}:// grammar"
            f" ({ADDRESS_SCHEME}:// followed by [A-Za-z0-9_./+-], no template braces)"
        )
    if address[-1] in _TRAILING_PUNCTUATION:
        raise ValueError(
            f"citation address {address!r} ends in {address[-1]!r}, which the xref"
            " extractor strips; the cited and indexed strings would disagree"
        )
    return address


def _validated_optional(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"citation field {name!r} must be a non-empty string or None")
    return value


def make_citation(
    address: str,
    *,
    model_id: str | None = None,
    revision: str | None = None,
    input_sha256: str | None = None,
    quantity: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Build an index-ready citation dict for one concrete ``ar://`` address.

    Only keys whose values were provided appear in the result, plus the
    ``saturn_pub_citation`` schema marker. All validation is fail-closed:
    ``ValueError`` on any malformed field, never a silently degraded dict.
    """

    citation: dict[str, Any] = {
        CITATION_KEY: CITATION_VERSION,
        "address": validate_address(address),
    }
    provided = {
        "model_id": model_id,
        "revision": revision,
        "quantity": quantity,
        "note": note,
    }
    for name in _OPTIONAL_STRING_FIELDS:
        value = _validated_optional(name, provided[name])
        if value is not None:
            citation[name] = value
    if input_sha256 is not None:
        if not isinstance(input_sha256, str) or not _SHA256_RE.match(input_sha256):
            raise ValueError(
                "citation field 'input_sha256' must be 64 lowercase hex characters or None"
            )
        citation["input_sha256"] = input_sha256
    return citation


def is_citation(value: Any) -> bool:
    """True when ``value`` carries the citation schema marker (shape unchecked)."""

    return isinstance(value, dict) and value.get(CITATION_KEY) == CITATION_VERSION


def _lenient_optional(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def citation_fields(value: Any) -> dict[str, Any] | None:
    """Extract normalized fields from a citation dict, or ``None``.

    This is the reader-side counterpart of :func:`make_citation` and never
    raises: a dict without the marker, or with an invalid address, yields
    ``None`` so callers can fall back to generic extraction. Optional fields
    that fail validation are dropped to ``None`` (never guessed at).
    """

    if not is_citation(value):
        return None
    address = value.get("address")
    try:
        validate_address(address)
    except ValueError:
        return None
    input_sha256 = value.get("input_sha256")
    if not isinstance(input_sha256, str) or not _SHA256_RE.match(input_sha256):
        input_sha256 = None
    return {
        "address": address,
        "model_id": _lenient_optional(value.get("model_id")),
        "revision": _lenient_optional(value.get("revision")),
        "input_sha256": input_sha256,
        "quantity": _lenient_optional(value.get("quantity")),
        "note": _lenient_optional(value.get("note")),
    }


def _cmd_make(args: argparse.Namespace) -> int:
    citation = make_citation(
        args.address,
        model_id=args.model_id,
        revision=args.revision,
        input_sha256=args.input_sha256,
        quantity=args.quantity,
        note=args.note,
    )
    print(json.dumps(citation, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit index-ready ar:// evidence citations (saturn_pub_citation v1)"
    )
    commands = parser.add_subparsers(dest="cite_command", required=True)

    make = commands.add_parser("make", help="print one validated citation as canonical JSON")
    make.add_argument("address", help="concrete ar:// address being cited")
    make.add_argument("--model-id", default=None)
    make.add_argument("--revision", default=None)
    make.add_argument("--input-sha256", default=None)
    make.add_argument("--quantity", default=None, help="what quantity the citation supports")
    make.add_argument("--note", default=None)
    make.set_defaults(handler=_cmd_make)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "ADDRESS_SCHEME",
    "CITATION_KEY",
    "CITATION_VERSION",
    "citation_fields",
    "is_citation",
    "make_citation",
    "validate_address",
]


if __name__ == "__main__":
    raise SystemExit(main())

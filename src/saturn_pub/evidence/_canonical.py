"""Canonical JSON and sha256 helpers shared by the evidence plane.

This is the one custody idiom every evidence module relies on: canonical JSON
(``sort_keys``, compact separators, ``allow_nan=False``) with non-finite floats
sanitized to ``"NaN"`` / ``"Infinity"`` / ``"-Infinity"`` strings first, plus a
sha256 over those exact bytes. Equal content always yields equal bytes, so a
hash chain can be re-verified from a saved record without re-running anything.

The module imports nothing beyond the standard library (and nothing from the
rest of ``saturn_pub``) so the evidence subpackage stays self-contained and can
later move into a neutral base package shared by more than one toolkit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

GENESIS_SHA256 = "0" * 64


def sanitize_non_finite(value: Any) -> Any:
    """Recursively replace non-finite floats with strict-JSON-safe strings.

    ``nan`` -> ``"NaN"``, ``inf`` -> ``"Infinity"``, ``-inf`` -> ``"-Infinity"``.
    Applied to every value before canonical serialization so hashing stays
    strict JSON (``allow_nan=False``) and a non-finite score or evidence entry
    cannot crash a probe or a seal recomputation.
    """

    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Mapping):
        return {key: sanitize_non_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_non_finite(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON encoding shared by the whole evidence plane.

    Strict JSON (``allow_nan=False``) with sorted keys and compact separators;
    non-finite floats are replaced by their string forms first.
    """

    return json.dumps(
        sanitize_non_finite(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """sha256 hex digest of raw bytes."""

    return hashlib.sha256(data).hexdigest()


def content_fingerprint(value: Any) -> str:
    """sha256 hex digest of :func:`canonical_json_bytes` of ``value``."""

    return sha256_hex(canonical_json_bytes(value))


__all__ = [
    "GENESIS_SHA256",
    "canonical_json_bytes",
    "content_fingerprint",
    "sanitize_non_finite",
    "sha256_hex",
]

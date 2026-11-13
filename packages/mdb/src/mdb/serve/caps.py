"""The 64 KiB response cap, canonical JSON, envelopes and explicit ``omitted`` bookkeeping.

Every API response is canonical JSON (sorted keys, no whitespace, UTF-8, no NaN). The ASGI
middleware ``CapMiddleware`` measures each ``/api/`` response body and replaces one that is over
the cap with ``{"refused": "over_cap", "bytes": N, "cap": CAP}`` (HTTP 413). It never truncates.
Views also bound their own lists (``take``) and say what they left out (``Omitted``).
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .schema import CAP_BYTES, SCHEMAS, Omitted, Refused, unresolved

NONFINITE = "non-finite in the stored body"


def nonfinite() -> dict:
    """The wire form of a NaN/inf from a stored body: an unresolved number that says why."""
    return unresolved(NONFINITE).model_dump(mode="json")


def canonical(value: Any) -> bytes:
    """Canonical JSON bytes. Text is UTF-8; a string no UTF-8 can hold (a lone surrogate in a
    stored body) makes the whole value fall back to ``\\uXXXX`` escapes, still valid JSON, so the
    bytes counted against the cap are the bytes sent."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except UnicodeEncodeError:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False).encode("ascii")


def dump(model) -> Any:
    """Plain JSON of a pydantic model (wire names)."""
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json", by_alias=True)
    if isinstance(model, list):
        return [dump(m) for m in model]
    return model


def envelope(view: str, data, omitted: list[Omitted] | None = None, *, state: str = "resolved",
             reason: str | None = None) -> dict:
    """``{schema, state, data, digest, bytes, omitted?, reason?}``; digest and bytes are over
    canonical data. ``state`` is resolved, unresolved or refused; a state other than resolved
    must carry a ``reason``."""
    if state != "resolved" and not reason:
        raise ValueError(f"a {state} envelope says why")
    plain = dump(data)
    raw = canonical(plain)
    out = {"schema": SCHEMAS[view], "state": state, "data": plain,
           "digest": hashlib.sha256(raw).hexdigest(),
           "bytes": len(raw)}
    if reason:
        out["reason"] = reason
    if omitted:
        out["omitted"] = [dump(o) for o in omitted]
    return out


def refused(reason: str, detail: str | None = None, **extra) -> dict:
    model = Refused(refused=reason, detail=None if detail is None else str(detail)[:2000], **extra)
    return model.model_dump(mode="json", exclude_none=True)


def take(items, limit: int, field: str, omitted: list[Omitted], *, reason: str | None = None):
    """The first ``limit`` items; records how many were left out under ``field``."""
    items = list(items)
    if len(items) > limit:
        omitted.append(Omitted(field=field, count=len(items) - limit, reason=reason))
        return items[:limit]
    return items


def bound_json(value: Any, *, field: str, omitted: list[Omitted], max_items: int = 16,
               max_text: int = 512, depth: int = 0) -> Any:
    """A JSON value with long lists and strings cut, each cut recorded in ``omitted``."""
    if depth > 8:
        omitted.append(Omitted(field=field, count=1, reason="nested deeper than 8 levels"))
        return None
    if isinstance(value, dict):
        out = {}
        keys = sorted(value)
        if len(keys) > 4 * max_items:
            omitted.append(Omitted(field=field, count=len(keys) - 4 * max_items,
                                   reason="keys beyond the first (sorted)"))
            keys = keys[:4 * max_items]
        for key in keys:
            out[key] = bound_json(value[key], field=f"{field}.{key}", omitted=omitted,
                                  max_items=max_items, max_text=max_text, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        items = list(value)
        if len(items) > max_items:
            omitted.append(Omitted(field=field, count=len(items) - max_items,
                                   reason="list items beyond the first"))
            items = items[:max_items]
        return [bound_json(v, field=f"{field}[{i}]", omitted=omitted, max_items=max_items,
                           max_text=max_text, depth=depth + 1) for i, v in enumerate(items)]
    if isinstance(value, str) and len(value) > max_text:
        omitted.append(Omitted(field=field, count=len(value) - max_text,
                               reason="characters beyond the first"))
        return value[:max_text]
    if isinstance(value, float) and not math.isfinite(value):
        return nonfinite()
    return value


class CapMiddleware:
    """Pure ASGI middleware: buffer ``/api/`` HTTP bodies and refuse ones over the cap."""

    def __init__(self, app, cap: int = CAP_BYTES):
        self.app, self.cap = app, cap

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").startswith("/api/"):
            await self.app(scope, receive, send)
            return
        start, chunks = {}, []

        async def capture(message):
            if message["type"] == "http.response.start":
                start.update(message)
                return
            if message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))
                if message.get("more_body"):
                    return
                body = b"".join(chunks)
                status, headers = start["status"], list(start.get("headers", []))
                if len(body) > self.cap:
                    body = canonical(refused("over_cap", bytes=len(body), cap=self.cap))
                    status = 413
                    headers = [(k, v) for k, v in headers if k.lower() != b"content-type"]
                    headers.append((b"content-type", b"application/json"))
                headers = [(k, v) for k, v in headers if k.lower() != b"content-length"]
                headers.append((b"content-length", str(len(body)).encode()))
                await send({"type": "http.response.start", "status": status,
                            "headers": headers})
                await send({"type": "http.response.body", "body": body})
                return
            await send(message)

        await self.app(scope, receive, capture)

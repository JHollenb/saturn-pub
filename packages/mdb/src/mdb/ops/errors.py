"""Typed operation errors that every front end renders the same way (re-exported from
``mdb.core.errors``, which is stdlib only)."""

from __future__ import annotations

from mdb.core.errors import BadArgs, OpError, Refused, UnknownOp, internal

__all__ = ["BadArgs", "OpError", "Refused", "UnknownOp", "internal"]

"""Typed operation errors (stdlib only). ``mdb.ops.errors`` re-exports them.

Every error renders the same way on every front end: ``to_json()`` is
``{"type", "message", "state", "reason"}``. ``state`` is ``refused`` when the request is valid
or malformed but cannot run (nothing changed), and ``unresolved`` for an internal failure
(the requested part could not be computed).
"""

from __future__ import annotations


class OpError(ValueError):
    """Base of operation errors. ``kind`` names the class on the wire."""

    kind = "error"
    state = "refused"

    def __init__(self, message: str, kind: str | None = None, state: str | None = None):
        self.message = message
        if kind is not None:
            self.kind = kind
        if state is not None:
            self.state = state
        super().__init__(message)

    def to_json(self) -> dict[str, str]:
        return {"type": self.kind, "message": self.message, "state": self.state,
                "reason": self.message}


class UnknownOp(OpError):
    kind = "unknown-op"


class BadArgs(OpError):
    kind = "bad-args"


class Refused(OpError):
    """The operation is valid but cannot run in this state (or is not built yet)."""

    kind = "refused"

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def internal(error: BaseException) -> OpError:
    """An unexpected exception as a typed error: ``internal``, state ``unresolved``."""
    return OpError(f"{type(error).__name__}: {error}", kind="internal", state="unresolved")

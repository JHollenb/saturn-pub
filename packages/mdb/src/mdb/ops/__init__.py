"""The single operation registry. Importing it loads no model library.

    from mdb.ops import Context, dispatch, describe
    ctx = Context(debugger)            # or debugger.ops(run)
    dispatch(ctx, "start", {"prompt": [1, 5, 9, 3]})
    dispatch(ctx, "print", {"address": "L0.p[h0]"})
"""

from . import builtin as _builtin  # noqa: F401 - registers the built-in ops
from . import predict as _predict  # noqa: F401 - explain/forecast/verify/whatif
from .context import Context
from .errors import BadArgs, OpError, Refused, UnknownOp
from .registry import INTEGER_KEYS, Arg, Op, describe, dispatch, register, registry

__all__ = [
    "Arg",
    "BadArgs",
    "Context",
    "INTEGER_KEYS",
    "Op",
    "OpError",
    "Refused",
    "UnknownOp",
    "describe",
    "dispatch",
    "register",
    "registry",
]

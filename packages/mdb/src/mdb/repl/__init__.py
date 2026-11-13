"""The gdb-style REPL over ``mdb.ops``. Importing it loads no model library."""

from .shell import Shell

__all__ = ["Shell"]

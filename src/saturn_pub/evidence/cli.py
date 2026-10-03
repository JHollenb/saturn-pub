"""`saturn-pub evidence <tool> ...` dispatcher for the evidence plane.

Each tool keeps its own argument parser and ``main(argv)``, so the tools stay
independently importable and testable; this module only routes the first token
to the right one.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from . import bisect, citations, claims, xref

_TOOLS: dict[str, Callable[[Sequence[str] | None], int]] = {
    "bisect": bisect.main,
    "claims": claims.main,
    "xref": xref.main,
    "cite": citations.main,
}

_USAGE = "usage: saturn-pub evidence {bisect|claims|xref|cite} ..."


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv) if argv is not None else []
    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE)
        print("\nSubcommands:")
        print("  bisect   first-divergence (and friends) over an ordered cut lattice")
        print("  claims   executable claim registry with drift-aware verification")
        print("  xref     cross-reference index over receipts (SQLite)")
        print("  cite     emit index-ready ar:// evidence citations")
        return 0 if argv else 2
    tool, rest = argv[0], argv[1:]
    handler = _TOOLS.get(tool)
    if handler is None:
        print(f"error: unknown evidence tool {tool!r}", file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

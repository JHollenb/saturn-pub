"""Symbol-table method v2: the frozen run-1 method (``method.py``, 3d0baff, unchanged) plus two
lessons read off its score. POST HOC: both were chosen after the run-1 key was scored
(SCORES.md lessons 2 and 4), so any score of v2 is a post-hoc score, not a held-out one.

1. **Multi-byte bindings.** The frozen method binds byte cells, so a u64 variable becomes eight
   bindings (DWARF recall@0.5 was 0 on bigadd and bsort). v2 merges the bindings of two cells
   when the same op instance wrote both (same defining tick), they are adjacent bytes of one site
   (bytes b, b+1 of one general register, or addresses a, a+1), and their liveness is identical
   (both live or both dead). Initial cells (no op instance) and flags are never merged this way.
2. **Dead copies are bindings.** The frozen method leaves every cell that no donor showed to be
   live unbound (70-85% of def-use-live occurrences). v2 lists each copy web (``Trace.webs``,
   this backend's own) that has at least two members as one binding over its not-live members,
   with ``status`` ``dead``; rule 1 then merges dead bindings too. Live bindings keep the frozen
   method's grouping (and gain rule 1's merges); a dead binding never absorbs a live cell.

``lessons(rec, words=True, dead_copies=True)`` returns ``RecoveryV2``: ``binding`` per cell (-1
unbound), ``status`` per binding (``live`` / ``dead``), and the frozen ``rec`` untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binary import FLAG0
from .method import Recovery
from .trace import INIT

METHOD = "method_v2"
SPEC = ("method_v2 (post hoc): frozen run-1 method; + merge bindings of adjacent bytes written by the "
        "same op instance with identical liveness; + dead copy-web members (webs of >= 2 cells) as "
        "bindings with status dead")


@dataclass
class RecoveryV2:
    rec: Recovery
    binding: np.ndarray        # int [N]: v2 binding id, -1 unbound
    status: list               # per v2 binding: "live" | "dead"
    web: np.ndarray            # int [N]: copy-web root cell id (Trace.webs)
    rules: dict

    @property
    def n_bindings(self) -> int:
        return len(self.status)


def _adjacent(a: int, b: int) -> bool:
    """Sites a, b (a < b) are neighbouring bytes of one register or of memory."""
    if b != a + 1:
        return False
    if a < FLAG0:
        return a // 8 == b // 8
    return a >= FLAG0 + 6


def lessons(rec: Recovery, *, words: bool = True, dead_copies: bool = True) -> RecoveryV2:
    cells = rec.base.cells
    n = len(cells)
    web = np.array(rec.base.webs(), np.int64)
    label = rec.binding.copy()                  # frozen live bindings 0..K-1
    k = int(label.max()) + 1 if n and label.max() >= 0 else 0
    live = rec.live.astype(bool)
    if dead_copies:
        size = np.bincount(web, minlength=n)
        roots: dict[int, int] = {}
        for i in range(n):
            if not live[i] and size[web[i]] >= 2:
                label[i] = roots.setdefault(int(web[i]), k + len(roots))
        k += len(roots)
    parent = list(range(k))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if words:
        by_tick: dict[int, list[int]] = {}
        for i, c in enumerate(cells):
            if c.def_ != INIT and label[i] >= 0:
                by_tick.setdefault(c.def_, []).append(i)
        for group in by_tick.values():
            group.sort(key=lambda i: cells[i].site)
            for x, y in zip(group, group[1:]):
                if _adjacent(cells[x].site, cells[y].site) and live[x] == live[y]:
                    rx, ry = find(int(label[x])), find(int(label[y]))
                    if rx != ry:
                        parent[max(rx, ry)] = min(rx, ry)
    first: dict[int, tuple[int, int]] = {}      # root -> earliest (def, cell): binding order
    for i in np.nonzero(label >= 0)[0]:
        r = find(int(label[i]))
        first[r] = min(first.get(r, (cells[i].def_, int(i))), (cells[i].def_, int(i)))
    renum = {r: j for j, r in enumerate(sorted(first, key=first.get))}
    binding = np.array([renum[find(int(b))] if b >= 0 else -1 for b in label], np.int64)
    status = ["dead"] * len(renum)
    for i in np.nonzero(binding >= 0)[0]:
        if live[i]:
            status[binding[i]] = "live"
    return RecoveryV2(rec, binding, status, web, {"words": words, "dead_copies": dead_copies,
                                                  "spec": SPEC})


__all__ = ["METHOD", "RecoveryV2", "SPEC", "lessons"]

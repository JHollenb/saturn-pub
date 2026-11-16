"""Machine backends beside the decoder engine: programs that are not Hugging Face decoders.

The decoder engine (``mdb.engine``, ``ports.FAMILIES``) is specialised to transformer families and
stays that way. A machine here is an *addressed step machine* (advisor memo 2026-11-16 §1): sites
holding values, ops with typed read and write sets, a schedule, and a hierarchy of clock domains.
``Program`` is the small protocol every backend implements; the verbs are defined once over it.

Backends: ``mdb.machines.x86`` (an x86-64 binary executed by torchx86, batched lanes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

# verdicts a patch on a machine can end in (beside the engine's effect / no effect)
EFFECT, NO_EFFECT = "effect", "no effect"
PATH_DIVERGED, FAULT, NONTERMINATING = "path-diverged", "fault", "nonterminating"
VERDICTS = (EFFECT, NO_EFFECT, PATH_DIVERGED, FAULT, NONTERMINATING)
EDGE_TYPES = ("data", "addr", "ctrl")


@dataclass(frozen=True)
class ClockDomain:
    """A unit for step / next / reverse-step: ``boundary(tick)`` is True when the tick ends a unit.
    ``over`` names the coarser domain it nests in."""
    name: str
    boundary: Callable[[Any], bool]
    over: str | None = None


class Program(Protocol):
    """What a machine exposes to mdb's verbs.

    sites()            every addressable site (register bytes, flag bits, touched memory bytes)
    step(n, unit)      advance n units of a clock domain; next / reverse_step likewise
    snapshot/restore   the whole machine state, for checkpoints and forks
    poke(site, value)  the ``set`` primitive at the cursor
    reads_of(t)        typed read set of tick t: [(site, bits, edge)] with edge in EDGE_TYPES
    writes_of(t)       write set of tick t: [(site, role, value)]
    domains()/define() the clock ladder, finest last, and user-defined domains
    """

    def sites(self) -> list[int]: ...
    def step(self, n: int = 1, unit: str = "instruction") -> Any: ...
    def next(self, n: int = 1, unit: str = "instruction") -> Any: ...
    def reverse_step(self, n: int = 1, unit: str = "instruction") -> Any: ...
    def snapshot(self) -> Any: ...
    def restore(self, snap: Any) -> None: ...
    def poke(self, site: Any, value: int) -> None: ...
    def reads_of(self, t: int) -> list: ...
    def writes_of(self, t: int) -> list: ...
    def domains(self) -> list[ClockDomain]: ...
    def define(self, domain: ClockDomain) -> None: ...

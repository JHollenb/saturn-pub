"""Verdict stability across kernel paths (SC6): a margin rule on top of the noise floor.

The floor (``supplied.judge``: ``floor_spacings`` x the dtype spacing at the consumer's
magnitude, 2 spacings by default) is a noise bound: it says whether one realisation of a forward
moved the consumer. It does not say the verdict survives another realisation of the same forward.
Showcase exp6 (``mdb-experiments/results/showcase-20261114/exp6`` in the repository, audit ``AUDIT-SHOWCASE.md`` correction 1)
measured, over mdb's above-floor patch effects on Qwen3-0.6B-Base and Qwen2.5-0.5B, how many
lose their verdict (sign reversed, or fall to or below the floor) when the same forward runs
another way: every loss sits at or below 4 floors, and no effect above 4 floors lost its verdict
(0/110 and 0/104) on any path. Any reordering moves 3-spacing verdicts: the kernel path (TL
batched vs mdb stepped: 60/274 Qwen3, 16/228 Qwen2.5), batch composition (TL batched vs TL
batch-1: 72/286, 19/241) and processing (processed vs unprocessed bridge: 46/257, 82/249). On
Qwen2.5 processing, not the kernel path, dominates; the SC6 issue text overstated the kernel
path.

``stability`` therefore marks a verdict ``stable`` only when ``|effect| > k`` floors, else
``kernel-sensitive`` (the verdict field itself is unchanged); ``k`` defaults to
``STABLE_FLOORS`` = 4 (``supplied``, from the exp6 table) or comes from a per-model
``Calibration`` measured by ``calibrate`` on paired effects from two realisations (e.g. mdb
stepped vs batched): the smallest ``k`` in floor units above which no verdict was lost, with the
loss table by bin. Every result names the kernel path its verdict was made on
(``mdb-stepped``, ``batched`` or a caller-declared string).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .finite import MEASURED, SUPPLIED, num

STABLE_FLOORS = 4.0
STABLE = "stable"
SENSITIVE = "kernel-sensitive"
MDB_STEPPED = "mdb-stepped"
BATCHED = "batched"
UNDECLARED = "undeclared"
BINS = (1.0, 1.5, 2.0, 4.0)      # floor units; the last bin is open (> 4 floors)
STABLE_SUPPORT = ("exp6 (showcase 2026-11-14, AUDIT-SHOWCASE.md correction 1): 0/110 (Qwen3-0.6B-"
                  "Base) and 0/104 (Qwen2.5-0.5B) above-floor effects > 4 floors lost their "
                  "verdict on any path (kernel path, batch composition, processing); every loss "
                  "was <= 4 floors")
RULE = ("stable when |effect| > k floors, else kernel-sensitive: a verdict within k floors of zero "
        "can move under another realisation of the same forward (kernel path, batch "
        "composition, processing); the verdict field is unchanged")
SCHEMA = "mdb-verdict-stability-calibration-v1"


@dataclass
class Calibration:
    """A per-model stability margin measured from paired effects of two realisations.

    ``k``: the smallest margin in floor units above which no verdict was lost (``measured``; at
    least 1, the floor itself). ``table``: per bin of ``|reference effect| / floor``, the count
    of reference effects, how many lost their verdict on the other path and how many of those
    reversed sign. ``paths``: ``[reference, other, ...]`` kernel-path names."""

    model: str
    k: float
    paths: list[str]
    n: int
    above_floor: int
    lost: int
    sign_reversed: int
    table: list[dict[str, Any]] = field(default_factory=list)
    gained: int = 0
    source: str = ""
    schema: str = SCHEMA

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Calibration:
        if data.get("schema", SCHEMA) != SCHEMA:
            raise ValueError(f"not a {SCHEMA} calibration: {data.get('schema')!r}")
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_json(), indent=1, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        return cls.from_json(json.loads(Path(path).read_text()))


def _bin(ratio: float, bins: tuple[float, ...]) -> str:
    for low, high in zip(bins, bins[1:]):
        if low < ratio <= high:
            return f"{low:g}-{high:g}"
    return f">{bins[-1]:g}"


def lost(reference: float, other: float, floor: float) -> bool:
    """The reference verdict ``effect`` (``|reference| > floor``) is lost on the other path: the
    sign reverses or the other effect is at or below the floor."""
    return abs(reference) > floor and (reference * other < 0 or abs(other) <= floor)


def calibrate(pairs: Iterable[tuple[float, float, float]], *, model: str,
              paths: tuple[str, ...] | list[str] = (MDB_STEPPED, BATCHED),
              bins: tuple[float, ...] = BINS, source: str = "") -> Calibration:
    """Calibrate the stability margin of one model from ``(reference, other, floor)`` triples:
    the same effect (e.g. a patch's margin change) measured on the reference path (the one
    verdicts are made on, e.g. mdb stepped) and on another realisation of the same forward
    (batched, another batch composition, a processed bridge), with the floor the verdict uses.
    Pairs from several other paths can be concatenated (name them all in ``paths``).

    Returns the smallest ``k`` (floor units) above which no reference verdict is lost (``k =
    max |reference| / floor`` over lost verdicts; 1.0 when none is lost) with the loss table by
    bin of ``|reference| / floor`` (``bins``: edges, last bin open). ``gained`` counts reference
    effects at or below the floor that the other path puts above it (reported, not used)."""
    rows: dict[str, dict[str, Any]] = {}
    order = [f"{a:g}-{b:g}" for a, b in zip(bins, bins[1:])] + [f">{bins[-1]:g}"]
    for name in order:
        rows[name] = {"bin": name, "n": 0, "lost": 0, "sign_reversed": 0}
    n = above = n_lost = reversed_ = gained = 0
    k = 1.0
    for reference, other, floor in pairs:
        reference, other, floor = float(reference), float(other), float(floor)
        if not floor > 0:
            raise ValueError("a floor must be positive")
        n += 1
        if abs(reference) <= floor:
            gained += abs(other) > floor
            continue
        above += 1
        ratio = abs(reference) / floor
        row = rows[_bin(ratio, bins)]
        row["n"] += 1
        if lost(reference, other, floor):
            n_lost += 1
            row["lost"] += 1
            k = max(k, ratio)
            if reference * other < 0:
                reversed_ += 1
                row["sign_reversed"] += 1
    return Calibration(model=model, k=k, paths=list(paths), n=n, above_floor=above, lost=n_lost,
                       sign_reversed=reversed_, table=list(rows.values()), gained=gained,
                       source=source)


def _margin(calibration: Calibration | dict | str | Path | None,
            k: float | None) -> tuple[float, str, str]:
    """``(k, authority, source)`` of the margin to use."""
    if calibration is not None:
        if isinstance(calibration, (str, Path)):
            calibration = Calibration.load(calibration)
        elif isinstance(calibration, dict):
            calibration = Calibration.from_json(calibration)
        return (float(calibration.k), MEASURED,
                f"calibrated for {calibration.model} on {' vs '.join(calibration.paths)}")
    if k is not None:
        return float(k), SUPPLIED, "caller"
    return STABLE_FLOORS, SUPPLIED, STABLE_SUPPORT


def stability(effect: float, floor: float, *, kernel_path: str | None = None,
              calibration: Calibration | dict | str | Path | None = None,
              k: float | None = None) -> dict[str, Any]:
    """The stability of a verdict on an effect of size ``|effect|`` against ``floor``:
    ``stable`` when ``|effect| > k x floor``, else ``kernel-sensitive``. ``k`` from the model's
    ``calibration`` (a ``Calibration``, its JSON, or a path to it), else the caller's ``k``,
    else ``STABLE_FLOORS``. ``kernel_path`` names the realisation the verdict was made on."""
    margin, authority, source = _margin(calibration, k)
    effect, floor = abs(float(effect)), float(floor)
    floors = effect / floor if floor > 0 else (math.inf if effect else 0.0)
    out: dict[str, Any] = {
        "stability": STABLE if floors > margin else SENSITIVE,
        "k": num(margin, authority), "k_source": source,
        "kernel_path": kernel_path or UNDECLARED, "rule": RULE,
    }
    if math.isfinite(floors):
        out["floors"] = num(floors, MEASURED)
    return out


__all__ = ["BATCHED", "BINS", "Calibration", "MDB_STEPPED", "RULE", "SENSITIVE", "STABLE",
           "STABLE_FLOORS", "STABLE_SUPPORT", "UNDECLARED", "calibrate", "lost", "stability"]

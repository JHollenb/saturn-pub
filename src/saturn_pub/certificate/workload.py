"""Frozen arbitrary-token relational-induction workload (stdlib only).

The assay presents sequences of arbitrary, distinct token IDs, then repeats a prefix::

    t0 t1 t2 t3 t4 t5  t0 t1 t2 -> t3

The query is the last repeated cue (``t2`` above); the induction answer is the token
that *followed* the earlier matching cue (``t3``). Tokens and cue positions change on
every example and every panel, so the claimed circuit role is not a phrase or a token
ID -- it is the operation "retrieve the token that followed the earlier matching cue".

Two causal positions are derived per item:

* ``source`` -- the earlier position holding the answer token (deleting the final
  query's edge to it is the necessity cut);
* ``match`` -- the earlier occurrence of the cue token itself, one position before
  ``source`` (deleting the edge to it is the neighbor-cue control).

Determinism comes from a stdlib :class:`random.Random` seed; no numpy. This module
imports no model framework.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InductionPanelSpec:
    """One sealed panel of the relational-induction workload."""

    panel_id: str
    seed: int
    examples: int = 192
    length: int = 12
    classes: int = 48
    token_low: int = 1000
    token_high: int = 20000

    def __post_init__(self) -> None:
        if not isinstance(self.panel_id, str) or not self.panel_id:
            raise ValueError("panel_id must be a non-empty string")
        if self.examples <= 0:
            raise ValueError("examples must be positive")
        if self.length < 2:
            raise ValueError("length must be at least two")
        if self.classes < self.length:
            raise ValueError("classes must cover a length >= 2 alphabet")
        if self.token_high - self.token_low < self.classes:
            raise ValueError("token range is too small for the requested alphabet")

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "seed": self.seed,
            "examples": self.examples,
            "length": self.length,
            "classes": self.classes,
            "token_low": self.token_low,
            "token_high": self.token_high,
        }


def build_relational_induction_workload(spec: InductionPanelSpec) -> tuple[list[dict[str, Any]], int]:
    """Build the exact arbitrary-token content-addressed retrieval assay for one panel.

    Returns ``(items, classes)``. Each item carries ``tokens`` (the input IDs),
    ``cands`` (the full alphabet, in candidate order), ``label`` (index into ``cands`` of
    the answer), and the derived ``match``/``source``/``query`` positions.
    """

    rng = random.Random(spec.seed)
    alphabet = sorted(rng.sample(range(spec.token_low, spec.token_high), spec.classes))
    alphabet_index = {token: index for index, token in enumerate(alphabet)}
    items: list[dict[str, Any]] = []
    for _ in range(spec.examples):
        sequence = rng.sample(alphabet, spec.length)
        cue_index = rng.randrange(0, spec.length - 1)
        tokens = sequence + sequence[: cue_index + 1]
        answer_token = sequence[cue_index + 1]
        match, source = cue_index, cue_index + 1
        items.append(
            {
                "tokens": tokens,
                "cands": list(alphabet),
                "label": alphabet_index[answer_token],
                "nclass": spec.classes,
                "base_length": spec.length,
                "match": match,
                "source": source,
                "query": len(tokens) - 1,
            }
        )
    return items, spec.classes


def relational_induction_positions(item: dict[str, Any]) -> tuple[int, int]:
    """Return the repeated cue (``match``) and its earlier answer (``source``) position."""

    length = int(item["base_length"])
    match = len(item["tokens"]) - length - 1
    if not 0 <= match < length - 1:
        raise ValueError("item is not a valid relational-induction sequence")
    return match, match + 1


def wrong_donor_indices(labels: Sequence[int]) -> tuple[int, ...]:
    """For each row, the index of the nearest later row with a different answer label.

    A wrong donor supplies a same-shaped repair whose *content* is a different answer;
    it probes content specificity. Requires at least two answer classes present.
    """

    labels = [int(value) for value in labels]
    result: list[int] = []
    for row, label in enumerate(labels):
        donor = next(
            (
                (row + offset) % len(labels)
                for offset in range(1, len(labels))
                if labels[(row + offset) % len(labels)] != label
            ),
            None,
        )
        if donor is None:
            raise ValueError("wrong-donor repair requires at least two answer classes present")
        result.append(donor)
    return tuple(result)


__all__ = [
    "InductionPanelSpec",
    "build_relational_induction_workload",
    "relational_induction_positions",
    "wrong_donor_indices",
]

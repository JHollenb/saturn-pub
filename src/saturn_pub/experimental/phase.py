"""A standalone 32-bit modular RoPE phase law; no SPF tensor-format dependency."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class PhaseLaw:
    """Transport is exact within quantized frequencies; decoded rotations are floating point."""

    frequencies: tuple[int, ...]
    origin: int = 0

    @classmethod
    def from_inv_freq(cls, inv_freq: Sequence[float]) -> PhaseLaw:
        if len(inv_freq) == 0 or any(not math.isfinite(float(x)) for x in inv_freq):
            raise ValueError("finite, nonempty frequencies required")
        return cls(tuple(round(float(x) / (2 * math.pi) * (1 << 32)) % (1 << 32) for x in inv_freq))

    def shift(self, delta: int) -> None:
        if not isinstance(delta, int):
            raise ValueError("origin shifts require integers")
        self.origin = (self.origin + delta) % (1 << 32)

    def theta(self, positions: Sequence[int]) -> np.ndarray:
        # Python integers keep huge and negative logical origins out of float arithmetic.
        return np.array(
            [[(int(p) + self.origin) * f % (1 << 32) for f in self.frequencies] for p in positions],
            dtype=np.uint64,
        )

    def cos_sin(self, positions: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        angles = self.theta(positions).astype(np.float64) * (2 * math.pi / (1 << 32))
        return np.cos(angles), np.sin(angles)

    @property
    def storage_bytes(self) -> int:
        return 4 * len(self.frequencies) + 8


def rotate_pairs(values: np.ndarray, law: PhaseLaw, positions: Sequence[int]) -> np.ndarray:
    """Reference rotation for [rows, channel-pairs, 2] vectors."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(positions), len(law.frequencies), 2):
        raise ValueError("vectors must match positions and phase channels")
    cos, sin = law.cos_sin(positions)
    x, y = values[..., 0], values[..., 1]
    return np.stack((x * cos - y * sin, x * sin + y * cos), axis=-1)


def relocated_scores(
    query: np.ndarray,
    local_keys: np.ndarray,
    law: PhaseLaw,
    *,
    query_position: int,
    page_origin: int,
) -> np.ndarray:
    """Transport the query to a page's local frame, leaving stored keys unchanged.

    local_keys were rotated once at local positions under the same zero-origin law.
    This preserves relative phase, not hidden-state causal closure or full-model semantics.
    """
    if law.origin != 0:
        raise ValueError("page-local score reference requires a zero-origin law")
    local_query = rotate_pairs(
        np.asarray(query).reshape(1, len(law.frequencies), 2), law, [query_position - page_origin]
    )[0]
    return np.sum(local_keys * local_query, axis=(-1, -2))

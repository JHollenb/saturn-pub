"""A standalone 32-bit modular RoPE phase law; no external tensor-format dependency."""

from __future__ import annotations

import math
import platform
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..contracts import ExecutionPoint, SlotSpec, SurfaceManifest
from ..core import Adapter, Session
from ..values import clone, digest


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

    def to_state_dict(self) -> dict[str, object]:
        return {"frequencies": list(self.frequencies), "origin": self.origin}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> PhaseLaw:
        if set(state) != {"frequencies", "origin"}:
            raise ValueError("phase state must contain exactly frequencies and origin")
        frequencies = state["frequencies"]
        origin = state["origin"]
        if (
            not isinstance(frequencies, list)
            or not frequencies
            or any(type(value) is not int or not 0 <= value < (1 << 32) for value in frequencies)
            or type(origin) is not int
            or not 0 <= origin < (1 << 32)
        ):
            raise ValueError("invalid quantized phase state")
        return cls(tuple(frequencies), origin)


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


class PhaseOriginAdapter(Adapter):
    """Executable metadata-only phase relocation reference with immutable local keys."""

    def __init__(self, law: PhaseLaw, *, shift: int = 512, query_offset: int = 5):
        if law.origin != 0:
            raise ValueError("adapter construction requires a zero-origin phase law")
        if type(shift) is not int or type(query_offset) is not int:
            raise ValueError("phase shift and query offset must be integers")
        self.frequencies = law.frequencies
        self.shift = shift
        self.query_offset = query_offset
        self._configuration_fingerprint = digest(
            {"frequencies": list(self.frequencies), "shift": shift, "query_offset": query_offset}
        )
        self.model_identity = digest(
            {"family": "rope-phase-reference", "frequencies": list(self.frequencies)}
        )
        self.execution = {
            "family": "rope-phase-reference",
            "adapter": "metadata-origin-v1",
            "granularity": "origin-shift",
            "environment_versions": {"numpy": np.__version__, "python": platform.python_version()},
            "model_program": "query-local-frame-transport->score->origin-metadata-shift",
            "kernel": {"rotation": "numpy-float64-reference"},
            "batch": {"queries": 1},
            "device": {"type": "cpu", "index": None},
            "numeric_scope": {
                "phase_arithmetic": "uint32-exact",
                "decoded_rotation": "float64-bounded",
                "key_rewrite": False,
                "shift": shift,
                "query_offset": query_offset,
            },
            "parity": "same-reference-program-exact-metadata",
        }

    def validate_execution(self):
        current = digest(
            {
                "frequencies": list(self.frequencies),
                "shift": self.shift,
                "query_offset": self.query_offset,
            }
        )
        if current != self._configuration_fingerprint:
            raise ValueError(
                "phase configuration changed after adapter identity was frozen; rewrap"
            )

    def session(self, raw_keys: np.ndarray, query: np.ndarray) -> Session:
        import torch

        self.validate_execution()
        raw_keys = np.asarray(raw_keys, dtype=np.float64)
        query = np.asarray(query, dtype=np.float64)
        if raw_keys.ndim != 3 or raw_keys.shape[1:] != (len(self.frequencies), 2):
            raise ValueError("raw keys must be [rows, phase_channels, 2]")
        if query.shape != (len(self.frequencies), 2):
            raise ValueError("query must be [phase_channels, 2]")
        keys = rotate_pairs(raw_keys, PhaseLaw(self.frequencies), range(len(raw_keys)))
        return Session(
            self,
            {
                "local_keys": torch.from_numpy(keys.copy()),
                "query": torch.from_numpy(query.copy()),
                "origin": 0,
                "step": 0,
                "scores": None,
            },
        )

    def boundary(self, state):
        return f"phase-origin:{state['origin']}/before:transport"

    def point(self, state):
        return ExecutionPoint(
            family="rope-phase-reference",
            logical_step=state["step"],
            phase="transport",
            operator="query-local-frame-transport",
            edge="before",
            next_operation="score_then_shift_origin_metadata",
            local_clock=state["origin"],
        )

    def surface(self, state):
        def schema(value, optional=False):
            if value is None:
                return {"kind": "tensor", "optional": optional}
            return {
                "kind": "tensor",
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "optional": optional,
            }

        return SurfaceManifest(
            slots=(
                SlotSpec(
                    "local_keys",
                    role="immutable-cache",
                    producer="local_rope",
                    consumer="phase_score",
                    schema=schema(state["local_keys"]),
                ),
                SlotSpec(
                    "query",
                    role="carrier",
                    producer="caller",
                    consumer="phase_score",
                    schema=schema(state["query"]),
                ),
                SlotSpec(
                    "origin",
                    role="phase-origin",
                    producer="metadata_shift",
                    consumer="query_transport",
                    schema={"kind": "uint32", "minimum": 0, "maximum": (1 << 32) - 1},
                ),
                SlotSpec(
                    "step",
                    role="clock",
                    producer="metadata_shift",
                    consumer="runtime",
                    schema={"kind": "integer", "minimum": 0},
                ),
                SlotSpec(
                    "scores",
                    role="observation",
                    producer="phase_score",
                    consumer="evaluator",
                    persistence="evidence",
                    schema=schema(state["scores"], optional=True),
                ),
            ),
            state_schema="saturn-pub-phase-origin-state-v1",
            consumer="phase-relative-score-reference",
            horizon="metadata-origin-shift",
        )

    def addresses(self, state):
        return ("local_keys", "query") + (("scores",) if state["scores"] is not None else ())

    def write(self, state, writes, invalidates):
        raise ValueError("phase-origin reference state is immutable; create a new session")

    def validate(self, state):
        import torch

        if set(state) != {"local_keys", "query", "origin", "step", "scores"}:
            raise ValueError("phase-origin state schema keys differ from the contract")
        keys, query = state["local_keys"], state["query"]
        if (
            not isinstance(keys, torch.Tensor)
            or keys.device.type != "cpu"
            or keys.dtype != torch.float64
            or keys.ndim != 3
            or tuple(keys.shape[1:]) != (len(self.frequencies), 2)
            or not torch.isfinite(keys).all()
        ):
            raise ValueError("invalid immutable phase keys")
        if (
            not isinstance(query, torch.Tensor)
            or query.device.type != "cpu"
            or query.dtype != torch.float64
            or tuple(query.shape) != (len(self.frequencies), 2)
            or not torch.isfinite(query).all()
        ):
            raise ValueError("invalid phase query")
        if type(state["origin"]) is not int or not 0 <= state["origin"] < (1 << 32):
            raise ValueError("invalid phase origin")
        if type(state["step"]) is not int or state["step"] < 0:
            raise ValueError("invalid phase clock")
        scores = state["scores"]
        if scores is not None and (
            not isinstance(scores, torch.Tensor)
            or scores.device.type != "cpu"
            or scores.dtype != torch.float64
            or tuple(scores.shape) != (keys.shape[0],)
            or not torch.isfinite(scores).all()
        ):
            raise ValueError("invalid phase score observation")

    def advance(self, state):
        import torch

        self.validate(state)
        law = PhaseLaw(self.frequencies)
        scores = relocated_scores(
            state["query"].numpy(),
            state["local_keys"].numpy(),
            law,
            query_position=state["origin"] + self.query_offset,
            page_origin=state["origin"],
        )
        out = clone(state)
        out["scores"] = torch.from_numpy(scores.copy())
        out["origin"] = (state["origin"] + self.shift) % (1 << 32)
        out["step"] += 1
        return out

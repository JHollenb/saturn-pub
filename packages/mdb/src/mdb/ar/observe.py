"""Cursor-indexed observations and the address refusal rule.

Every observable the AR lane emits carries a cursor ``{axis, index, phase}``
so recurrence is a coordinate, never a schema fork.  An ``ar://`` address is
resolvable only if a captured port observation covers it — the mapper-side
rule that stops the address grammar from outrunning its observables.

Observation cost is first-class: every log records its capture policy and
the wall time spent fingerprinting, so a change of observability is
provenance, never silence.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from mdb.ar.training import (
    EXACT_SCALAR_NUMERICAL_LANE,
    LAST_ROW_NUMERICAL_LANE,
)
from mdb.tensor_hash import prefixed_tensor_sha256

OBSERVATION_SCHEMA = "saturn-ar-port-observation-v1"
BOUNDED_FINGERPRINT_ELEMENTS = 4096
RESULT_RETENTION_SCHEMA = "saturn-ar-cycle-retention-policy-v1"
RESULT_RETENTION_MODES = ("evidence", "exact-token", "exact-fast")


class AddressRefused(ValueError):
    """Raised when an ar:// address has no producing observable."""


def tensor_sha256(value: torch.Tensor) -> str:
    meta = f"{value.dtype}|{tuple(value.shape)}|".encode()
    return prefixed_tensor_sha256(meta, value)


def bounded_sha256(
    value: torch.Tensor,
    limit: int = BOUNDED_FINGERPRINT_ELEMENTS,
) -> tuple[str, int]:
    window = value.detach().reshape(-1)[:limit]
    meta = f"{value.dtype}|{tuple(value.shape)}|bounded:{int(window.numel())}|".encode()
    return prefixed_tensor_sha256(meta, window), int(window.numel())


@dataclass(frozen=True, slots=True)
class Cursor:
    """Occurrence coordinate: which token, which phase."""

    axis: str
    index: int
    phase: str

    def __post_init__(self) -> None:
        if self.axis not in {"token", "denoise_step"}:
            raise ValueError(f"unsupported cursor axis {self.axis!r}")
        if self.phase not in {"prefill", "decode"}:
            raise ValueError(f"unsupported cursor phase {self.phase!r}")
        if self.index < 0:
            raise ValueError("cursor index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {"axis": self.axis, "index": self.index, "phase": self.phase}


@dataclass(frozen=True, slots=True)
class CycleRetentionPolicy:
    """Controls which *result* payloads survive an exact token cycle.

    The evidence lane is the backwards-compatible debugger contract: full
    readout, root/final/intermediate Frames, and all ActEvidence remain live.
    ``exact-token`` computes the historical full readout (preserving its exact
    greedy token), retains only the selected row, and drops Frame payloads.
    ``exact-fast`` additionally narrows compute to the native last-token
    readout.  The model execution and KV state remain exact, but the readout is
    assigned to the explicit ``last_row`` numerical lane because changing GEMM
    geometry can change floating-point rounding.  It must not be reused as an
    ``exact_scalar`` answer without an external parity closure.  Observations
    remain independently controlled by their existing explicit requests.
    """

    mode: str = "evidence"

    def __post_init__(self) -> None:
        if self.mode not in RESULT_RETENTION_MODES:
            raise ValueError(
                f"unsupported cycle retention mode {self.mode!r}; "
                f"expected one of {RESULT_RETENTION_MODES!r}"
            )

    @classmethod
    def evidence(cls) -> CycleRetentionPolicy:
        return cls(mode="evidence")

    @classmethod
    def exact_fast(cls) -> CycleRetentionPolicy:
        return cls(mode="exact-fast")

    @classmethod
    def exact_token(cls) -> CycleRetentionPolicy:
        return cls(mode="exact-token")

    @property
    def output_demand(self) -> str:
        return "last-token" if self.mode == "exact-fast" else "full-sequence"

    @property
    def retain_frames(self) -> bool:
        return self.mode == "evidence"

    @property
    def retain_full_logits(self) -> bool:
        return self.mode == "evidence"

    @property
    def numerical_lane(self) -> str:
        return LAST_ROW_NUMERICAL_LANE if self.mode == "exact-fast" else EXACT_SCALAR_NUMERICAL_LANE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESULT_RETENTION_SCHEMA,
            "mode": self.mode,
            "numerical_lane": self.numerical_lane,
            "output_demand": self.output_demand,
            "retain_frames": self.retain_frames,
            "retain_full_logits": self.retain_full_logits,
        }


@dataclass(frozen=True, slots=True)
class CapturePolicy:
    """What the sequencer observes and at what fidelity.

    ``sites`` is the armed site vocabulary (empty = macro digests only).
    ``layers`` restricts intra-layer capture to a layer range.
    ``full_fingerprint`` pays the full-tensor hash (GPU->CPU sync) instead of
    the bounded window; per-element addresses beyond the bounded window are
    refused unless it is set.
    ``retain_tensors`` keeps captured tensors alive for slice-view reads
    (evidence-grade only, never durable).
    ``result_retention`` independently chooses the legacy evidence result or
    the exact-fast token-native result; it never disables armed observations.
    """

    sites: tuple[str, ...] = ()
    layers: tuple[int, ...] = ()
    full_fingerprint: bool = False
    retain_tensors: bool = False
    bounded_limit: int = BOUNDED_FINGERPRINT_ELEMENTS
    retained_sequence_rows: tuple[int, ...] = ()
    result_retention: CycleRetentionPolicy = field(default_factory=CycleRetentionPolicy.evidence)

    def __post_init__(self) -> None:
        if not isinstance(self.result_retention, CycleRetentionPolicy):
            raise TypeError("result_retention must be a CycleRetentionPolicy")
        rows = tuple(sorted(set(self.retained_sequence_rows)))
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in rows
        ):
            raise ValueError("retained_sequence_rows must contain non-negative integers")
        if rows and self.result_retention.output_demand == "last-token":
            raise ValueError("last-row output demand cannot harvest absolute ancestor positions")
        object.__setattr__(self, "retained_sequence_rows", rows)

    @classmethod
    def exact_fast(cls, **capture: Any) -> CapturePolicy:
        """Request the explicit last-row lane with bounded Frame retention.

        ``exact`` here describes native model/KV execution, not equality to
        the full-sequence readout GEMM.  A caller that needs ``exact_scalar``
        authority must use :meth:`exact_token` or carry a parity proof and a
        fail-closed fallback.
        """

        if "result_retention" in capture:
            raise TypeError("exact_fast() sets result_retention")
        return cls(
            **capture,
            result_retention=CycleRetentionPolicy.exact_fast(),
        )

    @classmethod
    def exact_token(cls, **capture: Any) -> CapturePolicy:
        """Use full-readout scalar authority with selective row custody.

        The demanded final row is always retained.  Callers may additionally
        name exact ancestor positions through ``retained_sequence_rows``;
        unspecified rows are discarded after the same full-sequence readout.
        """

        if "result_retention" in capture:
            raise TypeError("exact_token() sets result_retention")
        return cls(
            **capture,
            result_retention=CycleRetentionPolicy.exact_token(),
        )

    def armed(self, site: str, layer: int | None) -> bool:
        if site not in self.sites:
            return False
        if layer is None or not self.layers:
            return bool(self.sites)
        return layer in self.layers

    def to_dict(self) -> dict[str, Any]:
        return {
            "sites": list(self.sites),
            "layers": list(self.layers),
            "full_fingerprint": self.full_fingerprint,
            "retain_tensors": self.retain_tensors,
            "bounded_limit": self.bounded_limit,
            "retained_sequence_rows": list(self.retained_sequence_rows),
            "result_retention": self.result_retention.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PortObservation:
    """One captured crossing of a declared port."""

    address: str
    cursor: Cursor
    site: str
    layer: int | None
    shape: tuple[int, ...]
    dtype: str
    bounded_sha256: str
    bounded_elements: int
    total_elements: int
    full_sha256: str | None
    produced_by: str
    # Axis carrying attention heads in `shape`, when the producer declared
    # one (None = this observation has no head coordinate at all).
    head_axis: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OBSERVATION_SCHEMA,
            "address": self.address,
            "cursor": self.cursor.to_dict(),
            "site": self.site,
            "layer": self.layer,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "bounded_sha256": self.bounded_sha256,
            "bounded_elements": self.bounded_elements,
            "total_elements": self.total_elements,
            "full_sha256": self.full_sha256,
            "produced_by": self.produced_by,
            "head_axis": self.head_axis,
        }


class ObservationLog:
    """Append-only observation log with cost provenance."""

    def __init__(self, policy: CapturePolicy) -> None:
        self.policy = policy
        self.observations: list[PortObservation] = []
        self.retained: dict[str, torch.Tensor] = {}
        self.fingerprint_ms = 0.0

    def observe(
        self,
        *,
        site: str,
        layer: int | None,
        cursor: Cursor,
        value: torch.Tensor,
        produced_by: str,
        head_axis: int | None = None,
    ) -> PortObservation | None:
        if not self.policy.armed(site, layer):
            return None
        started = time.perf_counter()
        bounded, covered = bounded_sha256(value, self.policy.bounded_limit)
        full = tensor_sha256(value) if self.policy.full_fingerprint else None
        self.fingerprint_ms += (time.perf_counter() - started) * 1000.0
        layer_part = f"/layer/{layer}" if layer is not None else ""
        address = f"ar://dense/cursor/{cursor.index}/phase/{cursor.phase}/site/{site}{layer_part}"
        observation = PortObservation(
            address=address,
            cursor=cursor,
            site=site,
            layer=layer,
            shape=tuple(int(dim) for dim in value.shape),
            dtype=str(value.dtype),
            bounded_sha256=bounded,
            bounded_elements=covered,
            total_elements=int(value.numel()),
            full_sha256=full,
            produced_by=produced_by,
            head_axis=head_axis,
        )
        self.observations.append(observation)
        if self.policy.retain_tensors:
            self.retained[address] = value.detach()
        return observation

    def cost_receipt(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "observation_count": len(self.observations),
            "fingerprint_ms": round(self.fingerprint_ms, 3),
            "retained_tensors": len(self.retained),
        }


class AddressRegistry:
    """Resolves ar:// addresses against producing observations only.

    Site addresses resolve when a matching observation exists.  Sliced
    addresses (``head=H``, ``channel=C``) additionally require coverage:
    either the full tensor was retained/fully fingerprinted, or the bounded
    window provably covers the requested slice.  Anything else is refused —
    never synthesized.
    """

    def __init__(self, log: ObservationLog) -> None:
        self._log = log

    def _find(self, address: str) -> PortObservation | None:
        base = address.split("?", 1)[0]
        for observation in self._log.observations:
            if observation.address == base:
                return observation
        return None

    def resolve(self, address: str) -> PortObservation:
        base, _, query = address.partition("?")
        observation = self._find(base)
        if observation is None:
            raise AddressRefused(
                f"no producing observable for {base!r}; "
                "an ar:// address exists only if a captured port covers it"
            )
        if query:
            self._require_slice_coverage(observation, query, address)
        return observation

    def _require_slice_coverage(
        self, observation: PortObservation, query: str, address: str
    ) -> None:
        if observation.full_sha256 is not None:
            return
        if address.split("?", 1)[0] in self._log.retained:
            return
        params = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
        if "head" in params:
            if observation.head_axis is None:
                raise AddressRefused(
                    f"{address!r}: producer declared no head axis for site {observation.site!r}"
                )
            # A bounded prefix window covers a head slice only when heads are
            # the leading varying axis of the flattened layout: head_axis == 1
            # with a singleton batch ([1, H, ...] contiguous).  Anywhere else
            # (e.g. attention output [B, S, H, dh]) head elements interleave
            # with other axes and a prefix window can never cover one head.
            if observation.head_axis != 1 or observation.shape[0] != 1:
                raise AddressRefused(
                    f"{address!r}: head slice not prefix-coverable for layout "
                    f"{observation.shape} with head_axis="
                    f"{observation.head_axis}. Re-capture with "
                    "full_fingerprint or retain_tensors."
                )
            per_head = observation.total_elements // observation.shape[1]
            head = int(params["head"])
            if not 0 <= head < observation.shape[1]:
                raise AddressRefused(
                    f"{address!r}: head {head} outside 0..{observation.shape[1] - 1}"
                )
            end = (head + 1) * per_head
            if end > observation.bounded_elements:
                raise AddressRefused(
                    f"{address!r}: bounded window covers {observation.bounded_elements} "
                    f"of {observation.total_elements} elements; head {head} slice ends at "
                    f"{end}. Re-capture with full_fingerprint or retain_tensors."
                )
            return
        raise AddressRefused(f"{address!r}: sliced address without coverage evidence")

    def known_addresses(self) -> tuple[str, ...]:
        return tuple(observation.address for observation in self._log.observations)


def observation_summary(observations: Iterable[PortObservation]) -> dict[str, Any]:
    by_site: dict[str, int] = {}
    for observation in observations:
        by_site[observation.site] = by_site.get(observation.site, 0) + 1
    return {"by_site": by_site, "total": sum(by_site.values())}


def jsonable_observations(observations: Iterable[PortObservation]) -> list[Mapping[str, Any]]:
    return [observation.to_dict() for observation in observations]

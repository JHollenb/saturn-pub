"""A small, frozen locator program for the next-composition assay.

The program describes two model-row locations.  It does not inspect a model,
decode token text, or claim that either row is a semantic concept address.  A
caller supplies the committed history and the two spans; the resolver then
returns the physical token rows that the native reader may inspect.

Both semantic roles are supplied declarations: A names the first token in the
preceding answer span, and B names the current follow-up copula.  The resolver
discovers only B's physical row by taking the last occurrence of the supplied
copula token inside the current follow-up span.  Keeping role provenance apart
from physical-row discovery prevents a discovered row from being mistaken for
a fixed semantic address when prompt length or tokenization changes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

READ_PROGRAM_SCHEMA = "saturn-mdb-frozen-read-program-v1"
PREDICTION_ASSISTANCE = "native-A-receiver"

# These limits keep a serialized declaration finite while leaving ample room
# for ordinary transformer configurations.  They are declaration limits only;
# this module never imports a model configuration or tensor runtime.
MAX_LAYER = 4095
MAX_KV_HEAD = 255

_PORT_NAMES = ("A", "B")
_ROLES = {
    "A": ("preceding-answer-first-generated", "supplied"),
    "B": ("current-followup-copula", "supplied"),
}


class ReadProgramError(ValueError):
    """Raised when a read-program declaration or resolution is invalid."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReadProgramError("read-program declaration is not canonical JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _plain_int(value: Any, *, label: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReadProgramError(f"{label} must be a plain integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f" in [{minimum}, {maximum}]" if maximum is not None else f" >= {minimum}"
        raise ReadProgramError(f"{label} must be{bound}")
    return int(value)


def _token_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ReadProgramError("history_token_ids must be a sequence of non-negative integers")
    result = tuple(
        _plain_int(token_id, label="history token id")
        for token_id in value
    )
    return result


def _span(value: Any, *, label: str, history_length: int) -> tuple[int, int]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ReadProgramError(f"{label} must be a (start, stop) sequence")
    if len(value) != 2:
        raise ReadProgramError(f"{label} must contain exactly (start, stop)")
    start = _plain_int(value[0], label=f"{label} start")
    stop = _plain_int(value[1], label=f"{label} stop")
    if start >= stop:
        raise ReadProgramError(f"{label} must be non-empty")
    if stop > history_length:
        raise ReadProgramError(f"{label} must lie fully within committed history")
    return start, stop


@dataclass(frozen=True, slots=True)
class ReadPort:
    """One immutable declared row locator role."""

    name: str
    layer: int
    kv_head: int
    row_role: str
    provenance: str

    @property
    def role_provenance(self) -> str:
        """A descriptive alias for callers that want the role's origin."""

        return self.provenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer,
            "kv_head": self.kv_head,
            "row_role": self.row_role,
            "provenance": self.provenance,
        }


def _normalize_ports(value: Any) -> tuple[ReadPort, ...]:
    if not isinstance(value, Mapping):
        raise ReadProgramError("ports must be an object keyed by A and B")
    if set(value) != set(_PORT_NAMES):
        raise ReadProgramError("ports must contain exactly A and B")

    ports: list[ReadPort] = []
    for name in _PORT_NAMES:
        raw = value[name]
        if isinstance(raw, ReadPort):
            data: dict[str, Any] = raw.to_dict()
        elif isinstance(raw, Mapping):
            data = dict(raw)
        else:
            raise ReadProgramError(f"port {name} must be an object")

        allowed = {"name", "layer", "kv_head", "row_role", "provenance"}
        unknown = set(data) - allowed
        if unknown:
            raise ReadProgramError(f"port {name} has unknown fields: {sorted(unknown)!r}")
        declared_name = data.get("name", name)
        if declared_name != name:
            raise ReadProgramError(f"port {name} has mismatched name")
        missing = [key for key in ("layer", "kv_head", "row_role", "provenance") if key not in data]
        if missing:
            raise ReadProgramError(f"port {name} is missing fields: {missing!r}")

        layer = _plain_int(data["layer"], label=f"port {name} layer", maximum=MAX_LAYER)
        kv_head = _plain_int(
            data["kv_head"], label=f"port {name} kv_head", maximum=MAX_KV_HEAD
        )
        role = data["row_role"]
        provenance = data["provenance"]
        if not isinstance(role, str) or not role:
            raise ReadProgramError(f"port {name} row_role must be a non-empty string")
        if not isinstance(provenance, str) or not provenance:
            raise ReadProgramError(f"port {name} provenance must be a non-empty string")
        expected_role, expected_provenance = _ROLES[name]
        if role != expected_role:
            raise ReadProgramError(f"unknown row role for port {name}: {role!r}")
        if provenance != expected_provenance:
            raise ReadProgramError(
                f"port {name} must declare {expected_provenance!r} role provenance"
            )
        ports.append(
            ReadPort(
                name=name,
                layer=layer,
                kv_head=kv_head,
                row_role=role,
                provenance=provenance,
            )
        )
    return tuple(ports)


@dataclass(frozen=True, slots=True, init=False)
class FrozenReadProgram:
    """Immutable declaration and resolver for the two-role read assay.

    ``ports`` is accepted as a mapping only to make malformed declarations
    fail early.  A normal caller should use :meth:`standard` or
    :func:`make_two_read_program`; no callback or native backend is involved.
    """

    _ports: tuple[ReadPort, ...] = field(repr=False)
    prediction_assistance: str
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = READ_PROGRAM_SCHEMA

    def __init__(
        self,
        ports: Mapping[str, Mapping[str, Any] | ReadPort] | None = None,
        *,
        prediction_assistance: str = PREDICTION_ASSISTANCE,
    ) -> None:
        if prediction_assistance != PREDICTION_ASSISTANCE:
            raise ReadProgramError(
                f"prediction_assistance must be {PREDICTION_ASSISTANCE!r}"
            )
        if ports is None:
            ports = {
                "A": {
                    "layer": 19,
                    "kv_head": 0,
                    "row_role": _ROLES["A"][0],
                    "provenance": _ROLES["A"][1],
                },
                "B": {
                    "layer": 22,
                    "kv_head": 0,
                    "row_role": _ROLES["B"][0],
                    "provenance": _ROLES["B"][1],
                },
            }
        normalized = _normalize_ports(ports)
        object.__setattr__(self, "_ports", normalized)
        object.__setattr__(self, "prediction_assistance", prediction_assistance)
        object.__setattr__(self, "fingerprint", _digest(self._body()))

    @classmethod
    def standard(cls) -> "FrozenReadProgram":
        """Return the fixed A/B declaration used by the next-composition assay."""

        return cls()

    @property
    def ports(self) -> Mapping[str, ReadPort]:
        """An immutable name-to-port view."""

        return MappingProxyType({port.name: port for port in self._ports})

    @property
    def declaration(self) -> Mapping[str, Any]:
        """A fresh immutable-friendly declaration view without its digest."""

        return self._body()

    def port(self, name: str) -> ReadPort:
        if name not in _PORT_NAMES:
            raise ReadProgramError(f"unknown read-program port: {name!r}")
        return self.ports[name]

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "ports": {port.name: port.to_dict() for port in self._ports},
            "prediction_assistance": self.prediction_assistance,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the declaration and its canonical SHA-256 fingerprint."""

        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenReadProgram":
        """Load a declaration only when its supplied fingerprint verifies."""

        if not isinstance(value, Mapping):
            raise ReadProgramError("read program must be an object")
        payload = dict(value)
        expected_keys = {"schema", "ports", "prediction_assistance", "fingerprint"}
        if set(payload) != expected_keys:
            raise ReadProgramError("read program has missing or unknown fields")
        if payload["schema"] != READ_PROGRAM_SCHEMA:
            raise ReadProgramError("read program schema mismatch")
        fingerprint = payload["fingerprint"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ReadProgramError("read program fingerprint must be a SHA-256 string")
        try:
            int(fingerprint, 16)
        except ValueError as exc:
            raise ReadProgramError("read program fingerprint must be hexadecimal") from exc
        body = {
            "schema": payload["schema"],
            "ports": payload["ports"],
            "prediction_assistance": payload["prediction_assistance"],
        }
        if _digest(body) != fingerprint:
            raise ReadProgramError("read program fingerprint mismatch")
        result = cls(
            ports=payload["ports"],
            prediction_assistance=payload["prediction_assistance"],
        )
        if result.fingerprint != fingerprint:
            # This should be unreachable, but keeps the digest check explicit
            # if the constructor's normalization ever evolves.
            raise ReadProgramError("read program normalized fingerprint mismatch")
        return result

    def resolve(
        self,
        history_token_ids: Sequence[int],
        preceding_answer_span: Sequence[int] = (0, 1),
        followup_span: Sequence[int] = (1, 2),
        copula_token_id: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Resolve A and B against one committed token history.

        Spans use Python's half-open ``(start, stop)`` convention.  A is the
        first row of the supplied preceding-answer span.  B is the last row
        whose token id equals ``copula_token_id`` *within* the supplied
        follow-up span; a missing in-span copula returns ``None`` as the second
        tuple item so a caller can abstain without inventing a row.  The tuple
        order is always ``(A, B)`` and each resolved address carries its port
        name for callers that prefer a name-keyed view.
        """

        history = _token_ids(history_token_ids)
        answer_span = _span(
            preceding_answer_span,
            label="preceding_answer_span",
            history_length=len(history),
        )
        current_span = _span(
            followup_span,
            label="followup_span",
            history_length=len(history),
        )
        if answer_span[1] > current_span[0]:
            raise ReadProgramError(
                "preceding_answer_span and followup_span must be disjoint and ordered"
            )
        copula_id = _plain_int(copula_token_id, label="copula_token_id")

        answer_start, answer_stop = answer_span
        followup_start, followup_stop = current_span
        answer_port = self.port("A")
        answer = {
            "port": answer_port.name,
            "layer": answer_port.layer,
            "kv_head": answer_port.kv_head,
            "row": answer_start,
            "source_token_id": history[answer_start],
            "role": answer_port.row_role,
            "provenance": answer_port.provenance,
            "lineage": {
                "provenance": answer_port.provenance,
                "role": answer_port.row_role,
                "source": "preceding_answer_span",
                "span": [answer_start, answer_stop],
                "occurrence": "first-generated",
            },
        }

        copula_row = next(
            (
                row
                for row in range(followup_stop - 1, followup_start - 1, -1)
                if history[row] == copula_id
            ),
            None,
        )
        if copula_row is None:
            return answer, None

        copula_port = self.port("B")
        copula = {
            "port": copula_port.name,
            "layer": copula_port.layer,
            "kv_head": copula_port.kv_head,
            "row": copula_row,
            "source_token_id": history[copula_row],
            "role": copula_port.row_role,
            "provenance": copula_port.provenance,
            "lineage": {
                "provenance": copula_port.provenance,
                "role": copula_port.row_role,
                "source": "followup_span",
                "span": [followup_start, followup_stop],
                "occurrence": "last-copula-in-span",
                "copula_token_id": copula_id,
            },
        }
        return answer, copula


def make_two_read_program() -> FrozenReadProgram:
    """Return the standard two-port declaration."""

    return FrozenReadProgram.standard()


__all__ = [
    "FrozenReadProgram",
    "MAX_KV_HEAD",
    "MAX_LAYER",
    "PREDICTION_ASSISTANCE",
    "READ_PROGRAM_SCHEMA",
    "ReadPort",
    "ReadProgramError",
    "make_two_read_program",
]

"""Optional, context-qualified debug information over existing typed ports."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import StateAddress
from .core import Session, StateCut
from .values import describe, digest


@dataclass(frozen=True)
class SymbolBinding:
    """A read-only local binding. Evidence labels confer no mutation authority.

    Context slots must cover the input dependencies relevant to the claim. The
    library checks their identity; it cannot infer whether the claim is complete.
    Each location is bound to an exact execution point and state schema.
    """

    symbol: str
    model_identity: str
    execution: str
    context: str
    context_slots: tuple[tuple[str, str], ...]
    locations: tuple[StateAddress, ...]
    evidence: tuple[str, ...]
    claim: str
    status: str = "supplied"
    abstain_reason: str | None = None
    confidence: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "locations", tuple(self.locations))
        object.__setattr__(self, "context_slots", tuple(tuple(row) for row in self.context_slots))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("symbol confidence must be a finite value in [0, 1]")
        if not all((self.symbol, self.model_identity, self.execution, self.context, self.claim)):
            raise ValueError("symbol requires identity, context and a bounded claim")
        if self.status not in {"supplied", "observed", "validated", "abstained"}:
            raise ValueError("unknown symbol status")
        if not self.evidence:
            raise ValueError("symbol requires evidence references")
        if self.status == "abstained":
            if not self.abstain_reason or self.locations:
                raise ValueError("abstention requires a reason and no executable locations")
        elif not self.locations or self.abstain_reason:
            raise ValueError("binding requires locations or explicit abstention")
        if any(not loc.execution_point or not loc.state_schema for loc in self.locations):
            raise ValueError("symbol locations require exact point and schema")
        if len({name for name, _ in self.context_slots}) != len(self.context_slots):
            raise ValueError("duplicate context slot")

    @classmethod
    def bind(
        cls,
        session: Session,
        symbol: str,
        locations: tuple[StateAddress, ...],
        *,
        context: str,
        context_slots: tuple[str, ...],
        evidence: tuple[str, ...],
        claim: str,
        status: str = "supplied",
    ) -> SymbolBinding:
        frame = session.inspect()
        point = digest(dict(frame.execution_point))
        schema = frame.surface["state_schema"]
        for location in locations:
            # Bare locations are qualified here; supplied qualifiers must not be erased.
            session.read_port(location)
        bound = tuple(
            StateAddress(loc.slot, loc.selector, loc.role, point, schema) for loc in locations
        )
        for location in bound:
            session.read_port(location)
        return cls(
            symbol,
            session.adapter.model_identity,
            digest(dict(session.adapter.execution)),
            context,
            tuple((name, digest(describe(session.read(name)))) for name in context_slots),
            bound,
            evidence,
            claim,
            status,
        )

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": "saturn-pub-debug-symbol-v1",
            "symbol": self.symbol,
            "model_identity": self.model_identity,
            "execution": self.execution,
            "context": self.context,
            "context_slots": dict(self.context_slots),
            "locations": [loc.to_dict() for loc in self.locations],
            "evidence": list(self.evidence),
            "claim": self.claim,
            "status": self.status,
            "abstain_reason": self.abstain_reason,
            "confidence": self.confidence,
            "authority": "read-only-debug-info",
        }
        return {**body, "fingerprint": digest(body)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SymbolBinding:
        body = {k: v for k, v in value.items() if k != "fingerprint"}
        if digest(body) != value.get("fingerprint"):
            raise ValueError("symbol seal mismatch")
        result = cls(
            value["symbol"],
            value["model_identity"],
            value["execution"],
            value["context"],
            tuple(value["context_slots"].items()),
            tuple(
                StateAddress(
                    loc["slot"],
                    tuple(loc["selector"]),
                    loc["role"],
                    loc["execution_point"],
                    loc["state_schema"],
                )
                for loc in value["locations"]
            ),
            tuple(value["evidence"]),
            value["claim"],
            value["status"],
            value["abstain_reason"],
            value["confidence"],
        )
        if result.to_dict() != value:
            raise ValueError("unsupported symbol schema or fields")
        return result

    def resolve(self, session: Session, *, context: str) -> tuple[StateAddress, ...]:
        if context != self.context or session.adapter.model_identity != self.model_identity:
            raise ValueError("symbol context or model mismatch")
        if digest(dict(session.adapter.execution)) != self.execution:
            raise ValueError("symbol execution mismatch")
        for name, expected in self.context_slots:
            if digest(describe(session.read(name))) != expected:
                raise ValueError(f"symbol input context changed: {name}")
        if self.status == "abstained":
            raise ValueError(f"symbol abstained: {self.abstain_reason}")
        for location in self.locations:
            session.read_port(location)
        return self.locations


class SymbolTable:
    """Several local bindings may describe one symbol at different safe points."""

    def __init__(self, bindings: tuple[SymbolBinding, ...] = ()):
        self.bindings = list(bindings)

    def save(self, path: str | Path, cut: StateCut) -> None:
        """Write sealed debug info beside a cut; never add execution authority."""
        cut.verify()
        for row in self.bindings:
            if row.model_identity != cut.model_identity or row.execution != digest(
                dict(cut.execution)
            ):
                raise ValueError("debug info belongs to another execution")
        body = {
            "schema": "saturn-pub-debug-info-v1",
            "cut": cut.fingerprint,
            "bindings": [row.to_dict() for row in self.bindings],
        }
        Path(path).write_text(json.dumps({**body, "fingerprint": digest(body)}, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path, cut: StateCut) -> SymbolTable:
        cut.verify()
        value = json.loads(Path(path).read_text())
        body = {k: v for k, v in value.items() if k != "fingerprint"}
        if (
            digest(body) != value.get("fingerprint")
            or body.get("cut") != cut.fingerprint
            or body.get("schema") != "saturn-pub-debug-info-v1"
            or set(body) != {"schema", "cut", "bindings"}
        ):
            raise ValueError("debug info seal or cut reference mismatch")
        rows = tuple(SymbolBinding.from_dict(row) for row in body["bindings"])
        if any(
            row.model_identity != cut.model_identity or row.execution != digest(dict(cut.execution))
            for row in rows
        ):
            raise ValueError("debug info execution mismatch")
        return cls(rows)

    def resolve(self, session: Session, symbol: str, *, context: str) -> SymbolBinding:
        candidates = [row for row in self.bindings if row.symbol == symbol]
        matches = []
        reasons = []
        for row in candidates:
            try:
                row.resolve(session, context=context)
                matches.append(row)
            except ValueError as exc:
                reasons.append(str(exc))
        if len(matches) != 1:
            raise ValueError(
                f"symbol {symbol!r} requires one qualified binding; "
                f"found {len(matches)}; {'; '.join(reasons) or 'no binding'}"
            )
        return matches[0]


def structural_symbols(session: Session) -> list[dict[str, Any]]:
    """Adapter declarations, not inferred semantic roles or measured writers."""
    frame = session.inspect()
    return [
        {
            **slot,
            "symbol": slot["name"],
            "status": "adapter-declared",
            "readable": slot["name"] in frame.slots,
            "authority": "structural-description",
        }
        for slot in frame.surface["slots"]
    ]


def backtrace(session: Session, slot: str) -> dict[str, Any]:
    """Walk recorded read/write footprints backwards from the current value.

    Unspecified native footprints stop the walk. Declared dependency edges are
    execution provenance, not proof of semantic causality. Fork-prefix receipts
    are not implicitly reconstructed from cut bytes.
    """
    surface = session.inspect().surface
    if slot not in {row["name"] for row in surface["slots"]}:
        raise ValueError(f"undeclared state slot: {slot}")
    needed = {slot}
    rows = []

    def expand(body, enclosing=None):
        if body["operation"] == "step" and body.get("micro_receipts"):
            for micro in reversed(body["micro_receipts"]):
                yield from expand(micro, body["fingerprint"])
        else:
            yield body, enclosing

    bodies = [item for receipt in reversed(session.receipts) for item in expand(receipt.to_dict())]
    for body, enclosing in bodies:
        if not needed:
            break
        receipt_id = body["fingerprint"]
        if body["operation"] == "restore":
            rows.append(
                {
                    "receipt": receipt_id,
                    "operation": "restore",
                    "status": "ancestry-boundary",
                    "cut": body["result"],
                }
            )
            break
        if body["operation"] == "apply":
            steps = [
                {
                    "native_operation": {
                        **body["act"],
                        "operation": body["act"]["name"],
                        "footprint": "declared",
                    },
                    "after": body.get("execution_point", {}),
                }
            ]
        elif body["operation"] == "continue":
            steps = body.get("transitions", [])
        else:
            rows.append(
                {
                    "receipt": receipt_id,
                    "operation": body["operation"],
                    "status": "unexpanded-history",
                }
            )
            break
        stop = False
        for index in range(len(steps) - 1, -1, -1):
            step = steps[index]
            spec = step.get("native_operation", {})
            if spec.get("footprint") != "declared":
                rows.append({"receipt": receipt_id, "status": "unknown-footprint"})
                stop = True
                break
            writes = set(spec.get("writes", [])) | set(spec.get("invalidates", []))
            used = needed & writes
            if used:
                rows.append(
                    {
                        "receipt": receipt_id,
                        "enclosing_receipt": enclosing,
                        "cut": body["result"] if index == len(steps) - 1 else None,
                        "cut_scope": "receipt-reference"
                        if index == len(steps) - 1
                        else "point-only",
                        "receipt_result": body["result"],
                        "operation": spec["operation"],
                        "writes_needed": sorted(used),
                        "reads": spec.get("reads", []),
                        "point": step["after"],
                        "status": "declared-dependency",
                    }
                )
                needed = (needed - writes) | set(spec.get("reads", []))
        if stop:
            break
    return {
        "slot": slot,
        "writers": rows,
        "unresolved": sorted(needed),
        "authority": "declared-provenance-not-semantic-proof",
    }

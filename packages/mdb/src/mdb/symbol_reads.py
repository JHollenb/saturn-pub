"""One writer for attention-derived symbol reads: "this head read this source with this mass".

Two observation surfaces produce the same records:

* Project-backed (``project_reads`` / ``reads_from_lifecycle``): an ordinary MDB Project with
  retained native cuts and receipts. Canonical records come from ``symbol_attention`` unchanged.
* Passive (``PassiveSurface`` + ``passive_*``): a non-resumable observation surface, either a
  passive ``StateCut`` (native discovery) or a saved capture with no cut at all (the FLUX
  read-graph converter). Canonical ``SymbolBinding`` / ``SymbolOccurrence`` / ``SymbolUse`` /
  ``SymbolLifecycle`` records are built here.

Both feed ``symbol_read`` / ``read_for_use``, which return a sealed, backend-neutral ``SymbolRead``:
source (token rows + role + how the role was assigned), reader (step, block, head, optional KV
group), observed mass, and a ``CausalStatus``. The causal status defaults to ``unknown`` (observed
only). Intervention results are attached explicitly, with cut type, floor, control, verdict and the
plan hash and run id they came from. A set-level result lives on a ``ReadSet``; it is never copied
onto the member reads.

Pointer rule: records never hold model weights or activations. Large data is referenced by
identity: model identity, parameter name, run/job id, file path + sha256 (+ array name). Every
record is checked by ``check_bounded``: no tensor-like values, no raw bytes, no sequence longer
than ``MAX_INLINE_ITEMS`` and no record larger than ``MAX_RECORD_BYTES``.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

READ_SCHEMA = "mdb-symbol-read-v1"
READ_SET_SCHEMA = "mdb-symbol-read-set-v1"
READS_DOCUMENT_SCHEMA = "mdb-symbol-reads-v1"
MAX_INLINE_ITEMS = 64
MAX_RECORD_BYTES = 64 * 1024
MAX_TEXT_BYTES = 4 * 1024

ROLE_ASSIGNMENTS = ("supplied", "nominated", "learned")
INTERVENTIONS = ("cut", "keep")
CUT_TYPES = ("renormalising", "no-renormalisation")
VERDICTS = ("supported", "not_supported", "unresolved", "control_missing")
UNKNOWN_LABEL = "unknown (observed only; no intervention)"


class SymbolReadError(ValueError):
    """A read record is malformed, unbounded, or claims more than its evidence."""


# --------------------------------------------------------------------------- bounds and seals

def check_bounded(value, label="record", *, _depth=0):
    """Return ``value`` as plain JSON, refusing arrays, tensors, bytes and oversized sequences."""
    if _depth > 16:
        raise SymbolReadError(label + " nests too deeply")
    if value is None or isinstance(value, (bool, str)):
        if isinstance(value, str) and len(value.encode()) > MAX_TEXT_BYTES:
            raise SymbolReadError(label + " text exceeds the inline limit")
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        if not math.isfinite(float(value)):
            raise SymbolReadError(label + " is not finite")
        return float(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise SymbolReadError(label + " holds raw bytes; reference the artifact by pointer")
    if any(hasattr(value, a) for a in ("shape", "ndim", "detach", "__array__")):
        raise SymbolReadError(label + " holds an array or tensor; reference it by pointer")
    if isinstance(value, Mapping):
        if len(value) > MAX_INLINE_ITEMS * 4:
            raise SymbolReadError(label + " has too many keys")
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SymbolReadError(label + " has a non-text key")
            out[key] = check_bounded(item, f"{label}.{key}", _depth=_depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_INLINE_ITEMS:
            raise SymbolReadError(
                f"{label} has {len(value)} items (> {MAX_INLINE_ITEMS}); store it by pointer")
        return [check_bounded(item, f"{label}[{i}]", _depth=_depth + 1) for i, item in enumerate(value)]
    raise SymbolReadError(f"{label} has unsupported type {type(value).__name__}")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def seal_digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sealed(body, label):
    body = check_bounded(body, label)
    if len(_canonical(body).encode()) > MAX_RECORD_BYTES:
        raise SymbolReadError(label + " exceeds the record byte budget")
    return {**body, "fingerprint": seal_digest(body)}


def _sha(value, label):
    text = str(value or "").removeprefix("sha256:")
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise SymbolReadError(label + " must be a sha256 hex digest")
    return text


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_pointer(path, *, array=None, selector=None, json_pointer=None, sha256=None, role=None):
    """Reference saved bytes by path + sha256 (computed when not supplied); never copies them."""
    path = Path(path)
    pointer = {"kind": "file", "path": str(path),
               "sha256": _sha(sha256, "file sha256") if sha256 else file_sha256(path)}
    for key, value in (("array", array), ("selector", selector), ("json_pointer", json_pointer),
                       ("role", role)):
        if value is not None:
            pointer[key] = list(value) if isinstance(value, tuple) else value
    return check_bounded(pointer, "file pointer")


def parameter_pointer(model_identity, parameter, *, selector=None, checkpoint_sha256=None):
    """Reference a model parameter by model identity + parameter name (+ optional selector)."""
    if not model_identity or not parameter:
        raise SymbolReadError("parameter pointer needs a model identity and parameter name")
    pointer = {"kind": "parameter", "model_identity": str(model_identity), "parameter": str(parameter)}
    if selector is not None:
        pointer["selector"] = list(selector)
    if checkpoint_sha256 is not None:
        pointer["checkpoint_sha256"] = _sha(checkpoint_sha256, "checkpoint sha256")
    return check_bounded(pointer, "parameter pointer")


def run_pointer(run_id, *, plan_sha256=None, path=None):
    pointer = {"kind": "run", "run_id": str(run_id)}
    if plan_sha256 is not None:
        pointer["plan_sha256"] = _sha(plan_sha256, "plan sha256")
    if path is not None:
        pointer["path"] = str(path)
    return check_bounded(pointer, "run pointer")


# --------------------------------------------------------------------------- read vocabulary

@dataclass(frozen=True)
class ReadSource:
    """Token rows read as one source, with a role name and how the role was assigned.

    ``supplied``: named by the experimenter (provenance needs ``by``). ``nominated``: chosen by a
    ranking (provenance needs ``ranking``, ``score`` and an ``evaluation`` that carries the
    mandatory trivial baselines of ``attention_ranking``). ``learned``: proposed by a fitted model
    (provenance needs ``predictor``).
    """
    name: str
    rows: tuple[int, ...]
    assignment: str = "supplied"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    binding_fingerprint: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "rows", tuple(check_bounded([int(r) for r in self.rows], "source rows")))
        object.__setattr__(self, "provenance", check_bounded(dict(self.provenance), "source provenance"))
        if not self.name or not self.rows or any(r < 0 for r in self.rows):
            raise SymbolReadError("a read source needs a name and non-negative token rows")
        if self.assignment not in ROLE_ASSIGNMENTS:
            raise SymbolReadError("role assignment must be one of " + ", ".join(ROLE_ASSIGNMENTS))
        p = self.provenance
        if self.assignment == "supplied" and not p.get("by"):
            raise SymbolReadError("a supplied role records who supplied it ('by')")
        if self.assignment == "nominated":
            from .attention_ranking import require_trivial_baselines
            if not p.get("ranking") or not isinstance(p.get("score"), (int, float)):
                raise SymbolReadError("a nominated role records its ranking and score")
            try:
                require_trivial_baselines(p.get("evaluation"))
            except ValueError as error:
                raise SymbolReadError("a nominated role carries the trivial baselines") from error
        if self.assignment == "learned" and not p.get("predictor"):
            raise SymbolReadError("a learned role records the predictor identity")
        if self.binding_fingerprint is not None:
            _sha(self.binding_fingerprint, "source binding fingerprint")

    def to_dict(self):
        return {"name": self.name, "rows": list(self.rows), "assignment": self.assignment,
                "provenance": dict(self.provenance), "binding_fingerprint": self.binding_fingerprint}

    @classmethod
    def from_dict(cls, value):
        return cls(value["name"], tuple(value["rows"]), value["assignment"], value["provenance"],
                   value.get("binding_fingerprint"))


@dataclass(frozen=True)
class Reader:
    """Who read: (step, block/layer, head), optional KV group and receiver description.

    ``step=None`` means aggregated over ``steps`` (stated explicitly). ``block_label`` keeps the
    backend's own name (e.g. ``single.7`` or ``native.layers.9:19``).
    """
    block: int
    head: int
    step: int | None = None
    steps: tuple[int, ...] | None = None
    kv_group: int | None = None
    block_label: str | None = None
    receivers: str | None = None
    receiver_row: int | None = None

    def __post_init__(self):
        for name in ("block", "head"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
                raise SymbolReadError(f"reader {name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        if self.steps is not None:
            object.__setattr__(self, "steps", tuple(int(s) for s in self.steps))
        if self.step is not None and self.steps is not None:
            raise SymbolReadError("give one step or the aggregated steps, not both")

    @classmethod
    def grouped(cls, block, head, *, n_heads, n_kv_heads, **kw):
        """Reader whose KV group is computed by ``attention_port.kv_group``."""
        from .attention_port import kv_group
        return cls(block, head, kv_group=kv_group(head, n_heads, n_kv_heads), **kw)

    def key(self):
        return (self.step if self.step is not None else -1, self.block, self.head)

    def to_dict(self):
        return {"block": self.block, "head": self.head, "step": self.step,
                "steps": None if self.steps is None else list(self.steps), "kv_group": self.kv_group,
                "block_label": self.block_label, "receivers": self.receivers,
                "receiver_row": self.receiver_row}

    @classmethod
    def from_dict(cls, value):
        return cls(value["block"], value["head"], value.get("step"),
                   None if value.get("steps") is None else tuple(value["steps"]), value.get("kv_group"),
                   value.get("block_label"), value.get("receivers"), value.get("receiver_row"))


@dataclass(frozen=True)
class CausalResult:
    """One intervention result with its provenance.

    ``intervention``: ``cut`` (the reads are removed) or ``keep`` (only these reads remain).
    ``effect``: the arm's score in ``effect_units`` (``None`` when the score is not usable; say
    why in ``reason``). ``control``: the matched control arm (``None`` if none was run).
    ``verdict``: ``supported`` (effect present and the control does not show it),
    ``not_supported`` (no effect, or the control shows the same), ``unresolved`` (mixed, or the
    score cannot be read), ``control_missing`` (effect present, no matched control).
    """
    intervention: str
    cut_type: str
    verdict: str
    plan_sha256: str
    run_id: str
    effect: float | None = None
    effect_units: str | None = None
    effect_per_seed: tuple | None = None
    floor: float | None = None
    floor_source: str | None = None
    control: Mapping[str, Any] | None = None
    reason: str | None = None
    claim: str | None = None
    evidence: tuple = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.intervention not in INTERVENTIONS:
            raise SymbolReadError("intervention must be 'cut' or 'keep'")
        if self.cut_type not in CUT_TYPES:
            raise SymbolReadError("cut type must be 'renormalising' or 'no-renormalisation'")
        if self.verdict not in VERDICTS:
            raise SymbolReadError("verdict must be one of " + ", ".join(VERDICTS))
        object.__setattr__(self, "plan_sha256", _sha(self.plan_sha256, "plan sha256"))
        if not self.run_id:
            raise SymbolReadError("an intervention result names the run it came from")
        if self.verdict == "supported" and not self.control:
            raise SymbolReadError("'supported' needs a control; use 'control_missing'")
        if self.verdict == "control_missing" and self.control:
            raise SymbolReadError("'control_missing' cannot carry a control")
        if self.effect is None and self.verdict == "supported":
            raise SymbolReadError("'supported' needs a readable effect")
        if self.effect_per_seed is not None:
            object.__setattr__(self, "effect_per_seed", tuple(self.effect_per_seed))
        object.__setattr__(self, "evidence", tuple(check_bounded(list(self.evidence), "evidence")))
        object.__setattr__(self, "metadata", check_bounded(dict(self.metadata), "result metadata"))
        if self.control is not None:
            object.__setattr__(self, "control", check_bounded(dict(self.control), "control"))

    def to_dict(self):
        return check_bounded({
            "intervention": self.intervention, "cut_type": self.cut_type, "verdict": self.verdict,
            "plan_sha256": self.plan_sha256, "run_id": self.run_id, "effect": self.effect,
            "effect_units": self.effect_units,
            "effect_per_seed": None if self.effect_per_seed is None else list(self.effect_per_seed),
            "floor": self.floor, "floor_source": self.floor_source, "control": self.control,
            "reason": self.reason, "claim": self.claim, "evidence": list(self.evidence),
            "metadata": dict(self.metadata)}, "causal result")

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "evidence": tuple(value.get("evidence") or ()),
                      "effect_per_seed": None if value.get("effect_per_seed") is None
                      else tuple(value["effect_per_seed"])})


@dataclass(frozen=True)
class CausalStatus:
    """``unknown`` until an intervention result is attached; the label says which."""
    results: tuple[CausalResult, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "results", tuple(
            r if isinstance(r, CausalResult) else CausalResult.from_dict(r) for r in self.results))

    @property
    def status(self):
        return "intervened" if self.results else "unknown"

    @property
    def label(self):
        if not self.results:
            return UNKNOWN_LABEL
        return "; ".join(f"{r.intervention} ({r.cut_type}): {r.verdict}" for r in self.results)

    def with_result(self, result):
        return CausalStatus(self.results + (result,))

    def to_dict(self):
        return {"status": self.status, "label": self.label,
                "results": [r.to_dict() for r in self.results]}

    @classmethod
    def from_dict(cls, value):
        status = cls(tuple(value.get("results") or ()))
        if value.get("status") != status.status or value.get("label") != status.label:
            raise SymbolReadError("causal status label differs from its results")
        return status


# --------------------------------------------------------------------------- records

@dataclass(frozen=True)
class SymbolRead:
    """One observed read with its canonical anchors and causal status (default ``unknown``)."""
    source: ReadSource
    reader: Reader
    mass: float
    mass_kind: str
    surface: str
    backend: str
    binding_fingerprint: str
    use_id: str | None = None
    use_fingerprint: str | None = None
    lifecycle_fingerprint: str | None = None
    sets: tuple[str, ...] = ()
    causal: CausalStatus = field(default_factory=CausalStatus)
    evidence: tuple = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.mass, bool) or not isinstance(self.mass, numbers.Real) or not math.isfinite(self.mass):
            raise SymbolReadError("observed mass must be a finite number")
        object.__setattr__(self, "mass", float(self.mass))
        if self.surface not in ("project-session", "passive-observation"):
            raise SymbolReadError("surface must be 'project-session' or 'passive-observation'")
        _sha(self.binding_fingerprint, "binding fingerprint")
        object.__setattr__(self, "sets", tuple(self.sets))
        object.__setattr__(self, "evidence", tuple(check_bounded(list(self.evidence), "read evidence")))
        object.__setattr__(self, "metadata", check_bounded(dict(self.metadata), "read metadata"))
        self.to_dict()

    def _body(self):
        return {"schema": READ_SCHEMA, "source": self.source.to_dict(), "reader": self.reader.to_dict(),
                "mass": self.mass, "mass_kind": self.mass_kind, "surface": self.surface,
                "backend": self.backend, "binding_fingerprint": self.binding_fingerprint,
                "use_id": self.use_id, "use_fingerprint": self.use_fingerprint,
                "lifecycle_fingerprint": self.lifecycle_fingerprint, "sets": list(self.sets),
                "causal": self.causal.to_dict(), "evidence": list(self.evidence),
                "metadata": dict(self.metadata)}

    def to_dict(self):
        return _sealed(self._body(), "symbol read")

    @property
    def fingerprint(self):
        return self.to_dict()["fingerprint"]

    def with_causal(self, result):
        """A per-read intervention result (e.g. one head cut alone). Never use for set results."""
        return replace(self, causal=self.causal.with_result(result))

    @classmethod
    def from_dict(cls, value):
        if value.get("schema") != READ_SCHEMA:
            raise SymbolReadError("unsupported read schema")
        record = cls(ReadSource.from_dict(value["source"]), Reader.from_dict(value["reader"]),
                     value["mass"], value["mass_kind"], value["surface"], value["backend"],
                     value["binding_fingerprint"], value.get("use_id"), value.get("use_fingerprint"),
                     value.get("lifecycle_fingerprint"), tuple(value.get("sets") or ()),
                     CausalStatus.from_dict(value["causal"]), tuple(value.get("evidence") or ()),
                     value.get("metadata") or {})
        if record.to_dict() != value:
            raise SymbolReadError("symbol read seal mismatch")
        return record


@dataclass(frozen=True)
class ReadSet:
    """A set of readers of one source, with set-level causal status.

    Membership is held by pointer (``members``: file pointer + count + digest); ``member_reads``
    optionally lists the fingerprints of member reads that exist as records. A set result says
    the SET was cut or kept. It does not say any member is individually necessary, and nothing
    here copies it onto member reads.
    """
    name: str
    source: ReadSource
    members: Mapping[str, Any]
    selection: Mapping[str, Any]
    causal: CausalStatus = field(default_factory=CausalStatus)
    member_reads: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        members = check_bounded(dict(self.members), "set members")
        if not members.get("pointer") or not isinstance(members.get("count"), int):
            raise SymbolReadError("set membership needs a pointer and a count")
        _sha(members.get("digest"), "member digest")
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "selection", check_bounded(dict(self.selection), "set selection"))
        object.__setattr__(self, "member_reads", tuple(self.member_reads))
        object.__setattr__(self, "metadata", check_bounded(dict(self.metadata), "set metadata"))
        self.to_dict()

    @staticmethod
    def membership(units, pointer):
        """Count + order-free digest of member units ((block, head) or (step, block, head))."""
        units = sorted(tuple(int(x) for x in u) for u in units)
        if len(set(units)) != len(units):
            raise SymbolReadError("set members repeat")
        return {"pointer": pointer, "count": len(units), "digest": seal_digest([list(u) for u in units]),
                "unit": "block,head" if units and len(units[0]) == 2 else "step,block,head"}

    @property
    def set_id(self):
        """Stable identity of (source, name, membership): independent of causal results."""
        return seal_digest({"source": self.source.to_dict(), "name": self.name,
                            "members": self.members["digest"], "selection": dict(self.selection)})

    def with_result(self, result):
        return replace(self, causal=self.causal.with_result(result))

    def _body(self):
        return {"schema": READ_SET_SCHEMA, "set_id": self.set_id, "name": self.name,
                "source": self.source.to_dict(), "members": dict(self.members),
                "selection": dict(self.selection), "causal": self.causal.to_dict(),
                "member_reads": list(self.member_reads), "metadata": dict(self.metadata),
                "scope": "set-level: results apply to the set as a whole, not to any member alone"}

    def to_dict(self):
        return _sealed(self._body(), "read set")

    @classmethod
    def from_dict(cls, value):
        if value.get("schema") != READ_SET_SCHEMA:
            raise SymbolReadError("unsupported read-set schema")
        record = cls(value["name"], ReadSource.from_dict(value["source"]), value["members"],
                     value["selection"], CausalStatus.from_dict(value["causal"]),
                     tuple(value.get("member_reads") or ()), value.get("metadata") or {})
        if record.to_dict() != value:
            raise SymbolReadError("read set seal mismatch")
        return record


# --------------------------------------------------------------------------- common constructor

def symbol_read(source, reader, mass, *, mass_kind, surface, backend, binding_fingerprint,
                use=None, lifecycle=None, sets=(), evidence=(), metadata=None):
    """The single constructor both surfaces use. Causal status always starts ``unknown``."""
    return SymbolRead(source, reader, mass, mass_kind, surface, backend, binding_fingerprint,
                      use_id=None if use is None else use.use_id,
                      use_fingerprint=None if use is None else use.fingerprint,
                      lifecycle_fingerprint=None if lifecycle is None else lifecycle.fingerprint,
                      sets=tuple(sets), evidence=tuple(evidence), metadata=metadata or {})


def read_for_use(use, lifecycle, *, source, reader, mass, mass_kind, surface, backend,
                 sets=(), evidence=(), metadata=None):
    if use.binding_fingerprint != lifecycle.binding_fingerprint:
        raise SymbolReadError("use belongs to another binding")
    if use.use_id not in {u.use_id for u in lifecycle.uses}:
        raise SymbolReadError("use is not part of this lifecycle")
    return symbol_read(source, reader, mass, mass_kind=mass_kind, surface=surface, backend=backend,
                       binding_fingerprint=lifecycle.binding_fingerprint, use=use, lifecycle=lifecycle,
                       sets=sets, evidence=evidence or tuple(use.evidence_refs), metadata=metadata)


# --------------------------------------------------------------------------- Project-backed variant

def reads_from_lifecycle(lifecycle, *, source, backend="qwen-gqa", step=None,
                         mass_observable="source_mass"):
    """Derive reads from an ``attach_attention`` lifecycle (one per query-head use)."""
    trace = dict(lifecycle.metadata).get("attention_trace") or {}
    layer = trace.get("layer")
    if layer is None:
        raise SymbolReadError("lifecycle carries no attention trace layer")
    if source.binding_fingerprint not in (None, lifecycle.binding_fingerprint):
        raise SymbolReadError("source names another binding")
    reads = []
    for use in lifecycle.uses:
        meta = dict(use.metadata)
        masses = [o["value"] for o in use.signed_observations
                  if isinstance(o, Mapping) and o.get("observable") == mass_observable]
        if len(masses) != 1:
            raise SymbolReadError("use lacks exactly one observed mass")
        reader = Reader(int(layer), int(meta["query_head"]), step=step, kv_group=meta.get("kv_head"),
                        block_label=f"layer.{layer}", receivers="query row",
                        receiver_row=meta.get("receiver_row"))
        reads.append(read_for_use(use, lifecycle, source=source, reader=reader, mass=masses[0],
                                  mass_kind="attention probability on source row",
                                  surface="project-session", backend=backend,
                                  metadata={"trace_fingerprint": meta.get("trace_fingerprint")}))
    return tuple(reads)


def project_reads(project, session_id, lifecycle, *, source, step=None, backend="qwen-gqa", **attach):
    """``symbol_attention.attach_attention`` (unchanged) plus the reads it observed."""
    from .symbol_attention import attach_attention
    joined = attach_attention(project, session_id, lifecycle, **attach)
    return joined, reads_from_lifecycle(joined, source=replace(
        source, binding_fingerprint=joined.binding_fingerprint), backend=backend, step=step)


# --------------------------------------------------------------------------- passive variant

@dataclass(frozen=True)
class PassiveSurface:
    """A non-resumable observation surface: a passive StateCut, or a saved capture with no cut.

    ``cut_ref`` is the StateCut fingerprint when one exists; otherwise a labelled reference to
    the capture (``passive-capture:sha256:<hex>``), never a fabricated cut id.
    """
    model_identity: str
    execution: Mapping[str, Any]
    execution_point: Mapping[str, Any]
    surface: Mapping[str, Any]
    cut_ref: str
    session_id: str
    branch: str = "ordinary-execution"
    slots: Mapping[str, Any] | None = None

    @classmethod
    def from_cut(cls, cut, *, session_id, branch="ordinary-execution", slots=None):
        return cls(cut.model_identity, dict(cut.execution), dict(cut.execution_point),
                   dict(cut.surface), cut.fingerprint, session_id, branch, slots)

    @property
    def point(self):
        from saturn_pub.values import digest
        return digest(dict(self.execution_point))

    @property
    def state_schema(self):
        return self.surface["state_schema"]

    def address(self, slot, selector, role):
        from saturn_pub.contracts import StateAddress
        return StateAddress(slot, tuple(selector), role, self.point, self.state_schema)


def passive_binding(surface, symbol, locations, *, context_slots, evidence, claim, status="observed"):
    """Binding over passive locations; qualified by metadata when the surface declares slots."""
    from saturn_pub.symbols import SymbolBinding
    from saturn_pub.values import digest
    binding = SymbolBinding(symbol, surface.model_identity, digest(dict(surface.execution)),
                            surface.session_id, tuple(context_slots), tuple(locations),
                            tuple(evidence), claim, status=status)
    if surface.slots is not None:
        binding.resolve_metadata(model_identity=surface.model_identity, execution=dict(surface.execution),
                                 context=surface.session_id, slots=dict(surface.slots),
                                 surface=dict(surface.surface),
                                 execution_point=dict(surface.execution_point))
    return binding


def passive_definition(surface, binding, *, occurrence_id, clock, evidence_refs, metadata, status="observed"):
    from .symbol_lifecycle import SymbolOccurrence
    check_bounded(dict(metadata), "definition metadata")
    return SymbolOccurrence.for_binding(binding, occurrence_id=occurrence_id, session_id=surface.session_id,
        source_cut=surface.cut_ref, locations=binding.locations, branch=surface.branch,
        execution=binding.execution, clock=clock, evidence_refs=tuple(evidence_refs), status=status,
        metadata=metadata)


def passive_use(surface, binding, *, use_id, consumer, signed_observations, evidence_refs,
                result_cut=None, receipt=None, locations=None, metadata=None, status="observed"):
    from .symbol_lifecycle import SymbolUse
    check_bounded(list(signed_observations), "signed observations")
    check_bounded(dict(metadata or {}), "use metadata")
    return SymbolUse(use_id, binding.to_dict()["fingerprint"], surface.session_id, consumer,
                     source_cut=surface.cut_ref, parent_cut=surface.cut_ref, result_cut=result_cut,
                     branch=surface.branch, execution=binding.execution, receipt=receipt,
                     locations=binding.locations if locations is None else tuple(locations),
                     evidence_refs=tuple(evidence_refs), status=status,
                     signed_observations=tuple(signed_observations), metadata=metadata or {})


def passive_lifecycle(binding, *, definitions, uses, claim_boundary, metadata=None):
    from .symbol_lifecycle import SymbolLifecycle
    return SymbolLifecycle.define(binding, definitions=tuple(definitions), uses=tuple(uses),
        mechanics_status="valid", instrument_status="observations-only", trend_status="not-assessed",
        terminal_status="not-assessed", claim_boundary=claim_boundary, metadata=metadata or {})


# --------------------------------------------------------------------------- persistence

def save_reads(path, reads=(), sets=(), *, meta=None):
    """Write reads and read sets to a new JSON file (refuses to overwrite)."""
    path = Path(path)
    if path.exists():
        raise SymbolReadError("reads file already exists: " + str(path))
    body = {"schema": READS_DOCUMENT_SCHEMA, "reads": [r.to_dict() for r in reads],
            "sets": [s.to_dict() for s in sets], "meta": check_bounded(dict(meta or {}), "meta")}
    known = {r["fingerprint"] for r in body["reads"]}
    set_ids = {s["set_id"] for s in body["sets"]}
    for s in body["sets"]:
        if not set(s["member_reads"]) <= known:
            raise SymbolReadError("read set names a member read that is not in this file")
    for r in body["reads"]:
        if not set(r["sets"]) <= set_ids:
            raise SymbolReadError("read names a set that is not in this file")
    document = {**body, "fingerprint": seal_digest(body)}
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": file_sha256(path), "reads": len(reads), "sets": len(sets)}


def load_reads(path):
    document = json.loads(Path(path).read_text())
    if document.get("schema") != READS_DOCUMENT_SCHEMA:
        raise SymbolReadError("unsupported reads document")
    body = {k: v for k, v in document.items() if k != "fingerprint"}
    if seal_digest(body) != document.get("fingerprint"):
        raise SymbolReadError("reads document seal mismatch")
    return {"reads": tuple(SymbolRead.from_dict(r) for r in document["reads"]),
            "sets": tuple(ReadSet.from_dict(s) for s in document["sets"]),
            "meta": document["meta"], "fingerprint": document["fingerprint"]}

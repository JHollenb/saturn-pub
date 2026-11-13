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
only). Intervention results are attached explicitly as typed ``CausalResult`` records with an
explicit ``scope``: ``read`` results go on a ``SymbolRead`` (``with_causal``), ``set`` results on a
``ReadSet`` (``with_result``); each refuses the other scope, so a set result is never copied onto
member reads.

Pointer rule: records never hold model weights or activations. Large data is referenced by a
locator (declared root name + relative path: where) and a digest (sha256 + size: what); moving
a file changes the root declaration, never the reference. A model name that stands in for a
verified checkpoint identity carries ``identity_status: "unresolved"`` and what would resolve it.

Bounds (``check_bounded``, applied to every record, result, passive use/definition metadata and
document ``meta``): no tensor-like values, no raw bytes, no non-finite numbers; at most
``MAX_INLINE_ITEMS`` items in a sequence, counting every item of sequences nested inside it; at
most ``MAX_RECORD_ITEMS`` values in total; at most ``MAX_RECORD_BYTES`` of canonical JSON
(``MAX_META_BYTES`` for ``meta``); text at most ``MAX_TEXT_BYTES``. numpy scalars (including
``np.bool_``) convert to Python scalars; numpy arrays (also 0-d) are refused.
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

try:
    import numpy as _np
except ImportError:  # pragma: no cover - numpy is a dependency of mdb
    _np = None

READ_SCHEMA = "mdb-symbol-read-v2"
READ_SET_SCHEMA = "mdb-symbol-read-set-v2"
READS_DOCUMENT_SCHEMA = "mdb-symbol-reads-v2"
MAX_INLINE_ITEMS = 64
MAX_RECORD_ITEMS = 4096
MAX_RECORD_BYTES = 64 * 1024
MAX_META_BYTES = 64 * 1024
MAX_TEXT_BYTES = 4 * 1024

ROLE_ASSIGNMENTS = ("supplied", "nominated", "learned")
# How a ReadSet's members were chosen. Deliberately disjoint from ROLE_ASSIGNMENTS: a set is
# "ranked", never "nominated" (that word means a source role chosen by a ranking with baselines).
SET_SELECTION_KINDS = ("ranked", "random", "group", "category", "category-unique", "control", "supplied")
INTERVENTIONS = ("cut", "keep")
CUT_TYPES = ("renormalising", "no-renormalisation")
VERDICTS = ("supported", "not_supported", "unresolved", "control_missing")
RESULT_SCOPES = ("read", "set")
# What decided the verdict. ``picture``: audited picture calls (scores attached as context);
# ``picture_only``: picture calls, score omitted as unusable; ``score``: the numeric score;
# ``audit_statement``: an audit sentence with no per-arm call; ``none``: nothing decides.
BASES = ("score", "picture", "picture_only", "audit_statement", "none")
# Vocabulary shared with attention_panels: claim per intervention, alternatives per claim.
CLAIMS = {"cut": "necessary", "keep": "sufficient"}
ALTERNATIVES = {"necessary": ("generic_damage", "mass", "set_size"), "sufficient": ("mass", "set_size")}
CLAIM_SCOPES = {"necessary": "needed_set", "sufficient": "sufficient_set"}
JUDGED_BY = ("picture", "score", "picture+score")
CONTROL_JUDGEMENTS = ("clean", "same", "mixed")
REFERENCE_EFFECTS = ("none", "downgrade_only")
SIZE_MATCH_MIN_RATIO = 0.9
UNKNOWN_LABEL = "unknown (observed only; no intervention)"


class SymbolReadError(ValueError):
    """A read record is malformed, unbounded, or claims more than its evidence."""


# --------------------------------------------------------------------------- bounds and seals

def check_bounded(value, label="record", *, max_bytes=MAX_RECORD_BYTES, max_items=MAX_RECORD_ITEMS):
    """Return ``value`` as plain JSON, refusing arrays, tensors, bytes and oversized data.

    Enforced recursively: no tensor-like value (0-d arrays included), no raw bytes, no
    non-finite number, text <= ``MAX_TEXT_BYTES``, mappings <= 4 x ``MAX_INLINE_ITEMS`` keys,
    any sequence <= ``MAX_INLINE_ITEMS`` items counting every item of the sequences nested in
    it (a 48 x 48 list holds 2352), <= ``max_items`` values in total, and <= ``max_bytes`` of
    canonical JSON. numpy scalars (``np.bool_``, ``np.int64``, ``np.float32``...) convert.
    """
    count = [0]
    out = _bounded(value, label, 0, count, max_items, None)
    if max_bytes is not None:
        size = len(_canonical(out).encode())
        if size > max_bytes:
            raise SymbolReadError(f"{label} is {size} bytes (> {max_bytes}); store it by pointer")
    return out


def _bounded(value, label, depth, count, max_items, run):
    if depth > 16:
        raise SymbolReadError(label + " nests too deeply")
    count[0] += 1
    if count[0] > max_items:
        raise SymbolReadError(f"{label}: more than {max_items} values in one record; store it by pointer")
    if _np is not None and isinstance(value, _np.generic):
        value = value.item()
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
            out[key] = _bounded(item, f"{label}.{key}", depth + 1, count, max_items, None)
        return out
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_INLINE_ITEMS:
            raise SymbolReadError(
                f"{label} has {len(value)} items (> {MAX_INLINE_ITEMS}); store it by pointer")
        # Items of directly nested sequences share the outermost sequence's item budget.
        run = [0] if run is None else run
        run[0] += len(value)
        if run[0] > MAX_INLINE_ITEMS:
            raise SymbolReadError(f"{label} holds more than {MAX_INLINE_ITEMS} items across nested "
                                  "sequences; store it by pointer")
        return [_bounded(item, f"{label}[{i}]", depth + 1, count, max_items,
                         run if isinstance(item, (list, tuple)) else None)
                for i, item in enumerate(value)]
    raise SymbolReadError(f"{label} has unsupported type {type(value).__name__}")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def seal_digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sealed(body, label):
    body = check_bounded(body, label)
    return {**body, "fingerprint": seal_digest(body)}


def _sha(value, label):
    text = str(value or "").removeprefix("sha256:")
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise SymbolReadError(label + " must be a sha256 hex digest")
    return text


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SymbolReadError(label + " must be non-empty text without surrounding whitespace")
    return value


def _number(value, label, *, optional=True):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise SymbolReadError(label + " must be a finite number")
    return float(value)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- pointers and identity

def declare_root(name, path):
    """A named storage root: pointers hold ``name`` + a relative path, never the absolute path."""
    return {"name": _text(name, "root name"), "path": str(Path(path).resolve())}


def file_pointer(path, *, root, array=None, selector=None, json_pointer=None, sha256=None, role=None):
    """Reference saved bytes: locator (root name + relative path) and digest (sha256 + size).

    ``root`` comes from ``declare_root``; the file must lie under it. ``sha256`` is computed when
    not supplied. The bytes are never copied.
    """
    path = Path(path).resolve()
    try:
        relative = path.relative_to(Path(root["path"]))
    except ValueError as error:
        raise SymbolReadError(f"{path} is not under the declared root {root['name']!r}") from error
    pointer = {"kind": "file", "locator": {"root": root["name"], "path": relative.as_posix()},
               "digest": {"sha256": _sha(sha256, "file sha256") if sha256 else file_sha256(path),
                          "size": path.stat().st_size}}
    for key, value in (("array", array), ("selector", selector), ("json_pointer", json_pointer),
                       ("role", role)):
        if value is not None:
            pointer[key] = list(value) if isinstance(value, tuple) else value
    return check_bounded(pointer, "file pointer")


def resolve_pointer(pointer, roots):
    """Path of a file pointer under ``roots`` ({name: path}); refuses an undeclared root."""
    locator = pointer["locator"]
    if locator["root"] not in roots:
        raise SymbolReadError(f"pointer names undeclared root {locator['root']!r}")
    return Path(roots[locator["root"]]) / locator["path"]


def verify_pointer(pointer, roots):
    """Resolve a file pointer and check its digest (size, then sha256)."""
    path = resolve_pointer(pointer, roots)
    digest = pointer["digest"]
    if path.stat().st_size != digest["size"] or file_sha256(path) != digest["sha256"]:
        raise SymbolReadError(f"{path} does not match its pointer digest")
    return path


def model_identity(name, *, checkpoint_sha256=None, resolve_by=None):
    """A model name plus whether it is a verified identity.

    With ``checkpoint_sha256`` the status is ``verified``. Without it the name only stands in
    for an identity: status ``unresolved``, and ``resolve_by`` must say what would resolve it.
    """
    record = {"name": _text(name, "model name")}
    if checkpoint_sha256 is not None:
        record.update(identity_status="verified", checkpoint_sha256=_sha(checkpoint_sha256, "checkpoint sha256"))
    else:
        if not resolve_by:
            raise SymbolReadError("an unverified model name says what would resolve it ('resolve_by')")
        record.update(identity_status="unresolved", resolve_by=str(resolve_by))
    return check_bounded(record, "model identity")


def _identity_record(value):
    if value is None:
        return None
    value = dict(value)
    if value.get("identity_status") == "verified":
        return model_identity(value.get("name"), checkpoint_sha256=value.get("checkpoint_sha256"))
    if value.get("identity_status") == "unresolved":
        return model_identity(value.get("name"), resolve_by=value.get("resolve_by"))
    raise SymbolReadError("model identity_status must be 'verified' or 'unresolved'")


def parameter_pointer(model_identity_name, parameter, *, selector=None, checkpoint_sha256=None, resolve_by=None):
    """Reference a model parameter by model identity + parameter name (+ optional selector).

    Without ``checkpoint_sha256`` the model name is not a verified identity: the pointer says
    ``identity_status: "unresolved"`` and needs ``resolve_by``.
    """
    if not model_identity_name or not parameter:
        raise SymbolReadError("parameter pointer needs a model identity and parameter name")
    pointer = {"kind": "parameter", "parameter": str(parameter),
               "model": model_identity(str(model_identity_name), checkpoint_sha256=checkpoint_sha256,
                                       resolve_by=resolve_by or ("checkpoint sha256 of the file holding "
                                                                 "this parameter"))}
    if selector is not None:
        pointer["selector"] = list(selector)
    return check_bounded(pointer, "parameter pointer")


def run_pointer(run_id, *, plan_sha256=None, plan=None):
    """Reference a run by id, its plan hash and optionally the plan's file pointer."""
    pointer = {"kind": "run", "run_id": _text(run_id, "run id")}
    if plan_sha256 is not None:
        pointer["plan_sha256"] = _sha(plan_sha256, "plan sha256")
    if plan is not None:
        if plan_sha256 is not None and plan["digest"]["sha256"] != pointer["plan_sha256"]:
            raise SymbolReadError("plan pointer digest differs from plan sha256")
        pointer["plan"] = dict(plan)
    return check_bounded(pointer, "run pointer")


def verdict_rule(rule_id, rule):
    """A verdict rule written in one place: ``{"id", "sha256", "rule"}`` (sha256 of ``rule``)."""
    rule = check_bounded(dict(rule), "verdict rule", max_bytes=MAX_META_BYTES)
    return {"id": _text(rule_id, "rule id"), "sha256": seal_digest(rule), "rule": rule}


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


def _picture(value):
    if value is None:
        return None
    value = check_bounded(dict(value), "picture")
    calls = value.get("calls")
    if not isinstance(calls, list) or not calls or not all(isinstance(c, str) and c.strip() for c in calls):
        raise SymbolReadError("picture needs a non-empty list of picture calls")
    return value


def _control(value, *, claim, run_id, counted):
    """Validate a control record. ``counted``: the one control the verdict uses (must be in-run)."""
    label = "control" if counted else "reference control"
    if not isinstance(value, Mapping):
        raise SymbolReadError(label + " must be a structured record")
    value = check_bounded(dict(value), label)
    _text(value.get("arm"), label + " arm")
    _text(value.get("run_id"), label + " run id")
    if not isinstance(value.get("in_run"), bool):
        raise SymbolReadError(label + " says whether it ran in the same run ('in_run')")
    matched = value.get("matched_on")
    if not isinstance(matched, list) or not matched or not all(isinstance(m, str) and m for m in matched):
        raise SymbolReadError(label + " says what it was matched on ('matched_on')")
    if value.get("judged_by") not in JUDGED_BY:
        raise SymbolReadError(label + " says what it was judged by: " + ", ".join(JUDGED_BY))
    judgement = value.get("judgement")
    if judgement is not None and judgement not in CONTROL_JUDGEMENTS:
        raise SymbolReadError(label + " judgement must be one of " + ", ".join(CONTROL_JUDGEMENTS))
    for key in ("effect",):
        _number(value.get(key), f"{label} {key}")
    rules_out = value.get("rules_out") or []
    if not set(rules_out) <= set(ALTERNATIVES[claim]):
        raise SymbolReadError(f"{label} rules_out must be within {ALTERNATIVES[claim]}")
    if rules_out and (not value["in_run"] or judgement != "clean"):
        raise SymbolReadError(label + " rules alternatives out only when in-run and judged clean")
    if "set_size" in rules_out:
        size, target = value.get("size"), value.get("set_size")
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in (size, target)) or \
                size < SIZE_MATCH_MIN_RATIO * target:
            raise SymbolReadError(label + f" rules out set_size only with size >= {SIZE_MATCH_MIN_RATIO} x set_size")
    if "mass" in rules_out:
        held, target = value.get("mass_held"), value.get("set_mass_held")
        more = (isinstance(held, list) and isinstance(target, list) and len(held) == len(target) > 0
                and all(_number(c, "mass") >= _number(t, "mass") for c, t in zip(held, target)))
        if "mass" not in matched and not more:
            raise SymbolReadError(label + " rules out mass only if matched on mass or holding at least "
                                  "the set's mass on every seed")
    if counted:
        if value["in_run"] is not True:
            raise SymbolReadError("a counted control must come from the same run (in_run true); "
                                  "record other runs under reference_controls")
        if value["run_id"] != run_id:
            raise SymbolReadError("a counted control names the result's own run id")
        if judgement is None:
            raise SymbolReadError("a counted control carries its judgement")
    else:
        if value.get("verdict_effect") not in REFERENCE_EFFECTS:
            raise SymbolReadError("a reference control says its verdict effect: " + ", ".join(REFERENCE_EFFECTS))
        if value["verdict_effect"] != "none" and not value["in_run"]:
            raise SymbolReadError("a control from another run never affects the verdict")
    return value


@dataclass(frozen=True)
class CausalResult:
    """One intervention result with its provenance (see docs/SYMBOL-READS.md, "Causal status").

    ``scope``: ``read`` (one read intervened alone) or ``set`` (a whole ReadSet); enforced by
    ``SymbolRead`` / ``ReadSet``. ``basis``: what decided the verdict (``BASES``).
    ``claim``: ``necessary`` (cut) or ``sufficient`` (keep). ``control``: the ONE in-run control
    matched for this claim, as a structured record (arm, run_id, in_run=True, matched_on,
    judged_by, judgement); ``reference_controls``: other arms (other runs, unmatched in-run
    arms) that never make a result ``supported``. ``ruled_out`` / ``remaining_alternatives``
    partition ``ALTERNATIVES[claim]``; ``claim_scope`` is set only on ``supported`` and is
    ``identity`` only when nothing remains.
    """
    intervention: str
    cut_type: str
    verdict: str
    plan_sha256: str
    run_id: str
    scope: str | None = None
    basis: str | None = None
    effect: float | None = None
    effect_units: str | None = None
    effect_per_seed: tuple | None = None
    floor: float | None = None
    floor_source: str | None = None
    floor_incomplete: bool | None = None
    control: Mapping[str, Any] | None = None
    reference_controls: tuple = ()
    picture: Mapping[str, Any] | None = None
    claim: str | None = None
    claim_text: str | None = None
    claim_scope: str | None = None
    ruled_out: tuple = ()
    remaining_alternatives: tuple | None = None
    contrast_id: str | None = None
    rule_id: str | None = None
    rule_sha256: str | None = None
    plan: Mapping[str, Any] | None = None
    reason: str | None = None
    evidence: tuple = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        setter = object.__setattr__
        if self.intervention not in INTERVENTIONS:
            raise SymbolReadError("intervention must be 'cut' or 'keep'")
        if self.cut_type not in CUT_TYPES:
            raise SymbolReadError("cut type must be 'renormalising' or 'no-renormalisation'")
        if self.verdict not in VERDICTS:
            raise SymbolReadError("verdict must be one of " + ", ".join(VERDICTS))
        if self.scope not in RESULT_SCOPES:
            raise SymbolReadError("result scope must be 'read' or 'set'")
        if self.basis not in BASES:
            raise SymbolReadError("basis must be one of " + ", ".join(BASES))
        setter(self, "plan_sha256", _sha(self.plan_sha256, "plan sha256"))
        if not isinstance(self.run_id, str) or not self.run_id.strip() or self.run_id != self.run_id.strip():
            raise SymbolReadError("an intervention result names the run it came from (non-empty run id)")
        claim = CLAIMS[self.intervention]
        if self.claim not in (None, claim):
            raise SymbolReadError(f"a {self.intervention} result claims '{claim}'")
        setter(self, "claim", claim)
        if self.plan is not None:
            setter(self, "plan", check_bounded(dict(self.plan), "plan pointer"))
            if self.plan.get("digest", {}).get("sha256") != self.plan_sha256:
                raise SymbolReadError("plan pointer digest differs from plan sha256")
        setter(self, "effect", _number(self.effect, "effect"))
        setter(self, "floor", _number(self.floor, "floor"))
        if self.effect_per_seed is not None:
            setter(self, "effect_per_seed", tuple(_number(v, "per-seed effect", optional=False)
                                                  for v in self.effect_per_seed))
        if self.floor is not None:
            if self.floor < 0:
                raise SymbolReadError("floor is a magnitude (>= 0)")
            if not self.floor_source:
                raise SymbolReadError("a floor says where it came from ('floor_source')")
        if self.floor_incomplete is not None and not isinstance(self.floor_incomplete, bool):
            raise SymbolReadError("floor_incomplete is a bool")
        setter(self, "picture", _picture(self.picture))
        if self.basis in ("picture", "picture_only") and self.picture is None:
            raise SymbolReadError(f"basis '{self.basis}' needs the picture calls")
        if self.control is not None:
            setter(self, "control", _control(self.control, claim=claim, run_id=self.run_id, counted=True))
        setter(self, "reference_controls", tuple(
            _control(c, claim=claim, run_id=self.run_id, counted=False) for c in self.reference_controls))
        # Alternatives: ruled out only by a clean in-run record that says so.
        ruled = tuple(sorted(set(self.ruled_out)))
        if not set(ruled) <= set(ALTERNATIVES[claim]):
            raise SymbolReadError(f"ruled_out must be within {ALTERNATIVES[claim]}")
        justified = set((self.control or {}).get("rules_out") or ())
        for c in self.reference_controls:
            justified |= set(c.get("rules_out") or ())
        if not set(ruled) <= justified:
            raise SymbolReadError("ruled_out " + ", ".join(sorted(set(ruled) - justified))
                                  + " is not backed by a clean in-run control that rules it out")
        remaining = tuple(a for a in sorted(ALTERNATIVES[claim]) if a not in ruled)
        if self.remaining_alternatives is not None and tuple(sorted(self.remaining_alternatives)) != remaining:
            raise SymbolReadError("remaining_alternatives must be the alternatives not ruled out")
        setter(self, "ruled_out", ruled)
        setter(self, "remaining_alternatives", remaining)
        if self.verdict == "supported":
            expected = "identity" if not remaining else CLAIM_SCOPES[claim]
            if self.claim_scope != expected:
                raise SymbolReadError(f"a supported {claim} result has claim_scope '{expected}' "
                                      f"(remaining: {', '.join(remaining) or 'none'})")
        elif self.claim_scope is not None:
            raise SymbolReadError("claim_scope is set only on supported results")
        if self.scope == "set" and not (isinstance(self.contrast_id, str) and self.contrast_id.strip()):
            raise SymbolReadError("a set-level result names its independent contrast ('contrast_id')")
        if (self.rule_id is None) != (self.rule_sha256 is None):
            raise SymbolReadError("rule_id and rule_sha256 come together")
        if self.rule_sha256 is not None:
            setter(self, "rule_sha256", _sha(self.rule_sha256, "rule sha256"))
        if self.verdict == "supported":
            self._check_supported()
        if self.verdict == "control_missing" and self.control is not None:
            raise SymbolReadError("'control_missing' cannot carry a counted control; "
                                  "record unmatched arms under reference_controls")
        setter(self, "evidence", tuple(check_bounded(list(self.evidence), "evidence")))
        setter(self, "metadata", check_bounded(dict(self.metadata), "result metadata"))
        self.to_dict()

    def _check_supported(self):
        if self.control is None:
            raise SymbolReadError("'supported' needs a control; use 'control_missing'")
        if self.control["judgement"] != "clean":
            raise SymbolReadError("'supported' needs a control judged clean")
        if self.basis not in ("score", "picture", "picture_only"):
            raise SymbolReadError(f"'supported' cannot rest on basis '{self.basis}'")
        if self.basis == "picture_only":
            return
        if self.effect is None or self.floor is None or self.floor_incomplete is None:
            raise SymbolReadError("'supported' needs a numeric effect, a numeric floor and floor_incomplete "
                                  "(or basis 'picture_only' with picture calls)")
        if self.intervention == "cut" and not self.effect > self.floor:
            raise SymbolReadError("'supported' cut effect must exceed its floor")
        if self.intervention == "keep":
            control_effect = self.control.get("effect")
            if control_effect is None or not control_effect - self.effect > self.floor:
                raise SymbolReadError("'supported' keep must separate from its control by more than the floor")

    def verify_plan(self, path):
        """Check the plan file at ``path`` hashes to ``plan_sha256``."""
        if file_sha256(path) != self.plan_sha256:
            raise SymbolReadError(f"plan file {path} does not hash to the recorded plan sha256")
        return True

    def to_dict(self):
        return check_bounded({
            "intervention": self.intervention, "cut_type": self.cut_type, "verdict": self.verdict,
            "plan_sha256": self.plan_sha256, "run_id": self.run_id, "scope": self.scope, "basis": self.basis,
            "effect": self.effect, "effect_units": self.effect_units,
            "effect_per_seed": None if self.effect_per_seed is None else list(self.effect_per_seed),
            "floor": self.floor, "floor_source": self.floor_source, "floor_incomplete": self.floor_incomplete,
            "control": self.control, "reference_controls": list(self.reference_controls),
            "picture": self.picture, "claim": self.claim, "claim_text": self.claim_text,
            "claim_scope": self.claim_scope, "ruled_out": list(self.ruled_out),
            "remaining_alternatives": list(self.remaining_alternatives), "contrast_id": self.contrast_id,
            "rule_id": self.rule_id, "rule_sha256": self.rule_sha256, "plan": self.plan,
            "reason": self.reason, "evidence": list(self.evidence), "metadata": dict(self.metadata)},
            "causal result")

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ("evidence", "reference_controls", "ruled_out"):
            value[key] = tuple(value.get(key) or ())
        for key in ("effect_per_seed", "remaining_alternatives"):
            if value.get(key) is not None:
                value[key] = tuple(value[key])
        return cls(**value)


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

    def require_scope(self, scope, owner):
        for r in self.results:
            if r.scope != scope:
                raise SymbolReadError(f"a {owner} holds only '{scope}'-scope results (got '{r.scope}')")

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
    """One observed read with its canonical anchors and causal status (default ``unknown``).

    ``model``: optional ``model_identity`` record (name + identity_status).
    """
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
    model: Mapping[str, Any] | None = None

    def __post_init__(self):
        if isinstance(self.mass, bool) or not isinstance(self.mass, numbers.Real) or not math.isfinite(self.mass):
            raise SymbolReadError("observed mass must be a finite number")
        object.__setattr__(self, "mass", float(self.mass))
        if not self.mass_kind:
            raise SymbolReadError("observed mass says what it measures ('mass_kind')")
        if self.surface not in ("project-session", "passive-observation"):
            raise SymbolReadError("surface must be 'project-session' or 'passive-observation'")
        _sha(self.binding_fingerprint, "binding fingerprint")
        self.causal.require_scope("read", "symbol read")
        object.__setattr__(self, "sets", tuple(self.sets))
        object.__setattr__(self, "evidence", tuple(check_bounded(list(self.evidence), "read evidence")))
        object.__setattr__(self, "metadata", check_bounded(dict(self.metadata), "read metadata"))
        object.__setattr__(self, "model", _identity_record(self.model))
        self.to_dict()

    def _body(self):
        return {"schema": READ_SCHEMA, "source": self.source.to_dict(), "reader": self.reader.to_dict(),
                "mass": self.mass, "mass_kind": self.mass_kind, "surface": self.surface,
                "backend": self.backend, "binding_fingerprint": self.binding_fingerprint,
                "use_id": self.use_id, "use_fingerprint": self.use_fingerprint,
                "lifecycle_fingerprint": self.lifecycle_fingerprint, "sets": list(self.sets),
                "causal": self.causal.to_dict(), "evidence": list(self.evidence),
                "metadata": dict(self.metadata), "model": self.model}

    def to_dict(self):
        return _sealed(self._body(), "symbol read")

    @property
    def fingerprint(self):
        return self.to_dict()["fingerprint"]

    def with_causal(self, result):
        """Attach a per-read intervention result (one read intervened alone). Refuses set results."""
        if result.scope != "read":
            raise SymbolReadError("a set-level result cannot be attached to a member read; "
                                  "attach it to the ReadSet")
        return replace(self, causal=self.causal.with_result(result))

    @classmethod
    def from_dict(cls, value):
        if value.get("schema") != READ_SCHEMA:
            raise SymbolReadError("unsupported read schema (expected " + READ_SCHEMA + ")")
        record = cls(ReadSource.from_dict(value["source"]), Reader.from_dict(value["reader"]),
                     value["mass"], value["mass_kind"], value["surface"], value["backend"],
                     value["binding_fingerprint"], value.get("use_id"), value.get("use_fingerprint"),
                     value.get("lifecycle_fingerprint"), tuple(value.get("sets") or ()),
                     CausalStatus.from_dict(value["causal"]), tuple(value.get("evidence") or ()),
                     value.get("metadata") or {}, value.get("model"))
        if record.to_dict() != value:
            raise SymbolReadError("symbol read seal mismatch")
        return record


@dataclass(frozen=True)
class ReadSet:
    """A set of readers of one source, with set-level causal status.

    Membership is held by pointer (``members``: file pointer + count + digest); ``member_reads``
    optionally lists the fingerprints of member reads that exist as records. ``selection.kind``
    is one of ``SET_SELECTION_KINDS``. A set result (scope ``set``) says the SET was cut or kept.
    It does not say any member is individually necessary; ``with_result`` refuses read-scope
    results and ``SymbolRead.with_causal`` refuses set-scope results.
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
        selection = check_bounded(dict(self.selection), "set selection")
        if selection.get("kind") not in SET_SELECTION_KINDS:
            raise SymbolReadError("set selection kind must be one of " + ", ".join(SET_SELECTION_KINDS)
                                  + " ('nominated' names a source role, not a set selection: use 'ranked')")
        object.__setattr__(self, "selection", selection)
        self.causal.require_scope("set", "read set")
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
        if result.scope != "set":
            raise SymbolReadError("a read-scope result belongs on its SymbolRead, not on a ReadSet")
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
            raise SymbolReadError("unsupported read-set schema (expected " + READ_SET_SCHEMA + ")")
        record = cls(value["name"], ReadSource.from_dict(value["source"]), value["members"],
                     value["selection"], CausalStatus.from_dict(value["causal"]),
                     tuple(value.get("member_reads") or ()), value.get("metadata") or {})
        if record.to_dict() != value:
            raise SymbolReadError("read set seal mismatch")
        return record


# --------------------------------------------------------------------------- common constructor

def symbol_read(source, reader, mass, *, mass_kind, surface, backend, binding_fingerprint,
                use=None, lifecycle=None, sets=(), evidence=(), metadata=None, model=None):
    """The single constructor both surfaces use. Causal status always starts ``unknown``."""
    return SymbolRead(source, reader, mass, mass_kind, surface, backend, binding_fingerprint,
                      use_id=None if use is None else use.use_id,
                      use_fingerprint=None if use is None else use.fingerprint,
                      lifecycle_fingerprint=None if lifecycle is None else lifecycle.fingerprint,
                      sets=tuple(sets), evidence=tuple(evidence), metadata=metadata or {}, model=model)


def read_for_use(use, lifecycle, *, source, reader, mass, mass_kind, surface, backend,
                 sets=(), evidence=(), metadata=None, model=None):
    if use.binding_fingerprint != lifecycle.binding_fingerprint:
        raise SymbolReadError("use belongs to another binding")
    if use.use_id not in {u.use_id for u in lifecycle.uses}:
        raise SymbolReadError("use is not part of this lifecycle")
    return symbol_read(source, reader, mass, mass_kind=mass_kind, surface=surface, backend=backend,
                       binding_fingerprint=lifecycle.binding_fingerprint, use=use, lifecycle=lifecycle,
                       sets=sets, evidence=evidence or tuple(use.evidence_refs), metadata=metadata,
                       model=model)


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
    the capture (``passive-capture:sha256:<hex>``), never a fabricated cut id. ``model_identity``
    is the name the canonical binding carries; whether it is a verified identity is recorded by
    the caller with ``model_identity(...)`` (records built here do not change).
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
    check_bounded(list(evidence_refs), "definition evidence")
    return SymbolOccurrence.for_binding(binding, occurrence_id=occurrence_id, session_id=surface.session_id,
        source_cut=surface.cut_ref, locations=binding.locations, branch=surface.branch,
        execution=binding.execution, clock=clock, evidence_refs=tuple(evidence_refs), status=status,
        metadata=metadata)


def passive_use(surface, binding, *, use_id, consumer, signed_observations, evidence_refs,
                result_cut=None, receipt=None, locations=None, metadata=None, status="observed"):
    from .symbol_lifecycle import SymbolUse
    check_bounded({"signed_observations": list(signed_observations), "metadata": dict(metadata or {}),
                   "evidence_refs": list(evidence_refs)}, "passive use")
    return SymbolUse(use_id, binding.to_dict()["fingerprint"], surface.session_id, consumer,
                     source_cut=surface.cut_ref, parent_cut=surface.cut_ref, result_cut=result_cut,
                     branch=surface.branch, execution=binding.execution, receipt=receipt,
                     locations=binding.locations if locations is None else tuple(locations),
                     evidence_refs=tuple(evidence_refs), status=status,
                     signed_observations=tuple(signed_observations), metadata=metadata or {})


def passive_lifecycle(binding, *, definitions, uses, claim_boundary, metadata=None):
    from .symbol_lifecycle import SymbolLifecycle
    check_bounded(dict(metadata or {}), "lifecycle metadata")
    return SymbolLifecycle.define(binding, definitions=tuple(definitions), uses=tuple(uses),
        mechanics_status="valid", instrument_status="observations-only", trend_status="not-assessed",
        terminal_status="not-assessed", claim_boundary=claim_boundary, metadata=metadata or {})


# --------------------------------------------------------------------------- persistence

def _check_rules(results, meta):
    rule = meta.get("verdict_rule")
    if rule is None:
        return
    if not isinstance(rule, Mapping) or seal_digest(rule.get("rule")) != rule.get("sha256"):
        raise SymbolReadError("meta.verdict_rule sha256 does not match its rule")
    for r in results:
        if (r["rule_id"], r["rule_sha256"]) != (rule["id"], rule["sha256"]):
            raise SymbolReadError("a result names a verdict rule other than meta.verdict_rule")


def save_reads(path, reads=(), sets=(), *, meta=None):
    """Write reads and read sets to a new JSON file (refuses to overwrite).

    ``meta`` is bounded like a record (``MAX_META_BYTES``). If ``meta`` declares a
    ``verdict_rule`` (from ``verdict_rule``), every result must name that rule and hash.
    """
    path = Path(path)
    if path.exists():
        raise SymbolReadError("reads file already exists: " + str(path))
    body = {"schema": READS_DOCUMENT_SCHEMA, "reads": [r.to_dict() for r in reads],
            "sets": [s.to_dict() for s in sets],
            "meta": check_bounded(dict(meta or {}), "meta", max_bytes=MAX_META_BYTES)}
    known = {r["fingerprint"] for r in body["reads"]}
    set_ids = {s["set_id"] for s in body["sets"]}
    for s in body["sets"]:
        if not set(s["member_reads"]) <= known:
            raise SymbolReadError("read set names a member read that is not in this file")
    for r in body["reads"]:
        if not set(r["sets"]) <= set_ids:
            raise SymbolReadError("read names a set that is not in this file")
    _check_rules([x for rec in (*body["reads"], *body["sets"]) for x in rec["causal"]["results"]], body["meta"])
    document = {**body, "fingerprint": seal_digest(body)}
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": file_sha256(path), "reads": len(reads), "sets": len(sets)}


def load_reads(path):
    document = json.loads(Path(path).read_text())
    if document.get("schema") != READS_DOCUMENT_SCHEMA:
        raise SymbolReadError("unsupported reads document (expected " + READS_DOCUMENT_SCHEMA + ")")
    body = {k: v for k, v in document.items() if k != "fingerprint"}
    if seal_digest(body) != document.get("fingerprint"):
        raise SymbolReadError("reads document seal mismatch")
    meta = check_bounded(document["meta"], "meta", max_bytes=MAX_META_BYTES)
    _check_rules([x for rec in (*document["reads"], *document["sets"]) for x in rec["causal"]["results"]], meta)
    return {"reads": tuple(SymbolRead.from_dict(r) for r in document["reads"]),
            "sets": tuple(ReadSet.from_dict(s) for s in document["sets"]),
            "meta": meta, "fingerprint": document["fingerprint"]}

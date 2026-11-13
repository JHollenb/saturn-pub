"""One sealed record envelope, generalised from ``circuit_investigation.save/record``.

::

    {schema: "mdb-record-v2", id, session_id, kind, model_identity, execution, authority,
     pointers: [...], body, digest}

``digest`` is ``seal_digest`` of the envelope without ``id`` and ``digest``; ``id`` equals
``digest`` (records are content-addressed, as circuit records were). The whole record, with
``id`` and ``digest``, is bounded (no arrays or tensors, at most 64 inline items per sequence)
and its *stored form* (``stored_form``: the canonical UTF-8 JSON that the digest covers and
that ``put_record`` writes and a reader transmits) is at most 64 KiB. One definition of that
form and its size serves sealing, verifying and indexing.

Authority rule, enforced by ``seal_record`` and ``verify_record`` for every kind: inside
``body`` every numeric leaf (int or float, never bool) must be the ``value`` of a labelled
number ``num(value, authority, ...)`` (exact key set, real finite value, one of the five
authorities), unless it sits at a path the kind declares structural (``plain_numeric``: shapes, indices,
counts, ids, byte sizes). A declared path exempts only a *leaf* that is a non-negative ``int``
(never a bool, at most 15 decimal digits, so it is exact in JSON/JS) or a list of such ints; it
covers a whole subtree only when it ends in an explicit ``**`` marker, and even then only
non-negative ints are exempt. Lists, tuples and mappings are walked fully. The
kind's ``numeric`` paths must exist and hold labelled numbers; a labelled-number mapping with
extra keys is an error; NaN and +-Inf are refused everywhere. Sealing or verifying a record of
an undeclared kind is refused. Pointers must be of a known kind (file, parameter, run,
occurrence, tensor) and their free fields are typed: ``selector`` is a list whose entries are ``None``
(the whole axis), a non-negative int or a list of non-negative ints; ``array`` is an array name
(text) or ``{shape: [int, ...], dtype: str}``; ``role`` and ``json_pointer`` are text; a run
pointer's ``plan`` is a file pointer; a parameter pointer's ``model`` is a canonical model
identity. ``execution`` and ``model_identity`` are scanned too: ints (>= 0) anywhere, floats
only at the adapters' configuration constants (``EXECUTION_FLOATS``), nothing else numeric.
Integers in records are capped at 15 decimal digits; a larger one (or a ``ValueError`` from a
huge int) is a ``RecordError``. A path component ``*`` matches every key of a mapping or index
of a list. Out of scope by design: a string that looks like a number is text, and a boolean is
not a numeric claim; neither is refused.

``mdb-circuit-investigation-v1`` records keep their own format and readers
(``circuit_investigation``); ``verify_legacy`` checks one the way ``circuit_investigation.record``
does, without the session binding.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass

from .pointers import (
    AUTHORITIES,
    MAX_RECORD_BYTES,
    RecordError,
    _canonical,
    _identity_record,
    _sha,
    check_bounded,
    seal_digest,
    validate_pointer,
)

SCHEMA = "mdb-record-v2"
LEGACY_SCHEMAS = ("mdb-circuit-investigation-v1",)
FIELDS = ("schema", "id", "session_id", "kind", "model_identity", "execution", "authority",
          "pointers", "body", "digest")
NUM_KEYS = ("value", "authority", "proof", "unit", "pointer")
POINTER_KINDS = ("file", "parameter", "run", "occurrence", "tensor")
_POINTER_FIELDS = {
    "file": {"kind", "locator", "digest", "array", "selector", "json_pointer", "role"},
    "parameter": {"kind", "parameter", "model", "selector"},
    "run": {"kind", "run_id", "plan_sha256", "plan"},
    "tensor": {"kind", "ref", "sha256", "shape", "dtype"},
}
_SHORT = 64


def stored_form(record):
    """The text a record is stored and transmitted as: its canonical JSON (the digest's form)."""
    return _canonical(record)


def stored_bytes(record):
    """UTF-8 bytes of ``stored_form``; text that is not encodable is a ``RecordError``."""
    try:
        return stored_form(record).encode("utf-8")
    except UnicodeEncodeError as error:
        raise RecordError("record holds text that cannot be encoded as UTF-8") from error


def _huge_int_guard(function):
    """A ``ValueError``/``OverflowError`` from a huge int (``10**5000``) is a ``RecordError``."""
    import functools

    @functools.wraps(function)
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except RecordError:
            raise
        except (ValueError, OverflowError) as error:
            raise RecordError(f"record holds a number that cannot be sealed: {error}") from error
    return guarded


def _bounded_record(value, label):
    """``check_bounded`` with the size measured on ``stored_bytes`` (one definition)."""
    try:
        out = check_bounded(value, label, max_bytes=None)
    except UnicodeEncodeError as error:
        raise RecordError(f"{label} holds text that cannot be encoded as UTF-8") from error
    size = len(stored_bytes(out))
    if size > MAX_RECORD_BYTES:
        raise RecordError(f"{label} is {size} bytes (> {MAX_RECORD_BYTES}); store it by pointer")
    return out


INT_DIGITS = 15
_INT_CAP = 10 ** INT_DIGITS


def _count(value):
    """A non-negative int below 10**15 (bool is not an int here)."""
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < _INT_CAP


def _selector(value, label):
    """``None`` (whole axis), a non-negative int or a list of them, per entry, in a list."""
    if not isinstance(value, list):
        raise RecordError(f"{label} selector must be a list")
    for item in value:
        if not (item is None or _count(item)
                or (isinstance(item, list) and all(_count(i) for i in item))):
            raise RecordError(f"{label} selector entries are None, a non-negative integer or a "
                              "list of non-negative integers")
    return value


def _array_field(value):
    if isinstance(value, str):
        return
    if (not isinstance(value, Mapping) or set(value) != {"shape", "dtype"}
            or not isinstance(value["shape"], list) or not all(_count(n) for n in value["shape"])
            or not isinstance(value["dtype"], str)):
        raise RecordError("file pointer array is its name (text) or {shape: [int, ...], dtype: text}")


def _typed_pointer(pointer):
    """Free fields of the three stored pointer kinds are text, structure or counts, never claims."""
    kind = pointer["kind"]
    if kind == "file":
        locator, digest = pointer["locator"], pointer["digest"]
        if not isinstance(locator["root"], str) or not isinstance(locator["path"], str):
            raise RecordError("file pointer locator root and path are text")
        if not _count(digest["size"]) or not isinstance(digest["sha256"], str):
            raise RecordError("file pointer digest is {sha256: text, size: non-negative integer}")
        if "array" in pointer:
            _array_field(pointer["array"])
        if "selector" in pointer:
            _selector(pointer["selector"], "file pointer")
        for field in ("json_pointer", "role"):
            if field in pointer and not isinstance(pointer[field], str):
                raise RecordError(f"file pointer {field} is text")
    elif kind == "parameter":
        if "selector" in pointer:
            _selector(pointer["selector"], "parameter pointer")
        if "model" in pointer and _canonical(_identity_record(pointer["model"])) != _canonical(
                pointer["model"]):
            raise RecordError("parameter pointer model is not a canonical model identity")
    elif kind == "run":
        if "plan_sha256" in pointer and (not isinstance(pointer["plan_sha256"], str)
                                         or _sha(pointer["plan_sha256"], "plan sha256")
                                         != pointer["plan_sha256"]):
            raise RecordError("run pointer plan_sha256 is a sha256 hex digest")
        if "plan" in pointer:
            plan = pointer["plan"]
            if not isinstance(plan, Mapping) or plan.get("kind") != "file":
                raise RecordError("run pointer plan is a file pointer")
            if _canonical(_record_pointer(plan)) != _canonical(plan):
                raise RecordError("run pointer plan is not in canonical form")


def _typed_occurrence(pointer):
    """An occurrence pointer's integers obey the 15-digit cap too (value shape, event)."""
    shape = pointer["value"]["shape"]
    if not isinstance(shape, list) or not all(_count(n) for n in shape):
        raise RecordError("occurrence value shape holds integers in [0, 10**15)")
    event = pointer["event"]
    if event is not None and not _count(event):
        raise RecordError("occurrence event is an integer in [0, 10**15)")


def _record_pointer(pointer):
    try:
        pointer = validate_pointer(pointer)
        kind = pointer["kind"]
        if kind not in POINTER_KINDS:
            raise RecordError(f"pointer kind {kind!r} is not one of " + ", ".join(POINTER_KINDS))
        extra = set(pointer) - _POINTER_FIELDS.get(kind, set(pointer))
        if extra:
            raise RecordError(f"{kind} pointer has unknown fields {sorted(extra)}")
        if kind == "occurrence":
            _typed_occurrence(pointer)
        elif kind != "tensor":    # validate_tensor_pointer already typed every field
            _typed_pointer(pointer)
    except (TypeError, KeyError, AttributeError) as error:
        raise RecordError(f"malformed pointer: {error}") from error
    except RecordError:
        raise
    except (ValueError, OverflowError) as error:
        raise RecordError(f"malformed pointer: {error}") from error
    return pointer


def _short_text(value, label):
    if not isinstance(value, str) or not value or len(value) > _SHORT or value != value.strip():
        raise RecordError(f"{label} must be short text (<= {_SHORT} chars, no surrounding space)")
    return value


def _authority(value, label="authority"):
    if value not in AUTHORITIES:
        raise RecordError(f"{label} must be one of " + ", ".join(AUTHORITIES))
    return value


@_huge_int_guard
def num(value, authority, *, proof=None, unit=None, pointer=None):
    """A numeric claim: ``{value, authority, proof?, unit?, pointer?}``.

    ``proof`` names the identity or theorem (``P14``), ``pointer`` the evidence the number was
    read from or derived on. Booleans and non-finite numbers are refused.
    """
    np = __import__("sys").modules.get("numpy")
    if np is not None and isinstance(value, np.generic):
        value = value.item()
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or (not isinstance(value, numbers.Integral) and not math.isfinite(value))):
        raise RecordError("a numeric claim needs a finite number")
    if isinstance(value, numbers.Integral) and not -_INT_CAP < int(value) < _INT_CAP:
        raise RecordError(f"a numeric claim holds at most {INT_DIGITS} decimal digits "
                          "(exact in JSON and JS); store larger integers as text")
    out = {"value": int(value) if isinstance(value, numbers.Integral) else float(value),
           "authority": _authority(authority)}
    if proof is not None:
        out["proof"] = _short_text(proof, "proof")
    if unit is not None:
        out["unit"] = _short_text(unit, "unit")
    if pointer is not None:
        out["pointer"] = _record_pointer(pointer)
    return out


def is_num(value):
    """A num-shaped mapping: only num keys, a ``value``, and an authority or a numeric value."""
    if not isinstance(value, Mapping) or "value" not in value or not set(value) <= set(NUM_KEYS):
        return False
    number = value["value"]
    return "authority" in value or (isinstance(number, numbers.Real)
                                    and not isinstance(number, bool))


@_huge_int_guard
def check_num(value, label="claim"):
    """Validate a stored numeric claim; returns it in canonical form."""
    if not isinstance(value, Mapping) or "value" not in value or not set(value) <= set(NUM_KEYS):
        raise RecordError(f"{label} must be a num(value, authority) claim, not "
                          f"{type(value).__name__}")
    if "authority" not in value:
        raise RecordError(f"{label} has no authority")
    rebuilt = num(value["value"], value["authority"], proof=value.get("proof"),
                  unit=value.get("unit"), pointer=value.get("pointer"))
    if _canonical(rebuilt) != _canonical(dict(value)):
        raise RecordError(f"{label} is not a canonical numeric claim")
    return rebuilt


# --------------------------------------------------------------------------- kinds

@dataclass(frozen=True)
class KindSpec:
    """What a record kind must hold.

    ``numeric``: dotted body paths whose values are ``num`` claims; ``*`` stands for every
    item of a list or every value of a mapping. Every declared path must exist and hold labelled
    numbers. ``required``: dotted body paths that must be present. ``pointers``: pointer kinds
    the record must carry at least one of each. ``plain_numeric``: dotted body paths (``*``
    allowed) of structural leaves (shapes, indices, counts, ids, byte sizes): a path exempts a
    leaf holding a non-negative int, or a list of such ints. It covers a whole subtree only
    with a trailing ``**`` marker (``sizes.**``), and even then only non-negative ints are
    exempt; a float, a negative number or an int past 15 digits is refused everywhere.
    """
    name: str
    numeric: tuple = ()
    required: tuple = ()
    pointers: tuple = ()
    plain_numeric: tuple = ()

    def __post_init__(self):
        for attr in ("numeric", "required", "pointers", "plain_numeric"):
            object.__setattr__(self, attr, tuple(getattr(self, attr)))
        _short_text(self.name, "record kind")
        for path in self.plain_numeric:
            parts = path.split(".")
            if "**" in parts[:-1] or any(not p for p in parts):
                raise RecordError(f"plain_numeric path {path!r}: '**' only as the last component")


_KINDS = {}


def declare_kind(name, *, numeric=(), required=(), pointers=(), plain_numeric=()):
    """Declare a record kind. Re-declaring the same spec is a no-op; a different one fails."""
    spec = KindSpec(name, numeric, required, pointers, plain_numeric)
    if name in _KINDS and _KINDS[name] != spec:
        raise RecordError(f"record kind {name!r} is already declared differently")
    _KINDS[name] = spec
    return spec


def kind_spec(name):
    return _KINDS.get(name)


def _claims(value, parts, label, path):
    """Yield (label, value) for every value at the dotted claim path; absent branches raise."""
    if not parts:
        yield label, value
        return
    head, rest = parts[0], parts[1:]
    if head == "*":
        if isinstance(value, Mapping):
            for key, item in value.items():
                yield from _claims(item, rest, f"{label}.{key}", path)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                yield from _claims(item, rest, f"{label}[{i}]", path)
        else:
            raise RecordError(f"{label} must be a mapping or list holding the claims at "
                              f"body.{path}, not {type(value).__name__}")
        return
    if not isinstance(value, Mapping) or head not in value:
        raise RecordError(f"{label}.{head} is missing (the kind declares a claim at body.{path})")
    yield from _claims(value[head], rest, f"{label}.{head}", path)


def _present(value, parts):
    if not parts:
        return True
    if isinstance(value, Mapping) and parts[0] in value:
        return _present(value[parts[0]], parts[1:])
    return False


def _is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


class _Rule:
    """Which numeric leaves a scan exempts: ``patterns`` (body) or the execution profile."""

    def __init__(self, patterns=(), *, any_count=False, floats=()):
        self.patterns, self.any_count, self.floats = patterns, any_count, tuple(floats)

    def covers(self, parts):
        """The path lies under a ``**`` subtree (or anywhere, for a profile)."""
        if self.any_count:
            return True
        return any(p[-1] == "**" and len(parts) >= len(p) - 1
                   and all(a == "*" or a == b for a, b in zip(p[:-1], parts)) for p in self.patterns)

    def leaf(self, parts):
        """The path is a declared structural leaf."""
        return self.covers(parts) or any(
            p[-1] != "**" and len(p) == len(parts) and all(a == "*" or a == b
                                                          for a, b in zip(p, parts))
            for p in self.patterns)

    def exempt(self, parts, number):
        if isinstance(number, float):
            return tuple(parts) in self.floats
        return _count(number) and self.leaf(parts)


def _scan_claims(value, label, parts=(), rule=None, depth=0):
    """Every numeric leaf is a labelled number's value or an exempt structural leaf."""
    rule = rule or _Rule()
    if depth > 32:
        raise RecordError(f"{label} nests too deeply")
    if isinstance(value, Mapping):
        if "value" in value and "authority" in value:
            check_num(value, label)  # exact key set, real value, known authority
            return
        if ("value" in value and set(value) <= set(NUM_KEYS)
                and (_is_number(value["value"]) or isinstance(value["value"], str))
                and not rule.covers(parts)):
            raise RecordError(f"{label} has no authority")   # a claim that dropped its label
        for key, item in value.items():
            _scan_claims(item, f"{label}.{key}", (*parts, key), rule, depth + 1)
    elif isinstance(value, (list, tuple)):
        if (value and all(_count(i) for i in value) and rule.leaf(parts)):
            return                                            # a declared list of counts
        for i, item in enumerate(value):
            _scan_claims(item, f"{label}[{i}]", (*parts, str(i)), rule, depth + 1)
    elif _is_number(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise RecordError(f"{label} is not finite")
        if isinstance(value, int) and not -_INT_CAP < value < _INT_CAP:
            raise RecordError(f"{label} holds more than {INT_DIGITS} decimal digits")
        if not rule.exempt(parts, value):
            raise RecordError(f"{label} is a bare number; a claim is num(value, authority) "
                              "and only declared structural leaves hold plain non-negative "
                              "integers")


# Floats an adapter's execution profile may carry: configuration constants, not measurements.
EXECUTION_FLOATS = (("batch", "guidance"), ("guidance",), ("eta",),
                    ("logit_softcap", "final"), ("logit_softcap", "attention"))
_PROFILE = _Rule(any_count=True, floats=EXECUTION_FLOATS)
_IDENTITY = _Rule(any_count=True)


@_huge_int_guard
def validate_body(kind, body, pointers=()):
    """Check ``body`` against the declared kind and the authority rule; raises ``RecordError``."""
    spec = _KINDS.get(kind)
    if spec is None:
        raise RecordError(f"record kind {kind!r} is not declared (declare_kind)")
    if not isinstance(body, Mapping):
        raise RecordError("record body must be a mapping")
    for path in spec.required:
        if not _present(body, path.split(".")):
            raise RecordError(f"record kind {kind!r} requires body.{path}")
    for path in spec.numeric:
        for label, value in _claims(body, path.split("."), "body", path):
            check_num(value, label)
    patterns = tuple(tuple(p.split(".")) for p in spec.plain_numeric)
    _scan_claims(body, "body", (), _Rule(patterns))
    have = {p.get("kind") for p in pointers}
    missing = [k for k in spec.pointers if k not in have]
    if missing:
        raise RecordError(f"record kind {kind!r} needs pointers of kind {missing}")


# --------------------------------------------------------------------------- seal / verify

def _check_envelope(record):
    """``model_identity`` and ``execution`` carry no measurements (see the module docstring)."""
    _scan_claims(record["model_identity"], "record.model_identity", (), _IDENTITY)
    _scan_claims(record["execution"], "record.execution", (), _PROFILE)


@_huge_int_guard
def seal_record(*, session_id, kind, model_identity, execution, authority, body, pointers=()):
    """Build and seal one record. ``model_identity``/``execution`` are the session's values."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise RecordError("a record needs a session id")
    pointers = [_record_pointer(p) for p in pointers]
    envelope = _bounded_record({
        "schema": SCHEMA, "session_id": session_id, "kind": _short_text(kind, "record kind"),
        "model_identity": model_identity, "execution": execution,
        "authority": _authority(authority), "pointers": pointers, "body": dict(body),
    }, "record")
    _check_envelope(envelope)
    validate_body(kind, envelope["body"], envelope["pointers"])
    digest = seal_digest(envelope)
    return _bounded_record({**envelope, "id": digest, "digest": digest}, "record")


@_huge_int_guard
def verify_record(record, *, require_kind=True):
    """Check a stored v2 record: fields, seal, bounds, authority, pointers, kind and body.

    The kind must be declared in this process (``require_kind`` is kept for callers and is
    always in force): a record of an undeclared kind is refused, sealed or not.
    """
    if not isinstance(record, Mapping) or set(record) != set(FIELDS):
        raise RecordError("record fields are " + ", ".join(FIELDS))
    if record["schema"] != SCHEMA:
        raise RecordError(f"record schema must be {SCHEMA!r}")
    record = _bounded_record(dict(record), "record")
    envelope = {k: v for k, v in record.items() if k not in ("id", "digest")}
    digest = seal_digest(envelope)
    if record["digest"] != digest or record["id"] != digest:
        raise RecordError("record seal differs from its content")
    _authority(record["authority"])
    if not isinstance(record["pointers"], list):
        raise RecordError("record pointers are a list")
    pointers = [_record_pointer(p) for p in record["pointers"]]
    if _canonical(pointers) != _canonical(record["pointers"]):
        raise RecordError("record pointers are not in canonical form")
    _check_envelope(record)
    validate_body(record["kind"], record["body"], pointers)
    return record


def verify_legacy(body, identifier):
    """Check an ``mdb-circuit-investigation-v1`` body against its id (saturn_pub digest)."""
    from saturn_pub.values import digest

    if not isinstance(body, Mapping) or body.get("schema") not in LEGACY_SCHEMAS:
        raise RecordError("not a legacy circuit-investigation record")
    if digest(body) != identifier:
        raise RecordError("legacy record seal differs from its id")
    return {"record_id": identifier, **body}


# --------------------------------------------------------------------------- built-in kinds

# An occurrence record (chapter 10 of the model-math series). Checkpoint -> model_identity;
# numerical profile -> execution; clock, port, addresses, parent cut and ordered edits ->
# the occurrence pointer's address and cut; hydration recipe -> the pointer's recipe/event.
# The body holds the rest: input/history, transform window and convention, retained terms
# with an explicit unlisted-term sum, the consumer address and the source evidence.
OCCURRENCE = declare_kind(
    "occurrence",
    numeric=("retained.*.value", "unlisted"),
    required=("history", "transform", "retained", "unlisted", "consumer", "evidence"),
    pointers=("occurrence",),
    # Structure, not claims, declared leaf by leaf: the token ids and the parent row of the
    # history, the window and head of the transform, source rows and ids of retained terms,
    # byte sizes and event indices of evidence. Anything else numeric there is a labelled number.
    plain_numeric=("history.token_ids", "history.parent_row", "transform.window",
                   "transform.head", "retained.*.src", "retained.*.row", "retained.*.token",
                   "retained.*.layer", "retained.*.head", "evidence.*.size",
                   "evidence.*.event", "evidence.*.index"),
)

# the analysis layer's kinds (torch-free), so any reader can verify them
from . import kinds as _kinds  # noqa: E402, F401

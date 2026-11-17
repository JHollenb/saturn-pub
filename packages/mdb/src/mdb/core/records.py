"""One sealed record envelope, generalised from ``circuit_investigation.save/record``.

A record, as every caller sees it (the *logical* record; the same fields in v2 and v3)::

    {schema: "mdb-record-v3", id, session_id, kind, model_identity, execution, authority,
     pointers: [...], body, digest}

Schema v3 (written by ``seal_record`` from mdb 0.2.2). Its *stored form* (``stored_form``: the
canonical UTF-8 JSON that the digest covers, that ``put_record`` writes and a sink appends) is
the logical record with two changes:

* ``execution`` is replaced by ``execution_sha256`` (``seal_digest(execution)``). The execution
  profile is the session's, so it is kept once: in the project's session row, or as one
  ``{"mdb_execution": ...}`` line per sink file. A reader passes it back (``load_record``).
* the ``values`` of a column claim (``nums``) are the base64 of their little-endian bytes.

``digest`` = ``id`` = ``seal_digest`` of the stored form without ``id`` and ``digest``, so the
id still binds every field, the execution profile included (through its sha256). The stored
form is at most ``MAX_RECORD_BYTES_V3`` (1 MiB); the structure outside column values is bounded
as in v2 (no arrays or tensors, at most 64 inline items per sequence, at most
``MAX_RECORD_ITEMS_V3`` values). A column claim holds up to ``MAX_NUMS_INLINE`` values inline;
more go to a tensor blob by pointer (``nums(..., store=)``).

Schema v2 (``mdb-record-v2``, mdb <= 0.2.1) is read and verified unchanged: its stored form is
the canonical JSON of the logical record, at most 64 KiB (``MAX_RECORD_BYTES``), and it has no
column claims. ``migrate_record`` writes the v3 twin of a v2 record (same content, new id);
``mdb.core.migrate`` does it for files.

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

import array
import base64
import binascii
import json
import math
import numbers
import sys
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

SCHEMA = "mdb-record-v3"
SCHEMA_V2 = "mdb-record-v2"
SCHEMAS = (SCHEMA, SCHEMA_V2)          # every schema ``verify_record`` and the index accept
LEGACY_SCHEMAS = ("mdb-circuit-investigation-v1",)
FIELDS = ("schema", "id", "session_id", "kind", "model_identity", "execution", "authority",
          "pointers", "body", "digest")
STORED_FIELDS = ("schema", "id", "session_id", "kind", "model_identity", "execution_sha256",
                 "authority", "pointers", "body", "digest")
EXECUTION_SCHEMA = "mdb-execution-v1"   # a sink's ``{"mdb_execution": ...}`` line
MAX_RECORD_BYTES_V3 = 1024 * 1024       # stored form of a v3 record (v2: ``MAX_RECORD_BYTES``)
MAX_RECORD_ITEMS_V3 = 65536             # JSON values outside column values (v2: 4096)
MAX_NUMS_INLINE = 65536                 # values one column claim holds inline
NUMS_KIND = "nums"
NUMS_DTYPES = {"float64": "d", "float32": "f", "int64": "q"}   # dtype -> ``array`` typecode
NUMS_KEYS = ("kind", "authority", "proof", "unit", "pointer", "shape", "dtype", "values")
_NUMS_REQUIRED = {"kind", "authority", "shape", "dtype", "values"}
NUM_KEYS = ("value", "authority", "proof", "unit", "pointer")
POINTER_KINDS = ("file", "parameter", "run", "occurrence", "tensor")
_POINTER_FIELDS = {
    "file": {"kind", "locator", "digest", "array", "selector", "json_pointer", "role"},
    "parameter": {"kind", "parameter", "model", "selector"},
    "run": {"kind", "run_id", "plan_sha256", "plan"},
    "tensor": {"kind", "ref", "sha256", "shape", "dtype"},
}
_SHORT = 64


def stored_record(record):
    """The stored form of a record as a mapping: v3 -> ``execution_sha256`` and encoded column
    values; v2 (and an already stored v3 mapping) -> unchanged."""
    if (isinstance(record, Mapping) and record.get("schema") == SCHEMA
            and "execution" in record):
        return _encode(record)
    return record


def stored_form(record):
    """The text a record is stored and transmitted as: canonical JSON of ``stored_record`` (the
    digest's form)."""
    return _canonical(stored_record(record))


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
    """v2: ``check_bounded`` with the size measured on ``stored_bytes`` (one definition)."""
    try:
        out = check_bounded(value, label, max_bytes=None)
    except UnicodeEncodeError as error:
        raise RecordError(f"{label} holds text that cannot be encoded as UTF-8") from error
    size = len(stored_bytes(out))
    if size > MAX_RECORD_BYTES:
        raise RecordError(f"{label} is {size} bytes (> {MAX_RECORD_BYTES}); store it by pointer")
    return out


def _is_nums(value):
    return (isinstance(value, Mapping) and value.get("kind") == NUMS_KIND and "values" in value
            and "value" not in value)


_LEAVES = frozenset((str, int, float, bool, type(None)))


def _column_paths(value, path=(), depth=0, out=None):
    """Paths (key / index tuples) of the column claims inside ``value`` (leaves are skipped
    without a call: this walk runs on every seal, verify and read)."""
    out = [] if out is None else out
    if depth > 32:
        raise RecordError("record nests too deeply")
    kind = type(value)
    if kind is dict or (kind is not list and kind is not tuple and isinstance(value, Mapping)):
        if value.get("kind") == NUMS_KIND and "values" in value and "value" not in value:
            out.append(path)
            return out
        items = value.items()
    elif kind is list or kind is tuple:
        items = enumerate(value)
    else:
        return out
    for key, item in items:
        if type(item) not in _LEAVES:
            _column_paths(item, (*path, key), depth + 1, out)
    return out


def _at(value, path):
    for part in path:
        value = value[part]
    return value


def _bounded_v3(value, label):
    """v3 bounds on a logical envelope: column values are checked by ``check_nums`` (at most
    ``MAX_NUMS_INLINE`` each) and set aside; everything else by ``check_bounded`` with
    ``MAX_RECORD_ITEMS_V3``. Returns the envelope in canonical form and the column paths."""
    if not isinstance(value, Mapping):
        raise RecordError(f"{label} must be a mapping")
    body = value.get("body")
    paths = _column_paths(body) if isinstance(body, Mapping) else []
    columns = [check_nums(_at(body, p), "body" + "".join(f".{k}" for k in p)) for p in paths]
    for path in paths:
        body = _replace_at(body, (*path, "values"), [])
    if paths:
        value = {**value, "body": body}
    try:
        out = check_bounded(value, label, max_bytes=None, max_items=MAX_RECORD_ITEMS_V3)
    except UnicodeEncodeError as error:
        raise RecordError(f"{label} holds text that cannot be encoded as UTF-8") from error
    for path, column in zip(paths, columns):
        _at(out["body"], path)["values"] = column["values"]
    return out, paths


def _replace_at(value, path, new):
    """A copy of ``value`` with ``new`` at ``path``; only the containers on the path are copied
    (a tuple on the path becomes a list, as ``check_bounded`` would make it)."""
    if not path:
        return new
    head = path[0]
    copy = dict(value) if isinstance(value, Mapping) else list(value)
    copy[head] = _replace_at(value[head], path[1:], new)
    return copy


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


# --------------------------------------------------------------------------- column claims (v3)

def _flat(values):
    """(flat row-major list, shape, dtype kind or None) of a nested list, numpy array or torch
    tensor (``"f4"`` for float32 and narrower floats, ``"f8"``, ``"i"``)."""
    if hasattr(values, "detach"):
        values = values.detach().cpu()
        if values.is_floating_point() and values.element_size() < 4:
            values = values.float()            # bfloat16 / float16 are exact in float32
        values = values.numpy()
    if hasattr(values, "tolist") and hasattr(values, "shape") and hasattr(values, "dtype"):
        kind = values.dtype.kind
        tag = "i" if kind in "iu" else ("f4" if kind == "f" and values.dtype.itemsize <= 4
                                        else "f8" if kind == "f" else None)
        return values.reshape(-1).tolist(), [int(n) for n in values.shape], tag
    shape, level = [], values
    while isinstance(level, (list, tuple)):
        shape.append(len(level))
        level = level[0] if level else None
    flat = list(values)
    for _ in range(len(shape) - 1):
        flat = [x for row in flat for x in row]
    return flat, shape, None


def _column_value(value, dtype, label):
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise RecordError(f"{label} values are numbers (no booleans, text or None)")
    if dtype == "int64":
        if not isinstance(value, numbers.Integral) or not -_INT_CAP < int(value) < _INT_CAP:
            raise RecordError(f"{label}: an int64 column holds integers of at most {INT_DIGITS} "
                              "decimal digits")
        return int(value)
    value = float(value)
    if not math.isfinite(value):
        raise RecordError(f"{label} values must be finite")
    if dtype == "float32" and array.array("f", [value])[0] != value:
        raise RecordError(f"{label}: {value!r} is not exactly a float32 value")
    return value


@_huge_int_guard
def nums(values, authority, *, proof=None, unit=None, pointer=None, shape=None, dtype=None,
         store=None, inline=None):
    """A column claim: many numbers with one authority, proof and unit (schema v3).

    ``{kind: "nums", authority, proof?, unit?, pointer?, shape, dtype, values}``. ``values`` is
    a nested list, a numpy array or a tensor (flattened row-major; ``shape`` defaults to its
    shape). ``dtype`` is ``float64`` (default), ``float32`` (every value must be exactly a
    float32) or ``int64`` (default for integer arrays). Values are finite.

    Up to ``MAX_NUMS_INLINE`` values are held inline (the stored form keeps their bytes as
    base64). More, or ``inline=False``, need ``store`` (a project or its ``LocalStore``): the
    values go to a tensor blob and ``values`` is a tensor pointer, checked by sha256 on read
    (``nums_values``). In the record a column costs about 8 bytes per float64 value (base64:
    10.7 characters) against about 70 for one ``num`` per value.
    """
    flat, found, tag = _flat(values)
    shape = found if shape is None else [int(n) for n in shape]
    if dtype is None:
        if tag is None:
            tag = "i" if flat and all(isinstance(v, numbers.Integral)
                                      and not isinstance(v, bool) for v in flat) else "f8"
        dtype = {"i": "int64", "f4": "float32"}.get(tag, "float64")
    if dtype not in NUMS_DTYPES:
        raise RecordError("a column dtype is one of " + ", ".join(NUMS_DTYPES))
    flat = [_column_value(v, dtype, "column") for v in flat]
    if math.prod(shape) != len(flat):
        raise RecordError(f"column shape {shape} does not hold {len(flat)} values")
    out = {"kind": NUMS_KIND, "authority": _authority(authority), "shape": shape,
           "dtype": dtype}
    if proof is not None:
        out["proof"] = _short_text(proof, "proof")
    if unit is not None:
        out["unit"] = _short_text(unit, "unit")
    if pointer is not None:
        out["pointer"] = _record_pointer(pointer)
    if inline is None:
        inline = len(flat) <= MAX_NUMS_INLINE
    if inline:
        out["values"] = flat
    else:
        if store is None:
            raise RecordError(f"a column of {len(flat)} values (> {MAX_NUMS_INLINE} inline) is "
                              "kept by pointer: pass store= (a project or its LocalStore)")
        import numpy as np

        from .pointers import tensor_pointer

        store = getattr(store, "store", store)
        data = np.asarray(flat, dtype=dtype).reshape(shape)
        try:
            stored = store.put_tensor(data)
        except (ValueError, OverflowError) as error:
            raise RecordError(f"the column store refused the values: {error}") from error
        out["values"] = tensor_pointer(stored["ref"], sha256=stored["sha256"],
                                       shape=stored["shape"], dtype=stored["dtype"])
    return check_nums(out)


@_huge_int_guard
def check_nums(value, label="column"):
    """Validate a column claim; returns it in canonical form (``values`` a list of Python
    numbers, or a tensor pointer whose shape and dtype match)."""
    if not isinstance(value, Mapping) or value.get("kind") != NUMS_KIND:
        raise RecordError(f"{label} must be a nums column claim")
    keys = set(value)
    if not _NUMS_REQUIRED <= keys or not keys <= set(NUMS_KEYS):
        raise RecordError(f"{label} column fields are {', '.join(sorted(_NUMS_REQUIRED))} and "
                          "optionally proof, unit, pointer")
    dtype, shape = value["dtype"], value["shape"]
    if dtype not in NUMS_DTYPES:
        raise RecordError(f"{label} dtype must be one of " + ", ".join(NUMS_DTYPES))
    if (not isinstance(shape, (list, tuple)) or not shape
            or not all(_count(n) for n in shape)):
        raise RecordError(f"{label} shape is a non-empty list of non-negative integers")
    out = {"kind": NUMS_KIND, "authority": _authority(value["authority"], f"{label} authority"),
           "shape": list(shape), "dtype": dtype}
    for field in ("proof", "unit"):
        if field in value:
            out[field] = _short_text(value[field], f"{label} {field}")
    if "pointer" in value:
        out["pointer"] = _record_pointer(value["pointer"])
    values = value["values"]
    if isinstance(values, Mapping):
        out["values"] = _record_pointer(values)
        if (out["values"]["kind"] != "tensor" or out["values"]["shape"] != out["shape"]
                or out["values"]["dtype"] != dtype):
            raise RecordError(f"{label} values pointer is a tensor pointer of the column's "
                              "shape and dtype")
        return out
    if not isinstance(values, (list, tuple)):
        raise RecordError(f"{label} values are a list or a tensor pointer")
    if len(values) > MAX_NUMS_INLINE:
        raise RecordError(f"{label} holds {len(values)} values inline (> {MAX_NUMS_INLINE}); "
                          "keep it by pointer (nums(..., store=))")
    if math.prod(out["shape"]) != len(values):
        raise RecordError(f"{label} shape {out['shape']} does not hold {len(values)} values")
    if dtype != "int64" and any(isinstance(v, numbers.Integral) and not isinstance(v, bool)
                                for v in values):
        # one logical form per content: a float column holds floats (1.0, never 1)
        raise RecordError(f"{label} is a float column holding integers; give floats")
    out["values"] = [_column_value(v, dtype, label) for v in values]
    return out


def nums_values(claim, *, store=None):
    """The values of a column claim as a numpy array of its shape and dtype (inline, or read
    from ``store`` and checked against the pointer's sha256)."""
    import numpy as np

    claim = check_nums(claim)
    values = claim["values"]
    if isinstance(values, Mapping):
        from .pointers import resolve_tensor

        return resolve_tensor(values, store=getattr(store, "store", store), framework="numpy")
    return np.asarray(values, dtype=claim["dtype"]).reshape(claim["shape"])


def _pack(values, dtype):
    data = array.array(NUMS_DTYPES[dtype], values)
    if sys.byteorder != "little":
        data.byteswap()
    return base64.b64encode(data.tobytes()).decode("ascii")


def _unpack(text, dtype, count, label):
    if not isinstance(text, str):
        raise RecordError(f"{label} stored values are base64 text")
    try:
        raw = base64.b64decode(text.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError) as error:
        raise RecordError(f"{label} stored values are not base64: {error}") from error
    data = array.array(NUMS_DTYPES[dtype])
    if len(raw) != count * data.itemsize:
        raise RecordError(f"{label} stored values are {len(raw)} bytes, not {count} x "
                          f"{data.itemsize} (truncated or corrupt)")
    data.frombytes(raw)
    if sys.byteorder != "little":
        data.byteswap()
    out = data.tolist()
    if dtype != "int64" and not all(math.isfinite(v) for v in out):
        raise RecordError(f"{label} stored values are not finite")
    if base64.b64encode(raw).decode("ascii") != text:
        raise RecordError(f"{label} stored values are not canonical base64")
    return out


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

    def __init__(self, patterns=(), *, any_count=False, floats=(), columns=False):
        self.patterns, self.any_count, self.floats = patterns, any_count, tuple(floats)
        self.columns = columns          # v3: a column claim (``nums``) is a labelled leaf

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
        if rule.columns and _is_nums(value):
            return                   # checked whole by ``check_nums`` in ``_bounded_v3``
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
def validate_body(kind, body, pointers=(), *, columns=False):
    """Check ``body`` against the declared kind and the authority rule; raises ``RecordError``.
    ``columns`` (schema v3): column claims (``nums``) are labelled numbers too."""
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
            if columns and _is_nums(value):
                check_nums(value, label)
            else:
                check_num(value, label)
    patterns = tuple(tuple(p.split(".")) for p in spec.plain_numeric)
    _scan_claims(body, "body", (), _Rule(patterns, columns=columns))
    have = {p.get("kind") for p in pointers}
    missing = [k for k in spec.pointers if k not in have]
    if missing:
        raise RecordError(f"record kind {kind!r} needs pointers of kind {missing}")


# --------------------------------------------------------------------------- seal / verify

def _check_envelope(record):
    """``model_identity`` and ``execution`` carry no measurements (see the module docstring)."""
    _scan_claims(record["model_identity"], "record.model_identity", (), _IDENTITY)
    _scan_claims(record["execution"], "record.execution", (), _PROFILE)


CONTENT_FIELDS = ("session_id", "kind", "model_identity", "execution", "authority", "pointers",
                  "body")


def _encode(record, paths=None):
    """The stored mapping of a logical v3 record (with or without ``id``/``digest``);
    ``paths``: the body's column paths when already known."""
    try:
        out = {k: v for k, v in record.items() if k != "execution"}
        out["execution_sha256"] = seal_digest(record["execution"])
        body = record["body"]
        if paths is None:
            paths = _column_paths(body) if isinstance(body, Mapping) else ()
        for path in paths:
            column = _at(body, path)
            if not isinstance(column["values"], Mapping):
                body = _replace_at(body, (*path, "values"),
                                   _pack(column["values"], column["dtype"]))
    except (TypeError, KeyError, OverflowError, ValueError) as error:
        raise RecordError(f"record cannot be encoded: {error}") from error
    out["body"] = body
    return out


def _decode(stored, execution):
    """The logical record of a stored v3 mapping, given its execution profile (unverified)."""
    if not isinstance(stored, Mapping) or set(stored) != set(STORED_FIELDS):
        raise RecordError("stored v3 record fields are " + ", ".join(STORED_FIELDS))
    if not isinstance(execution, Mapping):
        raise RecordError("a v3 record is read with its execution profile (a mapping)")
    if seal_digest(execution) != stored["execution_sha256"]:
        raise RecordError("the execution profile given differs from the record's "
                          "execution_sha256")
    body = stored["body"]
    for path in _column_paths(body) if isinstance(body, Mapping) else ():
        column = _at(body, path)
        label = "body" + "".join(f".{k}" for k in path)
        values = column["values"]
        if isinstance(values, Mapping):
            continue
        shape, dtype = column.get("shape"), column.get("dtype")
        if (dtype not in NUMS_DTYPES or not isinstance(shape, list) or not shape
                or not all(_count(n) for n in shape)):
            raise RecordError(f"{label} column has no valid shape and dtype")
        if math.prod(shape) > MAX_NUMS_INLINE:
            raise RecordError(f"{label} holds more than {MAX_NUMS_INLINE} values inline")
        body = _replace_at(body, (*path, "values"),
                           _unpack(values, dtype, math.prod(shape), label))
    out = {k: v for k, v in stored.items() if k != "execution_sha256"}
    out["execution"], out["body"] = dict(execution), body
    return out


def _size_v3(stored, label="record"):
    size = len(_canonical(stored).encode("utf-8"))
    if size > MAX_RECORD_BYTES_V3:
        raise RecordError(f"{label} is {size} bytes (> {MAX_RECORD_BYTES_V3}); store it by "
                          "pointer")
    return size


def _seal(*, session_id, kind, model_identity, execution, authority, body, pointers):
    """A v3 record of exactly this content (no custody added, no sink written)."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise RecordError("a record needs a session id")
    pointers = [_record_pointer(p) for p in pointers]
    envelope, paths = _bounded_v3({
        "schema": SCHEMA, "session_id": session_id, "kind": _short_text(kind, "record kind"),
        "model_identity": model_identity, "execution": execution,
        "authority": _authority(authority), "pointers": pointers, "body": body,
    }, "record")
    _check_envelope(envelope)
    validate_body(kind, envelope["body"], envelope["pointers"], columns=True)
    stored = _encode(envelope, paths)
    digest = seal_digest(stored)
    try:
        _size_v3({**stored, "id": digest, "digest": digest})
    except UnicodeEncodeError as error:
        raise RecordError("record holds text that cannot be encoded as UTF-8") from error
    return {**envelope, "id": digest, "digest": digest}


@_huge_int_guard
def seal_record(*, session_id, kind, model_identity, execution, authority, body, pointers=(),
                run=None):
    """Build and seal one record (schema v3). ``model_identity``/``execution`` are the
    session's values.

    Custody (P8, ``core.custody``): while custody is declared the body gains ``custody`` (script
    and prereg sha256, library versions, kernel path) unless it carries its own; every open
    record sink receives the sealed record. ``run``: the run (or runs) the record is about; when
    one is rooted at a checkpoint whose path is not recomputed, custody gains
    ``from_checkpoint`` (``custody.checkpoint_origin``)."""
    from . import custody

    if not isinstance(body, Mapping):
        raise RecordError("record body must be a mapping")
    record = _seal(session_id=session_id, kind=kind, model_identity=model_identity,
                   execution=execution, authority=authority, pointers=pointers,
                   body=custody.attach_custody(kind, dict(body), run=run))
    custody.emit(record)
    return record


def _verify_v2(record):
    record = _bounded_record(dict(record), "record")
    envelope = {k: v for k, v in record.items() if k not in ("id", "digest")}
    digest = seal_digest(envelope)
    if record["digest"] != digest or record["id"] != digest:
        raise RecordError("record seal differs from its content")
    return record, False


def _verify_v3(record):
    envelope, paths = _bounded_v3({k: v for k, v in record.items()
                                   if k not in ("id", "digest")}, "record")
    stored = _encode(envelope, paths)
    digest = seal_digest(stored)
    if record["digest"] != digest or record["id"] != digest:
        raise RecordError("record seal differs from its content")
    _size_v3({**stored, "id": digest, "digest": digest})
    return {**envelope, "id": digest, "digest": digest}, True


@_huge_int_guard
def verify_record(record, *, require_kind=True):
    """Check a logical record (v3 or v2): fields, seal, bounds, authority, pointers, kind and
    body. A stored v3 mapping (``execution_sha256``) is read with ``load_record``.

    The kind must be declared in this process (``require_kind`` is kept for callers and is
    always in force): a record of an undeclared kind is refused, sealed or not.
    """
    if isinstance(record, Mapping) and set(record) == set(STORED_FIELDS):
        raise RecordError("this is a stored v3 record (execution_sha256): read it with "
                          "load_record(stored, execution=...)")
    if not isinstance(record, Mapping) or set(record) != set(FIELDS):
        raise RecordError("record fields are " + ", ".join(FIELDS))
    if record["schema"] == SCHEMA:
        record, columns = _verify_v3(record)
    elif record["schema"] == SCHEMA_V2:
        record, columns = _verify_v2(record)
    else:
        raise RecordError(f"record schema must be one of {', '.join(map(repr, SCHEMAS))}")
    _authority(record["authority"])
    if not isinstance(record["pointers"], list):
        raise RecordError("record pointers are a list")
    pointers = [_record_pointer(p) for p in record["pointers"]]
    if _canonical(pointers) != _canonical(record["pointers"]):
        raise RecordError("record pointers are not in canonical form")
    _check_envelope(record)
    validate_body(record["kind"], record["body"], pointers, columns=columns)
    return record


MAX_LINE_BYTES = MAX_RECORD_BYTES_V3 + 4096     # a stored record line (v3 bound + framing slack)


def _no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise RecordError(f"duplicate key {key!r} in a stored record")
        out[key] = value
    return out


def parse_line(text, label="stored record", *, limit=MAX_LINE_BYTES):
    """Parse one stored JSON line: refused before parsing when longer than ``limit`` and on a
    duplicate key (the id covers the parsed value, so a first-wins reader must not see another
    one). Every error is a ``RecordError``."""
    if limit is not None and len(text) > limit:
        raise RecordError(f"{label} is {len(text)} bytes (> {limit}); refused before parsing")
    try:
        return json.loads(text, object_pairs_hook=_no_duplicates)
    except (ValueError, RecursionError) as error:
        raise RecordError(f"{label} is not readable JSON (truncated or corrupt): "
                          f"{error}") from error


def _parse(stored):
    if isinstance(stored, (bytes, bytearray)):
        if len(stored) > MAX_LINE_BYTES:
            raise RecordError(f"stored record is {len(stored)} bytes (> {MAX_LINE_BYTES}); "
                              "refused before parsing")
        try:
            stored = bytes(stored).decode("utf-8")
        except UnicodeDecodeError as error:
            raise RecordError(f"stored record is not UTF-8: {error}") from error
    if isinstance(stored, str):
        stored = parse_line(stored)
    if not isinstance(stored, Mapping):
        raise RecordError("a stored record is a JSON object")
    return stored


def load_record(stored, *, execution=None, executions=None, verify=True):
    """The logical record of a stored one (a JSON line, bytes or a parsed mapping), verified.

    v3 needs its execution profile: ``execution`` (checked against ``execution_sha256``) or
    ``executions`` (``{sha256: profile}``, e.g. a sink's ``mdb_execution`` lines). v2 records
    (and logical v3 records) are verified as they are. ``verify=False`` decodes only."""
    stored = _parse(stored)
    if set(stored) == set(STORED_FIELDS) and stored.get("schema") == SCHEMA:
        if execution is None:
            execution = (executions or {}).get(stored.get("execution_sha256"))
            if execution is None:
                raise RecordError(f"no execution profile {str(stored.get('execution_sha256'))[:12]}"
                                  " was given for this v3 record (execution= or executions=)")
        stored = _decode(stored, execution)
    return verify_record(stored) if verify else dict(stored)


def migrate_record(record):
    """The v3 record of a v2 record: the same content (every field but ``schema``, ``id`` and
    ``digest`` canonically equal, so every number bit-identical) under a new id. The v2 record
    is verified first; a v3 record is returned verified and unchanged. No custody is added and
    no sink receives it."""
    old = verify_record(record)
    if old["schema"] == SCHEMA:
        return old
    new = _seal(**{k: old[k] for k in CONTENT_FIELDS})
    same_content(old, new)
    return new


def same_content(a, b):
    """Raise ``RecordError`` unless two logical records hold the same content (canonical JSON
    of every field but ``schema``, ``id`` and ``digest``); returns the differing-field list
    (empty)."""
    differ = [k for k in CONTENT_FIELDS if _canonical(a[k]) != _canonical(b[k])]
    if differ:
        raise RecordError("records differ in " + ", ".join(differ))
    return differ


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

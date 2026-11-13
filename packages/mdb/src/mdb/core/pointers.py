"""The pointer rule: records hold locators and digests, never tensors.

Moved here unchanged from ``mdb.symbol_reads`` (which re-exports every name, so existing
imports and serialized records are byte-identical): the bounds (``check_bounded``), seals
(``seal_digest``) and the ``file`` / ``parameter`` / ``run`` pointers. The only code change is
that numpy is no longer imported at module load: a numpy scalar can only exist once numpy is
in ``sys.modules``, so ``_bounded`` looks it up there.

New here: the ``occurrence`` pointer (a value named by its canonical address and the program,
recipe event and cut that produce it) and a resolver registry. ``resolve(pointer, **context)``
dispatches on ``pointer["kind"]`` and raises ``Unresolved`` (never returns None). The
``occurrence`` resolver needs the engine and is registered by it, not here.

Pointer kinds and their fields:

* ``file``: ``locator {root, path}``, ``digest {sha256, size}``, optional ``array``,
  ``selector``, ``json_pointer``, ``role``. ``locator.path`` is relative and stays under its
  root: ``validate_pointer`` refuses an absolute path, a ``..`` component (``/`` or ``\\``
  separated), an empty path and a NUL byte; the ``file`` resolver also resolves the path and
  requires it to lie under the resolved root (a symlink out of the root is refused), without
  touching the target, with one outcome for an existing and a missing target.
* ``parameter``: ``parameter``, ``model`` (``model_identity`` record), optional ``selector``.
* ``run``: ``run_id``, optional ``plan_sha256`` and ``plan`` (a file pointer).
* ``occurrence``: ``address`` (canonical address JSON), ``program``, ``recipe``, ``event``,
  ``cut``, ``value {shape, dtype, sha256}``, optional ``authority``.
* ``tensor``: ``ref`` (the blob id in a saturn_pub ``LocalStore``), ``sha256`` (of the raw
  little-endian payload, ``tensor_digest``), ``shape``, ``dtype``. ``resolve_tensor`` reads it
  from a ``LocalStore``, checking both digests.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import sys
from collections.abc import Mapping
from pathlib import Path

MAX_INLINE_ITEMS = 64
MAX_RECORD_ITEMS = 4096
MAX_RECORD_BYTES = 64 * 1024
MAX_META_BYTES = 64 * 1024
MAX_TEXT_BYTES = 4 * 1024

# How a number was obtained. Never merged into one field with resolution state.
AUTHORITIES = ("measured", "derived-exact", "derived-local", "bound", "supplied")


class RecordError(ValueError):
    """A record or pointer is malformed, unbounded, or claims more than its evidence.

    ``mdb.symbol_reads.SymbolReadError`` is this class.
    """


class Unresolved(LookupError):
    """A pointer could not be turned into its value.

    ``refused`` is True when the target was found but failed verification (wrong digest or
    identity), False when it is simply not available in the given context.
    """

    def __init__(self, reason, *, refused=False):
        self.reason, self.refused = reason, refused
        super().__init__(reason)


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
            raise RecordError(f"{label} is {size} bytes (> {max_bytes}); store it by pointer")
    return out


def _bounded(value, label, depth, count, max_items, run):
    if depth > 16:
        raise RecordError(label + " nests too deeply")
    count[0] += 1
    if count[0] > max_items:
        raise RecordError(f"{label}: more than {max_items} values in one record; store it by pointer")
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str)):
        if isinstance(value, str) and len(value.encode()) > MAX_TEXT_BYTES:
            raise RecordError(label + " text exceeds the inline limit")
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        if not math.isfinite(float(value)):
            raise RecordError(label + " is not finite")
        return float(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise RecordError(label + " holds raw bytes; reference the artifact by pointer")
    if any(hasattr(value, a) for a in ("shape", "ndim", "detach", "__array__")):
        raise RecordError(label + " holds an array or tensor; reference it by pointer")
    if isinstance(value, Mapping):
        if len(value) > MAX_INLINE_ITEMS * 4:
            raise RecordError(label + " has too many keys")
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RecordError(label + " has a non-text key")
            out[key] = _bounded(item, f"{label}.{key}", depth + 1, count, max_items, None)
        return out
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_INLINE_ITEMS:
            raise RecordError(
                f"{label} has {len(value)} items (> {MAX_INLINE_ITEMS}); store it by pointer")
        # Items of directly nested sequences share the outermost sequence's item budget.
        run = [0] if run is None else run
        run[0] += len(value)
        if run[0] > MAX_INLINE_ITEMS:
            raise RecordError(f"{label} holds more than {MAX_INLINE_ITEMS} items across nested "
                              "sequences; store it by pointer")
        return [_bounded(item, f"{label}[{i}]", depth + 1, count, max_items,
                         run if isinstance(item, (list, tuple)) else None)
                for i, item in enumerate(value)]
    raise RecordError(f"{label} has unsupported type {type(value).__name__}")


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
        raise RecordError(label + " must be a sha256 hex digest")
    return text


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise RecordError(label + " must be non-empty text without surrounding whitespace")
    return value


def _number(value, label, *, optional=True):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise RecordError(label + " must be a finite number")
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
        raise RecordError(f"{path} is not under the declared root {root['name']!r}") from error
    pointer = {"kind": "file", "locator": {"root": root["name"], "path": relative.as_posix()},
               "digest": {"sha256": _sha(sha256, "file sha256") if sha256 else file_sha256(path),
                          "size": path.stat().st_size}}
    for key, value in (("array", array), ("selector", selector), ("json_pointer", json_pointer),
                       ("role", role)):
        if value is not None:
            pointer[key] = list(value) if isinstance(value, tuple) else value
    return check_bounded(pointer, "file pointer")


def _locate(pointer, roots):
    """``(path, real, inside)``: the root-joined path, its fully resolved real path and whether
    that stays under the resolved root. Nothing is opened or hashed, so an existing and a
    missing target give one outcome. A path that is absolute, climbs with ``..`` or cannot be
    resolved (a symlink loop) is not inside."""
    locator = pointer["locator"]
    if locator["root"] not in roots:
        raise RecordError(f"pointer names undeclared root {locator['root']!r}")
    root = Path(roots[locator["root"]])
    try:
        _relative_path(locator["path"])
        path = root / locator["path"]
        base = root.resolve()
        real = path.resolve()
    except (RecordError, OSError, RuntimeError, ValueError, TypeError):
        return root / "?", None, False
    return path, real, real.is_relative_to(base)


def resolve_pointer(pointer, roots):
    """Path of a file pointer under ``roots`` ({name: path}); refuses an undeclared root.

    Confined like ``resolve``: a path that leaves its root (absolute, ``..``, a symlink out) is
    a ``RecordError`` whether or not the target exists. The target is not opened.
    """
    path, real, inside = _locate(pointer, roots)
    if not inside:
        raise RecordError(ESCAPE)
    return path


def verify_pointer(pointer, roots):
    """Resolve a file pointer and check its digest (size, then sha256)."""
    path, real, inside = _locate(pointer, roots)
    if not inside:
        raise RecordError(ESCAPE)
    digest = pointer["digest"]
    if real.stat().st_size != digest["size"] or file_sha256(real) != digest["sha256"]:
        raise RecordError(f"{path} does not match its pointer digest")
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
            raise RecordError("an unverified model name says what would resolve it ('resolve_by')")
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
    raise RecordError("model identity_status must be 'verified' or 'unresolved'")


def parameter_pointer(model_identity_name, parameter, *, selector=None, checkpoint_sha256=None, resolve_by=None):
    """Reference a model parameter by model identity + parameter name (+ optional selector).

    Without ``checkpoint_sha256`` the model name is not a verified identity: the pointer says
    ``identity_status: "unresolved"`` and needs ``resolve_by``.
    """
    if not model_identity_name or not parameter:
        raise RecordError("parameter pointer needs a model identity and parameter name")
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
            raise RecordError("plan pointer digest differs from plan sha256")
        pointer["plan"] = dict(plan)
    return check_bounded(pointer, "run pointer")


# --------------------------------------------------------------------------- occurrence pointer

OCCURRENCE_FIELDS = ("kind", "address", "program", "recipe", "event", "cut", "value")


def _ref(value, label):
    """A short reference: non-empty text of [A-Za-z0-9_.:-], at most 128 characters."""
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:-")
    if not isinstance(value, str) or not 0 < len(value) <= 128 or set(value) - allowed:
        raise RecordError(label + " must be a short reference ([A-Za-z0-9_.:-], <= 128 chars)")
    return value


INT_CAP = 10 ** 15     # a sealed integer holds at most 15 decimal digits (records.INT_DIGITS)


def _count_ok(n):
    return isinstance(n, int) and not isinstance(n, bool) and 0 <= n < INT_CAP


def _value_descriptor(shape, dtype, sha256):
    if not isinstance(shape, (list, tuple)) or not all(_count_ok(n) for n in shape):
        raise RecordError("occurrence value shape must be a list of integers in [0, 10**15)")
    if not isinstance(dtype, str) or not dtype or len(dtype) > 32:
        raise RecordError("occurrence value dtype must be short text such as 'bfloat16'")
    return {"shape": list(shape), "dtype": dtype,
            "sha256": None if sha256 is None else _sha(sha256, "occurrence value sha256")}


def _occurrence_address(address, ports, global_ports):
    """An ``Occurrence`` from an object, text or JSON.

    Text is parsed against the port lists (default: the Qwen2 list). JSON and objects are
    checked against the lists only when they are given: stored pointers are validated without
    the engine's port program, so any well-formed port name passes there.
    """
    from . import address as addresses

    check = ports is not None or global_ports is not None
    try:
        if isinstance(address, str):
            return addresses.parse(address, ports=ports, global_ports=global_ports)
        if isinstance(address, Mapping):
            return addresses.from_json(dict(address), ports=ports, global_ports=global_ports,
                                       check_ports=check)
        if isinstance(address, (addresses.Occurrence, addresses.StateCell)):
            if check and isinstance(address, addresses.Occurrence):
                addresses.from_json(address.to_json(), ports=ports, global_ports=global_ports)
            return address
    except addresses.AddressError as error:
        raise RecordError(f"occurrence address: {error}") from error
    raise RecordError("an occurrence address is an Occurrence, its text or its JSON")


def occurrence_pointer(address, *, program, shape, dtype, sha256=None, recipe=None, event=None,
                       cut=None, authority=None, ports=None, global_ports=None):
    """Point at one occurrence value by address and the program that produced it.

    ``address`` is an ``Occurrence`` (not a state cell), its text or its JSON. ``program``
    names the producing program (``stepped``, ``prefill``...); it must agree with the
    address's own ``~program`` when that is set. ``recipe`` and ``cut`` say how to rebuild the
    value, ``event`` is the recipe event index, ``sha256`` the value's raw little-endian bytes
    (None until hashed). ``ports``/``global_ports``: see ``_occurrence_address``.
    """
    from . import address as addresses

    occurrence = _occurrence_address(address, ports, global_ports)
    if not isinstance(occurrence, addresses.Occurrence):
        raise RecordError("an occurrence pointer names a port occurrence, not a state cell")
    program = _ref(program, "occurrence program")
    if occurrence.program is not None and occurrence.program != program:
        raise RecordError("occurrence pointer program differs from its address program")
    if event is not None and not _count_ok(event):
        raise RecordError("occurrence event must be a recipe event index in [0, 10**15)")
    pointer = {"kind": "occurrence", "address": occurrence.to_json(), "program": program,
               "recipe": None if recipe is None else _ref(recipe, "occurrence recipe"),
               "event": event, "cut": None if cut is None else _sha(cut, "occurrence cut"),
               "value": _value_descriptor(shape, dtype, sha256)}
    if authority is not None:
        if authority not in AUTHORITIES:
            raise RecordError("authority must be one of " + ", ".join(AUTHORITIES))
        pointer["authority"] = authority
    return check_bounded(pointer, "occurrence pointer")


def validate_occurrence_pointer(pointer, *, ports=None, global_ports=None):
    """Re-validate a stored occurrence pointer; returns it in canonical form or raises."""
    if not isinstance(pointer, Mapping) or pointer.get("kind") != "occurrence":
        raise RecordError("not an occurrence pointer")
    keys = set(pointer)
    if not set(OCCURRENCE_FIELDS) <= keys <= set(OCCURRENCE_FIELDS) | {"authority"}:
        raise RecordError("occurrence pointer fields are " + ", ".join(OCCURRENCE_FIELDS)
                          + " and optional authority")
    value = pointer["value"]
    if not isinstance(value, Mapping) or set(value) != {"shape", "dtype", "sha256"}:
        raise RecordError("occurrence value is {shape, dtype, sha256}")
    if not isinstance(pointer["address"], Mapping):
        raise RecordError("a stored occurrence pointer holds the address JSON")
    address = _occurrence_address(pointer["address"], ports, global_ports)
    rebuilt = occurrence_pointer(address, program=pointer["program"], shape=value["shape"],
                                 dtype=value["dtype"], sha256=value["sha256"],
                                 recipe=pointer["recipe"], event=pointer["event"],
                                 cut=pointer["cut"], authority=pointer.get("authority"))
    if _canonical(rebuilt) != _canonical(check_bounded(dict(pointer), "occurrence pointer")):
        raise RecordError("occurrence pointer is not in canonical form")
    return rebuilt


# --------------------------------------------------------------------------- tensor pointer

TENSOR_FIELDS = ("kind", "ref", "sha256", "shape", "dtype")


def tensor_pointer(ref, *, sha256, shape, dtype):
    """Point at one tensor blob of a saturn_pub ``LocalStore`` (``put_tensor``'s result).

    ``ref`` is the blob id (sha256 of the stored file), ``sha256`` the payload digest, so the
    pointer is checked twice on resolve: the store checks the file, the resolver the payload.
    """
    descriptor = _value_descriptor(shape, dtype, sha256)
    if descriptor["sha256"] is None:
        raise RecordError("a tensor pointer needs the payload sha256")
    return check_bounded({"kind": "tensor", "ref": _sha(ref, "tensor blob ref"),
                          "sha256": descriptor["sha256"], "shape": descriptor["shape"],
                          "dtype": descriptor["dtype"]}, "tensor pointer")


def validate_tensor_pointer(pointer):
    """Re-validate a stored tensor pointer; returns it in canonical form or raises."""
    if not isinstance(pointer, Mapping) or set(pointer) != set(TENSOR_FIELDS):
        raise RecordError("tensor pointer fields are " + ", ".join(TENSOR_FIELDS))
    rebuilt = tensor_pointer(pointer["ref"], sha256=pointer["sha256"], shape=pointer["shape"],
                             dtype=pointer["dtype"])
    if _canonical(rebuilt) != _canonical(check_bounded(dict(pointer), "tensor pointer")):
        raise RecordError("tensor pointer is not in canonical form")
    return rebuilt


ESCAPE = "file pointer path must be non-empty, relative and stay under its root"


def _relative_path(path):
    """Refuse an absolute, empty, NUL-holding or ``..``-climbing path (text only; the message
    never repeats the path, so a refusal tells nothing about the filesystem)."""
    if not isinstance(path, str):
        return path
    flat = path.replace("\\", "/")
    if (not path.strip() or "\0" in path or flat.startswith("/")
            or (len(flat) > 1 and flat[0].isalpha() and flat[1] == ":")
            or ".." in flat.split("/")):
        raise RecordError(ESCAPE)
    return path


def validate_pointer(pointer):
    """Structural check of any pointer kind; unknown kinds need only bounded JSON and a kind."""
    if not isinstance(pointer, Mapping) or not isinstance(pointer.get("kind"), str):
        raise RecordError("a pointer is a mapping with a 'kind'")
    kind = pointer["kind"]
    if kind == "occurrence":
        return validate_occurrence_pointer(pointer)
    if kind == "tensor":
        return validate_tensor_pointer(pointer)
    pointer = check_bounded(dict(pointer), f"{kind} pointer")
    if kind == "file":
        locator, digest = pointer.get("locator"), pointer.get("digest")
        if (not isinstance(locator, dict) or set(locator) != {"root", "path"}
                or not isinstance(digest, dict) or set(digest) != {"sha256", "size"}):
            raise RecordError("file pointer needs locator {root, path} and digest {sha256, size}")
        _sha(digest["sha256"], "file sha256")
        _relative_path(locator["path"])
    elif kind == "parameter":
        if not isinstance(pointer.get("parameter"), str) or not pointer["parameter"]:
            raise RecordError("parameter pointer needs a parameter name")
        _identity_record(pointer.get("model"))
    elif kind == "run":
        _text(pointer.get("run_id"), "run id")
    return pointer


# --------------------------------------------------------------------------- resolvers

_RESOLVERS = {}


def register_resolver(kind, resolver, *, replace=False):
    """Register ``resolver(pointer, **context)`` for pointers of ``kind``."""
    if not isinstance(kind, str) or not kind:
        raise ValueError("resolver kind must be non-empty text")
    if kind in _RESOLVERS and not replace:
        raise ValueError(f"a resolver for {kind!r} is already registered")
    _RESOLVERS[kind] = resolver


def unregister_resolver(kind):
    _RESOLVERS.pop(kind, None)


def resolvers():
    return tuple(sorted(_RESOLVERS))


def resolve(pointer, **context):
    """The value a pointer names. Raises ``Unresolved`` when it cannot be produced."""
    kind = pointer.get("kind") if isinstance(pointer, Mapping) else None
    if isinstance(pointer, Mapping) and "kind" in pointer and not isinstance(kind, str):
        raise Unresolved("malformed pointer: kind must be text", refused=True)
    if kind not in _RESOLVERS:
        raise Unresolved(f"no resolver registered for pointer kind {kind!r}")
    try:
        value = _RESOLVERS[kind](pointer, **context)
    except Unresolved:
        raise
    except Exception as error:  # a resolver (built in or plug-in) never leaks a raw exception
        raise Unresolved(f"resolver {kind} failed: {type(error).__name__}", refused=True) from error
    if value is None:
        raise Unresolved(f"the {kind!r} resolver returned no value")
    return value


def _well_formed(pointer):
    """The pointer passes ``validate_pointer``; otherwise ``Unresolved`` (never KeyError)."""
    try:
        return validate_pointer(pointer)
    except (RecordError, TypeError, ValueError, KeyError, AttributeError) as error:
        raise Unresolved(f"malformed pointer: {error}", refused=True) from error


def _resolve_file(pointer, *, roots=None, **_):
    """Checked path of a file pointer; context ``roots`` = {root name: directory}."""
    pointer = _well_formed(pointer)
    locator, digest = pointer["locator"], pointer["digest"]
    if (not isinstance(locator["root"], str) or not isinstance(locator["path"], str)
            or isinstance(digest["size"], bool) or not isinstance(digest["size"], int)):
        raise Unresolved("malformed pointer: locator root/path are text and digest size an integer",
                         refused=True)
    if roots is None:
        raise Unresolved("file pointer needs roots={name: directory}")
    if pointer["locator"]["root"] not in roots:
        raise Unresolved(f"root {pointer['locator']['root']!r} is not declared in this context")
    path, real, inside = _locate(pointer, roots)   # follows symlinks; target not opened yet
    if not inside:
        raise Unresolved("outside root", refused=True)
    if not real.is_file():
        raise Unresolved(f"{path} is missing")
    if real.stat().st_size != digest["size"] or file_sha256(real) != digest["sha256"]:
        raise Unresolved(f"{path} does not match its pointer digest", refused=True)
    return real     # the path that was checked, not the unresolved one a symlink swap could redirect


def _resolve_parameter(pointer, *, parameters=None, model=None, checkpoint_sha256=None,
                       allow_unverified=False, **_):
    """A named parameter (selected), from ``parameters`` (a mapping) or ``model``.

    A verified pointer needs the context's ``checkpoint_sha256`` to equal its own; an
    unresolved identity is refused unless ``allow_unverified=True``.
    """
    pointer = _well_formed(pointer)
    if not isinstance(pointer.get("selector", []), (list, tuple)):
        raise Unresolved("malformed pointer: parameter selector must be a list", refused=True)
    identity = pointer.get("model")
    if not isinstance(identity, Mapping):
        raise Unresolved("malformed pointer: a parameter pointer names its model identity",
                         refused=True)
    if identity.get("identity_status") == "verified":
        if checkpoint_sha256 is None:
            raise Unresolved("verified parameter pointer needs the context's checkpoint_sha256")
        if checkpoint_sha256 != identity["checkpoint_sha256"]:
            raise Unresolved("checkpoint differs from the pointer's identity", refused=True)
    elif not allow_unverified:
        raise Unresolved("model identity is unresolved: " + str(identity.get("resolve_by")))
    if parameters is None and model is not None:
        if not hasattr(model, "named_parameters"):
            raise Unresolved("model= needs a named_parameters() method")
        parameters = dict(model.named_parameters())
    if parameters is None:
        raise Unresolved("parameter pointer needs parameters= or model=")
    if not isinstance(parameters, Mapping):
        raise Unresolved("parameters= must be a mapping")
    if pointer["parameter"] not in parameters:
        raise Unresolved(f"parameter {pointer['parameter']!r} is not in this model")
    value = parameters[pointer["parameter"]]
    for index in pointer.get("selector", ()):
        try:
            value = value[index]
        except (IndexError, KeyError, TypeError) as error:
            raise Unresolved("parameter selector is outside the parameter", refused=True) from error
    return value


def resolve_tensor(pointer, *, store=None, framework="torch"):
    """The tensor a tensor pointer names, from ``store`` (a ``LocalStore``).

    A missing blob is ``Unresolved``; a blob that fails the store's digest checks or whose
    payload digest, shape or dtype differs from the pointer is a refusal naming the blob. Not
    in the ``resolve`` registry (which stays file and parameter): call it directly.
    """
    pointer = _well_formed(pointer)
    if store is None:
        raise Unresolved("tensor pointer needs store= (the LocalStore holding its blob)")
    ref = pointer["ref"]
    if not store.has_tensor(ref):
        raise Unresolved(f"tensor blob {ref} is missing from the store")
    try:
        info = store.tensor_info(ref)
        value = store.get_tensor(ref, framework=framework)
    except (ValueError, OSError) as error:
        raise Unresolved(f"tensor blob {ref} is corrupt: {error}", refused=True) from error
    if (info["sha256"], info["shape"], info["dtype"]) != (
            pointer["sha256"], pointer["shape"], pointer["dtype"]):
        raise Unresolved(f"tensor blob {ref} differs from its pointer (payload digest, shape or "
                         "dtype)", refused=True)
    return value


register_resolver("file", _resolve_file)
register_resolver("parameter", _resolve_parameter)

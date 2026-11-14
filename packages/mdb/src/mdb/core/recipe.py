"""Replay recipe v2: a sealed, bounded record that rebuilds a debugger cursor from the model alone.

::

    {schema: "mdb-replay-recipe-v2", name, model_identity, layers, tokens, prompt_sha256,
     numeric_scope, lookahead, transitions: [{transition, layer, edits}], staged: [edit],
     target: {where, transitions, port}, cut, fingerprint}

It is ``Run.checkpoint()`` one to one (``from_checkpoint`` / ``to_checkpoint``): ``transitions``
are its ``path_edits`` (the edits committed with each native transition, in order), ``staged``
its ``edits`` (staged in the pending layer at the target), ``target`` its ``where``,
``transitions`` and ``port``, ``cut`` the boundary cut fingerprint the replay must reproduce.
What a checkpoint holds as a tensor digest the recipe holds as a ``tensor`` pointer
(``mdb.core.pointers.tensor_pointer``: blob ref + payload sha256) into the project store; an
edit is ``{port, op, value, mask?, mask_count?, selection?}`` with ``value`` either a tensor
pointer or ``{kind: "scalar", value: num(x, "supplied")}``. A state-cell edit
(``mdb.engine.edits.CellEdit``) adds ``cell`` ``{layer, tensor, rows, kv, d, timing,
staged_at}``, ``dose`` (``mix`` only, a supplied number) and ``origin`` (the pointer of a value
taken from another run: names and digests, never the array; the value itself is the edit's
tensor pointer). Recipes written before cells existed read unchanged. Token ids are inline up to
``MAX_INLINE_ITEMS`` and a tensor pointer (int64) beyond. Arrays are never inline: the recipe
passes ``check_bounded`` (no array, at most 64 KiB) before it is sealed and again when read.

The seal is ``mdb.core.index.recipe_fingerprint`` (the recipe table's own convention), so a
recipe is stored with ``mdb.core.index.put_recipe`` and its id is its fingerprint.

Replaying (``mdb.ops.debugger.Debugger.restart``) refuses with a ``RecipeError`` (a
``Refused``) whose ``code`` is one of ``CODES``, each with its own message: another model identity, another
numeric scope, a missing blob, a corrupt blob, a malformed recipe, a replay that reached
another cut fingerprint, or (``origin-edits``) a recipe whose edits at an occurrence are not
the ones an ``@cut+edit`` origin names.

Edited origins. A value read under edits is addressed ``@<cut>+<ref>+<ref>:<target>``: the
boundary cut it is computed from (whose fingerprint binds everything before it, path edits
included) and, in order, the reference (``edit_ref``: sha256 of the canonical manifest) of each
edit applied in the transition that computes it. ``occurrence_checkpoint(..., edits=refs)``
refuses a recipe whose edits there have other references, so a pointer never resolves through
edits it was not read under.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

from saturn_pub.values import digest

from . import index
from .errors import Refused
from .pointers import (
    MAX_INLINE_ITEMS,
    MAX_RECORD_BYTES,
    RecordError,
    Unresolved,
    _canonical,
    check_bounded,
    resolve_tensor,
    tensor_pointer,
    validate_tensor_pointer,
)
from .records import check_num, declare_kind, num, seal_record

SCHEMA = "mdb-replay-recipe-v2"
CHECKPOINT_SCHEMA = "mdb-engine-checkpoint-v1"   # mdb.engine.run.CHECKPOINT_SCHEMA
FIELDS = ("schema", "name", "model_identity", "layers", "tokens", "prompt_sha256",
          "numeric_scope", "lookahead", "transitions", "staged", "target", "cut", "fingerprint")
CODES = ("model-identity", "numeric-scope", "missing-blob", "corrupt-blob", "malformed",
           "fingerprint", "origin-edits")
_EDIT_KEYS = {"port", "op", "value", "mask", "mask_count", "selection", "cell", "dose", "origin"}
_CELL_KEYS = {"layer", "tensor", "rows", "kv", "d", "timing", "staged_at"}


class RecipeError(Refused):
    """A recipe cannot be built, read or replayed; ``code`` is one of ``CODES``."""

    def __init__(self, code, message):
        if code not in CODES:
            raise ValueError(f"unknown recipe refusal code {code!r}")
        self.code = code
        super().__init__(message)


def _malformed(message):
    return RecipeError("malformed", f"malformed replay recipe: {message}")


# --------------------------------------------------------------------------- build

def _put(store, tensor, sha256, label):
    """Store one tensor; its payload digest must be the one the checkpoint recorded."""
    stored = store.put_tensor(tensor)
    if stored["sha256"] != sha256:
        raise _malformed(f"{label} digest {sha256[:12]} does not match the tensor held for it")
    return tensor_pointer(stored["ref"], sha256=stored["sha256"], shape=stored["shape"],
                          dtype=stored["dtype"])


def _edit(manifest, tensors, store):
    """A checkpoint edit manifest as a recipe edit; its tensors go to ``store``."""
    described = manifest["value"]
    if described["kind"] == "scalar":
        value = {"kind": "scalar", "value": num(described["value"], "supplied")}
    else:
        sha = described["sha256"]
        if sha not in tensors:
            raise _malformed(f"edit value {sha[:12]} at {manifest['port']} is not held")
        value = _put(store, tensors[sha], sha, f"edit value at {manifest['port']}")
    out = {"port": manifest["port"], "op": manifest["op"], "value": value}
    if "selection" in manifest:
        out["selection"] = manifest["selection"]
    if "cell" in manifest:
        out["cell"] = dict(manifest["cell"])
    if "dose" in manifest:
        out["dose"] = num(manifest["dose"], "supplied")
    if "origin" in manifest:
        out["origin"] = dict(manifest["origin"])
    if "mask_sha256" in manifest:
        sha = manifest["mask_sha256"]
        if sha not in tensors:
            raise _malformed(f"edit mask {sha[:12]} at {manifest['port']} is not held")
        out["mask"] = _put(store, tensors[sha], sha, f"edit mask at {manifest['port']}")
        out["mask_count"] = int(manifest["mask_count"])
    return out


def from_checkpoint(checkpoint, *, tokens, layers, tensors, store):
    """Seal the recipe of a ``Run.checkpoint()``.

    ``tokens`` are the run's token ids, ``layers`` the model's layer count, ``tensors`` maps the
    payload sha256 of every edit value and mask the checkpoint names to its tensor
    (``Run.edit_tensors()``), ``store`` the project ``LocalStore`` the tensors are written to.
    Raises ``RecordError`` when the recipe would hold an array or exceed 64 KiB.
    """
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise _malformed("not an mdb engine checkpoint")
    tokens = [int(t) for t in tokens]
    if digest(tokens) != checkpoint["prompt_sha256"]:
        raise _malformed("the token ids are not the checkpoint's prompt")
    if len(tokens) > MAX_INLINE_ITEMS:
        import numpy as np

        stored = store.put_tensor(np.asarray(tokens, dtype=np.int64))
        tokens = tensor_pointer(stored["ref"], sha256=stored["sha256"], shape=stored["shape"],
                                dtype=stored["dtype"])
    body = {
        "schema": SCHEMA,
        "name": checkpoint.get("name"),
        "model_identity": checkpoint["model_identity"],
        "layers": int(layers),
        "tokens": tokens,
        "prompt_sha256": checkpoint["prompt_sha256"],
        "numeric_scope": dict(checkpoint.get("numeric_scope") or {}),
        "lookahead": checkpoint["lookahead"],
        "transitions": [{"transition": int(entry["transition"]), "layer": int(entry["layer"]),
                         "edits": [_edit(m, tensors, store) for m in entry["edits"]]}
                        for entry in checkpoint["path_edits"]],
        "staged": [_edit(m, tensors, store) for m in checkpoint["edits"]],
        "target": {"where": checkpoint["where"], "transitions": int(checkpoint["transitions"]),
                   "port": checkpoint["port"]},
        "cut": checkpoint["cut"],
    }
    return seal(body)


def seal(body):
    """Bound and seal a recipe body (``fingerprint`` is recomputed); raises ``RecordError``."""
    body = check_bounded({k: v for k, v in dict(body).items() if k != "fingerprint"},
                         "replay recipe", max_bytes=MAX_RECORD_BYTES)
    return verify({**body, "fingerprint": index.recipe_fingerprint(body)})


# --------------------------------------------------------------------------- read

def _check_edit(edit, label):
    if not isinstance(edit, Mapping) or not {"port", "op", "value"} <= set(edit) <= _EDIT_KEYS:
        raise _malformed(f"{label} is {{port, op, value, mask?, mask_count?, selection?, cell?, "
                         "dose?, origin?}")
    if "cell" in edit and (not isinstance(edit["cell"], Mapping)
                           or set(edit["cell"]) != _CELL_KEYS):
        raise _malformed(f"{label} cell is {{{', '.join(sorted(_CELL_KEYS))}}}")
    if ("dose" in edit or "origin" in edit) and "cell" not in edit:
        raise _malformed(f"{label}: dose and origin belong to a state-cell edit")
    if "dose" in edit and check_num(edit["dose"], f"{label} dose")["authority"] != "supplied":
        raise _malformed(f"{label} dose is a supplied number")
    if "origin" in edit and not isinstance(edit["origin"], Mapping):
        raise _malformed(f"{label} origin is an object")
    value = edit["value"]
    if isinstance(value, Mapping) and value.get("kind") == "scalar":
        if set(value) != {"kind", "value"}:
            raise _malformed(f"{label} scalar is {{kind, value}}")
        claim = check_num(value["value"], f"{label} value")
        if claim["authority"] != "supplied":
            raise _malformed(f"{label} scalar is a supplied number")
    else:
        validate_tensor_pointer(value)
    if ("mask" in edit) != ("mask_count" in edit):
        raise _malformed(f"{label} carries a mask and its count together")
    if "mask" in edit:
        validate_tensor_pointer(edit["mask"])
        if edit["mask"]["dtype"] != "bool":
            raise _malformed(f"{label} mask is a bool tensor")


def verify(recipe):
    """Check a recipe's fields, bounds, seal and edits; returns it or raises ``RecipeError``."""
    if not isinstance(recipe, Mapping) or set(recipe) != set(FIELDS):
        raise _malformed("fields are " + ", ".join(FIELDS))
    if recipe["schema"] != SCHEMA:
        raise _malformed(f"schema must be {SCHEMA!r}")
    try:
        bounded = check_bounded(dict(recipe), "replay recipe", max_bytes=MAX_RECORD_BYTES)
        body = {k: v for k, v in bounded.items() if k != "fingerprint"}
        if recipe["fingerprint"] != index.recipe_fingerprint(body):
            raise _malformed("its seal differs from its content")
        if not isinstance(recipe["tokens"], list):
            validate_tensor_pointer(recipe["tokens"])
        if not isinstance(recipe["transitions"], list) or not isinstance(recipe["staged"], list):
            raise _malformed("transitions and staged are lists")
        for n, entry in enumerate(recipe["transitions"]):
            if set(entry) != {"transition", "layer", "edits"}:
                raise _malformed("a transition is {transition, layer, edits}")
            for i, edit in enumerate(entry["edits"]):
                _check_edit(edit, f"transitions[{n}].edits[{i}]")
        for i, edit in enumerate(recipe["staged"]):
            _check_edit(edit, f"staged[{i}]")
        if set(recipe["target"]) != {"where", "transitions", "port"}:
            raise _malformed("target is {where, transitions, port}")
    except RecordError as error:
        raise _malformed(str(error)) from error
    return dict(recipe)


def _manifest(edit):
    """A recipe edit as the checkpoint edit manifest it came from."""
    value = edit["value"]
    if value["kind"] == "scalar":
        described = {"kind": "scalar", "value": value["value"]["value"]}
    else:
        described = {"kind": "tensor", "sha256": value["sha256"], "shape": list(value["shape"]),
                     "dtype": "torch." + value["dtype"]}
    out = {"port": edit["port"], "op": edit["op"], "value": described}
    if "cell" in edit:
        out["cell"] = dict(edit["cell"])
    if "dose" in edit:
        out["dose"] = edit["dose"]["value"]
    if "selection" in edit:
        out["selection"] = edit["selection"]
    if "origin" in edit:
        out["origin"] = dict(edit["origin"])
    if "mask" in edit:
        out.update(mask_sha256=edit["mask"]["sha256"], mask_count=edit["mask_count"],
                   mask_shape=list(edit["mask"]["shape"]))
    return out


def to_checkpoint(recipe):
    """The ``Run.checkpoint()`` dict the recipe was sealed from (``Run.restore`` input)."""
    recipe = verify(recipe)
    target = recipe["target"]
    return {
        "schema": CHECKPOINT_SCHEMA, "name": recipe["name"],
        "model_identity": recipe["model_identity"], "prompt_sha256": recipe["prompt_sha256"],
        "where": target["where"], "cut": recipe["cut"], "transitions": target["transitions"],
        "port": target["port"], "lookahead": recipe["lookahead"],
        "numeric_scope": dict(recipe["numeric_scope"]),
        "edits": [_manifest(e) for e in recipe["staged"]],
        "path_edits": [{"transition": t["transition"], "layer": t["layer"],
                        "edits": [_manifest(e) for e in t["edits"]]}
                       for t in recipe["transitions"]],
    }


def tensor_pointers(recipe):
    """Every tensor pointer of the recipe's edits (values and masks), in recipe order."""
    edits = [e for t in recipe["transitions"] for e in t["edits"]] + list(recipe["staged"])
    out = []
    for edit in edits:
        if edit["value"]["kind"] == "tensor":
            out.append(edit["value"])
        if "mask" in edit:
            out.append(edit["mask"])
    return out


def _load(pointer, store, label):
    try:
        return resolve_tensor(pointer, store=store, framework="torch")
    except Unresolved as error:
        code = "corrupt-blob" if error.refused else "missing-blob"
        raise RecipeError(code, f"{label} blob {pointer['ref']} (payload sha256 "
                                  f"{pointer['sha256'][:12]}): {error.reason}") from error


def load_tensors(recipe, store):
    """``{payload sha256: torch tensor}`` of every edit value and mask, each verified against
    the store's file digest and the recipe's payload digest. A missing blob is
    ``RecipeError("missing-blob")``, a corrupt one ``RecipeError("corrupt-blob")``, each naming
    the blob."""
    return {p["sha256"]: _load(p, store, "edit tensor") for p in tensor_pointers(recipe)}


def load_tokens(recipe, store):
    """The recipe's token ids (read from the store when held by pointer), digest-checked."""
    tokens = recipe["tokens"]
    if not isinstance(tokens, list):
        tokens = [int(t) for t in _load(tokens, store, "token ids").reshape(-1).tolist()]
    if digest(tokens) != recipe["prompt_sha256"]:
        raise _malformed("its token ids differ from its prompt_sha256")
    return tokens


def check_binding(recipe, *, model_identity, numeric_scope):
    """Refuse a replay on another model identity or under another numeric scope (a cut binds
    both, so the replay could not reproduce the recorded fingerprint)."""
    if _canonical(recipe["model_identity"]) != _canonical(model_identity):
        raise RecipeError("model-identity",
                          f"recipe {recipe['fingerprint'][:12]} was recorded on model identity "
                          f"{str(recipe['model_identity'])[:16]}; the attached model is "
                          f"{str(model_identity)[:16]} (wrong model identity)")
    recorded, mine = dict(recipe["numeric_scope"]), dict(numeric_scope or {})
    differing = sorted(k for k in set(recorded) | set(mine) if recorded.get(k) != mine.get(k))
    if differing:
        detail = ", ".join(f"{k}: recipe {recorded.get(k)!r}, this process {mine.get(k)!r}"
                           for k in differing)
        raise RecipeError("numeric-scope",
                          f"recipe {recipe['fingerprint'][:12]} was recorded under another "
                          f"numeric scope ({detail}); set the same scope and re-attach")


def fingerprint_mismatch(recipe, expected, reached):
    """The refusal for a replay of ``recipe`` that reached cut ``reached``, not ``expected``."""
    return RecipeError("fingerprint",
                       f"replaying recipe {recipe['fingerprint'][:12]} reached cut {reached[:12]}, "
                       f"not the recorded {expected[:12]} (fingerprint mismatch)")


def edit_ref(manifest):
    """The reference of one edit in an ``@cut+edit`` origin: the sha256 of its checkpoint
    manifest (``Run.origin_of``'s ``edits`` items; port, op, the value or its payload digest,
    mask, selection, cell, dose, origin) as canonical JSON. Any differing field gives another
    reference. 64 hex characters, so it is a valid origin edit id."""
    if not isinstance(manifest, Mapping) or not {"port", "op", "value"} <= set(manifest):
        raise _malformed("an edit manifest is {port, op, value, ...}")
    return digest(dict(manifest))


def occurrence_checkpoint(recipe, *, where, cut, event, port, edits=None):
    """The ``Run.checkpoint()`` whose cursor sits on one occurrence of the recipe's history.

    ``cut``/``event`` are the boundary cut and transition count the occurrence is read at or
    computed from (``Run.origin_of``), ``port`` its port inside that layer (None for a boundary
    value). The edits staged there are the recipe's edits of transition ``event`` (its staged
    edits when ``event`` is the target's transition); the path is every recorded transition
    before ``event``. Restoring it and reading at the cursor reproduces the occurrence.

    ``edits``: the ordered edit references of an ``@cut+edit`` origin (``edit_ref``). When given,
    the staged edits must have exactly those references, else ``RecipeError("origin-edits")``:
    the recipe would compute the value under other edits than it was read under.
    """
    recipe = verify(recipe)
    target = recipe["target"]["transitions"]
    if event > target:
        raise _malformed(f"event {event} is past the recipe target (transition {target})")
    layer = (recipe["staged"] if event == target else next(
        (t["edits"] for t in recipe["transitions"] if t["transition"] == event), []))
    out = to_checkpoint(recipe)
    out.update(name=None, where=where, cut=cut, transitions=event, port=port,
               edits=[_manifest(e) for e in layer] if port is not None else [],
               path_edits=[p for p in out["path_edits"] if p["transition"] < event])
    if edits is not None:
        mine = [edit_ref(m) for m in out["edits"]]
        if mine != list(edits):
            raise RecipeError(
                "origin-edits",
                f"recipe {recipe['fingerprint'][:12]} applies {len(mine)} edit(s) at transition "
                f"{event} ({', '.join(r[:12] for r in mine) or 'none'}); the address origin "
                f"names {len(edits)} ({', '.join(str(r)[:12] for r in edits) or 'none'}) (the "
                "edits differ: tampered, or another recipe)")
    return out


# --------------------------------------------------------------------------- project

CHECKPOINT = declare_kind("checkpoint", required=("name", "recipe", "where", "cut", "port"),
                          plain_numeric=("transitions", "serial"))


def save(project, session_id, recipe):
    """Verify and index a recipe in the project; returns its id (its fingerprint)."""
    return index.put_recipe(project, session_id, verify(recipe))


def load(project, recipe_id):
    """A recipe from the project index, seal and content verified."""
    return verify(index.get_recipe(project, recipe_id))


def checkpoints(project, session_id):
    """The session's checkpoint records (verified v2 records), oldest first."""
    rows = index.list_records(project, session_id, kind=CHECKPOINT.name, strict=True)
    return sorted(rows, key=lambda r: r["body"]["serial"])


def record_checkpoint(project, session_id, recipe):
    """Index a named checkpoint of a saved recipe and move the session cursor to it.

    The record (kind ``checkpoint``) holds the name, the recipe id, the target and a serial
    number, so re-saving a name later is a newer record even when the content repeats.
    """
    row = project.session(session_id)
    target = recipe["target"]
    record = seal_record(
        session_id=session_id, kind=CHECKPOINT.name, model_identity=json.loads(row["identity"]),
        execution=json.loads(row["execution"]), authority="measured",
        body={"name": recipe["name"], "recipe": recipe["fingerprint"], "where": target["where"],
              "cut": recipe["cut"], "transitions": target["transitions"], "port": target["port"],
              "serial": len(checkpoints(project, session_id))})
    identifier = index.put_record(project, record)
    index.set_cursor(project, session_id, identifier)
    return record


def find_checkpoint(project, name, *, session_id=None):
    """The newest checkpoint record named ``name``: in ``session_id`` first, else in exactly one
    other session of the project. None when no session has it; ``Refused`` when several other
    sessions do (name the session to choose)."""
    def newest(sid):
        named = [r for r in checkpoints(project, sid) if r["body"]["name"] == name]
        return named[-1] if named else None

    if session_id is not None:
        found = newest(session_id)
        if found is not None:
            return found
    others = [r["session_id"] for r in project.sessions() if r["session_id"] != session_id]
    found = [record for record in map(newest, others) if record is not None]
    if len(found) > 1:
        raise Refused(f"checkpoint {name!r} exists in sessions "
                      + ", ".join(r["session_id"] for r in found) + "; attach with session=")
    return found[0] if found else None

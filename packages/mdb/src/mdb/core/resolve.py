"""Resolve an occurrence pointer to its tensor (``mdb.resolve``), cheapest source first.

Tiers, tried in ``TIERS`` order (``tiers=`` picks a subset, in that order):

1. ``live``: the attached debugger's live run already holds the value in its port cache
   (``Run.row_captures``; nothing is computed). Only for the attached session's own origin, and
   only when the live run reached the occurrence from the pointer's cut with the recipe's edits.
2. ``blob``: the project's occurrence index (``mdb.core.index.find_occurrence``) names a tensor
   blob for the address, recorded under the pointer's recipe and cut.
3. ``cut``: the pointer's cut is retained in the project store; the model restarts from it and
   a read-only probe computes the value (needs the live model).
4. ``replay``: the recipe is replayed from the token ids (needs the live model).

Every answer is checked against the pointer's recorded payload digest (``value.sha256``, when
set): a different digest is a refusal and no later tier is tried. A resolved value carries the
tier that answered and its authority: what ``Run.read_info`` reports for the value in the
computing tiers (``measured`` for a value no edit wrote), the pointer's own ``authority`` (else
``measured``) for a stored blob.

Origins. An address with no origin, or with the attached session's id as its branch, reads the
attached session. Any other branch ``b`` is resolved only through a recipe for ``b`` held in
the project (a session ``b`` that indexed the pointer's recipe or has a cursor, or a checkpoint
named ``b``); otherwise the pointer is refused, never answered from the attached session. (The
core address mapping ``to_state_address`` does not look at origins; this module is the guard.)
An ``@cut`` origin must name the pointer's own cut and carry no edit references.

Without a live model tiers 3 and 4 cannot run: when 1 and 2 cannot answer either, the result is
``unresolved`` and ``needs`` says what would resolve it. This module needs no framework; the
computing tiers call the attached debugger (``mdb.ops.debugger.Debugger``) passed as
``session=``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import address as addresses
from . import index
from . import recipe as recipes
from .errors import Refused
from .pointers import RecordError, Unresolved, validate_occurrence_pointer

TIERS = ("live", "blob", "cut", "replay")


@dataclass
class Resolution:
    """The outcome of one ``resolve``: ``state`` is ``resolved``, ``unresolved`` or ``refused``.

    ``value`` (a tensor), ``tier`` and ``authority`` are set only when resolved; ``reason`` says
    why not otherwise and ``needs`` what would resolve an unresolved pointer. ``attempts`` lists
    each tier tried, ``{tier, state, reason}``.
    """
    state: str
    value: Any = None
    tier: str | None = None
    authority: str | None = None
    sha256: str | None = None
    reason: str | None = None
    needs: str | None = None
    attempts: list = field(default_factory=list)


class _Refusal(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _digest(value):
    """sha256 of the raw little-endian bytes (``mdb.engine.ports.tensor_digest``)."""
    raw = value.detach().cpu().contiguous().reshape(-1).view(__import__("torch").uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def _open(project, session):
    if project is None:
        project = getattr(session, "project", None)
    if project is None:
        raise _Refusal("resolving an occurrence pointer needs project= (or a debugger attached "
                       "with project=)")
    if isinstance(project, (str, Path)):
        mine = getattr(session, "project", None)
        if mine is not None and mine.root == Path(project).expanduser().resolve():
            return mine
        from .project import Project

        return Project(project, read_only=True)
    return project


def _recipes_for(project, branch, current):
    """Recipe ids that stand for origin ``branch``: recipes indexed by session ``branch``, the
    recipe under its cursor, and the recipes of checkpoints named ``branch``."""
    found = []
    sessions = {r["session_id"] for r in project.sessions()}
    if branch in sessions:
        found += index.recipe_ids(project, branch)
        cursor = index.get_cursor(project, branch)
        if cursor is not None:
            found.append(index.get_record(project, branch, cursor)["body"]["recipe"])
    named = recipes.find_checkpoint(project, branch, session_id=current)
    if named is not None:
        found.append(named["body"]["recipe"])
    return list(dict.fromkeys(found))


def _target(pointer, occurrence, project, session):
    """``(session id the value belongs to, recipe id, own)``: ``own`` when that is the attached
    session (so its live run may answer)."""
    current = getattr(session, "session_id", None)
    origin = occurrence.origin
    wanted = pointer["recipe"]
    if origin is not None and origin.cut is not None:
        if origin.edits or origin.cut != pointer["cut"]:
            raise _Refusal(f"origin @{origin.cut[:12]} is not this pointer's cut with no edits")
        origin = None
    if origin is None or origin.branch == current:
        if wanted is None and current is not None:
            cursor = index.get_cursor(project, current)
            if cursor is not None:
                wanted = index.get_record(project, current, cursor)["body"]["recipe"]
        return current, wanted, current is not None
    branch = origin.branch
    candidates = _recipes_for(project, branch, current)
    if not candidates:
        raise _Refusal(f"address origin {branch}: is not the attached session "
                       f"{current!r} and the project holds no recipe for {branch!r}; it is "
                       "not read from the attached session")
    if wanted is not None and wanted not in candidates:
        raise _Refusal(f"the pointer's recipe {wanted[:12]} is not a recipe for origin "
                       f"{branch}:")
    return branch, wanted or candidates[-1], False


def _layers(session, recipe):
    adapter = getattr(session, "adapter", None)
    return recipe["layers"] if recipe is not None else getattr(adapter, "layers", None)


def resolve(pointer, *, project=None, session=None, tiers=TIERS):
    """Resolve an occurrence pointer; returns a ``Resolution`` (see the module docstring).

    ``project``: an ``mdb.core.project.Project`` or its path (opened read-only), default the
    attached debugger's. ``session``: the attached debugger (``mdb.attach(..., project=)``),
    needed for the ``live``, ``cut`` and ``replay`` tiers.
    """
    unknown = [t for t in tiers if t not in TIERS]
    if unknown or not tiers:
        raise ValueError(f"tiers are a non-empty subset of {', '.join(TIERS)}")
    attempts = []

    def refused(reason):
        return Resolution("refused", reason=reason, attempts=attempts)

    try:
        pointer = validate_occurrence_pointer(pointer)
        occurrence = addresses.from_json(pointer["address"], check_ports=False)
        project = _open(project, session)
        sid, recipe_id, own = _target(pointer, occurrence, project, session)
        recipe = recipes.load(project, recipe_id) if recipe_id is not None else None
    except (_Refusal, RecordError, addresses.AddressError, index.NotIndexed) as error:
        return refused(str(getattr(error, "reason", error)))
    except ValueError as error:   # an unopenable project, an unknown session
        return refused(str(error))
    expected = pointer["value"]["sha256"]
    text = addresses.format_address(replace(occurrence, origin=None, program=None))
    port = (occurrence.port if occurrence.layer is not None and occurrence.port != "h_out"
            else None)
    context = {"pointer": pointer, "recipe": recipe, "address": text, "port": port,
               "selected": bool(occurrence.selector),
               "sid": sid, "own": own, "project": project, "session": session}
    for tier in tiers:
        try:
            value, authority = _TIER[tier](context)
        except (_Refusal, Refused) as error:
            attempts.append({"tier": tier, "state": "refused", "reason": error.reason})
            return refused(error.reason)
        except Unresolved as error:
            attempts.append({"tier": tier, "state": "unresolved", "reason": error.reason})
            continue
        got = _digest(value)
        if expected is not None and got != expected:
            reason = (f"the {tier} tier produced digest {got[:12]}, the pointer records "
                      f"{expected[:12]} (digest mismatch)")
            attempts.append({"tier": tier, "state": "refused", "reason": reason})
            return refused(reason)
        attempts.append({"tier": tier, "state": "resolved", "reason": None})
        state = "refused" if authority is None else "resolved"
        return Resolution(state, value=value if authority else None, tier=tier,
                          authority=authority, sha256=got, attempts=attempts,
                          reason=None if authority else f"the {tier} value's label was refused")
    model = getattr(getattr(session, "adapter", None), "model_identity", None)
    needs = None
    if model is None and any(t in tiers for t in ("cut", "replay")):
        identity = recipe["model_identity"] if recipe is not None else "the recorded model"
        needs = (f"attach the model with identity {str(identity)[:16]} (same numeric scope) "
                 "and pass session= so the cut or replay tier can compute the value")
    return Resolution("unresolved", reason="; ".join(f"{a['tier']}: {a['reason']}"
                                                     for a in attempts),
                      needs=needs, attempts=attempts)


# --------------------------------------------------------------------------- tiers

def _engine(context, tier):
    session = context["session"]
    if getattr(session, "adapter", None) is None:
        raise Unresolved(f"the {tier} tier needs the attached debugger (session=) and its model")
    if context["recipe"] is None:
        raise Unresolved("the pointer names no recipe and the session has no cursor recipe")
    if context["pointer"]["event"] is None:
        raise Unresolved("the pointer records no recipe event")
    if context["selected"]:
        raise Unresolved("an address with selectors is resolved from a stored blob only")
    return session


def _checkpoint(context):
    pointer = context["pointer"]
    try:
        return recipes.occurrence_checkpoint(context["recipe"], where=context["address"],
                                             cut=pointer["cut"], event=pointer["event"],
                                             port=context["port"])
    except recipes.RecipeError as error:
        raise _Refusal(str(error)) from error


def _live(context):
    session = _engine(context, "live")
    if not context["own"]:
        raise Unresolved("the live run belongs to another session than the address origin")
    return session.live_occurrence(_checkpoint(context))


def _blob(context):
    project, pointer, recipe = context["project"], context["pointer"], context["recipe"]
    layers = _layers(context["session"], recipe)
    if context["sid"] is None or layers is None:
        raise Unresolved("no session or layer count to look the occurrence up under")
    row = index.find_occurrence(project, context["sid"], pointer["address"], layers=layers)
    if row is None or row["blob_id"] is None:
        raise Unresolved("no stored blob is indexed for this occurrence")
    for column, mine in (("recipe_id", pointer["recipe"]), ("cut_id", pointer["cut"])):
        if mine is not None and row[column] is not None and row[column] != mine:
            raise Unresolved(f"the stored blob was recorded under another {column}")
    try:
        if not project.store.has_tensor(row["blob_id"]):
            raise Unresolved(f"blob {row['blob_id']} is missing from the store")
        value = project.store.get_tensor(row["blob_id"], framework="torch")
    except ValueError as error:
        raise _Refusal(f"blob {row['blob_id']} is corrupt: {error}") from error
    return value, pointer.get("authority", "measured")


def _cut(context):
    session = _engine(context, "cut")
    cut_id = context["pointer"]["cut"]
    if cut_id is None:
        raise Unresolved("the pointer records no cut")
    project = context["project"]
    holders = [s["session_id"] for s in project.sessions()]
    for sid in [context["sid"], *holders]:
        if sid is not None and cut_id in project.cut_ids(sid):
            return session.compute_occurrence(_checkpoint(context), context["recipe"],
                                              project.store, root=project.load_cut(sid, cut_id))
    raise Unresolved(f"cut {cut_id[:12]} is not retained in the project")


def _replay(context):
    session = _engine(context, "replay")
    return session.compute_occurrence(_checkpoint(context), context["recipe"],
                                      context["project"].store, root=None)


_TIER = {"live": _live, "blob": _blob, "cut": _cut, "replay": _replay}

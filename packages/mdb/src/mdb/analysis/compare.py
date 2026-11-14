"""Native measurement of a consumer under a set of edits, and the comparison of two branches.

``measure(run, edits, consumer)`` stages every edit at its layer (rewinding by layer
boundaries first when an edited layer is already committed), runs natively to the consumer,
reads it and restores the session; ``state_digest`` is checked before and after. It is the
multi-edit form of the measurement step of ``propagate.whatif`` and refuses what that refuses
(an edit that is not a layer port, a consumer that is not downstream, a rewind that would drop
staged or later committed edits).

State-cell edits (``Edit.cell``) are staged where ``propagate.cell_plan`` puts them (at the
cursor when the consumer is ahead, else at the entry of the consumer row's cell layer); a set
that would rewind past a cell staged at the cursor is refused. Two cell edits conflict in A+B
when they write overlapping elements of one cache tensor with ops that do not commute (``mix``
never commutes).

``compare(run, a, b, consumer)``: a side is a list of edits (``propagate.Edit``; applied on the
current session) or a checkpoint (an in-memory ``Run.checkpoint()`` dict, or a ``Run``
replayed from a durable recipe). The consumer is measured at ``base`` (the current session), at
``A``, at ``B`` and, when both sides are edits, at ``A+B`` (both sets applied together); each is
``measured``. The effects ``A - base``, ``B - base`` and the interaction ``(A+B) - A - B + base``
are arithmetic on measured values (``measured``). For a vector consumer the effects are
reported as max-abs and norm of the differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .finite import MEASURED, f64
from .forecast import Logit, Margin, Refusal
from .propagate import Edit, plan, stage, state_digest
from .reads import claim
from .verdict import decide, magnitude


def _guard(run: Any, checkpoints: list[dict[str, Any]]) -> None:
    """Hold every edit tensor the checkpoints name (so ``restore`` can rebuild them); refuse,
    before anything native runs, a checkpoint naming a tensor this run does not hold."""
    held = run.edit_tensors()
    for checkpoint in checkpoints:
        manifests = [*checkpoint["edits"], *(m for item in checkpoint["path_edits"]
                                             for m in item["edits"])]
        for manifest in manifests:
            needed = [manifest["value"].get("sha256"), manifest.get("mask_sha256")]
            missing = [sha for sha in needed if sha is not None and sha not in held]
            if missing:
                raise Refusal(f"the session cannot be restored: edit tensor {missing[0][:12]} at "
                              f"{manifest['port']} is not held by this run")
    run.hold(held)


_SAME = ("where", "cut", "transitions", "port", "edits", "path_edits")


def restore_exactly(run: Any, checkpoint: dict[str, Any]) -> None:
    """``run.restore(checkpoint)`` that finishes even under a ``KeyboardInterrupt`` (re-raised
    after it), then checks that the cursor, the cut and the committed and staged edits equal the
    checkpoint's; ``RuntimeError`` if not."""
    try:
        run.restore(checkpoint, defer_interrupt=True)
    finally:
        now = run.checkpoint()
        differ = [key for key in _SAME if now[key] != checkpoint[key]]
        if differ:
            raise RuntimeError(f"the restore to {checkpoint['where']} left {', '.join(differ)} "
                               f"different (now at {run.where})")


def consumer_stop(run: Any, consumer: Any) -> str:
    """The stop a consumer is read at: the row's ``logits`` for a margin or logit."""
    if isinstance(consumer, (Margin, Logit)):
        return f"t{consumer.row}.logits"
    key = run.parse(consumer)
    label = run.stops()[key[1]]
    if label in ("start", "sample"):
        raise Refusal(f"{label} is a token id, not a value with a change; use a margin M[a,b] "
                      "or a logit as the consumer")
    return f"t{key[0]}.{label}"


def consumer_value(consumer: Any, value: Any) -> Any:
    """The consumer's float64 value from the tensor read at its stop (index, then cast)."""
    flat = value.reshape(-1)
    if isinstance(consumer, Margin):
        return f64(flat[consumer.a]) - f64(flat[consumer.b])
    if isinstance(consumer, Logit):
        return f64(flat[consumer.i])
    return f64(flat)


def _read(run: Any, consumer: Any) -> tuple[Any, str, tuple[float, Any]]:
    """The consumer's float64 value, its label, and ``(magnitude, dtype)`` of the native tensor
    it came from (the noise floor of ``verdict.decide``)."""
    stop = consumer_stop(run, consumer)
    if run.parse(stop) > run.parse(run.where):
        run.until(stop)
    info = run.read_info(stop)
    native = run.read(stop)
    return (consumer_value(consumer, native), info["authority"] or MEASURED,
            (magnitude(consumer, native), native.dtype))


def _layer_key(run: Any, edit: Edit) -> tuple[int, int]:
    row, index = run.parse(edit.address)
    layer = int(run.stops()[index].split(".", 1)[0][1:])
    return row, layer


def _stage_key(run: Any, edit: Edit, p: Any) -> tuple[int, int]:
    """The order key of where an edit is staged: a cell at its plan's ``stage_at``, a port edit
    in its layer (keyed by the layer's entry; a pending layer is staged at the cursor)."""
    if edit.cell is not None:
        return run.parse(p.stage_at)
    row, layer = _layer_key(run, edit)
    return run.parse(f"t{row}.L{layer}.h")


def measure(run: Any, edits: list[Edit], consumer: Any) -> dict[str, Any]:
    """The consumer measured natively with every edit in ``edits`` applied (none: the base);
    the session is restored and checked unchanged. Returns ``value`` (float64), ``authority``,
    ``moves``."""
    plans = [plan(run, edit, consumer) for edit in edits]
    consumer_stop(run, consumer)
    keys = [_stage_key(run, edit, p) for edit, p in zip(edits, plans)]
    rewind = any(p.mode == "rewind" for p in plans)
    if rewind and any(e.cell is not None and p.mode == "pending" for e, p in zip(edits, plans)):
        raise Refusal("this set mixes a cell edit staged at the cursor with an edit that needs "
                      "stepping back: the cell would be staged earlier than at the cursor")
    before = state_digest(run)
    checkpoint = run.checkpoint()
    _guard(run, [checkpoint])
    moves: list[str] = []
    try:
        if rewind:
            entry = min(keys)
            while run.parse(run.where) >= entry:
                run.reverse_step("layer")
            moves.append(f"reverse-step layer to {run.where}")
        order = sorted(range(len(edits)), key=lambda i: keys[i])
        for i in order:
            edit = edits[i]
            if edit.cell is not None:
                if run.parse(run.where) != keys[i]:
                    run.until(run.address(keys[i]))
                    moves.append(f"until {run.where}")
                stage(run, edit)
                moves.append(f"set {edit.address}")
                continue
            row, layer = _layer_key(run, edit)
            status = run.status()
            here = run.parse(run.where)
            pending = (status["port"] is not None and status["pending_layer"] == layer
                       and here[0] == row)
            if not pending:
                run.until(f"t{row}.L{layer}.h")
                moves.append(f"until t{row}.L{layer}.h")
            run.set(edit.address, edit.value, edit.op, mask=edit.mask, selection=edit.selection,
                    origin=edit.origin)
            moves.append(f"set {edit.address}" + (f"[{edit.selection}]" if edit.selection
                                                  else ""))
        value, authority, scale = _read(run, consumer)
        moves.append(f"read {consumer_stop(run, consumer)}")
    finally:
        restore_exactly(run, checkpoint)
    moves.append(f"restore {checkpoint['where']}")
    after = state_digest(run)
    if after != before:
        raise RuntimeError(f"measure did not leave the session where it was ({before[:12]} -> "
                           f"{after[:12]})")
    return {"value": value, "authority": authority, "moves": moves,
            "scale": scale}


def measure_at(run: Any, checkpoint: dict[str, Any], consumer: Any) -> dict[str, Any]:
    """The consumer measured natively from an in-memory checkpoint of this run; the session is
    restored and checked unchanged."""
    before = state_digest(run)
    here = run.checkpoint()
    _guard(run, [here, checkpoint])
    try:
        run.restore(checkpoint)
        value, authority, scale = _read(run, consumer)
    finally:
        restore_exactly(run, here)
    if state_digest(run) != before:
        raise RuntimeError("measure_at did not leave the session where it was")
    return {"value": value, "authority": authority, "scale": scale,
            "moves": [f"restore {checkpoint['where']}", f"read {consumer_stop(run, consumer)}",
                      f"restore {here['where']}"]}


def measure_other(other: Any, consumer: Any) -> dict[str, Any]:
    """The consumer measured natively on another run (a durable checkpoint replayed)."""
    value, authority, scale = _read(other, consumer)
    return {"value": value, "authority": authority, "scale": scale,
            "moves": [f"replay to {other.where}", f"read {consumer_stop(other, consumer)}"]}


@dataclass
class Side:
    """One branch of a comparison: ``edits`` (a list of ``Edit``), ``checkpoint`` (an
    in-memory checkpoint of this run) or ``run`` (another run, e.g. a durable checkpoint
    replayed); ``text`` names it."""

    text: str
    edits: list[Edit] | None = None
    checkpoint: dict[str, Any] | None = None
    run: Any = None


def _side(run: Any, side: Side, consumer: Any) -> dict[str, Any]:
    if side.edits is not None:
        return measure(run, side.edits, consumer)
    if side.checkpoint is not None:
        return measure_at(run, side.checkpoint, consumer)
    return measure_other(side.run, consumer)


def _commute(x: Edit, y: Edit) -> bool:
    """Whether applying ``x`` then ``y`` at one port equals ``y`` then ``x`` (real arithmetic):
    two adds, two scales, or two replaces with the same scalar value (a ``mix`` never)."""
    if x.op == y.op and x.op in ("add", "scale"):
        return True
    return (x.op == y.op == "replace" and not hasattr(x.value, "shape")
            and not hasattr(y.value, "shape") and float(x.value) == float(y.value))


def _cell_overlap(x: Edit, y: Edit) -> bool:
    """Whether two cell edits write a common element of one cache tensor."""
    a, b = x.cell, y.cell
    if (a["layer"], a["tensor"]) != (b["layer"], b["tensor"]):
        return False
    return all(a.get(key) is None or b.get(key) is None or set(a[key]) & set(b[key])
               for key in ("rows", "kv", "d"))


def _overlap(x: Edit, y: Edit) -> bool:
    if x.cell is not None or y.cell is not None:
        return x.cell is not None and y.cell is not None and _cell_overlap(x, y)
    if x.mask is None or y.mask is None:
        return True
    import torch

    return bool(torch.logical_and(x.mask, y.mask).any())


def _order_conflict(a: list[Edit], b: list[Edit]) -> str | None:
    """Why A+B is order-dependent (same port, overlapping masks, ops that do not commute)."""
    for x in a:
        for y in b:
            same = (x.cell is not None and y.cell is not None) or x.address == y.address
            if same and _overlap(x, y) and not _commute(x, y):
                return (f"A and B both edit {x.address} on overlapping elements with ops that "
                        f"do not commute ({x.op}, {y.op}): A+B depends on the order")
    return None


def _effect(value: Any) -> dict[str, Any]:
    return claim(value, MEASURED)


def compare(run: Any, a: Side, b: Side, consumer: Any) -> dict[str, Any]:
    """Base, A, B and (both sides edits) A+B, each measured natively; effects and the
    interaction ``(A+B) - A - B + base``. Returns ``report`` (bounded JSON) and ``values``
    (float64 tensors)."""
    before = state_digest(run)
    base = measure(run, [], consumer)
    found = {"base": base, "A": _side(run, a, consumer), "B": _side(run, b, consumer)}
    joint: str | None = None
    if a.edits is not None and b.edits is not None:
        joint = _order_conflict(a.edits, b.edits)
        if joint is None:
            found["A+B"] = measure(run, [*a.edits, *b.edits], consumer)
    else:
        joint = ("A+B needs both sides to be edits (a checkpoint is a whole session state, not "
                 "a change that composes)")
    after = state_digest(run)
    if after != before:
        raise RuntimeError("compare did not leave the session where it was")
    values = {name: item["value"] for name, item in found.items()}
    vector = values["base"].dim() > 0
    report: dict[str, Any] = {
        "sides": {"A": a.text, "B": b.text},
        "measured": {name: {"value": claim(item["value"], item["authority"]),
                            "moves": item["moves"]} for name, item in found.items()},
        "effects": {"A": _effect(values["A"] - values["base"]),
                    "B": _effect(values["B"] - values["base"])},
        "session": {"before": before, "after": after, "unchanged": True},
    }
    if joint is None:
        report["order"] = ("A+B stages both sides' edits layer by layer; at one port A's edit "
                           "is applied before B's (where both edit one port, the ops commute)")
        interaction = values["A+B"] - values["A"] - values["B"] + values["base"]
        report["effects"]["A+B"] = _effect(values["A+B"] - values["base"])
        report["interaction"] = {"value": _effect(interaction),
                                 "formula": "(A+B) - A - B + base"}
        values["interaction"] = interaction
    else:
        report["interaction"] = {"state": "refused", "reason": joint}
        if a.edits is not None and b.edits is not None:
            found_joint = {"state": "refused", "reason": joint}
            report["measured"]["A+B"] = found_joint
            report["effects"]["A+B"] = found_joint
    # each effect against the dtype-spacing floor (native only: compare has no exact chain)
    report["verdicts"] = {}
    for name in [n for n in ("A", "B", "A+B") if n in values]:
        size = max(found[name]["scale"][0], found["base"]["scale"][0])
        report["verdicts"][name] = decide(values[name] - values["base"], None, None, size,
                                          found["base"]["scale"][1],
                                          "compare measures natively, with no exact chain")
    report["kind"] = "vector" if vector else "scalar"
    return {"report": report, "values": values}


__all__ = ["Side", "compare", "consumer_stop", "consumer_value", "measure", "measure_at",
           "measure_other", "restore_exactly"]

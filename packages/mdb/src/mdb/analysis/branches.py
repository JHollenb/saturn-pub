"""Branches: named native branches from one checkpoint, all measured at one consumer against one
floor (gap audit 2026-11-14, P1).

Eight experiment drivers wrote the same loop by hand: checkpoint, edit, run to the logits, read
the margin, restore, keyed by an arm label, with a matched-norm null drawn differently in four of
them and the floor attached by hand in every one. ``Branches`` is that loop as a debugger object:

    br = Branches(run, Margin(last, a, b), at=f"t{last}.L20.h")
    br.add("R1", [edit(f"t{last}.L20.v[kv6]", donor.ref(f"t{last}.L20.v[kv6]"))])
    br.add("R1-held", br.arm("R1").edits, holds=[f"t@last.L>20.p"])
    br.add_null("random", like="R1", seed=0)      # matched-norm random change, same address
    br.add_null("self", like="R1")                # self-patch: the native value back (dM = 0)
    br.add_null("answer", like="R1")              # matched norm along the answer direction
    br.run()                                      # stepped (mdb-stepped kernel path)
    br.table(); br.compare("R1", "R1-held"); br.seal(); br.restore()

Semantics (a debugger's): ``branch`` (the checkpoint taken at ``at``), ``add``/``add_null``
(named arms), ``run`` (each arm: restore, holds on, edits staged at their layers, run to the
consumer's logits, read, holds released, restore), ``compare`` (two arms' difference, in floors),
``restore`` (the run back at the branch point). Every arm records: its edits' manifests (origins
included: a ``ValueRef`` from another run or a ``Pool`` keeps its pointer), its holds, ``M``,
``dM = M - M0``, the argmax, ``|dM|`` in floors with the stability band (``stability.stability``:
stable above ``k`` floors, else kernel-sensitive; ``k`` from a calibration, the caller, or 4) and
the kernel path it was measured on.

The floor is ONE number for every arm: ``floor_spacings`` (2, the showcase wording) x the dtype
spacing at the base margin's larger ``|logit|`` (``verdict.margin_floor``) plus ``extra_floor``
(a measured kernel-path delta), or the caller's ``floor``. A callable consumer gives no
magnitude: pass ``floor`` or the arms carry no floor columns.

Built-in nulls (each derived from an arm ``like`` and recorded with its rule):

* ``random``: at every address the ``like`` arm edits, an ``add`` of a seeded Gaussian over the
  edited elements, rescaled to the norm of the change the arm makes there (``|requested -
  native|``, float64 on the reference run's native value), cast once to the model dtype.
* ``self``: every edited element replaced by its own native value (the self-patch): ``dM`` must
  be 0 on the stepped path; a nonzero ``dM`` measures the kernel path, not the model.
* ``answer``: residual-stream ports only (``h``, ``o``, ``r``, ``m``, ``h_out``): the change's norm
  written along the consumer's readout direction (``reads.readout(...).vector``, the
  answer-minus-foil unembedding through the final norm), signed like the change's own projection:
  the direct-effect ceiling of a write of that size (the answer-token null).

``batched`` measures arms in one batched forward per chunk (``sweep.cell_sweep``; kernel path
``batched``, with its measured null-vs-stepped delta added to the floor of those arms; twice the
delta: a margin is a difference of two logits): state-cell arms (``replace`` of whole cells) and
port arms at the pending row's native-module ports (``a q k v n g sg u f A o m``; any op, selector
or mask, applied per arm by a forward hook with the stepped run's own arithmetic). A batched
forward is a different kernel path from the stepped one, and the audited showcase found that
this only changes verdicts for effects within about 4 floors (0/214 above), so the default
``recheck="band"`` re-measures stepped only the arms that could flip: ``1/k <= |dM| / floor <= k``
(``k`` = the stability margin, 4 unless calibrated); arms above ``k`` floors (stable) or below
``1/k`` (clearly null) keep their batched row, which says why (``recheck.why``). ``recheck=0``
none, ``"all"`` every arm, an ``int`` the ``k`` arms of largest ``|dM|`` (the exp5 pattern).

``probes`` (amplitude programme 2026-11-14): named callables ``probe(run) -> number | list`` read
on every stepped arm (and the base) after the consumer's logits are computed and before the
restore, so an arm can report a value OTHER than the consumer (a cache cell's projection, a
port's norm) from the branch it ran; stored per row as ``probes`` (measured). Batched arms carry
no probes (their forward is not a ``Run``).

``interchange`` (subspace.py) builds on this object.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from mdb.core.kinds import BRANCHES, procedure  # noqa: F401 (declares the kind)
from mdb.core.records import seal_record

from .finite import EXACT, MEASURED, SUPPLIED, f64, num
from .forecast import Logit, Margin, Refusal, run_session_id
from .stability import MDB_STEPPED, stability
from .verdict import magnitude, margin_floor, spacing

SCHEMA = "mdb-branches-v1"
NULL_KINDS = ("random", "self", "answer")
RESIDUAL_PORTS = ("h", "o", "r", "m", "h_out")
_CAP = 64


def _torch():
    import torch

    return torch


@dataclass
class Edit:
    """One edit of an arm: ``target`` (an address, selector allowed; ``cache.L<l>.k|v[...]`` for
    a state cell), ``value`` (number, tensor or ``engine.refs.ValueRef``), ``op``, an optional
    bool ``mask`` (ports only), ``dose`` (cell ``mix``) and ``origin`` (overrides a ref's)."""

    target: str
    value: Any
    op: str = "replace"
    mask: Any = None
    dose: float | None = None
    origin: dict[str, Any] | None = None

    @property
    def cell(self) -> bool:
        return self.target.strip().startswith("cache.")


def edit(target: str, value: Any, op: str = "replace", **options: Any) -> Edit:
    return Edit(target, value, op, **options)


@dataclass
class Arm:
    name: str
    kind: str = "edit"
    edits: list[Edit] = field(default_factory=list)
    holds: list[Any] = field(default_factory=list)   # "ADDR" (source: reference) or (ADDR, run)
    final_scale: str = "native"                      # "held": final RMSNorm scale held too
    like: str | None = None
    seed: int | None = None
    rule: str | None = None
    result: dict[str, Any] | None = None


def _safe(value: Any, authority: str) -> Any:
    """A manifest made record-safe: floats (and negative ints) labelled, long lists capped."""
    import numbers

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value) if value >= 0 else num(float(value), authority)
    if isinstance(value, numbers.Real):
        return num(float(value), authority)
    if isinstance(value, dict):
        if set(value) >= {"value", "authority"} and isinstance(value.get("authority"), str):
            return value
        return {str(k): _safe(v, authority) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v, authority) for v in list(value)[:_CAP]]
    return str(value)


_KEEP = ("port", "op", "value", "selection", "mask_count", "origin", "row", "layer",
         "realised_exact", "realised_max_abs_error", "cell")


def _keep(entry: dict[str, Any]) -> dict[str, Any]:
    """The fields of a committed edit/cell record a branches record keeps."""
    entry = {k: v for k, v in entry.items() if k in _KEEP}
    if isinstance(entry.get("realised_max_abs_error"), float):
        entry["realised_max_abs_error"] = num(entry["realised_max_abs_error"], MEASURED)
    return entry


class Branches:
    """Named native branches of ``run`` from one checkpoint (module docstring)."""

    def __init__(self, run: Any, consumer: Any, *, at: str | None = None,
                 name: str = "branches", row: int | None = None, floor: float | None = None,
                 floor_spacings: float = 2.0, extra_floor: float = 0.0,
                 calibration: Any = None, stable_floors: float | None = None,
                 kernel_path: str = MDB_STEPPED, reference: Any = None,
                 probes: dict[str, Any] | None = None):
        self.debuggee, self.consumer, self.name = run, consumer, name
        self.probes = dict(probes or {})
        self.row = consumer.row if hasattr(consumer, "row") else row
        if self.row is None:
            raise Refusal("a callable consumer needs row= (the logits row it reads)")
        if at is not None:
            here, there = run.parse(run.where), run.parse(at)
            if there > here:
                run.until(at)
            elif there < here:
                run.rewind(at)
        self.at = run.where
        self.checkpoint = run.checkpoint(f"{name}:at")
        # edits and cells committed on the path BEFORE the branch point (an arm's own edits are
        # those fired from here on): recorded once under ``at``, never in an arm row
        self.path = {"edits": [_keep(e) for e in run.committed_edits()],
                     "cells": [_keep(c) for c in run.committed_cells()]}
        self.calibration, self.stable_floors = calibration, stable_floors
        self.kernel_path = kernel_path
        self.arms: dict[str, Arm] = {}
        self._reference = reference
        started = time.monotonic()
        logits = self._to_consumer()
        self.base_logits = logits.detach().clone()
        self.M0 = self._value(logits)
        self.base_argmax = int(logits.reshape(-1).argmax())
        self.base_probes = self._probe()
        self.base_seconds = time.monotonic() - started
        run.restore(self.checkpoint)
        self.extra_floor = abs(float(extra_floor))
        self.floor_spacings = float(floor_spacings)
        if floor is not None:
            self.floor, self.floor_rule = float(floor), "caller"
        elif isinstance(consumer, Margin):
            self.floor = margin_floor(logits, consumer.a, consumer.b, spacings=floor_spacings,
                                      extra=extra_floor)
            self.floor_rule = (f"{floor_spacings:g} x {str(logits.dtype).removeprefix('torch.')} "
                               "spacing at the larger |logit| of the base margin + extra")
        elif isinstance(consumer, Logit):
            size = abs(float(logits.reshape(-1)[consumer.i]))
            self.floor = floor_spacings * spacing(size, logits.dtype) + self.extra_floor
            self.floor_rule = f"{floor_spacings:g} x spacing at the base logit + extra"
        else:
            self.floor, self.floor_rule = None, "none (callable consumer, no floor given)"

    # --- the consumer ---------------------------------------------------------------------

    @property
    def text(self) -> str:
        return getattr(self.consumer, "text", f"t{self.row}.<callable>")

    def _value(self, logits: Any) -> float:
        lg = f64(logits).reshape(-1)
        if isinstance(self.consumer, Margin):
            return float(lg[self.consumer.a] - lg[self.consumer.b])
        if isinstance(self.consumer, Logit):
            return float(lg[self.consumer.i])
        return float(self.consumer(lg))

    def _probe(self) -> dict[str, Any]:
        """Every probe evaluated on the debuggee where it stands (after the consumer)."""
        out = {}
        for key, probe in self.probes.items():
            value = probe(self.debuggee)
            if isinstance(value, (list, tuple)):
                out[key] = [num(float(v), MEASURED) for v in value]
            else:
                out[key] = num(float(value), MEASURED)
        return out

    def _to_consumer(self, final_scale: str = "native") -> Any:
        stop = f"t{self.row}.logits"
        self.debuggee.until(stop)
        if final_scale == "held":
            from mdb.engine.holds import final_held_logits

            return final_held_logits(self.debuggee, self.row, self.reference)
        return self.debuggee.read(stop)

    @property
    def reference(self) -> Any:
        """The base continuation as its own run: a run at the root of the debuggee's lineage
        (``Run.fresh``: the same prompt and split, or the checkpoint it was rehydrated from)
        restored to the branch point (edits on the path included) and run to the consumer with
        no arm edit. Holds keep values from it; the nulls read native values from it."""
        if self._reference is None:
            run = self.debuggee
            # the root of the debuggee's lineage: its prompt, or the checkpoint it was
            # rehydrated from (no prefix recompute; the branch point restores from its marks)
            ref = run.fresh()
            ref.name = f"{self.name}:reference"
            ref.hold(run.edit_tensors())
            ref.restore(self.checkpoint)
            # the debuggee's active holds are part of its base: the reference runs under them
            # too, so ``reference == M0`` and the self-patch null reads 0 on a held run
            for held in run.hold_objects():
                if not held.cells:
                    ref.hold(held.text, source=held.source, name=held.name)
            ref.until(f"t{self.row}.logits")
            self._reference = ref
        return self._reference

    # --- arms -----------------------------------------------------------------------------

    def _shaped(self, e: Edit) -> Edit:
        """The edit with a cache-cell value in the block shape ``(1, kv, rows, d)`` the stepped
        ``Run.set`` wants, when it holds exactly that many numbers in another shape (a flat
        ``(d,)`` for one kv head and one row): normalised once here so the batched path and the
        stepped recheck accept the same arms (EX1)."""
        if not e.cell:
            return e
        from dataclasses import replace

        from mdb.engine import refs

        torch = _torch()
        found = refs.resolve(self.debuggee, e.target)
        kwargs = refs.cell_edit(self.debuggee, found, 0.0, None)
        adapter = self.debuggee.adapter
        block = (1, adapter.num_kv_heads if kwargs["kv"] is None else len(kwargs["kv"]),
                 len(kwargs["rows"]), adapter.head_dim if kwargs["d"] is None else len(kwargs["d"]))
        ref = e.value if isinstance(e.value, refs.ValueRef) else None
        value = ref.value if ref is not None else e.value
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) == block
                or value.numel() != block[0] * block[1] * block[2] * block[3]):
            return e
        value = value.reshape(block)
        if ref is not None:
            value = refs.ValueRef(value, {**ref.pointer, "shape": list(block)})
        return replace(e, value=value)

    def add(self, name: str, edits: Sequence[Edit] = (), *, holds: Sequence[Any] = (),
            final_scale: str = "native") -> Arm:
        """A named arm: ``edits`` staged from the branch point, ``holds`` (addresses held at the
        reference run, or ``(address, run)``) on while it runs, ``final_scale`` ``held`` to read
        the logits with the final RMSNorm scale held too."""
        if name in self.arms or name == "base":
            raise Refusal(f"an arm named {name!r} exists")
        if final_scale not in ("native", "held"):
            raise Refusal("final_scale is native or held")
        arm = Arm(name, "edit",
                  [self._shaped(e if isinstance(e, Edit) else Edit(*e)) for e in edits],
                  list(holds), final_scale)
        self.arms[name] = arm
        return arm

    def arm(self, name: str) -> Arm:
        if name not in self.arms:
            raise Refusal(f"no arm {name!r} (arms: {', '.join(self.arms) or 'none'})")
        return self.arms[name]

    def add_null(self, name: str, kind: str = "random", *, like: str, seed: int = 0,
                 holds: Sequence[Any] | None = None) -> Arm:
        """A built-in null derived from arm ``like`` (module docstring); it keeps ``like``'s
        holds and final scale unless ``holds`` is given."""
        if kind not in NULL_KINDS:
            raise Refusal(f"null kind must be one of {', '.join(NULL_KINDS)}")
        base = self.arm(like)
        if name in self.arms or name == "base":
            raise Refusal(f"an arm named {name!r} exists")
        edits = [self._null(kind, e, like, seed + i) for i, e in enumerate(base.edits)]
        arm = self.add(name, edits, holds=list(base.holds if holds is None else holds),
                       final_scale=base.final_scale)
        arm.kind, arm.like, arm.seed = f"null:{kind}", like, seed
        arm.rule = {"random": "matched-norm Gaussian add over the edited elements",
                    "self": "edited elements replaced by their native value (self-patch)",
                    "answer": "matched-norm add along the consumer readout direction"}[kind]
        return arm

    def _change(self, e: Edit) -> dict[str, Any]:
        """``{native, delta, mask, target, port}`` of an edit on the reference run (float64):
        the change the edit makes at its address."""
        from mdb.engine import refs

        torch = _torch()
        ref = self.reference
        value = e.value.value if isinstance(e.value, refs.ValueRef) else e.value
        found = refs.resolve(ref, e.target)
        if found["kind"] == "cell":
            raw = refs.ref(ref, e.target).value
            native = f64(raw).cpu()
            requested = f64(value).cpu() if isinstance(value, torch.Tensor) else \
                torch.full_like(native, float(value))
            delta = {"replace": lambda: requested - native, "add": lambda: requested,
                     "scale": lambda: native * (requested - 1.0),
                     "mix": lambda: float(e.dose or 0.0) * (requested - native)}[e.op]()
            return {"native": native, "raw": raw, "delta": delta, "mask": None, "cell": True,
                    "target": found["address"], "port": None}
        base = found["address"].partition("[")[0]
        port = found["label"].split(".", 1)[1]
        shape = ref.address_shape(base)
        raw = ref.read(base)
        native = f64(raw).cpu()
        request = refs.port_edit(ref, found, value, None, e.mask, shape=shape)
        mask = request["mask"]
        full = request["value"]
        requested = (f64(full).cpu().expand(shape) if isinstance(full, torch.Tensor)
                     else torch.full(shape, float(full), dtype=torch.float64))
        result = {"replace": requested, "add": native + requested,
                  "scale": native * requested}[e.op]
        delta = result - native
        if mask is not None:
            delta = torch.where(mask.cpu().broadcast_to(shape), delta, torch.zeros_like(delta))
        return {"native": native, "raw": raw, "delta": delta, "mask": mask, "cell": False,
                "target": base,
                "port": port}

    def _null(self, kind: str, e: Edit, like: str, seed: int) -> Edit:
        torch = _torch()
        change = self._change(e)
        delta, mask = change["delta"], change["mask"]
        size = float(delta.norm())
        dtype = self.debuggee.adapter.dtype
        origin = {"kind": "null", "null": kind, "like": like, "seed": seed,
                  "matched_norm": num(size, EXACT), "address": change["target"]}
        if kind == "self":
            value = change["raw"]               # the native value as read: no cast at all
            return Edit(e.target if change["cell"] else change["target"], value, "replace",
                        mask=None if change["cell"] else mask, origin=origin)
        if kind == "random":
            g = torch.randn(delta.shape, generator=torch.Generator().manual_seed(seed),
                            dtype=torch.float64)
            if mask is not None:
                g = torch.where(mask.cpu().broadcast_to(g.shape), g, torch.zeros_like(g))
            g = g * (size / float(g.norm())) if float(g.norm()) else g
            return Edit(e.target if change["cell"] else change["target"], g.to(dtype), "add",
                        mask=None if change["cell"] else mask, origin=origin)
        if change["cell"] or change["port"] not in RESIDUAL_PORTS:
            raise Refusal(f"an answer null writes along the readout direction: residual-stream "
                          f"ports only ({', '.join(RESIDUAL_PORTS)}), not {e.target}")
        from .reads import readout

        c = f64(readout(self.reference, self.consumer).vector).reshape(-1).cpu()
        u = c / float(c.norm())
        flat = delta.reshape(-1, delta.shape[-1])
        rows = []
        for vec in flat:
            n = float(vec.norm())
            sign = 1.0 if float(vec @ u) >= 0 else -1.0
            rows.append(sign * n * u)
        g = torch.stack(rows).reshape(delta.shape)
        origin["direction"] = "readout vector (reads.readout(...).vector), signed like the change"
        return Edit(change["target"], g.to(dtype), "add", origin=origin)

    # --- measuring --------------------------------------------------------------------------

    def _stage(self, e: Edit) -> None:
        run = self.debuggee
        value, origin = e.value, e.origin
        if e.cell:
            run.set(e.target, value, e.op, origin=origin) if e.dose is None else \
                self._mix(e)
            return
        from mdb.engine import refs

        found = refs.resolve(run, e.target)
        row, label = found["row"], found["label"]
        first = f"t{row}.{label.split('.')[0]}.{run.program.names[0]}"
        here, there = run.parse(run.where), run.parse(first)
        own = run.parse(found["address"].partition("[")[0])
        if own < run.parse(self.at):
            raise Refusal(f"{e.target} is before the branch point {self.at}")
        if there > here:
            run.until(first)
        run.set(e.target, value, e.op, mask=e.mask, origin=origin)

    def _mix(self, e: Edit) -> None:
        from mdb.engine import refs

        found = refs.resolve(self.debuggee, e.target)
        kwargs = refs.cell_edit(self.debuggee, found, e.value, e.origin)
        value = kwargs.pop("value")
        self.debuggee.set_cell(kwargs.pop("layer"), kwargs.pop("tensor"), value, "mix", dose=e.dose,
                          **kwargs)

    def _measure(self, arm: Arm) -> dict[str, Any]:
        run = self.debuggee
        started = time.monotonic()
        run.restore(self.checkpoint)
        names = []
        try:
            for i, held in enumerate(arm.holds):
                address, source = (held, None) if isinstance(held, str) else held
                hold_name = f"{arm.name}:{i}"
                run.hold(address, source=source if source is not None else self.reference,
                         name=hold_name)
                names.append(hold_name)
            cells_first = sorted(arm.edits, key=lambda e: not e.cell)
            from mdb.engine import refs

            def order(e: Edit) -> Any:
                return run.parse(refs.resolve(run, e.target)["address"].partition("[")[0])

            ports = sorted((e for e in cells_first if not e.cell), key=order)
            for e in [*(c for c in cells_first if c.cell), *ports]:
                self._stage(e)
            logits = self._to_consumer(arm.final_scale)
            probed = self._probe()
            start = int(self.checkpoint["transitions"])
            committed = [c for c in run.committed_edits() if c["transition"] >= start]
            hold_edits = [c for c in committed if c.get("origin", {}).get("kind") == "hold"]
            user = [c for c in committed if c.get("origin", {}).get("kind") != "hold"]
            cells = [c for c in run.committed_cells() if c["transition"] >= start]
            holds = [h for h in run.holds if h["name"] in names]
        finally:
            for hold_name in names:
                if any(h["name"] == hold_name for h in run.holds):
                    run.release(hold_name)
            run.restore(self.checkpoint)
        row = self._row(arm, logits, user, cells, holds, len(hold_edits),
                        time.monotonic() - started, self.kernel_path)
        if self.probes:
            row["probes"] = probed
        return row

    def _row(self, arm: Arm, logits: Any, edits: list, cells: list, holds: list, fired: int,
             seconds: float, kernel_path: str, extra: float = 0.0) -> dict[str, Any]:
        M = self._value(logits)
        dM = M - self.M0
        edits = [_keep(e) for e in edits]
        cells = [_keep(c) for c in cells]
        out: dict[str, Any] = {
            "name": arm.name, "kind": arm.kind, "like": arm.like, "rule": arm.rule,
            "M": num(M, MEASURED), "dM": num(dM, MEASURED),
            "argmax": int(logits.reshape(-1).argmax()),
            "edits": _safe(edits, SUPPLIED),
            "cells": _safe(cells, SUPPLIED),
            "holds": _safe([{**h, "edits_fired": fired} for h in holds], EXACT),
            "final_scale": arm.final_scale, "kernel_path": kernel_path,
            "seconds": num(seconds, MEASURED)}
        if isinstance(self.consumer, (Margin, Logit)):
            # the consumer's |logit| size on this arm: a floor over an item's arms (exp6's rule)
            out["magnitude"] = num(magnitude(self.consumer, logits), MEASURED)
        if arm.final_scale == "held":
            out["final_scale_authority"] = EXACT
        if self.floor is not None:
            floor = self.floor + extra
            stab = stability(dM, floor, kernel_path=kernel_path, calibration=self.calibration,
                             k=self.stable_floors)
            out["floor"] = num(floor, MEASURED)
            out["effect_floors"] = stab.pop("floors", None)   # None: a zero floor
            out["above_floor"] = abs(dM) > floor
            out.update({k: v for k, v in stab.items() if k != "kernel_path"})
        return out

    def run_arms(self, names: Sequence[str] | None = None, *, again: bool = False
                 ) -> list[dict[str, Any]]:
        """Measure the named arms (default: every arm not measured yet; ``again``: re-measure)
        on the stepped path; returns their rows. The run is back at the branch point after."""
        chosen = list(self.arms) if names is None else list(names)
        rows = []
        for name in chosen:
            arm = self.arm(name)
            if arm.result is None or again:
                arm.result = self._measure(arm)
            rows.append(arm.result)
        return rows

    def run(self, names: Sequence[str] | None = None, *, mode: str = "stepped",
            recheck: int | str | None = "band", again: bool = False, batch: int = 64
            ) -> list[dict[str, Any]]:
        """Measure arms: ``mode="stepped"`` (``run_arms``, the authority; ``recheck`` unused) or
        ``"batched"`` (``batched``, with ``recheck`` stepped re-measurements)."""
        if mode == "stepped":
            return self.run_arms(names, again=again)
        if mode == "batched":
            return self.batched(names, recheck=recheck, batch=batch)
        raise Refusal("mode is stepped or batched")

    def _batched_edit(self, arm: Arm, e: Edit) -> Any:
        """The ``Cell`` or ``PortPatch`` of one batched arm edit (refusing what cannot batch)."""
        from mdb.engine import refs
        from mdb.engine.ports import PortEdit

        from .sweep import Cell, PortPatch

        torch = _torch()
        found = refs.resolve(self.debuggee, e.target)
        if e.cell:
            if e.op != "replace" or e.dose is not None:
                raise Refusal(f"arm {arm.name}: a batched cell arm replaces whole cells, not "
                              f"{e.target} ({e.dose is not None and 'mix' or e.op})")
            keys = dict(found["selector"].keys)
            kv, row = keys.get("kv"), keys.get("row")
            if not isinstance(kv, int) or not isinstance(row, int) or "d" in keys:
                raise Refusal(f"{e.target}: one kv head and one row, all of head_dim")
            value = e.value.value if isinstance(e.value, refs.ValueRef) else e.value
            if not isinstance(value, torch.Tensor):
                raise Refusal(f"arm {arm.name}: batched cell {e.target} needs a tensor "
                              "value, not a number")
            return Cell(found["layer"], found["tensor"], row, kv, value)
        if e.dose is not None:
            raise Refusal(f"arm {arm.name}: a dose applies to cell edits only")
        if found["row"] != self.row:
            raise Refusal(f"arm {arm.name}: batched port edits are at the pending row "
                          f"t{self.row}, not {e.target}")
        layer_text, _, port = found["label"].partition(".")
        base = found["address"].partition("[")[0]
        if self.debuggee.parse(base) < self.debuggee.parse(self.at):
            raise Refusal(f"{e.target} is before the branch point {self.at}")
        request = refs.port_edit(self.debuggee, found, e.value, None, e.mask,
                                 shape=self.reference.address_shape(base))
        mask = request["mask"]
        shape = tuple(self.reference.address_shape(base))
        edit = PortEdit(port, request["value"], e.op,
                        mask=None if mask is None else mask.broadcast_to(shape).contiguous(),
                        selection=request["selection"])
        return PortPatch(int(layer_text[1:]), port, edit)

    def _recheck_plan(self, rows: list[dict[str, Any]], recheck: Any) -> dict[int, str]:
        """``{arm index: why}`` of the arms to re-measure stepped, and (in place) the
        ``recheck`` record of every row (``done``, ``why``)."""
        if recheck in (None, False, 0):
            plan: dict[int, str] = {}
            why = {i: "recheck off" for i in range(len(rows))}
        elif recheck == "all" or recheck is True:
            plan = {i: "recheck='all'" for i in range(len(rows))}
            why = {}
        elif recheck == "band":
            plan, why = {}, {}
            for i, row in enumerate(rows):
                f = (row.get("effect_floors") or {}).get("value")
                k = float(row["k"]["value"]) if "k" in row else None
                if f is None or k is None:
                    plan[i] = "no floor: the band cannot be judged, rechecked"
                elif f > k:
                    why[i] = (f"above the band: {f:.3g} floors > k={k:g} (stable; no kernel-path "
                              "difference measured above k floors)")
                elif f < 1.0 / k:
                    why[i] = f"clearly null: {f:.3g} floors < 1/k={1.0 / k:.3g}; verdict cannot flip"
                else:
                    plan[i] = f"in the kernel-sensitive band: 1/k <= {f:.3g} floors <= k={k:g}"
        elif isinstance(recheck, int) and not isinstance(recheck, bool) and recheck > 0:
            order = sorted(range(len(rows)), key=lambda i: -abs(rows[i]["dM"]["value"]))
            plan = {i: f"recheck={recheck}: among the {recheck} largest |dM|"
                    for i in order[:recheck]}
            why = {i: f"recheck={recheck}: not among the {recheck} largest |dM|"
                   for i in range(len(rows)) if i not in plan}
        else:
            raise Refusal("recheck is 0 (none), 'band', 'all' or an int k (the k largest)")
        for i, row in enumerate(rows):
            row["recheck"] = {"done": i in plan, "why": plan.get(i) or why.get(i, "")}
        return plan

    def batched(self, names: Sequence[str] | None = None, *, recheck: int | str | None = "band",
                batch: int = 64) -> list[dict[str, Any]]:
        """Arms measured in batched native forwards (``sweep.cell_sweep``): state-cell arms
        (every edit a ``replace`` of whole cells ``cache.L<l>.k|v[kv<g>,row=<r>]``: one kv head,
        one row, all of head_dim) and port arms at the pending row's native-module ports
        (``a q k v n g sg u f A o m``; any op/selector/mask; holds refused), the branch point the
        pending row's first port with no edits on the path. Kernel path ``batched``; the measured
        null-vs-stepped delta is added to those arms' floor.

        ``recheck`` (module docstring): ``"band"`` (default) re-measures stepped only the arms
        whose effect is within the kernel-sensitive band and could flip, ``"all"`` every arm,
        an int ``k`` the ``k`` of largest ``|dM|``, ``0`` none. A rechecked row keeps both
        (``stepped`` on the row, ``kernel_delta`` = batched - stepped); every row carries
        ``recheck = {done, why}``."""
        from .stability import BATCHED
        from .sweep import cell_sweep

        torch = _torch()
        chosen = [self.arm(n) for n in (list(self.arms) if names is None else names)]
        cells_of = []
        for arm in chosen:
            if arm.holds or arm.final_scale != "native":
                raise Refusal(f"arm {arm.name}: holds run on the stepped path only")
            cells_of.append([self._batched_edit(arm, e) for e in arm.edits])
        logits_all = self.base_logits
        run = self.debuggee
        first = f"t{self.row}.L0.{run.program.names[0]}"
        moved = run.parse(run.where) != run.parse(first)
        if moved:
            # the sweep runs the pending row whole from its first port: with nothing edited on
            # the path (cell_sweep refuses otherwise) the branch point's state is the same
            run.rewind(first)
        try:
            swept = cell_sweep(run, cells_of, batch=batch, stepped_logits=logits_all)
        finally:
            if moved:
                run.restore(self.checkpoint)
        extra = float(swept["null_vs_stepped"] or 0.0)
        # the same delta over the consumer's own tokens only (the showcase floors measured it
        # there; ``null_vs_stepped`` is the max over the whole vocabulary)
        own = None
        if isinstance(self.consumer, (Margin, Logit)) and len(swept["null"]):
            ids = ([self.consumer.a, self.consumer.b] if isinstance(self.consumer, Margin)
                   else [self.consumer.i])
            native = logits_all.reshape(-1).float().cpu()[ids]
            own = float(max((n[ids] - native).abs().max() for n in swept["null"]))
        rows = []
        for i, arm in enumerate(chosen):
            row_logits = swept["logits"][i]
            arm.result = self._row(arm, row_logits, [], [{"cell": e.target, "op": e.op,
                                                          "origin": getattr(e.value, "pointer",
                                                                            e.origin)}
                                                         for e in arm.edits], [], 0, 0.0, BATCHED,
                                   extra=2 * extra)
            arm.result["kernel_delta_null"] = num(extra, MEASURED)
            if own is not None:
                arm.result["kernel_delta_null_consumer_tokens"] = num(own, MEASURED)
            rows.append(arm.result)
        for i, why in self._recheck_plan(rows, recheck).items():
            stepped = self._measure(chosen[i])
            rows[i]["stepped"] = {"dM": stepped["dM"], "argmax": stepped["argmax"],
                                  "kernel_path": stepped["kernel_path"]}
            rows[i]["kernel_delta"] = num(rows[i]["dM"]["value"] - stepped["dM"]["value"],
                                          MEASURED)
        _ = torch
        return rows

    # --- debugger verbs -------------------------------------------------------------------------

    def restore(self) -> str:
        """Put the run back at the branch point."""
        return self.debuggee.restore(self.checkpoint)

    def compare(self, a: str, b: str) -> dict[str, Any]:
        """Arm ``a`` minus arm ``b`` at the consumer, in the shared floor's units."""
        ra, rb = (self.arm(n).result for n in (a, b))
        if ra is None or rb is None:
            raise Refusal("measure both arms first (run_arms)")
        diff = ra["M"]["value"] - rb["M"]["value"]
        out = {"a": a, "b": b, "difference": num(diff, MEASURED),
               "kernel_paths": [ra["kernel_path"], rb["kernel_path"]]}
        if self.floor:
            out["floors"] = num(abs(diff) / self.floor, MEASURED)
            out["above_floor"] = abs(diff) > self.floor
        return out

    def table(self) -> list[dict[str, Any]]:
        """One row per measured arm (base first): name, kind, dM, argmax, floors, stability."""
        rows = [{"name": "base", "kind": "base", "M": self.M0, "dM": 0.0,
                 "argmax": self.base_argmax, "effect_floors": 0.0, "kernel_path": self.kernel_path}]
        for arm in self.arms.values():
            r = arm.result
            if r is None:
                continue
            rows.append({"name": arm.name, "kind": arm.kind, "M": r["M"]["value"],
                         "dM": r["dM"]["value"], "argmax": r["argmax"],
                         "effect_floors": (r.get("effect_floors") or {}).get("value"),
                         "stability": r.get("stability"), "kernel_path": r["kernel_path"]})
        return rows

    def seal(self, *, session_id: str | None = None) -> dict[str, Any]:
        """A ``branches`` record of every measured arm (at most 64; ``arms_omitted`` counts the
        rest), with custody (P8)."""
        from mdb.core.custody import custody

        run = self.debuggee
        measured = [a.result for a in self.arms.values() if a.result is not None]
        row, index = run.parse(self.at)
        consumer: dict[str, Any] = {"text": self.text, "row": int(self.row)}
        if isinstance(self.consumer, Margin):
            consumer.update(kind="margin", a=int(self.consumer.a), b=int(self.consumer.b))
        elif isinstance(self.consumer, Logit):
            consumer.update(kind="logit", i=int(self.consumer.i))
        else:
            consumer["kind"] = "callable"
        body = {
            "schema": SCHEMA, "name": self.name,
            "at": {"where": self.at, "row": int(row), "cut": self.checkpoint["cut"],
                   "transitions": int(self.checkpoint["transitions"]),
                   "path_edits": _safe(self.path["edits"], SUPPLIED),
                   "path_cells": _safe(self.path["cells"], SUPPLIED)},
            "consumer": consumer,
            "base": {"M": num(self.M0, MEASURED), "argmax": self.base_argmax,
                     **({"probes": self.base_probes} if self.probes else {})},
            "probes": sorted(self.probes),
            "floor": ({"value": num(self.floor, MEASURED), "rule": self.floor_rule,
                       "spacings": num(self.floor_spacings, SUPPLIED),
                       "extra": num(self.extra_floor, MEASURED)} if self.floor is not None
                      else {"rule": self.floor_rule}),
            "arms": measured[:_CAP], "n_arms": len(measured),
            "arms_omitted": max(0, len(measured) - _CAP),
            "procedure": procedure(run),
            "custody": custody(self.kernel_path),
        }
        def sealed(b):
            return seal_record(session_id=session_id or run_session_id(run), kind="branches",
                               model_identity=run.adapter.model_identity,
                               execution=dict(run.adapter.execution), authority=MEASURED, body=b)
        from mdb.core.pointers import RecordError

        try:
            return sealed(body)
        except RecordError:
            # too large for one record (many arms with edits/probes): the full rows by pointer
            # (bounded.put_json, held on the run), a compact per-arm table in the record
            from mdb.core.bounded import put_json

            body["arms_payload"] = put_json(measured, run=run)
            body["arms"] = [{"name": r["name"], "kind": r["kind"], "M": r["M"], "dM": r["dM"],
                             "argmax": r["argmax"], "stability": r.get("stability"),
                             "effect_floors": r.get("effect_floors"),
                             "kernel_path": r["kernel_path"]} for r in measured[:_CAP]]
            body["arms_compact"] = True
            return sealed(body)


def branches(run: Any, consumer: Any, **options: Any) -> Branches:
    return Branches(run, consumer, **options)


__all__ = ["Arm", "Branches", "Edit", "NULL_KINDS", "RESIDUAL_PORTS", "SCHEMA", "branches",
           "edit"]

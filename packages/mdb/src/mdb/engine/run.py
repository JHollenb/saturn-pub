"""A port-level cursor over a native decoder Session: step, read, edit, checkpoint, rewind.

The cursor is plain data: the current layer-boundary cut, a port name inside the pending
layer (or none), and the ordered edits staged for that layer. Moving to a port probes the
native layer from the boundary on a throw-away candidate (``Session.probe``); moving past the
last port commits the native transition through ``Session.continue_`` so a receipt and cut
exist. ``lookahead`` decides what a move inside a layer computes. ``"none"`` (default) runs a
probe that halts at the cursor port: no port after the cursor is held, and no native code runs
past the cursor, with one disclosed exception: the native eager attention function cannot be
split, so at a ``p`` stop the value mix ``A`` is computed natively as a by-product of ``p``
(never held, checked or captured; ``status()["computed_past_cursor"]`` is ``["A"]`` there and
``[]`` elsewhere). ``"layer"`` runs one full-layer probe per layer and serves the ports from
that capture cache, so the native code has already computed everything after the cursor:
``status()["computed_past_cursor"]`` lists every such port. ``row_captures`` returns only ports
at or before the cursor in both modes.

Stops of one token row, in order: ``start`` (before the embedding), ``emb``, then
``L<i>.<port>`` for every layer and port (``L<i>.h_out`` is the committed layer boundary),
then ``fn``, ``logits``, ``sample``; the commit after ``sample`` leads to the next row's
``start``. Addresses are ``t<row>.<stop>``; within the current row ``<stop>`` alone works.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from saturn_pub.core import InPlaceWrite, Session, StateCut, held_bytes
from saturn_pub.values import clone, digest

from .edits import CELL_BOUNDARY, CellEdit, is_cell
from .ports import PortController, PortEdit, check_deterministic, program_for, tensor_digest

GLOBAL_STOPS = ("start", "emb", "fn", "logits", "sample")
CHECKPOINT_SCHEMA = "mdb-engine-checkpoint-v1"
LOOKAHEADS = ("none", "layer")
# Full-layer probes of committed, unedited layers kept for reads. One entry is one row's ports
# of one layer (about 85 KB + 48 bytes per token row at d=1536, d_ff=8960, bf16), so a whole
# 28-layer row costs a few MB; analysis verbs walk every layer of a row more than once.
_READ_CACHE = 64
# Receipts of the run's own Session kept (about 2 KB of host memory each, one per transition).
# Nothing reads older ones: each trail mark keeps its transition's fired edit records.
_RECEIPTS = 64
_ADDRESS = re.compile(r"(?:t(?P<row>\d+)\.)?(?P<rest>.+)")
_LAYER_PORT = re.compile(r"L(?P<layer>\d+)\.(?P<port>\w+)")
_SLOTS = {"start": "tokens", "logits": "logits", "sample": "sampled_token"}


class ReplayMismatch(ValueError):
    """A replay from the run root reached another cut than the checkpoint records."""

    def __init__(self, expected: str, reached: str):
        self.expected, self.reached = expected, reached
        super().__init__("replay from the run root did not reproduce the checkpoint cut")


@dataclass
class _Mark:
    fingerprint: str
    cut: StateCut | None
    row: int
    label: str
    phase: str
    layer: int
    transitions: int
    edits: tuple[PortEdit, ...] = ()
    fired: tuple[dict[str, Any], ...] = ()  # the receipt's port_edits for this layer transition
    cells: tuple[CellEdit, ...] = ()  # state-cell edits realised in this layer transition
    cell_fired: tuple[dict[str, Any], ...] = ()  # their realisation records, in order


def _coverage(
    writes: Sequence[tuple[str, str, Any, str]], select: torch.Tensor | None
) -> tuple[int, int, bool]:
    """Elementwise bookkeeping of a value's writes (``(kind, op, mask, source)`` in order) over
    the selected elements: ``(written, selected, replaced)`` where ``replaced`` says every
    selected element's last write is a ``replace`` edit (so it is the cast of a supplied tensor).
    ``mask`` None covers every element."""
    masks = [w[2] for w in writes if w[2] is not None]
    shape = tuple(select.shape) if select is not None else tuple(masks[0].shape) if masks else (1,)
    chosen = torch.ones(shape, dtype=torch.bool) if select is None else select.to("cpu", torch.bool)
    touched = torch.zeros(shape, dtype=torch.bool)
    replaced = torch.zeros(shape, dtype=torch.bool)
    for kind, op, mask, _ in writes:
        cover = torch.ones(shape, dtype=torch.bool) if mask is None else mask.to("cpu").broadcast_to(shape)
        touched |= cover
        if kind == "edit" and op == "replace":
            replaced |= cover
        else:
            replaced &= ~cover
    count = int((touched & chosen).sum())
    return count, int(chosen.sum()), count > 0 and bool((replaced | ~chosen).all())


def _cell_upstream(row: int, cell: CellEdit, staged: bool) -> dict[str, Any]:
    """An ``upstream_edits`` entry for a state-cell edit: ``row`` of the transition that realises
    it, the cell's ``layer``, ``port`` ``cache.L<l>.<k|v>``, ``op``, ``staged`` and ``cell`` (the
    address with its selector)."""
    return {"row": row, "layer": cell.layer, "port": cell.port, "op": cell.op, "staged": staged,
            "cell": cell.text}


class _Snapshot:
    """Everything one cursor move can change, so a move that is interrupted (``BaseException``,
    ``KeyboardInterrupt`` included) after the native session has moved puts the session back to
    the cut of the last trail mark and restores the trail, the staged edits and the cursor
    exactly as they were. Rolling back (rather than completing the record) is always possible:
    the last mark always holds its cut, and restoring a cut cannot fail the way a replay can."""

    def __init__(self, run: Run):
        self.run = run
        self.trail = list(run._trail)
        self.marks = [(mark, mark.cut, mark.edits, mark.fired, mark.cells, mark.cell_fired)
                      for mark in run._trail]
        self.edits, self.port, self.cache = list(run._edits), run._port, run._cache
        self.cells = list(run._cells)
        self.last = run._trail[-1]
        self.cut, self.fingerprint = self.last.cut, self.last.fingerprint

    def rollback(self) -> None:
        run = self.run
        for mark, cut, edits, fired, cells, cell_fired in self.marks:
            mark.cut, mark.edits, mark.fired = cut, edits, fired
            mark.cells, mark.cell_fired = cells, cell_fired
        run._trail = list(self.trail)
        run._edits, run._port, run._cache = list(self.edits), self.port, self.cache
        run._cells = list(self.cells)
        run._drop_reads()
        for attempt in range(3):  # a second interrupt during the rollback is retried
            try:
                try:
                    here = run.session.capture(retain=False).fingerprint
                except InPlaceWrite:
                    here = None  # the state was written in place: only a restore recovers it
                if here != self.fingerprint:
                    run.session.restore(self.cut)
                return
            except BaseException:
                if attempt == 2:
                    raise


class Run:
    """Port-level debugger cursor over ``DecoderAdapter(granularity="operation")``.

    Memory: a trail mark's cut shares every tensor its transition did not replace with the
    previous mark (``saturn_pub`` ``shares_state``) and holds a grown K/V layer as its appended
    rows, so a mark costs about what its transition wrote (one layer's new K/V row, the hidden
    carrier, the logits). ``held_bytes()`` measures it. Edit tensors no edit, checkpoint or
    ``hold`` names any more are dropped; the session keeps its last ``_RECEIPTS`` receipts.

    ``retain_rows`` bounds memory further: boundary cuts of rows older than that many rows
    behind the cursor are dropped (each row's ``start`` cut is kept) and rebuilt on demand by
    replaying the native transitions, with their committed edits, from the nearest retained
    cut.

    ``root`` (persistence): start the run at a stored boundary cut of a run over the same
    ``tokens`` instead of prefilling them; ``root_transitions`` is the transition count that
    cut had in the run that stored it, so ``restore`` finds a checkpoint taken at that cut.

    Read cache (reads of committed layers): a committed layer whose transition carried no port
    edit and no cell edit is read from ONE full-layer probe of its boundary cut, kept for the
    last ``_READ_CACHE`` such layers and keyed by (boundary cut fingerprint, layer): the probe is
    a pure function of that cut, so an entry cannot go stale. A committed layer with edits or
    cells keeps its per-port stop-probe, but on the Session hydrated from the same cut (one
    held, keyed by fingerprint; ``Session.probe`` runs on a clone and leaves it unchanged).
    Values are cloned out of the cache. Both are dropped on restore, reverse, prune (when a
    cut is dropped), a rolled-back move and ``close``; ``cache_reads = False`` bypasses both.
    """

    cache_reads = True

    def __init__(
        self,
        adapter: Any,
        tokens: Sequence[int],
        *,
        retain_rows: int | None = None,
        lookahead: str = "none",
        root: StateCut | None = None,
        root_transitions: int = 0,
    ):
        from saturn_pub.adapters.decoder import DecoderAdapter

        if not isinstance(adapter, DecoderAdapter):
            raise ValueError("Run requires a saturn_pub DecoderAdapter")
        if adapter.granularity != "operation":
            raise ValueError("Run requires DecoderAdapter(granularity='operation')")
        if adapter.execution.get("residency", "resident") != "resident":
            raise ValueError("Run supports resident execution only")
        if retain_rows is not None and (type(retain_rows) is not int or retain_rows < 0):
            raise ValueError("retain_rows must be a nonnegative integer or None")
        if lookahead not in LOOKAHEADS:
            raise ValueError(f"lookahead must be one of {', '.join(LOOKAHEADS)}")
        check_deterministic(adapter.model)
        self.lookahead = lookahead
        self.adapter = adapter
        self.name: str | None = None  # the debugger's name for this run (an origin of values)
        self.program = program_for(adapter)
        self.tokens = [int(token) for token in tokens]
        self.retain_rows = retain_rows
        if type(root_transitions) is not int or root_transitions < 0:
            raise ValueError("root_transitions must be a nonnegative integer")
        if root is None:
            self.session = adapter.session(self.tokens)
        else:
            self.session = Session.from_cut(adapter, root)
            held = [int(t) for t in self.session.read("tokens")[0].tolist()]
            if held[: len(self.tokens)] != self.tokens:
                raise ValueError("the root cut holds other token ids than this run's prompt")
        self._stops = (
            "start",
            "emb",
            *(f"L{i}.{port}" for i in range(adapter.layers) for port in self.program.names),
            "fn",
            "logits",
            "sample",
        )
        self._index = {label: index for index, label in enumerate(self._stops)}
        self._port: str | None = None
        self._edits: list[PortEdit] = []
        self._cells: list[CellEdit] = []
        self._cache: PortController | None = None
        self._values: dict[str, torch.Tensor] = {}
        self._masks: dict[str, torch.Tensor] = {}
        # edit tensors a plain-data checkpoint or pointer handed out may name, or that hold()
        # supplied: kept for restore; any other edit tensor is dropped once no edit uses it
        self._pinned: set[str] = set()
        self._trail: list[_Mark] = [self._mark(root_transitions)]
        self._reads: OrderedDict[tuple[str, int], PortController] = OrderedDict()
        self._hydrated: tuple[str, Session] | None = None

    # --- cursor -------------------------------------------------------------------------

    @property
    def where(self) -> str:
        mark = self._trail[-1]
        if self._port is not None:
            return f"t{mark.row}.L{mark.layer}.{self._port}"
        return f"t{mark.row}.{mark.label}"

    def status(self) -> dict[str, Any]:
        """Where the cursor is and the lookahead mode that governs what has been computed."""
        return {
            "where": self.where,
            "port": self._port,
            "pending_layer": self.pending_layer,
            "lookahead": self.lookahead,
            "computed_past_cursor": self._computed_past_cursor(),
        }

    def _computed_past_cursor(self) -> list[str]:
        """Ports after the cursor port whose value the native code has already computed:
        ``["A"]`` at a ``p`` stop in ``"none"`` mode (the eager call cannot be split), every
        later port of the cached full-layer probe in ``"layer"`` mode."""
        if self._port is None:
            return []
        if self.lookahead == "none":
            return ["A"] if self._port == "p" else []
        if self._cache is None:
            return []
        names = self.program.names
        after = names.index(self._port)
        return [name for name in names if names.index(name) > after and name in self._cache.captures]

    @property
    def pending_layer(self) -> int | None:
        mark = self._trail[-1]
        return mark.layer if mark.phase == "layer" else None

    @property
    def edits(self) -> list[dict[str, Any]]:
        return [edit.manifest() for edit in self._edits]

    @property
    def cells(self) -> list[dict[str, Any]]:
        """Manifests of the staged state-cell edits (``CellEdit.manifest``), in staging order."""
        return [cell.manifest() for cell in self._cells]

    @property
    def checks(self) -> dict[str, Any]:
        """Replica checks (``s`` and the value mix) from the current layer's capture pass."""
        return dict(self._cache.checks) if self._cache is not None else {}

    def stops(self) -> tuple[str, ...]:
        return self._stops

    def _label(self, phase: str, layer: int) -> str:
        last = self.adapter.layers - 1
        if phase == "layer":
            return "emb" if layer == 0 else f"L{layer - 1}.h_out"
        return {
            "embed": "start",
            "normalization": f"L{last}.h_out",
            "readout": "fn",
            "sample": "logits",
            "commit": "sample",
        }[phase]

    def _mark(self, transitions: int) -> _Mark:
        cut = self.session.capture(retain=False)
        point = cut.execution_point
        return _Mark(
            cut.fingerprint,
            cut,
            point["logical_step"],
            self._label(point["phase"], point["local_clock"]),
            point["phase"],
            point["local_clock"],
            transitions,
        )

    def _key(self) -> tuple[int, int]:
        mark = self._trail[-1]
        if self._port is not None:
            return mark.row, self._index[f"L{mark.layer}.{self._port}"]
        return mark.row, self._index[mark.label]

    def _mark_key(self, mark: _Mark) -> tuple[int, int]:
        return mark.row, self._index[mark.label]

    def _next(self, key: tuple[int, int]) -> tuple[int, int]:
        row, index = key
        return (row, index + 1) if index + 1 < len(self._stops) else (row + 1, 0)

    def _previous(self, key: tuple[int, int]) -> tuple[int, int]:
        row, index = key
        return (row, index - 1) if index else (row - 1, len(self._stops) - 1)

    def _in_pending(self, key: tuple[int, int]) -> str | None:
        """The port name if ``key`` is a non-final port of the pending layer."""
        mark = self._trail[-1]
        if mark.phase != "layer" or key[0] != mark.row:
            return None
        label = self._stops[key[1]]
        prefix = f"L{mark.layer}."
        if not label.startswith(prefix) or label == f"L{mark.layer}.h_out":
            return None
        return label[len(prefix) :]

    def parse(self, target: str) -> tuple[int, int]:
        """``t<row>.<stop>``, ``<stop>``, ``L<i>.<port>`` or a bare port of the pending layer."""
        match = _ADDRESS.fullmatch(target.strip())
        if match is None:
            raise ValueError(f"cannot parse address {target!r}")
        mark = self._trail[-1]
        row = mark.row if match["row"] is None else int(match["row"])
        rest = match["rest"]
        if rest in GLOBAL_STOPS:
            return row, self._index[rest]
        layer_port = _LAYER_PORT.fullmatch(rest)
        if layer_port is not None:
            label = f"L{int(layer_port['layer'])}.{layer_port['port']}"
        elif rest in self.program.names:
            if self.pending_layer is None or match["row"] is not None:
                raise ValueError(f"bare port {rest!r} needs a pending layer; use L<i>.{rest}")
            label = f"L{self.pending_layer}.{rest}"
        else:
            raise ValueError(
                f"unknown stop {rest!r}; use {', '.join(GLOBAL_STOPS)}, L<i>.<port> or a port "
                f"of {', '.join(self.program.names)}"
            )
        if label not in self._index:
            raise ValueError(f"no stop {label} in a {self.adapter.layers}-layer program")
        return row, self._index[label]

    def address(self, key: tuple[int, int]) -> str:
        return f"t{key[0]}.{self._stops[key[1]]}"

    # --- native motion ------------------------------------------------------------------

    def _controller(
        self, layer: int, edits: Sequence[PortEdit], cells: Sequence[CellEdit] = (),
        **options: Any,
    ) -> Any:
        return PortController(
            self.program, layer, edits=tuple(edits), layers=self.adapter.layers,
            cells=tuple(cells), **options,
        )

    def _advance(
        self, session: Session, layer: int, edits: Sequence[PortEdit],
        cells: Sequence[CellEdit] = (),
    ) -> tuple[Any, PortController | None]:
        """One native transition (with the edits and cell edits for its layer); returns the
        receipt and the controller that carried them (None when nothing was carried)."""
        check_deterministic(self.adapter.model)
        if edits or cells:
            controller = self._controller(layer, edits, cells, capture=False)
            return session.continue_(1, ports=controller), controller
        return session.continue_(1), None

    def _commit(self) -> Any:
        mark = self._trail[-1]
        layer = mark.phase == "layer"
        edits = tuple(self._edits) if layer else ()
        cells = tuple(self._cells) if layer else ()  # carried across non-layer transitions
        snapshot = _Snapshot(self)
        try:
            receipt, controller = self._advance(self.session, mark.layer, edits, cells)
            new = self._mark(mark.transitions + 1)
            fired = tuple(  # port edits only: cell entries (``cell`` key) are in cell_fired
                dict(entry) for entry in receipt.to_dict()["transitions"][0].get("port_edits", ())
                if "cell" not in entry
            ) if edits else ()
            mark.edits = edits
            if edits:
                mark.fired = fired
            if cells:
                mark.cells = cells
                mark.cell_fired = tuple(dict(controller.cell_records[i])
                                        for i in range(len(cells)))
                self._cells = []
            self._edits, self._port, self._cache = [], None, None
            self._trail.append(new)
        except BaseException:
            snapshot.rollback()  # the session is never left a transition ahead of the trail
            raise
        self._prune()
        return receipt

    def _probe(self, session: Session, controller: PortController) -> dict[str, Any]:
        check_deterministic(self.adapter.model)
        return session.probe(controller)

    def _upto(self, port: str) -> tuple[str, ...]:
        names = self.program.names
        return names[: names.index(self.program.spec(port).name) + 1]

    def _ensure_cache(self, port: str | None = None) -> PortController:
        """The capture cache for the cursor, at ``port`` (default: the current cursor port).

        ``lookahead="layer"``: one full-layer probe. ``"none"``: a probe that stops at ``port``
        and captures only ports up to it; with no cursor port yet (``port`` is None and the
        cursor is at a boundary) there is nothing to hold.
        """
        layer = self.pending_layer
        if layer is None:
            raise ValueError(f"no pending decoder layer at {self.where}")
        port = self._port if port is None else port
        if self.lookahead == "layer":
            if self._cache is None:
                controller = self._controller(layer, self._edits, self._cells)
                if self._probe(self.session, controller)["stopped"]:
                    raise RuntimeError("full-layer probe stopped unexpectedly")
                self._cache = controller
            return self._cache
        if port is None:
            raise ValueError(f"no port at {self.where}")
        stop = self.program.spec(port).name
        if self._cache is None or self._cache.stop != stop:
            controller = self._controller(
                layer, self._edits, self._cells, stop=stop, capture=self._upto(stop)
            )
            if not self._probe(self.session, controller)["stopped"]:
                raise RuntimeError("stop-probe did not stop at its port")
            unfired = controller.unfired_before_stop()
            if unfired:
                raise ValueError(f"staged edits were not realised: {', '.join(unfired)}")
            self._cache = controller
        return self._cache

    def port_shape(self, port: str) -> tuple[int, ...]:
        """Shape of ``port`` in the pending transition, derived from the program axes and the
        model config (executes nothing)."""
        src = int(self.session.read("tokens").shape[1])
        return self.program.shape(port, self.adapter.model.config, src=src)

    def _forward_to(self, target: tuple[int, int]) -> None:
        while True:
            current = self._key()
            if current == target:
                return
            if current > target:
                raise ValueError(
                    f"{self.address(target)} is behind the cursor {self.where}; "
                    "use reverse_step or restore"
                )
            port = self._in_pending(target)
            if port is not None:
                self._ensure_cache(port)
                self._port = port
                return
            self._commit()

    def _unit(self, unit: str):
        if unit == "port":
            return lambda label: True
        if unit == "sublayer":
            return lambda label: label in GLOBAL_STOPS or label.endswith((".r", ".h_out"))
        if unit == "layer":
            return lambda label: label in GLOBAL_STOPS or label.endswith(".h_out")
        if unit == "token":
            return lambda label: label == "start"
        raise ValueError("unit must be port, sublayer, layer or token")

    def step(self, unit: str = "port", n: int = 1) -> str:
        accept = self._unit(unit)
        if type(n) is not int or n < 1:
            raise ValueError("n must be a positive integer")
        for _ in range(n):
            key = self._next(self._key())
            while not accept(self._stops[key[1]]):
                key = self._next(key)
            self._forward_to(key)
        return self.where

    def next(self) -> str:
        return self.step("sublayer")

    def finish(self) -> str:
        layer = self.pending_layer
        if layer is None:
            return self.step("token")
        self._forward_to((self._trail[-1].row, self._index[f"L{layer}.h_out"]))
        return self.where

    def until(self, target: str) -> str:
        self._forward_to(self.parse(target))
        return self.where

    def commit(self) -> str:
        if self.pending_layer is None:
            raise ValueError(f"nothing to commit at {self.where}")
        return self.finish()

    def abort(self) -> str:
        """Drop every staged edit, port and state-cell alike."""
        saved = (self._edits, self._cells, self._cache)
        self._edits, self._cells = [], []
        self._cache = None
        try:
            if self._port is not None:
                self._ensure_cache()
        except BaseException:
            self._edits, self._cells, self._cache = saved  # a refused abort changes nothing
            raise
        self._sweep_tensors()
        return self.where

    # --- reads and edits ----------------------------------------------------------------

    def read(self, target: str | None = None) -> torch.Tensor:
        return self._read(target)[0]

    def read_info(
        self, target: str | None = None, *, select: torch.Tensor | None = None
    ) -> dict[str, Any]:
        """How the value ``read(target)`` returns came about. Always exactly these keys:

        ``edited`` (an edit wrote elements of this value), ``by`` (comma-joined port names whose
        edit or realisation produced it, or None), ``kind`` (``native | edit | realisation |
        replica``), ``authority`` (``measured | derived-exact | derived-local | supplied``, None
        when refused), ``state`` (``resolved`` | ``refused``) and ``reason`` (why, when refused),
        ``ops`` (every operation that wrote the value, in order), ``selective`` (any such edit
        has a mask), ``upstream_edits`` (every edit that fired before this value in program
        order: ``{row, layer, port, op, staged}``), ``replica_equal`` and ``max_abs_diff``
        (replica check of ``s`` / of the value mix at ``A``; None when not applicable or not
        reached) with ``max_abs_diff_authority``, ``realisation_exact`` and
        ``realisation_max_abs_error`` (``o`` and ``r`` under an ``r`` edit), and
        ``same_port_edit`` (``{port, ops}`` when an edit at this port wrote none of the elements
        read).

        Authority: ``supplied`` only when every element read is, bit for bit, the cast of a
        supplied tensor (the last operation covering each element is a ``replace`` and nothing
        rewrote it afterwards); a value written by any other combination of operations is
        ``derived-exact``; an element no edit wrote keeps the native label. ``select`` (a bool
        tensor over the port) restricts the claim to those elements.
        """
        return self._describe(self._read(target, value=False)[1], select)

    def _describe(self, context: dict[str, Any], select: torch.Tensor | None = None) -> dict[str, Any]:
        port, edits, committed = context["port"], context["edits"], context["committed"]
        controller, staged = context["controller"], context["staged"]
        names = self.program.names
        spec = self.program.spec
        boundary = context["boundary_edits"]
        skip = {id(e) for e in boundary}
        upstream = [
            {"row": mark.row, "layer": mark.layer, "port": e.port, "op": e.op, "staged": False}
            for mark in committed
            for e in mark.edits
            if id(e) not in skip
        ]
        upstream += [_cell_upstream(mark.row, cell, False) for mark in committed
                     for cell in mark.cells]
        upstream += [_cell_upstream(context["row"], cell, staged)
                     for cell in context.get("cells", ()) if cell.timing == "before"]
        info: dict[str, Any] = {
            "edited": False,
            "by": None,
            "kind": "native",
            "authority": "measured",
            "state": "resolved",
            "reason": None,
            "ops": [],
            "selective": False,
            "upstream_edits": upstream,
            "replica_equal": None,
            "replica_checked": None,
            "max_abs_diff": None,
            "max_abs_diff_authority": None,
            "realisation_exact": None,
            "realisation_max_abs_error": None,
            "same_port_edit": None,
        }
        if port is None and not boundary:
            return info
        index = names.index(port) if port is not None else len(names) - 1

        if port is not None:
            for e in sorted(edits, key=lambda e: names.index(spec(e.port).name)):
                if names.index(spec(e.port).name) < index:
                    upstream.append(
                        {
                            "row": context["row"], "layer": context["layer"], "port": e.port,
                            "op": e.op, "staged": staged,
                        }
                    )
        label_port = "h_out" if port is None else port
        here = [e for e in edits if spec(e.port).name == port] if port is not None else []
        edited_ports = {spec(e.port).name for e in edits}
        r_edits = [e for e in edits if spec(e.port).name == "r"]
        # (kind, op, mask, source port) of everything that wrote the value, in order
        writes: list[tuple[str, str, Any, str]] = [("edit", e.op, e.mask, "h_out") for e in boundary]
        writes += [("edit", e.op, e.mask, label_port) for e in here]
        base = None
        if port == "o" and r_edits:
            writes += [("realisation", e.op, e.mask, "r") for e in r_edits]
        elif port == "s" or (port == "p" and "s" in edited_ports):
            base = "replica"
        elif port == "A" and edited_ports & {"s", "p"}:
            base = "replica"
        realisation = (controller.realisation or {}) if controller is not None else {}
        inexact_r = port == "r" and bool(r_edits) and not realisation.get("exact", False)
        base_by = (
            ["s"] if port == "p" and base else sorted(edited_ports & {"s", "p"}, key=names.index)
            if port == "A" and base
            else []
        )
        info["ops"] = [w[1] for w in writes]
        info["selective"] = any(w[2] is not None for w in writes if w[0] == "edit")
        if port in ("o", "r") and r_edits and "exact" in realisation:
            info["realisation_exact"] = bool(realisation["exact"])
            info["realisation_max_abs_error"] = float(realisation["max_abs_error"])
        if base is not None:
            info.update(kind="replica", authority="derived-exact")
        written, _, _ = _coverage(writes, select)
        if written == 0:
            info["by"] = ",".join(base_by) or None
            if here or boundary:
                info["same_port_edit"] = {"port": label_port, "ops": [w[1] for w in writes if w[0] == "edit"]}
            self._check_replica(info, port, controller, edited_ports, base)
            return info
        own, _, _ = _coverage([w for w in writes if w[0] == "edit"], select)
        _, _, replaced = _coverage(writes, select)
        sources = [w[3] for w in writes if _coverage([w], select)[0]]
        info["by"] = ",".join(sorted(set(sources) | set(base_by), key=names.index))
        info["edited"] = own > 0
        info["kind"] = "edit" if own > 0 else "realisation"
        info["authority"] = "supplied" if replaced else "derived-exact"
        if inexact_r:
            info["authority"] = "measured"
            info["kind"] = "realisation"
        if info["authority"] != "supplied":
            self._check_replica(info, port, controller, edited_ports, base)
        return info

    @staticmethod
    def _check_replica(
        info: dict[str, Any], port: str | None, controller: Any, edited: set[str], base: str | None
    ) -> None:
        """Fill the replica fields of ``s``/``A``; refuse a label that rests on a replica whose check
        failed. A check not run yet leaves the label derived-exact with ``replica_checked`` False."""
        if controller is None or port not in ("s", "p", "A"):
            return
        checks = controller.checks
        needed = {"s": ["s"], "p": ["s"], "A": ["mix_replica"] + (["s"] if "s" in edited else [])}[port]
        diffs = []
        failed = missing = False
        for name in needed:
            check = checks.get(name)
            if check is None:
                missing = True
            elif not check["equal"]:
                failed = True
                diffs.append(float(check.get("max_abs_diff", 0.0)))
        if port in ("s", "A"):
            key = "s" if port == "s" else "mix_replica"
            check = checks.get(key)
            if check is not None:
                info["replica_equal"] = bool(check["equal"])
                info["max_abs_diff"] = float(check.get("max_abs_diff", 0.0))
                info["max_abs_diff_authority"] = "measured"
        if base is None:
            return
        if failed:
            info.update(
                state="refused", authority=None,
                reason=f"replica check failed: max |replica \u2212 native| = {max(diffs)}",
            )
        elif missing:  # derived from captured operands by the stated formula; compare pending
            info["replica_checked"] = False
        else:
            info["replica_checked"] = True

    def _committed_probe(self, index: int, name: str) -> PortController:
        """The probe that serves port ``name`` of the committed layer at trail mark ``index``:
        the cached full-layer probe when that transition carried no edits and no cells (see the
        class docstring), else a stop-probe with its edits and cells."""
        past = self._trail[index]
        if self.cache_reads and not past.edits and not past.cells:
            key = (past.fingerprint, past.layer)
            controller = self._reads.get(key)
            if controller is not None:
                self._reads.move_to_end(key)
                return controller
            controller = self._controller(past.layer, ())
            if self._probe(self._session_at(index), controller)["stopped"]:
                raise RuntimeError("full-layer probe stopped unexpectedly")
            self._reads[key] = controller
            while len(self._reads) > _READ_CACHE:
                self._reads.popitem(last=False)
            return controller
        # a committed read is not bound by the cursor: stop after s (at p) so the replica
        # check of s has run and its label can be resolved
        stop = "p" if name == "s" else name
        controller = self._controller(
            past.layer, past.edits, past.cells, stop=stop, capture=(name,)
        )
        self._probe(self._session_at(index), controller)
        return controller

    def _drop_reads(self) -> None:
        """Forget the read cache and the hydrated Session (see the class docstring)."""
        self._reads = OrderedDict()
        self._hydrated = None

    def close(self) -> None:
        """Release the read cache and the hydrated Session; the run stays usable."""
        self._drop_reads()

    def held_bytes(self) -> dict[str, int]:
        """Bytes of tensor storage this run holds, each storage counted once (measured):
        ``trail`` (the marks' cuts: consecutive cuts share what a transition did not replace,
        and a grown K/V layer is held as its appended rows), ``state`` (the live session),
        ``reads`` (the committed-read cache and the hydrated Session), ``edits`` (held edit
        values and masks) and ``total`` (their union)."""
        cuts = [mark.cut for mark in self._trail if mark.cut is not None]
        reads = [captures for controller in self._reads.values()
                 for captures in (controller.captures, controller.originals)]
        if self._hydrated is not None:
            reads.append(self._hydrated[1])
        edits = [self._values, self._masks]
        return {
            "trail": held_bytes(*cuts),
            "state": held_bytes(self.session),
            "reads": held_bytes(*reads),
            "edits": held_bytes(*edits),
            "total": held_bytes(*cuts, self.session, *reads, *edits),
            "marks": len(self._trail),
        }

    def _layer_mark(self, key: tuple[int, int]) -> tuple[int, str] | None:
        """(trail index, port) of a ``L<i>.<port>`` stop whose layer transition was committed."""
        match = _LAYER_PORT.fullmatch(self._stops[key[1]])
        if match is None:
            return None
        layer = int(match["layer"])
        for index, mark in enumerate(self._trail):
            if mark.row == key[0] and mark.phase == "layer" and mark.layer == layer:
                return index, match["port"]
        return None

    def _context(
        self,
        port: str | None,
        edits: Sequence[PortEdit],
        committed: Sequence[_Mark],
        staged: bool,
        controller: PortController | None,
        row: int,
        layer: int | None,
        *,
        boundary: bool,
        cells: Sequence[CellEdit] = (),
    ) -> dict[str, Any]:
        """What ``_describe`` needs. ``boundary_edits`` are the ``h_out`` edits of the committed
        transition that produced this value (``L<i>.h_out`` and ``L<i+1>.h`` are one value)."""
        producer = committed[-1] if committed else None
        carried = (
            tuple(
                e for e in producer.edits if self.program.spec(e.port).name == "h_out"
            )
            if producer is not None and producer.phase == "layer" and (boundary or port == "h")
            else ()
        )
        return {
            "port": port, "edits": list(edits), "committed": list(committed), "staged": staged,
            "controller": controller, "row": row, "layer": layer, "boundary_edits": carried,
            "cells": list(cells),
        }

    def _read(
        self, target: str | None, *, value: bool = True
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """``(a private copy of the value at target, its context for _describe)``; with
        ``value=False`` only the context is built and the value is neither copied nor read."""
        key = self._key() if target is None else self.parse(target)
        if key > self._key():
            raise ValueError(f"{self.address(key)} has not been computed at {self.where}")
        mark = self._trail[-1]
        port = self._in_pending(key)
        if port is not None:
            controller = self._ensure_cache(self._port)
            context = self._context(
                port, self._edits, self._trail[:-1], True, controller, mark.row, mark.layer,
                boundary=False, cells=self._cells,
            )
            return (clone(controller.captures[port]) if value else None), context
        for index in range(len(self._trail) - 1, -1, -1):
            if self._mark_key(self._trail[index]) == key:
                context = self._context(
                    None, [], self._trail[:index], False, None, key[0], None, boundary=True
                )
                return (self._boundary_value(index) if value else None), context
        located = self._layer_mark(key)
        if located is not None:
            index, name = located
            past = self._trail[index]
            controller = self._committed_probe(index, name)
            context = self._context(
                name, past.edits, self._trail[:index], False, controller, past.row, past.layer,
                boundary=False, cells=past.cells,
            )
            return (clone(controller.captures[name]) if value else None), context
        raise ValueError(f"{self.address(key)} is before the start of this run")

    def value(self) -> torch.Tensor:
        return self.read()

    def _boundary_value(self, index: int) -> torch.Tensor:
        """A private copy of the slot that trail mark ``index`` reads (one slot, not the cut)."""
        mark = self._trail[index]
        slot = _SLOTS.get(mark.label, "hidden")
        if index == len(self._trail) - 1 and slot != "sampled_token":
            return self.session.read(slot)
        return self._cut_of(index).slot(slot)

    def set(
        self,
        target: str,
        value: Any,
        op: str = "replace",
        *,
        mask: torch.Tensor | None = None,
        selection: str | None = None,
        origin: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Stage one edit at a port of the pending layer; returns the edit's request manifest.

        ``mask`` (bool, broadcastable to the port) limits the op to the elements where it is
        true (the rest stay bit-identical); ``selection`` is an opaque string and ``origin`` the
        pointer of a value taken from another run, both recorded verbatim in the manifest.
        Validation is by shape only (the port's shape is derived, nothing past the cursor runs);
        whether the edit fires is checked when the layer commits. Atomic: on any failure,
        ``KeyboardInterrupt`` included, nothing stays staged.
        """
        key = self.parse(target)
        mark = self._trail[-1]
        label = self._stops[key[1]]
        if mark.phase != "layer" or key[0] != mark.row or not label.startswith(f"L{mark.layer}."):
            pending = "there is none" if mark.phase != "layer" else f"it is L{mark.layer}"
            raise ValueError(
                f"edits apply to the pending layer only; {self.address(key)} is outside it "
                f"({pending} at {self.where})"
            )
        port = label.split(".", 1)[1]
        spec = self.program.spec(port)
        if not spec.editable:
            raise ValueError(f"port {port} is not editable: {spec.edit}")
        edit = PortEdit(port, value, op, mask=mask, selection=selection, origin=origin)
        shape = self.port_shape(port)
        edit.validate(shape)
        if edit.mask is not None:
            edit = PortEdit(
                port,
                edit.value,
                op,
                mask=edit.mask.broadcast_to(shape).contiguous(),
                selection=selection,
                origin=origin,
            )
        saved_edits, saved_cache = list(self._edits), self._cache
        try:
            self._edits.append(edit)
            self._cache = None
            if self.lookahead == "layer" or self._port is not None:
                self._ensure_cache(self._port)
        except BaseException:
            self._edits[:] = saved_edits
            self._cache = saved_cache
            raise
        if isinstance(edit.value, torch.Tensor):
            self._values[edit.key] = edit.value
        if edit.mask is not None:
            self._masks[tensor_digest(edit.mask)] = edit.mask
        return edit.manifest()

    def _written_by(self, target: str) -> tuple[str, list[PortEdit], int] | None:
        """``(port, edits that wrote it, row)`` for a computed address whose value an edit can
        have written: a port of the pending layer (staged edits, plus the committed ``h_out``
        edits for ``h``), a port of a committed layer (the edits its mark recorded) or a layer
        boundary produced by a committed layer (that layer's ``h_out`` edits). None otherwise."""
        key = self.parse(target)
        mark = self._trail[-1]
        port = self._in_pending(key)  # staged edits may sit past the cursor
        if port is not None:
            here = [e for e in self._edits if self.program.spec(e.port).name == port]
            if port == "h":
                here = [*self._producer_h_out(len(self._trail) - 1), *here]
            return port, here, mark.row
        if key > self._key():
            return None
        for index in range(len(self._trail) - 1, -1, -1):
            if self._mark_key(self._trail[index]) == key:
                carried = self._producer_h_out(index)
                return ("h_out", carried, key[0]) if carried else None
        located = self._layer_mark(key)
        if located is not None:
            index, name = located
            past = self._trail[index]
            here = [e for e in past.edits if self.program.spec(e.port).name == name]
            if name == "h":
                here = [*self._producer_h_out(index), *here]
            return name, here, past.row
        return None

    def _producer_h_out(self, index: int) -> list[PortEdit]:
        """The ``h_out`` edits of the committed transition that produced boundary ``index``."""
        if index == 0:
            return []
        producer = self._trail[index - 1]
        if producer.phase != "layer":
            return []
        return [e for e in producer.edits if self.program.spec(e.port).name == "h_out"]

    def address_shape(self, target: str) -> tuple[int, ...] | None:
        """Shape of the port at a computed ``L<i>.<port>`` address in its own row (``src`` is
        that row's source length), or None for a global stop; executes nothing."""
        key = self.parse(target)
        match = _LAYER_PORT.fullmatch(self._stops[key[1]])
        if match is None:
            return None
        return self.program.shape(match["port"], self.adapter.model.config, src=key[0] + 1)

    def edit_mask(self, target: str) -> torch.Tensor | None:
        """Union mask of the elements written by edits at ``target``, staged or committed: None
        when no edit wrote that port (or the target is not an edited port), else a bool tensor
        of the port's shape in its own row (all true for an unmasked edit). A committed layer's
        port is served from the edits recorded on that occurrence's mark; ``h`` and the layer
        boundary include the ``h_out`` edits of the layer that produced them."""
        found = self._written_by(target)
        if found is None:
            return None
        port, here, row = found
        if not here:
            return None
        shape = self.program.shape(port, self.adapter.model.config, src=row + 1)
        union = torch.zeros(shape, dtype=torch.bool)
        for edit in here:
            selected = edit.selected(shape)
            union |= True if selected is None else selected
        return union

    def staged_info(self) -> list[dict[str, Any]]:
        """One record per staged edit: the request (``port``, ``op``, ``value`` descriptor,
        ``selection``, mask fields), ``layer``, ``fired`` and, once the current probe fired it,
        the realised fields (``realised_*``, ``cast_*``, ``changed_fraction``, ``mix``,
        ``replica_verified``, ``realised_at`` ...)."""
        cache = self._cache
        records = []
        for edit in self._edits:
            entry = None if cache is None else cache.fired.get(id(edit))
            record = {"selection": None, **edit.manifest(), "layer": self.pending_layer}
            record["fired"] = entry is not None
            if entry is not None:
                record.update(entry)
            records.append(record)
        return records

    def read_cache(self, layer: int, kind: str, *, staged: bool = True) -> torch.Tensor:
        """The key (``"k"``) or value (``"v"``) cache tensor of ``layer`` at the current boundary,
        read straight from the state slot (no capture, no fingerprinting). ``staged`` (default):
        with the staged state-cell edits on it written in staging order, as the next read of the
        cell will see it (``False``: the state slot as committed)."""
        if kind not in ("k", "v"):
            raise ValueError("kind must be 'k' or 'v'")
        if type(layer) is not int or not 0 <= layer < self.adapter.layers:
            raise ValueError(f"layer must be in 0..{self.adapter.layers - 1}")
        value = self.session.read("keys" if kind == "k" else "values", layer)
        if staged:
            for cell in self._cells:
                if cell.layer == layer and cell.tensor == kind:
                    value = cell.realise(value)[0]
        return value

    def set_cell(
        self,
        layer: int,
        tensor: str,
        value: Any,
        op: str = "replace",
        *,
        rows: Sequence[int],
        kv: Sequence[int] | None = None,
        d: Sequence[int] | None = None,
        dose: float | None = None,
        selection: str | None = None,
        origin: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Stage one state-cell edit of ``cache.L<layer>.<tensor>`` (``engine.edits.CellEdit``);
        returns its request manifest.

        The cell is the key/value row as later reads see it. The edit is realised natively in
        the next layer transition, by copy-on-write of the cache tensor that transition hands
        the layer, and takes effect for every read of it not yet made: when the cursor is in
        the cell's own layer at or past the read boundary (``kr`` for k, ``p`` for v) the
        current row's read has happened and the edit is realised at that layer's exit (timing
        ``after``); otherwise at the entry of the transition (``before``). Reads already made
        are not recomputed. The rows must already be written into the cache at the cursor (the
        row the pending layer writes becomes a cell when it commits). Atomic: on any failure,
        ``KeyboardInterrupt`` included, nothing stays staged.
        """
        if tensor not in ("k", "v"):
            raise ValueError("a cache cell is k or v")
        if type(layer) is not int or not 0 <= layer < self.adapter.layers:
            raise ValueError(f"layer must be in 0..{self.adapter.layers - 1}")
        mark = self._trail[-1]
        timing = "before"
        if mark.phase == "layer" and mark.layer == layer and self._port is not None:
            names = self.program.names
            if names.index(self._port) >= names.index(CELL_BOUNDARY[tensor]):
                timing = "after"
        edit = CellEdit(layer, tensor, tuple(rows), value, op, kv=None if kv is None else
                        tuple(kv), d=None if d is None else tuple(d), dose=dose, timing=timing,
                        staged_at=mark.transitions, selection=selection, origin=origin)
        current = self.read_cache(layer, tensor, staged=False)
        length = int(current.shape[2])
        if max(edit.rows) >= length and mark.phase == "layer" and layer == mark.layer \
                and max(edit.rows) == mark.row:
            raise ValueError(f"row {mark.row} of {edit.port} is written by the pending layer "
                             "transition; it becomes a cell when the layer commits")
        edit.check(current.shape)
        saved_cells, saved_cache = list(self._cells), self._cache
        try:
            self._cells.append(edit)
            self._cache = None
            if self.lookahead == "layer" or self._port is not None:
                self._ensure_cache(self._port)
        except BaseException:
            self._cells[:] = saved_cells
            self._cache = saved_cache
            raise
        if isinstance(edit.value, torch.Tensor):
            self._values[tensor_digest(edit.value)] = edit.value
        return edit.manifest()

    def staged_cells(self) -> list[dict[str, Any]]:
        """One record per staged cell edit: its manifest, ``fired`` (the current probe realised
        it: a ``before`` edit once a port of the pending layer is held) and, then, the
        realisation record (``realised_sha256``, ``cast_*``, ``realised_*``,
        ``changed_fraction``, ``count``, ``realised_timing``, ``realised_in_layer``)."""
        records = []
        cache = self._cache
        for index, cell in enumerate(self._cells):
            entry = None if cache is None else cache.cell_records.get(index)
            record = {**cell.manifest(), "fired": entry is not None}
            if entry is not None:
                record.update(entry)
            records.append(record)
        return records

    def committed_cells(self) -> list[dict[str, Any]]:
        """One record per committed (realised) cell edit, in commit order: the manifest, the
        realisation record and ``row``/``layer``/``transition`` of the transition that realised
        it (``layer`` is that transition's layer; the cell's own layer is ``cell.layer``)."""
        return [
            {**cell.manifest(), **record, "row": mark.row, "layer": mark.layer,
             "transition": mark.transitions}
            for mark in self._trail[:-1]
            for cell, record in zip(mark.cells, mark.cell_fired)
        ]

    def committed_edits(self) -> list[dict[str, Any]]:
        """One record per committed (fired) edit of this run, in commit order: ``row``, ``layer``,
        ``transition``, ``selection`` (None if absent) and the receipt's manifest fields
        (``port``, ``op``, ``value``, ``realised_*``, ``cast_*``, ``changed_fraction``, ``mix``,
        ``replica_verified``, mask fields)."""
        records = []
        for mark in self._trail[:-1]:
            for entry in mark.fired:
                records.append(
                    {
                        "selection": None,
                        **entry,
                        "row": mark.row,
                        "layer": mark.layer,
                        "transition": mark.transitions,
                    }
                )
        return records

    def original(self, target: str, *, recompute: bool = False) -> torch.Tensor | None:
        """The pre-edit value of an edited port.

        Pending layer: the controller's recorded original (None if no edit changed this port;
        ``r`` and ``p`` under an ``s`` edit have originals too). Committed layer: None unless
        ``recompute=True``, which re-probes from the retained cut with the edits at that port
        removed (later retention is never added; the run's state, receipts and trail are
        untouched). A ``h_out`` target returns None.
        """
        key = self.parse(target)
        if key > self._key():
            raise ValueError(f"{self.address(key)} has not been computed at {self.where}")
        port = self._in_pending(key)
        if port is not None:
            value = self._ensure_cache(self._port).originals.get(port)
            return None if value is None else clone(value)
        located = self._layer_mark(key)
        if located is None or not recompute:
            return None
        index, name = located
        if name == "h_out":
            return None
        past = self._trail[index]
        # everything that rewrites this port: p follows s, o follows r, A follows s and p
        drop = {"p": {"p", "s"}, "o": {"o", "r"}, "A": {"A", "p", "s"}}.get(name, {name})
        kept = tuple(e for e in past.edits if self.program.spec(e.port).name not in drop)
        if len(kept) == len(past.edits):
            return None
        controller = self._controller(
            past.layer, kept, past.cells, stop="p" if name == "s" else name, capture=(name,)
        )
        self._probe(self._session_at(index), controller)
        return clone(controller.captures[name])

    def row_captures(self, row: int) -> dict[str, torch.Tensor] | None:
        """Values already retained for ``row``: the boundary values of marks whose cut is held
        (keyed by stop label, e.g. ``emb``, ``L0.h_out``, ``logits``) plus, for the current row,
        the pending layer's cached ports (``L<i>.<port>``). None when nothing is retained."""
        out: dict[str, torch.Tensor] = {}
        last = len(self._trail) - 1
        for index, mark in enumerate(self._trail):
            if mark.row == row and (mark.cut is not None or index == last):
                out[mark.label] = self._boundary_value(index)  # already a private copy
        if self._cache is not None and row == self._trail[-1].row and self.pending_layer is not None:
            names = self.program.names
            limit = -1 if self._port is None else names.index(self._port)
            for port, value in self._cache.captures.items():
                if names.index(port) <= limit:  # never a value past the cursor, in either mode
                    out[f"L{self.pending_layer}.{port}"] = clone(value)
        return out or None

    # --- checkpoints and rewind ---------------------------------------------------------

    @property
    def edited(self) -> bool:
        """True when an edit (port or state cell) is staged, or was committed on the path to
        the cursor: values read here are addressed ``@cut+edit`` (``origin_of``)."""
        return bool(self._edits or self._cells) or any(
            mark.edits or mark.cells for mark in self._trail[:-1])

    def checkpoint(self, name: str | None = None, *, pin: bool = True) -> dict[str, Any]:
        """The cursor as plain data (``restore`` input). ``pin``: keep the edit tensors it names
        held for its restore; False when they are stored elsewhere (a project recipe), so the
        run keeps nothing for it."""
        mark = self._trail[-1]
        if pin:
            self._pinned.update(self.edit_tensors())  # the checkpoint names them
        return {
            "schema": CHECKPOINT_SCHEMA,
            "name": name,
            "model_identity": self.adapter.model_identity,
            "prompt_sha256": digest(self.tokens),
            "where": self.where,
            "cut": mark.fingerprint,
            "transitions": mark.transitions,
            "port": self._port,
            "lookahead": self.lookahead,
            "numeric_scope": self._numeric_scope(),
            "edits": [*(edit.manifest() for edit in self._edits),
                      *(cell.manifest() for cell in self._cells)],
            "path_edits": self._path_manifest(len(self._trail) - 1),
        }

    def _path_manifest(self, index: int) -> list[dict[str, Any]]:
        """The committed edits on the path to trail mark ``index`` (as a checkpoint lists them):
        port edits then the state-cell edits realised in that transition."""
        return [
            {
                "transition": item.transitions,
                "layer": item.layer,
                "edits": [*(edit.manifest() for edit in item.edits),
                          *(cell.manifest() for cell in item.cells)],
            }
            for item in self._trail[:index]
            if item.edits or item.cells
        ]

    def edit_tensors(self) -> dict[str, torch.Tensor]:
        """Every tensor the edits of ``checkpoint()`` name, by payload sha256 (edit values and
        masks, committed on the path and staged): what a durable checkpoint must store."""
        out: dict[str, torch.Tensor] = {}
        for edit in [*(e for mark in self._trail[:-1] for e in mark.edits), *self._edits]:
            if isinstance(edit.value, torch.Tensor):
                out[tensor_digest(edit.value)] = edit.value
            if edit.mask is not None:
                out[tensor_digest(edit.mask)] = edit.mask
        for cell in [*(c for mark in self._trail[:-1] for c in mark.cells), *self._cells]:
            if isinstance(cell.value, torch.Tensor):
                out[tensor_digest(cell.value)] = cell.value
        return out

    def hold(self, tensors: dict[str, torch.Tensor]) -> None:
        """Hold edit values and masks (payload sha256 -> tensor) so ``restore`` can rebuild edits
        this process did not stage, e.g. a checkpoint read back from a project store. Every
        digest is checked first; nothing is held when one differs."""
        for sha, value in tensors.items():
            if tensor_digest(value) != sha:
                raise ValueError(f"tensor held as {sha[:12]} has another payload digest")
        for sha, value in tensors.items():
            (self._masks if value.dtype == torch.bool else self._values)[sha] = value
            self._pinned.add(sha)

    def _sweep_tensors(self) -> None:
        """Drop the held edit tensors no edit of the trail or the staged set uses and no
        checkpoint (``checkpoint(pin=True)``) or ``hold`` named: an aborted,
        rewound or replaced edit leaves nothing behind."""
        used = set(self.edit_tensors()) | self._pinned
        for table in (self._values, self._masks):
            for sha in [sha for sha in table if sha not in used]:
                del table[sha]

    def origin_of(self, target: str) -> dict[str, Any]:
        """Where the value at a computed address comes from (what an occurrence pointer records).

        ``cut``/``transitions``: the boundary mark the value is read at (``port`` None) or whose
        layer transition computes it (``port`` the port name); ``edits``: that transition's
        edit manifests (the staged ones for the pending layer, ``[]`` for a boundary value).
        """
        key = self.parse(target)
        if key > self._key():
            raise ValueError(f"{self.address(key)} has not been computed at {self.where}")
        port = self._in_pending(key)
        if port is not None:
            mark = self._trail[-1]
            return {"cut": mark.fingerprint, "transitions": mark.transitions, "port": port,
                    "edits": [*self.edits, *self.cells]}
        for mark in reversed(self._trail):
            if self._mark_key(mark) == key:
                return {"cut": mark.fingerprint, "transitions": mark.transitions, "port": None,
                        "edits": []}
        located = self._layer_mark(key)
        if located is None:
            raise ValueError(f"{self.address(key)} is before the start of this run")
        index, name = located
        past = self._trail[index]
        return {"cut": past.fingerprint, "transitions": past.transitions, "port": name,
                "edits": [*(edit.manifest() for edit in past.edits),
                          *(cell.manifest() for cell in past.cells)]}

    @property
    def numeric_scope(self) -> dict[str, Any]:
        """A copy of the numeric scope this run executes under (``adapter.execution
        ["numeric_scope"]``): thread count, determinism, matmul precision, TF32 and the CUDA
        reduced-precision reduction flags ``bf16_reduced_precision_reduction`` /
        ``fp16_reduced_precision_reduction`` (True: cuBLAS may accumulate bf16/fp16 matmuls in
        reduced precision; ``"absent"`` on a torch without the flag). Bound when the adapter
        was built and checked on every native call, so it is the scope of every value read;
        reading it touches no torch global."""
        return self._numeric_scope()

    def _numeric_scope(self) -> dict[str, Any]:
        """The adapter's bound numeric scope (thread count, determinism, matmul precision,
        reduced-precision reduction flags).

        A cut's fingerprint binds this scope, so a checkpoint replays only under the same one.
        """
        execution = getattr(self.adapter, "execution", None) or {}
        return dict(execution.get("numeric_scope") or {})

    def _refuse_other_numeric_scope(self, checkpoint: dict[str, Any]) -> None:
        recorded = checkpoint.get("numeric_scope")
        if not isinstance(recorded, dict):
            return  # a checkpoint written before the scope was recorded
        mine = self._numeric_scope()
        differing = sorted(
            name for name in set(recorded) | set(mine) if recorded.get(name) != mine.get(name)
        )
        if differing:
            detail = ", ".join(
                f"{name}: checkpoint {recorded.get(name)!r}, this process {mine.get(name)!r}"
                for name in differing
            )
            raise ValueError(
                "checkpoint was taken under another numeric scope "
                f"({detail}); a cut binds the numeric scope, so set the same scope and rewrap "
                "the model before restoring"
            )

    def _edit_from(self, manifest: dict[str, Any]) -> PortEdit | CellEdit:
        """The edit a checkpoint manifest describes (a ``CellEdit`` for a state-cell manifest),
        with its tensors from what this run holds."""
        described = manifest["value"]
        if described["kind"] == "scalar":
            value: Any = described["value"]
        else:
            value = self._values.get(described["sha256"])
            if value is None:
                raise ValueError(
                    f"edit value {described['sha256'][:12]} at {manifest['port']} is not held by "
                    "this process; restart it from a project (Debugger.restart) or Run.hold it"
                )
        if is_cell(manifest):
            return CellEdit.from_manifest(manifest, value)
        mask = None
        if "mask_sha256" in manifest:
            mask = self._masks.get(manifest["mask_sha256"])
            if mask is None:
                raise ValueError(
                    f"edit mask {manifest['mask_sha256'][:12]} at {manifest['port']} is not held "
                    "by this process; restart it from a project (Debugger.restart) or Run.hold it"
                )
        return PortEdit(
            manifest["port"], value, manifest["op"], mask=mask, selection=manifest.get("selection"),
            origin=manifest.get("origin"),
        )

    def restore(self, checkpoint: dict[str, Any], *, defer_interrupt: bool = False) -> str:
        """Put the cursor, trail and staged edits back to ``checkpoint`` (all or nothing).

        ``defer_interrupt``: a ``KeyboardInterrupt`` raised while restoring is remembered, the
        restore is run again until it completes, and the interrupt is re-raised afterwards. For
        a restore that undoes a temporary excursion (``whatif``): without it an interrupt
        rolls the restore back, leaving the run where the excursion left it.
        """
        if not defer_interrupt:
            return self._restore_atomic(checkpoint)
        interrupted = None
        while True:
            try:
                where = self._restore_atomic(checkpoint)
                break
            except KeyboardInterrupt as error:
                interrupted = error
        if interrupted is not None:
            raise interrupted
        return where

    def _restore_atomic(self, checkpoint: dict[str, Any]) -> str:
        snapshot = _Snapshot(self)
        try:
            where = self._restore(checkpoint)
        except BaseException:
            snapshot.rollback()  # all or nothing, replay and interrupts included
            raise
        self._sweep_tensors()
        return where

    def _restore(self, checkpoint: dict[str, Any]) -> str:
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("not an mdb engine checkpoint")
        if checkpoint["model_identity"] != self.adapter.model_identity:
            raise ValueError("checkpoint belongs to another model identity")
        if checkpoint["prompt_sha256"] != digest(self.tokens):
            raise ValueError("checkpoint belongs to another prompt")
        self._refuse_other_numeric_scope(checkpoint)
        self._drop_reads()
        staged = [self._edit_from(manifest) for manifest in checkpoint["edits"]]
        # A mark stands for the checkpoint only with the same cut AND the same committed edits
        # before it: an edit whose result is bit-identical (``add 0``) leaves the cut unchanged
        # but must not survive the restore.
        found = [i for i, m in enumerate(self._trail) if m.fingerprint == checkpoint["cut"]
                 and self._path_manifest(i) == checkpoint["path_edits"]]
        if found:
            self._restore_mark(found[-1])
        else:
            path = {
                entry["transition"]: tuple(self._edit_from(m) for m in entry["edits"])
                for entry in checkpoint["path_edits"]
            }
            self._restore_mark(0)
            while self._trail[-1].transitions < checkpoint["transitions"]:
                here = path.get(self._trail[-1].transitions, ())
                self._edits = [e for e in here if isinstance(e, PortEdit)]
                self._cells = [e for e in here if isinstance(e, CellEdit)]
                self._commit()
            if self._trail[-1].fingerprint != checkpoint["cut"]:
                raise ReplayMismatch(checkpoint["cut"], self._trail[-1].fingerprint)
        self._edits = [e for e in staged if isinstance(e, PortEdit)]
        self._cells = [e for e in staged if isinstance(e, CellEdit)]
        self._cache, self._port = None, None
        if checkpoint["port"] is not None:
            port = self.program.spec(checkpoint["port"]).name
            self._ensure_cache(port)
            self._port = port
        return self.where

    def reverse_step(self, unit: str = "port", n: int = 1) -> str:
        accept = self._unit(unit)
        if type(n) is not int or n < 1:
            raise ValueError("n must be a positive integer")
        root = self._mark_key(self._trail[0])
        self._drop_reads()
        for _ in range(n):
            key = self._previous(self._key())
            while key >= root and not accept(self._stops[key[1]]):
                key = self._previous(key)
            if key < root:
                raise ValueError(f"cannot reverse before the run root {self.address(root)}")
            self._goto(key)
        self._sweep_tensors()
        return self.where

    def _retime_cells(self, port: str | None) -> None:
        """After the cursor moved back to ``port`` (None: a boundary): a staged ``after`` cell
        whose boundary read (``kr``/``p``) is no longer behind the cursor becomes ``before``,
        as if it were staged here. Only downgrades: a cell staged ``before`` stays ``before``."""
        mark, names = self._trail[-1], self.program.names
        retimed = []
        for cell in self._cells:
            behind = (port is not None and mark.phase == "layer" and mark.layer == cell.layer
                      and names.index(port) >= names.index(CELL_BOUNDARY[cell.tensor]))
            retimed.append(replace(cell, timing="before")
                           if cell.timing == "after" and not behind else cell)
        if any(a is not b for a, b in zip(retimed, self._cells)):
            self._cells, self._cache = retimed, None

    def _goto(self, key: tuple[int, int]) -> None:
        if key >= self._key():
            self._forward_to(key)
            return
        port = self._in_pending(key)
        if port is not None:
            self._retime_cells(port)
            self._ensure_cache(port)
            self._port = port
            return
        index = max(i for i, m in enumerate(self._trail) if self._mark_key(m) <= key)
        self._restore_mark(index)
        if self._mark_key(self._trail[-1]) != key:
            port = self._in_pending(key)
            if port is None:
                raise RuntimeError(f"cannot place the cursor at {self.address(key)}")
            self._retime_cells(port)
            self._ensure_cache(port)
            self._port = port
        else:
            self._retime_cells(None)

    def _restaged_cells(self, index: int) -> list[CellEdit]:
        """The cell edits that were staged at trail mark ``index``: those realised by the first
        committed layer transition at or after it whose staging point is not after the mark."""
        at = self._trail[index].transitions
        for mark in self._trail[index:-1]:
            if mark.phase == "layer":
                return [cell for cell in mark.cells if cell.staged_at <= at]
        return [cell for cell in self._cells if cell.staged_at <= at]

    def _restore_mark(self, index: int) -> None:
        current = index == len(self._trail) - 1
        edits = list(self._edits) if current else list(self._trail[index].edits)
        cells = list(self._cells) if current else self._restaged_cells(index)
        if not current:
            snapshot = _Snapshot(self)
            try:
                cut = self._cut_of(index)
                self.session.restore(cut)
                self._trail = self._trail[: index + 1]
                mark = self._trail[index]
                mark.cut, mark.edits, mark.fired = cut, (), ()
                mark.cells, mark.cell_fired = (), ()
                self._cache = None
                self._drop_reads()
                self._edits, self._cells, self._port = edits, cells, None
            except BaseException:
                snapshot.rollback()
                raise
            return
        self._edits, self._cells, self._port = edits, cells, None

    def _cut_of(self, index: int) -> StateCut:
        mark = self._trail[index]
        if mark.cut is not None:
            return mark.cut
        base = max(i for i in range(index) if self._trail[i].cut is not None)
        session = Session.from_cut(self.adapter, self._trail[base].cut)
        for item in self._trail[base:index]:
            self._advance(session, item.layer, item.edits, item.cells)
        cut = session.capture(retain=False)
        if cut.fingerprint != mark.fingerprint:
            raise RuntimeError("replay from a retained cut did not reproduce the boundary")
        return cut

    def _session_at(self, index: int) -> Session:
        """A Session at trail mark ``index`` for probing only: the run's own session for the
        last mark, else the hydrated Session for that cut (reused while the fingerprint matches;
        ``Session.probe`` leaves it unchanged)."""
        if index == len(self._trail) - 1:
            return self.session
        fingerprint = self._trail[index].fingerprint
        if self.cache_reads and self._hydrated is not None and self._hydrated[0] == fingerprint:
            return self._hydrated[1]
        session = Session.from_cut(self.adapter, self._cut_of(index))
        if self.cache_reads:
            self._hydrated = (fingerprint, session)
        return session

    def _prune(self) -> None:
        receipts = self.session.receipts
        if len(receipts) > _RECEIPTS:
            del receipts[: len(receipts) - _RECEIPTS]
        if self.retain_rows is None:
            return
        horizon = self._trail[-1].row - self.retain_rows
        for mark in self._trail[1:-1]:
            if mark.row < horizon and mark.label != "start" and mark.cut is not None:
                mark.cut = None
                self._drop_reads()

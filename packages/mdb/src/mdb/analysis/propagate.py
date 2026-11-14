"""What-if: the change one edit makes in a downstream consumer, predicted at the cursor, then
measured natively, with the session left exactly where it was.

``whatif(run, edit, consumer)`` works in three steps.

1. **Prediction**, from values that exist at the cursor and the weights only (no native call):
   the row model of ``forecast`` (``RowModel``) is evaluated over the rows from the edit's row to
   the consumer's row, once without the edit (``base``) and once with the requested edit written
   at the edited port (``edited``); later rows read the earlier rows' keys and values from the
   same chain, and a sampled token is the chain's own argmax. ``exact = edited - base`` is
   ``derived-exact`` with a rounding ``bound`` (the forecast bounds plus the cast of the edit
   operand), and the first-order term ``linear = J delta`` (forward-mode derivative of the same
   chain at the unedited point, P14) is ``derived-local``. ``exact`` is ``F(x1) - F(x0)`` (P09
   section 1): both chains hold the edited port at ``x0`` or write the edit there and recompute
   the same stops (those the edit reaches) from the same operands; the base is never the native
   value (that would fold the base run's rounding into the exact change). The prediction is
   sealed as a record (kind ``whatif``) before anything native runs. When the consumer is ahead
   of the cursor, the operands past the cursor exist only as the chain's own float64 values: the
   sealed exact change says so (``prediction.exact.operands.source == "forecast"``). Beside them
   the **emulated** chains (``EmulatedRowModel``, forecast's premises: the dtype's rounding where
   eager HF rounds, backend kernels on the model's device) give the emulated base and edited
   values; their difference predicts the native change itself (``prediction.emulated``,
   ``derived-exact`` under those premises). The edit is written there by the engine's own
   arithmetic (``PortEdit.apply``, the cell write) on the emulated value. A staged edit ahead of
   the cursor refuses the emulated chains (they would not see it); a refused emulation leaves the
   rest of the prediction as it was. The emulated base does not depend on the edit: the what-ifs
   of one ``whatif_many`` call share it (per first row, consumer and cursor).
2. **Measurement**: checkpoint, stage the edit (moving to the edited layer first: forward, or back
   by layer boundaries when the layer is already committed), run natively to the consumer, read
   it, restore; the consumer's unedited value is read the same way. Labelled ``measured``. With
   the consumer ahead of the cursor the edited branch runs first, then the unedited branch to
   the consumer, where the exact chain is computed again on the measured operands (``report
   ["exact"]``, after the seal, never presented as sealed).
3. **Split** of the native change (P20): ``native = linear + nonlinear [+ operands] +
   realisation + rounding`` with ``nonlinear = exact - linear``, ``operands`` (forecast seal
   only) the exact change on measured operands minus the sealed one, ``realisation`` the change
   of the exact value on measured operands when the edit operand is the value the edit actually
   wrote (read back natively) instead of the requested one (all ``derived-exact``), ``rounding
   effect = emulated - exact(written, measured operands)`` (``derived-exact`` under the emulation
   premises; the worst-case interval ``bound`` on the native rounding is kept beside it only when
   finite) and ``residual = native - emulated`` (``measured``: 0 when the emulation holds).
   Without an emulated change the last two are one term, ``rounding = native - exact``
   (``measured``). The linear number is never reported as the effect.
4. **Verdict** (``report["effect"]``, ``verdict.decide``): the native change against the
   emulated one within the floor of ``verdict.FLOOR_SPACINGS`` (1, supplied, the constant verify
   uses) dtype spacings at the consumer. ``effect`` when they agree and the native change is not
   zero; ``no effect`` when they agree and it is zero (the note gives the exact change the dtype
   cannot show); beyond the floor verify's rule: ``not confirmed`` when a stage of the edited or
   base run lies outside the bound of its identity on its native operands (the parting stop
   named), else ``unresolved`` (rounding drift, or a chain over several rows: not stage-checked).
   With no emulated change (refused): ``effect`` iff the native edited and base values differ.
   The exact change is reported beside the verdict, never decides it. The result's ``state`` is
   the verdict's; the interval bound never sets it.

A state-cell edit (``Edit.cell``: ``cache.L<l>.k|v[kv,row,d]``, ops ``replace | add | scale |
mix``) is staged where the cursor is when the consumer is still ahead (``pending``), else at
the entry ``t<c>.L<l>.h`` of the consumer's row (``rewind``). It reaches every read of the cell
not yet made: the first is the cell layer's ``kr`` (keys) or ``A`` (values) of the cursor's row
when that read is still ahead (``first_read``), else of the next row, and the same read in every
later row. The prediction writes the cell into the cache rows the chain reads (float64, the
requested operation unrounded; the written value's distance is the edit's bound) and refuses
when the first affected row has not started at the cursor; the linear term is the forward-mode
derivative through the attention read (a value cell enters ``A`` linearly, P06/P07).

The session's ``state_digest`` (cursor, cut fingerprint, transition count, staged and committed
edits, lookahead state) is taken before and after; a difference raises. Every edit tensor the
checkpoint names is held (``Run.hold``) before anything native runs, so the restore can rebuild
them; a checkpoint naming a tensor the run does not hold is refused first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mdb.core.kinds import WHATIF
from mdb.core.pointers import seal_digest
from mdb.core.bounded import MAX_ITEMS, bound_list, load_json
from mdb.core.records import num, seal_record

from .accumulation import bound_state
from .finite import BOUND, EXACT, LOCAL, MEASURED, SUPPLIED, f64
from .forecast import (EMULATION_PREMISE, EMULATION_PREMISES, EmulatedRowModel, Logit, Margin,
                       Refusal, RowModel, _split, _staged_between, producers, run_session_id,
                       stage_parting)
from .verdict import FLOOR_SPACINGS, NOT_CONFIRMED, decide, magnitude, spacing

SCHEMA = "mdb-whatif-v1"
OPS = ("replace", "add", "scale")
CELL_OPS = ("replace", "add", "scale", "mix")
CELL_READ = {"k": "kr", "v": "A"}       # mdb.engine.edits.CELL_READ (engine is torch-heavy)
CELL_BOUNDARY = {"k": "kr", "v": "p"}   # mdb.engine.edits.CELL_BOUNDARY


def _torch():
    import torch

    return torch


@dataclass(frozen=True)
class Edit:
    """A requested edit at ``address`` (``t<row>.L<i>.<port>``): ``value`` a float or a tensor
    in the port's run shape, ``op`` one of ``replace | add | scale``, ``mask`` an optional bool
    tensor (broadcastable to the port) limiting the op, ``selection``/``source`` the texts it came
    from (recorded verbatim)."""

    address: str
    value: Any
    op: str = "replace"
    mask: Any = None
    selection: str | None = None
    source: str | None = None
    cell: dict[str, Any] | None = None   # {layer, tensor, rows, kv, d}: a state-cell edit
    dose: float | None = None            # mix only
    origin: dict[str, Any] | None = None  # pointer of a value from another run


@dataclass
class Plan:
    """The resolved request: edit site, consumer, the stops each row's edit reaches, and how the
    measurement reaches the edited layer (``pending``: stage now; ``ahead``: run forward to the
    layer; ``rewind``: back by layer boundaries, then forward)."""

    row: int
    label: str
    key: tuple[int, int]
    consumer_row: int
    consumer_label: str
    consumer_key: tuple[int, int]
    consumer_text: str
    margin: Margin | None
    logit: Logit | None
    affected: dict[int, set[str]]
    here: tuple[int, int]
    mode: str
    stage_at: str | None = None  # a cell edit: the cursor it is staged at

    @property
    def edit_stop(self) -> str:
        return f"t{self.row}.{self.label}"

    @property
    def consumer_stop(self) -> str:
        return f"t{self.consumer_row}.{self.consumer_label}"


def state_digest(run: Any) -> str:
    """Digest of what a what-if must leave unchanged: ``Run.checkpoint()`` without its name
    (cursor, cut fingerprint, transition count, staged edits, committed path edits, lookahead,
    numeric scope) and ``Run.status()``."""
    checkpoint = {key: value for key, value in run.checkpoint().items() if key != "name"}
    return seal_digest({"checkpoint": checkpoint, "status": run.status()})


# --------------------------------------------------------------------------- planning


def affected(run: Any, row: int, label: str, last_row: int, *, every_row: bool = False
             ) -> dict[int, set[str]]:
    """Stops of rows ``row..last_row`` whose value an edit at ``t<row>.<label>`` can change: the
    stops downstream of it in its own row, and in each later row the stops downstream of a cache
    row it changed (``L<i>.kr`` reads the key cache, ``L<i>.A`` the value cache) or of a sampled
    token it changed (``emb`` of a row past the prompt). ``every_row``: ``label`` is a cache
    read every later row makes again (a state-cell edit), so it seeds every row."""
    program, layers = run.program, run.adapter.layers
    children: dict[str, list[str]] = {}
    for stop in run.stops():
        for name in producers(program, layers, stop):
            children.setdefault(name, []).append(stop)

    def closure(seeds: set[str]) -> set[str]:
        found, todo = set(seeds), list(seeds)
        while todo:
            for child in children.get(todo.pop(), ()):
                if child not in found:
                    found.add(child)
                    todo.append(child)
        return found

    out = {row: closure({label})}
    keys: set[int] = set()
    values: set[int] = set()
    for j in range(row + 1, last_row + 1):
        previous = out[j - 1]
        keys |= {i for i in range(layers) if f"L{i}.kr" in previous}
        values |= {i for i in range(layers) if f"L{i}.v" in previous}
        seeds = {f"L{i}.kr" for i in keys} | {f"L{i}.A" for i in values}
        if every_row:
            seeds.add(label)
        if "sample" in previous and j >= len(run.tokens):
            seeds.add("emb")
        out[j] = closure(seeds)
    return out


def _consumer(run: Any, consumer: Any) -> tuple[int, str, Margin | None, Logit | None, str]:
    vocab = int(run.adapter.model.config.vocab_size)
    if isinstance(consumer, Margin):
        if not (0 <= consumer.a < vocab and 0 <= consumer.b < vocab) or consumer.a == consumer.b:
            raise Refusal(f"M[a,b] needs two different token ids in 0..{vocab - 1}")
        return consumer.row, "logits", consumer, None, consumer.text
    if isinstance(consumer, Logit):
        if not 0 <= consumer.i < vocab:
            raise Refusal(f"logit index must be in 0..{vocab - 1}")
        return consumer.row, "logits", None, consumer, consumer.text
    key = run.parse(consumer)
    label = run.stops()[key[1]]
    if label in ("start", "sample"):
        raise Refusal(f"{label} is a token id, not a value with a change; use a margin M[a,b] "
                      "or a logit as the consumer")
    return key[0], label, None, None, f"t{key[0]}.{label}"


def _read_ahead(run: Any, layer: int, tensor: str) -> bool:
    """Whether the cursor's row has still to read ``cache.L<layer>.<tensor>`` (an edit staged
    now is realised before that read)."""
    status = run.status()
    here = run.parse(run.where)
    if status["pending_layer"] == layer and status["port"] is not None:
        names = run.program.names
        return names.index(status["port"]) < names.index(CELL_BOUNDARY[tensor])
    return here < run.parse(f"t{here[0]}.L{layer}.{CELL_READ[tensor]}")


def _later_edits(run: Any, row: int, layer: int) -> str | None:
    """A committed edit or cell edit after ``(row, layer)``: rewinding there would drop it."""
    later = [e for e in run.committed_edits() if (int(e["row"]), int(e["layer"])) > (row, layer)]
    later += [e for e in run.committed_cells() if (int(e["row"]), int(e["layer"])) >= (row, layer)]
    if later:
        first = later[0]
        return (f"the committed edit at t{first['row']}.L{first['layer']} ({first['port']}) comes "
                "after the stage point: stepping back to it would drop that edit from the "
                "measured run")
    return None


def cell_plan(run: Any, edit: Edit, consumer: Any) -> Plan:
    """Resolve and check a state-cell what-if without running anything native: where it is
    staged, its first read, the stops it reaches and whether the consumer is among them."""
    if edit.op not in CELL_OPS:
        raise Refusal(f"a cell op is one of {', '.join(CELL_OPS)}")
    cell = edit.cell
    layer, tensor, rows = int(cell["layer"]), cell["tensor"], list(cell["rows"])
    read = f"L{layer}.{CELL_READ[tensor]}"
    row_c, label_c, margin, logit, text = _consumer(run, consumer)
    key_c = run.parse(f"t{row_c}.{label_c}")
    here = run.parse(run.where)
    if key_c > here:
        mode, stage_at = "pending", run.where
        first = here[0] if _read_ahead(run, layer, tensor) else here[0] + 1
        length = int(run.read_cache(layer, tensor, staged=False).shape[2])
        if max(rows) >= length:
            raise Refusal(f"{edit.address}: row {max(rows)} is not in the cache at {run.where} "
                          f"(it holds rows 0..{length - 1})")
    else:
        mode, first = "rewind", row_c
        stage_at = f"t{row_c}.L{layer}.h"
        if max(rows) >= row_c:
            raise Refusal(f"{text} is computed at the cursor and its row t{row_c} reads cache "
                          f"rows 0..{row_c - 1} only: the cell rows {rows} cannot reach it")
        if run.edits or run.cells:
            raise Refusal("the consumer is already computed and edits are staged: reaching the "
                          "stage point means stepping back, which drops the staged edits; commit "
                          "or abort them first")
        conflict = _later_edits(run, row_c, layer)
        if conflict is not None:
            raise Refusal(conflict)
    reach = affected(run, first, read, max(row_c, first), every_row=True)
    if row_c < first or label_c not in reach[row_c]:
        raise Refusal(f"{text} is not downstream of {edit.address} staged at {stage_at}: its "
                      f"first read is t{first}.{read}")
    return Plan(first, read, run.parse(f"t{first}.{read}"), row_c, label_c, key_c, text, margin,
                logit, reach, here, mode, stage_at)


def plan(run: Any, edit: Edit, consumer: Any) -> Plan:
    """Resolve and check a what-if request without running anything native."""
    if edit.cell is not None:
        return cell_plan(run, edit, consumer)
    if edit.op not in OPS:
        raise Refusal(f"op must be one of {', '.join(OPS)}")
    key = run.parse(edit.address)
    label = run.stops()[key[1]]
    if not label.startswith("L"):
        raise Refusal(f"edits apply to layer ports, not {label}")
    spec = run.program.spec(_split(label)[1])
    if not spec.editable:
        raise Refusal(f"port {spec.name} is not editable: {spec.edit}")
    row_c, label_c, margin, logit, text = _consumer(run, consumer)
    key_c = run.parse(f"t{row_c}.{label_c}")
    if key_c <= key:
        raise Refusal(f"{text} is not downstream of the edit at t{key[0]}.{label}: it is "
                      "computed before (or is) the edited port")
    reach = affected(run, key[0], label, row_c)
    if label_c not in reach[row_c]:
        raise Refusal(f"{text} is not downstream of the edit at t{key[0]}.{label}: no path of "
                      "the port program (or of the cache / sampled token) leads from it")
    here = run.parse(run.where)
    status = run.status()
    layer = _split(label)[0]
    entry = run.parse(f"t{key[0]}.L{layer}.h")
    if status["port"] is not None and status["pending_layer"] == layer and here[0] == key[0]:
        mode = "pending"
    elif entry > here:
        mode = "ahead"
    else:
        mode = "rewind"
        if run.edits:
            raise Refusal("the edited layer is already committed and edits are staged: reaching "
                          "it means stepping back, which drops the staged edits; commit or abort "
                          "them first")
        later = [e for e in run.committed_edits()
                 if (int(e["row"]), int(e["layer"])) > (key[0], layer)]
        if later:
            first = later[0]
            raise Refusal(f"the committed edit at t{first['row']}.L{first['layer']}."
                          f"{first['port']} comes after the edited layer: stepping back to it "
                          "would drop that edit from the measured run")
    return Plan(key[0], label, key, row_c, label_c, key_c, text, margin, logit, reach, here,
                mode)


# --------------------------------------------------------------------------- prediction


def _chain(run: Any, p: Plan, *, bounds: bool, overrides: dict[str, Any] | None = None,
           skip: bool = False, tokens: dict[int, int] | None = None,
           cell: Any = None, model: type = RowModel) -> list[RowModel]:
    """Row models for rows ``p.row .. p.consumer_row``. Stops at or before the cursor are taken
    as captured, except (``skip``) the ones the edit reaches; row ``j > p.row`` reads the cache
    rows ``p.row..j-1`` from the chain and its token from the prompt, from ``tokens`` or from the
    chain's argmax of row ``j - 1``. ``cell(layer, kind, value, spread)`` rewrites the cache
    rows ``0..p.row-1`` every row reads (a state-cell edit; the first row then reads them
    through it too). ``model``: ``RowModel`` (float64) or ``EmulatedRowModel`` (the dtype's
    rounding, no bounds)."""
    torch = _torch()
    models: list[RowModel] = []
    here = p.here
    rows: dict[tuple[int, str], tuple[Any, Any]] = {}   # cache rows 0..p.row-1, cast once

    def known_for(j: int):
        drop = p.affected.get(j, set()) if skip else set()

        def known(name: str) -> Any:
            if name == "start" or name in drop or run.parse(f"t{j}.{name}") > here:
                return None
            return run.read(f"t{j}.{name}")

        return known

    def cached(layer: int, kind: str) -> tuple[Any, Any]:
        """The cache rows every row of the chain reads (``cell`` written in), read and cast
        once per chain."""
        if (layer, kind) not in rows:
            base = f64(run.read_cache(layer, kind)[..., : p.row, :])[0]
            spread0 = torch.zeros_like(base)
            if cell is not None:
                base, spread0 = cell(layer, kind, base, spread0)
            rows[layer, kind] = base, spread0
        return rows[layer, kind]

    for j in range(p.row, p.consumer_row + 1):
        earlier = list(models)

        def past(layer: int, kind: str, earlier=earlier) -> Any:
            base, spread0 = cached(layer, kind)
            parts, spreads = [base], [spread0]
            for model in earlier:
                name = f"L{layer}.kr" if kind == "k" else f"L{layer}.v"
                value, spread = yield from model.need(name)
                shape = (model.kv_heads, -1, model.head_dim)
                value = value.reshape(shape)[:, -1:, :]
                parts.append(value)
                spreads.append(None if spread is None else spread.reshape(shape)[:, -1:, :])
            if any(s is None for s in spreads):
                return torch.cat(parts, 1), None
            return torch.cat(parts, 1), torch.cat(spreads, 1)

        def token(j=j, earlier=earlier) -> Any:
            if j < len(run.tokens):
                return run.tokens[j], None
            if tokens is not None:
                return tokens[j], None
            previous = earlier[-1]
            value, spread = yield from previous.need("sample")
            sampled = int(value)
            if bounds and spread is None:
                return sampled, (f"the sampled token of t{j - 1} is not certified: "
                                 + previous.unbounded.get("sample", "no bound on its logits"))
            return sampled, None

        first = j == p.row
        models.append(model(run, j, known_for(j), bounds=bounds,
                            overrides=(overrides or {}) if first else None,
                            past=None if first and cell is None else past,
                            token=None if first else token))
    return models


def _consumer_value(p: Plan, model: RowModel) -> tuple[Any, Any]:
    value, bound = model.get(p.consumer_label), model.bound(p.consumer_label)
    if p.margin is not None:
        value = value[p.margin.a] - value[p.margin.b]
        bound = None if bound is None else bound[p.margin.a] + bound[p.margin.b]
    elif p.logit is not None:
        value = value[p.logit.i]
        bound = None if bound is None else bound[p.logit.i]
    return value, bound


def _sampled_tokens(p: Plan, models: list[RowModel], run: Any) -> dict[int, int]:
    return {j: int(models[j - p.row - 1].get("sample"))
            for j in range(p.row + 1, p.consumer_row + 1) if j >= len(run.tokens)}


def _request(run: Any, edit: Edit, shape: tuple[int, ...], x0: Any, b0: Any, u: float
             ) -> tuple[Any, Any, Any]:
    """``(requested, delta, bound)`` at the edited port in float64: the exact result of
    ``x0 op value`` on the selected elements and its rounding bound against the value the native
    edit writes (the operand cast to the model dtype, one rounded op; a ``mix`` is float64
    rounded once and casts no operand; ``b0`` the bound of ``x0``
    itself when ``x0`` is a prediction)."""
    torch = _torch()
    port_shape = tuple(run.address_shape(edit.address))  # the edited row's frame, not the cursor's
    if isinstance(edit.value, torch.Tensor):
        value = f64(edit.value).broadcast_to(port_shape).reshape(shape)
    else:
        value = torch.full(shape, float(edit.value), dtype=torch.float64)
    mask = (torch.ones(shape, dtype=torch.bool) if edit.mask is None
            else edit.mask.to(device=x0.device, dtype=torch.bool).broadcast_to(port_shape).reshape(shape))
    on: Any = None
    if edit.op == "replace":
        full = value
        on = u * value.abs()
    elif edit.op == "add":
        full = x0 + value
        if b0 is not None:
            on = u * (full.abs() + u * value.abs()) + u * value.abs() + b0 * (1 + u)
    else:
        full = x0 * value
        if b0 is not None:
            on = (2 * u + u * u) * full.abs() + b0 * value.abs() * (1 + u) ** 2
    requested = torch.where(mask, full, x0)
    if bool(mask.all()):
        bound = on
    else:
        bound = None if on is None or b0 is None else torch.where(mask, on, b0)
    return requested, requested - x0, bound


def _edits_on_path(run: Any, p: Plan) -> str | None:
    """A staged or committed edit (other than at the edited port) on a stop the edit reaches:
    the chain recomputes those stops from identities and would not see it."""
    row = p.here[0]
    found = [(int(e["row"]), int(e["layer"]), e["port"]) for e in run.committed_edits()]
    found += [(row, int(e["layer"]), e["port"]) for e in run.staged_info()]
    for j, layer, port in found:
        stop = f"L{layer}.{run.program.spec(port).name}"
        if (j, stop) != (p.row, p.label) and stop in p.affected.get(j, set()):
            return (f"the edit at t{j}.{stop} lies on the path of this edit; the prediction "
                    "recomputes that stop from its identity and does not model the edit")
    return None


def _cell_edit(edit: Edit, value: Any = None) -> Any:
    """``edit`` as the engine's ``CellEdit`` (``value`` replaces its value when given)."""
    from mdb.engine.edits import CellEdit

    cell = edit.cell
    return CellEdit(int(cell["layer"]), cell["tensor"], tuple(cell["rows"]),
                    edit.value if value is None else value, edit.op, kv=cell.get("kv"),
                    d=cell.get("d"), dose=edit.dose, selection=edit.selection,
                    origin=edit.origin)


def _cell_writer(edit: Edit, written: Any, mode: str) -> Any:
    """``_chain``'s ``cell`` for a state-cell edit, on the cache rows ``0..row-1`` it reads.
    ``written`` is the whole cache tensor with the block the engine writes. ``mode``:
    ``ideal`` the float64 request on the selected block, its distance to the written block
    added to the spread; ``written`` the written block itself; ``tangent`` the unedited rows
    with the displacement ``ideal - base`` as forward-mode tangent (no spread)."""
    import torch.autograd.forward_ad as fwad

    torch = _torch()
    request = _cell_edit(edit)

    def write(layer: int, kind: str, value: Any, spread: Any) -> tuple[Any, Any]:
        if (layer, kind) != (request.layer, request.tensor):
            return value, spread
        x = value.unsqueeze(0)
        index = request.index(x.shape)
        ideal = request.ideal(x[index])
        actual = f64(written[..., : x.shape[2], :])[index]
        if mode == "tangent":
            direction = torch.zeros_like(x)
            direction[index] = ideal - x[index]
            return fwad.make_dual(value, direction[0]), None
        out = x.clone()
        out[index] = ideal if mode == "ideal" else actual
        if spread is None:
            return out[0], None
        grown = spread.unsqueeze(0).clone()
        if mode == "ideal":
            grown[index] = grown[index] + (ideal - actual).abs()
        return out[0], grown[0]

    return write


def _written_cell(run: Any, edit: Edit) -> Any:
    """The cache tensor at the cursor with the block the cell edit writes, by the engine's own
    arithmetic (``CellEdit.realise``, the code the native layer call runs)."""
    request = _cell_edit(edit)
    return request.realise(run.read_cache(request.layer, request.tensor))[0]


def _settle(run: Any, p: Plan, chain: list[RowModel], seen: dict[str, dict[str, Any]]
            ) -> dict[str, Any]:
    """Everything the report reads from one chain, so the chain (one float64 row model per row)
    can be dropped before the next is built: the consumer's value and bound, the premises, why
    its consumer is unbounded (the last row's reason, ``last``; the first row model's with a
    reason, ``first``), the sampled tokens, and its operand digests (added to ``seen``). Read
    in the order a whatif always read them, so every number and digest is unchanged."""
    value, bound = _consumer_value(p, chain[-1])
    out = {"value": value, "bound": bound,
           "premises": set().union(*(m.premises for m in chain)),
           "last": chain[-1].unbounded.get(p.consumer_label)}
    out["tokens"] = _sampled_tokens(p, chain, run)
    _operands(run, chain, seen)
    out["first"] = next((m.unbounded[p.consumer_label] for m in chain
                         if p.consumer_label in m.unbounded), None)
    return out


def _flip_reason(p: Plan, base: dict[str, Any], edited: dict[str, Any]) -> str | None:
    """Why the first-order term is refused: the edit flips a sampled token of the chain."""
    flipped = {j: (t, e) for (j, t), e in zip(base["tokens"].items(), edited["tokens"].values())
               if t != e}
    if not flipped:
        return None
    j, (was, now) = next(iter(flipped.items()))
    return (f"the edit flips the sampled token of t{j} ({was} -> {now}): the first-order term "
            "does not cross a token change")


def _found(base: dict[str, Any], edited: dict[str, Any], seen: dict[str, dict[str, Any]],
           shape: Any) -> dict[str, Any]:
    """The prediction of two settled chains (``_settle``)."""
    bound, base_bound = edited["bound"], base["bound"]
    return {
        "base": base["value"], "base_bound": base_bound, "exact": edited["value"] - base["value"],
        "exact_bound": None if bound is None or base_bound is None else bound + base_bound,
        "edit_shape": shape, "operands": list(seen.values()),
        "unbounded": next((r for r in (edited["last"], base["last"]) if r is not None),
                          "no rounding bound for this chain"),
        "operand_reason": base["first"] if base["first"] is not None else edited["first"],
        "premises": base["premises"] | edited["premises"],
    }


WRITE_PREMISE = ("the edit is written by the engine's own arithmetic (PortEdit.apply, "
                 "CellEdit.realise) on the emulated value of the edited port or cache")


def _emulated_base(run: Any, p: Plan, shared: dict | None) -> dict[str, Any]:
    """The emulated unedited chain (``EmulatedRowModel``: stops at or before the cursor as
    captured, the rest with the dtype's rounding): the consumer's value, the sampled tokens and
    the edited port's value ``x0`` (no row model is kept). The consumer's value does not depend
    on the edit, so the what-ifs of one call with the same first row, consumer and cursor share
    it (``shared``); ``x0`` is kept per edited port (a fresh first row model, evaluated only up
    to that port, for a port not seen yet)."""
    key = (p.row, p.consumer_stop, p.here)
    out = None if shared is None else shared.get(key)
    if out is None:
        chain = _chain(run, p, bounds=False, model=EmulatedRowModel)
        out = {"x0": {p.label: chain[0].get(p.label)},
               "value": _consumer_value(p, chain[-1])[0],
               "tokens": _sampled_tokens(p, chain, run)}
        del chain               # one chain of row models alive at a time
        if shared is not None:
            shared[key] = out
    if p.label not in out["x0"]:
        first = _chain(run, p, bounds=False, model=EmulatedRowModel)[0]
        out["x0"][p.label] = first.get(p.label)
        del first
    return out


def _emulated_written(run: Any, p: Plan, edit: Edit, x0: Any) -> Any:
    """The value a port edit writes, by the engine's arithmetic (``PortEdit.apply``) on the
    emulated (or captured) value of the port in the model dtype, as float64 in the chain's
    shape."""
    from mdb.engine.ports import PortEdit

    shape = tuple(run.address_shape(p.edit_stop))
    port = run.program.spec(_split(p.label)[1]).name
    x = x0.reshape(shape).to(getattr(run.adapter, "device", None) or "cpu", run.adapter.dtype)
    written = PortEdit(port, edit.value, edit.op, mask=edit.mask).apply(x)
    return f64(written).reshape(x0.shape)


def _emulated_edited(run: Any, p: Plan, edit: Edit, written: Any) -> list[RowModel]:
    """The emulated edited chain: the stops the edit reaches recomputed with the dtype's rounding,
    the edit written at its port (``written``: a port edit's value) or into the cache (a cell
    edit: the engine's own write, ``_written_cell``)."""
    if edit.cell is not None:
        return _chain(run, p, bounds=False, skip=True, model=EmulatedRowModel,
                      cell=_cell_writer(edit, _written_cell(run, edit), "written"))
    return _chain(run, p, bounds=False, skip=True, model=EmulatedRowModel,
                  overrides={p.label: lambda value, bound: (written, None)})


def _emulate(run: Any, p: Plan, edit: Edit, shared: dict | None) -> dict[str, Any]:
    """The emulated base and edited consumer values and their change, or ``emulated_reason``.
    A staged edit ahead of the cursor (not on the edit's path: that refuses the whole
    prediction) rewrites an operand the chains recompute; the emulated chains do not model it,
    so they are refused (as a forecast is) and the native measurement decides alone."""
    here = p.here
    staged = _staged_between(run, here[0], here, (here[0] + 1, -1))
    if staged:
        return {"emulated_reason": f"staged edit(s) at {', '.join(staged)} ahead of the cursor "
                                   "rewrite an operand the emulated chains recompute"}
    try:
        base = _emulated_base(run, p, shared)
        written = None if edit.cell is not None else _emulated_written(run, p, edit,
                                                                       base["x0"][p.label])
        edited = _consumer_value(p, _emulated_edited(run, p, edit, written)[-1])[0]
    except Refusal as error:
        return {"emulated_reason": str(error)}
    return {"emulated_base": base["value"], "emulated_edited": edited,
            "emulated": edited - base["value"], "emulated_written": written}


def _predict_cell(run: Any, p: Plan, edit: Edit, *, linear: bool = True,
                  emulate: bool = True, shared: dict | None = None) -> dict[str, Any]:
    """The prediction of a state-cell edit (``base``, ``exact`` and bounds, ``linear``), or
    ``reason``: the chain from the first affected row with the cell written into the cache rows
    every row reads. The base chain recomputes the same stops as the edited one, from the same
    operands (P09 section 1: ``F(x1) - F(x0)``, never ``F(x1)`` minus a native value)."""
    torch = _torch()
    if p.row > p.here[0]:
        return {"reason": f"the first read of {edit.address} is t{p.row}.{p.label}, in a row "
                          f"that has not started at the cursor {run.where}: none of its "
                          "operands exist yet"}
    conflict = _edits_on_path(run, p)
    if conflict is not None:
        return {"reason": conflict}
    written = _written_cell(run, edit)
    seen: dict[str, dict[str, Any]] = {}
    base = _settle(run, p, _chain(run, p, bounds=True, skip=True), seen)
    edited = _settle(run, p, _chain(run, p, bounds=True, skip=True,
                                    cell=_cell_writer(edit, written, "ideal")), seen)
    out = _found(base, edited, seen, None)
    if emulate:
        out.update(_emulate(run, p, edit, shared))
    reason = _flip_reason(p, base, edited)
    if reason is not None:
        out["linear_reason"] = reason
        return out
    if not linear:
        return out
    import torch.autograd.forward_ad as fwad

    with torch.inference_mode(False), fwad.dual_level():
        dual = _chain(run, p, bounds=False, skip=True, tokens=base["tokens"],
                      cell=_cell_writer(edit, written, "tangent"))
        primal, tangent = fwad.unpack_dual(_consumer_value(p, dual[-1])[0])
        out["linear"] = (torch.zeros_like(primal) if tangent is None else tangent).detach().clone()
    return out


def _predict(run: Any, p: Plan, edit: Edit, *, linear: bool = True, emulate: bool = True,
             shared: dict | None = None) -> dict[str, Any]:
    """The prediction's tensors (``base``, ``exact`` and bounds, ``linear``), or ``reason``.

    ``exact = F(x1) - F(x0)`` (P09 section 1, P20): both chains hold the edited port at its value
    at the cursor (``x0``: captured, or the chain's own value when the port is ahead) and
    recompute the same stops (those the edit reaches) from the same operands, every other stop
    taken as captured when it is at or before ``p.here``. ``linear`` (P14) is the forward-mode
    derivative of that base chain at ``x0``. ``linear=False`` skips it (the post-hoc exact
    chain on measured operands, ``_measured_exact``). With ``emulate`` the emulated base and
    edited chains (``_emulate``) predict the native change itself; ``shared`` holds the emulated
    base across the what-ifs of one call."""
    torch = _torch()
    if edit.cell is not None:
        return _predict_cell(run, p, edit, linear=linear, emulate=emulate, shared=shared)
    if p.row > p.here[0]:
        return {"reason": f"the edit site t{p.row}.{p.label} is in a row that has not started at "
                          f"the cursor {run.where}: none of its operands exist yet"}
    conflict = _edits_on_path(run, p)
    if conflict is not None:
        return {"reason": conflict}
    seen: dict[str, dict[str, Any]] = {}
    head = _chain(run, p, bounds=True)[:1]
    x0, b0, u = head[0].get(p.label), head[0].bound(p.label), head[0].u
    _operands(run, head, seen)
    del head                    # one chain of row models alive at a time
    shape = tuple(x0.shape)
    requested, delta, spread = _request(run, edit, shape, x0, b0, u)
    base = _settle(run, p, _chain(run, p, bounds=True, skip=True,
                                  overrides={p.label: lambda value, bound: (x0, b0)}), seen)
    edited = _settle(run, p, _chain(run, p, bounds=True, skip=True,
                                    overrides={p.label: lambda value, bound: (requested,
                                                                              spread)}), seen)
    out = _found(base, edited, seen, shape)
    if emulate:
        out.update(_emulate(run, p, edit, shared))
    reason = _flip_reason(p, base, edited)
    if reason is not None:
        out["linear_reason"] = reason
        return out
    if not linear:
        return out
    import torch.autograd.forward_ad as fwad

    with torch.inference_mode(False), fwad.dual_level():
        point = x0.clone()
        dual = _chain(run, p, bounds=False, skip=True, tokens=base["tokens"],
                      overrides={p.label: lambda value, bound: (fwad.make_dual(point, delta),
                                                                None)})
        primal, tangent = fwad.unpack_dual(_consumer_value(p, dual[-1])[0])
        out["linear"] = (torch.zeros_like(primal) if tangent is None else tangent).detach().clone()
    return out


# --------------------------------------------------------------------------- measurement


def stage(run: Any, edit: Edit) -> dict[str, Any]:
    """Stage ``edit`` on ``run`` at the cursor (``Run.set`` at its port, or ``Run.set_cell`` for
    a state-cell edit); returns the engine's manifest."""
    if edit.cell is None:
        return run.set(edit.address, edit.value, edit.op, mask=edit.mask,
                       selection=edit.selection, origin=edit.origin)
    cell = edit.cell
    return run.set_cell(int(cell["layer"]), cell["tensor"], edit.value, edit.op,
                        rows=cell["rows"], kv=cell.get("kv"), d=cell.get("d"), dose=edit.dose,
                        selection=edit.selection, origin=edit.origin)


def guard(run: Any, checkpoint: dict[str, Any]) -> None:
    """Hold every edit tensor the checkpoint names so ``restore`` can rebuild it; refuse, before
    anything native runs, a checkpoint naming a tensor this run does not hold."""
    held = run.edit_tensors()
    manifests = [*checkpoint["edits"], *(m for item in checkpoint["path_edits"]
                                         for m in item["edits"])]
    for manifest in manifests:
        needed = [manifest["value"].get("sha256"), manifest.get("mask_sha256")]
        missing = [sha for sha in needed if sha is not None and sha not in held]
        if missing:
            raise Refusal(f"the session cannot be restored after the what-if: edit tensor "
                          f"{missing[0][:12]} at {manifest['port']} is not held by this run")
    run.hold(held)


def _complete_restore(run: Any, checkpoint: dict[str, Any], state: list[bool]) -> None:
    """``run.restore(checkpoint)`` that completes whatever interrupts it, then re-raises the
    interrupt (``state[0]`` is set once the restore has completed)."""
    interrupted = None
    while True:
        try:
            run.restore(checkpoint, defer_interrupt=True)
            break
        except KeyboardInterrupt as error:      # before the restore ran, or inside it
            interrupted = error
    state[0] = True
    if interrupted is not None:
        raise interrupted


def _restoring(run: Any, checkpoint: dict[str, Any], body: Any) -> Any:
    """``body()``, then ``checkpoint`` restored whatever happens, interrupts included: the
    restore is unconditional, so an interrupt landing exactly at the entry of the ``finally``
    (before its first statement) still finds the restore done before it propagates. An interrupt
    during the restore cannot roll it back to the edited excursion either."""
    state = [False]
    try:
        try:
            return body()
        finally:
            _complete_restore(run, checkpoint, state)
    except KeyboardInterrupt:
        if not state[0]:
            _complete_restore(run, checkpoint, state)
        raise


def _check(run: Any, out: dict[str, Any], side: str) -> None:
    """Run the what-if's check of ``side`` (``edited`` or ``base``) while the native state of
    that run is at the consumer (``_checker``)."""
    check = out.get("check")
    if check is not None:
        out[f"check_{side}"] = check(run, side, out[side])


def _measure_cell(run: Any, p: Plan, edit: Edit, checkpoint: dict[str, Any],
                  check: Any = None) -> dict[str, Any]:
    """``_measure`` of a state-cell edit: staged at ``p.stage_at``; ``realised`` is the cache
    tensor the native layer call wrote, read back at the consumer."""
    layer, tensor = int(edit.cell["layer"]), edit.cell["tensor"]
    out: dict[str, Any] = {"moves": [], "check": check}

    def edited() -> None:
        stage(run, edit)
        out["moves"].append(f"set {edit.address} at {run.where}")
        run.until(p.consumer_stop)
        out["moves"].append(f"until {p.consumer_stop}")
        out["edited"] = run.read(p.consumer_stop).clone()
        out["edited_info"] = run.read_info(p.consumer_stop)
        out["realised"] = run.read_cache(layer, tensor, staged=False).clone()
        _check(run, out, "edited")

    def body() -> None:
        if p.mode == "pending":     # the consumer is ahead: edited branch first, then the base
            edited()
            run.restore(checkpoint)
            out["moves"].append(f"restore {checkpoint['where']}")
            _base_ahead(run, p, edit, out)
            return
        out["base"] = run.read(p.consumer_stop).clone()
        out["base_info"] = run.read_info(p.consumer_stop)
        while run.parse(run.where) >= run.parse(p.stage_at):
            run.reverse_step("layer")
        out["moves"].append(f"reverse-step layer to {run.where}")
        if run.where != p.stage_at:
            run.until(p.stage_at)
            out["moves"].append(f"until {p.stage_at}")
        edited()

    _restoring(run, checkpoint, body)
    out["moves"].append(f"restore {checkpoint['where']}")
    return out


def _measure(run: Any, p: Plan, edit: Edit, checkpoint: dict[str, Any],
             check: Any = None) -> dict[str, Any]:
    """Native consumer values without and with the edit, the value the edit wrote, and the
    native-call trail; restores ``checkpoint`` whatever happens. ``check(run, side, value)``
    (``_checker``) runs while each native run is at the consumer."""
    if edit.cell is not None:
        return _measure_cell(run, p, edit, checkpoint, check)
    layer = _split(p.label)[0]
    entry = f"t{p.row}.L{layer}.h"
    out: dict[str, Any] = {"moves": [], "check": check}

    def body() -> None:
        if p.consumer_key > p.here:  # edited branch first, then the base and its exact chain
            _edited(run, p, edit, entry, out)
            run.restore(checkpoint)
            out["moves"].append(f"restore {checkpoint['where']}")
            _base_ahead(run, p, edit, out)
            return
        out["base"] = run.read(p.consumer_stop).clone()
        out["base_info"] = run.read_info(p.consumer_stop)
        _edited(run, p, edit, entry, out)

    _restoring(run, checkpoint, body)
    out["moves"].append(f"restore {checkpoint['where']}")
    return out


def _base_ahead(run: Any, p: Plan, edit: Edit, out: dict[str, Any]) -> None:
    """From the checkpoint, with the consumer ahead of it: run the unedited branch to the
    consumer, read it, and compute there (every operand of the chain now measured) the exact
    change on measured operands (``out["posthoc"]``, ``_measured_exact``; the edited branch has
    already put the written operand in ``out["realised"]``). Runs after the prediction was
    sealed, so nothing it computes is presented as sealed."""
    run.until(p.consumer_stop)
    out["moves"].append(f"until {p.consumer_stop}")
    out["base"] = run.read(p.consumer_stop).clone()
    out["base_info"] = run.read_info(p.consumer_stop)
    _check(run, out, "base")
    out["posthoc"] = _measured_exact(run, p, edit, out["realised"])


def _written_exact(run: Any, p: Plan, edit: Edit, found: dict[str, Any], realised: Any
                   ) -> dict[str, Any]:
    """The exact change with the operand the edit wrote (``realised``, read back natively) on
    the operands of ``found``'s chains: ``at_realised`` (its consumer minus ``found["base"]``),
    its rounding ``spread`` (None when unbounded) and ``spread_reason``."""
    torch = _torch()
    if edit.cell is not None:
        chain = _chain(run, p, bounds=True, skip=True,
                       cell=_cell_writer(edit, realised, "written"))
    else:
        written = f64(realised).reshape(found["edit_shape"])
        chain = _chain(run, p, bounds=True, skip=True,
                       overrides={p.label: lambda value, bound: (written,
                                                                 torch.zeros_like(written))})
    value, bound = _consumer_value(p, chain[-1])
    return {"at_realised": value - found["base"],
            "spread": None if bound is None or found["base_bound"] is None else (
                bound + found["base_bound"]),
            "spread_reason": _realised_reason(p, chain[-1], found)}


def _measured_exact(run: Any, p: Plan, edit: Edit, realised: Any) -> dict[str, Any]:
    """The exact chains again with the cursor at (or past) the consumer, so every operand they
    read is the native value of the unedited run: ``base``, ``exact`` and bounds (requested
    operand) and ``at_realised`` (written operand), or ``reason``. P09 section 1 / P20 define
    ``native delta = F(x1) - F(x0) + residual`` on the retained operands; before the native run
    those operands past the cursor exist only as the chain's own forecast."""
    from dataclasses import replace

    q = replace(p, here=run.parse(run.where))
    try:
        post = _predict(run, q, edit, linear=False, emulate=False)
    except Refusal as error:
        return {"reason": str(error)}
    if post.get("reason") is None:
        post.update(_written_exact(run, q, edit, post, realised))
    return post


def _edited(run: Any, p: Plan, edit: Edit, entry: str, out: dict[str, Any],
            cut: dict[str, Any] | None = None, keep: bool = False) -> dict[str, Any] | None:
    """From the current cursor: reach ``entry`` (the layer's entry: back by layer boundaries in
    ``rewind`` mode, forward otherwise, or ``restore(cut)`` of a checkpoint taken at it), stage
    the edit, run to the consumer and read the edited value and the value the edit wrote into
    ``out``. With ``keep``, returns the checkpoint at ``entry`` taken before the edit (``cut``
    itself when given): restoring it drops the edit and keeps the edits committed at that layer, which
    stepping back stages again."""
    if cut is not None:
        run.restore(cut)
        out["moves"].append(f"restore the cut at {entry}")
    elif p.mode == "rewind":
        while run.parse(run.where) >= run.parse(entry):
            run.reverse_step("layer")
        out["moves"].append(f"reverse-step layer to {run.where}")
    if p.mode != "pending" and run.where != entry:
        run.until(entry)
        out["moves"].append(f"until {entry}")
    cut = cut or (run.checkpoint() if keep else None)
    run.set(p.edit_stop, edit.value, edit.op, mask=edit.mask, selection=edit.selection,
            origin=edit.origin)
    out["moves"].append(f"set {p.edit_stop}")
    if run.parse(run.where) < p.consumer_key:
        run.until(p.consumer_stop)
        out["moves"].append(f"until {p.consumer_stop}")
    out["edited"] = run.read(p.consumer_stop).clone()
    out["edited_info"] = run.read_info(p.consumer_stop)
    out["realised"] = run.read(p.edit_stop).clone()
    _check(run, out, "edited")
    return cut


def _measure_batch(run: Any, items: list[_Prepared]) -> list[dict[str, Any]]:
    """``_measure`` of several prepared what-ifs made at the same cursor, in one excursion.

    A what-if in ``rewind`` mode with the consumer at or before the cursor starts from
    a clean boundary of the original run, and the trail's marks below an edit are untouched by
    it: so they are measured from the latest entry to the earliest, each stepping back by layer
    boundaries from where the previous one ended, and the cursor is restored once at the end
    (the replay from the earliest entry) instead of after every one. The consumer's unedited
    value is read at the cursor before the first leaves it. The other what-ifs are measured one
    by one, first. Returns the native measurements in the order of ``items``; the session is
    restored whatever happens."""
    checkpoint = items[0].checkpoint
    results: list[dict[str, Any] | None] = [None] * len(items)
    batch = []
    for i, item in enumerate(items):
        if (item.edit.cell is None and item.p.mode == "rewind"
                and item.p.consumer_key <= item.p.here):
            batch.append(i)
        else:
            results[i] = _measure(run, item.p, item.edit, item.checkpoint, item.check)
    base = {i: (run.read(items[i].p.consumer_stop).clone(),
                run.read_info(items[i].p.consumer_stop)) for i in batch}
    entries = {i: f"t{items[i].p.row}.L{_split(items[i].p.label)[0]}.h" for i in batch}
    batch.sort(key=lambda i: (run.parse(entries[i]), i), reverse=True)

    def body() -> None:
        cuts: dict[str, dict[str, Any]] = {}   # entry -> checkpoint there (several heads per layer)
        for i in batch:
            out: dict[str, Any] = {"moves": [], "base": base[i][0], "base_info": base[i][1],
                                   "check": items[i].check}
            cuts[entries[i]] = _edited(run, items[i].p, items[i].edit, entries[i], out,
                                       cuts.get(entries[i]), keep=True)
            results[i] = out

    if batch:
        _restoring(run, checkpoint, body)
        for i in batch:
            results[i]["moves"].append(f"restore {checkpoint['where']}")
    return results


def whatif_many(run: Any, requests: list[tuple[Edit, Any]], *, store: Any = None
                ) -> list[WhatIf | Refusal]:
    """``whatif`` of each ``(edit, consumer)`` at the cursor, measured in one excursion: the
    predictions are sealed first (each needs the cursor's values), the native measurements share
    their rewinds and one final restore (``_measure_batch``), and each is then split and judged
    at the cursor again. Every result is the one ``whatif`` returns, number for number (the
    ``measured.moves`` trail names the shared steps); a request ``whatif`` refuses before
    anything native runs comes back as its ``Refusal``."""
    before = state_digest(run)
    prepared: list[_Prepared | Refusal] = []
    shared: dict = {}      # the emulated base chains, shared by the what-ifs of this call
    for edit, consumer in requests:
        try:
            prepared.append(_prepare(run, edit, consumer, before=before, store=store,
                                     shared=shared))
        except Refusal as error:
            prepared.append(error)
    shared.clear()         # nothing wider than the call: the row models go here
    ready = [item for item in prepared if isinstance(item, _Prepared)]
    natives = iter(_measure_batch(run, ready) if ready else [])
    after = state_digest(run) if ready else before
    return [item if isinstance(item, Refusal) else _finish(run, item, next(natives), after)
            for item in prepared]


# --------------------------------------------------------------------------- report


def _scalar(p: Plan, value: Any) -> Any:
    """The consumer's float64 value from the tensor read at its stop (index, then cast)."""
    flat = value.reshape(-1)
    if p.margin is not None:
        return f64(flat[p.margin.a]) - f64(flat[p.margin.b])
    if p.logit is not None:
        return f64(flat[p.logit.i])
    return f64(flat)


def _claim(value: Any, authority: str, proof: str | None = None) -> dict[str, Any]:
    """One labelled number for a scalar; ``{max_abs, norm}`` (both labelled) for a vector."""
    if value.dim() == 0:
        return num(float(value), authority, proof=proof)
    flat = value.reshape(-1)
    return {"max_abs": num(float(flat.abs().max()), authority, proof=proof),
            "norm": num(float(flat.norm()), authority, proof=proof)}


def _realised_reason(p: Plan, model: RowModel, found: dict[str, Any]) -> str:
    """Why the consumer of the realised chain has no rounding bound: its own reason, else the
    first one of the prediction's chains."""
    if p.consumer_label in model.unbounded:
        return model.unbounded[p.consumer_label]
    reason = found["operand_reason"]
    return reason if reason is not None else "no rounding bound for this chain"


def _within(gap: Any, bound: Any, reason: str) -> dict[str, Any]:
    if bound is None:
        return {"bound_note": f"no worst-case interval bound ({bound_state(reason)}: {reason})"}
    return {"bound": num(float(bound.reshape(-1).max()), BOUND),
            "within_bound": bool((gap.abs() <= bound).all())}


def _edit_view(run: Any, p: Plan, edit: Edit) -> dict[str, Any]:
    from mdb.engine.ports import tensor_digest

    torch = _torch()
    out: dict[str, Any] = {"address": p.edit_stop, "row": p.row, "layer": _split(p.label)[0],
                           "port": _split(p.label)[1], "op": edit.op}
    if edit.cell is not None:
        cell = edit.cell
        out.update(address=edit.address, layer=int(cell["layer"]),
                   port=f"cache.L{cell['layer']}.{cell['tensor']}",
                   cell={key: cell.get(key) for key in ("layer", "tensor", "rows", "kv", "d")},
                   stage_at=p.stage_at, first_read=p.edit_stop)
        if edit.dose is not None:
            out["dose"] = num(float(edit.dose), SUPPLIED)
    if edit.origin is not None:
        out["origin"] = dict(edit.origin)
    if isinstance(edit.value, torch.Tensor):
        out.update(sha256=tensor_digest(edit.value.contiguous()),
                   shape=list(edit.value.shape))
    else:
        out["value"] = num(float(edit.value), SUPPLIED)
    if edit.mask is not None:
        out["mask_count"] = int(edit.mask.sum())
    if edit.selection is not None:
        out["selection"] = edit.selection
    if edit.source is not None:
        out["source"] = edit.source
    return out


def _operands(run: Any, models: list[RowModel], seen: dict[str, dict[str, Any]]) -> None:
    """Add the address, digest and shape of every captured operand of ``models`` to ``seen``
    (the first chain to read an address names it)."""
    from mdb.engine.ports import tensor_digest

    for model in models:
        tensors = {**{f"t{model.row}.{n}": t for n, t in model.captured.items()},
                   **{f"{n}[row=0:{model.row}]": t for n, t in model.cache_reads.items()}}
        for address, tensor in tensors.items():
            seen.setdefault(address, {"address": address,
                                      "sha256": tensor_digest(tensor.contiguous()),
                                      "shape": list(tensor.shape)})


@dataclass
class WhatIf:
    """A what-if result: ``report`` (bounded JSON, every number labelled), the sealed prediction
    ``record`` and the float64 ``tensors`` behind the numbers (for in-process callers)."""

    report: dict[str, Any]
    record: dict[str, Any]
    tensors: dict[str, Any]


def _within_view(found: dict[str, Any]) -> dict[str, Any]:
    if found["exact_bound"] is None:
        return {"bound_note": f"no worst-case interval bound ({bound_state(found['unbounded'])}: "
                              f"{found['unbounded']})"}
    return {"bound": num(float(found["exact_bound"].reshape(-1).max()), BOUND),
            "premises": sorted(found["premises"] | {"round to nearest, no overflow or underflow",
                                                    "replace/add/scale: the operand is cast to "
                                                    "the model dtype and applied in one rounded "
                                                    "op; mix: float64, rounded once"})}


@dataclass
class _Prepared:
    """A what-if up to its sealed prediction: everything that needs the cursor's values."""

    p: Plan
    edit: Edit
    checkpoint: dict[str, Any]
    found: dict[str, Any]
    prediction: dict[str, Any]
    body: dict[str, Any]
    record: dict[str, Any]
    before: str
    check: Any = None


CHECK_ERRORS = (Refusal, KeyError, IndexError, ValueError, RuntimeError)


def _checker(p: Plan, edit: Edit, found: dict[str, Any]) -> Any:
    """``check(run, side, value)``, run while the native run of ``side`` (``edited`` or
    ``base``) stands at the consumer: the spacings between its native consumer value and the
    emulated one and, beyond the floor, where they part (``finding``, ``stage_parting`` on the
    consumer's row: not confirmed when a stage of the emulated chain does not reproduce its
    native value from its native operands, else unresolved). A chain over several rows is not
    stage-checked (the earlier rows' ports are not held at the consumer). None without an
    emulated prediction."""
    if found.get("emulated") is None:
        return None

    def check(run: Any, side: str, value: Any) -> dict[str, Any]:
        predicted = found[f"emulated_{side}"]
        native = _scalar(p, value).reshape(predicted.shape)
        step = spacing(magnitude(p.margin or p.logit, value), value.dtype)
        gap = float((native - predicted).abs().max())
        out: dict[str, Any] = {"spacings": gap / step}
        if gap <= FLOOR_SPACINGS * step:
            return out
        if p.row != p.consumer_row:
            out["finding"] = {"verdict": "unresolved", "state": "unresolved",
                              "reason": f"the {side} run parts from its emulation by "
                                        f"{gap / step:.3g} spacings; the chain spans rows "
                                        f"t{p.row}..t{p.consumer_row}, whose earlier ports are "
                                        "not held at the consumer: stages not checked"}
            return out
        try:
            chain = (_chain(run, p, bounds=False, model=EmulatedRowModel) if side == "base"
                     else _emulated_edited(run, p, edit, found["emulated_written"]))
            emulator = chain[-1]
            emulator.get(p.consumer_label)
            skip = {p.label: "reads the edited cell" if edit.cell is not None
                    else "written by the edit"} if side == "edited" else {}
            finding = stage_parting(run, p.consumer_row, emulator, p.consumer_label,
                                    MEASURED, skip)
        except CHECK_ERRORS as error:
            finding = {"verdict": "unresolved", "state": "unresolved",
                       "reason": f"stages not checked ({error})"}
        finding["reason"] = f"{side} run: {finding['reason']}"
        out["finding"] = finding
        return out

    return check


def _parting(native: dict[str, Any]) -> dict[str, Any] | None:
    """The verdict-relevant finding of the measurement's checks: a ``not confirmed`` one first,
    else the first ``unresolved`` one, else None."""
    findings = [native[key]["finding"] for key in ("check_edited", "check_base")
                if native.get(key) and "finding" in native[key]]
    return next((f for f in findings if f["verdict"] == NOT_CONFIRMED),
                findings[0] if findings else None)


def _prepare(run: Any, edit: Edit, consumer: Any, before_native: Any = None,
             before: str | None = None, store: Any = None,
             shared: dict | None = None) -> _Prepared:
    """Plan, hold the edit tensors, predict and seal; nothing native runs. ``before``: the
    ``state_digest`` of the cursor when the caller already has it. ``store``: a project store
    for the full operand list when it is over the record bound (see ``bound_operands``).
    ``shared``: the emulated base chains of one call (``_emulated_base``)."""
    p = plan(run, edit, consumer)
    if before is None:
        before = state_digest(run)
    checkpoint = run.checkpoint()
    guard(run, checkpoint)
    try:
        found = _predict(run, p, edit, shared=shared)
    except Refusal as error:
        found = {"reason": str(error)}
    reason = found.get("reason")
    prediction: dict[str, Any]
    if reason is not None:
        prediction = {"state": "refused", "reason": reason}
    else:
        exact_view = {"value": _claim(found["exact"], EXACT), **_within_view(found)}
        if p.consumer_key > p.here:
            exact_view["operands"] = {
                "source": "forecast", "at": run.where,
                "premise": f"the stops after {run.where} are this chain's own float64 values "
                           "from the operands captured there (a forecast), not native ones; "
                           "the exact change on measured operands is computed after the "
                           "native run (split term operands)"}
        else:
            exact_view["operands"] = {"source": "measured", "at": run.where}
        if "linear" in found:
            linear_view: dict[str, Any] = {"value": _claim(found["linear"], LOCAL, "P14"),
                                           "meaning": "gradient of the consumer times the edit "
                                                      "displacement; not the effect"}
        else:
            linear_view = {"state": "refused", "reason": found["linear_reason"]}
        if found.get("emulated") is not None:
            emulated_view: dict[str, Any] = {
                "value": _claim(found["emulated"], EXACT),
                "base": _claim(found["emulated_base"], EXACT),
                "edited": _claim(found["emulated_edited"], EXACT),
                "premise": EMULATION_PREMISE,
                "premises": [*EMULATION_PREMISES, WRITE_PREMISE],
                "meaning": "the native change predicted bit for bit: base and edited chains "
                           "with the dtype's rounding where eager HF rounds"}
        else:
            emulated_view = {"state": "refused",
                             "reason": found.get("emulated_reason", "not computed")}
        prediction = {"state": "resolved", "linear": linear_view, "exact": exact_view,
                      "emulated": emulated_view, "base": _claim(found["base"], EXACT)}
    body = {
        "schema": SCHEMA, "edit": _edit_view(run, p, edit),
        "consumer": {"address": p.consumer_text, "row": p.consumer_key[0],
                     "index": p.consumer_key[1],
                     "kind": "margin" if p.margin else "logit" if p.logit else "value",
                     **({"indices": [p.margin.a, p.margin.b]} if p.margin else {})},
        "sealed_at": {"where": run.where, "cut": checkpoint["cut"], "row": p.here[0],
                      "index": p.here[1], "transition": int(checkpoint["transitions"])},
        "mode": p.mode, "prediction": prediction,
        "method": {"exact": "F(x1) - F(x0): port-program identities on values at the cursor, "
                            "base and edited chains recomputing the same stops (those the edit "
                            "reaches) with the port at its value x0 or the edit written there "
                            "(rows chained through the cache)",
                   "linear": "forward-mode derivative of the same chain at the unedited point "
                             "(P14)",
                   "emulated": "the same chains with the dtype's rounding at every eager-HF "
                               "rounding point (EmulatedRowModel); the base takes the stops at "
                               "or before the cursor as captured"},
    }
    pointers: list[dict[str, Any]] = []
    if reason:
        body["operands"] = []
    else:
        body["operands"], summary = bound_list(
            found["operands"], MAX_ITEMS, store=store, run=run,
            keep=f"the first {MAX_ITEMS} in capture order (a row's ports, then its cache reads)")
        if summary is not None:
            tensor = summary["pointer"].pop("tensor", None)
            pointers = [] if tensor is None else [tensor]
            body["operands_summary"] = summary
    record = seal_record(session_id=run_session_id(run), kind="whatif",
                         model_identity=run.adapter.model_identity,
                         execution=dict(run.adapter.execution),
                         authority=LOCAL if reason is None else SUPPLIED, body=body,
                         pointers=pointers)
    # An evidence writer can persist/flush the sealed forecast before the first
    # counterfactual measurement. Writer failure prevents native release.
    if before_native is not None:
        before_native(record)
    return _Prepared(p, edit, checkpoint, found, prediction, body, record, before,
                     _checker(p, edit, found) if reason is None else None)


def load_operands(record: dict[str, Any], *, store: Any = None, run: Any = None
                  ) -> list[dict[str, Any]]:
    """The full operand list of a ``whatif`` record: ``body.operands`` itself within the record
    bound, else the blob ``body.operands_summary.pointer`` names (project ``store``, or the
    run's in-memory copy), checked against its sha256; ``ValueError`` if absent or changed."""
    body = record["body"]
    summary = body.get("operands_summary")
    if summary is None:
        return body["operands"]
    pointer = summary["pointer"]
    tensor = next((p for p in record["pointers"]
                   if p.get("kind") == "tensor" and p.get("ref") == pointer.get("ref")), None)
    return load_json(pointer, store=store, run=run, tensor=tensor)


def whatif(run: Any, edit: Edit, consumer: Any, *, before_native: Any = None,
           store: Any = None) -> WhatIf:
    """Predict, seal, measure and split the change of ``consumer`` (a ``Margin``, a ``Logit`` or
    a stop address) under ``edit``; the session is left exactly where it was (checked by
    ``state_digest``; a difference raises ``RuntimeError``). ``before_native(record)``
    may persist and flush the sealed prediction; an exception prevents measurement.
    """
    prep = _prepare(run, edit, consumer, before_native, store=store)
    native = _measure(run, prep.p, edit, prep.checkpoint, prep.check)
    return _finish(run, prep, native, state_digest(run))


def _finish(run: Any, prep: _Prepared, native: dict[str, Any], after: str) -> WhatIf:
    """Split and judge the native change of a prepared what-if (needs the cursor's values, so
    the session is back where it was); ``after``: the session digest then.

    The split and the verdict use the exact chain on measured operands: computed at the cursor
    when the consumer was already captured at the seal (the sealed chain itself), else at the
    consumer inside the measurement (``_base_ahead``); then ``operands`` (that minus the sealed
    exact change) joins the split, so ``native = linear + nonlinear + operands + realisation +
    rounding effect + residual`` still holds term by term. The verdict holds the native change
    against the sealed emulated one (``verdict.decide``), the same at any cursor."""
    p, edit, found, prediction = prep.p, prep.edit, prep.found, prep.prediction
    body, record, before = prep.body, prep.record, prep.before
    reason = found.get("reason")
    if after != before:
        raise RuntimeError("whatif did not leave the session where it was (state digest "
                           f"{before[:12]} -> {after[:12]})")
    base_label = native["base_info"]["authority"] or MEASURED
    edited_label = native["edited_info"]["authority"] or MEASURED
    base, edited = _scalar(p, native["base"]), _scalar(p, native["edited"])
    change = edited - base
    size = magnitude(p.margin or p.logit, native["base"], native["edited"])
    dtype = native["base"].dtype
    measured = {"base": _claim(base, base_label), "edited": _claim(edited, edited_label),
                "delta": _claim(change, MEASURED), "moves": native["moves"]}
    report: dict[str, Any] = {
        "edit": body["edit"], "consumer": body["consumer"], "mode": p.mode,
        "record": record["id"], "prediction": prediction, "measured": measured,
        "session": {"before": before, "after": after, "unchanged": True},
    }
    tensors = {"native_base": native["base"], "native_edited": native["edited"],
               "realised": native["realised"], "native_delta": change}
    forecast = "posthoc" in native   # the operands past the cursor were forecast at the seal
    post = native.get("posthoc")
    if reason is None and post is not None and post.get("reason") is not None:
        reason = f"the exact chain on measured operands is refused ({post['reason']})"
        report["exact"] = {"state": "refused", "reason": reason}
    if reason is None:
        if post is None:       # every operand was measured when the prediction was sealed
            post = {**found, **_written_exact(run, p, edit, found, native["realised"])}
        exact, at_realised, spread = found["exact"], post["at_realised"], post["spread"]
        native_change = change.reshape(exact.shape)
        terms = []
        if "linear" in found:
            terms += [{"name": "linear", "value": _claim(found["linear"], LOCAL, "P14")},
                      {"name": "nonlinear", "value": _claim(exact - found["linear"], EXACT,
                                                            "P20"),
                       "meaning": "exact minus linear: the curvature the first-order term misses"}]
        else:
            terms.append({"name": "exact", "value": _claim(exact, EXACT)})
        if forecast:
            terms.append({"name": "operands", "value": _claim(post["exact"] - exact, EXACT, "P20"),
                          "meaning": "exact change on the measured operands minus the sealed "
                                     "exact change on operands forecast from "
                                     f"{body['sealed_at']['where']}"})
        rounding = native_change - at_realised
        terms.append(
            {"name": "realisation", "value": _claim(at_realised - post["exact"], EXACT, "P20"),
             "meaning": "exact change with the operand the edit wrote (read back) minus the "
                        "exact change with the requested operand, both on measured operands"})
        emulated = found.get("emulated")
        if emulated is not None:
            emulated = emulated.reshape(native_change.shape)
            terms += [
                {"name": "rounding effect", "value": _claim(emulated - at_realised, EXACT, "P20"),
                 "meaning": "the emulated change (the dtype's rounding where eager HF rounds, "
                            "sealed) minus the exact change with the written operand; derived "
                            "under the emulation premises",
                 **_within(rounding, spread, post["spread_reason"])},
                {"name": "residual", "value": _claim(native_change - emulated, MEASURED, "P20"),
                 "meaning": "native change minus the emulated change"}]
        else:
            terms.append({"name": "rounding", "value": _claim(rounding, MEASURED, "P20"),
                          "meaning": "native change minus the exact change with the written "
                                     "operand (no emulated chain: "
                                     f"{found.get('emulated_reason', 'not computed')})",
                          **_within(rounding, spread, post["spread_reason"])})
        report["exact"] = {
            "requested": _claim(post["exact"], EXACT, "P20"),
            "written": _claim(at_realised, EXACT, "P20"),
            "operands": "measured",
            "meaning": "F(x1) - F(x0) (P09 section 1) on the native operands of the unedited "
                       "run; written: x1 the value the edit wrote (read back). The verdict "
                       "holds native against the written one",
        }
        report["difference"] = {
            "native_minus_exact": {"value": _claim(native_change - exact, MEASURED),
                                   **_within(native_change - exact, found["exact_bound"],
                                             found["unbounded"])},
            **({"native_minus_linear": _claim(native_change - found["linear"], MEASURED)}
               if "linear" in found else {}),
        }
        report["split"] = {"proof": "P20", "terms": terms,
                           "rule": "native delta = sum of the terms (float64)"}
        tensors.update(exact=exact, exact_bound=found["exact_bound"], exact_measured=post["exact"],
                       linear=found.get("linear"), at_realised=at_realised, rounding_bound=spread,
                       emulated=emulated)
        effect = decide(native_change, emulated, size, dtype, exact=at_realised,
                        why_native_only=f"the emulated chain is refused "
                                        f"({found.get('emulated_reason')})",
                        parting=_parting(native))
    else:
        report["split"] = {"state": "refused", "reason": reason}
        effect = decide(change, None, size, dtype,
                        why_native_only=f"the prediction is refused ({reason})")
    # the verdict follows the P20 accounting (measured), never the worst-case interval bound
    report["effect"] = effect
    report["state"] = effect["state"]
    return WhatIf(report, record, tensors)


_guard = guard  # old private name, kept for one commit

__all__ = ["CELL_OPS", "OPS", "WHATIF", "Edit", "Plan", "WhatIf", "affected", "cell_plan",
           "guard", "plan", "stage", "state_digest", "whatif"]

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
   chain at the unedited point, P14) is ``derived-local``. The prediction is sealed as a record
   (kind ``whatif``) before anything native runs.
2. **Measurement**: checkpoint, stage the edit (moving to the edited layer first: forward, or back
   by layer boundaries when the layer is already committed), run natively to the consumer, read
   it, restore; the consumer's unedited value is read the same way. Labelled ``measured``.
3. **Split** of the native change (P20): ``native = linear + nonlinear + realisation +
   rounding`` with ``nonlinear = exact - linear``, ``realisation`` the change of the exact value
   when the edit operand is the value the edit actually wrote (read back natively) instead of the
   requested one (both ``derived-exact``) and ``rounding = native - exact(realised)``
   (``measured``, with its ``bound``). The linear number is never reported as the effect.

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
from mdb.core.records import num, seal_record

from .finite import BOUND, EXACT, LOCAL, MEASURED, SUPPLIED, f64
from .forecast import Logit, Margin, Refusal, RowModel, _split, producers, run_session_id

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
           cell: Any = None) -> list[RowModel]:
    """Row models for rows ``p.row .. p.consumer_row``. Stops at or before the cursor are taken
    as captured, except (``skip``) the ones the edit reaches; row ``j > p.row`` reads the cache
    rows ``p.row..j-1`` from the chain and its token from the prompt, from ``tokens`` or from the
    chain's argmax of row ``j - 1``. ``cell(layer, kind, value, spread)`` rewrites the cache
    rows ``0..p.row-1`` every row reads (a state-cell edit; the first row then reads them
    through it too)."""
    torch = _torch()
    models: list[RowModel] = []
    here = p.here

    def known_for(j: int):
        drop = p.affected.get(j, set()) if skip else set()

        def known(name: str) -> Any:
            if name == "start" or name in drop or run.parse(f"t{j}.{name}") > here:
                return None
            return run.read(f"t{j}.{name}")

        return known

    for j in range(p.row, p.consumer_row + 1):
        earlier = list(models)

        def past(layer: int, kind: str, earlier=earlier) -> tuple[Any, Any]:
            base = f64(run.read_cache(layer, kind)[..., : p.row, :])[0]
            spread0 = torch.zeros_like(base)
            if cell is not None:
                base, spread0 = cell(layer, kind, base, spread0)
            parts, spreads = [base], [spread0]
            for model in earlier:
                name = f"L{layer}.kr" if kind == "k" else f"L{layer}.v"
                value, spread = model.get(name), model.bound(name)
                shape = (model.kv_heads, -1, model.head_dim)
                value = value.reshape(shape)[:, -1:, :]
                parts.append(value)
                spreads.append(None if spread is None else spread.reshape(shape)[:, -1:, :])
            if any(s is None for s in spreads):
                return torch.cat(parts, 1), None
            return torch.cat(parts, 1), torch.cat(spreads, 1)

        def token(j=j, earlier=earlier) -> tuple[int, str | None]:
            if j < len(run.tokens):
                return run.tokens[j], None
            if tokens is not None:
                return tokens[j], None
            previous = earlier[-1]
            sampled = int(previous.get("sample"))
            if bounds and previous.bound("sample") is None:
                return sampled, (f"the sampled token of t{j - 1} is not certified: "
                                 + previous.unbounded.get("sample", "no bound on its logits"))
            return sampled, None

        first = j == p.row
        models.append(RowModel(run, j, known_for(j), bounds=bounds,
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
    edit writes (the operand cast to the model dtype, one rounded op; ``b0`` the bound of ``x0``
    itself when ``x0`` is a prediction)."""
    torch = _torch()
    port_shape = tuple(run.address_shape(edit.address))  # the edited row's frame, not the cursor's
    if isinstance(edit.value, torch.Tensor):
        value = f64(edit.value).broadcast_to(port_shape).reshape(shape)
    else:
        value = torch.full(shape, float(edit.value), dtype=torch.float64)
    mask = (torch.ones(shape, dtype=torch.bool) if edit.mask is None
            else edit.mask.broadcast_to(port_shape).reshape(shape))
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


def _predict_cell(run: Any, p: Plan, edit: Edit) -> dict[str, Any]:
    """The prediction of a state-cell edit (``base``, ``exact`` and bounds, ``linear``), or
    ``reason``: the chain from the first affected row with the cell written into the cache rows
    every row reads."""
    torch = _torch()
    if p.row > p.here[0]:
        return {"reason": f"the first read of {edit.address} is t{p.row}.{p.label}, in a row "
                          f"that has not started at the cursor {run.where}: none of its "
                          "operands exist yet"}
    conflict = _edits_on_path(run, p)
    if conflict is not None:
        return {"reason": conflict}
    written = _written_cell(run, edit)
    base_chain = _chain(run, p, bounds=True)
    edited = _chain(run, p, bounds=True, skip=True, cell=_cell_writer(edit, written, "ideal"))
    base, base_bound = _consumer_value(p, base_chain[-1])
    value, bound = _consumer_value(p, edited[-1])
    out: dict[str, Any] = {
        "base": base, "base_bound": base_bound, "exact": value - base,
        "exact_bound": None if bound is None or base_bound is None else bound + base_bound,
        "edit_shape": None, "written": written, "operands": base_chain + edited,
        "unbounded": _unbounded(p, edited[-1], base_chain[-1]),
        "premises": set().union(*(m.premises for m in base_chain + edited)),
    }
    tokens = _sampled_tokens(p, base_chain, run)
    flipped = {j: (t, e) for (j, t), e in zip(tokens.items(),
                                              _sampled_tokens(p, edited, run).values())
               if t != e}
    if flipped:
        j, (was, now) = next(iter(flipped.items()))
        out["linear_reason"] = (f"the edit flips the sampled token of t{j} ({was} -> {now}): "
                                "the first-order term does not cross a token change")
        return out
    import torch.autograd.forward_ad as fwad

    with torch.inference_mode(False), fwad.dual_level():
        dual = _chain(run, p, bounds=False, skip=True, tokens=tokens,
                      cell=_cell_writer(edit, written, "tangent"))
        primal, tangent = fwad.unpack_dual(_consumer_value(p, dual[-1])[0])
        out["linear"] = (torch.zeros_like(primal) if tangent is None else tangent).detach().clone()
    return out


def _predict(run: Any, p: Plan, edit: Edit) -> dict[str, Any]:
    """The prediction's tensors (``base``, ``exact`` and bounds, ``linear``), or ``reason``."""
    torch = _torch()
    if edit.cell is not None:
        return _predict_cell(run, p, edit)
    if p.row > p.here[0]:
        return {"reason": f"the edit site t{p.row}.{p.label} is in a row that has not started at "
                          f"the cursor {run.where}: none of its operands exist yet"}
    conflict = _edits_on_path(run, p)
    if conflict is not None:
        return {"reason": conflict}
    base_chain = _chain(run, p, bounds=True)
    first = base_chain[0]
    x0, b0 = first.get(p.label), first.bound(p.label)
    shape = tuple(x0.shape)
    requested, delta, spread = _request(run, edit, shape, x0, b0, first.u)
    edited = _chain(run, p, bounds=True, skip=True,
                    overrides={p.label: lambda value, bound: (requested, spread)})
    base, base_bound = _consumer_value(p, base_chain[-1])
    value, bound = _consumer_value(p, edited[-1])
    out: dict[str, Any] = {
        "base": base, "base_bound": base_bound, "exact": value - base,
        "exact_bound": None if bound is None or base_bound is None else bound + base_bound,
        "edit_shape": shape, "operands": base_chain + edited,
        "unbounded": _unbounded(p, edited[-1], base_chain[-1]),
        "premises": set().union(*(m.premises for m in base_chain + edited)),
    }
    tokens = _sampled_tokens(p, base_chain, run)
    flipped = {j: (t, e) for (j, t), e in zip(tokens.items(),
                                              _sampled_tokens(p, edited, run).values())
               if t != e}
    if flipped:
        j, (was, now) = next(iter(flipped.items()))
        out["linear_reason"] = (f"the edit flips the sampled token of t{j} ({was} -> {now}): "
                                "the first-order term does not cross a token change")
        return out
    import torch.autograd.forward_ad as fwad

    with torch.inference_mode(False), fwad.dual_level():
        dual = _chain(run, p, bounds=False, skip=True, tokens=tokens,
                      overrides={p.label: lambda value, bound: (fwad.make_dual(value, delta),
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
                       selection=edit.selection)
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


def _measure_cell(run: Any, p: Plan, edit: Edit, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """``_measure`` of a state-cell edit: staged at ``p.stage_at``; ``realised`` is the cache
    tensor the native layer call wrote, read back at the consumer."""
    layer, tensor = int(edit.cell["layer"]), edit.cell["tensor"]
    out: dict[str, Any] = {"moves": []}
    try:
        if p.mode == "pending":
            run.until(p.consumer_stop)
            out["moves"].append(f"until {p.consumer_stop}")
        out["base"] = run.read(p.consumer_stop).clone()
        out["base_info"] = run.read_info(p.consumer_stop)
        if p.mode == "pending":
            run.restore(checkpoint)
            out["moves"].append(f"restore {checkpoint['where']}")
        else:
            while run.parse(run.where) >= run.parse(p.stage_at):
                run.reverse_step("layer")
            out["moves"].append(f"reverse-step layer to {run.where}")
            if run.where != p.stage_at:
                run.until(p.stage_at)
                out["moves"].append(f"until {p.stage_at}")
        stage(run, edit)
        out["moves"].append(f"set {edit.address} at {run.where}")
        run.until(p.consumer_stop)
        out["moves"].append(f"until {p.consumer_stop}")
        out["edited"] = run.read(p.consumer_stop).clone()
        out["edited_info"] = run.read_info(p.consumer_stop)
        out["realised"] = run.read_cache(layer, tensor, staged=False).clone()
    finally:
        run.restore(checkpoint, defer_interrupt=True)
    out["moves"].append(f"restore {checkpoint['where']}")
    return out


def _measure(run: Any, p: Plan, edit: Edit, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Native consumer values without and with the edit, the value the edit wrote, and the
    native-call trail; restores ``checkpoint`` whatever happens."""
    if edit.cell is not None:
        return _measure_cell(run, p, edit, checkpoint)
    layer = _split(p.label)[0]
    entry = f"t{p.row}.L{layer}.h"
    out: dict[str, Any] = {"moves": []}
    try:
        ahead = p.consumer_key > p.here
        if ahead:
            run.until(p.consumer_stop)
            out["moves"].append(f"until {p.consumer_stop}")
        out["base"] = run.read(p.consumer_stop).clone()
        out["base_info"] = run.read_info(p.consumer_stop)
        if ahead:
            run.restore(checkpoint)
            out["moves"].append(f"restore {checkpoint['where']}")
        if p.mode == "rewind":
            while run.parse(run.where) >= run.parse(entry):
                run.reverse_step("layer")
            out["moves"].append(f"reverse-step layer to {run.where}")
        if p.mode != "pending" and run.where != entry:
            run.until(entry)
            out["moves"].append(f"until {entry}")
        run.set(p.edit_stop, edit.value, edit.op, mask=edit.mask, selection=edit.selection)
        out["moves"].append(f"set {p.edit_stop}")
        if run.parse(run.where) < p.consumer_key:
            run.until(p.consumer_stop)
            out["moves"].append(f"until {p.consumer_stop}")
        out["edited"] = run.read(p.consumer_stop).clone()
        out["edited_info"] = run.read_info(p.consumer_stop)
        out["realised"] = run.read(p.edit_stop).clone()
    finally:
        # An interrupt (Ctrl-C) during this restore must not roll it back to the edited
        # excursion: the restore completes first and the interrupt is raised after it.
        run.restore(checkpoint, defer_interrupt=True)
    out["moves"].append(f"restore {checkpoint['where']}")
    return out


# --------------------------------------------------------------------------- report


def _scalar(p: Plan, value: Any) -> Any:
    flat = f64(value).reshape(-1)
    if p.margin is not None:
        return flat[p.margin.a] - flat[p.margin.b]
    if p.logit is not None:
        return flat[p.logit.i]
    return flat


def _claim(value: Any, authority: str, proof: str | None = None) -> dict[str, Any]:
    """One labelled number for a scalar; ``{max_abs, norm}`` (both labelled) for a vector."""
    if value.dim() == 0:
        return num(float(value), authority, proof=proof)
    flat = value.reshape(-1)
    return {"max_abs": num(float(flat.abs().max()), authority, proof=proof),
            "norm": num(float(flat.norm()), authority, proof=proof)}


def _unbounded(p: Plan, *models: RowModel) -> str:
    """Why the consumer of a chain has no rounding bound (the first reason found)."""
    return next((m.unbounded[p.consumer_label] for m in models
                 if p.consumer_label in m.unbounded), "no rounding bound for this chain")


def _within(gap: Any, bound: Any, reason: str) -> dict[str, Any]:
    if bound is None:
        return {"bound": {"state": "unresolved", "reason": reason}, "within_bound": None}
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


def _operands(run: Any, models: list[RowModel]) -> list[dict[str, Any]]:
    from mdb.engine.ports import tensor_digest

    seen: dict[str, dict[str, Any]] = {}
    for model in models:
        tensors = {**{f"t{model.row}.{n}": t for n, t in model.captured.items()},
                   **{f"{n}[row=0:{model.row}]": t for n, t in model.cache_reads.items()}}
        for address, tensor in tensors.items():
            seen.setdefault(address, {"address": address,
                                      "sha256": tensor_digest(tensor.contiguous()),
                                      "shape": list(tensor.shape)})
    return list(seen.values())


@dataclass
class WhatIf:
    """A what-if result: ``report`` (bounded JSON, every number labelled), the sealed prediction
    ``record`` and the float64 ``tensors`` behind the numbers (for in-process callers)."""

    report: dict[str, Any]
    record: dict[str, Any]
    tensors: dict[str, Any]


def _within_view(found: dict[str, Any]) -> dict[str, Any]:
    if found["exact_bound"] is None:
        return {"bound": {"state": "unresolved", "reason": found["unbounded"]}}
    return {"bound": num(float(found["exact_bound"].reshape(-1).max()), BOUND),
            "premises": sorted(found["premises"] | {"round to nearest, no overflow or underflow",
                                                    "the edit operand is cast to the model dtype "
                                                    "and applied in one rounded op"})}


def whatif(run: Any, edit: Edit, consumer: Any) -> WhatIf:
    """Predict, seal, measure and split the change of ``consumer`` (a ``Margin``, a ``Logit`` or
    a stop address) under ``edit``; the session is left exactly where it was (checked by
    ``state_digest``; a difference raises ``RuntimeError``)."""
    torch = _torch()
    p = plan(run, edit, consumer)
    before = state_digest(run)
    checkpoint = run.checkpoint()
    guard(run, checkpoint)
    try:
        found = _predict(run, p, edit)
    except Refusal as error:
        found = {"reason": str(error)}
    reason = found.get("reason")
    prediction: dict[str, Any]
    if reason is not None:
        prediction = {"state": "refused", "reason": reason}
    else:
        exact_view = {"value": _claim(found["exact"], EXACT), **_within_view(found)}
        if "linear" in found:
            linear_view: dict[str, Any] = {"value": _claim(found["linear"], LOCAL, "P14"),
                                           "meaning": "gradient of the consumer times the edit "
                                                      "displacement; not the effect"}
        else:
            linear_view = {"state": "refused", "reason": found["linear_reason"]}
        prediction = {"state": "resolved", "linear": linear_view, "exact": exact_view,
                      "base": _claim(found["base"], EXACT)}
    body = {
        "schema": SCHEMA, "edit": _edit_view(run, p, edit),
        "consumer": {"address": p.consumer_text, "row": p.consumer_key[0],
                     "index": p.consumer_key[1],
                     "kind": "margin" if p.margin else "logit" if p.logit else "value",
                     **({"indices": [p.margin.a, p.margin.b]} if p.margin else {})},
        "sealed_at": {"where": run.where, "cut": checkpoint["cut"], "row": p.here[0],
                      "index": p.here[1], "transition": int(checkpoint["transitions"])},
        "mode": p.mode, "prediction": prediction,
        "method": {"exact": "port-program identities on values at the cursor, the edit "
                            "written at the edited port (rows chained through the cache)",
                   "linear": "forward-mode derivative of the same chain at the unedited point "
                             "(P14)"},
        "operands": [] if reason else _operands(run, found["operands"]),
    }
    record = seal_record(session_id=run_session_id(run), kind="whatif",
                         model_identity=run.adapter.model_identity,
                         execution=dict(run.adapter.execution),
                         authority=LOCAL if reason is None else SUPPLIED, body=body)
    native = _measure(run, p, edit, checkpoint)
    after = state_digest(run)
    if after != before:
        raise RuntimeError("whatif did not leave the session where it was (state digest "
                           f"{before[:12]} -> {after[:12]})")
    base_label = native["base_info"]["authority"] or MEASURED
    edited_label = native["edited_info"]["authority"] or MEASURED
    base, edited = _scalar(p, native["base"]), _scalar(p, native["edited"])
    change = edited - base
    measured = {"base": _claim(base, base_label), "edited": _claim(edited, edited_label),
                "delta": _claim(change, MEASURED), "moves": native["moves"]}
    report: dict[str, Any] = {
        "edit": body["edit"], "consumer": body["consumer"], "mode": p.mode,
        "record": record["id"], "prediction": prediction, "measured": measured,
        "session": {"before": before, "after": after, "unchanged": True},
    }
    tensors = {"native_base": native["base"], "native_edited": native["edited"],
               "realised": native["realised"], "native_delta": change}
    if reason is None:
        if edit.cell is not None:
            chain = _chain(run, p, bounds=True, skip=True,
                           cell=_cell_writer(edit, native["realised"], "written"))
        else:
            shape = found["edit_shape"]
            realised = f64(native["realised"]).reshape(shape)
            chain = _chain(run, p, bounds=True, skip=True,
                           overrides={p.label: lambda value, bound: (realised,
                                                                     torch.zeros_like(realised))})
        value, bound = _consumer_value(p, chain[-1])
        at_realised = value - found["base"]
        spread = None if bound is None or found["base_bound"] is None else (
            bound + found["base_bound"])
        exact = found["exact"]
        native_change = change.reshape(exact.shape)
        terms = []
        if "linear" in found:
            terms += [{"name": "linear", "value": _claim(found["linear"], LOCAL, "P14")},
                      {"name": "nonlinear", "value": _claim(exact - found["linear"], EXACT,
                                                            "P20"),
                       "meaning": "exact minus linear: the curvature the first-order term misses"}]
        else:
            terms.append({"name": "exact", "value": _claim(exact, EXACT)})
        rounding = native_change - at_realised
        terms += [
            {"name": "realisation", "value": _claim(at_realised - exact, EXACT, "P20"),
             "meaning": "exact change with the operand the edit wrote (read back) minus the "
                        "exact change with the requested operand"},
            {"name": "rounding", "value": _claim(rounding, MEASURED, "P20"),
             "meaning": "native change minus the exact change with the written operand",
             **_within(rounding, spread, _unbounded(p, chain[-1], *found["operands"]))},
        ]
        report["difference"] = {
            "native_minus_exact": {"value": _claim(native_change - exact, MEASURED),
                                   **_within(native_change - exact, found["exact_bound"],
                                             found["unbounded"])},
            **({"native_minus_linear": _claim(native_change - found["linear"], MEASURED)}
               if "linear" in found else {}),
        }
        report["split"] = {"proof": "P20", "terms": terms,
                           "rule": "native delta = sum of the terms (float64)"}
        tensors.update(exact=exact, exact_bound=found["exact_bound"],
                       linear=found.get("linear"), at_realised=at_realised, rounding_bound=spread)
    else:
        report["split"] = {"state": "refused", "reason": reason}
    return WhatIf(report, record, tensors)


_guard = guard  # old private name, kept for one commit

__all__ = ["CELL_OPS", "OPS", "WHATIF", "Edit", "Plan", "WhatIf", "affected", "cell_plan",
           "guard", "plan", "stage", "state_digest", "whatif"]

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

The session's ``state_digest`` (cursor, cut fingerprint, transition count, staged and committed
edits, lookahead state) is taken before and after; a difference raises. Every edit tensor the
checkpoint names is held (``Run.hold``) before anything native runs, so the restore can rebuild
them; a checkpoint naming a tensor the run does not hold is refused first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mdb.core.pointers import seal_digest
from mdb.core.records import declare_kind, num, seal_record

from .finite import BOUND, EXACT, LOCAL, MEASURED, SUPPLIED, f64
from .forecast import Logit, Margin, Refusal, RowModel, _session_id, _split, producers

SCHEMA = "mdb-whatif-v1"
OPS = ("replace", "add", "scale")
WHATIF = declare_kind(
    "whatif",
    required=("schema", "edit", "consumer", "sealed_at", "mode", "prediction"),
    plain_numeric=("edit.row", "edit.layer", "edit.mask_count", "edit.shape", "consumer.row",
                   "consumer.index", "consumer.indices", "sealed_at.row", "sealed_at.index",
                   "sealed_at.transition", "operands.*.shape"),
)


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


def affected(run: Any, row: int, label: str, last_row: int) -> dict[int, set[str]]:
    """Stops of rows ``row..last_row`` whose value an edit at ``t<row>.<label>`` can change: the
    stops downstream of it in its own row, and in each later row the stops downstream of a cache
    row it changed (``L<i>.kr`` reads the key cache, ``L<i>.A`` the value cache) or of a sampled
    token it changed (``emb`` of a row past the prompt)."""
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


def plan(run: Any, edit: Edit, consumer: Any) -> Plan:
    """Resolve and check a what-if request without running anything native."""
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
           skip: bool = False, tokens: dict[int, int] | None = None) -> list[RowModel]:
    """Row models for rows ``p.row .. p.consumer_row``. Stops at or before the cursor are taken
    as captured, except (``skip``) the ones the edit reaches; row ``j > p.row`` reads the cache
    rows ``p.row..j-1`` from the chain and its token from the prompt, from ``tokens`` or from the
    chain's argmax of row ``j - 1``."""
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
            parts, spreads = [base], [torch.zeros_like(base)]
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
                               past=None if first else past, token=None if first else token))
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
    port_shape = tuple(run.port_shape(_split(run.stops()[run.parse(edit.address)[1]])[1]))
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


def _predict(run: Any, p: Plan, edit: Edit) -> dict[str, Any]:
    """The prediction's tensors (``base``, ``exact`` and bounds, ``linear``), or ``reason``."""
    torch = _torch()
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


def _guard(run: Any, checkpoint: dict[str, Any]) -> None:
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


def _measure(run: Any, p: Plan, edit: Edit, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Native consumer values without and with the edit, the value the edit wrote, and the
    native-call trail; restores ``checkpoint`` whatever happens."""
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
        run.restore(checkpoint)
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
    _guard(run, checkpoint)
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
    record = seal_record(session_id=_session_id(run), kind="whatif",
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


__all__ = ["OPS", "WHATIF", "Edit", "Plan", "WhatIf", "affected", "plan", "state_digest",
           "whatif"]

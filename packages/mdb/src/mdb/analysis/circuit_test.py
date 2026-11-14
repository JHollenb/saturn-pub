"""``circuit test``: a built circuit checked against a REFERENCE run by native activation patching.

Zero-ablation is not offered (issue M4): zeroing components is off-distribution, and on
Qwen2.5-1.5B zeroing the 26,924-node complement moved the margin by 27.75 against a circuit
effect of 1.81, so the old sufficiency verdict was always ``not sufficient``. Here every
patched component takes the value a reference computed instead: another run of the debugger
(the corrupted / counterfactual / opposite-answer prompt), or the per-component mean over a
prompt set's items.

Components (what is patched) live at the consumer's row ``r`` and are exactly what the engine
can stage there with ``Run.set``:

* ``embedding``: the row's input ``t<r>.L0.h`` (one component);
* ``head``: head ``h``'s block of ``t<r>.L<l>.A`` (the o_proj input), every head of every layer;
* ``mlp``: ``t<r>.L<l>.m`` (the MLP output) as one component, or, in a layer whose MLP the
  circuit expanded into neurons, ``neuron``: neuron ``i`` of ``t<r>.L<l>.f`` (the down_proj
  input), every neuron of that layer.

A circuit node maps to components: the embedding to ``L0.h``; a head node to its head; a source
row node under a head to that head (coarsened: the reference's attention row has another
length whenever the prompts differ, so ``p[h,src]`` is not patched); a neuron node to its
neuron; an attention or MLP term that the circuit did not expand (depth 1) to every head of
that layer, resp. its ``m``. Remainder, bias and edit nodes are not components. ``+N more``
groups are not in the circuit: their members are the complement, counted (``from_groups``),
and members whose own subtree was never built are counted again as ``unexamined``. Paged
groups were examined: the in-memory node tree holds every node, paged ones included (paging
applies to the record only).

Earlier rows reach the consumer row only through the cache: they stay this run's. The test is
over the consumer row's components (``rows: consumer``); once every head's ``A`` is patched
the cache is not read at all, which is why patching every component reproduces the reference
(``full``).

Measurements, each ONE native pass of the consumer row from layer 0 with all of its edits
staged layer by layer (``compare.measure``; the session is restored and its digest checked):

* ``base``: this run, nothing patched; ``ref``: the consumer read in the reference run at its
  ``@last`` row (for a prompt-set mean there is no single run: ``ref`` is then ``full``);
* ``ko`` (knock-out): only the circuit's components patched;
* ``keep`` (keep-only): every component NOT in the circuit patched, the circuit's components
  computed natively;
* ``full``: every component patched (a check: equal to ``ref`` bit for bit for a reference run).

``faithfulness = (keep - ref) / (base - ref)`` and ``knockout_share = (base - ko) / (base -
ref)``, both ``measured`` (arithmetic on measured values). They are refused when ``|base -
ref|`` is under one spacing of the consumer's dtype at its magnitude: the reference does not
change the consumer. There is no pass/fail bar; the words given are derived from the stated
one-spacing rule only.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from mdb.core.bounded import put_json
from mdb.core.kinds import CIRCUIT_TEST, procedure
from mdb.core.pointers import RecordError, check_bounded
from mdb.core.records import num, seal_record

from .finite import EXACT, MEASURED, f64
from .forecast import Logit, Margin, Refusal, run_session_id
from .propagate import Edit
from .reads import layer_module

SCHEMA = "mdb-circuit-test-v1"
_PORT = re.compile(r"t(?P<row>\d+)\.L(?P<layer>\d+)\.(?P<port>\w+)")

DEFINITION = (
    "components at the consumer's row: the row input L0.h (embedding), each head's block of "
    "L<l>.A (o_proj input), and per layer the MLP output L<l>.m, or each neuron of L<l>.f where "
    "the circuit expanded that MLP into neurons; source-row nodes count as their head (p is not "
    "patched); remainder, bias and edit nodes are not components")
ROWS = ("the test is over the consumer row's components; earlier rows are this run's (they "
        "reach the consumer row only through the cache, which a fully patched A does not read)")
RULE = ("one spacing = the gap between adjacent values of the consumer's dtype at the largest "
        "|logit| the consumer read in base, ref, ko and keep; a difference under one spacing is "
        "not measurable")


def _torch():
    import torch

    return torch


# --------------------------------------------------------------------------- components


@dataclass
class Port:
    """One patchable port at the consumer row: ``unit`` (embedding, head, mlp, neuron), ``size``
    components, the indices in the circuit and the indices held by ``+N more`` members
    (``grouped``; ``unexamined`` those whose subtree was never built)."""

    address: str              # t<r>.L<l>.<port>
    layer: int
    unit: str
    size: int
    circuit: set[int] = field(default_factory=set)
    grouped: set[int] = field(default_factory=set)
    unexamined: set[int] = field(default_factory=set)

    @property
    def complement(self) -> list[int]:
        return [i for i in range(self.size) if i not in self.circuit]


def _split(port: str) -> tuple[int, int, str]:
    found = _PORT.fullmatch(port)
    if found is None:
        raise Refusal(f"circuit node at {port!r} is not a layer port of the consumer row")
    return int(found["row"]), int(found["layer"]), found["port"]


def components(run: Any, circuit: Any) -> tuple[list[Port], dict[str, int]]:
    """The component table of ``circuit`` at its consumer row (layer order: ``L0.h``, then per
    layer ``A`` and ``m`` or ``f``) and counts of nodes that are not components or were
    coarsened (``source`` nodes listed under a head)."""
    config = run.adapter.model.config
    heads = int(config.num_attention_heads)
    d_ff = int(config.intermediate_size)
    row = circuit.consumer.row
    depth = int(circuit.record["body"]["depth"])
    expanded_mlp = set()
    stack = [circuit.root]
    while stack:   # layers whose MLP the circuit expanded into neurons (kept or grouped)
        node = stack.pop()
        stack += node.children
        members = [(node.kind, node.port)] + [(m.kind, m.port) for m in node.members]
        for kind, port in members:
            if kind == "neuron":
                expanded_mlp.add(_split(port)[1])
    table = {f"t{row}.L0.h": Port(f"t{row}.L0.h", 0, "embedding", 1)}
    for layer in range(run.adapter.layers):
        table[f"t{row}.L{layer}.A"] = Port(f"t{row}.L{layer}.A", layer, "head", heads)
        if layer in expanded_mlp:
            table[f"t{row}.L{layer}.f"] = Port(f"t{row}.L{layer}.f", layer, "neuron", d_ff)
        else:
            table[f"t{row}.L{layer}.m"] = Port(f"t{row}.L{layer}.m", layer, "mlp", 1)
    notes: dict[str, Any] = {"source_nodes_coarsened": 0, "grouped_sources_coarsened": 0,
                             "circuit_nodes": [], "group_nodes": []}

    def port_of(kind: str, port: str) -> tuple[Port, list[int]]:
        r, layer, name = _split(port)
        if r != row:
            raise Refusal(f"circuit node at {port} is not on the consumer row t{row}")
        if kind == "embedding":
            return table[f"t{row}.L0.h"], [0]
        if kind == "attn":
            return table[f"t{row}.L{layer}.A"], list(range(heads))
        if kind == "mlp":
            entry = table.get(f"t{row}.L{layer}.m") or table[f"t{row}.L{layer}.f"]
            return entry, list(range(entry.size))
        raise Refusal(f"no component for a {kind} node at {port}")

    def walk(node: Any) -> None:
        for child in node.children:
            if child.kind == "more":
                notes["group_nodes"].append(child.ref)
                expands = {"attn": depth > 1, "mlp": depth > 1, "head": depth > 2}
                for member in child.members:
                    if member.kind == "source":
                        notes["grouped_sources_coarsened"] += 1
                        continue
                    if member.kind in ("head", "neuron"):
                        entry = table[f"t{row}.L{_split(member.port)[1]}."
                                      f"{'A' if member.kind == 'head' else 'f'}"]
                        indices = [member.index[0]]
                    else:
                        entry, indices = port_of(member.kind, member.port)
                    entry.grouped.update(indices)
                    if expands.get(member.kind, False):
                        entry.unexamined.update(indices)
                continue
            if child.kind in ("remainder", "bias", "edit"):
                continue
            if child.kind in ("head", "neuron", "embedding", "attn", "mlp"):
                expanded = any(c.kind in ("head", "neuron", "more") for c in child.children)
                if child.kind in ("head", "neuron", "embedding") or not expanded:
                    notes["circuit_nodes"].append(child.ref)
            if child.kind == "head":
                table[f"t{row}.L{_split(child.port)[1]}.A"].circuit.add(child.index[0])
                notes["source_nodes_coarsened"] += sum(1 for c in child.children
                                                       if c.kind == "source")
                continue
            if child.kind == "neuron":
                table[f"t{row}.L{_split(child.port)[1]}.f"].circuit.add(child.index[0])
                continue
            if child.kind in ("attn", "mlp") and any(c.kind in ("head", "neuron", "more")
                                                     for c in child.children):
                walk(child)         # expanded: its heads / neurons are the components
                continue
            entry, indices = port_of(child.kind, child.port)
            entry.circuit.update(indices)

    walk(circuit.root)
    return list(table.values()), notes


def component_list(ports: list[Port]) -> list[dict[str, Any]]:
    """The full table as plain JSON (what the record's sha256-checked blob holds)."""
    return [{"address": p.address, "unit": p.unit, "size": p.size,
             "circuit": sorted(p.circuit), "from_groups": sorted(p.grouped - p.circuit),
             "unexamined": sorted(p.unexamined - p.circuit)} for p in ports]


def counts(ports: list[Port]) -> dict[str, Any]:
    """Component counts (``derived-exact``): total, circuit, complement, the complement held by
    ``+N more`` groups, and of those the ones in subtrees never built; per unit too."""
    def tally(selected: list[Port]) -> dict[str, int]:
        circuit = sum(len(p.circuit) for p in selected)
        total = sum(p.size for p in selected)
        grouped = sum(len(p.grouped - p.circuit) for p in selected)
        return {"total": total, "circuit": circuit, "complement": total - circuit,
                "from_groups": grouped,
                "unexamined": sum(len(p.unexamined - p.circuit) for p in selected),
                "not_listed": total - circuit - grouped}

    whole = tally(ports)
    out: dict[str, Any] = {key: num(value, EXACT) for key, value in whole.items()}
    out["by_unit"] = {unit: {key: num(value, EXACT) for key, value in tally(
        [p for p in ports if p.unit == unit]).items()}
        for unit in ("embedding", "head", "mlp", "neuron") if any(p.unit == unit for p in ports)}
    return out


# --------------------------------------------------------------------------- reference values


@dataclass
class Reference:
    """Reference values per component port (each the port's tensor at the reference row, in the
    run's dtype and device), what they came from, and the reference consumer when one run
    computed it."""

    kind: str                  # run | set-mean
    name: str
    values: dict[str, Any]     # port suffix L<l>.<port> -> tensor
    origin: dict[str, Any]     # the pointer fields every edit's origin shares
    consumer: Any = None       # (float64 value, magnitude) at the reference row, run only
    info: dict[str, Any] = field(default_factory=dict)


def _suffix(address: str) -> str:
    return address.split(".", 1)[1]


def _check_model(run: Any, other: Any, name: str) -> None:
    if other.adapter.model_identity != run.adapter.model_identity:
        raise Refusal(f"reference {name} is another model than this run: a component of one "
                      "cannot be patched into the other (same model required)")
    if other.program.family != run.program.family:
        raise Refusal(f"reference {name} is family {other.program.family}, this run is "
                      f"{run.program.family}")
    scope = run.adapter.execution.get("numeric_scope")
    if other.adapter.execution.get("numeric_scope") != scope:
        raise Refusal(f"reference {name} runs under another numeric scope than this run")


def _read_row(run: Any, other: Any, row: int, ports: list[Port], label: str) -> dict[str, Any]:
    """The component ports of ``other`` at ``row`` (``read`` copies; committed layers are served
    by one full-layer probe each), shape-checked against this run's ports."""
    out = {}
    for port in ports:
        address = f"t{row}.{_suffix(port.address)}"
        try:
            value = other.read(address)
        except ValueError as error:
            raise Refusal(f"{label}: {address} cannot be read ({error})") from None
        want = tuple(run.address_shape(port.address))
        if tuple(value.shape) != want:
            raise Refusal(f"{label}: {address} has shape {list(value.shape)}, this run's "
                          f"{port.address} has {list(want)} (shape mismatch: not the same model)")
        if value.dtype != run.adapter.dtype or value.device != run.adapter.device:
            value = value.to(device=run.adapter.device, dtype=run.adapter.dtype)
        out[_suffix(port.address)] = value
    return out


def from_run(run: Any, other: Any, name: str, ports: list[Port], consumer: Any) -> Reference:
    """Reference values from another run at its ``@last`` row; refused for this run itself,
    another model, or a reference that has not computed that row's logits."""
    from .compare import consumer_value
    from .verdict import magnitude

    if other is run:
        raise Refusal(f"reference {name} is this run: name another run (info runs)")
    _check_model(run, other, name)
    row = len(other.tokens) - 1
    stop = f"t{row}.logits"
    if other.parse(stop) > other.parse(other.where):
        raise Refusal(f"reference {name} has not computed {stop} (it is at {other.where}): "
                      f"run it there first (frame {name}: then until logits)")
    values = _read_row(run, other, row, ports, f"reference {name}")
    logits = other.read(stop)
    origin = {"origin": name, "via": "live", "where": other.where,
              "cut": other.checkpoint(pin=False)["cut"], "recipe": None, "row": row}
    return Reference("run", name, values, origin,
                     consumer=(consumer_value(consumer, logits), magnitude(consumer, logits)),
                     info={"row": row, "where": other.where,
                           "edited": bool(other.edited)})


def from_set(run: Any, name: str, prompts: list[tuple[str, list[int]]], ports: list[Port]
             ) -> Reference:
    """The per-component mean over a prompt set's items, each read at the item's ``@last``
    row: one fresh batch-1 run per item, made, read and released in turn. The sum is float64
    while the call runs; the mean is cast once to the run's dtype and nothing wider is kept."""
    from mdb.engine import Run

    torch = _torch()
    if not prompts:
        raise Refusal(f"set {name} has no items")
    sums: dict[str, Any] = {}
    rows = []
    for item, tokens in prompts:
        other = Run(run.adapter, tokens)
        try:
            row = len(tokens) - 1
            other.until(f"t{row}.L{run.adapter.layers - 1}.h_out")
            for key, value in _read_row(run, other, row, ports,
                                        f"set {name} item {item}").items():
                wide = value.to(dtype=torch.float64)
                sums[key] = wide if key not in sums else sums[key].add_(wide)
                del wide
            rows.append(row)
        finally:
            other.close()
            del other
    count = len(prompts)
    values = {}
    for key in list(sums):
        values[key] = sums.pop(key).div_(count).to(dtype=run.adapter.dtype)
    origin = {"origin": f"set:{name}", "via": "mean", "where": None, "cut": None,
              "recipe": None, "items": count}
    return Reference("set-mean", name, values, origin,
                     info={"items": count, "rows": sorted(set(rows))})


# --------------------------------------------------------------------------- patching


def _mask(run: Any, port: Port, indices: list[int]) -> Any:
    torch = _torch()
    mask = torch.zeros(tuple(run.address_shape(port.address)), dtype=torch.bool)
    if port.unit == "head":
        dh = run.adapter.head_dim
        for h in indices:
            mask[..., h * dh:(h + 1) * dh] = True
    else:
        mask[..., indices] = True
    return mask


def _selection(port: Port, indices: list[int]) -> str:
    if port.unit == "head":
        return "|".join(f"h{h}" for h in indices)
    return "|".join(f"i={i}" for i in indices)


def patches(run: Any, ports: list[Port], which: str, reference: Reference) -> list[Edit]:
    """The edits that patch ``which`` (``circuit``, ``complement`` or ``all``) from the
    reference: one ``replace`` per port with any selected component, masked to the selected
    heads / neurons unless the whole port is selected; each carries an origin pointer."""
    from mdb.engine.ports import tensor_digest

    edits = []
    for port in ports:
        chosen = (sorted(port.circuit) if which == "circuit" else port.complement
                  if which == "complement" else list(range(port.size)))
        if not chosen:
            continue
        suffix = _suffix(port.address)
        value = reference.values[suffix]
        source = (f"{reference.name}:t{reference.origin['row']}.{suffix}"
                  if reference.kind == "run" else f"set:{reference.name}:t@last.{suffix}")
        origin = {**{k: v for k, v in reference.origin.items() if k != "row"},
                  "address": source, "sha256": tensor_digest(value),
                  "shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
        if len(chosen) == port.size:
            edits.append(Edit(port.address, value, "replace", source=source, origin=origin))
        else:
            edits.append(Edit(port.address, value, "replace", _mask(run, port, chosen),
                              _selection(port, chosen), source=source, origin=origin))
    return edits


class _Calls:
    """Native decoder-layer calls counted by hooks on the layer modules while it is open:
    ``entered`` (every call, stop-probes that halt at the layer input included) and
    ``completed`` (calls that ran the whole layer)."""

    def __init__(self, adapter: Any):
        self.modules = [layer_module(adapter, i) for i in range(adapter.layers)]
        self.entered = self.completed = 0
        self.handles: list[Any] = []

    def __enter__(self) -> _Calls:
        def pre(*_):
            self.entered += 1

        def post(*_):
            self.completed += 1

        for module in self.modules:
            self.handles.append(module.register_forward_pre_hook(pre))
            self.handles.append(module.register_forward_hook(post))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def view(self) -> dict[str, Any]:
        return {"entered": num(self.entered, MEASURED), "completed": num(self.completed,
                                                                         MEASURED)}


def _measured(run: Any, edits: list[Edit], consumer: Any) -> tuple[dict[str, Any], dict]:
    from .compare import measure

    started = time.perf_counter()
    with _Calls(run.adapter) as calls:
        found = measure(run, edits, consumer)
    found["seconds"] = time.perf_counter() - started
    return found, {"calls": calls.view(), "seconds": num(found["seconds"], MEASURED),
                   "edits": num(len(edits), EXACT)}


# --------------------------------------------------------------------------- test


def _spacing(magnitude_: float, dtype: Any) -> float:
    from .verdict import spacing

    return spacing(magnitude_, dtype)


def test(run: Any, circuit: Any, reference: Any, *, session_id: str | None = None,
         store: Any = None, name: str | None = None,
         prompts: list[tuple[str, list[int]]] | None = None) -> dict[str, Any]:
    """Patch the circuit's components (knock-out) and, separately, every other component
    (keep-only) from the reference natively; seal a ``circuit-test`` record.

    ``reference`` is another ``Run`` (``name`` names it) or, with ``prompts`` (item name, token
    ids) and ``reference=None``, the mean over a prompt set ``name``. Returns ``{result,
    record}``; the session (and the reference run's) is restored and checked unchanged."""
    from .propagate import state_digest

    torch = _torch()
    consumer = circuit.consumer
    if not isinstance(consumer, (Margin, Logit)):
        raise Refusal(f"circuit {circuit.id}: the consumer {circuit.record['body']['consumer']} "
                      "is a vector; circuit test reports ratios of a scalar consumer (a margin "
                      "M[a,b] or a logit)")
    if run.where != circuit.where:
        raise Refusal(f"circuit {circuit.id} was built at {circuit.where}; the cursor is at "
                      f"{run.where} (restart there to test it)")
    started = time.perf_counter()
    before = state_digest(run)
    ports, notes = components(run, circuit)
    ref_before = None if reference is None else state_digest(reference)
    with _Calls(run.adapter) as reads:
        if reference is not None:
            ref = from_run(run, reference, name or getattr(reference, "name", None) or "ref",
                           ports, consumer)
        else:
            ref = from_set(run, name or "set", prompts or [], ports)
    if reference is not None and state_digest(reference) != ref_before:
        raise RuntimeError("reading the reference changed its session")
    found, cost = {}, {}
    for key, which in (("base", None), ("ko", "circuit"), ("keep", "complement"),
                       ("full", "all")):
        edits = [] if which is None else patches(run, ports, which, ref)
        found[key], cost[key] = _measured(run, edits, consumer)
    if state_digest(run) != before:
        raise RuntimeError("circuit test did not leave the session where it was")
    values = {key: float(item["value"]) for key, item in found.items()}
    if ref.consumer is not None:
        values["ref"] = float(ref.consumer[0])
        ref_label, ref_size = MEASURED, ref.consumer[1]
    else:
        values["ref"] = values["full"]
        ref_label, ref_size = found["full"]["authority"], found["full"]["scale"][0]
    dtype = found["base"]["scale"][1]
    magnitude_ = max([ref_size] + [found[k]["scale"][0] for k in ("base", "ko", "keep", "full")])
    resolution = _spacing(magnitude_, dtype)
    span = values["base"] - values["ref"]
    knock = values["base"] - values["ko"]
    margins = {key: num(values[key], found[key]["authority"] if key in found else ref_label)
               for key in ("base", "ref", "ko", "keep", "full")}
    meaning = {
        "base": "this run, nothing patched",
        "ref": ("the consumer read in the reference run at its @last row" if ref.kind == "run"
                else "every component set to the set's mean (there is no single reference run)"),
        "ko": "knock-out: only the circuit's components patched from the reference",
        "keep": ("keep-only: every component outside the circuit patched from the reference; "
                 "the circuit's components computed natively"),
        "full": "every component patched from the reference",
    }
    out: dict[str, Any] = {
        "schema": SCHEMA, "id": num(circuit.id, EXACT), "circuit": circuit.record["id"],
        "consumer": circuit.record["body"]["consumer"], "where": run.where,
        "reference": {"kind": ref.kind, "name": ref.name,
                      **{k: (num(v, EXACT) if isinstance(v, int) and not isinstance(v, bool)
                             else v) for k, v in ref.info.items() if k != "rows"}},
        "components": {"definition": DEFINITION, "rows": ROWS, **counts(ports),
                       "coarsened": {"source_nodes": num(notes["source_nodes_coarsened"], EXACT),
                                     "grouped_sources": num(notes["grouped_sources_coarsened"],
                                                            EXACT),
                                     "rule": "a source-row node is its head"},
                       "groups": "`+N more` groups and unexamined members are in the complement; "
                                 "paged groups were examined (enumerated from the node tree)"},
        "nodes": {key: {"refs": [r for r in notes[f"{key}_nodes"] if r][:64],
                        "count": num(len(notes[f"{key}_nodes"]), EXACT)}
                  for key in ("circuit", "group")},
        "margins": margins, "meaning": meaning,
        "resolution": {"spacing": num(resolution, EXACT), "dtype": str(dtype).removeprefix(
            "torch."), "magnitude": num(magnitude_, MEASURED), "rule": RULE},
    }
    if ref.kind == "set-mean":
        out["reference"]["rows"] = [num(r, EXACT) for r in ref.info["rows"][:16]]
    if ref.kind == "run":
        same = values["full"] == values["ref"]
        out["full_equals_ref"] = same
        out["full_meaning"] = ("patching every component reproduces the reference's consumer bit "
                               "for bit: the components are the whole row" if same else
                               "patching every component does NOT reproduce the reference's "
                               "consumer: the row has an input the components do not cover "
                               "(an edit of the reference, or another kernel path)")
    if abs(span) < resolution:
        reason = (f"the reference does not change the consumer: |base - ref| = {abs(span):.6g} "
                  f"is under one spacing ({resolution:.6g}); pick another reference")
        out["ratios"] = {"state": "refused", "reason": reason}
    else:
        out["ratios"] = {
            "faithfulness": num((values["keep"] - values["ref"]) / span, MEASURED),
            "knockout_share": num(knock / span, MEASURED),
            "formulas": {"faithfulness": "(keep - ref) / (base - ref)",
                         "knockout_share": "(base - ko) / (base - ref)"},
            "meaning": {"faithfulness": "the share of base - ref that the circuit keeps when "
                                        "everything else takes the reference's values (1: all "
                                        "of it, 0: none)",
                        "knockout_share": "the share of base - ref that patching only the circuit "
                                          "removes (1: all of it, 0: none)"}}
    out["reading"] = {
        "knockout": ("no measurable effect" if abs(knock) < resolution else "measurable effect"),
        "keep_vs_base": ("keep equals base within one spacing"
                         if abs(values["keep"] - values["base"]) < resolution
                         else "keep differs from base"),
        "keep_vs_ref": ("keep equals ref within one spacing"
                        if abs(values["keep"] - values["ref"]) < resolution
                        else "keep differs from ref"),
        "rule": "each word compares two measured margins at one spacing (resolution.rule); no "
                "other threshold"}
    held = put_json(component_list(ports), store=store, run=None if store is not None else run)
    out["components"]["list"] = {
        "state": held["state"], "sha256": held["sha256"],
        **({"ref": held["ref"], "tensor_sha256": held["tensor"]["sha256"],
            "bytes": num(held["bytes"], EXACT)} if held["state"] == "stored"
           else {"reason": held["reason"]})}
    total = time.perf_counter() - started
    out["cost"] = {
        "passes": num(4, EXACT),
        "per_pass": cost,
        "reference_read": {"calls": reads.view(),
                           "what": "layer calls to read the reference values" + (
                               " (one run per item, released in turn)" if ref.kind == "set-mean"
                               else " (committed reads, cached per layer)")},
        "seconds": num(total, MEASURED),
    }
    if torch.cuda.is_available() and str(run.adapter.device).startswith("cuda"):
        out["cost"]["peak_vram_bytes"] = num(int(torch.cuda.max_memory_allocated()), MEASURED)
        out["cost"]["peak_vram_meaning"] = "torch.cuda.max_memory_allocated since process start"
    out["session"] = {"before": before, "after": state_digest(run), "unchanged": True}
    out["procedure"] = procedure(run)
    del ref
    try:
        body = check_bounded(out, "circuit test record")
        record = seal_record(session_id=session_id or run_session_id(run), kind="circuit-test",
                             model_identity=run.adapter.model_identity,
                             execution=dict(run.adapter.execution), authority=MEASURED,
                             body=body)
    except RecordError as error:
        raise Refusal(f"the circuit test result does not fit one record ({error})") from None
    out["record"] = record["id"]
    return {"result": out, "record": record}


__all__ = ["CIRCUIT_TEST", "DEFINITION", "Port", "Reference", "SCHEMA", "component_list",
           "components", "counts", "from_run", "from_set", "patches", "test"]

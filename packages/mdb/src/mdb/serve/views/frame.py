"""Frame at the cursor (one occurrence address) and the branch tree of a recorded session."""
from __future__ import annotations

from ...core import address as addresses
from ..recorded import NOT_RECORDED
from ..schema import (
    BranchInfo,
    BranchNode,
    BranchTree,
    Frame,
    FrameFocus,
    LiveState,
    OccAddr,
    PointerRef,
)
from .common import (
    ViewError,
    address_cell,
    blob_ref,
    cell_address,
    cut_refs,
    layers_for,
    occurrence_row,
    parse,
    point_of,
    record_id,
    slot_ref,
    spec_digest,
)
from .symbols import symbol_cells

OFFLINE = "recorded project: no engine attached, nothing runs"


def _scope(model, origin):
    """Candidate cuts in history order, and the branch name they belong to."""
    if origin is None:
        return model.main, "main"
    if origin.cut is not None:
        matches = [c for c in model.cut_ids if c.startswith(origin.cut)]
        if len(matches) != 1:
            raise ViewError("unknown_cut", f"cut reference {origin.cut!r} matches "
                            f"{len(matches)} retained cuts", 404)
        return model.lineage(matches[0]), model.branch[matches[0]]
    cuts = model.branch_cuts(origin.branch)
    if not cuts:
        raise ViewError("unknown_branch", f"no branch {origin.branch!r} in {model.sid}", 404)
    return cuts, origin.branch


def _history(model, cuts):
    """Recorded cells in clock order along a lineage (cuts and the transitions between them)."""
    keep, cells = set(cuts), set()
    for cut in cuts:
        if model.cell_of_cut.get(cut):
            cells.add(model.cell_of_cut[cut])
    for receipt in model.receipts:
        if receipt.get("result") in keep:
            for step in receipt.get("transitions") or []:
                for key in ("before", "after"):
                    cell = model.place(step.get(key))
                    if cell:
                        cells.add(cell)
    return sorted(cells)


LAYERS_UNKNOWN = "unresolved: layer count unknown"
HIDDEN_PORTS = ("h", "h_out")


def _hidden(address):
    return isinstance(address, addresses.Occurrence) and address.port in HIDDEN_PORTS


def _needs_layers(model, address):
    """True when the address is a hidden-state port and the recording has no layer count."""
    return _hidden(address) and model.layers_status == "unknown"


def _state_point(model, address):
    if _needs_layers(model, address):
        return None, LAYERS_UNKNOWN
    try:
        return addresses.state_clock(address, granularity=model.granularity,
                                     layers=layers_for(model, address)), None
    except (addresses.NoStateSlot, ValueError) as error:
        return None, str(error)


def _matches(point, clock):
    have = {"logical_step": point.get("logical_step"), "phase": point.get("phase"),
            "index": point.get("index") if point.get("phase") == "layer" else None}
    return have == clock


def default_address(model):
    """The selected cut's cell, else the last addressable recorded cell of ``main``."""
    cell = model.cell_of_cut.get(model.selected)
    if cell is not None and cell_address(model, *cell):
        return cell_address(model, *cell)
    cells = [c for c in _history(model, model.main) if cell_address(model, *c)]
    if not cells:
        raise ViewError("empty_session", f"session {model.sid} has no addressable cut", 404)
    return cell_address(model, *cells[-1])


def frame(model, addr=None) -> Frame:
    text = addr or default_address(model)
    address = parse(text)
    canonical = addresses.format_address(address)
    cuts, branch = _scope(model, address.origin)
    x, y = address_cell(model, address)
    clock, no_slot = _state_point(model, address)
    frontier = [OFFLINE]
    chosen, note = None, None
    if clock is not None:
        exact = [c for c in cuts if _matches(model.cuts[c]["execution_point"], clock)]
        if not exact:
            exact = [c for c in model.cut_ids if _matches(model.cuts[c]["execution_point"], clock)]
            if exact:
                note = f"no retained cut on {branch} at this clock; showing branch " \
                       f"{model.branch[exact[-1]]}"
        chosen = exact[-1] if exact else None
        if chosen is None:
            frontier.append("no retained cut holds this state; replay needs the engine")
            at_cell = [c for c in cuts if model.cell_of_cut.get(c) == (x, y)]
            if at_cell:
                chosen = at_cell[-1]
                op = model.cuts[chosen]["execution_point"].get("next_operation")
                note = (f"this address's state is not retained; showing the retained boundary "
                        f"at this cell (before {op})")
    elif x is not None and y is not None:
        at_cell = [c for c in cuts if model.cell_of_cut.get(c) == (x, y)]
        chosen = at_cell[-1] if at_cell else None
        note = (f"{no_slot}; nearest retained boundary shown" if chosen else no_slot)
    else:
        note = no_slot or "address has no clock (a cache cell reads at the current clock)"
    if isinstance(address, addresses.StateCell):
        chosen = cuts[-1] if cuts else None
        note = "cache cell read at the latest cut of the branch"
    if chosen is None and _hidden(address) and model.layers_status == "inferred":
        note = NOT_RECORDED
    pointers = []
    if _needs_layers(model, address):
        # The value exists in the recording's terms but cannot be located without the layer
        # count: shown as an unresolved frontier pointer, never as absent.
        pointers.append(PointerRef(digest=spec_digest({"addr": canonical, "why": "layers"}),
                                   addr=canonical, status="frontier", reason=LAYERS_UNKNOWN))
    elif chosen is not None:
        limits = {}
        if model.layers_known:
            limits["layers"] = model.n_layers
        if model.first_step is not None:  # sample/logits need the session's first stepped row
            limits["first_step"] = model.first_step
        limits = limits or None
        state = None
        try:
            state = addresses.to_state_address(
                address, point=point_of(model, chosen), granularity=model.granularity,
                layers=layers_for(model, address), limits=limits)
        except addresses.NoStateSlot as error:
            if "stepped" in str(error):  # sample/logits of a row the session never stepped
                pointers.append(PointerRef(
                    digest=spec_digest({"addr": canonical, "why": "unstepped"}),
                    addr=canonical, status="frontier", reason=f"unresolved: {str(error)[:300]}"))
            # otherwise this address names no state slot at this point
        except (ValueError, addresses.AddressError) as error:
            # not "no slot": a refusal with its reason, shown as an unresolved pointer
            pointers.append(PointerRef(digest=spec_digest({"addr": canonical, "why": "refused"}),
                                       addr=canonical, status="frontier",
                                       reason=f"refused: {str(error)[:300]}"))
        if state is not None:
            sel = list(state.selector)
            item = sel.pop(0) if state.slot in ("keys", "values") else None
            if item is not None and sel[:1] == [0]:
                sel = sel[1:]
            pointers.append(slot_ref(model, chosen, state.slot, item=item,
                                     select=sel, addr=canonical))
    if isinstance(address, addresses.Occurrence):
        row = occurrence_row(model, address)
        if row and row.get("blob_id"):
            pointers.append(blob_ref(model, row["blob_id"], addr=canonical,
                                     via=["occurrence index"]))
        elif row:
            frontier.append("occurrence indexed without a blob: rebuilding it needs replay")
    sym = symbol_cells(model).get((x, y), []) if x is not None else []
    records = [record_id(r) for r in [*model.records(), *model.findings()]
               if chosen and chosen in cut_refs(r)]
    records = [r for r in records if r]
    receipts = [r["fingerprint"] for r in model.receipts
                if chosen and chosen in (r.get("result"), r.get("parent"))]
    focus = FrameFocus(cut=chosen, point=model.cuts[chosen]["execution_point"] if chosen else None,
                       receipts=receipts[:16], symbols=sym[:16], records=records[:16],
                       pointers=pointers[:8], note=note)
    history = [c for c in _history(model, cuts) if cell_address(model, *c)]
    prev = nxt = None
    if x is not None and y is not None:
        before = [c for c in history if c < (x, y)]
        after = [c for c in history if c > (x, y)]
        origin = "" if address.origin is None else addresses._format_origin(address.origin)
        prev = cell_address(model, *before[-1], origin=origin) if before else None
        nxt = cell_address(model, *after[0], origin=origin) if after else None
    sel = addresses._format_selector(address.selector) or None
    key = None
    if model.layers_known:  # aliases depend on the model's layer count: never from an inferred one
        try:
            key = addresses.format_address(addresses.value_key(address, layers=model.n_layers))
        except (addresses.AddressError, ValueError):
            key = None
    return Frame(addr=canonical, key=key,
                 occ=OccAddr(run=model.sid, branch=branch, cut=chosen,
                             clock={"token": x, "layer": y}, sel=sel),
                 axes=model.axes(), pos={"token": x, "layer": y},
                 live=LiveState(state="offline", frontier=frontier), focus=focus,
                 prev=prev, next=nxt, stateClock=clock)


def branches(model) -> BranchTree:
    nodes = []
    for cut in model.cut_ids:
        desc = model.cuts[cut]
        receipt = model.producer.get(cut)
        kind = receipt["operation"] if receipt else ("root" if not desc["parent"] else "capture")
        cell = model.cell_of_cut.get(cut)
        nodes.append(BranchNode(cut=cut, parent=desc["parent"], branch=model.branch[cut],
                                clock={"token": cell[0] if cell else None,
                                       "layer": cell[1] if cell else None},
                                kind=kind, retained=True,
                                receipt=receipt["fingerprint"] if receipt else None))
    infos = []
    for name in dict.fromkeys(model.branch[c] for c in model.cut_ids):
        members = [c for c in model.cut_ids if model.branch[c] == name]
        base = model.cuts[members[0]]["parent"]
        infos.append(BranchInfo(name=name, head=members[-1], base=base, cuts=len(members)))
    return BranchTree(run=model.sid, active=model.selected, nodes=nodes, branches=infos)

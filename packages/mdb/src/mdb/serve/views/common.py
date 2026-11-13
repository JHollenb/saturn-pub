"""Shared helpers for view projections: addresses, cells and pointer references."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from saturn_pub.values import digest

from ...core import address as addresses
from .. import _legacy, blobs
from ..schema import PointerRef
from ..stored import as_dict, as_list, as_str


class ViewError(ValueError):
    """A request the view cannot answer (bad address, unknown id): refused, not a crash."""

    def __init__(self, reason, detail, status=422):
        super().__init__(detail)
        self.reason, self.detail, self.status = reason, detail, status


def short(value, n=12):
    return value if isinstance(value, str) and len(value) <= n else str(value)[:n]


def spec_digest(spec: dict) -> str:
    raw = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def layers_for(model, address):
    """The ``layers=`` to give ``mdb.core.address``/``index`` for this address, or None.

    ``known``: the model's count. ``inferred`` (the highest recorded layer index, a lower bound):
    ``max(inferred, layer + 1)``, so a layer is never declared nonexistent from a guess; a lookup
    under it either misses (unresolved) or hits the row indexed under the true count. ``unknown``:
    None for a hidden-state port (``h``/``h_out``: no key without a count); ports whose key does
    not depend on the count get the placeholder 1."""
    if model.layers_status == "known":
        return model.n_layers
    if model.layers_status == "inferred":
        layer = getattr(address, "layer", None)
        return max(model.n_layers, layer + 1) if isinstance(layer, int) else model.n_layers
    hidden = isinstance(address, addresses.Occurrence) and address.port in ("h", "h_out")
    return None if hidden else 1


def occurrence_row(model, address):
    """The occurrence index row for an address, or None.

    Looked up under ``value_key`` (``tN.emb`` = ``tN.L0.h``; ``tN.Lk.h_out`` = ``tN.L{k+1}.h``),
    so one value has one key; core requires the layer count (``layers_for``). Absent index tables
    read as "not indexed", and so does a hidden-state address in a recording whose layer count is
    unknown.
    """
    from ...core.index import find_occurrence

    try:
        parsed = addresses.coerce(address)
    except ValueError:
        parsed = None  # find_occurrence below reads it with the pointer's own port list
    layers = layers_for(model, parsed)
    if layers is None:
        return None
    try:
        return find_occurrence(model.project, model.sid, address, layers=layers)
    except (ValueError, LookupError):
        return None


def parse(text):
    try:
        return addresses.parse(text)
    except addresses.AddressError as error:
        raise ViewError("bad_address", str(error)) from error


def point_of(model, cut):
    """The execution point a cut was recorded at, shaped like ``ExecutionPoint`` (what
    ``mdb.core.address.to_state_address(point=...)`` reads)."""
    p = model.cuts[cut]["execution_point"]
    return SimpleNamespace(logical_step=p.get("logical_step"), phase=p.get("phase"),
                           index=p.get("index"), fingerprint=p.get("fingerprint"))


def cell_address(model, x, y, *, origin=""):
    """Address text of timeline cell (x, y) from the server-supplied tick fragments, or None
    for a boundary no occurrence address names (``AxisTick.addressable`` False)."""
    token = f"t{x}"
    tick = model.depth.axis().ticks[y]
    if not tick.addressable:
        return None
    parts = [token] + ([tick.frag] if tick.frag else [])
    return origin + ".".join(parts) + "." + (tick.port or "h")


def address_cell(model, address):
    """``(x, y)`` of an occurrence address; y is None for a port the depth axis lacks."""
    if not isinstance(address, addresses.Occurrence):
        return None, None
    return address.token, model.depth.tick_of_address(address)


def location_address(model, location: dict) -> str:
    """Canonical address text of a ``StateAddress`` dict, or ``slot[selector]`` when the
    decoder mapping has no address for it."""
    from saturn_pub.contracts import StateAddress

    location = as_dict(location)
    selector = tuple(x for x in as_list(location.get("selector"))
                     if isinstance(x, int) and not isinstance(x, bool))
    slot = as_str(location.get("slot")) or "?"
    point_digest = as_str(location.get("execution_point"))
    point = None
    for cut, desc in model.cuts.items():
        if point_digest and digest(desc["execution_point"]) == point_digest:
            p = desc["execution_point"]
            point = SimpleNamespace(logical_step=p.get("logical_step"), phase=p.get("phase"),
                                    index=p.get("index"), fingerprint=None)
            break
    try:
        state = StateAddress(slot, selector, as_str(location.get("role")))
        if point is None and slot not in ("keys", "values"):
            raise addresses.NoStateSlot("no retained point")
        return addresses.format_address(addresses.from_state_address(
            state, point or SimpleNamespace(logical_step=0, phase="", index=None,
                                            fingerprint=None),
            layers=model.n_layers, granularity=model.granularity,
            limits=None if model.first_step is None else {"first_step": model.first_step}))
    except (addresses.NoStateSlot, addresses.AddressError, ValueError, TypeError):
        return slot + (json.dumps(list(selector)) if selector else "")


def slot_ref(model, cut, slot, *, item=None, select=(), addr=None) -> PointerRef:
    spec = {"kind": "slot", "cut": cut, "slot": slot, "item": item, "select": list(select)}
    via = [f"cut {short(cut)}", f"slot {slot}"]
    try:
        info = blobs.slot_info(model.store_root, cut, slot, item)
    except (blobs.BlobError, OSError, ValueError) as error:
        reason = getattr(error, "reason", str(error))
        return PointerRef(digest=spec_digest(spec), addr=addr, spec=spec, via=via,
                          status="frontier", reason=reason)
    shape = info["shape"]
    if shape is not None and select:
        if len(select) > len(shape) or any(not 0 <= s < n for s, n in zip(select, shape)):
            return PointerRef(digest=spec_digest(spec), addr=addr, spec=spec, via=via,
                              status="frontier", reason="selector is outside the slot tensor")
        shape = shape[len(select):]
    return PointerRef(digest=spec_digest(spec), addr=addr, spec=spec, via=via,
                      status="resolvable", shape=shape, dtype=info["dtype"])


def blob_ref(model, ref, *, addr=None, via=()) -> PointerRef:
    spec = {"kind": "blob", "ref": ref}
    try:
        info = _legacy.tensor_info(model.project, ref)
    except (OSError, ValueError) as error:
        return PointerRef(digest=spec_digest(spec), addr=addr, spec=spec,
                          via=[*via, f"tensor {short(ref)}"], status="frontier",
                          reason=str(error))
    return PointerRef(digest=spec_digest(spec), addr=addr, spec=spec,
                      via=[*via, f"tensor {short(ref)}"], status="resolvable",
                      shape=info["shape"], dtype=info["dtype"])


def descriptor_ref(desc: dict, *, addr=None, why="descriptor only") -> PointerRef:
    """A sealed tensor descriptor whose bytes the project does not retain."""
    desc = as_dict(desc)
    sha = as_str(desc.get("sha256")) or as_str(desc.get("tensor"))
    shape = desc.get("shape")
    if not (isinstance(shape, list) and len(shape) <= 16 and all(
            isinstance(n, int) and not isinstance(n, bool) for n in shape)):
        shape = None
    return PointerRef(digest=spec_digest({"kind": "descriptor", "sha256": sha}), addr=addr,
                      via=[f"sha256 {short(sha)}"], status="frontier",
                      reason=f"{why}: tensor bytes are not retained in the project store",
                      shape=shape, dtype=str(desc.get("dtype", ""))[:40].replace(
                          "torch.", "") or None)


def cut_refs(record: dict) -> list[str]:
    """Cut ids a record names (circuit records, v2 records, findings)."""
    out = []
    for key in ("supporting_cut", "input_cut", "result_cut", "cut_id", "source_cut"):
        if isinstance(record.get(key), str):
            out.append(record[key])
    body = record.get("body") if isinstance(record.get("body"), dict) else {}
    history = body.get("history") if isinstance(body.get("history"), dict) else {}
    if isinstance(history.get("parent_cut"), str):
        out.append(history["parent_cut"])
    for pointer in as_list(record.get("pointers")):
        if isinstance(pointer, dict) and isinstance(pointer.get("cut"), str):
            out.append(pointer["cut"])
    return list(dict.fromkeys(out))


def record_id(record: dict) -> str:
    """The record's id when it is a string, else ""."""
    return as_str(as_dict(record).get("id")) or as_str(as_dict(record).get("record_id"))

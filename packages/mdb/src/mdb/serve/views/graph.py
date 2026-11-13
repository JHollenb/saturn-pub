"""Graph view: a sealed slice graph (or the session graph) laid out on clock coordinates.

Nodes keep the ids of the recorded ``StateGraph``. Nodes beyond ``k`` (nearest to the center
first) are folded into one aggregate node per (kind, role) group, reported in ``pruned``.
"""
from __future__ import annotations

import sqlite3
from collections import deque

from ..schema import CircuitOverlay, GraphEdge, GraphNode, GraphView, PrunedGroup, SliceRef
from ..stored import as_str
from .common import ViewError, address_cell, cell_address, parse, record_id, short
from .records import _interaction

MAX_K = 64


def _slice_ref(sliced):
    return SliceRef(id=sliced["slice_id"], component=sliced["component"], cut=sliced["cut_id"],
                    complete=bool(sliced["complete"]))


def _role(kind, labels, meta, source_ref):
    if source_ref.startswith("symbol://") or "binding" in meta:
        return "symbol"
    if source_ref.startswith("slice-gap://"):
        return "gap"
    if source_ref.startswith(("circuit://", "slice://", "state-annotation://", "watch://")):
        return "record"
    if kind == "action":
        return "operation"
    if kind == "state":
        return "state"
    slot = meta.get("slot") or (labels[0] if labels else "")
    return {"keys": "cache", "values": "cache", "tokens": "cursor", "logits": "readout",
            "hidden": "carrier"}.get(slot, "value")


def _node(model, raw):
    content = raw.get("content") or {}
    meta = content.get("metadata") or {}
    labels = raw.get("labels") or []
    kind = raw["kind"]
    source_ref = str(content.get("source_ref") or content.get("act_ref") or "")
    cell = None
    native = meta.get("native_receipt") if isinstance(meta.get("native_receipt"), dict) else {}
    if native.get("operation"):
        meta = {**meta, "operation": native["operation"], "receipt": native.get("fingerprint")}
    if kind == "action":
        point = meta.get("point")
        cell = model.place(point) if point else None
        if cell is None and native.get("parent") in model.cell_of_cut:
            cell = model.cell_of_cut[native["parent"]]
    elif kind == "state":
        cell = model.cell_of_cut.get(content.get("state_fingerprint") or "")
    else:
        cut = meta.get("native_cut") or (meta.get("boundary") or {}).get("cut")
        cell = model.cell_of_cut.get(cut) if cut else None
    label = labels[0] if labels else (meta.get("operation") or kind)
    if kind == "state":
        label = "cut " + short(content.get("state_fingerprint", ""), 8)
    small = {k: meta[k] for k in ("slot", "operation", "receipt", "native_cut", "kind",
                                  "record_id", "terminal_status", "support") if k in meta
             and isinstance(meta[k], (str, bool))}
    return GraphNode(id=raw["node_id"], kind=kind, label=str(label)[:48],
                     role=_role(kind, labels, meta, source_ref),
                     x=cell[0] if cell else None, y=cell[1] if cell else None,
                     addr=cell_address(model, *cell) if cell else None,
                     ev="measured" if raw.get("evidence") and any(
                         e.get("class") == "observed" for e in raw["evidence"]) else None,
                     meta=small)


def _edge_type(edge, nodes):
    evidence = [e.get("class") for e in edge.get("evidence") or []]
    if edge["graph"] == "knowledge":
        return "attribution" if nodes.get(edge["source"], {}).get("role") == "symbol" \
            else "unknown"
    relation = edge["relation"]
    target = nodes.get(edge["target"], {})
    source = nodes.get(edge["source"], {})
    if relation == "writes" and target.get("role") == "cache":
        return "kv_write"
    if (source.get("meta") or {}).get("operation") == "apply":
        return "intervention"
    return "dataflow" if "observed" in evidence else "declared"


def _choose_slice(model, slices, addr, slice_id):
    if slice_id:
        match = [s for s in slices if s["slice_id"].startswith(slice_id)]
        if len(match) != 1:
            raise ViewError("unknown_slice", f"slice {slice_id!r} matches {len(match)} slices", 404)
        return match[0]
    if addr:
        cell = address_cell(model, parse(addr))
        at = [s for s in slices if model.cell_of_cut.get(s["cut_id"]) == cell]
        if at:
            return at[-1]
    return slices[-1] if slices else None


def _prune(nodes, edges, center, depth, k):
    adjacent = {}
    for e in edges:
        adjacent.setdefault(e.source, []).append(e.target)
        adjacent.setdefault(e.target, []).append(e.source)
    order, dist = [], {}
    starts = [center] if center in nodes else list(nodes)[:1]
    queue = deque((s, 0) for s in starts)
    while queue:
        node, d = queue.popleft()
        if node in dist:
            continue
        dist[node] = d
        order.append(node)
        for nxt in sorted(adjacent.get(node, [])):
            if nxt not in dist and d + 1 <= depth:
                queue.append((nxt, d + 1))
    order += sorted(n for n in nodes if n not in dist)
    kept = set(order[:k])
    groups = {}
    for node in order[k:]:
        n = nodes[node]
        groups.setdefault((n.kind, n.role), []).append(node)
    out_nodes = [nodes[n] for n in order[:k]]
    folded, pruned = {}, []
    for (kind, role), members in sorted(groups.items()):
        gid = f"more:{kind}:{role}"
        folded.update({m: gid for m in members})
        out_nodes.append(GraphNode(id=gid, kind="aggregate", label=f"+{len(members)} more {role}",
                                   role=role, count=len(members)))
        pruned.append(PrunedGroup(group=f"{kind}/{role}", count=len(members), node=gid))
    out_edges, seen = [], set()
    for e in edges:
        s, t = folded.get(e.source, e.source), folded.get(e.target, e.target)
        if s == t or (s not in kept and not s.startswith("more:")) or (
                t not in kept and not t.startswith("more:")):
            continue
        key = (s, t, e.type)
        if key in seen:
            continue
        seen.add(key)
        out_edges.append(e.model_copy(update={"source": s, "target": t,
                                              "id": e.id if (s, t) == (e.source, e.target)
                                              else f"{e.id[:16]}~{s[-8:]}~{t[-8:]}"}))
    return out_nodes, out_edges, pruned


def _overlays(model, sliced, node_ids):
    out = []
    if sliced is None:
        return out
    records = [r for r in model.records() if isinstance(r, dict)]
    focus = [r for r in records if r.get("kind") == "focus"
             and r.get("slice_id") == sliced["slice_id"]]
    for f in focus:
        members = [n for n in [f.get("node_id")] if isinstance(n, str) and n in node_ids]
        if not members:
            continue
        fid = record_id(f)
        out.append(CircuitOverlay(id="focus:" + fid, kind="focus",
                                  label=f"focus {as_str(f.get('component'))[:48]}",
                                  nodes=members,
                                  verdict=as_str(f.get("terminal_status")) or "unknown",
                                  record=fid))
        for r in records:
            if r.get("focus_id") == fid and r.get("kind") in ("comparison", "interface-test"):
                rid = record_id(r)
                # the number is a labelled field, never text in the label
                interaction = _interaction(r) if r["kind"] == "comparison" else None
                out.append(CircuitOverlay(id=f"{r['kind']}:{rid}", kind=r["kind"],
                                          label=r["kind"], interaction=interaction,
                                          nodes=members,
                                          verdict=as_str(r.get("terminal_status")) or "unknown",
                                          record=rid))
    return out[:16]


def graph(model, *, addr=None, zoom="slice", depth=3, k=48, slice_id=None):
    if zoom not in ("slice", "session"):
        raise ViewError("bad_zoom", "zoom is 'slice' or 'session'")
    if not 1 <= k <= MAX_K or not 0 <= depth <= 16:
        raise ViewError("bad_bounds", f"k is 1..{MAX_K} and depth 0..16")
    slices = model.slices()
    sliced = _choose_slice(model, slices, addr, slice_id) if zoom == "slice" else None
    if zoom == "slice" and sliced is None:
        zoom = "session"
    if zoom == "slice":
        raw_nodes, raw_edges = sliced["graph"]["nodes"], sliced["graph"]["edges"]
        center = (sliced.get("targets") or [None])[0]
        authority = sliced.get("authority", "declared-dependency")
    else:
        try:
            snap = model.session_graph()
        except (ValueError, KeyError, TypeError, AttributeError, LookupError, OSError,
                sqlite3.Error) as error:
            raise ViewError("graph_unreadable", "the session graph could not be read "
                            f"({type(error).__name__})", 409) from error
        raw_nodes, raw_edges = snap["nodes"], snap["edges"]
        center = None
        authority = "recorded history (execution + knowledge graphs)"
    nodes = {n["node_id"]: _node(model, n) for n in raw_nodes}
    plain = {i: n.model_dump() for i, n in nodes.items()}
    edges = [GraphEdge(id=e["edge_id"], source=e["source"], target=e["target"],
                       type=_edge_type(e, plain), relation=e["relation"],
                       evidence=",".join(sorted({x.get("class", "?") for x in e.get("evidence")
                                                 or []})) or "none")
             for e in raw_edges if e["source"] in nodes and e["target"] in nodes]
    if addr and zoom == "session":
        cell = address_cell(model, parse(addr))
        at = [n.id for n in nodes.values() if (n.x, n.y) == cell and n.kind == "state"]
        center = at[-1] if at else None
    out_nodes, out_edges, pruned = _prune(nodes, edges, center, depth, k)
    lanes = {}
    for n in out_nodes:
        if n.x is None:
            lanes.setdefault(n.role, len(lanes))
            n.lane = lanes[n.role]
    return GraphView(zoom=zoom, center=center, slice=_slice_ref(sliced) if sliced else None,
                     nodes=out_nodes, edges=out_edges, pruned=pruned,
                     overlays=_overlays(model, sliced, {n.id for n in out_nodes}),
                     slices=[_slice_ref(s) for s in slices][:32], authority=authority)

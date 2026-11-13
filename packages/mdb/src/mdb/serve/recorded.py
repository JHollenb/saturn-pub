"""Read-only model of one recorded project session: clocks, cuts, branches and index rows.

Everything here is derived from verified ``mdb.core.project_read`` reads and from sealed
cut metadata; nothing is executed. ``Recorded`` caches one ``SessionModel`` per session and
rebuilds it when the session's index rows change.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from saturn_pub.values import digest

from ..core import address as addresses
from . import _legacy, blobs
from .schema import AxisTick, ClockAxis
from .stored import as_dict, as_int, as_list, as_str

# What a read of stored rows may raise when a row is corrupt (bad JSON, broken seal, wrong shape).
READ_ERRORS = (ValueError, KeyError, TypeError, AttributeError, LookupError, OSError,
               sqlite3.Error, RecursionError, ArithmeticError)


class SessionUnreadable(Exception):
    """A session whose stored rows cannot be turned into a model: refused as a whole (the other
    sessions of the project are unaffected)."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason

LAYER_UNKNOWN = "layer count unknown"
NOT_RECORDED = "not recorded; model layer count unknown"


def layer_fallback(n):
    return (f"at least {n} recorded layers (inferred from the highest recorded layer index; the "
            "model's layer count is not recorded)")


class DepthTicks:
    """The second clock axis of an autoregressive decoder: boundaries inside one token step.

    Ticks follow ``mdb.core.address`` state-slot semantics: ``Lk`` is the hidden state entering
    layer k (``tN.Lk.h``; ``tN.emb`` is ``tN.L0.h``), ``out`` the state after the last layer
    (``tN.L{n-1}.h_out``), ``logits``/``sample`` the readout. ``embed`` (only when recorded) is
    the boundary before the embedding; no occurrence address names it. Under layer granularity
    the embed boundary of step t+1 holds token t's logits, so it is drawn in column t at
    ``logits``.
    """

    def __init__(self, n_layers, granularity, observed, status="known"):
        """``status``: ``known`` (the surface's layer clock), ``inferred`` (highest recorded layer
        index: a lower bound, not the model's count) or ``unknown`` (no layer ticks at all)."""
        self.n, self.granularity, self.status = n_layers or 0, granularity, status
        n_layers = self.n
        keys = ([("embed",)] if ("embed",) in observed else [])
        if status != "unknown":
            keys += [("L", k) for k in range(n_layers)] + [("out",)]
        keys += [k for k in (("fn",), ("logits",), ("sample",))
                 if k in observed or (k == ("logits",) and granularity == "layer")]
        self.keys = keys
        self.index = {key: i for i, key in enumerate(keys)}

    def place(self, point):
        """``(token column, tick key)`` of a native execution point, or None."""
        if not isinstance(point, dict) or not isinstance(point.get("logical_step"), int):
            return None
        t, phase, idx = point["logical_step"], point.get("phase"), point.get("index")
        if phase == "layer" and isinstance(idx, int):
            if self.status == "unknown":
                return None
            return (t, ("L", idx)) if idx < self.n else (t, ("out",)) if idx == self.n else None
        if phase == "embed":
            if self.granularity == "layer":
                return (t - 1, ("logits",)) if t >= 1 else (0, ("embed",))
            return t, ("embed",)
        mapped = {"normalization": ("out",), "readout": ("fn",), "sample": ("logits",),
                  "commit": ("sample",)}.get(phase)
        return None if mapped is None else (t, mapped)

    def tick_of_address(self, address):
        if not isinstance(address, addresses.Occurrence):
            return None
        port, layer = address.port, address.layer
        if port == "emb":
            key = ("L", 0)  # value_key: tN.emb is tN.L0.h (the embed boundary precedes it)
        elif port == "h_out" and layer is not None:
            if layer > self.n - 1:
                return None
            key = ("out",) if layer == self.n - 1 else ("L", layer + 1)
        elif layer is not None:
            key = ("L", layer)
        elif port == "fn":
            key = ("fn",) if ("fn",) in self.index else ("out",)
        elif port in ("logits", "M"):
            key = ("logits",)
        elif port == "sample":
            key = ("sample",) if ("sample",) in self.index else ("logits",)
        else:
            return None
        return self.index.get(key)

    def axis(self):
        ticks = []
        for i, key in enumerate(self.keys):
            inferred = self.status == "inferred"
            if key[0] == "L":
                ticks.append(AxisTick(i=i, label=f"L{key[1]}", frag=f"L{key[1]}", port="h",
                                      detail=f"hidden state entering layer {key[1]}",
                                      inferred=inferred))
            elif key == ("out",):
                last = max(self.n - 1, 0)
                detail = ("hidden state after the last layer" if self.status == "known" else
                          f"hidden state after layer {last}; whether it is the last layer is "
                          "not recorded")
                ticks.append(AxisTick(i=i, label=f"L{last} out", frag=f"L{last}", port="h_out",
                                      detail=detail, inferred=inferred))
            elif key == ("embed",):
                ticks.append(AxisTick(i=i, label="pre-embed", addressable=False,
                                      detail="before the embedding: the cut holds tokens and "
                                      "cache, no hidden state; no occurrence address names it "
                                      "(tN.emb is tN.L0.h)"))
            else:
                port = {"fn": "fn", "logits": "logits", "sample": "sample"}
                ticks.append(AxisTick(i=i, label=key[0], port=port[key[0]],
                                      detail={"fn": "final-norm output",
                                              "logits": "readout logits",
                                              "sample": "sampled token"}[key[0]]))
        if self.status == "known":
            note = f"{self.n} decoder layers; {self.granularity} granularity"
        elif self.status == "inferred":
            note = layer_fallback(self.n) + f"; {self.granularity} granularity"
        else:
            note = f"layer count unknown (no layer ticks drawn); {self.granularity} granularity"
        return ClockAxis(id="layer", label="depth", kind="ordinal", ticks=ticks, note=note)


class SessionModel:
    def __init__(self, project, sid: str):
        self.project, self.sid = project, sid
        self.store_root = _legacy.store_root(project)
        row = _legacy.session_row(project, sid)
        self.row = row
        self._cache, self._problems = {}, []
        self.identity = json.loads(row["identity"])
        self.execution = json.loads(row["execution"])
        self.selected = as_str(row["selected"])
        self.debug_info = json.loads(row["debug_info"])
        for name in ("execution", "debug_info"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"session {name} is not an object")
        if not isinstance(self.identity, (str, dict)):
            raise ValueError("session identity is neither text nor an object")
        self.granularity = as_str(self.execution.get("granularity")) or "layer"
        self.cut_ids = _legacy.cut_ids(project, sid)
        self.cuts = {c: _legacy.cut_descriptor(project, sid, c) for c in self.cut_ids}
        for cut, desc in self.cuts.items():
            parent = desc.get("parent") if isinstance(desc, dict) else 0
            if not (isinstance(desc, dict) and isinstance(desc.get("execution_point"), dict)
                    and isinstance(desc.get("surface"), dict)
                    and (parent is None or isinstance(parent, str))):
                raise ValueError(f"cut {cut[:12]} has a malformed descriptor")
        self.receipts = _legacy.receipts(project, sid)
        self.slots = {}
        for cut in self.cuts.values():
            for slot in as_list(cut["surface"].get("slots")):
                if isinstance(slot, dict) and isinstance(slot.get("name"), str):
                    self.slots.setdefault(slot["name"], slot)
        points = [c["execution_point"] for c in self.cuts.values()]
        steps = [as_int(p.get("logical_step")) for p in points]
        steps = [t for t in steps if t is not None and t >= 0]
        # the recording's first stepped row: the lowest logical step of any cut's execution point
        self.first_step = min(steps) if steps else None
        for receipt in self.receipts:
            for step in receipt.get("transitions") or []:
                points += [step.get("before"), step.get("after")]
        self.n_layers, self.layer_note, self.layers_status = self._layers(points)
        observed = set()
        probe = DepthTicks(self.n_layers, self.granularity, set(), self.layers_status)
        for point in points:
            placed = probe.place(point)
            if placed:
                observed.add(placed[1])
        self.depth = DepthTicks(self.n_layers, self.granularity, observed, self.layers_status)
        self.cell_of_cut = {c: self.place(d["execution_point"]) for c, d in self.cuts.items()}
        self.tokens, self.tokens_reason = self._tokens()
        self.point_cells = [cell for cell in map(self.place, points) if cell]
        columns = [x for x, _ in self.point_cells] + [len(self.tokens or []) - 1, 0]
        self.n_tokens = max(columns) + 1
        self._branches()

    @property
    def layers_known(self):
        return self.layers_status == "known"

    def _layers(self, points):
        """``(count, note, status)``: ``known`` only from the cut surface's layer clock; the
        highest recorded layer index is a lower bound (``inferred``), never the model's count."""
        schema = (self.slots.get("layer") or {}).get("schema") or {}
        if isinstance(schema.get("maximum"), int) and not isinstance(schema["maximum"], bool) \
                and schema["maximum"] >= 1:
            return schema["maximum"], "from the cut surface's layer clock", "known"
        seen = [p.get("index") for p in points if isinstance(p, dict) and p.get("phase") == "layer"
                and isinstance(p.get("index"), int) and p["index"] >= 1]
        if not seen:
            return None, LAYER_UNKNOWN, "unknown"
        return max(seen), layer_fallback(max(seen)), "inferred"

    def place(self, point):
        placed = self.depth.place(point) if hasattr(self, "depth") else None
        if placed is None or placed[1] not in self.depth.index or placed[0] < 0:
            return None
        return placed[0], self.depth.index[placed[1]]

    def _tokens(self):
        """Token ids from the ``tokens`` slot of the longest retained cut (numpy read)."""
        candidates = [c for c in self.cut_ids if "tokens" in self._slot_names(c)]
        if not candidates:
            return None, "no cut retains a tokens slot"
        best, best_len = None, -1
        for cut in candidates:
            try:
                shape = blobs.slot_info(self.store_root, cut, "tokens")["shape"] or []
            except (blobs.BlobError, OSError, ValueError):
                continue
            if shape and shape[-1] > best_len:
                best, best_len = cut, shape[-1]
        if best is None:
            return None, "tokens slot has no tensor descriptor"
        try:
            array, _ = blobs.cut_slot(self.store_root, best, "tokens")
        except blobs.BlobError as error:
            return None, error.reason
        return [int(v) for v in array.reshape(-1).tolist()], None

    def _slot_names(self, cut):
        path = Path(self.store_root) / "cuts" / f"{cut}.json"
        try:
            return set(json.loads(path.read_bytes()).get("slots") or {})
        except (OSError, ValueError):
            return set()

    def _branches(self):
        """Name branches: the selected cut's lineage is ``main``; each other fork ``b<n>``."""
        self.producer = {}
        for receipt in self.receipts:
            if receipt.get("result") in self.cuts and receipt.get("operation") not in (
                    "restore", "abort"):
                self.producer.setdefault(receipt["result"], receipt)
        main, cursor = [], self.selected
        while cursor in self.cuts and cursor not in main:
            main.append(cursor)
            cursor = self.cuts[cursor]["parent"]
        self.branch, taken, count = {c: "main" for c in main}, set(), 0
        for cut in self.cut_ids:
            if cut in self.branch:
                continue
            parent = self.cuts[cut]["parent"]
            inherited = self.branch.get(parent)
            if inherited and inherited != "main" and parent not in taken:
                self.branch[cut] = inherited
                taken.add(parent)
            else:
                count += 1
                self.branch[cut] = f"b{count}"
        self.main = list(reversed(main))

    def lineage(self, cut):
        out, cursor = [], cut
        while cursor in self.cuts and cursor not in out:
            out.append(cursor)
            cursor = self.cuts[cursor]["parent"]
        return list(reversed(out))

    def branch_cuts(self, name):
        heads = [c for c in self.cut_ids if self.branch.get(c) == name]
        return self.lineage(heads[-1]) if heads else []

    def axes(self):
        ticks = []
        for x in range(self.n_tokens):
            token = self.tokens[x] if self.tokens and x < len(self.tokens) else None
            ticks.append(AxisTick(i=x, label=f"t{x}" if token is None else f"t{x}:{token}",
                                  frag=f"t{x}"))
        token_axis = ClockAxis(id="token", label="token row", kind="ordinal", ticks=ticks,
                               note="token rows (logical_step)")
        ports = [AxisTick(i=i, label=name, detail=(slot.get("role") or ""))
                 for i, (name, slot) in enumerate(sorted(self.slots.items()))]
        port_axis = ClockAxis(id="port", label="recorded slot", kind="categorical", ticks=ports,
                              note="state slots retained in cuts; in-layer ports need the engine")
        return [token_axis, self.depth.axis(), port_axis]

    def records(self):
        """Every indexed record (legacy ``circuit_records`` and v2 ``records``), read one row at
        a time so one unverifiable row cannot hide the rest. A row whose kind this process has
        not declared, or that fails verification, becomes a placeholder carrying ``unreadable``
        (shown as unresolved). Absent tables read as empty. Sorted by (kind, id): the two tables
        share no insertion order."""
        from ..core import records as core_records  # declares the core kinds ("occurrence")
        from ..core.index import NotIndexed, get_record

        rows = []
        try:
            with _legacy.db(self.project) as db:
                tables = {r[0] for r in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                for table, schema in (("circuit_records", "legacy"), ("records", "v2")):
                    if table in tables:
                        rows += [(r[0], r[1], schema) for r in db.execute(
                            f"SELECT id, kind FROM {table} WHERE session_id=?", (self.sid,))]
        except sqlite3.Error:
            return []
        out = []
        for rid, kind, schema in rows:
            try:
                record = get_record(self.project, self.sid, rid)
                if isinstance(record, dict) and record.get("verified") is False:
                    # core returns a placeholder row for an undeclared kind or a bad seal
                    state = record.get("state") if record.get("state") in (
                        "unresolved", "refused") else "unresolved"
                    record = {"record_id": rid, "kind": kind, "schema": f"index-{schema}",
                              "unreadable": (f"{state}: "
                                             + str(record.get("reason") or "unverified"))[:300],
                              "unreadable_state": state}
                out.append(record)
            except Exception as error:  # noqa: BLE001 - one corrupt row is one placeholder row
                if schema == "v2" and core_records.kind_spec(kind) is None:
                    state, reason = "unresolved", f"unresolved: undeclared kind {kind}"
                else:
                    state, reason = "refused", ("refused: record failed verification "
                                                f"({str(error)[:160] or type(error).__name__})")
                out.append({"record_id": rid, "kind": kind, "schema": f"index-{schema}",
                            "unreadable": reason[:300], "unreadable_state": state})
        out.sort(key=lambda r: (str(r.get("kind")), str(r.get("id") or r.get("record_id"))))
        return out

    def _problem(self, kind, ident, error, reason=None):
        reason = reason or (f"refused: {kind} {as_str(ident)[:12] or '?'} could not be read "
                            f"({type(error).__name__})")
        if any(p["kind"] == kind and p["record_id"] == (as_str(ident) or kind)
               for p in self._problems):
            return
        self._problems.append({"record_id": as_str(ident) or f"{kind}", "kind": kind,
                               "schema": "index-unreadable", "unreadable": reason,
                               "unreadable_state": "refused"})

    def unreadable(self):
        """Placeholder rows for stored rows (slices, symbols, ...) that could not be read."""
        from .views.symbols import build_rows  # malformed bindings/lifecycles surface here

        self.symbols(), self.slices(), self.lifecycles(), self.findings()
        build_rows(self)
        try:
            self.session_graph()  # rows left out of the session graph are listed too
        except READ_ERRORS:
            pass
        return list(self._problems)

    def _refuse(self, kind, ident, error):
        """Per-row refusal from the project reader: the row is listed, the rest is served."""
        self._problem(kind, ident, error)

    def _graph_refuse(self, kind, ident, error):
        """A row left out of the session graph. Rows already listed as refused are not listed
        twice; a row that is readable on its own (e.g. a record whose reference does not
        resolve) is listed under ``graph:<id>``."""
        ident = as_str(ident) or kind
        if any(p["record_id"] == ident for p in self._problems):
            return
        self._problem("session-graph", "graph:" + ident, error, reason=(
            f"refused from the session graph: {kind} {ident[:12]} "
            f"({type(error).__name__}: {str(error)[:120]})")[:300])

    def session_graph(self):
        """The sealed session graph (cached); raises when the session history itself does not
        verify. Rows that cannot be placed are left out and listed by ``unreadable()``."""
        if "session_graph" not in self._cache:
            self.symbols(), self.slices(), self.lifecycles(), self.findings()
            try:
                self._cache["session_graph"] = (_legacy.session_graph(
                    self.project, self.sid, refuse=self._graph_refuse), None)
            except READ_ERRORS as error:
                self._cache["session_graph"] = (None, error)
        graph, error = self._cache["session_graph"]
        if error is not None:
            raise error
        return graph

    def _cached(self, key, read):
        if key not in self._cache:
            try:
                self._cache[key] = read()
            except READ_ERRORS as error:
                self._problem(key, key, error)
                self._cache[key] = []
        return self._cache[key]

    def lifecycles(self):
        return self._cached("lifecycles", lambda: [
            r for r in _legacy.symbol_lifecycles(self.project, self.sid, refuse=self._refuse)
            if isinstance(r, dict) and isinstance(r.get("lifecycle"), dict)
            and isinstance(r.get("joins"), dict) and isinstance(r.get("lifecycle_id"), str)])

    def findings(self):
        return self._cached("findings", lambda: [
            f for f in _legacy.findings(self.project, self.sid, refuse=self._refuse)
            if isinstance(f, dict) and isinstance(f.get("annotation_id"), str)])

    def symbols(self):
        return self._cached("symbols", lambda: [
            b for b in _legacy.symbols(self.project, self.sid, refuse=self._refuse)
            if isinstance(b, dict) and isinstance(b.get("fingerprint"), str)])

    def slices(self):
        if "slices" in self._cache:
            return self._cache["slices"]
        out = []
        try:
            ids = list(_legacy.slice_ids(self.project, self.sid))
        except READ_ERRORS as error:
            self._problem("slice", "slices", error)
            ids = []
        for slice_id in ids:
            try:
                sliced = _legacy.slice_record(self.project, self.sid, slice_id)
                graph = as_dict(as_dict(sliced).get("graph"))
                if not (isinstance(sliced.get("slice_id"), str)
                        and isinstance(sliced.get("cut_id"), str)
                        and isinstance(sliced.get("component"), str)
                        and isinstance(graph.get("nodes"), list)
                        and isinstance(graph.get("edges"), list)):
                    raise ValueError("malformed slice")
                out.append(sliced)
            except READ_ERRORS as error:
                self._problem("slice", slice_id, error)
        self._cache["slices"] = out
        return out

    def cells_of_point_digest(self, point_digest):
        return [self.cell_of_cut[c] for c, d in self.cuts.items()
                if self.cell_of_cut[c] and digest(d["execution_point"]) == point_digest]


class Recorded:
    """A project opened read-only, with cached per-session models."""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.project = _legacy.open_project(self.root)
        self._models, self._lock = {}, threading.Lock()

    def sessions(self):
        return _legacy.sessions(self.project)

    def _revision(self, sid):
        with _legacy.db(self.project) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master")}
            parts = [db.execute("SELECT revision FROM sessions WHERE id=?", (sid,)).fetchone()]
            for table in ("cuts", "receipts", "symbols", "slices", "records",
                          "circuit_records", "symbol_lifecycles"):
                if table in tables:
                    parts.append(db.execute(f"SELECT count(*) FROM {table} WHERE session_id=?",
                                            (sid,)).fetchone()[0])
        if parts[0] is None:
            raise KeyError(sid)
        return tuple([parts[0][0], *parts[1:]])

    def model(self, sid) -> SessionModel:
        revision = self._revision(sid)
        with self._lock:
            cached = self._models.get(sid)
            if cached and cached[0] == revision:
                return cached[1]
        try:
            built = SessionModel(self.project, sid)
        except Exception as error:  # noqa: BLE001 - a corrupt session is refused, not a crash
            raise SessionUnreadable(
                f"session {sid[:24]!r} is unreadable ({type(error).__name__}: "
                f"{str(error)[:120]})") from error
        with self._lock:
            self._models[sid] = (revision, built)
        return built

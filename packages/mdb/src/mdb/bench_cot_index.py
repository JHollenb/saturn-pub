"""Append-only metadata index with persistent native-parent links and bounded views."""
from __future__ import annotations

from .errors import BenchError


class RegisterIndex:
    def __init__(self):
        self.reset()
        self.stats = {"events_indexed": 0, "cuts_loaded": 0, "observations_verified": 0}

    def reset(self):
        self.offset = 0
        self.last_event = None
        self.config = (None, None)
        self.rows = {}
        self.errors = {}
        self.nodes = {}
        self.children = {}

    def sync(self, bench):
        events = bench.events
        # A rewind/failed transaction may replace the tail. The session owns an
        # append-only event prefix; new durable manifests preserve that prefix.
        if (len(events) < self.offset or self.offset
                and events[self.offset - 1] != self.last_event):
            self.reset()
        from .bench_cot import verify_artifact
        for event in events[self.offset:]:
            kind = event.get("kind")
            if kind == "cot-register-configured":
                ref = event["config_ref"]
                config = bench.artifacts.get(ref)
                verify_artifact(config, bench.artifacts,
                                {"identity": bench.identity, "session_id": bench.session_id})
                self.config = config, ref
            elif kind == "cot-register-observation":
                row = bench.artifacts.get(event["register_ref"])
                verify_artifact(row, bench.artifacts,
                                {"identity": bench.identity, "session_id": bench.session_id})
                self.rows.setdefault(event["source_cut"], []).append(
                    {"register_ref": event["register_ref"], **row})
                self._invalidate(event["source_cut"])
                self.stats["observations_verified"] += 1
            elif kind == "cot-register-error":
                self.errors.setdefault(event["source_cut"], []).append(dict(event))
                self._invalidate(event["source_cut"])
            self.stats["events_indexed"] += 1
        self.offset = len(events)
        self.last_event = dict(events[-1]) if events else None
        return self

    def _invalidate(self, ref):
        todo = [ref]
        while todo:
            current = todo.pop()
            self.nodes.pop(current, None)
            todo.extend(self.children.get(current, ()))

    def node(self, bench, ref):
        if ref is None:
            return {"native_count": 0, "count": 0, "latest": None, "latest_error": None}
        path, seen, current = [], set(), ref
        while current and current not in self.nodes:
            if current in seen:
                raise BenchError("CoT native lineage contains a cycle")
            seen.add(current)
            row = bench.artifacts.get(current)
            self.stats["cuts_loaded"] += 1
            path.append((current, row))
            current = row.get("parent_cut")
        parent = self.nodes[current] if current else self.node(bench, None)
        for current, row in reversed(path):
            previous = row.get("parent_cut")
            node = {"ref": current, "row": row, "parent": previous,
                    "native_count": parent["native_count"] + 1,
                    "count": parent["count"] + len(self.rows.get(current, ())),
                    "latest": current if current in self.rows else parent["latest"],
                    "latest_error": current if current in self.errors else parent["latest_error"]}
            self.nodes[current] = node
            if previous:
                self.children.setdefault(previous, set()).add(current)
            parent = node
        return self.nodes[ref]

    def lineage(self, bench, limit=None):
        result, ref = [], bench.branch.cut
        while ref and (limit is None or len(result) < limit):
            node = self.node(bench, ref)
            result.append((ref, node["row"]))
            ref = node["parent"]
        return list(reversed(result))

    def observations(self, bench, *, limit=None, cuts=None):
        if cuts is not None:
            return [row for ref, _ in self.lineage(bench, len(cuts)) if ref in cuts
                    for row in self.rows.get(ref, ())]
        result = []
        ref = self.node(bench, bench.branch.cut)["latest"]
        while ref and (limit is None or len(result) < limit):
            result.extend(reversed(self.rows[ref]))
            parent = self.node(bench, ref)["parent"]
            ref = self.node(bench, parent)["latest"]
        if limit is not None:
            result = result[:limit]
        return list(reversed(result))

    def recent_errors(self, bench, limit=3):
        result, ref = [], self.node(bench, bench.branch.cut)["latest_error"]
        while ref and len(result) < limit:
            result.extend(reversed(self.errors[ref]))
            ref = self.node(bench, self.node(bench, ref)["parent"])["latest_error"]
        return list(reversed(result[:limit]))


def index(bench):
    value = getattr(bench, "_cot_index", None)
    if value is None:
        value = RegisterIndex()
        bench._cot_index = value
    return value.sync(bench)

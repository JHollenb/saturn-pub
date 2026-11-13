"""Framework-free structural bindings for discovery's recorded operator program.

Names describe executable structure, not inferred semantics. A program binding
excludes input bytes and observed tensor digests; shape/control-flow changes
produce another binding. Occurrences remain qualified by their native trace.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict


class CatalogError(ValueError):
    pass


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _stage(operator):
    name = operator.split(".")[1]
    if name in {"native_layer_norm", "layer_norm", "native_group_norm", "group_norm",
                "rms_norm", "_native_batch_norm_legit_no_training"}:
        return "normalize"
    if name in {"addmm", "mm", "bmm", "matmul", "linear"}:
        return "affine"
    if name in {"gelu", "gelu_"}:
        return "gelu"
    return None


def _view(event):
    return event["operator"].split(".")[1] in {
        "view", "_unsafe_view", "reshape", "t", "transpose", "permute", "squeeze",
        "unsqueeze", "expand", "slice", "select", "detach", "alias", "as_strided"}


class StructuralCatalog:
    """Contiguous module, mathematical-stage, component and update regions."""

    def __init__(self, trace):
        from .forward_symbols import verify_trace

        verify_trace(trace)
        self.trace_fingerprint = trace["fingerprint"]
        self.event_count = len(trace["events"])
        events = trace["events"]
        structure = {
            "schema": "mdb-discovery-structural-program-v1",
            "model_identity": trace["model_identity"], "execution": trace["execution"],
            "events": [{k: row[k] for k in ("operator", "module_path", "inputs", "outputs",
                                            "arguments", "result")} for row in events],
            "values": {key: {k: row[k] for k in ("shape", "dtype", "origin", "producer")}
                       for key, row in trace["values"].items()},
            "outputs": trace["outputs"],
        }
        self.program_id = "program:" + _digest(structure)
        self.sites = []
        self._descriptors = trace["values"]
        intervals = defaultdict(list)
        for index, event in enumerate(events):
            path = event.get("module_path", "<root>")
            parts = [] if path == "<root>" else path.split(".")
            for module in ["<root>"] + [".".join(parts[:i]) for i in range(1, len(parts)+1)]:
                intervals[module].append(index)

        # Separate repeated calls rather than merging disjoint invocations.
        modules = []
        for module, indices in intervals.items():
            start = previous = indices[0]
            for index in indices[1:] + [None]:
                if index is None or index != previous + 1:
                    modules.append((module, start, previous+1))
                    start = index
                previous = index
        candidates = []
        for module, start, end in modules:
            stages = [_stage(e["operator"]) for e in events[start:end]]
            if "gelu" in stages and stages.count("affine") >= 2:
                candidates.append((module, start, end))
        mlps = [row for row in candidates if not any(
            child != row and row[1] <= child[1] and child[2] <= row[2]
            and (child[1], child[2]) != (row[1], row[2]) for child in candidates)]
        # Equal ranges can occur when a wrapper only delegates to a child.
        deduplicated = {}
        for row in sorted(mlps, key=lambda row: len(row[0]), reverse=True):
            deduplicated.setdefault((row[1], row[2]), row)
        for ordinal, (module, start, end) in enumerate(sorted(deduplicated.values(), key=lambda r:r[1]), 1):
            self._add(f"mlp_{ordinal:03d}", "mlp", "law", module, start, end, events)

        # A core stage owns implementation views derived from its output. Views
        # of parameters preparing the next affine belong to that next stage.
        cores = [(i, _stage(event["operator"])) for i, event in enumerate(events)
                 if _stage(event["operator"])]
        stage_counts = defaultdict(int)
        occupied = set()
        for index, kind in cores:
            start, end = index, index+1
            required = set(events[index]["inputs"])
            while (start > 0 and start-1 not in occupied and _view(events[start-1])
                   and events[start-1]["module_path"] == events[index]["module_path"]):
                prior = events[start-1]
                if not required.intersection(prior["outputs"]):
                    break
                required.update(prior["inputs"])
                start -= 1
            produced = set(events[index]["outputs"])
            while (end < len(events) and _view(events[end])
                   and events[end]["module_path"] == events[index]["module_path"]
                   and produced.intersection(events[end]["inputs"])):
                produced.update(events[end]["outputs"])
                end += 1
            occupied.update(range(start, end))
            stage_counts[kind] += 1
            self._add(f"{kind}_{stage_counts[kind]:03d}", kind, "math",
                      events[index]["module_path"], start, end, events)

        for law in [s for s in self.sites if s["unit"] == "law"]:
            # Follow the actual output port, not the numerical position of an
            # R3 block or the spelling of a caller's module.
            for index in range(law["exit"], len(events)):
                event = events[index]
                if set(law["outputs"]).intersection(event["inputs"]):
                    if (event["operator"].split(".")[1] in {"add", "add_"}
                            and set(law["inputs"]).intersection(event["inputs"])):
                        name = f"update_{1+sum(s['unit']=='update' for s in self.sites):03d}"
                        update = self._add(name, "residual", "update", event["module_path"],
                                           index, index+1, events)
                        update["component_id"] = law["id"]
                    break
        self.scopes = self.sites
        self.occurrences = []
        for site in self.sites:
            parents = [s for s in self.sites if s["id"] != site["id"] and
                       s["entry"] <= site["entry"] and s["exit"] >= site["exit"] and
                       (s["entry"],s["exit"]) != (site["entry"],site["exit"])]
            site["parent_scope"] = min(parents, key=lambda s:s["exit"]-s["entry"])["id"] if parents else None
            self.occurrences.append({"id":f"{site['id']}/trace/{self.trace_fingerprint}/at/{site['entry']}",
                "site_id":site["id"], "trace_fingerprint":self.trace_fingerprint,
                "entry":site["entry"], "exit":site["exit"],
                "identity_scope":"native-trace; lane/epoch/branch qualified by controller"})
        self._occurrence_by_site = {row["site_id"]:row["id"] for row in self.occurrences}
        self._values = set(trace["values"])

    def _add(self, name, kind, unit, module, start, end, events):
        produced = {v for event in events[start:end] for v in event["outputs"]}
        dependencies = list(dict.fromkeys(v for event in events[start:end] for v in event["inputs"] if v not in produced))
        parameters = [v for v in dependencies if self._descriptors[v]["origin"] == "parameter"]
        inputs = [v for v in dependencies if v not in parameters]
        outside = {v for event in events[end:] for v in event["inputs"]}
        outputs = list(dict.fromkeys(v for event in events[start:end] for v in event["outputs"] if v in outside))
        if not outputs:
            outputs = list(events[end-1]["outputs"])
        identifier = f"{self.program_id}/site/{name}"
        row = {"id":identifier, "site_id":identifier, "name":name, "kind":kind, "unit":unit,
               "module":module, "module_path":module, "entry":start, "exit":end,
               "inputs":inputs, "outputs":outputs, "dependency_inputs":dependencies,
               "parameters":parameters, "availability":"native-replay-safe-point"}
        self.sites.append(row)
        return row

    def resolve(self, selector):
        if isinstance(selector, dict):
            selector = selector.get("id", selector.get("site_id"))
        if not isinstance(selector, str):
            raise CatalogError("selector must be text")
        if selector in self._values:
            return {"id":selector, "value_id":selector, "kind":"port"}
        edge = None
        for suffix in (":output", ":input"):
            if selector.endswith(suffix):
                selector, edge = selector[:-len(suffix)], suffix[1:]
                break
        matches = [s for s in self.sites if selector in {s["id"],s["name"]} or
                   selector == "module:"+s["module"]]
        if len(matches) != 1:
            raise CatalogError(f"{'ambiguous' if matches else 'unknown'} structural selector: {selector}")
        row = copy.deepcopy(matches[0])
        if edge:
            values = row["outputs" if edge == "output" else "inputs"]
            if len(values) != 1:
                raise CatalogError(f"{edge} port has {len(values)} values; select an SSA ID")
            row.update({"id":row["id"]+":"+edge, "kind":"port", "value_id":values[0], "edge":edge})
        return row

    def boundaries_at(self, pc):
        return [{"site_id":s["id"], "scope_id":s["id"], "name":s["name"], "kind":s["kind"],
                 "occurrence_id":self._occurrence_by_site[s["id"]], "edge":edge, "pc":pc} for s in self.sites
                for edge in ("enter", "exit") if s["entry" if edge=="enter" else "exit"] == pc]

    def target(self, unit, pc, motion="step", scope=None):
        if unit == "op":
            return min(pc+1, self.event_count)
        if unit == "forward":
            return self.event_count
        if unit not in {"math", "law", "update"}:
            raise CatalogError(f"unsupported motion unit: {unit}")
        rows = [s for s in self.sites if s["unit"] == unit and s["exit"] > pc]
        if scope is not None:
            selected = self.resolve(scope)
            if motion in {"finish", "next"}:
                if not selected["entry"] <= pc < selected["exit"]:
                    raise CatalogError("selected scope is not active")
                return selected["exit"]
            rows = [s for s in rows if selected["entry"] <= s["entry"] and s["exit"] <= selected["exit"]]
        if motion == "into":
            entries = [s["entry"] for s in rows if s["entry"] > pc]
            if not entries:
                raise CatalogError("no next nested scope entry")
            return min(entries)
        active = [s for s in rows if s["entry"] <= pc]
        if active:
            return min(active, key=lambda s:s["exit"]-s["entry"])["exit"]
        if not rows:
            raise CatalogError(f"no remaining {unit} scope")
        return min(rows, key=lambda s:(s["entry"],s["exit"]))["exit"]

    def manifest(self):
        return {"schema":"mdb-discovery-catalog-v1", "program_id":self.program_id,
                "trace_fingerprint":self.trace_fingerprint, "sites":copy.deepcopy(self.sites),
                "occurrences":copy.deepcopy(self.occurrences)}


class AliasRegistry:
    """Revisioned descriptive metadata; pinning copies the resolved binding."""

    def __init__(self, catalog):
        self.catalog, self._records, self._revisions = catalog, {}, {}

    def bind(self, name, target, provenance=None, namespace="user"):
        if not isinstance(name, str) or not name.strip():
            raise CatalogError("alias name must be non-empty text")
        site = self.catalog.resolve(target)
        key = (namespace, name)
        revision = self._revisions.get(key, 0)+1
        self._revisions[key] = revision
        record = {"alias":name, "namespace":namespace, "revision":revision,
                  "target_id":site["id"], "target_type":site["kind"], "site":site,
                  "provenance":copy.deepcopy(provenance or {})}
        self._records[key] = record
        return copy.deepcopy(record)

    def resolve(self, name, namespace="user"):
        try:
            return copy.deepcopy(self._records[(namespace,name)])
        except KeyError as exc:
            raise CatalogError(f"unknown alias: {name}") from exc

    pin = resolve

    def remove(self, name, namespace="user"):
        self.resolve(name, namespace)
        del self._records[(namespace,name)]

    def list(self):
        return copy.deepcopy(list(self._records.values()))


DiscoveryCatalog = StructuralCatalog

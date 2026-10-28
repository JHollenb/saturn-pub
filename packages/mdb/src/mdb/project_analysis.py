"""Versioned, descriptive analysis over original project evidence, without models."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

from saturn_pub.symbols import backtrace
from saturn_pub.values import digest

from .project_navigation import _find, _operations, _symbols


@dataclass(frozen=True)
class Analyzer:
    name: str
    version: str
    run: object


REGISTRY = {}


def register(analyzer):
    """Register a framework-free analyzer; a changed implementation needs a new version."""
    if not analyzer.name or not analyzer.version or not callable(analyzer.run):
        raise ValueError("analyzer requires a name, version and callable")
    key = (analyzer.name, analyzer.version)
    if key in REGISTRY and REGISTRY[key] != analyzer:
        raise ValueError("analyzer version already registered")
    REGISTRY[key] = analyzer
    return analyzer


def inputs(project, sid):
    row = project.session(sid)
    return {
        "session_id": sid,
        "model_identity": json.loads(row["identity"]),
        "execution": json.loads(row["execution"]),
        "selected_cut": row["selected"],
        "cuts": [project.cut_descriptor(sid, cid) for cid in project.cut_ids(sid)],
        "receipts": [r.to_dict() for r in project.receipts(sid)],
        "symbols": _symbols(project, sid),
    }


def structural(project, sid, source):
    groups = {}
    gaps = []
    for cut in source["cuts"]:
        point = cut["execution_point"]
        key = digest(point)
        group = groups.setdefault(key, {"point": point, "cuts": [], "ports": [], "operations": []})
        group["cuts"].append(cut["fingerprint"])
        group["ports"] = sorted(
            set(group["ports"]) | {slot["name"] for slot in cut["surface"].get("slots", [])}
        )
        if not cut["surface"].get("slots"):
            gaps.append({"kind": "missing-surface", "cut": cut["fingerprint"]})
    for operation in _operations(project, sid):
        point = operation.get("before", {})
        key = digest(point)
        group = groups.setdefault(key, {"point": point, "cuts": [], "ports": [], "operations": []})
        group["operations"].append(operation)
        if operation["spec"].get("footprint") != "declared":
            gaps.append(
                {
                    "kind": "unknown-footprint",
                    "receipt": operation["receipt"],
                    "transition_index": operation["transition_index"],
                }
            )
        if not operation.get("before"):
            gaps.append({"kind": "unpositioned-operation", "receipt": operation["receipt"]})
    if not any(group["operations"] for group in groups.values()):
        gaps.append({"kind": "no-recorded-transitions"})
    return {
        "groups": [groups[key] for key in sorted(groups)],
        "gaps": gaps,
        "coverage": "recorded positions only; unvisited positions are unknown",
        "authority": "structural-description",
        "mechanics_status": "valid",
        "instrument_status": "applicable" if groups else "evidence-incomplete",
        "trend_status": "not-evaluated",
        "terminal_status": "not-evaluated",
    }


register(Analyzer("structure", "1", structural))


def analyze(project, sid, *, analyzer="structure", version=None):
    candidates = [
        a
        for (name, v), a in REGISTRY.items()
        if name == analyzer and (version is None or v == version)
    ]
    if len(candidates) != 1:
        raise ValueError("select a registered analyzer and explicit version if ambiguous")
    selected = candidates[0]
    source = inputs(project, sid)
    source_id = digest(source)
    with project._db() as db:
        rows = db.execute(
            "SELECT * FROM analyses WHERE session_id=? AND analyzer=? "
            "AND version=? AND input_digest=?",
            (sid, selected.name, selected.version, source_id),
        ).fetchall()
    if rows:
        return _verified(rows[0])
    body = {
        "schema": "mdb-project-analysis-v1",
        "session_id": sid,
        "analyzer": selected.name,
        "version": selected.version,
        "input_digest": source_id,
        "binding": {k: source[k] for k in ("model_identity", "execution", "selected_cut")},
        "evidence": {
            "cuts": [c["fingerprint"] for c in source["cuts"]],
            "receipts": [r["fingerprint"] for r in source["receipts"]],
            "symbols": [s["fingerprint"] for s in source["symbols"]],
        },
        "result": selected.run(project, sid, source),
    }
    if digest(inputs(project, sid)) != source_id:
        raise ValueError("project evidence changed during analysis; retry")
    identifier = digest(body)
    with project._db() as db:
        db.execute(
            "INSERT OR IGNORE INTO analyses VALUES (?, ?, ?, ?, ?, ?)",
            (identifier, sid, selected.name, selected.version, source_id, project._json(body)),
        )
    return {"analysis_id": identifier, **body}


def _verified(row):
    body = json.loads(row["body"])
    if digest(body) != row["id"]:
        raise ValueError("project analysis seal changed")
    if any(body[key] != row[key] for key in ("session_id", "analyzer", "version", "input_digest")):
        raise ValueError("project analysis index differs from its sealed body")
    return {"analysis_id": row["id"], **body}


def history(project, sid):
    current = digest(inputs(project, sid))
    with project._db() as db:
        rows = db.execute(
            "SELECT * FROM analyses WHERE session_id=? ORDER BY rowid", (sid,)
        ).fetchall()
    return [
        {**_verified(row), "current": json.loads(row["body"])["input_digest"] == current}
        for row in rows
    ]


def walk(project, sid, name, *, direction="upstream", depth=4):
    if direction not in {"upstream", "downstream"}:
        raise ValueError("direction must be upstream or downstream")
    if type(depth) is not int or not 1 <= depth <= 32:
        raise ValueError("depth must be between 1 and 32")
    item = _find(project, sid, name)
    starts = {item["id"]}
    if item["kind"] == "symbol":
        starts = {
            f"port:{loc['slot']}" for e in item["evidence"] for loc in e["binding"]["locations"]
        }
    edges = []
    for operation in _operations(project, sid):
        spec = operation["spec"]
        for relation in ("reads", "writes", "invalidates"):
            for port in spec.get(relation, []):
                source, target = f"port:{port}", f"operation:{spec['operation']}"
                if relation != "reads":
                    source, target = target, source
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "relation": relation,
                        "receipt": operation["receipt"],
                        "parent_cut": operation["parent_cut"],
                        "result_cut": operation["result_cut"],
                        "transition_index": operation["transition_index"],
                        "footprint": spec.get("footprint", "unknown"),
                    }
                )
    frontier, visited, found = starts, set(starts), []
    origin, dest = ("target", "source") if direction == "upstream" else ("source", "target")
    for hop in range(1, depth + 1):
        following = set()
        for edge in edges:
            if edge[origin] in frontier:
                found.append({**edge, "hop": hop})
                following.add(edge[dest])
        frontier = following - visited
        visited.update(following)
        if not frontier:
            break
    truncated = any(e[origin] in frontier for e in edges) if frontier else False
    return {
        "component": item["id"],
        "direction": direction,
        "depth": depth,
        "edges": found,
        "nodes": sorted(visited),
        "truncated": truncated,
        "authority": "declared-dependency",
        "causal_effect": "not-established",
        "scope": "structural reachability across recorded transactions; not a temporal or branch-specific path",
    }


def native_backtrace(project, sid, port, *, cut_id=None):
    """Supply only a linked receipt chain to the existing public backtrace."""
    cut_id = cut_id or project.session(sid)["selected"]
    cut = project.cut_descriptor(sid, cut_id)
    available = list(reversed(project.receipts(sid)))
    cursor, chain, visited = cut_id, [], set()
    while cursor not in visited:
        visited.add(cursor)
        receipt = next(
            (
                r
                for r in available
                if r.to_dict().get("result") == cursor
                and r.to_dict().get("status") != "refused"
                and r.to_dict()["operation"] != "abort"
            ),
            None,
        )
        if receipt is None:
            break
        chain.append(receipt)
        body = receipt.to_dict()
        if body["operation"] not in {"continue", "apply"}:
            break  # Public backtrace reports this history boundary explicitly.
        cursor = body["parent"]
    view = SimpleNamespace(
        receipts=list(reversed(chain)), inspect=lambda: SimpleNamespace(surface=cut["surface"])
    )
    return {"cut_id": cut_id, **backtrace(view, port)}


def finding(
    project, sid, component, title, *, kind="observation", note=None, evidence=(), cut_id=None
):
    if kind not in {"observation", "inference", "causal-claim"}:
        raise ValueError("unknown finding kind")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("finding requires a title")
    item = project.component(sid, component)
    cut_id = cut_id or project.session(sid)["selected"]
    if cut_id not in item["supporting_cuts"]:
        raise ValueError("finding cut must support its component")
    allowed = set(item["supporting_cuts"]) | {r["receipt"] for r in item["xrefs"]}
    allowed.update(e["fingerprint"] for e in item["observations"])
    evidence = tuple(evidence) or (cut_id,)
    if not set(evidence) <= allowed:
        raise ValueError(
            "finding evidence must reference supporting retained cuts, receipts or watches"
        )
    analysis = analyze(project, sid)
    return project.annotate(
        sid,
        title,
        cut_id=cut_id,
        namespace="findings",
        role=kind,
        tags=(item["id"], "finding"),
        note=note,
        details={
            "schema": "mdb-project-finding-v1",
            "component": item["id"],
            "kind": kind,
            "evidence": list(evidence),
            "analysis_id": analysis["analysis_id"],
            "terminal_status": "not-established",
        },
    )


def findings(project, sid):
    return [
        project._annotations(sid).show(row["annotation_id"])
        for row in project.annotations(sid, tag="finding")
    ]

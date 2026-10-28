"""Framework-free navigation over sealed public execution evidence."""

import argparse
import json

from saturn_pub.symbols import SymbolBinding
from saturn_pub.values import digest


def _operations(project, sid):
    for receipt in project.receipts(sid):
        body = receipt.to_dict()
        if body.get("status") == "refused":
            continue
        common = {
            "session_id": sid,
            "receipt": receipt.fingerprint,
            "parent_cut": body.get("parent"),
            "result_cut": body.get("result"),
            "transaction": body["operation"],
        }
        for index, step in enumerate(body.get("transitions", [])):
            yield {
                **common,
                "transition_index": index,
                "occurrence": step["occurrence"],
                "before": step["before"],
                "after": step["after"],
                "spec": step["native_operation"],
            }
        if body["operation"] == "apply" and "act" in body:
            yield {
                **common,
                "transition_index": None,
                "spec": {
                    **body["act"],
                    "operation": body["act"]["name"],
                    "footprint": "declared",
                },
            }


def _symbols(project, sid):
    return [
        SymbolBinding.from_dict(row).to_dict()
        for row in json.loads(project.session(sid)["debug_info"]).get("symbols", [])
    ]


def components(project, sid, *, query=None):
    project.session(sid)
    rows = {}

    def add(kind, name, **evidence):
        key = f"{kind}:{name}"
        row = rows.setdefault(
            key, {"id": key, "kind": kind, "name": name, "session_id": sid, "evidence": []}
        )
        if evidence not in row["evidence"]:
            row["evidence"].append(evidence)

    cuts = {
        identifier: project.cut_descriptor(sid, identifier) for identifier in project.cut_ids(sid)
    }
    for identifier, cut in cuts.items():
        for slot in cut["surface"].get("slots", []):
            add(
                "port",
                slot["name"],
                cut_id=identifier,
                slot=slot,
                execution_point=cut["execution_point"],
            )
    for op in _operations(project, sid):
        add(
            "operation",
            op["spec"]["operation"],
            **{k: v for k, v in op.items() if k != "spec"},
            spec=op["spec"],
        )
    for symbol in _symbols(project, sid):
        add("symbol", symbol["symbol"], binding=symbol)
        for identifier, cut in cuts.items():
            if any(
                loc["execution_point"] == digest(cut["execution_point"])
                and loc["state_schema"] == cut["surface"].get("state_schema")
                for loc in symbol["locations"]
            ):
                add("symbol", symbol["symbol"], cut_id=identifier, binding=symbol)
    return [
        row
        for key, row in sorted(rows.items())
        if query is None or query.casefold() in key.casefold()
    ]


def _find(project, sid, name):
    rows = components(project, sid)
    matches = [row for row in rows if row["id"] == name]
    if not matches:
        matches = [row for row in rows if row["name"] == name]
    if len(matches) != 1:
        raise ValueError(
            f"unknown or ambiguous component: {name}; use port:, operation: or symbol:"
        )
    return matches[0]


def xrefs(project, sid, name, *, relation=None):
    if relation not in (None, "reads", "writes", "invalidates"):
        raise ValueError("relation must be reads, writes or invalidates")
    item = _find(project, sid, name)
    ports = {item["name"]} if item["kind"] == "port" else set()
    if item["kind"] == "symbol":
        ports = {
            loc["slot"] for evidence in item["evidence"] for loc in evidence["binding"]["locations"]
        }
    result = []
    for op in _operations(project, sid):
        spec = op["spec"]
        for rel in ("reads", "writes", "invalidates"):
            if relation is not None and rel != relation:
                continue
            for port in spec.get(rel, []):
                if (
                    item["kind"] == "operation" and spec["operation"] == item["name"]
                ) or port in ports:
                    result.append(
                        {
                            **{k: v for k, v in op.items() if k != "spec"},
                            "operation": spec["operation"],
                            "port": port,
                            "relation": rel,
                            "authority": "declared-dependency",
                            "footprint": spec.get("footprint", "unknown"),
                            "scope": "whole-slot" if item["kind"] == "symbol" else "slot",
                            "causal_effect": "not-established",
                        }
                    )
    return result


def component(project, sid, name):
    item = _find(project, sid, name)
    refs = xrefs(project, sid, item["id"])
    cuts = {e["cut_id"] for e in item["evidence"] if "cut_id" in e}
    cuts.update(r[k] for r in refs for k in ("parent_cut", "result_cut") if r.get(k))
    retained = set(project.cut_ids(sid))
    return {
        **item,
        "xrefs": refs,
        "supporting_cuts": sorted(cuts & retained),
        "annotations": [a for a in project.annotations(sid) if a["cut_id"] in cuts],
        "observations": [e for e in project.watch_events(sid) if e.get("cut") in cuts],
        "coverage": "recorded execution only; missing references do not prove absence",
    }


def execute(
    project,
    sid,
    action,
    target=None,
    *,
    query=None,
    relation=None,
    cut_id=None,
    note=None,
    direction="upstream",
    depth=4,
    title=None,
    kind="observation",
    evidence=(),
    analyzer="structure",
    version=None,
):
    if sid is None:
        sessions = project.sessions()
        if len(sessions) != 1:
            raise ValueError("select --session-id for a project with zero or multiple sessions")
        sid = sessions[0]["session_id"]
    project.session(sid)
    if action == "analyze":
        return project.analyze(sid, analyzer=analyzer, version=version)
    if action == "analyses":
        return project.analyses(sid)
    if action == "findings":
        return project.findings(sid)
    if action == "list":
        return project.components(sid, query=query)
    if target is None:
        raise ValueError(f"{action} requires a target")
    if action == "show":
        return project.component(sid, target)
    if action == "xrefs":
        return project.xrefs(sid, target, relation=relation)
    if action in {"callers", "consumers", "trace"}:
        direction = {"callers": "upstream", "consumers": "downstream"}.get(action, direction)
        return project.trace(
            sid, target, direction=direction, depth=1 if action != "trace" else depth
        )
    if action == "backtrace":
        return project.backtrace(sid, target.removeprefix("port:"), cut_id=cut_id)
    if action == "finding":
        return project.finding(
            sid, target, title, kind=kind, note=note, evidence=evidence, cut_id=cut_id
        )
    if action == "cut":
        return project.cut_descriptor(sid, target)
    if action == "annotate":
        item = project.component(sid, target)
        selected = cut_id or project.session(sid)["selected"]
        if selected not in item["supporting_cuts"]:
            raise ValueError("annotation cut must support the selected component")
        return project.annotate(sid, item["id"], cut_id=selected, note=note, tags=(item["id"],))
    raise ValueError(f"unknown project action: {action}")


def add_arguments(parser, *, extra_actions=()):
    parser.add_argument(
        "action",
        choices=(
            *extra_actions,
            "list",
            "show",
            "xrefs",
            "cut",
            "annotate",
            "analyze",
            "analyses",
            "callers",
            "consumers",
            "trace",
            "backtrace",
            "finding",
            "findings",
        ),
    )
    parser.add_argument("target", nargs="?")
    parser.add_argument("--query")
    parser.add_argument("--relation", choices=("reads", "writes", "invalidates"))
    parser.add_argument("--cut-id")
    parser.add_argument("--note")
    parser.add_argument("--direction", choices=("upstream", "downstream"), default="upstream")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--title")
    parser.add_argument(
        "--kind", choices=("observation", "inference", "causal-claim"), default="observation"
    )
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--analyzer", default="structure")
    parser.add_argument("--version")


def command(project, sid, words):
    """Same queries in the attached debugger shell, with no second executor."""

    class Parser(argparse.ArgumentParser):
        def error(self, message):
            raise ValueError(message)

    parser = Parser(prog="project", add_help=False)
    add_arguments(parser)
    args = vars(parser.parse_args(words))
    result = execute(project, sid, **args)
    return result if isinstance(result, dict) else {"rows": result}

"""Framework-free navigation over sealed public execution evidence."""

import argparse

from saturn_pub.symbols import SymbolBinding

from .project_slice import slot_descriptors


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
    return project.symbols(sid)


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
            try:
                SymbolBinding.from_dict(symbol).resolve_metadata(
                    model_identity=cut["model_identity"],
                    execution=cut["manifest"]["execution"],
                    execution_point=cut["execution_point"],
                    surface=cut["surface"],
                    slots=slot_descriptors(cut),
                    context=symbol["context"],
                )
            except ValueError:
                continue
            add(
                "symbol",
                symbol["symbol"],
                cut_id=identifier,
                binding=symbol,
                support="metadata-support-verified",
            )
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
    if item["kind"] != "symbol":
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
    port=None,
    symbol=None,
    to_cut=None,
    context=None,
    binding_id=None,
    receipt_id=None,
    view="declared",
    slice_id=None,
    branch=None,
    node_id=None,
    consumer=None,
    steps=None,
    tokens=None,
    edits=(),
    run_id=None,
    record_id=None,
    cases=None,
    arguments=None,
    result=None,
    selector=None,
    live=None,
    linearize=False,
    recipe=None,
    occurrence=None,
):
    if sid is None:
        sessions = project.sessions()
        if len(sessions) != 1:
            raise ValueError("select --session-id for a project with zero or multiple sessions")
        sid = sessions[0]["session_id"]
    project.session(sid)
    if action in {"focus", "explain", "seek", "compare", "contract", "record", "records", "parameter", "replay-bind"}:
        import json

        from . import circuit_investigation as circuit

        def data(value):
            if value is None or not isinstance(value, str):
                return value
            if value.startswith("@"):
                from pathlib import Path

                path = Path(value[1:])
                if path.stat().st_size > 65536:
                    raise ValueError("circuit JSON file is too large")
                value = path.read_text()
            if len(value) > 65536:
                raise ValueError("circuit JSON argument is too large")
            return json.loads(value)

        if action == "focus":
            if target is None and slice_id is None:
                return circuit.select(project, sid, record_id) if record_id else circuit.current(project, sid)
            return project.focus(sid, target, cut_id=cut_id, slice_id=slice_id, node_id=node_id,
                                 consumer=data(consumer), steps=1 if steps is None else steps, context=context,
                                 binding_id=binding_id, receipt_id=receipt_id)
        if action == "replay-bind":
            return circuit.attach_replay(project, sid, data(recipe), occurrence)
        if action == "explain":
            return project.explain(sid, focus_id=record_id, view=view, live=live)
        if action == "seek":
            return project.seek(sid, focus_id=record_id, live=live)
        if action == "compare":
            if run_id:
                if edits or steps is not None or tokens is not None or linearize:
                    raise ValueError("select a sealed plan or new plan options, not both")
                return project.run_comparison(sid, run_id, live=live)
            return project.plan_comparison(sid, [data(e) for e in edits], focus_id=record_id,
                                           steps=steps, tokens=tokens, linearize=linearize)
        if action == "contract":
            return project.test_interface(sid, data(cases), data(arguments), data(result),
                                          name=title or "interface", live=live)
        if action == "record":
            return project.circuit_record(sid, target or record_id)
        if action == "records":
            return project.circuit_records(sid)
        if action == "parameter":
            from .circuit_native import parameter_view

            selected = circuit.current(project, sid, record_id)
            with circuit.materialized(project, sid, selected, live) as native:
                return parameter_view(native, target, data(selector) or [])
    if action == "finding" and target is None:
        from .circuit_investigation import finding

        return finding(project, sid, title, record_id=record_id, note=note)
    if action == "analyze":
        return project.analyze(sid, analyzer=analyzer, version=version)
    if action == "analyses":
        return project.analyses(sid)
    if action == "findings":
        return project.findings(sid)
    if action == "list":
        return project.components(sid, query=query)
    if action == "symbol-history":
        from .symbol_trace import trace_symbol

        if target and symbol:
            raise ValueError("select a history target or --symbol, not both")
        name = symbol or target
        if not name:
            raise ValueError("symbol-history requires a symbol")
        return trace_symbol(project, sid, name.removeprefix("symbol:"),
                            branch=branch, cut_id=cut_id)
    if action == "slice":
        if view != "declared":
            raise ValueError("only the declared slice view is implemented")
        if target:
            if port or symbol:
                raise ValueError("select a slice target or --symbol/--port, not both")
            if target.startswith("symbol:"):
                symbol = target.removeprefix("symbol:")
            else:
                port = target.removeprefix("port:")
        return project.slice(
            sid,
            port=port,
            symbol=symbol,
            cut_id=cut_id,
            to_cut=to_cut,
            context=context,
            binding_id=binding_id,
            receipt_id=receipt_id,
        )
    if action == "slice-show":
        return project.slice_record(sid, target or slice_id)
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
            sid,
            target,
            title,
            kind=kind,
            note=note,
            evidence=evidence,
            cut_id=cut_id,
            slice_id=slice_id,
            investigation_id=record_id,
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
            "slice",
            "slice-show",
            "symbol-history",
            "focus", "explain", "seek", "compare", "contract", "record", "records", "parameter", "replay-bind",
        ),
    )
    parser.add_argument("target", nargs="?")
    parser.add_argument("--query")
    parser.add_argument("--relation", choices=("reads", "writes", "invalidates"))
    parser.add_argument("--cut-id", "--cut", dest="cut_id")
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
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--symbol")
    source.add_argument("--port")
    parser.add_argument("--to-cut")
    parser.add_argument("--context")
    parser.add_argument("--binding", dest="binding_id")
    parser.add_argument("--receipt", dest="receipt_id")
    parser.add_argument("--view", choices=("declared", "computation", "mlp", "attention", "effect", "frequency"), default="declared")
    parser.add_argument("--slice-id")
    parser.add_argument("--branch")
    parser.add_argument("--node-id")
    parser.add_argument("--consumer", help="JSON port/selector or logits token/contrast")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--linearize", action="store_true", help="also compute a matched first-readout native adjoint")
    parser.add_argument("--edit", dest="edits", action="append", default=[], help="JSON port/mode/value/selector; repeat for B")
    parser.add_argument("--run", dest="run_id", nargs="?", const="@latest", help="execute a sealed comparison plan; defaults to this focus's latest")
    parser.add_argument("--record-id")
    parser.add_argument("--cases", help="JSON source/result cut pairs for an interface test")
    parser.add_argument("--arguments", help="JSON argument port/selector list")
    parser.add_argument("--result", help="JSON next-result port/selector")
    parser.add_argument("--selector", help="JSON integer coordinates for a parameter view")
    parser.add_argument("--recipe", help="sealed native replay recipe JSON or @FILE")
    parser.add_argument("--occurrence", help="expected recipe occurrence for the focused cut")


def command(project, sid, words, *, live=None):
    """Same queries in the attached debugger shell, with no second executor."""

    class Parser(argparse.ArgumentParser):
        def error(self, message):
            raise ValueError(message)

    parser = Parser(prog="project", add_help=False)
    add_arguments(parser)
    args = vars(parser.parse_args(words))
    result = execute(project, sid, **args, live=live)
    return result if isinstance(result, dict) else {"rows": result}

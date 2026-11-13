"""Sealed Project focus, native inspection and bounded circuit comparisons.

Metadata operations never import model libraries. Native execution is explicit
and uses the attached public Session; records contain descriptors, not tensors.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager, nullcontext

from saturn_pub.core import Act, Session
from saturn_pub.values import describe, digest

SCHEMA = "mdb-circuit-investigation-v1"


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _tables(project):
    if project.read_only:
        return
    with project._db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS circuit_records (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                kind TEXT NOT NULL, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS circuit_focus (
                session_id TEXT PRIMARY KEY, id TEXT NOT NULL);
        """)


def save(project, sid, kind, **fields):
    if project.read_only:
        raise ValueError("circuit records require a writable project")
    row = project.session(sid)
    body = _json(
        {
            "schema": SCHEMA,
            "session_id": sid,
            "kind": kind,
            "model_identity": json.loads(row["identity"]),
            "execution": json.loads(row["execution"]),
            **fields,
        }
    )
    identifier = digest(body)
    _tables(project)
    with project._db() as db:
        db.execute(
            "INSERT OR IGNORE INTO circuit_records VALUES (?,?,?,?)",
            (identifier, sid, kind, project._json(body)),
        )
        existing = db.execute(
            "SELECT body FROM circuit_records WHERE id=?", (identifier,)
        ).fetchone()[0]
        if existing != project._json(body):
            raise ValueError("circuit record identity collision")
    return {"record_id": identifier, **body}


def record(project, sid, identifier):
    row = project.session(sid)
    with project._db() as db:
        exists = db.execute(
            "SELECT name FROM sqlite_master WHERE name='circuit_records'"
        ).fetchone()
        saved = (
            db.execute(
                "SELECT * FROM circuit_records WHERE session_id=? AND id=?", (sid, identifier)
            ).fetchone()
            if exists
            else None
        )
    if saved is None:
        raise ValueError("circuit record is not retained in this project session")
    body = json.loads(saved["body"])
    if (
        digest(body) != identifier
        or body.get("schema") != SCHEMA
        or body.get("session_id") != sid
        or body.get("kind") != saved["kind"]
        or body.get("model_identity") != json.loads(row["identity"])
        or body.get("execution") != json.loads(row["execution"])
    ):
        raise ValueError("circuit record seal or session binding changed")
    return {"record_id": identifier, **body}


def records(project, sid, *, kind=None):
    project.session(sid)
    with project._db() as db:
        exists = db.execute(
            "SELECT name FROM sqlite_master WHERE name='circuit_records'"
        ).fetchone()
        ids = (
            [
                r[0]
                for r in db.execute(
                    "SELECT id FROM circuit_records WHERE session_id=? ORDER BY rowid", (sid,)
                )
            ]
            if exists
            else []
        )
    rows = [record(project, sid, identifier) for identifier in ids]
    return [r for r in rows if kind is None or r["kind"] == kind]


def current(project, sid, identifier=None):
    if identifier is None:
        project.session(sid)
        with project._db() as db:
            exists = db.execute(
                "SELECT name FROM sqlite_master WHERE name='circuit_focus'"
            ).fetchone()
            row = (
                db.execute("SELECT id FROM circuit_focus WHERE session_id=?", (sid,)).fetchone()
                if exists
                else None
            )
        if row is None:
            raise ValueError("select an occurrence with project focus first")
        identifier = row[0]
    value = record(project, sid, identifier)
    if value["kind"] != "focus":
        raise ValueError("expected a focus record")
    sliced = project.slice_record(sid, value["slice_id"])
    if not any(n["node_id"] == value["node_id"] for n in sliced["graph"]["nodes"]):
        raise ValueError("focused node is absent from its sealed slice")
    return value


def select(project, sid, identifier):
    value = record(project, sid, identifier)
    selected = current(project, sid, value.get("focus_id", identifier))
    if project.read_only:
        raise ValueError("selecting a saved focus requires a writable project")
    _tables(project)
    with project._db() as db:
        db.execute(
            "INSERT INTO circuit_focus VALUES (?,?) ON CONFLICT(session_id) DO UPDATE SET id=excluded.id",
            (sid, selected["record_id"]),
        )
    return selected


def _selector(value):
    if (
        not isinstance(value, (list, tuple))
        or len(value) > 8
        or any(type(i) is not int or i < 0 for i in value)
    ):
        raise ValueError("selector must contain at most eight nonnegative integers")
    return list(value)


def _consumer(value):
    if not isinstance(value, dict) or not set(value) <= {"port", "selector", "token", "contrast"}:
        raise ValueError("consumer requires port, optional selector or token/contrast")
    port = value.get("port")
    if not isinstance(port, str) or not port:
        raise ValueError("consumer requires a port")
    out = {"port": port, "selector": _selector(value.get("selector", []))}
    for name in ("token", "contrast"):
        if name in value:
            if type(value[name]) is not int or value[name] < 0:
                raise ValueError("consumer token IDs must be nonnegative integers")
            out[name] = value[name]
    if ("contrast" in out and "token" not in out) or ("token" in out and out["selector"]):
        raise ValueError("token contrasts require a token and no selector")
    if out.get("contrast") == out.get("token") and "token" in out:
        raise ValueError("consumer contrast must use different tokens")
    return out


def focus(
    project,
    sid,
    target=None,
    *,
    cut_id=None,
    slice_id=None,
    node_id=None,
    consumer=None,
    steps=1,
    context=None,
    binding_id=None,
    receipt_id=None,
):
    if type(steps) is not int or not 1 <= steps <= 128:
        raise ValueError("steps must be in 1..128")
    if slice_id:
        if target or cut_id or context or binding_id or receipt_id:
            raise ValueError("select an existing slice or a new component/cut")
        sliced = project.slice_record(sid, slice_id)
    else:
        if not target:
            raise ValueError("focus requires a port/symbol or --slice-id")
        item = project.component(sid, target)
        if item["kind"] not in {"port", "symbol"}:
            raise ValueError("focus an operation through an exact slice node")
        sliced = project.slice(
            sid,
            **{item["kind"]: item["name"]},
            cut_id=cut_id,
            context=context,
            binding_id=binding_id,
            receipt_id=receipt_id,
        )
    nodes = {n["node_id"]: n for n in sliced["graph"]["nodes"]}
    if node_id is None:
        if len(sliced["targets"]) != 1:
            raise ValueError("slice has multiple targets; select --node-id")
        node_id = sliced["targets"][0]
    if node_id not in nodes:
        raise ValueError("node is absent from selected slice")
    meta = nodes[node_id]["content"].get("metadata", {})
    boundary = meta.get("boundary")
    port = meta.get("slot")
    if not boundary and meta.get("receipt") is not None and meta.get("transition") is not None:
        boundary = {
            "receipt": meta["receipt"],
            "transition": meta["transition"],
            "point": meta.get("point"),
        }
    if boundary is None:
        raise ValueError("node is descriptive only; select a port version or native transition")
    selector = []
    if sliced["symbol_binding"] and node_id in sliced["targets"]:
        locations = [loc for loc in sliced["symbol_binding"]["locations"] if loc["slot"] == port]
        if len(locations) != 1:
            raise ValueError("symbol has ambiguous locations; select a port explicitly")
        selector = _selector(locations[0].get("selector", []))
    cut = project.cut_descriptor(sid, sliced["cut_id"])
    slots = {s["name"] for s in cut["surface"].get("slots", [])}
    consumer = _consumer(consumer or {"port": "logits" if "logits" in slots else port or "value"})
    value = save(
        project,
        sid,
        "focus",
        slice_id=sliced["slice_id"],
        node_id=node_id,
        component=sliced["component"],
        supporting_cut=sliced["cut_id"],
        boundary=boundary,
        port=port,
        selector=selector,
        consumer=consumer,
        steps=steps,
        expected_value=meta.get("value"),
        authority="declared-occurrence",
        terminal_status="not-established",
    )
    with project._db() as db:
        db.execute(
            "INSERT INTO circuit_focus VALUES (?,?) ON CONFLICT(session_id) DO UPDATE SET id=excluded.id",
            (sid, value["record_id"]),
        )
    return value


def explain(project, sid, *, focus_id=None, view="computation", live=None):
    selected = current(project, sid, focus_id)
    if view in {"mlp", "attention", "effect", "frequency"}:
        from .circuit_native import inspect_decoder

        with materialized(project, sid, selected, live) as native:
            result = inspect_decoder(native, selected["consumer"], view=view)
        return save(
            project, sid, "explanation", focus_id=selected["record_id"], view=view, result=result
        )
    if view not in {"computation", "declared"}:
        raise ValueError("unknown explanation view")
    sliced = project.slice_record(sid, selected["slice_id"])
    nodes = {n["node_id"]: n for n in sliced["graph"]["nodes"]}
    edges = [
        e for e in sliced["graph"]["edges"] if selected["node_id"] in (e["source"], e["target"])
    ]
    related = [r for r in records(project, sid) if r.get("focus_id") == selected["record_id"]]
    return {
        "focus": selected,
        "node": nodes[selected["node_id"]],
        "edges": edges,
        "frontier": sliced["gaps"],
        "complete": sliced["complete"],
        "state_materialized": False,
        "authority": "declared-dependency",
        "results": [{"record_id": r["record_id"], "kind": r["kind"]} for r in related],
        "next": ["seek", "explain --view mlp", "explain --view attention", "compare", "finding"],
    }


def attach_replay(project, sid, recipe, occurrence):
    """Associate a sealed root/history recipe with the selected retained identity."""
    from .native_replay_recipe import validate

    selected = current(project, sid)
    if not isinstance(recipe, dict) or not isinstance(occurrence, str):
        raise ValueError("replay binding requires a sealed recipe and occurrence ID")
    recipe = validate(recipe)
    event = next((e for e in recipe["events"] if e["id"] == occurrence), None)
    if (
        recipe["model_identity"] != selected["model_identity"]
        or recipe["execution"] != selected["execution"]
        or event is None
        or "cut" not in selected["boundary"]
        or event.get("expected_cut") != selected["boundary"]["cut"]
    ):
        raise ValueError(
            "recipe must bind the exact focused model, program and expected occurrence cut"
        )
    return save(
        project,
        sid,
        "replay-recipe",
        focus_id=selected["record_id"],
        recipe=recipe,
        occurrence=occurrence,
        authority="regenerate-then-verify",
    )


@contextmanager
def materialized(project, sid, selected, live):
    if live is None:
        raise ValueError(
            "native inspection requires an attached compatible public session; offline browsing does not launch a model"
        )
    if live.project is not project or live.session_id != sid:
        raise ValueError("live session belongs to another project/session")
    if live._pending:
        raise ValueError("commit or abort the pending intervention first")
    live._persist()  # Also checks optimistic Project writer revision.
    active = live.session.capture(retain=False)
    boundary = selected["boundary"]
    native = None
    try:
        recipes = [
            r
            for r in records(project, sid, kind="replay-recipe")
            if r["focus_id"] == selected["record_id"]
            and "cut" in boundary
            and any(
                e["id"] == r["occurrence"] and e.get("expected_cut") == boundary["cut"]
                for e in r["recipe"]["events"]
            )
        ]
        if recipes:
            from .native_replay_recipe import NativeReplay

            linked = recipes[-1]
            replay = NativeReplay(linked["recipe"], live.session.adapter).run()
            native = Session.from_cut(live.session.adapter, replay.cuts[linked["occurrence"]])
        elif "cut" in boundary:
            cut = project.load_cut(
                sid, boundary["cut"], device=str(getattr(live.session.adapter, "device", "cpu"))
            )
            native = Session.from_cut(live.session.adapter, cut)
        else:
            receipts = {r.fingerprint: r.to_dict() for r in project.receipts(sid)}
            body = receipts.get(boundary.get("receipt"))
            index = boundary.get("transition")
            if (
                body is None
                or body["operation"] != "continue"
                or type(index) is not int
                or not 0 <= index <= min(128, len(body.get("transitions", [])))
            ):
                raise ValueError("occurrence has no supported bounded transition replay")
            cut = project.load_cut(
                sid, body["parent"], device=str(getattr(live.session.adapter, "device", "cpu"))
            )
            native = Session.from_cut(live.session.adapter, cut)
            if index:
                repeated = native.continue_(index).to_dict()["transitions"]
                if repeated != body["transitions"][:index]:
                    raise ValueError(
                        "replayed occurrence differs from its native transition history"
                    )
            if (
                boundary.get("point")
                and dict(native.inspect().execution_point) != boundary["point"]
            ):
                raise ValueError("replayed occurrence clock differs")
        if selected.get("expected_value") is not None and selected.get("port"):
            if describe(native.read(selected["port"])) != selected["expected_value"]:
                raise ValueError("materialized operand differs from its sealed occurrence")
        yield native
    finally:
        # Re-activate the original external closure, and never commit a private fork.
        restored = live.session.capture(retain=False)
        if restored.fingerprint != active.fingerprint:
            raise ValueError("inspection changed the active native branch")


def seek(project, sid, *, focus_id=None, live=None, limit=16):
    selected = current(project, sid, focus_id)
    with materialized(project, sid, selected, live) as native:
        from saturn_pub.contracts import StateAddress

        value = (
            native.read_port(StateAddress(selected["port"], tuple(selected["selector"])))
            if selected["port"]
            else None
        )
        result = {
            "focus_id": selected["record_id"],
            "boundary": selected["boundary"],
            "execution_point": dict(native.inspect().execution_point),
            "value": describe(value),
            "sample": sample(value, limit),
            "inspect_only": "cut" not in selected["boundary"],
            "active_branch_preserved": True,
        }
    return save(project, sid, "materialization", **result)


def sample(value, limit=16):
    if type(limit) is not int or not 1 <= limit <= 64:
        raise ValueError("sample limit must be in 1..64")
    if hasattr(value, "shape"):
        return value.detach().reshape(-1)[:limit].cpu().tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return None


def _number(value):
    if hasattr(value, "numel"):
        if value.numel() != 1:
            raise ValueError("consumer must select one scalar or a token score")
        value = value.item()
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError("consumer is not a finite scalar")
    return float(value)


def measure(native, consumer):
    from saturn_pub.contracts import StateAddress

    raw = native.read_port(StateAddress(consumer["port"]))
    value = native.read_port(StateAddress(consumer["port"], tuple(consumer["selector"])))
    resolved = dict(consumer)
    decision = None
    if consumer["port"] == "logits":
        if raw is None or not hasattr(raw, "shape") or raw.ndim != 2 or raw.shape[0] != 1:
            raise ValueError("consumer requires batch-one native logits at this horizon")
        decision = int(raw.argmax(-1).item())
        if consumer["selector"]:
            scalar = _number(value)
        else:
            resolved.setdefault("token", decision)
            if max(resolved["token"], resolved.get("contrast", 0)) >= raw.shape[-1]:
                raise ValueError("consumer token is outside native vocabulary")
            scalar = _number(raw[0, resolved["token"]])
            if "contrast" in resolved:
                scalar -= _number(raw[0, resolved["contrast"]])
    else:
        if "token" in resolved:
            raise ValueError("token consumers require the logits port")
        scalar = _number(value)
    return {
        "value": scalar,
        "consumer": resolved,
        "decision": decision,
        "readout": describe(raw),
        "units": "logit-margin"
        if "contrast" in resolved
        else "logit"
        if consumer["port"] == "logits"
        else "native-scalar",
    }


def _edit(edit):
    if (
        not isinstance(edit, dict)
        or not set(edit) <= {"port", "selector", "mode", "value"}
        or not {"port", "value"} <= set(edit)
        or not isinstance(edit["port"], str)
    ):
        raise ValueError("edit requires port/value and optional selector/mode")
    mode = edit.get("mode", "add")
    if mode not in {"add", "replace", "scale"}:
        raise ValueError("edit mode must be add, replace or scale")
    value = _json(edit["value"])

    def valid(x):
        return (type(x) in {int, float} and math.isfinite(x)) or (
            isinstance(x, list) and 0 < len(x) <= 256 and all(valid(y) for y in x)
        )

    if not valid(value) or len(json.dumps(value)) > 16384:
        raise ValueError("edit values must be bounded finite numeric data")
    return {
        "port": edit["port"],
        "selector": _selector(edit.get("selector", [])),
        "mode": mode,
        "value": value,
    }


def plan(project, sid, edits, *, focus_id=None, steps=None, tokens=None, linearize=False):
    selected = current(project, sid, focus_id)
    if not isinstance(edits, list) or not 1 <= len(edits) <= 2:
        raise ValueError("comparison requires one or two edits")
    if steps is None and tokens is None and selected["consumer"]["port"] == "logits":
        steps, tokens = 128, 1
    steps = selected["steps"] if steps is None else steps
    if type(linearize) is not bool:
        raise ValueError("linearize must be boolean")
    if (
        type(steps) is not int
        or not 1 <= steps <= 128
        or (tokens is not None and (type(tokens) is not int or not 1 <= tokens <= 4))
    ):
        raise ValueError("budget requires 1..128 transitions and optional 1..4 own tokens")
    if tokens is not None and steps < 2:
        raise ValueError("own-token comparison needs an explicit transition budget")
    return save(
        project,
        sid,
        "comparison-plan",
        focus_id=selected["record_id"],
        edits=[_edit(e) for e in edits],
        steps=steps,
        tokens=tokens,
        consumer=selected["consumer"],
        linearize=linearize,
        order="A then B; edits operate on each resulting state",
        smooth_rectangle="not-established; additive fixed directions require separate qualification",
    )


def apply_edit(native, edit):
    value = native.read(edit["port"])
    indices = tuple(edit["selector"])
    old = value[indices] if hasattr(value, "shape") and indices else value
    replacement = edit["value"]
    if hasattr(old, "shape"):
        import torch

        replacement = torch.as_tensor(replacement, dtype=old.dtype, device=old.device)
        if replacement.ndim == 0:
            replacement = torch.ones_like(old) * replacement
        if replacement.shape != old.shape:
            raise ValueError("edit shape differs from native selection")
    elif indices:
        raise ValueError("selected edits currently require a tensor port")
    replacement = (
        old + replacement
        if edit["mode"] == "add"
        else old * replacement
        if edit["mode"] == "scale"
        else replacement
    )
    if indices:
        value[indices] = replacement
    else:
        value = replacement
    slot = next((s for s in native.inspect().surface["slots"] if s["name"] == edit["port"]), None)
    if slot is None:
        raise ValueError("edit port is unavailable")
    from dataclasses import replace

    return native.apply(
        replace(Act.replace(edit["port"], value), invalidates=tuple(slot["invalidates"]))
    )


def run_comparison(project, sid, plan_id, *, live=None):
    if plan_id == "@latest":
        selected = current(project, sid)
        plans = [
            r
            for r in records(project, sid, kind="comparison-plan")
            if r["focus_id"] == selected["record_id"]
        ]
        if not plans:
            raise ValueError("current focus has no comparison plan")
        plan_id = plans[-1]["record_id"]
    spec = record(project, sid, plan_id)
    if spec["kind"] != "comparison-plan":
        raise ValueError("expected a comparison plan")
    selected = current(project, sid, spec["focus_id"])
    rows, resolved = [], spec["consumer"]
    base_ports, gradient = None, None
    linearization = {"status": "not-requested"}
    with materialized(project, sid, selected, live) as source:
        parent = source.capture(retain=False)
        live._retain(parent)
        for mask in range(2 ** len(spec["edits"])):
            branch = source.fork(parent)
            cuts, tokens_before = [], None
            observed = None
            try:
                cut = live._retain(branch.capture(retain=False))
                cuts.append(cut.fingerprint)
                for i, edit in enumerate(spec["edits"]):
                    if mask & (1 << i):
                        apply_edit(branch, edit)
                        cuts.append(live._retain(branch.capture(retain=False)).fingerprint)
                input_cut = cuts[-1]
                observation = nullcontext()
                if branch.adapter.execution.get("adapter") == "native-decoder-v1":
                    from .circuit_native import _decoder, observe_layer

                    try:
                        _decoder(branch)
                        observation = observe_layer(branch)
                    except ValueError:
                        pass  # Native futures still run at other supported boundaries.
                if spec["tokens"]:
                    tokens_before = branch.read("tokens").shape[-1]
                for step in range(spec["steps"]):
                    with observation if step == 0 else nullcontext() as ports:
                        branch.continue_(1)
                    if step == 0 and ports is not None:
                        from .circuit_native import finish_observation

                        finish_observation(branch, ports)
                        observed = ports
                    cut = live._retain(branch.capture(retain=False))
                    cuts.append(cut.fingerprint)
                    if (
                        tokens_before is not None
                        and branch.read("tokens").shape[-1] - tokens_before >= spec["tokens"]
                    ):
                        break
                if (
                    tokens_before is not None
                    and branch.read("tokens").shape[-1] - tokens_before != spec["tokens"]
                ):
                    raise ValueError("transition budget ended before requested own-token horizon")
                measurement = measure(branch, resolved)
                if mask == 0:
                    resolved = measurement["consumer"]
                own = (
                    branch.read("tokens")[0, tokens_before:].tolist()
                    if tokens_before is not None
                    else []
                )
                row = {
                    "mask": mask,
                    "status": "measured",
                    "input_cut": input_cut,
                    "result_cut": cuts[-1],
                    "measurement": measurement,
                    "own_tokens": own,
                }
                if mask == 0:
                    base_ports = observed
                    if spec.get("linearize"):
                        from .circuit_native import consumer_gradient

                        try:
                            if spec["tokens"] not in (None, 1) or any(
                                e["port"] != "hidden" for e in spec["edits"]
                            ):
                                raise ValueError(
                                    "linearization supports first-readout hidden edits only"
                                )
                            details, candidate_gradient = consumer_gradient(source, resolved)
                            if details["logits"] != measurement["readout"]:
                                raise ValueError(
                                    "autograd primal differs from native consumer or horizon"
                                )
                            gradient = candidate_gradient
                            linearization = {
                                k: v
                                for k, v in details.items()
                                if k
                                not in {"neuron_product_partials", "neuron_activity_projection"}
                            }
                            linearization.update(
                                status="native-primal-matched", authority="local-autograd"
                            )
                        except (ValueError, RuntimeError) as error:
                            linearization = {
                                "status": "instrument-unavailable",
                                "reason": str(error),
                            }
                if observed is not None and base_ports is not None:
                    from .circuit_native import finite_accounting

                    row["local_accounting"] = finite_accounting(base_ports, observed)
                if gradient is not None:
                    # The just-created input cut is not yet indexed. Retained in-memory
                    # native cuts are authoritative until the arm is published.
                    actual = live._cut(input_cut).payload["hidden"]
                    delta = actual.double() - source.read("hidden").double()
                    row["local_linear_prediction"] = float((gradient.double() * delta).sum())
                    row["native_minus_local_prediction"] = (
                        measurement["value"]
                        - rows[0]["measurement"]["value"]
                        - row["local_linear_prediction"]
                        if rows
                        else 0.0
                    )
            except (ValueError, IndexError, TypeError) as error:
                cut = live._retain(branch.capture(retain=False))
                cuts.append(cut.fingerprint)
                row = {
                    "mask": mask,
                    "status": "evidence-incomplete",
                    "reason": str(error),
                    "result_cut": cut.fingerprint,
                }
            finally:
                live._extra_receipts.extend([*branch.receipts, *branch.attempts])
                live._persist()
            row["cuts"] = list(dict.fromkeys(cuts))
            row["receipts"] = [r.fingerprint for r in branch.receipts]
            retained = save(
                project,
                sid,
                "comparison-arm",
                focus_id=selected["record_id"],
                plan_id=plan_id,
                **row,
            )
            rows.append(retained)
        measured = {r["mask"]: r["measurement"]["value"] for r in rows if r["status"] == "measured"}
        interaction = (
            measured[3] - measured[1] - measured[2] + measured[0] if len(measured) == 4 else None
        )
        return save(
            project,
            sid,
            "comparison",
            focus_id=selected["record_id"],
            plan_id=plan_id,
            arms=[r["record_id"] for r in rows],
            consumer=resolved,
            effects={str(k): v - measured[0] for k, v in measured.items()} if 0 in measured else {},
            interaction=interaction,
            authority="native-finite-intervention",
            linearization=linearization,
            active_branch_preserved=True,
            terminal_status="not-established",
        )


def contract(project, sid, cases, arguments, result, *, name="interface", live=None):
    if not isinstance(cases, list) or not 1 <= len(cases) <= 64:
        raise ValueError("interface requires 1..64 source/result cut pairs")
    if not isinstance(arguments, list) or not 1 <= len(arguments) <= 8:
        raise ValueError("interface requires 1..8 argument selectors")
    arguments = [_consumer(a) for a in arguments]
    result = _consumer(result)
    if any("token" in a for a in [*arguments, result]):
        raise ValueError("interface coordinates use port/selector, not logit contrasts")
    selected = current(project, sid)
    observations, grouped = [], {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"source", "result"}:
            raise ValueError("interface cases require source and result cut IDs")
        # Reuse branch-history verification, not aggregate structural reachability.
        from .project_slice import _Slice

        history = _Slice(project, sid, case["source"])
        if case["source"] not in history.branch(case["result"]):
            raise ValueError("interface result is outside source branch history")
        cursor, path, visited = case["result"], [], set()
        while cursor != case["source"]:
            if cursor in visited or len(path) >= 128:
                raise ValueError("interface native path is cyclic or exceeds its budget")
            visited.add(cursor)
            receipt = history.producer(cursor)
            if receipt is None or receipt["operation"] != "continue":
                raise ValueError("interface cases require an unambiguous native continuation path")
            for transition in reversed(receipt["transitions"]):
                point = transition["before"]
                path.append(
                    {
                        "operation": transition["native_operation"],
                        "point": {
                            k: point.get(k) for k in ("family", "phase", "operator", "index")
                        },
                    }
                )
            cursor = receipt["parent"]
        path.reverse()
        values = []
        for cid, selectors in ((case["source"], arguments), (case["result"], [result])):
            local = {**selected, "boundary": {"cut": cid}, "expected_value": None}
            with materialized(project, sid, local, live) as native:
                from saturn_pub.contracts import StateAddress

                values.append(
                    [
                        describe(native.read_port(StateAddress(s["port"], tuple(s["selector"]))))
                        for s in selectors
                    ]
                )
        row = {**case, "arguments": values[0], "result_value": values[1][0], "native_path": path}
        observations.append(row)
        grouped.setdefault(digest([values[0], path]), []).append(len(observations) - 1)
    witnesses = []
    for group in grouped.values():
        for i in group[1:]:
            if observations[group[0]]["result_value"] != observations[i]["result_value"]:
                witnesses.append(
                    {"left": group[0], "right": i, "reason": "same-arguments-different-next-result"}
                )
    return save(
        project,
        sid,
        "interface-test",
        focus_id=selected["record_id"],
        name=name,
        arguments=arguments,
        result=result,
        cases=observations,
        witnesses=witnesses,
        repeated_argument_groups=sum(len(g) > 1 for g in grouped.values()),
        scope="exact coordinate equality within matched native continuation programs/horizons; supplied cases only, no universal proof",
        terminal_status="not-established",
    )


def finding(project, sid, title, *, record_id=None, note=None):
    selected = current(project, sid)
    candidates = [
        r
        for r in records(project, sid)
        if r.get("focus_id") == selected["record_id"]
        and r["kind"] in {"comparison", "explanation", "interface-test", "materialization"}
    ]
    value = (
        record(project, sid, record_id) if record_id else candidates[-1] if candidates else selected
    )
    selected = current(
        project, sid, value["focus_id"] if "focus_id" in value else value["record_id"]
    )
    return project.finding(
        sid,
        selected["component"],
        title,
        cut_id=selected["supporting_cut"],
        slice_id=selected["slice_id"],
        note=note,
        investigation_id=value["record_id"],
    )


def add_graph_records(project, sid, graph, cuts, annotations):
    """One knowledge graph over original evidence; never new execution edges."""
    from .state_graph import NodeBinding

    rows = records(project, sid)
    nodes = {
        r["record_id"]: graph.add_source(
            source_ref="circuit://sha256/" + r["record_id"],
            binding=NodeBinding(session=sid),
            labels=("circuit-" + r["kind"],),
            metadata={
                "record_id": r["record_id"],
                "kind": r["kind"],
                "focus_id": r.get("focus_id"),
                "terminal_status": r.get("terminal_status", "not-established"),
            },
        )
        for r in rows
    }
    for row in rows:
        node = nodes[row["record_id"]]
        for key in ("focus_id", "plan_id"):
            if row.get(key):
                if row[key] not in nodes:
                    raise ValueError("circuit graph has an unresolved record reference")
                graph.add_edge(node, nodes[row[key]], "related_to", graph="knowledge")
        for key in ("supporting_cut", "input_cut", "result_cut"):
            if row.get(key) in cuts:
                graph.add_edge(node, cuts[row[key]], "describes", graph="knowledge")
        for arm in row.get("arms", []):
            if arm not in nodes:
                raise ValueError("circuit graph has an unresolved arm")
            graph.add_edge(node, nodes[arm], "related_to", graph="knowledge")
    for annotation, identifier in annotations:
        if identifier not in nodes:
            raise ValueError("finding has an unresolved circuit record")
        graph.add_edge(annotation, nodes[identifier], "related_to", graph="knowledge")

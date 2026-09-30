"""Trace deletion, native repair, failed repair and collateral; replay in a new process.

Run: python examples/debugging_repair.py
Only public Saturn and torch are used. The circuit and labels are supplied, not learned.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from repair_circuit import RepairCircuit

from saturn_pub import Act, Session, StateAddress
from saturn_pub.causal import CausalPath, PortObservation
from saturn_pub.cli import Debugger
from saturn_pub.paths import PathSchedule, TimedAct, Trajectory, compare_trajectories
from saturn_pub.store import LocalStore
from saturn_pub.symbols import SymbolBinding, SymbolTable, backtrace
from saturn_pub.values import describe


def evaluate(session):
    output = session.read("output").tolist()
    return {"target_contact": output[0], "collateral_contact": output[1]}


def replay_child(directory, cut_id, sidecar, suppress):
    cut = LocalStore(directory).load(cut_id)
    session = Session.from_cut(RepairCircuit(), cut)
    table = SymbolTable.load(sidecar, cut)
    binding = table.resolve(session, "action.target", context="supplied-left-contact-v1")
    session.step()
    if suppress:
        session.apply(Act.zero("carrier"))
    session.step()
    print(
        json.dumps(
            {
                "payload": describe(session.capture(retain=False).payload),
                "metrics": evaluate(session),
                "symbol": binding.to_dict(),
                "cut": session.capture(retain=False).fingerprint,
            }
        )
    )


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    adapter = RepairCircuit()
    session = adapter.session()
    parent = session.capture()
    plans = {
        "native": PathSchedule(),
        "no-op": PathSchedule((TimedAct(1, Act.add("carrier", torch.zeros(2))),)),
        "early-delete": PathSchedule((TimedAct(1, Act.zero("carrier")),)),
        "delete-both": PathSchedule(
            (TimedAct(1, Act.zero("carrier")), TimedAct(2, Act.zero("carrier")))
        ),
        "late-delete": PathSchedule((TimedAct(2, Act.zero("carrier")),)),
        "late-collateral": PathSchedule(
            (TimedAct(2, Act.add("carrier", torch.tensor([0.0, 1.0]))),)
        ),
    }
    traces = {
        name: Trajectory.run(
            session,
            name=name,
            parent=parent,
            steps=3,
            ports=("carrier", "output"),
            schedule=plan,
            evaluator=evaluate,
        )
        for name, plan in plans.items()
    }
    comparisons = {
        name: compare_trajectories(traces["native"], trace)
        for name, trace in traces.items()
        if name != "native"
    }
    assert comparisons["no-op"]["first_recorded_divergence"] is None
    assert comparisons["early-delete"]["observable_windows"]["carrier"] == [
        {"diverged_at": 1, "recovered_at": 2}
    ]
    assert traces["early-delete"].points[-1].metrics == traces["native"].points[-1].metrics
    assert traces["delete-both"].points[-1].metrics["target_contact"] == 0
    assert traces["late-delete"].points[-1].metrics["target_contact"] == 0
    assert traces["late-collateral"].points[-1].metrics == {
        "target_contact": 1.0,
        "collateral_contact": 1.0,
    }
    assert session.capture(retain=False).fingerprint == parent.fingerprint

    # One supplied, point-qualified binding per supported safe point; no role discovery.
    bindings = []
    for point in traces["native"].points:
        at_point = Session.from_cut(adapter, point.cut)
        bindings.append(
            SymbolBinding.bind(
                at_point,
                "action.target",
                (StateAddress("carrier", (0,), role="carrier"),),
                context="supplied-left-contact-v1",
                context_slots=("source",),
                evidence=("examples/repair_circuit.py: supplied first-channel role",),
                claim="First carrier channel in this supplied two-channel fixture; not learned.",
            )
        )
    table = SymbolTable(tuple(bindings))

    # The ordinary debugger uses the existing Observer engine to watch a resolved symbol.
    debugger = Debugger(session)
    debugger.symbols, debugger.symbol_context = table, "supplied-left-contact-v1"
    debugger.trajectories = traces
    transcript = []
    commands = (
        "info steps",
        "info symbols",
        "resolve action.target",
        "diff native early-delete --first-divergence",
        "capture parent",
        "fork early parent",
        "use early",
        "watch symbol:action.target change",
        "continue",
        "zero carrier",
        "capture deleted",
        "continue",
        "backtrace carrier",
        "unwatch all",
        "step",
        "read output",
    )
    for command in commands:
        transcript.append({"command": command, "result": debugger.execute(command)})
    assert debugger.session.read("output").tolist() == [1.0, 0.0]
    deleted = debugger.cuts["deleted"]
    store = LocalStore(output / "store")
    store.save(parent)
    for trace in traces.values():
        for point in trace.points:
            store.save(point.cut)
        for _, cut in trace.events:
            store.save(cut)
    store.save(deleted)
    sidecar = output / "deleted-debug-info.json"
    table.save(sidecar, deleted)

    # Reopen the watched/deleted state in two new OS processes with reconstructed weights.
    fresh = {}
    for name, suppress in (("repair-enabled", False), ("repair-suppressed", True)):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--replay",
            str(output / "store"),
            deleted.fingerprint,
            str(sidecar),
        ]
        if suppress:
            command.append("--suppress")
        fresh[name] = json.loads(subprocess.check_output(command, text=True))
    expected = {"repair-enabled": "early-delete", "repair-suppressed": "delete-both"}
    for name, trace_name in expected.items():
        final = traces[trace_name].points[-1].cut
        assert fresh[name]["payload"] == describe(final.payload)
        assert fresh[name]["cut"] == final.fingerprint

    repair_trace = traces["early-delete"]
    rows = []
    for kind, offset, port, value in (
        ("first-divergence", 1, "carrier[0]", {"native": 1.0, "candidate": 0.0}),
        (
            "repair",
            2,
            "carrier[0]",
            {"native": 1.0, "candidate": 1.0, "status": "observed recovery in supplied circuit"},
        ),
        ("collateral", 3, "output[1]", {"native": 0.0, "candidate": 0.0}),
        (
            "native-consumer",
            3,
            "output",
            evaluate(Session.from_cut(adapter, repair_trace.points[-1].cut)),
        ),
    ):
        point = repair_trace.points[offset]
        at_point = Session.from_cut(adapter, point.cut)
        rows.append(
            PortObservation.record(
                kind,
                port,
                value,
                at_point,
                point.receipts[-1],
                parent=parent.fingerprint,
                cut=point.cut,
            )
        )
    path = CausalPath.from_observations(
        source="source",
        address="carrier[0]",
        carrier="carrier",
        writer="writer.repair",
        consumer="consumer.threshold",
        observations=tuple(rows),
    )
    report = {
        "schema": "saturn-pub-repair-demo-v1",
        "parent": parent.fingerprint,
        "specimen": "supplied two-writer neural circuit; CPU float32; no training",
        "traces": {name: trace.to_dict() for name, trace in traces.items()},
        "comparisons": comparisons,
        "debugger": transcript,
        "backtrace": backtrace(debugger.session, "output"),
        "causal_path": path.to_dict(),
        "fresh_process": fresh,
        "limitations": "Source, weights, symbols and roles supplied; no semantic "
        "discovery or real-model generalization. Recovery is evaluated "
        "against exact native carrier and the declared threshold consumer.",
        "exactness": "same-program exact selected state and fresh-process cut identity",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Debugging repair: measured walkthrough",
        "",
        report["specimen"],
        "",
        "| Branch | Carrier after first writer/edit | Carrier after repair/edit | "
        "Target contact | Collateral contact |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, trace in traces.items():
        metrics = trace.points[-1].metrics
        lines.append(
            f"| {name} | {trace.points[1].values['carrier'].tolist()} | "
            f"{trace.points[2].values['carrier'].tolist()} | "
            f"{metrics['target_contact']} | {metrics['collateral_contact']} |"
        )
    lines += [
        "",
        "The early deletion diverged at offset 1 and recovered at offset 2. "
        "Suppressing the second writer's result removed target contact. A late "
        "second-channel addition preserved target contact and created collateral.",
        "",
        "Both suffixes replayed in fresh processes with identical complete payloads "
        "and final cut fingerprints. No-op had no observed divergence.",
        "",
        "## Debugger commands",
        "",
        "```text",
        *commands,
        "```",
        "",
        report["limitations"],
    ]
    (output / "walkthrough.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:13]))
    print(f"\nFresh-process replay: exact for both suffixes. Report: {output / 'report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/debugging-repair"))
    parser.add_argument("--replay", nargs=3, metavar=("STORE", "CUT", "DEBUG_INFO"))
    parser.add_argument("--suppress", action="store_true")
    args = parser.parse_args()
    if args.replay:
        replay_child(*args.replay, args.suppress)
    else:
        main(args.output)

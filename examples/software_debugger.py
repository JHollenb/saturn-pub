"""One broad CPU proof of Saturn's model-as-software debugger mechanics.

Run from the checkout root: ``python examples/software_debugger.py``.
The graph, rules, memory and cursor are supplied execution state. The adapter has no
learned parameters, so numeric results demonstrate runtime mechanics rather than semantics.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
from software_machine import GraphMachine

from saturn_pub import Act
from saturn_pub.causal import CausalPath, PortObservation
from saturn_pub.cli import Debugger
from saturn_pub.investigation import Arm, Investigation
from saturn_pub.program import Instruction, Program, ReplayBundle
from saturn_pub.store import LocalStore
from saturn_pub.values import describe, digest


def record(kind, port, value, session, receipt, parent, cut):
    return PortObservation.record(kind, port, value, session, receipt, parent=parent, cut=cut)


def main() -> None:
    output = Path("outputs/software-debugger")
    output.mkdir(parents=True, exist_ok=True)
    adapter = GraphMachine()
    session = adapter.session()
    parent = session.capture()

    # Watches are branch-local diagnostics. A fire retains the exact safe point.
    debugger = Debugger(session)
    debugger.cuts["parent"] = parent
    watch = debugger.execute("watch register change")
    watched = debugger.execute("continue")
    watch_cut = watched["events"][0]["cut"]
    debugger.execute("restore parent")
    trace = debugger.execute("trace 3")
    debugger.execute("restore parent")

    # A masked write uses the live payload and must preserve every unmasked memory cell.
    memory = session.read("memory")
    patch = Act.patch(
        "memory",
        torch.tensor([10.0, 0.0, 0.0, 0.0]),
        torch.tensor([True, False, False, False]),
        expected_digest=digest(describe(memory)),
    )
    patch = replace(patch, preserve=("graph", "rules", "cursor", "register", "output"))
    candidate = session.fork(parent)
    patch_receipt = candidate.apply(patch)
    patched_memory = candidate.read("memory")
    patched_cut = candidate.capture()
    assert torch.equal(patched_memory[1:], memory[1:])
    candidate_receipt = candidate.continue_(3)
    candidate_cut = candidate.capture()
    native = session.replay(parent, steps=3)
    assert session.capture(retain=False).fingerprint == parent.fingerprint
    assert float(native.read("output")[0]) == 7.0
    assert float(candidate.read("output")[0]) == 21.0

    # Run the same-parent native/candidate panel and compile only its measured arm.
    panel = Investigation(
        "How does one masked memory write change the graph's native output?",
        (Arm("native", role="native"), Arm("masked-memory", (patch,), "candidate")),
        3,
    ).run(session, lambda branch: {"output": float(branch.read("output")[0])}, parent=parent)
    program = Program.compile(
        session,
        panel,
        "masked-memory",
        (Instruction(0, patch),),
        context="supplied-graph-v1",
    )
    replay = program.run(session, context="supplied-graph-v1")
    assert torch.equal(replay.read("output"), candidate.read("output"))

    # Keep source/address/writer/carrier and behavioral evidence in distinct graph planes.
    rows = (
        record(
            "source",
            "supplied.memory[0]",
            10.0,
            candidate,
            patch_receipt,
            parent.fingerprint,
            patched_cut,
        ),
        record(
            "address",
            "memory[0]",
            [0],
            candidate,
            patch_receipt,
            parent.fingerprint,
            patched_cut,
        ),
        record(
            "writer",
            "masked-replace",
            patch.manifest(),
            candidate,
            patch_receipt,
            parent.fingerprint,
            patched_cut,
        ),
        record(
            "carrier",
            "memory",
            patched_memory,
            candidate,
            patch_receipt,
            parent.fingerprint,
            patched_cut,
        ),
        record(
            "first-divergence",
            "memory[0]",
            {"native": 3.0, "candidate": 10.0},
            candidate,
            patch_receipt,
            parent.fingerprint,
            patched_cut,
        ),
        record(
            "repair",
            "graph-output",
            {"status": "not-assessed", "reason": "fixture declares no broken target"},
            candidate,
            candidate_receipt,
            parent.fingerprint,
            candidate_cut,
        ),
        record(
            "collateral",
            "memory[1:]",
            {"preserved": True},
            candidate,
            candidate_receipt,
            parent.fingerprint,
            candidate_cut,
        ),
        record(
            "native-consumer",
            "graph-output",
            {"native": 7.0, "candidate": 21.0},
            candidate,
            candidate_receipt,
            parent.fingerprint,
            candidate_cut,
        ),
    )
    causal_path = CausalPath.from_observations(
        source="supplied.memory[0]",
        address="memory[0]",
        carrier="memory",
        writer="masked-replace",
        consumer="graph-output",
        observations=rows,
    )

    # Saving the same immutable cut twice is a content-addressed no-op.
    store = LocalStore(output / "store")
    first = store.save(parent)
    second = store.save(parent)
    assert first == second == parent.fingerprint
    store.save(patched_cut)
    store.save(candidate_cut)
    bundle = ReplayBundle.create(
        program,
        store,
        branch_heads={"parent": parent.fingerprint, "candidate": candidate_cut.fingerprint},
        receipts=(patch_receipt, candidate_receipt),
        adapter_factory={
            "kind": "python",
            "module": "software_machine",
            "qualname": "create_graph_machine",
            "kwargs": {},
        },
        include_data=True,
    )
    bundle_path = output / "replay-bundle.json"
    bundle.save(bundle_path)
    loaded_bundle = ReplayBundle.load(bundle_path)
    loaded_bundle.verify()
    restored_store = LocalStore(output / "restored-store")
    restored_adapter = loaded_bundle.restore(restored_store, allow_data=True)
    restored_cut = restored_store.load(candidate_cut.fingerprint)
    restored = restored_adapter.session().fork(restored_cut)
    assert torch.equal(restored.read("output"), candidate.read("output"))

    report = {
        "specimen": "parameter-free supplied graph/rules/memory/cursor, CPU float32",
        "parent": parent.fingerprint,
        "execution_point": dict(parent.execution_point or {}),
        "surface": dict(parent.surface or {}),
        "watch": watch,
        "watch_stop": watched,
        "watch_safe_point": watch_cut,
        "trace": trace["trace"],
        "masked_write": {
            "receipt": patch_receipt.to_dict(),
            "memory_before": memory.tolist(),
            "memory_after": patched_memory.tolist(),
            "unmasked_preserved": torch.equal(patched_memory[1:], memory[1:]),
        },
        "same_parent_futures": {"native": 7.0, "candidate": 21.0},
        "measured_panel": panel,
        "program_replay_exact": torch.equal(replay.read("output"), candidate.read("output")),
        "cas_deduplicated": first == second,
        "replay_bundle": {
            "fingerprint": bundle.fingerprint,
            "fresh_store_restored": torch.equal(restored.read("output"), candidate.read("output")),
        },
        "causal_path": causal_path.to_dict(),
        "exactness": {
            "same_program_state_replay": "exact",
            "cross-kernel_or_cross-device": "not assessed",
            "semantic_capability": "not assessed; supplied arithmetic fixture",
        },
        "limitations": (
            "The caller supplied graph, rules, target address and intervention. Watch fires are "
            "diagnostic events; signed output movement is an observation, not a semantic verdict."
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ("same_parent_futures", "cas_deduplicated", "exactness")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

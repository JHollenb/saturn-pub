"""Tiny CPU workflow proof. Run with an installed MDB wheel; no downloads/fleet.

python -I circuit_understanding.py --out /absolute/new/result-directory
"""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a fresh output directory")
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    import torch
    from saturn_pub.adapters.decoder import DecoderAdapter

    from mdb import BenchSession, Project
    from mdb_lanes.circuit_investigation import attach_replay, finding
    from mdb_lanes.native_replay_recipe import SCHEMA, seal

    torch.set_num_threads(1)
    native = DecoderAdapter.tiny("qwen3", seed=17)
    project = Project(args.out / "project")
    with BenchSession.from_saturn_pub(
        native.session([1, 2, 3]), project=project, session_id="tiny"
    ) as owner:
        owner.debugger().step(1)
        live = owner.adapter
        parent = live.session.capture(retain=False).fingerprint
        focus = project.focus(
            "tiny", "hidden", consumer={"port": "logits", "token": 4, "contrast": 5}
        )
        recipe = seal(
            {
                "schema": SCHEMA,
                "model_identity": native.model_identity,
                "execution": dict(native.execution),
                "roots": [{"id": "root", "token_ids": [1, 2, 3]}],
                "events": [
                    {
                        "id": "embedded",
                        "session": "root",
                        "kind": "continue",
                        "steps": 1,
                        "expected_cut": parent,
                    }
                ],
            }
        )
        attach_replay(project, "tiny", recipe, "embedded")
        materialization = project.seek("tiny", live=live)
        views = {
            view: project.explain("tiny", view=view, live=live)
            for view in ("mlp", "attention", "frequency", "effect")
        }
        plan = project.plan_comparison(
            "tiny",
            [
                {"port": "hidden", "selector": [0, 0, 1], "value": 0.0001},
                {"port": "hidden", "selector": [0, 0, 1], "value": -0.0001},
            ],
            linearize=True,
        )
        panel = project.run_comparison("tiny", plan["record_id"], live=live)
        arms = [project.circuit_record("tiny", a) for a in panel["arms"]]
        if any(a["status"] != "measured" for a in arms):
            raise RuntimeError("tiny panel has an incomplete native arm")
        cases = [{"source": a["input_cut"], "result": a["result_cut"]} for a in arms]
        interface = project.test_interface(
            "tiny",
            cases,
            [{"port": "hidden", "selector": [0, 0, 0]}],
            {"port": "logits", "selector": [0, 4]},
            name="one-coordinate-hypothesis",
            live=live,
        )
        annotation = finding(
            project, "tiny", "Tiny native circuit workflow", record_id=panel["record_id"]
        )
        preserved = live.session.capture(retain=False).fingerprint == parent
    reopened = Project(project.root, read_only=True)
    assert reopened.focus("tiny") == focus
    assert reopened.circuit_record("tiny", panel["record_id"]) == panel
    graph = reopened.graph("tiny")
    assert graph.verify().ok
    report = {
        "schema": "mdb-tiny-circuit-workflow-v1",
        "device": "cpu",
        "specimen": "untrained random Qwen3; seed17; no semantic capability claim",
        "parameters": sum(p.numel() for p in native.model.parameters()),
        "model_identity": native.model_identity,
        "execution": dict(native.execution),
        "focus": focus["record_id"],
        "materialization": materialization["record_id"],
        "explanations": {k: v["record_id"] for k, v in views.items()},
        "comparison": panel["record_id"],
        "finding": annotation,
        "active_parent_preserved": preserved,
        "reopen_exact": True,
        "root_recipe_target_exact": materialization["value"] == focus["expected_value"],
        "native_primal_exact": views["effect"]["result"]["effect"]["primal_logits_exact"],
        "gradient_norm": views["effect"]["result"]["effect"]["gradient_norm"],
        "source_frequency_closure": views["frequency"]["result"]["frequency"][
            "return_closure_max_abs"
        ],
        "arms": [
            {
                "mask": a["mask"],
                "value": a["measurement"]["value"],
                "decision": a["measurement"]["decision"],
                "own_tokens": a["own_tokens"],
                "local_prediction": a.get("local_linear_prediction"),
                "prediction_residual": a.get("native_minus_local_prediction"),
                "mlp_closure": a["local_accounting"]["mlp"]["closure_max_abs"],
                "attention_closure": a["local_accounting"]["attention"]["closure_max_abs"],
            }
            for a in arms
        ],
        "interface_witnesses": interface["witnesses"],
        "interface_repeated_groups": interface["repeated_argument_groups"],
        "graph_circuit_nodes": sum(
            n.content.get("source_ref", "").startswith("circuit://") for n in graph.nodes.values()
        ),
        "seconds": time.perf_counter() - started,
        "large_model_qualification": "deferred: Beast mrun unavailable",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Generate the recorded project that ``test_serve_*.py`` serves. Needs torch; run once.

Random tiny CPU models only (no downloads): the records exercise the debugger's mechanics, not
learned behaviour. Two sessions in one project:

* ``tiny``: random tiny Qwen3 (``DecoderAdapter.tiny``). Cuts and receipts from a one-layer step,
  a supplied symbol bound on ``hidden`` with a symbol slice, a focus, explanations, a two-edit
  comparison (sham arm first), an interface test and a finding (the workflow of
  ``examples/project-navigation/circuit_understanding.py``).
Records satisfy core's strict authority rule: every measurement in a body is a labelled
``num(value, authority)``; bare numbers appear only at the paths the kind declares structural.

* ``qwen``: random tiny Qwen2 (``QwenAdapter.tiny``) with eager attention observed over five
  layer transitions. A supplied K/V source symbol, a logits symbol, a lifecycle whose uses are
  per-head attention reads (``symbol_attention.attach_attention``), the observed arrays stored
  as tensor blobs (``LocalStore.put_tensor``), one v2 ``occurrence`` record with an occurrence
  pointer, and an occurrence index row for the attention-probability blob.

    W=/path/to/worktree
    env PYTHONPATH=$W/src:$W/packages/mdb/src OMP_NUM_THREADS=1 \
        python packages/mdb/tests/fixtures/serve_fixture_project.py --out NEW_DIRECTORY

Writes NEW_DIRECTORY/project and NEW_DIRECTORY/fixture.json (what was recorded, with ids).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def circuit_session(project, summary):
    from saturn_pub.adapters.decoder import DecoderAdapter

    from mdb import BenchSession
    from mdb.circuit_investigation import attach_replay, finding
    from mdb.native_replay_recipe import SCHEMA, seal
    from mdb.project_slice import bind_live_symbol

    native = DecoderAdapter.tiny("qwen3", seed=17)
    with BenchSession.from_saturn_pub(
        native.session([1, 2, 3]), project=project, session_id="tiny"
    ) as owner:
        owner.debugger().step(1)
        live = owner.adapter
        parent = live.session.capture(retain=False).fingerprint
        bound = bind_live_symbol(live, [
            "demo.carrier", "--port", "hidden", "--context", "demo",
            "--context-slot", "tokens", "--claim", "Supplied carrier at this boundary"])
        sliced = project.slice("tiny", symbol="demo.carrier", context="demo")
        focus = project.focus(
            "tiny", "hidden", consumer={"port": "logits", "token": 4, "contrast": 5}
        )
        recipe = seal({
            "schema": SCHEMA,
            "model_identity": native.model_identity,
            "execution": dict(native.execution),
            "roots": [{"id": "root", "token_ids": [1, 2, 3]}],
            "events": [{"id": "embedded", "session": "root", "kind": "continue", "steps": 1,
                        "expected_cut": parent}],
        })
        attach_replay(project, "tiny", recipe, "embedded")
        project.seek("tiny", live=live)
        for view in ("mlp", "attention", "effect"):
            project.explain("tiny", view=view, live=live)
        plan = project.plan_comparison(
            "tiny",
            [{"port": "hidden", "selector": [0, 0, 1], "value": 0.0001},
             {"port": "hidden", "selector": [0, 0, 1], "value": -0.0001}],
            linearize=True,
        )
        panel = project.run_comparison("tiny", plan["record_id"], live=live)
        arms = [project.circuit_record("tiny", a) for a in panel["arms"]]
        cases = [{"source": a["input_cut"], "result": a["result_cut"]} for a in arms
                 if a["status"] == "measured"]
        project.test_interface(
            "tiny", cases, [{"port": "hidden", "selector": [0, 0, 0]}],
            {"port": "logits", "selector": [0, 4]}, name="one-coordinate-hypothesis", live=live,
        )
        note = finding(project, "tiny", "Tiny native circuit workflow",
                       record_id=panel["record_id"])
    summary["tiny"] = {
        "model": "random tiny Qwen3 seed 17, CPU",
        "symbol_binding": bound["binding"]["fingerprint"],
        "slice": sliced["slice_id"],
        "focus": focus["record_id"],
        "comparison": panel["record_id"],
        "arm_status": [a["status"] for a in arms],
        "finding": note["annotation_id"],
    }


def attention_session(project, summary):
    import numpy as np
    import torch
    from saturn_pub import StateAddress
    from saturn_pub.adapters.qwen import QwenAdapter

    from mdb.attention_observer import AttentionObserver
    from mdb.core.index import index_occurrence
    from mdb.core.pointers import occurrence_pointer
    from mdb.core.records import num, seal_record
    from mdb.prompt_qwen import PromptQwenAdapter
    from mdb.saturn_pub_adapter import SaturnPubDebuggerAdapter
    from mdb.symbol_attention import attach_attention, definition_at, retained_binding

    native = PromptQwenAdapter(QwenAdapter.tiny().model).session([5, 7, 11])
    adapter = SaturnPubDebuggerAdapter(native, project=project, session_id="qwen")
    parent = adapter.checkpoint()["cut_id"]
    with AttentionObserver(native.adapter.model) as observed:
        receipt = native.continue_(5)
    result = adapter.checkpoint()["cut_id"]
    layer = 1
    source = retained_binding(
        project, "qwen", parent, "description.L1",
        tuple(StateAddress(slot, (layer, 0, kv, 0)) for slot in ("keys", "values")
              for kv in range(2)),
        context="example", context_slots=("tokens", "keys", "values"),
        claim="Supplied source role at token row 0; observed native K/V location.")
    output = retained_binding(project, "qwen", result, "output.logits",
                              (StateAddress("logits"),), context="example",
                              claim="Observed native readout.")
    lifecycle = definition_at(project, "qwen", parent, source)
    captured = next(row for row in observed.rows if row["layer"] == layer)
    blobs = {}
    for key in ("query", "key", "value", "attention", "head_return"):
        array = captured[key].detach().to(torch.float32).cpu().numpy()
        blobs[key] = project.store.put_tensor(np.ascontiguousarray(array))
    trace = {"layer": layer, **{
        key: {"shape": blobs[key]["shape"], "sha256": blobs[key]["sha256"],
              "path": "tensors/" + blobs[key]["ref"] + ".safetensors",
              "dtype": blobs[key]["dtype"]}
        for key in ("query", "key", "value", "attention", "head_return")}}
    attention = captured["attention"]
    heads = [{"head": h, "kv_head": h // 2,
              "source_mass": float(attention[0, h, 0, 0]),
              "entropy_nats": float(-(attention[0, h, 0] * attention[0, h, 0].clamp_min(1e-12)
                                      .log()).sum()),
              "source_value_return_norm": float(captured["head_return"][0, 0, h].norm())}
             for h in range(attention.shape[1])]
    joined = attach_attention(project, "qwen", lifecycle, receipt_id=receipt.fingerprint,
                              trace=trace, heads=heads, source_row=0, output_binding=output)
    saved = project.save_symbol_lifecycle("qwen", joined)
    adapter.symbols.bindings.append(output)
    adapter._persist()

    # A larger blob so reduce has something to tile: the hidden state at the result cut.
    hidden = native.read("hidden").detach().to(torch.float32).cpu().numpy()
    hidden_blob = project.store.put_tensor(np.ascontiguousarray(hidden))
    row = int(joined.uses[0].metadata["receiver_row"])
    address = f"t{row}.L{layer}.p"
    pointer = occurrence_pointer(address, program="stepped", shape=blobs["attention"]["shape"],
                                 dtype=blobs["attention"]["dtype"],
                                 sha256=blobs["attention"]["sha256"], cut=parent,
                                 authority="measured")
    index_occurrence(project, "qwen", address, cut_id=parent, blob_id=blobs["attention"]["ref"],
                     layers=2)  # core requires the model's layer count (value_key)
    probabilities = attention[0, 0, 0].tolist()
    order = sorted(range(len(probabilities)), key=lambda i: -probabilities[i])
    retained = [{"src": i, "value": num(probabilities[i], "measured", unit="probability")}
                for i in order[:2]]
    session = project.session("qwen")
    record = seal_record(
        session_id="qwen", kind="occurrence",
        model_identity=json.loads(session["identity"]),
        execution=json.loads(session["execution"]), authority="measured",
        body={
            "history": {"parent_cut": parent, "receipt": receipt.fingerprint},
            "transform": {"port": "p", "head": 0, "convention": "softmax over source rows"},
            "retained": retained,
            "unlisted": num(sum(probabilities[i] for i in order[2:]), "derived-exact",
                            unit="probability"),
            "consumer": f"t{row}.L{layer}.A[h0]",
            "evidence": ["cas://sha256/" + receipt.fingerprint],
        },
        pointers=[pointer],
    )
    from mdb.core.index import put_record

    put_record(project, record)
    summary["qwen"] = {
        "model": "random tiny Qwen2 (4 heads, 2 KV heads, 2 layers), CPU",
        "lifecycle": saved["lifecycle_id"],
        "reads": len(joined.uses),
        "blobs": {key: value["ref"] for key, value in blobs.items()},
        "hidden_blob": hidden_blob["ref"],
        "hidden_shape": hidden_blob["shape"],
        "occurrence_address": address,
        "occurrence_record": record["id"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("choose a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    import torch

    torch.set_num_threads(1)
    from mdb.project import Project

    started = time.perf_counter()
    project = Project(args.out / "project")
    summary = {"schema": "mdb-serve-fixture-v1",
               "specimen": "untrained random tiny models; mechanics only"}
    circuit_session(project, summary)
    attention_session(project, summary)
    reopened = Project(project.root, read_only=True)
    assert reopened.graph().verify().ok
    summary["sessions"] = [s["session_id"] for s in reopened.sessions()]
    summary["seconds"] = round(time.perf_counter() - started, 2)
    (args.out / "fixture.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Tiny native nine-future campaign from sealed complete MLP predictions."""
from __future__ import annotations

import hashlib
import json
import runpy
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from saturn_pub.store import LocalStore

from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
from mdb_lanes.forward_recording import tensor_descriptor
from mdb_lanes.tensor_archive import load_tensor_archive, write_tensor_archive

HERE = Path(__file__).parents[1]


def test_sealed_full_fields_drive_all_nine_native_futures_from_fresh_prepared_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("flux_single_contexts.py", "flux_single_kv.py", "flux_single_mlp_campaign.py", "flux_single_contraction.py", "flux_mlp_replacement.py"):
        shutil.copyfile(HERE / "qualification" / name, name)
    tiny = runpy.run_path(str(HERE / "tests/test_flux_single_contexts.py"))
    engine = runpy.run_path("flux_single_contexts.py")
    prep_util = runpy.run_path("flux_single_mlp_campaign.py")
    mlp = runpy.run_path(str(HERE / "qualification/flux_single_mlp.py"))
    helper = runpy.run_path("flux_mlp_replacement.py")
    campaign = runpy.run_path(str(HERE / "qualification/flux_mlp_replacement_campaign.py"))
    run_panel = campaign["run_panel"]
    monkeypatch.setitem(run_panel.__globals__, "QUERIES", [0, 3])
    monkeypatch.setitem(run_panel.__globals__, "NEURONS", [0, 3])
    model, adapter, session = tiny["tiny_session"]()
    cuts, builder = engine["build_sources"](adapter=adapter, session=session, output=tmp_path / "builder", context={"id": "tiny-p71"})
    cuts, prep = prep_util["prepare_sources"](adapter=adapter, cuts=cuts, output=tmp_path / "preparation",
        builder_ref={"path": "builder/builder.json"}, wall=lambda: None)
    assert builder["counts"]["adapter_steps"] + prep["counts"]["adapter_steps"] == 108
    assert builder["counts"]["acts"] + prep["counts"]["acts"] == 4
    assert builder["counts"]["receipt_emissions"] + prep["counts"]["receipt_emissions"] == 112

    block = model.single_transformer_blocks[6]
    h, i = block.attn.query_dim, block.attn.mlp_hidden_dim
    fused = block.attn.to_qkv_mlp_proj
    weight = fused.weight[3 * h:3 * h + 2 * i].detach().float().numpy().copy()
    bias = fused.bias[3 * h:3 * h + 2 * i].detach().float().numpy().copy() if fused.bias is not None else None
    inputs = {role: (cut.payload["image"].float().numpy().copy(), cut.payload["context"]["single_mod"].float().numpy().copy()) for role, cut in cuts.items()}
    for value in [weight, *([bias] if bias is not None else []), *(value for operands in inputs.values() for value in operands)]:
        value.setflags(write=False)
    output = tmp_path / "result"
    output.mkdir()
    save = prep_util["save"]
    save(output / "source.json", {"scope": "Tiny native module/Act mechanics; synthetic image consumer, no VAE or real prompt encode claim",
        "builder": {"path": "../builder/builder.json"}, "preparation": {"path": "../preparation/preparation.json"},
        "cuts": {role: cut.fingerprint for role, cut in cuts.items()}})
    arrays = {}
    def array(value):
        descriptor = tensor_descriptor(value)
        key = hashlib.sha256(json.dumps({key: descriptor[key] for key in ("dtype", "shape", "sha256")}, sort_keys=True).encode()).hexdigest()
        if key not in arrays:
            arrays[key] = write_tensor_archive(output / "arrays" / (key + ".npy"), value, root=output)
        return arrays[key]
    # This deterministic synthetic consumer makes endpoint identity observable
    # without a trained VAE. Every transformer/typed-suffix step remains native.
    def consumer(state, **kwargs):
        digest = hashlib.sha256(state.read("latent").float().numpy().tobytes()).digest()
        return Image.fromarray(np.full((2, 2, 3), list(digest[:3]), dtype=np.uint8))
    adapter.decode = consumer
    counts = {key: 0 for key in campaign["COUNTS"]}
    # Only executed builder/Act work is claimed here; no checkpoint load or real
    # prompt encoding is performed by this unit-test fixture.
    counts.update(adapter_steps=108, acts=4, receipt_emissions=112, physical_batch=1)
    starts = dict(counts)
    events = []
    def event(kind, **fields):
        events.append({"kind": kind, "counts": dict(counts), **fields})
    branches = [{**row, "roll": 2 if row["roll"] == 512 else 0} for row in helper["BRANCHES"]]
    config = {"branches": branches}
    release = {"native": False, "calls": 0}
    def guard(module, args):
        assert release["native"], "current target entered before full-field seal"
        release["calls"] += 1
    guard_handle = block.register_forward_pre_hook(guard)
    store = LocalStore(output / "typed-store")
    try:
        with ForecastTransaction(output / "transaction", model_identity=adapter.model_identity, input_identity="fresh-pre6",
                source_identity="tiny-worker", frontier={"kind": "architectural"}) as tx:
            binding = {"inputs": {role: {"cut": cut.fingerprint, "image": array(cut.payload["image"]),
                "single_mod": array(cut.payload["context"]["single_mod"])} for role, cut in cuts.items()},
                "weight_rows": array(fused.weight[3 * h:3 * h + 2 * i]), "neuron_chunk": 5}
            tx.frontier("source_binding", binding)
            tx.predict("mlp_field", lambda _: {"predictions": {"cases": {role: helper["predict_mlp_field"](*operands, weight, bias,
                eps=block.norm.eps, hidden_width=h, mlp_width=i, neuron_chunk=5, query_indices=[0, 3], neuron_indices=[0, 3])
                for role, operands in inputs.items()}}})
            assert release["calls"] == 0
            seal = tx.seal()
            release["native"] = True
            tx.execute("native", lambda context: run_panel(pipe=SimpleNamespace(transformer=model, vae=torch.nn.Identity()), adapter=adapter,
                cuts=cuts, output=output, config=config, store=store, array=array, wall=lambda: None, event=event,
                counts=counts, helper=helper, capture=mlp["capture_source_calls"], predictions=context.predictions["mlp_field"]["cases"]))
    finally:
        guard_handle.remove()
    assert verify_transaction(tx.root, expected_seal_sha256=seal["seal_sha256"])["verified"]
    result = tx.observation("native")
    assert result["target_calls"] == release["calls"] == 11
    assert counts["adapter_steps"] == 245 and counts["acts"] == 11 and counts["receipt_emissions"] == 256
    assert counts["adapter_steps"] - starts["adapter_steps"] == 137
    assert counts["acts"] - starts["acts"] == 7 and counts["receipt_emissions"] - starts["receipt_emissions"] == 144
    assert counts["candidate_calls"] == counts["images"] == counts["vae_calls"] == 9
    expected_ids = [row["id"] for row in branches]
    assert [row["id"] for row in result["rows"]] == expected_ids
    assert len(list(output.glob("*.png"))) == 9 and len(list((output / "rows").glob("*.json"))) == 9
    assert result["control_observations"]["sham_png_equals_zero"]
    candidates = json.loads((output / "candidates.json").read_text())["candidates"]
    prefixes = json.loads((output / "prefixes.json").read_text())["prefixes"]
    predictions = tx.prediction("mlp_field")["cases"]
    rows = {row["id"]: row for row in result["rows"]}
    emissions = [json.loads(line) for line in (output / "receipt-emissions.jsonl").read_text().splitlines()]
    assert [row["sequence"] for row in emissions] == list(range(144))
    assert sum(len(row["acts"]) for row in rows.values()) == 7
    for spec in branches:
        name, role = spec["id"], spec["source"]
        row, candidate = rows[name], candidates[name]
        assert candidate["input_parent"] == cuts[spec["base"]].fingerprint
        assert row["parent"] == prefixes[spec["base"]]["post_single6"]
        assert len(row["continuation_receipts"]) == 15
        assert row["continuation_receipts"][-1]["result"] == row["final_cut"]
        if spec["feature"] == "native":
            assert candidate["replacement"] is None and not row["acts"]
            continue
        replacement = candidate["replacement"]
        actual = load_tensor_archive(output, replacement["payload"])
        if spec["feature"] == "predicted":
            expected = np.roll(predictions[role]["activation"], spec["roll"], axis=1)
        else:
            source = candidates[role]
            native = load_tensor_archive(output, source["ports"]["original_concat"])
            expected = np.roll(native[:, source["text_rows"]:, source["attention_width"]:], spec["roll"], axis=1)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        used = load_tensor_archive(output, candidate["ports"]["used_concat"])
        np.testing.assert_array_equal(used[:, candidate["text_rows"]:, candidate["attention_width"]:], expected)
        assert replacement["source"] == role and replacement["roll"] == spec["roll"]
        assert candidate["source_parent"] == cuts[role].fingerprint
        assert row["acts"][0]["slot"] == "image" and row["acts"][0]["undeclared_slots_exact"]
        assert row["acts"][0]["parent"] == row["parent"]
        assert row["acts"][0]["post_act"] == row["continuation_receipts"][0]["parent"]
    observed = json.loads((output / "entry-observations.json").read_text())
    assert observed["calls"] == 11 and set(observed["rows"]) == {"zero", "text"}
    assert [observed["rows"][role]["call_index"] for role in ("zero", "text")] == [0, 1]
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())

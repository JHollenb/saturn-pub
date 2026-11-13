"""Tiny native nine-future campaign from sealed attention and MLP predictions."""
from __future__ import annotations

import hashlib
import json
import runpy
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from saturn_pub.store import LocalStore

from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
from mdb_lanes.forward_recording import tensor_descriptor
from mdb_lanes.tensor_archive import load_tensor_archive, write_tensor_archive

HERE = Path(__file__).parents[1]


def test_sealed_full_fields_drive_all_nine_native_futures_from_fresh_prepared_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("flux_single_contexts.py", "flux_single_kv.py", "flux_single_mlp_campaign.py", "flux_single_contraction.py", "flux_mlp_replacement.py", "flux_attention_reference.py"):
        shutil.copyfile(HERE / "qualification" / name, name)
    tiny = runpy.run_path(str(HERE / "tests/test_flux_single_contexts.py"))
    engine = runpy.run_path("flux_single_contexts.py")
    prep_util = runpy.run_path("flux_single_mlp_campaign.py")
    mlp = runpy.run_path(str(HERE / "qualification/flux_single_mlp.py"))
    helper = runpy.run_path("flux_mlp_replacement.py")
    attention_helper = runpy.run_path("flux_attention_reference.py")
    campaign = runpy.run_path(str(HERE / "qualification/flux_attention_replacement_campaign.py"))
    run_panel = campaign["run_panel"]
    monkeypatch.setitem(run_panel.__globals__, "QUERIES", [0, 3])
    monkeypatch.setitem(run_panel.__globals__, "NEURONS", [0, 3])
    model, adapter, session = tiny["tiny_session"]()
    cuts, builder = engine["build_sources"](adapter=adapter, session=session, output=tmp_path / "builder", context={"id": "tiny-p73"})
    cuts, prep = prep_util["prepare_sources"](adapter=adapter, cuts=cuts, output=tmp_path / "preparation",
        builder_ref={"path": "builder/builder.json"}, wall=lambda: None)
    assert builder["counts"]["adapter_steps"] + prep["counts"]["adapter_steps"] == 108
    assert builder["counts"]["acts"] + prep["counts"]["acts"] == 4
    assert builder["counts"]["receipt_emissions"] + prep["counts"]["receipt_emissions"] == 112

    block = model.single_transformer_blocks[6]
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
    branches = campaign["BRANCHES"]
    config = {"branches":branches,"branch_execution_order":campaign["BRANCH_EXECUTION_ORDER"],
        "observation_plan":campaign["OBSERVATION_PLAN"],"neuron_chunk":5,
        "attention_query_chunk":2,"attention_projection_chunk":5}
    release = {"native": False, "calls": 0}
    def guard(module, args):
        assert release["native"], "current target entered before full-field seal"
        release["calls"] += 1
    guard_handle = block.register_forward_pre_hook(guard)
    store = LocalStore(output / "typed-store")
    try:
        with ForecastTransaction(output / "transaction", model_identity=adapter.model_identity, input_identity="fresh-pre6",
                source_identity="tiny-worker", frontier={"kind": "architectural"}) as tx:
            binding,operands=campaign["bind_prediction_inputs"](block,cuts,array=array,counts=counts,
                builder_ref={"path":"builder/builder.json"},preparation_ref={"path":"preparation/preparation.json"},config=config)
            tx.frontier("source_binding",binding)
            tx.predict("attention_mlp_fields",lambda _:{"predictions":{"cases":{
                role:campaign["predict_fields"](case,operands["parameters"],binding,helper=helper,attention_helper=attention_helper)
                for role,case in operands["cases"].items()}}})
            assert release["calls"] == 0
            seal = tx.seal()
            release["native"] = True
            tx.execute("native", lambda context: run_panel(pipe=SimpleNamespace(transformer=model, vae=torch.nn.Identity()), adapter=adapter,
                cuts=cuts, output=output, config=config, store=store, array=array, wall=lambda: None, event=event,
                counts=counts, helper=helper, capture=mlp["capture_source_calls"], predictions=context.predictions["attention_mlp_fields"]["cases"]))
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
    predictions = tx.prediction("attention_mlp_fields")["cases"]
    rows = {row["id"]: row for row in result["rows"]}
    emissions = [json.loads(line) for line in (output / "receipt-emissions.jsonl").read_text().splitlines()]
    assert [row["sequence"] for row in emissions] == list(range(144))
    specs={row["id"]:row for row in branches}
    expected_emission_order=["prefix.zero","prefix.text"]+[
        name for name in campaign["BRANCH_EXECUTION_ORDER"] for _ in range(15+(specs[name]["feature"]!="native"))]
    assert [row["branch"] for row in emissions]==expected_emission_order
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
        used=load_tensor_archive(output,candidate["ports"]["used_concat"])
        original=load_tensor_archive(output,candidate["ports"]["original_concat"])
        t,h=candidate["text_rows"],candidate["attention_width"]
        np.testing.assert_array_equal(used[:,:t],original[:,:t])
        for field,sl in (("A",slice(None,h)),("M",slice(h,None))):
            if field not in spec["selection"]:
                np.testing.assert_array_equal(used[:,t:,sl],original[:,t:,sl])
                continue
            actual=load_tensor_archive(output,replacement["payloads"][field])
            if spec["feature"]=="predicted":
                expected=predictions[role]["attention"]["attention"] if field=="A" else predictions[role]["mlp"]["activation"]
            else:
                donor=load_tensor_archive(output,candidates[role]["ports"]["original_concat"])
                expected=donor[:,t:,sl]
            np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
            np.testing.assert_array_equal(used[:,t:,sl],expected)
        assert replacement["source"]==role and replacement["selection"]==spec["selection"]
        assert candidate["source_parent"]==cuts[role].fingerprint
        assert row["acts"][0]["slot"] == "image" and row["acts"][0]["undeclared_slots_exact"]
        assert row["acts"][0]["parent"] == row["parent"]
        assert row["acts"][0]["post_act"] == row["continuation_receipts"][0]["parent"]
    observed = json.loads((output / "entry-observations.json").read_text())
    assert observed["calls"] == 11 and set(observed["rows"]) == {"zero", "text"}
    assert [observed["rows"][role]["call_index"] for role in ("zero", "text")] == [0, 1]
    observations=json.loads((output/"continuation-observations.json").read_text())
    assert observations["branch_execution_order"]==campaign["BRANCH_EXECUTION_ORDER"]
    assert set(observations["rows"])==set(expected_ids)
    for name,points in observations["rows"].items():
        assert len(points)==16
        assert [row["index"] for row in points]==list(range(16))
        refs=observations["reference_rows"]
        eligible=[ref for ref in refs if campaign["BRANCH_EXECUTION_ORDER"].index(ref)<=campaign["BRANCH_EXECUTION_ORDER"].index(name)]
        for index,point in enumerate(points):
            assert set(point["comparisons"])==set(eligible)
            assert point["incoming_receipt"]==(rows[name]["continuation_receipts"][index-1]["fingerprint"] if index else (rows[name]["acts"][-1]["receipt_id"] if rows[name]["acts"] else prefixes[rows[name]["base"]]["receipt"]["fingerprint"]))
            for slot,record in point["slots"].items():
                if record["present"]:
                    assert ("array" in record)==(name in refs)
                    if name in refs:
                        retained=load_tensor_archive(output,record["array"])
                        assert list(retained.shape)==record["descriptor"]["shape"]
                    if slot=="latent":
                        assert record["meaning"]["constant_before_commit"]==(index<15)
            if index==15:
                assert not point["slots"]["image"]["present"] and not point["slots"]["text"]["present"]
                assert point["slots"]["latent"]["meaning"]["phase"]=="committed-Z4"
    native_attention=json.loads((output/"attention-stage-observations.json").read_text())
    assert native_attention["calls"]==11
    for role,row in native_attention["rows"].items():
        assert set(row["query"])==set(row["keys"])=={"q_raw","k_raw","v","q_norm","k_norm"}
        for group in ("query","keys"):
            assert all(load_tensor_archive(output,record).dtype==np.float32 for record in row[group].values())
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


def test_continuation_metrics_count_changed_values_and_keep_signed_zero():
    campaign=runpy.run_path(str(HERE/"qualification/flux_attention_replacement_campaign.py"))
    actual=np.array([0,0x8000,0x3f80],np.uint16)
    reference=np.zeros(3,np.uint16)
    result=campaign["bf16_distance"](actual,reference,chunk=1)
    assert result["changed_value_count"]==2 and result["sample_count"]==3
    assert not result["exact_bits"] and result["l2"]==1 and result["signed_mean"]==pytest.approx(1/3)
    assert result["rms"]==pytest.approx(np.sqrt(1/3))
    with pytest.raises(ValueError,match="geometry"):
        campaign["bf16_distance"](actual,reference[:2])


def test_reference_buffers_are_immutable_and_candidate_norms_have_only_fixed_witnesses():
    campaign=runpy.run_path(str(HERE/"qualification/flux_attention_replacement_campaign.py"))
    point={"logical_step":3,"phase":"single","index":7}
    reference=torch.zeros((1,2,4),dtype=torch.bfloat16)
    payload={"image":reference,"text":reference[:,:1],"latent":reference,"noise_prediction":None}
    observer=campaign["ContinuationObserver"](array=lambda value:{"retained":True},plan=campaign["OBSERVATION_PLAN"])
    observer.observe("zero",SimpleNamespace(payload=payload,execution_point=point,fingerprint="reference"),index=0,incoming_receipt="receipt0")
    stored=observer.references["zero"][0]["snapshots"]["image"]
    assert not stored.flags.writeable
    with pytest.raises(ValueError):
        stored[0,0,0]=1
    altered=reference.clone();altered[0,0,0]=2
    row=observer.observe("A.predicted",SimpleNamespace(payload={**payload,"image":altered},execution_point=point,fingerprint="candidate"),index=0,incoming_receipt="receipt1")
    assert row["comparisons"]["zero"]["slots"]["image"]["l2"]==2
    assert row["slots"]["image"]["witness"]["values_fp32"][0]==2
    assert "array" not in row["slots"]["image"]
    assert row["slots"]["latent"]["meaning"]["constant_before_commit"]
    assert campaign["slot_meaning"]({"logical_step":3,"phase":"scheduler"},"image")["stale"]
    assert campaign["slot_meaning"]({"logical_step":3,"phase":"scheduler"},"noise_prediction")["meaningful_propagation_observable"]
    assert campaign["slot_meaning"]({"logical_step":4,"phase":"project"},"latent")["meaningful_propagation_observable"]


def test_fixed_plan_refuses_changed_reference_order_and_selector():
    campaign=runpy.run_path(str(HERE/"qualification/flux_attention_replacement_campaign.py"))
    config={"context":campaign["CONTEXT"],"height":512,"width":512,"steps":4,"single_index":6,
        "selector":{"logical_step":3,"phase":"single","index":6},"query_indices":campaign["QUERIES"],
        "neuron_indices":campaign["NEURONS"],"observation_plan":campaign["OBSERVATION_PLAN"],
        "attention_query_chunk":64,"attention_projection_chunk":256,"branch_execution_order":campaign["BRANCH_EXECUTION_ORDER"],
        "neuron_chunk":256,"array_storage":"bf16-bits","archive_integer_ids":"native-int64","branches":campaign["BRANCHES"],
        "max_wall_seconds":1500,"cpu_threads":4,"archive_compression_level":1,
        **{"max_"+key:campaign["COUNTS"][key] for key in ("adapter_steps","acts","candidate_calls","images","vae_calls")}}
    campaign["validate_config"](config)
    for changed in ({"branch_execution_order":list(reversed(config["branch_execution_order"]))},
            {"selector":{"logical_step":3,"phase":"single","index":7}},
            {"observation_plan":{**config["observation_plan"],"witness_count":65}}):
        with pytest.raises(ValueError):
            campaign["validate_config"]({**config,**changed})

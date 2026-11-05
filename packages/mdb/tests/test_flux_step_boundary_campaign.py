"""Tiny real fourteen-block prediction, complete next-step input and custody."""
from __future__ import annotations

import hashlib
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from saturn_pub.store import LocalStore

from mdb.forecast_transaction import verify_transaction
from mdb.tensor_archive import load_tensor_archive

HERE = Path(__file__).parents[1]


def plan(campaign):
    return {"context":campaign["CONTEXT"],"block_indices":campaign["BLOCK_INDICES"],
        "branches":campaign["BRANCHES"],"branch_execution_order":campaign["ORDER"],
        "selector":{"logical_step":2,"phase":"single","index":6},"height":512,"width":512,"steps":4,
        "sample_indices":campaign["SAMPLE_INDICES"],"image_sample_indices":campaign["IMAGE_SAMPLE_INDICES"],
        "query_chunk":64,"projection_chunk":256,"neuron_chunk":256,"retain_fields":False,
        "array_storage":"bf16-bits","archive_integer_ids":"native-int64","cpu_threads":4,
        "max_wall_seconds":900,"transaction_max_bytes":768<<20,
        "observation_plan":campaign["OBSERVATION_PLAN"],"transport_method":"planned-authenticated-ssh-manifest",
        "transport_limits":{"max_expanded_bytes":8<<30,"max_member_bytes":1<<30},
        "hypotheses":{"scope":"Conditional own-state numerical forecasts; native/RGB deviations remain observations."},
        "scope":"Tiny actual public native transformer/scheduler mechanics; synthetic image consumer.",
        **{"max_"+name:campaign["COUNTS"][name] for name in ("adapter_steps","acts","candidate_calls","images","vae_calls")}}


def stage(tmp_path):
    for name in ("flux_single_contexts.py","flux_single_contexts_campaign.py","flux_single_mlp_campaign.py",
            "flux_single_kv.py","flux_single_contraction.py","flux_mlp_replacement.py",
            "flux_attention_replacement_campaign.py","flux_transition_reference.py","flux_step_boundary_reference.py",
            "flux_step_boundary_campaign.py"):
        shutil.copyfile(HERE / "qualification" / name,tmp_path / name)


@pytest.mark.parametrize("corruption",[None,"commit-bits","chain-join"])
def test_complete_step_forecast_is_sealed_then_consumed_without_native_refresh(tmp_path,monkeypatch,corruption):
    monkeypatch.chdir(tmp_path)
    stage(tmp_path)
    campaign = runpy.run_path("flux_step_boundary_campaign.py")
    tiny = runpy.run_path(str(HERE / "tests/test_flux_single_contexts.py"))
    model,adapter,session = tiny["tiny_session"]()
    output = tmp_path / "result"
    config = plan(campaign)
    config.update(sample_indices=[0,2,3,6],image_sample_indices=[0,1,3],query_chunk=2,projection_chunk=5,neuron_chunk=5)
    counts = {key:0 for key in campaign["COUNTS"]}
    counts["physical_batch"] = 1
    request = {"job_id":"tiny-step89","request_sha256":"tiny-request","source_hashes":{"fixture":"fixed"}}
    events = []
    def event(kind,**fields):
        events.append({"kind":kind,**fields})
    def consumer(state,**kwargs):
        digest = hashlib.sha256(state.read("latent").float().numpy().tobytes()).digest()
        return Image.fromarray(np.full((2,2,3),list(digest[:3]),dtype=np.uint8))
    adapter.decode = consumer
    original = campaign["predict_step"]
    def bad_prediction(*args,**kwargs):
        value = original(*args,**kwargs)
        if corruption == "commit-bits":
            value["commit"]["latent"] = value["commit"]["latent"].copy()
            value["commit"]["latent"].view(np.uint32)[0,0,0] |= np.uint32(1)
        elif corruption == "chain-join":
            value["steps"][8]["input_descriptor"] = {**value["steps"][8]["input_descriptor"],"bf16_le_sha256":"0"*64}
        return value
    if corruption:
        monkeypatch.setitem(campaign["run"].__globals__,"predict_step",bad_prediction)
        with pytest.raises(ValueError,match="exact BF16 bits" if corruption == "commit-bits" else "own-state join"):
            campaign["run"](model=model,adapter=adapter,session=session,output=output,config=config,
                request=request,source={"context":{"id":"tiny89"}},counts=counts,event=event,production_profile=False)
        assert counts["adapter_steps"] == 84 and counts["acts"] == 0
        assert (output / "native-segment.json").is_file()
        assert not any(module._forward_pre_hooks or module._forward_hooks for module in model.modules())
        return
    summary = campaign["run"](model=model,adapter=adapter,session=session,output=output,config=config,
        request=request,source={"context":{"id":"tiny89"}},counts=counts,event=event,production_profile=False)
    assert summary["counts"] == {**campaign["COUNTS"],"model_loads":0,"prompt_encodings":0,"noise_draws":0}
    assert summary["target_calls"] == {**{str(i):4 for i in range(6,20)},"norm_out":4,"proj_out":4}
    assert summary["control_observations"] == {"sham_final_latent_exact_native":True,"sham_png_exact_native":True}
    source_abi = json.loads((output / "native-source-abi.json").read_text())
    assert source_abi["verified"] and source_abi["expected"] == source_abi["actual"]
    assert verify_transaction(output / "transaction",expected_seal_sha256=summary["seal"]["seal_sha256"])["verified"]
    sealed = next(event for event in events if event["kind"] == "step_boundary_sealed")
    assert sealed["blocked_entries"] == 0 and set(sealed["target_entries_before_seal"].values()) == {0}
    assert [event["block_index"] for event in events if event["kind"] == "block_forecast_complete"] == list(range(6,20))
    prefix = json.loads((output / "prefix.json").read_text())
    assert len(prefix["receipts"]) == 68
    store = LocalStore(output / "typed-store")
    source = store.load(prefix["pre6"])
    segment = json.loads((output / "native-segment.json").read_text())
    assert len(segment["boundaries"]) == len(segment["receipts"]) == 16
    for position,boundary in enumerate(segment["boundaries"]):
        assert boundary["cut"] == segment["receipts"][position]["result"]
        actual = store.load(boundary["cut"])
        for name,record in boundary["arrays"].items():
            np.testing.assert_array_equal(load_tensor_archive(output,record),actual.payload[name].float().numpy())
    z3 = store.load(segment["next_input"]["cut"])
    assert z3.payload["step"] == 3 and z3.payload["phase"] == "project" and z3.payload["cursor"] == 0
    assert all(z3.payload[name] is None for name in ("image","text","context","noise_prediction"))
    assert z3.payload["conditioning"].equal(source.payload["conditioning"])
    observations = json.loads((output / "continuation-observations.json").read_text())
    assert set(observations["rows"]) == set(campaign["ORDER"])
    for name in campaign["ORDER"]:
        row = json.loads((output / "rows" / (name+".json")).read_text())
        assert row["parent"] == z3.fingerprint and len(row["continuation_receipts"]) == 28
        assert len(row["acts"]) == int(name != "native")
        if row["acts"]:
            action = row["acts"][0]
            assert action["slot"] == "latent" and action["parent"] == z3.fingerprint
            installed = store.load(action["post_act"])
            assert installed.payload["latent"].stride() == z3.payload["latent"].stride()
        boundaries = observations["rows"][name]
        assert len(boundaries) == 29 and boundaries[0]["point"]["logical_step"] == 3
        assert boundaries[0]["slots"]["latent"]["meaning"]["phase"] == "next-step-input-Z3"
        assert boundaries[-1]["slots"]["latent"]["meaning"]["phase"] == "committed-Z4"
        assert all(item["complete_cut_retained"] == (i in {0,28}) for i,item in enumerate(boundaries))
        assert all(slot["array"] for item in observations["rows"]["native"] for slot in item["slots"].values() if slot["present"])
    emissions = [json.loads(line) for line in (output / "receipt-emissions.jsonl").read_text().splitlines()]
    assert [item["sequence"] for item in emissions] == list(range(170))
    assert [item["branch"] for item in emissions] == ["prefix"]*68+["reference"]*16+["native"]*28+["sham"]*29+["predicted"]*29
    manifest = campaign["finalize_manifest"](output,job_id=request["job_id"],request_sha256=request["request_sha256"])
    # Exercise the actual worker completion envelope, including transaction
    # observations formerly left as MappingProxy in the returned summary.
    completion = json.loads(json.dumps({"summary":summary,"manifest":manifest},sort_keys=True))
    assert completion["summary"]["control_observations"]["sham_png_exact_native"] is True
    saved = json.loads((output / "remote-manifest.json").read_text())
    actual_files = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    assert set(saved["files"]) == actual_files-{"remote-manifest.json"}
    assert manifest["files"] == len(saved["files"])
    for name,value in saved["files"].items():
        raw = (output / name).read_bytes()
        assert value == {"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw),"filetype":"regular"}
    assert not any(module._forward_pre_hooks or module._forward_hooks for module in model.modules())


def test_finite_plan_and_output_custody_refuse_wrong_geometry_and_nonregular_files(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    stage(tmp_path)
    campaign = runpy.run_path("flux_step_boundary_campaign.py")
    config = plan(campaign)
    campaign["validate_config"](config)
    for change in ({"context":{**config["context"],"seed":90}},{"block_indices":[6,7]},
            {"max_adapter_steps":169},{"transaction_max_bytes":1},{"max_wall_seconds":901},
            {"transport_method":"stdout-archive"},{"cpu_threads":True}):
        with pytest.raises(ValueError):
            campaign["validate_config"]({**config,**change})
    for kind in ("symlink","member","count"):
        output = tmp_path / kind
        output.mkdir()
        (output / "a").write_bytes(b"some bytes")
        if kind == "symlink":
            (output / "link").symlink_to(output / "a")
        limits = {"max_member_bytes":1} if kind == "member" else {"max_files":1} if kind == "count" else {}
        with pytest.raises(ValueError):
            campaign["finalize_manifest"](output,job_id="tiny",request_sha256="tiny",**limits)


def test_native_source_abi_refuses_hostile_reference_pin_before_model_execution(tmp_path,monkeypatch):
    from diffusers.models.normalization import AdaLayerNormContinuous
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    monkeypatch.chdir(tmp_path)
    stage(tmp_path)
    campaign = runpy.run_path("flux_step_boundary_campaign.py")
    boundary = runpy.run_path("flux_step_boundary_reference.py")
    kwargs = {"norm_type":AdaLayerNormContinuous,"scheduler_type":FlowMatchEulerDiscreteScheduler,
        "adapter_type":Flux2KleinAdapter}
    result = campaign["verify_native_source_contract"](boundary=boundary,**kwargs)
    assert result["verified"] and result["model_calls"] == 0 and len(result["actual"]) == 5
    for key in ("AdaLayerNormContinuous.module_sha256","AdaLayerNormContinuous.forward_source_sha256",
            "FlowMatchEulerDiscreteScheduler.module_sha256","FlowMatchEulerDiscreteScheduler.step_source_sha256"):
        hostile = {**boundary,"SOURCE_CONTRACT":{**boundary["SOURCE_CONTRACT"],key:"0"*64}}
        with pytest.raises(ValueError,match="native source ABI differs"):
            campaign["verify_native_source_contract"](boundary=hostile,**kwargs)
    monkeypatch.setitem(campaign["verify_native_source_contract"].__globals__,"NATIVE_ADAPTER_MODULE_SHA256","0"*64)
    with pytest.raises(ValueError,match="Flux2KleinAdapter.module_sha256"):
        campaign["verify_native_source_contract"](boundary=boundary,**kwargs)

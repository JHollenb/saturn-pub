"""Tiny own-state nineteen-stage chain and atomic joint1/2 native futures."""
from __future__ import annotations

import hashlib
import inspect
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from PIL import Image
from saturn_pub.store import LocalStore

from mdb_lanes.flux_forecast_closure import FluxForecastClosureAdapter
from mdb_lanes.flux_joint_forecast_closure import FluxJointForecastClosureAdapter
from mdb_lanes.forecast_transaction import verify_transaction
from mdb_lanes.tensor_archive import load_tensor_archive

HERE = Path(__file__).parents[1]
DEPENDENCIES = ("flux_step_boundary_campaign.py","flux_step_boundary_reference.py","flux_transition_reference.py",
    "flux_mlp_replacement.py","flux_single_contexts.py","flux_attention_replacement_campaign.py",
    "flux_reentry_reference.py","flux_reentry_campaign.py","flux_joint_chain_campaign.py")


def stage(tmp_path):
    for name in DEPENDENCIES:
        shutil.copyfile(HERE/"qualification"/name,tmp_path/name)


def config(campaign):
    return {"context":campaign["CONTEXT"],"block_indices":campaign["BLOCK_INDICES"],"branches":campaign["BRANCHES"],
        "branch_execution_order":campaign["ORDER"],"selector":{"logical_step":2,"phase":"single","index":6},
        "height":512,"width":512,"steps":4,"sample_indices":campaign["SAMPLE_INDICES"],
        "image_sample_indices":campaign["IMAGE_SAMPLE_INDICES"],"query_chunk":64,"projection_chunk":256,
        "neuron_chunk":256,"retain_fields":False,"array_storage":"bf16-bits","archive_integer_ids":"native-int64",
        "cpu_threads":4,"max_wall_seconds":900,"transaction_max_bytes":768<<20,
        "transport_method":"planned-authenticated-ssh-manifest","transport_limits":{"max_expanded_bytes":8<<30,"max_member_bytes":1<<30},
        "observation_plan":campaign["OBSERVATION_PLAN"],
        "expected_closure_module_sha256":hashlib.sha256(Path(inspect.getfile(FluxForecastClosureAdapter)).read_bytes()).hexdigest(),
        "expected_joint_closure_module_sha256":hashlib.sha256(Path(inspect.getfile(FluxJointForecastClosureAdapter)).read_bytes()).hexdigest(),
        "scope":"Tiny installed public native model/typed mechanics, synthetic2x2 consumer; no checkpoint/CUDA job.",
        "hypotheses":{"scope":"Own-state whole-stage predictions before native targets; every numerical miss retained."},
        **{"max_"+name:campaign["COUNTS"][name] for name in ("adapter_steps","acts","candidate_calls","images","vae_calls")}}


def tiny_session():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(101)
        model = Flux2Transformer2DModel(in_channels=4,num_layers=5,num_single_layers=20,
            attention_head_dim=8,num_attention_heads=1,joint_attention_dim=6,timestep_guidance_channels=8,
            axes_dims_rope=(2,2,2,2),guidance_embeds=False).eval().to(torch.bfloat16)
        adapter = FluxJointForecastClosureAdapter(model,FlowMatchEulerDiscreteScheduler(),granularity="operation")
        session = adapter.session(latent=torch.randn(1,4,4,dtype=torch.bfloat16),conditioning=torch.randn(1,3,6,dtype=torch.bfloat16),
            img_ids=torch.zeros(1,4,4),txt_ids=torch.zeros(1,3,4),
            timesteps=torch.tensor([1000.,800.,600.,300.]),sigmas=torch.tensor([1.,.8,.6,.3,0.]))
    return model,adapter,session


@pytest.mark.parametrize("corruption",[None,"context-bits","joint-join","image-bits","text-bits"])
def test_own_state_forecast_admits_complete_closures_and_four_native_futures(tmp_path,monkeypatch,corruption):
    monkeypatch.chdir(tmp_path)
    stage(tmp_path)
    campaign = runpy.run_path("flux_joint_chain_campaign.py")
    model,adapter,session = tiny_session()
    cfg = config(campaign)
    cfg.update(sample_indices=[0,2,3,6],image_sample_indices=[0,1,3],query_chunk=2,projection_chunk=5,neuron_chunk=5)
    output = tmp_path/"result"
    counts = {key:0 for key in campaign["COUNTS"]}
    counts["physical_batch"] = 1
    request = {"job_id":"tiny-joint-chain101","request_sha256":"tiny-seal","source_hashes":{"fixture":"fixed"}}
    events,captured = [],{}
    def event(kind,**fields):
        events.append({"kind":kind,**fields})
    def consumer(state,**kwargs):
        digest = hashlib.sha256(state.read("latent").float().numpy().tobytes()).digest()
        return Image.fromarray(np.full((2,2,3),list(digest[:3]),dtype=np.uint8))
    adapter.decode = consumer
    original = campaign["predict_chain"]
    def predict(*args,**kwargs):
        value = original(*args,**kwargs)
        captured["prediction"] = value
        if corruption == "context-bits":
            value["prefix"]["project"]["context"]["single_mod"] = value["prefix"]["project"]["context"]["single_mod"].copy()
            value["prefix"]["project"]["context"]["single_mod"].view(np.uint32)[0,0] |= np.uint32(1)
        elif corruption == "joint-join":
            value["joint1"]["input_text_descriptor"] = {**value["joint1"]["input_text_descriptor"],"bf16_le_sha256":"0"*64}
        elif corruption in {"image-bits","text-bits"}:
            slot = corruption.split("-")[0]
            value["joint1"][slot] = value["joint1"][slot].copy()
            value["joint1"][slot].view(np.uint32).flat[0] |= np.uint32(1)
        return value
    monkeypatch.setitem(campaign["run"].__globals__,"predict_chain",predict)
    kwargs = {"model":model,"adapter":adapter,"session":session,"output":output,"config":cfg,
        "request":request,"source":{"context":{"id":"tiny101"}},"counts":counts,"event":event,"production_profile":False}
    if corruption:
        with pytest.raises(ValueError,match="exact BF16 bits" if corruption != "joint-join" else "own-state"):
            campaign["run"](**kwargs)
        assert counts["adapter_steps"] == 87 and counts["acts"] == 0
        assert (output/"native-segment.json").is_file()
        assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
        return
    summary = campaign["run"](**kwargs)
    assert summary["counts"] == {**campaign["COUNTS"],"model_loads":0,"prompt_encodings":0,"noise_draws":0}
    assert summary["target_calls"] == campaign["contract"]()["target_calls"]
    assert summary["control_observations"] == {"sham_final_latent_exact_native":True,"sham_png_exact_native":True}
    assert verify_transaction(output/"transaction",expected_seal_sha256=summary["seal"]["seal_sha256"])["verified"]
    sealed = next(item for item in events if item["kind"] == "joint_chain_sealed")
    assert sealed["blocked_entries"] == 0 and set(sealed["target_entries_before_seal"].values()) == {0}
    assert [item["kind"] for item in events].index("joint1_forecast_complete") < [item["kind"] for item in events].index("joint_chain_sealed")
    prefix = json.loads((output/"prefix.json").read_text())
    segment = json.loads((output/"native-segment.json").read_text())
    assert len(prefix["receipts"]) == 68 and len(segment["receipts"]) == len(segment["boundaries"]) == 19
    store = LocalStore(output/"typed-store")
    prediction = captured["prediction"]
    def equal_tree(actual,expected):
        if isinstance(actual,torch.Tensor):
            np.testing.assert_array_equal(actual.float().numpy().view(np.uint32),expected.view(np.uint32))
        elif isinstance(actual,dict):
            assert set(actual) == set(expected)
            for key in actual:
                equal_tree(actual[key],expected[key])
        else:
            assert len(actual) == len(expected)
            for left,right in zip(actual,expected,strict=True):
                equal_tree(left,right)
    rows = {name:json.loads((output/"rows"/(name+".json")).read_text()) for name in campaign["ORDER"]}
    for name,row in rows.items():
        assert len(row["continuation_receipts"]) == 25
        assert row["common_post_joint1"]["point"]["index"] == 2
        assert len(row["acts"]) == int(name != "native")
        expected_bridges = int(row["mode"] == "inherited")
        assert len(row["bridge_receipts"]) == len(row["bridge_boundaries"]) == expected_bridges
        for bridge,receipt in zip(row["bridge_boundaries"],row["bridge_receipts"],strict=True):
            assert bridge["cut"] == receipt["result"]
            assert store.load(bridge["cut"]).fingerprint == bridge["cut"]
        assert len(row["bridge_records"]) == expected_bridges
        for bridge in row["bridge_records"]:
            assert bridge["before"]["cut"] == row["installed"]["cut"]
            assert bridge["after"]["cut"] == bridge["receipt"]["result"]
            assert bridge["before"]["point"]["index"] == 1
            assert bridge["after"]["point"]["index"] == 2
            assert store.load(bridge["before"]["cut"]).fingerprint == bridge["before"]["cut"]
        if row["mode"] in {"inherited","joint1"}:
            installed = store.load(row["installed"]["cut"])
            expected = prediction["prefix"]["joint0"] if row["mode"] == "inherited" else prediction["joint1"]
            equal_tree(installed.payload["latent"],prediction["prefix"]["remaining_step"]["commit"]["latent"])
            equal_tree(installed.payload["context"],prediction["prefix"]["project"]["context"])
            equal_tree(installed.payload["image"],expected["image"])
            equal_tree(installed.payload["text"],expected["text"])
            action = row["acts"][0]
            assert set(action["act"]["writes"]) == {"latent","context","image","text"}
            assert len(action["act"]["preconditions"]) == 13
            parent = store.load(action["parent"])
            for slot in ("latent","image","text"):
                assert installed.payload[slot].stride() == parent.payload[slot].stride()
    observations = json.loads((output/"continuation-observations.json").read_text())
    for name,boundaries in observations["rows"].items():
        assert len(boundaries) == 26
        assert boundaries[0]["cut"] == rows[name]["common_post_joint1"]["cut"]
        assert boundaries[-1]["slots"]["latent"]["meaning"]["phase"] == "committed-Z4"
    emissions = [json.loads(line) for line in (output/"receipt-emissions.jsonl").read_text().splitlines()]
    assert [item["sequence"] for item in emissions] == list(range(191))
    expected = ["prefix"]*68+["reference"]*19
    for name,row in rows.items():
        expected += [name]*(25+len(row["acts"])+len(row["bridge_receipts"]))
    assert [item["branch"] for item in emissions] == expected
    manifest = campaign["step"]()["finalize_manifest"](output,job_id=request["job_id"],request_sha256=request["request_sha256"])
    assert json.loads(json.dumps({"summary":summary,"manifest":manifest}))["summary"]["counts"]["images"] == 4
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    # Archived native/bridge closures remain available without replay.
    for entry in segment["boundaries"]:
        record = entry["closure"]["latent"]
        np.testing.assert_array_equal(load_tensor_archive(output,record),store.load(entry["cut"]).payload["latent"].float().numpy())


def test_contract_and_public_closure_source_pin_are_explicit(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    stage(tmp_path)
    campaign = runpy.run_path("flux_joint_chain_campaign.py")
    cfg = config(campaign)
    campaign["validate_config"](cfg)
    assert campaign["contract"]()["counts"] == campaign["COUNTS"]
    assert campaign["contract"]()["panel_counts"]["adapter_steps"] == 120
    for change in ({"context":{**cfg["context"],"seed":98}},{"max_adapter_steps":220},
            {"expected_closure_module_sha256":None},{"cpu_threads":True},{"max_wall_seconds":901}):
        with pytest.raises(ValueError):
            campaign["validate_config"]({**cfg,**change})
    boundary = runpy.run_path("flux_step_boundary_reference.py")
    reentry = runpy.run_path("flux_reentry_reference.py")
    with pytest.raises(ValueError,match="source ABI"):
        campaign["verify_source_abi"](boundary=boundary,reentry=reentry,closure_type=FluxJointForecastClosureAdapter,
            expected_joint_closure_module_sha256="0"*64)

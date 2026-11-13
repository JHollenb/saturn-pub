"""Installed tiny44-stage prediction and live/terminal atomic closure mechanics."""
from __future__ import annotations

import hashlib
import inspect
import io
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

from mdb_lanes.flux_final_latent_closure import FluxFinalLatentClosureAdapter
from mdb_lanes.forecast_transaction import verify_transaction
from mdb_lanes.tensor_archive import load_tensor_archive

HERE = Path(__file__).parents[1]
DEPENDENCIES = ("flux_step_boundary_campaign.py","flux_step_boundary_reference.py","flux_transition_reference.py",
    "flux_mlp_replacement.py","flux_single_contexts.py","flux_attention_replacement_campaign.py",
    "flux_reentry_reference.py","flux_reentry_campaign.py","flux_joint_chain_campaign.py",
    "flux_joint_tail_campaign.py","flux_final_latent_campaign.py")


def stage(tmp_path):
    for name in DEPENDENCIES:
        shutil.copyfile(HERE/"qualification"/name,tmp_path/name)


def config(campaign):
    return {"context":campaign["CONTEXT"],"block_indices":campaign["BLOCK_INDICES"],
        "final_block_indices":campaign["FINAL_BLOCK_INDICES"],"branches":campaign["BRANCHES"],
        "branch_execution_order":campaign["ORDER"],"selector":{"logical_step":2,"phase":"single","index":6},
        "height":512,"width":512,"steps":4,"sample_indices":campaign["SAMPLE_INDICES"],
        "image_sample_indices":campaign["IMAGE_SAMPLE_INDICES"],"query_chunk":64,"projection_chunk":256,
        "neuron_chunk":256,"retain_fields":False,"array_storage":"bf16-bits","archive_integer_ids":"native-int64",
        "cpu_threads":4,"max_wall_seconds":1050,"transaction_max_bytes":1024<<20,
        "transport_method":"planned-authenticated-ssh-manifest","transport_limits":{"max_expanded_bytes":12<<30,"max_member_bytes":1<<30},
        "observation_plan":campaign["OBSERVATION_PLAN"],"expected_closure_module_sha256":campaign["OLD_CLOSURE_SHA256"],
        "expected_joint_closure_module_sha256":campaign["PRIOR_JOINT_CLOSURE_SHA256"],
        "expected_tail_closure_module_sha256":campaign["PRIOR_TAIL_CLOSURE_SHA256"],
        "expected_final_closure_module_sha256":hashlib.sha256(Path(inspect.getfile(FluxFinalLatentClosureAdapter)).read_bytes()).hexdigest(),
        "scope":"Tiny installed public native CPU model/typed mechanics with synthetic2x2 consumer; no checkpoint/CUDA job.",
        "hypotheses":{"scope":"All44 own-state transitions sealed before selected native targets; errors are observations."},
        **{"max_"+name:campaign["COUNTS"][name] for name in ("adapter_steps","acts","candidate_calls","images","vae_calls")}}


def tiny_session():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(107)
        model = Flux2Transformer2DModel(in_channels=4,num_layers=5,num_single_layers=20,
            attention_head_dim=8,num_attention_heads=1,joint_attention_dim=6,timestep_guidance_channels=8,
            axes_dims_rope=(2,2,2,2),guidance_embeds=False).eval().to(torch.bfloat16)
        adapter = FluxFinalLatentClosureAdapter(model,FlowMatchEulerDiscreteScheduler(),granularity="operation")
        session = adapter.session(latent=torch.randn(1,4,4,dtype=torch.bfloat16),conditioning=torch.randn(1,3,6,dtype=torch.bfloat16),
            img_ids=torch.zeros(1,4,4),txt_ids=torch.zeros(1,3,4),
            timesteps=torch.tensor([1000.,800.,600.,300.]),sigmas=torch.tensor([1.,.8,.6,.3,0.]))
    return model,adapter,session


def equal_tree(actual,expected):
    if isinstance(actual,torch.Tensor):
        np.testing.assert_array_equal(actual.float().numpy().view(np.uint32),expected.view(np.uint32))
    elif isinstance(actual,dict):
        assert set(actual) == set(expected)
        for key in actual:
            equal_tree(actual[key],expected[key])
    elif actual is None:
        assert expected is None
    else:
        assert len(actual) == len(expected)
        for left,right in zip(actual,expected,strict=True):
            equal_tree(left,right)


@pytest.mark.parametrize("corruption",[None,"context","final-join","output-bits","terminal-clock","terminal-reset"])
def test_whole_forecast_and_four_native_vae_futures(tmp_path,monkeypatch,corruption):
    monkeypatch.chdir(tmp_path)
    stage(tmp_path)
    campaign = runpy.run_path("flux_final_latent_campaign.py")
    model,adapter,session = tiny_session()
    cfg = config(campaign)
    cfg.update(sample_indices=[0,2,3,6],image_sample_indices=[0,1,3],query_chunk=2,projection_chunk=5,neuron_chunk=5)
    output = tmp_path/"result"
    counts = {key:0 for key in campaign["COUNTS"]}
    counts["physical_batch"] = 1
    request = {"job_id":"tiny-final-latent107","request_sha256":"tiny-seal","source_hashes":{"fixture":"fixed"}}
    events,captured = [],{}
    def event(kind,**fields):
        events.append({"kind":kind,**fields})
    def consumer(state,**kwargs):
        digest = hashlib.sha256(state.read("latent").float().numpy().tobytes()).digest()
        return Image.fromarray(np.full((2,2,3),list(digest[:3]),dtype=np.uint8))
    adapter.decode = consumer
    original = campaign["predict_final_latent"]
    def predict(*args,**kwargs):
        value = original(*args,**kwargs)
        captured["prediction"] = value
        final = value["final_step"]
        if corruption == "context":
            final["context_input_descriptors"]["single_mod"]["sha256"] = "0"*64
        elif corruption == "final-join":
            final["steps"][7]["input_descriptor"] = {**final["steps"][7]["input_descriptor"],"bf16_le_sha256":"0"*64}
        elif corruption == "output-bits":
            final["commit"]["latent"] = final["commit"]["latent"].copy()
            final["commit"]["latent"].view(np.uint32).flat[0] |= np.uint32(1)
        elif corruption == "terminal-clock":
            final["next_state_contract"]["clock"]["step"] = 3
        elif corruption == "terminal-reset":
            final["next_state_contract"]["reset"]["context"] = {"silently-retained":True}
        return value
    monkeypatch.setitem(campaign["run"].__globals__,"predict_final_latent",predict)
    kwargs = {"model":model,"adapter":adapter,"session":session,"output":output,"config":cfg,
        "request":request,"source":{"context":{"id":"tiny107"}},"counts":counts,"event":event,"production_profile":False}
    if corruption:
        with pytest.raises(ValueError,match="exact BF16 bits" if corruption == "output-bits" else "own-state"):
            campaign["run"](**kwargs)
        assert counts["adapter_steps"] == 112 and counts["acts"] == counts["images"] == 0
        assert (output/"native-segment.json").is_file()
        assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
        return
    summary = campaign["run"](**kwargs)
    assert summary["counts"] == {**campaign["COUNTS"],"model_loads":0,"prompt_encodings":0,"noise_draws":0}
    assert summary["target_calls"] == campaign["contract"]()["target_calls"]
    assert summary["control_observations"] == {"sham_final_latent_exact_native":True,"sham_png_exact_native":True}
    assert verify_transaction(output/"transaction",expected_seal_sha256=summary["seal"]["seal_sha256"])["verified"]
    sealed = next(item for item in events if item["kind"] == "final_latent_sealed")
    assert sealed["blocked_entries"] == 0 and set(sealed["target_entries_before_seal"].values()) == {0}
    assert [item["block_index"] for item in events if item["kind"] == "final_single_forecast_complete"] == list(range(20))
    kinds = [item["kind"] for item in events]
    assert kinds.index("final_readout_commit_forecast_complete") < kinds.index("final_latent_sealed") < kinds.index("reference_transition_complete")
    storage = next(item for item in events if item["kind"] == "final_forecast_storage")
    assert 0 < storage["payload_bytes"] < storage["npy_bytes"] < 1024<<20
    segment = json.loads((output/"native-segment.json").read_text())
    assert len(segment["boundaries"]) == len(segment["receipts"]) == 44
    assert segment["native_final"]["cut"] == segment["boundaries"][43]["cut"]
    store = LocalStore(output/"typed-store")
    prediction = captured["prediction"]
    rows = {name:json.loads((output/"rows"/(name+".json")).read_text()) for name in campaign["ORDER"]}
    for name,row in rows.items():
        assert len(row["acts"]) == int(name != "native") and row["continuation_receipts"] == []
        assert len(row["bridge_receipts"]) == len(row["bridge_records"]) == len(row["bridge_boundaries"]) == (22 if name == "bridge.predicted" else 0)
        final = store.load(row["final_cut"])
        assert final.payload["step"] == 4 and final.payload["phase"] == "project" and final.payload["cursor"] == 0
        assert all(final.payload[key] is None for key in ("context","image","text","noise_prediction"))
        if name != "native":
            action = row["acts"][0]
            assert set(action["act"]["writes"]) == {"latent","context","image","text"} and len(action["act"]["preconditions"]) == 13
        if name == "bridge.predicted":
            installed = store.load(row["installed"]["cut"])
            prefix = prediction["prefix"]["prefix"]["prefix"]
            equal_tree(installed.payload["latent"],prefix["remaining_step"]["commit"]["latent"])
            equal_tree(installed.payload["context"],prefix["project"]["context"])
            for slot in ("image","text"):
                equal_tree(installed.payload[slot],prediction["prefix"]["joints"]["4"][slot])
            for index,bridge in enumerate(row["bridge_records"]):
                parent = row["installed"]["cut"] if index == 0 else row["bridge_records"][index-1]["after"]["cut"]
                assert bridge["before"]["cut"] == parent and bridge["after"]["cut"] == bridge["receipt"]["result"]
                assert store.load(bridge["after"]["cut"]).fingerprint == bridge["after"]["cut"]
        elif name == "full.predicted":
            equal_tree(final.payload["latent"],prediction["final_step"]["commit"]["latent"])
    binding = json.loads((output/"binding.json").read_text())
    for index in range(20):
        bank = binding["final_blocks"][str(index)]
        assert bank["block_index"] == index and bank["native_parameter_prefix"] == f"single_transformer_blocks.{index}."
        if index >= 6:
            assert bank["parameters"] == binding["blocks"][str(index)]["parameters"]
            assert bank["native_parameters"] == binding["blocks"][str(index)]["native_parameters"]
        np.testing.assert_array_equal(load_tensor_archive(output,bank["parameters"]["qkv_mlp_weight"]),
            model.single_transformer_blocks[index].attn.to_qkv_mlp_proj.weight.detach().float().numpy())
    observations = json.loads((output/"continuation-observations.json").read_text())
    assert set(observations["rows"]) == {"native","bridge.predicted"}
    for trajectory in observations["rows"].values():
        assert len(trajectory) == 23
        for item in trajectory:
            cut = store.load(item["cut"])
            assert item["complete_cut_retained"] is True
            for key in ("latent","image","text","noise_prediction"):
                archived = item["closure"][key]
                if archived is None:
                    assert cut.payload[key] is None
                else:
                    np.testing.assert_array_equal(load_tensor_archive(output,archived),cut.payload[key].float().numpy())
    emissions = [json.loads(line) for line in (output/"receipt-emissions.jsonl").read_text().splitlines()]
    assert [item["sequence"] for item in emissions] == list(range(137))
    assert [item["branch"] for item in emissions] == ["prefix"]*68+["reference"]*44+["sham"]+["bridge.predicted"]*23+["full.predicted"]
    manifest = campaign["step"]()["finalize_manifest"](output,job_id=request["job_id"],request_sha256=request["request_sha256"],
        max_expanded_bytes=cfg["transport_limits"]["max_expanded_bytes"],max_member_bytes=cfg["transport_limits"]["max_member_bytes"])
    assert json.loads(json.dumps({"summary":summary,"manifest":manifest}))["summary"]["counts"]["images"] == 4
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


def test_contract_source_and_storage_geometry(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    stage(tmp_path)
    campaign = runpy.run_path("flux_final_latent_campaign.py")
    cfg = config(campaign)
    campaign["validate_config"](cfg)
    assert campaign["contract"]()["panel_counts"]["adapter_steps"] == 66
    for change in ({"transaction_max_bytes":1536<<20},{"expected_final_closure_module_sha256":None},
            {"expected_tail_closure_module_sha256":None},{"final_block_indices":list(range(19))},
            {"max_wall_seconds":1051},{"cpu_threads":True}):
        with pytest.raises(ValueError):
            campaign["validate_config"]({**cfg,**change})
    boundary,reentry = (runpy.run_path(name) for name in ("flux_step_boundary_reference.py","flux_reentry_reference.py"))
    with pytest.raises(ValueError,match="source ABI"):
        campaign["verify_source_abi"](boundary=boundary,reentry=reentry,closure_type=FluxFinalLatentClosureAdapter,
            expected_final_closure_module_sha256="0"*64)
    a,b = np.ones((2,3),np.float32),np.zeros((4,),np.float32)
    profile = campaign["array_storage_profile"]({"same1":a,"same2":a.copy(),"different":b})
    expected = []
    for value in (a,b):
        buffer = io.BytesIO()
        np.save(buffer,value,allow_pickle=False)
        expected.append(len(buffer.getvalue()))
    assert profile["unique_arrays"] == 2 and profile["array_occurrences"] == 3 and profile["npy_bytes"] == sum(expected)

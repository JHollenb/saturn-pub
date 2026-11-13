"""Installed tiny own-state two-pass and predicted-first typed admission mechanics."""
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

from mdb_lanes import flux_derived_terminal
from mdb_lanes.flux_final_latent_closure import FluxFinalLatentClosureAdapter
from mdb_lanes.forecast_transaction import verify_transaction
from mdb_lanes.tensor_archive import load_tensor_archive

HERE = Path(__file__).parents[1]
DEPENDENCIES = ("flux_step_boundary_campaign.py","flux_step_boundary_reference.py","flux_transition_reference.py",
    "flux_mlp_replacement.py","flux_single_contexts.py","flux_attention_replacement_campaign.py",
    "flux_reentry_reference.py","flux_reentry_campaign.py","flux_joint_chain_campaign.py",
    "flux_joint_tail_campaign.py","flux_final_latent_campaign.py","flux_two_pass_campaign.py")


def stage(tmp_path):
    for name in DEPENDENCIES:
        shutil.copyfile(HERE/"qualification"/name,tmp_path/name)


def config(campaign):
    return {"context":campaign["CONTEXT"],"block_indices":campaign["BLOCK_INDICES"],
        "logical_steps":[2,3],"branches":campaign["BRANCHES"],
        "branch_execution_order":campaign["EXECUTION_ORDER"],"selector":{"logical_step":2,"phase":"project","index":None},
        "height":512,"width":512,"steps":4,"sample_indices":campaign["SAMPLE_INDICES"],
        "image_sample_indices":campaign["IMAGE_SAMPLE_INDICES"],"query_chunk":64,"projection_chunk":256,
        "neuron_chunk":256,"retain_fields":False,"array_storage":"bf16-bits","archive_integer_ids":"native-int64",
        "cpu_threads":4,"max_wall_seconds":1200,"transaction_max_bytes":1536<<20,
        "transport_method":"planned-authenticated-ssh-manifest","transport_limits":{"max_expanded_bytes":14<<30,"max_member_bytes":1536<<20},
        "observation_plan":campaign["OBSERVATION_PLAN"],"expected_closure_module_sha256":campaign["OLD_CLOSURE_SHA256"],
        "expected_joint_closure_module_sha256":campaign["PRIOR_JOINT_CLOSURE_SHA256"],
        "expected_tail_closure_module_sha256":campaign["PRIOR_TAIL_CLOSURE_SHA256"],
        "expected_final_closure_module_sha256":campaign["PRIOR_FINAL_CLOSURE_SHA256"],
        "expected_derived_terminal_module_sha256":hashlib.sha256(Path(inspect.getfile(flux_derived_terminal)).read_bytes()).hexdigest(),
        "scope":"Tiny installed public CPU native model; synthetic2x2 consumer, no checkpoint/CUDA job.",
        "hypotheses":{"scope":"All56 own-state stages seal and P actual PNG is saved before selected native targets. Errors observations."},
        **{"max_"+name:campaign["COUNTS"][name] for name in ("adapter_steps","acts","candidate_calls","images","vae_calls")}}


def tiny_session():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(109)
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
        for left,right in zip(actual,expected,strict=True):
            equal_tree(left,right)


@pytest.mark.parametrize("corruption",[None,"context","cross-pass","output-bits","terminal-reset","native-before-seal","transformer-during-first-vae"])
def test_two_complete_passes_predicted_image_first(tmp_path,monkeypatch,corruption):
    monkeypatch.chdir(tmp_path)
    stage(tmp_path)
    campaign = runpy.run_path("flux_two_pass_campaign.py")
    model,adapter,session = tiny_session()
    vae = torch.nn.Identity()
    cfg = config(campaign)
    cfg.update(sample_indices=[0,2,3,6],image_sample_indices=[0,1,3],query_chunk=2,projection_chunk=5,neuron_chunk=5)
    output = tmp_path/"result"
    counts = {key:0 for key in campaign["COUNTS"]}
    counts["physical_batch"] = 1
    request = {"job_id":"tiny-two-pass109","request_sha256":"tiny-seal","source_hashes":{"fixture":"fixed"}}
    events,captured,decode_order = [],{},[]
    def event(kind,**fields):
        events.append({"kind":kind,**fields})
    def consumer(state,**kwargs):
        decode_order.append(dict(counts))
        if len(decode_order) == 1:
            assert counts["adapter_steps"] == 56 and counts["acts"] == 1
            assert not (output/"native-segment.json").exists()
            if corruption == "transformer-during-first-vae":
                model.x_embedder(torch.zeros(1,4,4,dtype=torch.bfloat16))
        value = vae(state.read("latent"))
        digest = hashlib.sha256(value.float().numpy().tobytes()).digest()
        return Image.fromarray(np.full((2,2,3),list(digest[:3]),dtype=np.uint8))
    adapter.decode = consumer
    original = campaign["predict_two_pass"]
    def predict(*args,**kwargs):
        if corruption == "native-before-seal":
            model.x_embedder(torch.zeros(1,4,4,dtype=torch.bfloat16))
        value = original(*args,**kwargs)
        captured["prediction"] = value
        second = value["passes"][1]
        if corruption == "context":
            second["context_input_descriptors"]["single_mod"]["sha256"] = "0"*64
        elif corruption == "cross-pass":
            second["input_latent_descriptor"] = {**second["input_latent_descriptor"],"bf16_le_sha256":"0"*64}
        elif corruption == "output-bits":
            second["commit"]["latent"] = second["commit"]["latent"].copy()
            second["commit"]["latent"].view(np.uint32).flat[0] |= np.uint32(1)
        elif corruption == "terminal-reset":
            second["next_state_contract"]["reset"]["context"] = {"silently-retained":True}
        return value
    monkeypatch.setitem(campaign["run"].__globals__,"predict_two_pass",predict)
    kwargs = {"model":model,"adapter":adapter,"session":session,"output":output,"config":cfg,
        "request":request,"source":{"context":{"id":"tiny109"}},"counts":counts,"event":event,
        "guard_components":(model,vae),"production_profile":False}
    if corruption:
        with pytest.raises(ValueError):
            campaign["run"](**kwargs)
        assert counts["adapter_steps"] == 56
        assert counts["acts"] == int(corruption == "transformer-during-first-vae")
        assert counts["images"] == 0
        assert not (output/"native-segment.json").exists()
        assert not any(module._forward_hooks or module._forward_pre_hooks for component in (model,vae) for module in component.modules())
        return
    summary = campaign["run"](**kwargs)
    assert summary["counts"] == {**campaign["COUNTS"],"model_loads":0,"prompt_encodings":0,"noise_draws":0}
    assert summary["target_calls"] == campaign["contract"]()["target_calls"]
    assert summary["control_observations"] == {"sham_final_latent_exact_native":True,"sham_png_exact_native":True}
    assert verify_transaction(output/"transaction",expected_seal_sha256=summary["seal"]["seal_sha256"])["verified"]
    kinds = [item["kind"] for item in events]
    assert kinds.index("two_pass_sealed") < kinds.index("predicted_image_complete_before_native_reference") < kinds.index("reference_transition_complete")
    for kind in ("two_pass_sealed","predicted_image_complete_before_native_reference"):
        eventrow = next(item for item in events if item["kind"] == kind)
        targets = eventrow.get("target_entries_before_seal",eventrow.get("target_entries"))
        assert set(targets.values()) == {0}
    assert [(item["logical_step"],item["block_index"]) for item in events if item["kind"] == "single_forecast_complete"] == [(logical,index) for logical in (2,3) for index in range(20)]
    storage = next(item for item in events if item["kind"] == "two_pass_forecast_storage")
    assert 0 < storage["payload_bytes"] < storage["npy_bytes"] < 1536<<20
    first = json.loads((output/"first-predicted-image.json").read_text())
    assert first["counts"]["adapter_steps"] == 56 and first["counts"]["acts"] == first["counts"]["images"] == first["counts"]["vae_calls"] == 1
    derived = json.loads((output/"derived-terminal.json").read_text())
    assert derived["native_steps"] == derived["acts_applied"] == derived["execution_receipt_emissions"] == 0
    segment = json.loads((output/"native-segment.json").read_text())
    assert len(segment["boundaries"]) == len(segment["receipts"]) == 56
    assert segment["common_post_joint4"]["cut"] == segment["boundaries"][33]["cut"]
    assert segment["native_final"]["cut"] == segment["boundaries"][55]["cut"]
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
            equal_tree(installed.payload["latent"],prediction["passes"][0]["commit"]["latent"])
            equal_tree(installed.payload["context"],prediction["passes"][1]["project"]["context"])
            for slot in ("image","text"):
                equal_tree(installed.payload[slot],prediction["passes"][1]["joints"]["4"][slot])
            for index,bridge in enumerate(row["bridge_records"]):
                parent = row["installed"]["cut"] if index == 0 else row["bridge_records"][index-1]["after"]["cut"]
                assert bridge["before"]["cut"] == parent and bridge["after"]["cut"] == bridge["receipt"]["result"]
                assert store.load(bridge["after"]["cut"]).fingerprint == bridge["after"]["cut"]
        elif name == "full.predicted":
            equal_tree(final.payload["latent"],prediction["passes"][1]["commit"]["latent"])
            assert row["parent"] == derived["scaffold_cut"]
            assert not row["bridge_receipts"]
    binding = json.loads((output/"binding.json").read_text())
    assert set(binding["pass_schedules"]) == {"2","3"} and "context" not in binding["inputs"]
    for index in range(20):
        bank = binding["final_blocks"][str(index)]
        assert bank["block_index"] == index and bank["native_parameter_prefix"] == f"single_transformer_blocks.{index}."
        np.testing.assert_array_equal(load_tensor_archive(output,bank["parameters"]["qkv_mlp_weight"]),
            model.single_transformer_blocks[index].attn.to_qkv_mlp_proj.weight.detach().float().numpy())
    observations = json.loads((output/"continuation-observations.json").read_text())
    assert set(observations["rows"]) == {"native","bridge.predicted"}
    for trajectory in observations["rows"].values():
        assert len(trajectory) == 23
        for item in trajectory:
            cut = store.load(item["cut"])
            for key in ("latent","image","text","noise_prediction"):
                archived = item["closure"][key]
                if archived is None:
                    assert cut.payload[key] is None
                else:
                    np.testing.assert_array_equal(load_tensor_archive(output,archived),cut.payload[key].float().numpy())
    emissions = [json.loads(line) for line in (output/"receipt-emissions.jsonl").read_text().splitlines()]
    assert [item["sequence"] for item in emissions] == list(range(137))
    assert [item["branch"] for item in emissions] == ["prefix"]*56+["full.predicted"]+["reference"]*56+["sham"]+["bridge.predicted"]*23
    assert [row["adapter_steps"] for row in decode_order] == [56,112,112,134]
    manifest = campaign["step"]()["finalize_manifest"](output,job_id=request["job_id"],request_sha256=request["request_sha256"],
        max_expanded_bytes=cfg["transport_limits"]["max_expanded_bytes"],max_member_bytes=cfg["transport_limits"]["max_member_bytes"])
    assert json.loads(json.dumps({"summary":summary,"manifest":manifest}))["summary"]["counts"]["images"] == 4
    assert not any(module._forward_hooks or module._forward_pre_hooks for component in (model,vae) for module in component.modules())


def test_finite_contract_and_source_pin(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    stage(tmp_path)
    campaign = runpy.run_path("flux_two_pass_campaign.py")
    cfg = config(campaign)
    campaign["validate_config"](cfg)
    contract = campaign["contract"]()
    assert contract["prefix_steps"] == contract["reference_steps"] == 56
    assert contract["counts"]["adapter_steps"] == 134 and contract["panel_counts"]["adapter_steps"] == 78
    for change in ({"transaction_max_bytes":1024<<20},{"expected_derived_terminal_module_sha256":None},
            {"logical_steps":[2]},{"branch_execution_order":campaign["ORDER"]},{"max_wall_seconds":1201},{"cpu_threads":True}):
        with pytest.raises(ValueError):
            campaign["validate_config"]({**cfg,**change})
    boundary,reentry = (runpy.run_path(name) for name in ("flux_step_boundary_reference.py","flux_reentry_reference.py"))
    with pytest.raises(ValueError,match="source ABI"):
        campaign["verify_source_abi"](boundary=boundary,reentry=reentry,closure_type=FluxFinalLatentClosureAdapter,
            expected_derived_terminal_module_sha256="0"*64)

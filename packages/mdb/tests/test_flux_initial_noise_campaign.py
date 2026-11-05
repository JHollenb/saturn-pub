"""Installed CPU mechanics for two prospective origins and recipe-only parameters."""
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
from safetensors.torch import save_file
from saturn_pub.store import LocalStore

from mdb import parameter_artifacts
from mdb.flux_final_latent_closure import FluxFinalLatentClosureAdapter
from mdb.forecast_transaction import verify_transaction
from mdb.tensor_archive import load_tensor_archive

HERE = Path(__file__).parents[1]
DEPENDENCIES = ("flux_step_boundary_campaign.py", "flux_step_boundary_reference.py", "flux_transition_reference.py",
    "flux_mlp_replacement.py", "flux_single_contexts.py", "flux_attention_replacement_campaign.py",
    "flux_reentry_reference.py", "flux_reentry_campaign.py", "flux_joint_chain_campaign.py",
    "flux_joint_tail_campaign.py", "flux_final_latent_campaign.py", "flux_two_pass_campaign.py",
    "flux_initial_noise_campaign.py")


def stage(tmp_path):
    for name in DEPENDENCIES:
        shutil.copyfile(HERE / "qualification" / name, tmp_path / name)
    return runpy.run_path(str(tmp_path / "flux_initial_noise_campaign.py"))


def config(c):
    return {"context":c["CONTEXT"], "block_indices":c["BLOCK_INDICES"], "logical_steps":[0,1,2,3],
        "matched_logical_steps":[2,3], "branches":c["BRANCHES"], "branch_execution_order":c["ORDER"],
        "selector":{"logical_step":0,"phase":"project","index":None},
        "matched_selector":{"logical_step":2,"phase":"project","index":None},
        "height":512,"width":512,"steps":4,"sample_indices":c["SAMPLE_INDICES"],
        "image_sample_indices":c["IMAGE_SAMPLE_INDICES"],"query_chunk":64,"projection_chunk":256,
        "neuron_chunk":256,"retain_fields":False,"array_storage":"bf16-bits","archive_integer_ids":"native-int64",
        "parameter_storage":"checkpoint-recipe-v1","parameter_checkpoint_file":"transformer/diffusion_pytorch_model.safetensors",
        "cpu_threads":4,"max_wall_seconds":1800,"full_transaction_max_bytes":3<<30,"matched_transaction_max_bytes":1536<<20,
        "transport_method":"planned-authenticated-ssh-manifest","transport_limits":{"max_expanded_bytes":10<<30,"max_member_bytes":3<<30},
        "observation_plan":c["OBSERVATION_PLAN"],"expected_closure_module_sha256":c["OLD_CLOSURE_SHA256"],
        "expected_joint_closure_module_sha256":c["PRIOR_JOINT_CLOSURE_SHA256"],
        "expected_tail_closure_module_sha256":c["PRIOR_TAIL_CLOSURE_SHA256"],
        "expected_final_closure_module_sha256":c["PRIOR_FINAL_CLOSURE_SHA256"],
        "expected_derived_terminal_module_sha256":c["DERIVED_TERMINAL_SHA256"],
        "expected_parameter_artifacts_module_sha256":c["PARAMETER_MODULE_SHA256"],
        "scope":"Tiny installed CPU random model and synthetic2x2 VAE consumer; no real checkpoint or model job.",
        "hypotheses":{"scope":"112 own transitions before first denoiser; matched56 before logical2/3 targets. Errors retained."},
        **{"max_"+name:c["COUNTS"][name] for name in ("adapter_steps","acts","candidate_calls","images","vae_calls")}}


@pytest.mark.parametrize("history,limit",[("full",3<<30),("matched",1536<<20)])
def test_verifier_uses_declared_history_budget_and_seal_not_legacy_default(tmp_path,monkeypatch,history,limit):
    import mdb.forecast_transaction as transaction

    c = stage(tmp_path)
    calls = []
    # Simulate a valid transaction just above1GiB without writing a giant file.
    def geometry(root,*,expected_seal_sha256=None,max_bytes=1<<30):
        assert (1<<30)+1 <= max_bytes, "offline transaction geometry exceeds bound"
        calls.append((root,expected_seal_sha256,max_bytes))
        return {"verified":True}
    monkeypatch.setattr(transaction,"verify_transaction",geometry)
    root = tmp_path/"transaction"/history
    assert c["verify_history_transaction"](root,config(c),history,{"seal_sha256":"a"*64}) == {"verified":True}
    assert calls == [(root,"a"*64,limit)]


def tiny_session():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(113)
        model = Flux2Transformer2DModel(in_channels=4,num_layers=5,num_single_layers=20,
            attention_head_dim=8,num_attention_heads=1,joint_attention_dim=6,timestep_guidance_channels=8,
            axes_dims_rope=(2,2,2,2),guidance_embeds=False).eval().to(torch.bfloat16)
        adapter = FluxFinalLatentClosureAdapter(model,FlowMatchEulerDiscreteScheduler(),granularity="operation")
        session = adapter.session(latent=torch.randn(1,4,4,dtype=torch.bfloat16),conditioning=torch.randn(1,3,6,dtype=torch.bfloat16),
            img_ids=torch.zeros(1,4,4),txt_ids=torch.zeros(1,3,4),
            timesteps=torch.tensor([1000.,800.,600.,300.]),sigmas=torch.tensor([1.,.8,.6,.3,0.]))
    return model,adapter,session


def checkpoint(c,tmp_path,model):
    path = tmp_path / "checkpoint" / "transformer" / "diffusion_pytorch_model.safetensors"
    path.parent.mkdir(parents=True)
    save_file({name:value.detach().contiguous().cpu() for name,value in model.named_parameters()},path)
    return c["CheckpointParameters"](path.parent.parent,"transformer/"+path.name,c["sha"](path),model)


@pytest.mark.parametrize("corruption",[None,"native-before-full","native-before-matched","transformer-during-first-vae","cross-pass","terminal-reset","output-bits"])
def test_complete_initial_and_matched_histories(tmp_path,monkeypatch,corruption):
    monkeypatch.chdir(tmp_path)
    c = stage(tmp_path)
    model,adapter,session = tiny_session()
    cp = checkpoint(c,tmp_path,model)
    vae = torch.nn.Identity()
    cfg = config(c)
    cfg.update(sample_indices=[0,2,3,6],image_sample_indices=[0,1,3],query_chunk=2,projection_chunk=5,neuron_chunk=5)
    counts = {key:0 for key in c["COUNTS"]}
    counts["physical_batch"] = 1
    output = tmp_path / "result"
    events,predictions,decode_order = [],{},[]
    def event(kind,**fields):
        events.append({"kind":kind,**fields})
    def consumer(state,**kwargs):
        decode_order.append(dict(counts))
        if len(decode_order) == 1 and corruption == "transformer-during-first-vae":
            model.x_embedder(torch.zeros(1,4,4,dtype=torch.bfloat16))
        value = vae(state.read("latent"))
        digest = hashlib.sha256(value.float().numpy().tobytes()).digest()
        return Image.fromarray(np.full((2,2,3),list(digest[:3]),dtype=np.uint8))
    adapter.decode = consumer
    original = c["predict_passes"]
    def predict(binding,**kwargs):
        history = binding["history"]
        if corruption == "native-before-"+history:
            model.x_embedder(torch.zeros(1,4,4,dtype=torch.bfloat16))
        value = original(binding,**kwargs)
        predictions[history] = value
        last = value["passes"][-1]
        if corruption == "cross-pass":
            value["passes"][1]["input_latent_descriptor"]["bf16_le_sha256"] = "0"*64
        elif corruption == "terminal-reset":
            last["next_state_contract"]["reset"]["context"] = {"stale":True}
        elif corruption == "output-bits":
            last["commit"]["latent"] = last["commit"]["latent"].copy()
            last["commit"]["latent"].view(np.uint32).flat[0] |= np.uint32(1)
        return value
    monkeypatch.setitem(c["run"].__globals__,"predict_passes",predict)
    kwargs = {"model":model,"adapter":adapter,"session":session,"checkpoint":cp,"output":output,"config":cfg,
        "request":{"job_id":"tiny-initial113","request_sha256":"tiny-seal","source_hashes":{"fixture":"fixed"}},
        "source":{"context":{"id":"tiny113"}},"counts":counts,"event":event,"guard_components":(model,vae),"production_profile":False}
    if corruption:
        with pytest.raises(ValueError):
            c["run"](**kwargs)
        assert counts["adapter_steps"] == (56 if corruption == "native-before-matched" else 0)
        assert counts["images"] == int(corruption == "native-before-matched")
        assert counts["acts"] == int(corruption in ("native-before-matched","transformer-during-first-vae"))
        if corruption == "native-before-matched":
            assert (output/"rows/full.predicted.json").is_file()
            assert len(json.loads((output/"native-prefix.json").read_text())["receipts"]) == 56
        assert not (output/"native-segment.json").exists()
        assert not any(module._forward_hooks or module._forward_pre_hooks for component in (model,vae) for module in component.modules())
        return
    summary = c["run"](**kwargs)
    assert summary["counts"] == {**c["COUNTS"],"model_loads":0,"prompt_encodings":0,"noise_draws":0}
    assert summary["target_calls"] == c["contract"]()["target_calls"]
    assert summary["target_calls_by_step"] == c["contract"]()["target_calls_by_step"]
    assert summary["control_observations"] == {"sham_final_latent_exact_native":True,"sham_png_exact_native":True}
    assert [item["adapter_steps"] for item in decode_order] == [0,56,112,112]
    kinds = [item["kind"] for item in events]
    assert kinds.index("history_full_sealed") < kinds.index("predicted_image_complete") < kinds.index("reference_transition_complete")
    for history,logical,steps,acts in (("full",0,0,1),("matched",2,56,2)):
        assert verify_transaction(output/"transaction"/history,expected_seal_sha256=summary["seals"][history]["seal_sha256"],
            max_bytes=cfg[history+"_transaction_max_bytes"])["verified"]
        first = json.loads((output/("first-image-"+history+".json")).read_text())
        assert first["counts"]["adapter_steps"] == steps and first["counts"]["acts"] == acts
        assert all(not any(row.values()) for row in first["clock_qualified_targets_remaining"].values())
        assert sum(first["target_calls"].values()) == sum(c["target_counts"](2).values()) * int(logical == 2)
        frontier = json.loads((output/("binding-"+history+".json")).read_text())
        assert "origins" not in frontier and frontier["origin"]["logical_step"] == logical
        assert frontier["logical_steps"] == list(range(logical,4))
        assert len(predictions[history]["passes"]) == 4-logical
    initial_binding = json.loads((output/"binding-initial.json").read_text())
    binding = json.loads((output/"binding.json").read_text())
    assert set(initial_binding["origins"]) == {"full"} and set(binding["origins"]) == {"full","matched"}
    native = json.loads((output/"native-segment.json").read_text())
    assert len(native["boundaries"]) == len(native["receipts"]) == 112
    assert native["native_Z2"]["cut"] == native["boundaries"][55]["cut"] == binding["origins"]["matched"]["cut"]
    assert native["native_final"]["cut"] == native["boundaries"][-1]["cut"]
    store = LocalStore(output/"typed-store")
    for item in native["boundaries"]:
        cut = store.load(item["cut"])
        for slot in ("latent","image","text","noise_prediction"):
            record = item["closure"][slot]
            if record is None:
                assert cut.payload[slot] is None
            else:
                np.testing.assert_array_equal(load_tensor_archive(output,record),cut.payload[slot].float().numpy())
    for row in summary["rows"]:
        final = store.load(row["final_cut"])
        assert final.payload["step"] == 4 and all(final.payload[key] is None for key in ("context","image","text","noise_prediction"))
        assert len(row["acts"]) == int(row["id"] != "native") and row["continuation_receipts"] == []
        if row["id"] != "native":
            act = row["acts"][0]["act"]
            assert len(act["preconditions"]) == 13 and set(act["writes"]) == {"latent","image","text","context"}
        if row["id"].endswith(".predicted"):
            expected = predictions[row["history"]]["passes"][-1]["commit"]["latent"]
            np.testing.assert_array_equal(final.payload["latent"].float().numpy().view(np.uint32),expected.view(np.uint32))
    emissions = [json.loads(line) for line in (output/"receipt-emissions.jsonl").read_text().splitlines()]
    assert [item["sequence"] for item in emissions] == list(range(115))
    assert [item["branch"] for item in emissions] == ["full.predicted"]+["reference"]*56+["matched.predicted"]+["reference"]*56+["sham"]
    manifest = parameter_artifacts.validate_manifest(json.loads((output/"parameter-artifacts.json").read_text()))
    assert c["sha"](output/"retrieve_weights.py") == c["PARAMETER_MODULE_SHA256"]
    assert all(not (output/item["path"]).exists() for item in manifest["artifacts"])
    external = tmp_path/"shared-parameter-cache"
    parameter_artifacts.restore(manifest,external,cp.path.parent.parent)
    assert parameter_artifacts.verify_present(manifest,external)["complete"]
    bank = binding["final_blocks"]["0"]["parameters"]["qkv_mlp_weight"]
    np.testing.assert_array_equal(load_tensor_archive(external,bank),model.single_transformer_blocks[0].attn.to_qkv_mlp_proj.weight.detach().float().numpy())
    assert not any(module._forward_hooks or module._forward_pre_hooks for component in (model,vae) for module in component.modules())
    assert json.loads(json.dumps(summary))["counts"]["images"] == 4


def test_checkpoint_allowlist_mutation_and_hash(tmp_path):
    c = stage(tmp_path)
    model,_,_ = tiny_session()
    cp = checkpoint(c,tmp_path,model)
    value = model.single_transformer_blocks[0].attn.to_qkv_mlp_proj.weight
    record = cp.record(value)
    decoded = cp.load(record)
    assert not decoded.flags.writeable
    np.testing.assert_array_equal(decoded,value.detach().float().numpy())
    with pytest.raises(ValueError,match="complete file"):
        c["CheckpointParameters"](cp.path.parent.parent,cp.filename,"0"*64,model)
    with pytest.raises(ValueError,match="allowlist"):
        cp.load({**record,"path":"unbound.npy"})
    with cp.path.open("ab") as stream:
        stream.write(b"hostile")
    with pytest.raises(ValueError,match="changed"):
        cp.load(record)


def test_finite_contract_and_immutable_source_pins(tmp_path):
    c = stage(tmp_path)
    cfg = config(c)
    c["validate_config"](cfg)
    assert c["contract"]()["mathematical_steps"] == 168
    assert c["contract"]()["counts"]["receipt_emissions"] == 115
    assert c["sha"](Path(inspect.getfile(parameter_artifacts))) == c["PARAMETER_MODULE_SHA256"]
    for change in ({"logical_steps":[0,1]},{"matched_logical_steps":[3]},{"max_wall_seconds":1801},
            {"cpu_threads":True},{"parameter_storage":"float-cache"},{"full_transaction_max_bytes":4<<30},
            {"expected_parameter_artifacts_module_sha256":"0"*64}):
        with pytest.raises(ValueError):
            c["validate_config"]({**cfg,**change})
    (tmp_path/"flux_reentry_reference.py").write_text("changed")
    with pytest.raises(ValueError,match="immutable predecessor"):
        c["prior"]()

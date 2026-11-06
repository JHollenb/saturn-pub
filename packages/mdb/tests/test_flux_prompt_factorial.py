"""Installed tiny CPU tests; no real checkpoint, encoder, VAE or mrun job."""
from __future__ import annotations

import hashlib
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

from mdb.flux_final_latent_closure import FluxFinalLatentClosureAdapter

HERE = Path(__file__).parents[1]
DEPENDENCIES = ("flux_step_boundary_campaign.py", "flux_step_boundary_reference.py", "flux_transition_reference.py",
    "flux_mlp_replacement.py", "flux_single_contexts.py", "flux_attention_replacement_campaign.py",
    "flux_reentry_reference.py", "flux_reentry_campaign.py", "flux_joint_chain_campaign.py",
    "flux_joint_tail_campaign.py", "flux_final_latent_campaign.py", "flux_two_pass_campaign.py",
    "flux_initial_noise_campaign.py", "flux_prompt_factorial.py")


def stage(tmp_path):
    for name in DEPENDENCIES:
        shutil.copyfile(HERE/"qualification"/name,tmp_path/name)
    return runpy.run_path(str(tmp_path/"flux_prompt_factorial.py"))


def tiny():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(127)
        model = Flux2Transformer2DModel(in_channels=4,num_layers=5,num_single_layers=20,
            attention_head_dim=8,num_attention_heads=1,joint_attention_dim=6,timestep_guidance_channels=8,
            axes_dims_rope=(2,2,2,2),guidance_embeds=False).eval().to(torch.bfloat16)
        adapter = FluxFinalLatentClosureAdapter(model,FlowMatchEulerDiscreteScheduler(),granularity="operation")
        latent = torch.randn(1,4,4,dtype=torch.bfloat16)
        conditions = [torch.randn(1,3,6,dtype=torch.bfloat16) for _ in range(4)]
        sessions = {key:adapter.session(latent=latent.clone(),conditioning=conditions[index],
            img_ids=torch.zeros(1,4,4,dtype=torch.int64),txt_ids=torch.zeros(1,3,4,dtype=torch.int64)+index,
            timesteps=torch.tensor([1000.,800.,600.,300.]),sigmas=torch.tensor([1.,.8,.6,.3,0.]))
            for index,key in enumerate(("c0-o0","c0-o1","c1-o0","c1-o1"))}
    return model,adapter,sessions


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("factorial")
    c = stage(root)
    model,adapter,sessions = tiny()
    path = root/"checkpoint/transformer/diffusion_pytorch_model.safetensors"
    path.parent.mkdir(parents=True)
    save_file({name:value.detach().contiguous().cpu() for name,value in model.named_parameters()},path)
    cp = c["prior"]()["CheckpointParameters"](path.parent.parent,"transformer/"+path.name,c["sha"](path),model)
    cfg = c["make_config"]()
    cfg.update(sample_indices=[0,2,3,6],image_sample_indices=[0,1,3],query_chunk=2,projection_chunk=5,neuron_chunk=5)
    counts = dict.fromkeys(c["COUNTS"],0)
    counts["physical_batch"] = 1
    events,decoded = [],[]
    vae = torch.nn.Identity()
    def consume(state,**kwargs):
        decoded.append(dict(counts))
        value = vae(state.read("latent"))
        digest = hashlib.sha256(value.float().numpy().tobytes()).digest()
        return Image.fromarray(np.full((2,2,3),list(digest[:3]),dtype=np.uint8))
    adapter.decode = consume
    output = root/"result"
    summary = c["run"](model=model,adapter=adapter,sessions=sessions,checkpoint=cp,output=output,config=cfg,
        request={"job_id":"tiny-factorial","request_sha256":"fixture","source_hashes":{"fixture":"fixed"}},
        source={"scope":"Tiny random model, synthetic2x2 pixel consumer; no native text encoder or real VAE."},counts=counts,
        guard_components=(model,vae),event=lambda kind,**fields:events.append({"kind":kind,**fields}),production_profile=False)
    return {"c":c,"data":output,"summary":summary,"counts":counts,"events":events,"decoded":decoded,
        "sessions":sessions,"root":root,"checkpoint":path.parent.parent}


def test_all_seals_and_four_actual_consumers_precede_any_native(completed):
    x = completed
    c,data = x["c"],x["data"]
    assert x["counts"] == {**c["COUNTS"],"model_loads":0,"prompt_encodings":0,"noise_draws":0}
    assert [row["adapter_steps"] for row in x["decoded"]] == [0,0,0,0,112,224,336,448,448]
    kinds = [row["kind"] for row in x["events"]]
    assert kinds.count("condition_sealed") == 4
    assert kinds.index("cohort_sealed") < kinds.index("native_transition")
    cohort = json.loads((data/"cohort-seal.json").read_text())
    assert cohort["counts"]["acts"] == cohort["counts"]["adapter_steps"] == 0
    for i,key in enumerate(c["contract"]()["forecast_order"]):
        first = json.loads((data/"first-images"/(key+".json")).read_text())
        assert first["counts"]["images"] == i+1
        assert first["counts"]["adapter_steps"] == 0 and set(first["target_calls"].values()) == {0}
        assert first["cohort_seal_sha256"] == c["sha"](data/"cohort-seal.json")
    assert x["summary"]["target_calls"] == c["target_counts"](16)
    assert x["summary"]["control_observations"] == {"sham_png_exact":True,"sham_latent_exact":True}
    assert all(not session.history for session in x["sessions"].values())


def test_exact_shared_noise_and_permitted_prompt_changes_only(completed):
    x,c = completed,completed["c"]
    binding = json.loads((x["data"]/"binding.json").read_text())
    from mdb.tensor_archive import load_tensor_archive
    origins = binding["origins"]
    for slot in ("latent","img_ids","timesteps","sigmas"):
        values = [load_tensor_archive(x["data"],row["inputs"][slot]) for row in origins.values()]
        assert all(np.array_equal(value,values[0]) for value in values)
    ids = [load_tensor_archive(x["data"],row["inputs"]["txt_ids"]) for row in origins.values()]
    assert all(value.dtype == np.int64 for value in ids)
    assert not np.array_equal(ids[0],ids[-1])
    for row in origins.values():
        assert row["point"]["logical_step"] == 0 and row["point"]["phase"] == "project"
    assert c["CONDITIONS"][0]["seed"] == 127


def test_storage_has_receipts_every_step_but_only_declared_complete_cuts(completed):
    data,c = completed["data"],completed["c"]
    assert len(list((data/"typed-store/receipts").glob("*.json"))) == 453
    retained,nonsaved,metadata = set(),set(),set()
    for key in c["contract"]()["native_order"]:
        lines = [json.loads(line) for line in (data/"native"/(key+".jsonl")).read_text().splitlines()]
        assert len(lines) == 112
        for row in lines:
            capture = row["capture"]
            metadata.add(capture["cut"])
            assert c["identity"](capture["manifest"]) == capture["cut"]
            if row["local_index"] in c["LANDMARKS"]:
                retained.add(capture["cut"])
                assert all(value["archive"] for value in row["outputs"].values())
            else:
                nonsaved.add(capture["cut"])
                assert all(value["archive"] is None for value in row["outputs"].values())
                assert all(error["full_field"] is None for error in row["errors"].values())
                assert "not retained" in capture["scope"]
    saved = {path.stem for path in (data/"typed-store/cuts").glob("*.json")}
    assert len(metadata) == 448 and len(retained) == 80 and len(nonsaved) == 368
    assert retained <= saved and not nonsaved & saved
    recipes = json.loads((data/"parameter-artifacts.json").read_text())
    assert recipes["artifacts"] and all(not (data/row["path"]).exists() for row in recipes["artifacts"])
    total = sum(path.stat().st_size for path in data.rglob("*") if path.is_file())
    assert total < 64<<20  # Tiny mechanics only, not a production storage estimate.


@pytest.mark.parametrize("change",["origin","order","own-state","coverage","terminal","Z0","conditioning"])
def test_program_rejects_stale_state_or_invented_coverage(completed,change):
    from mdb.forecast_transaction import load_forecast_tree

    c,data = completed["c"],completed["data"]
    binding = json.loads((data/"binding.json").read_text())
    raw = json.loads((data/"transactions/c0-o0/forecasts/program.json").read_text())
    program = json.loads(json.dumps(c["prior"]()["util"]()["json_value"](
        load_forecast_tree(data/"transactions/c0-o0",raw["predictions"]))))
    if change == "origin":
        program["origin_cut"] = "0"*64
    elif change == "order":
        program["transitions"][28]["logical_step"] = 0
    elif change == "own-state":
        program["transitions"][9]["inputs"]["image"]["fp32_le_sha256"] = "0"*64
    elif change == "coverage":
        program["transitions"][9]["full_arrays_retained"] = True
    elif change == "terminal":
        program["terminal_latent"]["descriptor"]["fp32_le_sha256"] = "0"*64
    elif change == "Z0":
        program["transitions"][0]["inputs"]["latent"]["bf16_le_sha256"] = "0"*64
    else:
        program["transitions"][0]["inputs"]["origin_conditioning"]["sha256"] = "0"*64
    with pytest.raises(ValueError):
        c["validate_program"](program,binding["origins"]["c0-o0"])


def test_fixed_contract_avoids_inherited_seed113_and_rejects_geometry(tmp_path):
    c = stage(tmp_path)
    config = c["make_config"]()
    c["validate_config"](config)
    assert "context" not in config and "matched_selector" not in config
    assert c["contract"]()["counts"]["adapter_steps"] == 448
    mutable = c["make_config"]()
    mutable["conditions"][0]["seed"] = 113
    with pytest.raises(ValueError):
        c["validate_config"](mutable)
    assert c["CONDITIONS"][0]["seed"] == 127
    for key,value in (("seed",113),("cpu_threads",1),("max_acts",4),("transaction_max_bytes",1<<30)):
        bad = {**config,key:value}
        with pytest.raises(ValueError):
            c["validate_config"](bad)


def test_independent_four_program_rebuild_and_exact_endpoint_joins(completed):
    auditor = runpy.run_path(str(HERE/"qualification/analyze_flux_prompt_factorial.py"))
    report = auditor["analyze_data"](completed["data"],parameter_root=completed["root"]/"external-parameter-cache",
        checkpoint=completed["checkpoint"],expected_job="tiny-factorial",production_profile=False)
    assert report["mechanics_status"] == "numeric-core-valid-custody-unassessed"
    assert report["custody_status"] == "unassessed-by-pure-core"
    assert report["mathematical_transitions"] == 448
    assert report["instrument_status"] == "independent-full-hashes-qualified"
    assert report["coverage"]["full_native_landmark_transitions"] == 80
    assert report["coverage"]["native_witness_only_transitions"] == 368
    assert not report["coverage"]["full_native_replay_retained"]
    assert report["endpoint_comparisons"]["controls"] == {"sham_png_exact":True,"sham_latent_exact":True}
    assert set(report["endpoint_comparisons"]["factorial_contrasts"]["rgb"]) == {
        "color_at_o0","color_at_o1","order_at_c0","order_at_c1","color","order","interaction"}


def test_factorial_signed_math_and_zero_denominator_are_observations():
    a = runpy.run_path(str(HERE/"qualification/analyze_flux_prompt_factorial.py"))
    values = {"c0-o0":np.array([0.]),"c0-o1":np.array([1.]),"c1-o0":np.array([2.]),"c1-o1":np.array([7.])}
    result = a["factorial"](values)
    assert result["color"].item() == 4 and result["order"].item() == 3 and result["interaction"].item() == 4
    assert [result[key].item() for key in ("color_at_o0","color_at_o1","order_at_c0","order_at_c1")] == [2,6,1,5]
    error = a["compare"](np.array([-5.]),np.array([0.]))
    assert error["signed_mean"] == -5 and error["max_abs"] == 5 and error["relative_rms"] is None
    assert error["changed_value_count"] == 1 and error["absolute_error_p99"] == 5 and error["alignment_cosine"] is None


def test_hostile_native_entry_before_first_seal_is_blocked_and_hooks_removed(tmp_path,monkeypatch):
    c = stage(tmp_path)
    model,adapter,sessions = tiny()
    path = tmp_path/"checkpoint/transformer/diffusion_pytorch_model.safetensors"
    path.parent.mkdir(parents=True)
    save_file({name:value.detach().contiguous().cpu() for name,value in model.named_parameters()},path)
    cp = c["prior"]()["CheckpointParameters"](path.parent.parent,"transformer/"+path.name,c["sha"](path),model)
    counts = dict.fromkeys(c["COUNTS"],0)
    counts["physical_batch"] = 1
    def hostile(*args,**kwargs):
        model.x_embedder(torch.zeros(1,4,4,dtype=torch.bfloat16))
        pytest.fail("native target body must not execute")
    monkeypatch.setitem(c["run"].__globals__,"predict_program",hostile)
    with pytest.raises(ValueError,match="native module forbidden"):
        c["run"](model=model,adapter=adapter,sessions=sessions,checkpoint=cp,output=tmp_path/"result",
            config=c["make_config"](),request={"job_id":"hostile","request_sha256":"fixture","source_hashes":{}},
            source={},counts=counts,production_profile=False)
    assert counts["adapter_steps"] == counts["acts"] == counts["images"] == 0
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())

"""Tiny native consumer checks for the exact production carrier interventions."""
import json
import runpy
from pathlib import Path

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from saturn_pub.adapters.flux2 import Flux2KleinAdapter

import mdb.job_support
import mdb.payload

Q = Path(__file__).parents[1] / "qualification"
branch_carriers = runpy.run_path(str(Q / "flux_concept_routes.py"))["branch_carriers"]


def test_exact_hypothesis_helper_seals_executes_and_closes_public_transaction(tmp_path):
    from mdb.forecast_transaction import verify_transaction

    helper = runpy.run_path(str(Q / "flux_concept_routes.py"))
    config = {"hypotheses": {"R-C": "later prompt reread"}, "max_wall_seconds": 840}
    tx, seal = helper["prepare_hypotheses"](tmp_path / "tx", model_identity="tiny", input_identity="input",
        source_identity="source", cuts={"base": {"0": {"parent": "a" * 64}}}, source_sha256="b" * 64, config=config)
    assert tx.declarations["frontier"]["kind"] == "architectural"
    assert tx.declarations["budgets"]["max_wall_ms"] == 840000
    assert dict(tx.prediction("route_hypotheses")) == config["hypotheses"]
    with tx:
        tx.execute("native", lambda context: {"rows": [{"id": "tiny", "image": "native.png"}], "counts": {"images": 1}})
    assert tx.report()["phase"] == "closed"
    assert verify_transaction(tmp_path / "tx", expected_seal_sha256=seal["seal_sha256"])["verified"]


def organism(dtype=torch.float32):
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(97)
        model = Flux2Transformer2DModel(in_channels=4, num_layers=1, num_single_layers=1,
            attention_head_dim=8, num_attention_heads=2, joint_attention_dim=6,
            timestep_guidance_channels=8, axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval().to(dtype)
        latent, cond = torch.randn(1, 4, 4), torch.randn(1, 3, 6)
    adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    args = dict(latent=latent, conditioning=cond, img_ids=torch.zeros(1, 4, 4), txt_ids=torch.zeros(1, 3, 4),
                timesteps=torch.tensor([1000., 500.]), sigmas=torch.tensor([1., .5, 0.]))
    base = adapter.session(**args)
    donor = adapter.session(**dict(args, conditioning=cond + 1.))
    return adapter, base, donor


@pytest.mark.parametrize("arm", ["sham", "text", "conditioning", "both"])
def test_actual_two_act_branch_isolates_carriers_and_continues_own_state(arm):
    adapter, base, donor = organism()
    parent = base.capture()
    base.continue_(1)
    donor.continue_(1)
    projected, donor_projected = base.capture(), donor.capture()
    branch, receipts = branch_carriers(base, parent, projected, donor_projected, arm, name="tiny")
    assert len(receipts) == 2
    for slot in ("conditioning", "text"):
        origin = donor_projected if arm == "both" or arm == slot else projected
        assert torch.equal(branch.read(slot), origin.payload[slot])
    assert torch.equal(branch.read("image"), projected.payload["image"])
    assert parent.payload["phase"] == "project" and parent.payload["text"] is None
    adapter.finish(branch)
    assert branch.capture().payload["step"] == 2
    if arm in {"sham", "both"}:
        reference = base if arm == "sham" else donor
        adapter.finish(reference)
        assert torch.equal(branch.read("latent"), reference.read("latent"))


def test_wrong_arm_and_cross_time_donor_rejected():
    adapter, base, donor = organism()
    parent = base.capture()
    base.continue_(1)
    projected = base.capture()
    donor.continue_(6)
    with pytest.raises(ValueError, match="unknown"):
        branch_carriers(base, parent, projected, donor.capture(), "bad", name="tiny")
    with pytest.raises(ValueError, match="timestep"):
        branch_carriers(base, parent, projected, donor.capture(), "text", name="tiny")


def test_submission_freezes_prompts_geometry_and_finite_single_lease(tmp_path, monkeypatch):
    source, staged = runpy.run_path(str(Q / "submit_flux_concept_routes.py")), {}

    def stage(path, **values):
        staged.update(values)
        assert json.loads(values["config_bytes"]) == values["request"]["config"]

    def launch(command, **values):
        config = values["request"]["config"]
        assert config["seed"] == 31 and len(config["prompts"]) == 4
        assert config["max_images"] == 4 + 2 * (3 * 3 + 1) == 24
        assert config["cut_steps"] == [0, 1] and config["max_adapter_steps"] == 3000
        assert values["preflight"] and not values["retry_on_kill"] and values["needs"] == {"cuda": True}
        assert values["request"] == staged["request"]
        assert values["request"]["scheduling_contract"]["model_loads"] == 1
        return "job-tiny"

    monkeypatch.setattr(mdb.payload, "stage_worker", stage)
    monkeypatch.setattr(mdb.payload, "worker_command", lambda name: ["python", name])
    monkeypatch.setattr(mdb.job_support, "guarded_mrun_launch", launch)
    assert source["stage"](tmp_path / "run", {"model_path": "/explicit/model", "checkpoint_sha256": "a" * 64})["job_id"] == "job-tiny"


def test_offline_reader_recomputes_actual_bfloat16_cut_without_torch_loading(tmp_path):
    import numpy as np
    from saturn_pub.store import LocalStore

    analyzer = runpy.run_path(str(Q / "analyze_flux_concept_routes.py"))
    typed = runpy.run_path(str(Q / "analyze_flux_mlp_forecast.py"))
    _, base, _ = organism(torch.bfloat16)
    cut = base.capture()
    store = LocalStore(tmp_path / "store")
    store.save(cut)
    typed["audit_store"](store.root)
    values = analyzer["slot_array"](store.root, cut.fingerprint, "latent", typed)
    assert np.array_equal(values, cut.payload["latent"].float().numpy())
    compare = analyzer["comparison"]
    target, initial = np.array([3., -2.]), np.array([1., 0.])
    result = compare(target, initial, target)
    assert result["exact_target"] and not result["exact_base"] and result["target_projection"] == 1.
    assert compare(initial, initial, initial)["target_projection"] is None
    assert compare(initial - (target - initial), initial, target)["target_projection"] == -1.
    with pytest.raises(ValueError, match="nonfinite"):
        compare(initial * float("nan"), initial, target)

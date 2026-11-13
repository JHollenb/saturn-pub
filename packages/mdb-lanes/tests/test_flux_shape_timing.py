"""Actual tiny native consumer: unread source, source switch and typed closure."""
import json
import runpy
from pathlib import Path

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from saturn_pub import Session
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

from mdb_lanes.forecast_transaction import verify_transaction
from mdb_lanes.native_navigation import NativeCutNavigator

replace = runpy.run_path(str(Path(__file__).parents[1] / "qualification/flux_shape_timing.py"))["replace_field"]
prepare_hypotheses = runpy.run_path(str(Path(__file__).parents[1] / "qualification/flux_shape_timing.py"))["prepare_hypotheses"]
result_summary = runpy.run_path(str(Path(__file__).parents[1] / "qualification/flux_shape_timing.py"))["result_summary"]
branch_carriers = runpy.run_path(str(Path(__file__).parents[1] / "qualification/flux_concept_routes.py"))["branch_carriers"]


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(147)
    model = Flux2Transformer2DModel(in_channels=4, num_layers=1, num_single_layers=1, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=6, timestep_guidance_channels=8,
        axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval()
    adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    args = dict(latent=torch.randn(1, 4, 4), conditioning=torch.randn(1, 3, 6),
        img_ids=torch.zeros(1, 4, 4), txt_ids=torch.zeros(1, 3, 4),
        timesteps=torch.tensor([1000., 500.]), sigmas=torch.tensor([1., .5, 0.]))
    base = adapter.session(**args)
    preproject = base.capture()
    base.continue_(1)
    changed = adapter.session(**dict(args, conditioning=args["conditioning"] + .4))
    changed.continue_(1)
    return adapter, preproject, base.capture(), changed.capture()


def test_source_not_consumed_until_projection_and_switch_one_reconstructs_coherent(tmp_path):
    adapter, preproject, parent, donor = fixture()
    store = LocalStore(tmp_path)
    base = Session.from_cut(adapter, preproject)
    early, _ = branch_carriers(base, preproject, parent, donor, "text", name="early", on_receipt=store.receipt)
    coherent, _ = branch_carriers(base, preproject, parent, donor, "both", name="coherent", on_receipt=store.receipt)
    for s in (early, coherent):
        NativeCutNavigator(s, max_steps=6).until(logical_step=1, phase="project")
    assert torch.equal(early.read("latent"), coherent.read("latent"))
    assert not torch.equal(early.read("conditioning"), coherent.read("conditioning"))
    replace(early, "conditioning", donor.payload["conditioning"], name="switch.C", store=store)
    assert describe(early.capture().payload) == describe(coherent.capture().payload)
    adapter.finish(early)
    adapter.finish(coherent)
    assert torch.equal(early.read("latent"), coherent.read("latent"))


def test_typed_edit_retains_other_fields_and_rejects_invalid_shape(tmp_path):
    adapter, parent, _, donor = fixture()
    store = LocalStore(tmp_path)
    session = Session.from_cut(adapter, parent)
    row = replace(session, "conditioning", donor.payload["conditioning"], name="C", store=store)
    assert row["undeclared_slots_exact"]
    restored = store.load(row["after"])
    expected = dict(parent.payload)
    expected["conditioning"] = donor.payload["conditioning"]
    assert describe(restored.payload) == describe(expected)
    before = describe(session.capture().payload)
    with pytest.raises((ValueError, RuntimeError)):
        replace(session, "conditioning", torch.zeros(1, 1, 1), name="invalid", store=store)
    assert describe(session.capture().payload) == before


def test_production_hypothesis_transaction_seals_then_executes(tmp_path):
    config = {"hypotheses": {"H1": "earlier source changes affect later output"}, "max_wall_seconds": 30}
    tx, seal = prepare_hypotheses(tmp_path / "transaction", model_identity="tiny", parent="observed-parent",
        source_hashes={"worker": "frozen-source"}, config=config)
    assert tx.prediction("shape_timing_hypotheses") == config["hypotheses"]
    with tx:
        tx.execute("native", lambda context: {"observed": True})
    verified = verify_transaction(tmp_path / "transaction", expected_seal_sha256=seal["seal_sha256"])
    assert verified["phase"] == "closed"


def test_production_summary_roundtrips_shared_publisher_and_collector(tmp_path, monkeypatch, capsys):
    import mdb_lanes.job_support

    support = runpy.run_path(str(Path(__file__).parents[1] / "qualification/discovery_upgrades.py"))
    monkeypatch.chdir(tmp_path)
    output = Path("instrument-fixture")
    output.mkdir()
    (output / "artifact.json").write_text('{"purpose":"model-free publishing check"}\n')
    request_hash = "a" * 64
    summary = result_summary(job_id="job-instrument-fixture", request_sha256=request_hash,
        counts={"images": 0}, rows={}, seal={}, seconds=0, verification={"verified": True})
    support["emit"](output, summary)
    logs = capsys.readouterr().out
    monkeypatch.setattr(mdb_lanes.job_support, "terminal_job", lambda job: (None,
        {"job_id": job, "state": "succeeded"}, logs))
    run_dir = tmp_path / "collected"
    run_dir.mkdir()
    (run_dir / "request.json").write_text(json.dumps({"request_sha256": request_hash}))
    collected = support["collect"](run_dir, "job-instrument-fixture", log_storage="gzip")
    report = json.loads((run_dir / "report.json").read_text())
    assert collected["state"] == "succeeded"
    assert report["request_sha256"] == request_hash
    assert (run_dir / "extracted/instrument-fixture/artifact.json").read_bytes() == (output / "artifact.json").read_bytes()

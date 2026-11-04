"""Tiny native source builder, reusable two-seal panel and fresh applicability."""
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
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from PIL import Image
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

from mdb.forecast_transaction import verify_transaction

HERE = Path(__file__).parents[1] / "qualification"
engine = runpy.run_path(str(HERE / "flux_single_contexts.py"))
campaign = runpy.run_path(str(HERE / "flux_single_contexts_campaign.py"))


def config():
    return {"contexts": campaign["CONTEXTS"], "branches": engine["BRANCHES"], "height": 512, "width": 512,
        "steps": 4, "selector": {"logical_step": 3, "phase": "single", "index": 0}, "cpu_threads": 1,
        "max_adapter_steps": 861, "max_acts": 39, "max_candidate_calls": 15, "max_images": 27,
        "max_vae_calls": 27, "max_wall_seconds": 900, "archive_compression_level": 1,
        "hypotheses": {"scope": "structural conditional equalities; RGB observed only"}, "scope": "tiny mechanics"}


def test_fixed_connected_contexts_survive_sorted_config_and_refuse_recipe_drift():
    value = json.loads(json.dumps(config(), sort_keys=True))
    campaign["validate_config"](value)
    assert len({item["prompt"] for item in value["contexts"]}) == 2
    assert [(item["prompt_id"], item["seed"]) for item in value["contexts"]] == [("p0", 53), ("p1", 47), ("p1", 53)]
    for field, changed in (("max_acts", 40), ("max_candidate_calls", 16), ("archive_compression_level", True),
            ("selector", {"logical_step": 2, "phase": "single", "index": 0}), ("attention_query_indices", [0])):
        with pytest.raises(ValueError):
            campaign["validate_config"]({**value, field: changed})
    value["contexts"][2]["seed"] = 47
    with pytest.raises(ValueError):
        campaign["validate_config"](value)


def tiny_session():
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(53)
        model = Flux2Transformer2DModel(in_channels=4, num_layers=5, num_single_layers=20,
            attention_head_dim=8, num_attention_heads=1, joint_attention_dim=6, timestep_guidance_channels=8,
            axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval().to(torch.bfloat16)
        adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
        session = adapter.session(latent=torch.randn(1, 4, 4, dtype=torch.bfloat16),
            conditioning=torch.randn(1, 3, 6, dtype=torch.bfloat16), img_ids=torch.zeros(1, 4, 4), txt_ids=torch.zeros(1, 3, 4),
            timesteps=torch.tensor([1000., 800., 600., 300.]), sigmas=torch.tensor([1., .8, .6, .3, 0.]))
    return model, adapter, session


def test_fresh_native_builder_and_complete_reusable_panel_keep_true_parent_and_two_seals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copyfile(HERE / "flux_single_kv.py", "flux_single_kv.py")
    model, adapter, session = tiny_session()
    initial = session.capture()
    context = campaign["CONTEXTS"][0]
    cuts, builder = engine["build_sources"](adapter=adapter, session=session, output=tmp_path / "builder",
        context=context, job_id="one-real-parent")
    assert builder["initial"] == initial.fingerprint and builder["job_id"] == "one-real-parent"
    assert builder["counts"] == {"adapter_steps": 96, "acts": 3, "receipt_emissions": 99}
    assert [len(builder["paths"][key]) for key in ("to_Z2", "to_Z3", "zero_to_single0", "own_to_single0")] == [56, 28, 6, 6]
    store = LocalStore(tmp_path / "builder" / "typed-store")
    own = store.load(builder["own_single0"])
    zero = cuts["zero"]
    assert torch.count_nonzero(zero.payload["conditioning"]) == 0
    assert torch.equal(own.payload["conditioning"], initial.payload["conditioning"])
    assert torch.equal(cuts["text"].payload["text"], own.payload["text"])
    assert not torch.equal(zero.payload["text"], cuts["text"].payload["text"])
    for key in zero.payload:
        if key != "text":
            assert describe(zero.payload[key]) == describe(cuts["text"].payload[key])
    assert builder["acts"]["transplant_text"]["parent"] == zero.fingerprint
    assert builder["acts"]["transplant_text"]["post_act"] == cuts["text"].fingerprint

    # The test consumer is deliberately synthetic; native block/Act/clock and
    # attention predictions are real, but this makes no tiny VAE parity claim.
    def test_consumer(state, **kwargs):
        digest = hashlib.sha256(state.read("latent").float().numpy().tobytes()).digest()
        return Image.fromarray(np.full((2, 2, 3), list(digest[:3]), dtype=np.uint8))
    adapter.decode = test_consumer
    pipe = SimpleNamespace(transformer=model, vae=torch.nn.Identity())
    cfg = {**config(), "context": context, "attention_query_indices": [0, 3], "max_adapter_steps": 191,
        "max_acts": 10, "max_candidate_calls": 5, "max_images": 9, "max_vae_calls": 9}
    root = tmp_path / "panel"
    result = engine["run_panel"](pipe=pipe, adapter=adapter, cuts=cuts, output=root, config=cfg,
        request={"job_id": "one-real-parent", "request_sha256": "r" * 64, "source_hashes": {}},
        source_metadata={"context_id": context["id"], "source_selection": {"kind": "fresh-native-builder"},
            "parent_counts_after_builder": {"adapter_steps": 96, "acts": 3}})
    assert result["job_id"] == "one-real-parent" and result["context_id"] == context["id"]
    assert {key: result["counts"][key] for key in ("adapter_steps", "acts", "candidate_calls", "vae_calls", "images")} == {
        "adapter_steps": 191, "acts": 10, "candidate_calls": 5, "vae_calls": 9, "images": 9}
    assert result["counts"]["model_loads"] == result["counts"]["prompt_encodings"] == result["counts"]["noise_draws"] == 0
    assert builder["counts"]["receipt_emissions"] + result["counts"]["receipt_emissions"] == 300
    assert verify_transaction(root / "transaction")["verified"]
    assert verify_transaction(root / "attention-transaction")["verified"]
    timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    seal_event = next(row for row in timeline if row["kind"] == "attention_forecast_sealed")
    assert seal_event["counts"]["candidate_calls"] == 2 and seal_event["counts"]["adapter_steps"] == seal_event["counts"]["acts"] == 0
    assert result["control_observations"]["sham_png_equals_zero"]
    assert result["control_observations"]["coherent_png_equals_text"]
    assert len(list(root.glob("*.png"))) == 9
    assert not any(module._forward_pre_hooks for module in model.modules())
    assert session.capture().fingerprint == initial.fingerprint


def test_report_json_converts_nested_transaction_mappings_without_changing_source(tmp_path):
    from types import MappingProxyType

    nested = MappingProxyType({"contexts": MappingProxyType({"p0-s53": (1, 2)})})
    campaign["save"](tmp_path / "report.json", nested)
    assert json.loads((tmp_path / "report.json").read_text()) == {"contexts": {"p0-s53": [1, 2]}}
    assert isinstance(nested["contexts"], MappingProxyType)

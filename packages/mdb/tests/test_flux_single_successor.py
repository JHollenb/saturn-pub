"""Declared single1 reader mechanics, exact source partitions and trajectories."""
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
from saturn_pub import Act, Session
from saturn_pub.store import LocalStore

from mdb.forecast_transaction import verify_transaction
from mdb.native_navigation import NativeCutNavigator

HERE = Path(__file__).parents[1]
q = runpy.run_path(str(HERE / "qualification/flux_single_kv.py"))
engine = runpy.run_path(str(HERE / "qualification/flux_single_contexts.py"))
worker = runpy.run_path(str(HERE / "qualification/flux_single_successor_campaign.py"))
tiny = runpy.run_path(str(HERE / "tests/test_flux_single_contexts.py"))


def pair(dtype=torch.bfloat16):
    model, adapter, original = tiny["tiny_session"]()
    model.to(dtype)
    # Build the tiny source directly at the actual single1 entry. The real assay
    # imports two sealed prior occurrences instead of generating these histories.
    n = NativeCutNavigator(original, max_steps=100)
    n.until(logical_step=3, phase="single", index=1)
    zero = n.capture()
    text_nav = NativeCutNavigator(Session.from_cut(adapter, zero), max_steps=2)
    text_nav.apply(Act.replace("text", zero.payload["text"] + .375, name="tiny-own-working-text"))
    return model, adapter, {"zero": zero, "text": text_nav.capture()}


def test_declared_index1_capture_equals_native_and_keeps_recipient_image_partitions():
    model, adapter, cuts = pair()
    q["validate_paired_cuts"](**cuts, single_index=1)
    with pytest.raises(ValueError, match="pre-single0"):
        q["validate_paired_cuts"](**cuts)
    native = {name: q["capture_candidate"](model.single_transformer_blocks[1], cut, single_index=1)
        for name, cut in cuts.items()}
    source = {"parent": cuts["text"].fingerprint, "text_rows": native["text"]["text_rows"], **native["text"]["observed_qkv"]}
    edited = q["capture_candidate"](model.single_transformer_blocks[1], cuts["zero"], source=source, selection="KV", single_index=1)
    assert torch.equal(edited["image"], native["text"]["image"])
    assert not torch.equal(edited["text"], native["text"]["text"])
    assert torch.equal(edited["ports"]["image_mlp_preactivation"], native["zero"]["ports"]["image_mlp_preactivation"])
    for name, cut in cuts.items():
        n = NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=2)
        n.step()
        assert torch.equal(n.session.read("image"), native[name]["image"])
        assert torch.equal(n.session.read("text"), native[name]["text"])
    for invalid in (-1, 2, True):
        with pytest.raises(ValueError):
            q["validate_paired_cuts"](**cuts, single_index=invalid)
    block = model.single_transformer_blocks[1]
    with pytest.raises(ValueError, match="geometry"):
        q["capture_candidate"](block, cuts["zero"], source={**source, "k": torch.zeros(1)}, selection="K", single_index=1)
    assert not block.attn._forward_hooks and not block.attn.mlp_act_fn._forward_hooks


def test_complete_index1_panel_two_seals_and_full_reference_trajectory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copyfile(HERE / "qualification/flux_single_kv.py", "flux_single_kv.py")
    model, adapter, cuts = pair()
    def consumer(session, **kwargs):
        digest = hashlib.sha256(session.read("latent").float().numpy().tobytes()).digest()
        return Image.fromarray(np.full((2, 2, 3), list(digest[:3]), dtype=np.uint8))
    adapter.decode = consumer  # Synthetic test consumer; no VAE parity claim.
    config = {**tiny["config"](), "single_index": 1, "capture_reference_trajectory": True,
        "attention_query_indices": [0, 3], "max_adapter_steps": 182, "max_acts": 10, "max_candidate_calls": 5,
        "max_images": 9, "max_vae_calls": 9}
    output = tmp_path / "panel"
    output.mkdir()
    engine["save"](output / "plan.json", config)
    result = engine["run_panel"](pipe=SimpleNamespace(transformer=model, vae=torch.nn.Identity()), adapter=adapter,
        cuts=cuts, output=output, config=config, request={"job_id": "same-parent", "request_sha256": "r" * 64, "source_hashes": {}},
        source_metadata={"context_id": "p1-s53", "source_selection": {"kind": "tiny-import-equivalent"}}, output_prepared=True)
    assert result["counts"]["adapter_steps"] == 182 and result["counts"]["receipt_emissions"] == 192
    assert result["counts"]["acts"] == 10 and result["counts"]["candidate_calls"] == 5
    assert result["counts"]["images"] == result["counts"]["vae_calls"] == 9
    source = json.loads((output / "source.json").read_text())
    assert source["selector_index"] == source["selector"]["index"] == source["selector"]["local_clock"] == 1
    prefixes = json.loads((output / "prefixes.json").read_text())["prefixes"]
    store = LocalStore(output / "typed-store")
    for name in ("zero", "text"):
        row = json.loads((output / "rows" / (name + ".json")).read_text())
        trajectory = row["reference_trajectory"]
        assert len(trajectory) == 21 and sum(item["retained_full_cut"] for item in trajectory) == 6
        assert trajectory[0]["cut"] == prefixes[name]["post_single1"] == row["parent"]
        assert trajectory[0]["point"]["index"] == trajectory[0]["point"]["local_clock"] == 2
        assert trajectory[-1]["cut"] == row["final_cut"] and set(trajectory[-1]["arrays"]) == {"latent"}
        assert all(set(item["arrays"]) == {"image", "text", "latent"} for item in trajectory[:-1]
            if item["point"]["phase"] != "scheduler")
        scheduler = next(item for item in trajectory if item["point"]["phase"] == "scheduler")
        assert scheduler["retained_full_cut"] and set(scheduler["arrays"]) == {"image", "text", "latent", "noise_prediction"}
        for entry in trajectory:
            for slot, record in entry["arrays"].items():
                value = np.load(output / record["full_array"]["path"], allow_pickle=False)
                wide = value.astype(np.float64)
                assert record["l2"] == pytest.approx(float(np.linalg.norm(wide.ravel())))
                assert record["rms"] == pytest.approx(float(np.sqrt(np.mean(wide * wide))))
                if entry["retained_full_cut"]:
                    np.testing.assert_array_equal(value, store.load(entry["cut"]).payload[slot].float().numpy())
    assert "reference_trajectory" not in json.loads((output / "rows/K.image.json").read_text())
    assert verify_transaction(output / "transaction")["verified"]
    assert verify_transaction(output / "attention-transaction")["verified"]
    timeline = [json.loads(line) for line in (output / "timeline.jsonl").read_text().splitlines()]
    second = next(row for row in timeline if row["kind"] == "attention_forecast_sealed")
    assert second["counts"]["candidate_calls"] == 2 and second["counts"]["adapter_steps"] == second["counts"]["acts"] == 0
    assert not any(module._forward_pre_hooks for module in model.modules())


def test_successor_config_is_fixed_and_single0_defaults_remain_explicit():
    config = {**tiny["config"](), "single_index": 1, "capture_reference_trajectory": True,
        "selector": {"logical_step": 3, "phase": "single", "index": 1}, "input_cuts": worker["INPUT_CUTS"],
        "input_store": "input-store", "source_selection": "source-selection.json", "max_adapter_steps": 182,
        "max_acts": 10, "max_candidate_calls": 5, "max_images": 9, "max_vae_calls": 9,
        "max_wall_seconds": 600, "attention_query_indices": worker["QUERIES"]}
    worker["validate_config"](json.loads(json.dumps(config, sort_keys=True)))
    for field, value in (("single_index", 0), ("capture_reference_trajectory", False), ("attention_query_indices", [0]),
            ("max_adapter_steps", 191), ("input_cuts", {"zero": "wrong", "text": "wrong"})):
        with pytest.raises(ValueError):
            worker["validate_config"]({**config, field: value})


def test_native_implementation_identity_uses_unique_basenames_and_allows_unused_encoder():
    prior = {"/old/env/transformer_flux2.py": "a" * 64, "/old/env/pipeline_flux2_klein.py": "b" * 64,
        "/old/env/modeling_qwen3.py": "c" * 64}
    current = {"/new/env/transformer_flux2.py": "a" * 64, "/new/env/pipeline_flux2_klein.py": "b" * 64}
    assert worker["validate_implementation_sources"](prior, current) == {
        "transformer_flux2.py": "a" * 64, "pipeline_flux2_klein.py": "b" * 64}
    with pytest.raises(ValueError, match="differs"):
        worker["validate_implementation_sources"](prior, {**current, "/new/env/transformer_flux2.py": "d" * 64})
    with pytest.raises(ValueError, match="ambiguous"):
        worker["validate_implementation_sources"]({**prior, "/another/transformer_flux2.py": "a" * 64}, current)

"""Native single6 operand footprint and conditional projection mechanics."""
from __future__ import annotations

import hashlib
import json
import math
import runpy
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from saturn_pub import Act, Session

from mdb_lanes.forecast_transaction import verify_transaction
from mdb_lanes.native_navigation import NativeCutNavigator

HERE = Path(__file__).parents[1]
q = runpy.run_path(str(HERE / "qualification/flux_single_contraction.py"))
worker = runpy.run_path(str(HERE / "qualification/flux_single_contraction_campaign.py"))
tiny = runpy.run_path(str(HERE / "tests/test_flux_single_contexts.py"))


def source_fixture():
    model, adapter, session = tiny["tiny_session"]()
    n = NativeCutNavigator(session, max_steps=100)
    n.until(logical_step=3, phase="single", index=3)
    low = n.capture()
    high = NativeCutNavigator(Session.from_cut(adapter, low), max_steps=2)
    change = torch.linspace(-.5, .5, low.payload["text"].shape[-1], dtype=low.payload["text"].dtype)
    high.apply(Act.replace("text", low.payload["text"] + change, name="tiny-own-history-T"))
    high.apply(Act.replace("image", low.payload["image"] + change, name="tiny-own-history-I"))
    return model, adapter, {"zero": low, "text": high.capture()}


def advance_sources(adapter, inputs):
    from mdb_lanes.forward_recording import tensor_descriptor

    cuts, expected = {}, {"expected_frontiers": {}, "expected_outputs": {}}
    for role, cut in inputs.items():
        n = NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=4)
        for _ in range(3):
            receipt = n.step()
        current = n.capture()
        cuts[role] = current
        expected["expected_frontiers"][role] = {"cut": current.fingerprint, "point": n.point, "receipt_id": receipt.fingerprint,
            "arrays": {key: tensor_descriptor(current.payload[key]) for key in ("image", "text", "latent")}}
        receipt = n.step()
        after = n.capture()
        expected["expected_outputs"][role] = {"cut": after.fingerprint, "point": n.point, "receipt_id": receipt.fingerprint,
            "arrays": {key: tensor_descriptor(after.payload[key]) for key in ("image", "text", "latent")}}
    expected.update(expected_pngs={"zero": "unknown", "text": "unknown"}, expected_final_latents={"zero": {}, "text": {}})
    return cuts, expected


def test_native_projection_operand_partitions_bias_residual_and_hook_cleanup():
    model, adapter, inputs = source_fixture()
    cuts, _ = advance_sources(adapter, inputs)
    q["validate_sources"](cuts["zero"], cuts["text"])
    block = model.single_transformer_blocks[6]
    native = {role: q["capture_candidate"](block, cut) for role, cut in cuts.items()}
    high = native["text"]
    donor = {"parent": cuts["text"].fingerprint, "concat": high["ports"]["original_concat"], "text_rows": high["text_rows"], "attention_width": high["attention_width"]}
    t, h = donor["text_rows"], donor["attention_width"]
    for name in ("A", "M", "AM"):
        value = q["capture_candidate"](block, cuts["zero"], source=donor, selection=name)
        used, original = value["ports"]["used_concat"], value["ports"]["original_concat"]
        assert torch.equal(used[:, :t], original[:, :t])
        for key, part in (("A", slice(None, h)), ("M", slice(h, None))):
            selected = donor["concat"] if key in name else original
            assert torch.equal(used[:, t:, part], selected[:, t:, part])
        assert value["equation_exact"] and torch.equal(value["text"], native["zero"]["text"])
        if name == "AM":
            assert torch.equal(value["ports"]["projection"][:, t:], high["ports"]["projection"][:, t:])
    for role, cut in cuts.items():
        n = NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=1)
        n.step()
        assert torch.equal(n.session.read("image"), native[role]["image"])
        assert torch.equal(n.session.read("text"), native[role]["text"])
    with pytest.raises(ValueError, match="geometry"):
        q["capture_candidate"](block, cuts["zero"], source={**donor, "concat": torch.zeros(1)}, selection="A")
    assert not block.attn.to_out._forward_hooks and not block.attn.to_out._forward_pre_hooks
    with pytest.raises(ValueError, match="clock"):
        q["capture_candidate"](block, inputs["zero"])


def test_fp64_projection_samples_match_scalar_equation_and_bf16_primitive_casts():
    rng = np.random.default_rng(68)
    low = {"concat": rng.normal(size=(1, 5, 7)).astype(np.float32), "residual": rng.normal(size=(1, 5, 3)).astype(np.float32),
        "gate": rng.normal(size=(1, 1, 3)).astype(np.float32)}
    high = {"concat": rng.normal(size=(1, 5, 7)).astype(np.float32)}
    weight, bias = rng.normal(size=(2, 7)).astype(np.float32), rng.normal(size=2).astype(np.float32)
    frozen = low["concat"].copy()
    result = q["forecast_samples"](low, high, weight, bias, query_indices=[0, 2], output_indices=[0, 2], text_rows=2, attention_width=3)
    def cast(value):
        return torch.from_numpy(np.array(value, copy=True)).float().to(torch.bfloat16).float().numpy()
    for name, record in result["cases"].items():
        for query, position in enumerate((2, 4)):
            operands = [float(high["concat"][0, position, column] if (column < 3 and "A" in name) or (column >= 3 and "M" in name)
                else low["concat"][0, position, column]) for column in range(7)]
            expected = [math.fsum(value * float(coefficient) for value, coefficient in zip(operands, row, strict=True)) + float(b) for row, b in zip(weight, bias, strict=True)]
            np.testing.assert_allclose(record["projection_fp64"][0, query], expected, rtol=1e-14, atol=1e-14)
        p = cast(record["projection_fp64"])
        product = cast(result["recipient_gate_samples"] * p)
        updated = cast(result["recipient_residual_samples"] + product)
        np.testing.assert_array_equal(record["projection_bf16"], p)
        np.testing.assert_array_equal(record["product_bf16"], product)
        np.testing.assert_array_equal(record["updated_bf16"], updated)
        assert not record["updated_fp64"].flags.writeable
    np.testing.assert_array_equal(low["concat"], frozen)
    with pytest.raises(ValueError):
        q["forecast_samples"](low, high, weight, bias, query_indices=[0, 0], output_indices=[0, 2], text_rows=2, attention_width=3)
    with pytest.raises(ValueError):
        q["forecast_samples"](low, high, weight.astype(np.float64), bias, query_indices=[0, 2], output_indices=[0, 2], text_rows=2, attention_width=3)


def test_complete_tiny_panel_census_two_seals_sample_json_and_typed_suffix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copyfile(HERE / "qualification/flux_single_contraction.py", "flux_single_contraction.py")
    model, adapter, inputs = source_fixture()
    _, selection = advance_sources(adapter, inputs)
    def consumer(session, **kwargs):
        digest = hashlib.sha256(session.read("latent").float().numpy().tobytes()).digest()
        return Image.fromarray(np.full((2, 2, 3), list(digest[:3]), dtype=np.uint8))
    adapter.decode = consumer  # Synthetic image consumer, no VAE qualification.
    cfg = {"branches": q["BRANCHES"], "max_adapter_steps": 143, "max_acts": 10, "max_candidate_calls": 5,
        "max_images": 9, "max_vae_calls": 9, "max_wall_seconds": 300, "query_indices": [0, 3], "output_dims": [0, 7],
        "hypotheses": {"law": "conditional projection"}, "scope": "tiny native mechanics"}
    output = tmp_path / "panel"
    result = worker["run_panel"](pipe=SimpleNamespace(transformer=model, vae=torch.nn.Identity()), adapter=adapter, inputs=inputs,
        output=output, config=cfg, request={"job_id": "same-parent", "request_sha256": "r" * 64, "source_hashes": {}}, metadata={}, selection=selection)
    assert {key: result["counts"][key] for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls", "receipt_emissions")} == {
        "adapter_steps": 143, "acts": 10, "candidate_calls": 5, "images": 9, "vae_calls": 9, "receipt_emissions": 153}
    timeline = [json.loads(line) for line in (output / "timeline.jsonl").read_text().splitlines()]
    main = next(row for row in timeline if row["kind"] == "hypotheses_sealed")
    numeric = next(row for row in timeline if row["kind"] == "projection_forecast_sealed")
    assert main["counts"]["adapter_steps"] == numeric["counts"]["adapter_steps"] == 6
    assert main["counts"]["candidate_calls"] == 0 and numeric["counts"]["candidate_calls"] == 2
    assert main["counts"]["acts"] == numeric["counts"]["acts"] == 0
    assert verify_transaction(output / "transaction")["verified"] and verify_transaction(output / "projection-transaction")["verified"]
    assert result["control_observations"]["sham_png_equals_zero"] and result["control_observations"]["coherent_png_equals_text"]
    assert result["control_observations"]["AM_projection_image_exact_text_candidate"]
    projection = json.loads((output / "projection-forecast.json").read_text())
    for sample in projection["samples"].values():
        for key in ("projection", "updated"):
            assert np.load(output / sample[key]["path"], allow_pickle=False).shape == (1, 2, 2)
    assert len(list(output.glob("*.png"))) == 9
    assert not any(module._forward_pre_hooks or module._forward_hooks for module in model.modules())


def test_fixed_config_and_native_source_identity_mismatch_refuse():
    cfg = {"single_index": 6, "selector": {"logical_step": 3, "phase": "single", "index": 6}, "input_store": "input-store",
        "source_selection": "source-selection.json", "input_cuts": {"zero": "z", "text": "t"}, "query_indices": worker["QUERIES"],
        "output_dims": worker["OUTPUT_DIMS"], "height": 512, "width": 512, "steps": 4, "branches": q["BRANCHES"],
        "max_adapter_steps": 143, "max_acts": 10, "max_candidate_calls": 5, "max_images": 9, "max_vae_calls": 9,
        "max_wall_seconds": 540, "cpu_threads": 4, "archive_compression_level": 1}
    worker["validate_config"](json.loads(json.dumps(cfg, sort_keys=True)))
    for field, value in (("single_index", 5), ("output_dims", [0]), ("query_indices", [0]), ("max_adapter_steps", 142)):
        with pytest.raises(ValueError):
            worker["validate_config"]({**cfg, field: value})
    assert worker["validate_native_sources"]({"/old/model.py": "a", "/old/encoder.py": "b"}, {"/new/model.py": "a"}) == {"model.py": "a"}
    with pytest.raises(ValueError):
        worker["validate_native_sources"]({"/old/model.py": "a"}, {"/new/model.py": "b"})

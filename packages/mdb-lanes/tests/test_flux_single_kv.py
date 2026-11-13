"""Tiny native parallel ports, typed routes, and conditional attention forecast."""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import math
import runpy
import shutil
import tarfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from diffusers.models.transformers import transformer_flux2
from saturn_pub import Act, Session
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
from mdb_lanes.native_navigation import NativeCutNavigator

HERE = Path(__file__).parents[1] / "qualification"
q = runpy.run_path(str(HERE / "flux_single_kv.py"))
campaign = runpy.run_path(str(HERE / "flux_single_kv_campaign.py"))


def specimen(dtype=torch.float32):
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(117)
        model = Flux2Transformer2DModel(in_channels=4, num_layers=1, num_single_layers=2,
            attention_head_dim=8, num_attention_heads=2, joint_attention_dim=6, timestep_guidance_channels=8,
            axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval().to(dtype)
        adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
        session = adapter.session(latent=torch.randn(1, 4, 4, dtype=dtype), conditioning=torch.zeros(1, 3, 6, dtype=dtype),
            img_ids=torch.zeros(1, 4, 4), txt_ids=torch.zeros(1, 3, 4),
            timesteps=torch.tensor([1000., 800., 600., 300.]), sigmas=torch.tensor([1., .8, .6, .3, 0.]))
    zero = NativeCutNavigator(session, max_steps=32)
    zero.until(logical_step=3, phase="single", index=0)
    cut = zero.capture()
    changed = NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=8)
    changed.apply(Act.replace("text", cut.payload["text"] + .375, name="own-history-working-T"))
    return model, adapter, cut, changed.capture()


def candidate(model, cut, source=None, selection="native"):
    return q["capture_candidate"](model.single_transformer_blocks[0], cut, source=source, selection=selection)


def source_from(value, cut):
    return {"parent": cut.fingerprint, "text_rows": value["text_rows"], **value["observed_qkv"]}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_independent_k_v_selection_native_parallel_mlp_and_residual_closure(dtype):
    model, adapter, zero, text = specimen(dtype)
    q["validate_paired_cuts"](zero, text)
    original = transformer_flux2.dispatch_attention_fn
    baseline, native_text = candidate(model, zero), candidate(model, text)
    edits = {name: candidate(model, zero, source_from(native_text, text), name) for name in ("K", "V", "KV")}
    count = baseline["text_rows"]
    for selection, value in edits.items():
        assert torch.equal(value["used_qkv"]["q"], baseline["observed_qkv"]["q"])
        for key in ("k", "v"):
            selected = native_text if key.upper() in selection else baseline
            assert torch.equal(value["used_qkv"][key][:, :count], selected["observed_qkv"][key][:, :count])
            assert torch.equal(value["used_qkv"][key][:, count:], baseline["observed_qkv"][key][:, count:])
        assert all(value["partition_controls"].values())
    for value in (baseline, native_text, *edits.values()):
        assert value["equation_exact"]
        assert torch.equal(value["ports"]["residual"] + value["ports"]["gate"] * value["ports"]["fused_projection"], value["ports"]["updated"])
        for key in ("image_mlp_preactivation", "image_mlp_activation"):
            assert torch.equal(value["ports"][key], baseline["ports"][key])
    assert torch.equal(edits["KV"]["image"], native_text["image"])
    assert not torch.equal(edits["KV"]["text"], native_text["text"])
    for cut, predicted in ((zero, baseline), (text, native_text)):
        native = NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=2)
        native.step()
        assert torch.equal(native.session.read("image"), predicted["image"])
        assert torch.equal(native.session.read("text"), predicted["text"])
    assert transformer_flux2.dispatch_attention_fn is original


def test_dispatch_and_mlp_hooks_restore_on_invalid_source():
    model, _, zero, text = specimen()
    original = transformer_flux2.dispatch_attention_fn
    with pytest.raises(ValueError, match="geometry"):
        candidate(model, zero, {"parent": text.fingerprint, "text_rows": 3, "k": torch.zeros(1), "v": torch.zeros(1)}, "K")
    assert transformer_flux2.dispatch_attention_fn is original
    block = model.single_transformer_blocks[0]
    assert not block.attn._forward_hooks and not block.attn.mlp_act_fn._forward_hooks
    assert not block.attn.mlp_act_fn._forward_pre_hooks


def test_common_post_single_parent_stride_closure_and_coherent_future(tmp_path):
    _, adapter, zero, text = specimen(torch.bfloat16)
    fronts = {}
    for name, cut in (("zero", zero), ("text", text)):
        n = NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=8)
        n.step()
        fronts[name] = n.capture()
    mixed = NativeCutNavigator(Session.from_cut(adapter, fronts["zero"]), max_steps=8)
    store = LocalStore(tmp_path)
    actions = []
    for slot in ("image", "text"):
        reference = fronts["zero"].payload[slot]
        actions.append(q["install_port"](mixed, slot, fronts["text"].payload[slot].contiguous(), name="coherent-" + slot, store=store))
        assert mixed.session.read(slot).stride() == reference.stride()
    expected = {**fronts["zero"].payload, "image": mixed.session.read("image"), "text": mixed.session.read("text")}
    assert describe(mixed.capture().payload) == describe(expected)
    assert actions[0]["parent"] == fronts["zero"].fingerprint and actions[1]["parent"] == actions[0]["post_act"]
    own = NativeCutNavigator(Session.from_cut(adapter, fronts["text"]), max_steps=8)
    mixed.until(logical_step=4, phase="project")
    own.until(logical_step=4, phase="project")
    assert torch.equal(mixed.session.read("latent"), own.session.read("latent"))
    assert torch.equal(mixed.session.read("conditioning"), zero.payload["conditioning"])


def test_equal_text_and_wrong_write_clock_refused(tmp_path):
    _, adapter, zero, _ = specimen()
    with pytest.raises(ValueError, match="actually differ"):
        q["validate_paired_cuts"](zero, zero)
    n = NativeCutNavigator(Session.from_cut(adapter, zero), max_steps=2)
    before = n.capture().fingerprint
    with pytest.raises(ValueError, match="post-single0"):
        q["install_port"](n, "image", zero.payload["image"], name="early", store=LocalStore(tmp_path))
    assert n.capture().fingerprint == before


def test_numpy_attention_forecast_matches_scalar_softmax_and_cast_without_inputs_mutation():
    rng = np.random.default_rng(22)
    zero = {key: rng.normal(size=(1, 5, 2, 4)).astype(np.float32) for key in ("q", "k", "v")}
    text = {key: value.copy() for key, value in zero.items()}
    for value in text.values():
        value[:, :2] += .4
    copies = {key: value.copy() for key, value in zero.items()}
    result = q["forecast_attention_samples"](zero, text, text_rows=2, query_indices=[0, 2])
    for selection, record in result["cases"].items():
        for qi, position in enumerate((2, 4)):
            for head in range(2):
                keys = [text["k"][0, row, head] if "K" in selection and row < 2 else zero["k"][0, row, head] for row in range(5)]
                vals = [text["v"][0, row, head] if "V" in selection and row < 2 else zero["v"][0, row, head] for row in range(5)]
                scores = [math.fsum(float(a) * float(b) for a, b in zip(zero["q"][0, position, head], key, strict=True)) / 2. for key in keys]
                weights = [math.exp(value - max(scores)) for value in scores]
                weights = [value / math.fsum(weights) for value in weights]
                expected = [math.fsum(weight * float(value[column]) for weight, value in zip(weights, vals, strict=True)) for column in range(4)]
                np.testing.assert_allclose(record["attention"][0, qi, head], expected, rtol=1e-14, atol=1e-14)
                assert record["text_mass"][0, head, qi] == pytest.approx(math.fsum(weights[:2]), abs=1e-15)
        expected_bf16 = torch.from_numpy(record["attention"].copy()).float().to(torch.bfloat16).float().numpy()
        np.testing.assert_array_equal(record["attention_bf16"], expected_bf16)
        assert not record["attention"].flags.writeable and not record["text_mass"].flags.writeable
    assert result["cases"]["K"]["role"] == "novel conditional forecast"
    assert result["cases"]["KV"]["role"] == "native-text represented consistency control"
    for key in zero:
        np.testing.assert_array_equal(zero[key], copies[key])


def test_nested_observed_source_forecast_seals_before_three_edited_native_calls(tmp_path):
    model, _, zero, text = specimen(torch.bfloat16)
    calls = []
    with ForecastTransaction(tmp_path / "outer", model_identity="tiny", input_identity="pair", source_identity="source",
                             frontier={"kind": "architectural"}) as outer:
        outer.predict("hypotheses", lambda _: {"predictions": {"scope": "conditional"}})
        outer.seal()

        def execute(_):
            base, donor = candidate(model, zero), candidate(model, text)
            calls.extend(["zero", "text"])
            arrays = [{key: value.float().numpy().copy() for key, value in result["observed_qkv"].items()} for result in (base, donor)]
            with ForecastTransaction(tmp_path / "attention", model_identity="tiny", input_identity="observedQKV", source_identity="source",
                                     frontier={"kind": "architectural"}) as secondary:
                secondary.frontier("source_binding", {"observed_calls": len(calls), "edited_calls": 0})
                secondary.predict("sampled_attention", lambda context: {"predictions": q["forecast_attention_samples"](
                    *arrays, text_rows=3, query_indices=[0, 3])})
                seal = secondary.seal()
                assert calls == ["zero", "text"]

                def edited(context):
                    actual = {}
                    for selection in ("K", "V", "KV"):
                        value = candidate(model, zero, source_from(donor, text), selection)
                        calls.append(selection)
                        actual[selection] = value["attention_dispatch_output"][:, [3, 6]].float().numpy().copy()
                    assert context.predictions["sampled_attention"]["cases"]["K"]["attention"].shape == actual["K"].shape
                    return {"samples": actual}

                secondary.execute("edited_candidates", edited)
            assert verify_transaction(secondary.root, expected_seal_sha256=seal["seal_sha256"])["verified"]
            return {"calls": calls}

        outer.execute("native", execute)
    assert calls == ["zero", "text", "K", "V", "KV"] and verify_transaction(outer.root)["verified"]


def test_fixed_panel_census_and_query_contract():
    config = {"branches": campaign["BRANCHES"], "input_cuts": {"zero": "a" * 64, "text": "b" * 64},
        "input_store": "input-store", "source_selection": "source-selection.json", "steps": 4,
        "height": 512, "width": 512, "seed": 47, "max_adapter_steps": 240, "max_acts": 10,
        "max_candidate_calls": 5, "max_images": 9, "max_vae_calls": 9, "max_wall_seconds": 600, "archive_compression_level": 1}
    campaign["validate_config"](config)
    assert 2 + 9 * 21 == 191
    assert sum(int(row["image_source"] is not None) + int(row["text_source"] is not None)
               for row in config["branches"] if row["kind"] != "reference") == 10
    for change in ({"attention_query_indices": [0]}, {"max_candidate_calls": 6}, {"max_acts": 11}, {"branches": []}):
        with pytest.raises(ValueError):
            campaign["validate_config"]({**config, **change})


def test_final_emit_serializes_nested_public_observation_and_preserves_completed_archive(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    shutil.copyfile(HERE / "discovery_upgrades.py", tmp_path / "campaign_support.py")
    output = Path("result")
    with ForecastTransaction(output / "transaction", model_identity="tiny", input_identity="input", source_identity="source",
                             frontier={"kind": "architectural"}) as tx:
        tx.predict("conditional", lambda _: {"predictions": {"scope": "observed-QKV"}})
        seal = tx.seal()
        tx.execute("native", lambda _: {"comparisons": {"K": {"comparisons": {"fp64": {"rms": .125}}}}})
    observation = tx.observation("native")
    nested = observation["comparisons"]["K"]["comparisons"]
    assert isinstance(nested, Mapping) and not isinstance(nested, dict)
    summary = {"counts": {"adapter_steps": 191, "candidate_calls": 5, "acts": 10, "images": 9},
               "numerical_attention_forecast": {"comparisons": observation["comparisons"]}}
    with pytest.raises(TypeError, match="mappingproxy"):
        json.dumps(summary)  # Exact former failure, after numerical execution.
    campaign["save"](output / "transaction-report.json", tx.report())
    campaign["save"](output / "summary.json", summary)
    before = (output / "summary.json").read_bytes()
    campaign["emit_report"](output, summary, compression_level=1)
    support = runpy.run_path(str(HERE / "discovery_upgrades.py"))
    lines = capsys.readouterr().out.splitlines()
    report = json.loads(gzip.decompress(base64.b64decode(next(line[len(support["REPORT"]):]
        for line in lines if line.startswith(support["REPORT"])))))
    raw = base64.b64decode(next(line[len(support["ARCHIVE"]):] for line in lines if line.startswith(support["ARCHIVE"])))
    assert report == {**json.loads(before), "archive_sha256": hashlib.sha256(raw).hexdigest()}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        assert archive.extractfile("result/summary.json").read() == before
        assert json.loads(archive.extractfile("result/transaction-report.json").read())["phase"] == "closed"
    assert (output / "summary.json").read_bytes() == before
    assert isinstance(nested, Mapping) and not isinstance(nested, dict)
    assert verify_transaction(tx.root, expected_seal_sha256=seal["seal_sha256"])["verified"]

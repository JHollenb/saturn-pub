"""Tiny installed CPU prefix/seal/consumer checks, never trained model weights."""
from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers.models.transformers import transformer_flux2 as native

from saturn_pub.core import StateCut
from saturn_pub.store import LocalStore

HERE = Path(__file__).parents[1]
PATH = HERE / "qualification/flux_image_reader_forecast.py"
C = runpy.run_path(str(PATH))
OLD = runpy.run_path(str(HERE / "tests/test_flux_image_tail_microscope.py"))


def fixture(root):
    config, parents, weights = OLD["fixture"](root)
    p = parents["sham"].payload
    delta = torch.tensor([.25, -.5, .125, -.25, .5, -.125, .25, -.5], dtype=torch.bfloat16)
    edit = StateCut(parents["sham"].model_identity, parents["sham"].execution, parents["sham"].boundary,
                    parents["sham"].fingerprint, {**p, "image": p["image"] + delta})
    parents["edit"] = edit
    LocalStore(Path(config["input_root"]) / "typed-store").save(edit)
    config["inputs"]["edit"]["cut"] = edit.fingerprint
    config.update(midpoint_recipe=C["MIDPOINT_RECIPE"].copy(), fixed_head_control=True, transaction_max_bytes=16 << 20)
    config["helper_source_sha256"][C["MICROSCOPE"]] = OLD["C"]["sha"](HERE / "qualification" / C["MICROSCOPE"])
    return config, parents, weights


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("reader")
    config, parents, weights = fixture(root)
    events = []
    summary = C["run"](config, progress=lambda value: events.append(value))
    return config, parents, weights, summary, events


def test_two_prefixes_never_call_original_sdpa_or_enter_tail(tmp_path):
    config, parents, _ = fixture(tmp_path)
    block = native.Flux2TransformerBlock(**config["module_config"]).to(torch.bfloat16).eval()
    original, calls = native.dispatch_attention_fn, []
    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    native.dispatch_attention_fn = spy
    counts = dict.fromkeys(("block_entries", "dispatch_boundaries", "linear", "layernorm", "rmsnorm", "native_attention", "completed_blocks", "output_projection", "mlp"), 0)
    try:
        with torch.inference_mode():
            prefix = C["capture_qkv_prefix"](block, parents["sham"].payload, native, counts=counts)
        assert not calls and native.dispatch_attention_fn is spy
        assert counts == dict(block_entries=1, dispatch_boundaries=1, linear=6, layernorm=2, rmsnorm=4,
                              native_attention=0, completed_blocks=0, output_projection=0, mlp=0)
        assert prefix["values"]["q"].shape == (1, 7, 2, 4)
        assert not prefix["native_complete_outputs_computed"]
    finally:
        native.dispatch_attention_fn = original


def test_input_and_forecast_seals_precede_their_native_boundaries(completed):
    config, _, _, summary, events = completed
    data = Path(config["output"])
    kinds = [row["kind"] for row in events]
    assert kinds == ["input_sealed", "prefix_stopped", "prefix_stopped", "image_cohort_sealed", "target_completed", "target_completed"]
    assert all(row["target_entries"] == 0 for row in events[:4])
    assert summary["complete_native_targets"] == 2
    assert summary["prefix_counts"] == dict(block_entries=2, dispatch_boundaries=2, linear=12, layernorm=4, rmsnorm=8,
                                            native_attention=0, completed_blocks=0, output_projection=0, mlp=0)
    assert summary["forecast_primitive_counts"] == dict(native_attention=2, linear=9, layernorm=3, swiglu=3)
    assert summary["target_counts"] == dict(native_blocks=2, native_attention=2, linear=24, layernorm=8, rmsnorm=8, swiglu=4)
    assert summary["vae_calls"] == summary["encodes"] == summary["noise_draws"] == summary["suffix_steps"] == 0
    seal = json.loads((data / "input-seal.json").read_text())
    assert seal["target_entries"] == seal["prefix_entries"] == 0
    assert seal["changed_slots"] == ["image"] and seal["novel_input"]
    from mdb.forecast_transaction import verify_transaction
    assert verify_transaction(data / "transaction", expected_seal_sha256=summary["forecast_seal_sha256"])["verified"] is True
    assert summary["transaction_array_storage"] == "files-v1"
    assert list((data / "transaction").rglob("*.npy")) and not list((data / "transaction").rglob("*.bin"))


def test_midpoint_bits_layout_carry_and_original_source_preserved(completed):
    config, parents, _, summary, _ = completed
    m, core = C["helpers"](config)
    data = Path(config["output"])
    seal = json.loads((data / "input-seal.json").read_text())
    expected = C["midpoint_image"](parents["sham"].payload["image"], parents["edit"].payload["image"], parents["sham"].payload["image"])
    decoded = core["H"]["load_array"](data, seal["inputs"]["midpoint"]["image"])
    assert np.array_equal(decoded, expected.float().numpy())
    assert summary["input_comparisons"]["midpoint_minus_sham"]["changed_bf16_values"] > 0
    assert summary["input_comparisons"]["midpoint_minus_edit"]["changed_bf16_values"] > 0
    assert expected.stride() == parents["sham"].payload["image"].stride()
    for cut in parents.values():
        cut.verify()
    assert m["sha"](data / "retrieve_weights.py") == m["PARAMETER_RETRIEVER_SHA256"]
    assert summary["immutable_sources_and_parameters_unchanged"]


def test_forecasts_match_tiny_native_and_wrong_head_is_an_observation(completed):
    config, _, _, _, _ = completed
    data = Path(config["output"])
    for role in ("sham", "midpoint"):
        target = json.loads((data / "targets" / (role + ".json")).read_text())
        assert target["forecast_minus_native"]["changed_values"] == 0
        assert all(row["changed_values"] == 0 for row in target["prefix_qkv_replay"].values())
        if role == "midpoint":
            assert target["wrong_fixed_head_minus_prediction"]["changed_values"] > 0
        else:
            assert target["historical_replay"]["image"]["exact_value_identity"]


@pytest.mark.parametrize("mode", ("exception", "foreign-sentinel"))
def test_prefix_cleanup_does_not_swallow_unrelated_exceptions(tmp_path, mode):
    config, parents, _ = fixture(tmp_path)
    block = native.Flux2TransformerBlock(**config["module_config"]).to(torch.bfloat16).eval()
    original = native.dispatch_attention_fn
    error = RuntimeError("unrelated") if mode == "exception" else C["_BeforeSDPA"](object())
    def fail(module, args):
        raise error
    handle = block.norm1.register_forward_pre_hook(fail)
    before = {name: (len(module._forward_hooks), len(module._forward_pre_hooks)) for name, module in block.named_modules()}
    counts = dict.fromkeys(("block_entries", "dispatch_boundaries", "linear", "layernorm", "rmsnorm", "native_attention", "completed_blocks", "output_projection", "mlp"), 0)
    with pytest.raises(type(error)), torch.inference_mode():
        C["capture_qkv_prefix"](block, parents["sham"].payload, native, counts=counts)
    assert native.dispatch_attention_fn is original
    assert before == {name: (len(module._forward_hooks), len(module._forward_pre_hooks)) for name, module in block.named_modules()}
    handle.remove()


@pytest.mark.parametrize("caught", (False, True))
def test_forecast_guard_rejects_full_block_and_restores_on_failure(tmp_path, monkeypatch, caught):
    config, parents, weights = fixture(tmp_path)
    old_dispatch = native.dispatch_attention_fn
    created = []
    original_forward = native.Flux2TransformerBlock.forward
    def track(self, *args, **kwargs):
        if self not in created:
            created.append(self)
        return original_forward(self, *args, **kwargs)
    monkeypatch.setattr(native.Flux2TransformerBlock, "forward", track)
    calls = []
    def bad_dispatch(*args, **kwargs):
        calls.append(True)
        # Prefix capture replaces this function without invoking it; this first
        # call is the forecast primitive, at which point a full block is denied.
        p = parents["sham"].payload
        try:
            created[0](p["image"], p["text"], p["context"]["img_mod"], p["context"]["txt_mod"], p["context"]["rope"])
        except ValueError:
            if not caught:
                raise
        return old_dispatch(*args, **kwargs)
    monkeypatch.setattr(native, "dispatch_attention_fn", bad_dispatch)
    with pytest.raises(ValueError, match="forecast phase|attempted during forecast"):
        C["run"](config, parents=parents, weights=weights)
    assert len(calls) == (2 if caught else 1)
    assert native.dispatch_attention_fn is bad_dispatch
    assert not (Path(config["output"]) / "targets").exists()


def test_corrupted_seal_prevents_native_release_and_keeps_prefix_evidence(tmp_path):
    config, _, _ = fixture(tmp_path)
    completed_blocks = []
    handle = torch.nn.modules.module.register_module_forward_hook(lambda module, args, result: completed_blocks.append(True) if isinstance(module, native.Flux2TransformerBlock) else None)
    def tamper(event):
        if event["kind"] == "image_cohort_sealed":
            (Path(config["output"]) / "transaction/seal.json").write_text("{}")
    try:
        with pytest.raises(Exception, match="seal changed"):
            C["run"](config, progress=tamper)
    finally:
        handle.remove()
    assert not completed_blocks
    assert (Path(config["output"]) / "prefixes/midpoint.json").is_file()


def test_rounding_collapse_is_reported_without_tuning_or_abort(tmp_path):
    config, parents, weights = fixture(tmp_path)
    parents["edit"] = parents["sham"]
    config["inputs"]["edit"]["cut"] = parents["sham"].fingerprint
    config["inputs"]["sham"]["historical_image_descriptor"]["bf16_le_sha256"] = "0" * 64
    result = C["run"](config, parents=parents, weights=weights)
    assert not result["novel_input"] and result["complete_native_targets"] == 2
    assert result["input_comparisons"]["midpoint_minus_sham"]["exact_bf16_equal"]
    target = json.loads((Path(config["output"]) / "targets/sham.json").read_text())
    assert not target["historical_replay"]["image"]["exact_value_identity"]


def test_wrong_recipe_or_source_policy_refused_before_observation(tmp_path):
    config, _, _ = fixture(tmp_path)
    config["midpoint_recipe"]["dose"] = .75
    with pytest.raises(ValueError, match="recipe"):
        C["validate_config"](config)
    config["midpoint_recipe"] = C["MIDPOINT_RECIPE"].copy()
    config["source_reduction_policy"] = False
    with pytest.raises(ValueError, match="policy"):
        C["validate_config"](config)


def test_worker_cli_and_exported_finalizer_reach_metadata_only_marker(tmp_path):
    config, _, _ = fixture(tmp_path)
    config_path = tmp_path / "config.json"
    OLD["C"]["save"](config_path, config)
    result = subprocess.run([sys.executable, "-I", "-B", str(PATH), "--config", str(config_path)],
                            cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    marker = json.loads(next(line.split(" ", 1)[1] for line in result.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")))
    assert marker["files_manifest_sha256"] == OLD["C"]["sha"](Path(config["output"]) / "files.json")
    assert marker["expanded_bytes"] < 16 << 20
    assert "base64" not in marker
    with pytest.raises(FileExistsError):
        C["finalize_directory"](config["output"])

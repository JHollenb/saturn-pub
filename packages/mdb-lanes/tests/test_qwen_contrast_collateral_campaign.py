"""Small installed native mechanics and independent contrast/readout equations."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import inspect
import json
import runpy
import shutil
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

C = runpy.run_path(str(Path(__file__).parents[1] / "qualification/qwen_contrast_collateral_campaign.py"))
A = runpy.run_path(str(Path(__file__).parents[1] / "qualification/qwen_contrast_components_math.py"))


def tiny_model():
    torch.manual_seed(17); torch.set_num_threads(1)
    config = Qwen2Config(vocab_size=31, hidden_size=8, intermediate_size=12, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).eval().requires_grad_(False)


def test_fixed_source_protocol_has_fresh_contexts_signed_axes_and_bounded_geometry():
    config = C["make_config"]({}); counts = C["validate_config"](config)
    assert [row["id"] for row in config["prompts"]] == ["card", "sequence", "code", "translation"]
    assert config["prompts"][0]["text"] == "A card has the word blue written in black ink. The word on the card is"
    assert len(config["cells"]) == 9 and config["future_cells"] == C["FUTURES"]
    assert all(row["carrying"]*row["neutral"] == 0 for row in config["cells"])
    assert counts["component_readout_targets"] == 36 and counts["native_physical_entries_max"] == 244
    assert counts["autonomous_cached_calls_max"] == 160 and len(C["dependencies"]()) == 5
    config["cells"][1]["carrying"] = -.5
    with pytest.raises(ValueError, match="fixed contrast protocol"):
        C["validate_config"](config)


def test_minimum_norm_contrast_dual_and_neutral_share_norm_without_unit_gain_claim():
    rng = np.random.default_rng(123); P = rng.normal(size=(7, 2)).astype(np.float32); c = rng.normal(size=7)
    result = A["decompose"](dict(plus=P[:, 0], minus=P[:, 1]), c)
    a = P.astype(np.float64).T @ c; G = P.astype(np.float64).T @ P
    coefficients = np.linalg.solve(G, a); expected = P @ (coefficients/(a @ coefficients))
    np.testing.assert_allclose(result["unit_contrast_q1"], expected, atol=1e-14)
    assert result["unit_q1_contrast"] == pytest.approx(1.)
    assert result["ideal_neutral_dot"] == pytest.approx(0., abs=1e-14)
    assert result["ideal_orthogonality"] == pytest.approx(0., abs=1e-14)
    assert result["carrying_gain"] != pytest.approx(1.)
    for axis in ("carrying", "neutral"):
        assert np.linalg.norm(result["ideal"][axis]) == pytest.approx(result["common_norm"])
        assert result["realized"][axis].dtype == np.float32 and not result["realized"][axis].flags.writeable
    direction = P.astype(np.float64) @ np.array([-a[1], a[0]])
    assert np.linalg.norm(expected+direction) >= np.linalg.norm(expected)


def test_neutral_can_change_vocabulary_winner_while_pairwise_numerator_is_preserved():
    head = A["readonly"]([[1., 0., 0.], [0., 0., 0.], [0., 2., 0.], [-1., 0., 0.]], np.float32)
    P = np.array([[1, 2], [2, -1], [0, 0]], dtype=np.float32); c = np.array([1., 0., 0.])
    axes = A["decompose"](dict(plus=P[:, 0], minus=P[:, 1]), c)
    result = A["readout"](np.array([.21, .1, .3]), axes, np.ones(3), head, 1e-6, C["CELLS"], [0, 1], collateral_ids=dict(other=2))
    assert result["rows"]["baseline"]["greedy_token"] == 0
    assert result["rows"]["neutral-p1"]["greedy_token"] == 2
    for row in result["rows"].values():
        x = np.array([.21, .1, .3])+row["carrying"]*axes["realized"]["carrying"]+row["neutral"]*axes["realized"]["neutral"]
        direct = np.array([sum(float(w)*float(v) for w, v in zip(h, x, strict=True)) for h in head])/np.sqrt(sum(float(v)**2 for v in x)/3+1e-6)
        np.testing.assert_allclose(row["logits"], direct, atol=2e-15)
        assert row["coefficient_greedy_token"] == row["greedy_token"]
        assert row["scale_displacement_closure_maxabs"] < 2e-15
        if row["carrying"] == 0:
            assert row["ideal_contrast_numerator"] == result["baseline_contrast_numerator"]
    assert result["rows"]["neutral-p1"]["denominator"] != result["baseline_denominator"]


def test_degenerate_span_is_an_explicit_observation_and_corrupt_source_is_rejected():
    x = np.array([1., 2., 3.], dtype=np.float32)
    result = A["decompose"](dict(plus=x, minus=x), np.array([1., 0., 0.]))
    assert result["numeric_rank"] == 1 and "collapsed" in result["applicability"]
    assert np.linalg.norm(result["realized"]["neutral"]) == 0
    with pytest.raises(ValueError, match="FP32 writes"):
        A["decompose"](dict(plus=x.astype(np.float64), minus=x), np.ones(3))
    with pytest.raises(ValueError, match="finite source"):
        A["decompose"](dict(plus=x, minus=x), np.array([np.nan, 0., 0.]))


def test_source_bound_default_separator_request_rejects_tamper(tmp_path):
    folder = Path(__file__).parents[1] / "qualification"
    names = ["qwen_contrast_collateral_campaign.py", *C["HELPERS"]]
    for name in names:
        shutil.copyfile(folder / name, tmp_path / name)
    config = C["make_config"]({})
    request = dict(config=config, source_hashes={name: C["sha"](tmp_path / name) for name in names})
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    C["authenticate_request"](request, config, tmp_path)
    changed = copy.deepcopy(request); changed["config"]["seed"] += 1
    with pytest.raises(ValueError, match="request/config identity"):
        C["authenticate_request"](changed, config, tmp_path)
    (tmp_path / names[0]).write_text("changed source")
    with pytest.raises(ValueError, match="request source changed"):
        C["authenticate_request"](request, config, tmp_path)


def test_actual_tiny_global_seal_original_targets_ordinary_sham_and_measured_futures(tmp_path, monkeypatch):
    model = tiny_model()
    settings = dict(seed=37, lanczos_steps=8, max_prompt_tokens=8, max_prefix_ops=2048, cells=C["CELLS"], future_cells=C["FUTURES"], continuation_tokens=2,
        max_native_entries=64, max_wall_seconds=60, transaction_bytes=16 << 20)
    root = tmp_path / "qwen-contrast-collateral-result"
    result = C["campaign"](model, [dict(id="tiny", text="untrained fixture only", ids=torch.tensor([[2, 4, 6]]))], [7, 11], root, settings,
        model_identity=dict(name="tiny-random-Qwen2-no-pretrained"), source_identity=dict(fixture=C["digest"]("new-source-only-tiny")), collateral_ids=dict(other=9))
    counts = result["summary"]["counts"]
    assert counts["prefixes"] == counts["native_MLP_bases"] == counts["ordinary_shams"] == 1
    assert counts["parity_targets"] == 4 and counts["component_readouts"] == 9 and counts["ordinary_prefills"] == 5
    assert counts["cached_futures"] == 10 and counts["physical_entries"] == 31 and counts["denied_preseal_entries"] == 0
    assert counts["final_MLP_entries"] == counts["final_postnorm_entries"] == 21
    assert counts["head_entries"] == counts["final_norm_entries"] == 25
    assert result["summary"]["transaction"]["verified"]
    events = [json.loads(row) for row in (root / "timeline.jsonl").read_text().splitlines()]
    assert [row["kind"] for row in events] == ["genuine_prefix_complete", "global_forecasts_sealed", "context_targets_complete"]
    assert events[1]["pretarget_counts"]["final_MLP_entries"] == events[1]["pretarget_counts"]["head_entries"] == 0
    observed = result["observations"]["contexts"]["tiny"]; predicted = result["predictions"]["contexts"]["tiny"]
    assert observed["sham_vs_baseline"]["max_abs"] == 0 and not result["predictions"]["continuation_forecast"]
    for name, row in observed["ordinary"].items():
        assert row["earlier_rows_exact"] and row["cache_exact_vs_observed_prefix"]
        assert row["component_vs_ordinary"]["max_abs"] < 1e-6
        if name != "sham":
            assert len(row["continuation"]) == 2 and row["continuation"][0]["consumed_token"] == row["greedy_token"]
            assert row["continuation"][1]["consumed_token"] == row["continuation"][0]["greedy_token"]
    for name, row in observed["component"].items():
        vector = row["carrier"].reshape(-1).astype(np.float64)
        assert row["geometry"]["addition_error_l2"] >= 0 and row["logits"].shape == (31,)
        assert predicted["readout"]["rows"][name]["logits"].shape == (31,) and np.isfinite(vector).all()

    def blocked(*_args, **_kwargs):
        raise AssertionError("Publisher must not call native modules")

    monkeypatch.setattr(torch.nn.Module, "__call__", blocked)
    expression = next(node.value for node in ast.walk(ast.parse(inspect.getsource(C["worker"]))) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "execution" for target in node.targets))
    execution = eval(compile(ast.Expression(expression), "actual_worker_profile", "eval"), dict(torch=torch, np=np, importlib=importlib))
    assert type(execution["torch"]) is str
    accounts = C["publish_accounts"](result, root, execution=execution, source_identity=dict(fixture=C["digest"]("new-source-only-tiny")), run_id="tiny-no-pretrained")
    assert len(accounts["rows"]) == len(accounts["atlas_ids"]) == 9 and accounts["coverage"]["independent_contexts"] == 1
    saved = json.loads(next((root / "catalog").glob("*.json")).read_text())
    assert saved["law_id"] == "rms-reader.contrast-dual-signed" and type(saved["binding"]["execution"]["torch"]) is str
    ready = C["dependencies"]()["flux_image_tail_microscope.py"]["finalize"](root, job_id="tiny-mechanics-only", request_sha256=C["digest"]("tiny"))
    assert ready["file_count"] > 30 and not (root / "parameters").exists()


def test_previous_full_capture_is_released_before_next_genuine_prefix(tmp_path, monkeypatch):
    """Captures retain cloned parameter/intermediate banks, unlike saved cuts."""
    helpers = C["dependencies"](); base = helpers["qwen_spectral_readout_campaign.py"]
    native_prefix = base["prefix_with_cache"]; references = []

    def watched_prefix(*args, **kwargs):
        assert all(reference() is None for reference in references), "previous full capture survives into the next prefix"
        capture = native_prefix(*args, **kwargs)
        assert len(capture.observed_values) > 100 and capture.initial_values
        references.append(weakref.ref(capture))
        return capture

    base["prefix_with_cache"] = watched_prefix
    monkeypatch.setitem(C["campaign"].__globals__, "dependencies", lambda: helpers)
    settings = dict(seed=37, lanczos_steps=8, max_prompt_tokens=8, max_prefix_ops=2048, cells=C["CELLS"], future_cells=C["FUTURES"], continuation_tokens=0,
        max_native_entries=64, max_wall_seconds=60, transaction_bytes=16 << 20)
    prompts = [dict(id=name, text="untrained lifetime fixture only", ids=torch.tensor([values])) for name, values in (("z-first", [2, 4, 6]), ("a-second", [3, 5]))]
    result = C["campaign"](tiny_model(), prompts, [7, 11], tmp_path / "lifetime-result", settings,
        model_identity=dict(name="tiny-random-Qwen2-no-pretrained"), source_identity=dict(fixture=C["digest"]("capture-lifetime-only")))
    assert len(references) == 2 and all(reference() is None for reference in references)
    assert result["summary"]["counts"]["prefixes"] == 2 and result["summary"]["counts"]["physical_entries"] == 42
    assert list(result["predictions"]["order"]) == ["z-first", "a-second"]
    assert result["summary"]["transaction"]["verified"]


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
def test_same_original_recipe_helper_restores_exact_factors_and_tied_head(tmp_path, source_dtype):
    from safetensors.torch import save_file

    import mdb_lanes.parameter_artifacts as artifacts

    model = tiny_model(); selected = {name: value for name, value in model.named_parameters() if name in ("model.embed_tokens.weight", "model.norm.weight") or name.startswith(("model.layers.1.post_attention_layernorm.", "model.layers.1.mlp."))}
    with torch.no_grad():
        for value in selected.values():
            value.copy_(value.to(source_dtype).float())
    checkpoint = tmp_path / "checkpoint"; checkpoint.mkdir(); file = checkpoint / "model.safetensors"
    save_file({name: value.detach().to(source_dtype).contiguous() for name, value in selected.items()}, file)
    output = tmp_path / "recipes"; output.mkdir()
    bindings = C["dependencies"]()["qwen_spectral_parameter_recipes.py"]["prepare"](model, checkpoint, {file.name: dict(bytes=file.stat().st_size, sha256=C["sha"](file))}, output, inspect.getfile(artifacts))
    assert bindings["head"]["original_tensor"] == "model.embed_tokens.weight" and not list(output.rglob("*.npy"))
    restored = tmp_path / "restored"; receipt = artifacts.restore(json.loads((output / "parameter-artifacts.json").read_text()), restored, checkpoint)
    assert receipt["model_calls"] == receipt["conversions"] == 0
    for name, binding in bindings.items():
        array = np.load(restored / ("parameters/"+name+".npy"), allow_pickle=False)
        if source_dtype == torch.bfloat16:
            array = (array.astype(np.uint32) << 16).view(np.float32)
        assert array.tobytes() == model.get_parameter(binding["native_name"]).detach().numpy().tobytes()

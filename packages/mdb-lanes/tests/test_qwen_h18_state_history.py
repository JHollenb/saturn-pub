"""Synthetic installed-Qwen parity and native-entry custody mechanics only."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).parents[1] / "qualification"
H = runpy.run_path(str(ROOT / "qwen_h18_state_history.py"))
G = H["load_prefix"](ROOT / "flux_functional_qwen_prefix.py")
DISPATCH = {"backend": "sdpa", "gqa_policy": "repeat_kv", "source": "tiny-native-sdpa"}
ROPE = {"init_device": "cpu", "init_dtype": "float32", "placement_dtype": "float32", "source": "tiny-native-device-only"}
GEOMETRY = {"edit_index": 2, "horizon": 4, "selected": (1, 2, 4)}


def fixture():
    torch.manual_seed(187)
    config = transformers.Qwen3Config(vocab_size=23, hidden_size=6, intermediate_size=12,
        num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2, head_dim=2,
        max_position_embeddings=32, rope_theta=1_000_000, attention_dropout=0.0,
        attention_bias=False, pad_token_id=0, use_cache=False)
    model = transformers.Qwen3ForCausalLM(config).eval().to(dtype=torch.bfloat16)
    model.config._attn_implementation = "sdpa"
    model.model.rotary_emb = type(model.model.rotary_emb)(config=model.config, device=torch.device("cpu"))
    ids = torch.tensor([[1, 3, 0, 4, 0]], dtype=torch.int64)
    mask = torch.tensor([[1, 1, 0, 1, 0]], dtype=torch.int64)
    return model, ids, mask


def forecast(model, ids, mask, observe=None):
    return H["functional_state_history"](G, ids, mask, model.state_dict(), model.config,
        attention_dispatch=DISPATCH, rope_buffer_policy=ROPE, observe=observe, **GEOMETRY)


def native(model, ids, mask, **kwargs):
    return H["native_state_history"](model, ids, mask, authorize_native=lambda: None, **GEOMETRY, **kwargs)


def test_common_own_prefix_and_edited_suffix_match_full_native_without_refresh():
    model, ids, mask = fixture()
    saved = {}
    result = forecast(model, ids, mask, lambda lane, name, value: saved.setdefault(lane, {}).update({name: value.clone()}))
    assert result["counts"] == {"blocks": 6, "embedding": 1, "linear": 42, "RMSNorm": 24,
        "SDPA": 6, "eager_attention": 0, "rotary_applications": 12, "SiLU_products": 6, "native_module_forwards": 0,
        "residual_additions": 12, "state_negations": 1, "conditioning_concats": 2, "rope_context_formations": 1}
    frozen = G["functional_qwen_prefix"](ids, mask, model.state_dict(), model.config,
        attention_dispatch=DISPATCH, rope_buffer_policy=ROPE, prefix_layers=4, selected_hidden_states=(1, 2, 4))
    for index in range(5):
        assert torch.equal(result["common"][index], frozen["hidden_states"][index])
    actual = native(model, ids, mask)
    for index in range(5):
        assert torch.equal(result["common"][index], actual["common"][index])
    for index in range(2, 5):
        assert torch.equal(result["minus"][index], actual["minus"][index])
    assert torch.equal(result["direct"], actual["direct"])
    assert torch.equal(result["coherent"], actual["coherent"])
    assert set(saved["plus"]) == {"h", "norm", "q", "k", "v", "head", "projection", "R", "postnorm", "gate", "up", "activation", "product", "M", "Hnext"}
    assert actual["counts"]["SDPA"] == 7
    assert actual["counts"]["baseline_blocks"] == 5
    assert actual["counts"]["suffix_blocks"] == 2
    assert result["mask_profile"]["has_padding"] is True
    assert actual["genuine_qwen_act"] is False


def test_native_sham_replay_and_actual_postedit_consumption_are_exact():
    model, ids, mask = fixture()
    before = H["_versions"](model)
    sham = native(model, ids, mask, suffix_sign=1)
    for index in range(2, 5):
        assert torch.equal(sham["minus"][index], sham["common"][index])
    edited = native(model, ids, mask)
    assert torch.equal(edited["consumed"][0]["hidden"], -edited["common"][2])
    assert [row["index"] for row in edited["consumed"]] == [2, 3]
    assert torch.equal(edited["consumed"][1]["hidden"], edited["minus"][3])
    assert edited["source_context"]["position_ids"].tolist() == [list(range(5))]
    assert edited["source_context"]["attention_mask"].shape == (1, 1, 5, 5)
    assert H["_versions"](model) == before


def test_native_ports_are_actual_and_global_hooks_restore_after_callback_failure():
    model, ids, mask = fixture()
    sdpa = torch.nn.functional.scaled_dot_product_attention
    hooks = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    def fail(lane, name, value):
        if lane == "minus" and name == "postnorm":
            raise RuntimeError("deliberate observer failure")
    with pytest.raises(RuntimeError, match="deliberate"):
        native(model, ids, mask, observe=fail)
    assert torch.nn.functional.scaled_dot_product_attention is sdpa
    assert hooks == [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    ports = {}
    native(model, ids, mask, observe=lambda lane, name, x: ports.setdefault(lane, {}).update({name: x.clone()}))
    assert torch.equal(ports["minus"]["h"], -ports["plus"]["h"])
    assert ports["minus"]["k_consumed"].shape[1] == 4
    assert {"product", "M", "R", "Hnext"} <= ports["minus"].keys()


def test_separate_native_entry_can_refuse_before_any_model_call():
    model, ids, mask = fixture()
    called = []
    handle = model.register_forward_pre_hook(lambda *args: called.append(True))
    def deny():
        raise RuntimeError("cohort unsealed")
    try:
        with pytest.raises(RuntimeError, match="unsealed"):
            H["native_state_history"](model, ids, mask, authorize_native=deny, **GEOMETRY)
    finally:
        handle.remove()
    assert called == []


def test_forecast_global_module_guard_releases_and_catches_callback_native_call():
    model, ids, mask = fixture()
    def violate(*args):
        model.model.embed_tokens(ids)
    with pytest.raises(RuntimeError, match="forbidden"):
        forecast(model, ids, mask, violate)
    assert model.model.embed_tokens(ids).shape == (1, 5, 6)
    assert forecast(model, ids, mask)["counts"]["native_module_forwards"] == 0


def test_edit_index_and_source_policy_refusals_are_mechanics_not_numeric_gates():
    model, ids, mask = fixture()
    with pytest.raises(ValueError, match="selected"):
        H["functional_state_history"](G, ids, mask, model.state_dict(), model.config,
            attention_dispatch=DISPATCH, rope_buffer_policy=ROPE, edit_index=3, horizon=4, selected=(1, 2, 4))
    with pytest.raises(ValueError, match="FP32"):
        H["functional_state_history"](G, ids, mask, model.state_dict(), model.config,
            attention_dispatch=DISPATCH, rope_buffer_policy={**ROPE, "placement_dtype": "bfloat16"}, **GEOMETRY)
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32


def test_signed_zero_numeric_vs_bit_defects_and_nonlinear_writer_are_observations():
    p = torch.tensor([0., -0., 1.], dtype=torch.bfloat16)
    m = torch.tensor([0., 0., -1.], dtype=torch.bfloat16)
    report = H["oddness_diagnostic"](p, m)
    assert report["numerical_changed"] == 0 and report["differing_bytes"] > 0
    model, ids, mask = fixture()
    ports = {}
    forecast(model, ids, mask, lambda lane, name, x: ports.setdefault(lane, {}).update({name: x.clone()}))
    diagnostics = H["writer_diagnostic"](ports["plus"], ports["minus"])
    assert diagnostics["extra_learned_projections"] == 0
    assert diagnostics["status"] == "observation-only"
    assert diagnostics["oddness"]["h"]["numerical_changed"] == 0
    assert "minus_finite_addition_remainder" in diagnostics


def test_fully_attended_native_dispatch_is_explicit_and_retains_parity():
    model, ids, _ = fixture()
    mask = torch.ones_like(ids)
    result = H["functional_state_history"](G, ids, mask, model.state_dict(), model.config,
        attention_dispatch={**DISPATCH, "gqa_policy": "enable_gqa"}, rope_buffer_policy=ROPE, **GEOMETRY)
    actual = native(model, ids, mask)
    for index in range(2, 5):
        assert torch.equal(result["minus"][index], actual["minus"][index])
    assert all(row["mask_shape"] is None for row in actual["sdpa_calls"])
    with pytest.raises(ValueError, match="enable_gqa"):
        forecast(model, ids, mask)


def test_prefix_source_pin_refuses_changed_helper_before_execution(tmp_path):
    altered = tmp_path / "prefix.py"
    altered.write_text("raise RuntimeError('must not execute')\n")
    with pytest.raises(ValueError, match="source differs"):
        H["load_prefix"](altered)


def test_functional_weight_mutation_is_detected_and_guard_is_released():
    model, ids, mask = fixture()
    changed = False
    def mutate(lane, name, value):
        nonlocal changed
        if not changed:
            model.model.embed_tokens.weight.add_(1)
            changed = True
    with pytest.raises(ValueError, match="parameter mapping changed"):
        forecast(model, ids, mask, mutate)
    assert model.model.embed_tokens(ids).shape == (1, 5, 6)

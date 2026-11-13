"""Tiny synthetic Qwen3 prefix parity tests; no checkpoint access or model jobs."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

FUNCTIONAL = runpy.run_path(
    str(Path(__file__).parents[1] / "qualification/flux_functional_qwen_prefix.py")
)


def _tiny_model(backend: str):
    config = transformers.Qwen3Config(
        vocab_size=67,
        # Deliberately exercise Qwen's valid unequal residual and attention
        # widths: 20-wide residual stream, 4*4 = 16 attention channels.
        hidden_size=20,
        intermediate_size=40,
        # Keep one suffix block so Transformers' output-capturing decorator
        # does not overwrite hidden_states[27] with the final model norm.  The
        # functional prefix still consumes only blocks 0..26.
        num_hidden_layers=28,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rope_theta=1_000_000,
        pad_token_id=0,
        attention_dropout=0.0,
        use_sliding_window=False,
    )
    model = transformers.Qwen3ForCausalLM(config).eval().to(dtype=torch.bfloat16)
    # The installed Transformers implementation chooses this interface at
    # runtime.  The functional call receives the corresponding inspected
    # contract explicitly; it never infers a backend from the class name.
    model.config._attn_implementation = backend
    return model


def _inputs():
    input_ids = torch.tensor([[3, 7, 11, 19, 23, 0, 0, 0, 29]], dtype=torch.int64)
    # The interior zero is a real padding position.  Keeping a later token
    # valid exercises key masking independently from the causal triangle.
    attention_mask = torch.tensor([[1, 1, 1, 0, 1, 0, 0, 0, 1]], dtype=torch.int64)
    return input_ids, attention_mask


def _bound_parameters(model):
    names = {"model.embed_tokens.weight"}
    names.update(
        f"model.layers.{layer}.{suffix}"
        for layer in range(27)
        for suffix in FUNCTIONAL["QWEN_LAYER_PARAMETER_SUFFIXES"]
    )
    state = model.state_dict()
    assert names <= set(state)
    return {name: state[name] for name in names}


def _run_forecast(model, backend, rope_buffer_policy=None):
    input_ids, attention_mask = _inputs()
    parameters = _bound_parameters(model)
    before_ids = input_ids.clone()
    before_mask = attention_mask.clone()
    dispatch = {
        "backend": backend,
        "gqa_policy": "repeat_kv",  # native masked SDPA also repeats KV
        "source": "transformers-5.14.1-qwen3-attention-interface-v1",
    }
    with FUNCTIONAL["module_entry_guard"](model) as entries:
        result = FUNCTIONAL["functional_qwen_prefix"](
            input_ids,
            attention_mask,
            parameters,
            model.config,
            attention_dispatch=dispatch,
            rope_buffer_policy=rope_buffer_policy,
        )
    assert entries == []
    torch.testing.assert_close(input_ids, before_ids, rtol=0, atol=0)
    torch.testing.assert_close(attention_mask, before_mask, rtol=0, atol=0)
    return result, input_ids, attention_mask


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_functional_qwen_prefix_matches_native_hidden_states_with_padding_and_gqa(backend):
    torch.manual_seed(41)
    model = _tiny_model(backend)
    result, input_ids, attention_mask = _run_forecast(model, backend)

    with torch.inference_mode():
        native = model.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # Hidden-state indexing is the native capture convention: index 0 is the
    # embedding and index n is the output after decoder block n-1.  Compare all
    # prefix states, not just the selected conditioning ports.
    assert len(native.hidden_states) >= 28
    for index in range(28):
        torch.testing.assert_close(
            result["hidden_states"][index], native.hidden_states[index], rtol=0, atol=0
        )
    torch.testing.assert_close(result["ports"]["embedding"], native.hidden_states[0], rtol=0, atol=0)
    for index in (9, 18, 27):
        torch.testing.assert_close(
            result["ports"]["selected"][index], native.hidden_states[index], rtol=0, atol=0
        )
    expected_conditioning = torch.cat(
        [native.hidden_states[index] for index in (9, 18, 27)], dim=-1
    )
    torch.testing.assert_close(result["conditioning"], expected_conditioning, rtol=0, atol=0)
    assert result["conditioning"].shape == (1, input_ids.shape[1], 60)
    assert result["profile"]["conditioning_width"] == 60
    assert result["profile"]["attention_width"] == 16
    assert result["profile"]["num_key_value_groups"] == 2
    assert result["profile"]["gqa_policy"] == "repeat_kv"
    assert result["profile"]["qualification_scope"] == "tiny synthetic CPU parity only; original CUDA pending"
    assert result["profile"]["mask"]["has_padding"] is True
    assert result["profile"]["mask"]["shape"] == [1, 1, input_ids.shape[1], input_ids.shape[1]]
    assert result["position_ids"].tolist() == [list(range(input_ids.shape[1]))]


def test_cpu_float32_rope_policy_matches_device_only_native_buffer():
    torch.manual_seed(59)
    model = _tiny_model("eager")
    # Reproduce the historical loader policy without reading any observed
    # activation: constructor CPU FP32 RoPE, then device-only placement.
    native_rotary = type(model.model.rotary_emb)(config=model.config, device=torch.device("cpu"))
    model.model.rotary_emb = native_rotary
    policy = {
        "init_device": "cpu",
        "init_dtype": "float32",
        "placement_dtype": "float32",
        "source": "synthetic-native-device-only",
    }
    result, input_ids, attention_mask = _run_forecast(model, "eager", policy)
    with torch.inference_mode():
        native = model.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32
    for index in range(28):
        torch.testing.assert_close(
            result["hidden_states"][index], native.hidden_states[index], rtol=0, atol=0
        )


def test_prospective_tokenizer_stage_is_explicit_and_cannot_repair_history():
    input_ids, attention_mask = _inputs()
    binding = {
        "cohort": "prospective-new",
        "parser_stage": "tokenizer",
        "implementation_pinned": True,
        "tokenizer_files_sha256": {"chat_template.jinja": "future-bound"},
    }
    bound = FUNCTIONAL["bind_prospective_tokenized_input"](
        input_ids, attention_mask, tokenizer_binding=binding
    )
    assert bound["historical_repair"] is False
    assert bound["tokenizer_binding"] == binding
    with pytest.raises(ValueError, match="prospective|historical"):
        FUNCTIONAL["bind_prospective_tokenized_input"](
            input_ids,
            attention_mask,
            tokenizer_binding={
                "cohort": "historical",
                "parser_stage": "tokenizer",
                "implementation_pinned": True,
            },
        )


def test_dispatch_and_gqa_policy_are_required_and_guard_is_live():
    torch.manual_seed(43)
    model = _tiny_model("eager")
    input_ids, attention_mask = _inputs()
    parameters = _bound_parameters(model)
    with pytest.raises(ValueError, match="attention_dispatch"):
        FUNCTIONAL["functional_qwen_prefix"](
            input_ids,
            attention_mask,
            parameters,
            model.config,
            attention_dispatch={"backend": "eager", "gqa_policy": "repeat_kv"},
        )
    with pytest.raises(ValueError, match="eager.*repeat_kv"):
        FUNCTIONAL["functional_qwen_prefix"](
            input_ids,
            attention_mask,
            parameters,
            model.config,
            attention_dispatch={
                "backend": "eager",
                "gqa_policy": "enable_gqa",
                "source": "synthetic-wrong-contract",
            },
        )
    with pytest.raises(FUNCTIONAL["ModuleForwardForbidden"]):
        with FUNCTIONAL["module_entry_guard"](model):
            model.model(input_ids, attention_mask=attention_mask, use_cache=False)


def test_unmasked_sdpa_requires_explicit_enable_gqa_policy():
    torch.manual_seed(47)
    model = _tiny_model("sdpa")
    input_ids, _ = _inputs()
    parameters = _bound_parameters(model)
    with pytest.raises(ValueError, match="enable_gqa"):
        FUNCTIONAL["functional_qwen_prefix"](
            input_ids,
            None,
            parameters,
            model.config,
            attention_dispatch={
                "backend": "sdpa",
                "gqa_policy": "repeat_kv",
                "source": "synthetic-unmasked-native-contract",
            },
        )


def test_unmasked_sdpa_enable_gqa_matches_native_when_contract_is_bound():
    torch.manual_seed(53)
    model = _tiny_model("sdpa")
    input_ids, _ = _inputs()
    parameters = _bound_parameters(model)
    dispatch = {
        "backend": "sdpa",
        "gqa_policy": "enable_gqa",
        "source": "transformers-5.14.1-sdpa-mask-dispatch-v1",
    }
    with FUNCTIONAL["module_entry_guard"](model):
        result = FUNCTIONAL["functional_qwen_prefix"](
            input_ids,
            None,
            parameters,
            model.config,
            attention_dispatch=dispatch,
        )
    with torch.inference_mode():
        native = model.model(input_ids, output_hidden_states=True, use_cache=False)
    for index in range(28):
        torch.testing.assert_close(
            result["hidden_states"][index], native.hidden_states[index], rtol=0, atol=0
        )
    assert result["profile"]["mask"]["kind"] == "native-causal-flag"
    assert result["profile"]["gqa_policy"] == "enable_gqa"

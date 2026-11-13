"""One synthetic loader-to-prefix parity check; no original model or CUDA."""
from __future__ import annotations

import hashlib
import io
import runpy
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
HERE = Path(__file__).parents[1] / "qualification"
WORKER = runpy.run_path(str(HERE / "flux_functional_qwen_prefix_worker.py"))
FUNCTIONAL = runpy.run_path(str(HERE / "flux_functional_qwen_prefix.py"))


def raw_bits(value):
    return value.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes()


class TinyStreamingRecipeBank:
    """Read one actual synthetic checkpoint tensor per loader request, uncached."""

    def __init__(self, path, rows):
        self.path = path
        self.rows = {row["source"]["tensor"]: row for row in rows}
        self.calls = []

    def load(self, record):
        name = record["parameter_tensor"]
        row = self.rows[name]
        assert record == WORKER["_recipe"](row, name)
        with safe_open(self.path, framework="pt", device="cpu") as checkpoint:
            value = checkpoint.get_tensor(name)
        assert value.dtype == torch.bfloat16 and list(value.shape) == row["source"]["shape"]
        assert hashlib.sha256(raw_bits(value)).hexdigest() == row["source"]["payload_sha256"]
        self.calls.append(name)
        # Qualified RecipeBank returns FP32 values carrying exact BF16 bits.
        # Preserve that ABI while streaming from a real tiny checkpoint.
        return value.float().numpy().copy()


def test_actual_qwen_hydration_then_guarded_prefix_matches_pretrained_native(tmp_path, monkeypatch):
    config = transformers.Qwen3Config(vocab_size=37, hidden_size=6, intermediate_size=12,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        max_position_embeddings=64, rope_theta=1_000_000, tie_word_embeddings=True,
        pad_token_id=0, attention_dropout=0.0, use_sliding_window=False)
    checkpoint_dir = tmp_path / "synthetic-checkpoint"
    with torch.random.fork_rng():
        torch.manual_seed(7103)
        source = transformers.Qwen3ForCausalLM(config).eval().to(dtype=torch.bfloat16)
        source.save_pretrained(checkpoint_dir, safe_serialization=True)
    del source
    checkpoint_path = checkpoint_dir / "model.safetensors"
    checkpoint_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    rows = []
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        names = set(checkpoint.keys())
        assert "lm_head.weight" not in names
        for name in sorted(names):
            value = checkpoint.get_tensor(name)
            assert value.dtype == torch.bfloat16
            encoded = io.BytesIO()
            np.save(encoded, value.contiguous().view(torch.uint16).numpy(), allow_pickle=False)
            digest = hashlib.sha256(encoded.getvalue()).hexdigest()
            rows.append(dict(path="synthetic/" + digest + ".npy", sha256=digest,
                source=dict(tensor=name, shape=list(value.shape), dtype="BF16", payload_sha256=hashlib.sha256(raw_bits(value)).hexdigest())))
    assert len(rows) == 2 + 4 * len(FUNCTIONAL["QWEN_LAYER_PARAMETER_SUFFIXES"]) == 46
    bank = TinyStreamingRecipeBank(checkpoint_path, rows)
    cuda_entries = []
    def no_cuda(*args, **kwargs):
        cuda_entries.append(True)
        raise AssertionError("synthetic CPU hydration attempted CUDA initialization")
    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    dispatch = dict(backend="sdpa", gqa_policy="repeat_kv", source="installed-transformers-masked-SDPA-synthetic-hydration")
    restored = WORKER["_load_qwen"](dict(qwen_config=str(checkpoint_dir / "config.json"), attention_dispatch=dispatch), torch.device("cpu"), bank, dict(artifacts=rows))
    reference = transformers.Qwen3ForCausalLM.from_pretrained(checkpoint_dir, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).eval().requires_grad_(False)

    # Check every learned tensor and the intentionally unarchived tied head
    # before either model has performed a forward. This catches FP32 meta
    # allocation, missing/scrambled restoration and broken ties directly.
    expected_parameters = dict(reference.named_parameters(remove_duplicate=False))
    actual_parameters = dict(restored.named_parameters(remove_duplicate=False))
    assert set(actual_parameters) == set(expected_parameters) == names | {"lm_head.weight"}
    for name, value in actual_parameters.items():
        expected = expected_parameters[name]
        assert value.dtype == expected.dtype == torch.bfloat16
        assert value.device.type == expected.device.type == "cpu" and not value.is_meta
        assert raw_bits(value) == raw_bits(expected), name
    for model in (restored, reference):
        assert model.lm_head.weight is model.model.embed_tokens.weight
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        for _, value in model.named_buffers():
            assert not value.is_meta and value.device.type == "cpu" and bool(torch.isfinite(value).all())
        for name in ("inv_freq", "original_inv_freq"):
            value = getattr(model.model.rotary_emb, name)
            expected = getattr(reference.model.rotary_emb, name)
            assert value.dtype == expected.dtype == torch.float32
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            # head_dim=4 gives a nontrivial frequency, so this also rejects a
            # BF16-rounded frequency merely recast to FP32.
            assert not torch.equal(value, value.bfloat16().float())
    assert bank.calls == sorted(names) and len(set(bank.calls)) == 46
    assert WORKER["ROPE_POLICY"]["placement_dtype"] == "float32"

    input_ids = torch.tensor([[3, 5, 0, 11, 13, 17, 19]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 0, 1, 1, 0, 1]], dtype=torch.int64)
    before_ids, before_mask = input_ids.clone(), attention_mask.clone()
    native_entries, native_sdpa = [], []
    native_handle = reference.register_forward_pre_hook(lambda *_: native_entries.append("full-native-four-layer"))
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    def observed_sdpa(query, key, value, *args, **kwargs):
        native_sdpa.append(dict(query=tuple(query.shape), key=tuple(key.shape), mask=kwargs.get("attn_mask"), enable_gqa=kwargs.get("enable_gqa", False)))
        return original_sdpa(query, key, value, *args, **kwargs)
    try:
        with torch.inference_mode(), FUNCTIONAL["module_entry_guard"]([restored, reference]) as entries:
            forecast = FUNCTIONAL["functional_qwen_prefix"](input_ids, attention_mask, actual_parameters, restored.config,
                attention_dispatch=dispatch, rope_buffer_policy=WORKER["ROPE_POLICY"], prefix_layers=3, selected_hidden_states=(1, 2, 3))
        assert entries == [] and native_entries == []
        assert set(forecast["hidden_states"]) == {0, 1, 2, 3}
        assert forecast["profile"]["attention_width"] == 16 and forecast["profile"]["hidden_size"] == 6
        assert forecast["profile"]["mask"]["has_padding"] is True
        assert forecast["profile"]["gqa_policy"] == "repeat_kv"
        # Only now execute the matched full native reference. Its suffix
        # keeps selected state3 an interior pre-final-norm state.
        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", observed_sdpa)
        with torch.inference_mode():
            native = reference(input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        assert native_entries == ["full-native-four-layer"] and len(native_sdpa) == 4
        assert all(row["query"][1] == row["key"][1] == 4 and row["mask"] is not None and row["enable_gqa"] is False for row in native_sdpa)
        assert len(native.hidden_states) == 5
        for index in range(4):
            assert raw_bits(forecast["hidden_states"][index]) == raw_bits(native.hidden_states[index]), index
        expected_e = torch.cat([native.hidden_states[index] for index in (1, 2, 3)], dim=-1)
        assert forecast["conditioning"].shape == (1, 7, 18)
        assert raw_bits(forecast["conditioning"]) == raw_bits(expected_e)
        assert raw_bits(forecast["ports"]["embedding"]) == raw_bits(native.hidden_states[0])
        assert not torch.equal(native.hidden_states[3], native.hidden_states[-1])
        for index in (1, 2, 3):
            assert raw_bits(forecast["ports"]["selected"][index]) == raw_bits(native.hidden_states[index])
        assert torch.equal(input_ids, before_ids) and torch.equal(attention_mask, before_mask)
        assert cuda_entries == []
        for name, value in actual_parameters.items():
            assert raw_bits(value) == raw_bits(expected_parameters[name]), name
        assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_digest
    finally:
        native_handle.remove()

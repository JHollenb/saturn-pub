"""Tiny random CPU canary for the exact port hooks shipped to the fleet."""
import importlib.util
from pathlib import Path

import pytest


def test_native_ports_sham_cache_and_attention_are_compatible():
    torch = pytest.importorskip("torch")
    from transformers import Qwen2Config, Qwen2ForCausalLM
    path = Path(__file__).parents[1] / "qualification/live_scaffold_worker.py"
    spec = importlib.util.spec_from_file_location("live_scaffold_native_canary", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    torch.manual_seed(113)
    config = Qwen2Config(vocab_size=32, hidden_size=24, intermediate_size=32,
                        num_hidden_layers=28, num_attention_heads=12, num_key_value_heads=2,
                        max_position_embeddings=64)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        plain = model(input_ids=ids, use_cache=True, output_attentions=False, logits_to_keep=1)
        with worker.Capture(model) as capture:
            observed = model(input_ids=ids, use_cache=True, output_attentions=True, logits_to_keep=1)
        with worker.Capture(model, {"layer": 22, "kind": "replace", "heads": [6, 8],
                                    "value": capture.heads[22]}) as sham:
            repeated = model(input_ids=ids, use_cache=True, output_attentions=True, logits_to_keep=1)
        assert torch.equal(plain.logits, observed.logits)
        assert torch.equal(plain.logits, repeated.logits)
        assert observed.attentions[22].shape == (1, 12, 3, 3)
        assert observed.past_key_values.layers[22].values.shape == (1, 2, 3, 2)
        assert torch.equal(sham.writes[23], capture.writes[23])
        token = observed.logits[:, -1].argmax(-1)[:, None]
        future = model(input_ids=token, past_key_values=observed.past_key_values,
                       use_cache=True, logits_to_keep=1)
        assert future.past_key_values.get_seq_length() == 4

"""adapters.load(base, lora=...) merges a PEFT/LoRA adapter via peft (never hand-rolled)."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from saturn_pub.adapters import load
from saturn_pub.adapters.decoder import DecoderAdapter


@pytest.fixture(autouse=True)
def _no_grad():
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        yield
    finally:
        torch.set_grad_enabled(prev)


def test_lora_without_peft_raises_friendly_error():
    if importlib.util.find_spec("peft") is not None:
        pytest.skip("peft installed; the missing-extra error path cannot trigger")
    model = DecoderAdapter.tiny("llama", seed=1).model
    with pytest.raises(ImportError, match="peft"):
        load(model, lora="/any/path", device=None)


def test_lora_merge_round_trip(tmp_path):
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    base = DecoderAdapter.tiny("llama", seed=1).model
    config = LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(base, config)
    # Make the adapter non-trivial so the merged model differs from the base.
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.1)
    peft_model.save_pretrained(str(tmp_path / "lora"))

    merged = load(DecoderAdapter.tiny("llama", seed=1).model, lora=tmp_path / "lora", device=None)
    assert isinstance(merged, DecoderAdapter)
    plain = DecoderAdapter.tiny("llama", seed=1)
    prompt = [5, 7, 11, 3]
    # the merged LoRA changes the logits vs the un-merged base
    assert not torch.allclose(merged.native_logits(prompt), plain.native_logits(prompt))
    # and the fast lane runs on the merged model
    session = merged.session(prompt)
    session.generate(3)
    assert session.read("tokens").shape[1] == len(prompt) + 3

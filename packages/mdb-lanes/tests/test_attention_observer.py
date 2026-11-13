"""The observer returns the unchanged native result and restores dispatch."""
import pytest
import torch
from saturn_pub.adapters.qwen import QwenAdapter
from transformers.models.qwen2 import modeling_qwen2

from mdb_lanes.attention_observer import AttentionObserver


def test_native_parity_qk_allocation_gqa_and_no_mutation():
    adapter = QwenAdapter.tiny()
    original = modeling_qwen2.eager_attention_forward
    native = adapter.session([5, 7, 11])
    parent = native.capture(retain=False)
    native.continue_(adapter.layers + 2)
    logits = native.read("logits")
    native.restore(parent)
    with AttentionObserver(adapter.model) as observer:
        native.continue_(adapter.layers + 2)
        with pytest.raises(ValueError, match="nested"):
            with AttentionObserver(adapter.model):
                pass
    assert modeling_qwen2.eager_attention_forward is original
    assert torch.equal(native.read("logits"), logits)
    assert len(observer.rows) == adapter.layers
    for row in observer.rows:
        q, k, v, attention = [row[key] for key in ("query", "key", "value", "attention")]
        repeated = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        expected = (q @ repeated.transpose(-1, -2) * q.shape[-1] ** -.5).softmax(-1)
        torch.testing.assert_close(attention, expected)
        torch.testing.assert_close(attention.sum(-1), torch.ones_like(attention.sum(-1)))
        value = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        torch.testing.assert_close(row["head_return"], (attention @ value).transpose(1, 2))


def test_exception_restores_native_dispatch():
    adapter = QwenAdapter.tiny()
    original = modeling_qwen2.eager_attention_forward
    with pytest.raises(RuntimeError, match="test"):
        with AttentionObserver(adapter.model):
            raise RuntimeError("test")
    assert modeling_qwen2.eager_attention_forward is original

"""pyvene Qwen3 shim: after register_qwen3() an IntervenableModel builds on a tiny Qwen3 and a
vanilla interchange of v_proj.output at one position changes the logits only from that row."""

import pytest

torch = pytest.importorskip("torch")
pv = pytest.importorskip("pyvene")

from mdb.interop.pyvene_compat import register_qwen3  # noqa: E402


def test_qwen3_interchange_runs_and_is_positional():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    assert set(register_qwen3()) <= {"Qwen3ForCausalLM", "Qwen3Model", "Qwen3ForSequenceClassification"}
    register_qwen3()                                          # idempotent
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg).eval()
    im = pv.IntervenableModel({"layer": 1, "component": "model.layers[1].self_attn.v_proj.output",
                               "unit": "pos", "max_number_of_units": 1,
                               "intervention_type": pv.VanillaIntervention}, model=m)
    base, src = torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[6, 7, 8, 9, 10]])
    _, out = im({"input_ids": base}, [{"input_ids": src}], {"sources->base": ([[[2]]], [[[2]]])})
    clean = m(base).logits
    assert torch.equal(out.logits[0, :2], clean[0, :2])      # rows before the patched row
    assert (out.logits[0, 2:] - clean[0, 2:]).abs().max() > 0

"""``load_local`` gives the model Hugging Face gives: weights in the asked dtype, buffers as built.

``module.to(dtype=bf16)`` also casts buffers. RoPE's ``inv_freq`` is float32 by construction; a
bf16 copy has a relative error up to 3.7e-3 and moved the declared margin of Qwen2.5-1.5B by one
bf16 step (mrun job-6fe2324701e4)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.engine.models import load_local, tiny_model  # noqa: E402


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
def test_load_local_casts_the_weights_and_leaves_the_buffers(tmp_path, family):
    built = tiny_model(family, "fp32")
    built.save_pretrained(tmp_path)
    reference = {name: value.clone() for name, value in built.named_buffers()}
    assert any(name.endswith("inv_freq") for name in reference)

    loaded = load_local(tmp_path, dtype="bf16")
    assert {p.dtype for p in loaded.parameters()} == {torch.bfloat16}
    assert not loaded.training
    for name, value in loaded.named_buffers():
        assert value.dtype == reference[name].dtype == torch.float32, name
        assert torch.equal(value, reference[name]), name
    # the weights are the checkpoint's, rounded once
    for name, value in loaded.named_parameters():
        assert torch.equal(value, built.get_parameter(name).to(torch.bfloat16)), name

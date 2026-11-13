"""Synthetic worker custody tests; no checkpoint reads or native jobs."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

WORKER = runpy.run_path(
    str(Path(__file__).parents[1] / "qualification/flux_functional_qwen_prefix_worker.py")
)


def _tiny_config():
    return transformers.Qwen3Config(
        vocab_size=67,
        hidden_size=20,
        intermediate_size=40,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rope_theta=1_000_000,
        dtype="bfloat16",
    )


def test_native_rope_binding_uses_constructor_cpu_then_device_only_placement():
    config = _tiny_config()
    with torch.device("meta"):
        model = transformers.Qwen3ForCausalLM(config)
    model = model.to(dtype=torch.bfloat16).to_empty(device="cpu")
    expected = type(model.model.rotary_emb)(config=config, device=torch.device("cpu"))
    # The production profile matches the historical
    # from_pretrained(dtype=BF16).to(cuda) path: nonpersistent RoPE buffers
    # remain constructor FP32 while parameters are BF16.
    WORKER["_bind_native_rope_buffers"](
        model, config, torch.device("cpu"), placement_dtype="float32"
    )
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32
    assert model.model.rotary_emb.inv_freq.device.type == "cpu"
    torch.testing.assert_close(model.model.rotary_emb.inv_freq, expected.inv_freq, rtol=0, atol=0)
    torch.testing.assert_close(
        model.model.rotary_emb.original_inv_freq, expected.original_inv_freq, rtol=0, atol=0
    )


def test_worker_contract_exposes_physical_encoder_and_component_calls():
    contract = WORKER["contract"]({})
    assert contract["cases"] == 2
    assert contract["forecast"]["functional_module_forwards"] == 0
    assert contract["forecast"]["context_embedder_forecast_linear_calls"] == 2
    assert contract["native"]["native_encoder_calls"] == 2
    assert contract["native"]["retained_hidden_states"] == 56
    assert contract["consumers"]["context_embedder_native_calls"] == 4
    assert contract["consumers"]["rgb_forecasts"] == 0


def test_bfloat16_raw_bit_array_roundtrip_is_explicit(tmp_path):
    value = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.bfloat16)
    record = WORKER["_save_bits"](
        tmp_path, "arrays/case/conditioning.npy", value, kind="conditioning", case="case"
    )
    import numpy as np

    bits = np.load(tmp_path / record["path"], allow_pickle=False)
    restored = torch.from_numpy(bits.copy()).view(torch.bfloat16)
    torch.testing.assert_close(restored, value, rtol=0, atol=0)
    assert record["logical_dtype"] == "torch.bfloat16"
    assert record["shape"] == [1, 3]


def test_forecast_transaction_keeps_uint16_bfloat16_bits_without_unsupported_cast(tmp_path, monkeypatch):
    import sys
    import types

    import numpy as np

    from mdb_lanes.forecast_transaction import ForecastPayload, ForecastTransaction

    # The shared-file extension is a separate optional runtime package; this
    # unit test exercises the public transaction encoding contract without
    # requiring that storage service in the isolated qualification wheel.
    monkeypatch.setitem(
        sys.modules,
        "saturn_pub.shared_store",
        types.SimpleNamespace(publish_configured=lambda _path, _raw: None),
    )
    transaction = ForecastTransaction(
        tmp_path / "transaction",
        model_identity="qwen-test",
        input_identity="input-test",
        source_identity="source-test",
        frontier={"kind": "raw-input"},
    )
    transaction.frontier("input", {"rows": 1})
    bits = np.asarray([[0x3F80, 0xC000]], dtype=np.uint16)
    transaction.predict(
        "forecast",
        lambda _context: {
            "predictions": {"logical_dtype": "bfloat16"},
            "payloads": {"bits": ForecastPayload(bits, dtype=None, provenance={"logical_dtype": "bfloat16"})},
        },
    )
    assert transaction.payload("forecast", "bits").dtype == np.uint16
    np.testing.assert_array_equal(transaction.payload("forecast", "bits"), bits)
    transaction.seal()
    transaction.close()

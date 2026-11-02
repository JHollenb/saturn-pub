"""Native full-model wiring checks, without a reference-drift quality gate."""

import numpy as np
import pytest
import torch
from models import build_spf_model
from native_bridge import forward_native_hybrid_reference, generate_native_hybrid_reference


@pytest.mark.parametrize("accumulator", ["packed", "clocked"])
@torch.no_grad()
def test_native_previous_layer_readout_drives_later_ingress(monkeypatch, accumulator):
    pytest.importorskip("spf.tensor.sequence_mixer")
    from spf.tensor.core import decode_words

    model = build_spf_model(d_model=4, n_layers=2, state_size=2, seed=17)
    ids = torch.tensor([[2, 3, 4]])
    first = model.layers[0]
    hidden = model.embedding(ids)
    ingress = first.export_native_inputs(
        {"addresses": first.control_addresses(hidden), "v": first.v_values(hidden)}
    )
    result = ingress["program"].execute(
        ingress["addresses"], ingress["values"], accumulator=accumulator
    )
    expected = first.apply_readout(decode_words(result.words), hidden)
    seen = []
    original = model.layers[1].control_addresses

    def record_input(actual_hidden):
        seen.append(actual_hidden.detach().clone())
        return original(actual_hidden)

    monkeypatch.setattr(model.layers[1], "control_addresses", record_input)
    logits, report = forward_native_hybrid_reference(model, ids, accumulator=accumulator)
    assert logits.shape == (1, 3, 256)
    torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
    assert report["core_fallback_conversions"] == 0
    assert len(report["layers"]) == 2
    assert all(row["core_execution"] == "canonical_spf_word_planes" for row in report["layers"])


@pytest.mark.parametrize("accumulator", ["packed", "clocked"])
def test_conversion_accounting_matches_actual_outer_codec_calls(monkeypatch, accumulator):
    pytest.importorskip("spf.tensor.sequence_mixer")
    import spf.tensor.core as core

    model = build_spf_model(d_model=4, n_layers=2, state_size=2, seed=17)
    ids = torch.tensor([[2, 3, 4], [7, 8, 9]])
    encoded, decoded = [], []
    original_encode, original_decode = core.encode_words, core.decode_words

    def encode_count(value, *args, **kwargs):
        encoded.append(np.asarray(value).size)
        return original_encode(value, *args, **kwargs)

    def decode_count(words, *args, **kwargs):
        assert np.asarray(words).dtype == np.uint32
        decoded.append(np.asarray(words).size)
        return original_decode(words, *args, **kwargs)

    monkeypatch.setattr(core, "encode_words", encode_count)
    monkeypatch.setattr(core, "decode_words", decode_count)
    _, report = forward_native_hybrid_reference(model, ids, accumulator=accumulator)
    # Per layer: address ingress, payload ingress, static scalar diagonal;
    # then one decode of the final word output. No core codec calls occur.
    assert len(encoded) == 3 * len(model.layers)
    assert len(decoded) == len(model.layers)
    assert report["interop_fallback_conversions_elements"] == sum(encoded) + sum(decoded)
    assert sum(row["ingress_elements_encoded"] for row in report["layers"]) == sum(encoded)
    assert sum(row["readout_elements_decoded"] for row in report["layers"]) == sum(decoded)
    assert report["core_fallback_conversions"] == 0


@pytest.mark.parametrize("accumulator", ["packed", "clocked"])
def test_native_causality_and_generated_history(accumulator):
    pytest.importorskip("spf.tensor.sequence_mixer")
    model = build_spf_model(d_model=4, n_layers=2, state_size=2, seed=17)
    ids = torch.tensor([[2, 3, 4, 5]])
    changed = ids.clone()
    changed[:, 2:] = 19
    full = forward_native_hybrid_reference(model, ids, accumulator=accumulator)[0]
    altered = forward_native_hybrid_reference(model, changed, accumulator=accumulator)[0]
    torch.testing.assert_close(full[:, :2], altered[:, :2], rtol=0, atol=0)
    expected = ids[:, :2].clone()
    for _ in range(3):
        logits, _ = forward_native_hybrid_reference(model, expected, accumulator=accumulator)
        expected = torch.cat((expected, logits[:, -1].argmax(-1)[:, None]), 1)
    generated, rows = generate_native_hybrid_reference(
        model, ids[:, :2], accumulator=accumulator, tokens=3
    )
    assert torch.equal(generated, expected)
    assert len(rows) == 3

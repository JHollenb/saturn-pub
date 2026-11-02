"""Small CPU mechanics checks; these do not make a learned-quality claim."""

import numpy as np
import pytest
import torch
from models import build_spf_model, numpy_complex_reference


def small_model(seed=7):
    return build_spf_model(vocab_size=256, d_model=6, n_layers=2, state_size=3, seed=seed)


def test_shapes_batch_independence_and_seed_rng_closure():
    torch.manual_seed(3)
    before = torch.random.get_rng_state().clone()
    model = small_model()
    assert torch.equal(before, torch.random.get_rng_state())
    assert model.model_sha256() == small_model().model_sha256()
    ids = torch.tensor([[1, 2, 3], [8, 9, 10]])
    logits = model(ids)
    assert logits.shape == (2, 3, 256)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits[:1], model(ids[:1]))


def test_no_future_leakage_and_streaming_cut_equivalence():
    model = small_model()
    ids = torch.tensor([[1, 2, 3, 4, 5], [8, 9, 10, 11, 12]])
    full = model(ids)
    changed = ids.clone()
    changed[:, 3:] = 99
    torch.testing.assert_close(full[:, :3], model(changed)[:, :3])
    carrier = model.capture_carrier(ids, 3)
    torch.testing.assert_close(full[:, 3:], model.continue_from_carrier(carrier, ids[:, 3:]))
    assert set(carrier["state_planes"]) == {"layer.0.state", "layer.1.state"}
    assert carrier["state_planes"]["layer.0.state"].shape == (2, 3, 6, 2)
    assert not carrier["state_planes"]["layer.0.state"].requires_grad


def test_read_before_write_matches_independent_strict_causal_matrix():
    model = small_model()
    layer = model.layers[0]
    x = model.embedding(torch.tensor([[2, 3, 4, 5]]))
    _, final, details = layer.forward_reference(x, capture=True)
    a, b, c, v = [numpy_complex_reference(details[key]) for key in ("a", "b", "c", "v")]
    matrix = np.zeros((4, 4), np.complex128)
    for i in range(4):
        for j in range(i):
            matrix[i, j] = np.sum(c[0, i] * b[0, j] * np.prod(a[0, j + 1 : i + 1], axis=0))
        matrix[i, i] = layer.diagonal.item()
    np.testing.assert_allclose(
        numpy_complex_reference(details["mixed"])[0], matrix @ v[0], rtol=2e-5, atol=2e-6
    )
    assert torch.count_nonzero(details["strict_read"][:, 0]) == 0
    # Final state includes the last write even though that write was not read.
    expected = np.zeros((3, 6), np.complex128)
    for t in range(4):
        expected = a[0, t, :, None] * expected + b[0, t, :, None] * v[0, t, None]
    np.testing.assert_allclose(numpy_complex_reference(final)[0], expected, rtol=2e-5, atol=2e-6)


def test_resets_remove_prior_segment_at_each_layer():
    model = small_model()
    ids = torch.tensor([[2, 3, 4, 5]])
    reset = torch.tensor([[False, False, True, False]])
    segmented = model.forward_reference(ids, reset_before=reset)[0]
    torch.testing.assert_close(segmented[:, 2:], model(ids[:, 2:]))


def test_embedding_controller_gain_decay_and_head_receive_gradients():
    model = small_model()
    ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    loss = torch.nn.functional.cross_entropy(model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten())
    loss.backward()
    names = (
        "embedding.weight",
        "layers.0.address_projection.weight",
        "layers.0.value_projection.weight",
        "layers.0.decay_raw",
        "layers.0.write_log2_gain",
        "layers.0.read_log2_gain",
        "head.weight",
    )
    for name in names:
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0, name


def test_generation_consumes_generated_history():
    model = small_model()
    ids = torch.tensor([[12, 20, 21], [18, 16, 17]])
    expected = ids.clone()
    for _ in range(4):
        token = model(expected)[:, -1].argmax(-1)
        expected = torch.cat((expected, token[:, None]), 1)
    assert torch.equal(model.generate(ids, 4), expected)


def test_carrier_is_checkpoint_bound_and_can_be_restored():
    model = small_model()
    ids = torch.tensor([[12, 20, 21, 22, 23]])
    carrier = model.capture_carrier(ids, 3)
    original = model.continue_from_carrier(carrier, ids[:, 3:])
    saved = {key: value.clone() for key, value in carrier["state_planes"].items()}
    for value in carrier["state_planes"].values():
        value.zero_()
    ablated = model.continue_from_carrier(carrier, ids[:, 3:])
    assert not torch.equal(original, ablated)
    carrier["state_planes"] = saved
    torch.testing.assert_close(
        original, model.continue_from_carrier(carrier, ids[:, 3:]), rtol=0, atol=0
    )
    with torch.no_grad():
        model.head.bias[0] += 1
    with pytest.raises(ValueError, match="checkpoint"):
        model.continue_from_carrier(carrier, ids[:, 3:])


def test_empty_chunk_and_invalid_input_contract():
    model = small_model()
    assert model(torch.empty((2, 0), dtype=torch.long)).shape == (2, 0, 256)
    with pytest.raises(ValueError, match="integer"):
        model(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="boolean"):
        model.forward_reference(torch.ones(2, 3, dtype=torch.long), reset_before=torch.zeros(2, 3))


def test_native_export_uses_canonical_feature_program_and_codec():
    pytest.importorskip("spf.tensor.sequence_mixer")
    from spf.tensor.core import decode_words, unpack_theta
    from spf.tensor.sequence_mixer import execute_sequence_mixer
    from spf.tensor.sequence_mixer_reference import apply_sequence_mixer_reference

    model = small_model()
    ids = torch.tensor([[1, 2, 3, 4]])
    _, _, details = model.forward_reference(ids, capture=True)
    layer = model.layers[0]
    native = layer.export_native_inputs(details["layers"][0])
    assert native["values"].dtype == np.uint32
    assert native["fallback_conversions"] == native["values"].size + native["addresses"].size + 1
    program = native["program"]
    features = program.features.execute(native["addresses"])
    # Fixed omega=1 and selectors; confirms executable sparse feature semantics.
    np.testing.assert_array_equal(
        unpack_theta(native["parameters"].forward.a), unpack_theta(features)[..., :3]
    )
    packed = execute_sequence_mixer(native["values"], native["parameters"], accumulator="packed")
    clocked = program.execute(native["addresses"], native["values"], accumulator="clocked")
    ideal = apply_sequence_mixer_reference(native["values"], native["parameters"])
    assert packed.words.shape == clocked.words.shape == ideal.values.shape == (1, 4, 6)
    assert packed.counters.fallback_conversions == clocked.counters.fallback_conversions == 0
    assert np.isfinite(decode_words(packed.words)).all()
    assert np.isfinite(decode_words(clocked.words)).all()
    # This test verifies mechanics only; quantization drift is an observation.

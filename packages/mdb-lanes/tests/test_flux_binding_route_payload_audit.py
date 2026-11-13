"""Independent source-state transport and observed factorial arithmetic."""
from pathlib import Path
import runpy

import numpy as np
import pytest


HERE = Path(__file__).parents[1] / 'qualification'
H = runpy.run_path(str(HERE / 'flux_binding_route_payload_audit.py'))
BF = H['helpers']()
RGB = runpy.run_path(str(HERE / 'flux_binding_matched_dose_audit.py'))['rgb_response']


def test_transport_preserves_exact_endpoint_bits_and_ordered_delta_rows():
    base = np.array([[[0, 0x3f80], [0x4000, 0]]], np.uint16)
    source = np.array([[[0x8000, 0x4000], [0x4080, 0x3f80]]], np.uint16)
    np.testing.assert_array_equal(H['transport_rows'](base, base, source, 0., BF), base)
    np.testing.assert_array_equal(H['transport_rows'](base, base, source, 1., BF), source)
    destination = BF['round_bf16'](np.array([[[10, 20], [30, 40]]], np.float32))
    source_base = np.zeros((1, 2, 2), np.uint16)
    source_value = BF['round_bf16'](np.array([[[1, 2], [3, 4]]], np.float32))
    actual = H['transport_rows'](destination, source_base, source_value, .5, BF)
    np.testing.assert_array_equal(BF['decode_bf16'](actual), np.array([[[10.5, 21], [31.5, 42]]], np.float32))


def test_transport_refuses_nonfinite_or_inconsistent_source_geometry():
    base = np.zeros((1, 2, 2), np.uint16)
    with pytest.raises(ValueError):
        H['transport_rows'](base, base[:, :1], base, 1., BF)
    with pytest.raises(ValueError):
        H['transport_rows'](base, base, base, float('nan'), BF)
    bad = base.copy(); bad[0, 0, 0] = 0x7f80
    with pytest.raises(ValueError):
        H['transport_rows'](base, base, bad, 1., BF)


def test_factorial_signs_separate_payload_destination_and_interaction():
    sham = np.zeros((2, 4, 3), np.uint8)
    source = sham.copy(); source[:, :, 0] = 10
    def red(value):
        image = sham.copy(); image[:, :, 0] = value
        return image
    images = {key: red(value) for key, value in [('A@A', 5), ('A@B', 2), ('B@A', 1), ('B@B', 0)]}
    result = H['factorial_metrics'](images, sham, sham, source, RGB)
    expected = dict(payload_effect_at_A=.4, payload_effect_at_B=.2,
                    destination_effect_for_A=.3, destination_effect_for_B=.1, interaction=.2)
    for name, value in expected.items():
        actual = result['contrasts'][name]['whole']
        assert np.isclose(actual['signed_increment_on_natural_axis'], value)
        assert actual['axis_orthogonal_response_rms_0_255'] == 0


def test_ordered_direction_cosine_norms_and_row_order():
    base = np.zeros((1, 2, 2), np.uint16)
    left = BF['round_bf16'](np.array([[[3, 4], [0, 0]]], np.float32))
    right = BF['round_bf16'](np.array([[[0, 0], [0, 5]]], np.float32))
    result = H['ordered_direction_comparison'](base, left, base, right, BF)
    assert result['ordered_delta_cosine'] == 0.
    assert result['left_l2'] == result['right_l2'] == 5.
    assert result['left_ordered_row_l2'] == [5., 0.]
    assert result['right_ordered_row_l2'] == [0., 5.]
    twice = BF['round_bf16'](np.array([[[6, 8], [0, 0]]], np.float32))
    result = H['ordered_direction_comparison'](base, twice, base, left, BF)
    assert result['ordered_delta_cosine'] == 1.
    assert result['left_to_right_l2_ratio'] == 2.


def test_ordered_direction_zero_and_nonfinite_handling():
    base = np.zeros((1, 2, 2), np.uint16)
    one = BF['round_bf16'](np.ones((1, 2, 2), np.float32))
    result = H['ordered_direction_comparison'](base, one, base, base, BF)
    assert result['ordered_delta_cosine'] is None
    assert result['cosine_status'] == 'undefined-zero-delta-direction'
    assert result['left_to_right_l2_ratio'] is None
    with pytest.raises(ValueError):
        H['ordered_direction_comparison'](base, one[:, :1], base, base, BF)
    bad = base.copy(); bad[0, 0, 0] = 0x7f80
    with pytest.raises(ValueError):
        H['ordered_direction_comparison'](base, bad, base, one, BF)

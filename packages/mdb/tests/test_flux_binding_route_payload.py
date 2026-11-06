"""Small source-only mechanics for destination-relative payload transport."""
from pathlib import Path
import runpy
import numpy as np
import pytest

HERE = Path(__file__).parents[1] / 'qualification'
H = runpy.run_path(str(HERE / 'flux_binding_route_payload_math.py'))
BF = H['helpers']()


def bits(value):
    return BF['round_bf16'](np.asarray(value, np.float32).reshape(1, 2, 2))


def test_transport_preserves_direction_at_another_destination():
    base = bits([1, 2, 3, 4]); source = bits([2, 4, 3, 4]); destination = bits([8, 6, 4, 2])
    result = H['transport'](destination, base, source, .5)
    np.testing.assert_array_equal(BF['decode_bf16'](result), np.array([[[8.5, 7], [4, 2]]]))
    np.testing.assert_array_equal(base, bits([1, 2, 3, 4]))


def test_exact_endpoint_retains_signed_zero_and_norm():
    base = bits([0, 1, 0, 0]); source = bits([0, 2, 0, 0]); source[0, 0, 0] = 0x8000
    alpha, actual, measured = H['match'](base, base, source, 1.)
    assert alpha == 1. and measured == 1.
    np.testing.assert_array_equal(actual, source)
    np.testing.assert_array_equal(H['transport'](base, base, source, 0.), base)


def test_match_uses_landed_bits_and_handles_zero():
    base = bits([0, 0, 0, 0]); source = bits([3, 4, 0, 0]); destination = bits([8, 8, 8, 8])
    alpha, value, measured = H['match'](destination, base, source, 2.5)
    assert alpha > 0 and measured == H['norm'](destination, value)
    assert abs(measured - 2.5) < .04
    a, value, measured = H['match'](destination, base, source, 0)
    assert a == measured == 0
    np.testing.assert_array_equal(value, destination)


def test_nonfinite_and_impossible_doses_refuse():
    base = bits([0, 0, 0, 0])
    with pytest.raises(ValueError): H['match'](base, base, base, 1.)
    with pytest.raises(ValueError): H['match'](base, base, base, float('nan'))
    with pytest.raises(ValueError): H['transport'](base, base, base, -.1)

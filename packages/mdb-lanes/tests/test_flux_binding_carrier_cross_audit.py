"""Saved carrier-cross image arithmetic and typed trajectory distinctions."""
from pathlib import Path
import runpy

import numpy as np


HERE = Path(__file__).parents[1] / 'qualification'
H = runpy.run_path(str(HERE / 'flux_binding_carrier_cross_audit.py'))
BF = H['helpers']()
MH = runpy.run_path(str(HERE / 'flux_binding_matched_dose_audit.py'))


def test_carrier_arm_selection_and_reference_closeness_are_signed():
    assert H['source_letters']('I.Q_T.P') == ('Q', 'P')
    base = np.zeros((2, 4, 3), np.uint8)
    source = np.full_like(base, 20)
    result = H['reference_closeness'](source, source, base)
    assert result['rms_to_same_text_reference_0_255'] == 0
    assert result['other_minus_same_rms_0_255'] == 20


def test_carrier_factorial_separates_text_image_and_interaction():
    base = np.zeros((2, 4, 3), np.uint8)
    source = base.copy(); source[:, :, 0] = 10
    def red(value):
        result = base.copy(); result[:, :, 0] = value
        return result
    images = {arm: red(value) for arm, value in [('I.P_T.P', 5), ('I.P_T.Q', 1), ('I.Q_T.P', 3), ('I.Q_T.Q', 0)]}
    result = H['carrier_factorial'](images, base, source, MH['rgb_response'])['contrasts']
    expected = dict(text_effect_with_I_P=.4, text_effect_with_I_Q=.3,
                    image_effect_with_T_P=.2, image_effect_with_T_Q=.1, interaction=.1)
    for name, value in expected.items():
        assert np.isclose(result[name]['whole']['signed_increment_on_natural_axis'], value)


class FakeEvidence:
    def __init__(self, descriptors):
        self.descriptors = descriptors
    def cut(self, fingerprint):
        return {}, {'text': self.descriptors[fingerprint]}


def test_project_boundary_nontensor_is_not_a_tensor_zero():
    tensor = dict(kind='tensor', dtype='torch.bfloat16', shape=[1, 2, 2], sha256='same')
    absent = dict(kind='scalar', value=None)
    evidence = FakeEvidence(dict(P=absent, Q=absent, T=tensor))
    def unavailable(*args):
        raise AssertionError('nontensor/exact-descriptor comparison must not read a tensor')
    result = H['state_field_comparison'](evidence, 'P', 'Q', 'text', BF, MH, unavailable)
    assert result['status'] == 'not-applicable-nontensor'
    assert result['descriptors_exact'] and result['l2'] is None and result['raw_bits_exact'] is None
    mixed = H['state_field_comparison'](evidence, 'P', 'T', 'text', BF, MH, unavailable)
    assert mixed['status'] == 'not-comparable-tensor-versus-nontensor'
    exact = H['state_field_comparison'](evidence, 'T', 'T', 'text', BF, MH, unavailable)
    assert exact['status'] == 'compared-tensor' and exact['l2'] == 0 and exact['raw_bits_exact']


def test_trajectory_tensor_delta_and_nonmatching_geometry_are_explicit():
    descriptor = lambda sha, shape: dict(kind='tensor', dtype='torch.bfloat16', shape=shape, sha256=sha)
    evidence = FakeEvidence(dict(P=descriptor('left', [1, 2, 2]), Q=descriptor('right', [1, 2, 2]), U=descriptor('other', [1, 3, 2])))
    bits = dict(P=np.zeros((1, 2, 2), np.uint16), Q=BF['round_bf16'](np.array([[[3, 4], [0, 0]]], np.float32)))
    def slot_bits(owner, fingerprint, slot):
        return bits[fingerprint]
    result = H['state_field_comparison'](evidence, 'P', 'Q', 'text', BF, MH, slot_bits)
    assert result['status'] == 'compared-tensor' and result['l2'] == 5 and result['rms'] == 2.5
    different = H['state_field_comparison'](evidence, 'P', 'U', 'text', BF, MH, slot_bits)
    assert different['status'] == 'not-comparable-tensor-geometry-or-dtype' and different['l2'] is None

"""Saved BF16 dose diagnostics do not certify an energy-matched control."""
from pathlib import Path
import runpy

import numpy as np
import pytest


H = runpy.run_path(str(Path(__file__).parents[1] / 'qualification/flux_binding_context_controls_audit.py'))


def test_conditioning_dose_has_explicit_norms_bytes_and_zero_denominator():
    base = np.zeros((1, 4, 2), np.uint16)
    source = (np.array([[[2, 0], [0, 0], [3, 4], [0, 0]]], np.float32).view(np.uint32) >> 16).astype(np.uint16)
    result = H['conditioning_source_dose'](base, source, [2, 3], [0, 1])
    assert result['contextual_rows']['l2'] == 5
    assert result['contextual_rows']['rms'] == 2.5
    assert result['contextual_to_original_l2_ratio'] == 2.5
    assert result['contextual_to_original_rms_ratio'] == 2.5
    assert result['contextual_rows']['different_bf16_elements'] == 2
    # 3.0 is 0x4040, 4.0 is 0x4080: four actually different bytes.
    assert result['contextual_rows']['actually_different_payload_bytes'] == 4
    assert result['contextual_rows']['storage_bytes_occupied_by_different_bf16_elements'] == 4
    # 2.0 is 0x4000: one changed byte, occupying one two-byte BF16 word.
    assert result['original_changed_id_rows']['actually_different_payload_bytes'] == 1
    assert result['original_changed_id_rows']['storage_bytes_occupied_by_different_bf16_elements'] == 2
    source[:, :2] = 0
    result = H['conditioning_source_dose'](base, source, [2, 3], [0, 1])
    assert result['contextual_to_original_l2_ratio'] is None
    assert result['ratio_status'] == 'undefined-original-changed-id-dose-is-zero'


def test_conditioning_dose_refuses_bad_shape_or_nonfinite_source():
    base = np.zeros((1, 4, 2), np.uint16)
    with pytest.raises(ValueError):
        H['conditioning_source_dose'](base, base[:, :3], [2], [0])
    source = base.copy()
    source[0, 2, 0] = 0x7f80
    with pytest.raises(ValueError):
        H['conditioning_source_dose'](base, source, [2], [0])

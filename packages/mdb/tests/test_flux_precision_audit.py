"""Small endpoint errors must not hide errors in a weak receiver effect."""
import runpy
from pathlib import Path

import numpy as np


def test_receiver_difference_is_measured_against_its_own_scale():
    metric = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'qualification' / 'analyze_flux_precision.py'))['error']
    actual = np.array([100., 100.125])
    predicted = np.array([100., 101.])
    assert metric(predicted, actual)['relative_rms'] < .01
    effect = metric(np.diff(predicted), np.diff(actual))
    assert effect['relative_rms'] == 7.
    assert effect['cosine'] == 1.
    zero = metric(np.zeros(2), np.zeros(2))
    assert zero['exact'] and zero['relative_rms'] is None
    assert zero['cosine'] is None

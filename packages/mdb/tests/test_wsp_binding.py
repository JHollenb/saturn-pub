import numpy as np
import pytest

from mdb.wsp_binding import fit_response_selector


def test_selector_generalizes_role_condition_while_retaining_response_direction():
    # Known-wiring organism: one role feature plus a nuisance coordinate.
    # Held-out nuisance values test the role condition rather than interpolation.
    h = np.array([[1., 0., 2.], [1., 0., -2.]])
    k = np.array([[0., 1., 2.], [0., 1., -2.]])
    u = np.array([3., -4.])
    left, right, _ = fit_response_selector(h, u, k, ridge=1e-8)
    np.testing.assert_array_equal(left[:, 0], u)
    np.testing.assert_allclose(np.array([[1., 0., 19.]]) @ right.T @ left.T, [u], atol=3e-8)
    np.testing.assert_allclose(np.array([[0., 1., -31.]]) @ right.T @ left.T, [[0., 0.]], atol=1e-12)


def test_selector_refuses_broken_instruments():
    with pytest.raises(ValueError):
        fit_response_selector([[1., 2.]], [0., 0.], [[2., 1.]])
    with pytest.raises(ValueError):
        fit_response_selector([[1., float("nan")]], [1.], [[2., 1.]])
    with pytest.raises(ValueError):
        fit_response_selector([[1., 2.]], [1.], [[2.]], ridge=0.)

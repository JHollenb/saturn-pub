"""Finite MLP/write equations; no model framework, fit or target lookup."""
import numpy as np


def same_bits(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype and
        np.array_equal(np.ascontiguousarray(left).view(np.uint8), np.ascontiguousarray(right).view(np.uint8)))


def equality(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return dict(numeric_exact=False, raw_bits_exact=False, shape_dtype_match=False)
    return dict(numeric_exact=bool(np.array_equal(left,right)), raw_bits_exact=bool(same_bits(left,right)), shape_dtype_match=True,
        different_values=int(np.count_nonzero(left!=right)),
        signed_zero_differences=int(np.count_nonzero((left==0)&(right==0)&(np.signbit(left)!=np.signbit(right)))))


def mlp(value, stages, weights, first, reference, law):
    if first == 'anchored':
        paired = law['affine_delta'](value, stages['mlp_input'], stages['mlp_paired'], weights['in'], reference['bf16'])[0]
    elif first == 'direct':
        paired = reference['_linear'](value, {'w.weight': weights['in']}, 'w', 256)
    else:
        raise ValueError('predeclared first-affine law required')
    u, v = np.split(paired, 2, axis=-1)
    activation = reference['bf16'](reference['_silu'](u) * v)
    return law['affine_delta'](activation, stages['mlp_activation'], stages['mlp'], weights['out'], reference['bf16'])[0]


def writeback(updated, gate, mlp_write, bf16):
    if updated.shape != mlp_write.shape or gate.shape != (1, 1, updated.shape[-1]):
        raise ValueError('receiving-state/time-gate geometry differs')
    gated = bf16(gate * mlp_write)
    return dict(after_attention=updated, mlp=mlp_write, gated=gated, exit=bf16(updated + gated))


def power_account(components, total):
    total = np.asarray(total, np.float64)
    terms = {name: np.asarray(value, np.float64) for name, value in components.items()}
    if not total.size or not np.isfinite(total).all() or not terms or any(value.shape != total.shape or not np.isfinite(value).all() for value in terms.values()):
        raise ValueError('finite same-space terms required')
    names = list(terms)
    powers = {name: float(np.mean(value * value)) for name, value in terms.items()}
    cross = {a + ' × ' + b: float(2 * np.mean(terms[a] * terms[b])) for i, a in enumerate(names) for b in names[i + 1:]}
    return dict(component_mse=powers, twice_signed_cross_moments=cross, total_mse=float(np.mean(total * total)),
        summed_power=sum(powers.values()) + sum(cross.values()),
        identity_residual_rms=float(np.sqrt(np.mean((total - sum(terms.values())) ** 2))),
        scope='Finite same-output-space vector account. Cross terms retain cancellation; terms are not positive causal shares.')


def differences(zero, one):
    U0, V0, Y0 = [np.asarray(zero[key], np.float64) for key in ('after_attention', 'gated', 'exit')]
    U1, V1, Y1 = [np.asarray(one[key], np.float64) for key in ('after_attention', 'gated', 'exit')]
    return dict(after_attention=U1-U0, time_gated_mlp=V1-V0,
        finite_addition=(Y1-U1-V1)-(Y0-U0-V0)), Y1-Y0


def transport(native0, native1, predicted0, predicted1):
    native, nd = differences(native0, native1)
    predicted, pd = differences(predicted0, predicted1)
    return dict(native_response=power_account(native, nd), predicted_response=power_account(predicted, pd),
        prediction_error=power_account({key: predicted[key]-native[key] for key in native}, pd-nd))

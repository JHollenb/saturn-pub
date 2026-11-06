"""Source-only BF16 payload transport; no model execution or future fitting."""
import math
from pathlib import Path
import runpy


def helpers():
    return runpy.run_path(str(Path(__file__).with_name('flux_binding_generalization_audit.py')))


def transport(destination_base, source_base, source_target, alpha):
    import numpy as np
    arrays = (destination_base, source_base, source_target)
    if (any(x.dtype != np.uint16 or x.ndim != 3 for x in arrays)
            or any(x.shape != destination_base.shape for x in arrays)
            or not destination_base.size or not math.isfinite(alpha) or alpha < 0):
        raise ValueError('matched finite BF16 row arrays and nonnegative alpha required')
    h = helpers()
    d, b, s = [h['decode_bf16'](x).astype(np.float64) for x in arrays]
    if not all(np.isfinite(x).all() for x in (d, b, s)):
        raise ValueError('nonfinite transport operand')
    if alpha == 0:
        return destination_base.copy()
    if alpha == 1 and np.array_equal(destination_base, source_base):
        return source_target.copy()
    return h['round_bf16']((d + float(alpha) * (s - b)).astype(np.float32))


def norm(base, value):
    import numpy as np
    decode = helpers()['decode_bf16']
    delta = decode(value).astype(np.float64) - decode(base).astype(np.float64)
    if not np.isfinite(delta).all():
        raise ValueError('nonfinite delivered text dose')
    return float(np.sqrt(np.sum(delta * delta)))


def match(destination_base, source_base, source_target, target):
    """Choose an amplitude from supplied arrays, preserving exact endpoints."""
    if not math.isfinite(target) or target < 0:
        raise ValueError('finite nonnegative target norm required')
    endpoint = transport(destination_base, source_base, source_target, 1.)
    endpoint_norm = norm(destination_base, endpoint)
    if target == 0:
        return 0., destination_base.copy(), 0.
    if norm(source_base, source_target) == 0:
        raise ValueError('zero source direction cannot supply a nonzero dose')
    if endpoint_norm == target:
        return 1., endpoint, endpoint_norm
    low, high = 0., max(1., 2 * target / norm(source_base, source_target))
    while norm(destination_base, transport(destination_base, source_base, source_target, high)) < target:
        high *= 2
        if high > 1024:
            raise ValueError('dose bracket exceeds declared source-only bound')
    choices = [(abs(endpoint_norm - target), 1., endpoint, endpoint_norm)]
    for _ in range(48):
        alpha = (low + high) / 2
        value = transport(destination_base, source_base, source_target, alpha)
        actual = norm(destination_base, value)
        choices.append((abs(actual - target), alpha, value, actual))
        if actual < target:
            low = alpha
        else:
            high = alpha
    _, alpha, value, actual = min(choices, key=lambda row: (row[0], row[1]))
    return alpha, value, actual

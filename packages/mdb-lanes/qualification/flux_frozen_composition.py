"""Frozen source-only contextual composition and supplied-window color diagnostics."""
import numpy as np


def forecast_frontier(binding):
    return {'kind': 'architectural', 'binding': binding,
            'visibility': 'Observed three-source conditioning and actual projection weights; new joint encoding and native denoiser states unavailable'}


def bf16(value):
    bits = np.asarray(value, np.float32).view(np.uint32)
    return ((bits + np.uint32(32767) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xffff0000)).view(np.float32)


def compose(values, old_joint, old_additive):
    if set(values) != {'base', 'color', 'shape'}:
        raise ValueError('Only three source encodings may enter the frozen rule')
    arrays = [*values.values(), old_joint, old_additive]
    if any(v.shape != arrays[0].shape or v.dtype != np.float32 or not np.isfinite(v).all() for v in arrays):
        raise ValueError('Finite, equal-shape widened BF16 operands required')
    additive = bf16(values['color'].astype(np.float64) + values['shape'] - values['base'])
    correction = old_joint.astype(np.float64) - old_additive
    return {'additive': additive, 'plus_old': bf16(additive.astype(np.float64) + correction), 'minus_old': bf16(additive.astype(np.float64) - correction)}


def hue_observer(rgb):
    """Uncalibrated chromatic pixel counts; no object or geometry authority."""
    v = np.asarray(rgb, np.float64) / 255
    maximum, minimum = v.max(-1), v.min(-1)
    delta = maximum - minimum
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 0)
    hue = np.zeros_like(delta)
    for channel, offset in ((0, 0), (1, 2), (2, 4)):
        mask = (v.argmax(-1) == channel) & (delta > 0)
        a, b = ((1, 2), (2, 0), (0, 1))[channel]
        hue[mask] = ((v[..., a][mask] - v[..., b][mask]) / delta[mask] + offset) * 60
    hue %= 360
    chromatic = (saturation > .20) & (maximum > .10)
    return {'pixels': int(hue.size), 'chromatic': int(chromatic.sum()), 'gold_band': int((chromatic & (hue >= 20) & (hue < 80)).sum()), 'violet_band': int((chromatic & (hue >= 240) & (hue < 320)).sum()), 'scope': 'Supplied display window; saturation/value and hue bands are diagnostic choices, not calibrated semantic gates'}

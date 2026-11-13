"""Chunked full-vocabulary Qwen readout and exact affine winner envelope.

No model/native imports. The positive RMS denominator cannot change the
winning vocabulary row. Hull intersections use exact rational arithmetic on
the supplied FP64 affine coefficients; numerical logits remain FP64.
"""
from __future__ import annotations

from fractions import Fraction
from numbers import Real

import numpy as np


def upper_envelope(a, b):
    """O(V log V) upper envelope on [0,1], lowest token ID at every tie."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite nonempty affine coefficient vectors required")
    order = sorted(range(len(a)), key=lambda index: (float(b[index]), -float(a[index]), index))
    unique, equivalent = [], {}
    for index in order:
        if unique and b[index] == b[unique[-1]]:
            if a[index] == a[unique[-1]]:
                equivalent[unique[-1]].append(index)
            continue
        unique.append(index)
        equivalent[index] = [index]
    hull, touches = [], {}
    for index in unique:
        start = None
        while hull:
            previous, previous_start = hull[-1]
            start = (Fraction(float(a[previous])) - Fraction(float(a[index]))) / (Fraction(float(b[index])) - Fraction(float(b[previous])))
            if previous_start is None or start > previous_start:
                break
            if start == previous_start:
                touches.setdefault(start, set()).update((previous, index))
            hull.pop()
        hull.append((index, start if hull else None))
    segments, points = [], {Fraction(0), Fraction(1)}
    for position, (index, start) in enumerate(hull):
        left = max(Fraction(0), start) if start is not None else Fraction(0)
        stop = hull[position + 1][1] if position + 1 < len(hull) else None
        right = min(Fraction(1), stop) if stop is not None else Fraction(1)
        if left < right:
            segments.append({"left": float(left), "right": float(right), "token_id": int(index),
                             "slope": float(b[index]), "intercept": float(a[index]),
                             "equivalent_token_ids": sorted(equivalent[index])})
            points.update((left, right))
    breakpoints = []
    cursor = 0
    for point in sorted(points):
        while cursor + 1 < len(hull) and hull[cursor + 1][1] < point:
            cursor += 1
        active = [hull[cursor][0]]
        if cursor + 1 < len(hull) and hull[cursor + 1][1] == point:
            active.append(hull[cursor + 1][0])
        candidates = set(active) | touches.get(point, set())
        reference = active[0]
        height = Fraction(float(a[reference])) + point * Fraction(float(b[reference]))
        tied = sorted(token for index in candidates
                      if Fraction(float(a[index])) + point * Fraction(float(b[index])) == height
                      for token in equivalent[index])
        breakpoints.append({"dose": float(point), "dose_exact": str(point), "token_id": min(tied), "tied_token_ids": tied})
    return {"domain": [0., 1.], "segments": segments, "breakpoints": breakpoints,
            "tie_policy": "lowest vocabulary token ID; exact rational intersections of FP64 coefficients",
            "normalizer_scope": "positive shared RMS denominator does not change the affine winner"}


def full_readout(cut, delta, gamma, head, eps, doses=(1., 0., .5), *, head_bias=None,
                 include_logits=False, chunk_rows=4096):
    """Return a=W(gamma*r), b=W(gamma*m), doses and the full winner envelope.

    ``head`` must be a caller-owned read-only FP32 [V,H] array. It is scanned
    in chunks of at most 4096 rows; no full FP64 head matrix is materialized or
    returned. ``cut`` and ``delta`` are [1,1,H], gamma is [H]. Each dose logit
    array, when requested, has shape [1,1,V]. All returned arrays are read-only.
    """
    if head_bias is not None:
        raise ValueError("the readout law requires an explicitly bias-free head")
    if not isinstance(head, np.ndarray) or head.dtype != np.float32 or head.ndim != 2 or not head.shape[0] or head.flags.writeable:
        raise ValueError("head must be a read-only nonempty FP32 NumPy [V,H] matrix")
    if type(chunk_rows) is not int or not 1 <= chunk_rows <= 4096:
        raise ValueError("head chunks must contain 1..4096 rows")
    if isinstance(eps, bool) or not isinstance(eps, Real) or not np.isfinite(eps) or eps <= 0:
        raise ValueError("positive finite RMS epsilon required")
    r, m, gamma = (np.array(value, dtype=np.float64, copy=True) for value in (cut, delta, gamma))
    if r.shape != (1, 1, head.shape[1]) or m.shape != r.shape or gamma.shape != (head.shape[1],) or not head.shape[1]:
        raise ValueError("cut/delta [1,1,H], gamma [H] and head [V,H] geometry differs")
    if any(not np.isfinite(value).all() for value in (r, m, gamma)):
        raise ValueError("nonfinite cut, delta or gamma")
    doses = tuple(doses)
    if not doses or any(isinstance(dose, bool) or not isinstance(dose, Real) or not np.isfinite(dose) for dose in doses):
        raise ValueError("declare finite numerical doses")
    keys = [format(float(dose), ".17g") if dose else "0" for dose in doses]
    if len(keys) != len(set(keys)):
        raise ValueError("dose values must be distinct")
    a, b = np.empty(head.shape[0], dtype=np.float64), np.empty(head.shape[0], dtype=np.float64)
    weighted_r, weighted_m = (gamma * value.reshape(-1) for value in (r, m))
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, head.shape[0], chunk_rows):
            block = np.array(head[start:start + chunk_rows], dtype=np.float64, copy=True)
            if not np.isfinite(block).all():
                raise ValueError("nonfinite readout matrix")
            a[start:start + chunk_rows] = block @ weighted_r
            b[start:start + chunk_rows] = block @ weighted_m
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite affine readout arithmetic")
    predictions = {}
    for key, dose in zip(keys, doses, strict=True):
        with np.errstate(over="ignore", invalid="ignore"):
            normalizer = float(np.sqrt(np.mean(np.square(r + float(dose) * m), dtype=np.float64) + eps))
            logits = (a + float(dose) * b) / normalizer
        if not np.isfinite(normalizer) or normalizer <= 0 or not np.isfinite(logits).all():
            raise ValueError("nonfinite RMS/readout dose arithmetic")
        ranking = np.argsort(-logits, kind="stable")
        first, second = int(ranking[0]), int(ranking[1]) if len(ranking) > 1 else None
        predictions[key] = {"dose": float(dose), "token_id": first, "runner_up_token_id": second,
                            "top2gap": None if second is None else float(logits[first] - logits[second]),
                            "normalizer": normalizer}
        if include_logits:
            logits = logits.reshape(1, 1, -1)
            logits.flags.writeable = False
            predictions[key]["logits"] = logits
    a.flags.writeable = b.flags.writeable = False
    return {"a": a, "b": b, "vocab_size": int(head.shape[0]), "doses": predictions,
            "envelope": upper_envelope(a, b), "rms_epsilon": float(eps),
            "arithmetic": "independent FP64 affine/RMS law; native FP32 rounding not emulated",
            "matrix_storage": "caller-bound read-only FP32 head, chunked widening; head not returned",
            "max_chunk_rows": chunk_rows}

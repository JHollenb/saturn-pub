"""Shared analysis metrics that resist the usual single-number traps (numpy only).

* ``relative_error``: uncentred relative MSE (the historical definition) returned together with
  centred relative MSE, both again with position 0 excluded, and the fraction of target energy at
  position 0. A residual stream whose first token carries a massive activation makes the
  uncentred number a position-0 score; the companions make that visible.
* ``fit_mean_writes`` / ``mean_write_forecast``: global mean write and position-split mean write
  (one mean for position 0, one for the rest), fit on discovery documents only.
* ``Bounded`` / ``margin`` / ``combine`` / ``flag_effects``: margin arithmetic with a rounding
  bound that propagates through differences and interaction terms; any reported effect not
  larger than its bound is flagged unresolved.
* ``correlation_report``: Pearson, Spearman and the leave-one-out range together.
"""
from __future__ import annotations

import math
import numpy as np

# --------------------------------------------------------------------------- relative error


def _rows(value, position_axis):
    a = np.moveaxis(np.asarray(value, dtype=np.float64), position_axis, 0)
    return a.reshape(a.shape[0], -1)


def relative_error(prediction, target, *, position_axis=0):
    """Error of ``prediction`` against ``target`` with position-robust companions.

    ``relative_mse`` = mse / mean(target^2) (uncentred; the historical field).
    ``centered_relative_mse`` = mse / mean((target - per-feature mean over positions)^2), i.e.
    error relative to the target's variation across positions (1 - R^2 against the document's
    own per-feature constant). ``*_excluding_position_0`` repeat both on positions 1.. only.
    ``position_0_energy_fraction`` = sum(target[0]^2) / sum(target^2).
    """
    p, t = _rows(prediction, position_axis), _rows(target, position_axis)
    if p.shape != t.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("metric requires matching finite arrays")
    if t.size == 0:
        raise ValueError("metric requires at least one scalar")
    err = np.square(p - t)
    energy = np.square(t)

    def ratio(num, den):
        return float(num / den) if den > 1e-20 else None

    def block(e, tt):
        mse = float(e.mean())
        centered = np.square(tt - tt.mean(0, keepdims=True)).mean()
        return mse, ratio(mse, float(np.square(tt).mean())), ratio(mse, float(centered))

    mse, _, cen = block(err, t)
    if len(t) > 1:
        _, rel1, cen1 = block(err[1:], t[1:])
    else:
        rel1 = cen1 = None
    total = float(energy.sum())
    frac0 = float(energy[0].sum() / total) if total > 0 else None
    return {"mse": mse, "relative_mse": mse / max(float(energy.mean()), 1e-20),
            "max_abs": float(np.abs(p - t).max()), "scalars": int(t.size),
            "centered_relative_mse": cen,
            "relative_mse_excluding_position_0": rel1,
            "centered_relative_mse_excluding_position_0": cen1,
            "position_0_energy_fraction": frac0,
            "positions": int(len(t))}


def fit_mean_writes(states, *, split_position=0):
    """Mean boundary-to-boundary writes from discovery documents only.

    ``states``: iterable of arrays ``[boundaries, positions, width]`` (one per discovery
    document). Returns ``global`` ``[spans, width]`` (mean over all positions, the historical
    baseline), ``position_0`` and ``rest`` (means for position ``split_position`` and for all
    other positions).
    """
    states = [np.asarray(s, np.float64) for s in states]
    if not states:
        raise ValueError("at least one discovery document is required")
    spans = states[0].shape[0] - 1
    if spans < 1 or any(s.ndim != 3 or s.shape[0] != spans + 1 for s in states):
        raise ValueError("states must be [boundaries, positions, width] with matching boundaries")
    writes = [s[1:] - s[:-1] for s in states]
    glob = np.stack([np.concatenate([w[i] for w in writes]).mean(0) for i in range(spans)])
    first = np.stack([np.stack([w[i][split_position] for w in writes]).mean(0) for i in range(spans)])
    rest = np.stack([np.concatenate([np.delete(w[i], split_position, axis=0) for w in writes]).mean(0)
                     for i in range(spans)])
    return {"global": glob, "position_0": first, "rest": rest, "split_position": split_position,
            "documents": len(states), "fit_scope": "discovery documents only"}


def mean_write_forecast(initial, fitted, *, split=False):
    """Cumulative mean-write forecast ``[spans, positions, width]`` from the initial frontier."""
    initial = np.asarray(initial, np.float64)
    if not split:
        return initial[None] + fitted["global"].cumsum(0)[:, None]
    out = initial[None] + fitted["rest"].cumsum(0)[:, None]
    k = fitted["split_position"]
    out[:, k] = initial[k] + fitted["position_0"].cumsum(0)
    return out


def mean_write_baselines(initial, native_states, fitted, *, position_axis=0):
    """``relative_error`` for global and position-split mean writes at every later boundary."""
    native_states = np.asarray(native_states)
    result = {}
    for name, split in (("mean_write", False), ("position_split_mean_write", True)):
        forecast = mean_write_forecast(initial, fitted, split=split)
        result[name] = [{"stage": i, **relative_error(forecast[i], native_states[i + 1],
                                                      position_axis=position_axis)}
                        for i in range(len(forecast))]
    return result


# --------------------------------------------------------------------------- margin bounds

_MANTISSA = {"bfloat16": 7, "bf16": 7, "float16": 10, "fp16": 10, "half": 10,
             "float32": 23, "fp32": 23, "float": 23, "float64": 52, "fp64": 52, "double": 52}
_MIN_EXP = {7: -126, 10: -14, 23: -126, 52: -1022}


def float_spacing(values, dtype):
    """Spacing between adjacent representable values at each ``|value|`` (one ULP).

    Uses the upper binade at exact powers of two, which is the conservative choice.
    """
    key = str(dtype).replace("torch.", "").lower()
    if key not in _MANTISSA:
        raise ValueError("unsupported dtype for spacing: " + str(dtype))
    m = _MANTISSA[key]
    v = np.abs(np.asarray(values, np.float64))
    exponent = np.floor(np.log2(np.where(v > 0, v, 1.0)))
    exponent = np.maximum(np.where(v > 0, exponent, _MIN_EXP[m]), _MIN_EXP[m])
    return np.ldexp(1.0, (exponent - m).astype(int))


class InferredStep(float):
    """A grid spacing inferred from values; a LOWER bound on the spacing of the larger values."""


def infer_step_lower_bound(values, *, max_power=8, min_power=-30):
    """Largest power of two dividing every value, as an ``InferredStep``.

    This is a lower bound on the true rounding spacing only: it is set by the finest-grained
    value, so it under-states the spacing of larger values (``[0.50390625, 20.0]`` gives 2^-8
    although 20.0 sits on a much coarser grid). ``rounding_bound(step=...)`` refuses it unless
    ``allow_inferred_step=True``; use ``dtype=`` for a worst-case bound.
    """
    v = np.asarray(values, np.float64).ravel()
    for p in range(max_power, min_power - 1, -1):
        step = math.ldexp(1.0, p)
        if np.all(np.mod(v, step) == 0):
            return InferredStep(step)
    return None


def infer_step(*args, **kwargs):
    """Deprecated alias of ``infer_step_lower_bound`` (a lower bound, not the spacing)."""
    return infer_step_lower_bound(*args, **kwargs)


def rounding_bound(values, *, dtype=None, step=None, allow_inferred_step=False):
    """Half-spacing round-to-nearest bound per value, from a dtype or a known step.

    A ``step`` produced by ``infer_step_lower_bound`` is refused (it under-states the spacing of
    larger values) unless ``allow_inferred_step=True``.
    """
    if (dtype is None) == (step is None):
        raise ValueError("pass exactly one of dtype or step")
    if isinstance(step, InferredStep) and not allow_inferred_step:
        raise ValueError("an inferred step is only a lower bound on the spacing; pass dtype= or a "
                         "known step, or allow_inferred_step=True to accept an under-stated bound")
    if step is not None:
        if step <= 0:
            raise ValueError("step must be positive")
        return np.full(np.shape(values), step / 2.0)
    return float_spacing(values, dtype) / 2.0


class Bounded:
    """A value with a worst-case absolute bound that propagates through linear arithmetic.

    Each base quantity is one term ``key -> (coefficient, unit_bound)``. Linear combinations
    merge terms by key, so a quantity that enters twice (for example the native margin in both
    the full effect and every singleton effect) cancels exactly instead of double-counting its
    bound. ``bound = sum(|coefficient| * unit_bound)``. Products of two bounded values are refused.
    """

    __slots__ = ("value", "terms", "base")
    _counter = [0]

    def __init__(self, value, bound=0.0, *, key=None, terms=None, base=None):
        self.value = float(value)
        self.base = dict(base or {})  # explicit key -> stored value, to catch one key, two values
        if terms is not None:
            self.terms = dict(terms)
            return
        if bound < 0 or not math.isfinite(bound):
            raise ValueError("bound must be finite and non-negative")
        if key is None:
            Bounded._counter[0] += 1
            key = ("anonymous", Bounded._counter[0])
        else:
            self.base[key] = self.value
        self.terms = {key: (1.0, float(bound))} if bound else {}

    @property
    def bound(self):
        return float(sum(abs(c) * b for c, b in self.terms.values()))

    def _merge(self, other, sign):
        other = _bounded(other)
        terms = dict(self.terms)
        base = dict(self.base)
        for key, value in other.base.items():
            if key in base and base[key] != value:
                raise ValueError(f"key {key!r} is reused with a different value "
                                 f"({base[key]!r} vs {value!r}); one key names one quantity")
            base[key] = value
        for key, (c, b) in other.terms.items():
            old = terms.get(key, (0.0, b))
            if old[1] != b:
                raise ValueError("one key carries two different unit bounds")
            terms[key] = (old[0] + sign * c, b)
        return Bounded(self.value + sign * other.value, terms=terms, base=base)

    def __add__(self, other):
        return self._merge(other, 1.0)

    __radd__ = __add__

    def __sub__(self, other):
        return self._merge(other, -1.0)

    def __rsub__(self, other):
        return _bounded(other) - self

    def __neg__(self):
        return self * -1.0

    def __mul__(self, scalar):
        if isinstance(scalar, Bounded):
            raise TypeError("product of two bounded values is not linear; bound it explicitly")
        return Bounded(self.value * scalar, terms={k: (c * scalar, b) for k, (c, b) in self.terms.items()},
                       base=self.base)

    __rmul__ = __mul__

    @property
    def resolved(self):
        return abs(self.value) > self.bound

    def as_dict(self):
        return {"value": self.value, "bound": self.bound, "resolved": self.resolved,
                "status": "resolved" if self.resolved else "inside rounding bound"}

    def __repr__(self):
        return f"Bounded({self.value!r} +/- {self.bound!r})"


def _bounded(value):
    return value if isinstance(value, Bounded) else Bounded(float(value), 0.0)


def bounded_values(values, *, dtype=None, step=None, keys=None):
    """Wrap each stored value with its rounding bound (``keys`` name the base quantities)."""
    v = np.asarray(values, np.float64)
    b = rounding_bound(v, dtype=dtype, step=step)
    keys = [None] * v.size if keys is None else list(keys)
    return [Bounded(float(x), float(y), key=k)
            for x, y, k in zip(v.ravel(), np.ravel(b), keys, strict=True)]


def margin(logits, a, b, *, dtype=None, step=None, source=None):
    """``logits[a] - logits[b]`` with the sum of both rounding bounds.

    ``source`` names the logit vector (e.g. an arm) so that the same logit reused in several
    margins is recognised as one quantity.
    """
    logits = np.asarray(logits, np.float64).reshape(-1)
    keys = None if source is None else [(source, int(a)), (source, int(b))]
    la, lb = bounded_values(logits[[a, b]], dtype=dtype, step=step, keys=keys)
    return la - lb


def combine(terms):
    """Linear combination ``sum(coef * Bounded)`` from ``(coef, Bounded)`` pairs."""
    total = Bounded(0.0, 0.0)
    for coef, value in terms:
        total = total + coef * _bounded(value)
    return total


def flag_effects(effects):
    """Rows ``{name, value, bound, resolved, status}`` for a mapping of named Bounded effects."""
    return [{"name": name, **_bounded(effect).as_dict()} for name, effect in effects.items()]


# --------------------------------------------------------------------------- correlation


def _ranks(x):
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def pearson(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    x, y = x - x.mean(), y - y.mean()
    den = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y / den) if den > 0 else None


def spearman(x, y):
    return pearson(_ranks(x), _ranks(y))


def correlation_report(x, y, *, labels=None):
    """Pearson, Spearman (average ranks for ties) and the leave-one-out range, together."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 3:
        raise ValueError("correlation needs two matching 1-D arrays with at least three rows")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("correlation needs finite values")
    labels = list(range(len(x))) if labels is None else list(labels)
    if len(labels) != len(x):
        raise ValueError("one label per row is required")
    rows = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        rows.append({"dropped": labels[i], "pearson": pearson(x[keep], y[keep]),
                     "spearman": spearman(x[keep], y[keep])})

    def span(key):
        vals = [(r[key], r["dropped"]) for r in rows if r[key] is not None]
        if not vals:
            return {"min": None, "max": None}
        lo, hi = min(vals, key=lambda v: v[0]), max(vals, key=lambda v: v[0])
        return {"min": lo[0], "min_dropped": lo[1], "max": hi[0], "max_dropped": hi[1]}

    return {"n": int(len(x)), "pearson": pearson(x, y), "spearman": spearman(x, y),
            "leave_one_out": {"pearson": span("pearson"), "spearman": span("spearman"), "rows": rows}}

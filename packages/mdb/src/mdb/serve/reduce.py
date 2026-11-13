"""Server-side reductions of one recorded tensor (numpy only). The browser never gets a tensor.

Ops: ``stats``; ``topk {axis, k<=64, by: abs|signed}``; ``slice {selector}`` (<= 4096 values,
float32 little-endian base64); ``histogram {bins<=256, range?}``; ``tile {axes, fixed, x0, y0,
level}`` (one 64 x 64 u8 tile, mean of 2^level x 2^level blocks, with its domain and scale);
``project {direction: pointer}`` (inner products over the last axis). Unknown parameter names
are refused (422 ``bad_params``). Results are cached by (session, pointer digest, op, sha256 of
the canonical known params), bounded by entry count and by total bytes of cached results.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import threading
from collections import OrderedDict

from ..core.address import AddressError, format_address
from ..core.address import parse as parse_address
from . import blobs
from .schema import (
    HistogramResult,
    NumList,
    ProjectResult,
    SliceResult,
    StatsResult,
    TileResult,
    TopkEntry,
    TopkResult,
    num,
    unresolved,
)
from .views.common import occurrence_row, spec_digest

MAX_TOPK, MAX_SLICE, MAX_BINS, TILE = 64, 4096, 256, 64
CACHE_BYTES = 16 * 1024 * 1024
PARAMS = {"stats": (), "topk": ("k", "by", "axis"), "slice": ("selector",),
          "histogram": ("bins", "range"), "tile": ("axes", "fixed", "x0", "y0", "level"),
          "project": ("direction",)}


class ReduceError(ValueError):
    def __init__(self, reason, detail, status=422):
        super().__init__(detail)
        self.reason, self.detail, self.status = reason, detail, status


def resolve(model, spec: dict):
    """``(array, info)`` for a pointer spec; raises ``ReduceError`` (unresolved or refused)."""
    import numpy as np

    store = model.store_root
    try:
        if spec["kind"] == "slot":
            array, info = blobs.cut_slot(store, spec["cut"], spec["slot"], spec.get("item"))
            for index in spec.get("select") or []:
                if array.ndim == 0 or not 0 <= index < array.shape[0]:
                    raise ReduceError("bad_selector", "select index is outside the slot tensor")
                array = array[index]
            return np.asarray(array), info
        if spec["kind"] == "blob":
            return blobs.tensor_blob(store, spec["ref"])
        if spec["kind"] == "occurrence":
            try:
                address = format_address(parse_address(spec["address"]))
            except AddressError as error:
                raise ReduceError("bad_address", str(error)) from error
            row = occurrence_row(model, address)
            if row is None:
                raise ReduceError("unresolved", f"{address} is not materialised in this project "
                                  "(no occurrence index row); producing it needs the engine")
            if not row.get("blob_id"):
                raise ReduceError("unresolved", f"{address} is indexed without a stored blob; "
                                  "rebuilding it needs recipe replay in the engine")
            array, info = blobs.tensor_blob(store, row["blob_id"])
            return array, {**info, "via": ["occurrence index", *info["via"]]}
    except blobs.BlobError as error:
        raise ReduceError("over_limit" if error.limit else
                          "verification_failed" if error.refused else "unresolved",
                          error.reason) from error
    raise ReduceError("bad_pointer", f"unknown pointer kind {spec.get('kind')!r}")


def _f(value):
    return float(value)


def _encode_f32(array):
    import numpy as np

    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode()


def _int(params, key, default, lo, hi):
    value = params.get(key, default)
    if value is None and default is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ReduceError("bad_params", f"{key} must be an integer in {lo}..{hi}")
    return value


def _stat(value, **kw):
    """A measured statistic, or ``unresolved`` when it overflows float64."""
    value = float(value)
    if not math.isfinite(value):
        return unresolved("overflows float64 (the exact value is not finite in double precision)",
                          **kw)
    return num(value, "measured", **kw)


def stats(array, base):
    import numpy as np

    values = np.asarray(array, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    count = num(values.size, "measured", unit="values")
    if finite.size == 0:
        none = unresolved("no finite values")
        return StatsResult(**base, count=count, finite=num(0, "measured", unit="values"),
                           min=none, max=none, mean=none, std=none, absmax=none, l2=none)
    m = dict(ptr=base["pointer"])
    top = float(np.abs(finite).max())
    with np.errstate(all="ignore"):
        mean = finite.mean()
        std = finite.std()
        l2 = top * np.sqrt(((finite / top) ** 2).sum()) if top > 0 else 0.0
    return StatsResult(**base, count=count, finite=num(finite.size, "measured", unit="values"),
                       min=_stat(finite.min(), **m), max=_stat(finite.max(), **m),
                       mean=_stat(mean, **m), std=_stat(std, **m), absmax=_stat(top, **m),
                       l2=_stat(l2, **m))


def _top_indices(score, k):
    """Indices of the ``k`` largest scores, ties broken by the lower index: exactly the first
    ``k`` of a stable descending full sort, without sorting everything (argpartition, then a sort
    of only the winners and the boundary ties)."""
    import numpy as np

    n = score.size
    if n <= k:
        return np.argsort(-score, kind="stable")
    top = np.argpartition(score, n - k)[n - k:]
    threshold = score[top].min()
    above = np.flatnonzero(score > threshold)
    ties = np.flatnonzero(score == threshold)[:k - above.size]
    keep = np.concatenate([above, ties])
    return keep[np.lexsort((keep, -score[keep]))]


def topk(array, params, base):
    import numpy as np

    k = _int(params, "k", 8, 1, MAX_TOPK)
    by = params.get("by", "abs")
    if by not in ("abs", "signed"):
        raise ReduceError("bad_params", "by is 'abs' or 'signed'")
    values = np.asarray(array, dtype=np.float64)
    axis = params.get("axis")
    if axis is not None:
        if values.ndim == 0:
            raise ReduceError("bad_params", "axis needs a tensor with at least one axis")
        axis = _int(params, "axis", None, -values.ndim, values.ndim - 1)
    if values.size == 0:
        what = "element value" if axis is None else f"sum over the other axes ({by})"
        return TopkResult(**base, axis=axis, by=by, k=k, entries=[], score=what)
    if axis is None:
        flat = values.reshape(-1)
        score = np.abs(flat) if by == "abs" else flat.copy()
        ev, what = "measured", "element value"
    else:
        moved = np.moveaxis(values, axis, 0).reshape(values.shape[axis], -1)
        with np.errstate(all="ignore"):
            flat = (np.abs(moved) if by == "abs" else moved).sum(axis=1)
        score = flat.copy()
        ev, what = "derived-exact", f"sum over the other axes ({by})"
    score[~np.isfinite(score)] = -np.inf
    entries = []
    for i in _top_indices(score, k):
        if not np.isfinite(flat[i]):
            continue
        index = ([int(j) for j in np.unravel_index(int(i), values.shape)] if axis is None
                 else [int(i)])
        entries.append(TopkEntry(index=index, v=num(flat[i], ev, ptr=base["pointer"])))
    return TopkResult(**base, axis=axis, by=by, k=k, entries=entries, score=what)


def _selector(array, selector):
    if not isinstance(selector, list) or len(selector) > array.ndim:
        raise ReduceError("bad_params", "selector is a list of at most ndim entries")
    index = []
    for dim, item in zip(array.shape, selector):
        if item is None:
            index.append(slice(None))
        elif isinstance(item, int) and not isinstance(item, bool) and -dim <= item < dim:
            index.append(item)
        elif (isinstance(item, list) and len(item) == 2
              and all(isinstance(v, int) and not isinstance(v, bool) for v in item)
              and 0 <= item[0] <= item[1] <= dim):
            index.append(slice(item[0], item[1]))
        else:
            raise ReduceError("bad_params", f"selector entry {item!r} is not an index, a "
                              f"[start, stop] range or null within size {dim}")
    return tuple(index)


def slice_(array, params, base):
    import numpy as np

    selector = params.get("selector", [])
    view = array[_selector(array, selector)]  # a view: nothing is copied before the size check
    if view.size > MAX_SLICE:
        raise ReduceError("over_limit", f"slice has {view.size} values (> {MAX_SLICE}); "
                          "narrow the selector or use tile/histogram")
    part = np.asarray(view, dtype=np.float64)
    note = None if base["dtype"] == "float32" else f"{base['dtype']} values sent as float32"
    return SliceResult(**{**base, "note": note}, selector=selector, outShape=list(part.shape),
                       encoding="f32-le-base64", data=_encode_f32(part),
                       nonfinite=int((~np.isfinite(part)).sum()), ev="measured")


def histogram(array, params, base):
    import numpy as np

    bins = _int(params, "bins", 32, 1, MAX_BINS)
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    rng = params.get("range")
    if rng is not None:
        real = (isinstance(rng, list) and len(rng) == 2 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in rng))
        if not real or not all(math.isfinite(v) for v in rng) or not rng[0] < rng[1] or \
                not math.isfinite(float(rng[1]) - float(rng[0])):
            raise ReduceError("bad_params", "range is [lo, hi]: two finite reals with lo < hi and "
                              "a finite hi - lo")
    if finite.size == 0:
        counts, edges = np.zeros(bins, dtype=int), np.linspace(0.0, 1.0, bins + 1)
    else:
        counts, edges = np.histogram(finite, bins=bins, range=None if rng is None
                                     else (float(rng[0]), float(rng[1])))
    edge_list = NumList(vs=[_f(e) for e in edges], ev="derived-exact", unit="bin edge",
                        proof="numpy.histogram edges over the finite values"
                        if rng is None else "requested range split into equal bins")
    return HistogramResult(**base, bins=bins, edges=edge_list,
                           counts=[int(c) for c in counts],
                           nonfinite=int(values.size - finite.size), ev="measured")


def tile(array, params, base):
    import numpy as np

    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    axes = params.get("axes", [values.ndim - 2, values.ndim - 1])
    if (not isinstance(axes, list) or len(axes) != 2 or len(set(axes)) != 2
            or any(not isinstance(a, int) or not 0 <= a < values.ndim for a in axes)):
        raise ReduceError("bad_params", "axes is two distinct axis indices")
    others = [d for d in range(values.ndim) if d not in axes]
    level = _int(params, "level", 0, 0, 16)
    x0 = _int(params, "x0", 0, 0, 1 << 30)
    y0 = _int(params, "y0", 0, 0, 1 << 30)
    if values.size == 0:  # nothing recorded to draw: a resolved empty tile, not an error
        return TileResult(**base, axes=axes, fixed=[0] * len(others), x0=x0, y0=y0, level=level,
                          w=0, h=0, full=[values.shape[axes[0]], values.shape[axes[1]]],
                          encoding="u8-base64", data="",
                          domain=NumList(vs=[0.0, 0.0], ev="derived-exact",
                                         proof="empty tensor: no block means"),
                          scale="linear", agg=f"mean of {1 << level}x{1 << level} blocks of "
                          "finite values (no values)", nonfinite=0, ev="measured")
    fixed = params.get("fixed", [0] * len(others))
    if (not isinstance(fixed, list) or len(fixed) != len(others)
            or any(not isinstance(f, int) or not 0 <= f < values.shape[d]
                   for f, d in zip(fixed, others))):
        raise ReduceError("bad_params", f"fixed gives one index for each of axes {others}")
    plane = values[tuple(fixed[others.index(d)] if d in others else slice(None)
                         for d in range(values.ndim))]
    if axes[0] > axes[1]:
        plane = plane.T
    s = 1 << level
    rows, cols = plane.shape
    h = max(0, min(TILE, -(-(rows - y0) // s)))
    w = max(0, min(TILE, -(-(cols - x0) // s)))
    out = np.full((h, w), np.nan)
    for i in range(h):
        block_rows = plane[y0 + i * s:y0 + (i + 1) * s]
        for j in range(w):
            block = block_rows[:, x0 + j * s:x0 + (j + 1) * s]
            good = block[np.isfinite(block)]
            if good.size:
                out[i, j] = good.mean()
    finite = out[np.isfinite(out)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
    # Byte 0 = no finite value in the block (drawn empty); 1..255 span the domain. Scaled by
    # halves so a domain near the float64 limit does not overflow.
    if lo < 0 < hi:
        m = max(-lo, hi)
        domain, scale = [-m, m], "diverging"
        proof = ("diverging: symmetric about 0, [-m, m] with m = max(|min|, |max|) of the "
                 "block means")
        q = 1 + np.round((out + m) / (2 * m) * 254 if math.isfinite(2 * m)
                         else (out / 2 / m + 0.5) * 254)
    else:
        domain, scale = [lo, hi], "linear"
        proof = "linear: [min, max] of the block means"
        if not hi > lo:
            q = np.full(out.shape, 128.0)
        elif math.isfinite(hi - lo):
            q = 1 + np.round((out - lo) / (hi - lo) * 254)
        else:
            q = 1 + np.round((out / 2 - lo / 2) / (hi / 2 - lo / 2) * 254)
    q[~np.isfinite(out)] = 0
    q = np.clip(q, 0, 255)
    data = base64.b64encode(q.astype(np.uint8).tobytes()).decode()
    return TileResult(**base, axes=axes, fixed=fixed, x0=x0, y0=y0, level=level, w=w, h=h,
                      full=[rows, cols], encoding="u8-base64", data=data,
                      domain=NumList(vs=domain, ev="derived-exact", proof=proof),
                      scale=scale, agg=f"mean of {s}x{s} blocks of finite values",
                      nonfinite=int((~np.isfinite(out)).sum()), ev="measured")


def project(model, array, params, base):
    import numpy as np

    spec = params.get("direction")
    if not isinstance(spec, dict):
        raise ReduceError("bad_params", "direction is a pointer spec")
    direction, _ = resolve(model, spec)
    direction = np.asarray(direction, dtype=np.float64).reshape(-1)
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != direction.size:
        raise ReduceError("bad_params", f"direction has {direction.size} values; the tensor's "
                          f"last axis has {values.shape[-1] if values.ndim else 0}")
    out = values @ direction
    out = np.asarray(out).reshape(values.shape[:-1])
    if out.size > MAX_SLICE:
        raise ReduceError("over_limit", f"projection has {out.size} values (> {MAX_SLICE})")
    return ProjectResult(**base, direction=spec_digest(spec),
                         directionNorm=_stat(np.linalg.norm(direction)),
                         outShape=list(out.shape), encoding="f32-le-base64",
                         data=_encode_f32(out), ev="derived-exact")


def _check_params(op, params):
    known = PARAMS.get(op)
    if known is None:
        raise ReduceError("bad_op", f"unknown reduce op {op!r}")
    unknown = sorted(str(k) for k in params if k not in known)
    if unknown:
        shown = ", ".join(repr(k[:40]) for k in unknown[:8]) + (
            f" (+{len(unknown) - 8} more)" if len(unknown) > 8 else "")
        raise ReduceError("bad_params", f"unknown parameter(s) for {op}: {shown}; "
                          f"{op} takes {', '.join(known) if known else 'no parameters'}")


def params_digest(params):
    """sha256 of the canonical params (known names only, so it is the whole key)."""
    try:
        raw = json.dumps(params, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (RecursionError, ValueError, TypeError) as error:
        raise ReduceError("bad_params", "params are not plain JSON of bounded depth") from error
    return hashlib.sha256(raw.encode()).hexdigest()


class Reducer:
    """Runs reductions and caches results by (session, pointer, op, params digest).

    The cache is bounded by entry count (``size``) and by the summed size of the cached results
    (``max_bytes``); a result larger than a quarter of the byte budget is not cached."""

    def __init__(self, size=256, max_bytes=CACHE_BYTES):
        self.size, self.max_bytes = size, max_bytes
        self._cache, self._lock, self.bytes = OrderedDict(), threading.Lock(), 0

    def _store(self, key, result):
        weight = len(result.model_dump_json())
        if weight > self.max_bytes // 4:
            return
        with self._lock:
            if key in self._cache:
                self.bytes -= self._cache.pop(key)[1]
            self._cache[key] = (result, weight)
            self.bytes += weight
            while len(self._cache) > self.size or self.bytes > self.max_bytes:
                self.bytes -= self._cache.popitem(last=False)[1][1]

    @staticmethod
    def _pointer_key(model, spec: dict) -> str:
        """The cache key of a pointer: an occurrence pointer is keyed on the blob its index row
        names now (so a re-indexed address never serves a stale result), the rest on the spec."""
        if isinstance(spec, dict) and spec.get("kind") == "occurrence":
            try:
                address = format_address(parse_address(spec["address"]))
                row = occurrence_row(model, address)
            except (AddressError, KeyError, TypeError, ValueError):
                row = None
            blob = row.get("blob_id") if isinstance(row, dict) else None
            return spec_digest({"kind": "occurrence-blob", "blob": blob if isinstance(
                blob, str) else None, "address": None if isinstance(blob, str) else spec.get(
                    "address")})
        return spec_digest(spec)

    def run(self, model, spec: dict, op: str, params: dict):
        _check_params(op, params)
        key = (model.sid, self._pointer_key(model, spec), op, params_digest(params))
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key][0]
        array, info = resolve(model, spec)
        base = {"op": op, "pointer": spec_digest(spec), "shape": list(info["shape"]),
                "dtype": info["dtype"], "note": None}
        try:
            if op == "stats":
                result = stats(array, base)
            elif op == "topk":
                result = topk(array, params, base)
            elif op == "slice":
                result = slice_(array, params, base)
            elif op == "histogram":
                result = histogram(array, params, base)
            elif op == "tile":
                result = tile(array, params, base)
            else:
                result = project(model, array, params, base)
        except ReduceError:
            raise
        except (ValueError, ArithmeticError, TypeError, IndexError) as error:
            # numpy rejected a well-formed request (bad axis, bad range, ...): the caller's
            # parameters, not a server fault.
            raise ReduceError("bad_params", f"{op}: {type(error).__name__}: "
                              f"{str(error)[:300]}") from error
        self._store(key, result)
        return result


__all__ = ["Reducer", "ReduceError", "resolve"]

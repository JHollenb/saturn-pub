"""Compiled-state residual view: how far a linear state compiler reproduces native.

A *prompt family* is a shared prompt prefix plus a fixed-length suffix whose
positions carry declared symbols. Each symbol is a finite value with the first
suffix position whose causal context determines it. A *compiled state* for one
feature set and layer is a per-position ridge map from the visible one-hot
symbol features to the residual entering that layer. The residual view reports,
per (position, layer): mean native norm, mean residual norm and pooled R²; per
layer: native top-1 agreement after splicing the compiled state and running the
remaining layers; per feature set: the agreement horizon.

The family supplies symbol values; mdb never derives them from tokens, and an
evaluator reference is used only for scoring. This module is framework-free:
reports render without numpy, and the map math imports numpy only when called.
The model runner lives in ``compiled_state_runner``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from saturn_pub.values import digest

FAMILY_SCHEMA = "mdb-prompt-family-v1"
MAP_SCHEMA = "mdb-compiled-state-map-v1"
REPORT_SCHEMA = "mdb-compiled-state-residual-v1"


class CompiledStateError(ValueError):
    pass


# --------------------------------------------------------------------------- family


@dataclass(frozen=True)
class Symbol:
    """A finite value fixed by the tokens at suffix positions <= ``visible_from``."""

    name: str
    cardinality: int
    visible_from: int


@dataclass(frozen=True)
class FeatureSet:
    """Symbols a compiler may read; ``switch`` fits one map per value of that symbol
    at the positions that can see it (earlier positions keep the pooled map)."""

    symbols: tuple[str, ...]
    switch: str | None = None


@dataclass(frozen=True)
class PromptFamily:
    name: str
    prefix: str
    positions: tuple[str, ...]
    symbols: tuple[Symbol, ...]
    feature_sets: Mapping[str, FeatureSet]
    score_positions: tuple[int, ...]
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        T = len(self.positions)
        names = [s.name for s in self.symbols]
        if not self.name or T == 0 or len(set(self.positions)) != T:
            raise CompiledStateError("family needs a name and distinct position names")
        if len(set(names)) != len(names) or not names:
            raise CompiledStateError("family symbols need distinct names")
        for s in self.symbols:
            if (type(s.cardinality) is not int or s.cardinality < 1
                    or type(s.visible_from) is not int or not 0 <= s.visible_from < T):
                raise CompiledStateError(f"symbol {s.name!r} needs cardinality >= 1 and a suffix position")
        if not self.score_positions or any(type(p) is not int or not 0 <= p < T for p in self.score_positions):
            raise CompiledStateError("score positions must index the suffix")
        if not self.feature_sets:
            raise CompiledStateError("family declares no feature sets")
        for fname, fs in self.feature_sets.items():
            if not fs.symbols or any(n not in names for n in fs.symbols) or len(set(fs.symbols)) != len(fs.symbols):
                raise CompiledStateError(f"feature set {fname!r} names unknown or repeated symbols")
            if fs.switch is not None and fs.switch not in names:
                raise CompiledStateError(f"feature set {fname!r} switches on an unknown symbol")

    # layout: feature 0 is the bias; then one one-hot block per symbol, in declared order
    @property
    def width(self) -> int:
        return 1 + sum(s.cardinality for s in self.symbols)

    def offsets(self) -> dict[str, int]:
        out, at = {}, 1
        for s in self.symbols:
            out[s.name] = at
            at += s.cardinality
        return out

    def symbol(self, name: str) -> Symbol:
        return next(s for s in self.symbols if s.name == name)

    def to_dict(self) -> dict[str, Any]:
        body = {"schema": FAMILY_SCHEMA, "name": self.name, "prefix": self.prefix,
                "positions": list(self.positions),
                "symbols": [{"name": s.name, "cardinality": s.cardinality, "visible_from": s.visible_from}
                            for s in self.symbols],
                "feature_sets": {k: {"symbols": list(v.symbols), "switch": v.switch}
                                 for k, v in self.feature_sets.items()},
                "score_positions": list(self.score_positions), "notes": dict(self.notes)}
        return {**body, "fingerprint": digest(body)}

    @property
    def fingerprint(self) -> str:
        return self.to_dict()["fingerprint"]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PromptFamily:
        body = dict(value)
        fp = body.pop("fingerprint", None)
        if body.get("schema") != FAMILY_SCHEMA or (fp is not None and fp != digest(body)):
            raise CompiledStateError("prompt family schema or fingerprint mismatch")
        return cls(name=body["name"], prefix=body["prefix"], positions=tuple(body["positions"]),
                   symbols=tuple(Symbol(**s) for s in body["symbols"]),
                   feature_sets={k: FeatureSet(tuple(v["symbols"]), v.get("switch"))
                                 for k, v in body["feature_sets"].items()},
                   score_positions=tuple(body["score_positions"]), notes=body.get("notes", {}))


def visibility(family: PromptFamily):
    """bool [T, width]: feature f is determined by the tokens at positions <= p."""
    import numpy as np

    V = np.zeros((len(family.positions), family.width), bool)
    V[:, 0] = True
    off = family.offsets()
    for s in family.symbols:
        V[s.visible_from:, off[s.name]:off[s.name] + s.cardinality] = True
    return V


def columns(family: PromptFamily, feature_set: str):
    """bool [width]: the bias plus the blocks of the feature set's symbols."""
    import numpy as np

    cols = np.zeros(family.width, bool)
    cols[0] = True
    off = family.offsets()
    for name in family.feature_sets[feature_set].symbols:
        cols[off[name]:off[name] + family.symbol(name).cardinality] = True
    return cols


def design(family: PromptFamily, values):
    """float32 [N, width] one-hot features from integer symbol values [N, S] (unmasked)."""
    import numpy as np

    values = np.asarray(values)
    S = len(family.symbols)
    if values.ndim != 2 or values.shape[1] != S or not np.issubdtype(values.dtype, np.integer):
        raise CompiledStateError(f"symbol values must be an integer array [N, {S}]")
    card = np.array([s.cardinality for s in family.symbols])
    if values.size and ((values < 0) | (values >= card)).any():
        raise CompiledStateError("symbol value outside its declared cardinality")
    X = np.zeros((len(values), family.width), np.float32)
    X[:, 0] = 1
    rows = np.arange(len(values))
    for j, (name, at) in enumerate(family.offsets().items()):
        X[rows, at + values[:, j]] = 1
    return X


# --------------------------------------------------------------------------- fit


def _einsum(x):
    if type(x).__module__.startswith("torch"):
        import torch
        return torch.einsum
    import numpy as np
    return np.einsum


class CompilerFit:
    """Streaming ridge sufficient statistics for every feature set of a family at
    the given layers. ``add`` takes numpy rows; a device runner may accumulate the
    same moments itself and hand them over once with ``add_moments``."""

    def __init__(self, family: PromptFamily, layers: Sequence[int], hidden_size: int,
                 feature_sets: Sequence[str] | None = None):
        import numpy as np

        self.family, self.layers, self.D = family, tuple(sorted(set(layers))), int(hidden_size)
        T, F = len(family.positions), family.width
        self.n = 0
        self.gram = np.zeros((T, F, F))
        self.xty = {L: np.zeros((T, F, self.D)) for L in self.layers}
        fsets = family.feature_sets if feature_sets is None else {f: family.feature_sets[f] for f in feature_sets}
        self.switches = sorted({fs.switch for fs in fsets.values() if fs.switch})
        # per switch symbol: [V, T - s0, F, F] and {L: [V, T - s0, F, D]}, positions >= its visible_from
        self.sw_gram, self.sw_xty, self.sw_n = {}, {}, {}
        for name in self.switches:
            s = family.symbol(name)
            k = T - s.visible_from
            self.sw_gram[name] = np.zeros((s.cardinality, k, F, F))
            self.sw_xty[name] = {L: np.zeros((s.cardinality, k, F, self.D)) for L in self.layers}
            self.sw_n[name] = np.zeros(s.cardinality, np.int64)

    def add(self, values, states: Mapping[int, Any]):
        import numpy as np

        values = np.asarray(values)
        P = design(self.family, values)[:, None, :] * visibility(self.family)[None]
        gram = np.einsum("bpf,bpg->pfg", P, P)
        xty = {L: np.einsum("bpf,bpd->pfd", P, np.asarray(states[L], np.float64)) for L in self.layers}
        sw = {}
        for name in self.switches:
            s0, j = self.family.symbol(name).visible_from, self._index(name)
            groups = {}
            for g in range(self.family.symbol(name).cardinality):
                m = values[:, j] == g
                if m.any():
                    Pg = P[m, s0:]
                    groups[g] = (int(m.sum()), np.einsum("bpf,bpg->pfg", Pg, Pg),
                                 {L: np.einsum("bpf,bpd->pfd", Pg, np.asarray(states[L], np.float64)[m, s0:])
                                  for L in self.layers})
            sw[name] = groups
        self.add_moments(len(values), gram, xty, sw)

    def add_moments(self, n: int, gram, xty: Mapping[int, Any], switched: Mapping[str, Mapping[int, tuple]] = ()):
        """Accumulate summed moments. ``switched[name][value] = (rows, gram, {L: xty})``."""
        import numpy as np

        self.n += int(n)
        self.gram += np.asarray(gram, np.float64)
        for L in self.layers:
            self.xty[L] += np.asarray(xty[L], np.float64)
        for name, groups in dict(switched).items():
            for g, (rows, gg, gx) in groups.items():
                self.sw_n[name][g] += int(rows)
                self.sw_gram[name][g] += np.asarray(gg, np.float64)
                for L in self.layers:
                    self.sw_xty[name][L][g] += np.asarray(gx[L], np.float64)

    def _index(self, name: str) -> int:
        return [s.name for s in self.family.symbols].index(name)

    def solve(self, feature_set: str, layer: int, lam: float = 1.0) -> CompiledStateMap:
        """Per-position ridge (G + lam I) W = X^T Y over the visible feature-set columns.
        The bias is regularized like every other column."""
        import numpy as np

        if not lam > 0 or not math.isfinite(lam):
            raise CompiledStateError("ridge lambda must be positive and finite")
        if layer not in self.xty:
            raise CompiledStateError(f"layer {layer} was not accumulated")
        if self.n == 0:
            raise CompiledStateError("no fit rows")
        fam, fs = self.family, self.family.feature_sets[feature_set]
        if fs.switch and fs.switch not in self.switches:
            raise CompiledStateError(f"feature set {feature_set!r} switch moments were not accumulated")
        V, cols = visibility(fam), columns(fam, feature_set)
        T, F = V.shape
        W = np.zeros((T, F, self.D), np.float32)
        for p in range(T):
            W[p] = _ridge(self.gram[p], self.xty[layer][p], V[p] & cols, lam)
        groups = None
        if fs.switch:
            s = fam.symbol(fs.switch)
            groups = np.repeat(W[None], s.cardinality, 0)
            for g in range(s.cardinality):
                if self.sw_n[fs.switch][g] == 0:
                    continue  # unseen switch value keeps the pooled map
                for i, p in enumerate(range(s.visible_from, T)):
                    groups[g, p] = _ridge(self.sw_gram[fs.switch][g, i], self.sw_xty[fs.switch][layer][g, i],
                                          V[p] & cols, lam)
        return CompiledStateMap(family=fam, feature_set=feature_set, layer=int(layer), lam=float(lam),
                                n_fit=self.n, weights=groups if groups is not None else W,
                                switch_rows=self.sw_n[fs.switch].tolist() if fs.switch else None)


def _ridge(G, XtY, keep, lam):
    import numpy as np

    out = np.zeros((G.shape[0], XtY.shape[1]), np.float32)
    idx = np.flatnonzero(keep)
    out[idx] = np.linalg.solve(G[np.ix_(idx, idx)] + lam * np.eye(len(idx)), XtY[idx]).astype(np.float32)
    return out


@dataclass
class CompiledStateMap:
    """Fitted map for one (feature set, layer): weights [T, F, D], or [V, T, F, D]
    when the feature set switches on a symbol."""

    family: PromptFamily
    feature_set: str
    layer: int
    lam: float
    n_fit: int
    weights: Any
    switch_rows: list[int] | None = None

    @property
    def switch(self) -> str | None:
        return self.family.feature_sets[self.feature_set].switch

    def masked_features(self, values):
        import numpy as np

        return design(self.family, values)[:, None, :] * visibility(self.family)[None].astype(np.float32)

    def compile(self, values):
        """float32 [N, T, D] compiled residual entering ``layer``."""
        import numpy as np

        values = np.asarray(values)
        groups = values[:, [s.name for s in self.family.symbols].index(self.switch)] if self.switch else None
        return compile_state(self.masked_features(values), self.weights, groups,
                             self.family.symbol(self.switch).visible_from if self.switch else 0)

    def manifest(self, weights_digest: str) -> dict[str, Any]:
        fs = self.family.feature_sets[self.feature_set]
        body = {"schema": MAP_SCHEMA, "family": self.family.fingerprint, "feature_set": self.feature_set,
                "symbols": list(fs.symbols), "switch": fs.switch, "switch_rows": self.switch_rows,
                "layer": self.layer, "lam": self.lam, "n_fit": self.n_fit,
                "shape": list(self.weights.shape), "dtype": "float32", "weights_sha256": weights_digest}
        return {**body, "fingerprint": digest(body)}

    def save(self, directory: str | Path) -> dict[str, Any]:
        import numpy as np

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{_slug(self.feature_set)}-L{self.layer}"
        weights = np.ascontiguousarray(self.weights, np.float32)
        np.save(d / f"{stem}.npy", weights)
        manifest = self.manifest(weights_sha256(weights))
        (d / f"{stem}.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        fam = d / "family.json"
        if not fam.exists():
            fam.write_text(json.dumps(self.family.to_dict(), indent=1, sort_keys=True) + "\n")
        return manifest

    @classmethod
    def load(cls, directory: str | Path, feature_set: str, layer: int) -> CompiledStateMap:

        d = Path(directory)
        family = PromptFamily.from_dict(json.loads((d / "family.json").read_text()))
        stem = f"{_slug(feature_set)}-L{layer}"
        manifest, weights = _verified(d / f"{stem}.json", family)
        return cls(family=family, feature_set=feature_set, layer=int(layer), lam=manifest["lam"],
                   n_fit=manifest["n_fit"], weights=weights, switch_rows=manifest["switch_rows"])


def compile_state(P, weights, groups=None, switch_from: int = 0):
    """Compiled states [B, T, D] from masked features P [B, T, F] (numpy or torch).
    With ``groups`` [B], weights are [V, T, F, D] and positions >= ``switch_from``
    use the row's group map."""
    einsum = _einsum(P)
    if groups is None:
        return einsum("bpf,pfd->bpd", P, weights)
    x = einsum("bpf,pfd->bpd", P, weights[0])
    for g in range(weights.shape[0]):
        m = groups == g
        if m.any():
            x[m, switch_from:] = einsum("bpf,pfd->bpd", P[m, switch_from:], weights[g, switch_from:])
    return x


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def weights_sha256(weights) -> str:
    """sha256 of the contiguous little-endian float32 weight bytes (file-format independent)."""
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(weights, "<f4").tobytes()).hexdigest()


def _verified(path: Path, family: PromptFamily):
    import numpy as np

    body = json.loads(path.read_text())
    fp = body.pop("fingerprint", None)
    if body.get("schema") != MAP_SCHEMA or fp != digest(body) or body["family"] != family.fingerprint:
        raise CompiledStateError(f"{path.name}: map schema, fingerprint or family mismatch")
    weights = np.load(path.with_suffix(".npy"))
    if list(weights.shape) != body["shape"] or weights_sha256(weights) != body["weights_sha256"]:
        raise CompiledStateError(f"{path.name}: weights do not match the manifest shape or digest")
    return {**body, "fingerprint": fp}, weights


def verify_maps(directory: str | Path) -> list[dict[str, Any]]:
    """Check every map manifest in a directory against the family and the weight bytes."""
    d = Path(directory)
    family = PromptFamily.from_dict(json.loads((d / "family.json").read_text()))
    return [_verified(p, family)[0] for p in sorted(d.glob("*.json")) if p.name != "family.json"]


# --------------------------------------------------------------------------- residual statistics


def batch_residual_moments(native, compiled):
    """Per-position reductions of native [B, T, D] and native - compiled (numpy or torch):
    (sum_x [T, D], sum_x2 [T], sse [T], sum_norm [T], sum_err_norm [T])."""
    err = native - compiled
    sq, esq = (native * native).sum(2), (err * err).sum(2)
    return native.sum(0), sq.sum(0), esq.sum(0), (sq ** 0.5).sum(0), (esq ** 0.5).sum(0)


class ResidualAccumulator:
    """Pooled per-position R² = 1 - SSE / SST with SST = sum ||x - mean||^2 over rows
    and dimensions, plus mean native and residual L2 norms. A position whose native
    state does not vary (SST below 1e-9 of sum ||x||^2) has R² None."""

    def __init__(self, positions: int, hidden_size: int):
        import numpy as np

        self.n = 0
        self.sum_x = np.zeros((positions, hidden_size))
        self.sum_x2, self.sse = np.zeros(positions), np.zeros(positions)
        self.sum_norm, self.sum_err_norm = np.zeros(positions), np.zeros(positions)

    def add(self, native, compiled):
        import numpy as np

        native, compiled = np.asarray(native, np.float64), np.asarray(compiled, np.float64)
        self.add_moments(len(native), *batch_residual_moments(native, compiled))

    def add_moments(self, n, sum_x, sum_x2, sse, sum_norm, sum_err_norm):
        import numpy as np

        self.n += int(n)
        self.sum_x += np.asarray(sum_x, np.float64)
        self.sum_x2 += np.asarray(sum_x2, np.float64)
        self.sse += np.asarray(sse, np.float64)
        self.sum_norm += np.asarray(sum_norm, np.float64)
        self.sum_err_norm += np.asarray(sum_err_norm, np.float64)

    def summary(self, position_names: Sequence[str]) -> list[dict[str, Any]]:
        if self.n == 0:
            raise CompiledStateError("no residual rows")
        sst = self.sum_x2 - (self.sum_x ** 2).sum(1) / self.n
        out = []
        for p, name in enumerate(position_names):
            native, resid = self.sum_norm[p] / self.n, self.sum_err_norm[p] / self.n
            out.append({"position": name, "native_norm": float(native), "residual_norm": float(resid),
                        "relative_residual": float(resid / native) if native > 0 else None,
                        "r2": float(1 - self.sse[p] / sst[p]) if sst[p] > 1e-9 * self.sum_x2[p] else None})
        return out


# --------------------------------------------------------------------------- agreement and horizon


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


def agreement(native_top, compiled_top, reference=None) -> dict[str, Any]:
    """Top-1 agreement at the score positions ([N, K] token ids). A row agrees when
    every score position matches native. With an evaluator ``reference`` [N, K]:
    error recall = share of native-wrong rows reproduced exactly; error precision =
    share of compiler-wrong rows that equal native."""
    import numpy as np

    nat, comp = np.asarray(native_top), np.asarray(compiled_top)
    same = nat == comp
    row = same.all(1)
    n, k = len(row), int(row.sum())
    out = {"n": n, "row_agreement": k / n if n else None, "row_agreement_ci": wilson(k, n),
           "position_agreement": same.mean(0).tolist() if n else None}
    if reference is not None:
        ref = np.asarray(reference)
        nat_wrong, comp_wrong = (nat != ref).any(1), (comp != ref).any(1)
        nw, cw = int(nat_wrong.sum()), int(comp_wrong.sum())
        out.update({"native_accuracy": 1 - nw / n, "compiled_accuracy": 1 - cw / n,
                    "n_native_wrong": nw, "n_compiled_wrong": cw,
                    "error_recall": float(row[nat_wrong].mean()) if nw else None,
                    "error_recall_ci": wilson(int(row[nat_wrong].sum()), nw),
                    "error_precision": float(row[comp_wrong].mean()) if cw else None})
    return out


def horizon(by_layer: Mapping[int, float], threshold: float) -> dict[str, Any]:
    """``horizon``: deepest tested layer such that it and every shallower tested layer
    reach the threshold (None if the shallowest fails). ``first_break``: shallowest
    failing layer. ``deepest_pass`` can exceed ``horizon`` when agreement recovers."""
    layers = sorted(int(L) for L in by_layer)
    passing = [L for L in layers if by_layer[L] is not None and by_layer[L] >= threshold]
    first_break = next((L for L in layers if L not in passing), None)
    held = [L for L in passing if first_break is None or L < first_break]
    return {"threshold": threshold, "horizon": held[-1] if held else None, "first_break": first_break,
            "deepest_pass": passing[-1] if passing else None,
            "monotone": all(L not in passing for L in layers if first_break is not None and L > first_break)}


# --------------------------------------------------------------------------- report


def seal_report(body: Mapping[str, Any]) -> dict[str, Any]:
    body = {**body, "schema": REPORT_SCHEMA}
    body.pop("fingerprint", None)
    return {**body, "fingerprint": digest(body)}


def verify_report(report: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(report)
    fp = body.pop("fingerprint", None)
    if body.get("schema") != REPORT_SCHEMA or fp != digest(body):
        raise CompiledStateError("compiled-state report schema or fingerprint mismatch")
    return dict(report)


def _by_layer(layers: Mapping[str, Any]):
    """Layer rows in numeric order (JSON object keys are strings and may be key-sorted)."""
    return sorted(layers.items(), key=lambda item: int(item[0]))


def _fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def render(report: Mapping[str, Any], *, feature_set: str | None = None, layer: int | None = None,
           position: str | None = None) -> str:
    fam = report["family"]
    lines = [f"family {fam['name']}  ({fam['fingerprint'][:12]})  model {report.get('model', {}).get('name', '?')}",
             f"rows fit {report['rows']['fit']}  eval {report['rows']['eval']}  r2 {report['rows']['r2']}  "
             f"lambda {report['lam']}  threshold {report['threshold']}"]
    nc, ctl = report.get("noise_ceiling"), report.get("native_splice_control")
    if nc:
        lines.append(f"noise ceiling (native vs batch-shifted native): agreement {_fmt(nc['row_agreement'])}  "
                     f"error recall {_fmt(nc.get('error_recall'))}")
    if ctl and ctl["max_abs_dlogit"]:
        lines.append(f"native splice control: max |dlogit| {max(ctl['max_abs_dlogit'].values()):.3g} "
                     f"over {ctl['rows']} rows")
    names = fam["positions"]
    score = [names[p] for p in fam["score_positions"]]
    for fname, fs in report["feature_sets"].items():
        if feature_set and fname != feature_set:
            continue
        h = fs["horizon"]
        lines.append("")
        lines.append(f"[{fname}] horizon L{h['horizon']}  first break L{h['first_break']}  "
                     f"deepest pass L{h['deepest_pass']}{'' if h['monotone'] else '  (non-monotone)'}")
        if position is not None:
            if position not in names:
                raise CompiledStateError(f"unknown position {position!r}; family positions: {', '.join(names)}")
            p = names.index(position)
            lines.append(f"  {'L':>3}  {'|native|':>9}  {'|resid|':>9}  {'rel':>6}  {'R2':>6}")
            for L, row in _by_layer(fs["layers"]):
                if layer is not None and int(L) != layer:
                    continue
                s = row["positions"][p]
                lines.append(f"  {L:>3}  {_fmt(s['native_norm'], 2):>9}  {_fmt(s['residual_norm'], 2):>9}  "
                             f"{_fmt(s['relative_residual']):>6}  {_fmt(s['r2']):>6}")
            continue
        head = f"  {'L':>3}  {'agree':>6}  {'recall':>6}  " + "  ".join(f"R2@{n:<4}" for n in score) + "  R2 min..max"
        lines.append(head)
        for L, row in _by_layer(fs["layers"]):
            if layer is not None and int(L) != layer:
                continue
            a = row["agreement"]
            r2 = [s["r2"] for s in row["positions"]]
            lines.append(f"  {L:>3}  {_fmt(a['row_agreement']):>6}  {_fmt(a.get('error_recall')):>6}  "
                         + "  ".join(f"{_fmt(r2[p]):>7}" for p in fam["score_positions"])
                         + f"  {_fmt(min(x for x in r2 if x is not None))}..{_fmt(max(x for x in r2 if x is not None))}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mdb compiled-state",
                                 description="Render a compiled-state residual report (framework-free).")
    ap.add_argument("report", help="mdb-compiled-state-residual-v1 JSON")
    ap.add_argument("--feature-set")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--position", help="per-layer norms and R² at one suffix position")
    ap.add_argument("--maps", help="directory of compiled maps to verify against the report")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        report = verify_report(json.loads(Path(a.report).read_text()))
        checked = None
        if a.maps:
            manifests = verify_maps(a.maps)
            listed = {m["fingerprint"] for fs in report["feature_sets"].values() for m in fs["maps"].values()}
            missing = listed - {m["fingerprint"] for m in manifests}
            if missing:
                raise CompiledStateError(f"{len(missing)} report maps are absent from {a.maps}")
            checked = len(listed)
        if a.json:
            print(json.dumps(report, indent=1, sort_keys=True))
        else:
            print(render(report, feature_set=a.feature_set, layer=a.layer, position=a.position))
            if checked is not None:
                print(f"\nmaps verified: {checked}")
    except (CompiledStateError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f"mdb compiled-state: {e}", file=sys.stderr)
        return 2
    return 0


__all__ = ["CompiledStateError", "CompiledStateMap", "CompilerFit", "FeatureSet", "PromptFamily",
           "ResidualAccumulator", "Symbol", "agreement", "batch_residual_moments",
           "columns", "compile_state", "design", "horizon", "main", "render", "seal_report",
           "verify_maps", "verify_report", "visibility", "weights_sha256", "wilson"]

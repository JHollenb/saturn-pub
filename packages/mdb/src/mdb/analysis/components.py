"""Component bindings: a value cell split into named parts by a feature basis (advisor memo
2026-11-14 "showcase", shared build B-a).

A value cell ``cache.L<l>.v[kv<g>,row=<r>]`` of a model with no V norm is a linear image of the
residual entering layer ``l`` at row ``r`` (``h = t<r>.L<l>.h``):

    v_lin = W_V^g (rho gamma * h) (+ b_V^g),   rho = (mean(h^2) + eps)^-1/2 at the native h

(Qwen3: no V bias and no v_norm, checked here; a V bias is its own ``v_bias`` component). A
residual feature basis whose site holds ``h`` (Qwen-Scope ``resid_post`` of layer ``l - 1``)
splits ``h = b_dec + sum_f a_f d_f + e`` exactly (``sae.FeatureBasis.decompose``), so

    v_lin = sum_f v_f + v_bdec + v_err (+ b_V^g),   v_f = W_V^g (rho gamma * a_f d_f)

in float64, every term at the same native ``rho``. A ``ComponentBinding`` holds those parts
for one cell: one component per listed feature (``feature[f]``), the rest of the active
features lumped (``other_features``), the decoder bias (``bdec``) and the error (``error``, the
cell's ``Unbound``). ``closure`` is ``|sum of components - v_lin|`` (float64 summation only);
``rounding`` is ``v_native - v_lin`` (the bf16 V projection against its float64 image,
measured, never folded into a component).

At fixed attention weights the cell reaches a scalar consumer read at row ``j`` through the
heads of KV group ``g`` only (P06/P07): ``to_consumer(v) = sum_{h in kappa^-1(g)} p[h, r]
c . W_O,h v`` with ``c`` the consumer's readout vector (P19). It is linear in ``v``, so the
components' shares add to the cell's (``derived-exact``); ``mass`` (``sum_h p[h, r]``) is the
cell's and is reported as inherited. ``content`` reads each component's write
``sum_h p[h, r] W_O,h v_i`` through ``symbols.content``.

Offline: ``component_binding`` needs ``h = t<r>.L<l>.h`` and the native cell, which a live run
has only for a row it stepped. ``ResidualCapture`` saves ``h`` at declared (row, layer) pairs
(and, optionally, the native value cache of those layers) from a run, to disk if wanted
(``save``/``load``); ``component_binding(None, ..., capture=cap, adapter=adapter)`` then builds
the same binding with no forward pass. K/V-only captures cannot: ``h`` is not recoverable from
them. ``ComponentBinding.provenance`` says where ``h`` and the native cell came from.

Refusals: row 0 (the first token is off a residual SAE's training distribution, Qwen-Scope
report section 2.2; ``sae.FIRST_ROW_CAVEAT``), a prefilled row (its residual was never
stopped at), a basis whose site does not hold ``h``, a model with a V norm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .finite import EXACT, MEASURED, f64, num
from .forecast import Refusal
from .symbols import Binding, Candidates, content, readout_basis

_CELL = re.compile(r"^cache\.L(\d+)\.v\[kv(\d+),row=(\d+)\]$")


def _torch():
    import torch

    return torch


@dataclass
class ResidualCapture:
    """The residual ``h = t<r>.L<l>.h`` at declared (row, layer) pairs, native dtype, and
    optionally the native value cache (``values[l]``: (1, H_KV, T, d_h)) of the captured layers.
    ``provenance`` records where they were read (``Run.read`` / ``Run.read_cache(staged=False)``,
    the tokens and the cursor at capture); ``adapter`` is only needed to build a binding."""

    h: dict[tuple[int, int], Any] = field(default_factory=dict)
    values: dict[int, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    adapter: Any = None

    @classmethod
    def from_run(cls, run: Any, rows: list[int], layers: list[int], *,
                 values: bool = True) -> "ResidualCapture":
        """Save ``h`` of every (row, layer) in ``rows x layers`` the run has stepped, and (with
        ``values``) the committed value cache of those layers. Refuses a row the run prefilled
        or has not reached (its residual was never stopped at)."""
        here = int(run.where.split(".")[0][1:])
        cap = cls(adapter=run.adapter)
        for row in rows:
            if int(row) < int(run.first_row) or int(row) > here:
                raise Refusal(f"t{row}: not stepped by this run (first stepped row "
                              f"t{run.first_row}, cursor {run.where})")
            for layer in layers:
                cap.h[(int(row), int(layer))] = run.read(f"t{row}.L{layer}.h").detach().clone()
        if values:
            for layer in layers:
                cap.values[int(layer)] = run.read_cache(int(layer), "v",
                                                        staged=False).detach().clone()
        cap.provenance = {"h": "Run.read(t<r>.L<l>.h) at capture", "rows": [int(r) for r in rows],
                          "layers": [int(x) for x in layers], "cursor": run.where,
                          "values": "Run.read_cache(v, staged=False)" if values else None,
                          "tokens": [int(t) for t in run.session.read("tokens")[0].tolist()]}
        return cap

    def save(self, path: str) -> None:
        _torch().save({"h": {f"{r}.{l}": v for (r, l), v in self.h.items()},
                       "values": {str(l): v for l, v in self.values.items()},
                       "provenance": self.provenance}, path)

    @classmethod
    def load(cls, path: str, adapter: Any = None) -> "ResidualCapture":
        state = _torch().load(path, map_location="cpu", weights_only=True)
        return cls({tuple(int(x) for x in k.split(".")): v for k, v in state["h"].items()},
                   {int(k): v for k, v in state["values"].items()}, state["provenance"], adapter)


@dataclass
class Component:
    """One part of a cell: ``residual`` (d,) is its part of ``h``, ``value`` (d_h,) its image in
    the cell (both float64); ``activation`` for a feature."""

    name: str
    feature: int | None
    activation: float | None
    residual: Any
    value: Any


@dataclass
class ComponentBinding(Binding):
    """A value cell split into components (module docstring). ``map`` is ``W_V^g diag(rho
    gamma)`` (d_h, d) in float64, so ``map @ x`` is the cell image of a residual vector ``x`` at
    the native ``rho``."""

    row: int = -1
    layer: int = -1
    kv: int = -1
    rho: float = 0.0
    basis: dict[str, Any] = field(default_factory=dict)
    components: list[Component] = field(default_factory=list)
    native: Any = None
    linear: Any = None
    v_bias: Any = None
    map: Any = None
    active: list[int] = field(default_factory=list)
    acts: dict[int, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    site: str = "value-cell"       # "value-cell" (a V cell) | "mlp-write" (t<r>.L<l>.m)

    @property
    def cell(self) -> str:
        if self.site == "mlp-write":
            return f"t{self.row}.L{self.layer}.m"
        return f"cache.L{self.layer}.v[kv{self.kv},row={self.row}]"

    def get(self, name: str) -> Component:
        for c in self.components:
            if c.name == name:
                return c
        raise Refusal(f"{self.name} has no component {name}")

    def closure(self) -> dict[str, Any]:
        """``sum of components (+ b_V) - v_lin`` (float64 order only) and ``v_native - v_lin``."""
        total = sum(c.value for c in self.components)
        if self.v_bias is not None:
            total = total + self.v_bias
        gap = float((total - self.linear).abs().max())
        scale = float(self.linear.abs().max()) or 1.0
        rounding = self.native - self.linear
        return {"max_gap": num(gap, EXACT), "relative_gap": num(gap / scale, EXACT),
                "within_f64": gap <= 64 * 2.0 ** -53 * scale * max(1, len(self.components)),
                "rounding_norm": num(float(rounding.norm()), MEASURED),
                "rounding_relative": num(float(rounding.norm() / (self.linear.norm() or 1.0)),
                                         MEASURED),
                "meaning": "max_gap: components + b_V against W_V^g(rho gamma h) in float64 "
                           "(an identity; checks the split); rounding: the native bf16 cell "
                           "against its float64 image (measured, not a component)"}

    def image(self, residual: Any) -> Any:
        """The cell image ``W_V^g (rho gamma * x)`` of a residual vector at the native rho (an
        MLP-write binding's site is the residual stream itself: the identity)."""
        if self.map is None:
            return f64(residual).reshape(-1)
        return self.map @ f64(residual).reshape(-1)

    def feature_value(self, feature: int, activation: float, basis: Any) -> Any:
        """The site image of ``activation * d_f`` (a supplied activation, e.g. a donor's)."""
        return self.image(basis.direction(int(feature)) * float(activation))

    def edited(self, *, remove: tuple[str, ...] = (), add: Any = None) -> Any:
        """The native cell minus the named components plus ``add`` (d_h, float64 or None):
        the value a ``set_cell`` writes (cast by the caller to the cache dtype)."""
        out = self.native.clone()
        for name in remove:
            out = out - self.get(name).value
        if add is not None:
            out = out + f64(add).reshape(-1)
        return out


def component_binding(run: Any, name: str, cell: str, basis: Any, *,
                      features: list[int] | None = None, capture: ResidualCapture | None = None,
                      adapter: Any = None) -> ComponentBinding:
    """Split ``cell`` (``cache.L<l>.v[kv<g>,row=<r>]``) over ``basis``; ``features`` lists the
    features kept as their own components (default: every active one).

    With ``capture`` (a ``ResidualCapture``) ``h`` is read from it, not from ``run``; ``run`` may
    then be None and the weights come from ``adapter`` (default ``capture.adapter``). The native
    cell comes from ``capture.values`` when held, else from ``run.read_cache``; with neither the
    refusal says so. The result is bitwise the live binding when the capture holds the live
    run's ``h`` and cache."""
    from .reads import layer_module

    torch = _torch()
    match = _CELL.match(cell.replace(" ", ""))
    if not match:
        raise Refusal(f"a component binding is over one value cell "
                      f"cache.L<l>.v[kv<g>,row=<r>], not {cell!r}")
    layer, kv, row = (int(x) for x in match.groups())
    adapter = adapter or (run.adapter if run is not None else None) \
        or (capture.adapter if capture is not None else None)
    if adapter is None:
        raise Refusal("component_binding needs a run, or a capture plus an adapter")
    if capture is not None and run is not None and adapter is not run.adapter:
        raise Refusal("the supplied adapter is not the run's")
    if row == 0:
        from .sae import FIRST_ROW_CAVEAT

        raise Refusal(f"{cell}: {FIRST_ROW_CAVEAT}")
    if capture is not None:
        if (row, layer) not in capture.h:
            raise Refusal(f"{cell}: the capture holds no h at t{row}.L{layer} (it holds "
                          f"{sorted(capture.h)})")
    elif run is None:
        raise Refusal("component_binding needs a run or a capture")
    elif row < int(run.first_row):
        raise Refusal(f"{cell}: t{row} was prefilled (first stepped row t{run.first_row}); its "
                      f"residual was never stopped at")
    if not basis.covers(layer, "h"):
        raise Refusal(f"basis {basis.name} at {basis.site} does not hold L{layer}.h")
    module = layer_module(adapter, layer)
    if getattr(module.self_attn, "v_norm", None) is not None:
        raise Refusal("component binding: the model normalises V (not a linear image of h)")
    dh = adapter.head_dim
    rows = slice(kv * dh, (kv + 1) * dh)
    w_v = f64(module.self_attn.v_proj.weight[rows]).cpu()                       # (dh, d)
    b = getattr(module.self_attn.v_proj, "bias", None)
    v_bias = None if b is None else f64(b[rows]).cpu()
    norm = module.input_layernorm
    gamma, eps = f64(norm.weight).reshape(-1).cpu(), float(norm.variance_epsilon)
    h = (capture.h[(row, layer)] if capture is not None
         else run.read(f"t{row}.L{layer}.h")).reshape(-1)
    h64 = f64(h).cpu()
    rho = float((h64.pow(2).mean() + eps) ** -0.5)
    vmap = w_v * (rho * gamma)[None, :]
    dec = basis.decompose(h)
    acts = {int(f): float(a) for f, a in zip(dec.ids, dec.acts.tolist())}
    keep = list(dec.ids) if features is None else [int(f) for f in features]
    comps: list[Component] = []
    lumped = torch.zeros_like(h64)
    for col, f in enumerate(dec.ids):
        if f in keep:
            continue
        lumped = lumped + dec.parts[:, col]
    for f in keep:
        if f in acts:
            part = dec.parts[:, dec.ids.index(f)]
        else:
            part = torch.zeros_like(h64)
        comps.append(Component(f"feature[{f}]", f, acts.get(f, 0.0), part, vmap @ part))
    comps.append(Component("other_features", None, None, lumped, vmap @ lumped))
    comps.append(Component("bdec", None, None, dec.bias, vmap @ dec.bias))
    comps.append(Component("error", None, None, dec.error, vmap @ dec.error))
    linear = vmap @ h64 + (0 if v_bias is None else v_bias)
    if capture is not None and layer in capture.values:
        cache, cache_from = capture.values[layer], "capture.values"
    elif run is not None:
        cache, cache_from = run.read_cache(layer, "v", staged=False), "Run.read_cache(staged=False)"
    else:
        raise Refusal(f"{cell}: the capture holds no value cache of L{layer} and there is no run")
    native = f64(cache[0, kv, row]).cpu()
    provenance = {"h": (f"capture ({capture.provenance.get('h')}; cursor "
                        f"{capture.provenance.get('cursor')})" if capture is not None
                        else "live Run.read"),
                  "native_cell": cache_from, "offline": capture is not None}
    return ComponentBinding(name, "component", [], [cell], row=row, layer=layer, kv=kv, rho=rho,
                            basis=basis.describe(), components=comps, native=native,
                            linear=linear, v_bias=v_bias, map=vmap, active=list(dec.ids),
                            acts=acts, provenance=provenance)


def transcoder_binding(run: Any, name: str, site: str, basis: Any, *,
                       features: list[int] | None = None) -> ComponentBinding:
    """The MLP write ``t<r>.L<l>.m`` split by a transcoder (``sae.TranscoderBasis``, gap audit
    P3): one component per listed feature (default: every active one), ``other_features``,
    ``bdec`` and ``error`` (= native ``m`` minus the reconstruction: the binding's ``Unbound``).
    The same decomposition family as the value-cell binding and ``SubspaceBinding.decompose``:
    ``component_table`` reads it (the write reaches a scalar consumer at row ``r`` directly, P19),
    ``edited(remove=...)`` gives the value a ``Run.set`` at ``m`` writes, ``feature_value`` a
    supplied activation's write. ``native`` is the measured ``m``; ``linear`` equals it (the
    error closes the identity)."""
    match = re.match(r"^t(\d+)\.L(\d+)\.m$", site.replace(" ", ""))
    if not match:
        raise Refusal(f"a transcoder binding is over one MLP write t<r>.L<l>.m, not {site!r}")
    row, layer = (int(x) for x in match.groups())
    if getattr(basis, "kind", None) != "transcoder":
        raise Refusal(f"{getattr(basis, 'name', basis)} is not a transcoder (reads L<l>.n, "
                      "writes L<l>.m)")
    if basis.layer != layer:
        raise Refusal(f"transcoder {basis.name} is of L{basis.layer}, not L{layer}")
    if row < int(run.first_row):
        raise Refusal(f"t{row} was prefilled (first stepped row t{run.first_row}): its MLP "
                      "input was never stopped at")
    torch = _torch()
    n = run.read(f"t{row}.L{layer}.n")
    m = run.read(f"t{row}.L{layer}.m")
    dec = basis.decompose(n, m)
    acts = {int(f): float(a) for f, a in zip(dec.ids, dec.acts.tolist())}
    keep = list(dec.ids) if features is None else [int(f) for f in features]
    comps: list[Component] = []
    lumped = torch.zeros_like(dec.x)
    for col, f in enumerate(dec.ids):
        if f not in keep:
            lumped = lumped + dec.parts[:, col]
    for f in keep:
        part = dec.parts[:, dec.ids.index(f)] if f in acts else torch.zeros_like(dec.x)
        comps.append(Component(f"feature[{f}]", f, acts.get(f, 0.0), part, part))
    comps.append(Component("other_features", None, None, lumped, lumped))
    comps.append(Component("bdec", None, None, dec.bias, dec.bias))
    comps.append(Component("error", None, None, dec.error, dec.error))
    return ComponentBinding(name, "component", [], [site], row=row, layer=layer, kv=-1, rho=0.0,
                            basis=basis.describe(), components=comps, native=dec.x,
                            linear=dec.x, v_bias=None, map=None, active=list(dec.ids),
                            acts=acts, site="mlp-write",
                            provenance={"decomposition": "transcoder", "n": "live Run.read",
                                        "native_write": "live Run.read", "offline": False})


def _group(run: Any, binding: ComponentBinding, reader_row: int) -> tuple[Any, list[int], Any]:
    """``(p[h, r] for heads of the group, heads, W_O blocks (H_g, d, d_h))`` at reader row."""
    from .reads import head_mix, layer_module

    mix = head_mix(run, reader_row, binding.layer, lazy=True)
    heads = [h for h, g in enumerate(mix.kappa) if g == binding.kv]
    if binding.row >= mix.p.shape[1]:
        raise Refusal(f"t{reader_row} does not read row {binding.row}")
    p = mix.p[heads, binding.row]
    dh = run.adapter.head_dim
    weight = layer_module(run.adapter, binding.layer).self_attn.o_proj.weight
    blocks = [f64(weight[:, h * dh:(h + 1) * dh]).cpu() for h in heads]
    return p.cpu(), heads, blocks


def write_of(run: Any, binding: ComponentBinding, value: Any, reader_row: int,
             group: tuple | None = None) -> Any:
    """``sum_h p[h, r] W_O,h value`` (d,) float64: what a cell value writes into the reader row
    at the native attention weights. An MLP-write binding writes its value into its own row
    (the reader row must be that row)."""
    if getattr(binding, "site", "value-cell") == "mlp-write":
        if reader_row != binding.row:
            raise Refusal(f"{binding.cell} writes row {binding.row} only; a consumer at "
                          f"t{reader_row} reads it through later layers (not a direct term)")
        return f64(value).reshape(-1)
    p, heads, blocks = group or _group(run, binding, reader_row)
    v = f64(value).reshape(-1)
    return sum(float(p[i]) * (blocks[i] @ v) for i in range(len(heads)))


def component_table(run: Any, binding: ComponentBinding, consumer: Any = None, *,
                    sets: list[Candidates] | None = None, tokenizer: Any = None) -> dict[str, Any]:
    """Per component: the exact share of a scalar consumer at fixed attention weights, the
    share of the cell's native read, and optionally the content readout of its write."""
    from .reads import readout

    if consumer is None:
        raise Refusal("component_table needs a scalar consumer (Margin or Logit)")
    reader = readout(run, consumer)
    if not reader.scalar:
        raise Refusal("component_table needs a scalar consumer (Margin or Logit)")
    j = int(reader.row)
    mlp = getattr(binding, "site", "value-cell") == "mlp-write"
    group = None if mlp else _group(run, binding, j)
    if mlp and j != binding.row:
        raise Refusal(f"{binding.cell} reaches t{j} only through later layers: use a consumer "
                      f"at t{binding.row}")
    p = None if mlp else group[0]
    c = f64(reader.vector).reshape(-1).cpu()
    native_effect = float(c @ write_of(run, binding, binding.native, j, group))
    linear_effect = float(c @ write_of(run, binding, binding.linear, j, group))
    basis = readout_basis(run, j) if sets else None
    rows = []
    for comp in binding.components:
        w = write_of(run, binding, comp.value, j, group)
        effect = float(c @ w)
        entry: dict[str, Any] = {
            "component": comp.name, "to_consumer": num(effect, EXACT, "P19"),
            "share_of_cell": (num(effect / native_effect, EXACT) if native_effect else None),
            "value_norm": num(float(comp.value.norm()), EXACT)}
        if comp.feature is not None:
            entry["feature"] = comp.feature
            entry["activation"] = num(float(comp.activation or 0.0), EXACT)
        if sets:
            entry["content"] = content(run, w, sets, basis, tokenizer=tokenizer)
        rows.append(entry)
    effects = [float(r["to_consumer"]["value"]) for r in rows]
    vb = 0.0
    if binding.v_bias is not None:
        vb = float(c @ write_of(run, binding, binding.v_bias, j, group))
    gap = abs(sum(effects) + vb - linear_effect)
    return {
        "cell": binding.cell, "binding": binding.name, "consumer": reader.text,
        "reader_row": j,
        "mass": ({"state": "not applicable", "reason": "an MLP write is added to its own row's "
                  "residual stream (no attention weight)"} if mlp else
                 {**num(float(p.sum()), EXACT), "inherited": "the cell's mass (sum of p over "
                  "the KV group's heads at the reader row); a component has no mass of its own"}),
        "cell_to_consumer": num(native_effect, EXACT, "P19"),
        "linear_to_consumer": num(linear_effect, EXACT, "P19"),
        "rounding_to_consumer": num(native_effect - linear_effect, MEASURED),
        "components": rows,
        "partition": {"gap": num(gap, EXACT),
                      "holds": gap <= 1e-9 * max(1.0, abs(linear_effect)),
                      "meaning": "sum of component shares (+ V bias) = the float64 image's "
                                 "share; the native cell differs by rounding_to_consumer"},
        "closure": binding.closure(),
        "basis": {k: binding.basis.get(k) for k in ("name", "site", "k")},
    }


__all__ = ["ResidualCapture", "Component", "ComponentBinding", "component_binding", "component_table",
           "transcoder_binding", "write_of"]

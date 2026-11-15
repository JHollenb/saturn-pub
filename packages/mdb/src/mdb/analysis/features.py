"""Derived feature addresses and feature watch predicates (issue S4; advisor memo 2026-11-14
"showcase", shared build B-b).

Address forms (``parse_feature_address``):

* ``feature[f]@t<r>.L<l>.h``: the activation ``a_f`` of feature ``f`` of a residual basis whose
  site holds ``t<r>.L<l>.h`` (0 when ``f`` is not active; float64, ``derived-exact`` over the
  native residual);
* ``feature[f]@cache.L<l>.v[kv<g>,row=<r>]``: feature ``f``'s component of that value cell
  (``analysis.components``): its value (d_h,) or, with a ``CellTarget``-style token pair, its
  content readout ``d . W_g v_f``.

``FeatureCondition`` is a watch/break predicate (an ``engine.breakpoints.Condition``) that also
understands ``feature[f]`` (activation at the context's site), ``best_feature(SET)`` (the label
of the most active feature of a declared set, -1 when none is active; the label is the
feature's declared name token, else its id), ``content(SET)`` (the best token of a declared
candidate set read from the context cell's current value through ``gamma_F * W_U`` and the
KV group's ``W_O``; rho-free, a ranking) and declared numeric constants (``truth``). It reads
the run it is evaluated on (``ConditionEnv.run``), so it fires on edited values as well.

Multi-site watches (gap fix 2026-11-14): a ``FeatureContext`` is tied to one site (one basis,
one layer). ``FeatureCondition(text, contexts)`` also takes several contexts (a sequence, or
``{layer: context}``; ``feature_sites`` builds one per layer from a ``{layer: basis}``): the
condition then reads, at each stop, the context of that stop's layer (``ConditionEnv.site``),
the item stops only at those layers, keeps one watch memory per layer and reports the stop
(``hit["site"]``, ``hit["layer"]``). A watch ``t4.L*.h`` over contexts for L0..Ln equals n
single-site watches ``t4.L<l>.h`` run one by one, in edge, level and change modes alike. Named
sets, labels, candidates and constants are per context, so every context must declare the
names the text uses.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from mdb.engine.breakpoints import Condition, ConditionError

from .finite import f64
from .forecast import Refusal

_ADDRESS = re.compile(r"^feature\[(\d+)\]@(?:t(\d+)\.L(\d+)\.h|cache\.L(\d+)\.v\[kv(\d+),row=(\d+)\])$")
FUNCTIONS = ("best_feature", "content")


@dataclass(frozen=True)
class FeatureAddress:
    feature: int
    row: int
    layer: int
    form: str               # "h" | "cell"
    kv: int | None = None

    @property
    def text(self) -> str:
        if self.form == "h":
            return f"feature[{self.feature}]@t{self.row}.L{self.layer}.h"
        return f"feature[{self.feature}]@cache.L{self.layer}.v[kv{self.kv},row={self.row}]"

    @property
    def site(self) -> str:
        """The residual the basis reads: ``t<r>.L<l>.h`` (also for the cell form)."""
        return f"t{self.row}.L{self.layer}.h"


def parse_feature_address(text: str) -> FeatureAddress:
    m = _ADDRESS.match(text.replace(" ", ""))
    if not m:
        raise Refusal(f"a feature address is feature[f]@t<r>.L<l>.h or "
                      f"feature[f]@cache.L<l>.v[kv<g>,row=<r>], not {text!r}")
    f = int(m[1])
    if m[2] is not None:
        return FeatureAddress(f, int(m[2]), int(m[3]), "h")
    return FeatureAddress(f, int(m[6]), int(m[4]), "cell", int(m[5]))


def activations(run: Any, basis: Any, row: int, layer: int) -> dict[int, float]:
    """``{feature: a_f}`` of the active set at ``t<row>.L<layer>.h`` (float64); for a transcoder
    (``read_port`` ``n``) at its read site ``t<row>.L<layer>.n``."""
    port = getattr(basis, "read_port", "h")
    if not basis.covers(layer, port):
        raise Refusal(f"basis {basis.name} at {basis.site} does not hold L{layer}.{port}")
    if row == 0 and port == "h":
        from .sae import FIRST_ROW_CAVEAT

        raise Refusal(f"t0: {FIRST_ROW_CAVEAT}")
    enc = basis.encode(run.read(f"t{row}.L{layer}.{port}"))
    return {int(f): float(a) for f, a in zip(enc["ids"], enc["acts"].tolist())}


def feature_value(run: Any, address: FeatureAddress | str, basis: Any, *,
                  target: Any = None) -> Any:
    """The value at a feature address (module docstring). ``target``: a ``symbols.CellTarget``
    for the cell form's scalar content readout."""
    if isinstance(address, str):
        address = parse_feature_address(address)
    if address.form == "h":
        return activations(run, basis, address.row, address.layer).get(address.feature, 0.0)
    from .components import component_binding

    cb = component_binding(run, "watch", f"cache.L{address.layer}.v[kv{address.kv},"
                           f"row={address.row}]", basis, features=[address.feature])
    value = cb.components[0].value
    if target is None:
        return value
    from .symbols import _cell_direction

    return float(_cell_direction(run, target) @ value)


@dataclass
class FeatureContext:
    """What a ``FeatureCondition`` reads: the basis, the site row/layer, optionally the cell's KV
    group for ``content``, named feature sets, labels, candidate token sets and constants."""

    basis: Any
    row: int
    layer: int
    kv: int | None = None
    sets: dict[str, list[int]] = field(default_factory=dict)
    labels: dict[int, int] = field(default_factory=dict)
    candidates: dict[str, list[int]] = field(default_factory=dict)
    constants: dict[str, float] = field(default_factory=dict)
    _memo: dict = field(default_factory=dict, repr=False)

    def acts(self, run: Any) -> dict[int, float]:
        key = (id(run), run.where, len(run.edits) if hasattr(run, "edits") else 0)
        if key not in self._memo:
            self._memo.clear()
            self._memo[key] = activations(run, self.basis, self.row, self.layer)
        return self._memo[key]

    def best_feature(self, run: Any, name: str) -> int:
        acts = self.acts(run)
        live = [(acts[f], f) for f in self.sets[name] if acts.get(f, 0.0) > 0]
        if not live:
            return -1
        f = max(live)[1]
        return int(self.labels.get(f, f))

    def content(self, run: Any, name: str) -> int:
        """Best token of candidate set ``name`` read from the cell's current value (ranking:
        ``gamma_F * W_U[t] . W_g v``, rho-free)."""
        from .circuit_map import group_writers
        from .reads import layer_module

        if self.kv is None:
            raise ConditionError("content(SET) needs a context cell (kv)")
        ad = run.adapter
        v = f64(run.read_cache(self.layer, "v", staged=False)[0, self.kv, self.row]).cpu()
        w_g = group_writers(layer_module(ad, self.layer).self_attn.o_proj.weight,
                            ad.num_heads, ad.num_kv_heads)[self.kv]
        write = f64(w_g).cpu() @ v
        final = getattr(ad.backbone, ad.spec.final_norm)
        gamma = f64(final.weight).reshape(-1).cpu()
        toks = self.candidates[name]
        import torch

        rows = f64(ad.model.lm_head.weight.index_select(
            0, torch.tensor(toks, device=ad.model.lm_head.weight.device))).cpu()
        scores = (rows * gamma) @ write
        return int(toks[int(scores.argmax())])


def feature_sites(bases: Any, row: int, layers: Any = None, **shared: Any) -> dict[int, FeatureContext]:
    """``{layer: FeatureContext}`` for a multi-site ``FeatureCondition``: one context per layer of
    ``bases`` (``{layer: basis}``, or a callable ``layer -> basis``; ``layers`` then names them),
    all at ``row``, sharing ``kv``, ``sets``, ``labels``, ``candidates`` and ``constants``."""
    if callable(bases):
        if layers is None:
            raise Refusal("feature_sites(callable) needs layers=")
        found = {int(layer): bases(int(layer)) for layer in layers}
    else:
        found = {int(layer): basis for layer, basis in dict(bases).items()
                 if layers is None or layer in set(layers)}
    return {layer: FeatureContext(basis, row, layer, **shared) for layer, basis in
            sorted(found.items())}


class FeatureCondition(Condition):
    """A ``Condition`` that also reads features (module docstring). ``context``: one
    ``FeatureContext`` (a single-site condition), or several (a sequence, or ``{layer:
    context}``; one per distinct layer): a multi-site condition (``sites`` = their layers)."""

    def __init__(self, text: str, context: Any):
        if isinstance(context, FeatureContext):
            self.contexts = {context.layer: context}
            self.sites = None
        else:
            items = list(context.values()) if isinstance(context, dict) else list(context)
            if not items or not all(isinstance(c, FeatureContext) for c in items):
                raise ConditionError("a feature condition takes a FeatureContext or several")
            layers = [c.layer for c in items]
            if len(set(layers)) != len(layers):
                raise ConditionError("one context per layer in a multi-site feature condition")
            if isinstance(context, dict) and any(int(k) != c.layer for k, c in context.items()):
                raise ConditionError("a context dict is keyed by its context's layer")
            self.contexts = {c.layer: c for c in items}
            self.sites = frozenset(layers)
        self.context = next(iter(self.contexts.values()))
        super().__init__(text)

    def _context(self, env: Any = None) -> FeatureContext:
        """The context of the stop ``env`` is at (a single-site condition: its only one)."""
        if self.sites is None:
            return self.context
        match = re.search(r"\.L(\d+)\.", getattr(env, "site", None) or "")
        if match is None or int(match[1]) not in self.contexts:
            raise ConditionError(f"no feature context for the stop "
                                 f"{getattr(env, 'site', None)!r} (sites: L"
                                 f"{', L'.join(map(str, sorted(self.sites)))})")
        return self.contexts[int(match[1])]

    def _check(self, node: ast.AST, *, value_ok: bool = False) -> None:
        contexts = list(self.contexts.values())
        if isinstance(node, ast.Name) and any(node.id in c.constants for c in contexts):
            if not all(node.id in c.constants for c in contexts):
                raise ConditionError(f"constant {node.id!r} is not declared at every site")
            return
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and \
                node.value.id == "feature":
            idx = node.slice
            if not (isinstance(idx, ast.Constant) and type(idx.value) is int):
                raise ConditionError("feature[...] takes one integer feature id")
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                node.func.id in FUNCTIONS:
            if node.keywords or len(node.args) != 1 or not isinstance(node.args[0], ast.Name):
                raise ConditionError(f"{node.func.id}(SET) takes one declared set name")
            name = node.args[0].id
            for ctx in contexts:
                pool = ctx.sets if node.func.id == "best_feature" else ctx.candidates
                if name not in pool:
                    raise ConditionError(f"{node.func.id}: no declared set {name!r}"
                                         + ("" if self.sites is None else f" at L{ctx.layer}"))
            return
        super()._check(node, value_ok=value_ok)

    def _eval(self, node: ast.AST, env: Any) -> Any:
        ctx = self._context(env)
        run = getattr(env, "run", None)
        if isinstance(node, ast.Name) and node.id in ctx.constants:
            return ctx.constants[node.id]
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and \
                node.value.id == "feature":
            return ctx.acts(self._need(run)).get(int(node.slice.value), 0.0)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                node.func.id in FUNCTIONS:
            name = node.args[0].id
            if node.func.id == "best_feature":
                return ctx.best_feature(self._need(run), name)
            return ctx.content(self._need(run), name)
        return super()._eval(node, env)

    @staticmethod
    def _need(run: Any) -> Any:
        if run is None:
            raise ConditionError("a feature predicate needs the run (ConditionEnv.run)")
        return run


__all__ = ["FeatureAddress", "FeatureCondition", "FeatureContext", "activations",
           "feature_sites", "feature_value", "parse_feature_address"]

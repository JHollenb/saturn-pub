"""Who reads a value: the attention reads of one source row, with mass, exact contribution and a
natively measured causal verdict.

The attention value mix is exact (P06/P07): at reader row ``j`` and layer ``l`` head ``h``
writes ``A_h = sum_s p[h, s] v[kappa(h), s]`` and the layer writes ``o = sum_h W_o,h A_h``. So
the part of ``o`` that source ``s`` sends through head ``h`` is ``p[h, s] W_o,h v[kappa(h), s]``
(``derived-exact``, float64 on captured operands). A declared consumer reads it through an exact
linear map: a margin ``M[a,b]`` or a logit through the final-norm readout with the final-norm
scale ``rho`` evaluated once at the native final state (P19, the convention of
``explain_margin``), a residual-stream port of the same row (``h``, ``r``, ``h_out`` of a later
point) as itself, and ``o`` of the read's own layer (the default consumer) as itself.

The causal verdict of a read is a native measurement: the source's weight in that head is set to
zero (``p[h, src] = 0``, a masked ``replace`` at the ``p`` port) through
``analysis.propagate.whatif``, which measures the consumer natively and restores the session.
The verdict follows whatif's P20 verdict (``verdict.decide``: the native change against the
emulated change, the base and edited chains with the dtype's rounding where eager HF rounds,
within one spacing at the consumer): ``causal`` when whatif finds an effect (the native change is
not zero, as the emulation predicts) and the change exceeds ``min_effect`` (default 0); ``not
causal at this dose`` when whatif finds no effect (the dtype shows no change at the consumer);
``unresolved`` (with the reason) otherwise, or when the measurement could not be made. Both facts
are kept when they differ: a read whose exact change is real but below the dtype's resolution
at the consumer is ``not causal at this dose`` with the reason ``no change at bfloat16
resolution; exact ΔM 0.33``, and a native change that is rounding only carries its exact change.
No worst-case rounding bound gates it. Mass and effect are always reported side by side; a small
mass never hides a read.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .accumulation import FP32_PREMISE, Accumulation, accumulation
from .finite import BOUND, EXACT, MEASURED, SUPPLIED, f64, num
from .forecast import Logit, Margin, Refusal
from .verdict import EFFECT, NO_EFFECT

RESIDUAL_PORTS = ("h", "r", "h_out")
VERDICTS = ("causal", "not causal at this dose", "unresolved")
_ROW = re.compile(r"t(?P<row>\d+)")
_SOURCE = re.compile(r"t(?P<row>\d+)(?:\.L(?P<layer>\d+)\.(?P<port>kr|kn|k|v))?")
_CELL = re.compile(r"cache\.L(?P<layer>\d+)\.(?P<port>k|v)\[(?P<sel>[^\]]*)\]")
_U32 = 2.0 ** -24


def _torch():
    import torch

    return torch


def _gamma(n: int, u: float) -> float:
    return n * u / (1.0 - n * u)


def unit_roundoff(dtype: Any) -> float:
    """Unit roundoff of a floating dtype (half its machine epsilon)."""
    return float(_torch().finfo(dtype).eps) / 2.0


def claim(value: Any, authority: str, proof: str | None = None) -> dict[str, Any]:
    """A labelled number for a scalar tensor; ``{max_abs, norm}`` (both labelled) for a vector."""
    value = f64(value)
    if value.dim() == 0:
        return num(float(value), authority, proof)
    flat = value.reshape(-1)
    return {"max_abs": num(float(flat.abs().max()), authority, proof),
            "norm": num(float(flat.norm()), authority, proof)}


def size(value: Any) -> float:
    """The magnitude used to rank contributions: ``|x|`` for a scalar, the norm for a vector."""
    value = f64(value)
    return float(value.abs()) if value.dim() == 0 else float(value.reshape(-1).norm())


def refused(reason: str) -> dict[str, Any]:
    return {"state": "refused", "reason": reason}


def unresolved(reason: str) -> dict[str, Any]:
    return {"state": "unresolved", "reason": reason}


def layer_module(adapter: Any, layer: int) -> Any:
    return getattr(adapter.backbone, adapter.spec.layers)[layer]


def computed(run: Any, address: str) -> bool:
    """Whether ``address`` (``t<row>.<stop>``) is at or before the cursor and inside this run."""
    key = run.parse(address)
    return key <= run.parse(run.where) and key[0] >= first_row(run)


def first_row(run: Any) -> int:
    """The first token row this run stepped (earlier rows were prefilled into the cache):
    ``Run.first_row``, the row of its root (``prefill=K`` steps rows from ``K``)."""
    return int(run.first_row)


# --------------------------------------------------------------------------- consumers


@dataclass
class Readout:
    """A consumer read through an exact linear map of residual-stream vectors.

    ``kind``: ``margin`` / ``logit`` (scalar: ``rho c . x`` plus ``bias``), ``residual`` (a
    residual port ``t<row>.L<L>.h|r|h_out``, read as itself) or ``o`` (``t<row>.L<L>.o``, read
    as itself). ``target`` is what ``propagate.whatif`` takes (``Margin``, ``Logit`` or a stop
    address). ``native`` is the consumer's native value (float64; None when not computed).
    """

    kind: str
    text: str
    row: int
    target: Any
    layer: int | None = None
    port: str | None = None
    vector: Any = None
    rho: float | None = None
    bias: float = 0.0
    native: Any = None
    native_authority: str | None = None
    premises: list[str] = field(default_factory=list)

    @property
    def scalar(self) -> bool:
        return self.kind in ("margin", "logit")

    def apply(self, x: Any) -> Any:
        """The consumer's share of a residual-stream vector ``x`` (float64)."""
        x = f64(x).reshape(-1)
        return self.vector @ x if self.scalar else x

    def reaches(self, layer: int, part: str) -> str | None:
        """None when the term ``part`` (``attn``/``mlp``) of ``layer`` at the consumer's row
        is an additive summand of the consumer, else why not."""
        if self.scalar:
            return None
        if self.kind == "o":
            if part == "attn" and layer == self.layer:
                return None
            return f"{self.text} is the attention output of L{self.layer} only"
        index = {"h": 0, "r": 1, "h_out": 2}[self.port]
        before = layer < self.layer or (layer == self.layer and (
            (part == "attn" and index >= 1) or (part == "mlp" and index >= 2)))
        if before:
            return None
        return f"L{layer}.{part} is not upstream of {self.text} in the residual stream"


def readout(run: Any, consumer: Any) -> Readout:
    """The ``Readout`` of a ``Margin``, a ``Logit`` or a stop address (``t<row>.L<L>.<port>``,
    a residual port or ``o``). A margin or logit needs the row's logits computed (the final
    state fixes ``rho``); ``Refusal`` otherwise and for any other port (no additive path)."""
    adapter = run.adapter
    if isinstance(consumer, (Margin, Logit)):
        row = consumer.row
        stop = f"t{row}.logits"
        if not computed(run, stop):
            raise Refusal(f"{stop} has not been computed at {run.where}: {consumer.text} needs "
                          f"it (the final state fixes the readout scale); run `until {stop}`")
        last = adapter.layers - 1
        h_last = f64(run.read(f"t{row}.L{last}.h_out")).reshape(-1)
        norm = getattr(adapter.backbone, adapter.spec.final_norm)
        gamma, eps = f64(norm.weight), float(norm.variance_epsilon)
        rho = float((h_last.pow(2).mean() + eps) ** -0.5)
        weight = adapter.model.lm_head.weight
        bias_vec = getattr(adapter.model.lm_head, "bias", None)
        logits = run.read(stop).reshape(-1)     # two entries are read: index, then cast
        info = run.read_info(stop)
        if isinstance(consumer, Margin):
            w = f64(weight[consumer.a]) - f64(weight[consumer.b])
            bias = 0.0 if bias_vec is None else float(f64(bias_vec[consumer.a])
                                                      - f64(bias_vec[consumer.b]))
            native = f64(logits[consumer.a]) - f64(logits[consumer.b])
            kind = "margin"
        else:
            w = f64(weight[consumer.i])
            bias = 0.0 if bias_vec is None else float(f64(bias_vec[consumer.i]))
            native = f64(logits[consumer.i])
            kind = "logit"
        return Readout(kind, consumer.text, row, consumer, vector=rho * gamma * w, rho=rho,
                       bias=bias, native=native, native_authority=info["authority"],
                       premises=["rho^F evaluated once at the native final state h_L (P19); "
                                 "not attributed to components"])
    if not isinstance(consumer, str):
        raise Refusal(f"unknown consumer {consumer!r}")
    key = run.parse(consumer)
    label = run.stops()[key[1]]
    match = re.fullmatch(r"L(?P<layer>\d+)\.(?P<port>\w+)", label)
    text = f"t{key[0]}.{label}"
    if match is None or match["port"] not in (*RESIDUAL_PORTS, "o"):
        raise Refusal(f"{text}: a consumer is a margin M[a,b], a logit, a residual port "
                      "(h, r, h_out) or o; other ports have no additive path from a head write")
    layer, port = int(match["layer"]), match["port"]
    native = authority = None
    if computed(run, text):
        native = f64(run.read(text)).reshape(-1)
        authority = run.read_info(text)["authority"]
    return Readout("o" if port == "o" else "residual", text, key[0], text, layer=layer,
                   port=port, native=native, native_authority=authority)


# --------------------------------------------------------------------------- the value mix


@dataclass
class HeadMix:
    """The value mix of reader row ``row`` at ``layer`` in float64 (P06/P07).

    ``p (H, T)`` native attention weights; ``values (H_KV, T, d_h)`` the value rows read
    (cache rows before ``row`` and the row's own ``v``); ``unit (H, T, d)`` is ``W_o,h
    v[kappa(h), s]`` so the part of ``o`` that source ``s`` sends through head ``h`` is ``p[h,
    s] unit[h, s]``. ``A`` (H, d_h) and ``o`` (d,) are the native captures when computed, else
    None. ``kappa`` maps a head to its KV head.
    """

    row: int
    layer: int
    p: Any
    values: Any
    unit: Any
    weight: Any
    kappa: list[int]
    A: Any = None
    o: Any = None
    dtype: Any = None
    info: dict[str, Any] = field(default_factory=dict)
    acc: Any = None          # accumulation.Accumulation of the run (None: float32 premise)

    def head_out(self) -> Any:
        """``A_h = sum_s p[h, s] v[kappa(h), s]`` in float64, shape (H, d_h)."""
        torch = _torch()
        v = self.values[self.kappa]                      # (H, T, d_h)
        return torch.einsum("hs,hse->he", self.p, v)

    def source_parts(self) -> Any:
        """``p[h, s] v[kappa(h), s]`` in float64, shape (H, T, d_h)."""
        return self.p[..., None] * self.values[self.kappa]

    def o_parts(self) -> Any:
        """``p[h, s] W_o,h v[kappa(h), s]`` in float64, shape (H, T, d)."""
        return self.p[..., None] * self.unit

    def checks(self) -> dict[str, Any]:
        """Native ``A`` against the float64 mix and native ``o`` against ``sum_h W_o,h A_h``:
        ``eta`` (measured) and its rounding ``bound`` (round to nearest; matmul partial sums kept
        at the precision of ``acc``, float32 by default, in any order, then one rounding to the
        model dtype). ``bound`` is None, with ``refused`` the reason, when that premise fails
        (``accumulation``: reduced-precision reductions and ``n u >= 1``)."""
        out: dict[str, Any] = {}
        u = unit_roundoff(self.dtype)
        acc = self.acc if self.acc is not None else Accumulation(_U32, FP32_PREMISE)
        premises = ["round to nearest, no overflow or underflow",
                    f"{acc.premise}, then one rounding to the model dtype"]

        def check(eta: Any, n: int, spread: Any, exact: Any, proof: str) -> dict[str, Any]:
            gamma = acc.gamma(n)
            if gamma is None:
                return {"eta": eta, "bound": None, "refused": acc.refusal(n),
                        "premises": premises, "proof": proof}
            return {"eta": eta, "bound": gamma * spread * (1 + u) + u * exact.abs(),
                    "premises": premises, "proof": proof}

        if self.A is not None:
            exact = self.head_out()
            spread = (self.p.abs()[..., None] * self.values[self.kappa].abs()).sum(1)
            out["value_mix"] = check(self.A - exact, self.p.shape[1], spread, exact, "P06")
        if self.o is not None and self.A is not None:
            flat = self.A.reshape(-1)
            exact = self.weight @ flat
            spread = self.weight.abs() @ flat.abs()
            out["head_write"] = check(self.o - exact, flat.numel(), spread, exact, "P07")
        return out


def head_mix(run: Any, row: int, layer: int, weights: dict[int, Any] | None = None) -> HeadMix:
    """The ``HeadMix`` of reader row ``row`` at ``layer``; ``Refusal`` when its ``p`` has not
    been computed in this run. ``weights``: a cache of float64 ``o_proj`` weights by layer,
    owned by the calling verb (each layer is cast once per verb, not once per reader row, and
    the cache is dropped when the verb returns)."""
    torch = _torch()
    adapter = run.adapter
    where = f"t{row}.L{layer}.p"
    if row < first_row(run):
        raise Refusal(f"t{row} was prefilled before this run started (first stepped row "
                      f"t{first_row(run)}): its attention weights were never stopped at")
    if not computed(run, where):
        raise Refusal(f"{where} has not been computed at {run.where}")
    heads, kv, dh = adapter.num_heads, adapter.num_kv_heads, adapter.head_dim
    p = run.read(where)
    native_dtype = p.dtype
    p64 = f64(p).reshape(heads, -1)
    cache = run.read_cache(layer, "v")[0]       # (H_KV, T_now, d_h): sliced, then cast
    if cache.shape[1] > row:
        values = f64(cache[:, : row + 1, :])
    else:
        own = f64(run.read(f"t{row}.L{layer}.v")).reshape(kv, 1, dh)
        values = torch.cat([f64(cache[:, :row, :]), own], 1)
    module = layer_module(adapter, layer).self_attn.o_proj
    if weights is None:
        weight = f64(module.weight)                      # (d, H d_h)
    else:
        if layer not in weights:
            weights[layer] = f64(module.weight)
        weight = weights[layer]
    blocks = weight.reshape(weight.shape[0], heads, dh).permute(1, 0, 2)   # (H, d, d_h)
    kappa = [h // (heads // kv) for h in range(heads)]
    unit = torch.einsum("hde,hse->hsd", blocks, values[kappa])
    mix = HeadMix(row, layer, p64, values, unit, weight, kappa, dtype=native_dtype,
                  acc=accumulation(run, native_dtype))
    mix.info["p"] = run.read_info(where)
    for port in ("A", "o"):
        address = f"t{row}.L{layer}.{port}"
        if computed(run, address):
            value = f64(run.read(address)).reshape(-1)
            setattr(mix, port, value.reshape(heads, dh) if port == "A" else value)
            mix.info[port] = run.read_info(address)
    if getattr(module, "bias", None) is not None and mix.o is not None:
        mix.o = mix.o - f64(module.bias)                 # the bias is not a read of a source
        mix.info["o_bias"] = True
    return mix


def head_mask(run: Any, row: int, layer: int, head: int, source: int | None = None) -> Any:
    """A bool mask over ``t<row>.L<layer>.p`` (source given) or ``.A`` (one head's block)."""
    torch = _torch()
    port = "p" if source is not None else "A"
    shape = tuple(run.address_shape(f"t{row}.L{layer}.{port}"))
    mask = torch.zeros(shape, dtype=torch.bool)
    if source is not None:
        mask[0, head, 0, source] = True
    else:
        dh = run.adapter.head_dim
        mask[..., head * dh:(head + 1) * dh] = True
    return mask


def edited_block(mix: HeadMix, run: Any, head: int) -> str | None:
    """Why head ``head``'s value mix does not reach ``o`` natively (an edit wrote its block of
    ``A``, or ``o`` itself), else None."""
    row, layer = mix.row, mix.layer
    address = f"t{row}.L{layer}.A"
    if "A" in mix.info:
        info = run.read_info(address, select=head_mask(run, row, layer, head))
        if info["edited"]:
            return f"{address}[h{head}] was written by an edit ({','.join(info['ops'])}): " \
                   "the value mix does not reach o"
    o = mix.info.get("o")
    if o is not None and o["kind"] != "native":
        return f"t{row}.L{layer}.o is {o['kind']} ({o['by']}): the head writes do not reach " \
               "the consumer as computed"
    return None


# --------------------------------------------------------------------------- sources


@dataclass(frozen=True)
class Source:
    """Cache cells of one source row: ``layers`` / ``kv`` / ``ports`` (``k``, ``v``) None means
    all. ``covers(layer, kv)`` asks about a read (any port); with ``port`` about one cell."""

    row: int
    layers: tuple[int, ...] | None = None
    kv: tuple[int, ...] | None = None
    text: str = ""
    ports: tuple[str, ...] | None = None

    def covers(self, layer: int, kv: int, port: str | None = None) -> bool:
        return ((self.layers is None or layer in self.layers)
                and (self.kv is None or kv in self.kv)
                and (port is None or self.ports is None or port in self.ports))


def parse_source(run: Any, text: str) -> Source:
    """``t<r>``, ``t<r>.L<l>.k|v|kn|kr`` or ``cache.L<l>.k|v[row=<r>]`` (with ``kv<i>``)."""
    text = text.strip()
    layers = run.adapter.layers
    match = _SOURCE.fullmatch(text)
    if match is not None:
        row = int(match["row"])
        layer = None if match["layer"] is None else int(match["layer"])
        if layer is not None and layer >= layers:
            raise Refusal(f"no layer {layer} in a {layers}-layer model")
        port = match["port"]
        return Source(row, None if layer is None else (layer,), None, text,
                      None if port is None else ("v",) if port == "v" else ("k",))
    match = _CELL.fullmatch(text)
    if match is None:
        raise Refusal(f"a read source is a token row t<r>, a cache-row occurrence t<r>.L<l>.k|v "
                      f"or a cache cell cache.L<l>.k|v[row=<r>], not {text!r}")
    layer = int(match["layer"])
    if layer >= layers:
        raise Refusal(f"no layer {layer} in a {layers}-layer model")
    row = kv = None
    for part in filter(None, (p.strip() for p in match["sel"].split(","))):
        key, _, value = part.partition("=")
        if not value and key.startswith("kv") and key[2:].isdigit():
            kv = int(key[2:])
        elif key == "kv" and value.isdigit():
            kv = int(value)
        elif key == "row" and value.isdigit():
            row = int(value)
        else:
            raise Refusal(f"{text}: a cache-cell source takes row=<r> and kv<i> only")
    if row is None:
        raise Refusal(f"{text}: name the source row (row=<r>)")
    if kv is not None and kv >= run.adapter.num_kv_heads:
        raise Refusal(f"{text}: no KV head {kv}")
    return Source(row, (layer,), None if kv is None else (kv,), text, (match["port"],))


def reader_rows(run: Any, source: Source, rows: str | None) -> list[int]:
    """Reader rows: the cursor's row by default; ``rows`` ``R``, ``A:B`` (half-open) or
    ``all`` (every stepped row from the source row to the cursor's row)."""
    here = int(_ROW.match(run.where)["row"])
    if rows is None:
        found = [here]
    elif rows == "all":
        found = list(range(max(source.row, first_row(run)), here + 1))
    elif re.fullmatch(r"\d+", rows):
        found = [int(rows)]
    elif re.fullmatch(r"\d+:\d+", rows):
        low, high = map(int, rows.split(":"))
        found = list(range(low, high))
    else:
        raise Refusal(f"rows is R, A:B or all, not {rows!r}")
    if not found:
        raise Refusal(f"rows={rows} names no row")
    late = [r for r in found if r > here]
    if late:
        raise Refusal(f"t{late[0]} has not been computed at {run.where}")
    early = [r for r in found if r < source.row]
    if early:
        raise Refusal(f"t{early[0]} cannot read t{source.row}: a row reads only itself and "
                      "earlier rows")
    return found


# --------------------------------------------------------------------------- reads


@dataclass
class Reads:
    """A ``reads`` result: ``report`` (bounded JSON, every number labelled) and the float64
    ``tensors`` behind it (for in-process callers and tests)."""

    report: dict[str, Any]
    tensors: dict[str, Any]


def _consumer_for(run: Any, consumer: Any, row: int, layer: int) -> Readout:
    """The declared consumer, or the default: ``o`` of the read's own row and layer."""
    if consumer is None:
        return readout(run, f"t{row}.L{layer}.o")
    return readout(run, consumer)


def _request(address: str, mask: Any, selection: str, target: Any) -> tuple[Any, Any]:
    """The what-if of one read: ``p`` at ``address`` set to 0 on ``mask``."""
    from .propagate import Edit

    return (Edit(address, 0.0, "replace", mask, selection,
                 "remove the source's weight in this head"), target)


def _verdict(result: Any, min_effect: float) -> dict[str, Any]:
    """The causal verdict of one read from its what-if ``result`` (a ``WhatIf``, or the
    ``Refusal`` the what-if gave before anything native ran)."""
    if isinstance(result, Refusal):
        return {"verdict": "unresolved", "reason": str(result)}
    delta = f64(result.tensors["native_delta"]).reshape(-1)
    bound = result.tensors.get("rounding_bound")
    out: dict[str, Any] = {
        "delta": result.report["measured"]["delta"],
        "min_effect": num(float(min_effect), SUPPLIED),
        "record": result.report["record"],
        "dose": "p[h, src] = 0 (the source removed from this head; the other weights kept)",
    }
    prediction = result.report["prediction"]
    out["predicted"] = (prediction["exact"]["value"] if prediction["state"] == "resolved"
                        else {"state": prediction["state"], "reason": prediction["reason"]})
    emulated = prediction.get("emulated")
    if emulated is not None:   # the emulated change (M51): what the verdict holds native against
        out["emulated"] = (emulated["value"] if "value" in emulated
                           else {"state": emulated["state"], "reason": emulated["reason"]})
    if bound is not None:   # the worst-case interval bound: shown beside, never decides
        out["rounding_bound"] = claim(f64(bound).reshape(-1).max(), BOUND)
    effect = result.report["effect"]
    out["effect"] = {key: value for key, value in effect.items() if key != "state"}
    out.update(_decide(delta, effect, min_effect))
    return out


def verdict(run: Any, address: str, mask: Any, selection: str, target: Any,
            min_effect: float) -> dict[str, Any]:
    """The causal verdict of one read: ``p`` at ``address`` set to 0 on ``mask`` through
    ``propagate.whatif`` (prediction sealed, measured natively, session restored)."""
    from .propagate import whatif

    try:
        result = whatif(run, *_request(address, mask, selection, target))
    except Refusal as error:
        result = error
    return _verdict(result, min_effect)


def _decide(delta: Any, effect: dict[str, Any], min_effect: float) -> dict[str, Any]:
    """The read's verdict from whatif's P20 verdict ``effect`` (``verdict.decide``): ``causal``
    on an effect beyond ``min_effect``; ``not causal at this dose`` on no effect (the reason
    says the dtype shows no change and gives the exact change); else ``unresolved`` with the
    whatif verdict and its reason (``not confirmed`` or ``unresolved``)."""
    if effect["verdict"] == NO_EFFECT:
        return {"verdict": "not causal at this dose",
                "reason": effect.get("note", f"no change at {effect['dtype']} resolution")}
    if effect["verdict"] == EFFECT:
        if size(delta) <= min_effect:
            return {"verdict": "unresolved", "reason": "changed, at or below min_effect"}
        return {"verdict": "causal", **({"reason": effect["note"]} if "note" in effect else {})}
    return {"verdict": "unresolved", "reason": f"whatif {effect['verdict']}: {effect['reason']}"}


def reads(run: Any, source: str | Source, *, rows: str | None = None, consumer: Any = None,
          top: int = 8, causal: bool = True, min_effect: float = 0.0) -> Reads:
    """The attention reads of ``source`` made at the reader rows (the cursor's row by default),
    one entry per (reader row, layer, head), ranked by the size of the exact contribution to
    the consumer (default: ``o`` of the read's own layer). The first ``top`` entries carry a
    native causal verdict (``causal``); the rest are summed into ``omitted`` (count and sums).
    Layers whose ``p`` has not been computed are listed under ``not_computed``."""
    if not isinstance(source, Source):
        source = parse_source(run, source)
    if top < 1:
        raise Refusal("top must be at least 1")
    if not math.isfinite(min_effect) or min_effect < 0:
        raise Refusal("min_effect must be a finite number >= 0")
    found = reader_rows(run, source, rows)
    layers = run.adapter.layers
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    checks: list[dict[str, Any]] = []
    weights: dict[int, Any] = {}        # float64 o_proj by layer, for this call only
    for j in found:
        for layer in range(layers):
            if source.layers is not None and layer not in source.layers:
                continue
            try:
                mix = head_mix(run, j, layer, weights)
            except Refusal as error:
                missing.append(f"t{j}.L{layer}: {error}")
                continue
            checks.append(mix.checks())
            try:
                reader = _consumer_for(run, consumer, j, layer)
            except Refusal as error:
                reader = error
            parts = mix.o_parts()
            heads = [h for h in range(mix.p.shape[0]) if source.covers(layer, mix.kappa[h])]
            for h in heads:
                s = source.row
                part = parts[h, s]
                entry: dict[str, Any] = {
                    "name": f"t{j}.L{layer}.h{h}", "address": f"t{j}.L{layer}.p[h{h},src={s}]",
                    "row": j, "layer": layer,
                    "mass": num(float(mix.p[h, s]), source_label(run, mix, h, s)),
                    "to_head": {"norm": num(float(mix.source_parts()[h, s].norm()), EXACT,
                                            "P06")},
                }
                blocked = edited_block(mix, run, h)
                entry["to_o"] = refused(blocked) if blocked else claim(part, EXACT, "P07")
                value = None
                if isinstance(reader, Refusal):
                    entry["to_consumer"] = refused(str(reader))
                elif blocked:
                    entry["to_consumer"] = refused(blocked)
                elif reader.reaches(layer, "attn") is not None or reader.row != j:
                    why = reader.reaches(layer, "attn") or (
                        f"{reader.text} is in another row: t{j} reaches it only through the "
                        "cache (no additive path)")
                    entry["to_consumer"] = refused(why)
                else:
                    value = reader.apply(part)
                    entry["to_consumer"] = claim(value, EXACT, "P19" if reader.scalar else "P07")
                entry["_rank"] = -1.0 if value is None else size(value)
                entry["_mask"] = (j, layer, h, s)
                # an edit that blocks the value mix decides the verdict too: not measured
                entry["_target"] = (None if isinstance(reader, Refusal) or blocked
                                    else reader.target)
                entry["_value"] = value
                entries.append(entry)
    weights.clear()
    mix = parts = None      # the last value mix is not needed by the verdicts' whatifs
    entries.sort(key=lambda e: (-e["_rank"], e["row"], e["layer"], e["name"]))
    listed, rest = entries[:top], entries[top:]
    measured: list[dict[str, Any]] = []
    requests = []
    for entry in listed:
        j, layer, h, s = entry["_mask"]
        if not causal:
            entry["causal"] = {"verdict": "unresolved", "reason": "causal=false: not measured"}
        elif entry["_target"] is None:
            entry["causal"] = {"verdict": "unresolved", "reason": entry["to_consumer"]["reason"]}
        else:
            measured.append(entry)
            requests.append(_request(f"t{j}.L{layer}.p", head_mask(run, j, layer, h, s),
                                     f"h{h},src={s}", entry["_target"]))
    if requests:    # one excursion for all the listed reads (the rewinds and the restore shared)
        from .propagate import whatif_many

        # the per-stage containment is not part of a read's verdict: not computed (M56c)
        for entry, result in zip(measured, whatif_many(run, requests, stages=False),
                                 strict=True):
            entry["causal"] = _verdict(result, min_effect)
    report: dict[str, Any] = {
        "source": {"row": source.row, "address": source.text or f"t{source.row}",
                   "token": int(run.tokens[source.row]) if source.row < len(run.tokens) else
                   None},
        "rows": found,
        "consumer": ("o of each read's layer (default)" if consumer is None
                     else getattr(consumer, "text", consumer)),
        "reads": [_public(e) for e in listed],
        "total": len(entries),
    }
    if report["source"]["token"] is None:
        report["source"].pop("token")
    if rest:
        values = [e["_value"] for e in rest if e["_value"] is not None]
        report["omitted"] = {
            "count": len(rest),
            "mass": num(sum(float(e["mass"]["value"]) for e in rest), EXACT),
            "to_consumer": _omitted_sum(rest, values, consumer),
            "meaning": f"+{len(rest)} more reads (ranked below the top {top}), summed",
        }
    report["checks"] = _check_summary(checks)
    if missing:
        report["not_computed"] = missing[:16]
        if len(missing) > 16:
            report["not_computed_total"] = len(missing)
    return Reads(report, {"entries": entries})


def _omitted_sum(rest: list[dict[str, Any]], values: list[Any], consumer: Any) -> dict:
    """The omitted reads' summed contribution, only when they share one consumer (a declared
    one, or by default ``o`` of one row and layer)."""
    if not values:
        return unresolved("no omitted read reaches the consumer")
    if consumer is None and len({(e["row"], e["layer"]) for e in rest}) > 1:
        return unresolved("the omitted reads have different consumers (o of their own layers): "
                          "their contributions do not add")
    if len(values) < len(rest):
        return unresolved("some omitted reads do not reach the consumer")
    return claim(sum(values[1:], values[0]), EXACT)


def source_label(run: Any, mix: HeadMix, head: int, source: int) -> str:
    info = run.read_info(f"t{mix.row}.L{mix.layer}.p",
                         select=head_mask(run, mix.row, mix.layer, head, source))
    return info["authority"] or MEASURED


def _public(entry: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in entry.items() if not k.startswith("_")}


def _check_summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    """The value-mix (P06) and head-write (P07) identities over every (row, layer) visited:
    the largest measured ``|eta|``, the largest ratio ``|eta| / bound`` and whether every
    element is within its bound."""
    out: dict[str, Any] = {}
    for name in ("value_mix", "head_write"):
        found = [c[name] for c in checks if name in c]
        if not found:
            out[name] = unresolved("not computed at the cursor (the port is past it)")
            continue
        failed = next((c["refused"] for c in found if c["bound"] is None), None)
        if failed is not None:
            out[name] = refused(failed)
            continue
        eta = max(float(c["eta"].abs().max()) for c in found)
        ratio = max(float((c["eta"].abs() / c["bound"].clamp_min(1e-300)).max())
                    for c in found)
        out[name] = {"eta_max_abs": num(eta, MEASURED, found[0]["proof"]),
                     "worst_ratio": num(ratio, MEASURED, found[0]["proof"]),
                     "within_bound": all(bool((c["eta"].abs() <= c["bound"]).all())
                                         for c in found),
                     "count": len(found), "premises": found[0]["premises"]}
    return out


# old private names, kept for one commit
_edited_block, _first_row, _label = edited_block, first_row, source_label

__all__ = ["HeadMix", "Readout", "Reads", "Source", "VERDICTS", "claim", "edited_block",
           "first_row", "head_mask", "head_mix", "parse_source", "reader_rows", "readout", "reads",
           "source_label", "verdict"]

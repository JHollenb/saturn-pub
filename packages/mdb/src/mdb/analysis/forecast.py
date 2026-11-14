"""Sealed forecasts of native values at the cursor, verified after the native step.

A forecast is computed only from values that exist at the cursor of an ``engine.run.Run``: the
ports at or before the cursor (``Run.read``), the key/value cache rows of earlier tokens
(``Run.read_cache``) and the weights. ``RowModel`` evaluates the port program's identities
(``finite``: P01 lookup and adds, P02 RMS, P04 affine reads and RoPE, P05 scores and softmax,
P10 SwiGLU, P19 readout) in float64 from those values, one stop of the token row at a time;
nothing native runs for the target. The prediction is ``derived-exact`` (exact real arithmetic
on captured operands) and carries a per-element rounding ``bound``: how far the native value can
sit from it under stated premises (round to nearest, the matmul accumulation precision of
``accumulation.accumulation`` (float32 unless the run's numeric scope allows CUDA reduced-precision
reductions, when a matmul bound is refused past ``n u >= 1``), float32 library functions within ``LIBRARY_ULPS`` ulp, no overflow or underflow). The bound of a
stop whose operands are themselves predictions propagates the operands' bounds through the
identity (linear maps exactly, SiLU by its Lipschitz constant, RMS and softmax by their exact
range over the operand box), so a forecast several stops ahead keeps a bound.

``EmulatedRowModel`` evaluates the same chain with the run's dtype rounding at every point where
eager HF rounds, each step with the backend's own kernels: a second prediction, of the native
bits (``derived-exact`` under ``EMULATION_PREMISE``). ``verify`` decides on it (M42): the exact
chain can sit tens of spacings from native where rounded attention scores tie, and that
``rounding_effect`` (native - exact) is reported beside the verdict, not hidden in a floor.

``seal`` stores a sealed record (``mdb.core.records``, kind ``forecast``, no arrays: the
prediction is reached by an occurrence pointer and its digest) in the run's ledger. ``verify``
compares every open forecast whose target the cursor has passed with the native value, proves
order (the sealing stop precedes the target, and the session's receipts after sealing lead from
the sealed cut to the current one) and reports an edit staged or committed after sealing instead
of counting it as a failure. A forecast whose target was never reached stays open.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mdb.core.bounded import MAX_ITEMS, bound_list, load_json, put_json
from mdb.core.pointers import occurrence_pointer, seal_digest
from mdb.core.records import declare_kind, num, seal_record

from .accumulation import accumulation, bound_state
from .explain import _readout_refusal, _rope_refusal
from . import verdict as _verdict
from .finite import BOUND, EXACT, f64

SCHEMA = "mdb-forecast-v1"
U32 = 2.0 ** -24          # unit roundoff of float32, the accumulation precision
LIBRARY_ULPS = 4          # premise: float32 exp, rsqrt, sin, cos within 4 ulp
SILU_LIPSCHITZ = 1.1      # sup |silu'(g)| = 1.0998 (attained near g = 2.4)
NO_IDENTITY = {"start": "the token list is an input of the row, not a computed value"}

FORECAST = declare_kind(
    "forecast",
    required=("schema", "target", "sealed_at", "method", "operands", "predicted", "bound"),
    plain_numeric=("sealed_at.row", "sealed_at.index", "sealed_at.transition",
                   "sealed_at.receipts.count", "target.row", "target.index", "target.indices",
                   "operands.*.shape", "predicted.shape", "predicted.token",
                   "edits_at_seal.*.row", "edits_at_seal.*.layer",
                   "method.summary.count", "method.summary.global_steps.count",
                   "method.summary.proof_counts.*.count", "method.summary.layer_counts.*.layer",
                   "method.summary.layer_counts.*.count", "method.summary.chain_pointer.bytes",
                   "operands_summary.count", "operands_summary.listed",
                   "operands_summary.rest.count", "operands_summary.pointer.bytes",
                   "emulated.shape", "emulated.token", "emulated.rounding_origin.head"),
)

# (function, proof) of the identity that computes each port from its producers
IDENTITY = {
    "h": ("copy", "P01"), "a": ("rms_norm", "P02"), "q": ("affine", "P04"),
    "k": ("affine", "P04"), "v": ("affine", "P04"), "qn": ("rms_norm", "P02"),
    "kn": ("rms_norm", "P02"), "qr": ("rope_rotate", "P04"), "kr": ("rope_rotate", "P04"),
    "s": ("scores", "P05"), "p": ("softmax", "P05"), "A": ("transport", "P01"),
    "o": ("head_write", "P01"), "r": ("residual_add", "P01"), "n": ("rms_norm", "P02"),
    "g": ("affine", "P10"), "u": ("affine", "P10"), "sg": ("swiglu_gate", "P10"),
    "f": ("swiglu_product", "P10"), "m": ("affine", "P10"), "h_out": ("residual_add", "P01"),
    "emb": ("embedding_lookup", "P01"), "fn": ("rms_norm", "P02"),
    "logits": ("readout_logits", "P19"), "sample": ("argmax", None),
}
GLOBAL_OPERATORS = {
    "emb": "emb = embed(token)", "fn": "fn = final_norm(h_out of the last layer)",
    "logits": "logits = lm_head(fn)", "sample": "sample = argmax(logits) (greedy)",
}


class Refusal(ValueError):
    """A forecast or prediction whose premise fails; ``reason`` says which."""


def _torch():
    import torch

    return torch


# Elements of a weight cast to float64 at one time by ``_affine`` (2^25 = 256 MiB). A larger
# weight (the unembedding: vocabulary x width) is multiplied a block of rows at a time, so a
# readout never holds a float64 copy of the whole matrix.
_CHUNK = 1 << 25


def _gamma(n: int, u: float) -> float:
    """``gamma_n = n u / (1 - n u)`` (Higham): relative error of an n-term float sum."""
    return n * u / (1.0 - n * u)


@dataclass(frozen=True)
class Margin:
    """The declared margin ``M[a,b] = logits[a] - logits[b]`` of token row ``row``."""

    row: int
    a: int
    b: int

    @property
    def text(self) -> str:
        return f"t{self.row}.M[{self.a},{self.b}]"


@dataclass(frozen=True)
class Logit:
    """One logit ``logits[i]`` of token row ``row``."""

    row: int
    i: int

    @property
    def text(self) -> str:
        return f"t{self.row}.logits[i={self.i}]"


# --------------------------------------------------------------------------- row model


def producers(program: Any, layers: int, label: str) -> tuple[str, ...]:
    """Stops whose values the identity of stop ``label`` reads in its own token row (the key /
    value cache rows of earlier tokens aside)."""
    if label in ("start", "emb"):
        return ()
    if label == "fn":
        return (f"L{layers - 1}.h_out",)
    if label in ("logits", "sample"):
        return ("fn",) if label == "logits" else ("logits",)
    layer, port = _split(label)
    if port == "h":
        return ("emb",) if layer == 0 else (f"L{layer - 1}.h_out",)
    spec = program.spec(port)
    return tuple(f"L{layer}.{name}" for name in spec.inputs if name not in ("cache",
                                                                           "prev_layer"))


class RowModel:
    """Float64 values of the stops of one token row, each from the identity on its producers.

    ``known(label)`` returns the native tensor of a stop (``emb``, ``L<i>.<port>``, ``fn``,
    ``logits``, ``sample``) when it is to be taken as captured, else None; every other stop is
    computed from its producers. ``overrides`` maps a stop to a function ``(value, bound) ->
    (value, bound)`` applied to it (the edits of ``propagate``; bound None when unknown).
    ``past(layer, kind)`` supplies the earlier rows' cache ``(value, bound)`` (default: the run's
    cache, bound 0) and ``token()`` the row's token id ``(id, reason)`` (reason None when the id
    is certain; default: the run's token list). Both are generator functions: a value of
    another row model is read with ``yield from model.need(label)``. With ``bounds=True`` each
    value carries a per-element rounding bound (``bound(label)``; None with a reason in
    ``unbounded`` when it cannot be stated). Values use flat shapes: ``(d,)`` for residual
    ports, ``(H, d_h)`` for ``qn``/``qr``, ``(H_KV, T, d_h)`` for ``kr``, ``(H, T)`` for
    ``s``/``p``, ``()`` for ``sample``.

    Evaluation is iterative: every step is a generator that yields the ``(model, label)`` it
    needs, and ``get`` runs them from one explicit stack, in the order a recursive evaluation
    would. A chain of row models (``propagate``: each row reads the earlier rows' cache and
    sampled token) therefore has no depth limit.
    """

    def __init__(self, run: Any, row: int, known: Callable[[str], Any], *, bounds: bool = True,
                 overrides: dict[str, Callable[[Any, Any], tuple[Any, Any]]] | None = None,
                 past: Callable[[int, str], tuple[Any, Any]] | None = None,
                 token: Callable[[], tuple[int, str | None]] | None = None):
        torch = _torch()
        adapter = run.adapter
        self.run, self.row, self.known = run, row, known
        self.program = run.program
        self.adapter = adapter
        self.config = adapter.model.config
        self.layers = adapter.layers
        self.heads, self.kv_heads, self.head_dim = (adapter.num_heads, adapter.num_kv_heads,
                                                    adapter.head_dim)
        dtype = adapter.dtype
        self.out16 = dtype in (torch.bfloat16, torch.float16)
        self.u = float(torch.finfo(dtype).eps) / 2.0
        self.acc = accumulation(run)   # matmul accumulation premise, from the numeric scope
        self._failed: str | None = None
        self.bounds = bounds
        self.overrides = dict(overrides or {})
        self.past, self.token = past, token
        self.values: dict[str, Any] = {}
        self.bnds: dict[str, Any] = {}
        self.unbounded: dict[str, str] = {}
        self.captured: dict[str, Any] = {}
        self.computed: list[str] = []
        self.premises: set[str] = set()
        self.cache_reads: dict[str, Any] = {}

    # --- structure -----------------------------------------------------------------------

    def producers(self, label: str) -> tuple[str, ...]:
        return producers(self.program, self.layers, label)

    def shape(self, label: str) -> tuple[int, ...]:
        d, heads, kv, dh = self.adapter.hidden_size, self.heads, self.kv_heads, self.head_dim
        if label in ("emb", "fn"):
            return (d,)
        if label == "logits":
            return (int(self.config.vocab_size),)
        if label == "sample":
            return ()
        port = _split(label)[1]
        width = {"q": heads * dh, "A": heads * dh, "k": kv * dh, "v": kv * dh,
                 "g": self.config.intermediate_size, "u": self.config.intermediate_size,
                 "sg": self.config.intermediate_size, "f": self.config.intermediate_size}
        if port in ("qn", "qr"):
            return (heads, dh)
        if port == "kn":
            return (kv, dh)
        if port == "kr":
            return (kv, self.row + 1, dh)
        if port in ("s", "p"):
            return (heads, self.row + 1)
        return (int(width.get(port, d)),)

    # --- values ---------------------------------------------------------------------------

    def get(self, label: str) -> Any:
        if label not in self.values:
            _evaluate(self, label)
        return self.values[label]

    def bound(self, label: str) -> Any:
        self.get(label)
        return self.bnds.get(label)

    def need(self, label: str) -> Any:
        """Generator: ``(value, bound)`` of ``label`` once it is computed (``yield from`` it in
        a step, a ``past`` or a ``token``; the driver computes the stop first when needed)."""
        yield self, label
        return self.values[label], self.bnds.get(label) if self.bounds else None

    def _compute(self, label: str) -> Any:
        torch = _torch()
        native = self.known(label)
        if native is not None:
            value = f64(native)
            if value is native:     # already float64 on the CPU: keep the capture unshared
                value = value.clone()
            value = value.reshape(self.shape(label))
            self.captured[label] = native
            bound = torch.zeros_like(value) if self.bounds else None
        else:
            if label == "start":
                raise Refusal(NO_IDENTITY["start"])
            value, bound = yield from self._step(label)
            self.computed.append(label)
            if bound is None and self._failed is not None:
                self.unbounded[label] = self._failed     # a premise of this step's bound fails
            self._failed = None
            if bound is not None and not bool((torch.isfinite(bound) & (bound >= 0)).all()):
                bound = None
                self.unbounded[label] = ("the propagated bound is not finite: the operands' "
                                         "bounds are too wide for this identity")
        override = self.overrides.get(label)
        if override is not None:
            value, bound = override(value, bound)
            if bound is None:
                self.unbounded[label] = "value written by an edit with no stated bound"
        self.values[label] = value
        if bound is not None:
            self.bnds[label] = bound
        elif self.bounds and label not in self.unbounded:
            self.unbounded[label] = next(
                (self.unbounded[p] for p in self.producers(label) if p in self.unbounded),
                "no rounding bound for this step")

    def _weight(self, module: Any, name: str = "weight") -> Any:
        """A float64 copy of one weight, for the caller's one use. Nothing is kept: every stop
        is computed once per row model, so a kept copy would only hold memory (a whole layer is
        about 390 MB in float64 at d=1536, d_ff=8960)."""
        tensor = getattr(module, name, None)
        return None if tensor is None else f64(tensor)

    def _layer(self, layer: int) -> Any:
        return getattr(self.adapter.backbone, self.adapter.spec.layers)[layer]

    def _past(self, layer: int, kind: str) -> Any:
        """Generator: ``(value, bound)`` of cache rows ``0..row-1`` of ``layer`` (post-RoPE keys
        or values), ``(H_KV, row, d_h)``: from ``past`` when given, else as captured (bound
        0)."""
        if self.past is not None:
            value, bound = yield from self.past(layer, kind)
            return value, bound if self.bounds else None
        name = f"cache.L{layer}.{kind}"
        if name not in self.cache_reads:
            self.cache_reads[name] = self.run.read_cache(layer, kind)[..., : self.row, :]
        value = f64(self.cache_reads[name])[0]
        return value, _torch().zeros_like(value) if self.bounds else None

    # --- steps ----------------------------------------------------------------------------

    def _step(self, label: str) -> Any:
        """Generator: ``(value, bound)`` of ``label`` from the identity on its producers."""
        torch = _torch()
        if label == "emb":
            embed = getattr(self.adapter.backbone, self.adapter.spec.embed)
            if self.token is None:
                token, doubt = int(self.run.read(f"t{self.row}.start").reshape(-1)[-1]), None
            else:
                token, doubt = yield from self.token()
            value = f64(embed.weight[token])  # the one row, not the vocabulary
            if doubt is not None:
                self.unbounded["emb"] = doubt
                return value, None
            return value, torch.zeros_like(value) if self.bounds else None
        if label == "fn":
            norm = getattr(self.adapter.backbone, self.adapter.spec.final_norm)
            x, bx = yield from self.need(f"L{self.layers - 1}.h_out")
            return self._rms(x, bx, norm, self.adapter.hidden_size)
        if label == "logits":
            refusal = _readout_refusal(self.config)
            if refusal is not None:
                raise Refusal(refusal)
            head = self.adapter.model.lm_head
            x, bx = yield from self.need("fn")
            return self._affine(x, bx, head)
        if label == "sample":
            x, bx = yield from self.need("logits")
            return self._argmax(x, bx)
        layer, port = _split(label)
        module = self._layer(layer)
        attn, mlp = module.self_attn, module.mlp
        here = f"L{layer}."
        if port == "h":
            return (yield from self.need(self.producers(label)[0]))
        if port == "a":
            x, bx = yield from self.need(here + "h")
            return self._rms(x, bx, module.input_layernorm, self.adapter.hidden_size)
        if port in ("q", "k", "v"):
            x, bx = yield from self.need(here + "a")
            return self._affine(x, bx, getattr(attn, f"{port}_proj"))
        if port in ("qn", "kn"):
            source = getattr(attn, f"{port[0]}_norm")
            value, bound = yield from self.need(here + port[0])
            rows = self.heads if port == "qn" else self.kv_heads
            shaped = None if bound is None else bound.reshape(rows, self.head_dim)
            return self._rms(value.reshape(rows, self.head_dim), shaped, source, self.head_dim)
        if port == "qr":
            source = here + ("qn" if "qn" in self.program.names else "q")
            value, bound = yield from self.need(source)
            return self._rope(value.reshape(self.heads, self.head_dim),
                              None if bound is None else bound.reshape(self.heads, self.head_dim))
        if port == "kr":
            source = here + ("kn" if "kn" in self.program.names else "k")
            value, bound = yield from self.need(source)
            shape = (self.kv_heads, self.head_dim)
            new, new_bound = self._rope(value.reshape(shape),
                                        None if bound is None else bound.reshape(shape))
            past, past_bound = yield from self._past(layer, "k")
            out = torch.cat((past, new.unsqueeze(1)), dim=1)
            if new_bound is None or past_bound is None:
                return out, None
            return out, torch.cat((past_bound, new_bound.unsqueeze(1)), dim=1)
        if port == "s":
            self._attention_premises()
            q, bq = yield from self.need(here + "qr")
            k, bk = yield from self.need(here + "kr")
            return self._scores(q, bq, k, bk, float(attn.scaling))
        if port == "p":
            self._attention_premises()
            s, bs = yield from self.need(here + "s")
            return self._softmax(s, bs)
        if port == "A":
            v, bv = yield from self.need(here + "v")
            past, past_bound = yield from self._past(layer, "v")
            values = torch.cat((past, v.reshape(self.kv_heads, 1, self.head_dim)), dim=1)
            bound = None if bv is None or past_bound is None else torch.cat(
                (past_bound, bv.reshape(self.kv_heads, 1, self.head_dim)), dim=1)
            p, bp = yield from self.need(here + "p")
            return self._transport(p, bp, values, bound)
        if port == "o":
            x, bx = yield from self.need(here + "A")
            return self._affine(x, bx, attn.o_proj)
        if port in ("r", "h_out"):
            left, right = ("h", "o") if port == "r" else ("r", "m")
            x, bx = yield from self.need(here + left)
            y, by = yield from self.need(here + right)
            return self._add(x, bx, y, by)
        if port == "n":
            x, bx = yield from self.need(here + "r")
            return self._rms(x, bx, module.post_attention_layernorm, self.adapter.hidden_size)
        if port in ("g", "u"):
            x, bx = yield from self.need(here + "n")
            return self._affine(x, bx, mlp.gate_proj if port == "g" else mlp.up_proj)
        if port == "sg":
            x, bx = yield from self.need(here + "g")
            return self._silu(x, bx)
        if port == "f":
            x, bx = yield from self.need(here + "sg")
            y, by = yield from self.need(here + "u")
            return self._product(x, bx, y, by)
        if port == "m":
            x, bx = yield from self.need(here + "f")
            return self._affine(x, bx, mlp.down_proj)
        raise Refusal(f"no identity for port {port!r}")

    def _attention_premises(self) -> None:
        if getattr(self.config, "use_sliding_window", False):
            raise Refusal("sliding-window attention is not modelled")
        self.premises.add("single pending row: every cached source is visible and the "
                          "attention mask adds 0")

    # --- identities with their rounding bounds --------------------------------------------

    def _out_round(self) -> float:
        """Relative error of one rounding to the model dtype after a float32 computation."""
        return self.u if self.out16 else 0.0

    def _gamma(self, n: int) -> float | None:
        """``gamma_n`` of an ``n``-term native matmul sum under ``self.acc``; None (the reason
        kept for ``_compute``) when its premise fails."""
        gamma = self.acc.gamma(n)
        if gamma is None:
            self._failed = self.acc.refusal(n)
        else:
            self.premises.add(self.acc.premise)
        return gamma

    def _affine(self, x: Any, bx: Any, module: Any) -> tuple[Any, Any]:
        torch = _torch()
        native, bias = module.weight, self._weight(module, "bias")
        want = self.bounds and bx is not None
        step = max(1, _CHUNK // int(native.shape[1]))
        values, sizes, spreads = [], [], []
        for start in range(0, int(native.shape[0]), step):
            weight = f64(native[start:start + step])  # one block of output rows
            values.append(x @ weight.T)
            if want:
                absw = weight.abs()
                sizes.append((x.abs() + bx) @ absw.T)
                spreads.append(bx @ absw.T)
            del weight
        one = len(values) == 1
        value = (values[0] if one else torch.cat(values, -1)) + (0 if bias is None else bias)
        if not want:
            return value, None
        size = ((sizes[0] if one else torch.cat(sizes, -1))
                + (0 if bias is None else bias.abs()))
        spread = spreads[0] if one else torch.cat(spreads, -1)
        gamma = self._gamma(x.shape[-1] + (bias is not None))
        if gamma is None:
            return value, None
        rounds = (1 if self.out16 else 0) + (1 if bias is not None else 0)
        output = (1 + (self.u if self.out16 else U32)) ** rounds - 1
        rounding = gamma * size + output * (1 + gamma) * size
        return value, spread + rounding + torch.zeros_like(value)

    def _add(self, x: Any, bx: Any, y: Any, by: Any) -> tuple[Any, Any]:
        value = x + y
        if not self.bounds or bx is None or by is None:
            return value, None
        spread = bx + by
        rel = (1 + self.u) * (1 + (U32 if self.out16 else 0.0)) - 1
        return value, spread + rel * (value.abs() + spread)

    def _rms(self, x: Any, bx: Any, module: Any, width: int) -> tuple[Any, Any]:
        torch = _torch()
        gain, eps = self._weight(module), float(module.variance_epsilon)
        shape = x.shape
        rows = x.reshape(-1, width)
        rho = 1.0 / torch.sqrt(eps + (rows * rows).mean(-1, keepdim=True))
        value = (gain * rho * rows).reshape(shape)
        if not self.bounds or bx is None:
            return value, None
        brows = bx.reshape(-1, width)
        norm = rows.norm(dim=-1, keepdim=True)
        spread = brows.norm(dim=-1, keepdim=True)

        def rho_at(length):
            return 1.0 / torch.sqrt(eps + length * length / width)

        drho = torch.maximum(rho_at((norm - spread).clamp_min(0)) - rho,
                             rho - rho_at(norm + spread))
        prop = gain.abs() * (drho * (rows.abs() + brows) + rho * brows)
        cast = (1 + self.u) ** 2 if self.out16 else 1 + U32
        rel = (1 + _gamma(width + 2 + LIBRARY_ULPS, U32)) * (1 + U32) * cast - 1
        self.premises.add("RMSNorm in float32: mean of squares, rsqrt, scale, cast, gain")
        return value, (prop + rel * ((gain * rho * rows).abs() + prop)).reshape(shape)

    def _rope(self, x: Any, bx: Any) -> tuple[Any, Any]:
        """Default RoPE at the row's position on ``(heads, d_h)`` (half-split pairs)."""
        torch = _torch()
        refusal = _rope_refusal(self.config)
        if refusal is not None:
            raise Refusal(refusal)
        half = x.shape[-1] // 2
        angle = self.row * self._inv_freq(x.shape[-1])
        cos, sin = torch.cos(angle), torch.sin(angle)
        first, second = x[..., :half], x[..., half:]
        value = torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
        if not self.bounds or bx is None:
            return value, None
        bfirst, bsecond = bx[..., :half], bx[..., half:]
        trig = 32 * U32 * angle.abs() + LIBRARY_ULPS * U32
        cast = self.u if self.out16 else 0.0
        ec, es = trig + cast * (cos.abs() + trig), trig + cast * (sin.abs() + trig)
        mag_first, mag_second = first.abs() + bfirst, second.abs() + bsecond
        rel = (1 + self.u) ** 2 - 1

        def half_bound(own, own_b, other, other_b, c_abs, s_abs):
            prop = c_abs * own_b + s_abs * other_b
            size = own * (c_abs + ec) + other * (s_abs + es)
            return prop + own * ec + other * es + rel * size

        self.premises.add("RoPE: cos/sin of float32 angles (|angle error| <= 32 u32 |angle|), "
                          "cast to the model dtype, two products and one sum rounded")
        bound = torch.cat((
            half_bound(mag_first, bfirst, mag_second, bsecond, cos.abs(), sin.abs()),
            half_bound(mag_second, bsecond, mag_first, bfirst, cos.abs(), sin.abs()),
        ), dim=-1)
        return value, bound

    def _rotary(self) -> Any:
        return getattr(self.adapter.backbone, "rotary_emb", None)

    def _inv_freq(self, width: int) -> Any:
        """The RoPE inverse frequencies the model runs with: float64 of its float32
        ``rotary_emb.inv_freq`` buffer (M49: the buffer, not ``theta^(-2i/d)`` recomputed in
        float64, which differs from it by the buffer's float32 rounding); the formula only for a
        model without the buffer."""
        buffer = getattr(self._rotary(), "inv_freq", None)
        if buffer is not None:
            return f64(buffer)
        torch = _torch()
        rope = getattr(self.config, "rope_parameters", None) or {}
        theta = float(rope.get("rope_theta", getattr(self.config, "rope_theta", 10000.0)))
        return theta ** (-2.0 * torch.arange(width // 2, dtype=torch.float64) / width)

    def _kappa(self, tensor: Any) -> Any:
        groups = self.heads // self.kv_heads
        return tensor.repeat_interleave(groups, dim=0)

    def _scores(self, q: Any, bq: Any, k: Any, bk: Any, scaling: float) -> tuple[Any, Any]:
        torch = _torch()
        keys = self._kappa(k)
        value = scaling * torch.einsum("hd,htd->ht", q, keys)
        if not self.bounds or bq is None or bk is None:
            return value, None
        bkeys = self._kappa(bk)
        size = torch.einsum("hd,htd->ht", q.abs() + bq, keys.abs() + bkeys)
        prop = scaling * (size - torch.einsum("hd,htd->ht", q.abs(), keys.abs()))
        gamma = self._gamma(q.shape[-1])
        if gamma is None:
            return value, None
        dot = gamma * size + self._out_round() * (1 + gamma) * size
        product = (1 + U32) ** 2 * (1 + (self.u if self.out16 else U32))
        return value, prop + scaling * dot * product + (product - 1) * scaling * size * (
            1 + gamma) * (1 + self._out_round())

    def _softmax(self, s: Any, bs: Any) -> tuple[Any, Any]:
        torch = _torch()
        value = torch.softmax(s, dim=-1)
        if not self.bounds or bs is None:
            return value, None
        spread = bs.max(dim=-1, keepdim=True).values
        prop = value * torch.expm1(2 * spread)
        shift = (s - s.max(dim=-1, keepdim=True).values).abs() + 2 * spread
        each = torch.expm1(U32 * shift) + LIBRARY_ULPS * U32 * (1 + torch.expm1(U32 * shift))
        worst = each.max(dim=-1, keepdim=True).values
        gamma = _gamma(s.shape[-1], U32)
        rel = ((1 + each) * (1 + gamma) * (1 + U32) ** 2 / (1 - worst - gamma)
               * (1 + self._out_round()) - 1)
        self.premises.add("softmax in float32: exp(s - max) / sum, cast to the model dtype")
        return value, prop + rel * value * torch.exp(2 * spread)

    def _transport(self, p: Any, bp: Any, values: Any, bv: Any) -> tuple[Any, Any]:
        torch = _torch()
        mix = self._kappa(values)
        value = torch.einsum("ht,htd->hd", p, mix).reshape(-1)
        if not self.bounds or bp is None or bv is None:
            return value, None
        bmix = self._kappa(bv)
        size = torch.einsum("ht,htd->hd", p.abs() + bp, mix.abs() + bmix).reshape(-1)
        prop = size - torch.einsum("ht,htd->hd", p.abs(), mix.abs()).reshape(-1)
        gamma = self._gamma(p.shape[-1])
        if gamma is None:
            return value, None
        return value, prop + gamma * size + self._out_round() * (1 + gamma) * size

    def _silu(self, g: Any, bg: Any) -> tuple[Any, Any]:
        torch = _torch()
        value = g * torch.sigmoid(g)
        if not self.bounds or bg is None:
            return value, None
        prop = SILU_LIPSCHITZ * bg
        rel = (1 + _gamma(3 + LIBRARY_ULPS, U32)) * (1 + self._out_round()) - 1
        self.premises.add("SiLU in float32 as g / (1 + exp(-g)), cast to the model dtype")
        return value, prop + rel * (value.abs() + prop)

    def _product(self, x: Any, bx: Any, y: Any, by: Any) -> tuple[Any, Any]:
        value = x * y
        if not self.bounds or bx is None or by is None:
            return value, None
        prop = x.abs() * by + bx * y.abs() + bx * by
        return value, prop + self.u * (value.abs() + prop)

    def _argmax(self, logits: Any, bl: Any) -> tuple[Any, Any]:
        torch = _torch()
        index = int(torch.argmax(logits))
        value = torch.tensor(float(index), dtype=torch.float64)
        if not self.bounds or bl is None:
            return value, None
        if float(bl.max()) == 0.0:
            return value, torch.zeros_like(value)  # captured logits: the same first-index argmax
        rest = torch.cat((logits[:index] + bl[:index], logits[index + 1:] + bl[index + 1:]))
        if float(logits[index] - bl[index]) > float(rest.max()):
            return value, torch.zeros_like(value)
        self.unbounded["sample"] = "argmax not certified: the top logit's bound overlaps another"
        return value, None


EMULATION_PREMISE = "rounding points of eager HF attention/MLP on this backend"
EMULATION_PREMISES = (
    "every step runs as eager HF writes it, on the dtype values of its operands, with the "
    "torch kernels of the run's device: affine maps as F.linear (bias fused), scores as "
    "matmul(q, k^T) x scaling and the value mix as matmul(p, v) at the native call's batch-1 "
    "shapes, RMSNorm in float32 (cast, then gain), RoPE with float32 cos/sin of float32 angles "
    "from the model's inv_freq buffer, softmax in float32 cast to the dtype, SiLU, gate x up, "
    "residual adds",
    "the backend's kernels are deterministic at those shapes (the same operands give the same "
    "bits)",
    "single pending row: every cached source is visible and the attention mask adds 0",
)
EMULATION_SUMS = ("kernel", "round-once")


class EmulatedRowModel(RowModel):
    """The chain of ``RowModel`` with the run's dtype rounding at every point where eager HF
    rounds (``EMULATION_PREMISES``): a prediction of the native bits, not of the real-number
    value; ``derived-exact`` under ``EMULATION_PREMISE``, no rounding bound.

    ``sums="kernel"`` (default) runs each matmul with the backend's own kernel at the native
    shapes; ``"round-once"`` takes its exact float64 sum rounded once to the dtype (float32
    accumulation then one rounding). Measured, every stop from the sealing stop to the logits:
    tiny qwen2/qwen3/llama/mistral (CPU, 2-28 layers, 3 prompts) ``kernel`` bit-exact in bf16
    and fp32, ``round-once`` bit-exact in bf16 and 3-10.5 spacings off in fp32; Qwen2.5-1.5B
    on CUDA (3 prompts, sealed at L0.h and L14.h; bf16 28 layers and an fp32 2-layer cut)
    ``kernel`` bit-exact (100% of elements at every stop, M[a,b] equal), ``round-once`` 0-8
    spacings in bf16 (1-ulp sum differences carried forward) and 1-35 in fp32, where the
    exact chain is 35-78 spacings off at L0.m in bf16 (the rounded-score ties of M42)."""

    def __init__(self, run: Any, row: int, known: Callable[[str], Any], *,
                 sums: str = "kernel", **options: Any):
        if sums not in EMULATION_SUMS:
            raise ValueError(f"sums must be one of {EMULATION_SUMS}")
        options["bounds"] = False
        super().__init__(run, row, known, **options)
        self.dtype = self.adapter.dtype
        self.sums = sums
        self.device = getattr(self.adapter, "device", None) or "cpu"

    def _on(self, tensor: Any, dtype: Any = None) -> Any:
        """``tensor`` on the run's device in ``dtype`` (default the model dtype): an exact cast
        for a value that is already a dtype value."""
        return tensor.to(self.device, self.dtype if dtype is None else dtype)

    def _back(self, tensor: Any) -> Any:
        return tensor.to("cpu", _torch().float64)

    def _param(self, module: Any, name: str = "weight") -> Any:
        tensor = getattr(module, name, None)
        return None if tensor is None else tensor.detach().to(self.device)

    def _affine(self, x: Any, bx: Any, module: Any) -> tuple[Any, Any]:
        if self.sums == "round-once":
            value, _ = RowModel._affine(self, x, None, module)
            return self._back(self._on(value)), None
        out = _torch().nn.functional.linear(self._on(x).reshape(1, 1, -1), self._param(module),
                                            self._param(module, "bias"))
        return self._back(out).reshape(-1), None

    def _add(self, x: Any, bx: Any, y: Any, by: Any) -> tuple[Any, Any]:
        return self._back(self._on(x) + self._on(y)), None

    def _rms(self, x: Any, bx: Any, module: Any, width: int) -> tuple[Any, Any]:
        torch = _torch()
        shape = x.shape
        rows = self._on(x).reshape(-1, width).to(torch.float32)
        variance = rows.pow(2).mean(-1, keepdim=True)
        rows = rows * torch.rsqrt(variance + float(module.variance_epsilon))
        return self._back(self._param(module) * rows.to(self.dtype)).reshape(shape), None

    def _rope(self, x: Any, bx: Any) -> tuple[Any, Any]:
        torch = _torch()
        refusal = _rope_refusal(self.config)
        if refusal is not None:
            raise Refusal(refusal)
        rotary = self._rotary()
        buffer = getattr(rotary, "inv_freq", None)
        inv = self._on(self._inv_freq(x.shape[-1]) if buffer is None else buffer.detach(),
                       torch.float32)
        freqs = inv * torch.tensor(float(self.row), dtype=torch.float32, device=inv.device)
        angles = torch.cat((freqs, freqs))
        scale = float(getattr(rotary, "attention_scaling", 1.0) or 1.0)
        cos = (angles.cos() * scale).to(self.dtype)
        sin = (angles.sin() * scale).to(self.dtype)
        q = self._on(x)
        half = q.shape[-1] // 2
        rotated = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
        return self._back(q * cos + rotated * sin), None

    def _scores(self, q: Any, bq: Any, k: Any, bk: Any, scaling: float) -> tuple[Any, Any]:
        torch = _torch()
        keys = self._kappa(k)                                     # (H, T, d_h)
        if self.sums == "round-once":
            raw = self._on(torch.einsum("hd,htd->ht", q, keys))
        else:
            raw = torch.matmul(self._on(q).unsqueeze(0).unsqueeze(2),       # (1, H, 1, d_h)
                               self._on(keys).unsqueeze(0).transpose(2, 3))  # (1, H, d_h, T)
        return self._back(raw * scaling).reshape(q.shape[0], -1), None

    def _softmax(self, s: Any, bs: Any) -> tuple[Any, Any]:
        torch = _torch()
        return self._back(torch.softmax(self._on(s), dim=-1, dtype=torch.float32)
                          .to(self.dtype)), None

    def _transport(self, p: Any, bp: Any, values: Any, bv: Any) -> tuple[Any, Any]:
        if self.sums == "round-once":
            value, _ = RowModel._transport(self, p, None, values, None)
            return self._back(self._on(value)), None
        out = _torch().matmul(self._on(p).unsqueeze(0).unsqueeze(2),         # (1, H, 1, T)
                              self._on(self._kappa(values)).unsqueeze(0))    # (1, H, T, d_h)
        return self._back(out).reshape(-1), None

    def _silu(self, g: Any, bg: Any) -> tuple[Any, Any]:
        return self._back(_torch().nn.functional.silu(self._on(g))), None

    def _product(self, x: Any, bx: Any, y: Any, by: Any) -> tuple[Any, Any]:
        return self._back(self._on(x) * self._on(y)), None


def _evaluate(model: RowModel, label: str) -> None:
    """Compute ``model``'s stop ``label`` and every stop it needs, in any row model, from one
    explicit stack of step generators (no recursion: a chain of rows has no depth limit)."""
    stack = [model._compute(label)]
    while stack:
        try:
            owner, wanted = stack[-1].send(None)
        except StopIteration:
            stack.pop()
            continue
        if wanted not in owner.values:
            stack.append(owner._compute(wanted))


def _split(label: str) -> tuple[int, str]:
    layer, port = label.split(".", 1)
    return int(layer[1:]), port


# --------------------------------------------------------------------------- ledger


@dataclass
class Forecast:
    """One sealed forecast and the in-process tensors its record points at."""

    id: int
    record: dict[str, Any]
    label: str
    margin: Margin | None
    key: tuple[int, int]
    predicted: Any
    bound: Any
    edits: frozenset
    status: str = "open"
    verdict: dict[str, Any] | None = field(default=None)
    chain: list = field(default_factory=list)   # the full identity chain (the record keeps a summary)
    emulated: Any = None   # the emulated prediction (EmulatedRowModel): what verify decides on


_LEDGERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def ledger(run: Any) -> dict[int, Forecast]:
    """The forecasts sealed on ``run`` in this process, by id."""
    if run not in _LEDGERS:
        _LEDGERS[run] = {}
    return _LEDGERS[run]


def _labelled_edits(run: Any) -> list[dict[str, Any]]:
    """Every edit present at sealing: committed ones and the staged ones of the pending layer."""
    out = [{"row": int(e["row"]), "layer": int(e["layer"]), "port": e["port"], "op": e["op"]}
           for e in run.committed_edits()]
    row = run.parse(run.where)[0]
    for edit in run.staged_info():
        out.append({"row": row, "layer": int(edit["layer"]), "port": edit["port"],
                    "op": edit["op"]})
    out += [{"row": int(e["row"]), "layer": int(e["cell"]["layer"]), "port": e["port"],
             "op": e["op"]} for e in run.committed_cells()]
    out += [{"row": row, "layer": int(e["cell"]["layer"]), "port": e["port"], "op": e["op"]}
            for e in run.staged_cells()]
    return out


def _staged_between(run: Any, row: int, low: tuple[int, int], high: tuple[int, int]) -> list[str]:
    """Staged edits at ports strictly after ``low`` and at or before ``high``."""
    found = []
    for edit in run.staged_info():
        port = run.program.spec(edit["port"]).name
        key = run.parse(f"t{row}.L{edit['layer']}.{port}")
        realised = run.parse(f"t{row}.L{edit['layer']}.o") if port == "r" else key
        if low < min(key, realised) <= high:
            found.append(f"t{row}.L{edit['layer']}.{port}")
    return found


def _resolve_target(run: Any, target: str | Margin | None) -> tuple[str, Margin | None,
                                                                 tuple[int, int]]:
    here = run.parse(run.where)
    row = here[0]
    if isinstance(target, Margin):
        if target.row != row:
            raise Refusal(f"forecasts stay in the cursor's row t{row}; {target.text} is not")
        vocab = int(run.adapter.model.config.vocab_size)
        if not (0 <= target.a < vocab and 0 <= target.b < vocab) or target.a == target.b:
            raise Refusal(f"M[a,b] needs two different token ids in 0..{vocab - 1}")
        return "logits", target, run.parse(f"t{row}.logits")
    if target is None:
        stops = run.stops()
        index = here[1] + 1
        if index >= len(stops):
            raise Refusal(f"the next stop t{row + 1}.start is the token list of the next row: "
                          + NO_IDENTITY["start"])
        key = (row, index)
    else:
        key = run.parse(target)
    if key[0] != row:
        raise Refusal(f"forecasts stay in the cursor's row t{row}; t{key[0]} is another row")
    return run.stops()[key[1]], None, key


def seal(run: Any, target: str | Margin | None = None, *, store: Any = None) -> dict[str, Any]:
    """Predict ``target`` (default: the next stop) from values at the cursor and seal it.

    The record keeps a bounded summary of the identity chain (``method.summary``); the full
    chain is stored as a JSON blob in ``store`` (a project ``LocalStore``) and the record points
    at it (``method.chain_pointer``, a tensor pointer to the uint8 bytes). Without a store the
    pointer is absent with its reason and the full chain stays on the ledger entry.

    Returns the forecast view (``id``, ``target``, ``method``, ``predicted``, ``bound``,
    ``record``); raises ``Refusal`` when the target is already computed (at or before the cursor,
    or past it by the native call: ``Run.status()["computed_past_cursor"]``), lies in another
    row, has no identity, or a staged edit sits between the cursor and the target.
    """
    here = run.parse(run.where)
    row = here[0]
    label, margin, key = _resolve_target(run, target)
    address = margin.text if margin is not None else f"t{row}.{label}"
    if label in NO_IDENTITY:
        raise Refusal(f"{address}: {NO_IDENTITY[label]}")
    if key <= here:
        raise Refusal(f"{address} is already computed at the cursor {run.where}: a forecast "
                      "must be sealed before the native value exists")
    status = run.status()
    if status["pending_layer"] is not None and label.startswith(f"L{status['pending_layer']}."):
        port = label.split(".", 1)[1]
        if port in status["computed_past_cursor"]:
            raise Refusal(f"{address} was already computed by the native call at {run.where} "
                          "(computed_past_cursor); seal before that call")
    staged = _staged_between(run, row, here, key)
    if staged:
        raise Refusal(f"staged edit(s) at {', '.join(staged)} rewrite an operand between the "
                      "cursor and the target; use whatif to predict an edit")

    known = _known(run, row, here)
    model = RowModel(run, row, known)
    value = model.get(label)
    bound = model.bound(label)
    if margin is not None:
        value = value[margin.a] - value[margin.b]
        bound = None if bound is None else bound[margin.a] + bound[margin.b]
    emulator = EmulatedRowModel(run, row, known)
    emulated = _target(emulator, label, margin)
    checkpoint = run.checkpoint()
    entries = ledger(run)
    number = max(entries, default=0) + 1
    chain = _chain(run, model, row)
    summary, chain_pointer = _store_chain(chain, store)
    body = _body(run, model, label, margin, address, key, here, value, bound, checkpoint,
                 status, summary)
    body["emulated"] = _emulated_view(emulated, label, margin, emulator,
                                      _rounding_origin(model, emulator, row, run.adapter.dtype))
    body["digests"] = {"exact": _digest(value), "emulated": _digest(emulated)}
    pointers = [] if chain_pointer is None else [chain_pointer]
    body["operands"], operands_summary = bound_list(
        body["operands"], MAX_ITEMS, store=store, run=run,
        keep=f"the first {MAX_ITEMS} in capture order (a row's ports, then its cache reads)")
    if operands_summary is not None:       # M40: the operand list grows with the layers too
        tensor = operands_summary["pointer"].pop("tensor", None)
        pointers += [] if tensor is None else [tensor]
        body["operands_summary"] = operands_summary
    if margin is None and label != "sample":
        from mdb.engine.ports import tensor_digest

        pointers.append(occurrence_pointer(
            f"t{row}.{label}", program="forecast", shape=list(value.shape), dtype="float64",
            sha256=tensor_digest(value.contiguous()), cut=checkpoint["cut"], authority=EXACT,
            ports=run.program.names))
    record = seal_record(session_id=run_session_id(run), kind="forecast",
                         model_identity=run.adapter.model_identity,
                         execution=dict(run.adapter.execution), authority=EXACT, body=body,
                         pointers=pointers)
    # the row model is dropped on return: the entry owns these tensors, no copy needed
    entry = Forecast(number, record, label, margin, key, value.detach(),
                     None if bound is None else bound.detach(),
                     frozenset(_edit_key(e) for e in body["edits_at_seal"]), chain=chain,
                     emulated=emulated.detach())
    entries[number] = entry
    return view(entry)


def _known(run: Any, row: int, here: tuple[int, int]) -> Callable[[str], Any]:
    """The operands of a forecast sealed at ``here``: the native stops of ``row`` at or before
    it (None for later stops and the token list)."""

    def known(name: str) -> Any:
        if name == "start" or run.parse(f"t{row}.{name}") > here:
            return None
        return run.read(f"t{row}.{name}")

    return known


def _target(model: RowModel, label: str, margin: Margin | None) -> Any:
    value = model.get(label)
    return value if margin is None else value[margin.a] - value[margin.b]


def _digest(value: Any) -> str:
    """sha256 of a prediction as float64 (vector, margin scalar or token id alike)."""
    from mdb.engine.ports import tensor_digest

    torch = _torch()
    return tensor_digest(torch.as_tensor(value, dtype=torch.float64).reshape(-1).contiguous())


def _emulated_view(value: Any, label: str, margin: Margin | None, model: EmulatedRowModel,
                   origin: dict[str, Any]) -> dict[str, Any]:
    """The second sealed prediction: the emulated chain (``derived-exact`` under its premise)."""
    out: dict[str, Any] = {"authority": EXACT, "premise": EMULATION_PREMISE,
                           "premises": list(EMULATION_PREMISES), "sums": model.sums,
                           "sha256": _digest(value)}
    if margin is not None:
        out["margin"] = num(float(value), EXACT, proof="P19")
    elif label == "sample":
        out["token"] = int(value)
    else:
        out.update(shape=list(value.shape), **_summary(value, EXACT))
    out["rounding_origin"] = origin
    return out


def _rounding_origin(model: RowModel, emulator: EmulatedRowModel, row: int,
                     dtype: Any) -> dict[str, Any]:
    """The first stop of the chain where the exact and the emulated values part by more than one
    spacing of the dtype at the stop's magnitude (scores and softmax: the head, and the score
    magnitude and spacing that rounding acts on). Sealed with the forecast: where the rounding
    effect ``native - exact`` is born, before the native value exists."""
    for name in emulator.computed:
        if name == "sample" or name not in model.values:
            continue
        exact, emulated = model.values[name], emulator.values[name]
        size = float(emulated.abs().max())
        step = _verdict.spacing(size, dtype)
        gap = (exact - emulated).abs()
        parts = float(gap.max()) / step
        if parts <= 1.0:
            continue
        out: dict[str, Any] = {"state": "found", "stop": f"t{row}.{name}",
                               "spacings": num(parts, EXACT), "magnitude": num(size, EXACT),
                               "spacing": num(step, EXACT)}
        port = name.rsplit(".", 1)[-1]
        if port in ("s", "p"):
            head = int(gap.reshape(gap.shape[0], -1).max(-1).values.argmax())
            scores = emulator.values.get(name[:-1] + "s", emulated) if port == "p" else emulated
            score = float(scores[head].abs().max())
            score_step = _verdict.spacing(score, dtype)
            out.update(head=head, score_magnitude=num(score, EXACT),
                       score_spacing=num(score_step, EXACT))
            out["text"] = (f"t{row}.{name}[h{head}]: score spacing {score_step:g} at |s| "
                           f"{score:.3g}; exact and emulated part by {parts:.3g} spacings")
        else:
            out["text"] = (f"t{row}.{name}: exact and emulated part by {parts:.3g} spacings at "
                           f"|x| {size:.3g}")
        return out
    return {"state": "none", "text": "exact and emulated agree within one spacing at every "
                                     "stop of the chain"}


def run_session_id(run: Any) -> str:
    """The record session id of an in-process run: its model identity and prompt digest."""
    checkpoint = run.checkpoint()
    return "mdb-run:" + seal_digest({"model": checkpoint["model_identity"],
                                     "prompt": checkpoint["prompt_sha256"]})[:32]


def _edit_key(edit: dict[str, Any]) -> tuple[int, int, str, str]:
    return int(edit["row"]), int(edit["layer"]), str(edit["port"]), str(edit["op"])


def _operator(program: Any, label: str) -> tuple[str, list[str]]:
    if label in GLOBAL_OPERATORS:
        return GLOBAL_OPERATORS[label], []
    spec = program.spec(_split(label)[1])
    return spec.operator, list(spec.proofs)


def _summary(value: Any, authority: str) -> dict[str, Any]:
    flat = value.reshape(-1)
    return {"max_abs": num(float(flat.abs().max()), authority),
            "norm": num(float(flat.norm()), authority)}


def _chain(run: Any, model: RowModel, row: int) -> list[dict[str, Any]]:
    """The identities the row model evaluated, in order (can be hundreds on a real model)."""
    chain = []
    for name in model.computed:
        operator, proofs = _operator(run.program, name)
        function, proof = IDENTITY[name.split(".", 1)[-1]]
        chain.append({"stop": f"t{row}.{name}", "operator": operator, "proofs": proofs,
                      "identity": function, "proof": proof})
    return chain


EDGE_STEPS = 3          # identities kept at each end of the chain in the record summary
MAX_LAYER_COUNTS = 64   # per-layer totals are listed only up to the record's sequence bound


def _store_chain(chain: list[dict[str, Any]], store: Any) -> tuple[dict[str, Any], Any]:
    """``(summary, tensor pointer or None)``: the bounded summary the record holds (count, proof
    use, first and last identities, per-layer totals) and, with a store, the pointer to the full
    chain blob (JSON bytes as a uint8 tensor)."""
    proofs: dict[str, int] = {}
    for step in chain:
        for proof in {step["proof"], *step["proofs"]} - {None}:
            proofs[proof] = proofs.get(proof, 0) + 1
    layers: dict[int, int] = {}
    for step in chain:
        port = step["stop"].split(".", 1)[1]
        if port in GLOBAL_OPERATORS:
            continue
        layer = _split(port)[0]
        layers[layer] = layers.get(layer, 0) + 1
    summary: dict[str, Any] = {
        "count": len(chain),
        "proof_counts": [{"proof": k, "count": v} for k, v in sorted(proofs.items())],
        "first": chain[:EDGE_STEPS], "last": chain[-EDGE_STEPS:],
        "global_steps": {"count": sum(1 for s in chain
                                      if s["stop"].split(".", 1)[1] in GLOBAL_OPERATORS)}}
    if layers and len(layers) <= MAX_LAYER_COUNTS:
        summary["layer_counts"] = [{"layer": i, "count": layers.get(i, 0)}
                                   for i in range(max(layers) + 1)]
    held = put_json(chain, store=store, absent_reason=(
        "no project store: the full chain is held in memory on the forecast entry "
        "(Forecast.chain), not stored"))
    summary["chain_sha256"] = held["sha256"]
    if held["state"] != "stored":
        summary["chain_pointer"] = {"state": "absent", "reason": held["reason"]}
        return summary, None
    summary["chain_pointer"] = {"state": "stored", "ref": held["ref"], "bytes": held["bytes"]}
    return summary, held["tensor"]


def load_chain(record: dict[str, Any], store: Any) -> list[dict[str, Any]]:
    """The full identity chain a forecast record points at, from ``store``; checked against the
    record's ``chain_sha256`` (``Unresolved`` or a refusal when absent or changed)."""
    summary = record["body"]["method"]["summary"]
    if summary["chain_pointer"]["state"] != "stored":
        raise Refusal("the record holds no chain pointer: " + summary["chain_pointer"]["reason"])
    pointer = next(p for p in record["pointers"] if p.get("kind") == "tensor"
                   and p["ref"] == summary["chain_pointer"]["ref"])
    try:
        return load_json({"state": "stored", "sha256": summary["chain_sha256"]}, store=store,
                         tensor=pointer)
    except ValueError:
        raise Refusal("the stored chain differs from the record's chain_sha256") from None


def _body(run, model, label, margin, address, key, here, value, bound, checkpoint,
          status, summary) -> dict[str, Any]:
    from mdb.engine.ports import tensor_digest

    row = here[0]
    operands = [{"address": f"t{row}.{name}", "sha256": tensor_digest(tensor.contiguous()),
                 "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}
                for name, tensor in model.captured.items()]
    operands += [{"address": f"{name}[row=0:{row}]" if row else name,
                  "sha256": tensor_digest(tensor.contiguous()), "shape": list(tensor.shape),
                  "dtype": str(tensor.dtype).removeprefix("torch.")}
                 for name, tensor in model.cache_reads.items()]
    if margin is not None:
        predicted: dict[str, Any] = {"value": num(float(value), EXACT, proof="P19"), "shape": []}
    elif label == "sample":
        predicted = {"token": int(value), "shape": []}
    else:
        predicted = {"shape": list(value.shape), "dtype": "float64",
                     "sha256": tensor_digest(value.contiguous()), **_summary(value, EXACT)}
    if bound is None:
        reason = model.unbounded.get(label, "no rounding bound for this step")
        bound_view: dict[str, Any] = {"state": bound_state(reason), "reason": reason}
    else:
        bound_view = {"max": num(float(bound.reshape(-1).max()), BOUND), "elementwise": True,
                      "premises": sorted(model.premises | {
                          "round to nearest, no overflow or underflow",
                          f"float32 library functions within {LIBRARY_ULPS} ulp",
                          f"model dtype {str(run.adapter.dtype).removeprefix('torch.')}"})}
    return {
        "schema": SCHEMA,
        "target": {"address": address, "stop": f"t{row}.{label}", "row": key[0],
                   "index": key[1],
                   "kind": "margin" if margin is not None else "token" if label == "sample"
                   else "value",
                   **({"indices": [margin.a, margin.b]} if margin is not None else {})},
        "sealed_at": {"where": run.where, "cut": checkpoint["cut"], "row": here[0],
                      "index": here[1], "transition": int(checkpoint["transitions"]),
                      "receipts": {"count": len(run.session.receipts),
                                   "last": (run.session.receipts[-1].fingerprint
                                            if run.session.receipts else None)},
                      "lookahead": status["lookahead"],
                      "computed_past_cursor": list(status["computed_past_cursor"])},
        "method": {"authority": EXACT, "summary": summary,
                   "rule": "identities of the port program on values that exist at the cursor; "
                           "no native call for the target"},
        "operands": operands,
        "predicted": predicted,
        "bound": bound_view,
        "edits_at_seal": _labelled_edits(run),
    }


def view(entry: Forecast) -> dict[str, Any]:
    """The bounded JSON of one forecast: its id, record digest and sealed body."""
    body = entry.record["body"]
    out = {"id": entry.id, "record": entry.record["id"], "target": body["target"]["address"],
           "status": entry.status, "sealed_at": body["sealed_at"]["where"],
           "method": entry.chain[:MAX_ITEMS], "method_summary": body["method"]["summary"],
           "predicted": body["predicted"],
           "emulated": body.get("emulated"),
           "bound": body["bound"]}
    if len(entry.chain) > MAX_ITEMS:       # M40: the view obeys the record bound too
        out["method_listed"] = {
            "count": len(entry.chain), "listed": MAX_ITEMS,
            "keep": f"the first {MAX_ITEMS} identities; all of them are in Forecast.chain and "
                    "(with a project) behind method_summary.chain_pointer (load_chain)"}
    if entry.verdict is not None:
        out["verdict"] = entry.verdict
    return out


# --------------------------------------------------------------------------- verify


def _lineage(run: Any, sealed: dict[str, Any]) -> dict[str, Any]:
    """Whether the sealed cut is an ancestor of the current cut: the sealed cut (its fingerprint
    at its transition count) is a mark of the run's trail, which holds the whole path of the
    current cut however deep the model is and is rebuilt identically by a restore (``whatif``).
    A cut on another path (an edit or a rewind before the seal) is not on the trail."""
    trail = run.cut_trail()
    current = trail[-1][0]
    wanted = (sealed["cut"], sealed.get("transition"))
    at = next((i for i in range(len(trail) - 1, -1, -1)
               if trail[i][0] == wanted[0] and (wanted[1] is None or trail[i][1] == wanted[1])),
              None)
    after = 0 if at is None else trail[-1][1] - trail[at][1]
    return {"sealed_cut": sealed["cut"], "current_cut": current,
            "receipts_after_seal": {"count": after}, "descends": at is not None,
            "same_cut": current == sealed["cut"]}


def _rounded_steps(entry: Forecast) -> int:
    """Rounded operations between the sealed operands and the target: the identities of the chain
    up to and including the target's (a margin: through the logits). Each rounds its result to the
    model dtype, so their errors accumulate; ``derived-exact``, a count of the chain."""
    stops = [step["stop"] for step in entry.chain]
    address = f"t{entry.key[0]}.{entry.label}"
    return stops.index(address) + 1 if address in stops else max(len(stops), 1)


def _native(run: Any, entry: Forecast) -> tuple[Any, dict[str, Any], Any]:
    """The native value as float64, its read info, and the raw native tensor (its dtype and
    magnitude set the verdict floor)."""
    row = entry.key[0]
    stop = f"t{row}.{entry.label}"
    value = run.read(stop)
    info = run.read_info(stop)
    if entry.margin is not None:
        flat = value.reshape(-1)
        return f64(flat[entry.margin.a]) - f64(flat[entry.margin.b]), info, value
    if entry.label == "sample":
        return f64(value).reshape(()), info, value
    return f64(value).reshape(entry.predicted.shape), info, value


def verify(run: Any, ids: list[int] | None = None) -> list[dict[str, Any]]:
    """Compare open forecasts (or the given ids) whose target the cursor has passed with the
    native value. Each result: ``verdict`` (``confirmed | not confirmed | unresolved | edited |
    open``), decided by the measured residual native - emulated (P20) against ``floor`` =
    ``verdict.FLOOR_SPACINGS`` (supplied) x the dtype spacing at the consumer's magnitude (a
    margin: the sum of its two logits' spacings), judged per element; above the floor, where
    emulation and native part (``_parting``); ``rounding_effect`` = native - exact
    (``measured``); the sealed ``rounding_origin``; ``bound_max`` / ``within_bound`` against the
    exact prediction only when the bound is finite (information, not the verdict); and
    ``order`` (sealing stop before the target; lineage of cuts)."""
    entries = ledger(run)
    here = run.parse(run.where)
    chosen = [entries[i] for i in ids] if ids else [e for e in entries.values()
                                                    if e.status == "open"]
    results = []
    for entry in chosen:
        body = entry.record["body"]
        sealed = body["sealed_at"]
        out: dict[str, Any] = {"id": entry.id, "target": body["target"]["address"],
                               "record": entry.record["id"], "sealed_at": sealed["where"]}
        if entry.status != "open":
            results.append({**out, **(entry.verdict or {}), "status": entry.status})
            continue
        if entry.key > here:
            results.append({**out, "status": "open", "verdict": "open",
                            "reason": f"target not reached (cursor {run.where})"})
            continue
        lineage = _lineage(run, sealed)
        sealed_key = (sealed["row"], sealed["index"])
        order = {"sealed": {"row": sealed_key[0], "index": sealed_key[1]},
                 "target": {"row": entry.key[0], "index": entry.key[1]},
                 "sealed_before_target": sealed_key < entry.key, **lineage}
        if not lineage["descends"]:
            results.append({**out, "status": "open", "verdict": "open", "order": order,
                            "reason": "the current cut does not descend from the sealed cut "
                                      "(rewound onto another path); the sealed target was not "
                                      "reached"})
            continue
        native, info, raw = _native(run, entry)
        later = [e for e in info["upstream_edits"] if _edit_key(e) not in entry.edits]
        verdict: dict[str, Any] = {"order": order}
        if info["edited"] or info["kind"] in ("edit", "realisation") or later:
            what = [f"t{e['row']}.L{e['layer']}.{e['port']} {e['op']}" for e in later]
            if info["edited"] or info["kind"] in ("edit", "realisation"):
                what.insert(0, f"{entry.label} written by {info['by']}")
            verdict.update(verdict="edited", edited_after_seal=what,
                           note="an edit reached the target after sealing; not a forecast "
                                "failure")
        else:
            label = info["authority"]  # measured, or derived-exact for the replica s
            if label is None:
                verdict.update(verdict="unresolved", state="unresolved", reason=info["reason"])
            else:
                verdict.update(_decide(run, entry, native, raw, label))
        entry.status = "closed"
        entry.verdict = verdict
        results.append({**out, "status": "closed", **verdict})
    return results


def _decide(run: Any, entry: Forecast, native: Any, raw: Any, label: str) -> dict[str, Any]:
    """The verdict of one reached, unedited forecast from its native value (``label`` the
    native value's authority)."""
    out: dict[str, Any] = {}
    gap = (native - entry.emulated).abs()
    out["residual_max_abs"] = num(float(gap.max()), label)
    out["native"] = (num(float(native), label) if native.dim() == 0
                     else _summary(native, label))
    # the consumer's magnitude: the margin's two logits (both spacings), else the largest element
    size = _verdict.magnitude(entry.margin if entry.margin is not None else object(), raw)
    partner = None
    if entry.margin is not None:
        flat = raw.reshape(-1).detach().abs().float()
        size, partner = float(flat[entry.margin.a]), float(flat[entry.margin.b])
    decided = _verdict.confirm(gap, size, raw.dtype, partner=partner)
    out.update({k: decided[k] for k in (
        "verdict", "state", "floor", "floor_spacings", "spacing", "residual_spacings", "dtype",
        "criterion", "index") if k in decided}, basis="measured residual native - emulated, P20")
    step = decided["spacing"]["value"]
    effect = float((native - entry.predicted).abs().max())
    out["rounding_effect"] = {"max_abs": num(effect, label),
                              "basis": "native - exact (the float64 chain), measured"}
    if step:
        out["rounding_effect"]["spacings"] = num(effect / step, label)
    out["rounding_origin"] = entry.record["body"]["emulated"]["rounding_origin"]
    out["rounded_steps"] = num(_rounded_steps(entry), EXACT)
    if decided["verdict"] != _verdict.CONFIRMED:
        out.update(_parting(run, entry, label))
    bound = entry.bound
    if bound is not None and bool(bound.isfinite().all()):
        # information only: the worst-case propagated bound of the exact prediction
        exact_gap = (native - entry.predicted).abs()
        within = bool((exact_gap <= bound).all())
        out.update(within_bound=within, bound_max=num(float(bound.max()), BOUND))
        if not within:
            out["worst_excess"] = num(float((exact_gap - bound).max()), label)
    return out


MAX_TIES = 3


def _parting(run: Any, entry: Forecast, label: str) -> dict[str, Any]:
    """Where emulation and native part, for a residual above the floor.

    ``not confirmed`` when the prediction is not the emulated chain of the sealed operands (the
    held prediction or an operand read now differs from its sealed digest) or when a stage of
    the chain, from the first stop where emulated and native part by more than the floor to the
    target, does not reproduce its native value from its native operands within the rounding
    bound of its identity (``bound``, premises as for a forecast): the identity does not
    describe that native step. Otherwise ``unresolved``: every such stage is the native step,
    and the residual is rounding drift between stages (one-spacing differences of the sums
    carried forward); native top-score ties where the emulated softmax parts are named."""
    body = entry.record["body"]
    sealed_digest = body["digests"]["emulated"]
    row = entry.key[0]
    if _digest(entry.emulated) != sealed_digest:
        return {"verdict": _verdict.NOT_CONFIRMED, "state": "resolved",
                "reason": "the held prediction differs from its sealed digest"}
    sealed = (body["sealed_at"]["row"], body["sealed_at"]["index"])
    emulator = EmulatedRowModel(run, row, _known(run, row, sealed))
    if _digest(_target(emulator, entry.label, entry.margin)) != sealed_digest:
        return {"verdict": _verdict.NOT_CONFIRMED, "state": "resolved",
                "reason": "the emulated chain of the operands read now differs from the sealed "
                          "prediction: an operand is not the one sealed"}
    dtype = run.adapter.dtype
    floor = _verdict.FLOOR_SPACINGS

    def read(name: str) -> Any:
        return f64(run.read(f"t{row}.{name}")).reshape(emulator.shape(name))

    chain = [name for name in emulator.computed if name != "sample"]
    first, parts, step = None, 0.0, 0.0
    for name in chain:
        native = read(name)
        step = _verdict.spacing(float(native.abs().max()), dtype)
        parts = float((emulator.values[name] - native).abs().max()) / step
        if parts > floor:
            first = name
            break
    if first is None:       # every stop within the floor; the target (margin, token) is not
        first = chain[-1] if chain else entry.label
    unchecked, ties = [], []
    stages = (emulator.computed[emulator.computed.index(first):] if first in emulator.computed
              else [entry.label])
    for name in stages:
        stage = RowModel(run, row, lambda n, at=name: None if n == at else run.read(
            f"t{row}.{n}"))
        value, bound = stage.get(name), stage.bound(name)
        native = (f64(run.read(f"t{row}.sample")).reshape(()) if name == "sample"
                  else read(name))
        if bound is None:
            unchecked.append(f"t{row}.{name} ({stage.unbounded.get(name, 'no bound')})")
            continue
        excess = float(((native - value).abs() - bound).max())
        if excess > 0:
            return {"verdict": _verdict.NOT_CONFIRMED, "state": "resolved",
                    "parting": {"stop": f"t{row}.{first}", "spacings": num(parts, label),
                                "stage": f"t{row}.{name}", "excess": num(excess, label)},
                    "reason": f"t{row}.{name}: the native value lies {excess:.3g} outside the "
                              "rounding bound of its identity on the native operands"}
        if name.endswith(".p"):
            scores, emulated = read(name[:-1] + "s"), emulator.values[name]
            top = scores.topk(min(2, scores.shape[-1]), dim=-1).values
            apart = (emulated != native).any(-1)
            for head in range(scores.shape[0]):
                if top.shape[-1] == 2 and top[head, 0] == top[head, 1] and bool(apart[head]):
                    size = float(top[head, 0].abs())
                    ties.append(f"t{row}.{name}[h{head}]: native top scores tie at "
                                f"{float(top[head, 0]):g} (score spacing "
                                f"{_verdict.spacing(size, dtype):g})")
    reason = (f"rounding drift: emulated and native part first at t{row}.{first} by "
              f"{parts:.3g} spacings; every stage from there to the target reproduces its native "
              "value from its native operands within its rounding bound")
    if ties:
        reason += "; at native ties: " + "; ".join(ties[:MAX_TIES]) + (
            f" (+{len(ties) - MAX_TIES} more)" if len(ties) > MAX_TIES else "")
    if unchecked:
        reason += "; not checked (no bound): " + ", ".join(unchecked[:MAX_TIES]) + (
            f" (+{len(unchecked) - MAX_TIES} more)" if len(unchecked) > MAX_TIES else "")
    return {"verdict": _verdict.UNRESOLVED, "state": "unresolved", "reason": reason,
            "parting": {"stop": f"t{row}.{first}", "spacings": num(parts, label),
                        "spacing": num(step, label), "ties": ties[:MAX_ITEMS],
                        "stages_checked": {"count": len(stages) - len(unchecked)}}}


_session_id = run_session_id  # old private name, kept for one commit

__all__ = ["FORECAST", "IDENTITY", "Logit", "Margin", "NO_IDENTITY", "Refusal", "RowModel",
           "ledger", "run_session_id", "seal", "verify", "view"]

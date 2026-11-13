"""Exact identities of the model-math proof series, evaluated on captured operands.

Every function takes plain tensors (port captures, weight tensors) as arguments and never reads a
model. Operands are upcast to float64 on CPU so that "exact in real arithmetic" is testable to
roundoff. Each function returns an ``IdentityResult``:

* ``terms``      the decomposition terms in the chapter's notation;
* ``total``      their sum (or the forward value for a forward identity);
* ``reference``  an independent float64 evaluation of the same quantity (e.g. ``F(x1) - F(x0)``);
* ``closure``    ``max|total - reference|`` (real-arithmetic closure; roundoff only);
  ``closure_rel`` divides by the largest of reference, total and any term, so cancellation
  between large terms is measured against the quantities actually summed;
* ``native``     the native value as captured (a difference of captures for finite identities);
* ``eta``        ``native - reference``: the measured arithmetic remainder of chapters 01 and 09.

``eta`` is a measurement, never a bound. Sources: obsidian/model-math chapters 01-05, 07, 09 and
the proof index. Torch is imported inside functions only, so ``IDENTITIES`` is listable without it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

EXACT = "derived-exact"
LOCAL = "derived-local"
BOUND = "bound"
MEASURED = "measured"
SUPPLIED = "supplied"
STATES = ("resolved", "unresolved", "refused")


def _torch():
    import torch

    return torch


def f64(x: Any) -> Any:
    """Float64 CPU copy of a tensor or array-like (exact for bf16/fp16/fp32 inputs).

    Model weights (and slices of them) are detached; float64 tensors keep autograd tracking so a
    caller's float64 replica can be differentiated through these functions.
    """
    torch = _torch()
    if isinstance(x, torch.Tensor) and (isinstance(x, torch.nn.Parameter)
                                        or (x.requires_grad and x.dtype != torch.float64)):
        x = x.detach()
    if isinstance(x, torch.Tensor):
        return x.to(device="cpu", dtype=torch.float64)
    return torch.as_tensor(x, dtype=torch.float64)


def f64_at(x: Any, index: Any) -> Any:
    """``f64(x)[index]`` without casting the rest of ``x`` (index, then cast: the same values).
    A Parameter is detached first, as ``f64`` detaches it."""
    torch = _torch()
    if isinstance(x, torch.nn.Parameter):
        x = x.detach()
    return f64(x[index])


def _maxabs(x: Any) -> float:
    if x is None:
        return float("nan")
    torch = _torch()
    x = torch.as_tensor(x)
    return float(x.abs().max()) if x.numel() else 0.0


AUTHORITIES = ("measured", "derived-exact", "derived-local", "bound", "supplied")
_RECORD_NUM: list[Any] = []


def num(value: float, authority: str, proof: str | None = None) -> dict[str, Any]:
    """A labelled number ``{value, authority, proof}``.

    Uses ``mdb.core.records.num`` when that module imports cleanly (it does in this repository);
    the local fallback below validates identically (booleans, non-numbers and non-finite floats
    refused, integers kept as integers, numpy scalars unwrapped, authority checked against the
    same vocabulary) and is used only if that import fails.
    """
    if not isinstance(value, (bool, int, float)) and hasattr(value, "item"):
        value = value.item()  # 0-d tensors and numpy scalars
    if not _RECORD_NUM:
        try:
            from mdb.core.records import num as record_num
        except Exception:  # noqa: BLE001 - any import-time failure means "not available"
            record_num = None
        _RECORD_NUM.append(record_num)
    if _RECORD_NUM[0] is not None:
        return _RECORD_NUM[0](value, authority, proof=proof)
    import numbers

    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or (not isinstance(value, numbers.Integral) and not math.isfinite(value))):
        raise ValueError("a numeric claim needs a finite number")
    if authority not in AUTHORITIES:
        raise ValueError(f"authority must be one of {', '.join(AUTHORITIES)}")
    out: dict[str, Any] = {
        "value": int(value) if isinstance(value, numbers.Integral) else float(value),
        "authority": authority}
    if proof is not None:
        if not isinstance(proof, str) or not proof or proof != proof.strip():
            raise ValueError("proof must be short text")
        out["proof"] = proof
    return out


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def premise(holds: bool, value: float, tolerance: float | None, proof: str, *,
            authority: str = MEASURED) -> dict[str, Any]:
    """One checked premise: ``{holds, value: num, tolerance: num | None}``.

    ``value`` is the checked quantity (``measured`` unless stated); ``tolerance`` is the limit the
    check used, a constant chosen here (``supplied``), or ``None`` when no limit applies.
    """
    return {"holds": bool(holds),
            "value": num(float(value), authority, proof) if _is_finite(value) else None,
            "tolerance": None if tolerance is None else num(float(tolerance), SUPPLIED, proof)}


@dataclass
class IdentityResult:
    """One identity evaluated on captured operands (see module docstring for the fields).

    ``authority`` is the authority of the result's own numbers (``derived-exact`` for an
    identity in real arithmetic, ``derived-local`` for a first-order or quadrature result,
    ``bound`` for a certificate). ``state`` is ``resolved``, ``unresolved`` (a quantity is not
    finite; see ``summary()``) or ``refused`` (a premise of a T-kind identity failed beyond its
    tolerance; ``reason`` is ``premise <name> violated``; ``terms`` stay available for
    inspection but carry no claim). ``premises`` maps a premise name to
    ``{holds, value, tolerance}`` (labelled numbers). ``approximation`` is the structured
    quadrature rule (``{"rule": "gauss-legendre", "k": int}``) of a ``derived-local`` result.
    ``eta_authority`` labels the ``eta`` numbers of ``summary()``: ``measured`` against a native
    tensor, ``derived-exact`` when the compared value is a replica computed from captured operands.
    """

    proof: str
    name: str
    kind: str
    authority: str
    terms: dict[str, Any]
    total: Any
    reference: Any = None
    native: Any = None
    native_dtype: str | None = None
    native_scale: float | None = None
    assumptions: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    premises: dict[str, dict[str, Any]] = field(default_factory=dict)
    state: str = "resolved"
    reason: str | None = None
    approximation: dict[str, Any] | None = None
    eta_authority: str = MEASURED  # ``derived-exact`` when ``native`` is itself a replica

    @property
    def closure(self) -> float | None:
        if self.reference is None:
            return None
        return _maxabs(f64(self.total) - f64(self.reference))

    @property
    def closure_rel(self) -> float | None:
        if self.reference is None:
            return None
        scale = max([_maxabs(f64(self.reference)), _maxabs(f64(self.total)), 1e-300]
                    + [_maxabs(f64(v)) for v in self.terms.values()])
        return self.closure / scale

    @property
    def eta(self) -> Any:
        if self.native is None:
            return None
        target = self.reference if self.reference is not None else self.total
        return f64(self.native) - f64(target)

    @property
    def eta_max_abs(self) -> float | None:
        return None if self.native is None else _maxabs(self.eta)

    @property
    def eta_rel(self) -> float | None:
        """``max|eta|`` over the native magnitude (endpoint magnitude for a difference)."""
        if self.native is None:
            return None
        scale = self.native_scale if self.native_scale is not None else _maxabs(f64(self.native))
        return self.eta_max_abs / max(scale, 1e-300)

    @property
    def eta_rel_to_change(self) -> float | None:
        """``max|eta|`` over ``max|native|`` (the native change, for a finite identity)."""
        if self.native is None:
            return None
        return self.eta_max_abs / max(_maxabs(f64(self.native)), 1e-300)

    def refuse_on(self, name: str, holds: bool, value: float, tolerance: float | None) -> None:
        """Record a premise of a T-kind identity; the first violated one refuses the result."""
        self.premises[name] = premise(holds, value, tolerance, self.proof)
        if not holds and self.state == "resolved":
            self.state, self.reason = "refused", f"premise {name} violated"

    def summary(self) -> dict[str, Any]:
        """Bounded view; every number is a labelled number (tensors summarised by max-abs).

        ``state``/``reason`` sit at the top level. Only a ``resolved`` result makes claims: under
        ``refused`` or ``unresolved`` the top-level ``authority`` is ``None`` and every term, the
        total, the closures and ``eta`` entries are inspection-only records
        ``{value, authority: None, inspection_only: True, proof}`` (the numbers are visible, none
        is labelled). Premises keep their own labels (``measured`` checks, ``supplied``
        tolerances). A quantity that is not finite is left out and makes the state ``unresolved``
        with a reason naming it (nothing raises).
        """
        own = self.authority
        unresolved: list[str] = []
        raw: list[tuple[dict[str, Any], str, float, str]] = []  # (target, key, value, authority)

        def put(target: dict[str, Any], key: str, name: str, fn: Callable[[], Any],
                authority: str) -> None:
            value = fn()
            if value is None:
                return
            if not _is_finite(value):
                unresolved.append(name)
                return
            raw.append((target, key, float(value), authority))

        closure_authority = own
        if self.approximation is not None and own == EXACT:
            closure_authority = LOCAL  # exact total, but the reference is a quadrature
        terms: dict[str, Any] = {}
        for key, value in self.terms.items():
            put(terms, key, f"terms_max_abs.{key}", lambda v=value: _maxabs(f64(v)), own)
        out: dict[str, Any] = {
            "proof": self.proof,
            "name": self.name,
            "kind": self.kind,
            "authority": own,
            "assumptions": list(self.assumptions),
            "premises": {k: dict(v) for k, v in self.premises.items()},
            "terms_max_abs": terms,
        }
        put(out, "total_max_abs", "total_max_abs", lambda: _maxabs(f64(self.total)), own)
        if self.reference is not None:
            put(out, "closure_max_abs", "closure_max_abs", lambda: self.closure,
                closure_authority)
            put(out, "closure_rel", "closure_rel", lambda: self.closure_rel, closure_authority)
        if self.native is not None:
            put(out, "eta_max_abs", "eta_max_abs", lambda: self.eta_max_abs, self.eta_authority)
            put(out, "eta_rel", "eta_rel", lambda: self.eta_rel, self.eta_authority)
            put(out, "eta_rel_to_change", "eta_rel_to_change", lambda: self.eta_rel_to_change,
                self.eta_authority)
            out["native_dtype"] = self.native_dtype
        if self.approximation is not None:
            out["approximation"] = dict(self.approximation)
        if self.notes:
            out["notes"] = list(self.notes)
        if "direction" in self.extras:
            out["direction"] = self.extras["direction"]
        state, reason = self.state, self.reason
        if unresolved and state == "resolved":
            state = "unresolved"
            reason = "not finite: " + ", ".join(unresolved)
        claims = state == "resolved"
        for target, key, value, authority in raw:
            target[key] = (num(value, authority, self.proof) if claims else
                           {"value": value, "authority": None, "inspection_only": True,
                            "proof": self.proof})
        if not claims:
            out["authority"] = None
        out["state"], out["reason"] = state, reason
        return out


def _native(native: Any = None, native0: Any = None, native1: Any = None):
    """(value, dtype, scale) for a native capture or the float64 difference of two captures.

    ``scale`` is the largest native magnitude involved (both endpoints for a difference): the
    dtype's rounding acts at that scale, so ``eta_rel`` is measured against it.
    """
    if native is not None:
        value = f64(native)
        return value, str(native.dtype) if hasattr(native, "dtype") else None, _maxabs(value)
    if native0 is not None and native1 is not None:
        a, b = f64(native0), f64(native1)
        return b - a, str(getattr(native1, "dtype", None)), max(_maxabs(a), _maxabs(b))
    return None, None, None


# --------------------------------------------------------------------------- elementwise maps


def silu(g: Any) -> Any:
    """S(g) = g * sigma(g) (chapter 05)."""
    torch = _torch()
    return g * torch.sigmoid(g)


def silu_d1(g: Any) -> Any:
    """S'(g) = sigma + g sigma (1 - sigma) (chapters 05 and 09 section 7)."""
    torch = _torch()
    s = torch.sigmoid(g)
    return s + g * s * (1 - s)


def silu_d2(g: Any) -> Any:
    """S''(g) = sigma (1 - sigma) [2 + g (1 - 2 sigma)] (chapters 05 and 09 section 7)."""
    torch = _torch()
    s = torch.sigmoid(g)
    return s * (1 - s) * (2 + g * (1 - 2 * s))


def _rho(x: Any, eps: float) -> Any:
    torch = _torch()
    return 1.0 / torch.sqrt(eps + (x * x).mean(-1, keepdim=True))


# --------------------------------------------------------------------------- P01 forward DAG


def embedding_lookup(embedding: Any, token: int, *, native: Any = None) -> IdentityResult:
    """P01 token lookup. Chapter 01 "Token lookup" (A).

    ``h_{0,t} = Emb e_{w_t}``: the selected embedding row. Port: ``L0.h``. ``embedding`` is the
    ``(V, d)`` weight of the token embedding module.
    """
    total = f64_at(embedding, int(token))     # the one row, not the vocabulary
    value, dtype, scale = _native(native)
    return IdentityResult("P01", "embedding_lookup", "A", EXACT, {}, total, native=value,
                          native_dtype=dtype, native_scale=scale)


def residual_add(x: Any, y: Any, *, native: Any = None) -> IdentityResult:
    """P01 residual skip. Chapters 04 ("Transport and write") and 05: ``r = h + o``,
    ``h_{l+1} = r + m``. Ports: ``r`` from ``h``,``o``; ``h_out`` from ``r``,``m`` (A).

    ``eta`` is the native rounding of the add.
    """
    total = f64(x) + f64(y)
    value, dtype, scale = _native(native)
    return IdentityResult("P01", "residual_add", "A", EXACT, {"x": f64(x), "y": f64(y)}, total,
                          native=value, native_dtype=dtype, native_scale=scale)


def head_write(head_concat: Any, out_weight: Any, out_bias: Any = None, *,
               native: Any = None) -> IdentityResult:
    """P01 head return. Chapter 04 "Transport and write": ``o_t = sum_h O_h A_{h,t}`` (A).

    Ports: ``o`` from ``A`` (flat head concat) and ``o_proj.weight`` (``d x H d_h``).
    """
    total = f64(head_concat) @ f64(out_weight).T
    if out_bias is not None:
        total = total + f64(out_bias)
    value, dtype, scale = _native(native)
    return IdentityResult("P01", "head_write", "A", EXACT, {}, total, native=value,
                          native_dtype=dtype, native_scale=scale)


def transport(probs: Any, values: Any, *, native: Any = None) -> IdentityResult:
    """P01 value transport. Chapter 04: ``A_{h,t} = sum_j p_{h,tj} v_{kappa(h),j}`` (A).

    ``probs (..., J)``, ``values (..., J, e)`` with heads already mapped through kappa.
    Ports: ``A`` from ``p`` and the value cache rows.
    """
    torch = _torch()
    total = torch.einsum("...j,...je->...e", f64(probs), f64(values))
    value, dtype, scale = _native(native)
    return IdentityResult("P01", "transport", "A", EXACT, {}, total, native=value,
                          native_dtype=dtype, native_scale=scale)


# --------------------------------------------------------------------------- P02 / P03 RMS


def rms_norm(x: Any, gamma: Any, eps: float, *, components: Mapping[str, Any] | None = None,
             native: Any = None) -> IdentityResult:
    """P02 RMS normalization. Chapter 02 "Scale and gain" (A).

    ``N_gamma(x) = rho(x) Gamma x``, ``rho(x) = (eps + d^-1 sum_i x_i^2)^(-1/2)``. Ports:
    ``a = N(h)``, ``n = N(r)``, final ``x^F = N(h_L)``; Qwen3 ``qn``/``kn`` per head with width
    ``d_h`` (chapter 03). With ``components`` (``x = sum_a xi_a``) the terms are
    ``rho(x) Gamma xi_a`` at the observed shared scale; ``reference`` is ``rho Gamma x`` and the
    unlisted part ``x - sum xi`` is a separate term. The chapter warns this is not
    ``sum_a N(xi_a)``.
    """
    x, g = f64(x), f64(gamma)
    rho = _rho(x, eps)
    value, dtype, scale = _native(native)
    if components is None:
        return IdentityResult("P02", "rms_norm", "A", EXACT, {}, rho * g * x, native=value,
                              native_dtype=dtype, native_scale=scale, extras={"rho": rho})
    terms = {name: rho * g * f64(xi) for name, xi in components.items()}
    unlisted = x - sum(f64(xi) for xi in components.values())
    terms["unlisted"] = rho * g * unlisted
    total = sum(terms.values())
    return IdentityResult("P02", "rms_norm_components", "A", EXACT, terms, total,
                          reference=rho * g * x, native=value, native_dtype=dtype,
                          native_scale=scale,
                          extras={"rho": rho},
                          assumptions=("shared scale rho(x) evaluated at the full state x",))


def rms_jacobian(x: Any, gamma: Any, eps: float) -> IdentityResult:
    """P02 local coupling and radial response. Chapter 02 "Local coupling" (D, radial A).

    ``J_N(x) = Gamma (rho I - rho^3/d x x^T)``; ``J_N(x) x = eps rho^3 Gamma x``. ``total`` is the
    radial response ``J_N x`` as its two cancelling terms ``rho Gamma x`` and
    ``-(rho^3/d) Gamma x (x^T x)``; ``reference`` is the closed form ``eps rho^3 Gamma x`` so
    closure checks the radial identity (the terms cancel to O(eps)). ``extras['jacobian']`` is
    ``J_N`` (``(..., d, d)``), a first-order (``derived-local``) object.
    """
    torch = _torch()
    x, g = f64(x), f64(gamma)
    d = x.shape[-1]
    rho = _rho(x, eps)
    eye = torch.eye(d, dtype=torch.float64)
    outer = x.unsqueeze(-1) * x.unsqueeze(-2)
    jac = g.unsqueeze(-1) * (rho.unsqueeze(-1) * eye - (rho**3).unsqueeze(-1) / d * outer)
    radial = torch.einsum("...ij,...j->...i", jac, x)
    terms = {"scaled_identity": rho * g * x,
             "radial_coupling": -(rho**3) / d * g * x * (x * x).sum(-1, keepdim=True)}
    closed = eps * rho**3 * g * x
    return IdentityResult("P02", "rms_jacobian", "D", LOCAL, terms, sum(terms.values()),
                          reference=closed, extras={"jacobian": jac, "rho": rho,
                                                    "jacobian_times_x": radial})


def rms_finite_change(x0: Any, x1: Any, gamma: Any, eps: float, *, native0: Any = None,
                      native1: Any = None) -> IdentityResult:
    """P03 finite RMS change. Chapter 02 "Finite change"; chapter 09 section 2 (F).

    ``Delta N = Gamma [rho_0 Delta x + (rho_1 - rho_0) x_0 + (rho_1 - rho_0) Delta x]`` with the
    rationalized ``rho_1 - rho_0 = -(c_1 - c_0) / (sqrt(c_0 c_1)(sqrt c_0 + sqrt c_1))`` and
    ``c_1 - c_0 = (2 x_0^T Delta x + ||Delta x||^2)/d`` (baseline alignment + edit energy).
    Ports: ``h -> a`` or ``r -> n`` at two endpoints. Assumes the same gain and positive
    denominators. ``extras['two_term_total']`` is chapter 09's equivalent
    ``Gamma [rho_1 Delta x + (rho_1 - rho_0) x_0]``.
    """
    torch = _torch()
    x0, x1, g = f64(x0), f64(x1), f64(gamma)
    d = x0.shape[-1]
    dx = x1 - x0
    c0 = eps + (x0 * x0).sum(-1, keepdim=True) / d
    c1 = eps + (x1 * x1).sum(-1, keepdim=True) / d
    alignment = 2 * (x0 * dx).sum(-1, keepdim=True) / d
    energy = (dx * dx).sum(-1, keepdim=True) / d
    dc = alignment + energy
    drho = -dc / (torch.sqrt(c0) * torch.sqrt(c1) * (torch.sqrt(c0) + torch.sqrt(c1)))
    rho0, rho1 = c0.rsqrt(), c1.rsqrt()
    terms = {
        "rho0_dx": g * rho0 * dx,
        "drho_x0": g * drho * x0,
        "drho_dx": g * drho * dx,
    }
    reference = g * _rho(x1, eps) * x1 - g * _rho(x0, eps) * x0
    value, dtype, scale = _native(native0=native0, native1=native1)
    return IdentityResult(
        "P03", "rms_finite_change", "F", EXACT, terms, sum(terms.values()), reference=reference,
        native=value, native_dtype=dtype, native_scale=scale,
        assumptions=("same fixed gain gamma at both endpoints", "positive denominators"),
        extras={
            "dc_alignment": alignment, "dc_energy": energy, "rho0": rho0, "rho1": rho1,
            "drho_rationalized_minus_direct": _maxabs(drho - (rho1 - rho0)),
            "two_term_total": g * (rho1 * dx + drho * x0),
        },
    )


# --------------------------------------------------------------------------- P04 Q,K,V, RoPE pairs


def affine(a: Any, weight: Any, bias: Any = None, *, native: Any = None,
           proof: str = "P04") -> IdentityResult:
    """P04 affine read. Chapter 03 "Affine reads" (A).

    ``q~ = Q a + b^Q``, ``k~ = K a + b^K``, ``v = V a + b^V`` (rows of ``weight`` are output
    coordinates). Ports: ``q``/``k``/``v`` from ``a``. The same map with ``proof="P10"`` gives
    ``g = G n``, ``u = U n`` and ``m = D^M f`` (chapter 05).
    """
    total = f64(a) @ f64(weight).T
    if bias is not None:
        total = total + f64(bias)
    value, dtype, scale = _native(native)
    return IdentityResult(proof, "affine", "A", EXACT, {}, total, native=value,
                          native_dtype=dtype, native_scale=scale)


def gqa_kappa(num_heads: int, num_kv_heads: int) -> list[int]:
    """P04 contiguous GQA grouping ``kappa(h) = floor(h / (H / H_KV))`` (chapter 03)."""
    if num_heads % num_kv_heads:
        raise ValueError("H must be a multiple of H_KV for contiguous grouping")
    size = num_heads // num_kv_heads
    return [h // size for h in range(num_heads)]


def head_route_average_drop(probs: Any, values: Any, out_weight: Any,
                            kappa: Sequence[int]) -> IdentityResult:
    """P04 head-route averaging. Chapter 03 "Affine reads" (A).

    Replacing every head's route by ``p_bar = H^-1 sum_h p_h`` drops the residual write
    ``sum_h O_h sum_j (p_{h,tj} - p_bar_tj) v_{kappa(h),j}``. ``probs (H, J)``,
    ``values (H_KV, J, d_h)``, ``out_weight (d, H d_h)``. ``reference`` is ``o(p) - o(p_bar)``
    computed by two direct head writes.
    """
    torch = _torch()
    p, v, o = f64(probs), f64(values), f64(out_weight)
    heads, d_h = p.shape[0], v.shape[-1]
    vk = v[list(kappa)]
    pbar = p.mean(0, keepdim=True).expand_as(p)
    blocks = o.reshape(o.shape[0], heads, d_h)
    terms = {
        f"head{h}": blocks[:, h] @ ((p[h] - pbar[h]) @ vk[h]) for h in range(heads)
    }

    def write(route):
        return torch.einsum("ihe,he->i", blocks, torch.einsum("hj,hje->he", route, vk))

    return IdentityResult("P04", "head_route_average_drop", "A", EXACT, terms,
                          sum(terms.values()), reference=write(p) - write(pbar))


def rope_inv_freq(theta_base: float, head_dim: int) -> Any:
    """``theta_a = Theta^(-2a/d_h)``, ``a = 0..d_h/2-1`` (chapter 03 "Position pairs")."""
    torch = _torch()
    a = torch.arange(head_dim // 2, dtype=torch.float64)
    return float(theta_base) ** (-2.0 * a / head_dim)


def rope_matrix(position: float, theta_base: float, head_dim: int) -> Any:
    """``R_t`` of the half-split pairing ``(a, a + d_h/2)`` as a ``(d_h, d_h)`` matrix."""
    torch = _torch()
    half = head_dim // 2
    angle = position * rope_inv_freq(theta_base, head_dim)
    cos, sin = torch.cos(angle), torch.sin(angle)
    r = torch.zeros(head_dim, head_dim, dtype=torch.float64)
    idx = torch.arange(half)
    r[idx, idx] = cos
    r[idx, idx + half] = -sin
    r[idx + half, idx] = sin
    r[idx + half, idx + half] = cos
    return r


def rope_rotate(x: Any, position: int, theta_base: float, *, native: Any = None) -> IdentityResult:
    """P04 rotary position (positional pair expansion). Chapter 03 "Position pairs" (A).

    ``(R_t x)_a = x_a cos(t theta_a) - x_{a+d_h/2} sin(t theta_a)``,
    ``(R_t x)_{a+d_h/2} = x_a sin(t theta_a) + x_{a+d_h/2} cos(t theta_a)``. Ports: ``qr`` from
    ``q`` (Qwen2) or ``qn`` (Qwen3) at the receiver position; the pending row of ``kr`` from
    ``k``/``kn``. Assumes default unscaled RoPE with full rotary width.
    """
    torch = _torch()
    x = f64(x)
    half = x.shape[-1] // 2
    angle = position * rope_inv_freq(theta_base, x.shape[-1])
    cos, sin = torch.cos(angle), torch.sin(angle)
    first, second = x[..., :half], x[..., half:]
    total = torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
    value, dtype, scale = _native(native)
    return IdentityResult("P04", "rope_rotate", "A", EXACT, {}, total, native=value,
                          native_dtype=dtype, native_scale=scale,
                          assumptions=("default RoPE, full rotary width, half-split pairs",))


def rope_relative(t: int, j: int, theta_base: float, head_dim: int) -> IdentityResult:
    """P04 relative rotation. Chapter 03 "Position pairs" (A): ``R_t^T R_j = R_{j-t}``."""
    total = rope_matrix(t, theta_base, head_dim).T @ rope_matrix(j, theta_base, head_dim)
    return IdentityResult("P04", "rope_relative", "A", EXACT, {}, total,
                          reference=rope_matrix(j - t, theta_base, head_dim))


def rope_score_pairs(qbar: Any, kbar: Any, t: int, positions: Sequence[int], theta_base: float,
                     *, scaling: float = 1.0, native: Any = None) -> IdentityResult:
    """P04 score in cos/sin form. Chapter 03 "Position pairs" (A).

    ``qbar_t^T R_{j-t} kbar_j = sum_a [C_a cos(delta theta_a) + S_a sin(delta theta_a)]``,
    ``C_a = q_x k_x + q_y k_y``, ``S_a = q_y k_x - q_x k_y``, ``delta = j - t``. ``qbar (d_h,)``
    and ``kbar (J, d_h)`` are pre-RoPE operands (after Qwen3 head norm); ``scaling`` is
    ``1/sqrt(d_h)`` to compare with the native pre-softmax score ``s``. ``reference`` is
    ``scaling (R_t qbar)^T (R_j kbar)``. ``extras`` holds the per-pair ``C``, ``S`` arrays.
    """
    torch = _torch()
    q, k = f64(qbar), f64(kbar)
    half = q.shape[-1] // 2
    qx, qy = q[:half], q[half:]
    kx, ky = k[:, :half], k[:, half:]
    c = qx * kx + qy * ky
    s = qy * kx - qx * ky
    delta = torch.tensor([j - t for j in positions], dtype=torch.float64)
    angle = delta[:, None] * rope_inv_freq(theta_base, q.shape[-1])[None, :]
    terms = {
        "cos": scaling * (c * torch.cos(angle)).sum(-1),
        "sin": scaling * (s * torch.sin(angle)).sum(-1),
    }
    qt = rope_matrix(t, theta_base, q.shape[-1]) @ q
    kj = torch.stack([rope_matrix(j, theta_base, q.shape[-1]) @ k[i]
                      for i, j in enumerate(positions)])
    value, dtype, scale = _native(native)
    return IdentityResult("P04", "rope_score_pairs", "A", EXACT, terms, sum(terms.values()),
                          reference=scaling * (kj @ qt), native=value, native_dtype=dtype,
                          native_scale=scale,
                          extras={"C": c, "S": s})


# --------------------------------------------------------------------------- P05 softmax


def scores(query: Any, keys: Any, *, scaling: float, native: Any = None) -> IdentityResult:
    """P05 scores. Chapter 04 "Scores and competition" (A): ``s_j = q^T k_j / sqrt(d_h)``.

    ``query (..., d_h)`` positioned, ``keys (..., J, d_h)`` positioned and mapped through kappa.
    Port: ``s`` from ``qr``, ``kr``. A single pending row sees every cached source (no mask).
    """
    torch = _torch()
    total = scaling * torch.einsum("...d,...jd->...j", f64(query), f64(keys))
    value, dtype, scale = _native(native)
    return IdentityResult("P05", "scores", "A", EXACT, {}, total, native=value,
                          native_dtype=dtype, native_scale=scale)


def softmax(score: Any, *, native: Any = None) -> IdentityResult:
    """P05 probabilities. Chapter 04: ``p_j = e^{s_j} / sum_{u in J_t} e^{s_u}`` (A).

    Port: ``p`` from ``s``. Max subtraction leaves real-arithmetic probabilities unchanged.
    """
    torch = _torch()
    value, dtype, scale = _native(native)
    return IdentityResult("P05", "softmax", "A", EXACT, {}, torch.softmax(f64(score), -1),
                          native=value, native_dtype=dtype, native_scale=scale)


def softmax_contrasts(query: Any, keys: Any, j0: int, *, scaling: float,
                      native_probs: Any = None) -> IdentityResult:
    """P05 contrasts. Chapter 04 "Contrast and sensitivity" (A).

    ``log(p_j / p_{j0}) = q^T (k_j - k_{j0}) / sqrt(d_h)``. ``reference`` is the float64 log-ratio
    of ``softmax(s)``; ``native`` is the log-ratio of the captured ``p`` (if given). Assumes fixed
    keys and visible set.
    """
    torch = _torch()
    q, k = f64(query), f64(keys)
    total = scaling * (k - k[j0]) @ q
    logp = torch.log_softmax(scaling * (k @ q), -1)
    native = None
    if native_probs is not None:
        lp = torch.log(f64(native_probs))
        native = lp - lp[..., j0:j0 + 1]
    return IdentityResult("P05", "softmax_contrasts", "A", EXACT, {}, total,
                          reference=logp - logp[j0], native=native,
                          native_dtype=None if native_probs is None else str(native_probs.dtype),
                          assumptions=("fixed keys", "fixed visible set"))


def anchored_transport(probs: Any, values: Any, j0: int) -> IdentityResult:
    """P05 anchored return. Chapter 04: ``A = v_{j0} + sum_j p_j (v_j - v_{j0})`` (A).

    Requires ``sum_j p_j = 1``. Rounded native ``p`` has ``sum p = 1 - eps``; the premise is
    reported as ``premises['mass_defect']`` (``measured``, ``eps``) and the identity keeps closing
    through the explicit term ``premise_remainder`` (``derived-exact``). Derivation: with
    ``sum p = 1 - eps``, ``v_{j0} + p.(v - v_{j0}) = v_{j0} + p.v - (1 - eps) v_{j0}
    = p.v + eps v_{j0}``, so ``p.v = anchor + contrast - eps v_{j0}`` and
    ``premise_remainder = -eps v_{j0}``. ``extras['mass_defect']`` is ``eps`` as a float.
    """
    p, v = f64(probs), f64(values)
    eps = 1 - p.sum()
    terms = {"anchor": v[j0], "contrast": p @ (v - v[j0]), "premise_remainder": -eps * v[j0]}
    return IdentityResult("P05", "anchored_transport", "A", EXACT, terms, sum(terms.values()),
                          reference=p @ v, extras={"mass_defect": float(eps)},
                          premises={"mass_defect": premise(abs(float(eps)) <= 1e-12, abs(float(eps)),
                                                           1e-12, "P05")},
                          assumptions=("probabilities sum to one",))


def query_shift_invariance(query: Any, keys: Any, direction: Any, *, scaling: float, j0: int = 0,
                           doses: Sequence[float] = (1.0, 10.0, 100.0),
                           project: bool = False) -> IdentityResult:
    """P05 query invariance. Chapter 04 "Finite source edits"; chapter 09 section 3 (T).

    At fixed keys, a query edit orthogonal to every contrast ``k_j - k_{j0}`` preserves the
    probabilities at every dose. The premise ``edit_orthogonal_to_contrasts`` is checked on the
    direction AS GIVEN: ``leak = max|(k_j - k_{j0}) . xi|`` against ``1e-9 max||k_j - k_{j0}|| ||xi||``.
    If it fails the result is ``refused`` (reason carries ``removed_norm``, the norm of the part of
    ``xi`` inside the contrast span, and the tolerance); the terms are still computed for the given
    direction, for inspection only. With ``project=True`` the direction is first projected onto the
    orthogonal complement of the contrast span; the result then answers the PROJECTED question
    (``extras['direction'] == 'projected'``, shown in ``summary()['direction']``) and
    ``removed_norm`` is a ``measured`` premise value (``premises['removed_norm']``).
    Terms are ``p(q + alpha xi) - p(q)`` per dose; ``reference`` is zero, so closure is the largest
    probability change. Assumes fixed keys and visible set.
    """
    torch = _torch()
    q, k, xi = f64(query), f64(keys), f64(direction)
    contrasts = torch.cat([k[:j0], k[j0 + 1:]]) - k[j0]
    left, sing, _ = torch.linalg.svd(contrasts.T, full_matrices=False)
    rank = int((sing > sing.max() * 1e-12).sum()) if sing.numel() else 0
    basis = left[:, :rank]
    removed = float((basis @ (basis.T @ xi)).norm())
    used = xi - basis @ (basis.T @ xi) if project else xi
    p0 = torch.softmax(scaling * (k @ q), -1)
    terms = {f"dose_{alpha:g}": torch.softmax(scaling * (k @ (q + alpha * used)), -1) - p0
             for alpha in doses}
    stacked = torch.stack(list(terms.values()))
    extras = {"perp_norm": float(used.norm()), "removed_norm": removed, "contrast_rank": rank,
              "direction": "projected" if project else "as_given"}
    result = IdentityResult("P05", "query_shift_invariance", "T", EXACT, terms, stacked,
                            reference=torch.zeros_like(stacked), extras=extras,
                            assumptions=("fixed keys", "fixed visible set",
                                         "edit orthogonal to every key contrast"))
    if project:
        result.premises["removed_norm"] = premise(True, removed, None, "P05")
    leak = _maxabs(contrasts @ used)
    tol = 1e-9 * max(float(contrasts.norm(dim=-1).max()) * float(used.norm()), 1e-300)
    result.refuse_on("edit_orthogonal_to_contrasts", leak <= tol, leak, tol)
    if result.state == "refused":
        result.reason = (f"premise edit_orthogonal_to_contrasts violated: the direction as given "
                         f"has a component of norm removed_norm={removed:.6g} inside the contrast "
                         f"span (max|contrast . xi| = {leak:.6g} > tolerance {tol:.6g}); pass "
                         f"project=True to ask the projected question instead")
    return result


# --------------------------------------------------------------------------- P06 differential


def attention_differential(query: Any, keys: Any, values: Any, *, scaling: float,
                           dquery: Any = None, dkeys: Any = None, dvalues: Any = None,
                           adjoint: Any = None) -> IdentityResult:
    """P06 attention differential. Chapter 04 "Contrast and sensitivity"; chapter 09 section 4 (D).

    ``delta A = sum_j p_j delta v_j + sum_j p_j (v_j - A) delta s_j``,
    ``delta s_j = (delta q^T k_j + q^T delta k_j)/sqrt(d_h)``;
    ``J_q = sum_j p_j (v_j - A) k_j^T / sqrt(d_h)``, ``J_{k_j} = p_j (v_j - A) q^T / sqrt(d_h)``,
    ``J_{v_j} = p_j I``. With ``adjoint = lambda_A``: ``dM/ds_j = p_j lambda^T (v_j - A)``,
    ``grad_{k_j} M = p_j [lambda^T (v_j - A)] q / sqrt(d_h)``. One head: ``query (d_h,)``,
    ``keys (J, d_h)``, ``values (J, e)``. Ports treated independently (fixed visible set); an
    upstream edit must compose producer Jacobians. Output is first-order (``derived-local``).
    """
    torch = _torch()
    q, k, v = f64(query), f64(keys), f64(values)
    p = torch.softmax(scaling * (k @ q), -1)
    a = p @ v
    centred = v - a
    dq = torch.zeros_like(q) if dquery is None else f64(dquery)
    dk = torch.zeros_like(k) if dkeys is None else f64(dkeys)
    dv = torch.zeros_like(v) if dvalues is None else f64(dvalues)
    ds = scaling * (k @ dq + dk @ q)
    terms = {"payload": p @ dv, "routing": (p * ds) @ centred}
    extras: dict[str, Any] = {
        "probs": p, "A": a, "ds": ds,
        "J_q": scaling * torch.einsum("j,je,jd->ed", p, centred, k),
        "J_k": scaling * torch.einsum("j,je,d->jed", p, centred, q),
    }
    if adjoint is not None:
        lam = f64(adjoint)
        dm_ds = p * (centred @ lam)
        extras["dM_ds"] = dm_ds
        extras["grad_k"] = scaling * dm_ds[:, None] * q[None, :]
        extras["grad_q"] = extras["J_q"].T @ lam
        extras["grad_v"] = p[:, None] * lam[None, :]
    return IdentityResult("P06", "attention_differential", "D", LOCAL, terms, sum(terms.values()),
                          extras=extras,
                          assumptions=("fixed visible set", "Q/K/V ports varied independently"))


# --------------------------------------------------------------------------- P07 finite split


def score_finite_change(query0: Any, query1: Any, keys0: Any, keys1: Any, *, scaling: float,
                        native0: Any = None, native1: Any = None) -> IdentityResult:
    """P07 finite score split. Chapter 04 "Finite source edits"; chapter 09 section 3 (A/F).

    ``Delta s_j = (Delta q^T k_j + q^T Delta k_j + Delta q^T Delta k_j)/sqrt(d_h)`` with positioned
    q, k (``qr``, ``kr``) at matched receiver and visible set. ``query (..., d_h)``,
    ``keys (..., J, d_h)``. ``native`` is ``s1 - s0`` from the captured ``s`` port.
    """
    torch = _torch()
    q0, q1, k0, k1 = map(f64, (query0, query1, keys0, keys1))
    dq, dk = q1 - q0, k1 - k0

    def dot(a, b):
        return scaling * torch.einsum("...d,...jd->...j", a, b)

    terms = {"query": dot(dq, k0), "key": dot(q0, dk), "interaction": dot(dq, dk)}
    value, dtype, scale = _native(native0=native0, native1=native1)
    return IdentityResult("P07", "score_finite_change", "F", EXACT, terms, sum(terms.values()),
                          reference=dot(q1, k1) - dot(q0, k0), native=value, native_dtype=dtype,
                          native_scale=scale,
                          assumptions=("matched receiver", "matched visible set"))


def transport_finite_split(probs0: Any, probs1: Any, values0: Any, values1: Any, *,
                           native0: Any = None, native1: Any = None) -> IdentityResult:
    """P07 finite routing/payload split. Chapter 04; chapter 09 section 4 (A/F).

    ``Delta A = sum_j [p_{0,j} Delta v_j + Delta p_j v_{0,j} + Delta p_j Delta v_j]``: old-route
    payload change, old-payload routing change, finite interaction. ``probs (..., J)``,
    ``values (..., J, e)``. ``native`` is the ``A`` port difference; ``eta`` then contains the
    native mix rounding at both endpoints.
    """
    torch = _torch()
    p0, p1, v0, v1 = map(f64, (probs0, probs1, values0, values1))
    dp, dv = p1 - p0, v1 - v0

    def mix(p, v):
        return torch.einsum("...j,...je->...e", p, v)

    terms = {"payload": mix(p0, dv), "routing": mix(dp, v0), "interaction": mix(dp, dv)}
    value, dtype, scale = _native(native0=native0, native1=native1)
    return IdentityResult("P07", "transport_finite_split", "F", EXACT, terms,
                          sum(terms.values()), reference=mix(p1, v1) - mix(p0, v0), native=value,
                          native_dtype=dtype,
                          native_scale=scale, assumptions=("matched visible set",))


# --------------------------------------------------------------------------- P08 reweighting


def softmax_reweighting(scores0: Any, dscores: Any, *, native1: Any = None) -> IdentityResult:
    """P08 exact finite softmax reweighting. Chapter 04; chapter 09 section 3 (F).

    ``p'_j = p_j e^{Delta s_j} / sum_u p_u e^{Delta s_u}`` with ``p = softmax(s0)``. Exact at any
    dose for the same visible set and finite logits; ``reference`` is ``softmax(s0 + Delta s)``.
    ``native1`` is the captured ``p`` at the changed endpoint.
    """
    torch = _torch()
    s0, ds = f64(scores0), f64(dscores)
    p0 = torch.softmax(s0, -1)
    w = p0 * torch.exp(ds - ds.max(-1, keepdim=True).values)
    total = w / w.sum(-1, keepdim=True)
    value, dtype, scale = _native(native1)
    return IdentityResult("P08", "softmax_reweighting", "F", EXACT, {}, total,
                          reference=torch.softmax(s0 + ds, -1), native=value, native_dtype=dtype,
                          native_scale=scale,
                          assumptions=("same visible source set", "finite allowed logits"))


def one_source_assumptions(scores0: Any, scores1: Any, values0: Any, values1: Any, source: int,
                           *, query0: Any = None, query1: Any = None) -> dict[str, Any]:
    """Check P08's one-source premises on captured operands, bitwise.

    Premises (chapter 04 "Finite source edits", chapter 09 section 5): only source ``j``'s score
    and value change, the receiver query is fixed, the visible set is fixed.
    """
    torch = _torch()
    s0, s1, v0, v1 = scores0, scores1, values0, values1
    others = [i for i in range(s0.shape[-1]) if i != source]
    same_set = tuple(s0.shape) == tuple(s1.shape) and tuple(v0.shape) == tuple(v1.shape)
    out: dict[str, Any] = {
        "visible_set_fixed": same_set,
        "other_scores_fixed": same_set and bool(torch.equal(s0[..., others], s1[..., others])),
        "other_values_fixed": same_set and bool(torch.equal(v0[..., others, :],
                                                            v1[..., others, :])),
        "other_score_max_abs_change": _maxabs(f64(s1[..., others]) - f64(s0[..., others]))
        if same_set else float("nan"),
        "other_value_max_abs_change": _maxabs(f64(v1[..., others, :]) - f64(v0[..., others, :]))
        if same_set else float("nan"),
        "query_fixed": None,
    }
    if query0 is not None and query1 is not None:
        out["query_fixed"] = bool(torch.equal(query0, query1))
    out["holds"] = bool(out["visible_set_fixed"] and out["other_scores_fixed"]
                        and out["other_values_fixed"] and out["query_fixed"] is not False)
    return out


def _one_source_premises(checks: Mapping[str, Any], scores0: Any, scores1: Any, values0: Any,
                         values1: Any, query0: Any, query1: Any) -> dict[str, dict[str, Any]]:
    """Labelled premises of ``one_source_response`` (bitwise checks, tolerance 0)."""
    sizes = abs(int(scores0.shape[-1]) - int(scores1.shape[-1])) + abs(
        int(values0.shape[-2]) - int(values1.shape[-2]))
    out = {
        "visible_set_fixed": premise(checks["visible_set_fixed"], sizes, 0.0, "P08"),
        "other_scores_fixed": premise(checks["other_scores_fixed"],
                                      checks["other_score_max_abs_change"], 0.0, "P08"),
        "other_values_fixed": premise(checks["other_values_fixed"],
                                      checks["other_value_max_abs_change"], 0.0, "P08"),
    }
    if query0 is not None and query1 is not None:
        out["query_fixed"] = premise(checks["query_fixed"],
                                     _maxabs(f64(query1) - f64(query0)), 0.0, "P08")
    return out


def one_source_response(scores0: Any, scores1: Any, values0: Any, values1: Any, source: int, *,
                        query0: Any = None, query1: Any = None, native0: Any = None,
                        native1: Any = None) -> IdentityResult:
    """P08 boxed one-source closed form. Chapter 09 section 5; chapter 04 (F under T premises).

    ``Delta a = [p_j (e^delta - 1)(v_j - a) + p_j e^delta Delta v_j] / Z``,
    ``Z = 1 + p_j (e^delta - 1)``, ``delta = s_{1,j} - s_{0,j}``, ``a = sum_i p_i v_i``,
    ``p = softmax(s0)``. One head: ``scores (J,)``, ``values (J, e)``. Premises: fixed receiver
    query, fixed visible set, only source ``j``'s score and value change. They are checked
    bitwise (``extras['assumptions']``, labelled in ``premises``); outside them the result is
    ``refused`` (``premise <name> violated``) and its terms are for inspection only.
    ``reference`` is the full float64 ``softmax(s1) v1 - softmax(s0) v0``, so a violated premise
    also shows as a closure failure.
    """
    torch = _torch()
    s0, s1, v0, v1 = map(f64, (scores0, scores1, values0, values1))
    p0 = torch.softmax(s0, -1)
    a = p0 @ v0
    delta = s1[source] - s0[source]
    pj = p0[source]
    z = 1 + pj * torch.expm1(delta)
    terms = {
        "routing": pj * torch.expm1(delta) * (v0[source] - a) / z,
        "payload": pj * torch.exp(delta) * (v1[source] - v0[source]) / z,
    }
    reference = torch.softmax(s1, -1) @ v1 - a
    value, dtype, scale = _native(native0=native0, native1=native1)
    checks = one_source_assumptions(scores0, scores1, values0, values1, source, query0=query0,
                                    query1=query1)
    result = IdentityResult(
        "P08", "one_source_response", "F", EXACT, terms, sum(terms.values()), reference=reference,
        native=value, native_dtype=dtype, native_scale=scale,
        assumptions=("fixed receiver query", "fixed visible set",
                     "only source j's score and value change"),
        extras={"assumptions": checks, "delta": float(delta), "p_j": float(pj), "Z": float(z)},
    )
    result.premises = _one_source_premises(checks, scores0, scores1, values0, values1, query0,
                                           query1)
    for name, entry in result.premises.items():
        if not entry["holds"]:
            result.state, result.reason = "refused", f"premise {name} violated"
            break
    return result


# --------------------------------------------------------------------------- P09 cuts


def cut_sources(probs: Any, values: Any, cut: Iterable[int], *, native_base: Any = None,
                native_zero: Any = None, native_mask: Any = None) -> dict[str, IdentityResult]:
    """P09 zero-cut versus mask-and-renormalize. Chapter 04 "Cuts and image gates" (A).

    ``A_S = sum_{j in S} p_j v_j``, ``mu = sum_{j in S} p_j``; ``Delta A_zero = -A_S``;
    ``Delta A_mask = (mu A - A_S)/(1 - mu)`` for ``mu < 1`` and probabilities summing to one.
    Rounded native ``p`` has ``sum p = 1 - eps``: ``premises['mass_defect']`` reports
    ``eps`` (``measured``) and the mask result carries the explicit term ``premise_remainder``
    (``derived-exact``) so it closes to float64 roundoff on native ``p``. Derivation: with
    ``A_K = A - A_S`` (kept payload) the chapter form is ``A_K/(1 - mu) - A`` (since
    ``A_K/(1-mu) - A = (mu A - A_S)/(1 - mu)``) while the renormalised native value is
    ``A_K/(sum p - mu) - A`` with ``sum p - mu = 1 - mu - eps``; the difference is
    ``premise_remainder = A_K eps / ((1 - mu)(1 - mu - eps))``. The zero cut needs no premise.
    ``probs (..., J)``, ``values (..., J, e)``. References: ``p_{not S} v - A`` and
    ``(p_{not S}/sum p_{not S}) v - A`` evaluated directly. ``native_*`` are ``A`` captures (base,
    zero-cut, mask-cut runs). ``mu = 1`` (all mass cut) makes the mask numbers non-finite; the
    summary then reports ``unresolved``.
    """
    torch = _torch()
    p, v = f64(probs), f64(values)
    keep = torch.ones(p.shape[-1], dtype=torch.float64)
    keep[list(cut)] = 0.0

    def mix(pp):
        return torch.einsum("...j,...je->...e", pp, v)

    a = mix(p)
    a_s = mix(p * (1 - keep))
    mu = (p * (1 - keep)).sum(-1, keepdim=True)
    zero_ref = mix(p * keep) - a
    mask_ref = mix(p * keep / (p * keep).sum(-1, keepdim=True)) - a
    zv, zd, zs = _native(native0=native_base, native1=native_zero)
    mv, md, ms = _native(native0=native_base, native1=native_mask)
    defect = 1 - p.sum(-1, keepdim=True)
    remainder = (a - a_s) * defect / ((1 - mu) * (1 - mu - defect))
    mask_terms = {"removed_payload": -a_s / (1 - mu), "reallocation": mu * a / (1 - mu),
                  "premise_remainder": remainder}
    mask = IdentityResult(
        "P09", "cut_mask_renormalize", "A", EXACT, mask_terms, sum(mask_terms.values()),
        reference=mask_ref, native=mv, native_dtype=md, native_scale=ms,
        extras={"mu": mu, "mass_defect": defect},
        assumptions=("mu < 1", "probabilities sum to one"))
    worst = _maxabs(defect)
    mu_max = float(mu.max())
    mask.premises = {
        "mass_defect": premise(worst <= 1e-12, worst, 1e-12, "P09"),
        "mu_lt_1": premise(mu_max < 1.0, mu_max, 1.0, "P09"),
    }
    return {
        "zero": IdentityResult("P09", "cut_zero", "A", EXACT, {"removed_payload": -a_s}, -a_s,
                               reference=zero_ref, native=zv, native_dtype=zd, native_scale=zs,
                               extras={"mu": mu}),
        "mask": mask,
    }


# --------------------------------------------------------------------------- P10 / P11 SwiGLU


def swiglu_forward(n: Any, gate_weight: Any, up_weight: Any, down_weight: Any, *,
                   native: Mapping[str, Any] | None = None) -> dict[str, IdentityResult]:
    """P10 SwiGLU write. Chapter 05 "Two arguments and a write" (A).

    ``g = G n``, ``u = U n``, ``f = S(g) * u``, ``m = D^M f`` (bias-free). Ports ``g``, ``sg``,
    ``u``, ``f``, ``m`` from ``n``. ``native`` maps port names to captures. Each stage is
    evaluated from the float64 upstream value, so ``eta`` accumulates the chain's native rounding.
    """
    n = f64(n)
    g = n @ f64(gate_weight).T
    u = n @ f64(up_weight).T
    sg = silu(g)
    f = sg * u
    m = f @ f64(down_weight).T
    native = native or {}
    out = {}
    for port, value in (("g", g), ("sg", sg), ("u", u), ("f", f), ("m", m)):
        nv, nd, ns = _native(native.get(port))
        out[port] = IdentityResult("P10", f"swiglu_{port}", "A", EXACT, {}, value, native=nv,
                                   native_dtype=nd, native_scale=ns)
    return out


def swiglu_gate(g: Any, *, native: Any = None) -> IdentityResult:
    """P10 activated gate ``sg = S(g) = g / (1 + e^-g)`` (chapter 05; port ``sg`` from ``g``)."""
    value, dtype, scale = _native(native)
    return IdentityResult("P10", "swiglu_gate", "A", EXACT, {}, silu(f64(g)), native=value,
                          native_dtype=dtype, native_scale=scale)


def swiglu_product(sg: Any, u: Any, *, native: Any = None) -> IdentityResult:
    """P10 product ``f = S(g) * u`` (chapter 05; port ``f`` from ``sg``, ``u``)."""
    value, dtype, scale = _native(native)
    return IdentityResult("P10", "swiglu_product", "A", EXACT, {}, f64(sg) * f64(u),
                          native=value, native_dtype=dtype, native_scale=scale)


def mlp_state_operator(r: Any, gate_weight: Any, up_weight: Any, down_weight: Any, gamma: Any,
                       eps: float, *, native_m: Any = None) -> IdentityResult:
    """P10 state-conditioned factorization. Chapter 05 "Quadratic geometry" (A).

    ``m = B(r) r``, ``B(r) = D^M diag(S(G n)) U rho^M(r) Gamma^M``. One row ``r (d,)``.
    ``reference`` is the forward ``D^M (S(G n) * U n)`` with ``n = N(r)``. ``B`` is a
    state-conditioned operator, not the Jacobian.
    """
    r, gm = f64(r), f64(gamma)
    gw, uw, dw = f64(gate_weight), f64(up_weight), f64(down_weight)
    rho = _rho(r, eps)
    n = rho * gm * r
    b_op = dw @ (silu(gw @ n)[:, None] * uw) * rho * gm[None, :]
    value, dtype, scale = _native(native_m)
    return IdentityResult("P10", "mlp_state_operator", "A", EXACT, {}, b_op @ r,
                          reference=dw @ (silu(gw @ n) * (uw @ n)), native=value,
                          native_dtype=dtype, native_scale=scale, extras={"B": b_op})


def neuron_quadratic_form(gate_row: Any, up_row: Any, n: Any, *,
                          native_f: Any = None) -> IdentityResult:
    """P11 rank-two quadratic form. Chapter 05 "Quadratic geometry" (A).

    ``f_a(n) = sigma(g_*^T n) n^T H_a n``, ``H_a = (g_* u_*^T + u_* g_*^T)/2``, eigenvalues
    ``lambda_pm = (g_*^T u_* pm ||g_*|| ||u_*||)/2``. ``reference`` is ``S(g_*^T n)(u_*^T n)``.
    ``extras`` holds ``H``, the closed-form and numerical extreme eigenvalues, their max
    difference, and the numerical rank (tolerance ``1e-12 * max|eig|``).
    """
    torch = _torch()
    g, u, x = f64(gate_row), f64(up_row), f64(n)
    h = (torch.outer(g, u) + torch.outer(u, g)) / 2
    total = torch.sigmoid(g @ x) * (x @ h @ x)
    eig = torch.linalg.eigvalsh(h)
    closed = torch.stack([(g @ u - g.norm() * u.norm()) / 2, (g @ u + g.norm() * u.norm()) / 2])
    numeric = torch.stack([eig[0], eig[-1]])
    rank = int((eig.abs() > 1e-12 * eig.abs().max()).sum())
    value, dtype, scale = _native(native_f)
    return IdentityResult("P11", "neuron_quadratic_form", "A", EXACT, {}, total,
                          reference=silu(g @ x) * (u @ x), native=value, native_dtype=dtype,
                          native_scale=scale,
                          extras={"H": h, "eig_closed": closed, "eig_numeric": numeric,
                                  "eig_closure": _maxabs(closed - numeric), "rank": rank})


def neuron_read_plane(gate_row: Any, up_row: Any, n: Any) -> IdentityResult:
    """P12 normalized read plane. Chapter 05 "Read plane and derivatives" (D).

    With ``a = g_*^T n``, ``b = u_*^T n``: ``grad_n f = b S'(a) g_* + S(a) u_*``,
    ``Hess_n f = b S''(a) g_* g_*^T + S'(a)(g_* u_*^T + u_* g_*^T)``. ``total`` is the gradient,
    ``extras['hessian']`` the Hessian, ``extras['plane_residual']`` the largest component of
    either outside ``span{g_*, u_*}`` (zero in real arithmetic). First-order object.
    """
    torch = _torch()
    g, u, x = f64(gate_row), f64(up_row), f64(n)
    a, b = g @ x, u @ x
    terms = {"gate_branch": b * silu_d1(a) * g, "up_branch": silu(a) * u}
    hess = b * silu_d2(a) * torch.outer(g, g) + silu_d1(a) * (torch.outer(g, u)
                                                              + torch.outer(u, g))
    basis, _ = torch.linalg.qr(torch.stack([g, u], 1))
    grad = sum(terms.values())

    def outside(m):
        return _maxabs(m - basis @ (basis.T @ m))

    return IdentityResult("P12", "neuron_read_plane", "D", LOCAL, terms, grad,
                          extras={"hessian": hess, "S": silu(a), "S1": silu_d1(a),
                                  "S2": silu_d2(a),
                                  "plane_residual": max(outside(grad[:, None]), outside(hess))})


def neuron_raw_hessian(gate_row: Any, up_row: Any, gamma: Any, eps: float,
                       r: Any) -> IdentityResult:
    """P13 raw-residual Hessian. Chapter 05 "Read plane and derivatives" (D).

    ``alpha = (Gamma g_*)^T r``, ``beta = (Gamma u_*)^T r``, ``c = r^T r``,
    ``Phi = S(rho(c) alpha) rho(c) beta``; ``Hess_r f = J_s^T (Hess_s Phi) J_s + 2 Phi_c I`` with
    rows of ``J_s`` = ``(Gamma g_*)^T, (Gamma u_*)^T, 2 r^T``. ``Hess_s Phi`` and ``Phi_c`` are
    evaluated by float64 autograd of the chapter's ``Phi`` (no new closed form is introduced).
    """
    torch = _torch()
    g, u, gm, x = f64(gate_row), f64(up_row), f64(gamma), f64(r)
    d = x.shape[-1]
    js = torch.stack([gm * g, gm * u, 2 * x])

    def phi(s):
        rho = 1.0 / torch.sqrt(eps + s[2] / d)
        return silu(rho * s[0]) * rho * s[1]

    s = torch.stack([(gm * g) @ x, (gm * u) @ x, x @ x])
    with torch.enable_grad():
        grad_phi = torch.autograd.functional.jacobian(phi, s)
        hess_phi = torch.autograd.functional.hessian(phi, s)
    chain = js.T @ hess_phi @ js
    terms = {"chain": chain, "isotropic": 2 * grad_phi[2] * torch.eye(d, dtype=torch.float64)}
    return IdentityResult("P13", "neuron_raw_hessian", "D", LOCAL, terms, sum(terms.values()),
                          extras={"chain_rank": int(torch.linalg.matrix_rank(chain)),
                                  "Phi_c": float(grad_phi[2])},
                          notes=("Hess_s Phi, Phi_c by float64 autograd of the chapter's Phi",))


# --------------------------------------------------------------------------- P14 Jacobian


def mlp_jacobian(r: Any, gate_weight: Any, up_weight: Any, down_weight: Any, gamma: Any,
                 eps: float) -> IdentityResult:
    """P14 MLP Jacobian. Chapter 05 "Whole-block response" (D).

    ``J_M = D^M [diag(u * S'(g)) G + diag(S(g)) U] J_N(r)`` for one row ``r (d,)``. Terms are the
    gate and up branches. First-order object (``derived-local``).
    """
    r = f64(r)
    gw, uw, dw = f64(gate_weight), f64(up_weight), f64(down_weight)
    jn = rms_jacobian(r, gamma, eps).extras["jacobian"]
    n = rms_norm(r, gamma, eps).total
    g, u = gw @ n, uw @ n
    terms = {
        "gate_branch": dw @ ((u * silu_d1(g))[:, None] * gw) @ jn,
        "up_branch": dw @ (silu(g)[:, None] * uw) @ jn,
    }
    return IdentityResult("P14", "mlp_jacobian", "D", LOCAL, terms, sum(terms.values()))


def mlp_pullback(adjoint_out: Any, r: Any, gate_weight: Any, up_weight: Any, down_weight: Any,
                 gamma: Any, eps: float) -> IdentityResult:
    """P14 consumer pullback through the MLP block with its skip. Chapters 05 and 07 (D).

    ``lambda_r = (I + J_M)^T lambda_{h_{l+1}}``; ``extras['neuron_multiplier']`` is
    ``d_a^T lambda_{h_{l+1}}`` for every neuron (chapter 05 "Whole-block response").
    """
    lam = f64(adjoint_out)
    jm = mlp_jacobian(r, gate_weight, up_weight, down_weight, gamma, eps).total
    terms = {"skip": lam, "mlp": jm.T @ lam}
    return IdentityResult("P14", "mlp_pullback", "D", LOCAL, terms, sum(terms.values()),
                          extras={"neuron_multiplier": f64(down_weight).T @ lam})


# --------------------------------------------------------------------------- P15 finite SwiGLU


def swiglu_finite_split(g0: Any, g1: Any, u0: Any, u1: Any, *, native0: Any = None,
                        native1: Any = None) -> IdentityResult:
    """P15 finite gate/up decomposition. Chapter 05 "Finite products"; chapter 09 section 6 (A/F).

    ``Delta f = u_0 * Delta s + s_0 * Delta u + Delta s * Delta u`` with ``s_i = S(g_i)``.
    ``extras['symmetric']`` is the equally exact split
    ``(u_0+u_1)/2 * Delta s + (s_0+s_1)/2 * Delta u`` and ``extras['symmetric_closure']`` its
    closure. ``native`` is the ``f`` port difference. Neither split is a unique causal
    decomposition.
    """
    g0, g1, u0, u1 = map(f64, (g0, g1, u0, u1))
    s0, s1 = silu(g0), silu(g1)
    ds, du = s1 - s0, u1 - u0
    terms = {"gate": u0 * ds, "up": s0 * du, "interaction": ds * du}
    reference = s1 * u1 - s0 * u0
    symmetric = {"gate": (u0 + u1) / 2 * ds, "up": (s0 + s1) / 2 * du}
    value, dtype, scale = _native(native0=native0, native1=native1)
    return IdentityResult("P15", "swiglu_finite_split", "F", EXACT, terms, sum(terms.values()),
                          reference=reference, native=value, native_dtype=dtype, native_scale=scale,
                          extras={"symmetric": symmetric,
                                  "symmetric_closure": _maxabs(sum(symmetric.values())
                                                               - reference)})


def gauss_legendre(k: int) -> tuple[Any, Any]:
    """``k``-point Gauss-Legendre nodes and weights on [0, 1] (float64)."""
    import numpy as np

    torch = _torch()
    x, w = np.polynomial.legendre.leggauss(int(k))
    return torch.tensor((x + 1) / 2, dtype=torch.float64), torch.tensor(w / 2,
                                                                         dtype=torch.float64)


def integral_mean_slope(g0: Any, g1: Any, k: int) -> IdentityResult:
    """P15 integral-mean slope. Chapter 09 section 6 (F identity, quadrature APPROXIMATION).

    ``Delta s_a = Sbar'_a Delta g_a``, ``Sbar'_a = int_0^1 S'(g_{0,a} + tau Delta g_a) d tau``.
    The integral is evaluated by ``k``-point Gauss-Legendre: ``total`` is the quadrature
    ``Sbar'_k Delta g``, ``reference`` is ``S(g_1) - S(g_0)``, and closure is the QUADRATURE
    ERROR, not roundoff. ``extras['slope']`` is ``Sbar'_k`` (no division by ``Delta g``).
    """
    g0, g1 = f64(g0), f64(g1)
    tau, w = gauss_legendre(k)
    dg = g1 - g0
    slope = sum(wi * silu_d1(g0 + ti * dg) for ti, wi in zip(tau, w))
    return IdentityResult("P15", "integral_mean_slope", "F", LOCAL, {}, slope * dg,
                          reference=silu(g1) - silu(g0), extras={"slope": slope},
                          approximation={"rule": "gauss-legendre", "k": int(k)},
                          notes=(f"{k}-point quadrature approximation of an exact integral",))


def swiglu_source_pairs(gate_row: Any, up_row: Any, components: Mapping[str, Any], *,
                        native_f: Any = None) -> IdentityResult:
    """P15 source-component products. Chapter 05 "Finite products and source pairs" (A).

    For ``n = sum_b n^[b]``: ``f_a = sigma(g_*^T n) sum_{b,c} (g_*^T n^[b])(u_*^T n^[c])``. Terms
    are keyed ``"b|c"``; the shared sigmoid uses the whole state. ``reference`` is
    ``S(g_*^T n)(u_*^T n)``. Pair terms are not independent counterfactuals.
    """
    torch = _torch()
    g, u = f64(gate_row), f64(up_row)
    comps = {k: f64(v) for k, v in components.items()}
    n = sum(comps.values())
    sig = torch.sigmoid(g @ n)
    terms = {f"{b}|{c}": sig * (g @ nb) * (u @ nc)
             for (b, nb), (c, nc) in itertools.product(comps.items(), repeat=2)}
    value, dtype, scale = _native(native_f)
    return IdentityResult("P15", "swiglu_source_pairs", "A", EXACT, terms, sum(terms.values()),
                          reference=silu(g @ n) * (u @ n), native=value, native_dtype=dtype,
                          native_scale=scale,
                          assumptions=("complete decomposition n = sum of components",))


def down_finite_split(down0: Any, down1: Any, f0: Any, f1: Any, *, native0: Any = None,
                      native1: Any = None) -> IdentityResult:
    """P15 down-map edit. Chapter 09 section 6 (A/F).

    ``Delta m = Delta D^M f_0 + D^M_0 Delta f + Delta D^M Delta f``. ``native`` is the ``m`` port
    difference.
    """
    d0, d1, f0, f1 = map(f64, (down0, down1, f0, f1))
    dd, df = d1 - d0, f1 - f0
    terms = {"down_edit": f0 @ dd.T, "product_change": df @ d0.T, "interaction": df @ dd.T}
    value, dtype, scale = _native(native0=native0, native1=native1)
    return IdentityResult("P15", "down_finite_split", "F", EXACT, terms, sum(terms.values()),
                          reference=f1 @ d1.T - f0 @ d0.T, native=value, native_dtype=dtype,
                          native_scale=scale)


# --------------------------------------------------------------------------- P19 / P20 readout


def readout_logits(h: Any, out_weight: Any, gamma: Any, eps: float, *, out_bias: Any = None,
                   native: Any = None) -> IdentityResult:
    """P19 final readout. Chapter 07 "Final readout" (A).

    ``lambda_{t,w} = sum_i W^out_{wi} gamma^F_i rho^F_t h_{L,t,i}`` (plus an output bias if the
    architecture has one; Qwen has none).
    """
    x = rms_norm(h, gamma, eps).total
    total = x @ f64(out_weight).T
    if out_bias is not None:
        total = total + f64(out_bias)
    value, dtype, scale = _native(native)
    return IdentityResult("P19", "readout_logits", "A", EXACT, {}, total, native=value,
                          native_dtype=dtype, native_scale=scale)


def readout_margin(h: Any, out_weight: Any, gamma: Any, eps: float, a: int, b: int, *,
                   components: Mapping[str, Any] | None = None, native_margin: Any = None,
                   native_normed: Any = None, out_bias: Any = None) -> IdentityResult:
    """P19 margin and its decomposition over residual components. Chapters 07 and 02 (A).

    ``M_ab(h) = (W_a - W_b) N_gamma^F(h) = rho^F(h) c^T h``, ``c = Gamma^F (W_a - W_b)^T``. For
    ``h = sum_k xi_k`` the terms are ``rho^F(h) c^T xi_k`` with ``rho^F`` evaluated ONCE at the full
    final state (chapter 02: not ``sum N(xi_k)``); the part of ``h`` not covered by the listed
    components is the term ``unlisted``. ``reference`` is ``(W_a - W_b) . (rho Gamma h)``.
    ``native_margin`` is ``logits[a] - logits[b]`` as captured. With ``native_normed`` (the
    captured final-norm output) ``extras`` splits eta into
    ``eta_final_norm = (W_a - W_b).(x^F_native - rho Gamma h)`` and
    ``eta_head = M_native - (W_a - W_b).x^F_native - (b_a - b_b)``. With ``out_bias`` (the head's
    bias) the margin carries the exact term ``head_bias = b_a - b_b`` (in ``terms``, ``total``
    and ``reference``).
    ``extras['gradient']`` is the local ``J_N(h)^T (W_a - W_b)^T`` (``derived-local``).
    """
    h, gm = f64(h), f64(gamma)
    diff = f64_at(out_weight, a) - f64_at(out_weight, b)     # two rows, not the vocabulary
    c = gm * diff
    rho = _rho(h, eps)
    if components is None:
        terms = {"h": rho * (c @ h)}
    else:
        terms = {name: rho * (c @ f64(xi)) for name, xi in components.items()}
        terms["unlisted"] = rho * (c @ (h - sum(f64(xi) for xi in components.values())))
    bias = 0.0 if out_bias is None else f64_at(out_bias, a) - f64_at(out_bias, b)
    if out_bias is not None:
        terms["head_bias"] = bias
    reference = diff @ (rho * gm * h) + bias
    jac = rms_jacobian(h, gamma, eps).extras["jacobian"]
    extras: dict[str, Any] = {"rho": rho, "c": c, "gradient": jac.T @ diff}
    value = None if native_margin is None else f64(native_margin)
    if native_normed is not None:
        xf = f64(native_normed)
        extras["eta_final_norm"] = diff @ xf + bias - reference
        if value is not None:
            extras["eta_head"] = value - diff @ xf - bias
    return IdentityResult("P19", "readout_margin", "A", EXACT, terms, sum(terms.values()),
                          reference=reference, native=value,
                          native_dtype=str(getattr(native_margin, "dtype", None)),
                          extras=extras,
                          assumptions=("rho^F is a function of the full final state; it is not "
                                       "attributed to components",))


# Rows of c_b = Gamma (W_a - W_b) formed at one time by ``winner_certificate`` (2^25 floats).
_WINNER_CHUNK = 1 << 25


def winner_certificate(h: Any, out_weight: Any, gamma: Any, eps: float, *,
                       delta: Any = None, bias: Any = None,
                       bias_free: bool = False) -> IdentityResult:
    """P19 decision-cell retention. Chapter 07 "Equal decisions and unequal states" (T).

    For the unique float64 winner ``a`` at ``h``: ``m_b = lambda_a - lambda_b``,
    ``c_b = Gamma^F (W_a - W_b)^T``. The winner is retained at ``h + delta`` exactly when
    ``c_b^T delta > -m_b / rho^F(h)`` for every ``b != a``; the sufficient radius is
    ``min_b m_b / (rho^F ||c_b||)`` (``bound``). ``total`` is the radius; ``extras`` report the
    exact condition and the actual float64 argmax at ``h + delta``. Real arithmetic, ties excluded.

    The head must be bias-free or its bias supplied: ``bias`` (shape ``(V,)``) is added to the
    logits exactly, so ``m_b`` includes ``b_a - b_b`` (the winner, the radius and the argmax all
    use it). The premise ``bias_free_or_supplied`` holds when ``bias`` is given or
    ``bias_free=True`` is asserted by the caller; when neither is passed the result is ``refused``
    (the premise is not checked silently).
    """
    torch = _torch()
    h, w, gm = f64(h), f64(out_weight), f64(gamma)
    b_vec = None if bias is None else f64(bias).reshape(-1)
    rho = _rho(h, eps)
    logits = (rho * gm * h) @ w.T
    if b_vec is not None:
        logits = logits + b_vec
    a = int(torch.argmax(logits))
    others = [i for i in range(w.shape[0]) if i != a]
    margins = logits[a] - logits[others]
    dv = None if delta is None else f64(delta)
    # c_b = Gamma (W_a - W_b) a block of rows at a time: its norms and c_b . delta, never a
    # second (V, d) float64 matrix; the winner's own row (c_a = 0) is dropped after
    step = max(1, _WINNER_CHUNK // int(w.shape[1]))
    norms, steer = [], []
    for start in range(0, int(w.shape[0]), step):
        block = gm[None, :] * (w[a][None, :] - w[start:start + step])
        norms.append(block.norm(dim=-1))
        if dv is not None:
            steer.append(block @ dv)
        del block
    norms = torch.cat(norms)
    norms = torch.cat((norms[:a], norms[a + 1:]))
    usable = norms > 0
    radius = (margins[usable] / (rho * norms[usable])).min()
    extras: dict[str, Any] = {"winner": a, "unique": bool((margins > 0).all()),
                              "bias": "supplied" if b_vec is not None else
                              ("bias_free" if bias_free else "unchecked")}
    if dv is not None:
        along = torch.cat(steer)
        along = torch.cat((along[:a], along[a + 1:]))
        extras["condition_holds"] = bool((along > -margins / rho).all())
        moved = h + dv
        after = (_rho(moved, eps) * gm * moved) @ w.T
        extras["argmax_after"] = int(torch.argmax(after if b_vec is None else after + b_vec))
        extras["delta_norm"] = float(dv.norm())
    result = IdentityResult("P19", "winner_certificate", "T", BOUND, {}, radius, extras=extras,
                            assumptions=("bias-free or supplied final head", "unique winner",
                                         "real arithmetic"))
    given = b_vec is not None or bias_free
    result.refuse_on("bias_free_or_supplied", given,
                     float(b_vec.norm()) if b_vec is not None else 0.0, None)
    if not given:
        result.reason = ("premise bias_free_or_supplied violated: pass bias=<head bias> or "
                         "bias_free=True")
    smallest = float(margins.min()) if margins.numel() else 0.0
    result.refuse_on("unique_winner", smallest > 0.0, smallest, 0.0)
    return result


def margin_error_bound(out_weight: Any, a: int, b: int, dy: Any) -> IdentityResult:
    """P19 pairwise margin bound. Chapter 07: ``|dM| <= ||W_a - W_b|| ||delta y||`` (T, bound)."""
    dv = f64(dy)
    diff = f64_at(out_weight, a) - f64_at(out_weight, b)
    return IdentityResult("P19", "margin_error_bound", "T", BOUND, {},
                          diff.norm() * dv.norm(), extras={"actual": float(diff @ dv)},
                          assumptions=("pairwise only; other competitors not covered",))


def prediction_error_split(step: Callable[[Any], Any], step_hat: Callable[[Any], Any], x: Any,
                           x_hat: Any) -> IdentityResult:
    """P20 recursive prediction error. Chapter 07 "Predicted trajectories" (A).

    ``eps_{i+1} = [F(x_i + eps_i) - F(x_i)] + [F^(x_i + eps_i) - F(x_i + eps_i)]``: propagated
    incoming error plus local mismatch on the actually predicted input. ``step`` and ``step_hat``
    are supplied callables (operands, not models read implicitly).
    """
    x, xh = f64(x), f64(x_hat)
    fx, fxh, fhat = f64(step(x)), f64(step(xh)), f64(step_hat(xh))
    terms = {"propagated": fxh - fx, "local": fhat - fxh}
    return IdentityResult("P20", "prediction_error_split", "A", EXACT, terms,
                          sum(terms.values()), reference=fhat - fx)


def energy_weighted_error(predicted: Any, target: Any) -> IdentityResult:
    """P20 energy weighting of relative state error. Chapter 07 "Predicted trajectories" (A).

    ``R = sum_t ||h^_t - h_t||^2 / sum_t ||h_t||^2 = sum_t w_t e_t`` with
    ``w_t = ||h_t||^2 / sum ||h_u||^2``, ``e_t`` the row relative error; zero target rows keep
    their squared error as a separate term over the total target energy. Rows are axis 0.
    """
    p, t = f64(predicted), f64(target)
    err = ((p - t) ** 2).reshape(p.shape[0], -1).sum(-1)
    energy = (t**2).reshape(t.shape[0], -1).sum(-1)
    total_energy = energy.sum()
    if float(total_energy) == 0.0:
        raise ValueError("relative error is undefined when every target row is zero")
    nz = energy > 0
    terms = {"weighted_rows": ((energy[nz] / total_energy) * (err[nz] / energy[nz])).sum(),
             "zero_rows": err[~nz].sum() / total_energy}
    return IdentityResult("P20", "energy_weighted_error", "A", EXACT, terms,
                          sum(terms.values()), reference=err.sum() / total_energy)


# --------------------------------------------------------------------------- P27 second order


def swiglu_second_order(g: Any, u: Any, g_a: Any, g_b: Any, u_a: Any, u_b: Any, g_ab: Any,
                        u_ab: Any) -> IdentityResult:
    """P27 mixed second derivative of a SwiGLU product. Chapter 09 section 7 (D).

    ``f_ab = u S''(g) g_a g_b + S'(g)(g_a u_b + g_b u_a) + u S'(g) g_ab + S(g) u_ab``. The last
    two terms vanish only for affine gate/up trajectories; upstream normalization and attention
    can make them nonzero. Arguments are the trajectory derivatives at the evaluation point
    (supplied operands). Second-order object (``derived-local``).
    """
    g, u, ga, gb, ua, ub, gab, uab = map(f64, (g, u, g_a, g_b, u_a, u_b, g_ab, u_ab))
    terms = {
        "gate_curvature": u * silu_d2(g) * ga * gb,
        "mixed_gate_up": silu_d1(g) * (ga * ub + gb * ua),
        "gate_trajectory": u * silu_d1(g) * gab,
        "up_trajectory": silu(g) * uab,
    }
    return IdentityResult("P27", "swiglu_second_order", "D", LOCAL, terms, sum(terms.values()))


def consumer_second_order(grad_h: Any, hess_h: Any, f_a: Any, f_b: Any,
                          f_ab: Any) -> IdentityResult:
    """P27 consumer curvature. Chapter 09 section 7 (D).

    ``M_ab = grad H^T f_ab + f_a^T Hess H f_b`` for ``M = H(f)`` with ``f`` the complete varying
    consumer input. Projecting the local Hessian through the gradient alone omits the second
    term.
    """
    gh, hh, fa, fb, fab = map(f64, (grad_h, hess_h, f_a, f_b, f_ab))
    terms = {"gradient_term": gh @ fab, "consumer_curvature": fa @ hh @ fb}
    return IdentityResult("P27", "consumer_second_order", "D", LOCAL, terms, sum(terms.values()))


# --------------------------------------------------------------------------- P28 path integral


def path_integral(grad_fn: Callable[[Any], Any], x0: Any, x1: Any, k: int, *,
                  endpoint_fn: Callable[[Any], Any] | None = None) -> IdentityResult:
    """P28 path-gradient completeness. Chapter 09 section 8 (F formal identity; QUADRATURE).

    ``M(x_1) - M(x_0) = int_0^1 grad M(x_0 + tau xi)^T xi d tau``, ``xi = x_1 - x_0``. ``total`` is
    the ``k``-point Gauss-Legendre APPROXIMATION of the integral; with ``endpoint_fn`` the
    ``reference`` is the endpoint difference and closure is the quadrature error (it is not a
    guarantee of the formal identity's completeness for any chosen rule). Smooth reference only:
    rounded native execution is discontinuous.
    """
    x0, x1 = f64(x0), f64(x1)
    xi = x1 - x0
    tau, w = gauss_legendre(k)
    terms = {f"node{i}": wi * (f64(grad_fn(x0 + ti * xi)) * xi).sum()
             for i, (ti, wi) in enumerate(zip(tau, w))}
    reference = None
    if endpoint_fn is not None:
        reference = f64(endpoint_fn(x1)) - f64(endpoint_fn(x0))
    return IdentityResult("P28", "path_integral", "F", LOCAL, terms, sum(terms.values()),
                          reference=reference,
                          approximation={"rule": "gauss-legendre", "k": int(k)},
                          notes=(f"{k}-point quadrature approximation of the formal integral",))


def parseval_split(grad: Any, xi: Any, *, axis: int = 0) -> IdentityResult:
    """P28 Parseval form of the path integrand at one point. Chapter 09 section 8 (A).

    ``grad M^T xi = sum_k N^-1 Re[conj(G_k) Xi_k]`` with the DFT along the token ``axis``; terms
    are per frequency ``k`` summed over the other axes.
    """
    torch = _torch()
    gr, x = f64(grad), f64(xi)
    n = gr.shape[axis]
    gk, xk = torch.fft.fft(gr, dim=axis), torch.fft.fft(x, dim=axis)
    per = (gk.conj() * xk).real / n
    per = per.movedim(axis, 0).reshape(n, -1).sum(-1)
    terms = {f"k{i}": per[i] for i in range(n)}
    return IdentityResult("P28", "parseval_split", "A", EXACT, terms, per.sum(),
                          reference=(gr * x).sum())


def rectangle(fn: Callable[[Any], Any], x: Any, a: Any, b: Any, *,
              hess_fn: Callable[[Any], Any] | None = None, k: int = 8) -> IdentityResult:
    """P28 mixed-Hessian rectangle. Chapter 09 section 9 (F under T premises).

    ``M(x+a+b) - M(x+a) - M(x+b) + M(x) = int int a^T Hess M(x + s a + t b) b ds dt`` for fixed
    additive directions and a C^2 margin. ``total`` is the four-corner value (exact arithmetic on
    four evaluations); with ``hess_fn`` the ``reference`` is the ``k x k`` Gauss-Legendre
    APPROXIMATION of the double integral, so closure is a quadrature error: the four-corner
    total stays ``derived-exact`` while the closure numbers are ``derived-local`` and the summary
    carries a structured ``approximation``.
    """
    x, va, vb = f64(x), f64(a), f64(b)
    corners = {"ab": f64(fn(x + va + vb)), "a": -f64(fn(x + va)), "b": -f64(fn(x + vb)),
               "base": f64(fn(x))}
    reference = None
    if hess_fn is not None:
        tau, w = gauss_legendre(k)
        reference = sum(wi * wj * (va @ f64(hess_fn(x + si * va + tj * vb)) @ vb)
                        for si, wi in zip(tau, w) for tj, wj in zip(tau, w))
    return IdentityResult("P28", "rectangle", "F", EXACT, corners,
                          sum(corners.values()), reference=reference,
                          approximation=None if hess_fn is None else {"rule": "gauss-legendre",
                                                                      "k": int(k)},
                          assumptions=("fixed additive directions", "C^2 margin on the rectangle"))


# --------------------------------------------------------------------------- P29 coalitions


def _subset_key(key: Any) -> frozenset:
    if isinstance(key, frozenset):
        return key
    if isinstance(key, str):
        return frozenset([key]) if key else frozenset()
    return frozenset(key)


def mobius(outcomes: Mapping[Any, Any]) -> dict[frozenset, Any]:
    """P29 Boolean-lattice Moebius inversion. Chapter 09 section 9 (A).

    ``I(T) = sum_{U subset T} (-1)^{|T|-|U|} F(U)``. ``outcomes`` maps a subset (frozenset, tuple,
    or set of edit names) to a measured outcome. ``I(T)`` is returned only when every subset of
    ``T`` was measured: an unmeasured corner gets no inferred coefficient.
    """
    table = {_subset_key(k): v for k, v in outcomes.items()}
    out = {}
    for t in table:
        subsets = [frozenset(c) for r in range(len(t) + 1) for c in itertools.combinations(t, r)]
        if all(s in table for s in subsets):
            out[t] = sum((-1) ** (len(t) - len(s)) * table[s] for s in subsets)
    return out


def mobius_inverse(interactions: Mapping[Any, Any]) -> dict[frozenset, Any]:
    """P29 ``F(S) = sum_{T subset S} I(T)`` for every ``S`` whose subsets all have coefficients."""
    table = {_subset_key(k): v for k, v in interactions.items()}
    out = {}
    for s in table:
        subsets = [frozenset(c) for r in range(len(s) + 1) for c in itertools.combinations(s, r)]
        if all(t in table for t in subsets):
            out[s] = sum(table[t] for t in subsets)
    return out


def walsh_hadamard(outcomes: Mapping[Any, Any],
                   items: Sequence[Any] | None = None) -> dict[frozenset, Any]:
    """P29 Walsh-Hadamard coefficients. Chapter 09 section 9 (A).

    ``F^(T) = 2^-n sum_b F(b) chi_T(b)``, ``chi_T(b) = (-1)^{sum_{i in T} b_i}``, on the COMPLETE
    cube over ``items`` (default: union of the keys); a missing corner is refused.
    """
    table = {_subset_key(k): v for k, v in outcomes.items()}
    universe = tuple(sorted(set().union(*table) if items is None else items, key=repr))
    cube = [frozenset(c) for r in range(len(universe) + 1)
            for c in itertools.combinations(universe, r)]
    missing = [s for s in cube if s not in table]
    if missing:
        names = sorted(map(sorted, missing))
        raise ValueError(f"Walsh-Hadamard needs the complete cube; missing {names}")
    scale = 2.0 ** -len(universe)
    return {t: scale * sum(table[b] * (-1) ** len(t & b) for b in cube) for t in cube}


def walsh_hadamard_inverse(coefficients: Mapping[Any, Any]) -> dict[frozenset, Any]:
    """P29 ``F(b) = sum_T F^(T) chi_T(b)`` on the complete cube of the coefficient keys."""
    table = {_subset_key(k): v for k, v in coefficients.items()}
    universe = tuple(sorted(set().union(*table), key=repr))
    cube = [frozenset(c) for r in range(len(universe) + 1)
            for c in itertools.combinations(universe, r)]
    return {b: sum(table.get(t, 0.0) * (-1) ** len(t & b) for t in cube) for b in cube}


# --------------------------------------------------------------------------- catalogue


def _entry(fn: Callable, kind: str, ports: Sequence[str], authority: str,
           section: str, assumptions: Sequence[str] = (),
           closure_authority: str | None = None) -> dict[str, Any]:
    """Catalogue entry. ``authority`` is that of the result's own numbers;
    ``closure_authority`` (when it differs) is that of its closure."""
    entry = {"function": fn.__name__, "callable": fn, "kind": kind, "ports": tuple(ports),
             "authority": authority, "section": section, "assumptions": tuple(assumptions)}
    if closure_authority is not None:
        entry["closure_authority"] = closure_authority
    return entry


# (proof ID, title as in obsidian/model-math/proof-index.md, kind, implemented entries). An empty
# entry tuple is a catalogued proof that this module does not implement (status "not implemented").
_CATALOGUE: tuple[tuple[str, str, str, tuple[dict[str, Any], ...]], ...] = (
    ("P01", "Token lookup and complete forward expression DAG", "A", (
        _entry(embedding_lookup, "A", ("h",), EXACT, "01 Token lookup"),
        _entry(residual_add, "A", ("h", "o", "r", "m", "h_out"), EXACT,
               "04 Transport and write; 05 Two arguments"),
        _entry(transport, "A", ("p", "values", "A"), EXACT, "04 Transport and write"),
        _entry(head_write, "A", ("A", "o"), EXACT, "04 Transport and write"),
    )),
    ("P02", "RMS denominator, full Jacobian, radial response", "A/D", (
        _entry(rms_norm, "A", ("h", "a", "r", "n", "q", "qn", "k", "kn"), EXACT,
               "02 Scale and gain"),
        _entry(rms_jacobian, "D", ("h", "r"), LOCAL, "02 Local coupling"),
    )),
    ("P03", "Finite RMS change, alignment and edit energy", "F", (
        _entry(rms_finite_change, "F", ("h", "a", "r", "n"), EXACT,
               "02 Finite change; 09 section 2", ("same gain", "positive denominators")),
    )),
    ("P04", "Affine Q/K/V, GQA, positional pair expansion", "A", (
        _entry(affine, "A", ("a", "q", "k", "v"), EXACT, "03 Affine reads"),
        _entry(gqa_kappa, "A", (), EXACT, "03 Affine reads"),
        _entry(head_route_average_drop, "A", ("p", "values", "o"), EXACT, "03 Affine reads"),
        _entry(rope_rotate, "A", ("q", "qn", "qr", "k", "kn", "kr"), EXACT,
               "03 Position pairs", ("default RoPE",)),
        _entry(rope_relative, "A", (), EXACT, "03 Position pairs"),
        _entry(rope_score_pairs, "A", ("q", "qn", "k", "kn", "s"), EXACT, "03 Position pairs"),
    )),
    ("P05", "Softmax contrasts and query invariance at fixed keys/visibility", "A/T", (
        _entry(scores, "A", ("qr", "kr", "s"), EXACT, "04 Scores and competition"),
        _entry(softmax, "A", ("s", "p"), EXACT, "04 Scores and competition"),
        _entry(softmax_contrasts, "A", ("qr", "kr", "p"), EXACT, "04 Contrast and sensitivity",
               ("fixed keys", "fixed visible set")),
        _entry(anchored_transport, "A", ("p", "values"), EXACT, "04 Contrast and sensitivity",
               ("sum p = 1",)),
        _entry(query_shift_invariance, "T", ("qr", "kr"), EXACT,
               "04 Finite source edits; 09 section 3",
               ("fixed keys", "fixed visible set", "edit orthogonal to key contrasts")),
    )),
    ("P06", "Full attention differential, centered values, source adjoint", "D", (
        _entry(attention_differential, "D", ("qr", "kr", "values", "A"), LOCAL,
               "04 Contrast and sensitivity; 09 section 4",
               ("fixed visible set", "ports varied independently")),
    )),
    ("P07", "Finite score and allocation/payload/interaction splits", "A/F", (
        _entry(score_finite_change, "F", ("qr", "kr", "s"), EXACT,
               "04 Finite source edits; 09 section 3", ("matched receiver and visible set",)),
        _entry(transport_finite_split, "F", ("p", "values", "A"), EXACT, "09 section 4",
               ("matched visible set",)),
    )),
    ("P08", "Exact finite softmax reweighting and one-source closed form", "F", (
        _entry(softmax_reweighting, "F", ("s", "p"), EXACT, "04; 09 section 3",
               ("same visible set", "finite logits")),
        _entry(one_source_response, "F", ("s", "p", "values", "A"), EXACT, "09 section 5; 04",
               ("fixed receiver query", "fixed visible set", "one changed source")),
    )),
    ("P09", "Zero-contribution cut versus mask-and-renormalize", "A", (
        _entry(cut_sources, "A", ("p", "values", "A"), EXACT, "04 Cuts and image gates",
               ("mu < 1 for the mask form",)),
    )),
    ("P10", "Full SwiGLU and state-conditioned operator factorization", "A", (
        _entry(swiglu_forward, "A", ("n", "g", "sg", "u", "f", "m"), EXACT,
               "05 Two arguments and a write"),
        _entry(affine, "A", ("n", "g", "u", "f", "m"), EXACT, "05 Two arguments and a write"),
        _entry(swiglu_gate, "A", ("g", "sg"), EXACT, "05 Two arguments and a write"),
        _entry(swiglu_product, "A", ("sg", "u", "f"), EXACT, "05 Two arguments and a write"),
        _entry(mlp_state_operator, "A", ("r", "m"), EXACT, "05 Quadratic geometry"),
    )),
    ("P11", "Rank-two quadratic form and its two eigenvalues", "A", (
        _entry(neuron_quadratic_form, "A", ("n", "f"), EXACT, "05 Quadratic geometry"),
    )),
    ("P12", "Normalized-neuron read plane, gradient, Hessian", "D/T", (
        _entry(neuron_read_plane, "D", ("n",), LOCAL, "05 Read plane and derivatives"),
    )),
    ("P13", "Raw-residual Hessian: isotropic term plus rank-at-most-three correction", "D", (
        _entry(neuron_raw_hessian, "D", ("r",), LOCAL, "05 Read plane and derivatives"),
    )),
    ("P14", "Complete MLP Jacobian, residual skips, and consumer pullback", "D", (
        _entry(mlp_jacobian, "D", ("r", "m"), LOCAL, "05 Whole-block response"),
        _entry(mlp_pullback, "D", ("r", "h_out"), LOCAL, "05 Whole-block response; 07"),
    )),
    ("P15", "Finite gate/up decomposition and source-component products", "A/F", (
        _entry(swiglu_finite_split, "F", ("g", "u", "f"), EXACT,
               "05 Finite products; 09 section 6"),
        _entry(integral_mean_slope, "F", ("g", "sg"), LOCAL, "09 section 6",
               ("quadrature approximation with k nodes",)),
        _entry(swiglu_source_pairs, "A", ("n", "f"), EXACT, "05 Finite products and source pairs",
               ("complete decomposition of n",)),
        _entry(down_finite_split, "F", ("f", "m"), EXACT, "09 section 6"),
    )),
    ("P19", "Native readout, decision cells, competitor-margin bounds", "A/T", (
        _entry(readout_logits, "A", ("h_out", "logits"), EXACT, "07 Final readout"),
        _entry(readout_margin, "A", ("h", "o", "m", "h_out", "fn", "logits"), EXACT,
               "07 Final readout; 02 Scale and gain",
               ("final-norm scale evaluated at the full final state",)),
        _entry(winner_certificate, "T", ("h_out",), BOUND,
               "07 Equal decisions and unequal states",
               ("bias-free head", "unique winner", "real arithmetic")),
        _entry(margin_error_bound, "T", ("fn",), BOUND, "07 Equal decisions and unequal states"),
    )),
    ("P20", "Recursive prediction error: propagated input error plus local mismatch", "A", (
        _entry(prediction_error_split, "A", (), EXACT, "07 Predicted trajectories"),
        _entry(energy_weighted_error, "A", (), EXACT, "07 Predicted trajectories"),
    )),
    ("P26", "RoPE score kernel's two-frequency transform and off-grid caveat", "A", ()),
    ("P27", "Mixed gate/up derivatives, upstream terms, downstream curvature", "D", (
        _entry(swiglu_second_order, "D", ("g", "u"), LOCAL, "09 section 7"),
        _entry(consumer_second_order, "D", ("f",), LOCAL, "09 section 7"),
    )),
    ("P28", "Path-gradient/Parseval completeness and mixed-Hessian rectangle", "F/T", (
        _entry(path_integral, "F", (), LOCAL, "09 section 8", ("smooth path; quadrature k",)),
        _entry(parseval_split, "A", (), EXACT, "09 section 8"),
        _entry(rectangle, "F", (), EXACT, "09 section 9",
               ("fixed additive directions", "C^2 margin"),
               closure_authority=LOCAL),  # closure is derived-local when the reference is a quadrature
    )),
    ("P29", "Boolean Moebius inversion and Walsh-Hadamard representation", "A", (
        _entry(mobius, "A", (), EXACT, "09 section 9"),
        _entry(mobius_inverse, "A", (), EXACT, "09 section 9"),
        _entry(walsh_hadamard, "A", (), EXACT, "09 section 9", ("complete cube",)),
        _entry(walsh_hadamard_inverse, "A", (), EXACT, "09 section 9", ("complete cube",)),
    )),
)

IDENTITIES: dict[str, dict[str, Any]] = {
    proof: {
        "title": title,
        "kind": kind,
        "function": entries[0]["function"] if entries else None,
        "status": "implemented" if entries else "not implemented",
        "functions": {e["function"]: e for e in entries},
        "ports": tuple(sorted({p for e in entries for p in e["ports"]})),
        "assumptions": tuple(dict.fromkeys(a for e in entries for a in e["assumptions"])),
        "authority": tuple(dict.fromkeys(e["authority"] for e in entries)),
    }
    for proof, title, kind, entries in _CATALOGUE
}


def list_identities() -> list[dict[str, Any]]:
    """JSON-safe listing of ``IDENTITIES`` (no callables, no torch)."""
    return [
        {"proof": proof, "title": v["title"], "kind": v["kind"], "status": v["status"],
         "functions": [{k: (list(x) if isinstance(x, tuple) else x) for k, x in e.items()
                        if k != "callable"} for e in v["functions"].values()]}
        for proof, v in IDENTITIES.items()
    ]


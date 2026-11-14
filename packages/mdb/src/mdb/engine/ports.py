"""Port programs for native pre-norm SwiGLU decoder layers (Qwen2, Qwen3, Llama, Mistral) and
the controller that drives them.

A port is a named value produced inside one native decoder-layer call. The controller
captures ports with forward hooks plus an intercept of the family's module-level
``eager_attention_forward``, applies staged edits by returning replacement tensors, and
raises a private ``PortStop`` at a target port. It never re-implements the layer: every
value comes from the native modules, except the two ports marked ``replica`` (``s``, and the
value mix after an ``s``/``p`` edit), which are replicas checked against native output on
every complete pass.

Authority labels used by this package (fixed vocabulary):
``measured | derived-exact | derived-local | bound | supplied``.
"""

from __future__ import annotations

import hashlib
import importlib
import threading
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from saturn_pub.core import PortStop

ATTN = "attn"
MLP = "mlp"
EDIT_OPS = ("replace", "add", "scale")

_D = ("batch", "row", "d_model")
_QF = ("batch", "row", "head*head_dim")
_KF = ("batch", "row", "kv_head*head_dim")
_FF = ("batch", "row", "d_ff")
_SCORE = ("batch", "head", "row", "src")

CACHE = "cache"
PREV_LAYER = "prev_layer"


@dataclass(frozen=True)
class PortSpec:
    """One port of a layer program: where it is captured and how an edit there is realised.

    ``inputs`` are the producer edges: names of earlier ports of the same layer, or
    ``"cache"`` (the key/value cache rows of earlier tokens) or ``"prev_layer"`` (the previous
    layer's committed output).
    """

    name: str
    sublayer: str
    site: str
    hook: str
    kind: str
    axes: tuple[str, ...]
    operator: str
    proofs: tuple[str, ...]
    edit: str
    editable: bool = True
    inputs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sublayer": self.sublayer,
            "site": self.site,
            "hook": self.hook,
            "kind": self.kind,
            "axes": list(self.axes),
            "operator": self.operator,
            "proofs": list(self.proofs),
            "edit": self.edit,
            "editable": self.editable,
            "inputs": list(self.inputs),
        }


def model_dims(config: Any) -> dict[str, int]:
    """Axis sizes of one decoder layer from a Hugging Face config (``src`` is added per call)."""
    heads = int(config.num_attention_heads)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    return {
        "batch": 1,
        "row": 1,
        "d_model": int(config.hidden_size),
        "head": heads,
        "kv_head": int(config.num_key_value_heads),
        "head_dim": int(head_dim),
        "d_ff": int(config.intermediate_size),
    }


@dataclass(frozen=True)
class PortProgram:
    """Ordered ports of one decoder layer, in native execution order."""

    family: str
    modeling: str
    ports: tuple[PortSpec, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(port.name for port in self.ports)

    def spec(self, name: str) -> PortSpec:
        for port in self.ports:
            if port.name == name:
                return port
        raise ValueError(f"{self.family} has no port {name!r}; ports: {', '.join(self.names)}")

    def index(self, name: str) -> int:
        return self.names.index(self.spec(name).name)

    def shape(self, name: str, config: Any, *, src: int) -> tuple[int, ...]:
        """Shape of a port for one decoder-step transition, from its axes and the model config.

        ``src`` is the number of source positions attention sees (tokens in the state).
        """
        dims = {**model_dims(config), "src": int(src)}
        out = []
        for axis in self.spec(name).axes:
            size = 1
            for part in axis.split("*"):
                size *= dims[part]
            out.append(size)
        return tuple(out)

    def modeling_module(self) -> Any:
        return importlib.import_module(self.modeling)

    def sites(self) -> tuple[str, ...]:
        return tuple(sorted({p.site for p in self.ports if p.hook in {"pre", "out"} and p.site}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mdb-port-program-v1",
            "family": self.family,
            "modeling": self.modeling,
            "ports": [port.to_dict() for port in self.ports],
        }


_H_EDIT = (
    "layer pre-hook replaces the layer input; the native layer binds residual to it, so the "
    "skip path and the attention branch both see the edit (the same computation as writing "
    "the boundary hidden for this one transition)"
)
_KR_REFUSAL = (
    "kr holds cached rows and the native layer writes the cache before the eager call, so an "
    "edit here could not reach the stored keys; edit k (or kn on Qwen3) to change the key row "
    "this transition writes"
)


def _proj(
    name: str,
    site: str,
    axes: tuple[str, ...],
    operator: str,
    inputs: tuple[str, ...],
    extra: str = "",
) -> PortSpec:
    return PortSpec(
        name,
        ATTN,
        site,
        "out",
        "native",
        axes,
        operator,
        ("P04",),
        f"{site} output replaced natively{extra}",
        inputs=inputs,
    )


@dataclass(frozen=True)
class Family:
    """One row of the family table. Every registered family runs the same native block shape:
    ``h -> input_layernorm (RMSNorm) -> q/k/v_proj [-> q_norm/k_norm] -> RoPE -> eager attention
    -> o_proj -> + h -> post_attention_layernorm (RMSNorm) -> act_fn(gate_proj) * up_proj ->
    down_proj -> + r``, with the same module names, so one port program serves all of them. The
    columns are what differs.

    ``qk_norm``: per-head RMSNorm on q and k before RoPE (ports ``qn``/``kn``). ``qkv_bias``:
    the q/k/v projections always carry a bias (``"always"``), or carry one only under
    ``config.attention_bias`` (``"config"``); the analysis reads ``Linear.bias`` of the loaded
    module either way, so an absent bias is an absent (exactly zero) term, never a guess.
    ``premises``: ``(config attribute, allowed values, why)``; a loaded config outside them is
    refused at attach (``config_refusal``) instead of being stepped under a wrong operator text.
    """

    name: str
    qk_norm: bool = False
    qkv_bias: str = "config"
    premises: tuple[tuple[str, tuple[Any, ...], str], ...] = ()


_SILU = ("hidden_act", ("silu",), "the MLP port program is sg = silu(g)")
_NO_MLP_BIAS = ("mlp_bias", (False, None),
                "the MLP port program (g = W_gate n, u = W_up n, m = W_down f) is bias-free")

FAMILIES: dict[str, Family] = {
    row.name: row
    for row in (
        Family("qwen2", qkv_bias="always", premises=(_SILU,)),
        Family("qwen3", qk_norm=True, premises=(_SILU,)),
        Family("llama", premises=(_SILU, _NO_MLP_BIAS)),
        Family("mistral", premises=(_SILU,)),
    )
}


def config_refusal(family: str, config: Any) -> str | None:
    """Why a loaded ``config`` falls outside its family row's premises (None when it does not)."""
    row = FAMILIES.get(family)
    if row is None:
        return f"no port program for model_type {family!r}; supported: {', '.join(FAMILIES)}"
    for name, allowed, why in row.premises:
        value = getattr(config, name, allowed[0])
        if value not in allowed:
            return (f"{family} config.{name} = {value!r} is outside the port table "
                    f"({name} must be one of {list(allowed)}: {why})")
    return None


def _decoder_program(row: Family) -> PortProgram:
    family, qk_norm = row.name, row.qk_norm
    bias = {"always": " + b_{}", "config": " (+ b_{} when attention_bias)"}[row.qkv_bias]
    head = ("batch", "row", "head", "head_dim")
    kv_head = ("batch", "row", "kv_head", "head_dim")
    q_src, k_src = ("qn", "kn") if qk_norm else ("q", "k")
    attention: list[PortSpec] = [
        PortSpec(
            "h",
            ATTN,
            "",
            "pre",
            "native",
            _D,
            "h = layer input (residual stream)",
            ("P01",),
            _H_EDIT,
            inputs=(PREV_LAYER,),
        ),
        PortSpec(
            "a",
            ATTN,
            "input_layernorm",
            "out",
            "native",
            _D,
            "a = RMSNorm(h) * w_in",
            ("P02", "P03"),
            "input_layernorm output replaced; reaches q/k/v only, the skip path keeps h",
            inputs=("h",),
        ),
        _proj(
            "q", "self_attn.q_proj", _QF, f"q = W_q a{bias.format('q')} (pre-RoPE, flat)", ("a",)
        ),
    ]
    if qk_norm:
        attention.append(
            PortSpec(
                "qn",
                ATTN,
                "self_attn.q_norm",
                "out",
                "native",
                head,
                "qn = RMSNorm_head(q) * w_q (per head, pre-RoPE)",
                ("P02", "P04"),
                "q_norm output replaced natively",
                inputs=("q",),
            )
        )
    attention.append(
        _proj(
            "k",
            "self_attn.k_proj",
            _KF,
            f"k = W_k a{bias.format('k')} (pre-RoPE, flat)",
            ("a",),
            "; flows through RoPE into the cache row this transition writes",
        )
    )
    if qk_norm:
        attention.append(
            PortSpec(
                "kn",
                ATTN,
                "self_attn.k_norm",
                "out",
                "native",
                kv_head,
                "kn = RMSNorm_head(k) * w_k (per kv head, pre-RoPE)",
                ("P02", "P04"),
                "k_norm output replaced natively; enters the cache via RoPE",
                inputs=("k",),
            )
        )
    attention += [
        _proj(
            "v",
            "self_attn.v_proj",
            _KF,
            f"v = W_v a{bias.format('v')} (flat)",
            ("a",),
            "; enters the cache row this transition writes",
        ),
        PortSpec(
            "qr",
            ATTN,
            "eager",
            "eager-arg",
            "native",
            ("batch", "head", "row", "head_dim"),
            f"qr = RoPE({q_src}) (query argument of eager attention)",
            ("P04",),
            "query argument of the eager call replaced; native eager attention recomputes "
            "s, p and the value mix",
            inputs=(q_src,),
        ),
        PortSpec(
            "kr",
            ATTN,
            "eager",
            "eager-arg",
            "native",
            ("batch", "kv_head", "src", "head_dim"),
            f"kr = [{CACHE}; RoPE({k_src})] (key argument of eager attention, cached rows "
            "included)",
            ("P04",),
            _KR_REFUSAL,
            editable=False,
            inputs=(k_src, CACHE),
        ),
        PortSpec(
            "s",
            ATTN,
            "eager",
            "eager-replica",
            "replica",
            _SCORE,
            "s = scaling * qr repeat_kv(kr)^T + mask (replica of the native expression)",
            ("P05", "P07", "P08", "P09"),
            "replica: p' = softmax(s', float32).to(dtype) and the value mix "
            "matmul(p', repeat_kv(v)) are recomputed by replica",
            inputs=("qr", "kr"),
        ),
        PortSpec(
            "p",
            ATTN,
            "eager",
            "eager-out",
            "native",
            _SCORE,
            "p = softmax(s, float32).to(dtype) (native attention weights)",
            ("P05", "P06", "P07", "P08", "P09"),
            "native weights replaced after the eager call; the value mix "
            "matmul(p', repeat_kv(v)) is recomputed by replica",
            inputs=("s",),
        ),
        PortSpec(
            "A",
            ATTN,
            "self_attn.o_proj",
            "pre",
            "native",
            _QF,
            f"A = concat_heads(p repeat_kv([{CACHE}; v])) (o_proj input)",
            ("P06", "P07"),
            "o_proj input replaced natively",
            inputs=("p", "v", CACHE),
        ),
        PortSpec(
            "o",
            ATTN,
            "self_attn.o_proj",
            "out",
            "native",
            _D,
            "o = W_o A",
            ("P06", "P07"),
            "o_proj output replaced natively; reaches r, hence both the skip path and the MLP",
            inputs=("A",),
        ),
        PortSpec(
            "r",
            ATTN,
            "post_attention_layernorm",
            "pre",
            "native",
            _D,
            "r = h + o",
            ("P06", "P07", "P14"),
            "realised as an o edit: o' = op(h + o) - h, so the native sum h + o' feeds both "
            "the skip path and the MLP; the realised r is compared with the request and the "
            "max abs error is recorded (floating-point sums need not hit it exactly)",
            inputs=("h", "o"),
        ),
    ]
    mlp = [
        PortSpec(
            "n",
            MLP,
            "mlp",
            "pre",
            "native",
            _D,
            "n = RMSNorm(r) * w_post (mlp input)",
            ("P02", "P03"),
            "mlp input replaced; reaches the MLP only, the skip path keeps r",
            inputs=("r",),
        ),
        PortSpec(
            "g",
            MLP,
            "mlp.gate_proj",
            "out",
            "native",
            _FF,
            "g = W_gate n",
            ("P10", "P15"),
            "gate_proj output replaced natively",
            inputs=("n",),
        ),
        PortSpec(
            "sg",
            MLP,
            "mlp.act_fn",
            "out",
            "native",
            _FF,
            "sg = silu(g)",
            ("P10", "P15"),
            "act_fn output replaced natively",
            inputs=("g",),
        ),
        PortSpec(
            "u",
            MLP,
            "mlp.up_proj",
            "out",
            "native",
            _FF,
            "u = W_up n",
            ("P10", "P15"),
            "up_proj output replaced natively",
            inputs=("n",),
        ),
        PortSpec(
            "f",
            MLP,
            "mlp.down_proj",
            "pre",
            "native",
            _FF,
            "f = sg * u (down_proj input)",
            ("P10", "P15"),
            "down_proj input replaced natively",
            inputs=("sg", "u"),
        ),
        PortSpec(
            "m",
            MLP,
            "mlp.down_proj",
            "out",
            "native",
            _D,
            "m = W_down f",
            ("P14",),
            "down_proj output replaced natively; the skip path keeps r",
            inputs=("f",),
        ),
        PortSpec(
            "h_out",
            MLP,
            "",
            "out",
            "native",
            _D,
            "h_out = r + m (layer output)",
            ("P14",),
            "layer output replaced; it becomes the committed hidden",
            inputs=("r", "m"),
        ),
    ]
    return PortProgram(
        family, f"transformers.models.{family}.modeling_{family}", tuple(attention + mlp)
    )


PROGRAMS: dict[str, PortProgram] = {name: _decoder_program(row) for name, row in FAMILIES.items()}


def program_for(adapter_or_family: Any) -> PortProgram:
    """The port program of a family name, or of an adapter (whose loaded config must then lie
    inside the family row's premises)."""
    family = adapter_or_family
    if not isinstance(family, str):
        family = adapter_or_family.spec.model_type
    try:
        program = PROGRAMS[family]
    except KeyError:
        raise ValueError(
            f"no port program for {family!r}; registered: {', '.join(sorted(PROGRAMS))}"
        ) from None
    config = getattr(getattr(adapter_or_family, "model", None), "config", None)
    refusal = None if config is None else config_refusal(family, config)
    if refusal is not None:
        raise ValueError(refusal)
    return program


def check_deterministic(module: Any) -> None:
    """Refuse a model or layer that would run in training mode (dropout applies there).

    The one implementation of the eval-mode rule: ``PortController`` installs call it on the
    pending layer, ``Run`` on the whole model before construction, every commit and every probe.
    """
    training = [name or type(module).__name__ for name, m in module.named_modules() if m.training]
    if not training:
        return
    dropout = sorted(
        {
            float(p)
            for m in module.modules()
            if m.training
            for p in (getattr(m, "attention_dropout", 0.0), getattr(m, "p", 0.0))
            if type(p) in (int, float) and p
        }
    )
    raise ValueError(
        f"port stepping requires an eval-mode model (no dropout): {len(training)} module(s) are "
        f"in training mode (e.g. {training[0]!r}); non-zero dropout probabilities seen: "
        f"{dropout or 'none'}; call model.eval()"
    )


def tensor_digest(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy()).hexdigest()  # hashes the buffer, not a bytes copy


def changed_fraction(before: torch.Tensor, after: torch.Tensor) -> float:
    """Fraction of elements whose bit pattern differs (same shape and dtype)."""
    if before.numel() == 0:
        return 0.0
    width = before.element_size()
    left = before.detach().cpu().contiguous().reshape(-1).view(torch.uint8).reshape(-1, width)
    right = after.detach().cpu().contiguous().reshape(-1).view(torch.uint8).reshape(-1, width)
    return float((left != right).any(dim=1).double().mean())


MEASURED_FIELDS = (
    "cast_max_abs_error",
    "realised_max_abs_error",
    "realisation_max_abs_error",
    "changed_fraction",
    "row_sum_min",
    "row_sum_max",
    "min",
)
SUPPLIED_FIELDS = ("row_sum_tolerance",)


def with_labels(entry: dict[str, Any]) -> dict[str, Any]:
    """Attach ``labels`` ``{field: authority}`` for the numeric quantities of a manifest entry
    (measurements are ``measured``, the row-sum tolerance is ``supplied``). Counts and shapes
    (``mask_count``, ``mask_shape``) are structure and stay unlabelled."""
    labels = {key: "measured" for key in MEASURED_FIELDS if key in entry}
    labels.update({key: "supplied" for key in SUPPLIED_FIELDS if key in entry})
    entry["labels"] = labels
    return entry


def max_abs(value: torch.Tensor) -> float:
    return float(value.max()) if value.numel() else 0.0


def wide(value: torch.Tensor) -> torch.Tensor:
    """``value`` in float64 on its own device (on the host for a device without float64)."""
    if value.device.type == "mps":
        return value.detach().to("cpu", torch.float64)
    return value.detach().to(torch.float64)


@dataclass(frozen=True)
class PortEdit:
    """A staged edit: ``replace``, ``add`` or ``scale`` the value at one port.

    ``mask`` (bool, broadcastable to the port) limits the op to the elements where it is true;
    the other elements stay bit-identical. ``selection`` is an opaque description of what the
    mask selects, recorded verbatim. ``origin`` is the pointer of a value taken from another run
    (``{origin, via, address, sha256, ...}``, names and digests), recorded verbatim.
    """

    port: str
    value: Any
    op: str = "replace"
    mask: Any = None
    selection: str | None = None
    origin: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.op not in EDIT_OPS:
            raise ValueError(f"edit op must be one of {', '.join(EDIT_OPS)}")
        value = self.value
        if isinstance(value, torch.Tensor):
            if not value.is_floating_point():
                raise ValueError("edit tensors must be floating point")
            object.__setattr__(self, "value", value.detach().clone())
        elif type(value) in (int, float):
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError("edit scalars must be finite")
            object.__setattr__(self, "value", float(value))
        else:
            raise ValueError("edit value must be a tensor or a finite number")
        if self.mask is not None:
            if not isinstance(self.mask, torch.Tensor) or self.mask.dtype != torch.bool:
                raise ValueError("edit mask must be a bool tensor")
            object.__setattr__(self, "mask", self.mask.detach().to("cpu", copy=True))
        if self.selection is not None and not isinstance(self.selection, str):
            raise ValueError("edit selection must be a string")

    def validate(self, shape: Sequence[int]) -> None:
        """Shape-only validation against a port shape; executes nothing."""
        shape = tuple(shape)
        value = self.value
        if isinstance(value, torch.Tensor):
            if self.op == "replace":
                if tuple(value.shape) != shape:
                    raise ValueError(
                        f"replace at port {self.port} needs shape {shape}, "
                        f"got {tuple(value.shape)}"
                    )
            else:
                try:
                    joint = tuple(torch.broadcast_shapes(tuple(value.shape), shape))
                except RuntimeError:
                    joint = None
                if joint != shape:
                    raise ValueError(
                        f"{self.op} value at port {self.port} does not broadcast to the port: "
                        f"port shape {shape}, value shape {tuple(value.shape)}"
                    )
        if self.mask is not None:
            try:
                joint = tuple(torch.broadcast_shapes(tuple(self.mask.shape), shape))
            except RuntimeError:
                joint = None
            if joint != shape:
                raise ValueError(
                    f"mask at port {self.port} does not broadcast to the port: "
                    f"port shape {shape}, mask shape {tuple(self.mask.shape)}"
                )

    def selected(self, shape: Sequence[int], device: Any = "cpu") -> torch.Tensor | None:
        """The mask broadcast to the port shape (None for a whole-port edit)."""
        if self.mask is None:
            return None
        return self.mask.to(device).broadcast_to(tuple(shape))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """The edited value alone (``realise`` without the record)."""
        self.validate(x.shape)
        return self._result(x, self._operands(x.dtype)[0])

    def _operands(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """``(written, requested)`` on the host: the request cast once to the port dtype (the
        operand the arithmetic uses) and the request in float64. The value crosses to the host
        at most once (a scalar or a host tensor not at all)."""
        value = self.value
        if isinstance(value, torch.Tensor):
            host = value.detach().to("cpu")
            return host.to(dtype), host.to(torch.float64)
        return torch.tensor(value, dtype=dtype), torch.tensor(value, dtype=torch.float64)

    def _result(self, x: torch.Tensor, written: torch.Tensor) -> torch.Tensor:
        """``x op written`` on the selected elements, on ``x``'s device (``written`` moved there
        once)."""
        written = written.to(x.device)
        if self.op == "replace":
            full = torch.empty_like(x)
            full.copy_(written)
        elif self.op == "add":
            full = x + written
        else:
            full = x * written
        sel = self.selected(x.shape, x.device)
        return full if sel is None else torch.where(sel, full, x)

    def realise(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """The edited value and the record of what entered the model.

        The arithmetic uses the recorded cast operand (``written``, the request cast to the
        port dtype): ``x op written``. ``cast_*`` compare the requested operand with ``written``
        (float64, over the selected elements); ``realised_*`` compare the result with the ideal
        ``x_f64 op requested_f64`` (the effect error; for ``replace`` it equals the cast error);
        ``changed_fraction`` is the fraction of elements whose bits changed.
        """
        self.validate(x.shape)
        written, requested = self._operands(x.dtype)
        result = self._result(x, written)
        return result, self._record(x, result, written, requested)

    def measure(self, x: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
        """The record of this edit applied to ``x`` when the value that actually entered the
        model is ``actual``: ``realised_*`` is ``actual`` against the exact float64 result of
        ``x op requested`` (selected elements; the others must equal ``x``)."""
        return self._record(x, actual, *self._operands(x.dtype))

    def _record(
        self, x: torch.Tensor, actual: torch.Tensor, written: torch.Tensor, requested: torch.Tensor
    ) -> dict[str, Any]:
        """``measure`` given the host operands; ``x`` and ``actual`` cross to the host once each,
        and the mask is the stored host mask."""
        shape = tuple(x.shape)
        before, after = x.detach().to("cpu"), actual.detach().to("cpu")
        base = before.to(torch.float64)
        wanted = requested.expand(shape)
        ideal = (
            wanted
            if self.op == "replace"
            else (base + wanted if self.op == "add" else base * wanted)
        )
        sel = self.selected(shape)
        if sel is not None:
            ideal = torch.where(sel, ideal, base)
        effect = max_abs((after.to(torch.float64) - ideal).abs())
        gap = (written.to(torch.float64).expand(shape) - wanted).abs()
        if sel is not None:
            gap = gap[sel]
        cast_error = max_abs(gap)
        record: dict[str, Any] = {
            "realised_dtype": str(written.dtype),
            "realised_sha256": tensor_digest(written),
            "cast_exact": cast_error == 0.0,
            "cast_max_abs_error": cast_error,
            "realised_exact": effect == 0.0,
            "realised_max_abs_error": effect,
            "changed_fraction": changed_fraction(before, after),
        }
        if sel is not None:
            record["mask_count"] = int(sel.sum())
        return record

    def manifest(self) -> dict[str, Any]:
        value = self.value
        if isinstance(value, torch.Tensor):
            described = {
                "kind": "tensor",
                "sha256": tensor_digest(value),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        else:
            described = {"kind": "scalar", "value": value}
        entry: dict[str, Any] = {"port": self.port, "op": self.op, "value": described}
        if self.selection is not None:
            entry["selection"] = self.selection
        if self.mask is not None:
            entry["mask_sha256"] = tensor_digest(self.mask)
            entry["mask_count"] = int(self.mask.sum())
            entry["mask_shape"] = list(self.mask.shape)
        if self.origin is not None:
            entry["origin"] = dict(self.origin)
        return entry

    @property
    def key(self) -> str:
        """Content key under which a process-local registry can hold the value."""
        manifest = self.manifest()["value"]
        return manifest["sha256"] if manifest["kind"] == "tensor" else f"scalar:{manifest['value']}"


def _module(layer: Any, path: str) -> Any:
    module = layer
    for part in path.split(".") if path else ():
        module = getattr(module, part, None)
        if module is None:
            raise ValueError(f"native layer has no module {path!r}")
    if not isinstance(module, torch.nn.Module):
        raise ValueError(f"native layer attribute {path!r} is not a module")
    return module


# --- the one process-wide eager-attention interceptor ------------------------------------
#
# The family's modeling module has one module-level ``eager_attention_forward``. It is patched
# once, by the first controller to install, with a single interceptor that dispatches through a
# lock-protected registry keyed by attention-module identity, and restored when the registry
# empties. The interceptor is a pass-through unless the calling thread is inside the
# controller-driven transition (``installed()``) of the controller registered for that module,
# so a plain forward of the same model from user code or another thread is never staged or
# captured. Hooks on the shared native modules carry the same guard.

_LOCK = threading.RLock()
_LOCAL = threading.local()


class _Patch:
    def __init__(self, modeling: Any):
        self.modeling = modeling
        self.original = modeling.eager_attention_forward
        self.repeat_kv = modeling.repeat_kv
        self.owners: dict[int, PortController] = {}
        patch = self

        def eager(module, query, key, value, attention_mask, *args, **kwargs):
            owner = patch.owners.get(id(module))
            if owner is None or not owner._engaged():
                return patch.original(module, query, key, value, attention_mask, *args, **kwargs)
            return owner._eager(
                patch, module, query, key, value, attention_mask, *args, **kwargs
            )

        eager._mdb_attention_observer = True
        self.interceptor = eager


_PATCHES: dict[str, _Patch] = {}


def _acquire(controller: PortController, modeling: Any, attention: Any) -> _Patch:
    with _LOCK:
        patch = _PATCHES.get(modeling.__name__)
        created = patch is None
        if patch is None:
            patch = _Patch(modeling)
        if id(attention) in patch.owners:
            raise ValueError(
                "an eager-attention interceptor is already installed on this attention module "
                "(nested installs on one layer are unsupported)"
            )
        patch.owners[id(attention)] = controller
        if created:
            _PATCHES[modeling.__name__] = patch
            modeling.eager_attention_forward = patch.interceptor
        return patch


def _release(patch: _Patch, attention: Any) -> None:
    with _LOCK:
        patch.owners.pop(id(attention), None)
        if patch.owners:
            return
        if patch.modeling.eager_attention_forward is patch.interceptor:
            patch.modeling.eager_attention_forward = patch.original
        _PATCHES.pop(patch.modeling.__name__, None)


class _Stop(PortStop):
    def __init__(self, port: str):
        super().__init__(f"stopped at port {port}")
        self.port = port


class PortController:
    """Hooks for one native decoder-layer call.

    ``layer=None`` observes every layer (captures are replaced per layer) and may not carry
    edits. ``capture`` selects which ports are cloned (``True`` = all, ``False`` = none, or an
    iterable of names); the stop port is always captured. ``layers`` (optional) is the model's
    layer count: a layer at or beyond it is refused at construction (and always at install).

    Per-transition lifecycle used by ``saturn_pub.Session``: ``begin()`` clears what the previous
    transition recorded, ``manifest()`` then lists exactly the edits whose hooks fired during the
    transition's native call (a probe or a transition that did not install the controller
    contributes nothing), and raises if an edit staged for the layer that ran did not fire.

    Stopping at ``p`` with the native eager attention function: the function cannot be split,
    so the value mix ``A`` is computed natively as a by-product of producing ``p``; it is not
    held, checked or captured (``computed_past_cursor`` on ``Run.status`` discloses it).

    Thread safety: the interceptor and hooks only act inside this controller's own
    ``installed()`` window in the calling thread; a forward of the same model from another
    thread, or outside a transition, passes through untouched. Two controllers on different
    attention modules (different models, or different layers) coexist; two on one module are
    refused.
    """

    def __init__(
        self,
        program: PortProgram,
        layer: int | None,
        *,
        stop: str | None = None,
        edits: Sequence[PortEdit] = (),
        capture: bool | Iterable[str] = True,
        layers: int | None = None,
        cells: Sequence[Any] = (),
    ):
        self.program = program
        self.layer = layer
        if layer is not None and (type(layer) is not int or layer < 0):
            raise ValueError("layer must be a nonnegative integer")
        if layer is not None and layers is not None and layer >= layers:
            raise ValueError(f"layer {layer} does not exist in a {layers}-layer model")
        self.stop = None if stop is None else program.spec(stop).name
        if stop == "h_out":
            raise ValueError("h_out is the layer output; commit the layer instead of stopping")
        self.edits = tuple(edits)
        if self.edits and layer is None:
            raise ValueError("edits need an explicit layer")
        self.cells = tuple(cells)  # engine.edits.CellEdit, realised in this layer's call
        if self.cells and layer is None:
            raise ValueError("cell edits need an explicit layer")
        self.cell_records: dict[int, dict[str, Any]] = {}
        self._adapter: Any = None
        self._edits_at: dict[str, list[PortEdit]] = {}
        for edit in self.edits:
            spec = program.spec(edit.port)
            if not spec.editable:
                raise ValueError(f"port {spec.name} is not editable: {spec.edit}")
            self._edits_at.setdefault(spec.name, []).append(edit)
        names = program.names
        if capture is True:
            self._capture = set(names)
        elif capture is False:
            self._capture = set()
        else:
            self._capture = {program.spec(name).name for name in capture}
        if self.stop is not None:
            self._capture.add(self.stop)
        self.captures: dict[str, torch.Tensor] = {}
        self.originals: dict[str, torch.Tensor] = {}
        self.order: list[str] = []
        self.checks: dict[str, Any] = {}
        self.installs = 0
        self.edited_captures: dict[str, dict[str, str]] = {}
        self.capture_notes: dict[str, dict[str, Any]] = {}
        self.replica_verified: bool | None = None
        self.replica_max_abs_diff: float | None = None
        self.realisation: dict[str, Any] | None = None
        self.fired: dict[int, dict[str, Any]] = {}
        self._realised: list[dict[str, Any]] = []
        self._live: dict[str, Any] = {}
        self._active = False
        self._seen_layer: int | None = None
        self._installed_now = False
        self._layer_index = layer
        self._local = threading.local()  # per-thread depth of entries into the pending layer

    # --- protocol used by saturn_pub ----------------------------------------------------

    def begin(self) -> None:
        """Start of one transition (probe or commit): forget what an earlier one recorded."""
        self._realised, self._seen_layer, self._installed_now = [], None, False
        self.fired, self.cell_records = {}, {}

    def _carries(self) -> bool:
        return self.layer is not None and bool(self._edits_at or self.stop or self.cells)

    def check_ahead(self, ahead: Sequence[Any]) -> None:
        """Before a multi-step ``continue_``: refuse when the controller carries edits (or a
        stop) for a layer that none of the coming transitions runs (``ahead`` is the adapter's
        ``ports_ahead``), so nothing is silently dropped."""
        if self._carries() and self.layer not in ahead:
            what = "edits" if self._edits_at or self.cells else "a stop"
            raise ValueError(
                f"port controller carries {what} for layer {self.layer} but the coming "
                f"transition(s) run {self._ahead_text(ahead)}; nothing was run"
            )

    @staticmethod
    def _ahead_text(ahead: Sequence[Any]) -> str:
        layers = sorted({a for a in ahead if a is not None})
        return ("layer " + ", ".join(map(str, layers))) if layers else "no decoder layer"

    def begin_call(self) -> None:
        """Start of one ``continue_`` call."""
        self._call_installs = self.installs

    def end_call(self) -> None:
        """End of one ``continue_`` call: a controller with edits must have been installed."""
        if self._carries() and self.installs == getattr(self, "_call_installs", self.installs):
            raise ValueError(
                f"port controller carries edits for layer {self.layer} but the call never ran "
                "that layer; the edits were not applied"
            )

    def _missing(self) -> list[str]:
        if self._seen_layer is None or self._seen_layer != self.layer:
            return []
        cells = [cell.text for i, cell in enumerate(self.cells) if i not in self.cell_records]
        return sorted(set(self._edits_at) - {r["port"] for r in self._realised}) + cells

    def unfired_before_stop(self) -> list[str]:
        """Edited ports at or before the stop port whose edit did not fire (stop probes only)."""
        if self.stop is None:
            return []
        limit = self.program.index(self.stop)
        done = {r["port"] for r in self._realised}
        cells = [cell.text for i, cell in enumerate(self.cells)
                 if cell.timing == "before" and i not in self.cell_records]
        return sorted(
            port
            for port in self._edits_at
            if self.program.index(port) <= limit and port not in done
        ) + cells

    def manifest(self) -> list[dict[str, Any]] | None:
        """Edits whose hooks fired in the transition that just ran; consumed once.

        Raises if the transition ran the layer this controller addresses and a staged edit did
        not fire, so a receipt can never omit an edit that was meant to apply.
        """
        missing = self._missing()
        if missing:
            raise ValueError(f"staged edits were not realised: {', '.join(missing)}")
        failed = [
            entry["port"]
            for entry in self._realised
            if entry["port"] in ("s", "p") and entry.get("replica_verified") is False
        ]
        if failed:
            raise ValueError(
                f"layer {self._layer_index}: replica check failed (max |replica - native| = "
                f"{self.replica_max_abs_diff}); refusing to commit the {', '.join(failed)} edit "
                "(its value mix is a replica that does not match the native computation)"
            )
        realised, self._realised, self._seen_layer = self._realised, [], None
        # State-cell edits realised in this call ride in the same list (entries carry a
        # ``cell`` key and the cell's ``port`` ``cache.L<l>.<k|v>``): address, op, operand
        # digest, origin and the realisation record. ``cell_records`` stays for the run.
        cells = [
            {**self.cells[i].manifest(), **self.cell_records[i]}
            for i in sorted(self.cell_records)
        ]
        realised = [*realised, *cells]
        return realised or None

    def probe_check(self) -> None:
        """After a probe: the controller must have met the layer it addresses."""
        if self.layer is not None and not self._installed_now:
            ran = "no decoder layer" if self._seen_layer is None else f"layer {self._seen_layer}"
            raise ValueError(
                f"port controller addresses layer {self.layer} but the probed transition ran "
                f"{ran}; nothing was captured"
            )

    @contextmanager
    def installed(self, adapter: Any, layer_index: int, layer_module: Any):
        if self.layer is not None and self.layer >= adapter.layers:
            raise ValueError(
                f"port controller addresses layer {self.layer} but the model has "
                f"{adapter.layers} layers"
            )
        self._seen_layer = layer_index
        if self.layer is not None and layer_index != self.layer:
            yield self
            return
        if self._active:
            raise ValueError("port controller is already installed")
        self._check(adapter, layer_module)
        self.captures, self.originals, self.order, self.checks = {}, {}, [], {}
        self.edited_captures, self.capture_notes, self.replica_verified = {}, {}, None
        self.replica_max_abs_diff = None
        self.realisation, self.fired, self.cell_records = None, {}, {}
        self._realised, self._live = [], {}
        self._layer_index = layer_index
        self._adapter = adapter
        self._local.depth = 0
        attention = layer_module.self_attn
        handles = self._register(layer_module)
        try:
            patch = _acquire(self, self.program.modeling_module(), attention)
        except Exception:
            for handle in handles:
                handle.remove()
            raise
        previous = getattr(_LOCAL, "active", None)
        self._active = True
        self._installed_now = True
        self.installs += 1
        completed = False
        _LOCAL.active = self
        try:
            yield self
            completed = True
        finally:
            _LOCAL.active = previous
            self._local.depth = 0
            self._live = {}  # call-scoped references (h, r operands, the native mix): not kept
            _release(patch, attention)
            for handle in handles:
                handle.remove()
            self._active = False
        if completed:
            missing = self._missing()
            if missing:
                raise ValueError(f"staged edits were not realised: {', '.join(missing)}")

    # --- installation ------------------------------------------------------------------

    def _check(self, adapter: Any, layer_module: Any) -> None:
        if adapter.spec.model_type != self.program.family:
            raise ValueError(
                f"{self.program.family} port program cannot drive {adapter.spec.model_type}"
            )
        if adapter.execution.get("attention") != "eager":
            raise ValueError("port stepping requires eager attention")
        if adapter.execution.get("residency", "resident") != "resident":
            raise ValueError("port stepping supports resident execution only")
        check_deterministic(layer_module)
        for site in self.program.sites():
            _module(layer_module, site)

    def _register(self, layer: Any) -> list[Any]:
        handles = []
        try:
            # entry counter first, exit counter (always called) last: hooks and the eager
            # interceptor act only on the outermost entry of the layer.
            handles.append(layer.register_forward_pre_hook(self._enter))
            if self.cells:
                handles.append(
                    layer.register_forward_pre_hook(self._cells_before, with_kwargs=True)
                )
            for port in self.program.ports:
                if port.hook not in {"pre", "out"}:
                    continue
                module = _module(layer, port.site)
                if port.name == "r":
                    hook = module.register_forward_pre_hook(self._r_hook)
                elif port.hook == "pre":
                    hook = module.register_forward_pre_hook(self._pre(port.name))
                else:
                    hook = module.register_forward_hook(self._out(port.name))
                handles.append(hook)
            if self.cells:  # not always_call: an interrupted call realises no "after" cell
                handles.append(layer.register_forward_hook(self._cells_after, with_kwargs=True))
            handles.append(layer.register_forward_hook(self._exit, always_call=True))
        except Exception:
            for handle in handles:
                handle.remove()
            raise
        return handles

    def _enter(self, module: Any, args: tuple[Any, ...]) -> None:
        self._local.depth = getattr(self._local, "depth", 0) + 1

    def _exit(self, module: Any, args: tuple[Any, ...], output: Any) -> None:
        self._local.depth = max(getattr(self._local, "depth", 0) - 1, 0)

    def _write_cells(self, kwargs: dict[str, Any], timing: str) -> None:
        """Realise the ``timing`` cell edits into the cache object the native layer was handed
        (copy-on-write: each written layer tensor is replaced by an edited copy)."""
        if not self._engaged():
            return
        cache = kwargs.get(self._adapter.spec.cache_kwarg)
        if cache is None:
            raise ValueError("the layer call carries no key/value cache to edit")
        for index, cell in enumerate(self.cells):
            if cell.timing != timing or index in self.cell_records:
                continue
            if cell.layer >= len(cache.layers):
                raise ValueError(f"{cell.port}: the cache holds no layer {cell.layer}")
            slot = cache.layers[cell.layer]
            name = "keys" if cell.tensor == "k" else "values"
            current = getattr(slot, name, None)
            if current is None:
                raise ValueError(f"{cell.port} holds no rows yet")
            written, record = cell.realise(current)
            setattr(slot, name, written)
            self.cell_records[index] = {**record, "realised_timing": timing,
                                        "realised_in_layer": self._layer_index}

    def _cells_before(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self._write_cells(kwargs, "before")

    def _cells_after(
        self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any
    ) -> None:
        self._write_cells(kwargs, "after")

    def _engaged(self) -> bool:
        """True only inside this controller's own transition, in the calling thread, at the
        outermost entry of the addressed layer (a nested forward of the same model from a user
        hook runs natively)."""
        return getattr(_LOCAL, "active", None) is self and getattr(self._local, "depth", 0) == 1

    def _keep(self, name: str, value: torch.Tensor) -> None:
        self.order.append(name)
        if name in self._capture:
            self.captures[name] = value.detach().clone()
        if name == self.stop:
            raise _Stop(name)

    def _mark_edited(self, port: str, by: str, kind: str) -> None:
        seen = self.edited_captures.get(port)
        if seen is not None and by not in seen["by"].split(","):
            by = f"{seen['by']},{by}"
        self.edited_captures[port] = {"by": by, "kind": kind}

    def _value(self, name: str, x: torch.Tensor) -> torch.Tensor:
        edits = self._edits_at.get(name)
        native_mix = self._live.pop("A_native", None) if name == "A" else None
        if native_mix is not None:
            self.originals["A"] = native_mix  # the native value mix, before the replica's
        if edits:
            self.originals.setdefault(name, x.detach().clone())
            for edit in edits:
                x, record = edit.realise(x)
                entry = {**edit.manifest(), "layer": self._layer_index, **record}
                if name in ("s", "p"):
                    entry["mix"] = "replica"
                    entry["replica_verified"] = None
                if name == "p":
                    entry.update(_row_statistics(x))
                with_labels(entry)
                self._realised.append(entry)
                self.fired[id(edit)] = entry
            self._mark_edited(name, name, "edit")
        if name == "h":
            self._live["h"] = x
        if name == "o" and "r" in self._edits_at:
            self.originals.setdefault("o", x.detach().clone())
            x = self._realise_r(x)
            self._mark_edited("o", "r", "realisation")
        self._keep(name, x)
        return x

    def _realise_r(self, o: torch.Tensor) -> torch.Tensor:
        h = self._live["h"]
        native = h + o  # a new tensor nothing writes: the original of r, held once uncopied
        target = native
        inputs, outputs = [], []
        for edit in self._edits_at["r"]:
            inputs.append(target)
            target = edit.apply(target)  # the records are made in _r_hook, once r is known
            outputs.append(target)
        self.originals["r"] = native
        self._live["r_target"] = target
        self._live["r_inputs"] = inputs
        self._live["r_outputs"] = outputs
        self._live["r_native"] = native
        edits = self._edits_at["r"]
        if any(edit.mask is None for edit in edits):
            new_o = target - h
        else:
            union = torch.zeros(o.shape, dtype=torch.bool, device=o.device)
            for edit in edits:
                union |= edit.selected(o.shape, o.device)
            new_o = torch.where(union, target - h, o)
        predicted = h + new_o
        self.realisation = {
            "exact": bool(torch.equal(predicted, target)),
            "max_abs_error": max_abs((wide(predicted) - wide(target)).abs()),
        }
        return new_o

    def _r_hook(self, module: Any, args: tuple[Any, ...]) -> None:
        if not self._engaged():
            return None
        r = args[0]
        target = self._live.get("r_target")
        if target is not None:
            error = max_abs((wide(r) - wide(target)).abs())
            native = self._live["r_native"]
            self.realisation = {"exact": bool(torch.equal(r, target)), "max_abs_error": error}
            edits = self._edits_at["r"]
            outputs = self._live["r_outputs"]
            for position, edit in enumerate(edits):
                # the last edit's effect is what actually entered the model at r (r = h + o')
                actual = r if position == len(edits) - 1 else outputs[position]
                record = edit.measure(self._live["r_inputs"][position], actual)
                # realised_* are measured against the exact float64 result of x op requested,
                # as at every port; the cost of realising r through o is recorded separately.
                entry = {
                    **edit.manifest(),
                    "layer": self._layer_index,
                    **{k: v for k, v in record.items() if k != "changed_fraction"},
                    "realised_at": "o",
                    "realisation_exact": self.realisation["exact"],
                    "realisation_max_abs_error": error,
                    "changed_fraction": changed_fraction(native, r),
                }
                with_labels(entry)
                self._realised.append(entry)
                self.fired[id(edit)] = entry
            self._mark_edited("r", "r", "edit")
        self._keep("r", r)

    def _pre(self, name: str):
        def hook(module: Any, args: tuple[Any, ...]):
            if not self._engaged():
                return None
            value = self._value(name, args[0])
            return None if value is args[0] else (value, *args[1:])

        return hook

    def _out(self, name: str):
        def hook(module: Any, args: tuple[Any, ...], output: Any):
            if not self._engaged():
                return None
            first = output[0] if isinstance(output, tuple) else output
            value = self._value(name, first)
            if value is first:
                return None
            return (value, *output[1:]) if isinstance(output, tuple) else value

        return hook

    def _eager(self, patch: _Patch, module, query, key, value, attention_mask, *args, **kwargs):
        original, repeat_kv = patch.original, patch.repeat_kv
        dropout = kwargs.get("dropout", args[0] if args else 0.0)
        if module.training or dropout:
            raise ValueError("port stepping requires dropout-free eval attention")
        query = self._value("qr", query)
        self._value("kr", key)
        scaling = kwargs.get("scaling", getattr(module, "scaling", None))
        groups = module.num_key_value_groups
        scores = torch.matmul(query, repeat_kv(key, groups).transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        used = self._value("s", scores)
        output, weights = original(module, query, key, value, attention_mask, *args, **kwargs)
        reference = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        s_check = _equality(reference, weights)
        self.checks["s"] = s_check
        self.capture_notes["s"] = {
            "replica_equal": s_check["equal"],
            "max_abs_diff": s_check.get("max_abs_diff", 0.0),
        }
        probs = weights
        if used is not scores:
            self.originals["p"] = weights.detach().clone()
            probs = torch.nn.functional.softmax(used, dim=-1, dtype=torch.float32).to(query.dtype)
            self._mark_edited("p", "s", "replica")
        edited = self._value("p", probs)  # a stop at p ends here; the mix is checked at A
        native_mix = torch.matmul(weights, repeat_kv(value, groups)).transpose(1, 2).contiguous()
        mix_check = _equality(native_mix, output)
        self.checks["mix_replica"] = mix_check
        self.replica_verified = bool(s_check["equal"] and mix_check["equal"])
        self.replica_max_abs_diff = max(
            s_check.get("max_abs_diff", 0.0), mix_check.get("max_abs_diff", 0.0)
        )
        mix = torch.matmul(edited, repeat_kv(value, groups)).transpose(1, 2).contiguous()
        for entry in self._realised:
            if entry["port"] in ("s", "p"):
                entry["replica_verified"] = self.replica_verified
        if edited is weights:
            self.checks["mix"] = mix_check
            return output, weights
        self.checks["mix"] = {"replica": True, "native_unedited": False}
        self._live["A_native"] = output.detach().reshape(*output.shape[:-2], -1).clone()
        return mix, edited


def _row_statistics(probs: torch.Tensor) -> dict[str, Any]:
    """Row sums and minimum of an edited ``p``. The row-sum tolerance is a supplied constant by
    dtype: the dtype's machine epsilon for bf16/f16, 1e-5 for fp32, 1e-12 for fp64."""
    if probs.dtype in (torch.bfloat16, torch.float16):
        tolerance = float(torch.finfo(probs.dtype).eps)
    elif probs.dtype == torch.float64:
        tolerance = 1e-12
    else:
        tolerance = 1e-5
    wide = probs.detach().to("cpu", torch.float64)
    sums = wide.sum(dim=-1)
    low, high, smallest = float(sums.min()), float(sums.max()), float(wide.min())
    return {
        "row_sum_min": low,
        "row_sum_max": high,
        "min": smallest,
        "row_sum_tolerance": tolerance,
        "is_distribution": bool(smallest >= 0.0 and max(abs(low - 1), abs(high - 1)) <= tolerance),
    }


def _equality(replica: torch.Tensor, native: torch.Tensor) -> dict[str, Any]:
    # bit-for-bit, with NaN positions counted as equal (an overflowing edit makes both NaN)
    equal = bool(torch.equal(replica, native)) or bool(
        torch.allclose(wide(replica), wide(native), rtol=0.0, atol=0.0, equal_nan=True)
    )
    result: dict[str, Any] = {"equal": equal}
    if not equal:  # measured: the float64 difference of the two native-dtype values (exact)
        result["max_abs_diff"] = float((wide(replica) - wide(native)).abs().max())
    return result


def observe_all(program: PortProgram) -> PortController:
    """An edit-free controller that captures every port of whichever layer runs."""
    return PortController(program, None)


# Old private names, kept for one commit: engine/edits.py imported them.
_with_labels = with_labels
_max_abs = max_abs

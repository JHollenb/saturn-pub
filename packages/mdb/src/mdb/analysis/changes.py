"""Finite computation changes between aligned native branches.

This experimental Python API accounts for endpoint differences, not isolated causal
credit. Consumer projections use the edited final RMS scale and the direct residual
readout; they are not suffix derivatives. Full tensor terms stay outside the JSON view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mdb.core.pointers import check_bounded

from . import finite
from .finite import EXACT, MEASURED, f64, num
from .forecast import Margin, Refusal
from .propagate import state_digest
from .reads import layer_module


@dataclass
class BranchChanges:
    report: dict[str, Any]
    tensors: dict[str, Any] = field(default_factory=dict)


def _norm(value: Any, authority: str = EXACT) -> dict:
    return num(float(f64(value).norm()), authority)


def _identity(result: Any) -> dict:
    out = {"proof": result.proof, "function": result.name,
           "terms": {name: {"norm": _norm(value)} for name, value in result.terms.items()},
           "algebra_closure": _norm(result.total - result.reference)}
    if result.native is not None:
        out["native_discrepancy"] = _norm(result.native - result.total, MEASURED)
    return out


def _rank(parts: dict[str, Any], top: int, address: Any, axis: str) -> dict:
    """Signed direct projections, with one exact aggregate of every omitted row."""
    torch = finite._torch()
    total = sum(parts.values())
    order = torch.argsort(total.abs(), descending=True, stable=True)[:top].tolist()
    entries = [{axis: i, "address": address(i), "total": num(float(total[i]), EXACT),
                "parts": {name: num(float(value[i]), EXACT) for name, value in parts.items()}}
               for i in order]
    tail = {name: num(float(value.sum() - value[order].sum()), EXACT)
            for name, value in parts.items()}
    return {"ranking": "absolute signed total under the fixed direct readout",
            "rows": entries, "omitted": total.numel() - len(order), "tail": tail,
            "total": num(float(total.sum()), EXACT)}


def explain_changes(base: Any, edited: Any, consumer: Margin, *, layer: int,
                    top: int = 5) -> BranchChanges:
    """Explain one block between completed runs of the same adapter and token history.

    Supports Qwen2/Qwen3 native ports. Edits to K/V producers or cache cells are
    supported. Direct edits of the explained intermediates are refused rather than
    being hidden inside an arithmetic discrepancy. Different prompts need a separate
    alignment contract. The supplied layer is not automatically discovered.
    """
    if base is edited:
        raise Refusal("use two distinct branch runs (an identical-copy sham is supported)")
    if base.adapter is not edited.adapter:
        raise Refusal("branches must share the same adapter and checkpoint instance")
    if not isinstance(consumer, Margin):
        raise Refusal("this prototype requires a declared Margin consumer")
    if not 0 <= layer < base.adapter.layers or not 1 <= top <= 8:
        raise Refusal("layer is outside the model or top is outside 1..8")
    before = [state_digest(run) for run in (base, edited)]
    tokens = [run.session.read("tokens").tolist() for run in (base, edited)]
    if tokens[0] != tokens[1]:
        raise Refusal("branches need identical consumed tokens and visibility")
    row = consumer.row
    if any(run.where not in (f"t{row}.logits", f"t{row}.sample") for run in (base, edited)):
        raise Refusal("both branches must be at the consumer row's logits or sample")
    if not 0 <= min(consumer.a, consumer.b) <= max(consumer.a, consumer.b) < (
            base.adapter.model.config.vocab_size):
        raise Refusal("consumer token is outside the vocabulary")
    names = ("h", "a", "p", "A", "o", "r", "n", "g", "sg", "u", "f", "m", "h_out")
    ports = []
    for run in (base, edited):
        captured = {}
        for name in names:
            address = f"t{row}.L{layer}.{name}"
            info = run.read_info(address)
            if info["kind"] in ("edit", "realisation"):
                raise Refusal(f"direct edit of explained intermediate {address} needs "
                              "explicit edit-displacement accounting")
            captured[name] = run.read(address)
            if not bool(finite._torch().isfinite(captured[name]).all()):
                raise Refusal(f"nonfinite native operand at {address}")
        ports.append(captured)
    p0, p1 = ports
    adapter = base.adapter
    module = layer_module(adapter, layer)
    torch = finite._torch()
    heads, width = adapter.num_heads, adapter.head_dim
    kappa = finite.gqa_kappa(heads, adapter.num_kv_heads)
    probs = [p["p"].reshape(heads, row + 1) for p in ports]
    values = [run.read_cache(layer, "v")[0, kappa, :row + 1, :]
              for run in (base, edited)]
    if any(tuple(v.shape) != (heads, row + 1, width) for v in values):
        raise Refusal("the cache does not cover the aligned visible source set")
    attn = finite.transport_finite_split(*probs, *values,
        native0=p0["A"].reshape(heads, width), native1=p1["A"].reshape(heads, width))
    mlp = finite.swiglu_finite_split(p0["g"].reshape(-1), p1["g"].reshape(-1),
        p0["u"].reshape(-1), p1["u"].reshape(-1),
        native0=p0["f"].reshape(-1), native1=p1["f"].reshape(-1))
    norms = {name: finite.rms_finite_change(p0[src], p1[src], norm.weight,
                    float(norm.variance_epsilon), native0=p0[dst], native1=p1[dst])
             for name, src, dst, norm in (
                 ("attention_norm", "h", "a", module.input_layernorm),
                 ("mlp_norm", "r", "n", module.post_attention_layernorm))}

    # Slice vocabulary rows before casting; do not materialize a float64 unembedding.
    head = adapter.model.lm_head
    final_norm = getattr(adapter.backbone, adapter.spec.final_norm)
    last = adapter.layers - 1
    final = [f64(run.read(f"t{row}.L{last}.h_out")).reshape(-1)
             for run in (base, edited)]
    rho = [(float(final_norm.variance_epsilon) + h.square().mean()).rsqrt() for h in final]
    readout = (f64(head.weight[consumer.a]) - f64(head.weight[consumer.b])) * f64(
        final_norm.weight)
    c = rho[1] * readout
    native_margin = [f64(run.read(f"t{row}.logits")).reshape(-1)[consumer.a] -
                     f64(run.read(f"t{row}.logits")).reshape(-1)[consumer.b]
                     for run in (base, edited)]
    delta = native_margin[1] - native_margin[0]
    final_terms = {"residual": c @ (final[1] - final[0]),
                   "scale": (rho[1] - rho[0]) * (readout @ final[0])}

    # One layer's matrices, cast once and released before returning.
    out_weight = f64(module.self_attn.o_proj.weight)
    head_readout = (c @ out_weight).reshape(heads, width)
    pv0, pv1, vv0, vv1 = map(f64, (*probs, *values))
    dp, dv = pv1 - pv0, vv1 - vv0
    source_parts = {
        "payload": (pv0 * torch.einsum("hd,hjd->hj", head_readout, dv)).sum(0),
        "routing": (dp * torch.einsum("hd,hjd->hj", head_readout, vv0)).sum(0),
        "interaction": (dp * torch.einsum("hd,hjd->hj", head_readout, dv)).sum(0)}
    attn_write = {name: out_weight @ value.reshape(-1) for name, value in attn.terms.items()}
    del out_weight
    down = f64(module.mlp.down_proj.weight)
    neuron_readout = c @ down
    neuron_parts = {name: neuron_readout * value for name, value in mlp.terms.items()}
    mlp_write = {name: down @ value for name, value in mlp.terms.items()}
    del down
    changes = {name: f64(p1[name]).reshape(-1) - f64(p0[name]).reshape(-1)
               for name in ("h", "o", "r", "m", "h_out")}
    attn_gap = changes["o"] - sum(attn_write.values())
    mlp_gap = changes["m"] - sum(mlp_write.values())
    add_gap = changes["h_out"] - changes["h"] - changes["o"] - changes["m"]
    projection = {name: c @ value for name, value in changes.items()}
    projection["arithmetic_discrepancy"] = c @ add_gap
    after = [state_digest(run) for run in (base, edited)]
    if before != after:
        raise RuntimeError("branch explanation changed a run state")
    report = {
        "schema": "mdb-branch-changes-v1", "row": row, "layer": layer,
        "scope": "observed endpoint accounting; direct final readout, not isolated suffix effects",
        "lineage": {"base": before[0], "edited": before[1], "session_unchanged": True,
                    "model_identity": adapter.model_identity,
                    "execution": dict(adapter.execution)},
        "consumer": {"address": consumer.text,
            "base": num(float(native_margin[0]), MEASURED),
            "edited": num(float(native_margin[1]), MEASURED),
            "delta": num(float(delta), MEASURED),
            "residual_change": num(float(final_terms["residual"]), EXACT),
            "normalization_scale_change": num(float(final_terms["scale"]), EXACT),
            "native_discrepancy": num(float(delta - sum(final_terms.values())), MEASURED)},
        "attention": {**_identity(attn),
            "sources": _rank(source_parts, top, lambda i: f"cache.L{layer}.v[row={i}]", "row"),
            "write_discrepancy_norm": _norm(attn_gap, MEASURED),
            "write_discrepancy_projection": num(float(c @ attn_gap), MEASURED)},
        "mlp": {**_identity(mlp),
            "neurons": _rank(neuron_parts, top, lambda i: f"t{row}.L{layer}.f[i={i}]", "index"),
            "write_discrepancy_norm": _norm(mlp_gap, MEASURED),
            "write_discrepancy_projection": num(float(c @ mlp_gap), MEASURED)},
        "normalization": {name: _identity(value) for name, value in norms.items()},
        "block_direct_projection": {name: num(float(value), MEASURED)
                                    for name, value in projection.items()},
        "equations": [
            {"output": f"L{layer}.A", "inputs": [f"L{layer}.p", f"cache.L{layer}.v"],
             "proof": "P07", "meaning": "payload + routing + interaction + native discrepancy"},
            {"output": f"L{layer}.n", "inputs": [f"L{layer}.r"], "proof": "P03"},
            {"output": f"L{layer}.f", "inputs": [f"L{layer}.g", f"L{layer}.u"], "proof": "P15"},
            {"output": f"L{layer}.h_out", "inputs": [f"L{layer}.h", f"L{layer}.o", f"L{layer}.m"],
             "proof": "P14"}],
        "limitations": ["top rows ranked by direct projection; cancellations can hide large terms",
                        "unlisted rows remain in the aggregate tail",
                        "native discrepancies are observations, not certified error bounds",
                        "this is not automatic circuit localization or semantic discovery"]}
    check_bounded(report, "branch-change report")
    return BranchChanges(report, {"attention": attn, "mlp": mlp, "normalization": norms,
        "source_parts": source_parts, "neuron_parts": neuron_parts,
        "attention_write": attn_write, "mlp_write": mlp_write,
        "attention_gap": attn_gap, "mlp_gap": mlp_gap, "readout": c})

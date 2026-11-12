"""One matched organism for representation, routing, writes, futures and decoding.

The finite panel is exploratory. Predictions and observations have distinct
provenance; chromatic connected components do not identify shape semantics.
"""
from __future__ import annotations

import itertools
import runpy
from pathlib import Path


def plan():
    histories = []
    for binding, relation in itertools.product((0, 1), repeat=2):
        first, second = ("red", "blue") if binding == 0 else ("blue", "red")
        prompt = (f"A {first} sphere to the {('left', 'right')[relation]} of a {second} cube "
                  "on a plain white background. Two separate objects, simple studio product photograph.")
        histories.append(dict(id=f"b{binding}r{relation}", binding=binding, relation=relation, prompt=prompt))
    return dict(histories=histories, seed=109, decoder_alphas=[-.1, -.05, 0., .05, .1],
        interventions=[
            dict(id="route.sham", fields=["q", "k", "v"], source="b0r0"),
            dict(id="route.qk", fields=["q", "k"], source="b1r0"),
            dict(id="route.v", fields=["v"], source="b1r0"),
            dict(id="route.qkv", fields=["q", "k", "v"], source="b1r0"),
            dict(id="write.attention", source="b1r0", attention=True),
            dict(id="write.mlp", source="b1r0", mlp=True),
            dict(id="write.both", source="b1r0", attention=True, mlp=True)],
        hypotheses={
            "equations": "Four full project and first-joint image/text forecasts, each consuming own predicted project/context, seal before any denoiser target. Compare full fields; numerical errors remain observations. Encoder conditioning is observed.",
            "binding": "Tentative: whole QKV or attention-plus-MLP transplant moves recipient toward binding-donor image more than the sham. QK-only and V-only are competing explanatory arms, with no assumed winner. Recipient residual/text/conditioning remain base at installation.",
            "composition": "Tentative additive rule at every observed boundary: X11 = X10 + X01 - X00. Retain relative interaction and all counterexamples. This is a tested hypothesis, not an inferred universal law.",
            "decoder": "Test continuous FP32 VAE response versus BF16-rounded input under the same FP32 consumer, then compare native BF16 decoder arithmetic separately. Symmetric dose curvature separates nonlinear response from latent-input rounding. FP32 is a diagnostic lane, not native parity.",
            "compliance": "Expect two visible colored components with requested image-plane ordering. This is prompt compliance, not internal object discovery; shape binding requires visual review.",
            "scope": "One checkpoint/backend/seed, four prompts with same color/shape vocabulary. No independent Qwen encoder forecast, no exact RGB forecast, no universal semantic names or speedup claim."})


def factorial(values):
    import numpy as np
    if set(values) != {"b0r0", "b0r1", "b1r0", "b1r1"}:
        raise ValueError("complete binding by relation panel required")
    x = {k: np.asarray(v, dtype=np.float64) for k, v in values.items()}
    if any(v.shape != x["b0r0"].shape or not np.isfinite(v).all() for v in x.values()):
        raise ValueError("matched finite arrays required")
    rms = lambda v: float(np.sqrt(np.mean(v*v)))
    interaction = x["b1r1"] - x["b1r0"] - x["b0r1"] + x["b0r0"]
    total = x["b1r1"] - x["b0r0"]
    return {"binding_r0_rms": rms(x["b1r0"]-x["b0r0"]),
        "binding_r1_rms": rms(x["b1r1"]-x["b0r1"]),
        "relation_b0_rms": rms(x["b0r1"]-x["b0r0"]),
        "relation_b1_rms": rms(x["b1r1"]-x["b1r0"]),
        "interaction_rms": rms(interaction), "joint_change_rms": rms(total),
        "interaction_over_joint": rms(interaction)/rms(total) if rms(total) else None,
        "scope": "descriptive finite differences; additive forecast error, not a causal decomposition"}


def port_candidate(block, cut, *, donor=None, fields=(), attention=None, mlp=None):
    """Counted native candidate; full positioned QK routes versus V payloads.

    Candidate text output is returned for verification but the experiment installs
    image only. Scoped native dispatch changes never affect adapter continuation.
    """
    from diffusers.models.transformers import transformer_flux2 as native
    if set(fields)-{"q", "k", "v"} or (fields and donor is None):
        raise ValueError("declared positioned Q/K/V fields and donor required")
    helper = runpy.run_path(str(Path(__file__).with_name("flux_joint_writes.py")))
    if type(block.attn.processor).__name__ != "Flux2AttnProcessor":
        raise ValueError("qualified joint processor required")
    original, captured = native.dispatch_attention_fn, {}
    def dispatch(q, k, v, **kwargs):
        if captured:
            raise ValueError("multiple candidate dispatches")
        observed = dict(q=q.detach().clone(), k=k.detach().clone(), v=v.detach().clone())
        effective = dict(observed)
        for name in fields:
            supplied = donor[name]
            if (supplied.shape, supplied.dtype, supplied.device) != (observed[name].shape, observed[name].dtype, observed[name].device):
                raise ValueError("source field contract differs")
            effective[name] = supplied.detach().clone()
        captured.update(observed=observed, effective=effective,
            dispatch=dict(backend=str(kwargs.get("backend")), parallel_config=str(kwargs.get("parallel_config")),
                          mask_is_none=kwargs.get("attn_mask") is None))
        return original(effective["q"], effective["k"], effective["v"], **kwargs)
    try:
        native.dispatch_attention_fn = dispatch
        value = helper["joint_candidate"](block, cut, attention=attention, mlp=mlp)
        if set(captured) != {"observed", "effective", "dispatch"}:
            raise ValueError("positioned Q/K/V not captured")
        value["qkv"] = captured
        return value
    finally:
        native.dispatch_attention_fn = original


def route_witness(qkv, text_rows, groups, *, queries=64):
    """Exact softmax equation on actual positioned Q/K; fixed spatial witnesses.

    The result is a route measurement. No token or attention weight is assigned
    an object meaning merely because its text contains a noun or color.
    """
    import numpy as np
    q, k = (np.asarray(qkv[name], dtype=np.float64) for name in ("q", "k"))
    if q.shape != k.shape or q.ndim != 4 or q.shape[0] != 1 or not 0 < text_rows < q.shape[1]:
        raise ValueError("B1 sequence/head/dim positioned fields required")
    rows = np.unique(np.linspace(text_rows, q.shape[1]-1, min(queries, q.shape[1]-text_rows)).astype(int))
    score = np.einsum("bqhd,bkhd->bhqk", q[:, rows], k) / np.sqrt(q.shape[-1])
    score -= score.max(-1, keepdims=True)
    p = np.exp(score); p /= p.sum(-1, keepdims=True)
    mass = {name: p[..., sorted(set(indices))].sum(-1).astype(np.float32)
            for name, indices in groups.items() if indices}
    mass.update(all_text=p[..., :text_rows].sum(-1).astype(np.float32),
                all_image=p[..., text_rows:].sum(-1).astype(np.float32))
    entropy = -(p*np.log(np.maximum(p, 1e-300))).sum(-1)
    return dict(rows=rows, probability=p.astype(np.float32), entropy=entropy.astype(np.float32), **mass)


def error(predicted, actual):
    import numpy as np
    p, a = np.asarray(predicted, np.float64), np.asarray(actual, np.float64)
    if p.shape != a.shape or not np.isfinite(p).all() or not np.isfinite(a).all():
        raise ValueError("finite matched prediction fields required")
    d = p-a; rms = lambda x: float(np.sqrt(np.mean(x*x)))
    return dict(rms=rms(d), relative_rms=rms(d)/rms(a) if rms(a) else None,
                max_abs=float(np.max(np.abs(d))), exact=bool(np.array_equal(p, a)))

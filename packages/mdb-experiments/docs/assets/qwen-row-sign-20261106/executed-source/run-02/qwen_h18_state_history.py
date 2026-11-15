"""Own-state Qwen sign-flip history; separate functional and native entrypoints.

This is a component composition, not a public Qwen Session or Act. Call native
verification only after the caller's complete forecast cohort has been sealed.
"""

from __future__ import annotations

import hashlib
import runpy
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F

PREFIX_SHA256 = "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a"
SCHEMA = "qwen-h18-state-history-v1"


def load_prefix(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != PREFIX_SHA256:
        raise ValueError("frozen prefix source differs")
    return runpy.run_path(str(path))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _geometry(edit_index, horizon, selected):
    _require(0 < edit_index < horizon, "edit_index must precede suffix horizon")
    _require(len(selected) == 3 and selected[1] == edit_index and selected[2] == horizon,
             "selected must be (earlier, edit_index, horizon)")
    _require(0 < selected[0] < edit_index, "first selected state must precede edit")


def _emit(observe, lane, name, value):
    if observe is not None:
        observe(lane, name, value)
    return value


@contextmanager
def _no_modules():
    def reject(module, args):
        raise RuntimeError(f"native module forward forbidden during forecast: {type(module).__name__}")
    handle = torch.nn.modules.module.register_module_forward_pre_hook(reject)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def _frozen_parameters(parameters):
    def versions():
        return tuple((name, id(t), t._version, t.data_ptr()) for name, t in parameters.items())
    before = versions()
    try:
        yield
    finally:
        _require(versions() == before, "functional parameter mapping changed")


def _tensor_metadata(value):
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "device": str(value.device)}


def _context(graph, ids, mask, profile, dispatch, rope_policy, dtype):
    _require(rope_policy.get("init_device") == "cpu" and rope_policy.get("init_dtype") == "float32"
             and rope_policy.get("placement_dtype") == "float32" and bool(rope_policy.get("source")),
             "source policy requires CPU FP32 RoPE and dtype-preserving placement")
    batch, length = ids.shape
    _require(batch == 1 and 0 < length <= profile["max_position_embeddings"], "unsupported query geometry")
    causal, flag, mask_profile = graph["_mask_for_attention"](
        mask, batch_size=batch, sequence_length=length, backend=dispatch.backend,
        dtype=dtype, device=ids.device)
    _require(dispatch.gqa_policy == "repeat_kv" or (causal is None and flag), "invalid grouped dispatch")
    if dispatch.backend == "sdpa" and causal is None and profile["num_key_value_groups"] > 1:
        _require(dispatch.gqa_policy == "enable_gqa", "unmasked native SDPA requires enable_gqa")
    positions = torch.arange(length, device=ids.device, dtype=torch.long).unsqueeze(0)
    inv = 1.0 / (profile["rope_theta"] ** (
        torch.arange(0, profile["head_dim"], 2, device="cpu", dtype=torch.float32) / profile["head_dim"]))
    inv = inv.to(ids.device)
    frequencies = (inv[None, :, None].expand(batch, -1, -1)
                   @ positions[:, None, :].float()).transpose(1, 2)
    rotary = torch.cat((frequencies, frequencies), dim=-1)
    return causal, flag, positions, rotary.cos().to(dtype), rotary.sin().to(dtype), mask_profile


def _block(graph, hidden, parameters, profile, index, context, dispatch, observe, lane):
    mask, causal, _, cos, sin, _ = context
    b, length, width = hidden.shape
    heads, kv, dim = (profile[x] for x in ("num_attention_heads", "num_key_value_heads", "head_dim"))
    root = f"model.layers.{index}."
    def p(name, shape):
        return graph["_check_parameter"](parameters, root + name + ".weight", shape, hidden.dtype)
    def emit(name, value):
        return _emit(observe, lane, name, value)
    emit("h", hidden)
    norm = emit("norm", graph["_rms_norm"](hidden, p("input_layernorm", (width,)), profile["rms_norm_eps"]))
    q = graph["_linear"](norm, p("self_attn.q_proj", (heads * dim, width))).view(b, length, heads, dim)
    k = graph["_linear"](norm, p("self_attn.k_proj", (kv * dim, width))).view(b, length, kv, dim)
    v = graph["_linear"](norm, p("self_attn.v_proj", (kv * dim, width))).view(b, length, kv, dim).transpose(1, 2)
    q = graph["_rms_norm"](q, p("self_attn.q_norm", (dim,)), profile["rms_norm_eps"]).transpose(1, 2)
    k = graph["_rms_norm"](k, p("self_attn.k_norm", (dim,)), profile["rms_norm_eps"]).transpose(1, 2)
    q = emit("q", q * cos.unsqueeze(1) + graph["_rotate_half"](q) * sin.unsqueeze(1))
    k = emit("k", k * cos.unsqueeze(1) + graph["_rotate_half"](k) * sin.unsqueeze(1))
    emit("v", v)
    head = emit("head", graph["_attention"](q, k, v, mask, is_causal=causal, scale=dim**-0.5,
                      groups=profile["num_key_value_groups"], dispatch=dispatch))
    projection = emit("projection", graph["_linear"](head.reshape(b, length, -1), p("self_attn.o_proj", (width, heads * dim))))
    residual = emit("R", hidden + projection)
    post = emit("postnorm", graph["_rms_norm"](residual, p("post_attention_layernorm", (width,)), profile["rms_norm_eps"]))
    inter = profile["intermediate_size"]
    gate = emit("gate", graph["_linear"](post, p("mlp.gate_proj", (inter, width))))
    up = emit("up", graph["_linear"](post, p("mlp.up_proj", (inter, width))))
    activation = emit("activation", F.silu(gate))
    product = emit("product", activation * up)
    mlp = emit("M", graph["_linear"](product, p("mlp.down_proj", (width, inter))))
    return emit("Hnext", residual + mlp)


def functional_state_history(graph, input_ids, attention_mask, parameters, config, *,
                             attention_dispatch, rope_buffer_policy, edit_index=18,
                             horizon=27, selected=(9, 18, 27), observe=None):
    """Form common and negative-H18 suffix states with no native forward/refresh.

    Callback tensors are live and must not be mutated. Only the first suffix
    block's full internal ports are emitted (plus/minus); all exits are returned.
    """
    _geometry(edit_index, horizon, selected)
    profile = graph["_native_profile"](config, horizon)
    dispatch = graph["AttentionDispatch"].from_mapping(attention_dispatch)
    _require(not graph["_config_value"](config, "attention_bias", False)
             and float(graph["_config_value"](config, "attention_dropout", 0.0)) == 0.0,
             "source requires bias-free deterministic attention")
    weight = parameters["model.embed_tokens.weight"]
    _require(weight.dtype == torch.bfloat16 and weight.device == input_ids.device, "source learned dtype/device differs")
    _require(input_ids.dtype in (torch.int32, torch.int64) and input_ids.ndim == 2, "invalid IDs")
    _require(bool((input_ids >= 0).all()) and bool((input_ids < profile["vocab_size"]).all()), "IDs outside vocabulary")
    with torch.inference_mode(), _no_modules(), _frozen_parameters(parameters):
        context = _context(graph, input_ids, attention_mask, profile, dispatch, rope_buffer_policy, weight.dtype)
        common = {0: F.embedding(input_ids, weight)}
        for index in range(horizon):
            common[index + 1] = _block(graph, common[index], parameters, profile, index, context, dispatch,
                                      observe if index == edit_index else None, "plus")
        minus = {edit_index: -common[edit_index]}
        _require(minus[edit_index].stride() == common[edit_index].stride(), "negation changed source layout")
        for index in range(edit_index, horizon):
            minus[index + 1] = _block(graph, minus[index], parameters, profile, index, context, dispatch,
                                     observe if index == edit_index else None, "minus")
        direct = torch.cat((common[selected[0]], minus[edit_index], common[horizon]), dim=-1)
        coherent = torch.cat((common[selected[0]], minus[edit_index], minus[horizon]), dim=-1)
    blocks = horizon + horizon - edit_index
    return {"schema": SCHEMA, "common": common, "minus": minus, "direct": direct, "coherent": coherent,
            "position_ids": context[2], "mask_profile": context[5],
            "profile": {"edit_index": edit_index, "horizon": horizon, "selected": list(selected),
                        "attention_dispatch": dispatch.to_dict(), "rope_buffer_policy": dict(rope_buffer_policy),
                        "intervention": "all-query all-feature BF16 negation of own residual",
                        "genuine_qwen_act": False},
            "counts": {"blocks": blocks, "embedding": 1, "linear": 7 * blocks, "RMSNorm": 4 * blocks,
                       "SDPA": blocks if dispatch.backend == "sdpa" else 0,
                       "eager_attention": blocks if dispatch.backend == "eager" else 0,
                       "rotary_applications": 2 * blocks, "SiLU_products": blocks,
                       "residual_additions": 2 * blocks, "state_negations": 1,
                       "conditioning_concats": 2, "rope_context_formations": 1,
                       "native_module_forwards": 0}}


def _versions(model):
    return tuple((name, id(t), t._version, t.data_ptr(), str(t.dtype), tuple(t.shape))
                 for name, t in (*model.named_parameters(), *model.named_buffers()))


def native_state_history(model, input_ids, attention_mask, *, authorize_native,
                         edit_index=18, horizon=27, selected=(9, 18, 27), observe=None,
                         suffix_sign=-1):
    """One full native baseline, then unchanged native suffix modules.

    authorize_native is invoked before the full model and each suffix layer. It must check
    the caller's global seal. No functional forecast tensor is accepted here.
    suffix_sign=+1 is only a sham replay qualification option. All input/context
    records below originate in this native baseline, never the forecast lane.
    """
    _geometry(edit_index, horizon, selected)
    _require(suffix_sign in (-1, 1), "suffix_sign must be +/-1")
    layers = model.model.layers
    _require(horizon <= len(layers), "horizon exceeds model")
    _require(model.config.model_type == "qwen3" and model.config._attn_implementation == "sdpa", "native Qwen3 SDPA required")
    _require(model.model.rotary_emb.inv_freq.dtype == torch.float32, "native RoPE must remain FP32")
    _require(not model.training and not model.config.attention_bias
             and float(model.config.attention_dropout) == 0.0, "native source requires eval and bias-free deterministic attention")
    _require(all(p.dtype == torch.bfloat16 for p in model.parameters()), "native learned tensors must be BF16")
    before = _versions(model)
    handles, common, consumed, kwargs_at_edit = [], {}, [], {}
    entries, block_entries, sdpa_calls = [], [], []
    active = {"lane": "baseline", "index": None}
    original_sdpa = F.scaled_dot_product_attention
    def sdpa(*args, **kwargs):
        result = original_sdpa(*args, **kwargs)
        sdpa_calls.append({"lane": active["lane"], "index": active["index"],
                           "QKV": [_tensor_metadata(x) for x in args[:3]],
                           "scale": kwargs.get("scale"), "dropout_p": kwargs.get("dropout_p", 0.0),
                           "is_causal": kwargs.get("is_causal", False),
                           "enable_gqa": kwargs.get("enable_gqa", False),
                           "mask_shape": None if (args[3] if len(args) > 3 else kwargs.get("attn_mask")) is None
                           else list((args[3] if len(args) > 3 else kwargs["attn_mask"]).shape)})
        if active["index"] == edit_index:
            lane = "plus" if active["lane"] == "baseline" else "minus"
            # Native SDPA operands may repeat KV; retain actual consumed geometry.
            for name, value in zip(("q_consumed", "k_consumed", "v_consumed"), args[:3], strict=True):
                _emit(observe, lane, name, value)
            _emit(observe, lane, "head_consumed", result)
        return result
    def entry(index):
        def hook(module, args, kwargs):
            active["index"] = index
            block_entries.append({"lane": active["lane"], "index": index})
            if active["lane"] == "baseline":
                common[index] = args[0].detach().clone()
                if index == edit_index:
                    kwargs_at_edit.update(kwargs)
            else:
                consumed.append({"index": index, "hidden": args[0].detach().clone(), "kwargs": dict(kwargs)})
            if index == edit_index:
                _emit(observe, "plus" if active["lane"] == "baseline" else "minus", "h", args[0])
        return hook
    def exit_hook(index):
        def hook(module, args, output):
            if active["lane"] == "baseline":
                common[index + 1] = output.detach().clone()
            if index == edit_index:
                _emit(observe, "plus" if active["lane"] == "baseline" else "minus", "Hnext", output)
        return hook
    names = {"input_layernorm": "norm", "self_attn.o_proj": "projection", "post_attention_layernorm": "postnorm",
             "mlp.gate_proj": "gate", "mlp.up_proj": "up", "mlp.act_fn": "activation", "mlp.down_proj": "M"}
    def port(name):
        def hook(module, args, output):
            if active["index"] == edit_index:
                lane = "plus" if active["lane"] == "baseline" else "minus"
                _emit(observe, lane, name, output)
                if name == "postnorm":
                    _emit(observe, lane, "R", args[0])
                if name == "M":
                    _emit(observe, lane, "product", args[0])
        return hook
    try:
        for name, module in model.named_modules():
            handles.append(module.register_forward_pre_hook(lambda m, a, n=name: entries.append({"lane": active["lane"], "name": n})))
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(entry(index), with_kwargs=True))
            handles.append(layer.register_forward_hook(exit_hook(index)))
        for name, label in names.items():
            module = layers[edit_index].get_submodule(name)
            handles.append(module.register_forward_hook(port(label)))
        F.scaled_dot_product_attention = sdpa
        with torch.inference_mode():
            authorize_native()
            baseline = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            _require(kwargs_at_edit.get("past_key_values") is None and not kwargs_at_edit.get("use_cache", False), "native source context unexpectedly cached")
            active["lane"] = "suffix"
            minus = {edit_index: -common[edit_index] if suffix_sign == -1 else common[edit_index].clone()}
            _require(minus[edit_index].stride() == common[edit_index].stride(), "native edit changed source layout")
            for index in range(edit_index, horizon):
                authorize_native()
                minus[index + 1] = layers[index](minus[index], **kwargs_at_edit)
            direct = torch.cat((common[selected[0]], minus[edit_index], common[horizon]), dim=-1)
            coherent = torch.cat((common[selected[0]], minus[edit_index], minus[horizon]), dim=-1)
    finally:
        F.scaled_dot_product_attention = original_sdpa
        for handle in reversed(handles):
            handle.remove()
        _require(_versions(model) == before, "native parameters/buffers changed")
    return {"schema": SCHEMA, "common": common, "minus": minus, "direct": direct, "coherent": coherent,
            "consumed": consumed, "source_context": kwargs_at_edit, "baseline_logits_shape": list(baseline.logits.shape),
            "counts": {"full_model_calls": 1, "baseline_blocks": len(layers), "suffix_blocks": horizon - edit_index,
                       "native_module_entries": len(entries), "SDPA": len(sdpa_calls),
                       "state_negations": int(suffix_sign == -1), "sham_clones": int(suffix_sign == 1),
                       "conditioning_concats": 2},
            "module_entries": entries, "block_entries": block_entries, "sdpa_calls": sdpa_calls,
            "genuine_qwen_act": False, "context_source": "own native full baseline; use_cache=False",
            "port_provenance": "R is actual postnorm input; M is actual down_proj return; Hnext actual block return. q/k/v_consumed are actual SDPA operands (KV after native repetition), head_consumed is SDPA return before transpose. No reconstructed attention residual."}


def oddness_diagnostic(plus, minus):
    """Observe numerical antisymmetry separately from signed-zero/raw-bit identity."""
    _require(plus.shape == minus.shape and plus.dtype == minus.dtype, "port schemas differ")
    expected = -plus
    delta = minus.double() - expected.double()
    bits = minus.contiguous().view(torch.uint8) != expected.contiguous().view(torch.uint8)
    zero = (plus == 0) & (minus == 0)
    return {"elements": plus.numel(), "numerical_changed": int(torch.count_nonzero(delta)),
            "max_abs": float(delta.abs().max()), "rms": float(delta.square().mean().sqrt()),
            "differing_bytes": int(bits.sum()),
            "bit_changed_elements": int(bits.reshape(-1, plus.element_size()).any(dim=-1).sum()),
            "zero_signbit_mismatches": int((zero & (torch.signbit(minus) != torch.signbit(expected))).sum()),
            "shared_zero_elements": int(zero.sum()),
            "status": "observation-only", "algebra": "SwiGLU M(-x)+M(x)=Wd(G*U) in real arithmetic; no extra projection executed"}


def writer_diagnostic(plus, minus):
    """Saved-port reductions only; never executes the extra learned Wd(G*U)."""
    result = {"status": "observation-only", "extra_learned_projections": 0,
              "MLP_even_component_max_abs": float((plus["M"].double() + minus["M"].double()).abs().max())}
    for lane, ports in (("plus", plus), ("minus", minus)):
        remainder = ports["Hnext"].double() - ports["R"].double() - ports["M"].double()
        result[lane + "_finite_addition_remainder"] = {
            "max_abs": float(remainder.abs().max()), "rms": float(remainder.square().mean().sqrt())}
    result["oddness"] = {name: oddness_diagnostic(plus[name], minus[name]) for name in plus.keys() & minus.keys()}
    return result

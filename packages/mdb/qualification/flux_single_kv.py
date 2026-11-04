"""Observe native parallel single0 and substitute only paired text K/V.

No full fused QKV/MLP projection or parameter array is retained. The attention
module return is its fused attention+MLP projection, not attention alone.
"""
from __future__ import annotations

import threading

_DISPATCH_LOCK = threading.RLock()


def validate_paired_cuts(zero, text):
    import torch
    from saturn_pub.values import describe

    if zero.model_identity != text.model_identity or zero.execution != text.execution:
        raise ValueError("single source model/execution differs")
    for cut in (zero, text):
        point = cut.execution_point
        if (point["logical_step"] != 3 or point["phase"] != "single" or point["index"] != 0 or point["local_clock"] != 0):
            raise ValueError("source must be the declared pre-single0 clock")
    if set(zero.payload) != set(text.payload) or any(describe(value) != describe(text.payload[key])
            for key, value in zero.payload.items() if key != "text"):
        raise ValueError("paired single sources differ outside working text")
    if torch.equal(zero.payload["text"], text.payload["text"]):
        raise ValueError("paired single sources must actually differ in working text")


def capture_candidate(block, cut, *, source=None, selection="native"):
    import torch
    from diffusers.models.transformers import transformer_flux2
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation

    if (cut.execution_point["phase"] != "single" or cut.execution_point["index"] != 0 or block.training
            or selection not in {"native", "K", "V", "KV"} or (selection == "native") != (source is None)):
        raise ValueError("candidate requires native single0 or a declared K/V source")
    text_rows = cut.payload["text"].shape[1]
    observed, used, ports, outputs, handles = {}, {}, {}, [], []
    original = None

    def dispatch(query, key, value, *args, **kwargs):
        if (observed or query.ndim != 4 or key.shape != query.shape or value.shape != query.shape
                or not 0 < text_rows < query.shape[1] or kwargs.get("backend") is not None or kwargs.get("parallel_config") is not None):
            raise ValueError("unsupported or repeated native single attention dispatch")
        for name, tensor in (("q", query), ("k", key), ("v", value)):
            if tensor.dtype != query.dtype or tensor.device != query.device or not bool(torch.isfinite(tensor).all()):
                raise ValueError("invalid positioned QKV profile")
            observed[name] = tensor.detach().clone()
        selected = {"q": query, "k": key, "v": value}
        if source is not None:
            if not isinstance(source, dict) or not isinstance(source.get("parent"), str) or not source["parent"] or source.get("text_rows") != text_rows:
                raise ValueError("text-bank source parent/partition missing")
            for name in ("k", "v"):
                donor = source[name]
                if (not isinstance(donor, torch.Tensor) or donor.shape != query.shape or donor.dtype != query.dtype
                        or donor.device != query.device or not bool(torch.isfinite(donor).all())):
                    raise ValueError("text-bank source geometry/dtype/device/finite contract differs")
                if name.upper() in selection:
                    selected[name] = selected[name].clone()
                    selected[name][:, :text_rows] = donor[:, :text_rows]
        used.update({name: tensor.detach().clone() for name, tensor in selected.items()})
        output = original(selected["q"], selected["k"], selected["v"], *args, **kwargs)
        outputs.append(output.detach().clone())
        ports["dispatch_profile"] = {"extra_positional_arguments": len(args), "attn_mask_present": kwargs.get("attn_mask") is not None,
            "dropout_p": float(kwargs.get("dropout_p", 0.)), "is_causal": bool(kwargs.get("is_causal", False)),
            "scale": kwargs.get("scale"), "enable_gqa": bool(kwargs.get("enable_gqa", False)),
            "attention_kwargs_present": bool(kwargs.get("attention_kwargs")),
            "backend": None if kwargs.get("backend") is None else str(kwargs["backend"]),
            "parallel_config_present": kwargs.get("parallel_config") is not None}
        return output

    def preactivation(module, args):
        if "image_mlp_preactivation" in ports:
            raise ValueError("single MLP entered more than once")
        ports["image_mlp_preactivation"] = args[0][:, text_rows:].detach().clone()

    def activation(module, args, value):
        ports["image_mlp_activation"] = value[:, text_rows:].detach().clone()

    def projection(module, args, value):
        ports["fused_projection"] = value.detach().clone()

    with _DISPATCH_LOCK:
        original = transformer_flux2.dispatch_attention_fn
        try:
            transformer_flux2.dispatch_attention_fn = dispatch
            handles = [block.attn.mlp_act_fn.register_forward_pre_hook(preactivation),
                block.attn.mlp_act_fn.register_forward_hook(activation), block.attn.register_forward_hook(projection)]
            with torch.inference_mode():
                residual = torch.cat([cut.payload["text"], cut.payload["image"]], dim=1)
                gate = Flux2Modulation.split(cut.payload["context"]["single_mod"], 1)[0][2]
                updated = block(hidden_states=residual, encoder_hidden_states=None,
                    temb_mod=cut.payload["context"]["single_mod"], image_rotary_emb=cut.payload["context"]["rope"],
                    joint_attention_kwargs=None)
                ports.update(residual=residual.detach().clone(), gate=gate.detach().clone(), updated=updated.detach().clone(),
                    image=updated[:, text_rows:].detach(), text=updated[:, :text_rows].detach())
                profile = ports.pop("dispatch_profile")
                if set(ports) != {"residual", "gate", "updated", "image", "text", "image_mlp_preactivation", "image_mlp_activation", "fused_projection"}:
                    raise ValueError("native single parallel ports incomplete")
                if any(not bool(torch.isfinite(value).all()) for value in (*ports.values(), *outputs)):
                    raise ValueError("nonfinite native single candidate")
                equation_exact = torch.equal(residual + gate * ports["fused_projection"], updated)
        finally:
            transformer_flux2.dispatch_attention_fn = original
            for handle in handles:
                handle.remove()
    if set(observed) != {"q", "k", "v"} or len(outputs) != 1:
        raise ValueError("single attention observation incomplete")
    controls = {"q_exact_recipient": torch.equal(used["q"], observed["q"])}
    for name in ("k", "v"):
        controls[name + "_image_exact_recipient"] = torch.equal(used[name][:, text_rows:], observed[name][:, text_rows:])
        expected = source[name] if source is not None and name.upper() in selection else observed[name]
        controls[name + "_text_exact_selected"] = torch.equal(used[name][:, :text_rows], expected[:, :text_rows])
    if not all(controls.values()):
        raise ValueError("native single operand selection violated declared partitions")
    return {"image": ports["image"], "text": ports["text"], "ports": ports, "observed_qkv": observed,
        "used_qkv": used, "attention_dispatch_output": outputs[0], "dispatch_calls": 1, "text_rows": text_rows,
        "input_parent": cut.fingerprint, "source_parent": None if source is None else source["parent"],
        "selection": selection, "equation_exact": equation_exact, "partition_controls": controls, "dispatch_profile": profile}


def forecast_attention_samples(zero, text, *, text_rows, query_indices):
    """Independent FP64 softmax attention, conditional on observed native QKV.

Only eight predeclared image queries are evaluated. K/V-only are novel
conditional forecasts; KV is already represented by the native text-source
image attention and is a consistency control. No model/SDPA call occurs.
"""
    import math

    import numpy as np

    shape = zero["q"].shape
    if (len(shape) != 4 or shape[0] != 1 or not 0 < text_rows < shape[1]
            or any(value.shape != shape or not np.isfinite(value).all() for source in (zero, text) for value in source.values())
            or not query_indices or len(set(query_indices)) != len(query_indices)
            or any(type(index) is not int or not 0 <= index < shape[1] - text_rows for index in query_indices)):
        raise ValueError("invalid observed QKV or fixed image-query indices")

    def readonly(value):
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)

    def bf16(value):
        bits = value.astype(np.float32).view(np.uint32)
        rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xffff0000)
        return rounded.view(np.float32)

    cases = {}
    for selection in ("K", "V", "KV"):
        output = np.empty((1, len(query_indices), shape[2], shape[3]), dtype=np.float64)
        mass = np.empty((1, shape[2], len(query_indices)), dtype=np.float64)
        for head in range(shape[2]):
            q = np.asarray(zero["q"][0, [text_rows + index for index in query_indices], head], dtype=np.float64)
            k, v = (np.array(zero[name][0, :, head], dtype=np.float64, copy=True) for name in ("k", "v"))
            if "K" in selection:
                k[:text_rows] = text["k"][0, :text_rows, head]
            if "V" in selection:
                v[:text_rows] = text["v"][0, :text_rows, head]
            logits = (q @ k.T) / math.sqrt(shape[3])
            probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            output[0, :, head] = probabilities @ v
            mass[0, head] = probabilities[:, :text_rows].sum(axis=1)
        cases[selection] = {"attention": readonly(output), "attention_bf16": readonly(bf16(output)), "text_mass": readonly(mass),
            "role": "native-text represented consistency control" if selection == "KV" else "novel conditional forecast"}
    return {"instrument_status": "applicable", "query_indices": list(query_indices), "cases": cases,
        "arithmetic": "NumPy FP64 QK/sqrt(D), stable row-softmax, probability@V; BF16 reference is FP64->FP32->round-nearest-even BF16 expanded FP32",
        "text_mass_scope": "Forecast only; native fused dispatcher does not expose probability weights",
        "scope": "Source-reviewed observed-QKV-only dataflow, not hermetic process isolation; no independent MLP or RGB forecast"}


def install_port(nav, slot, value, *, name, store):
    import torch
    from saturn_pub import Act
    from saturn_pub.values import describe

    point = nav.point
    if (point["logical_step"] != 3 or point["phase"] != "single" or point["index"] != 1
            or slot not in {"image", "text"}):
        raise ValueError("single route write requires the post-single0 clock")
    before = nav.capture()
    store.save(before)
    reference = before.payload[slot]
    if value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device or not bool(torch.isfinite(value).all()):
        raise ValueError("single route payload geometry/dtype/device/finite contract differs")
    with torch.inference_mode():
        installed = torch.empty_strided(reference.shape, reference.stride(), dtype=reference.dtype, device=reference.device)
        installed.copy_(value)
    act = Act.replace(slot, installed, name=name)
    receipt = nav.apply(act)
    store.receipt(receipt)
    after = nav.capture()
    store.save(after)
    if describe(after.payload) != describe({**before.payload, slot: installed}):
        raise ValueError("single route Act changed undeclared closure")
    return {"slot": slot, "point": point, "parent": before.fingerprint, "post_act": after.fingerprint,
        "act": act.manifest(), "receipt": receipt.to_dict(), "receipt_id": receipt.fingerprint, "undeclared_slots_exact": True}

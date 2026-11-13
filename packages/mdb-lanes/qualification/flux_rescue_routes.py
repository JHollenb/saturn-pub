"""Scoped native joint-attention observations and same-history text-K/V edit.

Q/K have already passed native normalization and RoPE at this boundary. Only
text K/V are substituted; Q and image K/V remain the recipient's actual ports.
"""
from __future__ import annotations

import threading

_DISPATCH_LOCK = threading.RLock()


def validate_paired_cuts(zero, own):
    from saturn_pub.values import describe

    if zero.model_identity != own.model_identity or zero.execution != own.execution:
        raise ValueError("paired source model/execution differs")
    for cut in (zero, own):
        point = cut.execution_point
        if (point["logical_step"] != 3 or point["phase"] != "joint" or point["index"] != 0
                or point["local_clock"] != 0):
            raise ValueError("paired sources require the declared post-project joint0 clock")
    if set(zero.payload) != set(own.payload) or any(describe(value) != describe(own.payload[key])
            for key, value in zero.payload.items() if key not in {"text", "conditioning"}):
        raise ValueError("paired sources differ outside working text/stored conditioning")


def capture_candidate(block, cut, *, joint_candidate, text_kv_source=None):
    import torch
    from diffusers.models.transformers import transformer_flux2

    if cut.execution_point["phase"] != "joint" or cut.execution_point["index"] != 0:
        raise ValueError("candidate requires joint0")
    text_rows = cut.payload["text"].shape[1]
    observed, used, dispatch_outputs = {}, {}, []
    original = None

    def dispatch(query, key, value, *args, **kwargs):
        if observed:
            raise ValueError("joint candidate dispatched attention more than once")
        if (query.ndim != 4 or key.shape != query.shape or value.shape != query.shape
                or not 0 < text_rows < query.shape[1] or kwargs.get("backend") is not None
                or kwargs.get("parallel_config") is not None):
            raise ValueError("unsupported native positioned QKV profile")
        for name, tensor in (("q", query), ("k", key), ("v", value)):
            if tensor.dtype != query.dtype or tensor.device != query.device or not bool(torch.isfinite(tensor).all()):
                raise ValueError("QKV dtype/device/finite contract differs")
            observed[name] = tensor.detach().clone()
        selected_k, selected_v = key, value
        if text_kv_source is not None:
            if (not isinstance(text_kv_source, dict) or not isinstance(text_kv_source.get("parent"), str)
                    or not text_kv_source["parent"] or text_kv_source.get("text_rows") != text_rows):
                raise ValueError("text-KV source parent/partition binding missing")
            for name, tensor in (("k", key), ("v", value)):
                source = text_kv_source[name]
                if (not isinstance(source, torch.Tensor) or source.shape != tensor.shape or source.dtype != tensor.dtype
                        or source.device != tensor.device or not bool(torch.isfinite(source).all())):
                    raise ValueError("text-KV source geometry/dtype/device/finite contract differs")
            selected_k, selected_v = key.clone(), value.clone()
            selected_k[:, :text_rows] = text_kv_source["k"][:, :text_rows]
            selected_v[:, :text_rows] = text_kv_source["v"][:, :text_rows]
        used.update(q=query.detach().clone(), k=selected_k.detach().clone(), v=selected_v.detach().clone())
        output = original(query, selected_k, selected_v, *args, **kwargs)
        if not bool(torch.isfinite(output).all()):
            raise ValueError("nonfinite native attention output")
        dispatch_outputs.append(output.detach().clone())
        return output

    with _DISPATCH_LOCK:
        original = transformer_flux2.dispatch_attention_fn
        try:
            transformer_flux2.dispatch_attention_fn = dispatch
            result = joint_candidate(block, cut, attention_gain=1., mlp_gain=1., mlp_policy="recompute")
        finally:
            transformer_flux2.dispatch_attention_fn = original
    if set(observed) != {"q", "k", "v"} or len(dispatch_outputs) != 1:
        raise ValueError("native candidate QKV observation incomplete")
    result.update(observed_qkv=observed, used_qkv=used, attention_dispatch_output=dispatch_outputs[0],
        dispatch_calls=1, text_rows=text_rows, input_parent=cut.fingerprint,
        donor_parent=None if text_kv_source is None else text_kv_source["parent"])
    result["partition_controls"] = {"q_exact_recipient": torch.equal(used["q"], observed["q"]),
        "image_k_exact_recipient": torch.equal(used["k"][:, text_rows:], observed["k"][:, text_rows:]),
        "image_v_exact_recipient": torch.equal(used["v"][:, text_rows:], observed["v"][:, text_rows:]),
        "text_k_exact_selected": torch.equal(used["k"][:, :text_rows], (observed if text_kv_source is None else text_kv_source)["k"][:, :text_rows]),
        "text_v_exact_selected": torch.equal(used["v"][:, :text_rows], (observed if text_kv_source is None else text_kv_source)["v"][:, :text_rows])}
    if not all(result["partition_controls"].values()):
        raise ValueError("native attention operand selection violated its declared recipe")
    return result


def install_route_port(nav, slot, value, *, name, store, joint_install):
    point = nav.point
    if (point["logical_step"] != 3 or (point["phase"], point["index"]) not in {("joint", 1), ("single", 0)}
            or slot not in {"image", "text"}):
        raise ValueError("route write requires declared post-joint0 or pre-single0 clock")
    return joint_install(nav, slot, value, name=name, store=store, expected_point=point)

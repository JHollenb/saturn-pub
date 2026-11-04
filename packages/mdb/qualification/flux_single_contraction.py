"""Native single6 parallel operand substitution and sampled linear forecast."""
from __future__ import annotations

BRANCHES = [
    {"id": "zero", "kind": "reference", "image_source": "zero", "text_source": "zero"},
    {"id": "text", "kind": "reference", "image_source": "text", "text_source": "text"},
    {"id": "sham", "kind": "edited", "image_source": "zero", "text_source": "zero"},
    {"id": "A.image", "kind": "edited", "image_source": "A", "text_source": None},
    {"id": "M.image", "kind": "edited", "image_source": "M", "text_source": None},
    {"id": "AM.image", "kind": "edited", "image_source": "AM", "text_source": None},
    {"id": "text.text", "kind": "edited", "image_source": None, "text_source": "text"},
    {"id": "coherent", "kind": "edited", "image_source": "text", "text_source": "text"},
    {"id": "AM.text", "kind": "edited", "image_source": "AM", "text_source": "text"}]


def validate_sources(low, high):
    import torch
    from saturn_pub.values import describe

    if low.model_identity != high.model_identity or low.execution != high.execution or set(low.payload) != set(high.payload):
        raise ValueError("single6 source identity/schema differs")
    for cut in (low, high):
        point = cut.execution_point
        if any(point[key] != value for key, value in {"logical_step": 3, "phase": "single", "index": 6, "local_clock": 6}.items()):
            raise ValueError("source requires pre-single6 clock")
    if any(describe(value) != describe(high.payload[key]) for key, value in low.payload.items() if key not in {"image", "text"}):
        raise ValueError("single6 sources differ outside working image/text")
    if any(torch.equal(low.payload[key], high.payload[key]) for key in ("image", "text")):
        raise ValueError("single6 source image and text must both actually differ")


def capture_candidate(block, cut, *, source=None, selection="native"):
    import torch
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation

    point = cut.execution_point
    if (any(point[key] != value for key, value in {"logical_step": 3, "phase": "single", "index": 6, "local_clock": 6}.items())
            or block.training or selection not in {"native", "A", "M", "AM"} or (source is None) != (selection == "native")):
        raise ValueError("candidate requires declared pre-single6 clock/native/A/M/AM profile")
    text_rows, width = cut.payload["text"].shape[1], cut.payload["image"].shape[-1]
    ports, handles = {}, []
    def before(module, args):
        if ports or len(args) != 1:
            raise ValueError("single6 projection entered repeatedly or with unsupported args")
        original = args[0]
        if original.ndim != 3 or original.shape[0] != 1 or original.shape[2] <= width:
            raise ValueError("single6 native concatenation geometry differs")
        used = original
        if source is not None:
            donor = source["concat"]
            if (not source.get("parent") or source["text_rows"] != text_rows or source["attention_width"] != width
                    or donor.shape != original.shape or donor.dtype != original.dtype or donor.device != original.device
                    or not bool(torch.isfinite(donor).all())):
                raise ValueError("single6 selected operand geometry/dtype/device/finite differs")
            used = original.clone()
            if "A" in selection:
                used[:, text_rows:, :width] = donor[:, text_rows:, :width]
            if "M" in selection:
                used[:, text_rows:, width:] = donor[:, text_rows:, width:]
        ports.update(original_concat=original.detach().clone(), used_concat=used.detach().clone())
        return (used,)
    def after(module, args, value):
        ports["projection"] = value.detach().clone()
    try:
        handles = [block.attn.to_out.register_forward_pre_hook(before), block.attn.to_out.register_forward_hook(after)]
        with torch.inference_mode():
            residual = torch.cat([cut.payload["text"], cut.payload["image"]], dim=1)
            gate = Flux2Modulation.split(cut.payload["context"]["single_mod"], 1)[0][2]
            updated = block(hidden_states=residual, encoder_hidden_states=None, temb_mod=cut.payload["context"]["single_mod"],
                image_rotary_emb=cut.payload["context"]["rope"], joint_attention_kwargs=None)
            ports.update(residual=residual, gate=gate.detach().clone(), updated=updated.detach().clone(),
                image=updated[:, text_rows:].detach(), text=updated[:, :text_rows].detach())
    finally:
        for handle in handles:
            handle.remove()
    if set(ports) != {"original_concat", "used_concat", "projection", "residual", "gate", "updated", "image", "text"}:
        raise ValueError("single6 native operands incomplete")
    if any(not bool(torch.isfinite(value).all()) for value in ports.values()):
        raise ValueError("single6 observed nonfinite native operand")
    controls = {"text_concat_exact_recipient": torch.equal(ports["used_concat"][:, :text_rows], ports["original_concat"][:, :text_rows])}
    for name, sl in (("A", slice(None, width)), ("M", slice(width, None))):
        expected = source["concat"] if source is not None and name in selection else ports["original_concat"]
        controls[name + "_image_exact_selected"] = torch.equal(ports["used_concat"][:, text_rows:, sl], expected[:, text_rows:, sl])
    if not all(controls.values()):
        raise ValueError("single6 operand substitution violated its declared footprint")
    return {"image": ports["image"], "text": ports["text"], "input_parent": cut.fingerprint, "source_parent": source["parent"] if source is not None else None,
        "selection": selection, "text_rows": text_rows, "attention_width": width, "ports": ports,
        "equation_exact": torch.equal(ports["residual"] + ports["gate"] * ports["projection"], ports["updated"]),
        "partition_controls": controls}


def forecast_samples(low, high, weight_rows, bias_rows, *, query_indices, output_indices, text_rows, attention_width):
    """FP64 full concatenated projection and explicit BF16 rounding variants."""
    import numpy as np

    concat, donor = low["concat"], high["concat"]
    residual, gate = low["residual"], low["gate"]
    shape = concat.shape
    if (len(shape) != 3 or shape[0] != 1 or donor.shape != shape or not 0 < text_rows < shape[1]
            or not 0 < attention_width < shape[2] or residual.shape != (1, shape[1], attention_width)
            or gate.shape not in {(1, 1, attention_width), residual.shape}
            or weight_rows.shape != (len(output_indices), shape[2])
            or (bias_rows is not None and bias_rows.shape != (len(output_indices),))
            or not query_indices or not output_indices or len(set(query_indices)) != len(query_indices)
            or len(set(output_indices)) != len(output_indices)
            or any(type(index) is not int or not 0 <= index < shape[1] - text_rows for index in query_indices)
            or any(type(index) is not int or not 0 <= index < attention_width for index in output_indices)
            or any(value.dtype != np.float32 for value in (concat, donor, residual, gate, weight_rows))
            or (bias_rows is not None and bias_rows.dtype != np.float32)
            or any(not np.isfinite(value).all() for value in (concat, donor, residual, gate, weight_rows))
            or (bias_rows is not None and not np.isfinite(bias_rows).all())):
        raise ValueError("invalid sampled native projection source/indices/weight geometry")
    def readonly(value):
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    def bf16(value):
        bits = np.asarray(value, dtype=np.float32).view(np.uint32)
        return ((bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xffff0000)).view(np.float32)
    positions = [text_rows + index for index in query_indices]
    r = np.asarray(residual[:, positions][:, :, output_indices], dtype=np.float64)
    g = np.asarray(np.broadcast_to(gate, residual.shape)[:, positions][:, :, output_indices], dtype=np.float64)
    cases = {}
    for selection in ("A", "M", "AM"):
        value = np.array(concat[:, positions], dtype=np.float64, copy=True)
        selected = donor[:, positions]
        if "A" in selection:
            value[:, :, :attention_width] = selected[:, :, :attention_width]
        if "M" in selection:
            value[:, :, attention_width:] = selected[:, :, attention_width:]
        projection = value @ np.asarray(weight_rows, dtype=np.float64).T
        if bias_rows is not None:
            projection += np.asarray(bias_rows, dtype=np.float64)
        p16 = bf16(projection)
        product16 = bf16(g * p16)
        updated16 = bf16(r + product16)
        if not np.isfinite(updated16).all() or not np.isfinite(projection).all():
            raise ValueError("nonfinite sampled projection forecast")
        cases[selection] = {name: readonly(value) for name, value in {
            "projection_fp64": projection, "projection_bf16": p16, "product_bf16": product16,
            "updated_fp64": r + g * projection, "updated_bf16": updated16}.items()}
    return {"query_indices": list(query_indices), "output_indices": list(output_indices), "cases": cases,
        "recipient_residual_samples": readonly(r), "recipient_gate_samples": readonly(g),
        "arithmetic": "FP64 full selected concat @ actual weight rows.T + bias. BF16 variant rounds projection FP64→FP32→RNE-BF16, then gate product and residual addition with the same cast; a numerical reference, not a bit-exact CUDA GEMM emulator.",
        "scope": "Conditional sampled native fused-linear/gated-write forecast from two observed source operands, not attention, raw-prompt or final-RGB prediction"}


def install_port(nav, slot, value, *, name, store):
    import torch
    from saturn_pub import Act
    from saturn_pub.values import describe

    if (slot not in {"image", "text"} or any(nav.point[key] != expected for key, expected in
            {"logical_step": 3, "phase": "single", "index": 7, "local_clock": 7}.items())):
        raise ValueError("single6 image/text Act requires post-single6 clock")
    before = nav.capture()
    reference = before.payload[slot]
    if value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device or not bool(torch.isfinite(value).all()):
        raise ValueError("single6 installed operand geometry/dtype/device/finite differs")
    with torch.inference_mode():
        payload = torch.empty_strided(reference.shape, reference.stride(), dtype=reference.dtype, device=reference.device)
        payload.copy_(value)
    store.save(before)
    act = Act.replace(slot, payload, name=name)
    receipt = nav.apply(act)
    after = nav.capture()
    store.save(after)
    if describe(after.payload) != describe({**before.payload, slot: payload}):
        raise ValueError("single6 Act changed undeclared state")
    return {"slot": slot, "point": nav.point, "parent": before.fingerprint, "post_act": after.fingerprint,
        "act": act.manifest(), "receipt_id": receipt.fingerprint, "receipt": receipt.to_dict(), "undeclared_slots_exact": True}

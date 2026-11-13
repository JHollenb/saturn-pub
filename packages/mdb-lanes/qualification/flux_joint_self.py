"""Recipient-only native sequential joint writes and explicitly live VAE previews.

Direct block calculations use scoped native hooks and are counted separately
from unmodified public adapter receipts. No donor tensor or second prompt enters
the candidate. A held MLP port is bound to this same observed parent.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


def joint_candidate_self(block, cut, *, attention_gain=1., mlp_gain=1., mlp_policy="recompute", held_mlp=None):
    import torch
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation

    if any(type(value) not in {float, int} or not math.isfinite(value) or not 0 <= value <= 1
           for value in (attention_gain, mlp_gain)):
        raise ValueError("joint gains must be finite values in [0,1]")
    if mlp_policy not in {"recompute", "hold-native"}:
        raise ValueError("unknown own MLP policy")
    payload = cut.payload
    if cut.execution_point["phase"] != "joint" or block.training:
        raise ValueError("candidate requires an observed joint entry and eval block")
    held = None
    if mlp_policy == "hold-native":
        if (not isinstance(held_mlp, Mapping) or held_mlp.get("parent") != cut.fingerprint
                or held_mlp.get("attention_gain") != 1. or "value" not in held_mlp):
            raise ValueError("held MLP must come from this same parent's native attention path")
        held = held_mlp["value"]
        if (not isinstance(held, torch.Tensor) or held.shape != payload["image"].shape
                or held.dtype != payload["image"].dtype or held.device != payload["image"].device
                or not bool(torch.isfinite(held).all())):
            raise ValueError("held own MLP tensor contract differs")
    elif held_mlp is not None:
        raise ValueError("recompute policy does not accept a held tensor")
    ports, handles = {}, []

    def attention(module, args, output):
        if "attention_raw" in ports:
            raise ValueError("native image attention entered more than once")
        ports["attention_raw"] = output[0].detach().clone()
        ports["attention_used"] = (attention_gain * ports["attention_raw"]).detach().clone()
        return ports["attention_used"].clone(), output[1]

    def intermediate(module, args):
        ports["intermediate"] = args[0].detach().clone()

    def ff_input(module, args):
        ports["mlp_input"] = args[0].detach().clone()

    def ff_output(module, args, output):
        if "mlp_raw" in ports:
            raise ValueError("native image MLP entered more than once")
        ports["mlp_raw"] = output.detach().clone()
        ports["mlp_selected"] = (output if held is None else held).detach().clone()
        ports["mlp_used"] = (mlp_gain * ports["mlp_selected"]).detach().clone()
        return ports["mlp_used"].clone()

    try:
        handles = [block.attn.register_forward_hook(attention), block.norm2.register_forward_pre_hook(intermediate),
                   block.ff.register_forward_pre_hook(ff_input), block.ff.register_forward_hook(ff_output)]
        context = payload["context"]
        with torch.inference_mode():
            (_, _, ga), (_, _, gm) = Flux2Modulation.split(context["img_mod"], 2)
            text, image = block(hidden_states=payload["image"], encoder_hidden_states=payload["text"],
                temb_mod_img=context["img_mod"], temb_mod_txt=context["txt_mod"], image_rotary_emb=context["rope"],
                joint_attention_kwargs=None)
            required = {"attention_raw", "attention_used", "intermediate", "mlp_input", "mlp_raw", "mlp_selected", "mlp_used"}
            if set(ports) != required:
                raise ValueError("native joint ports missing")
            ports.update(residual=payload["image"].clone(), input_text=payload["text"].clone(),
                gate_attention=ga.clone(), gate_mlp=gm.clone(),
                attention_write=(ga * ports["attention_used"]).clone(), mlp_write=(gm * ports["mlp_used"]).clone(),
                image=image.detach().clone(), text=text.detach().clone())
            if (not torch.equal(payload["image"] + ports["attention_write"], ports["intermediate"])
                    or not torch.equal(ports["intermediate"] + ports["mlp_write"], image)):
                raise ValueError("native BF16 sequential joint equation does not close")
        result = {"image": image, "text": text, "ports": ports, "equation_exact": True,
                  "attention_gain": float(attention_gain), "mlp_gain": float(mlp_gain), "mlp_policy": mlp_policy,
                  "arithmetic": "native output hooks Aused=gainA*Araw; Mused=gainM*Mselected; U=R+gateA*Aused; I=U+gateM*Mused; original context/gates unchanged; native dtype rounding at each operation"}
        if attention_gain == 1. and mlp_policy == "recompute":
            result["held_mlp"] = {"parent": cut.fingerprint, "attention_gain": 1., "value": ports["mlp_raw"].clone()}
        return result
    finally:
        for handle in handles:
            handle.remove()


def install_joint_port(nav, slot, value, *, name, store, expected_point):
    from saturn_pub import Act
    from saturn_pub.values import describe

    if slot not in {"image", "text"} or nav.point != dict(expected_point):
        raise ValueError("joint post-block write slot or clock differs")
    before = nav.capture()
    store.save(before)
    act = Act.replace(slot, value, name=name)
    receipt = nav.apply(act)
    store.receipt(receipt)
    after = nav.capture()
    store.save(after)
    if describe(after.payload) != describe({**before.payload, slot: value}):
        raise ValueError("typed joint port changed undeclared closure")
    return {"slot": slot, "point": dict(expected_point), "parent": before.fingerprint, "post_act": after.fingerprint,
            "act": act.manifest(), "receipt": receipt.to_dict(), "receipt_id": receipt.fingerprint,
            "undeclared_slots_exact": True}


def decode_live_preview(pipe, cut, *, height, width):
    """Native VAE on a captured live latent; no fake completed cut or denoising.

    Mirrors the public Klein adapter's native unpack/BN/unpatchify/VAE/postprocess
    path. Caller counts every direct VAE call, including final-cut parity checks.
    This partial noised latent is not a completed diffusion output.
    """
    import torch

    payload = cut.payload
    point = cut.execution_point
    if point["phase"] != "project" or point.get("local_clock") != 0:
        raise ValueError("preview requires a post-scheduler/pre-project cut")
    with torch.inference_mode():
        latent = pipe._unpack_latents_with_ids(payload["latent"], payload["img_ids"],
            height // (pipe.vae_scale_factor * 2), width // (pipe.vae_scale_factor * 2))
        mean = pipe.vae.bn.running_mean.detach().cpu().view(1, -1, 1, 1).to(latent)
        std = (pipe.vae.bn.running_var.detach().cpu().view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).sqrt().to(latent)
        latent = pipe._unpatchify_latents(latent * std + mean)
        decoded = pipe.vae.decode(latent.to(next(pipe.vae.parameters()).device), return_dict=False)[0]
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]

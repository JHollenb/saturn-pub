"""Native joint image-port substitutions, installed into typed image-only Acts.

Hooks are scoped to a direct, counted candidate-block calculation. Adapter
continuations are always unmodified native execution; edited blocks are recorded
as calculations followed by Acts, never mislabeled as native receipts.
"""
from __future__ import annotations


def joint_candidate(block, cut, *, attention=None, mlp=None):
    import torch
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation

    payload, ports, handles = cut.payload, {}, []

    def attn_hook(module, args, output):
        ports["attention"] = output[0].detach().clone()
        return output if attention is None else (attention, output[1])

    def mlp_hook(module, args, output):
        ports["mlp"] = output.detach().clone()
        return output if mlp is None else mlp

    for value in (attention, mlp):
        if value is not None and (value.shape != payload["image"].shape or value.dtype != payload["image"].dtype
                                  or value.device != payload["image"].device):
            raise ValueError("joint replacement port contract differs")
    try:
        handles.append(block.attn.register_forward_hook(attn_hook))
        handles.append(block.ff.register_forward_hook(mlp_hook))
        context = payload["context"]
        with torch.inference_mode():
            text, image = block(hidden_states=payload["image"], encoder_hidden_states=payload["text"],
                temb_mod_img=context["img_mod"], temb_mod_txt=context["txt_mod"],
                image_rotary_emb=context["rope"], joint_attention_kwargs=None)
            (_, _, ga), (_, _, gm) = Flux2Modulation.split(context["img_mod"], 2)
            ports.update(gate_attention=ga.detach().clone(), gate_mlp=gm.detach().clone())
            a = ports["attention"] if attention is None else attention
            m = ports["mlp"] if mlp is None else mlp
            equation = (payload["image"] + ga * a) + gm * m
            if not torch.equal(equation, image):
                raise ValueError("native sequential BF16 joint equation does not close")
        return {"text": text, "image": image, "ports": ports,
                "equation_exact": True, "attention_replaced": attention is not None, "mlp_replaced": mlp is not None}
    finally:
        for handle in handles:
            handle.remove()


def install_image(nav, value, *, name, store):
    from saturn_pub import Act
    from saturn_pub.values import describe

    before = nav.capture()
    store.save(before)
    act = Act.replace("image", value, name=name)
    receipt = nav.apply(act)
    after = nav.capture()
    store.save(after)
    if describe(after.payload) != describe({**before.payload, "image": value}):
        raise ValueError("image write changed undeclared closure")
    return {"parent": before.fingerprint, "post_act": after.fingerprint,
            "act": act.manifest(), "receipt": receipt.to_dict(), "undeclared_slots_exact": True}

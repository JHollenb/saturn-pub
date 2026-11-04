"""Own-noise-relative latent and stored-conditioning recipes at project cuts.

N2 is a derived zero-endpoint noise anchor, and R2 is its residual to the
observed latent. Neither name asserts memory erasure or a model marginal.
"""
from __future__ import annotations


def validate_sources(initial, parent):
    import torch
    from saturn_pub.values import describe

    if initial.model_identity != parent.model_identity or initial.execution != parent.execution:
        raise ValueError("own source model/execution differs")
    for cut, step in ((initial, 0), (parent, 2)):
        point, payload = cut.execution_point, cut.payload
        if (point["logical_step"] != step or point["phase"] != "project" or point["local_clock"] != 0
                or payload["step"] != step or payload["cursor"] != 0
                or any(payload[name] is not None for name in ("text", "image", "context", "noise_prediction"))):
            raise ValueError("own source is not the declared closed project frontier")
        if (len(payload["timesteps"]) != 4 or payload["sigmas"].shape != (5,)
                or payload["sigmas"].dtype != torch.float32 or float(payload["sigmas"][0]) != 1.):
            raise ValueError("source is not the declared four-step unit-noise schedule")
    if set(initial.payload) != set(parent.payload) or any(
            describe(value) != describe(parent.payload[key]) for key, value in initial.payload.items()
            if key not in {"step", "latent"}):
        raise ValueError("own initial and carried sources differ outside latent/step")
    z0, z2 = initial.payload["latent"], parent.payload["latent"]
    if (z0.shape != z2.shape or z0.stride() != z2.stride() or z0.dtype != z2.dtype or z0.device != z2.device
            or z2.dtype != torch.bfloat16 or not bool(torch.isfinite(z0).all()) or not bool(torch.isfinite(z2).all())):
        raise ValueError("own latent source geometry/dtype/device differs")


def carrier_payloads(initial, parent, *, alpha, beta):
    import torch

    validate_sources(initial, parent)
    if any(type(value) not in {int, float} or value not in {0., .5, 1.} for value in (alpha, beta)):
        raise ValueError("declared alpha/beta must be 0, .5 or 1")
    z0, z2, own_c = initial.payload["latent"], parent.payload["latent"], parent.payload["conditioning"]

    def layout(value, reference):
        target = torch.empty_strided(reference.shape, reference.stride(), dtype=reference.dtype, device=reference.device)
        target.copy_(value.to(reference.dtype))
        return target

    with torch.inference_mode():
        sigma = parent.payload["sigmas"][2].to(torch.float32)
        anchor = sigma * z0.to(torch.float32)
        residual = z2.to(torch.float32) - anchor
        latent = z2.clone(memory_format=torch.preserve_format) if alpha == 1. else layout(anchor + float(alpha) * residual, z2)
        conditioning = own_c.clone(memory_format=torch.preserve_format) if beta == 1. else layout(float(beta) * own_c.to(torch.float32), own_c)
    return {"latent": latent, "conditioning": conditioning, "anchor": anchor, "residual": residual,
        "recipe": {"alpha": float(alpha), "beta": float(beta), "sigma_index": 2, "sigma2": float(sigma),
            "native_latent_copy": alpha == 1., "native_conditioning_copy": beta == 1.,
            "latent_stride": list(z2.stride()), "conditioning_stride": list(own_c.stride()),
            "arithmetic": "N2=F32(sigma[2])*F32(Z0); R2=F32(L2)-N2; alpha1 exact native copy, otherwise QBF16(N2+F32(alpha)*R2); beta1 exact own-C copy, otherwise QBF16(F32(beta)*F32(Cown)); original strides retained",
            "interpretation": "N2 derived zero-endpoint own-noise anchor; R2 anchor-relative latent residual, not proved memory erasure"}}


def install_carrier_port(nav, slot, value, *, name, store, expected_step):
    from saturn_pub import Act
    from saturn_pub.values import describe

    point = nav.point
    if (slot not in {"latent", "conditioning"} or point["logical_step"] != expected_step
            or point["phase"] != "project" or point["local_clock"] != 0 or expected_step not in {2, 3}):
        raise ValueError("carrier write requires its declared live project clock")
    before = nav.capture()
    store.save(before)
    act = Act.replace(slot, value, name=name)
    receipt = nav.apply(act)
    store.receipt(receipt)
    after = nav.capture()
    store.save(after)
    if describe(after.payload) != describe({**before.payload, slot: value}):
        raise ValueError("carrier Act changed undeclared closure")
    return {"slot": slot, "point": point, "parent": before.fingerprint, "post_act": after.fingerprint,
        "act": act.manifest(), "receipt": receipt.to_dict(), "receipt_id": receipt.fingerprint,
        "undeclared_slots_exact": True}

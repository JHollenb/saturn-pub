"""Validate the metadata-delta route on real or tiny weights.

The route lifecycle is one flow for every adapter:

1. Step a *donor* parent and a held-out *recipient* parent to the same boundary.
2. Measure the delta as donor carrier minus recipient carrier, and save it as a
   ``DeltaCard`` -- metadata plus the content address (sha256 + shape + dtype) of
   the delta tensor, never the bytes.
3. ``MetadataDeltaRoute.select`` the card by tag from metadata only (no bytes),
   then ``hydrate`` it twice through a caller-owned resolver to show the
   deduplicated, content-verified, byte-accounted custody.
4. ``apply_delta`` forks candidate and native branches from the recipient parent,
   applies the delta, continues, compares, and restores the parent exactly.
5. Record apply/rollback exactness, the measured carrier effect, and the route
   custody receipt.

``--adapter decoder`` drives a native HF decoder (``hidden`` carrier);
``--adapter flux2`` drives FLUX.2 Klein block suffixes (``text`` carrier). It
imports only ``saturn_pub`` and public frameworks. No scheduler code lives here.

    python run.py --adapter decoder --tiny --output out
    python run.py --adapter decoder --model Qwen/Qwen2.5-0.5B --device cuda --output out
    python run.py --adapter flux2 --model /path/FLUX.2-klein-4B --device cuda \
        --residency streamed --output out
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from saturn_pub import Act
from saturn_pub.route import DeltaCard, MetadataDeltaRoute, apply_delta

STEP_DTYPE = torch.bfloat16


def _cuda() -> bool:
    return torch.cuda.is_available()


def _peak_vram_mb() -> float | None:
    return None if not _cuda() else torch.cuda.max_memory_allocated() / 2**20


def _rss_mb() -> float | None:
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 2**10
    except Exception:
        return None


def _free() -> None:
    gc.collect()
    if _cuda():
        torch.cuda.empty_cache()


def _carrier_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).norm())


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


# --------------------------------------------------------------------------------------
# Decoder adapter: hidden-carrier delta
# --------------------------------------------------------------------------------------
def build_decoder(args):
    from transformers import AutoConfig

    from saturn_pub.adapters.decoder import DecoderAdapter, families

    if args.tiny:
        return DecoderAdapter.tiny("qwen2")
    model_type = AutoConfig.from_pretrained(
        args.model, local_files_only=args.local_files_only
    ).model_type
    spec = families()[model_type]
    model = spec.model_cls.from_pretrained(
        args.model,
        attn_implementation="eager",
        torch_dtype=_dtype(args.dtype),
        local_files_only=args.local_files_only,
    ).to(args.device)
    return DecoderAdapter(model, granularity="layer")


def decoder_spec(args):
    cap = 60 if args.tiny else 150_000
    donor_tokens = [t % cap for t in (3, 9, 1, 5, 7, 2)]
    recipient_tokens = [t % cap for t in (5, 7, 11, 13, 2, 29)]
    return {
        "address": "hidden",
        "boundary_steps": 1,
        "continuation_steps": args.continuation_steps or 4,
        "make_session": lambda adapter, which: adapter.session(
            donor_tokens if which == "donor" else recipient_tokens
        ),
        "family": "qwen2" if args.tiny else "decoder",
        "site": "layer:0",
        "stream": "hidden",
    }


# --------------------------------------------------------------------------------------
# FLUX.2 Klein adapter: text-carrier delta (reuses the public diffusion encode path)
# --------------------------------------------------------------------------------------
def _session_keys():
    return ["latent", "conditioning", "img_ids", "txt_ids", "timesteps", "sigmas"]


def encode_klein4b_two_latents(model_path, prompt, seeds, size, steps):
    """Encode one prompt, build two latents (donor/recipient seeds) sharing everything else.

    Same prompt -> identical conditioning shape, so the two ``text`` carriers are subtractable;
    different latents -> the joint blocks mix a different image into text, so the delta is nonzero.
    """
    import numpy as np
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )

    config = json.loads((Path(model_path) / "transformer" / "config.json").read_text())
    in_channels = int(config["in_channels"])
    device = "cuda" if _cuda() else "cpu"
    pipe = Flux2KleinPipeline.from_pretrained(
        model_path, transformer=None, torch_dtype=STEP_DTYPE, local_files_only=True
    )
    pipe.text_encoder.to(device)
    prompt_embeds = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1)[0]
    txt_ids = pipe._prepare_text_ids(prompt_embeds).to(device)
    encs = []
    img_ids = timesteps = sigmas = None
    for seed in seeds:
        generator = torch.Generator("cpu").manual_seed(seed)
        latent, this_img_ids = pipe.prepare_latents(
            1, in_channels // 4, size, size, prompt_embeds.dtype, device, generator
        )
        if img_ids is None:
            img_ids = this_img_ids
            raw_sigmas = (
                None
                if pipe.scheduler.config.get("use_flow_sigmas", False)
                else np.linspace(1.0, 0.25, steps)
            )
            timesteps, _ = retrieve_timesteps(
                pipe.scheduler,
                steps,
                device,
                sigmas=raw_sigmas,
                mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=steps),
            )
            sigmas = pipe.scheduler.sigmas
        enc = dict(
            latent=latent,
            conditioning=prompt_embeds,
            img_ids=img_ids,
            txt_ids=txt_ids,
            timesteps=timesteps,
            sigmas=sigmas,
        )
        encs.append(
            {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}
        )
    del pipe
    _free()
    return encs


def build_flux2(args):
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    if args.tiny:
        from diffusers import FlowMatchEulerDiscreteScheduler

        adapter = Flux2KleinAdapter.tiny(residency=args.residency, device=args.device)
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(args.steps)

        def make_inputs(seed):
            generator = torch.Generator().manual_seed(seed)
            return dict(
                latent=torch.randn(1, 8, 5, generator=generator).transpose(1, 2),
                conditioning=torch.randn(1, 3, 24, generator=generator),
                img_ids=torch.zeros(1, 5, 4),
                txt_ids=torch.zeros(1, 3, 4),
                timesteps=scheduler.timesteps,
                sigmas=scheduler.sigmas,
            )

        return adapter, make_inputs

    from diffusers import Flux2KleinPipeline

    # Encode first (encoder-only pipeline), free it, then load the stepping transformer. This
    # keeps peak VRAM at the encoder, never encoder + transformer together.
    donor_enc, recipient_enc = encode_klein4b_two_latents(
        args.model, args.prompt, (args.seed, args.seed + 1), args.size, args.steps
    )
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model, text_encoder=None, tokenizer=None, torch_dtype=STEP_DTYPE, local_files_only=True
    )
    pipe.vae.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, residency=args.residency, device=args.device)
    return adapter, (donor_enc, recipient_enc)


def flux2_spec(args):
    return {
        "address": "text",
        "boundary_steps": 4,
        "continuation_steps": args.continuation_steps or 1,
        "family": "flux2-klein",
        "site": "joint.2",
        "stream": "text",
    }


# --------------------------------------------------------------------------------------
# Shared route lifecycle
# --------------------------------------------------------------------------------------
def run_route(adapter, spec, sessions, *, source_handle):
    address = spec["address"]
    donor, recipient = sessions
    for _ in range(spec["boundary_steps"]):
        donor.continue_()
    for _ in range(spec["boundary_steps"]):
        recipient.continue_()

    parent = recipient.capture()
    base = recipient.read(address)
    donor_value = donor.read(address)
    delta = (donor_value.detach().float() - base.detach().float()).to(base.dtype)

    act = Act.add(address, delta, dose=1.0)
    card = DeltaCard.from_act(
        card_id=f"{spec['family']}-{spec['site']}-{spec['stream']}",
        parent=parent,
        act=act,
        tags={
            "family": spec["family"],
            "site": spec["site"],
            "stream": spec["stream"],
            "step": 0,
            "role": "family_delta",
        },
        source_handle=source_handle,
        effect={"source": "donor_minus_recipient_carrier"},
    )
    declared = card.ref("value")["bytes"]
    route = MetadataDeltaRoute([card], transport={"dense_bytes_estimate": declared * 64})

    selected = route.select(family=spec["family"], stream=spec["stream"])
    store = {card.ref("value")["sha256"]: delta}
    calls: list[str] = []

    def resolver(ref):
        calls.append(ref["sha256"])
        return store[ref["sha256"]]

    hydrated = route.hydrate(selected[0], resolver)
    route.hydrate(selected[0], resolver)  # deduplicated: no second resolve
    dedup_ok = calls == [card.ref("value")["sha256"]]

    k = spec["continuation_steps"]
    result = apply_delta(
        recipient,
        hydrated,
        continuation_steps=k,
        evaluator=lambda s: {"carrier_l2_norm": float(s.read(address).detach().float().norm())},
    )

    # Measured effect: re-fork candidate and native from the same parent and diff the carrier.
    native = recipient.fork(result["parent"])
    candidate = recipient.fork(result["parent"])
    candidate.apply(hydrated)
    for _ in range(k):
        native.continue_()
        candidate.continue_()
    effect_l2 = _carrier_l2(candidate.read(address), native.read(address))

    receipt = result["receipt"]
    return {
        "address": address,
        "selected": [c.card_id for c in selected],
        "card_fingerprint": card.fingerprint,
        "route_fingerprint": route.fingerprint,
        "artifact_ref": {k2: card.ref("value")[k2] for k2 in ("sha256", "shape", "dtype", "bytes")},
        "dedup_ok": dedup_ok,
        "route_receipt": route.receipt(),
        "changed_vs_native": receipt["effect"]["changed_vs_native"],
        "effect_carrier_l2": effect_l2,
        "rollback": {
            "verified_exact": receipt["rollback"]["verified_exact"],
            "matches_parent": receipt["rollback"]["matches_parent"],
            "restored_slots": receipt["rollback"]["restored_slots"],
            "restored_elements": receipt["rollback"]["restored_elements"],
        },
        "raw_payloads_embedded": receipt["raw_payloads_embedded"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=("decoder", "flux2"), required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--residency", default="streamed", choices=("resident", "streamed"))
    parser.add_argument("--continuation-steps", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=611)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--prompt", default="a photorealistic red fox sitting in fresh snow at dawn"
    )
    parser.add_argument("--output", default="metadata-delta-route")
    args = parser.parse_args()

    started = time.time()
    if args.adapter == "decoder":
        adapter = build_decoder(args)
        spec = decoder_spec(args)
        donor = spec["make_session"](adapter, "donor")
        recipient = spec["make_session"](adapter, "recipient")
    else:
        adapter, built = build_flux2(args)
        spec = flux2_spec(args)
        if args.tiny:
            make_inputs = built
            donor = adapter.session(**make_inputs(23))
            recipient = adapter.session(**make_inputs(11))
        else:
            donor_enc, recipient_enc = built
            # The adapter's session() places each slot on its device/dtype; pass enc directly
            # (pre-casting the flow schedule breaks the adapter's schedule dtype/device check).
            donor = adapter.session(**{k: donor_enc[k] for k in _session_keys()})
            recipient = adapter.session(**{k: recipient_enc[k] for k in _session_keys()})

    source_handle = "store://metadata-delta-route"
    route_report = run_route(adapter, spec, (donor, recipient), source_handle=source_handle)

    report = {
        "schema": "saturn-pub-metadata-delta-route-validation-v1",
        "adapter": args.adapter,
        "tiny": args.tiny,
        "model_identity": adapter.model_identity,
        "execution": dict(adapter.execution),
        "device": args.device,
        "dtype": args.dtype if args.adapter == "decoder" else str(STEP_DTYPE),
        "residency": None if args.adapter == "decoder" else args.residency,
        "route": route_report,
        "wall_s": round(time.time() - started, 3),
        "peak_vram_mb": _peak_vram_mb(),
        "peak_rss_mb": _rss_mb(),
        "limitations": [
            "Exactness is about the mechanics of one program on one device and dtype.",
            "The measured effect is a carrier change signal, not a semantic label.",
            "Decoder prompts are fixed token ids, not a tokenized natural-language battery.",
        ],
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("METADATA_DELTA_ROUTE_REPORT=" + json.dumps(report))
    ok = (
        route_report["rollback"]["verified_exact"]
        and route_report["rollback"]["matches_parent"]
        and route_report["changed_vs_native"]
        and route_report["dedup_ok"]
        and not route_report["raw_payloads_embedded"]
    )
    if not ok:
        raise SystemExit("metadata-delta route validation failed its exactness checks")


if __name__ == "__main__":
    main()

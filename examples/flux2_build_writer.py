"""Build a recipient writer from paired native FLUX.2 Klein traces.

Run this first, then flux2_hotfix.py with the resulting writer.npz. The base
pipeline renders with CFG; Saturn captures only its first conditional block prefix.
No claim is made that the adapter executes the base pipeline's complete CFG loop.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from flux2_hotfix import count_red, place_for_blocks

from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.store import LocalStore
from saturn_pub.values import describe
from saturn_pub.writers import fit_low_rank_writer, save_low_rank_writer


@torch.inference_mode()
def capture(args, model, revision, name, steps, guidance):
    root = Path(args.output)
    pipe = Flux2KleinPipeline.from_pretrained(
        model,
        revision=revision,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_model_cpu_offload()
    captured = {}
    calls = 0

    def release_encoder(module, _inputs, output):
        module.to("cpu")
        torch.cuda.empty_cache()
        return output

    def release_denoiser(_module, _inputs):
        pipe.transformer.to("cpu")
        torch.cuda.empty_cache()

    def enter(_module, _inputs, kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            captured["inputs"] = {
                key: kwargs[key].detach().cpu().clone()
                for key in (
                    "hidden_states",
                    "encoder_hidden_states",
                    "img_ids",
                    "txt_ids",
                    "timestep",
                )
            }

    def route(_module, _inputs, output):
        if calls == 1:
            captured["route"] = tuple(value.detach().cpu().clone() for value in output)

    handles = [
        pipe.text_encoder.register_forward_hook(release_encoder),
        pipe.vae.register_forward_pre_hook(release_denoiser),
        pipe.transformer.register_forward_pre_hook(enter, with_kwargs=True),
        pipe.transformer.transformer_blocks[2].register_forward_hook(route),
    ]
    print(f"construction: render {name} steps={steps} guidance={guidance}", flush=True)
    initial = torch.randn(
        (1, 128, args.size // 16, args.size // 16),
        generator=torch.Generator().manual_seed(args.seed),
        dtype=torch.float32,
    )
    image = (
        pipe(
            prompt=args.prompt,
            latents=initial,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=args.size,
            width=args.size,
            max_sequence_length=512,
        )
        .images[0]
        .convert("RGB")
    )
    image.save(root / f"{name}.png")
    for handle in handles:
        handle.remove()
    place_for_blocks(pipe)
    # Direct constructor models an individual guidance-one denoiser branch.
    # Its prefix is also the conditional prefix inside the native CFG pipeline.
    adapter = Flux2KleinAdapter(pipe.transformer, pipe.scheduler)
    inp = captured["inputs"]
    session = adapter.session(
        latent=inp["hidden_states"],
        conditioning=inp["encoder_hidden_states"],
        img_ids=inp["img_ids"],
        txt_ids=inp["txt_ids"],
        timesteps=pipe.scheduler.timesteps,
        sigmas=pipe.scheduler.sigmas,
    )
    session.continue_(4)  # projection, joint.0, joint.1, joint.2
    exact = describe((session.read("text").cpu(), session.read("image").cpu())) == describe(
        captured["route"]
    )
    if not exact:
        raise RuntimeError(f"{name} conditional prefix differs from native blocks")
    cut = session.capture()
    store = LocalStore(root / "construction-state")
    store.save(cut)
    for receipt in session.receipts:
        store.receipt(receipt)
    record = {
        "model": model,
        "revision": revision,
        "cut": cut.fingerprint,
        "native_prefix_exact": exact,
        "steps": steps,
        "guidance": guidance,
        "captured_scope": "first conditional denoiser prefix through joint.2",
        "native_transformer_calls": calls,
        **count_red(image),
    }
    print("construction: " + json.dumps(record), flush=True)
    identifier = cut.fingerprint
    del session, adapter, cut, handles
    pipe = None
    captured = None
    gc.collect()
    torch.cuda.empty_cache()
    return store.load(identifier), record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=611)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--output", default="outputs/flux2-hotfix")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    donor, donor_record = capture(args, args.base_model, args.base_revision, "donor", 50, 4.0)
    recipient, recipient_record = capture(
        args, args.model, args.revision, "recipient-before", 4, 1.0
    )
    print("construction: fit rank-small writer from paired cuts", flush=True)
    arrays, metadata = fit_low_rank_writer(recipient, donor, address="text", rank=args.rank)
    metadata = save_low_rank_writer(root / "writer.npz", arrays, metadata)
    report = {
        "donor": donor_record,
        "recipient": recipient_record,
        "writer": metadata,
        "supplied": {
            "prompt": args.prompt,
            "seed": args.seed,
            "site": "joint.2",
            "rank": args.rank,
        },
        "learned": "PCA basis and ridge-fitted state-dependent residual weights from one paired trace",
        "renderer": "unchanged native diffusers pipelines; recipient feedback follows separately",
        "terminal_status": "not-assessed",
    }
    (root / "construction-report.json").write_text(json.dumps(report, indent=2))
    print("construction complete: " + json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()

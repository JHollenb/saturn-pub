"""Validate the FLUX diffusion adapters on real or tiny weights: stepped-vs-native
parity, streamed-vs-resident residency, mid-trajectory fresh-process replay, and decode.

One process runs one ``--stage``:

``full``
    Encode the prompt once (encoder-only pipeline, no transformer), free the encoders,
    then load a transformer+VAE pipeline and run the validations. Never holds two large
    models at once, so peak host memory is one model. Klein-4B additionally runs resident
    and streamed and asserts bitwise-identical latent trajectories (the residency claim).

``replay``
    Load only the transformer+VAE, hydrate a saved mid-trajectory ``StateCut`` from a
    fresh interpreter, continue to the end, and print the final-latent digest so the
    parent ``full`` run can confirm exact fresh-process replay.

``--tiny`` swaps every checkpoint load for the adapters' tiny random fixtures on CPU and
runs the identical validation control flow (decode is skipped). It is the local smoke test.

Outputs (``--output``): ``report.json`` and small downscaled PNGs. No latents or weights
are written into the repo. Real runs need CUDA; tiny runs are CPU-only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch

from saturn_pub import Session
from saturn_pub.store import LocalStore

STEP_DTYPE = torch.bfloat16


# --------------------------------------------------------------------------------------
# Memory / reporting helpers
# --------------------------------------------------------------------------------------
def _cuda() -> bool:
    return torch.cuda.is_available()


def _reset_peak() -> None:
    if _cuda():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _peak_vram_mb() -> float | None:
    return None if not _cuda() else torch.cuda.max_memory_allocated() / 2**20


def _rss_mb() -> float | None:
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports ru_maxrss in bytes; Linux reports kibibytes.
        return peak / 2**20 if sys.platform == "darwin" else peak / 2**10
    except Exception:
        return None


def _free() -> None:
    gc.collect()
    if _cuda():
        torch.cuda.empty_cache()


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


# --------------------------------------------------------------------------------------
# Native reference trajectories (the diffusers forward, one full transformer call/step)
# --------------------------------------------------------------------------------------
def _flux2_native_step(model, latent, enc, timestep):
    return model(
        hidden_states=latent,
        encoder_hidden_states=enc["conditioning"],
        timestep=timestep.expand(1).to(latent.dtype) / 1000,
        img_ids=enc["img_ids"],
        txt_ids=enc["txt_ids"],
        return_dict=False,
    )[0]


def _flux1_native_step(model, latent, enc, timestep):
    kwargs = {}
    if enc.get("guidance") is not None:
        kwargs["guidance"] = torch.full(
            [1], enc["guidance"], device=latent.device, dtype=torch.float32
        )
    return model(
        hidden_states=latent,
        encoder_hidden_states=enc["conditioning"],
        pooled_projections=enc["pooled"],
        timestep=timestep.expand(1).to(latent.dtype) / 1000,
        img_ids=enc["img_ids"],
        txt_ids=enc["txt_ids"],
        return_dict=False,
        **kwargs,
    )[0]


def native_trajectory(adapter, enc, scheduler, family):
    step = _flux1_native_step if family == "schnell" else _flux2_native_step
    latent = enc["latent"].to(adapter.device, adapter.dtype)
    per_step = []
    with torch.inference_mode():
        for timestep in scheduler.timesteps:
            noise = step(adapter.model, latent, _on(enc, adapter), timestep)
            latent = scheduler.step(noise, timestep, latent, return_dict=False)[0]
            per_step.append(_digest(latent))
    return latent, per_step


def _on(enc, adapter):
    moved = {}
    for name, value in enc.items():
        if isinstance(value, torch.Tensor):
            moved[name] = (
                value.to(adapter.device, adapter.dtype)
                if value.is_floating_point()
                else value.to(adapter.device)
            )
        else:
            moved[name] = value
    return moved


# --------------------------------------------------------------------------------------
# Adapter trajectory with per-step latent digests
# --------------------------------------------------------------------------------------
def adapter_trajectory(adapter, enc, *, cut_at=None, store=None):
    session = adapter.session(**{k: enc[k] for k in _session_keys(adapter)})
    per_step, saved_cut = [], None
    total = len(enc["timesteps"])
    for target in range(1, total + 1):
        while adapter.point(session._state).logical_step < target:
            session.continue_()
        per_step.append(_digest(session.read("latent")))
        if cut_at is not None and target == cut_at and store is not None:
            saved_cut = store.save(session.capture())
    return session, per_step, saved_cut


def _session_keys(adapter):
    keys = ["latent", "conditioning", "img_ids", "txt_ids", "timesteps", "sigmas"]
    if adapter.execution["family"] == "flux1":
        keys += ["pooled", "guidance"]
    return keys


# --------------------------------------------------------------------------------------
# Encode (encoder-only pipeline) — real weights
# --------------------------------------------------------------------------------------
def _encode_device(family):
    # Klein-4B's encoder fits the 16 GB card and keeps the resident CUDA encode that produced
    # its measured result. Klein-9B's Qwen3-8B (16 GB bf16) and schnell's T5-XXL + CLIP do
    # not: even accelerate sequential CPU offload peaked at 4.8 / 5.6 GB on an RTX 4080
    # (job-ae99a277e513, job-98d8bfa8554e). They encode the single prompt on CPU instead;
    # stepped and native trajectories share these embeddings, so parity is unaffected.
    return "cuda" if family == "klein4b" else "cpu"


def encode_real(family, model_path, prompt, seed, size, steps, guidance_scale):
    import numpy as np

    device = _encode_device(family)
    config = json.loads((Path(model_path) / "transformer" / "config.json").read_text())
    in_channels = int(config["in_channels"])
    if family == "schnell":
        from diffusers import FluxPipeline
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

        pipe = FluxPipeline.from_pretrained(
            model_path, transformer=None, torch_dtype=STEP_DTYPE, local_files_only=True
        )
        pipe.text_encoder.to(device)
        pipe.text_encoder_2.to(device)
        prompt_embeds, pooled, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        generator = torch.Generator("cpu").manual_seed(seed)
        latent, img_ids = pipe.prepare_latents(
            1, in_channels // 4, size, size, prompt_embeds.dtype, device, generator
        )
        sched = pipe.scheduler
        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = calculate_shift(
            latent.shape[1],
            sched.config.get("base_image_seq_len", 256),
            sched.config.get("max_image_seq_len", 4096),
            sched.config.get("base_shift", 0.5),
            sched.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(sched, steps, device, sigmas=sigmas, mu=mu)
        enc = dict(
            latent=latent,
            conditioning=prompt_embeds,
            pooled=pooled,
            img_ids=img_ids.unsqueeze(0) if img_ids.ndim == 2 else img_ids,
            txt_ids=text_ids.unsqueeze(0) if text_ids.ndim == 2 else text_ids,
            timesteps=timesteps,
            sigmas=sched.sigmas,
            guidance=None,
        )
    else:
        from diffusers import Flux2KleinPipeline
        from diffusers.pipelines.flux2.pipeline_flux2_klein import (
            compute_empirical_mu,
            retrieve_timesteps,
        )

        pipe = Flux2KleinPipeline.from_pretrained(
            model_path, transformer=None, torch_dtype=STEP_DTYPE, local_files_only=True
        )
        pipe.text_encoder.to(device)
        prompt_embeds = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1)[0]
        txt_ids = pipe._prepare_text_ids(prompt_embeds).to(device)
        generator = torch.Generator("cpu").manual_seed(seed)
        latent, img_ids = pipe.prepare_latents(
            1, in_channels // 4, size, size, prompt_embeds.dtype, device, generator
        )
        sigmas = (
            None
            if pipe.scheduler.config.get("use_flow_sigmas", False)
            else np.linspace(1.0, 0.25, steps)
        )
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            steps,
            device,
            sigmas=sigmas,
            mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=steps),
        )
        enc = dict(
            latent=latent,
            conditioning=prompt_embeds,
            img_ids=img_ids,
            txt_ids=txt_ids,
            timesteps=timesteps,
            sigmas=pipe.scheduler.sigmas,
        )
    enc = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}
    del pipe
    _free()
    return enc


def load_denoise_pipeline(family, model_path, residency, device):
    kwargs = dict(torch_dtype=STEP_DTYPE, local_files_only=True)
    if family == "schnell":
        from diffusers import FluxPipeline

        from saturn_pub.adapters.flux1 import Flux1Adapter

        pipe = FluxPipeline.from_pretrained(
            model_path,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            **kwargs,
        )
        pipe.vae.to("cpu")
        adapter = Flux1Adapter.from_pipeline(pipe, residency=residency, device=device)
    else:
        from diffusers import Flux2KleinPipeline

        from saturn_pub.adapters.flux2 import Flux2KleinAdapter

        pipe = Flux2KleinPipeline.from_pretrained(
            model_path,
            text_encoder=None,
            tokenizer=None,
            **kwargs,
        )
        if type(pipe).__name__ == "Flux2KleinKVPipeline":
            raise SystemExit("refused: Flux2KleinKVPipeline uses a distinct reference-KV ABI")
        pipe.vae.to("cpu")
        adapter = Flux2KleinAdapter.from_pipeline(pipe, residency=residency, device=device)
    return pipe, adapter


# --------------------------------------------------------------------------------------
# Tiny fixtures (local smoke) — same control flow, CPU
# --------------------------------------------------------------------------------------
def build_tiny(family, residency, device):
    from diffusers import FlowMatchEulerDiscreteScheduler

    if family == "schnell":
        from saturn_pub.adapters.flux1 import Flux1Adapter

        adapter = Flux1Adapter.tiny(residency=residency, device=device)
    else:
        from saturn_pub.adapters.flux2 import Flux2KleinAdapter

        adapter = Flux2KleinAdapter.tiny(residency=residency, device=device)
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(3)
    generator = torch.Generator().manual_seed(11)
    enc = dict(
        latent=torch.randn(1, 8, 5, generator=generator).transpose(1, 2),
        conditioning=torch.randn(1, 3, 24, generator=generator),
        img_ids=torch.zeros(1, 5, adapter.rope_axes if family == "schnell" else 4),
        txt_ids=torch.zeros(1, 3, adapter.rope_axes if family == "schnell" else 4),
        timesteps=scheduler.timesteps,
        sigmas=scheduler.sigmas,
    )
    if family == "schnell":
        enc["pooled"] = torch.randn(1, 16, generator=generator)
        enc["guidance"] = None
    return adapter, enc, scheduler


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------
def stage_full(args):
    family = args.family
    device = "cuda" if _cuda() else "cpu"
    scheduler = None
    report = {"schema": "saturn-pub-diffusion-family-v1", "family": family, "tiny": args.tiny}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    store = LocalStore(args.state_dir)
    started = time.perf_counter()

    if args.tiny:
        resident_adapter, enc, scheduler = build_tiny(family, "resident", device)
    else:
        enc = encode_real(
            family, args.model, args.prompt, args.seed, args.size, args.steps, args.guidance_scale
        )
        report["encode_rss_mb"] = _rss_mb()

    # Streamed trajectory (the family's residency path) with a mid-trajectory cut.
    _reset_peak()
    t0 = time.perf_counter()
    if args.tiny:
        streamed_adapter, enc_s, _ = build_tiny(family, "streamed", device)
    else:
        try:
            pipe, streamed_adapter = load_denoise_pipeline(family, args.model, "streamed", device)
        except (ValueError, SystemExit) as exc:
            # A refused family (e.g. a 9B-KV pipeline) records a clean refusal, not a crash.
            report["refused"] = str(exc)
            report["elapsed_s"] = time.perf_counter() - started
            report["peak_rss_mb"] = _rss_mb()
            (output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
            return report
        enc_s = enc
    cut_at = max(1, len(enc_s["timesteps"]) // 2)
    streamed_session, streamed_steps, cut_digest = adapter_trajectory(
        streamed_adapter, enc_s, cut_at=cut_at, store=store
    )
    streamed_final_latent = streamed_session.read("latent")
    streamed_final_digest = _digest(streamed_final_latent)
    report["streamed"] = {
        "residency": streamed_adapter.execution["residency"],
        "per_step_digests": streamed_steps,
        "final_digest": streamed_final_digest,
        "peak_vram_mb": _peak_vram_mb(),
        "wall_s": time.perf_counter() - t0,
        "cut_digest": cut_digest,
        "cut_at_step": cut_at,
        "model_identity": streamed_adapter.model_identity,
    }

    # Decode the streamed final latent (before any offload perturbs the frozen model).
    if not args.tiny:
        image = streamed_adapter.decode(streamed_session, height=args.size, width=args.size)
        thumb = image.convert("RGB")
        thumb.thumbnail((args.thumb, args.thumb))
        png = output / f"{family}.png"
        thumb.save(png)
        report["image"] = {
            "path": png.name,
            "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "thumb_px": args.thumb,
        }

    # Klein-4B residency claim: resident vs streamed, bitwise identical full trajectory.
    # Building a resident adapter moves the transformer onto the card, so this runs after
    # the streamed decode and gives the native reference a resident (on-card) model.
    if family == "klein4b" or (args.tiny and args.residency_check):
        _reset_peak()
        t0 = time.perf_counter()
        if not args.tiny:
            resident_adapter = type(streamed_adapter).from_pipeline(
                pipe, residency="resident", device=device
            )
        _, resident_steps, _ = adapter_trajectory(resident_adapter, enc_s)
        report["resident"] = {
            "per_step_digests": resident_steps,
            "peak_vram_mb": _peak_vram_mb(),
            "wall_s": time.perf_counter() - t0,
        }
        report["streamed_equals_resident_exact"] = streamed_steps == resident_steps

    # Native reference: the ordinary diffusers forward (one full transformer call per step).
    # 4B/tiny run resident; 9B and schnell do not fit, so this comparison alone uses
    # accelerate sequential CPU offload. It runs last because offload hooks perturb the
    # frozen transformer, which no adapter reuses afterwards.
    native_adapter = resident_adapter if (family == "klein4b" or args.tiny) else streamed_adapter
    if not args.tiny and family != "klein4b":
        pipe.enable_sequential_cpu_offload()
    _reset_peak()
    t0 = time.perf_counter()
    native_final, native_steps = native_trajectory(
        native_adapter, enc_s, _clone_scheduler(native_adapter, enc_s, scheduler), family
    )
    report["native"] = {
        "per_step_digests": native_steps,
        "peak_vram_mb": _peak_vram_mb(),
        "wall_s": time.perf_counter() - t0,
    }
    report["stepped_vs_native_exact"] = streamed_steps == native_steps
    if not report["stepped_vs_native_exact"]:
        report["stepped_vs_native_final_max_abs"] = _max_abs(streamed_final_latent, native_final)
    if not args.tiny and family != "klein4b":
        pipe.remove_all_hooks()

    # Free the large models before the fresh-process replay so two transformers are never
    # co-resident, then hydrate the mid-trajectory cut in a separate interpreter.
    if cut_digest is not None:
        if not args.tiny:
            streamed_session = streamed_adapter = native_adapter = pipe = None
            resident_adapter = None
            _free()
        report["fresh_process_replay"] = _replay_check(
            args, family, cut_digest, streamed_final_digest, store
        )

    report["elapsed_s"] = time.perf_counter() - started
    report["peak_rss_mb"] = _rss_mb()
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {k: report[k] for k in report if k not in {"streamed", "native", "resident"}}, indent=2
        )
    )
    return report


def _clone_scheduler(adapter, enc, tiny_scheduler):
    if tiny_scheduler is not None:
        return tiny_scheduler
    from diffusers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerDiscreteScheduler.from_config(adapter.scheduler_config)
    sched.timesteps = enc["timesteps"].to(adapter.device)
    sched.sigmas = enc["sigmas"].to(adapter.device)
    sched.set_begin_index(0)
    return sched


def _max_abs(a, b):
    return float((a.detach().float().cpu() - b.detach().float().cpu()).abs().max())


def _replay_check(args, family, cut_digest, in_process_final, store):
    """Run a fresh interpreter (subprocess) that hydrates the cut and finishes it."""
    import subprocess
    import sys

    if args.tiny:
        # No subprocess needed for tiny: hydrate in a fresh adapter object here.
        fresh_adapter, enc, _ = build_tiny(family, "streamed", "cpu")
        cut = store.load(cut_digest)
        resumed = Session(fresh_adapter, cut.payload, parent=cut.parent)
        resumed.restore(cut)
        fresh_adapter.finish(resumed)
        return {"exact": _digest(resumed.read("latent")) == in_process_final, "mode": "in-object"}
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        "replay",
        "--family",
        family,
        "--model",
        args.model,
        "--cut",
        cut_digest,
        "--state-dir",
        str(args.state_dir),
        "--size",
        str(args.size),
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    fresh = json.loads(out.stdout.strip().splitlines()[-1])
    return {
        "exact": fresh["final_digest"] == in_process_final,
        "mode": "subprocess",
        "final_digest": fresh["final_digest"],
    }


def stage_replay(args):
    _, adapter = load_denoise_pipeline(
        args.family, args.model, "streamed", "cuda" if _cuda() else "cpu"
    )
    cut = LocalStore(args.state_dir).load(args.cut, device=str(adapter.device))
    session = Session(adapter, cut.payload, parent=cut.parent)
    session.restore(cut)
    adapter.finish(session)
    print(
        json.dumps(
            {
                "final_digest": _digest(session.read("latent")),
                "boundary": cut.boundary,
                "peak_vram_mb": _peak_vram_mb(),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", required=True, choices=["klein4b", "klein9b", "klein9b-kv", "schnell"]
    )
    parser.add_argument("--stage", default="full", choices=["full", "replay"])
    parser.add_argument("--model", default="")
    parser.add_argument("--prompt", default="a small red boat on a white background")
    parser.add_argument("--seed", type=int, default=611)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--output", default="outputs/diffusion-families")
    parser.add_argument("--state-dir", default="outputs/diffusion-families/state")
    parser.add_argument("--thumb", type=int, default=256)
    parser.add_argument("--cut", default="")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--residency-check", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.stage == "replay":
        stage_replay(args)
    else:
        stage_full(args)


if __name__ == "__main__":
    main()

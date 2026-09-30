# Diffusion family validation

Validates the FLUX diffusion adapters on real or tiny weights and writes a JSON report
plus small downscaled PNGs. It never writes latents or model weights into the repository.

```bash
# Local smoke test (CPU, tiny random fixtures, no checkpoints, no downloads):
python experiments/diffusion_family_validation/run.py --family klein4b --tiny --residency-check
python experiments/diffusion_family_validation/run.py --family schnell --tiny --residency-check
```

Real runs need CUDA and a local checkpoint directory:

```bash
python experiments/diffusion_family_validation/run.py \
  --family klein4b --model /path/to/FLUX.2-klein-4B --local-files-only \
  --prompt 'a small red boat on a white background' --steps 4
```

`--family` is one of `klein4b`, `klein9b`, `klein9b-kv`, `schnell`.

## What each run checks

1. **Stepped vs native.** The adapter steps the native transformer one projection / joint
   block / single block / readout at a time; the reference is the ordinary diffusers
   forward (one full transformer call per step). Klein-4B runs the reference resident;
   Klein-9B and FLUX.1-schnell do not fit the 16 GB card, so the reference uses accelerate
   sequential CPU offload. The report records whether every per-step latent digest matched.
2. **Residency (Klein-4B).** The same trajectory is run resident and block-streamed and the
   full per-step latent digest sequence must be identical — the bitwise residency claim.
3. **Fresh-process replay.** A mid-trajectory `StateCut` is saved, then a *separate* Python
   interpreter (`--stage replay`) hydrates it and finishes the trajectory; its final-latent
   digest must equal the in-process one.
4. **Decode.** The streamed final latent is decoded with the native VAE and saved as a small
   PNG thumbnail.

## Memory discipline

The prompt is encoded by an encoder-only pipeline (the transformer is skipped with
`transformer=None`); the encoders are freed before a transformer+VAE pipeline is loaded, so
the process never holds two large models at once. Klein-4B encodes on the GPU; Klein-9B
(Qwen3-8B, 16 GB bf16) and FLUX.1-schnell (T5-XXL + CLIP) encode on CPU, since even
sequential offload peaked at 4.8 / 5.6 GB of VRAM. Block-streamed residency keeps the
transformer weights in host memory and copies one native module to the GPU at a time, so
peak VRAM is a single block plus activations rather than the whole transformer.

The job launch and collection scripts for the RTX 4080 (16 GB) validation are not part of
this public package; the measured [`results/`](results/README.md) JSON and
thumbnails committed here are the public record.

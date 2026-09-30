# Measured results — RTX 4080 (16 GB), bf16, batch 1

Real weights, 512×512, 4 steps, seed 611, prompt `a single small red boat on a plain white
background`, diffusers 0.39.0 / transformers 5.14.1 / accelerate 1.14.0, run on an RTX 4080
host. Each `<family>.json` is the runner's full report (per-step latent digests for every
trajectory, model identity, cut digest, peak memory); each `<family>.png` is a 256 px
thumbnail of the decoded streamed trajectory.

| Family | Job | Source | Stepped = native | Streamed = resident | Fresh-process replay |
| --- | --- | --- | --- | --- | --- |
| Klein-4B | `job-ece5315bc64b` | `34f533d` | exact (4/4 steps) | exact (4/4 steps) | exact |
| Klein-9B | `job-6be0bc01cc95` | `949e4f3` | exact (4/4 steps) | — | exact |
| FLUX.1-schnell | `job-69b0e456925b` | `949e4f3` | exact (4/4 steps) | — | exact |

`949e4f3` changes only how Klein-9B and schnell encode the prompt (on CPU); the Klein-4B path is
byte-identical to `34f533d`. Replay hydrates the step-2 `StateCut` in a separate interpreter
and must reproduce the in-process final-latent digest.

| Family | Streamed peak VRAM | Native-reference peak VRAM | Resident peak VRAM | Encode RSS | Peak RSS | Streamed 4-step wall | Job wall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Klein-4B | 0.65 GB | 7.44 GB (resident) | 7.50 GB | 5.5 GB | 9.3 GB | 26 s | 184 s |
| Klein-9B | 1.07 GB | 0.46 GB (sequential offload) | — | 19.5 GB | 22.2 GB | 49 s | 135 s |
| FLUX.1-schnell | 0.81 GB | 0.23 GB (sequential offload) | — | 15.2 GB | 26.3 GB | 79 s | 180 s |

Klein-9B and schnell do not fit the card resident, so there is no resident trajectory and the
native reference runs through accelerate sequential CPU offload. Peak VRAM is
`torch.cuda.max_memory_allocated` per phase. Encoding Klein-9B / schnell prompts through
sequential offload on the GPU peaked at 4.8 / 5.6 GB (`job-ae99a277e513`, `job-98d8bfa8554e`,
killed at a 4.4 GB ceiling), which is why those two encode on CPU.

These are exactness claims about the mechanics of one program on one device and dtype. Image
quality, prompt adherence, and portability across kernels, dtypes, and devices are not claimed.

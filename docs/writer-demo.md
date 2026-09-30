# Build, debug, and replay a writer

This example builds a small residual writer from fresh paired FLUX.2 Klein states.
It then uses real recipient images to select a dose, retains rejected futures, restores
the accepted writer, and verifies serving from a durable cut in another interpreter.
Requires CUDA and separately acquired base and distilled Klein 4B checkpoints.

The [public experiment](../experiments/flux2_writer/README.md) runs all three stages
and verifies the replay automatically, using pinned Hugging Face checkpoints:

```bash
python experiments/flux2_writer/run.py
```

The individual stages below are useful when inspecting or changing the experiment.

```bash
PROMPT='A clean flat illustration on a pure white background showing exactly five separate red apples arranged in a single horizontal row with equal spacing. No other objects, fruit, marks, or text.'
python examples/flux2_build_writer.py \
  --base-model /path/to/FLUX.2-klein-base-4B --base-revision BASE_REVISION \
  --model /path/to/FLUX.2-klein-4B --revision RECIPIENT_REVISION \
  --prompt "$PROMPT" --local-files-only
python examples/flux2_hotfix.py \
  --model /path/to/FLUX.2-klein-4B --revision RECIPIENT_REVISION \
  --package outputs/flux2-hotfix/writer.npz \
  --prompt "$PROMPT" --local-files-only
```

The construction process renders the unchanged base pipeline with 50 steps and CFG 4,
then the distilled recipient with four steps and guidance 1. It captures the first
conditional transformer inputs and output at `joint.2`. A new Saturn Session executes
projection and the three native joint blocks independently; its text and image state
must exactly match the captured native output before a construction cut is saved.
The adapter represents an individual denoiser branch. The base pipeline's complete
CFG loop remains a native diffusers operation, outside this adapter's continuation claim.

`fit_low_rank_writer` learns a PCA basis from the recipient text rows and fits a ridge
regression to the paired donor-minus-recipient residual. The installed write is:

```text
delta = ((text - mean) @ basis) @ weights + bias
text  = text + dose * delta
```

Rank eight produces 55,297 FP16 serving values including dose. `writer.json` records
both construction cuts, model identities, fitting parameters, reconstruction observations,
and the package checksum. `construction-state/` contains verified cuts and receipts.
The package contains learned arrays; it contains no donor model weights.

The second process loads only the recipient and this new package. It captures a common
parent, runs native and candidate futures, and evaluates red connected components in
the final RGB images. The controller tries doses 0.5, 1, 2, and 4. Rejection restores
its parameter, optimizer, RNG, and cursor closure; the rejected image and measurements
remain evidence. Zero dose, wrong timing, uninstall, and ordinary-pipeline RGB parity
are measured separately. All model parameters remain frozen.

For a third interpreter, take `parent` and `selected_dose` from `report.json`:

```bash
python examples/flux2_hotfix.py \
  --model /path/to/FLUX.2-klein-4B --revision RECIPIENT_REVISION \
  --package outputs/flux2-hotfix/writer.npz --prompt "$PROMPT" \
  --local-files-only --replay CUT_SHA256 --dose SELECTED_DOSE
```

This bypasses prompt encoding and feedback selection. Compare final latent descriptors
and pixel checksums in `replay-report.json` with the first recipient process.

The prompt, seed, site, rank, paired supervision, proposal doses, and target count are
supplied. The residual mapping is fitted; image feedback selects only the dose.
The renderer guarantees native continuation, not correct counting. The component
instrument measures colored regions, not semantic apple identity. This is one development
context: generalization and collateral behavior require separate experiments.

The examples and fitting API import `saturn_pub`, PyTorch, NumPy, diffusers, and other
public dependencies. A fleet scheduler can wrap the commands, but is not required by them.

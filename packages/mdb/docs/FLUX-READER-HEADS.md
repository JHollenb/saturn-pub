# Image-head coalitions, fresh contexts and persistent latent mediation

Three completed CUDA BF16 assays narrow the candidate value/image-response
family in original FLUX.2 Klein4B distilled. All use B1,512×512, four denoising
passes, and the final-pass joint0 reader. The unchanged consumer is the complete
remaining native transformer, flow commit and VAE.

The [earlier family](FLUX-READER-FAMILIES.md) established a V-band→image-output
route. This work identifies stronger tested head coalitions, retains fresh
signed-color counterexamples, reconstructs immediate V formation, and connects
the route to persistent latent state. The earlier writer of that latent remains
open. This is exploratory evidence, with terminal status unassessed.

## Completed panels

| Panel | Native job | CPU saved-array audit | Cases / exact text fields |
|---|---|---|---|
| Four fixed six-head coalitions and complements, seed193 | job-d54b62ed5588 | job-0dde11e4e138 | 44 /176 |
| New seed241 and four new paraphrases; six individual heads, two coalitions, stream factorial | job-6a35eb9ce352 | job-9f59b44b3595 | 80 /320 |
| Two recipients, typed latent source × whole-V-band mediation | job-ae92501eadcd | job-ffb6fd00556e | 8 /32 |

These are132 case occurrences, including repeated protected text and historical
controls, not132 independent prompts or discoveries. Complete text head,
post-attention residual, MLP and next-text fields match at all528 comparisons:
830,472,192 BF16 values. The fresh panel additionally reconstructs all12,582,912
image-V elements exactly. Numerical reader forecasts use original native
Flash/linear/LayerNorm/SwiGLU primitives, not independent finite CUDA arithmetic.
Actual target image futures are observations, not numerical image forecasts.

## Explicit image-head intervention

For baseline and band2-value-edited full native attention returns H0 and HV,
install `[H0_text, HV_image,S, H0_image,complement(S)]`. The paired removal
uses changed image heads outside S and baseline heads inside S. Text remains
baseline. Source band2 is image rows512–767, global K/V rows1024–1279 after
512 text rows. Original Q/K, all unselected V rows and the receiving state stay
fixed. The exact native attention geometry/strides and original downstream
projection are preserved; there are no small-head attention calls or separately
summed projection columns.

Each group route computes original full attention under baseline and changed V
at the first target, then mixes explicit image-output heads. The protected text
closure is checked. Numerical text forecasts seal as one cohort before any full
target suffix enters. Earlier native prefixes, when present, are separately
counted and remain outside the prediction frontier.
These attention-port edits use the qualified campaign dispatcher and explicit
footprints. They are not first-class registered head Acts; the upstream latent
replacements below use the ordinary typed Act API.

The fixed groups0–5,6–11,12–17,18–23 rank18–23 >0–5 >6–11 >12–17 in all
four seed193 contexts under local image alignment, final RGB alignment/RMS and
negative green-score effect. Every coalition and complement contributes.
Group18–23 add RGB projection is0.232–0.441; its complement retains0.334–0.540.
Group/complement RGB interaction relative RMS is0.210–0.575. Projections are
signed alignments, not causal percentages; this is a coarse coalition map.

## Fresh transfer and retained misses

The new prompts reverse mention order and paraphrase the original scenes. One
exact input is:

> Studio photograph on white: a yellow cylinder stands to the right of a green cone. Show both objects separately.

The other three cross color assignment and left/right order. The original
conditioner performs four new encodes, then is released before transformer GPU
residency. One shared seed241 noise draw supplies four honest native prefixes.
The selection/hypotheses are bound before encoding; local numerical predictions
start later, at the declared observed prefix. Seed and wording change jointly.

| Frozen consumer prediction | Per-context agreement |
|---|---|
| Full V-only decreases green score | 4/4 |
| Image-only is negative and larger in magnitude than text-only | 4/4 |
| Coalition18–23 exceeds0–5 in complete-image RGB alignment | 4/4 |
| Coalition18–23 is more negative in green score than0–5 | 2/4 |

The last selected-coalition color effect is+0.000105, while its RGB alignment
remains positive and larger than the outside group. Head20 has the strongest
RGB alignment among tested singletons18–23 in all four new contexts; head23 is
second. Signed color ranks vary. Do not call these universal color heads or
claim an exhaustive24-head map. The outside0–5 coalition remains in the panel.

Image-only RGB projections onto complete V-only effects are0.9854–1.0026;
text projections0.00165–0.00723. Additive RGB error remains11.98–15.59%.
Baseline-derived color masks are instruments, with empty-mask abstention and
visual baseline inspection. All four native baseline scenes conform, but the
source prompts also change geometry. Band2 is not an object mask.

The immediate value equation is `V=to_v((1+scale)*norm1(I)+shift)` in the
original sequence of BF16 operations and original full tensor geometry.
All four complete image-value tensors match exactly. Native V already ran
during prefix capture; call this reconstruction from parent state sealed before
the downstream consumer, not first-execution forecasting of V. Receiving native
V and reconstructed V remain separate. A discrepancy is an observation; the
base arm would then need to be described as reconstructed-V rather than untouched.

## Backward source cut

At two opposite-color recipients in one layout, replay84 native steps from
actual initial cuts to step3/project. Use normal `Act.replace('latent',source)`
at that writable typed boundary, preserving own conditioning. Execute the
original project and capture the first block's actual operands. No fabricated
StateCut, clock write or bypass of readout/scheduler admission is used.

Four own/swapped projected parents exactly match their latent source in image
register and full image V; all text and text Q/K/V exactly match their recipient.
This directly links `committed Z3 → original image projection → modulated norm
→ image V`. It does not explain which prior component wrote the relevant Z3.

Per recipient the square is:

1. Own latent with own V band.
2. Own latent with source V band.
3. Source latent with restored own V band.
4. Source latent with its natural V band.

This changes the whole V band, affecting both attention streams, and is not a
selected-head mediation test. Four historical own-corner PNGs, terminal latents
and text exits replay exactly. All32 new text-field comparisons are exact.

Full latent swaps produce green-score changes−0.9226/−0.9504 on the own-baseline
green masks; restoring the band leaves−0.9134/−0.9376. Band-only effects are
−0.0240/−0.00575. RGB band-only projections onto full latent effects are
0.0187/0.00875, with removal projections0.00888/0.0193. Signed interactions and
all corners remain in the audit. Most of this source-latent effect travels
through other paths. A tested local route is not the whole scene state.

## Run and inspect

Use the installed pinned public wheel environment outside a checkout; scripts
are explicit qualification payloads. `RUN` directories must be new.

```sh
python -I qualification/stage_flux_reader_streams.py RUN \
  --fresh COMPLETED_FRESH_NATIVE_RUN --audit COMPLETED_FRESH_STUDY --panel heads
python -I qualification/stage_flux_reader_streams.py RUN --submit
python -I qualification/flux_reader_fresh_audit.py AUDIT --study RUN
python -I qualification/flux_reader_fresh_audit.py AUDIT --submit
python -I qualification/recover_flux_directory.py AUDIT COMPACT_COLLECTION

# Explicit singleton/group lists for a bounded dissection:
python -I qualification/stage_flux_reader_streams.py NEXT_RUN \
  --fresh COMPLETED_FRESH_NATIVE_RUN --audit COMPLETED_FRESH_STUDY \
  --panel heads --head-groups '[[18],[19],[20],[21],[22],[23],[0,1,2,3,4,5]]'

# Fresh seed/wording: explicit frozen JSON campaign; original conditioner optional.
python -I qualification/flux_reader_fresh.py FRESH_RUN \
  --prior PRIOR_READER_RUN --origin-local AUDITED_ORIGIN_TREE \
  --factorial AUDITED_FACTORIAL_TREE --campaign CAMPAIGN.json
python -I qualification/flux_reader_fresh.py FRESH_RUN --submit

python -I qualification/stage_flux_reader_streams.py UPSTREAM_RUN \
  --fresh COMPLETED_FRESH_NATIVE_RUN --audit COMPLETED_FRESH_STUDY --panel upstream

python -I -m mdb equations find --catalog COMPACT_COLLECTION/extracted/binding-minimal-result/catalog/equations \
  --applicable-to COMPACT_COLLECTION/extracted/binding-minimal-result/catalog/context.json
```

The authored campaign is retained at
`artifacts/mdb-flux-reader-heads-20261106/generalization-campaign-v1.json`.
The consolidated `index-v2/equations` in that study holds208 records: prior68,
132 new numerical occurrences and8 new frozen consumer-prediction occurrences.
All208 supplied-context lookups match and all208 changed-horizon lookups abstain.
Installed CLI queries retrieve a retained signed-green counterexample and return
zero matches for changed horizon. These are curated conditions, not inferred
applicability. `completion-v1.json` and its reproducible verifier bind all panels.
The request, staged source bytes, source inventory, transaction and full counts
bind execution. One finite mrun lease per assay, strict preflight, no retries,
sequential own CUDA, native branches in process. Temporary per-case K/V copies
are no longer retained across the complete cohort. Thirteen focused checks pass
in the isolated installed-wheel environment, including a large orthogonal-effect
control showing that RGB RMS and signed alignment need different rankings.

## Cost, custody and next cut

| Panel | Complete blocks / navigation steps / VAE | Scheduler seconds | Peak RSS / VRAM MB |
|---|---|---|---|
| Coalitions | 1100 /1188 /44 |162.51 |3381 /9306 |
| Fresh contexts |2300 /2500 /80 |257.69 |8889.3 /9372 |
| Latent mediation |350 /388 /8 |87.96 |7218.5 /9412 |

These are measured costs, not a speedup claim. Fresh totals include300 prefix
blocks/340 prefix steps,4 expected pre-attention stops,4 encodes and1 noise draw.
Upstream totals include150 prefix blocks/172 prefix/project steps,4 expected
stops and2 typed latent Acts; no new encodes/noise. Original-primitive forecasts
are additional real work, separately counted in each request/summary.

Raw unique evidence remains near compute:377,873,953 +842,896,422 +276,155,693
bytes, with compact collections and checksum-bound raw retrievers local.
Original parameters remain recipe-backed; every run includes `retrieve_weights.py`
and `parameter-artifacts.json`. Fresh encoding also retains original encoder/
tokenizer file hashes and `retrieve_encoder.py`; no persistent parameter bank is
duplicated. Unsubmitted staging failures/revisions and original requests remain
separate from completed scientific evidence.

Next: original-wording/new-seed and new-wording/original-seed bridge contrasts;
an outside-group individual-head or alternate-partition audit; and backward
break/rescue from the previous readout/commit into the established carrier.
Automatic family induction, full independent CUDA parity and universal object
semantics remain open. The holistic and upstream Astra reviews are retained with
the blog's evidence pointers; their static review is distinct from array audits.

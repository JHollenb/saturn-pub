# Attribute binding across receiving states and contexts

The timed conditioning-read experiment transfers a red/blue assignment across a fresh seed, a paraphrase and new object shapes, but its applicability is narrower than a general attribute-binding algorithm. Among eight contexts, visual review finds four clear assignment transfers, two partial or mixed transfers, and two cases that retain the base assignment. All eight natural source references render the alternative assignment visibly.

The experiment uses the original FLUX.2 Klein4B distilled checkpoint, CUDA BF16, batch 1, 512×512, four diffusion steps and guidance1. Zero-based clocks are used throughout. Original Qwen conditioning is observed. Sources provide contextual rows and image priming; this is source-assisted causal research.

## What was executed

Native job `job-cffb90d8d322` acquired one original pipeline, encoded 16 prompts, released the encoder, drew two shared noise states, and generated 88 native images. Its eight contexts were the historical-style red/blue anchor at seed 109, a fresh seed 347, reversed mention order, green/yellow colors, cone/cylinder objects, reversed layout, wooden/metal materials and striped/dotted textures. Seven contexts share seed 109; these are dependent contrasts.

Each context includes two natural references and nine matched suffixes. Image priming replaces only the image stream at step 1/joint 4, followed by23 honest native operations to step 2/project. The nine suffixes cross priming with a conditioning read, shift the read to step 3, test unchanged rows, or restore image, text, or both immediately after step 2/joint 0. All suffixes reset base conditioning at project3 unless their declared late-read arm supplies source rows there.

Restoration values come from the matched image-only branch's actual joint 0 exit, executed after sealing as a fixed declared dependency. They are source-assisted controls, not predicted hidden states.

There are 6,008 navigation operations,5,368 block entries,184 typed Acts and 6,192 receipts. Each future consumes its actual preceding native state. Native VAE pixels are observed outputs.

## What was predicted

Before the 72 added suffixes, one MDB `ForecastTransaction` seals144 complete projected image/text fields and the semantic hypotheses. The observed frontier explicitly includes16 natural paths and eight priming prefixes:1,976 operations,1,768 blocks, eight Acts and 16 images. It does not claim zero earlier model execution.

The numerical predictions are:

```text
I2 = x_embedder(Z2)
T2 = context_embedder(C2)
```

They use original native Linear primitives and weights, with target-block entry forbidden during forecasting. The144 projection comparisons are bit-exact. This qualifies supplied-state projection composition; it is not an independent attention, denoiser-suffix or pixel forecast.

The saved-only auditor separately reconstructs72 first-joint image writes from captured native operands:

```text
R1 = BF16(R0 + BF16(g_attention * A))
R2 = BF16(R1 + BF16(g_mlp * M))
```

NumPy performs the explicit BF16 rounding independently of Torch. All 72 complete fields match bit-for-bit. These are conditional writer reconstructions, rather than prospective forecasts of attention or MLP returns.

Actual image Q/K/V descriptors remain equal in 56 matched comparisons holding image input, modulation and IDs fixed while changing conditioning. The text-dependent read and subsequent native write can change. This localizes a dataflow distinction; it does not label a color head.

## What transferred

| Context | Early priming plus changed-row read | Restoration result |
| --- | --- | --- |
| Red/blue anchor | Clear blue/purple sphere and red cube | Image restore keeps assignment; text restore returns base assignment |
| Fresh seed 347 | Clear assignment transfer | Same categorical asymmetry |
| Paraphrase | Clear assignment transfer | Same categorical asymmetry |
| Green/yellow | Lighter green sphere, yellow cube; no swap | Base assignment remains |
| Cone/cylinder | Clear blue/purple cone-like object and red cylinder | Same categorical asymmetry |
| Reversed layout | Mixed red/blue sphere; cube remains blue/purple | Image restore preserves mixture; text restore removes it |
| Wooden/metal | Wooden sphere and metal cube remain | No requested material swap |
| Striped/dotted | Sphere becomes dotted; cube mixes stripes and dots | Image restore preserves partial effect; text restore returns base texture assignment |

Hue, geometry and coloration collateral remain visible in the actual PNGs. The cone is truncated, the reversed-layout source sphere has a small cavity, and image priming removes multicolor stripes in the texture case. Manual review is neither a blinded semantic benchmark nor a calibrated object detector.

The whole-image signed projection onto the natural source-minus-base contrast is 0.789–0.828 in the four clear-transfer cases. It is 0.245 for green/yellow,0.346 for reversed layout,0.297 for material and 0.649 for texture. Material's image-only baseline is already0.299 on this axis, illustrating why pixel alignment alone cannot certify attribute transfer.

All eight baseline replays match their natural base full payload, latent and PNG exactly. Joint image/text restoration after joint 0 leaves only conditioning different immediately afterward. Resetting that conditioning at project3 rejoins the matched image-only branch's full payload; all eight final latents and PNGs are exact. Restoring either stream alone does not establish complete state closure.

## Computation records and generality

The useful family is a receiving-state-dependent, timed contextual read with working-state maintenance. Its parts have different evidence scopes:

- The projection and gated residual equations are already-known architectural transformations, now qualified across these contexts.
- Paired-token addressing moves from rows 4/11 to9/17 under the paraphrase. All eight changed-ID pairs are token-aligned; this is a source-assisted resolver, not automatic semantic address induction.
- The early image-plus-conditioning recipe has clear semantic transfer in four contexts and explicit counterexamples in four.
- Working text sustains the complete or partial early-read effect in several contexts after the immediate image write is removed. This is a conditional carrier role, rather than a universal text-only symbol.
- Joint restoration supplies exact continuation closure independently of whether a requested semantic transfer succeeds.

Eight `EquationCatalog` entries bind exact prompt, seed, receiver cut, source request and primitive profile. Queries abstain on changed seed, source, receiver or numerical profile. The separate computation records retain semantic hypotheses, native image effects and limitations. No automatic rule induction or universal dictionary is implied.

The initial unchanged-row selector chose chat-prefix rows 0/1. That is a weak sham, because causal contextualization cannot read the later attributes. All eight corresponding PNGs equal the matched image-only PNG exactly.

The completed companion assay (`job-12bcee179489`) selects two unchanged prompt words after the last changed attribute row: rows 12/13, usually “cube” and “on,” or rows 18/19 (“sphere” and punctuation) under the paraphrase. It reuses the actual saved primed parents, with zero new encodes, noise draws or prefixes. Sixteen projection fields seal before 448 native operations, 400 blocks, 16 conditioning Acts and eight VAE outputs. The saved-only follow-up audit (`job-fd98a63a4d6e`) verifies all 16 projections exactly, both conditioning-write footprints per context, selected-source bytes and base-conditioning resets.

Every selected pair carries a real source difference: 15,179–15,308 of 15,360 BF16 coordinates differ. All eight visible outputs retain the base assignment. Their RGB changes relative to the matched image-only branch are 0.368–0.580 levels RMS in seven contexts and 2.675 in the texture case. Contextual state differs without reproducing the attribute-row transfer.

The contextual source-minus-base delta has only 0.174–0.400 of the original attribute-row delta's L2 norm. These are source-coherent, equal-row-count controls, not energy-matched interventions. They weaken an arbitrary-row explanation under the tested doses; they do not prove exclusive semantic localization to the attribute rows. A matched-dose curve is a useful next discriminator.

The companion takes 61.52s elapsed with 2,104.2MB peak RSS and 9,238MB VRAM, and retains 326,895,504bytes near compute. Its compact audited directory is 3,240,804bytes, including all eight images. A first saved-only audit completed but lacked the request-sealed custody helper in its payload; the corrected publisher adds that helper and the dose diagnostics without rerunning the native model. The artifact retriever now accepts aliases declared by the manifest, including `control` and `main`, while retaining checksum verification.

## Reproduce the MDB workflow

Use an installed, qualified MDB/Saturn-pub environment outside the source checkout. Stage and inspect a finite request before submission:

```bash
python -I qualification/stage_flux_binding_generalization.py RUN \
  --prior LAST_WRITER_RUN --schedule-prior READER_GENERALIZATION_RUN
python -I qualification/stage_flux_binding_generalization.py RUN --submit
python -I qualification/run_flux_binding_generalization_audit.py AUDIT --native RUN
python -I qualification/run_flux_binding_generalization_audit.py AUDIT --submit
python -I qualification/recover_flux_directory.py AUDIT COMPACT
python -I qualification/flux_binding_context_controls.py CONTROLS --main-run RUN
python -I qualification/flux_binding_context_controls.py CONTROLS --submit
python -I qualification/flux_binding_context_controls_audit.py stage CONTROL_AUDIT --controls CONTROLS
python -I qualification/flux_binding_context_controls_audit.py submit CONTROL_AUDIT
python -I qualification/recover_flux_directory.py CONTROL_AUDIT CONTROL_COMPACT
```

Here the qualification paths refer to absolute files under `packages/mdb`; the admitted worker imports installed public packages without checkout path injection. The stage inputs are source-bound prior requests and parameter recipes, not a copied weight bank. `MRUN_URL` must name the configured scheduler.

Raw BF16 arrays, CAS states and receipts stay near compute. The compact result includes images, audits, source request and plan, sealed transaction, searchable records, parameter recipes and `retrieve_evidence.py`. Recover individual raw members with their manifest hashes. Recover original named parameters using `retrieve_weights.py`; recover encoder files using `retrieve_flux_encoder.py --manifest source-plan.json --checkpoint ORIGINAL_CHECKPOINT`.

Native elapsed time was426.12s, with 6,761.2MB peak RSS and 9,210MB peak VRAM. Raw evidence is 1,801,707,145bytes. The compact audited directory is 17,299,763bytes; its transported archive is 14,459,381bytes. Saved-only audit job `job-dd2e7936a55e` took 22.0s and 436.2MB RSS, with zero models or new images.

An earlier request `job-3d73d2e6c529` failed the bootstrap request-identity check before the experiment entered. The corrected request uses MDB's original JSON hash convention; the failed attempt remains separate.

Fifteen focused mechanics tests pass using installed public packages outside the checkout. The standard Saturn evidence sweep and verification complete with `target-streams-only`: two JSONL streams and 116 records, but zero typed receipts, claims or address references. The dedicated auditors supply the full-field and CAS/Act checks here; zero generic watch fires are not semantic certification.

## Evidence

- [Worker](../qualification/flux_binding_generalization.py), [stager](../qualification/stage_flux_binding_generalization.py), [independent auditor](../qualification/flux_binding_generalization_audit.py), [compact publisher](../qualification/run_flux_binding_generalization_audit.py), [contextual control](../qualification/flux_binding_context_controls.py).
- [Contextual-control auditor and publisher](../qualification/flux_binding_context_controls_audit.py), [audited companion data](../../../../artifacts/mdb-flux-binding-generalization-20261106/context-controls-compact-02/extracted/binding-minimal-result).
- [Study and requests](../../../../artifacts/mdb-flux-binding-generalization-20261106), [audited compact data](../../../../artifacts/mdb-flux-binding-generalization-20261106/compact-01/extracted/binding-minimal-result), [manual semantic review](../../../../artifacts/mdb-flux-binding-generalization-20261106/semantic-review.jsonl).
- [Earlier minimal binding cut](FLUX-BINDING-MINIMAL.md), [computation-family evidence](FLUX-COMPUTATION-FAMILIES.md), [known computation inventory](../../../../obsidian/research/known-computations/known-computations.jsonl).

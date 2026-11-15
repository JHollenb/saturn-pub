# Attribute binding at matched conditioning dose

The matched-dose campaign weakens an amplitude-only explanation of the earlier binding result. At nearly equal conditioning-input delta norm, attribute rows have a larger signed image response toward the natural source than later unchanged contextual rows in all four frozen primary cases, and in the other four descriptive cases. Reduced attribute doses retain the base assignment; intermediate doses transfer three primary assignments clearly and the paraphrase partially. Contextual rows amplified to the full attribute-input norm retain the base assignment in every context.

This is new dose and applicability evidence in the existing contextual-binding family. Payload direction and receiving row remain coupled. The experiment does not identify an exclusive color address or introduce a new mathematical primitive.

## Model and observed frontier

The original FLUX.2 Klein4B distilled checkpoint runs on CUDA in BF16, batch1, 512×512, four denoising steps, guidance1. All clocks are zero based. The eight paired contexts and actual image-primed step2/project parents come from [the generalization campaign](FLUX-BINDING-GENERALIZATION.md): anchor, fresh noise seed, paraphrase, green/yellow, cone/cylinder, reversed layout, wooden/metal, and striped/dotted. Seven contexts share seed109; one uses347. These are dependent contrasts and already observed contexts.

Source prompts supply original Qwen conditioning and the image priming. This run performs zero new encodings, noise draws or priming prefixes. It starts from those actual saved parents; it does not predict the prompt encoder or image prime.

Let A denote the two changed attribute-token rows, normally4/11, and B the two later unchanged contextual rows, normally12/13. The paraphrase uses A9/17 and B18/19. The later B rows carry real contextual differences between prompts, despite unchanged token IDs.

## The dose panel

For base conditioning C, observed source conditioning S and row support R, install

```text
C_R(alpha) = BF16(float32(float64(C_R) + alpha*(float64(S_R)-float64(C_R))))
```

Endpoints0/1 preserve the original BF16 bits exactly, including signed zeros. Other coordinates remain base conditioning. Alpha is solved using only supplied C/S arrays, with explicit ties-to-even rounding. The measured delivered dose is

```text
d_C(R, alpha) = || decode(C_R(alpha)) - decode(C_R) ||_2
```

Eight arms per context give64 suffixes:

| Arm | Intervention at step2/project |
| --- | --- |
| sham | Base C |
| A.half-context | A scaled to the actual dose of B at alpha0.5 |
| A.context | A scaled to the actual natural B dose |
| A.mid | Alpha midway between A.context and1 |
| A.full | Exact original source A rows |
| B.half | B at alpha0.5 |
| B.full | Exact original source B rows |
| B.attribute-norm | B scaled to the actual natural A dose |

Every branch resets base C at step3/project and consumes its own actual native state through the original remaining blocks, scheduler and VAE. B.attribute-norm extrapolates beyond the natural source, with alpha2.50–5.74. It is a stress test of scalar-strength sufficiency, not a natural prompt continuation.

The24 matched-pair relative input-norm residuals have maximum absolute magnitude3.986e-5, or0.003986%. For the primary A.context/B.full contrast, the maximum is3.443e-6. These are diagnostics; quantized BF16 plateaus are preserved rather than used as arbitrary scientific gates.

## Frozen predictions and numerical scope

One MDB ForecastTransaction seals128 complete image/text projection predictions and the semantic/ordinal hypotheses before any new target block or image executes. Counts at seal are one model load and zero encodings, noise draws, block entries, navigation operations, Acts, receipts or images.

The numerical horizon is precisely:

```text
I2 = x_embedder(Z2)
T2 = context_embedder(C2)
```

These predictions use original installed Linear primitives and checkpoint parameters, with no fitting and target-block entry prohibited. All128 compared fields are bit-exact. Forty new dose futures contribute80 field comparisons. Twenty-four already-known replay controls contribute48 comparisons; they are reported separately from new outcomes. No attention return, MLP return, complete suffix or final image is numerically forecast in this assay.

The frozen primary ordinal prediction is that A.context has a greater signed RGB increment on the fixed natural-source-minus-base axis than B.full, measured relative to same-case sham. Its four cases are anchor, seed, wording and objects. Reduced A doses are predicted to retain base assignment; amplified B is predicted to retain base assignment or produce collateral instead of a clean swap. A.mid is allowed to be mixed. Those hypotheses cannot be revised using the images.

## Native observations

| Context | A.context / B.full signed RGB increment | A.mid visible result | Amplified B result |
| --- | --- | --- | --- |
| Anchor | .007097 / .001178 | Blue/purple sphere, red cube | Red sphere, blue cube |
| Fresh seed347 | .004362 / -.000265 | Blue sphere, red cube | Base assignment |
| Paraphrase | .002220 / .000405 | Purple sphere, red cube; partial | Base assignment |
| Cone/cylinder | .024284 / .000053 | Blue/purple cone, red cylinder | Base assignment |
| Green/yellow | .039186 / .002076 | Lighter green sphere, yellow cube | Base assignment |
| Reversed layout | .002370 / -.000035 | Mixed sphere, purple cube | Base assignment |
| Wooden/metal | .001120 / -.002885 | Wooden sphere, metal cube | Base assignment |
| Striped/dotted | .042381 / .002545 | Dotted sphere, mixed cube pattern | Base assignment |

The four frozen primary RGB inequalities hold. The other four show the same ordering descriptively; they are not four extra preregistered successes. Effects at low matched dose are small, and pixel-axis projection is not a semantic classifier or causal percentage. Both reduced A arms retain the base assignment in all eight visual panels. All eight amplified B arms also retain it. Full A exactly replays the earlier four-clear/two-partial/two-no-swap panel.

Three primary A.mid outputs already show clear swaps at alpha0.6600,0.6600 and0.6998. The paraphrase is partial at0.5871 and clear at the already-known full dose. These observations bracket changes between tested doses under fixed receivers. They do not estimate an exact phase-transition threshold or a monotone response curve. Reversed layout is visibly mixed at mid dose and has a smaller whole-image source-axis increment at full dose; green/yellow also peaks above its full-dose increment without swapping. Retain these nonmonotone observations.

## Projection gain and what amplitude cannot explain

Conditioning norm is matched before the learned context projection. The audit therefore separately measures full actual T2 minus same-case sham:

```text
d_T = || T2(arm) - T2(sham) ||_2
gain = d_T / d_C
```

At A.context/B.full matched C norm, A has1.0255–1.0956 times B's projected-text norm. Different learned projection gains remain a concrete explanation for part of the low-dose response. Equal input norm does not establish equal reader input.

A second comparison is more discriminating. Amplified B has1.31–1.56 times the projected-text delta norm of A.mid across the four primary contexts, and also has greater conditioning delta norm. Yet B keeps the original assignment while A.mid gives three clear and one partial transfer. A simple monotone rule depending only on total C norm, or only on total T norm, cannot explain those assignment differences. Direction/content, row-specific reading, local normalization and receiver compatibility remain candidates. Different norm distributions across rows/features may matter even when the total norm is larger.

The result supports a direction-and-receiver-dependent contextual read with a dose-sensitive continuation. It does not separate payload semantics from spatial/token address, or certify an object-binding mechanism shared universally across attributes.

## MDB execution and audit

Native job `job-8a2851b5d95b` completes in273.9s, with7059.6MB peak RSS and9220MB VRAM. One original pipeline load produces64 actual VAE images,3584 navigation operations,3200 block entries,128 typed conditioning Acts and3712 receipts. There are zero new encodings, noise draws or prefixes.

Saved-only CPU audit `job-b2c3dd9b831f` takes48.36s and472.2MB RSS, with zero models, GPU memory or new images. It independently reconstructs all installed BF16 dose rows, checks128 Act footprints and base-C resets, joins all64 final native latents, and checks the three complete native/source inventories. All24 sham/A.full/B.full replay controls match the historical full payload, final latent and PNG bytes exactly. The global transaction is verified closed, and all64 target images follow the seal.

Native request SHA256 is `2f811566bf674a3055ba510055d68ee3b085588a46773b5f92fc18b08630203c`; seal is `ce827adac90b3e416401e2e2b95b91571c85e70930ac2d560a1ac4a3fcadf859`. Native inventory SHA256 is `29ab738de7fc8c76a08348a4e235ba41aff8e5a4bf59236927869217091395a4`. Raw native evidence occupies1,030,501,456bytes near compute. The compact audited directory is13,644,995bytes, with10,765,448bytes in the transported archive. Original parameters remain retrieval recipes, not another weight bank.

Eight focused installed-environment mechanics tests cover BF16 interpolation, signed-zero endpoints, quantized norm plateaus, independent delivered-dose arithmetic, signed/orthogonal RGB response, and projected-text gain/row counts. The first local audit staging attempt exceeded the finite local-work cap before submission. The corrected contract names64 per-arm audit units and explicitly lists their128 projection/128 Act/64 image observations. No native rerun was needed; the partial staging directory remains retained.

## Run the workflow

Use absolute qualification paths with an installed qualified environment outside the checkout, and set MRUN_URL to the configured scheduler:

```bash
python -I qualification/flux_binding_matched_dose.py RUN \
  --main-run GENERALIZATION_RUN --context-run CONTEXT_CONTROL_RUN
python -I qualification/flux_binding_matched_dose.py RUN --submit
python -I qualification/flux_binding_matched_dose_audit.py stage AUDIT --native RUN
python -I qualification/flux_binding_matched_dose_audit.py submit AUDIT
python -I qualification/recover_flux_directory.py AUDIT COMPACT
python -I qualification/render_flux_binding_matched_dose.py \
  COMPACT/extracted/binding-minimal-result FIGURES
```

The compact includes all64 native PNGs, the sealed source transaction, audited rows, dose recipes, plans, parameter recipes and checksum-verified `retrieve_evidence.py` for aliases native/main/context. `retrieve_weights.py` recovers original named parameters; `retrieve_flux_encoder.py` recovers the original encoder checkpoint files when needed. Large raw arrays, CAS states and receipts remain near compute.

The next useful discriminator is a small direct-T factorial at a common joint0-entry parent: transport the same two-row payload direction to A and B support, and compare both payload directions on each support at matched delivered T norm. Preserve a zero-delta sham and exact native-T replay. This can distinguish location effects from payload effects and their interaction at that boundary. It would not by itself establish a complete prompt-to-binding program.

## Evidence

- [Worker and stager](../qualification/flux_binding_matched_dose.py), [saved-only auditor](../qualification/flux_binding_matched_dose_audit.py), [native contact-sheet renderer](../qualification/render_flux_binding_matched_dose.py).
- [Full audited result](../../../../artifacts/mdb-flux-binding-matched-dose-20261106/compact-01/extracted/binding-minimal-result/audit.json), [sealed requests and compact custody](../../../../artifacts/mdb-flux-binding-matched-dose-20261106), [manual image review](../../../../artifacts/mdb-flux-binding-matched-dose-20261106/semantic-review.jsonl).
- [Earlier cross-context campaign](FLUX-BINDING-GENERALIZATION.md), [known computation inventory](../../../../obsidian/research/known-computations/known-computations.jsonl).

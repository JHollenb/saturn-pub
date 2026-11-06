# Binding payload and receiving rows

The direct-text factorial supports conditional payload–receiver compatibility in the existing contextual-binding family. Moving the successful attribute delta to later contextual rows removes the visible assignment transfer. Putting the other contextual delta on the attribute rows also retains the base assignment, at matched delivered projected-text norm. All twelve new crossed/scaled continuations show this pattern across the four declared historical contexts.

This result narrows the [matched-dose account](FLUX-BINDING-MATCHED-DOSE.md). It does not establish an exclusive semantic row, a universal binding vector, or failure of recipient-aware payload translation. Destination changes include local baseline state and positional treatment. The crossed states are controlled internal interventions, not naturally encoded prompts.

## Organism and boundary

The original FLUX.2 Klein4B distilled checkpoint runs on CUDA BF16, batch 1, 512×512, four denoising steps, guidance 1. Clocks are zero based. The four contexts are red/blue sphere–cube at seed 109, the same prompt at seed 347, a paraphrase with reversed mention order, and red/blue cone–cylinder at seed 109. These give three distinct paired conditioning inputs and two noise seeds; they are dependent historical contrasts, not held-out prompt generalization.

Each case starts from the exact saved sham step 2/joint 0 entry of the earlier campaign. That state already includes the observed source-assisted image prime and base projected text. There is no new encoding, noise draw, projection prefix or prime. Image input, latent, time/context, position IDs and base conditioning are held fixed at the intervention boundary.

Let A be attribute rows 4/11, or 9/17 under the paraphrase. Let B be later contextual rows 12/13, or 18/19 under the paraphrase. The two ordered source directions are:

```text
Delta_A = T(A.mid)[A] - T(sham)[A]
Delta_B = T(B.full)[B] - T(sham)[B]
```

These are actual earlier projected text arrays. Delta_A's intermediate dose previously produced three clear transfers and one partial transfer. Delta_B comes from unchanged token IDs whose contextual state differs between prompts.

## The two by two intervention

For payload p and destination d, install ordered row 1 at destination row 1 and row 2 at row 2:

```text
T'[R_d] = BF16(FP32(FP64(T_base[R_d]) + alpha * Delta_p))
```

All other text rows remain base. Source-only bisection chooses alpha to match the actual installed BF16 delta norm to the same case's original A.mid text norm. Exact same-support endpoints preserve source bits, including signed zeros. No target continuation or image enters this search.

| Cell | Payload | Receiving rows |
| --- | --- | --- |
| A@A | Attribute delta | Attribute rows |
| A@B | Attribute delta | Later contextual rows |
| B@A | Contextual delta | Attribute rows |
| B@B | Contextual delta | Later contextual rows |

Each case also has sham, exact full-A text replay and exact full-B text replay. Seven arms per case yield 28 native suffixes. A@A is itself an exact known A.mid replay: there are 16 known endpoint controls and 12 new crossed/scaled futures. Every branch resets base conditioning at step 3/project, consumes its own actual preceding state for 55 native operations, and ends with the original scheduler and VAE.

The sixteen factorial cells' maximum absolute relative text-norm residual is 2.389e-5, or 0.002389%. Installed-versus-intended ordered-delta cosines are at least 0.9999818. These diagnostics show precise transport; no arbitrary norm threshold gates scientific collection. The A/B source directions are different: ordered-row cosines are -.03174 in anchor/seed, .01282 for objects and -.31320 for wording. Absolute row positions are deliberately excluded from these feature-vector cosines.

## Frozen prediction and observed images

The global MDB transaction freezes all recipes and twelve ordinal inequalities before any new native block executes: per case, A@A has a larger signed RGB source-axis increment than each alternative cell. The frozen semantic prediction is that crossed/scaled cells retain the base assignment or produce a mixed/collateral response instead of a clean swap. Every miss would remain in the record.

Define the diagnostic scalar

```text
s(Y) = dot(Y - Y_sham, Y_natural-source - Y_natural-base)
       / ||Y_natural-source - Y_natural-base||^2
```

| Context | A@A | A@B | B@A | B@B | A@A visible assignment |
| --- | --- | --- | --- | --- | --- |
| Anchor | .649393 | -.003489 | .001999 | -.000146 | Blue/purple sphere, red cube |
| Seed 347 | .615702 | -.001714 | .001754 | -.001206 | Blue sphere, red cube |
| Paraphrase | .529417 | .000866 | -.001821 | .000681 | Purple sphere, red cube; partial |
| Cone/cylinder | .492942 | .000837 | -.000297 | .000771 | Blue/purple cone, red cylinder |

All twelve inequalities hold. Visual inspection of every cell finds the original red-first-object/blue-second-object assignment in all twelve new futures. Shading and geometry differences remain; retained assignment does not mean zero effect. The paraphrase stays partial at A@A, and the cone remains truncated. The four A@A assignment outcomes were already known and are not four newly predicted successes.

For the complete RGB arrays, the factorial interaction is Y_AA − Y_AB − Y_BA + Y_BB. Its projection on the fixed natural axis is .650736, .614455, .531053 and .493172 for anchor, seed, wording and objects. These are finite image contrasts, not semantic mediation percentages or probabilities. Payload effects depend strongly on destination in this panel; destination alone does not turn the tested B direction into a successful payload.

## A large immediate response can lead to the original assignment

The first joint block exposes the modulated normalized text presented to add_q_proj, raw text Q/K/V returns, both attention returns, and image/text states after the block. These are native observations, not independently predicted operands. In symbols, the captured normalized text includes both layer normalization and time modulation:

```text
N_T = (1 + scale_T) * LayerNorm(T) + shift_T
Q_T = W_Q N_T; K_T = W_K N_T; V_T = W_V N_T
```

Matching T norm does not match all downstream port norms. Here the moved A payload produces *greater* response norms than A@A at every captured normalization/raw-QKV port, at first image attention return, and at the immediate image state, in all four contexts.

| Context | A@B / A@A modulated-normalized-text delta norm | Image attention delta norm ratio | Joint 0 image delta norm ratio | Joint 0 text delta norm ratio |
| --- | --- | --- | --- | --- |
| Anchor | 1.0311 | 1.0537 | 1.0273 | .6387 |
| Seed 347 | 1.0311 | 1.1432 | 1.1589 | .6482 |
| Paraphrase | 1.0609 | 1.0814 | 1.2655 | .7920 |
| Cone/cylinder | 1.0010 | 1.1966 | 1.2579 | .7102 |

Each norm is relative to the same-case sham at the same named port. Attenuation in those first input/image-response norms cannot explain the A@B assignment failure. The working text response after joint 0 is smaller, and the directions/distributions at every port differ. Later repair, meaningful state geometry, attention addressing and working-text maintenance remain live explanations. This assay does not identify a responsible head or independently reconstruct its complete attention equation.

The working inference is compatibility between a supplied direction and a receiving state at a particular clock. It is new causal applicability evidence in an existing computation family, not a newly discovered mathematical primitive. A more general symbol-table record needs receiver conditions, address resolution, payload lowering, dose, timing and continuation. Literal delta movement failing does not exclude a recipient-aware translation that succeeds.

The next small cut is to exchange image and text states after joint 0 between A@A and A@B. A four-cell I×T continuation with exact diagonal replays can test whether the final assignment follows working text, immediate image state, or their interaction. That would connect this receiver-compatibility result to the earlier [working-text maintenance assay](FLUX-BINDING-GENERALIZATION.md).

## Execution, audit and prediction coverage

Native job `job-3d4bdd4210d8` runs one original pipeline load, 28 images, 1540 native navigation operations, 1400 block entries, 56 typed Acts and 1596 receipts. No new encoding, noise or prefix executes. Elapsed time is 212.02s, with 7663.4MB peak RSS and 9222MB peak VRAM. The job waited for a healthy existing CUDA lease before admission.

Before target release, the transaction seals 28 complete expected T-entry states and 28 invariant image fields. All 56 witnesses are exact. They verify supplied-state transport and typed installation; there are **zero prospective denoiser-equation or pixel numerical forecasts** in this assay. The twelve native-outcome ordinal predictions are a separate evidence scope.

Saved-only CPU audit `job-6382a82b5455` completes in 60.46s and 446.4MB RSS, with zero models, GPU memory or new images. It verifies three complete inventories, independently reconstructs transported BF16 rows, checks all 56 typed footprints and all 28 final native latent joins. All 16 known controls reproduce full terminal payload, final latent and PNG bytes exactly. The transaction verifies closed; all 28 images follow sealing.

Native request SHA256: `4baa6bd41c8cadd32a541088f9c2f57e94cc4f9fffabe811ff5e834d7c4bd714`. Seal: `4375e524eb00fe86a402585723f13ab7f724519fcd3624c74f6acce56107ddd7`. Native inventory: `20ba3a1fc8298e30177f1c42cec8df38b3455c6983d8043bcbe077937f28a2d3`.

Raw native evidence occupies 1,298,205,647 bytes near compute, including observed reader ports and CAS states. The compact audited directory is 5,891,794 bytes, and its transported archive is 4,432,934 bytes. Original weights remain immutable retrieval recipes, with no repeated weight bank. Individual raw retrieval was exercised on the 12,416-byte A@B replacement array and verified its SHA256.

Nine focused tests pass in the qualified installed environment outside the checkout. They cover ordered delta transport, exact endpoints/signed zeros, destination-relative quantized norms, zero/nonfinite handling, factorial contrast signs and source-direction geometry. Generic Saturn evidence run/verify is a separate limited instrument: the completed run sees one timeline stream with 59 records and zero typed receipts, claims or addresses. The dedicated CAS/Act/transaction auditor supplies the full checks above. A first evidence sweep refused a watch root inside its declared results tree before collection; the corrected scope keeps ledgers outside the immutable result.

## Reproduce with MDB

Use an installed qualified environment outside the checkout, absolute qualification paths, and the configured MRUN_URL:

```bash
python -I qualification/flux_binding_route_payload.py RUN --prior-run MATCHED_DOSE_RUN
python -I qualification/flux_binding_route_payload.py RUN --submit
python -I qualification/flux_binding_route_payload_audit.py stage AUDIT --native RUN
python -I qualification/flux_binding_route_payload_audit.py submit AUDIT
python -I qualification/recover_flux_directory.py AUDIT COMPACT
python -I qualification/render_flux_binding_route_payload.py \
  COMPACT/extracted/binding-minimal-result FIGURES
```

The compact includes all 28 native PNGs, the exact source transaction and plans, recipes, observed-port references, parameter manifests and retrievers. `retrieve_evidence.py` uses native/prior/main aliases with checksum verification. `retrieve_weights.py` recovers original named parameter arrays; `retrieve_flux_encoder.py` recovers the original encoder files if needed.

## Evidence

- [Native worker/stager](../qualification/flux_binding_route_payload.py), [transport mathematics](../qualification/flux_binding_route_payload_math.py), [independent auditor](../qualification/flux_binding_route_payload_audit.py), [native renderer](../qualification/render_flux_binding_route_payload.py).
- [Audited arrays, contrasts and controls](../../../../artifacts/mdb-flux-binding-route-payload-20261106/compact-01/extracted/binding-minimal-result/audit.json), [study and immutable requests](../../../../artifacts/mdb-flux-binding-route-payload-20261106), [manual semantic review](../../../../artifacts/mdb-flux-binding-route-payload-20261106/semantic-review.jsonl).
- [Pre-result interpretation plan](../../../../artifacts/mdb-flux-binding-route-payload-20261106/interpretation-plan.md), [independent result review](../../../../artifacts/mdb-flux-binding-route-payload-20261106/independent-review.md), [computation inventory](../../../../obsidian/research/known-computations/known-computations.jsonl).

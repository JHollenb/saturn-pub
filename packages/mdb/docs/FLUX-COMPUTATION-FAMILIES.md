# Testing a diffusion computation family in one MDB run

This campaign tests six extensions of the [replicated single-block residual/attention record](FLUX-ROUTE-REPLICATION.md) within one resident FLUX.2 Klein 4B model process. MDB saves immutable typed cuts, rewinds to those cuts, applies declared image-state edits, and continues each branch through the native scheduler and VAE. Different prompt/seed histories have their own matched roots; sharing a model process does not make them one identical state.

The campaign has been submitted as `job-65ed4e0752b5`. Results are pending. The frozen plan has SHA `2822456cffc40f75de71430865f96a6fa29ae0d2c23c174bd23f12bce009b325`; the producer request has SHA `75280f926f039ae1dbe7d2d28ae63c16aaac34d4ebd504a080811a21fc2bfaf7`.

## Six questions, shared evidence

| Panel | Question | Declared contrasts |
|---|---|---|
| 1 | Does text attention act through keys, values, or their combination? | K-only, V-only, K+V, and base queries under changed residual |
| 2 | Does the response depend on hybrid prompt history? | Hybrid versus ordinary full-prompt histories, plus reciprocal transfer |
| 3 | Is the response specific to relative position? | Relation, color assignment, and shape assignment, with matched R/A/RA edits |
| 4 | Does it extend beyond the original object pair? | Held-out green-cone/yellow-cylinder scenes and a third-object scene |
| 5 | Which support and timing matter? | Top versus alternative quarters of heads, text rows and image rows; row omission; early/late/alternative-row rescue |
| 6 | Can a prefix select an applicable computation record prospectively? | Paired-prefix and recipient-only nearest-development selectors, sealed before held-out futures |

There are 47 dependent native image outcomes: 36 development and 11 held-out. Eight prompt encodings produce nine source histories and seven directed source/recipient pairs. References, three copy shams, and existing mixed-state controls are shared across panels. Every declared branch runs regardless of prediction, abstention, partial semantic contrast, or an unfavorable outcome.

## What MDB executes

The specimen is the distilled FLUX.2 Klein 4B checkpoint, CUDA BF16, physical batch one, 512 × 512, four denoising steps and guidance 1. Checkpoint inventory SHA is `dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f`. There are no training updates. Model weights remain resident throughout the campaign.

The selected boundary is logical step 0, before single block 4. Read-only hooks capture the actual module operands and returns. The native block writes

```text
R_next = R + gate * Linear(concat(A, M), W_merge, bias).
```

R/A/RA interventions replace source image-row residual and/or attention operands while keeping the recipient MLP operands, gate, post-block text and stored conditioning. The original projection shape is retained to preserve native BF16 rounding. The resulting image field is installed with a typed image `Act` at the block exit. MDB verifies that other state fields remain unchanged.

For the K/V panel, positioned and normalized native Q/K/V operands are reconstructed through the installed native attention dispatcher. Changed operands supply all unspecified ports. Only image-query outputs reach the installed image field; recomputed text-query outputs are discarded. K-only and V-only effects separate dependencies through attention weights and value vectors. They do not by themselves establish semantic address and content roles.

For fixed queries, write `P = softmax(Q Kᵀ / sqrt(d))` and `A = P V`. Let `P'` result from replacing text keys, and let `ΔV` replace only text values. Before native rounding, the three attention changes are

```text
K only:  ΔA_K  = (P' - P) V
V only:  ΔA_V  = P ΔV
K and V: ΔA_KV = (P' - P) V + P ΔV + (P' - P) ΔV.
```

This separates a local key/value interaction from the later nonlinear response of the remaining model. A visible effect of a key edit does not identify a particular noun or relation inside the key vectors. Naming a reusable semantic operation would require more contexts and controls than this source decomposition alone.

The support selector ranks contrast energy in the observed prefixes, using a stable index rule for ties. It chooses the top quarter and bottom quarter before observing any outcome. The omission branch reverts the selected attention output rows to recipient values within a full text-KV intervention. Rescue copies selected image-state rows from that clean future at a declared later boundary; it does not copy a whole final image. Late rescue inherits the actual earlier scheduler state of the broken branch.

## Prediction barriers and interpretation

The first transaction seals all development hypotheses before the first native image suffix. After development outcomes, two small empirical selectors are fitted using development prefixes and their R/A/RA outcomes. Both selectors are sealed before any of the 11 held-out reference, sham or edited suffixes execute. Native transformer and VAE entry hooks guard both prediction phases; model identity and storage/version signatures bind external frozen weights.

The paired selector uses residual, attention and MLP contrast magnitudes, residual/attention contrast cosine, and the recipient's stored-conditioning/current-text alignment. The companion uses only recipient-state features and that alignment. Both are standardized nearest-development-prefix models with an explicit distance-based abstention rule. Candidate predictions remain recorded even when a selector abstains, and all held-out arms still execute. These are small empirical applicability tests, not automatic rule discovery.

For a source final latent `z_S`, recipient `z_T`, and edited future `z`, report

```text
p(z) = dot(z - z_S, z_T - z_S) / dot(z_T - z_S, z_T - z_S).
```

Source has p=0 and recipient p=1. Negative values and values above one are valid. This signed coordinate measures motion along one latent contrast; it does not measure semantic accuracy, fraction of an image restored, or collateral orthogonal to that contrast. A zero native contrast makes this coordinate inapplicable. Native RGB inspection and recorded collateral remain separate evidence.

The local law and downstream interaction answer different questions. In real arithmetic, keeping the gate and MLP operand fixed makes residual and attention replacement affine at this merge:

```text
Δ_R  = R_source - R_recipient
Δ_A  = gate * W_attention * (A_source - A_recipient)
Δ_RA = Δ_R + Δ_A.
```

The worker evaluates the full native-shaped projection rather than this split expression, because BF16 rounding depends on the execution shape. The later consumer is a nonlinear state transition `G` that also receives the preserved text, conditioning, scheduler state and cursor. Thus an interaction measured as `p_RA - p_R - p_A + 1` after the native suffix can arise even when the local writer is affine. It names an observed downstream response; it does not turn the local merge into an independently discovered multiplicative relation operator.

Likewise, equality of working text/image ports would not establish equality of complete states. Persistent conditioning is part of the captured state and is reread by later projections. The hybrid/natural panel retains both the current working ports and that stored conditioning so an equal immediate update can still be followed by different futures. The audit reports the actual equality and divergence observations rather than assuming them.

The campaign predicts empirical effects from observed prefixes. It does not independently forecast every edited RGB pixel. Exact local equations, causal continuation, applicability predictions, and semantic image descriptions are distinct levels of coverage.

## Reproduce the workflow

Use the declared installed public MDB qualification environment and configured mrun fleet. From the MDB package directory:

```sh
# Freeze the 47-row plan from the prior pinned checkpoint plan.
python qualification/submit_flux_family.py /absolute/campaign/plan.json \
  --make-plan /absolute/prior-replication/plan.json

# One CUDA lease, one model load, all branches in-process. No automatic retry.
python qualification/submit_flux_family.py /absolute/campaign/run-01 \
  --plan /absolute/campaign/plan.json

# After that exact job finishes, perform a separate CPU-only audit.
python qualification/remote_flux_family_audit.py /absolute/campaign/audit-01 \
  --producer /absolute/campaign/run-01
python qualification/remote_flux_family_audit.py /absolute/campaign/audit-01 \
  --collect AUDIT_JOB_ID
python qualification/remote_flux_family_audit.py /absolute/campaign/audit-01 --verify
```

The separate audit performs zero model forwards. It verifies the full producer archive, typed CAS, native ports, image Acts, scheduler arrays, PNG hashes, prediction seals, support selection and selector fitting on the storage host. The compact view returns the original 47 PNGs and audited records, with request/archive/analysis bindings. Full arrays remain in the named storage-host evidence directory. Offline auditing does not rerun external-weight projection or attention kernels.

The producer uses strict preflight, a finite CUDA lease, no retries and bounded local work. Its resource envelope is 32 GB RAM, 13 GB VRAM and four CPU threads, based on the prior measured resident workload. The installed mrun kill ceiling is 35.2 GB. The worker allows 3,000 seconds and 10,000 adapter steps, requires 12 GiB free scratch, and caps retained CPU source ports at 1,536 MiB. Multipart custody independently limits compressed/expanded bytes, members, individual files and log transport; the serialization budget was measured before submission.

The execution source is [flux_family_campaign.py](../qualification/flux_family_campaign.py), the frozen geometry and selectors are in [flux_family_plan.py](../qualification/flux_family_plan.py), and the independent auditor is [analyze_flux_family.py](../qualification/analyze_flux_family.py). Generic computation IDs should record this exact scope, applicability, signed effects, counterexamples and unresolved semantics before receiving stronger names.

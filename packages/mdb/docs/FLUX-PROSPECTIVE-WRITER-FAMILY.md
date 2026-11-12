# A prospective FLUX computation family

The frozen support resolver and residual/MLP construction transferred to all six new paired contexts in this campaign. The four primary cases satisfied all twelve route inequalities and all four component predictions. The two separately scored shape probes satisfied six route inequalities and two component predictions. All 36 complete numerical futures matched separate native execution exactly. These are dependent conditions under one specified checkpoint, source policy and numerical profile.

The new result connects a source writer, token addresses, a receiving state and a consumed image continuation. It advances the earlier [incoming-state/net-write decomposition](FLUX-BINDING-RESIDUAL-WRITE.md) and complements the older [six-panel family study](FLUX-COMPUTATION-FAMILIES.md), whose missed predictions and semantic limits remain unchanged.

The model identity is `a09191c5ff813af77bf89d8e4b9a07237befb67c1aef3e722322d3491250fcd0`. The qualified profile is CUDA BF16, batch one, 512×512, four clocks, guidance 1, Python 3.14.6, Torch 2.13.0+cu130, Diffusers 0.39.0 and Transformers 5.14.1. Deterministic algorithms are enabled and TF32 is disabled. Changing this execution profile requires separate qualification.

## What was frozen

Two historical contexts supplied a finite R/A/M component menu. R is incoming residual, A is the ungated text-attention return, and M is the ungated text-MLP return. P and Q name two controlled text-delivery branches in a shared source-image-primed receiver. They are constructed states with paired prompt ancestry.

Before generating fresh prefixes, the rule selected `PQP`: P residual, Q direct attention return, P MLP return, and Q recipient gates. It minimized the mean historical full-image RGB RMS distance to support-P within the eight declared candidates. The historical distances were 0.623 and 12.823 for PQP, compared with Q-to-support distances of 50.664 and 41.345. Recomputing the MLP on the mixed residual/attention state (`PQM`) gave 1.235 and 16.215. These old outcomes are development evidence.

The address resolver was also frozen: A is every changed red/blue token span; B is the first two unchanged valid prompt-content rows after A under the fixed resolver; S=A∪B. It resolves four rows from tokenizer IDs and offsets. It performs no search over downstream activations or images. Rewording changed the actual positions from `[4,11,12,13]` to `[10,18,19,20]`.

The [rule](../../../../artifacts/mdb-computation-family-20261107/family-rule-v1.json), SHA `0609082da7815afb15a3780cb32300014ee6154501d0991e24095cddad9c226d`, was frozen at `2026-11-07T05:34:14.285480Z`, before the fresh-prefix job. It declares every prompt, seed, arm, applicability condition and prediction.

## Mathematical construction

The original joint text writer is sequential. With B denoting a BF16 result at each actual operation:

```text
U = B(R + B(gA * A))
X = B(B(LayerNorm(U) * B(1 + scaleM)) + shiftM)
(a, b) = linear_in(X)
M = linear_out(SiLU(a) * b)
T_out = B(U + B(gM * M))
```

The original named weight/bias tensors, normalization, rotary attention, full projection geometry, strides and rounding order are retained. The functional lane uses primitive operations rather than native module forwards. Native module inputs and returns are captured separately for comparison. This identifies the arithmetic on the actual operands; it does not introduce newly fitted weights.

For each corner, the selected rows combine the declared R/A/M ports using the recipient gates. Outside S, Q's output remains literal Q. `PPP` is therefore P text on S under the Q receiver; `whole_P` installs the entire P text output; `complement_P` installs P only outside S. The wrong-address arm installs the same four P payload rows at a tokenizer-selected disjoint four-row address. Matching payload bytes and row count does not guarantee equal landed intervention norm.

PQP's M was originally computed after P attention. PQP cannot show that attention is irrelevant. It is a transported endpoint construction, whereas historical PQM additionally evaluates the MLP on a crossed intermediate state. Isolated and joint interventions also have different magnitudes, leaving dose, thresholding and downstream nonlinearities as alternatives to a special cooperation gate.

All six arms enter the same Q joint1 image receiver. Each continuation advances its own state through 54 denoiser transitions, including 49 remaining transformer blocks, to the final latent. The original VAE decode follows and is counted separately. The step3 project uses the declared baseline-conditioning reset. No target state refreshes a predicted trajectory.

## Prospective tests

Four primary contexts crossed two wordings with seeds 2027 and 2063. The ordinary base prompt was:

> A red sphere on the left and a blue cube on the right, on a plain white background.

The reworded base prompt was:

> On a plain white background, a blue cube is to the right of a red sphere.

Each paired source swapped red and blue. Separate seed-2081 probes substituted cone/cylinder and sphere/cylinder. The prefix job produced twelve encodings and twelve joint0 input cuts, with zero joint0 target calls and zero VAE/image calls.

Write Q, S, C and W for the final native images of QQQ, support-P, complement-P and whole-P. On all pixels in RGB levels 0–255, the three frozen route predictions were:

```text
RMS(S,W) < RMS(S,Q)
RMS(C,Q) < RMS(C,W)
proj(S-Q, W-Q) > max(0, proj(C-Q, W-Q))
```

Here `proj(x,d)=dot(x,d)/dot(d,d)`. The component prediction was `RMS(PQP,S)<RMS(Q,S)`. These are numerical consumer predictions, with no fitted tolerance or semantic classifier.

| Context | Panel | Q→support RMS | PQP→support RMS | Support→whole RMS | Complement→Q RMS | Frozen route predicates |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Plain / 2027 | Primary | 40.866 | 0.397 | 0.444 | 0.464 | 3/3 |
| Plain / 2063 | Primary | 34.662 | 1.712 | 0.523 | 0.433 | 3/3 |
| Reworded / 2027 | Primary | 36.848 | 1.067 | 0.427 | 0.419 | 3/3 |
| Reworded / 2063 | Primary | 35.294 | 7.424 | 2.342 | 2.178 | 3/3 |
| Cone/cylinder / 2081 | Shape probe | 26.975 | 1.045 | 1.854 | 0.453 | 3/3 |
| Sphere/cylinder / 2081 | Shape probe | 35.786 | 3.286 | 2.594 | 0.408 | 3/3 |

All six component inequalities held. All whole-versus-Q contrasts were nonzero; there were no undefined directional outcomes or missed frozen predicates. The numerical closeness is heterogeneous: reworded/2063 is visibly and numerically less complete than plain/2027.

Wrong-address responses were not zero. Their Q-relative RMS ranged from 2.035 to 19.713, with signed projections of 0.017–0.562 on the support-minus-Q axis. Their partial effects prevent an exclusive-address or uniquely necessary-support conclusion. They were retained as exploratory controls, without inventing a new post-outcome gate.

Unblinded image inspection shows the primary spheres turning blue and cubes moving toward red or magenta. Shape probes retain red/blue striping on the cone and purple cylinders. Agreement with support-P is not equivalent to a clean semantic color swap. The [native image panel](../../../../artifacts/mdb-computation-family-20261107/fresh-native-contact-sheet-v2.png) shows every fresh arm.

## Numerical custody and executed work

The numerical transaction sealed all 36 futures at worker second 200.313, with zero native writer, suffix or decoder target calls. Its seal is `1a7522c736754643e10dd9d94028c53b408d5c48735c7b61708aa5cfda518d17`. It forecasts joint0 formation from inputs, selected endpoints, consumer ports, complete own-state suffixes, final latents, raw VAE outputs and RGB. Native execution starts after that seal.

Fresh producer `job-a2048028d7bf` succeeded in 477.31 seconds, peak RSS 10,851 MB and VRAM 8,472 MB. It made twelve source-joint0 evaluations per lane, 1,944 suffix transitions per lane and 36 VAE decodes per lane. Native execution used 1,776 blocks and 72 typed Acts; the forecast lane used another 72 Acts, for 144 total. There were six common-parent rewinds. The earlier prefix job is paid preparation and is counted separately.

Independent saved-only CPU audit `job-b04ddb750071` succeeded in 75.93 seconds, peak RSS 2,476 MB and zero VRAM/model forwards. It checked 792 complete array fields, 27,648 boundary tensors and 8,568 scalar fields, with no bit, layout or hash differences. It independently reconstructs every selected endpoint with NumPy BF16 rounding, checks source joins, typed Acts and first consumers, and compares the complete decoded outputs. The [audit](../../../../artifacts/mdb-computation-family-20261107/fresh-family-audit-01/compact/audit.json) SHA is `7648793037c898ae9fc3d13032d169927fc65dcb4957ad23be4a370b71b968dc`.

Raw fresh evidence retains 2,720 files / 6,881,397,911 bytes on the durable compute volume. Native plus audit compact collections total 22,997,070 bytes. Original parameters are externalized through 311 named retrieval recipes and a standalone `retrieve_weights.py`; no per-run pretrained weight bank was exported. Activations, intervention states, sealed predictions and measured outcomes remain evidence.

## Reuse through MDB

The [combined atlas](../../../../artifacts/mdb-computation-family-20261107/cross-run-family-01/campaign.json) joins twenty historical occurrences and 36 fresh occurrences. All 56 exact-context queries match uniquely; 112 changed-clock/receiver queries abstain. Eight PQP occurrences form a caller-selected comparison with two source cohorts preserved. The generic IDs are:

| Layer | ID |
| --- | --- |
| Writer law | `flux.joint.sequential-text-write.bf16-reference.v1` |
| Route | `flux.step2.joint0.text-support-to-image.v1` |
| Frozen prospective rule | `flux.step2.joint0.support-family.v1` |
| Curated record family | `diffusion.support-addressed-joint-writer.v1` |

The atlas stores source-qualified occurrences and preserves its generic `not-validated` comparison status. The separate [finite-test conclusion](../../../../artifacts/mdb-computation-family-20261107/family-conclusion-v1.json) records which prospective claims earned support. It does not rewrite raw run status or automatically promote arbitrary applicability.

From the domains workspace, the installed public API can inspect the combined records without a model:

```python
import json
from pathlib import Path
from mdb.computation_atlas import ComputationAtlas

root = Path("artifacts/mdb-computation-family-20261107")
atlas = ComputationAtlas(root / "cross-run-family-01/atlas")
rows = atlas.find(law_id="flux.joint.sequential-text-write.bf16-reference.v1")
print(len(rows))  # 56
campaign = json.loads((root / "cross-run-family-01/campaign.json").read_text())
print(campaign["panels"]["primary"]["route_correct"])  # 12
family = atlas.show(campaign["selected_component_family"])
print(family["rule_status"])  # not-validated: generic comparison does not certify a rule
```

The [offline viewer](../../../../artifacts/mdb-computation-family-20261107/fresh-view-01/index.html) joins six computation records to 36 branches / 72 native-and-forecast image views. Select the corresponding record, then PQP, whole_P, complement_P or wrong_address. Use the image-difference checkbox and continuation-stage buttons for numerical agreement; `Record & evidence` retains the frozen route measurements, counterexamples and source references. If changing the context while a different record is selected, choose the corresponding record or `All branches in this run` to clear that filter.

To reuse the research method, version a source-only address resolver, finite component language, checkpoint/profile, receiving state and consumer horizon. Freeze proposed route effects before fresh state generation; seal numerical futures before target execution; apply typed Acts from a common parent; run the native consumers; audit complete saved arrays; retain misses; then import explicit occurrences. A nearby input remains a new hypothesis until tested. The public [writer](../qualification/flux_binding_family_writer.py), [campaign](../qualification/flux_binding_family_campaign.py), [prefix builder](../qualification/flux_binding_family_fresh_sources.py), [auditor](../qualification/flux_binding_family_audit.py), [publisher](../qualification/publish_flux_binding_family.py) and [renderer](../qualification/render_flux_binding_family.py) implement this campaign.

## Carrier accounting and remaining scope

The saved late-latent/previous-velocity hold experiment was also completed. Its corrected original-order equations match all forty compared fields. A separate zero-forward CPU run, `job-8de56de4c57d`, accounts for the saved RGB contrasts on both the hold-program axis and the original native axis. Velocity-only has native-axis projection 0.960/0.959, versus latent-only 0.230/0.240; the interaction remainder's signed projection changes sign between contexts. These signed coordinates are descriptive, not additive causal shares. See the [full account](../../../../artifacts/mdb-computation-family-20261107/saved-carrier-axes-01/RGB-AXES.md).

The first carrier forecast widened a BF16 product too early and missed by up to six RGB levels. Those misses remain preserved; corrected same-case qualification is not fresh replication. A packaging-incomplete fresh-family attempt was stopped during initialization, and its replacement retained the unchanged rule and inputs. The engineering attempt history remains in the campaign artifacts.

The native Euler path consumes current latent, conditioning and clock to form a current velocity. A constructed hold program that explicitly retains previous velocity has a different dependency. Its result does not show that the next native denoiser reads previous velocity independently.

This closes the bounded transfer study: a specified writer/address/receiver family predicts fresh consumed futures, with exact numerical execution and separate prospective role tests. Source-assisted prefixes, paired prompt directions and original parameter arithmetic remain supplied. Wider clocks, unrelated prompts, independently formed source directions, automatic family induction, unique address necessity and calibrated object semantics remain separate research questions.

The [completed Astra review](../../../../artifacts/mdb-computation-family-20261107/advisor-final-family-review-v1.md), SHA `761f0f3d36a7efb71393dc2c99c8b5df8259676a31392adaf1e1a6b31b469288`, independently recalculates the frozen inequalities and checks all image/source joins. It supports the bounded conclusion and preserves the causal and semantic qualifications above. Its four reporting corrections have been incorporated; original evidence and reviewed draft snapshots remain unchanged.

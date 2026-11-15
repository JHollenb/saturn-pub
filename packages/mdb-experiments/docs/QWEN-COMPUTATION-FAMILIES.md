# Qwen computation families and own-state futures

This campaign ports MDB's generated attention and SwiGLU family protocol to
**Qwen2.5-0.5B-Instruct**, including full-vocabulary logits and autoregressive
KV state. The bounded execution profile is Linux CPU FP32, Transformers 5.14.1,
Torch 2.13.0, eager attention, batch one, default RoPE, and no padding.

The experiment targets zero-based decoder layer 12: the thirteenth of 24 layers.
Each family has natural P/Q references and a four-corner factor experiment under
the Q receiver. One development prompt pair selects a single-factor ranking;
three new pairs test the frozen ranking. Every branch predicts **six emitted
greedy choices**, using one prefill and five cached calls. The sixth choice is
pending: final cache length is prompt length plus five. EOS does not stop this
fixed recurrence experiment.

The exact design, inputs, public wheels, source hashes, qualification receipts,
requests, native publications and independent audits live under
`artifacts/mdb-qwen-families-20261107/`. This is a development snapshot, not a
new published package release.

## The equations

For residual row matrix R, Qwen first applies RMS normalization:

\[
N_w(R)=w\odot R\left(\operatorname{mean}(R^2)+\epsilon\right)^{-1/2}.
\]

Its attention computes biased Q/K/V projections, applies RoPE to Q/K, and
appends the receiver's K/V to private per-layer cache state. There are fourteen
query heads, two KV heads, and 64 channels per head. The grouped KV heads are
repeated seven times for eager attention:

\[
A(Q,K,V)=\operatorname{softmax}\left(Q\,\operatorname{repeat}(K)^T/8+M\right)
          \operatorname{repeat}(V),\qquad
S=R+\operatorname{merge}(A)W_o^T.
\]

M is the full causal additive mask. Finite execution retains the native order
of projections, RoPE, matrix multiplications, FP32 softmax, head merge and
output projection. The attention factor graph separates the positioned query
and key history, called `route`, from the value history, called `value`.
These are operational names, not discovered semantic labels.

The same layer then recomputes its MLP from S:

\[
G=N_{\rm post}(S)W_g^T,\quad U=N_{\rm post}(S)W_u^T,\quad
B=\operatorname{SiLU}(G)\odot U,\quad H=S+BW_d^T.
\]

G and U have 4,864 channels. The MLP factor graph separates `gate` G and
`value` U. Both families have the ideal conditional form
**a state-dependent linear action on a payload**: attention is linear in V
with Q/K fixed, and the MLP is linear in U with G fixed. This does not make
either complete block, its normalization, or its future linear. It does not
authorize reassociating floating-point operations.

At the last layer, the full vocabulary readout is

\[
z=N_{\rm final}(H^{24})W_{\rm head}^T,\qquad i_{t+1}=\arg\max_v z_v.
\]

Each forecast feeds its own i into the next functional call and uses its own
predicted cache. Each native branch independently feeds its own native argmax
and native cache. No native token or future cache repairs a prediction.

## What the interventions change

For attention, a corner combines the selected source's final positioned query
and visible key history with the other selected source's value history. Only
the resulting **final prompt-token head row** is installed before the original
output projection. Earlier output rows and the target layer's persistent cache
remain Q's. The changed residual then enters that same layer's MLP normally.

For SwiGLU, only the final prompt row of G and/or U is replaced. The recipient's
post-attention residual stays fixed. Both families recompute all upper layers,
including their final-token cache writes, and every subsequent token call.

The six roles are `recipient`, `source`, `baseline`, the two individual factors,
and `both`. `baseline` is the QQ no-op control. `both` is a component hybrid
inside Q's receiver; it is not a natural P execution. Attention hybrids use a
one-query eager kernel, so QQ versus natural Q is also a measured finite
arithmetic control; bitwise equality is not assumed.

This cache distinction matters: identical first tokens do not imply identical
computational states or identical later text. Upper-layer cache writes can carry
an intervention into later decisions.

## Prediction, effects and evidence are separate

Natural and hybrid numeric forecasts start from literal token IDs and original
checkpoint parameters. There is no observed activation prefix. All source ports,
formed endpoints, full logits and six-choice futures are computed before the
first native target call **within each family lease**. Actual module-entry guards
enforce that barrier. MDB's `ForecastTransaction` seals the references and
`execute_program` drives forecast/seal/native stages. These are trusted Python
instruments, not an adversarial operating-system sandbox.

The two development leases finish first. Both effect rules and both fresh
request payloads are frozen before either fresh family launches. The second
fresh family receives no first-family outcomes or retuning. There is one global
numeric seal per family lease, rather than a single seal shared between leases.

The prospective effect score is

\[
c=z_{\text{answer Q}}-z_{\text{answer P}},
\]

using single-token answer labels supplied before any forward. Since
\(c=\log(p_{\rm Q}/p_{\rm P})\), lower c favors P **within that supplied pair**.
It need not increase P's absolute probability. Development chooses the better individual factor; fresh contexts
test its strict ordering against the other. Natural headroom and probability
mass on the two labels remain diagnostics. Counterexamples remain evidence.
The ranking is distinct from predicting the model's numbers correctly, and
neither establishes a semantic assignment switch.

An independent saved-only mrun reader authenticates every retained file and
the forecast seal. It compares complete logits, every layer's K/V tensors,
formed endpoints and selected source/projection/residual ports. It checks each
lane's own-token recurrence and cache clocks, records signed errors, evaluates
frozen rankings, and emits public equation-comparison reports. Native Q versus
QQ, mixed finite differences and per-layer cache writes remain visible.

The public runtime cohort is qualified outside the workspace: 27 family-core
tests, 8 mathematical-adapter tests and 4 whole-worker/reader tests. The latter
use a generated tiny Qwen and tokenizer, with native calls forbidden during
forecasts and reader execution. Real-checkpoint error measurements come from
the subsequent model runs, not these fixtures.

## Measured results

All four real-checkpoint leases and four independent saved audits completed.
Each branch consumed its own generated tokens and private caches.

| Family | Development choices | Fresh choices | Max logit / KV forecast error | Frozen fresh ranking |
|---|---:|---:|---:|---|
| Attention route versus value | 36 / 36 | 108 / 108 | 0 / 0 | Route better in 2 pairs; 1 counterexample |
| SwiGLU gate versus value | 36 / 36 | 108 / 108 | 0 / 0 | Gate better in 1 pair; 2 counterexamples |

These are 48 dependent branch futures over four paired contexts and two family
types, not 48 independent tasks. The step-array census contains 14,736 pairs and
71,323,776 positions. A separate selected-port census contains 480 pairs and
8,713,752 positions. Both censuses are bit-exact under the measured profile.
Six-choice agreement says nothing about a seventh choice or another backend.

The frozen margins below are `other score - selected score`, so a negative
margin is a preserved counterexample:

| Fresh pair | Attention: route < value | MLP: gate < value |
|---|---:|---:|
| Swapped letter/value assignments | +0.143497 | -0.149035 |
| Reworded first/second query | +0.186390 | -0.528994 |
| Animals with reversed query direction | -0.241542 | +0.279617 |

Ranking success does not imply a useful answer change. In the first two pairs,
attention route has a lower score than attention value, but both increase the
Q-to-P contrast relative to natural Q. All 24 fresh hybrid futures retain Q's
six emitted choices despite nonzero logits and cache changes. Answer-label mass
also varies substantially: the reworded recipient assigns about 89.3% to the
supplied pair, versus roughly 0.7%–4.4% for the letter-query recipients. This
supports a numerical account and contextual effect records, not a universal
semantic assignment operation.

There is a concrete development branch with a visible continuation difference:

```text
Natural Q:     " B\nQuestion: What color"
Attention V_P: " B\n\nThe following is a"
```

Both emit token 425 (` B`) first. Their next choices differ: 198 versus 271.
The target layer's prompt cache remains exactly Q's, while the V_P branch's
layer-13 cache differs by up to 0.652754 in a retained scalar. All nontrivial
hybrid prefill cache changes begin at layer 13; layers 0–12 stay Q's. Forecasts
predict both the persistent state differences and the resulting choices exactly.

The attention QQ control itself differs slightly from natural Q because of the
one-query kernel geometry: the largest measured first-readout logit difference
is about 1.335e-5. Its forecast still matches its own native execution exactly.
The MLP QQ control matches natural Q exactly. These controls remain distinct
from forecast error.

The canonical atlas publications contain 48 exact occurrences, 48 successful
exact lookups and 96 clock/receiver abstention checks. They bind token IDs and
the original parameter/config/tokenizer manifest as well as source code,
execution profile and horizon. Counterexamples accompany the records. The
earlier development publication snapshots remain preserved; the canonical
development publications are version 02.

## Using the workflow

The maintained model-specific execution entry is
`qualification/submit_qwen_family_campaign.py`. The generic CLI compiles and
scores protocols; it does not have a `mdb family run` command. Execution uses
the registered worker and public `run_graph` / `execute_program` APIs.

From the domains workspace, a new request can be staged without executing it:

```sh
./common/bin/common exec python mdb/qualification/submit_qwen_family_campaign.py stage \
  --base artifacts/mdb-qwen-families-20261107 \
  --family attention --phase development --run /absolute/new-request-directory
```

The staging directory contains a reviewable request, exact command, script
closure, public wheels, finite counts and preflight configuration. The frozen
launcher submits one finite mrun lease after its exact review. Original
historical directories are immutable; use a new directory for a reproduction.

`evidence.py` authenticates the scheduler command, payload custody and output
inventory, collects compact metadata, and admits an independent saved audit.
The publication helper indexes audited equation reports and their continuation,
effects and counterexamples in `EquationCatalog` and `ComputationAtlas`.

For an indexed run, use the cohort's installed MDB CLI:

```sh
mdb atlas --atlas /absolute/publication/atlas find \
  --applicable-to /absolute/publication/example-applicability.json
```

Exact occurrence lookup is deliberately narrow. Changing the clock or receiver
must abstain. Wider family predictions are the separately frozen and tested
ranking rules, not a side effect of atlas lookup.

Each result includes `parameter-artifacts.json` and standalone
`retrieve_weights.py`. They bind original safetensors files, archive headers,
named tensor shapes/dtypes/hashes, tied aliases and FP32 runtime bindings. No
pretrained weight bank is exported per run. The helper can verify originals or
reconstruct a requested parameter archive when needed.

The compact collections keep metadata locally and retain the complete arrays
on the fleet. `retrieve_evidence.py` fetches one inventory-authenticated NPZ
member without model execution or parameter retrieval. The exercised example,
`examples/native-value-choice1.npz`, is 964,006 bytes: 151,936 logits, all 24
layers' K/V tensors, input token 425 and next choice 271. Its cache length is
14, one beyond this prompt's thirteen tokens.

The canonical fresh attention atlas was also queried through the installed CLI;
`examples/atlas-find-result-v1.json` contains its one exact match, and the adjacent
receipt binds the command, applicability, publication and output hashes. Both
examples use saved evidence and perform zero model forwards.

The four model campaign leases sum to 199.31 seconds of scheduler elapsed time;
the four independent audit leases sum to 24.57 seconds. Peak campaign RSS is
3,316 MB, with zero VRAM. These are job sums, not end-to-end research latency or
speedup. Native raw inventories retain 662,016,979 bytes remotely, excluding
their inventory wrappers. Native and audit compact collections total 6,585,473
bytes locally; this excludes wheels, publications and the separately retrieved
964,006-byte example. The measurement summary preserves the complete accounting.

## Astra review and the next causal cut

Astra reviewed the four audits, frozen rankings, cache changes, implementation
and preceding FLUX family posts. The immutable review is
`advisor-result-review-v1.md` in the artifact root. It distinguishes known
architectural algebra from the empirical source-to-consumer account. In ideal
arithmetic the attention four-corner mixed term is

\[
(\Pi_P-\Pi_Q)(V_P-V_Q),
\]

where \(\Pi\) is the positioned, causally masked routing matrix. The SwiGLU mixed
term is

\[
(\operatorname{SiLU}(G_P)-\operatorname{SiLU}(G_Q))\odot(U_P-U_Q).
\]

Saved finite differences also retain operation-order and rounding effects.
Downstream logit differences include the nonlinear receiver. Their magnitudes
do not rank the importance of differently sized endpoints.

The useful mechanistic inference is that the development branch's later token
difference passes through upper-layer cache state: the pending token and clock
are shared, layers 0–12 have identical caches, and the intervention has expired.
This identifies a carrier class under the declared state closure; it does not
localize a necessary layer, row, head or K-versus-V component.

The next proposed experiment is a four-corner restoration of upper-layer
final-prompt-row **K versus V**, after checking the saved row-level support.
Hold pending token 425, positions and every other cache entry fixed; combine
natural-Q versus edited K with natural-Q versus edited V. Seal each own-state
forecast before its native continuation and measure the 198-versus-271 next
choice together with full logits and KV. This proposal was unexecuted at the
November 7 campaign's completion. The [November 10 cache restoration follow-up](QWEN-CACHE-RESTORATION.md)
has now executed it at that later observed-cache frontier: V replacement alone
reproduces the edited branch's five choices, while K still changes logits and
state. All twenty new forecast/native choices and retained arrays match exactly.
That result concerns the observed formatting branch, without assigning an
object or language concept.

## Scope

This is a reusable discovery workflow across FLUX and one trained autoregressive
Qwen checkpoint, with model-specific adapters and measured execution profiles.
The numerical account still performs the original tensor computation and uses
the original learned parameters. It is not cheaper inference, a compressed
surrogate, arbitrary-Qwen qualification, automatic concept naming, or enumeration
of every future. Sampled generation, padded batches, alternative RoPE, Qwen3
autoregressive decoding and other model families need separate adapters and
qualification. Frozen scientific rankings can fail even when the full numerical
forecast succeeds.

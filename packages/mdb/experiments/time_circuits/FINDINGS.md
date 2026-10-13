# MDB E6/E20 reproductions

Completed through source-sealed MDB workers on Beast's RTX 4080, one resident
model per finite mrun lease, sequential CUDA, strict admission and no retries.
The native paper protocol copies retain their original bytes. MDB owns typed
consumer inputs, candidate writes, durable publication, cuts and receipts.

## E20: the seeded residual acts through formed late K/V

Full `job-a7c0623183af` completed all **42/42 rows**: Qwen2.5-1.5B base,
CUDA FP32 eager, TF32 disabled, 14 identities × 3 contexts. The checkpoint and
source identities are in [provenance](PROVENANCE.json) and the collected request.

| Observation | MDB result |
|---|---:|
| Median L12 residual-seed target logprob gain | +3.514793 nats |
| Median fraction of seed effect removed by L22–27 recipient-band block | 0.924189 |
| Median fraction recovered by seeded-band rescue into the unseeded recipient | 0.871323 |
| Wrong-donor candidate wins after wrong seeded-band transplant | 37/42 |
| Same-run native observable rows | Exact in all three contexts |
| Full-logit baseline replay from the stored input cut | Exact in all three contexts |

The paper's principal medians reproduce to the displayed precision. This is a
bounded source → formed memory → native answer result. The fractions describe
**target logprob gain**, not percentages of reasoning or explained variance.
All signed rows, controls, denominator diagnostics and counter-trends remain
in the report. The original worker's CIs resample items; the later paper
reviewer's identity-clustered intervals use 14 independent identity groups.

Recipient-band self-reinstall has zero effect; restoring the seeded band after
blocking restores the seed effect exactly. Residual self-noop has a tiny
FP32 split/full-forward discrepancy (median −2.38e−7 nats), retained as a
diagnostic. MDB's comparison to the same split-native calculation is exact.
The comparison does not assert full-logit equality for every intervention arm.

The source/state archives passed an offline audit of **39 cuts**, **30 native
receipts**, **28,394,560 referenced tensor bytes** and **406 sealed source files**.
Custody verification and WATCH → CLAIMS → XREF run/verification returned clean.
The worker took **34.34 seconds**; page encoding across the three sessions took
**0.87 seconds**. The earlier three-item smoke used per-file durability and
took 107.98 seconds. These different geometries do not define a speedup ratio.

## E6: the feature-only gap persists in an exact instrument lane

Full interchange `job-f953178a6b6e` completed **20/20 items**, zero errors:
12 identity and 8 color items, both attention-freeze modes. This is the
Gemma-2-2B circuit-tracer E6, using the paper's pinned BF16 nnsight instrument,
Gemma Scope transcoders and native clean reconstruction-error policy.

All **1,400 MDB/native full-logit comparisons** and **1,400 feature-table byte
checks** were exact. First and last saved intervention inputs replayed exactly
for all 20 prompts: **40/40 replays**. Clean-value reinstall had zero Δlogprob
on every item in both attention settings.

Median fraction moved toward the filler answer:

| Feature cells swapped | Identity frozen / unfrozen | Color frozen / unfrozen |
|---|---:|---:|
| Subject, second-half layers | 0.073 / 0.024 | 0.227 / 0.190 |
| Subject, all layers | 0.036 / 0.054 | 0.446 / 0.330 |
| Subject and later positions, all layers | 0.027 / 0.079 | 0.321 / 0.327 |
| All positions, all layers | 0.027 / 0.079 | 0.321 / 0.327 |
| Graph's top-50 features | −0.050 / −0.135 | −0.015 / −0.194 |

All-layer subject swaps flipped **0/12 identity candidates** in both modes,
and **1/8 frozen / 2/8 unfrozen color candidates**. All-position swaps flipped
**1/12 frozen / 0/12 unfrozen identity candidates**, and **3/8 / 4/8 colors**.
Graph top-50 swaps flipped no candidates. Signed negative fractions are
counter-direction observations, not rows to discard.

The broad paper trend reproduces. Individual floating values differ from the
historical nondeterministic lane: for example the frozen subject-color median
is 0.446 here versus about 0.42 historically. Equality is claimed against
same-run native calls in the declared deterministic lane.

The durable archive audit verified **1,460 cuts**, **1,440 native receipts**,
**20 graph references**, **24,564,424 unique tensor bytes** and **406 sealed
source files**. Custody and evidence run/verification were clean. The graph
references are frozen structural instrument artifacts; semantic bindings
remain unassigned.

### Original zero-ablation panels

Frozen `job-cf3ce91547f2` and unfrozen `job-4a0e39eeaca6` each completed
20/20 items, zero errors, **760/760 exact full-logit native comparisons**,
760/760 unchanged feature tables and **40/40 exact input-cut replays**.
Their worker times were **399.3 / 350.4 seconds**. Each panel retains its
graph-selected and three-seed size-matched random deletions, including effects
that increase the target's logprob.
Each archive independently verifies 820 cuts, 800 receipts and 20 graph
references. Referenced tensor bytes total 1,001,876 frozen and 1,001,896
unfrozen. Custody verification and evidence run/verification are clean in both.

Median target Δlogprob in nats:

| Zero-ablation arm | Identity frozen / unfrozen | Color frozen / unfrozen |
|---|---:|---:|
| Top-50 subject features, best single layer | +0.011 / −0.005 | −0.078 / −0.055 |
| Top-50 subject features, all layers | +0.025 / +0.041 | −0.185 / +0.083 |
| All selected subject-position features | +0.303 / +0.132 | −0.037 / +0.229 |
| Graph's top-50 features at any position | −0.164 / +0.161 | −0.294 / +0.178 |
| Size-matched random active features at any position | −0.015 / −0.008 | +0.009 / +0.013 |

No candidate flipped in these arms or the other top-5/top-20 arms in either
setting. Vocabulary flips remain separately recorded: some clean color
vocabulary predictions were already different from the requested color.
The graph's largest subject-influence layer is L0 in 11/12 identities and
8/8 colors; median replacement scores are 0.699 / 0.681. These reproduce the
original graph-localization trend, without treating influence as a semantic
binding or deletion effect as a universal necessity result.

### Native repeatability diagnostic

Historical-policy smoke `job-84904f2638ec` retained 140 feature interventions
and two ordinary native references per intervention. Full-logit equality was
**44/140 for MDB versus native and 44/140 for native versus native**. Final
answer-position equality was 48/140 in both comparisons. Maximum final-logit
drift was 1.378418 for MDB/native and 1.3125 for native/native, with no top-1
disagreements. This is a native instrument repeatability limitation, with
substantial floating differences that must remain visible.

The pinned implementation uses BF16 CUDA `index_add_` for feature decoder
deltas. Enabling PyTorch deterministic algorithms selects deterministic CUDA
execution without modifying the pinned instrument source. Deterministic smoke
`job-715744154f27` achieved 140/140 exact MDB/native and 140/140 exact ordinary
native-repeat comparisons. The full interchange proof uses that explicitly
different numerical contract and preserves the older diagnostic unchanged.

### Durability and boundaries

Full E6 worker time was **658.5 seconds**, including **89.66 seconds** of
additional native-reference calls. The measured publication regions total
**370.95 seconds**; page encode/decode totals are **1.75 / 0.55 seconds**.
These are serial measured regions, not a controlled overhead decomposition.
The cost points toward cut/manifest publication as the next optimization target.
It does not demonstrate faster execution than the original 188-second native
interchange assay, which had fewer reference calls and no MDB durability.

E6's native K/V comparator remains the historical, separate 42-identity/
25-color panel. A new paired 20-item K/V comparison was not run here. The
later reviewer features-plus-error experiment materially reframes color:
all-position joint swaps recovered 0.990 of its filler effect, while the
identity joint swap recovered 0.314. Those are historical reviewer results,
not new MDB outputs. The feature-only gap cannot establish a general failure
of the complete instrument or count reasoning circuits.

## Mechanics, retention and use

Early E6 attempts exposed a missing transcoder-width attribute, record-based
graph targets, and native nondeterminism. Their failed reports/logs and
available observations remain custodied; none are interpreted as a scientific
null. The corrected runs have new source/request identities and no automatic
retry lineage.

Focused tests cover typed address rejection before native execution, private
parent/sibling input tensors, abort evidence, publication rollback, replay,
journal recovery after a backing-page deletion, archive traversal rejection
and source/config tampering. The existing focused MDB regression suite passed
205 tests with five optional skips before the final archive-test additions;
the new research/archive suite passes 13 tests. Installed-wheel imports outside
the workspace load no Saturn, Saturn-pub or common checkout.

These are bounded research-input cuts, not a newly qualified Gemma interactive
BenchSession or arbitrary checkpointing inside circuit-tracer. The ordinary
Qwen virtual-context K/V surface remains read-only. The E20 FP32 native
formed-K/V adapter does not convert or extend that numerical lane.

Use the [run, collection and audit guide](README.md). Large archives and weights
stay in custody stores; the compact evidence manifest accompanies the source.

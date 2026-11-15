# Qwen: which cache component carries a changed continuation?

On **Qwen2.5-0.5B-Instruct**, restoring an edited group of cached V rows into
natural Q state reproduces the edited branch's next five greedy choices with
Q keys. Restoring those values to Q removes that trajectory even with edited
keys retained. This is conditional sufficiency and necessity within one
four-corner replacement experiment. It does not identify a minimal layer,
head, coordinate set or semantic operation.

The [preceding family campaign](QWEN-COMPUTATION-FAMILIES.md) predicted complete
attention/MLP interventions and their own-token futures from raw prompt IDs.
This follow-up starts later, from **two observed post-prefill caches**. It tests
what carries one already observed continuation difference. The model profile
remains CPU FP32, eager attention, default RoPE, batch one and all-attended.

## The shared state and four interventions

Natural Q used the literal prompt:

```text
A: red
B: blue
Read B.
Answer:
```

The earlier layer-12 attention value intervention produced an edited recipient
branch, E. E is a counterfactual Q execution, not natural P. The current
restoration experiment explicitly consumes saved E cache rows as its payload.
It makes no donor-free or raw-prompt forecasting claim at this new frontier.

Both saved states have thirteen cached prompt tokens and the same pending token
425, ` B`. The original layer-12 intervention has expired. A bytewise inventory
shows that their only cache differences are row 12 in layers 13–23. Each layer
has 128 changed K and 128 changed V coordinates: 1,408 FP32 coordinates per
component group. The V payload is 5,632 bytes; this is an intervention footprint,
not the size of a complete model or forecast.

Four typed MDB Acts select the K and V sources independently. All other cache
bytes, positions, weights and the initial pending token stay fixed. Keys are
already post-RoPE and are copied directly. Each branch then evolves its own
tokens and cache without further restoration.

| Corner | Starting K | Starting V | Next five decoded choices | First c = z271 − z198 |
|---|---|---|---|---:|
| QQ | Q | Q | `\nQuestion: What color` | −0.083610535 |
| EQ | E | Q | `\nQuestion: What color` | −0.104911804 |
| QE | Q | E | `\n\nThe following is a` | +0.134693146 |
| EE | E | E | `\n\nThe following is a` | +0.125324249 |

These strings follow the common ` B`. The first new choice is token 198 or 271,
respectively. Five cached calls consume positions 13–17 and finish with cache
length 18; the fifth new choice remains pending.

All **20/20** forecast/native choices agree. Full logits, all K/V tensors and
all recorded layer attention returns and block outputs are byte-equal:
**1,980 array pairs / 5,865,000 overlapping positions**, zero error. Two corners
are historical replay controls; the other two are new mixed-state futures.
These choices and repeated cache positions are dependent measurements.

QQ matches the original natural `recipient` continuation, and EE matches the
original `value` continuation, in all 51 shared fields across five calls:
255 exact array comparisons per control. This QQ is distinct from the older
one-query attention QQ hybrid and its finite arithmetic floor.

## The executable and mathematical account

Between cached calls, the declared state is

\[
S_t=(C_t,i_t,p_t;\theta,\text{mask/position policy}),
\qquad
(z_t,C_{t+1})=F_\theta(i_t,C_t,p_t),\quad
i_{t+1}=\arg\max z_t.
\]

C is the complete private K/V history; i is the pending input token. The next
call derives its residual stream from that token's embedding. The previous
call's transient residual need not be retained at this between-token boundary.
Each forecast feeds back its own argmax and cache, and the native execution
independently does the same. The account uses the original learned parameters
and full tensor equations; no cheaper surrogate is claimed.

At the first affected reader, layer 13, the current query and newly formed K/V
are shared because all lower-layer computations still agree. The head reads
the selected historical cache and appends the common new row. Writing Π for
its masked, positioned attention weights, the ideal head equation is

\[
A_{XY}=\Pi_X V_Y,
\quad
A_{QE}-A_{QQ}=\Pi_Q(V_E-V_Q),
\quad
A_{EQ}-A_{QQ}=(\Pi_E-\Pi_Q)V_Q.
\]

Here V includes the new common row and the grouped-query expansion. The
ideal mixed head term is \((\Pi_E-\Pi_Q)(V_E-V_Q)\). This is conditional payload
linearity in the attention primitive. Later layers and the vocabulary readout
are nonlinear; the actual forecast preserves native finite operation order.

For the common-input first readout, c is an exact log-odds contrast:

\[
c=z_{271}-z_{198}=\log(p_{271}/p_{198}).
\]

The V replacement changes c by +0.218303680 with Q keys and +0.230236053
with E keys. K replacement changes it by −0.021301270 with Q values and
−0.009368896 with E values. The finite mixed readout term is

\[
c_{EE}-c_{EQ}-c_{QE}+c_{QQ}=+0.011932373.
\]

These are deterministic signed effects in one state. Later readouts can also
include feedback from different emitted tokens. The mixed term does not by
itself establish a semantic conjunction or argmax synergy.

## What the carrier trace shows

All nonidentical comparisons first differ in the recorded attention return and
block output at layer 13. Their newly appended K and V rows first differ at
layer 14, on the first cached call. This is consistent with the observed chain:

```text
earlier layer-12 head intervention
  -> changed upper-layer final-prompt cache rows
  -> layer-13 attention read at common pending token425
  -> changed block output after native attention, residual and MLP computation
  -> layer-14 new K/V write
  -> own-token and own-cache continuation
```

The earliest affected reader is not thereby a sufficient reader. Edited
historical rows remain directly available to layers 14–23 as well.

K remains causally active. EQ versus QQ changes first-call logits by as much
as 0.178557873 while retaining all five choices. QE versus EE changes them by
up to 0.175084829 while likewise retaining all five choices. Thus V alone
reproduces the tested **choices**, not EE's full logits or state. V-only is
also an initial-cache intervention: later K and V writes recompute normally.

## MDB execution, evidence and reuse

`qualification/qwen_cache_restoration.py` constructs content-addressed Frames
and explicit Act read/write sets, verifies the complete source footprint and
exact landing, and forks private states. `ForecastTransaction` captures the
observed frontier and seals all twenty forecasts before original native module
entry is permitted. The barrier is trusted Python plus module hooks, not an
adversarial operating-system sandbox.

`qualification/qwen_cache_restoration_audit.py` independently authenticates
inventories, sealed predictions, Frame/Act identities, all arrays, token
recurrence and immutable cache history. It accesses the old control futures
only after the new native executions. Numerical differences remain measurements
and do not invalidate otherwise sound mechanics.

Twenty tests passed outside the workspace against the frozen installed public
MDB/Saturn wheels. They cover a generated tiny model's complete worker
transaction, private cache/token behavior, the saved-only reader and hostile
integrity mutations. Qualification loads no pretrained assets.

The canonical artifacts are under
`artifacts/mdb-qwen-cache-restoration-20261110/`:

- `campaign-01`: native job `job-0f87aa985a36`, 47.24 scheduler seconds,
  3,382 MB peak RSS, zero VRAM.
- `audit-01`: saved-only job `job-f233ab134239`, 6.36 seconds,
  268.6 MB peak RSS, zero model loads or forwards.
- `publication-02`: four exact-context atlas records, four exact lookups and
  eight changed-clock/state abstentions. Earlier `publication-01` is an
  incomplete metadata publication, preserved separately.

The native raw inventory is 50,838,075 bytes remotely; native compact metadata
is 621,529 bytes. Audit output is 4,935,261 bytes. These exclude input staging,
wheels and later atlas publication. No pretrained parameter bank was exported;
the output retains `parameter-artifacts.json` and `retrieve_weights.py`.

The entry point is `qualification/submit_qwen_cache_restoration.py`. Staging
creates a reviewable request without execution; use a new run directory:

```sh
./common/bin/common exec python mdb/qualification/submit_qwen_cache_restoration.py stage \
  --base artifacts/mdb-qwen-cache-restoration-20261110 \
  --run /absolute/new-cache-reproduction
```

The frozen launcher admits one finite mrun lease after its exact mechanics
review. Its `collect` action authenticates custody and retains compact metadata.
The same stager supports `--kind audit --source /absolute/native-run` for the
independent reader. Public `EquationCatalog` and `ComputationAtlas` record the
generic recurrence ID, exact state applicability, signed consumer effects and
continuation. They do not infer a semantic name or generalize the record to an
unobserved cache.

The installed CLI was exercised on the V-only record, returning one exact
match with its continuation and signed effects. Its command and output are
retained in `atlas-find-QE-receipt-v1.json` and `atlas-find-QE-result-v1.json`.
Using the cohort's installed MDB:

```sh
mdb atlas --atlas artifacts/mdb-qwen-cache-restoration-20261110/publication-02/atlas find \
  --applicable-to artifacts/mdb-qwen-cache-restoration-20261110/example-QE-applicability-v1.json
```

## Advisor interpretation and next cut

The immutable `advisor-result-review-v1.md` independently recomputes the
reported totals and effects. Astra supports conditional V-group sufficiency
and necessity in these replacement contrasts, while distinguishing token
agreement from state equality and the first reader from a sufficient layer.

The next proposed localization holds keys at Q and divides edited V into
**layer 13 versus layers 14–23**. QQ and QE supply the none/all controls; two new
variants would separate the first reader from the remaining direct reads.
That follow-up has not run. Fresh contexts, minimal head/coordinate support,
semantic binding switches and general K/V rankings remain open.

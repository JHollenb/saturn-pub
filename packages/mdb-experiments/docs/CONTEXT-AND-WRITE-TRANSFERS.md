# Instruction context, MLP writes and receiving state

The matched-context experiment reproduced the earlier RSI error and predicted
two reciprocal final-MLP write transfers. The mixed-context write repaired the
repeated-SUB recipient. The repeated-SUB write weakened the mixed-context
recipient without flipping its register bit. Every completed case matched all
eleven heads' sealed sign/argmax predictions.

This extends the [raw-input sequence account](RAW-INPUT-SEQUENCE-FORECASTS.md).
It supplies a bounded causal example of why computation records need the
receiving state, reader and input context alongside the law address.

## The comparison

Both contexts start with RDI=262, RSI=3, all other registers zero and flags zero.
The first instruction is the same seven-byte `SUB RDI,1` encoding. The actual
15-byte input windows are:

```text
Repeated SUB: 4881ef010000004881ef0100000000
Mixed:        4881ef010000004831f74881c70100
```

Only byte indices 8 through 13 differ. Those bytes enter the current model
input. This establishes sensitivity to trailing-window context; it does not
establish semantic execution of future instructions or isolate decoder
mediation from the other input paths.

The same complete R3 checkpoint and source equations were used: 87 parameter
arrays, 291,851 scalars, NumPy FP64 reference, native Torch FP32 CPU/thread1.
The plan fixed seven new validation calls and three source-bound mixed-context
controls from `job-62804158d4e4`. The existing controls were not rerun or counted
as new observations.

## Recipes and measured outcomes

`zero` replaces the early `post.0` return with zero. `U_swap` additionally
installs the same context's independently predicted native final-mixer U.
`U_swap_P2native` also installs its predicted native final-MLP P2. Every payload
is cast to FP32 and sealed before native release.

The reciprocal transfers keep the recipient's zero/U recipe unchanged and
install the other context's **U-conditioned predicted P2**. The donor is the
other `U_swap` forecast, not its unedited native forecast or a measured donor
activation.

| Context | Native | Zero early MLP | Restore U | Also native P2 | Other-context P2 |
|---|---:|---:|---:|---:|---:|
| Repeated SUB | +6.551817 | +9.660821 | −0.644844 | +1.539284 | +2.444199 |
| Mixed | +10.275504 | +10.972253 | +3.431013 | +5.961761 | +0.341964 |

Values are measured RSI bit1 logits. Positive means that bit remains set.
The original sealed forecasts predicted the negative repeated-SUB result and
both transfers before any native case was executed.

Both zero-only branches leave RDI at 262 instead of the native 261. Restoring U
recovers RDI 261 in both contexts but also changes RSI 3 to RSI 1 in the repeated-SUB
context. Either the native-P2 restoration or the mixed-context P2 transfer
repairs that RSI bit. The reverse transfer preserves RSI3. All remaining
registers and flags, instruction length and store/syscall decisions remain
unchanged under these two transfers. Auxiliary numerical heads still change;
full target and collateral reports are retained.

## Separating the write from its recipient

Both transfers preserve the recipient's actual raw roots, architectural
before-state, decoder z, final U and final carrier M exactly. Only the installed
P2 return changes. The final register reader is additive:

```text
register logit = copy prior + Wreg · (M + P2) + bias + FP32 residuals
M = S + U, with its FP32 addition residual retained
```

For RSI bit1, the measured carrier/prior/bias contribution is approximately
7.698886 in the repeated-SUB recipient and 8.685693 in the mixed recipient.
Their endogenous P2 contributions are −8.343730 and −5.254680 respectively.

The observed donor-write contrast projects to ±3.089049889 through the actual
register weight row. The measured transfer effects are +3.089042664 and
−3.089048386. The audit accounts separately for analytical donor-payload error,
FP32 final-add rounding and FP32 readout rounding; the full 512-reader component
table remains available. Recipient-carrier differences are zero within each
transfer.

The resulting asymmetry is therefore understandable: adding about 3.089 repairs
a margin near −0.645; subtracting it from +3.431 leaves a positive margin near +0.342.
The old-context write is not intrinsically an error write. Its consequence
depends on the receiving carrier and reader margin.

These are matched transfers with endogenous natural writes and frozen
analytical off-diagonal payloads. They are not a bit-exact fixed-payload2×2
factorial. A full 48-channel write transfer establishes a causal contribution
here; it does not identify a minimal feature or prove necessity of an individual
coordinate.

## A source-derived applicability prediction

Before execution, a separate hypothesis transferred the historical repeated-SUB
RDI258 observation to RDI262. Every register's lowest two bits, the complete
window and flags match. Rowwise input construction, causal bit-axis mixers and
rowwise MLPs imply that higher register bits cannot affect the register-reader
prefix through row 1.

That prediction held exactly for native S/U/M/P2/h/z and the RSI bit1 logit.
All three predicted signs also matched: native positive, U restoration negative,
common-P2 restoration positive.

The intervention recipes have an explicit numerical qualification. Historical
payloads came from equations conditional on an observed FP32 prefix; current
payloads come from the complete raw-input FP64 account. Their low-row U and P2
payloads differ by at most 1.90735e−6 and 1.07288e−6. Edited outputs consequently
need not match bit for bit; the new RSI logits differ from their historical
counterparts by 3.81470e−6 and 4.76837e−6. Those differences are preserved.

The rule is reader-local and source-derived. It makes no prediction about
pooled heads or higher bit positions, and it was not automatically induced from
examples. The mixed window supplies a concrete boundary on its applicability.

## Execution integrity and retained failure

Successful job `job-053353849ab7` completed seven native calls, reused three
prior controls, and executed 13 new typed Acts and 5,805 replay primitives. All
seven complete rows matched all head decisions and decoded state; the largest
head absolute error was 3.30452e−5. Native and typed replay matched exactly for
all eleven heads. The offline auditor verified 642 raw artifact files, source
and archive bindings, all 36 retained port arrays per row and all input-root
joins. The two additional prior S/h arrays are explicitly replay-derived at
original native ports; the original 34 retained arrays were equality-joined.

The first attempt, `job-3cfcb0e595d7`, completed `old.native`, then entered
`old.zero` and failed because the hook cloned a tensor constructed outside the
recorder. MDB rejected that undeclared root. The fix constructs the same frozen
NumPy payload inside the recording dispatch. It changes no equations, cases or
payload recipes. A tiny mechanics check reproduces the old failure and verifies
the corrected hook.

The repaired run's complete forecast and payload descriptors **and addressed
array bytes** match the first attempt's globally sealed predictions exactly.
The repeated native control also matches exactly. Thus the original prediction
precedes all exposures, while the successful run's seven calls include one
repeated completed row and one previously entered partial row. They must not
all be called first exposures.

| Custody item | SHA256 |
|---|---|
| Original forecast seal | `c3724d86a6b6740d21c9b098905223ef1564e5f7050c2f9a1691ff5d030416ce` |
| Successful execution seal | `3dd4154fe98925912ed0477babbe2fc618c3305bded18fd82117f42bc4f47135` |
| Successful final event head | `7f1a8bff99cb15b4e4fa25bd63647135ba335d51be8bec94180fa50c74bd1bfe` |
| Successful archive | `7ef8530cd07e0a0f549e6dee8bcd213bc68b38522a4c1f3148f9e880285263fa` |

The [plan](../qualification/context-window-plan-20261030.json),
[structural hypothesis](../qualification/context-prefix-hypothesis-20261030.json),
[campaign](../qualification/context_window_campaign.py),
[offline auditor](../qualification/summarize_context_window.py),
[historical comparison](../qualification/compare_context_low_prefix.py) and
[portable index](../qualification/context-window-20261030.json) retain the
experiment and its limits. Plans/harness were pushed in `da92e52`; the recording
repair was pushed in `9e28090`.

The public source, installed package and shipped wheel match across all 173
runtime modules. Runtime qualification remains the existing 111 installed-wheel
tests; this campaign additionally checks the corrected hook mechanics, source
hashes and the five qualification scripts with Ruff. No runtime module changed.
The offline Saturn evidence sweep reads ten derived numerical rows and verifies
their stream mechanics. It reports zero watch findings and zero automatic
claims; its scope is `target-streams-only`, not an independent certification of
the model mechanism or MDB receipts.

## Inspecting and reusing these records

The following commands were executed with the installed package outside the
workspace. Paths below assume the collected run is available locally:

```sh
mdb forecast verify artifacts/mdb-context-window-20261030/run-02/extracted/context-window-result/transaction \
  --expected-seal 3dd4154fe98925912ed0477babbe2fc618c3305bded18fd82117f42bc4f47135
mdb ports artifacts/mdb-context-window-20261030/run-02/extracted/context-window-result/captures/old.U_swap_P2other/trace.json \
  --select 'module:post.2:return' --first-consumers
mdb equations find --catalog artifacts/mdb-context-window-20261030/run-02/equations-source-qualified \
  --reader 'regs[6,1]' \
  --applicable-to artifacts/mdb-context-window-20261030/run-02/examples/exact-context.json
```

The port query identifies return `v000619`, with its first direct consumer the
residual `aten.add.Tensor` at PC512. The equation query returns exactly one
matching record. Changing the implementation digest in the query returns zero
matches. The query JSON is copied from that record's `tested_context`.

Ten immutable source-qualified records retain all head comparisons, native
baselines, additive terms, and explicit targets: all 64 RDI bits plus RSI bit1.
Aliases live separately in the versioned portable index. An earlier derived
publication had source identity in its binding but omitted it from the
applicability predicates. Its ten records and index are preserved separately;
the source-qualified revision adds explicit source and horizon conditions.
The scientific reports and raw evidence are unchanged.

For another campaign, fix the cases and competing predictions, seal the complete
numerical forecasts and replacement payloads, execute the native panel, check
replay independently, and compare every head and the whole committed state.
Then index the measured response with exact source, context, reader and horizon
conditions. A matching lookup is evidence retrieval; a new state outside those
conditions is a new prediction experiment.

## What makes the next campaign faster

Measured worker time was 74.35 seconds, scheduler elapsed 91.97 seconds, and peak
resident RAM 1028.2 MiB. The equation calculation itself took 0.05434 seconds.
Forecast validation, cloning, hashing and durable publication together took
28.75045 seconds; declaration/full-weight publication took 14.48764 seconds;
observation publication took 22.27237 seconds. Native execution, recording and
replay together took 6.80809 seconds.

The next performance work belongs in transaction artifact handling: pack
immutable arrays, avoid duplicate serialization, share source-bound weight and
trace references, and measure validation/copy/hash/serialization/fsync costs
separately. These timings identify a combined handling/publication bottleneck;
they do not isolate filesystem cost or establish a speedup.

For the symbol table, useful generic labels are
`register-readout/additive-write-response`, `post.2.return → regs[6,1]` and
`matched-window/register-prefix-0:1`. Exact evidence identity stays separate
from revisable aliases and applicability hypotheses. Further discovery should
test successor states after the RSI error and identify which input pathways
account for the trailing-window effect. This assay ends at the first forward;
it makes no new continuation claim.

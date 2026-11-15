# Predicting other inputs

The unchanged independent equations predicted all eleven output-head decisions
in **42 executed rows across two new MDB campaigns**. These include different
register values, initial flags, high bits, instruction bytes, immediate values,
register routing, and interventions. Maximum absolute head error was
`6.227518e-5` against the native CPU/FP32 model. There were no forecast/native
input-root disagreements or own-state chain errors.

The intervention's behavior did change across inputs. Its signed RSI-reader
effect reversed direction, and restoring the native final-MLP return sometimes
increased collateral differences. The equations predicted those outcomes too.

This supports the supplied full-weight numerical account over a broader finite
panel. The equations and frozen weights were supplied before these experiments;
they were not inferred from the outcomes. Generic transferable route rules and
ISA-correct execution are separate questions.

The specimen is the three-layer R3 Mamba ISA model. Its inputs are a fifteen-byte
instruction window, eight 64-bit register values encoded as bits, and four flag
bits. Internal mixer state resets on each forward; only committed architectural
state and the model-selected byte offset carry between instructions.

## Six different inputs

The [fixed input plan](../qualification/input-generalization-plan-20261030.json)
contains six cases and three branches per case:

1. Native execution.
2. `U_swap`: zero the first MLP return and restore that case's independently
   predicted native final-mixer projection return.
3. `U_swap_P2native`: also restore that case's independently predicted native
   final-MLP return.

Each intervention applies only to the first call. Subsequent calls consume the
branch's actual model-predicted registers, flags and instruction-length offset.
The independent forecast separately consumes its own previous predictions.

| Input | Native behavior | Intervention observation |
|---|---|---|
| SUB twice, RSI0/RDI0 | RDI wraps to `2^64-1`, then `2^64-2` | Both edits preserve the hard state |
| SUB twice, RSI7/RDI255 | RDI254, then253 | Both edits preserve the hard state |
| SUB twice, high RSI/RDI bits and varied other registers/flags | Both decrements and flags match the oracle | Both edits preserve the hard state |
| ADD immediate15, then XOR; RDI65535/RSI17 | RDI65550, then65567 | Both edits preserve the hard state |
| ADD RAX,RBX with trailing XOR bytes | RAX13,120,839 and length7; oracle expects1030 and length3 | U edit changes RAX bits8,10; native-P2 restoration leaves bit8 different |
| SHL RDI,1 with trailing SHR bytes | RDI445 and collateral RBX corruption; oracle expects RDI522 with RBX unchanged | U edit changes RBX bit20; native-P2 restoration instead changes bits33,40 relative to native |

Four cases execute two model steps per branch. The final two six-byte programs
emit length7 and stop after their first call. They supply **no evidence about
executing their advertised second instructions**. The planned upper bound was
36 calls; actual execution retained 30, without overriding model lengths.

All ten unedited native steps received a post-hoc Unicorn check. Eight match all
registers, defined flags and length; the register-ADD and SHL steps do not. All
thirty rows, including the incorrect native responses and edited responses,
match their frozen equation decisions. Oracle results never supply forecast,
intervention, or continuation inputs.

## The same intervention recipe has different effects

The table reports the first-step RSI-bit1 logit difference from native. It does
not imply that RSI changed as an architectural integer; none of these six cases
flips an RSI hard bit.

| Input | U edit minus native | Also restore native P2, minus native |
|---|---:|---:|
| SUB, zero state | -0.335095 | -2.332851 |
| SUB, different low bits | -6.836333 | -4.039868 |
| SUB, high bits and varied state | -6.312562 | -4.730519 |
| ADD immediate15 | -1.349468 | -1.612705 |
| Register ADD | +1.155766 | -0.216675 |
| Shift window | +1.245349 | -1.107544 |

The U recipe has negative effects in four cases and positive effects in two.
Native-P2 restoration reduces RAX disagreement from two bits to one in the
register-ADD case. In the shift case, disagreement from native increases from
one RBX bit to two. Neither edited branch supplies an ISA-correct repair there.
Distance from native, consumer margin, and ISA correctness remain distinct.

The high-bit SUB case also varies other registers and flags. It broadens tested
state coverage but does not isolate high-bit causality or prove prefix invariance.

## Equivalent instructions expose applicability boundaries

Astra advised a small matched follow-up after the first panel. The
[second plan](../qualification/encoding-window-plan-20261030.json) crosses two
encodings of `ADD RAX,RBX` with trailing XOR bytes versus isolated, zero-padded
ADD. Initial registers and flags remain fixed: RAX1023, RBX7, with distinct
distractor registers. The oracle result is RAX1030 and length3 in every cell.

| ADD bytes | Trailing context | Native RAX | Native length |
|---|---|---:|---:|
| `4801d8` | `4831c8` | 13,120,839 | 7 |
| `4801d8` | Zero padding | 1023 | 7 |
| `4803c3` | `4831c8` | 1023 | 4 |
| `4803c3` | Zero padding | 1023 | 4 |

All twelve rows, including the three intervention branches in each cell, match
their sealed predictions. This is one-forward evidence only. The original
trailing-byte corner deliberately repeats the first panel; the other three
literal inputs are new within this campaign. Across both panels there are nine
distinct starting inputs, rather than 42 independent specimens. The repeated
corner reproduces roots, all eleven heads, all 39 ports and decoded state exactly
in all three branches across the two jobs.

The matched native contrasts establish encoding and trailing-context dependence
for this fixed state. They do not identify the internal decoder pathway. Each
edited cell uses its own equation-computed U/P2 donor, so comparisons across
cells test a common recipe, not portability of one fixed donor tensor.

## What the computation records should retain

Generic IDs remain appropriate: the source equations use
`r3.full-raw-state-transition`, with `native`, `U_swap`, and
`U_swap_P2native` routes. An operation name alone cannot predict these results.
Records need the literal window, architectural state, checkpoint, execution
profile, source identity, intervention recipe, consumer, signed effect,
collateral, and continuation horizon.

The useful working inference is that a fixed numerical transition law can
produce very different consumer effects as its receiving state changes. A
statement such as “U restoration always lowers RSI bit1” has counterexamples.
Restoring one native write also need not reconstruct the full native receiving
state. This panel measures that limitation; it does not yet isolate its minimal
causal feature.

The next small causal cut should hold the register-ADD state fixed and cross
decoder conditioning with direct byte-value inputs for the isolated/trailing
contrast. That follows the earlier
[state-transition path experiment](STATE-TRANSITION-DISCOVERIES.md), now on a
different operation window with a much larger hard-output difference.

## Execution and verification

Both jobs used the same frozen R3 checkpoint, CPU/FP32, one thread, one model
load per lease, strict mrun preflight, and no retries. Every panel's complete
forecasts and FP32 replacement payloads were globally sealed before its first
native call. There was no observed prefix. Equations were unchanged between
panels; the second panel was explicitly selected using the first panel's result.

| Job | Native calls | Typed Acts | Replay primitives | Worker time | Peak RSS |
|---|---:|---:|---:|---:|---:|
| `job-77fe6391408d` | 30 | 30 | 17,400 | 46.24s | 1350.9MiB |
| `job-64a4b3cf6bb1` | 12 | 20 | 6,960 | 13.12s | 1173.3MiB |

The independent [saved-array auditor](../qualification/analyze_sequence_generalization.py)
checks archive/request/source custody, seal/release ordering, all eleven heads,
39 named module input/return ports, payload/cast identities, typed receipts,
literal input packing, and each forecast/native chain's own previous state.
It runs no model, replay or source equations. Numerical differences remain
observations rather than scientific pass thresholds. Twenty focused installed
MDB mechanics tests pass outside the workspace.

The [worker](../qualification/raw_isa_sequence_campaign.py) now accepts per-case
programs and derives scheduling geometry from the plan. It retains the existing
legacy single-program path. The bounded archive transport permits an explicitly
requested larger panel while preserving its original64MiB default. The MDB
runtime numerical contract is unchanged.

Raw artifacts live under
`artifacts/mdb-input-generalization-20261030/{run-01,run-02}`. Each contains its
plan/request, shipped payload, job/log/archive, transaction, captures, equation
catalog, independent `analysis-v1.json`, and `isa-score-all-arms-v1.json`.
See the [portable index](../qualification/input-generalization-20261030.json)
for artifact hashes, individual rows and computation-atlas records.

Both model jobs succeeded. Their auxiliary mrun object-store finalization records
connection errors to the configured service. The complete marker archives were
collected locally and independently verified against their hashes; remote S3
replication is not confirmed. This does not change the retained measurements.

The separate atlas imports all 42 source-qualified equation records and retains
four caller-selected response-family candidates. All 42 exact-context queries
match uniquely; changing the source identity matches none. These are searchable
observations and candidate comparisons, with no automatic rule induction or
applicability expansion. The preceding 38-record atlas remains unchanged.

To re-audit a collected run without another model call:

```bash
python qualification/analyze_sequence_generalization.py RUN_DIR NEW_AUDIT.json
python qualification/score_sequence_generalization_isa.py RUN_DIR NEW_SCORE.json \
  --all-arms --custody-source qualification/summarize_context_window.py
```

The oracle command needs Unicorn; scoring is explicitly post-hoc. Output files
are exclusive writes. New model execution must continue through mrun admission.

The offline Saturn evidence sweep and verification both returned 0 over 42 derived
numeric rows, with no findings. Its `target-streams-only` scope checks derived
stream mechanics; it adds no scientific certification or MDB receipt authority.

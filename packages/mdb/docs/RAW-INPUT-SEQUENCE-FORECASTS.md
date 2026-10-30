# Predicting three ISA instructions from raw inputs

The independent R3 equations now start at instruction bytes, registers and flags.
They predicted three successive model calls without an observed internal prefix.
For two fixed starting states and three branches, all 18 instruction-step results
matched every head's sign or argmax decision. Each branch's predicted registers,
flags and instruction length supplied its next predicted input before native
execution began.

This extends the [earlier conditional suffix account](EXPLICIT-COMPUTATION-FORECASTS.md).
It uses full supplied equations and frozen weights for one checkpoint. The finite
test establishes raw-input trajectory forecasting here; broad ISA coverage and
automatic discovery of semantic algorithms remain further research.

## What was frozen and executed

The supplied byte program is:

```text
48 81 ef 01 00 00 00    SUB RDI, 1
48 31 f7                XOR RDI, RSI
48 81 c7 01 00 00 00    ADD RDI, 1
```

RSI starts at 3; RDI starts at 262 or 266. Other registers and flags start at zero.
The model receives a fifteen-byte window at its current predicted offset, so
following instruction bytes participate in the current decoder computation.
These exact states and program were fixed before numerical forecasts were made.
The two states share a low-bit prefix and are related examples.

Three branches apply edits only on the first instruction:

| Branch | First-step intervention |
|---|---|
| `native` | Unmodified model |
| `U_swap` | Zero early post.0; replace the final mixer update with FP32 predicted-native U |
| `U_swap_P2native` | Same, then install the common FP32 predicted-native final MLP write |

The U and P2 payloads came from the full raw-input equations' unedited first
trajectory. They were frozen before any native call. Subsequent instructions
have no intervention hooks. There is no externally supplied after-state or repair.

The reference explicitly evaluates the instruction decoder, soft register reads,
two immediate attention reads, aligned input slots, input projection, all three
Mamba mixer/MLP pairs and all eleven heads. It uses NumPy FP64 and a sequential
scan; native execution uses Torch FP32 and its parallel scan. All 87 checkpoint
arrays are retained. No model compression is involved.

## Results and scope

| Measurement | Observed result |
|---|---:|
| Starting states × branches × instruction steps | 2 × 3 × 3 = 18 |
| Complete register, flag, length and effect decisions matching forecast | 18/18 |
| All eleven head decision comparisons | Every row matched |
| Literal window/register/flag root joins | 18/18 |
| Largest continuous-head absolute error | 0.0000276002 |
| Largest error across compared actual module ports | 0.0000188351 |
| Module-return arrays compared per step | 34 |
| Native recordings and exact MDB replay comparisons | 18 |
| Typed Acts / replay operations | 10 / 10,440 |

All eleven heads are genuine downstream forecasts in this assay: there is no
native prefix and no already-computed destination/source head. The register bank
contains 512 bit decisions. Signed numerical residuals remain available alongside
hard decisions; small FP64/FP32 differences were not hidden behind an arbitrary
scientific threshold.

Every branch follows the same decoded trajectory within its starting case:

| Initial RDI | After SUB | After XOR | After ADD | RSI throughout |
|---:|---:|---:|---:|---:|
| 262 | 261 | 262 | 263 | 3 |
| 266 | 265 | 266 | 267 | 3 |

Flags remain zero and predicted instruction lengths are 7, 3 and 7. The native
validation uses MDB's existing `ISADriver` for fetching, decoding and committing;
the independent interpreter has its own NumPy packing and decoding. This tests
their agreement rather than reusing one helper on both sides.

These are six coupled trajectories, not eighteen independent computation
discoveries. The 18 native recordings are the measured outcomes; replay provides
the instrument audit rather than eighteen additional independent native runs.

## A predicted exception to the earlier collateral

The earlier repeated-SUB experiment found an RSI bit flip under update restoration.
It does not recur in this mixed-program context, exactly as the raw equations
predicted. The first-step RSI bit-1 logits are identical for the two new starting
states:

| Branch | Forecast logit | Native logit |
|---|---:|---:|
| Native | +10.275513 | +10.275504 |
| U restored | +3.431005 | +3.431013 |
| U and common P2 restored | +5.961758 | +5.961761 |

The interventions remain substantial: update restoration moves first-step
register logits by as much as 9.0066 and immediate-head logits by 25.6557. Every
decoded register and flag is nevertheless unchanged. All branches then commit
the same architectural state, so their later inputs and heads are exactly equal.
Each model call resets its hidden scan state. This assay therefore demonstrates
autonomous continuation forecasting without a propagated hard-state error.

Low-bit membership alone cannot be carried from the old repeated-SUB program to
this new window as an applicability rule. The next smallest causal comparison
holds RDI=262, RSI=3, flags and the first SUB instruction fixed, changes only the
trailing window to the old repeated-SUB bytes, and compares native with U_swap.
The current-window counterparts already exist. That cut would isolate the
effect of trailing-byte context rather than rerunning the full sequence campaign.

## How the new MDB features were used

`ForecastTransaction.run_script` published raw inputs, computed all six numerical
trajectories, sealed the complete forecast set, and only then admitted native
execution. The global seal event records zero native calls and zero prefix
captures. Predictions were never reconditioned on measured states. Observations
were persisted before the comparison callback ran.

Opt-in native ports supplied the actual decoder, input-projection, mixer and MLP
returns for 34 comparisons per step. The actual final delta projection returns
SSA `v000506` at PC404, after the bias add at PC403; its next direct consumer is
softplus at PC404. Its named bias is the read-only root `v000059`. Inferred affine
scopes remain separate. First-step restoration Acts use actual return-port IDs.

`compare_equations` generated numerical residuals, declared decision comparisons,
target/collateral effects and supplied additive-term closure reports.
`EquationCatalog` retained 18 immutable records with exact checkpoint, source,
execution and architecture applicability predicates. Tested contexts remain
separate from those structural conditions. Additive terms here are forecast
decompositions, not independently measured native component contributions.

The [workflow](COMPUTATION-FORECAST-WORKFLOW.md) documents the APIs. For this run:

```sh
mdb forecast verify /path/to/sequence-result/transaction \
  --expected-seal 986cc001ae5694a0266c354b1144f15e64712e88f929ecd8f1e48440bdb488ea
mdb ports /path/to/sequence-result/captures/rdi_262.native.0/trace.json \
  --select 'module:mixers.2.dt_proj:return' --first-consumers
mdb equations find --catalog /path/to/sequence-result/equations --reader regs:6:1
```

`verify_transaction` also accepts an independently pinned final event head;
the offline audit checked both that head and the forecast seal. A condition match
is a supplied selector, not a generalization result or a discovered semantic name.

## Execution cost and evidence

Job `job-62804158d4e4` ran one CPU thread on Beast within one finite mrun lease.
Worker time was 191.60 seconds; scheduler elapsed time was 201.65 seconds; peak
process-tree RSS was 1,181.1 MB. The phase ledger places 125.52 seconds between
raw-frontier publication and forecast persistence. That interval includes both
equation calculation and durable serialization. It does not isolate an I/O cost
or establish a speedup. Per-array/directory flushes and repeated tree decoding are
plausible costs to profile; packed immutable arrays and phase telemetry remain
the next performance work.

The earlier `job-20115331d4e2` stopped before any model forward because a
configuration check referenced newer executor fields absent from the captured
source. Its metadata and logs remain under run-01. The corrected run used the
same fixed inputs and equation account, with no intervening native outcomes.

The [frozen plan](../qualification/raw-isa-sequence-plan-20261030.json),
[full equations](../qualification/raw_isa_equations.py),
[campaign](../qualification/raw_isa_sequence_campaign.py),
[offline auditor](../qualification/summarize_raw_isa_sequence.py) and
[portable result index](../qualification/raw-isa-sequence-20261030.json) make the
result reviewable. Local raw evidence is under
`artifacts/mdb-raw-isa-sequence-20261030/run-02/` in the domains workspace.
The audit verified 1,423 artifact files, numerical arrays, source hashes,
actual-port bindings, catalog seals and independent successor joins. All 173 MDB
modules matched source, installed wheel and staged worker wheel. The public
upgrade passed 111 installed-wheel tests and its documented toy transaction.
Saturn evidence run/verify found zero findings on the 18 derived measurement
records, with `target-streams-only` coverage and no automatically ingested claims.
The direct transaction, source, array and typed-port audits establish the detailed
checks above; that generic sweep does not independently certify model semantics.

The raw forecast seal is
`986cc001ae5694a0266c354b1144f15e64712e88f929ecd8f1e48440bdb488ea`;
the final event head is
`20bfb677e471ed3a9421db12637c0c4274d27a0f663d2109599da561157c4f9e`;
the archive SHA256 is
`71780024152c3adc1f02cf5b2f2364d7e3c129a7ab09dd4010a3ae43cb9a3fc3`.

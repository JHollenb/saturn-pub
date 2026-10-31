# Following a computation into a different future

The R3 experiment now predicts a persistent architectural divergence. A first-step
intervention changes RSI from 3 to 1; later native calls consume that altered state
and take a different RDI trajectory. Every one of the 28 tested rows matches all
eleven head decisions frozen before execution. Maximum head error against the
FP64 equations is 3.30453e-5. These equations use the supplied frozen weights and
source; they were not learned from these outcomes.

The same panel localizes the selected context effect to decoder-conditioned
paths, exposes a new hard-output response to the final MLP, and supplies records
for the [cross-run atlas](COMPUTATION-ATLAS.md). It also predicts native model
errors on the new OR/AND byte windows. Faithful computation prediction and ISA
correctness are separate measurements here.

## An altered state survives into later inputs

The first segment is the unchanged fourteen-byte two-SUB program:

```text
4881ef010000004881ef01000000
```

Its first zero-padded fifteen-byte window exactly matches the previous error
case. Appending XOR directly would change that window and potentially remove the
starting contrast. Instead, after finishing the two-SUB segment, the host supplies
three separate one-instruction segments: `4831f7`, `4809f7`, and `4821f7`.
They encode XOR, OR, and AND of RDI with RSI. Registers and flags carry forward;
each segment starts at byte offset zero. This is an explicitly segmented input
stream. The two SUBs are the contiguous portion.

The starting state is RSI=3, RDI=262, other registers and flags zero. `U_swap`
zeros the first MLP and installs its independently predicted native final-mixer
write. `U_swap_P2native` additionally restores the predicted native final-MLP
write. These interventions occur only on the first call.

| After supplied instruction | Native: RSI / RDI | U restoration: RSI / RDI | Also restore P2: RSI / RDI |
|---|---|---|---|
| First SUB | 3 / 261 | 1 / 261 | 3 / 261 |
| Second SUB | 3 / 260 | 1 / 260 | 3 / 260 |
| XOR segment | 3 / 263 | 1 / 261 | 3 / 263 |
| OR segment | 3 / 3 | 1 / 1 | 3 / 3 |
| AND segment | 3 / 3 | 1 / 1 | 3 / 3 |

The complete three-branch futures were sealed before any native call. Forecast
steps consume their own predicted registers, flags and instruction lengths.
Native steps independently consume their actual prior outputs. All fifteen
native inputs join their sealed forecast inputs exactly; neither chain is
reconditioned from the other. Stores and syscalls would stop before commit.

The XOR step makes the persistent RSI difference visible in a different register.
The later OR/AND rows describe what the model actually emits; the oracle
comparison below prevents treating those rows as correct ISA execution.

This connects two clocks. Within a forward, causal mixers transport information
along bit positions and MLPs write updates into the receiving state. Between
forwards, architectural commits determine the next inputs. Hidden state resets
on each call in this model; there is no carried natural-language prompt or chain
of thought in this input contract.

## Which trailing-byte pathway changes the selected reader?

The current instruction and initial registers remain fixed. Only trailing bytes
differ between the old repeated-SUB and new mixed windows. The path panel holds
the early-MLP zero and **one old-context U payload** fixed, then crosses decoder
source with direct byte-value source.

| Decoder source | Direct byte-value source | Measured RSI bit1 logit |
|---|---|---:|
| Old | Old | -0.644844055 |
| New | Old | +0.372279882 |
| Old | New | -0.644844055 |
| New | New | +0.372276068 |

Decoder restoration occurs at `module:decoder:return`. The direct-value arm
changes only input-token columns `10:12` (two immediate readouts), `284:292`
(eight aligned slots), and `292:293` (selected slot). Donor-window readouts are
computed using the selected corner's decoder-conditioned attention and offset
weights. They do not import the other corner's attention conditioning.
The remaining 282 token columns retain their live recipient bytes exactly.
Typed Acts and independent native hooks agree on every output head.

Changing decoder conditioning repairs this reader in the tested fixed-U context.
Changing direct-window values alone leaves its FP32 logit unchanged. The joint
and decoder-only results differ by about 4e-6; this small residual remains in the
record. This is a reader-local contrast. It does not establish that direct byte
paths are irrelevant to other bit positions, pooled heads, contexts or U recipes.
The joint corner is a hybrid with old U; it is not the earlier new-context U case.

## Receiving state and the final MLP response

Two more branches transplant the joint corner's predicted final carrier M into
the old recipient. One recomputes P2; the other retains the old recipient's P2.

| Old-recipient branch | RSI bit1 logit |
|---|---:|
| Original old U branch | -0.644844055 |
| Other carrier, old P2 retained | +0.286885500 |
| Other carrier, P2 recomputed | +0.372270823 |

Changing the carrier while keeping old P2 already repairs the bit. Recomputing P2
adds approximately +0.085385 to that repaired margin. Thus this repair does not
require replacing the old P2 write. The carrier and its induced MLP response have
separately measured effects under the same register reader. Whole-vector
transplants identify causal contributions; they do not identify minimal features.

## Other MLPs and new operation windows

Two matched four-arm panels compare native execution with zeroing each of the
three MLP returns. Initial architectural state is identical across arms.

| Byte window | Native RDI | Zero post0 | Zero post1 | Zero post2 |
|---|---:|---:|---:|---:|
| AND: `4821f7` | 261 | 260 | 261 | 263 |
| OR: `4809f7` | 3 | 3 | 3 | 3 |

For both windows, removing post0 or post2 lowers the RSI-bit1 margin; removing
post1 raises it. Those signed response patterns recur while hard-state responses
differ. In particular, final-MLP zeroing changes the AND-window hard output.
A candidate generalization that late-MLP changes always preserve hard outputs
does not survive this example. These are total intervention effects, including
downstream recomputation, rather than isolated direct weight contributions.

A post-hoc Unicorn check executes one actual x86-64 instruction at each saved
before-state. At RDI262/RSI3, AND should produce RDI2 with length3; the native model
emits RDI261 with length7. OR should produce RDI263; it emits RDI3. No tested MLP
zero supplies an ISA-correct repair. The oracle ran after collection and never
supplied model inputs, payloads, equations or forecasts.

The new result is that the sealed account predicts these erroneous computations
too. A complete ISA-correctness or general computation-discovery claim remains
open.

## Evidence and execution boundaries

Job `job-09d78c2b5a0f` retained 28 native calls, 28 typed Acts and 16,240 replay
primitives, with 39 actual module-port arrays per row. Independent offline
verification checks archive/source/seal custody, each chain's own state joins,
literal input packing, all head decisions, stage residuals, port identities,
Act bytes and native/replay equality. Numerical discrepancies remain per row.
The qualification wheel, installed package, shipped runtime and working source
match across 175 Python modules. All 172 focused installed-wheel tests pass,
including nineteen atlas tests and seven independent-auditor mechanics tests.
An offline Saturn evidence sweep verifies 28 derived numerical-stream rows with
zero findings. Its `target-streams-only` scope checks stream mechanics; it is not
a second scientific certification of the model or MDB receipts.

The job's recorded status is **failed**: its optional catalog-publication callback
omitted required input/frontier binding fields after all observations had been
persisted. An offline publisher creates corrected comparisons and source-qualified
catalog records in a separate derived directory. It makes no model calls and
does not rewrite the failure or raw observations.

The first attempt, `job-24a17251e30a`, failed before a native call because its
recorder metadata key was unsupported. Its oversized failure archive could not
be collected through the bounded marker. The second worker verifies its retained
transaction on the same host after sealing the new forecast: zero native events
and all 1,727 retained forecast/payload NPY identities match. Original unused
equation expansions remain in that remote failed directory with file hashes;
their full bytes were not copied into the smaller collected archive. The second
attempt publishes all declared comparison ports rather than unused expansions.

| Custody item | SHA256 |
|---|---|
| Global forecast seal | `7338228da35e0374ca0ecd1f1bb2bafa1983cdabdc0278587222bd3467291feb` |
| Final failed-job event head | `135125350ee3f2de030a893b97c272eeb32b2c1a807192b1e29a9d63b9592cf1` |
| Collected archive | `e4c22ee3e02a6755e56646594cd1c4e5f15ac27b9ea6a4430070515fe93dd3ae` |

See the [campaign](../qualification/state_transition_campaign.py),
[equations](../qualification/state_transition_equations.py),
[offline publisher](../qualification/publish_state_transition_records.py),
[independent auditor](../qualification/analyze_state_transition_campaign.py),
[post-hoc oracle scorer](../qualification/score_state_transition_isa.py) and
[portable result index](../qualification/state-transition-20261030.json).

## Next discriminating cut

Keep the new decoder, old U and early-MLP zero fixed. Separately restore old
selected-register input columns `8:10` and the remaining changed instruction
conditioning, then restore both. This would distinguish changed register
selection from other decoder-conditioned features on the route to the carrier.
The atlas can retain these candidate families and their boundaries; automatic
rule induction and applicability expansion remain unfinished.

The subsequent [input-generalization campaign](INPUT-GENERALIZATION.md) first
tests other architectural states and instruction windows. Its 42 additional rows
preserve equation prediction while exposing sign reversals, collateral changes
and limits on native-write restoration.

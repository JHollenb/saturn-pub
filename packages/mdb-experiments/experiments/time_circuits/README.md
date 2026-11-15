# Run the paper's E6 and E20 with MDB

These finite assays use MDB's Model OS admission, durable tensor pages, immutable
input cuts, private candidate writes, receipts and replay around the paper's
unchanged native consumers. Each mrun job loads one model and runs its branches
in process. No Saturn or Saturn-pub checkout is imported or shipped.

E6 here means **Gemma circuit-tracer head-to-head** from
`research/papers/space-and-time-circuits/paper.md`. Another historical experiment
also has the E6 label: Qwen serial feedback in the architecture paper. This
directory does not execute that older experiment.

| Assay | Specimen and consumer | Main intervention | Observable |
|---|---|---|---|
| E20 | Qwen2.5-1.5B base, CUDA FP32, native eager suffix | Seed the subject residual entering L12; block/rescue its endogenous L22–27 subject K/V | Signed target logprob gain and fraction of the actual seed effect |
| E6 interchange | Gemma-2-2B, CUDA BF16, pinned circuit-tracer nnsight consumer | Set declared transcoder features to filler-prompt values, with native clean reconstruction errors | Signed answer logprob change, fraction toward filler and candidate flips |
| E6 ablation | Same Gemma/instrument | Zero graph-selected features, with matched random controls | Signed answer logprob change and candidate flips |

E20's full panel has 14 identities × 3 contexts = 42 rows and nine arms per
context. E6 uses the original admitted 12 identity + 8 color items. Interchange
runs both frozen and unfrozen attention in one job; ablation uses one attention
setting per job. Attribution batch size remains 64 and the graph budget 4096
feature nodes. Controls and counter-direction effects remain in the reports.

## Install and select the environment

Install MDB from this repository, then use an installed mrun with configured
scheduler credentials. See [installation](../../docs/INSTALL.md). In the domains
workspace use `common exec` for the shared environment. Set the intended
scheduler explicitly when multiple endpoints are available:

```sh
export MRUN_URL=http://192.168.68.68:9025
```

The controller must import this repository's installed MDB package. Workers
receive sealed MDB/mrun wheels and explicit experiment files. The source packet
records every shipped file hash, configuration bytes and request identity.

E20 resolves the registered `qwen2.5-1.5b` checkpoint through mrun. E6 uses the
existing host-local offline Gemma/transcoder assets and isolated instrument
environment on Beast. That environment needs Transformers 4.57.3, nnsight
0.8.0rc1 and circuit-tracer commit `7f66876689f59e92fc3641650d38b8bd41749ec4`.
The ordinary MDB Bench environment uses Transformers 5.14.1. They remain
separate environments; the sealed packages are extracted into the E6 lease's
private payload directory without modifying its installed instrument.

E6 enables `torch.use_deterministic_algorithms(True)` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. This is an explicit numerical lane: the
historical CUDA BF16 feature consumer had measurable repeat variability.
`--allow-nondeterminism --smoke` retains that two-item diagnostic lane and records native repeats;
it cannot certify bit-exact native equality. Do not compare its individual
floating outputs as though the two lanes were identical.

## Preflight, smoke and run

Start from `packages/mdb` in the Saturn repository. Use the Python interpreter in which MDB and mrun
are installed. In this workspace prefix commands with `rtk proxy` and use
`common exec` where shared imports/credentials are needed.

```sh
python -I experiments/time_circuits/submit.py preflight e20 --smoke --assay-id my-e20-smoke-01
python -I experiments/time_circuits/submit.py submit e20 --smoke --assay-id my-e20-smoke-01
python -I experiments/time_circuits/submit.py collect e20 --job JOB_ID

python -I experiments/time_circuits/submit.py submit e20
python -I experiments/time_circuits/submit.py collect e20 --job JOB_ID

python -I experiments/time_circuits/submit.py submit e6 --smoke
python -I experiments/time_circuits/submit.py collect e6 --job JOB_ID
python -I experiments/time_circuits/submit.py submit e6
python -I experiments/time_circuits/submit.py collect e6 --job JOB_ID
```

Collect only after that exact job is terminal. Run CUDA assays sequentially;
there are no automatic retries or child leases. Admission uses measured native
profiles plus input-cut headroom: 12 GB RAM and 12 GB VRAM. The deterministic
E6 smoke supplies the full-panel timing estimate. A larger panel expected to
exceed 15 minutes requires independent configuration review before submission.
The scheduling contract counts fixed prompt/context panels as local work items;
the request separately records inner consumer calls and cut replays. Full E6
interchange has 20 panels, 1,400 MDB consumer calls and 40 cut replays.
Use a new assay identity for every independent reproduction. Omission generates
a fresh UUID. Supplying the same `--assay-id` binds preflight and submission to
the same request; reusing a completed identity does not authorize overwriting
its durable result root. This instance field does not change the model seed,
prompt panel or intervention protocol.

To reproduce the original zero-ablation panels:

```sh
python -I experiments/time_circuits/submit.py submit e6 --e6-mode ablation --smoke
python -I experiments/time_circuits/submit.py submit e6 --e6-mode ablation
python -I experiments/time_circuits/submit.py submit e6 --e6-mode ablation --unfreeze-attention
```

Inspect and collect each job before starting the next full assay. E6 is an
external instrument assay; `mdb open --model gemma...` is not an equivalent
qualified interactive model adapter.

## Retrieve, audit, take custody and sweep

Reports and logs collect under `results/time-circuits/JOB_ID`. Large state
archives and model weights stay out of Git. The result root is the first 16
characters of the request digest, under `/mnt/hdd8tb/mdb/time-circuits`.

```sh
python -I experiments/time_circuits/retrieve.py results/time-circuits/JOB_ID
python -I experiments/time_circuits/verify.py \
  results/time-circuits/JOB_ID/state/REQUEST_DIGEST_PREFIX \
  --source results/time-circuits/JOB_ID/source \
  --out results/time-circuits/JOB_ID/state-audit.json
python -I experiments/time_circuits/finish.py results/time-circuits/JOB_ID
```

Retrieval waits for archive completion, verifies the downloaded SHA-256 against
the closed host copy, and rejects tar traversal and links. The offline audit
verifies publication journals, cut/parent/receipt identities, native ABI,
referenced tensor bytes, graph references, wheel contents and configuration.
It uses no model and does not claim to replay a native consumer.
This archive/source audit also passed in the dependency-free MDB environment,
with neither Torch nor mrun installed.

`finish.py` checks the bounded native/replay mechanics, takes custody of the
report, environment, request, source/state archives and raw rows, then runs and
verifies MDB's WATCH → CLAIMS → XREF sweep. Its mechanics receipt declares no
scientific effect threshold. Exit 3 is evidence triage; exit 2 is broken or
incomplete input. Byte audit, numerical replay and scientific interpretation are
separate authorities.

## What MDB owns and where the cuts stop

[`NativeResearchSession`](../../src/mdb/research_session.py) is a bounded adapter
for a caller-owned native consumer. It requires the complete input closure,
declared writable addresses, typed validation and model/numerical identity.
The consumer returns private native output plus a JSON observation. Model OS
admits the writes; private hydration prevents parent/sibling tensor aliasing.
Publication uses the existing durable journal and preserves the previous
selection if publication fails. Aborted candidates retain evidence.

For E20, a cut holds completed subject-prefix K/V for every layer, unchanged
suffix IDs and actual positions. The residual seed is still performed by the
frozen paper prefix code; MDB captures its resulting recipient-native K/V.
Rescue transplants that formed band into the unseeded recipient. Block replaces
it with recipient K/V; reinstall restores the seed's formed band. Wrong-source
and earlier-band controls remain explicit. This does not qualify Qwen
virtual-context BF16 K/V edits or convert FP32 cuts into another lane.

For E6, a cut holds prompt, an `[N,4]` FP64 table of integer
`(layer, position, feature, value)` edits, and an attention-freeze flag. Typed
validation rejects fractional or out-of-range addresses before the native
instrument runs. Frozen graph references retain token IDs, selected feature
coordinates, influence and instrument identity. They are structural references,
with semantic bindings unassigned. The instrument's internal model tensors and
graph computation are not arbitrary resumable layer cuts.

These are finite native-input research sessions. They do not add Gemma to the
interactive BenchSession families, a new live debugger broker, full external
instrument checkpointing, or a universal reasoning-circuit detector. Existing
BenchSession rewind/replay, page ownership, source frames, typed edits and
symbol tools remain available in their separately qualified family lanes.

## Interpret the measurements

E20 measures how much a particular source edit reaches an answer through a
particular formed memory band. Its ratios are fractions of target **logprob
gain**, not percentages of model reasoning or explained variance. Values may
exceed one or run negative; preserve them. The 42 rows contain 14 identities,
so identity-clustered uncertainty is distinct from the paper worker's original
item bootstrap.

E6 measures what the chosen transcoder basis and its native intervention can
control. Graph influence is a localization diagnostic; answer changes are the
native consumer observation. The historical K/V comparator uses a separate
42-identity/25-color panel and is not a new paired 20-item K/V run here.

The later reviewer experiment that swaps both features and reconstruction error
substantially changes the **color** interpretation: all-position joint swap
reaches 0.990 of the filler effect, versus 0.359 for feature-only. Identity's
joint swap reaches 0.314. Those are historical reviewer observations, not new
MDB measurements. A feature-only gap cannot establish that the information is
absent from the complete instrument or quantify the model's reasoning.

See [findings](FINDINGS.md) and [source provenance](PROVENANCE.json) for the
completed records and exact scope.

# mdb examples

**Only experiments approved by the owner are added here.**

Numbered directories (`01-...`, `02-...`) are experiment examples: one measured experiment, a
notebook that walks a newcomer through it, an equivalent script for mrun, a launcher, and the
numbers the original run measured (`expected_outputs.md`, with job ids). The loose scripts below
are older primitive demos.

Runnable examples over caller-owned `BenchSession` state and saved scalar programs.
Examples load no model on import; real-model examples run inside an admitted mrun worker.

| Example | What it shows |
| --- | --- |
| `resident_lab.py` | Rewind/replay, residual preview/commit/abort, sibling memory, circuit panel (tiny random CPU model). |
| `pass_indexed_baseline.py` | The pass-indexed-baseline R0 drift recipe as BenchSession primitives. |
| `collect_session.py` | Collecting an exported session. |
| [`01-sae-census-qwen3-1.7b/`](01-sae-census-qwen3-1.7b/) | First SAELens x mdb experiment: SAE variable census on Qwen3-1.7B-Base with Qwen-Scope SAEs (features vs error term, error splice, steering). Notebook, script, Beast launcher. |
| [`02-tl-reparametrisation-floor/`](02-tl-reparametrisation-floor/) | 2026-11-14, owner-approved (showcase exp6). TransformerLens processed/unprocessed bridges as a second realisation of the same forward, judged against mdb's floor: an effect-size table (above ~4 floors 0/214 verdicts change). `names.tl_hook`, selectors, Branches, supplied-forecast judging. Qwen3-0.6B-Base and Qwen2.5-0.5B. |
| [`03-composer-neurons-sealed-forecasts/`](03-composer-neurons-sealed-forecasts/) | 2026-11-14, owner-approved (showcase exp5). nnsight gradients nominate composer neurons, mdb seals the forecasts and measures the knock-outs natively on held-out items (26/26, large effect). Pools (leave-one-out mean), f-port selectors, Branches nulls, custody. |
| [`04-watch-edit-continue-country-feature/`](04-watch-edit-continue-country-feature/) | 2026-11-14, owner-approved (showcase exp1). The debugger loop: FeatureCondition watch on the carrier's SAE country feature (false alarms shown), break, Branches with a one-component repair, a pool donor and a matched-norm null, continue. |
| [`05-sae-component-binding-carrier/`](05-sae-component-binding-carrier/) | 2026-11-14, owner-approved (showcase exp2). SAE features as component bindings of the carrier value cell: the named country component moves 0.59 of the cell, a random one 0.0. `Binding.component`, symbol table, Branches over layers. |
| [`06-interchange-subspace-pyvene/`](06-interchange-subspace-pyvene/) | 2026-11-14, owner-approved (showcase exp7). Interchange at the carrier cell: pyvene and mdb agree on flip verdicts; `SubspaceBinding` for the training-free SAE subspace (an honest negative) and readout spans, with the audited DAS comparison. |
| [`07-frozen-gradient-brackets-transcoders/`](07-frozen-gradient-brackets-transcoders/) | 2026-11-14, owner-approved (showcase exp4; presented on above-floor nodes). Transcoder nodes, nnsight frozen-gradient edge, mdb holds (pattern, norm scales) as Branches arms; the edge explains 0.92-0.97 of native, direct-edge overstatement ~1.6x. |
| [`read_weight_transport_demo.py`](../../mdb-experiments/examples/read_weight_transport_demo.py) | Framework-free six-head toy algebra for normalized read-weight transport; [worked docs and measured Qwen result](../../mdb-experiments/docs/READ-WEIGHT-TRANSPORT.md). |
| [`read_weight_taylor_demo.py`](../../mdb-experiments/examples/read_weight_taylor_demo.py) | Evaluate a measured, sealed second-order six-head program without loading a model; [math and scope](../../mdb-experiments/docs/READ-WEIGHT-TAYLOR.md). |
| [`read_weight_centered_demo.py`](../../mdb-experiments/examples/read_weight_centered_demo.py) | Replay measured midpoint coefficients and their zero residual without a framework; [native-profile state/KV closure and measured costs](../../mdb-experiments/docs/READ-WEIGHT-CLOSURE.md). |

## `pass_indexed_baseline.py`

Expresses the six-line recipe from
`research/papers/pass-indexed-baseline/paper.md` section 7 in mdb's own
primitives — `prefill` (the scene prefix becomes the pass-0/fixed parent cut),
`stage_input`+`step` (teacher-force the exact question tokens), `step` (commit
the model's own greedy answer), `checkpoint` (the per-turn cut), and
`fork`/`restore`/`checkout` (a retained-cache fork that shares the immutable
prefix ancestor). For each of the four preregistered turns it scores the turn's
answer token under (a) the pass-indexed native continuation of the retained
trunk and (b) a fork of the fixed pass-0 prefix, and reports

    Drift(t) = logp_rebaselined(answer_t) - logp_fixed(answer_t)

plus the fact-span K/V bit-identity canary (the prefix keys/values read on the
trunk and on the fixed fork, which must be bit-identical).

**Run it:**

```sh
# Mechanics only — tiny random CPU Qwen2, no download, no tokenizer. Validates
# that the primitives compose and the fact-span canary is exact. (drift numbers
# are meaningless on random weights.)
python pass_indexed_baseline.py --self-test --root ./selftest

# Real model, inside an admitted mrun worker (BF16 CUDA):
python pass_indexed_baseline.py --model-path /mnt/big/llm-models/Qwen2.5-1.5B \
    --root /mnt/usb-storage/mdb/pass-indexed --scenes S0        # smoke, one scene
python pass_indexed_baseline.py --model-path /mnt/big/llm-models/Qwen2.5-1.5B \
    --root /mnt/usb-storage/mdb/pass-indexed                    # all four scenes
```

Numeric regime: mdb's Qwen lane (CausalContextVM) runs **BF16**; it refuses
FP32. The paper's primary R0 numbers are FP32. This example reproduces the
*direction and magnitude* of the base R0 drift in BF16, not a bit match.

### Measured reproduction (MEASURED)

Run on Beast through mrun (Qwen2.5-1.5B base, BF16 CUDA eager, greedy,
`gen_budget=3`), tunnel `MRUN_URL=http://127.0.0.1:19025`, priority 100.

| per-pass **median** Drift (nats) | pass 1 | pass 2 | pass 3 | fact-span K/V canary |
| --- | ---: | ---: | ---: | --- |
| base FP32, 4 scenes (paper §3; `job-35d4adaeed36`) | +3.125 | +2.202 | +2.063 | max-abs 0.0 |
| **mdb BF16 example, 4 scenes** (`job-30443c340ead`) | **+3.327** | **+2.083** | **+2.217** | **max-abs 0.0 (all 4 scenes)** |
| mdb BF16 example, smoke scene S0 (`job-221789dc049b`) | +3.075 | +1.777 | +2.856 | max-abs 0.0 |

All three per-pass medians are positive and within ~0.1–0.2 nats of the FP32
base, and the fact-span K/V bit-identity canary is exact (max-abs 0.0) on every
scene. One individual scene×pass cell differs in sign under BF16 (scene S2, the
recap turn T3: −0.95), a BF16-regime difference on the one referential turn; the
per-pass medians remain positive. Full per-scene drift and the canary are in the
job report's `scenes` array (measured peak VRAM 3582 MB; wall ~520 s for four
scenes). These are BF16 numbers, a different numeric regime than the paper's FP32
primary and the same regime as the paper's 7B replication and BF16 VM
cross-check.

## Experiment examples

### How to run

Each experiment directory holds `<name>.ipynb` (the walkthrough), `<name>.py` (the same cells as a
plain script; the notebook is generated from it, cell for cell), `run.sh` (the payload entry point),
`run_on_beast.sh` (submit through mrun) and `expected_outputs.md`.

- Beast through mrun (the way the numbers were measured): open the tunnel
  (`ssh -f -N -L 127.0.0.1:19025:127.0.0.1:9025 beast`, `MRUN_URL=http://127.0.0.1:19025`), check
  the GPU is free (`~/domains/common/bin/common mrun jobs --state running`), then
  `bash run_on_beast.sh`. It stages the payload (the example, `packages/mdb/src`, `src`) and
  submits one job at priority 70. Results land in the job log (`RESULT_JSON ...`) and
  `$MRUN_ARTIFACTS_DIR`.
- Local GPU: install `mdb-runtime[autoregressive]` (torch, transformers 5.x), `sae-lens` and
  `matplotlib`, set `MDB_MODEL` and `MDB_SAE` to local checkpoints, and either open the notebook or
  run `python <name>.py`.

### Hardware

One CUDA GPU with about 8 GB free and about 12 GB of host RAM. The first example measured a 3.7 GB
VRAM peak (bf16 1.7B model, eager attention) and runs in under 10 minutes. Examples 02-07 each
validated in one Beast job under 10 minutes; peaks are in each `expected_outputs.md` (the largest,
03, measured 6.9 GB VRAM; 07 about 9.6 GB host RAM). The models are read
from local disk only (`local_files_only`); nothing is downloaded.

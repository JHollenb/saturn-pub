# mdb examples

Runnable examples over a caller-owned `BenchSession`. Each one loads no model on
import; the real-model examples run inside an admitted mrun worker.

| Example | What it shows |
| --- | --- |
| `resident_lab.py` | Rewind/replay, residual preview/commit/abort, sibling memory, circuit panel (tiny random CPU model). |
| `pass_indexed_baseline.py` | The pass-indexed-baseline R0 drift recipe as BenchSession primitives. |
| `collect_session.py` | Collecting an exported session. |

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

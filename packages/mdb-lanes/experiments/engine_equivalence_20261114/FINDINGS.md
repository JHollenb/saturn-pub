# Qwen read panel: lane path vs engine path, FP32 — findings (2026-11-14 11:03 PST)

- Prereg: `PREREG.md` (committed before each submission). One job scored: **job-0fcdf8414918**
  (mrun, Beast, RTX 4080, torch 2.13.0+cu130, transformers 5.14.1, Qwen2.5-1.5B FP32 eager,
  weight sha256 a961db72…; worker 136 s, mrun elapsed 152 s; peak VRAM 5,940 MB allocated by torch,
  6,350 MB observed by mrun; reservation 9,500 MB). An earlier attempt, job-48b15af5a630, failed in
  the worker before loading the model (`import mdb.engine.trace as ...` binds the function, not the
  module); fixed and resubmitted.
- Engine integrated: `origin/mdb-engine-refactor` bcdd070 (merged in fbb9a54). Module hashes are in
  `results/job-0fcdf8414918.json` (`modules`).
- Raw result: `results/job-0fcdf8414918.json` (the worker's `QWENENG_RESULT` line). Offline analysis:
  `analyze.py` → `results/analysis.json`.
- Labels: MEASURED = from those files; ASSERTED = my reading.

## Worst problem first

**The engine's zero cut is not bit-equal to either lane arithmetic on CUDA (MEASURED).** The prereg
predicted, from the tiny CPU test, that the engine cut equals the lane's `recompute` cut bit for bit.
On the real model it does in 4 of 29 arms; engine vs the lane's default `delta` cut is equal in 9 of
29. The largest margin difference is 5.7e-6 (about 1–4 units in the last place of the logits); the
lane's own twin gap |delta − recompute| on the same arms is up to the same size (largest 3.8e-6 on
`rank12`, 128 margin spacings because that margin is small). So the engine path is a third
arithmetic of the same cut, at the size of the lane's existing floor term, not a replacement that
reproduces the lane's bits. Not yet known: which kernel difference causes it (the engine edits `p`
and lets eager attention run `p @ v` over all heads; the lane's recompute runs the touched head
alone in FP32).

## Results (MEASURED)

| comparison (FP32, bit for bit) | result |
|---|---|
| lane native first-position logits vs engine stepped run `t21.logits` (prefill=19) | **equal** (sha256 061d0345… both) |
| lane native vs the same forward under `mdb.trace` / under the lane observer | equal / equal (trace passive) |
| lane FP32 recompute `Read.probs` vs `mdb.trace` native `L*.p` vs run `t21.L*.p` (28×12×22) | **all equal** |
| 24 single-layer zero cuts (`floor_cut_recompute` arms; 12 distinct cuts), engine `whatif_many` vs lane recompute | 4/24 equal, rest differ by ≤ 5.7e-6 |
| 5 multi-layer sets by engine staged runs vs lane recompute | 3/5 equal (`keep-named3`, `rand0/1-named3`); `named3` and `rank12` differ by 1.9e-6 and 3.8e-6 |
| engine vs lane `delta` (the plan's default) over the same 29 arms | 9/29 equal, ≤ 5.7e-6 |
| smoke lane margins vs the saved lease rows of jobs b33e6b98c328 / 4a433df3f30b / 7c3232ad65bb | 59/59 equal (the lane is deterministic across jobs) |
| Qwen FP32 read verdicts with the engine margins substituted for the 30 arms it measured | counts unchanged: 0 supported / 4 set_only / 13 not_supported / 4 control_missing / 3 unresolved; 0 verdicts changed |
| engine P20 verdict of each what-if | `effect` for all 24, residual native − emulated 0 spacings |
| exact contribution of the read to the margin (`analysis.reads`, float64; removing it predicts minus this, first order) | L20H5 −1.522 (mass 0.927), L21H0 −0.668 (0.673), L20H11 −0.076 (0.634); measured margin shifts of the zero cuts (native −4.647): +1.846, +0.210, +0.005 |
| cost | engine run 6.1 s; 24 what-ifs + 5 staged runs 71.8 s; contributions 0.5 s; peak VRAM +50 MB over the lane |

## What this means for the lane (ASSERTED)

- Attention weights and base logits can come from the engine (`mdb.trace` or the stepped run)
  with no change in bits for this case.
- Single-source cuts through the engine (`whatif_many`) and multi-layer sets (staged runs) measure
  the same effects to within the lane's existing twin-gap floor, and leave every verdict unchanged
  here, but they are not the lane's bits. A plan sealed with the lane's `delta` or `recompute`
  arithmetic should keep that arithmetic; a new plan can declare the engine path as its execution.
- The renormalising cut, keep arms built as plans of the lane, BF16, and FLUX stay on the lane path.
- The exact contribution, read as the first-order prediction of a cut (minus the contribution),
  has the sign and the order of the three measured shifts, but not their size: +1.522 vs +1.846,
  +0.668 vs +0.210, +0.076 vs +0.005.

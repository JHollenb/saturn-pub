# PREREG: re-check of single-donor and zero-ablation nulls in the mdb record

Written 2026-11-16T16:20-0800, before any job of this experiment ran. Sealed guesses: `GUESSES.txt`, sha256 `99029fc161ca80a13430f39c39ce9aae030d388e5a1816633be8f952e1fd87d6`.

Why: the Qwen back-test of the binary lessons (job-33fe7bcce697, `../2026-11-16-binary-ground-truth/LESSONS.md`) overturned 157/313 canonical single-donor residual nulls on Qwen3-1.7B with 16 donors. This re-runs the "no effect" claims in tracker entries 1–12 that carry a headline and rest on ≤ 2 donors or a zero/mean ablation. The full inventory, with the skipped claims and why, is in `RECHECK.md`.

Code: saturn-pub branch `null-recheck` (worktree `.claude/worktrees/null-recheck`), `packages/mdb-experiments/experiments/null-recheck/` (`q3_recheck.py`, `q25_recheck.py`, `common.py`, `run.sh`, `smoke.py`). mdb is main b94fb8e, unchanged. A laptop smoke on tiny random Qwen3/Qwen2 models checked the plumbing only (batched arm = stepped `set_cell` arm, to the bit, on 4/4 checks).

## Claims re-run

| id | tracker source | model | original intervention, donors | re-run |
|---|---|---|---|---|
| C1 | entry 1 l.43 "keys do not" (K-only single cell 0/123 flips, \|fraction\| 0.013) | Qwen3-1.7B-Base bf16 | single different-bridge donor per pair (1–2 per recipient, 123 pairs) | K cell at the carrier (gt row, L20 KV6), 16 donors |
| C2 | entry 1 l.43 K-only bands from L0..L16 0/123 | same | same | K over all span rows, layers s..27, all KV, s ∈ {0, 8, 16, 20} |
| C3 | entry 1 l.43 "bands from L24 0/123, so the read finishes between L20 and L23" | same | same | V band, span rows, L24..27 (L20..27 as positive control) |
| C4 | entry 1 Run 1 P2 + audit, entry 10 (1.7B) "short L17–L21 write window on the same row" | same | single donor cell, live = \|dM\| > 0.5 | every V cell (28 layers × 8 KV) at the gt row, 16 donors |
| C5 | entry 1 Run 1 P1 "67% of cells move M < 0.07, exactly 0" (descriptive) | same | single donor | every span V cell, 16 donors |
| C6 | entry 9 T3 "replacing t0 with its position mean costs nothing" (20/20 kept, median \|dM\| 0) | same | mean ablation | row-0 h at the L14 input from 16 prompts with a different first token |
| C7 | entry 5 J2 "L22 KV-head-1 key cell −0.02" (entry 4 Eiffel K −0.06) | Qwen2.5-1.5B bf16 | single donor per pair, 27 pairs | same cell, 16 donors |
| C8 | entry 5 J2 "L22 KV-head-0 value cell −0.00" | same | same | same cell, 16 donors |
| C9 | entry 5 J2 "L22 h7 zeroed −0.07" | same | zero ablation | head-7 slice of A at the answer row resampled from 16 donors |
| C10 | entry 5 J2 "L23 f4197 zeroed −0.04: the neuron was specific to the Eiffel prompt" | same | zero ablation | f[4197] at the answer row resampled from 16 donors |
| C11 | entry 5 "city info leaves its row between L22 and L24 (residual patch −7.3 at L22 → −0.06 at L26)" | same | single donor (Rome), one prompt | h at the L26 input, city row, 16 donors on the J2 panel (L22, L24 also reported) |

## Setup

- **Jobs.** Two mrun jobs on Beast, priority 70, ≤ 600 s, one model load each, resident model paths `/mnt/big/llm-models/Qwen3-1.7B-Base` and `/mnt/big/llm-models/Qwen2.5-1.5B`, bf16 eager, as the originals.
- **Prompts.** C1–C5: the 64 entry-1 panel recipients (`job1-panel.json`, `candidates.json`, carrier rows from `gt.json`). C6: the 20 top-1% rows of TP1 (all row 0) and the 96 TP1 prompts. C7–C11: the 27 J2 pairs (job-ec87a6e1c6f2), rebuilt from the same template and checked against the saved pair list.
- **Donors.** C1–C5: 16 items of the recipient's family with a different answer (currency 24 / person 40 / landmark 33 / city 52 candidates): the panel's canonical donors first, then same-length items, then the rest, seeded (`20261116:<id>`). Rows aligned from the end of the descriptor span. A donor key from another position is RoPE-rotated to the recipient's row (fp32, one bf16 rounding); same-length donors need no rotation and are reported separately. Values need no rotation. C6: 16 prompts whose first token differs from the recipient's (row 0 at L14 depends only on the first token). C7–C11: the J2 donor first, then the next 15 kept cities in J2's order (every donor a different country).
- **Instrument.** C1–C5, C7–C10: mdb `analysis.sweep.cell_sweep` (the batched native sweep job 2 used): cell edits and answer-row port patches read by the answer row only (J2's default reader scope), one null arm per chunk, dM against that chunk's null. C6, C11: batched full forwards with a layer pre-hook, dM against lane 0. A few arms are re-measured stepped (`Run.set_cell`, and mdb `donors.whatif` on the f4197 arm).
- **Consumer.** C1–C5, C7–C11: M = logit(answer) − logit(donor's answer), per donor, at the last row. C6: M = logit(t1) − logit(t2) of TP1.
- **Floor.** 2 bf16 spacings at the larger \|logit\| of the pair, plus the measured spread of the null across the item's batches (C6/C11: plus \|batched null − batch-1 native\|). Effect: \|dM\| > floor. Magnitudes are reported in floor units, with the count ≥ 4 floors (the exp6 audit rule for calling an effect real).
- **Verdicts.** mdb `donors.informative` (bitwise) and `donors.pooled` (effect when ≥ 1 donor has one). Unit: the recipient item (C1–C3, C6–C11); for C4 the item with any live (\|dM\| > 0.5) gt-row cell outside L17–L21; for C5 the (item, row, layer, KV) cell.
- **Rule.** A claim FLIPS if more than 20% of its units are pooled `effect`, else it SURVIVES. A flip whose median effect is < 4 floors is reported as "flips, within the kernel-sensitive band". The originals' own metric (answer flips to the donor) is reported beside it.

## Stated limits

- Single seed, by rule. Same models, dtypes and prompts as the originals, except C11 (the J2 panel, not the Eiffel prompt) and the donor rows of C1–C3 (end-aligned, RoPE-rotated keys across lengths).
- The floor is a bf16 kernel floor, not a behavioural threshold.
- Claims not re-run are listed in `RECHECK.md` with the reason; "not re-run" is "not yet known", not "survives".

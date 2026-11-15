# Expected outputs: TL weight processing as a reparametrisation null

Source experiment: an earlier, larger run of this experiment (48 items per model) with a sealed
preregistration (`prereg.json`, copied here and declared as custody). This example follows its
corrected reading: the result is an effect-size table, not "verdicts don't hold across kernel
paths".

Models: Qwen3-0.6B-Base (carrier L20 KV6, query heads 12, 13) and Qwen2.5-0.5B (carrier L21 KV0,
query heads 0-6); bf16, eager attention, one CUDA GPU; TransformerLens 4.4.0 `TransformerBridge`.
Single seed. Floor: 2 bf16 spacings at the largest `|logit|` of (answer, foil) over the item's
native arms (the prereg's rule). This is mdb's margin noise floor (`verdict.margin_floor`, 2
spacings), deliberately wider than the 1-spacing verdict floor that `verify` and `whatif` use,
because these effects are compared across different realisations (batched vs stepped, TL vs mdb),
not a prediction against its own native run. "Lost" = sign reversed, or at or below the floor, on the other path.

## Original run (48 items per model)

| Model | What |
|---|---|
| Qwen3-0.6B-Base | 48 items, 290 s, peak VRAM 11980 MB |
| Qwen2.5-0.5B | 48 items, 147 s, peak VRAM 10356 MB |

An earlier combined smoke run with all four bridges on the GPU was killed at 15.3 GB VRAM, so the
fp32 bridges run on the CPU.

Loss by `|effect| / floor` bin, total (mdb stepped vs TL processed batched) / kernel only (mdb
stepped vs TL unprocessed batched), from the original run:

| \|effect\| / floor | Qwen3 n | Qwen3 lost (total / kernel) | Qwen2.5 n | Qwen2.5 lost (total / kernel) |
|---|---|---|---|---|
| 1-1.5 | 89 | 50 / 51 | 41 | 32 / 14 |
| 1.5-2 | 32 | 9 / 8 | 30 | 16 / 2 |
| 2-4 | 43 | 1 / 1 | 53 | 18 / 0 |
| > 4 | 110 | 0 / 0 | 104 | 0 / 0 |

Headline: above about 4 floors, 0/214 effects lose their verdict on any path (110 + 104). Losses
sit at 1-2 floors. Sign reversals: 1/274 (Qwen3), 28/228 (Qwen2.5, all from processing). The kernel
path dominates on Qwen3 (60/274 kernel vs 46/257 processing); processing dominates on Qwen2.5
(82/249 vs 16/228). The 2-spacing floor is a sound noise bound (p95 of processed - unprocessed 2 /
3.5 spacings), but a verdict needs about 4 floors. fp32 pure-reparametrisation check: max
`|P32 - U32|` base margin 2.3e-5 (Qwen3) / 1.8e-5 (Qwen2.5), 48/48 within 1e-4.

Bins are `low < |effect|/floor <= high` (mdb's `stability.calibrate`); with this convention the
function reproduces the table above exactly from the original run's raw per-item output.

## This example

Subset: **24 of the 48 base items per model**, every second item sorted by id (6 per family:
city, currency, landmark, person); the same 24 for both models; `items.json` (cut by
`make_reference.py`, with the donors). `reference.json` holds the original run's raw margins for those
items; the script scores both with the same function.

What replaced the original driver's hand-rolled parts:

| Original driver | Here |
|---|---|
| `GH` hand-built query-head list; `crow_of` re-implementing the carrier rule | `cm.Slot(..., layers=(L_c,), kv=(g,), rule_heads=cm.group_heads(adapter, g))` + `cm.slot_rows` (new 8-line accessor `group_heads`) |
| checkpoint / zero head / until logits / restore loop | `dbg.branches(run, Margin(last, a, b), at="t<last>.L<L_c>.h")`, one arm per head, `br.run()` |
| 4-line head mask per arm | selector edit `edit("t@last.L20.A[h3]", 0.0)` (`Run.set` selector, `breakpoints.plan`) |
| hand-built KV-group mask and donor tensor, origin lost | `donor.ref("cache.L20.v[kv6,row=r]")` value + pointer, edited at `t<crow>.L20.v[kv6]` with `origin=` the pointer, in a second `Branches` sharing the floor |
| TL column indices by hand | `names.to_tl` + `names.tl_hook` (mdb layout, rank check) + the same `breakpoints.plan` for the columns |
| `floor_of` / `score.py` bins | `verdict.magnitude` / `verdict.spacing` for the floor, `stability.calibrate` for the loss tables, the `stability` band (k = 4) on every arm |
| judged record ids only | `supplied.seal` before the native arm, `supplied.judge` after; every record through `dbg.records_to` with `dbg.custody(script, prereg)` |

Kept as plain Python: the item loop and the kernel-path deltas. No `Pool`: the donor is fixed per
item by the panel.

### Validation run (measured)

One job on one CUDA GPU (RTX 4080) through our job scheduler: succeeded, 261.8 s wall, peak VRAM
11936 MB (torch peak allocated 11483 MB Qwen3 / 10791 MB Qwen2.5), peak RSS 16.3 GB. Validation
log: `validation-job-285ed57bae28.log` in this directory. **24/24 checks pass.** Smoke run: 2 items
per model, 37 s, peak VRAM 11930 MB, 24/24.

Original run vs now, same 24 items: **every margin is bitwise equal** (mdb native base, 2H head arms
and V arm; TL U16, P16, U16 batch 1; fp32 within 0, max diff 0), carrier rows and donor carrier rows
equal, so every table is identical:

| path | 1-1.5 | 1.5-2 | 2-4 | > 4 | all | k |
|---|---|---|---|---|---|---|
| Qwen3 total | 33/48 | 6/16 | 1/25 | **0/57** | 40/146 | 2.12 |
| Qwen3 kernel | 34/48 | 4/16 | 1/25 | **0/57** | 39/146 | 2.12 |
| Qwen3 batch (b1 vs batched) | 34/50 | 10/27 | 0/20 | 0/55 | 44/152 | 1.75 |
| Qwen3 processing | 22/32 | 2/14 | 0/23 | 0/56 | 24/125 | 1.75 |
| Qwen2.5 total | 9/12 | 10/13 | 9/27 | **0/55** | 28/107 | 3.75 |
| Qwen2.5 kernel | 3/12 | 1/13 | 0/27 | **0/55** | 4/107 | 1.53 |
| Qwen2.5 batch | 9/26 | 0/12 | 0/25 | 0/56 | 9/119 | 1.50 |
| Qwen2.5 processing | 21/27 | 6/8 | 10/31 | 0/54 | 37/120 | 3.50 |

(lost / n per bin; `k` = `calibrate`'s smallest margin in floors above which nothing was lost.)

- Above 4 floors: 0/57 (Qwen3) and 0/55 (Qwen2.5) effects lose their verdict on any path: 0/112
  on this subset (the original run's 0/214 on all 48).
- Sign reversals (total path): 0/146 Qwen3, 11/107 Qwen2.5; kernel path 0 on both. Processing
  reversals on Qwen2.5: 17/120.
- Kernel path dominates on Qwen3 (39 lost vs 24 by processing); processing on Qwen2.5 (37 vs 4).
- fp32: max `|P32 - U32|` = 2.29e-5 (Qwen3) / 1.62e-5 (Qwen2.5) on these items (same as the
  original run's on these items; the 48-item maxima are 2.3e-5 / 1.8e-5).
- Self-patch null (`add_null("self")`): `dM` exactly 0 on every item.
- Sealed TL claims: 146 / 107 judged native effects (= the above-floor counts), 57 / 55 `stable`.
  `sign_agrees` is False on 2 / 13: the 0 / 11 reversals plus 2 / 2 effects where the processed
  claim is exactly 0 and the native effect positive (`judge` counts `supplied > 0` vs
  `native > 0`; checked on the reference data).
- Records: 1488 `supplied-forecast` + 96 `branches`, all read back and verified, carrying the
  script and prereg sha256.

Differences from the original run, and why they do not move a number:

- **V arm `dM`.** `Branches` measures the V interchange against the base of its own run (split at
  the carrier row); the original run subtracted the unsplit run's base. The V margin itself is bitwise
  equal; the two bases differ by at most 0.25 (Qwen3) / 0.125 (Qwen2.5) logits (the split's kernel
  path). The V effects on these items range 0.4-18.8 logits (2.5-75 floors); on this subset the
  base shift moved no V effect across a bin edge or the floor: every table equals the original run's.
  On other items it could, by at most that base delta.
- **Floor rule.** `Branches`' own floor is 2 spacings at the *base* margin's `|logit|`; the original run
  (and these tables) use the largest `|logit|` over the item's arms, read from the arm rows' new
  `magnitude` field. The two rules differ on 1/24 (Qwen3) and 2/24 (Qwen2.5) items.
- The null "unprocessed bridge bitwise equal to mdb" was not met in the original run (13/48, 7/48)
  and is not a check here; TL batch 1 is a full-sequence forward, mdb's is stepped.
- `--vram`: measured peak 11936 MB x 1.3 = 15.5 GB exceeds what the scheduler admits (its
  ceiling is ask x 1.1 <= ~15.1 GB), so the ask is 13500 MB (x 1.13).
- Not rerun: the attention-pattern check and the per-head pattern lists.

## mdb source changes made for this example

- `analysis/circuit_map.py`: `group_heads(adapter, kv)`, the query heads of a KV group (the grouping
  `group_writers` sums over), for a per-model `Slot(rule_heads=...)`. `Slot` itself already took
  layers and kv per model; only the head list had no accessor.
- `analysis/branches.py`: every arm row of a `Margin`/`Logit` consumer records `magnitude` (the
  consumer's `|logit|` size on that arm, `verdict.magnitude`), so a floor over an item's arms comes
  from the rows.
- Tests: `test_analysis_circuit_map.py::test_group_heads_is_the_group_writers_grouping`,
  `test_analysis_branches.py::test_arm_magnitude_and_head_selector` (also checks the `A[h1] = 0`
  selector equals the hand-built head mask). Both files: 24 passed.

# Expected outputs: watch, break, edit, continue on a country feature (Qwen3-1.7B-Base)

Original experiment: an earlier, larger run on all 110 d3/d4 items (two shards of 55), with a
sealed preregistration (`prereg.json`) and a frozen country-feature census (`census.json`). Model
Qwen3-1.7B-Base, bf16, eager attention, one CUDA GPU; SAE Qwen-Scope
`SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`, layer 19 (k = 50), through mdb's `FeatureBasis`. One seed.

## Original numbers, full panel (110 items: 27 wrong, 83 correct)

| Number | Value |
|---|---|
| watch fires on wrong items | 18/27 |
| watch fires on correct items (false alarms) | 30/83 |
| `content(C)` names the true country, correct items | 78/83 |
| `content(C)` names the true country, wrong items | 5/27 |
| R1 (one named SAE component swapped) repairs | 12/27 (9/18 where the watch fired) |
| RND (random feature swap, norm-matched to R1) repairs | 2/26 |
| R2 (whole carrier cell from a same-country donor) repairs | 22/27 |
| median R1 margin move / R2 margin move (wrong items) | 0.387 (n = 26) |

The original run also sealed a margin forecast after the R1 edit and verified it at the logits
(26/26 confirmed). That checks mdb's float64 replay of layers 21-27 on an
edited state. It is not evidence that R1 repairs anything. This example does not run it.
The preregistered watch criteria (fires on 20 or more wrong items and 10 or fewer correct items) failed. The
example shows the false alarms on purpose.

## What this example runs

A subset of 39 items (`items.json`, `subset_rule`): all 27 wrong items, the first 6 correct items
(by id) on which the watch fired (false alarms: Anitkabir, Red Square, Retiro Park,
Shibuya Crossing, Tivoli Gardens, the Belvedere palace), and the first 6 correct items (by id) on
which it stayed quiet (Big Ben, Buckingham Palace, Gyeongbokgung Palace, Notre-Dame Cathedral,
Saint Basil's Cathedral, Schonbrunn Palace). `reference.json` holds the original run's per-item
outputs for these items, and the same scoring on the subset.

The R2 donor pool holds every correct d1-d4 item of the subset's 19 countries (117 cells). The
donor rule is the original run's code exactly. Its `order` dict literal lets the later `"d3-lm": 1`
override the own family's 0. So "same family first" (the prereg's words) never applied to these
d3/d4 items. What ran was "any d3/d4 item, then d2, then d1, by id", and that is what this
example reproduces. The random-feature null (RND) replays the original run's recorded draws, stored in
`items.json`, because the original `random.Random(20261114)` stream depends on shard order.
The error-component swap and the entity-copy check are not run.

## Validation run of this example (measured)

One job on one CUDA GPU (RTX 4080) through our job scheduler (bf16 eager, 39 items): succeeded in
291 s wall. The scheduler recorded a peak VRAM of 3842 MB and a peak RSS of 2775 MB; torch reported a peak
allocation of 3388 MB. Collecting the 117-cell pool took 122 s, and each item took 2.5-7 s.
Validation log: `validation-job-2d3c8107e3b9.log` in this directory. Smoke run (2 items, Anitkabir
and Franz Kafka): both items matched the original run on every field. The scheduler marked that job
failed only because the trailing `grep` in `run.sh` returned 1 (`SCRIPT_RC 0`). `run.sh` now exits with
the script's own code.

| Number (same 39 items) | Original run | This example |
|---|---|---|
| watch fires, wrong | 18/27 | 18/27 |
| watch fires, correct (false alarms) | 6/12 | 6/12 |
| `content(C)` true country, wrong | 5/27 | 5/27 |
| `content(C)` true country, correct | 11/12 | 11/12 |
| R1 repairs, wrong | 12/27 | 12/27 |
| R1 repairs where the watch fired | 9/18 | 9/18 |
| R2 repairs, wrong | 22/27 | 22/27 |
| RND (feature-swap null) repairs, wrong | 2/26 | 2/26 |
| R1 changes the argmax on a correct item | 1/12 | 1/12 |
| median R1 / R2 margin move | 0.387 | 0.387 |
| rnd (Branches matched-norm Gaussian null) repairs, wrong | not run | 1/26 |

Per item, all 39 items are identical to the original run on every compared field: carrier row, foil,
native logits, floor, watch hit, `best_feature` label, `content(C)`, wrong feature, R1 change
norm, donor id, and the R1/R2/RND `dM` and argmax. The kernel path is the same as the
original run's: stepped from the span, cells set at the `t<last>.L20.h` checkpoint and continued.
The new APIs change no number. The wrong items' numbers are the full-panel numbers, because all
27 wrong items are in the subset.

The one Gaussian-null "repair" is d4-work:the novel Palace Walk. There the native margin sits
inside the floor: every arm moves it by 1 floor or less, and R1 also "repairs" it at 0.5 floors.
On Tivoli Gardens (correct) the Gaussian null flips the argmax away at 0.5 floors.

Records: the run writes 40 sealed records to `$MRUN_ARTIFACTS_DIR/records.jsonl`: one `pool`
record (`48626fce1bb2...`) and one `branches` record per item. All carry custody (the script
sha256 and the `prereg.json` sha256 `0adac2733d6e...`). Each R2 edit's origin names the pool, its
record id, the rule `donor (same:country)` and the donor id.

## mdb source change made for this example

`analysis/pools.py`: `collect(..., prefill=callable)` collects each cell on the driver's own
prefill split, so that the span rows are stepped. With it, the pool's `kernel_path` is
`mdb-stepped`, provided every item's cell row is at or after its split. Without it, a cache-cell
pool reads the batched prefill, which is not the original run's kernel path. Test:
`tests/test_analysis_pools.py::test_collect_with_prefill_split_is_stepped` (tiny qwen3 fp32 and
qwen2 bf16); the file passes 10/10.

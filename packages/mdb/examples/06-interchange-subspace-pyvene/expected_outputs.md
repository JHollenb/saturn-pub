# Expected outputs: interchange at the carrier cell (pyvene x mdb, whole cell and subspaces)

Showcase experiment 7 (`results/showcase-20261114/exp7/`), with context from the DAS study (tracker
entry 11b, `results/showcase-20261114/das/`). Model Qwen3-1.7B-Base, bf16, eager attention, CUDA on
Beast, single seed. SAE: Qwen-Scope `SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`, layer 19. pyvene 0.1.8 with
the Qwen3 shim `mdb.interop.pyvene_compat`. Framing: `results/showcase-20261114/AUDIT-SHOWCASE.md`,
`das/AUDIT-DAS.md`, `das/followup/AUDIT-FOLLOWUP.md`.

## Jobs

| Job | What |
|---|---|
| job-c8144767543d | showcase exp7, all 64 pairs (whole cell mdb + pyvene, SAE pushforward, random) |
| job-8b01f1a12d84, job-33a4b67a28eb, job-60bce10afa0c, job-70ebd29d8f10 | DAS follow-up: readout spans `full69` and `a69` (the k=32 arms re-measured here) |
| job-a4989538826a | this example, smoke, 3 pairs (script OK; the job was marked failed by a `pipefail` exit in `run.sh`, since fixed) |
| job-4e15b86999c5 | this example, smoke, 3 pairs, with the diagnostic arms |
| **job-1c4d004eebc0** | **this example, validation, 34 pairs** (`validation-job-1c4d004eebc0.log`) |

## The showcase result (64 pairs, MEASURED, job-c8144767543d, audited)

- Whole-cell IIA: mdb 44/64, pyvene 43/64. The two agree on 63/64 flip verdicts; the one
  disagreement is `currency:rupiah` (mdb -0.75, pyvene -0.5, floor 0.5).
- pyvene minus mdb whole-cell margin: median 0.125, max 0.75. The no-intervention kernel-path gap
  (pyvene's batched forward minus mdb's stepped native) is median 0.125, max 0.375. Same median,
  heavier tail. They are not equal.
- SAE pushforward subspace (median rank 4, max 9): flips 4/64 and moves a median 0.34 of the
  whole-cell margin change. A random subspace of the same rank flips 0/64 and moves 0.01. This is
  an honest negative for the SAE subspace as a handle on the variable.
- I1 (whole-cell IIA 0.55-0.65) came from another panel. Do not read anything into it passing or
  failing. I2 fails and its falsifier fires (SAE/whole = 0.09).

DAS context (audited; not re-run, no DAS training in this example): at k=32, held-out DAS gets
16/64. The country readout span gets 35/64 with the test countries in the token set, 20-22/64
with no test country, and 22/64 with only the pair's own two countries removed. The SAE pushforward
gets 4/64 and random 0/64. At k<=4 the per-pair SAE set is the better handle (4 flips and 0.34 moved,
against 0 and 0.19 for the readout span). The readout span's lead is mostly per-pair identity.

## Subset used by this example

The first 34 of the 64 pairs in id order (`MDB_NPAIRS=34`; `make_subset.py` cuts `items.json` and
`reference.json`): 16 city, 16 currency, and 2 landmark pairs (Borobudur, Masada). The subset holds
the one pyvene/mdb disagreement (`currency:rupiah`) and all 4 SAE flips (Gdansk, yen, Borobudur,
Masada). The showcase reference on these 34 pairs is computed from the saved raw outputs
(`exp7/out/exp7-items.jsonl`, `das/followup/out/score-full*.jsonl`).

## Validation run (job-1c4d004eebc0, MEASURED)

Beast, priority 70, 241 s wall (script 240 s, about 5.4 s per pair), peak VRAM 3888 MB, peak RSS
3.4 GB. The log prints each pair's numbers next to the showcase's, then this comparison:

| number (34 pairs) | showcase | now |
|---|---|---|
| flips @1 floor: whole (mdb) | 27 | 27 |
| flips @1 floor: pyvene | 26 | 26 |
| flips @1 floor: SAE pushforward | 4 | 4 |
| flips @1 floor: random | 0 | 0 |
| flips @1 floor: readout span full69, k=32 | 22 | 22 |
| flips @1 floor: readout span a69, k=32 | 13 | 13 |
| flips @4 floors: whole / pyvene / SAE / random / full69 | 24 / 24 / 1 / 0 / 22 | 24 / 24 / 1 / 0 / 22 |
| flips @4 floors: a69 | 8 | **9** (explained below) |
| pyvene/mdb verdict agreement | 33/34 (rupiah) | 33/34 (rupiah) |
| pyvene - mdb margin gap, median / max | 0.125 / 0.75 | 0.125 / 0.75 |
| no-intervention kernel gap, median / max | 0.0 / 0.25 | 0.0 / 0.25 |
| SAE rank, median / max | 4 / 9 | 4 / 9 |
| median fraction moved: SAE / random | 0.355 / 0.010 | 0.354 / 0.018 |
| self-patch null, max abs dM | (not run) | 0.0 |

On this subset the kernel gap is median 0.0 and max 0.25. Over all 64 pairs it is 0.125 and 0.375.
The audited framing ("same median, heavier tail") is a 64-pair statement.

Per pair, the following equal the showcase exactly on 34/34 pairs: M0, carrier row, donor carrier
row, SAE rank and features, whole-cell M, pyvene M, pyvene clean M, and the branched base M0. The
pool's donor cells are bitwise equal to the `Run.ref` reads on all 31 donors. The remaining arms
differ by at most a few bf16 steps:

- SAE equal on 29/34 (max diff 0.125), full69 on 27/34 (0.125), a69 on 27/34 (0.125), random on
  8/34 (0.625).

**Why they differ.** The showcase built `v + Q Q^T (v_donor - v)` with `v` taken from the
recipient's native run (prefill = start of the bridge span). `subspace.interchange` and this
example use the `v` of the branched run itself (prefill = carrier row). That way, the part of the
cell outside the subspace keeps that run's own value. The two `v` differ on 18/34 pairs (max
|dv| 0.81): the rows between the span start and the carrier row are stepped in one run and batched
in the other. To confirm, the example adds two diagnostic arms that swap against the native run's
`v` (the showcase convention). They equal the showcase exactly: `sae_vnative` = showcase SAE on
34/34 and `a69_vnative` = showcase a69 on 34/34. So the differences come from the base-cell
convention, not from the model or the new APIs. The one verdict that moves is an a69 pair near the
4-floor threshold (8 -> 9). The random arm also uses different draws: one seed per pair
(`20261114 + index`) instead of one stream over all 64 pairs. Its verdicts are unchanged (0 flips).

Symbol table (`city:Gdansk`, carrier row 38, consumer `M[answer, donor answer]`, a pair the SAE
subspace flips): the native cell's direct consumer contribution is +0.718. The SAE subspace (k=4)
holds 0.236 of the cell's energy and 0.105 of that direct share. Random (k=4) holds 0.002 and 0.022.
The full69 readout span (k=32) holds 0.558 and 0.977. The a69 readout span holds 0.514 and 0.703.
Each binding carries its pair's interchange and the panel IIA as attached labels.

## What the example does not reproduce

- The other 30 pairs (set `MDB_NPAIRS=64` and a larger `BUDGET`: about 5.4 s per pair, not
  validated within the 10-minute budget).
- DAS training, the k sweeps, the lto arm and the rank-matched SAE sweep (the DAS study; quoted in
  the notebook's section 7).
- `subspace.from_pyvene` is not used: the pyvene arm here is a `VanillaIntervention` (a whole-cell
  copy), not a trained rotation.

# FLUX read panels: sealed attention panels on the FLUX.2 Klein lane

This lane runs an `mdb.attention_panels` plan on FLUX.2 Klein 4B through `mdb.attention_port`. It replaces the one-off worker `saturn/experiments/2026-11-11-flux-read-graph/read_graph_worker.py` for read-cut experiments. It has three parts:

- `src/mdb/flux_read_panels.py`: framework-free logic. It checks plans and leases, builds and verifies regeneration recipes, maps worker rows to panel results, and computes the headfree "gone" score.
- `qualification/flux_read_panel_worker.py`: the mrun worker. It has two modes, panel and `--regenerate`.
- `qualification/flux_read_panel_stage.py`: offline commands to build plans, stage, submit and report.

Unit tests are in `tests/test_flux_read_panel.py`. Sealed plans, tables and verdicts are in `qualification/flux_read_panels_20261112/`.

## End to end

Use the mdb interpreter, with `PYTHONPATH` pointing at the checkout under test. The payload wheel is built from whichever `mdb` is imported, and `stage` prints `mdb_source`.

```bash
PY=/Users/jakeholl/domains/saturn-pub/packages/mdb/.venv/bin/python
export PYTHONPATH=<worktree>/packages/mdb/src
cd packages/mdb/qualification
$PY flux_read_panel_stage.py plans flux_read_panels_20261112/plans          # 1 build + 2 seal + leases
$PY flux_read_panel_stage.py stage RUN --plans .../plans --job smoke         # 3 stage (also: legacy, full:N)
$PY flux_read_panel_stage.py submit RUN                                      # 4 submit (mrun, priority 70)
rtk proxy rsync -a beast:/mnt/hdd8tb/mdb/research/<job>/ <results>/<job>/    # 5 collect
$PY flux_read_panel_stage.py report --plans .../plans --results <results>/job-* --out OUT   # 6 verdicts
$PY flux_read_panel_stage.py stage RUN --plans .../plans --job regenerate \
    --recipes <recipes> --arms smoke-legacy@4242:z:own60:ball full-00@4242:m:own60:red   # 7 regenerate, then submit
```

1. **Build.** `plans` calls `read_panel` once per word and merges the results with `merge_panels`. One call per word is needed because `RandomControls.mass` is a single vector shared by every item; see Workarounds. It then adds the prompt-level arms with `prompt_level_arms`. It writes four plans:
   - `smoke-legacy` / `smoke-new`: the word `ball`, seed 4242;
   - `legacy`: the headfree arm list exactly as executed, 224 branches;
   - `full`: all six words and both seeds, every control, 2,568 branches.
2. **Seal.** Each plan is written in canonical JSON, so the file's sha256 is the plan hash, with `<name>.seal.json` beside it. `leases_within` splits the plan into leases. Each lease's arms plus its reference roots must fit in the scheduler's 256 work items (`scheduling_contract.MAX_LOCAL_WORK_ITEMS`). This gives a lease cap of 251 for `full`, which has 11 leases.
3. **Stage.** Each job carries one sealed plan file and lease per pass in the payload. The smoke job has two passes, `legacy` and `native` precision. The run config holds the plan's sha256 and the lease record. `source_hashes` declares the `mdb/attention_port.py`, `mdb/attention_panels.py` and `mdb/flux_read_panels.py` bytes, and the bootstrap checks them against the installed wheel.
4. **Run.** The worker does the following, in order:
   - It refuses any plan whose sha256 differs from the run config's, and any lease whose seal, plan or branches do not match (`load_plan`, `check_lease`).
   - It prints `FLUXPANEL_MODULES`: the file path and sha256 of each shared module as imported, and the payload seal's entry for it. It raises if they differ.
   - It loads Klein once and encodes every prompt the lease needs: the native prompt and each word-deleted prompt. Then it releases the text encoder.
   - For every (context = seed) it runs the reference roots without the port: native, `del` (word deleted from the prompt) and `iso`. `iso` is the read-graph `hybrid(..., "own")` conditioning: the word is present only in its own rows, the rows after it come from the deleted prompt's encoding shifted by the deleted token count, and the padding rows come from the deleted prompt.
   - It runs each lease branch through `with port.apply(*port_cuts(arm), forwards=4)` and checks `port.calls == 100` and `port.forward_count == 4`. Arms that cut nothing (`iso:`, `del:`, `native:`) also run through the port, and each is checked to be bit-identical to its root (`equals_root`).
   - There are no retries.
5. **Collect.** Results go to `beast:/mnt/hdd8tb/mdb/research/<job>/flux-read-panel/`.
6. **Report.** `report` merges the per-lease `results.<tag>.px_mad.json` files with `merge_results`, which refuses other plans and duplicate outcomes. It then runs `attach_results`, `arm_effects` and `verdicts`, and writes the legacy identity table, the gone tables, the regeneration table and the verdict records.
7. **Regenerate.** `--arms TAG@CONTEXT:ARM_ID` picks a recipe by pass tag, context and arm. The pass tag is needed because the same arm id exists in a legacy pass and a native pass. The chosen recipes are copied into a new payload, and the worker runs `--regenerate CONTEXT:ARM_ID ... --recipes recipes`.

## What is stored per arm

| object | stored | where |
|---|---|---|
| decoded image | yes (PNG) | `images/<pass>/<context>/<arm>.png` |
| recipe (sealed JSON, ~10-22 KB) | yes | `recipes/<pass>/<context>/<arm>.json` |
| scalars | yes | `metrics.<pass>.json` |
| panel results (px MAD, latent rel vs the arm's reference) | yes | `results.<pass>.<metric>.json` |
| final latent | **no** (hash only), unless `retain=["latents"]` | `latents/...npy` |
| attention captures | **no** (this worker runs no observers) | |
| weights | never | checkpoint records and parameter pointers only |

Each `metrics.<pass>.json` row holds the following scalars:

- distances (`px_mad`, `px_max`, `latent_rel`, `rgb_equal`, `latent_equal`) to the arm's reference arm, and to the native, `del` and `iso` roots;
- `vs_floor_of`, the direct delta/recompute twin gap, on `/cut_recompute` floors;
- the latent sha256 (float32 bytes), the decoded-RGB sha256 and the PNG sha256;
- `port_calls`, `forward_count` and the recompute counters from `port.log`;
- seconds and peak allocated VRAM;
- for legacy passes, the pixel comparison with the saved 2026-11-11 reference PNG.

Per job the worker also writes:

- `checkpoint.json`: every checkpoint file with its byte count and safetensors header sha256, plus the full file sha256 when `hash_checkpoint` is set (smoke and regenerate jobs);
- `parameters.json`: one pointer per attention tensor of the transformer, using the `parameter_artifacts` source fields `file`, `tensor`, `dtype`, `shape` and `data_offsets`. `payload_sha256` is not computed, so these are pointers, not restorable artifacts;
- `roots.json`, `tokens.json`, `modules.json`, `summary.json`, `timeline.json`, and `files.json` (the sha256 of every output file).

### The recipe

A recipe has the fields `schema`, `arm`, `model`, `software`, `input`, `noise`, `port` and `expected`, plus a `fingerprint`, which is the sha256 of the canonical JSON. `validate_recipe` refuses an edited or incomplete recipe.

- **Model**:
  - the adapter's `model_identity`, which hashes the content of the transformer and VAE;
  - the model path and pipeline class;
  - the sha256 of `checkpoint.json` and `parameters.json`;
  - the attention blocks the cut touches.
- **Input**:
  - the prompt text;
  - how the conditioning is built: `native`, `deletion` or `hybrid_own`, with the word, the deleted prompt and the row plan;
  - the tokenizer settings.
- **Noise**: the seed, the CUDA generator, the height, width and steps, and the schedule rule.
- **Port**:
  - the backend, `precision`, `zero_method`, `empty` and `forwards`;
  - the cut specs (`Cut.to_dict()`: receivers, sources, heads, blocks, steps, `renormalize`, `sham`, per-cut `zero_method`);
  - the row table, as runs, for every row name the cuts use.
- **Software**:
  - the Python, torch, diffusers, transformers, numpy, Pillow, saturn_pub, CUDA and cuDNN versions;
  - the GPU and host;
  - a fingerprint of the adapter's execution ABI;
  - the sha256 of the worker and of the three shared modules.
- **Expected**: the conditioning, initial noise, final latent, decoded RGB and PNG hashes.

### What `--regenerate` does and does not guarantee

`--regenerate` rebuilds each named arm from the root alone, meaning the checkpoint, prompt text and seed. It re-encodes the prompt, rebuilds the conditioning from the recipe's construction, draws the noise from the seed, and rebuilds the port from the recipe's precision, zero method, row table and cut specs (`cut_from_dict`). It then reports, hash by hash, whether the conditioning, initial noise, final latent, decoded RGB and PNG equal the recipe's. It also reports any software-lane field that differs (`lane_drift`). It refuses to run a recipe whose model identity differs from the loaded model.

It **does** guarantee that a match means the same bytes were produced from the root on this host and this software lane. That lane is the CUDA device, torch/diffusers/transformers versions, saturn_pub wheel, TF32 setting and shared-module bytes.

It **does not** guarantee:

- the same bytes on another GPU, driver or library version: BF16 kernels are not bit-stable across lanes, and a mismatch there is reported, not repaired;
- that the text encoder weights are unchanged. They are not part of `model_identity`; the conditioning hash catches a change in its output;
- the full checkpoint-file hash, except in jobs run with `hash_checkpoint`.

This is the FLUX counterpart of the Qwen hydration path in `native_replay_recipe`, which has its own Qwen-only profile and was not modified.

## Reproduction of the 2026-11-11 headfree run (job-a0bca63168b4)

Klein 4B, 256 px, 4 steps, BF16, TF32 off, prompts `cat` and `car`, seeds 4242 and 9001.

Every number below is MEASURED from the saved job outputs unless it says otherwise. Per-job wall time and VRAM are in `qualification/flux_read_panels_20261112/jobs.json`. The full text report is `report/report.txt`, written by the `report` command. Images and recipes are on Beast under `/mnt/hdd8tb/mdb/research/<job>/flux-read-panel/` and on the laptop under `artifacts/mdb-flux-read-panels-20261112/results/`. They are not committed.

The sealed plans are in `qualification/flux_read_panels_20261112/plans/`:

| plan | sha256 | arms | branches | contents |
|---|---|---|---|---|
| `smoke-legacy` | `ce7dc5cf...` | 20 | 20 | `ball`, seed 4242, the old arms |
| `smoke-new` | `7e14e78c...` | 218 | 218 | `ball`, seed 4242, the full panel |
| `legacy` | `b52a26a9...` | 112 | 224 | the headfree arm list as executed |
| `full` | `94ef887e...` | 1,284 | 2,568 | 156 test, 336 control, 762 floor and 30 reference arms; 11 leases of ≤251 branches |

The full panel holds, per word:

- every old arm;
- both cut types for every set;
- the floor arms of every cut (`/sham_recompute` and `/cut_recompute` for zero cuts, `/sham` for renormalising cuts);
- a matched random set for cut and keep (`rand0-*`, disjoint, stratum, block and mass matched with tolerance 0.02);
- the other-item control (`@rows-<next word>`);
- `own240-minus-own60`;
- prompt-level `native:<pid>` and `img<-alltext` in both types, with floors.

All 78 random draws were emitted, and `panel.identity` records whether each one separates identity from mass. A test checks that the `legacy` plan equals the 224 executed (seed, arm) pairs. Another checks that every one of those arms appears in `full` with the same kind, cut type, rows, units and condition.

### Jobs

Klein 4B on Beast (RTX 4080). Priority 70, 10,400 MB VRAM request (7,991 MB peak × 1.3; the smoke used 12,000 MB).

| job | what | mrun wall s | worker s | peak alloc MB | branches |
|---|---|---|---|---|---|
| job-a34e4d6fe9dc | smoke (`ball`, 4242): legacy pass + new pass | 620 | 540 | 7,991 | 20/20 + 140/218 (budget stop) |
| job-2b516ba78da6 | legacy, all 6 words, both seeds | 250 | 211 | 7,991 | 224/224 + 28 roots |
| job-68368184de20 | full lease 0 | 292 | 261 | 7,991 | 234/234 |
| job-9b9ec68fc42a | full lease 1 | 354 | 295 | 7,991 | 232/232 |
| job-c4b924f3295d | full lease 2 | 429 | 360 | 7,991 | 247/247 |
| job-ea0bcbfe960f | full lease 3 | 269 | 237 | 7,991 | 242/242 |
| job-17f61da18312 | full lease 4 | 400 | 382 | 7,991 | 242/242 |
| job-baa6927a1a6c | full lease 5 | 333 | 296 | 7,991 | 239/239 |
| job-de42c09696dd | full lease 6 | 253 | 239 | 7,991 | 242/242 |
| job-d8790bc10fbb | full lease 7 | (see below) | 955 | 7,991 | 0/249 (budget stop) |
| job-a41e454eefe1, job-2c64f0609f8f, job-0ef55a497c49 | full leases 8-10 | did not run | | | |
| job-af5bfaf0fde2 | regenerate 3 arms | did not run | | | |

**Leases 7-10 and the regeneration job did not complete.** The Beast mrun host agent was stopped by systemd at 16:40:11 PST, in the same second it leased job-d8790bc10fbb (`journalctl --user -u mrun-agent`: `Stopping mrun-agent.service`). It was still inactive when this was written. That job's worker spent 866 s importing from the HDD-backed uv cache, against 28.6 s in job-2b516ba78da6, so its 540 s budget stopped it before the first branch. It ran past the 900 s mrun timeout because no agent was supervising it. The remaining three leases and the regeneration job are staged and queued. They need the agent back and full lease 7 resubmitted.

Timing:

- Mean time per branch was 0.40 s on the legacy lane and 0.46 s on the new-precision lanes. Keep arms took about 0.6 s.
- The first run in each job took 10-29 s of CUDA warm-up.
- Imports took 28-866 s and dominate the variance in job wall time.

### (a) Legacy precision against the saved reference PNGs

| compared | pixel-identical | not identical |
|---|---|---|
| legacy arms, all 6 words × 2 seeds plus prompt-level (job-2b516ba78da6) | 224 / 224 | none |
| smoke legacy arms (job-a34e4d6fe9dc) | 20 / 20 | none |
| roots: native, del and iso per (word, seed) | 31 / 31 | none |

The comparison was done twice, both times against the saved images:

- in the job, against `beast:/mnt/hdd8tb/mdb/research/job-a0bca63168b4/read-graph-result/`;
- on the laptop, against `saturn/experiments/2026-11-11-flux-read-graph/runs/headfree/result/`, giving 244 of 244 identical.

`native:<pid>` equals both the old `native.<pid>.<seed>.png` and the old `arm.<pid>.<seed>.sham.png`. The gone table computed from the legacy pass equals `runs/headfree/headfree_table.json` with a maximum absolute difference of 0 over all 6 words, 2 seeds and 15 columns.

Determinism across jobs: 183 (context, arm or root) keys were computed in two or more jobs, 160 arms and 23 roots. Every one has identical latent and RGB hashes. Inside each lease, every `iso:`, `del:` and `native:` arm run through the port is bit-identical to its port-free root.

### (b) Gone scores at new precision against the old table

The gone score is `gone = (d(arm) - d(iso)) / (d(none) - d(iso))`, with `d = 1 - MAD(arm, del) / MAD(native, del)`, as in `headfree_table.py`. The "old" row is also the legacy-precision result, which is identical.

The "floor bound" row is the arm's floor in gone units: `floor_px / (MAD(native, del) × span)`. The floor in px is the larger of two distances: the recompute sham's distance from iso, and the direct delta-to-recompute twin gap. By the triangle inequality, a gone difference smaller than this bound cannot be separated from arithmetic noise.

`-` / "not run" marks arms in leases 7-10. The table shows a column subset; `report/b-gone-full.json` holds every column, including all `m:` twins, `own240-minus-own60`, the random controls and the other-item controls.

| word | seed | source | own60 | own240 | rand240 | modI | nounI | keep-own60 | keep-nounI | m:own60 |
|---|---|---|---|---|---|---|---|---|---|---|
| red | 4242 | old (= legacy, exact) | 0.48 | 0.66 | 0.12 | 0.25 | 0.03 | 0.16 | 0.66 | 0.31 |
| | | new precision | 0.47 | 0.64 | 0.13 | 0.27 | 0.02 | 0.14 | 0.66 | 0.29 |
| | | floor bound | 0.092 | 0.094 | 0.075 | 0.083 | 0.075 | 0.072 | 0.109 | 0.082 |
| red | 9001 | old (= legacy, exact) | 0.61 | 0.73 | 0.23 | 0.46 | -0.01 | 0.11 | 0.93 | 0.36 |
| | | new precision | 0.61 | 0.81 | 0.23 | 0.46 | 0.00 | 0.14 | 1.00 | 0.37 |
| | | floor bound | 0.140 | 0.198 | 0.099 | 0.097 | 0.115 | 0.118 | 0.119 | 0.111 |
| black | 4242 | old (= legacy, exact) | 0.31 | 0.72 | -0.03 | 0.14 | 0.11 | 0.03 | 0.58 | 0.45 |
| | | new precision | 0.31 | 0.71 | -0.05 | 0.13 | 0.12 | 0.04 | 0.58 | 0.42 |
| | | floor bound | 0.047 | 0.054 | 0.067 | 0.052 | 0.042 | 0.033 | 0.035 | 0.025 |
| black | 9001 | old (= legacy, exact) | 0.39 | 0.77 | 0.10 | 0.21 | 0.08 | 0.03 | 0.67 | 0.34 |
| | | new precision | 0.39 | 0.77 | 0.11 | 0.21 | 0.07 | 0.03 | - | 0.33 |
| | | floor bound | 0.035 | 0.034 | 0.031 | 0.034 | 0.032 | 0.033 | - | 0.024 |
| ball | 4242 | old (= legacy, exact) | 0.51 | 0.66 | 0.25 | -0.09 | 0.28 | -0.20 | 0.30 | 0.30 |
| | | new precision | 0.51 | 0.68 | 0.24 | -0.08 | 0.30 | -0.19 | 0.29 | 0.29 |
| | | floor bound | 0.071 | 0.060 | 0.083 | 0.057 | 0.089 | 0.064 | 0.086 | 0.071 |
| ball | 9001 | old (= legacy, exact) | 0.47 | 0.49 | 0.42 | -0.02 | 0.23 | -0.04 | 0.80 | 0.38 |
| | | new precision | not run | not run | not run | not run | not run | not run | not run | not run |
| snow | 4242 | old (= legacy, exact) | 0.27 | 0.46 | 0.49 | 0.27 | 0.04 | 0.65 | 0.81 | 0.32 |
| | | new precision | 0.26 | 0.45 | 0.46 | 0.27 | 0.03 | 0.66 | 0.81 | 0.32 |
| | | floor bound | 0.020 | 0.016 | 0.047 | 0.017 | 0.014 | 0.024 | 0.020 | 0.020 |
| snow | 9001 | old (= legacy, exact) | 0.13 | 0.17 | 0.16 | 0.11 | 0.09 | 0.56 | 0.74 | 0.17 |
| | | new precision | not run | not run | not run | not run | not run | not run | not run | not run |
| night | 4242 | old (= legacy, exact) | 0.56 | 0.85 | 0.45 | 0.20 | 0.09 | 0.69 | 1.05 | 0.54 |
| | | new precision | 0.56 | 0.85 | 0.46 | 0.19 | 0.07 | 0.69 | 1.06 | 0.55 |
| | | floor bound | 0.053 | 0.073 | 0.064 | 0.036 | 0.040 | 0.058 | 0.028 | 0.053 |
| night | 9001 | old (= legacy, exact) | 0.28 | 0.57 | 0.20 | 0.12 | -0.01 | 0.48 | 0.99 | 0.22 |
| | | new precision | not run | not run | not run | not run | not run | not run | not run | not run |
| rain | 4242 | old (= legacy, exact) | -0.10 | 0.22 | -0.06 | 0.15 | -0.24 | 0.38 | 0.89 | -0.12 |
| | | new precision | -0.10 | 0.22 | -0.06 | 0.16 | -0.25 | 0.40 | 0.90 | -0.12 |
| | | floor bound | 0.137 | 0.128 | 0.110 | 0.159 | 0.129 | 0.135 | 0.095 | 0.137 |
| rain | 9001 | old (= legacy, exact) | 0.09 | 0.66 | 0.27 | 0.23 | -0.15 | 1.41 | 1.49 | 0.03 |
| | | new precision | not run | not run | not run | not run | not run | not run | not run | not run |

### (c) Regeneration from recipes

**Not run.** The separate job job-af5bfaf0fde2 is staged and queued. It regenerates three arms:

- `smoke-legacy@4242:z:own60:ball` (legacy precision);
- `full-00@4242:m:own60:red` (renormalising cut);
- `full-00@4242:z:keep-own60:red` (delta zero cut, 540 heads).

It could not start because the mrun host agent is down. The regenerate path was exercised only locally, against a fake model lane (`compare_regeneration` and the recipe and port round trip). Hash equality on the GPU lane is therefore **not yet measured**. The cross-job determinism in (a), 183 keys with identical hashes, comes from re-executing the panel, not from recipes.

### (d) Verdicts: `attach_results` → `arm_effects` → `verdicts`

The outcome for each arm is its px MAD from its reference arm, `iso:<word>`. These rows are copied from `report/d-verdicts-selected.json` without editing. The picture-judgement slot is empty (`picture=None`): no picture judgement was made.

Seed 9001 is missing for `ball` and partly for `black` (leases 7-10), so those records are `control_missing` across contexts. For a keep claim the separating control must have a **larger** effect, so a positive margin means the control's effect exceeds the keep arm's. Random controls marked mass-confounded (`separates_identity=False`) are reported in the JSON but never counted, and are left out of the column below.

| item | unit set | claim | status | both types agree | zero cut 4242: effect / floor | renorm 4242: effect / floor | separating controls 4242 (zero): effect, margin, beyond floor | 9001 |
|---|---|---|---|---|---|---|---|---|
| red | own60 | necessary | not_supported | True | not_supported: 0.0311 / 0.0033 | not_supported: 0.0270 / 0.0029 | own60@rows-black (other_item) 0.0400, -0.0088, False | zero not_supported, renorm not_supported |
| red | modI | necessary | not_supported | True | not_supported: 0.0235 / 0.0030 | supported: 0.0204 / 0.0025 | modI@rows-black (other_item) 0.0255, -0.0020, False | zero not_supported, renorm not_supported |
| red | nounI | necessary | not_supported | True | not_supported: 0.0122 / 0.0027 | not_supported: 0.0112 / 0.0027 | rand0-nounI (random) 0.0122, +0.0000, False; nounI@rows-black (other_item) 0.0273, -0.0151, False | zero not_supported, renorm not_supported |
| red | keep-own60 | sufficient | not_supported | True | not_supported: 0.0131 / 0.0026 | not_supported: 0.0161 / 0.0025 | keep-own60@rows-black (other_item) 0.0126, -0.0005, False | zero not_supported, renorm not_supported |
| red | keep-modI | sufficient | not_supported | True | supported: 0.0219 / 0.0026 | not_supported: 0.0226 / 0.0026 | keep-modI@rows-black (other_item) 0.0262, +0.0042, True | zero not_supported, renorm not_supported |
| red | keep-nounI | sufficient | not_supported | True | not_supported: 0.0410 / 0.0039 | supported: 0.0327 / 0.0027 | keep-rand0-nounI (random) 0.0415, +0.0005, False; keep-nounI@rows-black (other_item) 0.0760, +0.0350, True | zero not_supported, renorm not_supported |
| black | own60 | necessary | supported | True | supported: 0.0615 / 0.0031 | supported: 0.0387 / 0.0017 | own60@rows-ball (other_item) 0.0144, +0.0471, True | zero supported, renorm supported |
| black | modI | necessary | not_supported | False | supported: 0.0373 / 0.0034 | not_supported: 0.0163 / 0.0019 | rand0-modI (random) 0.0143, +0.0230, True; modI@rows-ball (other_item) 0.0072, +0.0301, True | zero supported, renorm not_supported |
| black | nounI | necessary | not_supported | True | not_supported: 0.0170 / 0.0028 | not_supported: 0.0212 / 0.0028 | rand0-nounI (random) 0.0196, -0.0026, False; nounI@rows-ball (other_item) 0.0320, -0.0149, False | zero not_supported, renorm not_supported |
| black | keep-own60 | sufficient | supported | True | supported: 0.0101 / 0.0022 | supported: 0.0137 / 0.0018 | keep-own60@rows-ball (other_item) 0.0724, +0.0623, True | zero supported, renorm supported |
| black | keep-modI | sufficient | control_missing | True | supported: 0.0252 / 0.0064 | supported: 0.0262 / 0.0019 | keep-rand0-modI (random) 0.0766, +0.0514, True; keep-modI@rows-ball (other_item) 0.1009, +0.0757, True | not run |
| black | keep-nounI | sufficient | control_missing | True | not_supported: 0.1099 / 0.0023 | not_supported: 0.0855 / 0.0021 | keep-rand0-nounI (random) 0.0328, -0.0771, False; keep-nounI@rows-ball (other_item) 0.0686, -0.0413, False | not run |
| ball | own60 | necessary | control_missing | True | not_supported: 0.0537 / 0.0038 | not_supported: 0.0351 / 0.0038 | own60@rows-snow (other_item) 0.0536, +0.0001, False | not run |
| ball | modI | necessary | control_missing | True | not_supported: 0.0135 / 0.0031 | not_supported: 0.0138 / 0.0029 | rand0-modI (random) 0.0291, -0.0156, False; modI@rows-snow (other_item) 0.0667, -0.0532, False | not run |
| ball | nounI | necessary | control_missing | True | not_supported: 0.0338 / 0.0048 | not_supported: 0.0222 / 0.0048 | nounI@rows-snow (other_item) 0.0358, -0.0020, False | not run |
| ball | keep-own60 | sufficient | control_missing | True | supported: 0.0297 / 0.0035 | supported: 0.0372 / 0.0035 | keep-own60@rows-snow (other_item) 0.1990, +0.1694, True | not run |
| ball | keep-modI | sufficient | control_missing | True | not_supported: 0.0913 / 0.0026 | not_supported: 0.0681 / 0.0026 | keep-rand0-modI (random) 0.0554, -0.0359, False; keep-modI@rows-snow (other_item) 0.2117, +0.1204, True | not run |
| ball | keep-nounI | sufficient | control_missing | True | supported: 0.0447 / 0.0046 | supported: 0.0367 / 0.0030 | keep-nounI@rows-snow (other_item) 0.2291, +0.1843, True | not run |


## Workarounds and gaps in the shared modules

None of these blocked the run. The shared modules were not edited; each workaround lives in `flux_read_panels.py`.

1. **Per-item random-control mass** (`attention_panels.py:289-303`, `RandomControls`; `:342-349`, `_draw_random`). `RandomControls.mass` is one vector for every item. A word's own write-mass share is item-specific, so the mass-matched random draw needs one `read_panel` call per word. The per-word panels are then joined with `merge_panels`.
   - Needed: `RandomControls.mass` as `{item: vector}`, or a callable that takes the item.
2. **Prompt-level arms** (`attention_panels.py:561-584`). `read_panel` emits only per-item arms, with receivers and sources templated by the item. The read-graph `sham` and `img<-alltext` arms belong to the prompt as a whole, so `prompt_level_arms` builds them as `Arm` records that follow the generator's naming, floor and twin conventions.
   - Needed: a `prompt_references=` option, or extra all-cut templates without `{item}`.
3. **No `Cut.from_dict`** (`attention_port.py:157-166` has only `to_dict`). Regenerating a port from a recipe needs the inverse, so `flux_read_panels.cut_from_dict` provides it. It is tested on every cut kind, including shams and `floor_arms`, as a byte-equal round trip.
   - Needed: `Cut.from_dict` in the port.
4. **`metadata()` after exit loses the cuts** (`attention_port.py:435-443`). `__exit__` sets `self.cuts = ()`, so `port.metadata()` read after the `with` block reports no cuts. The worker therefore reads it inside the block.
   - Needed: keep the last apply's metadata, for example in `port.last_metadata`.
5. **References that are not arm ids** (`attention_panels.py:568, 574`). `iso:<w>` and `del:<w>` carry `reference="native"`, which is not an arm id unless `references` includes `"native"`. The worker resolves `native` to the root native image of the item's prompt.
   - Needed: either always emit a native reference arm, or document `"native"` as a reserved root name.
6. Already documented in `ATTENTION-PORT.md`: `port.log` recompute counters are empty under `precision="legacy"`, so legacy rows have `recomputed=None`.

Scheduler limit: one mrun job may run at most 256 local work items (`scheduling_contract.MAX_LOCAL_WORK_ITEMS`). One lease plus its reference roots is therefore one job, and `leases_within` sizes the lease cap so that every lease fits.

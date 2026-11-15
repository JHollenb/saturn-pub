# Lanes and the refactored engine: integration points

Written 2026-11-14 10:46 PST, before any lane code was changed, on branch
`claude/mdb-lanes-integration-20261114` = `claude/mdb-lanes-port-20261113` (55d6f8e) merged with
`origin/mdb-engine-refactor` at **bcdd0709b18227f4dc642c6b1aec23a3cbf0c048** (merge commit
fbb9a54). The merge had no conflicts: the two sides touch disjoint files (MEASURED:
`git diff --name-only` of each side against 8c019ee, intersection empty). Paths below are
relative to the repository root. Each line says MEASURED (read in the code or run) or ASSERTED.

## How lanes plug in

- `packages/mdb/src/mdb/core/plugins.py`: two entry-point groups, `mdb.lanes` (verbs; each a
  `main(argv)` that receives the verb first) and `mdb.lanes.exports` (names of the `mdb`
  package). MEASURED (code). Unchanged by the refactor (the file is not in its diff).
- `packages/mdb/src/mdb/cli.py`: `mdb lanes` lists verbs, `mdb lanes VERB ...` runs one, a bare
  `mdb VERB` runs it after a notice. MEASURED (code).
- `packages/mdb-lanes/src/mdb_lanes/lanes_compat.py`: the lane side. `TOOLS` (verb -> module
  whose `main(argv[1:])` runs it), `BENCH_VERBS`, `DEBUGGER_VERBS`, `EXPORTS`; `--write`
  regenerates the source-tree stub `packages/mdb-lanes/src/mdb_lanes-0.1.0.dist-info`, which
  `packages/mdb/tests/test_cli_plugins.py` requires to be current. `pyproject.toml` of the lane
  package declares the same groups for installs. MEASURED (code).
- Neither read panel lane had a verb: `flux_read_panels.py` and `qwen_panels.py` have no
  `main`; they are launched by `packages/mdb-lanes/qualification/{flux_read_panel,qwen_panel}_stage.py`
  (plans / stage / submit / report) and run by the matching `*_worker.py` on mrun. MEASURED.
- The engine's op registry (`packages/mdb/src/mdb/ops/registry.py`, `register`, `hook`) has no
  entry-point discovery: a lane op appears in the REPL only if something imports the lane module
  first, and the engine never does. MEASURED (code). So lanes plug in as verbs, not as REPL ops.
- `packages/mdb/AGENTS.md`: runtime imports, worker payloads and launchers must resolve installed
  packages and never discover sibling checkout paths or patch `sys.path`. MEASURED (text).

## Engine facilities that overlap lane code

| Engine facility | File | What it offers | Lane code it overlaps | Status |
|---|---|---|---|---|
| passive capture `mdb.trace` | `packages/mdb/src/mdb/engine/trace.py` | native `L<l>.p` (attention weights, native dtype), `s` replica, any port, batch > 1, during the caller's own forward; bitwise passive | `attention_port.observe` / `Read.probs()` (FP32 softmax recomputed from q/k) in the Qwen worker | Qwen2/Qwen3/Llama/Mistral only; eager attention needed for `p`. Equality with the lane's FP32 recompute is not yet known (no saved engine capture). |
| per-source contribution | `packages/mdb/src/mdb/analysis/reads.py` (`head_mix`, `readout`, `reads`) | `p[h,s] W_o,h v[kv(h),s]` in float64 on captured operands, through the P19 final-norm readout to a margin (`derived-exact`) | `predicted_write.zero_cut_delta` + `head_writes` (same identity, Qwen has no gate) and the experiment's per-unit scores | Single row, single source, one consumer; no mask (renormalising) cut. |
| measured cut | `reads.verdict` -> `propagate.whatif` | zero cut `p[h,src] = 0` (masked `replace` at the `p` port), native margin change, P20 verdict against the emulated change | lane zero cut on one (head, source) at the receiver row (`attention_port` `renormalize=False`, `zero_method="delta"` or `"recompute"`) | One edit per what-if. Arithmetic differs from the lane's delta method (lane: `native_out - P[cut]@V[cut]` in FP32; engine: eager `p@v` with the entry zeroed): bit equality not yet known. |
| what-if / compare | `packages/mdb/src/mdb/analysis/propagate.py` (`whatif`, `whatif_many`, `Edit`) | one edit, sealed prediction, native measurement, restore; `whatif_many` shares one excursion | lane arms' single-cut measurement | No multi-edit what-if: a set cut over several layers is several edits; `compare` takes two. Staged edits on a `Run` (`run.set`) can realise a multi-layer set natively, without the sealed prediction. |
| replay recipe v2 | `packages/mdb/src/mdb/core/recipe.py` | sealed recipe of a stepped `Run` (prefill split, transitions with edits, staged edits, port and cell edits, `mix DOSE` cell edits with `dose`, masks, origins) | `mdb_lanes/native_replay_recipe.py` (v1: roots + events through `saturn_pub.Session`; copy-values only, no dose, no attention edit) and `qwen_panels.native_recipe` | v2 now expresses a fractional dose (cell `mix`) and a zero read cut (port `replace` 0 with a mask). It does not express a renormalising cut as an op (only as a supplied tensor), FLUX, or the lane's FP32 `blend` arithmetic (v2 `mix` interpolates in float64; the lane in FP32). |
| state-cell edits | `packages/mdb/src/mdb/engine/edits.py` (`CellEdit`, `CELL_OPS`) | `replace/add/scale/mix DOSE` on `cache.L<l>.k|v` cells | `attention_panels.blend` on `DynamicCache` values in the Qwen value panel | `mix` rounds a float64 interpolation once; `blend` rounds an FP32 one: bit equality not yet known for doses other than 0 and 1. |
| projects | `packages/mdb/src/mdb/core/project.py` (+ `core/index.py`, `core/project_read.py`) | the schema-3 project the lane `mdb_lanes.project.Project` writes, without lane code | lane `project.py` | The panel lanes write plain JSON results, not projects. No overlap to remove now. |
| exact identities | `packages/mdb/src/mdb/analysis/finite.py` | float64 identities of the proof series with measured `eta` | `analysis_metrics` rounding bounds (`Bounded`) | Different questions (identity closure vs readout rounding bound); no overlap removed. |
| op registry | `packages/mdb/src/mdb/ops/registry.py` | every verb, JSON in/out, 64 KiB bound | none (the lanes have CLIs) | No plug-in group for ops (see above). |
| port programs | `packages/mdb/src/mdb/engine/ports.py` (`FAMILIES`, `PROGRAMS`) | qwen2, qwen3, llama, mistral | FLUX lane (`attention_port.FluxBackend`) | **No FLUX / diffusers port program** (MEASURED: `FAMILIES` lists four decoder families; `README.md` "Not built yet" names FLUX.2 Klein, issue M14). The FLUX read-panel lane keeps its own access layer. |

## Consequences for this integration (ASSERTED plan, in the task's order)

- 4a: add two lane verbs (`qwen-panels`, `flux-read-panels`) in `lanes_compat.py` + `pyproject.toml`
  + the stub, served by a new lane module; the verbs run the qualification launchers given by an
  explicit path (no sibling discovery), plus offline subcommands from the library.
- 4b: Qwen only (no FLUX program). Attention weights through `mdb.trace`; single-source zero cuts
  through `reads` / `whatif`; set cuts and the renormalising cut stay on the lane path; controls
  and verdicts stay in `attention_panels`. Equivalence needs a model run (no saved engine capture):
  one FP32 smoke job.
- 4c: emit v2 for the arms v2 can express; keep reading v1.
- 4d, 4e: lane-only changes (no engine files involved).

## What was done (appended 2026-11-14 11:03 PST)

- 4a done: `mdb lanes qwen-panels` / `mdb lanes flux-read-panels` (`mdb_lanes/panel_lanes.py`;
  `lanes_compat.PANEL_VERBS`; pyproject + stub, 48 verbs). MEASURED: plans rebuilt through the verb
  are byte-identical to the sealed value plans (sha256 6e62c42c…, bdba6c57…).
- 4b done for Qwen, with a measured caveat: `mdb_lanes/qwen_engine.py` (trace, contributions,
  `whatif_many` single cuts, staged multi-layer sets, `engine_rows` scored by the panel's own
  `read_results` / `verdicts`). Smoke job-0fcdf8414918 (FP32 CUDA): base logits and attention
  weights bit-equal to the lane; cuts within 5.7e-6 of the lane's recompute arithmetic but bit-equal
  in only 4/24 single cuts and 3/5 sets; no verdict changes. Renormalising cuts, delta-method arms,
  keep arms of sealed plans, BF16 and FLUX stay on `attention_port`.
  `experiments/engine_equivalence_20261114/FINDINGS.md`.
- 4c done: `qwen_engine.v2_checkpoint` writes replay recipe v2 through the engine's project (edits,
  masks, cell `mix` doses); `qwen_engine.read_recipe` reads v1 and v2; `qwen_engine.V2_GAPS` lists
  what v2 does not express (renormalising cut, delta zero cut, FP32 blend dose arithmetic, FLUX,
  other row groupings). v1 `native_replay_recipe` is unchanged and still read.
- 4d done: `predicted_write.receiver_write_scores`, `pw_shares`, `clean_random_draws` (plus the
  existing `gated_unit_mass`); docs `packages/mdb-experiments/docs/FLUX-READ-PANELS.md` ("Ranking heads") and
  `ATTENTION-PANELS.md`. MEASURED: reproduces the experiment's addendum-2 selection exactly.
- 4e done: `mdb_lanes/flux_nnsight.py` (import-guarded; extra `nnsight = ["nnsight==0.8.2"]`); its
  functions are `bridge.py`'s verbatim (tested), so job-8fd7bb3b1604's parity applies to them for the
  one cell it measured; nothing else is covered (see the module docstring).
- No engine file was changed.

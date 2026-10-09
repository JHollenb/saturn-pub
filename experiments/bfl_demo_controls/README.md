# BFL demo controls (FLUX.2 Klein 4B)

Two small, public-repo-reproducible control experiments that harden two existing BFL demo
pages. All GPU compute runs on a GPU worker (RTX 4080) through the public `mrun` client; the model
is `black-forest-labs/FLUX.2-klein-4B` (rev `e7b7dc27f91deacad38e78976d1f2b499d76a294`), BF16,
guidance 1.0, 4 steps, text-only, via `saturn_pub.adapters.flux2.Flux2KleinAdapter`.

- **Exp 1 — "Is a route edit more than a prompt swap?"** Reproduces the
  counterfactual-diffusion-futures route arm (text-state substitution at `joint.2/3/4` and
  the text rows of `single.0`, `R_branch = R_src + dose·(R_donor − R_src)`) and compares it,
  at cuts 0 and 2, to a mid-trajectory conditioning (prompt) swap. Objective MAD-progress
  scoring; pre-registered decision rule in `PREREG.md`.
- **Exp 2 — camera-contact replication (text-only, blinded).** Captures native `joint.3`
  text states from matched left/right prompts, forms `M=(H_L+H_R)/2`, `D=(H_R−H_L)/2`, the
  predicate projector `P`, and writes `M`, `M±D_rest`, `M±D_pred` arms into a
  neutral-conditioned trajectory over 16 fresh seeds (512²). Images are written under random
  ids with a separate key for **blinded** judging by the coordinator; the worker never judges.

## Files

- `PREREG.md` — pre-registration (committed before the scored jobs): prompts, route sites,
  cut steps, scores, decision rule, sign convention, blinding, smoke gate.
- `run.py` — the worker. `--experiment {exp1,exp2}` runs the frozen mechanics smoke gate in
  process, aborts on failure (non-zero exit → failed job), else runs the full panel and
  writes `report.json`, images, and contact sheet(s). `--tiny` runs the identical control
  flow on the adapter's tiny CPU fixtures (no weights/GPU) as the local unit test.
- `launch.py` — submits one experiment to a GPU worker through `mrun.client.submit.launch`
  (mrun-pub). Offline `uv` overlay on the operator `mrun` env; scratch/outputs on
  `<scratch-root>`; weights from `<model-root>`; `retry_on_kill=False`.
- `results/` — collected `report.json`, images, contact sheets, and (Exp 2) `judge/` +
  `judge-key.json`.

## Reproduce

Local unit test (CPU, no model):

```sh
PYTHONPATH=src python experiments/bfl_demo_controls/run.py --experiment exp1 --tiny \
  --output /tmp/exp1-tiny
```

Scored run (a GPU worker, through mrun):

```sh
MRUN_URL=http://<mrun-host>:9025 \
/path/to/mrun-pub/.venv/bin/python experiments/bfl_demo_controls/launch.py \
  --experiment exp1 --vram-mb 14200 --ram-mb 28000 --wall-s 600 --timeout-s 1800
# then rsync <worker>:<scratch-root>/bfl-controls/<label>/ into results/exp1/
```

The worker never judges Exp 2 contact; it emits blinded sheets + a key for the coordinator.

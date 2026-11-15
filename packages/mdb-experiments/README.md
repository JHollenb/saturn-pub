# mdb-experiments

Internal research that uses mdb. **This directory is not part of the `mdb-runtime` package**: it is
not built, not installed and not in the sdist or wheel. It holds the experiments we ran with mdb,
their results and audits, the internal issue and investigation ledgers, and the lane-era
documentation of the older `mdb` tool (now the `mdb-lanes` plug-in).

If you want to learn mdb, read [`../mdb/README.md`](../mdb/README.md), [`../mdb/docs/`](../mdb/docs/)
and [`../mdb/examples/`](../mdb/examples/) instead.

Everything here was moved out of `packages/mdb` with `git mv` on 2026-11-14 (branch
`mdb-docs-split`), keeping the old layout under this directory, so `packages/mdb/X` is now
`packages/mdb-experiments/X`. File contents were not changed, except path references in the
lane-era docs and tests that pointed at the old location. Results and experiment trees are
custody records: do not edit them in place.

## Index of what moved

| Here | Was | What it is |
|---|---|---|
| `experiments/` | `packages/mdb/experiments/` | Experiment code: `time_circuits` (E6/E20 paper adapters), `native_discovery`, `jacobian-reader-writer`, `cot-register`, `symbol-lifecycle`, `knowledge_interchange`, `interop_20261113`. |
| `results/` | `packages/mdb/results/` | Run outputs, preregistrations and audits: `showcase-20261114` (the source runs of examples 02-07), `sae-integration-20261113` (example 01), `scale-20261114`, `twohop-20261113`, `amplitude-20261114`, `cot-monitor-20261113`, `interop-20261113`, `project-analysis`, `release-gate-0.2.0`. |
| `docs/*.md` (188 files) | `packages/mdb/docs/` | Experiment reports (`FLUX-*`, `QWEN-*`, `COMPUTATION-*`, `AUDIT-*`, `SYMBOL-*` campaign notes, proposals, the branch-change trial) and the lane-era tool docs (`USAGE.md`, `LOCAL-DEBUGGER.md`, `BACKENDS.md`, `EXPERIMENTS.md`, `QUALIFICATION.md`, `TRANSFORMERLENS-DEBUGGER.md`, ...). |
| `docs/assets/`, `docs/design/`, `docs/qualification/` | `packages/mdb/docs/...` | Figures and JSON accounts of the experiment reports; ISA design examples; lane-era qualification records. |
| `examples/` | `packages/mdb/examples/` (loose files) | Older, unapproved demos, mostly for the lane plug-in: `resident_lab.py`, `pass_indexed_baseline.py`, `collect_session.py`, `read_weight_*_demo.py`, `discovery/`, `project-navigation/`, `symbol-trace/`, command JSONs, temporal-causal-viewer samples. |
| `tools/` | `packages/mdb/tools/` | Maintainer tools of the lane split and extraction (`split_lanes.py`, `extract_*.py`, `finalize_provenance.py`) and the branch-change trial drivers. |
| `PROVENANCE.json` | `packages/mdb/PROVENANCE.json` | Extraction provenance of the lane-era runtime. |
| `issues.jsonl` | `packages/mdb/issues.jsonl` | Internal issue ledger (ids M*, I*, S*, SC*, DC*, CU*, RG*, ...). User-facing known issues are in `../mdb/CHANGELOG.md` "Known issues". |
| `investigations.jsonl` | `packages/mdb/investigations.jsonl` | Investigation ideas (status open, investigated, closed, follow-up). |
| `features.json` | `packages/mdb/features.json` | Feature requests (F*). |
| `tests/test_exp6_calibration.py` | part of `packages/mdb/tests/test_analysis_stability.py` | Checks that `mdb.analysis.stability.calibrate` reproduces the showcase exp6 audit loss table from `results/showcase-20261114/exp6/out`. |

The ledgers (`issues.jsonl`, `investigations.jsonl`) keep their historical text, so old entries
still name `packages/mdb/...` paths; read those as `packages/mdb-experiments/...`. Append to them
line by line; never rewrite them through a JSON round trip.

## Running the moved test

From `packages/mdb`, with the mdb test environment:

```
PYTHONPATH=src:../../src python -m pytest -q ../mdb-experiments/tests
```

Lane-era tests that read files here (`../mdb-lanes/tests/...`) were updated to the new paths.

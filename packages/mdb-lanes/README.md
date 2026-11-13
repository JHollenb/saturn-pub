# mdb-lanes

`mdb-lanes` (import name `mdb_lanes`) holds the experiment code that used to live inside
`mdb`: bench sessions, projects, state graphs, circuit and symbol tools, FLUX and Mamba lanes,
weight-space programs and their qualification tree (`qualification/`). It was moved out of the
engine by a scripted import rewrite (`mdb.X` became `mdb_lanes.X`); nothing was deleted.

The engine (`../mdb`, distribution `mdb-runtime`) never imports this package. The lanes plug
into it through two entry-point groups declared in `pyproject.toml`:

- `mdb.lanes`: command verbs. `mdb lanes` lists them and `mdb lanes VERB ...` runs one
  (`mdb VERB` also works, after a notice on stderr).
- `mdb.lanes.exports`: names of the `mdb` package that the engine does not define, such as
  `mdb.BenchSession` or `mdb.Project`, resolved on first use.

`src/mdb_lanes-0.1.0.dist-info` makes those entry points resolve from the source tree without
an install. From the repository root `W`:

```
PYTHONPATH=$W/src:$W/packages/mdb/src:$W/packages/mdb-lanes/src python -m mdb lanes
PYTHONPATH=$W/src:$W/packages/mdb/src:$W/packages/mdb-lanes/src python -c "import mdb; print(mdb.BenchSession.__module__)"
```

The first prints the 46 lane verbs; the second prints `mdb_lanes.bench`.

## Tests

```
cd $W/packages/mdb-lanes
env PYTHONPATH=$W/src:$W/packages/mdb/src:$W/packages/mdb-lanes/src OMP_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 python -m pytest -q -p no:cacheprovider tests/<file>
```

This recipe was run here on `tests/test_analysis_metrics.py` (8 passed). Tests that start
`python -I` children ignore `PYTHONPATH`; they need an environment where `mdb-runtime` and
`mdb-lanes` resolve as installed distributions from this checkout.

## Documentation

The lane write-ups are still in `../mdb/docs/` (attention ports, circuit discovery, compiled
state, FLUX experiments, weight-space programming, qualification records and more), with
examples in `../mdb/examples/` and experiments in `../mdb/experiments/`. The pre-split README,
an index of those write-ups, is in git history: `git show 9ec645b:packages/mdb/README.md`.

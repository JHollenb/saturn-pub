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

The first prints the 48 lane verbs; the second prints `mdb_lanes.bench`.

The two attention read-panel lanes are verbs too (`src/mdb_lanes/panel_lanes.py`):

```
mdb lanes qwen-panels info                      # the case, schemas, launcher, worker, docs
mdb lanes flux-read-panels check PLAN.json --seal SEAL.json
mdb lanes qwen-panels --qualification $W/packages/mdb-lanes/qualification plans --profile smoke --out DIR
mdb lanes flux-read-panels --qualification $W/packages/mdb-lanes/qualification stage RUN --plans P --job smoke
```

`plans`, `stage`, `submit`, `report` (FLUX) and `analyze` (Qwen) run the hash-bound launchers in
`qualification/` unchanged; their directory is passed explicitly (`--qualification` or
`MDB_LANES_QUALIFICATION`), never searched for. How the lanes meet the refactored engine
(`mdb.trace`, `analysis/reads.py`, `propagate.whatif`, replay recipe v2) is in
[`docs/ENGINE-INTEGRATION.md`](docs/ENGINE-INTEGRATION.md).

## Publication mode

Qwen bench sessions (`BenchSession.from_resident` / `BenchSession.resume` on a Qwen2 model, and
`mdb lanes bench open|resume --publication ...`) take one setting, `publication`:

| value | behaviour |
| --- | --- |
| `"per_call"` (default, journal mode) | one durable publish (fsync) per `step`/`generate` call. Faster: on the developers' GPU host a 32-token Qwen2.5-0.5B decode took 2.60 s in total with per-token publication, 0.79 s of it in 128 fsyncs. Each token cut is still its own immutable artifact in the call's sealed pack and restorable by ref; the call is all-or-nothing. A crash mid-call loses the whole call's tokens. |
| `"per_token"` | durable after every token, so a crash loses at most the token in flight. Slower. |

```python
BenchSession.from_resident(model, tok, root=r)                          # per_call
BenchSession.from_resident(model, tok, root=r, publication="per_token")
BenchSession.resume(model, tok, root=r, publication="per_token")       # not stored in the session
```

Anything else is refused; `"per_call"` needs `publication_mode="journal"`. The older
`command_publication=True/False` still works as an alias for `"per_call"`/`"per_token"`.
`session.publication_policy` reports what is in force. Mamba and FLUX sessions refuse the
setting.

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

The lane write-ups are in `../mdb-experiments/docs/` (attention ports, circuit discovery,
compiled state, FLUX experiments, weight-space programming, qualification records and more), with
examples in `../mdb-experiments/examples/` and experiments in `../mdb-experiments/experiments/`;
the engine integration notes are in [`docs/ENGINE-INTEGRATION.md`](docs/ENGINE-INTEGRATION.md).
The pre-split README, an index of those write-ups, is in git history:
`git show 9ec645b:packages/mdb/README.md`.

# mdb examples

Each example is a directory with a script, and for 01-07 the same cells as a notebook. Only
examples approved by the maintainers are added here.

Start with **00**: it runs on a laptop CPU in a few seconds with nothing downloaded. Examples 01-07
are real experiments on Qwen3/Qwen2.5 checkpoints and need a CUDA GPU and local model files.

| Example | What it teaches | Needs |
|---|---|---|
| [`00-hello-tiny/`](00-hello-tiny/) | The debugger loop in about 60 lines: `step` through a layer's ports, `set` an edit, `watch` a value, `Branches` from one checkpoint with a null arm, the `symbols` table, `forecast` then `verify`. | CPU only; `mdb-runtime[autoregressive]`; random tiny Qwen2, no download. About 3 s. |
| [`01-sae-census-qwen3-1.7b/`](01-sae-census-qwen3-1.7b/) | SAEs with mdb: `sae load`, `features` with the error term, a variable census (feature vs error share), splicing out the SAE error, steering a feature. | GPU (peak 4.6 GB VRAM); Qwen3-1.7B-Base; Qwen-Scope SAE `SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`; `matplotlib`. |
| [`02-tl-reparametrisation-floor/`](02-tl-reparametrisation-floor/) | How large an effect must be before its verdict survives another realisation of the same forward: mdb vs TransformerLens processed and unprocessed bridges. `names.tl_hook`, selectors, `Branches`, sealed supplied forecasts, the stability band. | GPU (peak 12 GB VRAM, 16 GB host RAM); Qwen3-0.6B-Base and Qwen2.5-0.5B; TransformerLens 4.4 (`interop` extra). |
| [`03-composer-neurons-sealed-forecasts/`](03-composer-neurons-sealed-forecasts/) | Another library nominates, mdb measures: nnsight gradients nominate MLP neurons, mdb seals the forecasts and measures the knock-outs natively on held-out items. `Pool` (leave-one-out mean), `f`-port selectors, `Branches` nulls, custody. | GPU (peak 6.9 GB VRAM); Qwen3-1.7B-Base; nnsight (`interop` extra). |
| [`04-watch-edit-continue-country-feature/`](04-watch-edit-continue-country-feature/) | The gdb loop on a real model: a `FeatureCondition` watch on an SAE feature (false alarms shown), break, `Branches` with a one-component repair, a pool donor and a matched-norm null, continue. | GPU (peak 3.8 GB VRAM); Qwen3-1.7B-Base; the Qwen-Scope SAE; `matplotlib`. |
| [`05-sae-component-binding-carrier/`](05-sae-component-binding-carrier/) | SAE features as component bindings of one KV-cache value cell: the named country-feature component moves 0.59 of the cell, a random component 0.0. `Binding.component`, the symbol table, `Branches` over layers. | GPU (peak 4.0 GB VRAM, 11 GB host RAM); Qwen3-1.7B-Base; the Qwen-Scope SAE; SAELens; `matplotlib`. |
| [`06-interchange-subspace-pyvene/`](06-interchange-subspace-pyvene/) | Interchange interventions at one cell: pyvene and mdb on the same flips; `SubspaceBinding` for a training-free SAE subspace (a negative result) and readout subspaces. | GPU (peak 3.9 GB VRAM); Qwen3-1.7B-Base; the Qwen-Scope SAE; pyvene 0.1.8; `matplotlib`. |
| [`07-frozen-gradient-brackets-transcoders/`](07-frozen-gradient-brackets-transcoders/) | Transcoder nodes and a frozen-gradient edge from nnsight, measured natively with holds (frozen attention pattern and norm scales) as `Branches` arms: the edge explains 0.92-0.97 of the native effect; the direct edge overstates it about 1.6x. | GPU (peak 3.9 GB VRAM, 9.6 GB host RAM); Qwen3-1.7B-Base; Qwen3-1.7B transcoders; nnsight; `matplotlib`. |

## 00: run it

```
pip install . "./packages/mdb[autoregressive]"     # from the saturn-pub checkout, see ../README.md
python packages/mdb/examples/00-hello-tiny/hello_tiny.py
```

The output should equal [`00-hello-tiny/expected_output.txt`](00-hello-tiny/expected_output.txt)
except the `record` id of the forecast, which digests the installed library versions.
`tests/test_example_hello_tiny.py` checks this.

## 01-07: what is in a directory

`<name>.py` is the walkthrough as a script (percent-format cells); `<name>.ipynb` is the same cells
as a notebook, generated from the script. `expected_outputs.md` has the numbers the validation run
measured and what each one means; `validation-job-<id>.log` is that run's log. `items.json`,
`prereg.json` and `reference.json` are the example's inputs, its preregistered rules and the
numbers of the original, larger run it is compared against. `run.sh` is the job entry point.

## 01-07: run them

- Your own GPU: install `mdb-runtime[autoregressive]` plus the libraries in the table, set
  `MDB_MODEL` (and `MDB_SAE` or `MDB_TRANSCODERS`, or `MDB_MODELS` for 02) to local checkpoint
  directories, `MDB_DEVICE` if not `cuda`, and run `python <name>.py` or open the notebook.
  Models are read from local disk only (`local_files_only`); nothing is downloaded. Outputs go to
  `MDB_OUT` (default `out/`); records are appended to `out/records.jsonl`, so a rerun into the
  same directory needs `append=True`, a fresh `MDB_OUT` or deleting the old file.
- `run_on_beast.sh` is how the maintainers ran them: it stages the example with the package
  sources and submits one job to their GPU host through the mrun scheduler. It needs that
  scheduler and is not required to run an example.

The numbers in `expected_outputs.md` were measured on one RTX 4080 (16 GB), bf16, eager attention,
one seed; each run took under 10 minutes.

The validation runs sealed the sha256 of each script as it ran. The scripts' comments and
markdown cells were edited afterwards to remove internal references; the code is unchanged (the
syntax trees without comments and docstrings are identical), so a rerun computes the same numbers
but its custody records carry a different script sha256.

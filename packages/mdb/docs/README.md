# mdb documentation

The main reference is the package [README](../README.md): install, hello world, addresses, ports,
verbs, predictions, verdicts, prompt sets, the web viewer and known limits. These pages go deeper
on one subject each.

| Page | What it covers |
|---|---|
| [INSTALL.md](INSTALL.md) | Installing from the saturn-pub repository, the extras, checking the install, building the web app and the wheel, the separate lanes plug-in. |
| [BRANCHES-POOLS-HOLDS.md](BRANCHES-POOLS-HOLDS.md) | `Run.set` with selectors and cross-run values, holds, `Branches`, `Pool`, core files, transcoders, custody and record sinks, multi-site feature watches. |
| [INTEROP.md](INTEROP.md) | `mdb.interop`: one name table for mdb, TransformerLens and nnsight; importing their values; batched nnsight edits; re-measuring another tool's claim natively; circuit-tracer graphs. |
| [SAE-WORKED-EXAMPLE.md](SAE-WORKED-EXAMPLE.md) | Using a residual SAE suite with mdb: `sae load`, `features`, `feature`, error splicing, feature edits, unit matching, a variable census (numbers measured on a real model). |
| [SYMBOL-TABLE-SCHEMA.md](SYMBOL-TABLE-SCHEMA.md) | Column reference of the symbol table (`symbols`), its frame export and JSON schema. |
| [PROOF-IDS.md](PROOF-IDS.md) | What the `P01` to `P33` tags in mdb's output mean. |
| [PERFORMANCE.md](PERFORMANCE.md) | What is fast since 0.2.1 (measured), `lookahead="auto"`, the cost of foreign hooks, and the known remaining costs with planned fixes. |
| [RECORDS.md](RECORDS.md) | The record format (schema v3): stored form, ids, column claims, bounds, reading v2 records and `mdb records migrate`. |

Runnable examples are in [../examples](../examples/README.md); start with `00-hello-tiny`.

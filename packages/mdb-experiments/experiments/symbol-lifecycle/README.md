# Native symbol lifecycle assay

This is a deliberately small public MDB assay with one explicitly typed, reversible cache Act. It
loads one pretrained Qwen2.5-0.5B resident model inside one mrun CUDA lease and:

1. pre-fills a short lookup prompt and records a committed source cut;
2. reads one committed cache layer's token rows across every KV head and registers an exact
   public `SymbolBinding` source-row coalition;
3. forks same-history donor branches, zeroes each member and the coalition, and makes a half-scale
   coalition control through the experiment-local typed `DecoderMemoryAdapter` Act;
4. installs those donor rows into public `Session` candidates and probes one shared baseline plus
   seven one-token native suffix arms, including a matched donor and same-payload wrong-target-row
   control;
5. restores the recipient parent, saves the sealed public cuts/receipts and lifecycle sidecar in
   `Project`, copies that project as an export, and reopens both project and native `Session` state.

The worker records model identity, the changed execution contract, branch labels, execution-point
clocks, row selectors, donor cuts, registered Act receipts, the actual consumer string, lifecycle
fingerprints, and export/reopen checks. It signs Paris/Rome and Berlin readout values separately;
Berlin is a donor comparison, not evidence that the Paris/Rome labels are semantically correct.
The token text is a supplied row nomination, not semantic proof. `Project` remains
metadata/navigation storage; public `Session` owns K/V execution and native suffix authority.

The raw counted native suffix transitions plus registered Acts are 243: 232 one-token
transitions plus 11 Acts, with the baseline shared across all arms. The worker also reports two
independent prefills and nine restores (seven arm restores, one Project reopen restore, and one
export reopen restore), making 254 bounded operations under the 256-operation ceiling. The
final-eight-layer coalition is omitted from this bounded default assay.

The launcher uses one guarded finite lease, strict preflight, CUDA admission, no automatic retry,
and an 840-second hard timeout. Explicit worker cache paths must belong to the execution host.
This invocation uses Beast's materialized runtime and ignores the shared launcher's Python path
so the worker staging process imports installed public packages:

```bash
common exec env MRUN_URL=http://192.168.68.68:9025 \
  MDB_WORKER_UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache \
  MDB_WORKER_TORCH_HOME=/mnt/hdd8tb/mdb/torch \
  MDB_WORKER_MODEL_ROOTS=/mnt/big/llm-models:/mnt/hdd8tb/llm-models \
  /path/to/mdb/.venv/bin/python -I /path/to/submit.py

common exec env MRUN_URL=http://192.168.68.68:9025 \
  /path/to/mdb/.venv/bin/python -I /path/to/submit.py --collect JOB_ID
```

The local mechanics path avoids model download and CUDA:

```bash
python worker.py --mode mechanics --root ./mechanics-session \
  --project-root ./mechanics-project --output ./mechanics-report.json
```

If the local interpreter has no Torch, that mechanics command falls back to the public MDB counter
debugger only; its Bench cuts are not consumed by the native lifecycle path.

The native result has separate `mechanics_status`, `instrument_status`, `trend_status`, and
`terminal_status` fields. A completed probe is evidence for the recorded native consumer and
selected rows; `trend_status=observations-only` records measurements without promotion. It does
not establish semantic CoT, unique symbol identity, or general causal localization.

`qualification/symbol_lifecycle_archive.py RESULT_DIR` checks the collected archive seal, extracts
only bounded regular files/directories, reopens both native projects with frameworks and private
imports blocked, verifies native page/blob byte hashes, and repeats forward, reverse, and hostile
SAS identity lookups. It performs no tensor decode or fresh native forward. See `FINDINGS.md` for
actual fleet results and preserved unsuccessful attempts.

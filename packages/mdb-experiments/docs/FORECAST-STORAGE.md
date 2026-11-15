# Packed forecast storage

`ForecastTransaction` publishes its numerical arrays in immutable bundles by default.
This amortizes file and directory synchronization across a complete publication,
while preserving the original NPY bytes, array hashes, values and explicit casts.
Existing individual-file transactions remain readable.

```python
with ForecastTransaction(
    output,
    model=model,
    model_identity=checkpoint_sha256,
    input_identity=input_sha256,
    source_identity=source_hashes,
    frontier={"kind": "raw-input"},
) as transaction:
    # frontier -> predict -> seal -> execute uses the same API and guards.
    ...
```

Use `array_storage="files-v1"` for the legacy layout. Existing transactions remain
readable and keep their original identities. Packed descriptors require the new
public readers; external scripts that call `np.load(root / descriptor["path"])`
directly must switch to `load_forecast_tree` or `load_forecast_array`, or explicitly
select the legacy writer.

## Defaults across discovery

The default changes apply to new publications, without rewriting retained evidence.

| Operation | Default | Explicit alternative |
|---|---|---|
| `FrozenNativeBatch.screen(rows)` | Largest admitted compatible batch, up to four in the qualified R3 profile | `batch_size=1` for scalar screening |
| `save_frontier` / `save_session_frontier` | Packed backing storages, `bundle-v1` | `storage_layout="slots-v1"` |
| `ForecastTransaction` | Packed numerical publications, `bundle-v1` | `array_storage="files-v1"` |
| Context-window and raw-input sequence campaigns | Packed numerical publications, `bundle-v1` | `--array-storage files-v1` |

Native screening and selected-state saving already used these defaults; forecast
transactions now do so too. Component stops and stepping continue through scalar
replay. No profile is inferred for an arbitrary model, and screening never commits
architectural state. Campaign prior-array readers accept both numerical layouts.

These choices can be used in one discovery workflow. Their separately measured
phase gains are not a measured combined campaign multiplier. In particular, the
58.64× publication, 6.34× worker and 3.24× whole-job ratios below are nested views
of the same forecast-storage reproduction.

The [default-change validation](../qualification/fast-defaults-20261030.json)
records 144 installed-wheel tests outside the workspace. It includes one tiny
CPU composition check (B4 screening → selected B1 replay → bundled cut restore
→ default bundled forecast → sealed suffix execution), both campaign CLI defaults
and legacy overrides, and both campaign numerical readers. Offline rechecks
retain all 1,161 original NPY identities and audit the eighteen historical
raw-input sequence rows. These are engineering and reader checks; no new
real-model execution or combined speed measurement is claimed.

## Publication and integrity

Each publication creates at most one new content-addressed `.npyb` file. It
contains a versioned header followed by complete, unchanged NPY files, ordered
by their NPY hashes. Duplicate values share a descriptor throughout a transaction.
Every supplied array is snapshotted and hashed again, so mutating and reusing a
Python ndarray cannot produce a stale cache hit.

A packed descriptor retains its original `sha256`, `content_sha256`, `shape` and
`dtype`, and adds `storage`, `bundle_sha256`, `offset` and `nbytes`. Its `path`
names the complete bundle. The bundle hash and the individual leaf hash are
distinct identities.

The writer synchronizes newly created directory entries, writes and fsyncs the
temporary bundle, publishes an immutable hard link, and fsyncs its containing
directory. Only then does it publish the referencing JSON and phase receipt.
`predict()` still returns after durable publication. Later observations create
new bundles; a sealed forecast bundle is never appended to or rewritten.

If metadata publication fails after its bundle was published, an unreferenced
durable bundle may remain. It does not become a completed forecast or authorize
native release. Physical retained-byte accounting includes bundle framing and
charges a duplicate array only once; existing file-count checks still apply.

The reader verifies the complete bundle hash, top-level NPY boundaries, the
individual NPY hash, numeric dtype/shape and content hash. Offsets into an NPY
array's payload do not become independent leaves, even when those bytes encode
another valid NPY. Unsafe paths, symlink ancestors, malformed ranges, truncated
data and inconsistent descriptors are rejected.

Global sealing also checks numerical references. Offline transaction verification
checks framing and byte identities with the standard library, without importing
NumPy or a model framework. An externally pinned seal/event head or archive is
still required to authenticate provenance. Trusted Python callback and native-entry
guard boundaries are unchanged.

## Reading either format

```python
import json
from mdb.forecast_transaction import load_forecast_array, load_forecast_tree

record = json.loads((transaction_root / "forecasts/campaign.json").read_text())
predictions = load_forecast_tree(transaction_root, record["predictions"])
payload = load_forecast_array(
    transaction_root, record["payloads"]["replacement"]["value"]
)
```

Both functions support legacy and packed descriptors. They return fresh,
immutable byte-backed NumPy arrays. Tree containers are immutable mappings and
tuples. `load_forecast_tree` shares verified bundle bytes within that one call;
later calls read and verify the retained files again. For a panel analysis,
decode its tree once rather than calling the single-array reader for every leaf.

A small array query can cost more in the packed layout because its bundle is
read and framed as a unit. This is a publication optimization, with reader costs
measured separately. Retain the complete transaction or every referenced bundle:
later records may refer to arrays published in earlier phases.

The [computation workflow](COMPUTATION-FORECAST-WORKFLOW.md) uses the same
transaction methods. The matched-context and raw-input sequence campaigns default
to bundles and accept `--array-storage files-v1`; their plans, equations, native
inputs and intervention recipes remain unchanged.

## Measured storage benchmark

Job `job-00aecea1310f` ran three isolated child processes sequentially in one
Beast CPU1 lease: original code at `29669c7`, updated code with individual files,
and updated code with bundles. Each consumed the same saved ten-row numerical
workload: 87 weight arrays and 1,161 array occurrences representing 496 unique
array values. There were no model calls or equation evaluations in this panel.

| Measurement | Original files | Updated files | Updated bundles |
|---|---:|---:|---:|
| Weight/declaration initialization | 11.950 s | 12.054 s | 2.173 s |
| Forecast callback plus publication | 29.530 s | 31.867 s | 0.542 s |
| Saved-observation callback plus publication | 25.154 s | 24.839 s | 1.035 s |
| Forecast-phase fsync calls | 456 | 457 | 9 |
| All measured fsync calls | 1,042 | 1,050 | 64 |
| Time inside all measured fsync calls | 67.087 s | 70.003 s | 5.299 s |
| Transaction files | 521 | 521 | 28 |
| Transaction bytes | 9,262,771 | 9,262,773 | 9,381,230 |
| Isolated-child peak RSS | 136.61 MiB | 137.60 MiB | 138.88 MiB |

The original forecast phase spent 29.342 of its 29.530 seconds inside fsync.
Packing reduced its measured publication time by 54.48×. The updated individual
file mode did not improve wall time: fewer repeated serializations cannot remove
hundreds of synchronous durable writes.

All 12,709 decoded tree nodes and all 1,161 original NPY leaf identities match
exactly across the three modes, including payload originals, casts and
provenance. Physical bytes increase by approximately 1.28% because descriptors
carry bundle locations. This is an I/O-layout optimization, with complete arrays
and precision retained.

Point reads and stronger validation have a cost. A warm public single-array
read was about 0.36–0.45 ms with the original files, versus 6.86 ms with bundles.
The public forecast-tree read was about 0.096–0.098 s originally, 0.209–0.210 s with
updated files and 0.256–0.257 s with bundles. The updated readers perform explicit
framing and descriptor checks. Decode the panel once for analysis rather than
repeatedly querying individual leaves.

These are fixed-order, warm-cache observations on one filesystem. cProfile and
fsync instrumentation were enabled equally. They are not cold-disk estimates or
cross-machine performance guarantees.

## Actual model campaign reproduction

Job `job-69bf68a89381` repeated the
[matched-context campaign](CONTEXT-AND-WRITE-TRANSFERS.md) with packed storage.
It executed the same seven already-tested native cases and reused the same
three prior controls. This is a performance reproduction, not new scientific
specimens or unseen-state prediction.

| Measurement | Prior files run | Packed reproduction |
|---|---:|---:|
| Forecast equations | 0.05434 s | 0.05441 s |
| Forecast handling/publication | 28.75045 s | 0.49031 s |
| Observation handling/publication | 22.27237 s | 1.06374 s |
| Worker wall time | 74.35391 s | 11.73697 s |
| Scheduled-job elapsed time | 91.97194 s | 28.39902 s |
| Array storage files | 496 | 3 |
| Transaction files | 521 | 28 |
| Physical native calls / prior rows | 7 / 3 | 7 / 3 |
| Typed Acts / replay primitives | 13 / 5,805 | 13 / 5,805 |
| Peak resident RAM | 1,028.2 MiB | 1,033.1 MiB |

The observed gains are 58.64× for forecast publication, 6.34× for worker wall time
and 3.24× for the complete scheduled job. Native-recording time also varied
between runs; its phase includes capture I/O and filesystem/cache conditions, so
it is not evidence of faster neural kernels. The equation time is unchanged.

The independent comparison verifies byte-exact weights, numerical forecasts,
payload originals/values/casts/provenance, and all ten rows' native heads,
36 intermediate arrays, roots, decoded state and before-state. All 1,161 NPY
occurrences retain their original shape, dtype, NPY hash and content hash.
The separate context auditor verifies all eleven heads, literal roots, native
ports, typed Acts and replay receipts. Its numerical residuals are identical to
the previous report.

At the measured revision, the installed qualification wheel, benchmark payload,
reproduction payload and working source matched across 174 runtime modules. That
qualification contains 107 focused
installed-wheel tests: 51 storage tests, 16 transaction tests and 40 prefix/port/
equation tests, run outside the workspace. Coverage includes directory-entry
durability, interrupted publication, malformed or rehashed counterfeit NPY,
truncation, symlinks, byte budgets, no-overwrite rules, fresh immutable reads,
legacy compatibility and framework-free verification. No hardware power-cut
experiment is claimed.

Two setup failures are retained. `benchmark-01` refused the model-free scheduling
contract before submission. Job `job-bb480ff342af` in `benchmark-02` failed before
the workload because the launcher inherited a Mac cache directory on Beast.
The corrected `benchmark-03` declares analysis work with zero model loads and
uses explicit qualified worker cache paths. Neither failed attempt contributes
a performance measurement.

The [portable evidence index](../qualification/forecast-storage-20261030.json),
[benchmark](../qualification/forecast_storage_benchmark.py),
[run comparator](../qualification/compare_forecast_storage_runs.py) and
[publisher](../qualification/publish_forecast_storage.py) retain the measured
scope. Raw artifacts and profiles are under
`artifacts/mdb-forecast-storage-20261030/benchmark-03`; the native audit and
comparison are under `artifacts/mdb-forecast-storage-20261030/reproduction-01`.

The offline Saturn sweep verifies thirteen derived records: three storage-only
measurements and ten known-case model rows. The scoped run and verification
return zero, with zero watch findings and zero automatic claims. Its
`target-streams-only` scope verifies numerical-stream mechanics; it is not a
second certification of the model or MDB receipts. The initial overly broad
results-root refusal is retained separately from the successful scoped reports.

The next performance questions are the remaining fixed phase-receipt/JSON
synchronization cost, indexed point reads, and startup overhead. Incremental
write-ahead recovery and cross-transaction shared weight/trace storage remain
separate work; the new bundles do not claim to implement them.

# Shared tensor storage

MDB can share immutable numerical files across local runs while preserving every existing path, file hash, array descriptor and StateCut fingerprint. A content-addressed pool holds one read-only object per SHA-256. Run-local files are regular hardlinks to those objects, so existing NumPy, safetensors, forecast and replay readers work unchanged.

## Configured workspace

The domains workspace is configured at `/Users/jakeholl/domains/artifacts/.mdb-storage.json`, with its pool at `/Users/jakeholl/domains/artifacts/.tensor-store`. The local common environment and the qualified FLUX environment include the updated packages. Fleet workers have not been redeployed.

Configured `saturn_pub.store.LocalStore` tensor publications and MDB `ForecastTransaction` NPY/bundle publications share their complete file bytes automatically. Local StateCut admission and forecast byte budgets continue to count logical run-local bytes; they are not global physical-storage counters. Other writers can admit a closed export with `saturn_pub.shared_store.share_configured_file(path)`, or compact a collected campaign afterward.

Configuration is inherited from the nearest ancestor `.mdb-storage.json`. An explicit `MDB_SHARED_TENSOR_STORE` environment variable overrides it, which also supports worker-local pools when intentionally configured. The pool and its clients must be on the same filesystem. Moving the workspace requires updating its explicit configuration; pool paths are not part of scientific identities.

## Commands

Use the installed CLI, or prefix it with `common exec` in this workspace:

```sh
common exec mdb storage configure --scope /path/to/artifacts --store /path/to/artifacts/.tensor-store
common exec mdb storage plan --scope /path/to/artifacts --store /path/to/artifacts/.tensor-store --pattern 'mdb-flux-*' --report storage/plan.json
common exec mdb storage compact --plan storage/plan.json --journal storage/compaction.jsonl --report storage/compaction.json
common exec mdb storage verify --plan storage/plan.json --report storage/verification.json
```

Planning only reads data. It scans matching immediate campaign directories recursively for `.npy`, `.npyb` and `.safetensors` regular files, hashes every candidate, and freezes its path, length and digest. Select only closed evidence, never a live mutable checkpoint or actively written capture.

Compaction rechecks each file against the plan, seals its permissions, admits its canonical object and atomically replaces duplicate files with hardlinks. The source path stays available throughout. A content mismatch, symlink, filesystem mismatch or changed source stops that operation. An append-only intent/completion journal records partial progress. Resume with the same plan and new journal/report filenames; already shared files are idempotent. Output reports are exclusive and are never overwritten.

Verification checks every original path, size, read-only mode and canonical inode, and hashes each unique object. No model or tensor framework is needed for these commands.

## Measured cleanup on November 5

The [frozen plan](../../../../artifacts/mdb-shared-storage-20261105/plan-v1.json) covers 5,775 numerical files across the local FLUX campaigns. Their logical lengths total 50,440,466,952 bytes. Unique file content totals 21,676,330,160 bytes across 4,302 objects.

The [compaction receipt](../../../../artifacts/mdb-shared-storage-20261105/compaction-v1.json) records 1,473 relinked files and **28,764,136,792 bytes of duplicate file payload removed**. Reported free capacity changed from 13.22 GB at planning to 41.97 GB after compaction. These global free-space observations include other filesystem activity; inode allocation counters separately report 29.07 GB released and do not resolve APFS clone sharing.

The [verification](../../../../artifacts/mdb-shared-storage-20261105/verification-v1.json) checks all 5,775 paths and 4,302 canonical objects. Original numerical bytes, field layouts, cut manifests, images, failed runs, source payloads and archive/log custody remain available. The two six-area saved-only scientific auditors were rerun into a separate [audit directory](../../../../artifacts/mdb-shared-storage-20261105/audits), without model execution; both complete JSON results equal their original reports.

Installed-wheel tests run outside the checkout with pytest's source-path injection disabled: 16 core custody/sharing tests and 74 MDB storage/forecast tests pass. They include interruption, corruption, changed-source, concurrency, cross-device and symlink controls, real BF16 StateCut hydration, legacy NumPy reads and shared forecast seals in both files/bundle formats.

## Retention and boundaries

Whole-file sharing does not merge different dtypes, remove tensor slices, rewrite BF16/FP32 representations, alter numerical contracts or make cuts interchangeable. A matching payload hash means matching bytes; execution and model provenance still live in the separate occurrence records.

Managed files are immutable. Writers must publish a new file or use atomic replacement for new content, rather than modifying a shared inode. Explicitly changing permissions and editing an object in place can affect every hardlink; normal managed publication refuses mismatched existing bytes. There is no automatic garbage collection. Removing a run leaves its canonical objects available, and removing a pool name does not invalidate surviving run-local hardlinks.

Archives and archive-bearing scheduler logs were retained in this cleanup. Avoiding that separate transport duplication needs manifest/blob publication and a deliberate custody retention format; it is not accomplished by calling compressed containers duplicate tensor files.

## Complete remote custody and small local reports

The [fresh hybrid-futures qualification](../../../../artifacts/mdb-flux-hybrid-futures-20261105/README.md) adds a near-data workflow for a laptop with insufficient free space. It is a research helper, rather than a new generic storage CLI or fleet rollout.

`qualification/collect_flux_remote_directory.py RUN POINTER.json` keeps the original-format complete archive, extracted evidence, request/payload/wheels and exact transport re-emission log on Beast. It runs the existing strict custody verifier there and checks both emitted manifest and summary against the original scheduler publication. The original execution metadata log remains separate from the re-emission log. The local pointer contains the complete member inventory, remote record hash and original job/request binding; it is explicitly not a complete local scientific collection.

`--retrieve POINTER.json MEMBER OUTPUT` authenticates the retained remote record against the staged request/job, constrains paths, streams at most the inventoried byte count, and checks SHA256 before promoting the downloaded file. Failed partials are retained separately. Fetching a member never reruns the model. Both encoder and retrieval subprocesses are terminated and reaped on failure. A small historical archive and exact binary retrieval qualify the mechanics;9installed tests include the actual public BF16 archive writer/reader and storage integrity failures.

An independent CPU mrun audit then reads the full remote collection and returns a small report/image projection with its own custody. All tensor comparisons, typed stores, receipt joins and forecast seals are checked against the complete scientific evidence on Beast. The local projection labels absent raw arrays and stores explicitly, retains their exact remote identities and provides later retrieval. Moving the audit near the evidence reduces transfer and laptop capacity requirements; it does not establish a model inference speedup.

The [MLP primitive qualification](../../../../artifacts/mdb-flux-mlp-primitives-20261106/README.md) reuses this path for254.95MBof native intermediate evidence, returning a551KBscientific projection with4.01MBcomplete local projection custody. A saved-only affine panel retains321MBof full P0/P1 predictions remotely and fetches only its68KBreport. Actual native output retrieval restores and decodes3.15MBwith no model rerun.

For an executed worker that omitted a collection-only encoder, `--encoder-source RUN` explicitly selects `flux_family_custody.py` from another request-sealed run. Its source request and SHA256 are recorded as `recovery_encoder`; it is copied outside the original executed payload. Original request and payload bytes stay unchanged, and collection still checks the original scheduler publication and full legacy archive/log/member custody. The pointer retains this extra source binding for subsequent retrieval. This is an explicit recovery path, not an implicit search for checkout helpers.

`qualification/flux_numpy_archive.py` writes finite exact BF16 widenings using the full public archive schema without importing Torch. Byte and metadata parity with the actual native writer, signed zero/subnormal preservation, fresh-process framework exclusion, and explicit-encoder integrity controls pass in the19test instrument panel. The first failed exporter attempt remains recorded. The helper is a research utility; the existing public native writer still imports Torch by design.

The [own-state integration](../../../../artifacts/mdb-flux-own-state-mlp-20261106/README.md) retains548,272,303scientific bytes remotely, including547,384,064bytes in118unique new arrays. Historical operands and original weights remain references. Separate CPU recomputation verifies all36calculated MLPs/60receiving writes and returns a1,504,373-byte scientific projection, transported in a262,384-byte archive. Array references inside `calculation-proof` explicitly address the remote calculation root. The original441,690,659-byte raw archive and exact transport log remain additional custody copies; scientific-file deduplication does not remove these containers.

The collector CLI now accepts `--catalog /absolute/remote/study-parent`, exposing its existing explicit destination option without requiring a Python wrapper. Ten installed custody tests pass; current `--help` verifies the CLI argument. Collection, metadata projection, exact-member retrieval and independent mathematical recomputation remain distinct operations.

The [fresh-receiver image study](../../../../artifacts/mdb-flux-fresh-mlp-futures-20261106/README.md) uses the same workflow for 1,093,976,349 logical scientific bytes, including 943,737,600 bytes in 150 unique arrays and a typed 40-cut consumer history. Its independent CPU replay returns 3,995,266 scientific bytes, including nine original PNGs; the compressed projection archive is 2,504,985 bytes. Full raw transport, original scratch and extracted copies remain additional retained data. An 18-latent follow-up reads the existing remote store and returns a 5,510-byte transport archive, adding no native passes or images. This separates numerical investigation from repeatedly copying complete tensors to the laptop.

Implementation: [core shared store](../../../src/saturn_pub/shared_store.py), [MDB commands](../src/mdb/storage_cli.py), [native LocalStore publication](../../../src/saturn_pub/store.py) and [forecast publication](../src/mdb/forecast_transaction.py).

## Explicit bounds for larger near-data collections

The [native consumer trace](../../../../artifacts/mdb-flux-consumer-trace-20261106/README.md) retains2,374,889,432logical scientific bytes:2,053,037,716typed blob bytes and311,435,136bytes in63 intermediate arrays, plus metadata. Its234cut histories share484pages/457blobs within the typed store.130original parameter recipes represent6,418,101,504logical NPY bytes with zero persistent parameter arrays. Separate CPU tensor/receipt/parameter verification returns a2,749,903-byte scalar/provenance projection in a410,721-byte archive.

The unchanged1536MiB compressed-archive default was too small. The failed emitter left a partial collection and spool, then the same saved output was recollected without rerunning the model. `collect_flux_remote_directory.py --limits-json POLICY.json` accepts explicit positive finite bounds, validates them through the original encoder API and records them in separate `collection-limits.json` metadata and the authenticated remote record. The original worker request, source and publication remain unchanged. The completed trace uses3072MiB archive/expanded and4608MiB log limits. Independent raw verification must pass `record.get('limits')` to the legacy verifier; default collections retain their existing behavior.

Remote collection failures now produce bounded local error records pointing to the retained partial destination. Use a new catalog for a recollection so existing partial evidence is preserved. The full tiny near-data roundtrip checks explicit encoder/parser/offline-verifier bounds, including nondefault chunk sizes;14focused consumer-trace/custody tests pass.

The completed native archive is1,896,088,381bytes and its exact compressed re-emission log is1,953,164,314bytes, in addition to extracted and original scratch copies. Full collection/verification takes206.78s. These transport and integrity costs remain separate from the177.95s native process and188.34s CPU scientific audit; neither the small local projection nor fewer native calls establishes an inference speedup.

## Inherited qualification with fresh consumed-source checks

The [J2 stream-cross assay](../../../../artifacts/mdb-flux-stream-cross-20261106/README.md) uses a smaller declared observation contract: input/J2 exit, attention midpoints, projected noise and first commit.39cuts/18arrays total328,506,393scientific bytes; the local projection is1,265,502bytes. Current full archive/log custody takes26.71s, native94.24s and CPU47.40s. These costs have a different capture scope from the full trace, without a controlled inference-speed comparison.

`qualification/flux_qualified_source.py` pins the previous full-custody record and independent audit, then freshly checks the immutable inventory entry for every consumed metadata file, cut, page, blob and array. Cut lookup traverses the complete referenced page/blob closure. The CPU assay verifies222source members/356,658,696bytes and reports the exact consumed inventory. It inherits earlier full archive/store qualification; it does not claim to rescan the old archive/log. New current evidence still receives full custody and typed-store verification. Three focused tests include same-length source corruption and referenced-blob corruption. This is a qualification helper, not a new generic storage admission API.

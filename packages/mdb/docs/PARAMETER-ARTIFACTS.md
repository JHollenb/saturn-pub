# Exact parameter artifact retrieval

MDB qualification runs retain checkpoint-derived parameter arrays alongside observed inputs, forecasts, native activations, typed cuts and receipts. Parameter copies can be redundant across histories. The preparation tool identifies exclusive parameter archives and gives each run a standalone, source-bound recipe for reproducing their **exact original bytes**. It does not remove files.

The public [parameter_artifacts module](../src/mdb/parameter_artifacts.py) uses Python's standard library. A copied `retrieve_weights.py` needs no MDB, NumPy, Torch, model loading, repository checkout or persistent cache. It reconstructs the retained NPY prefix followed by the pinned safetensors tensor payload. It does not regenerate weights or convert their dtype.

## Prepare, prove, then thin

Build/install a public MDB wheel containing `mdb.parameter_artifacts`. After a run completes authenticated collection, prepare its recipes:

```bash
MDB_PY=/path/to/installed/python
MDB_Q=/path/to/saturn-pub/packages/mdb/qualification
MDB_RUN=/path/to/completed-run
rtk proxy "$MDB_PY" -I "$MDB_Q/prepare_flux_parameter_artifacts.py" "$MDB_RUN" \
  --checkpoint-header /path/to/checkpoint-header.json \
  --checkpoint-root /mnt/big/llm-models/FLUX.2-klein-4B \
  --ssh-host beast
```

The header input records `checkpoint_root`, relative `file`, `file_bytes`, `data_start` and the raw safetensors `header` map. Each tensor supplies its native dtype, shape and data offsets. The full checkpoint file SHA comes from the collected producer's bound `source.checkpoint_files`, rather than trusting the header query alone. During retrieval the complete source file is independently hashed before its tensor ranges are used.

Preparation writes only these additive files outside the original result tree:

- `RUN/parameter-artifacts.json`: schema `mdb-parameter-artifacts-v1`, exact archive recipes, default locations and custody/source provenance.
- `RUN/retrieve_weights.py`: a byte-exact copy of the installed public module, with its source hash recorded in the manifest.

The [preparer](../qualification/prepare_flux_parameter_artifacts.py) infers the result directory from the completed source/summary/job/collection bindings. It checks producer and authenticated before/after/local metadata inventories. Explicit parameter ports map to qualified checkpoint names through native descriptors and module prefixes, including single-block and readout aliases. Every proposed archive receives a full-file SHA check; its raw BF16 payload must match the native descriptor and checkpoint tensor geometry.

Preparation also scans result JSON/JSONL metadata for non-parameter uses. A file shared with an observed activation or any other non-parameter role remains retained. Only complete parameter port records established by the bound native descriptors receive an exemption, including those within exact plain or transaction-encoded copies of their bound parameter bank. An arbitrary key named `parameters` grants no exemption. Generic inventory keys are indexes, not evidence of an activation role. Unsupported encodings are not silently converted into removal candidates.

Before thinning, reconstruct into a separate empty directory and compare the restored archive hashes/bytes against the originals:

```bash
rtk proxy python "$MDB_RUN/retrieve_weights.py" --root /path/to/empty-roundtrip
```

This uses the manifest's default authenticated SSH host, checkpoint location and remote Python. Keep the successful roundtrip receipt and an additive thinning receipt that binds the original inventory, selected recipes, removed paths and retained non-parameter evidence. The preparation and retrieval commands perform no thinning. Actual removal is a separate, explicitly authorized operation after the roundtrip proof.

Every future compatible run should carry its retrieval script and manifest before any duplicate parameter bytes are thinned. A manifest alone is not proof that the checkpoint is available or that a roundtrip succeeded.

## Retrieve with bare Python

With the script and manifest together in the run directory, defaults locate the original result tree relative to that script:

```bash
python retrieve_weights.py
```

Without overrides this streams the selected tensors over authenticated SSH from the manifest's checkpoint location. It hashes each complete checkpoint file once per invocation, verifies tensor names/dtypes/shapes/offsets, then checks raw payload and full reconstructed archive hashes. Only missing files are installed; a wrong existing file or symlink is refused. No existing evidence file is overwritten.

To use a local copy of the exact checkpoint, provide `--checkpoint`. This suppresses default SSH unless an explicit `--ssh` is also supplied:

```bash
python retrieve_weights.py --checkpoint /path/to/local/FLUX.2-klein-4B
python retrieve_weights.py --checkpoint /remote/model/root --ssh beast --root /path/to/empty-stage
python retrieve_weights.py --verify-only
```

`--root` changes the archive destination, preserving archive-relative paths. `--verify-only` verifies present files and reports missing ones without contacting the checkpoint source. The bounded SSH timeout defaults to 3600 seconds and can be set with `--timeout`. Paths, directory ancestors, file types, framing, sizes and hashes are validated; failed or partial retrieval leaves historical evidence and completed exact files available for inspection.

## Custody and historical audits

Thinning does not rewrite an original manifest or relax an old auditor. Until retrieval restores all required files, a historical full-inventory audit is expected to report missing data. Restore into the original declared artifact root, run `--verify-only`, then rerun the unchanged auditor. A successful restore must reproduce the old inventory's bytes and hashes, not a numerically similar tensor serialization.

The manifest records the original job/request, collection and producer inventory hashes, binding/source identities, checkpoint inventory, header query and retrieval/preparer sources. Each artifact records its original path/byte count/SHA, exact NPY prefix, and source checkpoint file/SHA, tensor name, dtype, shape, offsets and payload SHA. Optional alias and port metadata explains deduplicated archives.

Only exclusive checkpoint-derived parameter files are eligible. Observed state, conditioning, activations, native outputs, sealed forecasts, typed CAS payloads, receipts, images, source snapshots, inventories and accepted audit reports remain independent evidence. This storage operation makes no scientific accuracy, inference-speed or model-compression claim.

Focused [preparation tests](../tests/test_prepare_flux_parameter_artifacts.py) cover module aliases, exact framing, standalone local roundtrip, shared activation roles, corrupted archives/descriptors/custody, and unsafe paths. The public module separately tests transport, reconstruction and hostile filesystem cases.

## Measured cleanup and qualification

On November 5, 2026, the delivered standalone script retrieved all 169 unique
FLUX.2 Klein transformer parameter archives from the pinned checkpoint over
authenticated SSH. A separate verifier checked the exact NPY prefix, raw tensor
payload and complete archive hash before discarding each temporary probe file.
The complete successful invocation reconstructed 7,751,110,784 bytes in 251.03
seconds, with no dtype conversion or model execution. The staging directory is
intentionally empty afterward; the receipt proves retrieval rather than a
persistently hydrated tree.

Six authenticated run bundles then externalized 719 parameter paths totaling
34,377,461,632 logical bytes. Every retained file was checked against its
unchanged original inventory. These files already shared filesystem hardlinks,
so their summed lengths were not physical disk savings. Collecting 157 proven
parameter cache objects after their last run link disappeared removed
7,015,616,512 allocated bytes, about 7.02 GB. Twelve shared objects supporting
three older custody formats remain intact. The source checkpoint remains intact.

Each supported run has its standalone script, pinned manifest and additive
`parameter-thinning.json`. The shared cache also has a standalone script and
recipe. Presence checks actually exercised all seven delivered scripts and
reported the expected missing files. Historical full-inventory audits require
rehydration first. The preparation helper currently runs after authenticated
collection; capture-time omission of parameter exports remains a separate
integration task.

An isolated installed wheel passed 43 retrieval tests; the strict preparation
helper passed 16 tests against that wheel. The real 169-file checkpoint
roundtrip is additional evidence. During cleanup, an initial saved-ctime check
stopped after removing one run's links because changing a hardlink count changes
ctime on the surviving names. The preserved interruption was followed by fresh
full-content hashes and exact per-unlink inode/stat checks. Last-link cache
removal used private quarantine and a fresh link-count check; shared objects
were never modified in place.

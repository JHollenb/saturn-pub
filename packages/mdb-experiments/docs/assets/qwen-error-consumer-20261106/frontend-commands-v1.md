# Root-only original run commands

These commands are reviewable handoffs. This agent has run only the metadata
review and isolated synthetic frontend checks. It has not staged or submitted an
original native or saved-audit job.

Authenticate source metadata and exact count plans without staging:

```bash
rtk proxy /Users/jakeholl/.cache/mdb/qualification/flux-initial-noise-corrected-20261105/bin/python -I -B artifacts/mdb-qwen-error-consumer-20261106/review_frontend_metadata.py
```

Root can stage the native request with default intent-first sizing:

```bash
rtk proxy /Users/jakeholl/.cache/mdb/qualification/flux-initial-noise-corrected-20261105/bin/python -I -B saturn-pub/packages/mdb/qualification/stage_qwen_error_consumer.py artifacts/mdb-qwen-error-consumer-20261106/run-01 \
  --world-run artifacts/mdb-qwen-row-sign-20261106/run-02 \
  --world-compact artifacts/mdb-qwen-row-sign-20261106/compact-collection-01/extracted/binding-minimal-result \
  --world-seal artifacts/mdb-qwen-error-consumer-20261106/source-metadata-v1/world_seal.json \
  --context-run artifacts/mdb-qwen-sign-shortcut-20261106/run-02 \
  --context-compact artifacts/mdb-qwen-sign-shortcut-20261106/compact-collection-01/extracted/binding-minimal-result \
  --context-seal artifacts/mdb-qwen-sign-shortcut-20261106/frontier-metadata-v1/transaction/seal.json \
  --selection artifacts/mdb-qwen-error-consumer-20261106/selection-v1 \
  --parameter-subset artifacts/mdb-qwen-error-consumer-20261106/planning-v1/qwen-block19-parameter-artifacts.json
```

If intent-first planning lacks this component geometry, root may add the explicit
`--resource-job artifacts/mdb-qwen-row-sign-20261106/run-02/job.json` option before
staging. That records the measured component exception separately: CPU4/RAM4GiB/
VRAM2GiB/disk3GB/60s estimate/300s outer. The source job measured47.22s,
RSS1795.4MB andVRAM826MB; the new block19 runtime and peak remain unmeasured.
The two roots and checkpoint are Beast-local, so the single lease is pinned there.

Inspect the staged request before creating its exact finite review:

```bash
rtk proxy /Users/jakeholl/.cache/mdb/qualification/flux-initial-noise-corrected-20261105/bin/python -I -B saturn-pub/packages/mdb/qualification/stage_qwen_error_consumer.py artifacts/mdb-qwen-error-consumer-20261106/run-01 --inspect
```

After root writes `run-01/finite-run-review-v1.json` with the exact request hash,
`review_status: ready-for-finite-run` and an empty `blocking_issues` list:

```bash
rtk proxy /Users/jakeholl/.cache/mdb/qualification/flux-initial-noise-corrected-20261105/bin/python -I -B saturn-pub/packages/mdb/qualification/stage_qwen_error_consumer.py artifacts/mdb-qwen-error-consumer-20261106/run-01 --submit
```

Only after that exact original native study succeeds, root can stage and submit
its near-data CPU saved audit. The frontend validates the actual completed native
command/publication through the frozen controller before staging:

```bash
rtk proxy /Users/jakeholl/.cache/mdb/qualification/flux-initial-noise-corrected-20261105/bin/python -I -B saturn-pub/packages/mdb/qualification/stage_qwen_error_consumer_saved_audit.py artifacts/mdb-qwen-error-consumer-20261106/audit-01 --study artifacts/mdb-qwen-error-consumer-20261106/run-01
rtk proxy /Users/jakeholl/.cache/mdb/qualification/flux-initial-noise-corrected-20261105/bin/python -I -B saturn-pub/packages/mdb/qualification/stage_qwen_error_consumer_saved_audit.py artifacts/mdb-qwen-error-consumer-20261106/audit-01 --submit
```

Saved audit uses CPU2/RAM3GiB/VRAM0/disk1GB/240s estimate/600s outer, one lease and
no retries. It retains compact account metadata and checksum retrievers; arrays
and checkpoint payloads stay near their authenticated source roots. The saved
frontend SHA256 is
`0556094604951913293e6c5704487754420d6328a35bea258d9f77bad61c0d87`;
native frontend is
`c7574da612f8548e0574e627496e273c6ac8f81e8b04f5984fb368759d21cf28`.
Combined installed qualification passed13 focused checks in1.02s from the explicit
owned overlays and frozen HEAD698fe49 helper closure. Canonical Ruff F/I is clean.
Full pins and receipt pointers are in `frontend-qualification-v1/qualification.json`.

# Saved-only block19 consumer reader

Frozen reader `qualification/audit_qwen_error_consumer.py` SHA256
`c0adfae449b426c52c91a5cb2e05fd27453cdd704354dbd7dd465a194d19723b`.
Producer `ad82b86dd837a1f2bc9134f2c5bda3dcb097e8f8583e7849fcfbe8d5c49b229b`;
shared evaluator `c93324ee76156f2fba16f70d7cef78b7accbf6020f0c899cb04d856c348b5762`.
The exact seven `HELPERS`/`PINS` entries are declared in the reader; all are saved-only.

CLI:

```text
python -I -B audit_qwen_error_consumer.py --root RAW --output FRESH \
  --expected-manifest NATIVE_FILES_SHA --expected-request NATIVE_REQUEST_SHA
```

Resident generic saved-audit config is unchanged:
`native_root`, `native_manifest_sha256`, `native_request_sha256`, `output_scratch`.
`SATURN_DEBUG_REQUEST`, `SATURN_RUN_CONFIG`, `MRUN_JOB_ID` and empty
`CUDA_VISIBLE_DEVICES` are required. Output is `binding-minimal-result` and
`MDB_DIRECTORY_READY` with actual CPU job/request, manifest and summary hashes.
Root alone stages/submits jobs.

`audit.json` includes `worlds.{full-input,cheap-input}.fields` for five endpoint
ports; `responses` for complete cheap-minus-full response errors and magnitudes;
`input_difference`; `residual_accounts.{functional,native}` containing whole and
partition windows, input-H19/output-H20 alignment, finite-addition defects,
squared norms and all signed pairwise cross terms. It also retains authenticated
parents, source recipes/runtime, transaction and actual census/operand receipts.
The production primary EquationCatalog comparison budget is 7,340,032 elements
per world. Head width is excluded from residual-space gain/direction accounts.

Coverage counts are ten endpoint fields and five response fields, with twenty
and ten dependent window views. Two conditional catalog/atlas records are
observations for the same route; no family induction or numerical pass gate.

Compact output copies original small source/request/plan/contract/response/port
descriptors and fourteen metadata members once. All full arrays remain remote.
`retrieve_evidence.py native PATH DEST` uses `external-evidence.json` for current
raw output; `reader` uses its inherited world root. To retrieve shared context,
use `--manifest external-evidence-context.json reader PATH DEST`. This preserves
the frozen retriever's two supported aliases without inventing unsupported CLI
aliases. No activations or pretrained weight banks are copied into the compact.

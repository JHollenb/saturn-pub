# Jacobian reader/writer proof

This directory keeps two different proof levels separate:

1. Tiny seeded Qwen mechanics test exact source binding, read parity, signed and
   zero-dose writes, commit/abort, stale-source refusal, export/reopen, and offline
   verification. The identity transport and random model have no semantic meaning.
2. [`real_bf16_worker.py`](real_bf16_worker.py) fits a lens and exercises the public
   MDB reader/writer on a pretrained Qwen resident in its native CUDA BF16 lane. Its
   model observations are evidence for the named specimen, prompts, layer boundary,
   controls, and consumer.

Neither panel makes a general semantic-writer claim. The authoritative mechanics
record is [VALIDATION.md](VALIDATION.md); the API and artifact contract are in the
[Jacobian workspace guide](../../docs/JACOBIAN-WORKSPACE.md).

## Install

From `packages/mdb`, install the language-model dependencies. Add the fleet client
for the real-model run:

```sh
python -m pip install '.[autoregressive]'
python -m pip install '.[autoregressive,mrun]'
```

MDB has no runtime dependency on a Saturn, Saturn-pub, `common`, or upstream
Jacobian-lens checkout. The worker receives an explicit MDB wheel and protocol. It
fits and consumes the transport in one finite admitted job.

## Run the tiny mechanics qualification

The focused development suite is:

```sh
PYTHONPATH=src python -m pytest -q \
  tests/test_jacobian_io.py \
  tests/test_jacobian_workspace.py \
  tests/test_bench_jacobian.py \
  tests/test_bench_jacobian_verify.py
```

The worker also has a small fitted-lens mechanics test. It uses a seeded width-16
Qwen fixture, fits a non-identity BF16 transport, runs all ten native/control arms,
checks both memory probes, and verifies the export:

```sh
PYTHONPATH=src python -m pytest -q tests/test_jacobian_real_worker.py
```

For distribution qualification, build the wheel, install it into a fresh target
outside the checkout, then run `tests/qualify_installed_jacobian.py` in its capture
and reopen modes. The retained
[installed-wheel qualification](installed-wheel/final-qualified-1120c5299ffe/qualification.json)
used two distinct processes, verified the installed import root, and blocked imports
of `saturn`, `saturn_pub`, `common`, `common_harness`, and `mrun`.

These tests qualify software mechanics. The installed-wheel qualifier uses an
identity transport. The worker fixture does fit a non-identity transport, but only on
a seeded random model with a synthetic vocabulary. Neither establishes a fitted lens
on pretrained semantics or shows that a prepared edit helps a meaningful later
consumer.

## Run the real CUDA BF16 assay

The reproducible entry points are:

- [`submit_real_bf16.py`](submit_real_bf16.py): `preflight`, `submit`, and `collect`
  actions for one intent-first mrun lease;
- [`bootstrap_real_bf16.py`](bootstrap_real_bf16.py): verifies the staged payload and
  qualified wheel before loading the worker from that wheel;
- [`real_bf16_worker.py`](real_bf16_worker.py) `--protocol protocol.json`: the finite
  worker executed inside that lease;
- [`protocol.json`](protocol.json): the sealed specimen, fit texts, prompts, source
  boundary, directions, doses, controls, output root, and budgets.

The `preflight` action prints the sealed payload and intent preview. The `submit`
action then asks the mrun engine for strict preflight before admission. Retain the
returned job ID and collect that exact job rather than launching a second copy:

```sh
python experiments/jacobian-reader-writer/submit_real_bf16.py preflight \
  --measured-admission-from job-c25c2663c1ab \
  --resource-equivalent-change \
  'Clear transient tokenizer padding/truncation before sealing; identical fit/consumer geometry.'
python experiments/jacobian-reader-writer/submit_real_bf16.py submit \
  --measured-admission-from job-c25c2663c1ab \
  --resource-equivalent-change \
  'Clear transient tokenizer padding/truncation before sealing; identical fit/consumer geometry.'
python experiments/jacobian-reader-writer/submit_real_bf16.py collect --job NEW_JOB_ID
```

The launcher requests CUDA, no automatic retries, and no guessed memory reservation.
The worker keeps fitting, native parents, candidate branches, controls, export, and
verification inside the one outer lease. Do not start a duplicate full assay on the
same accelerator.

### Retained execution history

The first admitted proof attempt, `job-c25c2663c1ab`, completed all 576 declared
backward invocations and then the scheduler killed it for VRAM. The intent-first plan
reserved 3,474 MiB, its admission ceiling was 3,821.4 MiB, and measured peak use was
4,888 MiB. Its [collection receipt](results/job-c25c2663c1ab/collection-receipt.json),
[scheduler record](results/job-c25c2663c1ab/job-record.json), and
[worker log](results/job-c25c2663c1ab/job.log) are retained unchanged.

That attempt emitted no report, result bundle, reader observation, residual Act, or
native-consumer row. Its mechanics and semantic evidence are incomplete; completion
of the fitting loop is only a sizing observation. A corrected retry must use the
launcher's sealed measured-admission path against this exact failed job and retain a
new job identity. That path refuses a different worker, protocol, wheel, model,
backend, dtype, device, or workload geometry. It requests 1.25 times the observed
peak, or 6,110 MiB VRAM, while recording that a resource-killed peak remains a lower
bound. Use an unqualified `preflight`/`submit` pair only for a new workload that has
no matching retained sizing result.

The measured-admission successor `job-2cf2d1202511` also completed the fit but failed
before the first read. Hugging Face fit tokenization had left a transient backend
truncation limit of 32 in the tokenizer. The manifest sealed that state, then assay
encoding cleared it; MDB correctly rejected the changed `tokenizer_sha256`. Its
[collection receipt](results/job-2cf2d1202511/collection-receipt.json),
[scheduler record](results/job-2cf2d1202511/job-record.json), and
[worker log](results/job-2cf2d1202511/job.log) retain the instrument failure. It has
no reader, Act, or consumer evidence.

The corrected worker restores no-truncation/no-padding serving state after fit and
before manifest construction. Its regression fixture recreates transient fit settings
and requires the backend identity to stay fixed through the full assay. The
`--resource-equivalent-change` flag allows only the worker hash to differ from the
retained sizing basis, records the human reason, and preserves the protocol, wheel,
model, backend, dtype, device, and workload geometry checks. This is an instrument
repair, not evidence for a semantic effect.

The corrected successor `job-4ba3cc18d86a` reached every declared arm and memory
probe, then exceeded the outer 900-second lease before it emitted its report, bundle,
export/reopen result, or offline audit. Its
[collection receipt](results/job-4ba3cc18d86a/collection-receipt.json) therefore
records `state: timeout`, `report_present: false`, and zero collected artifacts. The
[raw log](results/job-4ba3cc18d86a/job.log) preserves these native observations:

| Arm | Cat → fox margin effect | Bear → lion margin effect |
| --- | ---: | ---: |
| J plus | 0.0000 | +0.0625 |
| J minus | -0.2500 | +0.5000 |
| J half | 0.0000 | +0.5625 |
| Raw plus | 0.0000 | +0.5625 |
| Raw minus | -0.1250 | +0.5000 |
| Norm-matched random plus | 0.0000 | +0.4375 |
| Norm-matched random minus | -0.0625 | +0.5625 |
| Complete donor | +8.6875 | +12.4375 |

The native delayed margins were `-5.6875` and `-5.2500`. Every sampled J, raw, and
random arm kept the recipient answer; the complete donor changed the answers to fox
and lion. The directions are heterogeneous and nonmonotonic across these two BF16
rows. BF16 score granularity is one plausible limitation, but this timed-out record
does not identify the cause. Memory-probe completion markers alone do not replace the
missing receipts and verifier closure. These are raw observations with
`evidence-incomplete` instrument and terminal status, not a qualified native proof.

A separate model-free CPU recovery, `job-d47b087e4124`, copied the already closed
export and fitted-lens directory from the timed-out worker without rerunning the model.
Its [collection receipt](results/recovery-job-4ba3cc18d86a/job-d47b087e4124/collection-receipt.json)
binds 22,460 recovered artifacts and the original timeout job record. An independent
installed-wheel [audit receipt](results/recovered-audit-job-4ba3cc18d86a-v2/receipt.json)
then verified 621 cuts, 14 Jacobian artifacts, 2,224 JSON artifacts, four memory
probes, eight memory writes, 304 VM bundles, and 648 unique payloads with zero
failures or misses. All four retained probes record exact reinstall closure and exact
selected-parent restoration, with sibling selections preserved. The recovered
formed-memory probe records give J-plus late-K/V effects of `0.0000` for cat→fox and
`+0.0625` for bear→lion. Complete-donor late-K/V effects were `+6.6875` and
`+10.4375`, and both donor candidates produced the donor answers. For comparison,
the corresponding whole-donor delayed effects in the raw log were `+8.6875` and
`+12.4375`. These two local probes do not show an advantage for the J direction over
the retained controls. The separately derived
[recovered analysis](results/recovered-audit-job-4ba3cc18d86a-v2/recovered-analysis.json)
records these observations, realized post-cast candidate norms, trend interpretation,
counter-trends, and limitations without changing the immutable audit receipt.

This qualifies the recovered export's byte integrity, references, sidecar/native-Act
linkage, and recorded memory-probe mechanics. A separate pure-stdlib, read-only
[future-parity audit](results/recovered-audit-job-4ba3cc18d86a-v2/future-parity.json)
verified all 36 native-versus-zero future pairs exactly: 12 cat in-run, 12 bear in-run,
and 12 bear after export/reopen. Every pair matches token IDs, pending token, logits
payload digest, VM checkpoint reference and delta digests, model identity, lane, and
numerical contract. All three zero-dose candidates are exact no-ops. The in-run and
reopened bear writes have distinct attempt and publication identities, while their
unchanged state deterministically reuses the same 12 content-addressed cuts and
endpoint. The selected close cut is exact and parents the reopened endpoint.

The parity audit loaded no model, imported no MDB runtime, and mutated no store. The
recovery did not reconstruct the missing worker report or rerun a native consumer.
The separate
[recovered mechanics qualification](results/recovered-audit-job-4ba3cc18d86a-v2/qualification.json)
records all five exact software checks as true and assigns terminal status
`bounded-native-reader-writer-mechanics-qualified-via-recovered-evidence`. The
original worker report remains incomplete and the original job remains `timeout`;
its semantic-writer status remains unqualified.

Reproduce the audit from `packages/mdb` with the pure-stdlib
[`verify_recovered_futures.py`](verify_recovered_futures.py):

```sh
python experiments/jacobian-reader-writer/verify_recovered_futures.py \
  --output /tmp/future-parity.json
```

The retained result reports `mechanics_status: valid`, `pair_count: 36`, and canonical
`result_sha256: 22803854d2a48cd955125802e304e9bc7f616cb53d913bcf06b17c851956bc1b`.

The assay fits one BF16 transport from three independent declared fit texts with the
recorded estimator, then reads and writes at source block 11 / completed layer
boundary 12. Two animal-memory contexts compare normalized signed J and raw
directions, norm-matched deterministic random controls, untouched native futures, and
a complete donor replacement as a positive reference. Each candidate runs from the
same retained parent through ordinary MDB residual Acts and the unchanged native
future consumer.

## Inspect the evidence

For a normally completed run, treat `report.json` and its referenced export/receipts
as the result, rather than the process exit code alone. For the retained timeout,
keep the scheduler status, raw observations, recovered-export receipt, and independent
audit as separate evidence records. Check at least:

- model, revision, resolved weight and tokenizer identities;
- CUDA BF16 eager lane and the lens's permitted application lane;
- fit corpus and text identities, estimator, source block, completed boundary, and
  transport shape/dtype/hash;
- common-parent cut, Act, candidate, continuation, and offline-verifier links;
- realized post-cast amplitude from `candidate.float() - source.float()`; retain the
  work-dtype recipe norm and stored BF16 delta norm as separate diagnostics;
- signed immediate readout and later-consumer effects for every dose and control;
- exact reinstall/restore observations and any per-row refusal or error;
- `mechanics_status`, `instrument_status`, `trend_status`, and `terminal_status` as
  separate fields.

A positive directional pattern is a trend for this bounded panel. A clean manifest,
successful Act, or changed readout is not evidence of a useful semantic writer by
itself. A failed or heterogeneous consumer row remains evidence and does not erase
the mechanically valid rows.

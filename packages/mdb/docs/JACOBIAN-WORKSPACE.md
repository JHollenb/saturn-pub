# Jacobian workspace: frozen reads and admitted writes

MDB exposes a bounded Jacobian workspace for reading an already fitted dense transport and preparing residual edits against it. It does not fit a lens, infer semantic roles, or replace native model state. The native residual, StateCut, K/V state, decoder, and continuation remain authoritative.

The architecture has two layers:

- `FrozenJacobianLens`, `JacobianReader`, `JacobianWriter`, and `PreparedJWrite` are standalone objects. They verify identities, calculate readouts, and prepare source-bound candidate tensors without mutating a model.
- Qwen `BenchSession` methods bind those objects to a live native boundary and admit a prepared candidate through the ordinary residual Act transaction.

## Install the model support

The base distribution intentionally has no model-library dependencies. Install the
autoregressive extra before using the reader/writer with Qwen:

```sh
python -m pip install '.[autoregressive]'
```

Add the `mrun` extra only when submitting the real-model proof through the fleet:

```sh
python -m pip install '.[autoregressive,mrun]'
```

The installed runtime imports `mdb`, Torch, Transformers, NumPy, and safetensors for
this path. It does not discover or import a Saturn, Saturn-pub, `common`, or upstream
Jacobian-lens checkout. Lens fitting is an experiment concern; the runtime consumes a
sealed fitted artifact.

## Frozen lens and standalone reader

`FrozenJacobianLens` takes the canonical manifest/matrix pair: one
`saturn-jacobian-lens-v1` mapping and one dense transport tensor. The tensor is the
matrix in `z = J @ h`; its shape, Torch dtype, and raw-byte SHA-256 must exactly match
`manifest["transport"]`. The manifest also binds the model and tokenizer identities,
fitting and application lanes, block-output coordinate, decoder, estimator, and
dictionary convention. This pair is the public input schema; a loose matrix file or a
manifest with an inferred dtype/lane is not a lens artifact.

Construction checks strict JSON, schema, model digests, coordinate map, shape, dtype,
frozen status, raw transport hash, permitted application lanes, decoder geometry,
estimator, and dictionary convention. `save()` writes that pair as `manifest.json` and
`tensors.safetensors`; `open()` verifies both files and returns a private transport
snapshot. The artifact is a transport lens, not a cloned decoder or model.

Tokenizer identity includes backend JSON and its configured truncation/padding state.
If fitting temporarily enables a maximum length or padding, restore the tokenizer to
the declared serving configuration before constructing the manifest and keep that
configuration unchanged for pin, read, write, export, and reopen. Do not weaken or
recompute the reader's tokenizer guard to accommodate fit-time state leaking into
serving; normalize the experiment boundary and seal the stable serving identity.

The reader uses caller-supplied decoder, final-normalization, and output-head tensors. Those remain caller-owned views and are hash-checked again on every read. Supplying `final_norm` and `output_head` selects the native normalized readout; the raw `W_U J h` score remains a separately labeled diagnostic.

```python
from mdb.jacobian_io import (
    FrozenJacobianLens,
    JacobianReader,
    JacobianWriter,
    PreparedJWrite,
)

lens = FrozenJacobianLens.open("lens")
reader = JacobianReader(
    lens,
    model.lm_head.weight.detach(),
    model_identity=lens.manifest["model_identity"],
    application_lane="cuda-float32-eager-tf32off",
    rms_weight=model.model.norm.weight.detach(),
    final_norm=model.model.norm,
    output_head=model.lm_head,
    execution_binding={
        "model_abi_sha256": model_abi_sha256,
        "numerical_contract": numerical_contract,
        "session_id": session_id,
    },
)

source = {
    "model_abi_sha256": model_abi_sha256,
    "numerical_contract": numerical_contract,
    "session_id": session_id,
    "cut_ref": cut_ref,
    "branch": branch,
    "token_position": token_position,
    "completed_layer_boundary": 12,
    "activation": residual_identity,
    "input_tokens_sha256": input_tokens_sha256,
    "cursor": cursor,
}

def one_token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"expected one token for {text!r}, got {ids}")
    return ids[0]

read_tokens = [
    one_token_id(tokenizer, " cat"),
    one_token_id(tokenizer, " fox"),
]
observation = reader.read(
    residual,
    source=source,
    token_subset=read_tokens,
    top_k=2,
    logical_address={"symbol": "animal", "instance": "one", "role": "remembered"},
)
```

The manifest fixes the source layer and completed-layer boundary. The source binds the exact activation bytes, model ABI, cut, branch, position, and any supplied token/cursor metadata. An explicit `execution_binding` additionally requires the numerical contract and optionally the session. Without it, standalone operation relies on the caller to maintain source custody; it does not independently prove that a named cut, cursor, or branch came from a live execution.

`logical_address` is opaque JSON metadata. MDB records it but does not infer that the residual contains the named concept or role. Likewise, a writer's `consumer={"name": ..., "horizon": ...}` records the requested evaluation target. It is not evidence that the edit works for that consumer.

## Preparing a write

`JacobianWriter.prepare_steer()` adds a signed dose of one normalized `W_U @ J` dictionary row. `prepare_span()` edits coordinates in the span of explicitly selected rows. Both accept exactly one residual token, retain the source and lens identities, record their algebra and stored delta, and return a detached `PreparedJWrite`. BF16 algebra is calculated in `torch.float32` and cast back to the native dtype; a zero dose preserves exact source bytes.

For BF16, `recipe.algebra.actual_delta_norm` is the work-dtype value before the
candidate cast. The stored BF16 `delta` is obtained by another native-dtype
subtraction and can round differently. When comparing realized intervention
amplitudes, use `candidate.float() - source.float()` as the authoritative post-cast
delta. Keep the work-dtype norm and stored-delta norm as separately labeled
diagnostics; neither is exact post-cast amplitude.

```python
writer = JacobianWriter(reader)
prepared = writer.prepare_steer(
    residual,
    source=source,
    token_id=read_tokens[1],
    dose=0.5,
    consumer={"name": "later-query", "horizon": 2},
    logical_address={"symbol": "animal", "instance": "one"},
)
prepared.save("candidate")

reopened = PreparedJWrite.open("candidate")
candidate = writer.validate(reopened, residual=residual, source=source)
```

`PreparedJWrite.save()` also uses strict JSON plus safetensors. `validate()` rejects changed source bytes, identity, lane, lens, execution binding, candidate bytes, dtype, or shape. It returns a clone; it still performs no native mutation.

## Qwen Bench transaction

Journal-mode Qwen Bench provides the execution authority:

```python
pin = bench.pin_jacobian_lens(lens_manifest, transport)
boundary = lens_manifest["coordinate_map"]["completed_layer_boundary"]
next_layer = bench.cursor()["next_layer"]
if next_layer is None or next_layer > boundary:
    raise RuntimeError(f"cannot reach lens boundary {boundary} from {bench.cursor()}")
if next_layer < boundary:
    bench.step("layer", count=boundary - next_layer)

token_cat = one_token_id(tokenizer, " cat")
token_fox = one_token_id(tokenizer, " fox")

read = bench.read_jacobian(
    pin["lens_ref"],
    token_subset=[token_cat, token_fox],
    top_k=2,
    logical_address={"symbol": "animal", "instance": "one"},
)
preview = bench.preview_jacobian_edit(
    read["observation_ref"],
    {
        "operation": "steer",
        "token_id": token_fox,
        "dose": 0.5,
        "consumer": {"name": "later-query", "horizon": 2},
    },
)

# Preview leaves the native residual and K/V unchanged.
receipt = bench.commit_edit(preview["act_id"])
# Or reject it and restore the exact parent:
# bench.abort_edit(preview["act_id"])
```

`read_jacobian()` checkpoints the exact source and records model ABI, numerical contract, session, cut, branch, cursor, token history, completed boundary, intervention lineage, and residual identity. Reading changes only sidecar bookkeeping; it does not change residual or K/V tensors. Preview verifies that the observation is still current, prepares a replacement, and stages a normal typed residual Act without installing it. `commit_edit()` admits that native tensor and creates a child cut. `abort_edit()` rejects the Act and restores its parent. Branch, cut, boundary, or residual drift makes the observation stale. Export is refused while an Act is pending; after commit or abort, lens/read/candidate references survive export and resume. `mdb.bench_verify.verify()` audits their hash and native-Act linkage.

## Compatibility and claim boundary

Lane names use Torch dtype spellings, such as `cuda-float32-eager-tf32off` and `cuda-bfloat16-eager`; `fp32` and `bf16` abbreviations are not manifest lane identities. The real Qwen2.5-1.5B J-space PoC lens is sealed only for `cuda-float32-eager-tf32off`. MDB's current native Qwen Bench executes BF16, so it correctly rejects that lens. MDB does not silently cast its transport, widen its permitted lanes, or claim an implicit refit.

The tiny BF16 Qwen tests establish identity binding, exact native read parity for an identity transport, source staleness, transactional commit/abort, export/resume, and offline verification. They do not establish a learned semantic lens or a useful memory writer. The real FP32 PoC observed modest directional J-span effects and stronger donor/complement effects; it remains exploratory and does not qualify sparse interpretation, learned role resolution, cross-lane portability, or a general semantic writer.

A fitted native-BF16 Qwen2.5-1.5B attempt later executed two reader/writer rows but
timed out before publishing its report. A model-free recovery copied its already
closed export, and an independent installed-wheel audit verified 621 cuts, 14
Jacobian artifacts, 2,224 JSON artifacts, four memory probes, eight memory writes,
304 VM bundles, and 648 unique payloads with no failures or misses. All four retained
memory probes record exact reinstall closure, selected-parent restoration, and
preserved sibling selection. In the two retained formed-memory probes, the normalized
J-plus late-K/V effects were `0.0000` and `+0.0625`; complete-donor effects were
`+6.6875` and `+10.4375`, and only the donor candidates produced the donor answers.
These local observations do not show a J-direction advantage over the controls. See the
[recovered audit receipt](../experiments/jacobian-reader-writer/results/recovered-audit-job-4ba3cc18d86a-v2/receipt.json)
and its separate
[recovered analysis](../experiments/jacobian-reader-writer/results/recovered-audit-job-4ba3cc18d86a-v2/recovered-analysis.json).
This is qualified byte/reference/software closure for the recovered export. The
separate read-only
[future-parity audit](../experiments/jacobian-reader-writer/results/recovered-audit-job-4ba3cc18d86a-v2/future-parity.json)
also verified all 36 native-versus-zero future pairs exactly: 12 cat in-run, 12 bear
in-run, and 12 bear after export/reopen. Token IDs, pending tokens, logits payload
digests, VM checkpoint references and delta digests, model identity, lane, and
numerical contract all match. The three zero-dose candidates are exact no-ops; the
two bear attempts have distinct attempt/publication identities while deterministically
reusing the same 12 content-addressed cuts and endpoint. The selected close cut is
exact and parents the reopened endpoint. This audit loaded no model and mutated no
store. The separate
[recovered mechanics qualification](../experiments/jacobian-reader-writer/results/recovered-audit-job-4ba3cc18d86a-v2/qualification.json)
records all five software checks as true and assigns
`bounded-native-reader-writer-mechanics-qualified-via-recovered-evidence`. The origin
job remains a timeout with no worker report, and this bounded mechanics qualification
does not qualify a BF16 semantic writer.

Reproduce the read-only parity analysis from `packages/mdb` with the linked
[`verify_recovered_futures.py`](../experiments/jacobian-reader-writer/verify_recovered_futures.py):

```sh
python experiments/jacobian-reader-writer/verify_recovered_futures.py \
  --output /tmp/future-parity.json
```

The retained result reports `mechanics_status: valid`, `pair_count: 36`, and canonical
`result_sha256: 22803854d2a48cd955125802e304e9bc7f616cb53d913bcf06b17c851956bc1b`.

The [experiment runbook](../experiments/jacobian-reader-writer/README.md) separates
that tiny mechanics qualification from the fitted-lens BF16 assay implemented by
[`real_bf16_worker.py`](../experiments/jacobian-reader-writer/real_bf16_worker.py).
The [validation record](../experiments/jacobian-reader-writer/VALIDATION.md)
includes 38 focused tests, including the fitted tiny-worker fixture, and the
separately installed wheel's two-process reopen/audit. Publication failures abandon
the attempted Act and restore its parent; obtain a fresh read before preparing
another edit from that resulting cut. Repeating an exact common-parent edit retains
a distinct sidecar attempt identity while preserving the native Act ID.

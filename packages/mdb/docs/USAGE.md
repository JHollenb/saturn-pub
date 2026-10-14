# MDB usage guide

For caller-owned or custom execution backends, see the
[public backend interface](BACKENDS.md): forward, capture and interventions,
plus optional native stateful debugger adapters without a scheduler.

MDB runs a model as an inspectable session: pause at a typed execution boundary,
save a cut, branch a future, inspect or edit permitted state, then replay or
restore it. Models remain native backend executors; MDB owns the coordinated
session, history and publication boundary. Start with a finite script, then try
the [experiment walkthroughs](EXPERIMENTS.md).

MDB supports the extracted Qwen2/Qwen2.5, Mamba v1, FLUX and SDXL adapters.
Resident capability catalogs are authoritative: compatibility depends on the
actual model or pipeline components, checkpoint and numerical lane. Mamba2 and
hybrid architectures are unsupported. Native and virtual Qwen lanes have separate
numerical contracts; cross-lane restoration requires a qualified conversion.

## Getting started

The local console works after a base install, without a job or model libraries:

```sh
python -m pip install .
mdb debug
mdb doctor
mdb help
```

This opens the pure-Python Counter demo. With `.[autoregressive]`, use
`mdb debug --tiny qwen2` for random CPU BF16 weights, or
`mdb open --local /path/to/hf-checkpoint --root ./session` for a supported local
checkpoint. Both use the same debugger commands and compact frame display;
`inspect --full` or `inspect --json` exposes the complete response.
The [local debugger walkthrough](LOCAL-DEBUGGER.md) covers native breakpoints,
finite continuation, edits, export and resume. Counter cuts are in-memory toy
state; native session exports retain the existing independent verification path.

For managed fleet execution:

From `packages/mdb` in the Saturn repository, in an environment with the desired
extras installed:

```sh
python -m pip install '.[all]'
mdb version
mdb capabilities
```

The `all` extra includes the public mrun-pub client pinned over HTTPS.
Configure mrun's scheduler and credentials
through its normal configuration. The [installation guide](INSTALL.md) covers
smaller installs and the qualified worker environment. An independent installed
MDB environment needs no common workspace or sibling checkout.

To learn without a model download or fleet connection, install the
`autoregressive` extra and run:

```sh
python -I "$PWD/examples/resident_lab.py" --tiny qwen2 --out ./results/first-lab
```

This tiny random model demonstrates replay, intervention, sibling memory and
circuit-panel mechanics. Its `report.json` and independently verifiable `export/`
are the outputs. It is not a semantic model experiment.

For real models, use mrun admission. The built-in launcher's current execution
device is CUDA; it requests CUDA admission and stages one resident worker.
`--dry-run` prints a sealed plan without submitting it. The Python resident API
uses the model or pipeline already loaded by the caller and creates no lease.

| Registry model | Adapter | Qualified numerical lane | First experiment |
|---|---|---|---|
| `qwen2.5-0.5b-instruct` | Qwen2 | BF16/CUDA; virtual and native qualified separately | Layer rewind and token replay |
| `mamba-130m` | Mamba v1 | FP32/CUDA native slow path | Recurrent-state siblings and memory-mediated response |
| `flux2-klein-4b-distilled` | FLUX | BF16/CUDA | Denoiser block cut and exact image replay |
| `illustrious-xl` | SDXL | FP16/CUDA | UNet block and scheduler closure replay |

See the [qualification record](QUALIFICATION.md) for measured results and exact
boundaries. A package capability inventory lists available code; use
`session.capabilities()` for the loaded resident's actual stages and writes.

For a fitted Jacobian lens, use Qwen `pin_jacobian_lens`, `read_jacobian`, and
`preview_jacobian_edit`, followed by ordinary residual `commit_edit` or
`abort_edit`. The [Jacobian reader/writer guide](JACOBIAN-WORKSPACE.md) covers
exact source binding, safe artifact reopen, compatible numerical lanes, and
the difference between an explicit tensor edit and a measured semantic writer.

## Execution and history

A finite CLI script uses the same methods as Python. Paths passed to `open` and
`resume` refer to the selected worker's durable filesystem:

```sh
mdb open --model qwen2.5-0.5b-instruct --prompt 'The capital of France is' \
  --root /absolute/worker/path/qwen --commands examples/qwen-commands.json
mdb resume --model qwen2.5-0.5b-instruct --root /absolute/worker/path/qwen \
  --commands examples/resume-commands.json
mdb open --model flux2-klein-4b-distilled --prompt 'A red cube on a white table.' \
  --root /absolute/worker/path/image --seed 17 --height 512 --width 512 --steps 4 \
  --commands examples/diffusion-commands.json
```

For Mamba use `--model mamba-130m`; for SDXL use `--model illustrious-xl`.
The pipeline components select the actual diffusion family. `open` without a
command script creates a bounded live-attach window; it requires an mrun worker
advertising the existing debugger credential protocol.

For a caller-owned language model, run this inside an already admitted worker:

```python
from mdb import BenchSession
from mdb.bench_loading import load_bench_model

spec, model, tokenizer = load_bench_model("qwen2.5-0.5b-instruct", device="cuda")
with BenchSession.from_resident(model, tokenizer, root="./qwen-session",
        model_id=spec.model_id, revision=spec.revision,
        continuation_capacity=16) as session:
    parent = session.prefill("The capital of France is")
    session.step("layer")
    cut = session.checkpoint("first-layer")
    print(session.inspect("residual"))
    rows = session.generate(4)
    print(tokenizer.decode([row["token_id"] for row in rows]))
    session.replay(cut, max_new_tokens=4)
    session.export("./qwen-export")
```

`load_bench_model` resolves locally materialized weights through installed mrun.
It loads a resident model; admission belongs to the enclosing worker. You can
instead pass your own compatible model and tokenizer directly. Use a new root
for `from_resident`; use `resume` for an existing session.

Caller-owned diffusion pipelines use the same session boundary:

```python
from mdb import BenchSession

with BenchSession.from_resident(pipeline, root="./image-session") as session:
    parent = session.prepare("A red cube on a white table.", seed=17,
                             height=512, width=512, num_inference_steps=4)
    session.step("block")
    cut = session.checkpoint("first-block")
    print(session.inspect("memory"))
    session.generate()
    image = session.resolve_value("image")
    session.restore(cut)
    session.generate()
    session.export("./image-export")
```

Autoregressive stages include embedding, individual layers, readout, sampling and
commit. A partial cut retains pending-token state and staged KV or recurrent state.
Token history is published after a successful transaction. Diffusion stages include
conditioning, internal blocks, denoising, scheduler, VAE and commit. Diffusion cuts
retain the actual scheduler state, timetable, RNG state and transient block closure.

Use `checkpoint`, `fork`, `checkout`, `restore`, `replay` and `export` on the same
session. Exports include retained cuts and their payload closure; keep the original
model weights independently available. `BenchSession.resume(model, tokenizer,
root=...)` validates identity before continuing. `mdb verify --root ...` checks the
offline closure without loading model weights.

Byte verification reads payloads from disk and returns a manifest-bound aggregate
of verified descriptors, payload digests, bytes and integrity checks. It avoids
per-read durable receipt writes; a failure persists an invalid aggregate under
`byte-audits/` before refusing the closure. Model-consumer hydration continues to
publish its normal durable receipts. Preserve the successful verification output
with your job report or custody record.

Older Mamba v1 exports using family-manifest schema v2 may omit the numerical environment. Default resume refuses
that incomplete contract. An independently qualified caller can supply
`legacy_numerical_environment=expected_environment` to `BenchSession.resume`;
the resident environment must match every field exactly. The old identity and
cut addresses remain intact, and the enclosing manifest retains the explicit
qualification for later resumes. The CLI accepts the same JSON through
`--legacy-numerical-environment`. A historical file-based checkpoint digest also
requires the caller's authoritative `model_sha256` in Python; this is distinct
from the launcher's parameter-based identity. Byte verification alone does not
qualify that legacy numerical contract.

## Memory and addresses

For causal inspection and intervention on Qwen committed K/V rows, use
`memory_read(spec)`, `memory_write(spec)`, `memory_diff(spec)`, and
`memory_probe(spec)`. The corresponding debugger commands are
`memory read|write|diff|probe SPEC|@FILE`. Selectors name native layers, a
half-open token-row span, and K/V components. Reads include decoded token text,
addresses, tensor summaries, and bounded numeric contents. Writes atomically
copy, zero, scale, or replace the selection through Model OS. A probe runs the
same fixed query through untouched, candidate, and exact-reinstall native
futures and restores the selected parent before returning. See
[Reading and writing hidden memory](HIDDEN-MEMORY.md) for the complete loaded-model
example, exact JSON schemas, budgets, outputs, and evidence boundary.

Memory inspection reports logical rows, physical sharing, private ownership,
resident/backed allocations, capacity, transfers and eviction/hydration events.
Model weights and workspace are distinct from virtual state. Copy-on-write and
durable backing allow siblings to share immutable ancestors while retaining private
continuations. Retained checkpoints consume storage; durability has measurable
overhead. Pages are logical tensor storage, not integrated CUDA virtual memory.

Spectral addresses identify `(symbol_id, scene_id, instance_id, role_id, frame_id)`.
They identify model execution objects and source context, not raw device pointers.
Tensor pages and checkpoint references provide the physical payload links.

| Output | What it gives you |
|---|---|
| `inspect("status")` | Session ID, active branch, execution cursor, current cut and capabilities |
| `inspect("residual")` | Available tensor shape, dtype, device where reported, byte digest and summary statistics |
| `inspect("memory")` | Family-specific logical views, allocations, sharing, backing and residency events |
| `inspect("frames")` | Source scopes, selected spectral bindings and their links to execution cuts |
| `inspect("symbols")` | Structural names and context-bound candidate/measured bindings |
| `inspect("measurements")` | Retained circuit reports, CAS references and current context compatibility |
| `inspect("performance")` | Instrumented stage timings, publication and payload/page counters |
| `checkpoint()` / `export()` | Durable executable cuts / a portable retained-state closure |

Inspect family capabilities before reading or writing an address. A residual is
available during the relevant layer/block execution, and the final image is
available only after native VAE and commit. The allocation table links payload
digests, tensor geometry and owners; a spectral address names the logical source.

## Fact editing (MEMIT, AlphaEdit)

`mdb.memit` is the default editing method. It ports MEMIT (Meng et al., ICLR 2023) and
AlphaEdit (Fang et al., ICLR 2025) with EasyEdit's Qwen2.5 hyperparameters, and applies the
weight update as a removable `down_proj` overlay. AlphaEdit projects each update onto the
null space of the preserved-key statistics C0, so preserved keys are left approximately
unchanged, and it accounts for earlier batches when editing sequentially.

```python
from mdb.memit import (AlphaEditHyperParams, MEMITHyperParams, alphaedit,
                       layer_second_moments, memit)

C0, n = layer_second_moments(model, tok, wiki_texts, [4, 5, 6, 7, 8], n_tokens=1_000_000)
reqs = [{"prompt": "The capital of {} is", "subject": "France", "target_new": " Tokyo"}]
patch, diag = memit(model, tok, reqs, C0, MEMITHyperParams())
# patch, diag = alphaedit(model, tok, reqs, C0, AlphaEditHyperParams())
# sequential: patch2, _ = alphaedit(model, tok, more_reqs, C0, hp, base=patch)
with patch.install(model):   # exact undo on exit; bench: bench.mlp_memory_install(patch)
    ...
```

Measured on Qwen2.5-1.5B, capital edits. Paraphrases are T1 "X's capital city is" /
T2 "The capital city of X is" / T3 "Q: What is the capital of X?":

| Method | Edits | Efficacy | Paraphrases T1 / T2 / T3 | Locality | Same-subject other relations |
|---|---:|---:|---|---:|---:|
| MEMIT | 12 | 12/12 | 5 / 12 / 12 | 0.92 | 0.50-1.00 (mean 0.65) |
| MEMIT | 44 | 44/44 | 25 / 44 / 44 | 0.94 | 0.36-1.00 |
| AlphaEdit | 12 | 12/12 | 5 / 12 / 12 | 0.91 | 0.08-1.00 (mean 0.50) |
| AlphaEdit | 44 | 44/44 | 30 / 44 / 44 | 0.62 | 0.16-1.00 (mean 0.42) |
| MEMIT, 4 sequential batches of 11 | 44 | 39/44 | 20 / 38 / 36 | 0.75 | 0.34-1.00 (mean 0.53) |
| AlphaEdit, 4 sequential batches of 11 | 44 | 44/44 | 26 / 44 / 44 | 0.88 | 0.25-1.00 (mean 0.49) |

Use MEMIT for a single batch. Use AlphaEdit when edits arrive in batches over time: that is
where its null-space projection measurably helps. At the default threshold, single-batch
AlphaEdit lost locality on Qwen2.5-1.5B.

Neither method protects the edited subject's other facts (language, currency). Check them
explicitly, or use the relation-gated `MLPMemory` below. The full guide (algorithms,
hyperparameters, deviations from the reference code, sequential editing, bench integration,
limits) is [Fact editing](FACT-EDITING.md).

## MLP associative memory

`mdb.mlp_memory.MLPMemory` adds a removable linear readout to the MLP output at chosen
decoder layers. It never writes model parameters.

- **Edits.** An edit names a prompt position and a target. The target is either a donor
  prompt's native MLP output there, or explicit per-layer vectors.
- **Preservation prompts.** These ask every row to keep its native output.
- **Fitting.** Layers are fitted in order (weighted ridge on normalized MLP inputs). Each
  layer's keys are captured with the lower layers' readouts installed.
- **Incremental writes.** `insert` and `delete` update a float64 inverse Gram matrix exactly
  (Sherman-Morrison-Woodbury) without a refit. `save` and `load` use safetensors plus a
  content-digested manifest.

```python
from mdb.mlp_memory import MLPMemory

edits = [{"id": "france", "ids": france_ids, "position": subject_last,
          "donor_ids": japan_ids, "donor_position": japan_subject_last}]
memory = MLPMemory.fit(model, edits, preserve=other_prompt_ids,
                       layers=[0, 1, 2, 3, 4, 19], alpha=1e-4, edit_weight=1.0)
with memory.install(model):      # hooks; leaving the block is exact undo
    ...
memory.insert(model, more_edits)  # native model required (no overlay installed)
memory.delete("france")
```

In a Qwen bench session, `bench.mlp_memory_install(memory)` and `bench.mlp_memory_remove()`
are lineage Acts:

- Cuts taken while an overlay is live record its digest, and a cut can only be restored under
  the overlay it was cut under.
- History committed before the install keeps its native K/V rows.
- Removal witnesses bit-identical native probe logits.
- `resume` reinstalls a persisted overlay, and `close` removes its hooks from the caller's model.

Qualified on Qwen2.5-1.5B for 12 country-to-capital edits at layers 0-4 and 19:
- 11/12 efficacy, 24/36 paraphrase generalization and 1.00 locality;
- exact undo;
- insert and delete within 4e-8 of a batch fit;
- bench and HF logits identical under BF16.

How to make a fact edit, not a subject swap (measured, Qwen2.5-1.5B, 12 edits):

- **A donor target is a subject swap.** It changed all 12 edited countries' official-language
  answers too.
- **Don't add zero rows for the edited subject's other relations under a linear key.** Store
  keys are about 97% subject + relation, so this is an XOR problem. The fit solves it only
  through the small interaction residue, at 4-10x gain, and held-out prompts collapse
  (locality 0.36).
- **Make the value relation-specific instead.** Optimize per-layer deltas with
  `attribute_delta(model, ids, pos, new_token, keep=[same subject, other relations],
  also=[other phrasings of this relation], layers=...)`, then fit them as `deltas` edits with a
  linear key. Result: 12/12 edits, 23/24 paraphrases, locality 0.97, other relations 0.75-1.00.
- **Conjunctive keys** stop the collapse. `fit_spectral_readers` trains calibrated subject and
  relation posteriors on several wordings per relation and binds them as spectral addresses
  (`phase(symbol) + phase(role)`, magnitude = confidence). With a donor target: 11/12 edits,
  21/24 paraphrases (T2+T3), locality 1.00, other relations 1.00. The older
  `fit_address_readers` phasor keys (one wording per relation) bind the prompt wording
  instead ("capital city of" is not "capital of").

Validate every edit on held-out prompts.

## Debugger, symbols and circuits

The debugger adapter uses the session's cut registry. Layer/token or block/step
breakpoints, inspection, typed residual edits, preview, commit/abort and checkpoint
navigation use the backend capability catalog. Some memory surfaces are read-only;
formed-state interventions are separately validated and consumed by native execution.
SourceFrameStack provides a logical scoped source stack alongside cuts. It is not a
native CPU/CUDA machine stack.

Hyperion distinguishes structural labels, candidate discoveries and measured
bindings. Compatible measured artifacts are identity-bound; changed contexts can
invalidate bindings. Bounded discovery runs actual same-parent interventions and
retains signed effects and controls. Qwen and Mamba circuit measurement/replay are
available where the resident catalog advertises them. Diffusion timesteps are not
given autoregressive token semantics. Candidate evidence does not establish a
universal semantic circuit or a complete explanation of reasoning.

`mdb attach JOB_ID` speaks the existing mrun debugger protocol. The corresponding
Saturn surface is live-qualified through the deployed broker on one
Qwen2.5-1.5B CUDA BF16 job, including read, atomic write, diff, three unique
fixed-query probes, exact reinstall, parent restoration, export and offline
verification. See the [hidden-memory guide](HIDDEN-MEMORY.md) for the retained
transcript, client correction and repeated-panel plan variance. The MDB package
does not itself deploy or administer the fleet service.

## Offline tools

`mdb cache`, `hyperion`, `context`, `bisect`, `xref`, `claims`, `watch`, `custody`,
`cite` and `evidence` expose the extracted evidence and memory tools. Use each
command's `--help` for its exact arguments. Writable default indexes live beneath
`MDB_HOME` (default `~/.local/share/mdb`), outside installed package directories.
Watch fires are diagnostics, not semantic verdicts. Retain raw receipts and add
interpretation separately.

Typed Thott context-register operations require the autoregressive extra for
tensor-format checks. Bench export byte verification and the evidence index tools
work with the dependency-free core.

Current operating boundary: single writer, serialized execution, local durable
backing. No production continuous batching, automatic semantic discovery for every
prompt, compact model replacement, or integrated CUDA VMM is claimed.

For step-by-step experiments and expected outputs, continue to
[MDB experiment walkthroughs](EXPERIMENTS.md).

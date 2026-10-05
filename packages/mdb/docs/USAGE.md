# MDB usage guide

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

From the MDB repository, in an environment with the desired extras installed:

```sh
python -m pip install '.[all]'
mdb version
mdb capabilities
```

The `all` extra includes the pinned installed mrun client; GitHub SSH access is
needed for its private dependencies. Configure mrun's scheduler and credentials
through its normal configuration. The [installation guide](INSTALL.md) covers
smaller installs and the qualified worker environment. In the domains workspace,
use `common exec` for shared credentials, with the installed MDB interpreter when
checking checkout independence.

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

`mdb attach JOB_ID` speaks the existing mrun debugger protocol. In-process protocol
mechanics and deployed broker qualification are separate. This extraction includes
no fleet refresh; live deployment proof remains pending.

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

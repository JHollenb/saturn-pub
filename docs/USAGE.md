# Sessions, memory and debugging

MDB supports the extracted Qwen2/Qwen2.5, Mamba v1, FLUX and SDXL adapters.
Resident capability catalogs are authoritative: compatibility depends on the
actual model or pipeline components, checkpoint and numerical lane. Mamba2 and
hybrid architectures are unsupported. Native and virtual Qwen lanes have separate
numerical contracts; cross-lane restoration requires a qualified conversion.

## Execution and history

A finite CLI script uses the same methods as Python. Paths passed to `open` and
`resume` refer to the selected worker's durable filesystem:

```sh
mdb open --model qwen2.5-0.5b-instruct --prompt 'The capital of France is' \
  --root /absolute/worker/path/qwen --commands examples/qwen-commands.json
mdb resume --model qwen2.5-0.5b-instruct --root /absolute/worker/path/qwen \
  --commands examples/qwen-commands.json
mdb open --model flux2-klein-4b-distilled --prompt 'A red cube on a white table.' \
  --root /absolute/worker/path/image --seed 17 --height 512 --width 512 --steps 4 \
  --commands examples/diffusion-commands.json
```

For Mamba use `--model mamba-130m`; for SDXL use `--model illustrious-xl`.
The pipeline components select the actual diffusion family. `open` without a
command script creates a bounded live-attach window; it requires an mrun worker
advertising the existing debugger credential protocol.

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

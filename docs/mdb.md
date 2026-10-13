# MDB: durable model debugging

MDB is the repository's `mdb-runtime` distribution, imported as `mdb` and built
from `packages/mdb`. It coordinates a native model's execution, retained cuts,
branches, permitted edits and durable backing. Its model adapters expose typed
layer/token or block/denoising boundaries.

Start with [installation](../packages/mdb/docs/INSTALL.md), then the
[usage guide](../packages/mdb/docs/USAGE.md) or
[experiment walkthroughs](../packages/mdb/docs/EXPERIMENTS.md).
The [documentation index](README.md) covers both packages.

## When to choose MDB

| Job | Why MDB fits | Walkthrough |
| --- | --- | --- |
| Save an in-flight model session and continue in another process | Durable checkpoints and portable exports bind state, model identity, history and execution contract | [Execution and history](../packages/mdb/docs/USAGE.md#execution-and-history) |
| Compare several futures while inspecting their storage | Copy-on-write branches, immutable ancestor sharing, local backing, residency accounting, eviction and hydration | [Memory and addresses](../packages/mdb/docs/USAGE.md#memory-and-addresses) |
| Debug a token or denoiser failure at a native boundary | Breakpoints, scoped source frames, inspection and preview/abort/commit of supported writes | [Debugger, symbols and circuits](../packages/mdb/docs/USAGE.md#debugger-symbols-and-circuits) |
| Rewind a layer, perturb a residual and retain the comparison | Same-parent native continuations, circuit measurements, receipts and independently verifiable exports | [Experiment walkthroughs](../packages/mdb/docs/EXPERIMENTS.md) |
| Read a frozen J-space lens and test an explicit residual write | Source-bound lens observations and candidates lower through native preview, commit and abort; later memory and output are measured separately | [Jacobian workspace](../packages/mdb/docs/JACOBIAN-WORKSPACE.md) |
| Stage a finite model job with reproducible runtime custody | Optional mrun integration seals installed wheels, declared files and configuration into an admitted worker | [Installation and worker environment](../packages/mdb/docs/INSTALL.md) |
| Audit an exported run without loading the model | Dependency-free byte verification and evidence/index tools | [Offline tools](../packages/mdb/docs/USAGE.md#offline-tools) |

Choose [`saturn_pub`](usage.md) for its broader decoder-family adapters,
SAELens/TransformerLens and circuit-tracer integrations, causal panels, measured
programs or reversible training. Its block-streamed residency moves frozen
weights one block at a time; MDB's managed paging concerns execution state and
branch ownership. Those mechanisms solve different memory problems.

Both base packages have no required dependencies. The extra libraries and
retained state determine the cost of a model workflow; the package name alone
does not establish a memory or speed advantage. Their APIs, cuts and numerical
contracts remain separate. See the [repository guide](monorepo.md).

## Install and try it offline

From the repository root, install either the offline tools or language-model support:

```sh
python -m pip install ./packages/mdb
python -m pip install './packages/mdb[autoregressive]'
```

| Install | Enables | Dependencies |
| --- | --- | --- |
| Base package | CLI metadata, export byte verification and evidence tools | No required model libraries |
| `autoregressive` | Caller-owned supported language models and the tiny CPU lab | Public model libraries |
| `diffusion` | Supported caller-owned FLUX/SDXL pipelines | Public model libraries plus pinned mrun-pub |
| `all` | Model extras and scheduler launches | Public dependencies and scheduler configuration for launches |

Use [MDB's installation guide](../packages/mdb/docs/INSTALL.md) for exact pins
and the qualified worker environment. Separate environments preserve each
package's model qualification lane.

With the autoregressive extra installed, run this from the repository root:

```sh
mdb version
mdb capabilities
python -I "$PWD/packages/mdb/examples/resident_lab.py" --tiny qwen2 --out ./results/mdb-first-lab
mdb verify --root ./results/mdb-first-lab/export
```

Choose a new output directory for each run. This random CPU model needs no
download or scheduler. It demonstrates rewind, residual edits, sibling memory
and circuit-panel mechanics; its vocabulary IDs do not have trained semantic meaning.

## Wrap a caller-owned model

With a compatible native `model` and `tokenizer` already loaded:

```python
from mdb import BenchSession

session = BenchSession.from_resident(model, tokenizer, root="./session")
session.prefill("Explain gravity.")
session.step("layer")
parent = session.checkpoint("after-layer")
session.fork("candidate")
print(session.inspect("memory"))
edit = session.preview_edit("residual", 1.0, mode="scale")
session.abort_edit(edit["act_id"])
session.restore(parent)
session.generate(8)
session.export("./portable-export")
session.close()
```

This wraps the resident model without creating another lease. For real-model
experiments in the domains workspace, load and run it inside an admitted mrun
worker. The [usage guide](../packages/mdb/docs/USAGE.md) covers finite CLI scripts,
image preparation, resume, address capabilities and explicit legacy conversion.

## Qualification and boundaries

MDB has bounded model-backed records for Qwen2/Qwen2.5, Mamba v1, FLUX and SDXL.
Read the [qualification record](../packages/mdb/docs/QUALIFICATION.md) for the
exact checkpoints, numerical lanes, native comparisons and remaining work.
The [E6/E20 paper adapters](../packages/mdb/experiments/time_circuits/README.md)
have separate research-input closures; they do not expand the interactive
BenchSession family catalog.

Restoring a cut requires compatible weights and execution identity. Native and
virtual Qwen lanes have distinct contracts. Logical tensor paging does not
establish integrated CUDA VMM or production continuous batching, and deployed
fleet debugger attachment remains unqualified. Existing qualification receipts
retain their recorded source identities; repository co-location adds no new
model/backend certification.

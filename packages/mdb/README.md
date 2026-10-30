# MDB

This is the `mdb-runtime` package in the Saturn repository, at `packages/mdb`.
Run the commands below from this directory. From the repository root, use
`pip install ./packages/mdb`. The import name remains `mdb`; installing this
package declares the public `saturn-pub` core as a dependency. See the [MDB overview](../../docs/mdb.md),
[documentation index](../../docs/README.md) and [repository guide](../../docs/monorepo.md).

Standalone model virtual memory and reversible model debugger, extracted from
Saturn's Open Bench. MDB may reuse public `saturn_pub` contracts, adapters and APIs
through declared package dependencies. Private Saturn and common workspace imports
remain forbidden. The distribution depends on the public Saturn-pub core.
Optional `mrun` integration provides model resolution, resource admission and resident jobs.

For a Saturn-pub Session, use `BenchSession.from_saturn_pub(session, project=Project(path))`.
`mdb public --tiny qwen3 --project ./investigation` opens its decoder in the same
console. Public symbols, backtraces and watches join durable graphs and annotations
in one project; see [public-session projects](docs/SATURN-PUB-PROJECTS.md).

`mdb circuits` browses trace-backed circuits, seals branch plans and predictions,
and retains verified native outcomes. Explicit Python backends execute the plans
through the public replay APIs; supplied role labels remain distinct from native
dataflow evidence. See the [circuit explorer](docs/CIRCUIT-EXPLORER.md).

`mdb discovery` adds generic computation names, MLP/math/update stepping,
breakpoints, resident scalar lanes, branches and bounded action scripts over
native ATen replay. See the [discovery debugger walkthrough](docs/DISCOVERY-DEBUGGER.md)
and [executed R3 script](examples/discovery/post1.mdb).

Start the local GDB-style console after a base install:

```sh
pip install .
mdb debug                      # pure-Python Counter; no server or model libraries
mdb doctor                     # local installation check
mdb help                       # debugger command groups before any session
```

With `.[autoregressive]`, `mdb debug --tiny qwen2` opens a random CPU BF16
model. `mdb open --local /path/to/hf-checkpoint --root ./session` opens a
supported local checkpoint in the same console. Start with the
[local debugger walkthrough](docs/LOCAL-DEBUGGER.md) for breakpoints, compact
frames, edits, export and resume.

For caller-owned execution, `BenchSession.from_backend` accepts custom runners
or existing debugger adapters. Built-in Transformers and TransformerLens
wrappers expose forward passes, activation capture and scoped interventions
without mrun. See the [backend guide](docs/BACKENDS.md).

`TransformerLensDebuggerAdapter` adds stateful CPU layer debugging to the same
runner: retained residual cuts, staged attention/MLP hook edits, forks and durable
export/resume. Try `mdb debug --tiny transformerlens` with `.[transformerlens]`.
See its [scope and walkthrough](docs/TRANSFORMERLENS-DEBUGGER.md).

Committed Qwen K/V memory is directly readable and writable through `memory_read`,
`memory_write`, `memory_diff`, and `memory_probe`. Select native layers, token rows,
and K/V components; inspect bounded numeric values and token text; apply atomic
copy/zero/scale/replace Acts; and compare paired native futures with exact parent
restore. Start with the [hidden-memory walkthrough](docs/HIDDEN-MEMORY.md).

Frozen Jacobian lenses can read exact residual cuts and prepare signed atom or
selected-span edits through `pin_jacobian_lens`, `read_jacobian`, and
`preview_jacobian_edit`. Writes use the existing residual Act commit/abort path.
See the [Jacobian reader/writer guide](docs/JACOBIAN-WORKSPACE.md) for source
binding, numerical-lane compatibility, persistence, and the semantic claim boundary.

Factual associations can be edited with MEMIT and AlphaEdit (`mdb.memit`), applied as
removable `down_proj` overlays that install into a bench session as lineage Acts. Removal is
exact. See [fact editing](docs/FACT-EDITING.md).

**Extraction status:** installed-wheel mechanics and bounded model-backed proofs
pass for Qwen, Mamba v1, FLUX and SDXL. Independent archive checks, custody and
evidence sweeps accompany the proofs. See the [qualification record](docs/QUALIFICATION.md)
for exact checkpoints, numerical lanes and limits. The corresponding Saturn
hidden-memory CLI is live-qualified through the deployed broker on one
Qwen2.5-1.5B CUDA BF16 job; this package does not deploy the fleet service.

Install the offline tools with `pip install .`. For caller-owned language models,
install `.[autoregressive]`; for diffusion, `.[diffusion]`. Scheduler launches also
require the installed `mrun` package and its configured scheduler credentials.

```sh
mdb capabilities
mdb inspect --root /path/to/export
mdb verify --root /path/to/export
mdb open --model qwen2.5-0.5b-instruct --prompt 'Explain gravity.' \
  --root /absolute/worker/path/session --dry-run
```

`open --model` and `resume --model` submit one resident worker through mrun.
The `--local` and `--tiny` paths create no lease. A caller-owned model
uses no additional lease:

```python
from mdb import BenchSession

session = BenchSession.from_resident(model, tokenizer, root="./session")
parent = session.prefill("Explain gravity.")
session.step("layer")
layer_cut = session.checkpoint("after-layer")
session.fork("candidate")
print(session.inspect("memory"))
candidate = session.preview_edit("residual", 1.0, mode="scale")
session.abort_edit(candidate["act_id"])
session.restore(layer_cut)
session.generate(8)
session.export("./portable-export")
session.close()
```

Diffusion pipelines use `prepare(prompt, seed=..., height=..., width=...,
num_inference_steps=...)` and `generate()` over the prepared scheduler timetable.
Restore requires the same qualified model identity and numerical contract.

See [installation](docs/INSTALL.md), [usage and boundaries](docs/USAGE.md) and
[extraction provenance](PROVENANCE.json).

Try the [experiment walkthroughs](docs/EXPERIMENTS.md) for layer rewind/replay,
residual preview/abort/commit, sibling memory, circuit measurement and diffusion
block replay. A tiny CPU demonstration requires no downloaded model or fleet:

```sh
python -I "$PWD/examples/resident_lab.py" --tiny qwen2 --out ./results/first-lab
```

The [E6/E20 paper assays](experiments/time_circuits/README.md) run the frozen
native protocols through bounded MDB research sessions, with private tensor
edits, durable input cuts, native comparison, replay and custody. These research
adapters have a narrower closure than the interactive BenchSession families.

## Compatibility

Legacy schema names, content-addressed identifiers and debugger protocol markers
retain their Saturn spelling. These are serialized ABI identifiers; they do not
require Saturn code. Tensor state stays backend-owned. MDB coordinates checkpoints,
branches, publication and Model OS transactions around one session identity.

The extraction tools in `tools/` are maintainer tools that read an upstream
checkout. Installed runtime code and sealed worker wheels do not use those tools.

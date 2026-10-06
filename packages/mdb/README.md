# MDB

Standalone model virtual memory and reversible model debugger, extracted from
Saturn's Open Bench. MDB owns the runtime source. It does not import Saturn or
Saturn-pub. `mrun` provides model resolution, resource admission and resident jobs.

Committed Qwen K/V memory is directly readable and writable through `memory_read`,
`memory_write`, `memory_diff`, and `memory_probe`. Select native layers, token rows,
and K/V components; inspect bounded numeric values and token text; apply atomic
copy/zero/scale/replace Acts; and compare paired native futures with exact parent
restore. Start with the [hidden-memory walkthrough](docs/HIDDEN-MEMORY.md).

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

`open` and `resume` submit one resident worker through mrun. A caller-owned model
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

## Compatibility

Legacy schema names, content-addressed identifiers and debugger protocol markers
retain their Saturn spelling. These are serialized ABI identifiers; they do not
require Saturn code. Tensor state stays backend-owned. MDB coordinates checkpoints,
branches, publication and Model OS transactions around one session identity.

The extraction tools in `tools/` are maintainer tools that read an upstream
checkout. Installed runtime code and sealed worker wheels do not use those tools.

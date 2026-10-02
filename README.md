# MDB

Standalone model virtual memory and reversible model debugger, extracted from
Saturn's Open Bench. MDB owns the runtime source. It does not import Saturn or
Saturn-pub. `mrun` provides model resolution, resource admission and resident jobs.

**Extraction status:** installed-wheel mechanics pass; model-backed qualification
is in progress. Upstream results do not certify this distribution. See the
[qualification record](docs/QUALIFICATION.md). No fleet service has been deployed.

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

## Compatibility

Legacy schema names, content-addressed identifiers and debugger protocol markers
retain their Saturn spelling. These are serialized ABI identifiers; they do not
require Saturn code. Tensor state stays backend-owned. MDB coordinates checkpoints,
branches, publication and Model OS transactions around one session identity.

The extraction tools in `tools/` are maintainer tools that read an upstream
checkout. Installed runtime code and sealed worker wheels do not use those tools.

# Saturn

**Run models as inspectable, branchable software.**

Saturn wraps native model execution with typed state, declared interventions, reproducible
continuations, and reversible training. A researcher can stop at a layer or denoising step,
change one carrier, run the real remaining model, compare futures, and commit or restore.

```text
capture → fork → inspect → intervene → native continuation → compare → commit / restore
```

This is the standalone `saturn-pub` toolkit. It has no private repositories, fleet services,
credentials, model weights, or historical experiment code. The import name is `saturn_pub`.

## Install

Python 3.10+, CPU supported. From this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[ar,diffusion,train,dev]'
```

With uv: `uv sync --extra ar --extra diffusion --extra train --extra dev`.
The checkout selects Python 3.12 for uv; the package also supports Python 3.10+.
The checked-in lockfile pins a development environment; `pip` installs the public dependencies.
Install only `.[ar]`, `.[diffusion]`, or `.[train]` when you need one lane. Bare installation
supports metadata inspection and JSON-only custom adapters without importing PyTorch.

## One complete transaction

```python
from saturn_pub import Act
from saturn_pub.adapters.qwen import QwenAdapter

adapter = QwenAdapter.tiny()  # offline, random native Qwen2; no download
session = adapter.session([5, 7, 11])
session.continue_()  # token embedding boundary
session.continue_()  # native decoder layer 0
parent = session.capture()
candidate = session.fork(parent)
candidate.apply(Act.zero("hidden"))
adapter.generate(candidate, tokens=2)
native = session.fork(parent)
adapter.generate(native, tokens=2)
comparison = native.compare(candidate)
session.commit(candidate)  # explicit decision; no automatic scientific verdict
session.restore(parent)  # verified restoration; retained futures stay intact
```

Steps are adapter micro-transitions; `adapter.generate` is an AR token-budget helper.
Captured state includes model identity, execution contract, and all declared continuation state.
Exactness means a named state boundary and numerical program, not agreement across all kernels.

## Run offline examples

```bash
python examples/ar_debug.py
python examples/debugger_replay.py
python examples/diffusion_debug.py
python examples/causal_panel.py
python examples/measured_program.py
python examples/dependency_reuse.py
python examples/reversible_training.py
python examples/shared_memory.py
saturn-pub debug --family ar
```

The tiny examples demonstrate mechanics on random native models and supplied synthetic tasks.
They do not demonstrate pretrained semantic capabilities. AR and diffusion share the session
grammar; their adapters own different physical state. Images and receipts go to `outputs/`.

`debugger_replay.py` records a readable layer-by-layer debugger walkthrough, compares two
futures, commits and restores, and verifies both futures from a saved cut in a fresh process.
Read `outputs/debugger-replay/walkthrough.md` after running it. The random diffusion images
are 8×8 mechanics fixtures; use the pretrained recipe for meaningful generated images.

## Research tools

[Hands-on notebooks](notebooks/README.md) show GDB-style model stepping, boundary
breakpoints, state inspection, intervention, rewind, and fresh-process replay.
[Model OS and memory scope](docs/model-os-memory.md) maps what is included and what
remains outside this release.

Run the [complete writer experiment](experiments/flux2_writer/README.md), including
construction, native-image feedback, rollback, and fresh-process replay, without a scheduler:

```bash
python experiments/flux2_writer/run.py
```

It loads pinned Hugging Face checkpoints or your local cache. The
[experiment catalog](experiments/README.md) includes the measured reference results.

| Tool | Purpose |
| --- | --- |
| Session, Frame, Act, StateCut, Receipt | Native execution, declared changes, branches, custody |
| Local debugger | Step, inspect, read, capture, fork, intervene, continue, compare, restore |
| FLUX.2 Klein adapter | Step native joint/single blocks, write text/image carriers, replay saved suffixes |
| Investigation | Same-parent causal panels with per-arm evidence and errors |
| Program | Replay measured interventions within explicit recipient/context support |
| DependencyGraph | Declared dirty closure and exact instance-local reuse |
| TrainingTransaction | Inspect candidate updates; promote or restore the registered closure |
| Experimental memory | Immutable shared Qwen ancestry, private tails, modular phase transport |

Mimas-inspired investigation and Hyperion-inspired programs are small public mechanisms,
not copies of the entire private research workbench. The native downstream consumer remains
the behavioral authority. A failed exploratory score does not erase a measured trend.

## Guides

Start with [when and how to use each capability](docs/usage.md), then choose a
[runnable example](examples/README.md) or [notebook](notebooks/README.md).

- [Architecture and scope](docs/architecture.md)
- [Debugger](docs/debugging.md)
- [Experiments and programs](docs/experiments.md)
- [Reversible training](docs/training.md)
- [Write an adapter](docs/adapters.md)
- [Pretrained model recipes](docs/pretrained.md)
- [Build a writer, debug its futures, and replay](docs/writer-demo.md)
- [Capabilities and numerical contracts](docs/capabilities.md)
- [Release validation](docs/validation.md)
- [Extraction provenance](PROVENANCE.md)

## Development

```bash
pytest -q
ruff check .
python -m build
```

Apache 2.0. Model weights and upstream libraries retain their own licenses.

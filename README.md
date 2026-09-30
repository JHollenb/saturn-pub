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

## Supported models

`saturn_pub.adapters.load(path_or_model)` dispatches by `model_type`. Autoregressive decoder
families run through `adapters/decoder.py` (Qwen2 also through the dedicated `adapters/qwen.py`);
Mamba-1 runs through `adapters/mamba.py`. Every adapter uses eager attention, batch one, and greedy
decoding, and refuses an out-of-rule config by naming the rule. Real-weight rows below were
measured fp32, `tf32` off; each reproduced the native 16-token greedy decode exactly and replayed a
mid-layer cut exactly in a fresh interpreter. Full numbers are in
[`experiments/family_validation/results.json`](experiments/family_validation/results.json).

| Family | Adapter | Real-weight checkpoints (max abs logit Δ vs native) | Parity | Limits |
| --- | --- | --- | --- | --- |
| gpt2 | Decoder | gpt2 — MacBook CPU (1.8e-4) | 16/16 greedy exact, fresh-process replay exact | learned absolute positions; gelu_new; eager |
| gpt_neox (Pythia) | Decoder | pythia-70m/160m/410m — `job-beeea1a95e3c` (1.6e-5 / 1.7e-6 / 3.8e-6) | 16/16 greedy exact, replay exact | MHA, parallel residual, partial rotary; untied |
| llama | Decoder | SmolLM2-360M — `job-d75df96810fe` (2.2e-5) | 16/16 greedy exact, replay exact | `pretraining_tp==1`; default RoPE |
| qwen2 | Qwen / Decoder | Qwen2.5-0.5B — `job-d75df96810fe` (1.8e-5) | 16/16 greedy exact, replay exact | full attention only, no sliding window |
| qwen3 | Decoder | Qwen3-0.6B — `job-d75df96810fe` (1.4e-5) | 16/16 greedy exact, replay exact | attention qk-norm required |
| mamba (Mamba-1) | Mamba | mamba-130m/370m-hf — `job-d75df96810fe` (1.0e-4 / 9.4e-5) | 16/16 greedy exact, replay exact; single-token zero-state step bit-exact | no attention/positions; numeric (not bit-exact) parity |
| phi | Decoder | tiny fixture only | tiny-fixture parity | no cached checkpoint available |
| mistral | Decoder | tiny fixture only | tiny-fixture parity | only Mistral-7B cached (exceeds the 4080 VRAM guardrail); accepted only while context <= `sliding_window` |
| mixtral | Decoder | tiny fixture only | bounded MoE parity | no cached checkpoint available |
| gemma (Gemma-1) | Decoder | tiny fixture only | tiny-fixture parity | no cached checkpoint available |

Real-weight runs used cached Beast (RTX 4080) checkpoints via one batched mrun job plus a
supplementary pythia lease; gpt2 was measured on the MacBook CPU because it is not cached on Beast.
phi, gemma, mixtral, and mistral have no usable cached checkpoint (mistral only as a 7B that exceeds
the VRAM budget), so they are exercised by the tiny-fixture suite only. See
[family validation](experiments/family_validation/README.md).

## Run offline examples

```bash
python examples/ar_debug.py
python examples/debugger_replay.py
python examples/software_debugger.py
python examples/debugging_repair.py
python examples/diffusion_debug.py
python examples/causal_panel.py
python examples/measured_program.py
python examples/dependency_reuse.py
python examples/reversible_training.py
python examples/structural_rewind.py
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

`software_debugger.py` is the broad no-download proof: a parameter-free graph machine exposes
typed execution points and ports, fires branch-local watches at retained cuts, patches one live
memory cell while preserving the rest, compares native/candidate consumers, compiles measured
replay, emits a cut/receipt-linked causal path, deduplicates state pages, and restores a
data-bearing ReplayBundle into a fresh store. `structural_rewind.py` separately snapshots and
reopens graph/rules/memory/cursor futures through an empty-model, optimizer-free training boundary.

`debugging_repair.py` follows an early deletion through a supplied redundant-writer neural
circuit, observes native repair, suppresses that repair, measures collateral, and replays both
suffixes in fresh processes. [Symbols and causal paths](docs/debug-symbols-paths.md) explains
qualified symbols, operation backtraces, temporal schedules, and recovery-window comparisons.

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
| Observer and WatchEvent | Branch-local change/threshold/nonfinite watches with retained safe-point cuts |
| CausalPath and PortObservation | Model/parent/clock-validated route and effect evidence graph |
| SymbolBinding and SymbolTable | Context/clock-qualified read-only debug symbols and sealed sidecars |
| Trajectory and PathSchedule | Bounded temporal assays, first recorded divergence, observable recovery |
| FLUX.2 Klein adapter | Step native joint/single blocks, write text/image carriers, replay saved suffixes |
| Investigation | Same-parent causal panels with per-arm evidence and errors |
| Program | Replay measured interventions within explicit recipient/context support |
| ReplayBundle | Seal a measured program, local cut graph, receipts, environment and adapter provenance |
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
- [Symbols, provenance, and debugging repair](docs/debug-symbols-paths.md)
- [Experiments and programs](docs/experiments.md)
- [Reversible training](docs/training.md)
- [Write an adapter](docs/adapters.md)
- [SAELens and TransformerLens interop](docs/interop.md)
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

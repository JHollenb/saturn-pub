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
Add `.[interop]` for the SAELens / TransformerLens example.

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
| gpt2 | Decoder | gpt2 — macOS arm64 CPU (1.8e-4) | 16/16 greedy exact, fresh-process replay exact | learned absolute positions; gelu_new; eager |
| gpt_neox (Pythia) | Decoder | pythia-70m/160m/410m — `job-beeea1a95e3c` (1.6e-5 / 1.7e-6 / 3.8e-6) | 16/16 greedy exact, replay exact | MHA, parallel residual, partial rotary; untied |
| llama | Decoder | SmolLM2-360M — `job-d75df96810fe` (2.2e-5) | 16/16 greedy exact, replay exact | `pretraining_tp==1`; default RoPE |
| qwen2 | Qwen / Decoder | Qwen2.5-0.5B — `job-d75df96810fe` (1.8e-5) | 16/16 greedy exact, replay exact | full attention only, no sliding window |
| qwen3 | Decoder | Qwen3-0.6B — `job-d75df96810fe` (1.4e-5) | 16/16 greedy exact, replay exact | attention qk-norm required |
| mamba (Mamba-1) | Mamba | mamba-130m/370m-hf — `job-d75df96810fe` (1.0e-4 / 9.4e-5) | 16/16 greedy exact, replay exact; single-token zero-state step bit-exact | no attention/positions; numeric (not bit-exact) parity |
| phi | Decoder | tiny fixture only | tiny-fixture parity | no cached checkpoint available |
| mistral | Decoder | tiny fixture only | tiny-fixture parity | only Mistral-7B cached (exceeds the 4080 VRAM guardrail); accepted only while context <= `sliding_window` |
| mixtral | Decoder | tiny fixture only | bounded MoE parity | no cached checkpoint available |
| gemma (Gemma-1) | Decoder | tiny fixture only | tiny-fixture parity | no cached checkpoint available |
| gemma2 (Gemma-2) | Decoder | gemma-2-2b — `job-9d26c9585494` (3.8e-5) | 16/16 greedy exact, replay exact | soft-capped attention/logits; alternating local/global sliding-window (context ≤ `sliding_window`); GeGLU; pre+post norms; tied, scaled embeddings |

Real-weight runs used cached checkpoints on an RTX 4080 (16 GB) host via one batched job plus a
supplementary pythia job; gpt2 was measured on macOS arm64 CPU because it was not cached on that host.
phi, gemma, mixtral, and mistral have no usable cached checkpoint (mistral only as a 7B that exceeds
the VRAM budget), so they are exercised by the tiny-fixture suite only. See
[family validation](experiments/family_validation/README.md).

### Block-streamed residency (debug a model larger than the card)

Every decoder family, the dedicated Qwen adapter, and Mamba take `residency="streamed"` (via
`load(path, residency="streamed", device="cuda")` or an adapter's `from_pretrained`/`tiny`). Frozen
weights stay in host memory and one native block -- embedding, each decoder/mixer layer, final
norm, lm_head -- is copied to the device at a time, so a 7B+ LM can be stepped and debugged layer
by layer on a 16 GB card. The key/value (Mamba conv/recurrent) cache stays resident on the
execution device; only frozen weights stream. The session grammar is unchanged
(capture/fork/Act/compare/commit/restore and fresh-process replay). Resident execution keeps the
historical zero-overhead path and byte-identical receipts; streamed execution adds a `residency`
field to the contract. Because `Tensor.to` is a byte-preserving move and `functional_call`
substitutes without mutating the module, **streamed output equals resident bitwise on the same
device wherever the model fits resident**. Rows measured fp32/bf16, `tf32` off, on an RTX 4080
(16 GB); exactness checks the 16-token greedy decode and a mid-layer fresh-process replay. Full
numbers and the plain-Python runner are in
[`experiments/lm_block_residency/`](experiments/lm_block_residency/README.md).

| Model | Adapter | Fits resident? | Streamed == native | Fresh-process replay | Peak VRAM (resident / streamed) | Job |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B (fp32) | Decoder (qwen2) | yes | bitwise equal to resident (0.0 logit Δ), 16/16 greedy exact | exact | 1.90 GB / 0.53 GB (3.5x) | `job-16a9286db5ab` |
| Qwen3-8B (bf16) | Decoder (qwen3) | no (15.26 GB weights vs 15.54 GB usable) | 16/16 greedy exact vs accelerate CPU-offload reference | exact | — / 1.21 GB | `job-6afb6e14cc9a` |

The 8 B weights (15.26 GB) leave no room for the CUDA context, activations, and key/value cache on
the 15.54 GB usable card, so resident inference is infeasible; streamed, its decode peaks at
**1.21 GB**. For a model that does not fit resident the reference is HF Transformers `generate` with
accelerate `device_map` CPU-offload (eager, same checkpoint and dtype, greedy) and the check is
exact equality of the generated token ids. Streaming trades speed for footprint: the per-layer
Python stepping over an external cache is already slower than a fused forward, and the host->device
copy adds more (0.5 tok/s for the streamed 8 B here), so it is a debugging/inspection path, not a
serving path. Streamed-vs-resident bitwise equality is proven on tiny fixtures for all families
(`tests/test_residency.py`) and on the resident-fitting real model above.

Diffusion adapters step the native FLUX transformer one projection / joint block / single block /
readout at a time with the native flow Euler update and VAE. A transformer that does not fit the
card runs **block-streamed**: weights stay in host memory and one native block is copied to the
device at a time. Rows below were measured bf16, 512×512, 4 steps on an RTX 4080 (16 GB); "exact"
means every per-step latent digest is equal. Reports and thumbnails are in
[`experiments/diffusion_family_validation/results/`](experiments/diffusion_family_validation/results/README.md).

| Family | Adapter | Real-weight checkpoint | Stepped = native | Streamed = resident | Fresh-process replay | Streamed peak VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| FLUX.2 Klein-4B | `Flux2KleinAdapter` | `job-ece5315bc64b` | exact | exact | exact | 0.65 GB |
| FLUX.2 Klein-9B | `Flux2KleinAdapter` | `job-6be0bc01cc95` | exact | — (does not fit resident) | exact | 1.07 GB |
| FLUX.1-schnell | `Flux1Adapter` | `job-69b0e456925b` | exact | — (does not fit resident) | exact | 0.81 GB |
| FLUX.1-dev | `Flux1Adapter` | tiny fixture only | tiny-fixture parity | — | — | no real checkpoint available |
| FLUX.2 Klein 9B-KV | — | refused with a reason | — | — | — | reference-KV trajectory ABI |

Text-only, batch one; no reference images, CFG, or LoRA. See [FLUX diffusion families](docs/diffusion-families.md).

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
python examples/instrument_trial.py
saturn-pub debug --family ar
saturn-pub-trial-verify
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

`instrument_trial.py` re-derives the six shipped Instrument Trial case verdicts from a
hash-pinned receipt bundle (stdlib, no torch), then arbitrates the same distributed-store
ablation on the supplied redundant-writer circuit under two declared readings: a naive
single-site patching reading inverts against the native consumer, and a consumer-gated
reading agrees. See [the trial guide](docs/trial.md).

## Research tools

[Hands-on notebooks](notebooks/README.md) show GDB-style model stepping, boundary
breakpoints, state inspection, intervention, rewind, and fresh-process replay.
[Runtime and memory scope](docs/runtime-memory-scope.md) maps what is included and what
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
| FLUX.2 Klein and FLUX.1 adapters | Step native joint/single blocks resident or block-streamed, write text/image carriers, replay saved suffixes |
| Model loader | `adapters.load()` dispatches decoder, Mamba-1, and Qwen checkpoints by `model_type`; `residency="streamed"` steps a model larger than the device one native block at a time |
| SAELens / TransformerLens interop | Find a feature with their hooks, intervene on Saturn's aligned carrier, replay the receipt |
| circuit-tracer interop (`saturn_pub.interop.circuit_tracer`) | Take an attribution graph, re-run each feature edge (and whole-group ablate / −2× steer) as a native intervention on the real weights at the transcoder's write carrier, label native-necessity and compare it to circuit-tracer's own predicted drop (agree / invert) under a frozen rule, report error nodes as uncovered, seal an offline-re-derivable verdict table |
| Investigation | Same-parent causal panels with per-arm evidence and errors |
| Instrument Trial (`saturn_pub.trial`) | Arbitrate a standard-instrument reading (patching/probe/cosine/SAE) against native continuation into an agree/invert/inconclusive row with a frozen decision rule; offline `verify_bundle` re-derives the six case verdicts from a hash-pinned bundle |
| Induction certificate (`saturn_pub.certificate`) | Freeze a candidate direct-source attention circuit, control arms, held-out panels and thresholds in advance; supply measured outcomes and get a signed (content-hashed) certificate or a named failing gate. Stdlib-only gate; a torch measurement helper runs the relational-induction causal battery on a native decoder adapter; registers as an evidence claim |
| Program | Replay measured interventions within explicit recipient/context support |
| ReplayBundle | Seal a measured program, local cut graph, receipts, environment and adapter provenance |
| DependencyGraph | Declared dirty closure and exact instance-local reuse |
| TrainingTransaction | Inspect candidate updates; promote or restore the registered closure |
| Experimental memory | Immutable shared Qwen ancestry, private tails, modular phase transport |
| Evidence plane (`saturn_pub.evidence`) | Stdlib-only, torch-free: first-divergence bisect over a cut lattice, hash-chained claim registry with drift audit, SQLite cross-reference index, and index-ready citations |
| Metadata-delta route (`saturn_pub.route`) | Payload-free catalog of candidate deltas: select by tag before hydrating any bytes, resolve the chosen delta through a caller-owned content-verified resolver (deduplicated, byte-accounted), then apply, replay vs native, and roll back exactly |

Investigation and Program are small public mechanisms,
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
- [circuit-tracer: arbitrate attribution-graph edges against the native consumer](docs/circuit-tracer.md)
- [FLUX diffusion families and block-streamed residency](docs/diffusion-families.md)
- [Pretrained model recipes](docs/pretrained.md)
- [Build a writer, debug its futures, and replay](docs/writer-demo.md)
- [Instrument Trial: arbitrate a reading against the native consumer](docs/trial.md)
- [Induction certificate: preregister, measure, and sign a bounded circuit claim](docs/certificate.md)
- [Capabilities and numerical contracts](docs/capabilities.md)
- [The evidence plane: bisect, claims, xref, cite](docs/evidence.md)
- [The metadata-delta route: select a delta before hydrating, apply, roll back](docs/route.md)
- [Release validation](docs/validation.md)
- [Extraction provenance](PROVENANCE.md)

## Development

```bash
pytest -q
ruff check .
python -m build
```

Apache 2.0. Model weights and upstream libraries retain their own licenses.

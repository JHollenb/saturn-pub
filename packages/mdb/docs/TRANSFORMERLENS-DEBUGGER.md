# Stateful TransformerLens debugging

MDB's abstract `ModelRunner` already wraps TransformerLens forward hooks. The
`TransformerLensDebuggerAdapter` adds retained layer execution to the same
runner/session API, using MDB's existing family runtime for tensor pages,
Model OS edit transactions, durable publication, branches and offline verification.

## Start the console

From the repository root:

```sh
python -m pip install './packages/mdb[transformerlens]'
mdb debug --tiny transformerlens --root ./lens-session
```

This is a deterministic, random two-layer CPU FP32 model with input IDs
`[1,2,3]`. Try:

```text
checkpoint input
break layer 1
continue
inspect residual
checkpoint parent
fork candidate
set hooks.blocks.1.mlp.hook_out '{"op":"suppress"}'
preview
commit
inspect hooks
continue
inspect logits
restore parent
set residual 0.9 scale
abort
export ./lens-export
quit
```

`step` executes one decoder layer; `step embedding` retains the input residual
before layer zero. `continue` finishes the current sequence, checking breaks
and watches at each actual boundary. The final layer step also computes native
final normalization and logits. No token is sampled or appended. The frame's
`tokens` count is the supplied sequence length.

```sh
mdb resume --tiny transformerlens --root ./lens-session
mdb verify --root ./lens-export
```

The root retains cuts, named checkpoints, branches, committed/rejected Acts and
source frames. Reverse-step and reverse-continue use the retained parent chain.
`--max-steps` bounds native transitions per motion command. The default model
seed is 113; resuming with another seed refuses the changed identity.

## Bring a caller-owned bridge

Use a local TransformerLens 4.2 bridge in eval mode, then construct the stateful
adapter. The adapter owns its MDB root; the caller retains ownership of the bridge:

```python
from mdb import BenchSession, TransformerLensDebuggerAdapter

bridge.eval()
with TransformerLensDebuggerAdapter(
    bridge, tokens, attention_mask=mask, root="./session", model_id="my-checkpoint"
) as adapter:
    with BenchSession.from_backend(
        "transformerlens", model=bridge,
        debugger_adapter=adapter, identity=adapter.identity,
    ) as session:
        debug = session.debugger()   # local, paused, ready; no scheduler
        debug.step()
        parent = debug.checkpoint("parent")["cut_id"]
        debug.fork("candidate")
        debug.set_value("hooks.blocks.1.hook_in", {"op": "scale", "value": 0.9})
        debug.commit()
        debug.continue_run()
        logits = session.resolve_value("logits")
        debug.replay(parent)
        debug.call("export", {"destination": "./export"})
```

Choose hook names from `session.capabilities()["intervention_points"]`; do not
assume aliases from another TransformerLens version. Durable `hooks.<name>`
edits accept `suppress`, `set`, `add`, or `scale` specifications, applied through
TransformerLens's actual forward hooks when that layer executes. Shape, dtype,
device and finite-value checks run against the native activation. Hook specs
must be JSON-serializable, target a supported decoder-block hook, and be staged
before its layer passes. Python callbacks remain available on ordinary
`session.forward` calls, but are outside the durable debugger closure.

Residual edits use the existing preview/commit/abort path. Preview leaves live
execution state untouched. Committed hook specs and residual values survive
forks and export; abort retains its rejection receipt and restores the parent.
Failed execution restores the prior acknowledged cut. The runner still exposes
ordinary full forwards, captures and scoped interventions beside the debugger.
Do not mutate the caller-owned model through another session or external code.

Reopen with compatible weights/configuration and no new inputs:

```python
with TransformerLensDebuggerAdapter(bridge, root="./export", resume=True) as adapter:
    with BenchSession.from_backend(
        "transformerlens", model=bridge,
        debugger_adapter=adapter, identity=adapter.identity,
    ) as session:
        session.debugger().continue_run()
```

## State and qualification boundary

The retained state includes token IDs, attention mask, CPU Torch RNG state,
residual, logits, layer/phase cursor and declarative hook edits. Model identity
binds parameters and buffers, configuration, compatibility mode, hook catalog,
adapter/bridge execution sources, library versions and the numerical environment.
Weights are not copied into the export. Changed weights/buffers, training mode,
numerical settings, configuration or persistent Python hooks refuse continued
execution. A second writer cannot open the same root.

The numerical contract is `transformerlens-cpu-cachefree-layer-reentry-v1`.
Current admitted architectures are native TransformerLens models, HF GPT-2 and
HF Qwen2/Qwen2.5 bridges. They require a local standard decoder block stack,
one sequence, CPU tensors and eval mode. Tiny mechanics cover uninterrupted versus
stepped logits, residual and attention/MLP hook edits, abort, fork isolation,
fresh-process resume, bounded motion, history and corrupt-byte refusal. These
checks add no real-checkpoint or GPU qualification to the existing record.

TransformerLens `start_at_layer` on an HF bridge may internally recompute earlier
blocks before injecting the retained residual. Stopping also clears incomplete
internal KV cache. MDB therefore makes no efficient incremental-decode claim:
KV caching, token generation, GPU lanes, other architectures, training/autograd,
arbitrary interior-hook pause/resume and remote drivers need separate closures.

`mdb verify` checks exported bytes and reference closure without Torch,
TransformerLens, weights or mrun. It does not execute a model or certify semantic
intervention utility. See [backend registration](BACKENDS.md),
[the local console guide](LOCAL-DEBUGGER.md) and [qualification](QUALIFICATION.md).

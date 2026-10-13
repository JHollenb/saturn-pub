# Caller-owned model backends

`BenchSession.from_backend` accepts a runner instance, an existing
`DebuggerAdapter`, or a registered factory name. It does not acquire a model,
contact a scheduler, or import Saturn. The base interface is independent of
PyTorch and mrun. Existing `from_resident` and `resume` family sessions retain
their original state schemas and numerical contracts.

## Forward, capture and intervention

Load a model with your chosen framework, then wrap it:

```python
from mdb import BenchSession

# model is caller-loaded; inputs is its Transformers input dictionary.
with BenchSession.from_backend(
    "transformers", model=model,
    identity={"model_id": "my-checkpoint", "revision": "my-pinned-revision"},
) as session:
    point = "module:model.embed_tokens:output"
    result = session.forward(inputs, capture=[point], interventions={
        point: {"op": "scale", "value": 0.9},
    })
    logits = result.output.logits
    changed_embedding = result.captured[point]
```

Choose addresses from `session.capabilities()["capture_points"]`. PyTorch
module addresses use `module:<named_modules path>:output`, including `<root>`
for the root. Tensor outputs and the first tensor in tuple outputs support
edits; incompatible outputs are refused when their hook fires. Captures are
detached snapshots after the intervention; native outputs retain autograd.
Temporary hooks are removed on success or failure, leaving caller hooks intact.

Declarative edits are `{"op": "suppress"}`, and `set`, `add` or `scale` with a
`value`. Unknown fields are refused. A callback receives one native value and
returns a replacement; `None` retains its changes to the private copy. Scope
local edits inside that callback. Built-in tensor edits preserve shape, dtype,
device and finite values. Other runners own their native value contracts.

One session serializes runner and debugger calls. Use one session per mutable
executor: external calls or additional sessions over that executor are the
caller's responsibility. Forward adapters do not roll back arbitrary model side
effects, such as changes to a caller-supplied mutable KV cache.

## TransformerLens and SAELens

From the repository root, install MDB's optional integration environment with:

```sh
python -m pip install './packages/mdb[transformerlens]'
```

This extra pins TransformerLens 4.2.0 independently of Saturn-pub's 3.9.0
interop environment. Construct your own local bridge or model, then:

```python
session = BenchSession.from_backend(
    "transformerlens", model=bridge,
    identity={"model_id": "my-checkpoint", "revision": "my-pinned-revision"},
)
point = "blocks.0.hook_in"  # Choose from this bridge's capability catalog.
result = session.forward(tokens, capture=[point], interventions={
    point: {"op": "suppress"},
})
session.close()
```

The adapter accepts a local v4 `TransformerBridge` or legacy v3
`HookedTransformer` hook surface. A `HookedSAETransformer` can use the same
surface when the caller has attached its SAEs; the caller retains control over
SAE reconstruction settings and TransformerLens compatibility mode.

Local adapters use `run_with_hooks` for actual forward edits. Declarative-only
v4 remote drivers use `run_with_cache(intervene=...)`: their observation
callbacks cannot modify computation. MDB translates `scale.value` to TL's
`scale.factor`, excludes non-fireable hooks, and refuses Python callbacks before
dispatch. Local tiny-model mechanics and driver lowering have tests; serving
performance, real remote GPU execution and SAE semantics are separate claims.

## Register your own runner

Subclass `ModelRunner`, or supply an object with the same `capabilities()` and
`forward()` methods. This interface imports no model framework:

```python
from mdb import ModelRunner, RunnerCapabilities, ForwardResult, register_backend

class MyRunner(ModelRunner):
    def __init__(self, executor):
        self.executor = executor

    def capabilities(self):
        return RunnerCapabilities(
            backend="my-engine", capture_points=("residual",),
            intervention_points=("residual",), intervention_specs=True,
        )

    def forward(self, inputs, *, capture=(), interventions=None, **kwargs):
        capture, interventions = self.capabilities().validate(capture, interventions)
        output, values = self.executor.run(inputs, capture=capture, edits=interventions)
        return ForwardResult(output=output, captured=values)

register_backend("my-engine", MyRunner)
session = BenchSession.from_backend(
    "my-engine", executor=my_executor, identity={"model_id": "my-model"},
)
```

Your executor must actually apply each declared edit and report every requested
capture. Factories are process-local and explicit; duplicate and built-in names
are protected. `available_backends()` lists factory names, not installed
dependencies. Named factory-created runners are session-owned. Instances remain
caller-owned unless `owns_backend=True`; their `close()` releases only resources
they own. Built-in wrappers leave caller-owned models and bridges open.
Identity is caller-declared, not independently verified weight/tokenizer custody.

## Stateful debugging through existing adapters

Use MDB's existing `DebuggerAdapter` ABC for native stepping, checkpoints,
restore, transactional Acts and replay. Supply it directly or alongside a runner:

```python
session = BenchSession.from_backend(
    runner, debugger_adapter=native_debugger_adapter,
    identity={"model_id": "my-model"},
)
debugger = session.debugger()  # Existing DebugSession, in-memory transport.
parent = debugger.checkpoint("parent")
debugger.pause()
session.safe_point("layer", 3)  # The runtime calls this at its actual safe point.
debugger.inspect("residual")
```

The executor publishes real safe points and owns the complete state required by
its advertised operations. Adapter capabilities preserve its existing schema,
addresses and exact-replay declaration. The transactional `ModelBackend`
protocol remains the seam for `ModelOSKernel` users behind a debugger adapter.

For an existing resident MDB session, supply
`BenchDebuggerAdapter(resident_session)`. Its original cut registry stays
authoritative. Durable export and resume remain operations of that original
session; this registration never converts its cuts or changes its numerical ABI.

Forward-only runners advertise no stateful operations and refuse checkpoint,
restore and debugger requests. Activation capture is not a complete KV,
recurrent-state, scheduler or RNG checkpoint. Cross-backend replay and general
durable export require separately implemented and qualified native state support.

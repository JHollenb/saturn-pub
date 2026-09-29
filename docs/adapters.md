# Add your own model

Implement Adapter's four methods:

```python
class MyAdapter(Adapter):
    model_identity = ...  # actual checkpoint/config fingerprint
    execution = ...  # backend, dtype, numerical program, sampler

    def boundary(self, state): ...
    def validate(self, state): ...
    def addresses(self, state): ...
    def advance(self, state): ...
```

`advance` consumes one declared execution boundary and returns a new state mapping. It must not
mutate model weights, input state, shared parent payloads, or undeclared global execution state.
Session publishes a continuation only after all requested transitions validate. Side effects inside
an adapter cannot be undone automatically; register/capture their closure or keep them out.

Declare every value needed to continue: carriers, cursors, caches, conditioning, RNG, and solver
history. Do not put recomputable observables into durable state unless they affect continuation
or are explicitly retained as evidence. Use simple JSON values and tensors for LocalStore.

`addresses` lists inspectable state. Override `write` to restrict read-only slots, validate types,
enforce invalidations, and recompute derived values. The base write implementation checks tensor
shape/dtype and validates the resulting state. You own dependency completeness and semantic ABI.

The pure-JSON `examples/custom_adapter.py` runs without torch. For a PyTorch model, use
`saturn_pub.values.identity(model, configuration)` to bind parameter/buffer content. Identity is
computed at adapter construction, so create a new adapter after any model change. Do not share
one mutable resident model across concurrent debugger/training sessions.

## Required adapter checks

- Native untouched execution versus instrumented execution under the same numerical program.
- Same-parent branch isolation, failed continuation atomicity, and meaningful intervention effects.
- Capture at every supported boundary and continue after restore, including a fresh process.
- Refusal of incompatible checkpoints, invalid shapes/dtypes, and non-finite state.
- Complete scheduler/cache/RNG closure and a consumer beyond the changed carrier.

Supported adapters should name exact versus bounded numerical parity. Do not silently accept a
new model family because its tensors have the same shapes. Source addresses and lowering remain
family-local even when operation names are shared.

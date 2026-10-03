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

## Block-streamed residency

For a model larger than the execution device, build the adapter under
`adapters._residency.BlockResidency`. Park the frozen weights in host memory with
`BlockResidency(device, mode="streamed").place(model)`, record the frozen guard *after* `place`
(host pinning reassigns storage), and route every native-module call through `residency.run(module,
*args, **kwargs)`. In resident mode `run` calls the module directly (zero overhead, byte-identical
to before); in streamed mode it copies exactly that module's parameters and buffers to the device
and runs it with `torch.func.functional_call`, so the transient device copy falls out of scope
afterward and the host weights never move. Because `Tensor.to` is byte-preserving and
`functional_call` substitutes without mutating the module, a streamed block computes the same bits
a resident block would on the same device. Keep per-step caches (key/value, recurrent) resident on
the execution device; only frozen weights stream. A full-model forward that the device cannot hold
resident -- a multi-token prefill or an uninstrumented comparator -- must itself be streamed block
by block rather than called as one `model(...)`. Add the residency descriptor to the execution
contract only in streamed mode so resident receipts stay byte-identical. The decoder, Qwen, and
Mamba adapters show the pattern end to end.

## Required adapter checks

- Native untouched execution versus instrumented execution under the same numerical program.
- Same-parent branch isolation, failed continuation atomicity, and meaningful intervention effects.
- Capture at every supported boundary and continue after restore, including a fresh process.
- Refusal of incompatible checkpoints, invalid shapes/dtypes, and non-finite state.
- Complete scheduler/cache/RNG closure and a consumer beyond the changed carrier.

Supported adapters should name exact versus bounded numerical parity. Do not silently accept a
new model family because its tensors have the same shapes. Source addresses and lowering remain
family-local even when operation names are shared.

## Shipped native adapters

`adapters/qwen.py` is the reference decoder adapter. `adapters/decoder.py` generalizes it to a
per-family spec registry (gpt2, gpt_neox, phi, llama, mistral, mixtral, gemma-1, qwen2, qwen3):
same state grammar, surface manifest, execution/numerical contract, frozen-model guards,
`validate`/`write` rules, layer/operation granularity, and `generate`/`native_logits`. Each family
declares its module paths and position style (absolute learned embeddings for gpt2, rotary
otherwise), and a fail-closed config validator that refuses an out-of-rule checkpoint by naming the
rule (for example gpt2 requires `gelu_new` and head-dim scaling; llama requires `pretraining_tp==1`;
gpt_neox and phi require untied embeddings; qwen3 requires attention qk-norm; mixtral requires a
valid routed-expert count; gemma-1 requires `gelu_pytorch_tanh` and biasless projections). Eager
attention is required and single-token decode runs `attention_mask=None` over an external
`DynamicCache`; the prefix is prefilled by one native full forward. `adapters/mamba.py` is a
separate adapter for Mamba-1: it cuts the native `MambaBlock` stack per token and per layer over
external causal-convolution and recurrent-state slots. Use `saturn_pub.adapters.load(path_or_model)`
to dispatch by `model_type`. See [pretrained recipes](pretrained.md) for one-line usage.

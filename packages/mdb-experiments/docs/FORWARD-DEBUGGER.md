# Pause a native forward graph

The bounded `mdb.forward_debugger` adapter executes one original PyTorch FX
node per public Saturn transition. Automatic capture uses no semantic dictionary.
Inputs are the model's ordinary arguments; graph values receive
automatic `vNNN` identifiers.

```python
from mdb.forward_debugger import create_forward_debugger

debugger = create_forward_debugger(
    frozen_model.eval(), input_tuple, model_identity=checkpoint_sha256,
    concrete_args={"capture": None},  # only when the model has this argument
)
debugger.execute("step")
debugger.execute("info forward")
debugger.execute("info symbols")
```

`info forward` shows the actual cursor, producer, input dependencies, readers,
shape, dtype, content digest and numeric samples. `info symbols` refreshes the
automatically generated public `SymbolTable` at the current point. Existing
Saturn capture, restore and branch commands retain their ordinary semantics.
The same native Session is available through `debugger.session`.

Only produced tensor values whose direct consumers are all still pending
can be edited. Inputs, parameters, previously consumed values and runtime
clocks are read-only. The adapter rejects supported stochastic and mutation
patterns, seals parameters, all buffers and supported module attributes, and
checks them for drift. FX tracing and Python process snapshotting are
different capabilities; unsupported dynamic behavior requires its own adapter.
Produced views adopt the public tensor snapshot layout immediately, so a
multi-node continuation preserves the same state contract as individual steps.
Native output parity must still be qualified for each model/backend.
Graph-node dependencies identify graph values. Module parameter and buffer
reads live in the sealed module closure; the native ATen trace exposes those
finer tensor reads.

Every internal semantic role remains `unresolved`. Structural module paths
and neural operations identify executable code: a neural `operator.add` does
not establish a latent x86 ADD instruction. Model output ABI fields remain
separate from inferred meanings of hidden state. Exact native-versus-stepped
output comparison and same-parent replay qualify the execution instrument.

The finer [native ATen trace](FORWARD-SYMBOLS.md) records functional tensor
primitives inside a model call. It complements the public Session graph cuts:
both can show actual producer/use structure without manually naming concepts.
Operation clocks from these two levels must not be collapsed.

The frozen direct-effect ISA NN consumes instruction-window bits and register/
flag before-state, then predicts next-state outputs. Its serving path needs no
instruction parser, supplied ALU or operation-family vocabulary. Its bit layout,
output ABI, architecture and training supervision remain supplied design
choices. It consumes machine-code bytes; it does not generate instruction bytes.

Separately measured, sealed `SymbolTable` sidecars can be attached through
`debugger.attach_annotations(symbols)` at their supported cut. `info symbols`
includes an annotation only while its input, model, clock and schema resolve.
These observed labels do not enter execution or grant edit authority. The
[readable ISA recovery](../../../../isa/docs/readable-symbol-recovery.md)
records a byte-effect rule and measured route/value/writer roles, including
the distinction between exact hard register words and differing float logits.

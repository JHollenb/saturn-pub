# Symbols observed during native forward execution

The [algebraic ISA follow-up](../../../../isa/docs/algebraic-symbol-acquisition.md)
adds a recipient-context polarity modulator, state-mode reader and nonlinear
amplifier as scoped working interpretations. Native factors compile the tested
384-coordinate state edit into a 48-coordinate mixer edit with decoded register,
flag and length parity in eight cases; numerical differences remain.

The [ISA causal atlas](../../../../isa/docs/forward-causal-atlas.md) adds a
human-readable working interpretation backed by typed native suffix branches:
a carry-conditioned state difference transfers between recipient contexts and
from bit row 8 to row 9. Its provenance and limitations remain visible beside
the source-derived equations and anonymous value identities.

`mdb forward-symbols` navigates values and operations recorded from an actual
PyTorch forward pass. It requires no caller-supplied symbol names or semantic
vocabulary. Values receive anonymous SSA identities such as `v000042`; each
records its native tensor shape, dtype, exact content digest, preview,
producer, and consumers. Operation names and module paths identify executable
code. They do not assign meanings to hidden states.

This is separate from `SymbolBinding` and `symbol-trace`, whose existing
contracts index supplied semantic annotations at declared locations. The
forward table can serve as a physical foundation for later qualified
interpretations without silently turning a label into an observation.

## Record and execute

```python
from mdb.forward_recording import ATenStepper, record_forward
from mdb.forward_symbols import ForwardSymbolTable

capture = record_forward(
    frozen_model.eval(), (input_tensor,),
    model_identity=checkpoint_sha256,
    execution={"backend": "cpu-fp32"},
    input_identity={"sha256": input_sha256},
)
machine = ATenStepper(capture.trace, capture.initial_values)
step = machine.step()   # Executes one recorded native ATen operation.
live_values = machine.symbols()  # Only values computed so far, no future activations.
table = ForwardSymbolTable(capture.trace)
print(table.format_step(step["pc"]))
output = machine.run()
assert all(row["exact_outputs"] for row in machine.steps)
```

The recorder intercepts native ATen calls, including functional operations
that module hooks omit. It observes the unchanged model and removes its hooks
on success or error. The stepper starts with only original inputs, frozen
parameters, buffers, and native tensor literals. Every intermediate is
recomputed by an operation. Supplying captured intermediate values as roots is
rejected. Each step compares its computed outputs with the native observation.
Random operators and mutable operations require additional closure contracts
and are refused by this bounded implementation.
The current tensor replay contract requires dense contiguous CPU inputs and
parameter roots; the native operators can then produce their own tensor views.

The tensor executor is optional and imports PyTorch. Its trace must come from
an owned, trusted native execution with a bounded operation/resource plan.
The sealed metadata navigator imports no model framework and executes no
model operations:

```bash
mdb forward-symbols trace.json --list
mdb forward-symbols trace.json --step 12
mdb forward-symbols trace.json --value v000042 --json
mdb forward-symbols trace.json --readable --scope mixers.2 --annotations annotations.json
mdb forward-symbols trace.json --readable --output-path /regs
mdb forward-symbols trace.json --algebra --json
```

`--algebra` nominates structural candidates from executed operator arguments
and def/use edges. The framework-free `mdb.forward_algebra` API supplies affine
scan composition, clock identities, finite/factorial contrasts, corner Möbius
coefficients and midpoint prediction. Structural identities, graph observations
and empirical bindings have separate provenance. A corner interpolant alone
does not establish a continuous law or a semantic symbol.

`branch_forward_many(..., observe_values=(...))` optionally returns cloned
candidate tensors at named SSA values and binds their digests in the typed
receipt. They are observations of the replay, never additional replay roots.

`--step` in the CLI navigates a captured execution. `ATenStepper.step()` runs
the actual recorded operation. These are distinct capabilities. The table
retains value definitions and uses; visibility is a dataflow property, not a
claim about allocator lifetime or learned semantic identity.

`ForwardReadableProgram` renders operator equations, source-name provenance,
tensor types, backward slices and structural patterns without importing a model
framework. External interpretations are a separate `mdb-forward-annotations-v1`
document bound to an exact trace fingerprint and value/event/module/output scope.
Evidence references are retained as references; the view does not certify their
semantic truth. `machine.symbols(readable=True, annotations=document)` joins the
same expressions to live computed values while keeping actual replay previews.

`mdb.forward_interventions.branch_forward_many` accepts produced SSA values and
replacement payloads, uses public Saturn `Act.replace` and `Receipt`, verifies
native roots and the exact prefix, and continues through the original recorded
operators. Changes after the first intervention are observations. Optional
native output tensors are hash-bound comparison observations, never replay roots.
The current contract replaces selected SSA handles; it does not implement global
alias mutation or arbitrary Python process checkpointing.

## Interpretation and clock scope

An ATen instruction is an executed tensor primitive, such as matrix multiply,
selection, addition, or exponentiation. It is not evidence that a network
emitted a discrete internal x86 program. A model's original output channels
and ABI can be shown separately with their provenance. A new semantic name,
probe, decoded thought, or causal-role interpretation requires its own
evidence; none is fabricated by this table.

Keep forward-call index, operation index, program-instruction index, and
recurrent sequence position separate. A tensor operation may compute many
positions in parallel. A recurrent field over bits or bytes is not automatically
an instruction clock, bank age, or expiry rule.

## Measured native Mamba use

The [ISA R3 Mamba assay](../../../../isa/docs/native-forward-symbols.md) recorded
three forwards on raw instruction bits and model-generated previous state. Each
contained 571 ATen instructions and 676 tensor identities. All 1,713 operation
outputs replayed exactly from inputs and weights; the three complete output trees
also matched untraced controls. The model-generated register trajectory was
`7 → 8 → 9 → 10`, with predicted four-byte advances. The scan's learned timestep
field is directly visible with its producer and consumers.

This qualification binds the executed source hashes in the archived report. It
covers deterministic CPU FP32 execution on one frozen checkpoint and short program.
The table retains source-authored module/output names alongside anonymous value
IDs. A later generic guard makes `run(require_exact=True)` reject mismatches in
previously stepped instructions too; its tiny regression test is separate from
the unchanged exact native replay evidence. The installed-wheel metadata proof
uses the exact wheels executed by the native job.

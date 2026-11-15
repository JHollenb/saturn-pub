# ISA discovery debugger inputs and execution design

Date: 2026-10-30. The input and current-capability sections describe existing
code. The session, lane, boundary and command interfaces below are proposed.
They are not a report of newly implemented or qualified execution.

The [full interface specification](ISA-DEBUGGER-INTERFACE.md) fixes object
ownership, versioned commands, motion/stop semantics, instruction commit,
action scripts, lane scheduling, accounting and recovery. Its
[case inputs](design/isa-discovery-cases.example.json),
[session plan](design/isa-discovery-plan.example.json) and
[action script](design/isa-discovery.example.mdb) are reviewable proposed
formats, not files accepted by the current runtime.

The immediate objective is to explore a model as an addressable program:
identify components with generic names, pause at their boundaries, inspect and
branch state, and automate repeated experiments. Recovering names for latent
ISA operations is optional research, not a prerequisite for debugger utility.

## Exact input to the current model

These facts apply to the frozen R3 direct-effect ISA Mamba used by the recent
computation-route experiments. Other ISA profiles and models have different
interfaces. This model consumes no natural-language prompt or tokenizer.

The ordinary forward signature is
`model(window_bits, reg_bits, flags)`:

| Tensor | Shape | Meaning |
|---|---|---|
| `window_bits` | `[B,120]` | Fifteen raw bytes, least-significant bit first within each byte |
| `reg_bits` | `[B,8,64]` | Eight 64-bit register words, least-significant bit first |
| `flags` | `[B,4]` | Incoming CF, ZF, SF, OF |

The register order is RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI. Thus the
experiment names R3, R6 and R7 refer to RBX, RSI and RDI respectively.
The current scientific adapter uses `B=1`, dense CPU FP32 roots.

For example, the recent imm8 source condition is:

```json
{
  "window_hex": "4883c7010000000000000000000000",
  "register_words": [0, 0, 0, 0, 0, 0, 3, 255],
  "flags_CF_ZF_SF_OF": [0, 0, 0, 0]
}
```

The hexadecimal window has fifteen bytes: `48 83 c7 01` followed by eleven
zeros. The network emits next-register and flag logits, instruction-length
logits and additional heads. The serving loop thresholds register/flag logits
at zero and decodes length as `argmax + 1`. The recent causal runs retain eleven
heads, including store enable/address/value outputs; this three-input ABI does
not expose a general RAM input.

Sources: [model and bit layout](../../../../isa/direct_effect/model.py),
[register and flag order](../../../../isa/direct_effect/profile.py),
[native input packing and decoding](../../../../isa/isa_lab/native_forward_symbol_assay.py).

## What context means here

One forward predicts the effects of the instruction at the beginning of its
window. When the host fetches from a concatenated program, the same window can
contain bytes from following instructions. The decoder sees all fifteen bytes;
it is not given pre-segmented instructions. Isolated training examples also use
random trailing bytes, while recent intervention cases use explicit zero tails.

The host owns program bytes and the current byte offset/RIP. Absolute RIP is
not another tensor in this forward signature. Prior architectural history
enters through the supplied register/flag state. There is no prompt history,
CoT, KV cache or retained Mamba hidden state between these ordinary forwards.

Inside a forward, the register representation becomes a sequence of **64 bit
positions**. Each position includes that bit of all eight registers and learned
instruction-dependent features. The Mamba scan runs along this bit axis. Its
state resets on the next call. These positions are not sixty-four program
instructions. Persisting this scan state across calls would change the model's
execution contract.

## Multiple instructions

The existing [rollout driver](../../../../isa/direct_effect/train.py) repeats:

1. Fetch fifteen bytes at the current model-driven instruction pointer.
2. Run a forward on those bytes and the current predicted registers/flags.
3. Commit its decoded register/flag state and advance by its predicted length.
4. Repeat within `max_steps`, stopping according to the driver's policy.

There is no neural context-window limit on the number of such calls. The host
budget and correctness of the accumulated predicted state/length constrain a
rollout. Evaluation code includes lengths16,128,1024; the existence of those
panels does not establish reliable execution at those lengths.

The native forward-symbol assay already demonstrates three predicted-state,
predicted-length-driven calls. Other bounded experiments retain four-instruction
paths. The most recent MLP causal panels instead run a main instruction and a
separately supplied next ADD1. They establish literal own-state consumption,
not an arbitrary program runner or persistent hidden-state continuation.

## Current debugger pieces

| Surface | Implemented behavior | Boundary |
|---|---|---|
| `ForwardSymbolTable` / `ForwardReadableProgram` | Anonymous values, operator expressions, module paths, producers, uses and output bindings | Metadata navigation alone does not execute or pause a model |
| `ATenStepper` | Executes one recorded functional tensor primitive per step | Requires a native capture; current qualified R3 lane is deterministic CPU FP32 |
| `ForwardAdapter` / `ForwardDebugger` | A public Session over an FX graph, executing original graph nodes | FX and ATen clocks differ; model/backend qualification is adapter-specific |
| Public `Debugger` | Break/until/watch, capture, restore, fork, compare and command execution | Breaks currently match a boundary string; the FX adapter emits `node`/`complete`, not named MLP boundaries |
| Command scripts and Python APIs | Existing CLI command files and callable session APIs | Unified ISA instruction sessions and breakpoint action blocks are not implemented |
| `PreparedForwardParent` / `ExperimentPanel` | Reuses a root-produced frontier and seals bounded intervention panels | Branches run serially; no physical branch batch or cross-job resident model service |

The ATen trace used by this R3 has571 primitive events and676 tensor identities
per forward. An FX node may invoke several ATen events. These clocks must never
be treated as interchangeable integer offsets.

See [forward debugging](FORWARD-DEBUGGER.md),
[ATen execution](FORWARD-SYMBOLS.md),
[local commands](LOCAL-DEBUGGER.md), and [prepared panels](PREPARED-PANELS.md).

## Structural identities and aliases

Provide three distinct identities:

- **Site:** a component or port in a model/program version, such as `post.1:exit`.
- **Occurrence:** that site in a particular lane, input context, instruction
  call, branch and execution clock.
- **Route:** an identified set of producer/use edges with an associated scope;
  observed effects are separate records attached to that route.

Use generic display names such as `mlp_001`, `value_00412`, `route_0007` and
`state_update_003`. Back each display ID with an immutable qualified identity;
an ordinal alone is not a portable address. User aliases such as
`interesting_write` or `early_branch` are editable metadata. Exact activation
content digests identify observed values, not stable component sites.

Source module names are useful associations. Semantic descriptions such as
"carry reader" can be added separately, with their evidence and limits.
Ordinary aliases should not require an observed/validated semantic claim or
pass through the existing evidence-qualified annotation API. Naming grants no
additional mutation or execution authority.

## Proposed execution hierarchy

Expose explicit units rather than silently changing the meaning of `step`:

| Unit | Completion boundary | Example |
|---|---|---|
| `op` | One ATen tensor primitive | Matrix multiply, add, reshape, exp |
| `math` | One declared mathematical stage | LayerNorm, affine transform, GELU |
| `law` | A declared component's output, before its caller applies the result | Complete `post.1` MLP |
| `update` | A declared state-update transaction | Residual `x + F(x)` has been applied |
| `forward` | The model call has produced all output heads | Raw next-state logits exist |
| `instruction` | Decode/commit the predicted architectural state and select the next window | Registers/flags and byte offset advance |

An MLP is a composite function. Treating its complete evaluation as one law
step is useful. Mathematical stepping can enter its LayerNorm, first affine,
GELU and second affine. A primitive step operates on whole tensor operands;
it is not a separate debugger step for each scalar element.

The law's output and the residual update that consumes it are different
stopping points. For each boundary record its inputs, outputs, live frontier,
upcoming consumers, structural site and before/after edge.

For example, the retained imm32/R6=3 source11 capture in
`isa/runs/native-forward-symbols-20261030T090600.257799Z` already exposes these
distinct post1 endpoints:

| Site | Captured value | ATen event | Next-event cursor after production |
|---|---|---:|---:|
| MLP input available | `v000412` | 358, previous mixer residual add | 359, before LayerNorm |
| MLP output available | `v000430` | 368, final projection view | 369, before residual add |
| Residual update applied | `v000431` | 369, residual add | 370 |

These zero-based event indices and value IDs belong to that recorded program;
the proposed boundary resolver must bind the trace/program identity rather
than assume the same numbers apply to every model or graph. This table is a
metadata inspection, not a new live-breakpoint qualification.

The current selective scan uses a parallel doubling implementation. A logical
recurrence position `H[t]` is not automatically a physical native pause point.
Initially inspect selected positions at available scan boundaries. A truly
resumable bit-by-bit scan needs a separately qualified execution adapter.

## Proposed breakpoints and motion

Use structured boundary descriptors with model/program identity, site/module,
edge, forward/instruction index, graph-node or ATen event index, and optional
tensor selectors. Resolve an MLP entry to before its first consumer and its
exit to after the final projection but before the residual addition. Keep
component kind and generic alias searchable.

These examples are **proposed syntax**, not current CLI commands:

```text
break module:post.1 enter
break module:post.1 exit
break update:post.1.residual after
step --unit op
step --unit math
step --unit law
step --unit update
step --unit forward
step --unit instruction
```

The full specification places these behaviors in the explicit `discovery-v1`
interface. Legacy `next` and `continue N` retain their existing meanings.

`next` should step over the current selected scope; `finish` should reach its
exit; `until` should execute to a chosen boundary. Each must still honor
intermediate breakpoints and watchpoints. The existing generic `next` is an
alias for ordinary transition stepping and does not yet implement this scope
behavior.

Reserve history navigation for retained state. `restore` resumes a compatible
snapshot; reverse motion restores a preceding snapshot or replays from one.
A jump cannot simply change the operator cursor while leaving required tensor
producers unexecuted. It must restore a valid live frontier or execute the
necessary prefix. Inspecting an old capture and executing from it should be
visibly distinct actions.

Start by adapting the already qualified R3 ATen execution into the public
Session contracts. Make the initial capture visible: a recorder that first
completes a forward and then replays it is not pausing that first execution.
Direct eager safe-point capture can be added as a separate execution mode.

## Proposed resident lanes and physical batching

Keep a frozen model resident inside a finite admitted mrun lease. Each logical
lane owns its program/window, registers, flags, byte offset, execution cursor,
branch ancestry and retained frontier. Inputs can be independent experiments
or branches from a shared parent. Selecting a lane changes debugger focus;
replacing its input creates a new execution context.

Changing instruction bytes or before-state invalidates downstream cached
values. A prepared parent belongs to its exact inputs and execution contract;
an old prefix cannot be reused for a different input merely because the tensor
shapes match. Forks from an unchanged parent can reuse that parent's valid
frontier. Persist the original lane occurrence before reusing its physical slot.

Implement multiple resident scalar lanes first, then group compatible lanes
into physical batches. The model itself accepts a batch dimension, and its
ordinary evaluation already uses batches. Its current causal recorder,
decoder helper and prepared panels are not a batch-lane debugger. A recorded
batch-one graph also contains shape and scalar constants that cannot be
silently reinterpreted for larger batches.

Initially batch lanes at a common execution boundary and park paused lanes.
Later scheduling can compact runnable lanes or batch suffixes at matching
cuts, with independent masks/state and per-lane receipts. Lane refilling must
reset its state and ancestry rather than inherit another experiment's hidden
values. A stopped lane remains stopped while other lanes run.

Measure throughput and end-to-end cost at several batch sizes; this tiny model
may be limited by recording, cloning, hashing and publication rather than
matrix arithmetic. Retain a fast, explicitly scoped observation path and a
full-recording audit path. Qualify scalar/batched numerical behavior, lane
permutation and refill isolation, breakpoint behavior and same-parent edits.
Batch-dependent differences remain measured outcomes rather than being
silently called exact scalar execution.

Cross-job resident serving requires additional scheduler admission, ownership,
custody and invalidation work. A bounded resident lease is the first useful
step and must not be presented as that cross-job service.

## Proposed scripting model

Use one typed command/API layer for the console, files and Python automation.
Reuse public Session/Act/Receipt and prepared-panel operations. Existing command
files remain valid; add bounded action blocks at breakpoints and allowlisted
predicates for automatic snapshots, summaries, logging, branches and continuation.

For example, this is a proposed action script:

```text
interface mdb-discovery-v1
plan load design/isa-discovery-plan.example.json
break module:post.1 exit
on-hit
    capture auto
    log inputs outputs route --summary
    continue
end
run --all --stop-policy lane
```

Trusted Python callers can orchestrate richer loops over the same API. A saved
declarative experiment must not import arbitrary callbacks. Bound action depth,
commands, branches and total instruction/operation work; preserve per-action
errors and all completed evidence. Log script version/hash, command sequence,
resolved site IDs, lane/branch/input identity, installed edits and outputs.

## Implementation order

1. Structural alias and boundary catalog, with separate stable sites and runtime
   occurrences. No semantic operation inference required.
2. Qualified ATen-to-Session adapter for the current R3, MLP entry/exit and
   residual-update boundaries, explicit stepping units and scope-aware motion.
3. ISA instruction driver plus shared command/Python API, breakpoint actions
   and reproducible scripts inside one resident finite lease.
4. Multiple independent resident scalar lanes, input replacement and valid
   parent/frontier reuse.
5. Physical batching and refill/paused-lane scheduling after numerical,
   isolation and performance measurements.

This extends the [discovery execution roadmap](DISCOVERY-EXECUTION-UPGRADES.md)
with a debugger interface that is useful before a semantic ontology is known.

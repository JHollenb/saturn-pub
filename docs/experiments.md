# Causal experiments and measured programs

For a complete pretrained experiment, run the
[FLUX.2 writer recipe](../experiments/flux2_writer/README.md). It builds the writer,
compares native image futures, restores rejected updates, and checks fresh-process replay.
It needs only the public package, model checkpoints, and CUDA; no fleet runner is required.

Start with one specimen, one parent, a bounded continuation horizon, and the actual consumer.
The execution grammar is `producer → address → payload → carrier → consumer → future`.

```python
from saturn_pub.investigation import Investigation

plan = Investigation.doses(
    "Does this hidden write change the next token?",
    "hidden",
    payload,
    [-1, 0, 0.25, 1],
    steps=1,
)
panel = plan.run(
    session,
    lambda branch: {
        "next_token": int(branch.read("tokens")[0, -1]),
    },
)
```

Choose a boundary whose continuation budget reaches the named consumer. One Qwen micro-step
may only reach an embedding or layer, while one DDIM step reaches the next latent. Generation
examples call the adapter's token-budget helper to avoid confusing layers with generated tokens.

Add Arm objects for wrong-source, wrong-site/time, matched-random, reverse-order, and coalition
controls as applicable. Unsupported sites fail per arm and remain `instrument-error` rows.
Native, zero-dose, and candidate branches share one parent and one resident model. They are
paired conditions, not independent specimens. Replication requires new seeds, inputs, or models.

Preserve full signed measurements, distributions, trajectories, disagreement, and rejected arms.
The planner sets `terminal_status=not-assessed`. A continuation policy chooses a next branch;
it is not an authority on whether a scientific effect exists. Keep mechanics, instrument validity,
trends, and terminal claims distinct when writing your own report.

## Programs

Program.compile accepts a completed measured arm and the corresponding ordered Acts. V1
accepts only zero-delay interventions performed before the panel's continuation. It checks the
Act descriptors and parameters, parent fingerprint, recipient identity, boundary, execution ABI,
named context, and measured continuation budget. Its evidence establishes structural support,
not semantic certification.

Callable Acts execute trusted caller code. Bind a custom implementation version and its payload
in `Act.parameters`; the runtime cannot certify an arbitrary Python closure's semantics. Built-in
Acts bind their payload fingerprint and dose. Programs are in-process objects, not serialized
Python functions. Durable executable packages and learned temporal lowering are future work.

## Incremental execution

DependencyGraph cells must be topologically ordered, declare every input, and have an explicit
implementation version. Change that version whenever implementation semantics change. Inputs
cannot shadow computed cells. Reuse is admitted by input content, cell name, and version inside
one graph instance; mutable closure inputs belong in the declared read set.

The final consumer always executes. If native continuation feeds changed state back into the
graph, pass that changed state as a new input; its downstream dirty closure recomputes. Logical
reuse is not automatically a speedup. Measure native and cached paths with matched budgets,
including lookup, verification, capture, copying, hydration, and decode.

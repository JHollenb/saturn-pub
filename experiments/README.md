# Reproducible experiments

These recipes run with the public package and public framework dependencies. Model
weights are acquired separately. No private checkout, fleet service, or `mrun` is required.
Generated states and results belong in `outputs/`, which is ignored by Git.

| Experiment | Process | Consumer |
|---|---|---|
| [FLUX.2 writer](flux2_writer/README.md) | Capture paired states, fit a writer, evaluate futures, restore, replay | Native rendered RGB |

Each recipe declares supplied inputs, fitted quantities, observations, exact mechanics,
and the scope of its claims. Reference results illustrate one measured development
context; they are not an expected universal outcome or a gate for your experiments.

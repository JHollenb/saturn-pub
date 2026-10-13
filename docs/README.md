# Saturn documentation

This repository ships two independent Python packages. Both wrap native model
execution with inspectable state, branches, interventions and replay. Choose the
package whose workflow and numerical contract fit the job.

| Package | Starting point | Best fit |
| --- | --- | --- |
| `saturn-pub` / `saturn_pub` | [Capability usage guide](usage.md) | Local causal experiments, interpretability integrations and reversible training |
| `mdb-runtime` / `mdb` | [MDB overview and quickstart](mdb.md) | Durable debugger sessions, managed branch memory and resumable model execution |

See [two packages, one repository](monorepo.md) for installation, builds, separate
test suites and migration details. Installing one package does not install the other.

## MDB

- [Overview, use cases and quickstart](mdb.md)
- [Installation and optional dependencies](../packages/mdb/docs/INSTALL.md)
- [Custom backends: forward, capture, intervention and local debugger](../packages/mdb/docs/BACKENDS.md)
- [Usage: execution, history, memory, debugger and offline tools](../packages/mdb/docs/USAGE.md)
- [Experiment walkthroughs: rewind, residual edits, sibling memory and diffusion replay](../packages/mdb/docs/EXPERIMENTS.md)
- [J-space reader/writer API](../packages/mdb/docs/JACOBIAN-WORKSPACE.md) and [reproducible proof](../packages/mdb/experiments/jacobian-reader-writer/README.md)
- [Qualification records and model/backend boundaries](../packages/mdb/docs/QUALIFICATION.md)
- [E6/E20 paper experiments and research-session boundaries](../packages/mdb/experiments/time_circuits/README.md)
- [Package README](../packages/mdb/README.md) and [extraction provenance](../packages/mdb/PROVENANCE.json)

The detailed MDB guides live with its package so its source distribution retains
its documentation. The links above are the repository-wide entry points.

## Saturn toolkit

- [Choose a capability by job](usage.md), [runnable examples](../examples/README.md) and [notebooks](../notebooks/README.md)
- [Architecture](architecture.md), [capabilities and numerical contracts](capabilities.md), [runtime and memory scope](runtime-memory-scope.md)
- [Debugger](debugging.md) and [symbols, provenance and debugging repair](debug-symbols-paths.md)
- [Experiments and measured programs](experiments.md) and [reversible training](training.md)
- [Adapters](adapters.md), [pretrained recipes](pretrained.md) and [diffusion families](diffusion-families.md)
- [SAELens / TransformerLens](interop.md) and [circuit-tracer](circuit-tracer.md) integrations
- [Writer demonstration](writer-demo.md), [Instrument Trial](trial.md) and [induction certificate](certificate.md)
- [Evidence plane](evidence.md) and [metadata-delta route](route.md)
- [Release validation](validation.md) and [extraction provenance](../PROVENANCE.md)

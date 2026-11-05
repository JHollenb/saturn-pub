# MDB development

MDB is the public model debugger and durable runtime at `packages/mdb` in the
Saturn-pub repository. MDB may depend on and import the public `saturn_pub`
package. Prefer reusing its typed contracts, model adapters, analysis primitives
and execution APIs over copying implementations or adding a separate bridge
solely to preserve package isolation. Keep the dependency direction
`mdb -> saturn_pub`; the Saturn-pub core must not depend on MDB.

MDB remains a separately built, installable distribution. Declare any required
Saturn-pub dependency in MDB's package metadata; use optional extras and lazy
imports for optional integrations. Runtime imports, worker payloads and launchers
must resolve declared installed packages, never discover sibling checkout paths
or patch `sys.path`. Private `saturn`, `common` and `common_harness` imports remain
forbidden. Maintainer extraction tools require an explicit upstream source.

Run MDB's tests in a separate process from the root Saturn-pub suite to avoid
test-module collisions. Test guards must allow public `saturn_pub` imports and
continue blocking private workspace packages. Test installed wheels with their
declared dependencies outside the workspace, including Saturn-pub whenever used.
Framework-free commands must remain usable without importing model frameworks.

Preserve legacy serialized schema/address/CAS and numerical-contract identities.
Do not change Hyperion compiler bytes casually: measured artifacts bind its digest.
Sharing code does not make existing cuts interchangeable or qualify a new model
execution lane. Reconcile dependency pins before sharing model execution, and
qualify the resulting model/backend/dtype contract separately from tiny mechanics.

Use mrun admission for real models, sequential CUDA, one finite lease per assay,
strict preflight and no retries. Independently review runs expected over 15 minutes.
No fleet refresh or private Saturn consumer migration is implied by this policy.

## Pretrained parameter artifacts

New research runs must include a standalone `retrieve_weights.py` and a
`parameter-artifacts.json` binding the required named tensors to immutable
checkpoint files, original archive headers, shapes, dtypes and hashes. Use
`qualification/prepare_flux_parameter_artifacts.py` for the qualified FLUX
capture format. Keep pretrained constants outside persistent per-run evidence
once exact retrieval has been qualified; retain new activations, states,
predictions, controls and native outcomes. Temporary weights used for execution
or audit may be hydrated when needed.

Before removing historical parameter copies, prove exact original archive-byte
restoration and exclude any file also serving a nonparameter role. Preserve the
original manifests, inventories, seals and audits; put retrieval metadata and a
separate thinning receipt outside their originally inventoried result tree.
Record that an externalized tree is incomplete until rehydrated. A legacy full
inventory audit must run only after exact restoration. Never remove the pinned
source checkpoint or unique experiment state as parameter cleanup.

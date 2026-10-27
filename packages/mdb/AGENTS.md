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

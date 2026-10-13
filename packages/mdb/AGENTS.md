# MDB development

MDB is the standalone canonical runtime extraction. Runtime imports, worker
payloads and launchers must not discover or import Saturn, Saturn-pub or common
checkout paths. Maintainer extraction tools require an explicit upstream source.

MDB now lives at `packages/mdb` in the Saturn-pub repository and remains an
independently built distribution. Run its tests in a separate process from the
root Saturn suite. Co-location does not authorize cross-package runtime imports
or changes to the qualified model execution and serialized identities.

Preserve legacy serialized schema/address/CAS and numerical-contract identities.
Do not change Hyperion compiler bytes casually: measured artifacts bind its digest.
Run the focused suite with upstream imports blocked and test an installed wheel
outside the workspace. Model-backed qualification is separate from tiny mechanics.

Use mrun admission for real models, sequential CUDA, one finite lease per assay,
strict preflight and no retries. Independently review runs expected over 15 minutes.
No fleet refresh or Saturn consumer migration belongs to this extraction.

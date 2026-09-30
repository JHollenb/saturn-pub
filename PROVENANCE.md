# Extraction provenance

This repository is an independent curated toolkit, not an export of the research tree.
Public contracts are intentionally smaller; private experiment code, result archives, indexes,
model weights, credentials, host configuration, and Git history are excluded.

The implementation was informed by the Saturn research working tree on 2026-09-29,
whose latest committed base was `7dcccb311ef54e65acd60c70b88415f7fdd50229`.
Uncommitted working-tree sources were included in the review; the commit alone is not their identity.

## Direct adaptation

`training/_checkpoint.py` adapts the author's private training checkpoint controller.
Source SHA-256: `6302204520f4f139ff55db6e401601f301c859479435a273ee672713d9fd9e0b`.
The original controller is already family-neutral internally. Public changes rename schema IDs,
remove a workspace-specific comment, and make the all-score-source policy requirement opt-in.
All score views remain recorded; users choose which objectives control continuation.
The registered-mutable-state wrapper is new.
The corresponding original mechanics tests were adapted to the public import namespace.

This retains tested handling of optimizer topology, tied tensors, gradients, nonpersistent buffers,
module modes, RNG, evaluation isolation, state drift, rollback, and immutable decision receipts.

## Independent small implementations

Core transactions, native adapters, local storage, debugger, investigations, measured programs,
dependency reuse, shared-memory reference, and modular phase reference were written for this
repository. They draw on the author's documented state-transaction, causal-panel, measured-program, runtime,
branch-search, partial-evaluation, and phase-transport contracts without copying experiment harnesses.

The phase reference implements the mathematical modular law directly. It does not vendor the private phase
tensor format, its block formats, kernels, or MIT-licensed source. The shared-history reference has no Triton or
scheduler dependency. Optional upstream frameworks are installed separately and keep their own licenses.

All bundled owned code is Apache 2.0. No historical scientific result is granted by source extraction.

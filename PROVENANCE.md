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

`saturn_pub/evidence/` adapts the author's private evidence-plane modules, which the original
working tree keeps stdlib-only and import-clean (no torch/transformers/scheduler/object-store).
Near-identical extracted copies existed in two private trees; the cleaner originals were adapted:

- `evidence/bisect.py` and `evidence/_canonical.py` adapt `divergence_bisect.py`.
  Source SHA-256: `031b15ad2d9ffc23587442e068e38c55cc3bccc383741c4a12c6444281d8d661`.
- `evidence/claims.py` adapts `claims_registry.py`.
  Source SHA-256: `5a7ee4d0f4bcd59b4950d7f92baa85fda10703101f0eceaf2626082f9b279215`.
- `evidence/xref.py` adapts `xref_index.py`.
  Source SHA-256: `62ac2d68c976cf45185de82639f895cb1191549bbaf682ac441d7b6a89986c02`.
- `evidence/citations.py` adapts `evidence_citations.py`.
  Source SHA-256: `53d44593f1b23f6f0967845d627b70f69da8472de5096f5507777e69772a5103`.

Public changes rename schema IDs to the `saturn-pub-*` family and the citation marker to
`saturn_pub_citation`; factor the shared canonical-JSON/sha256 custody idiom into a self-contained
`_canonical` module; remove workspace-specific default paths and comments (the registry and index
require explicit paths); and replace the private checkout's repo-root/repo-src declared-reference
resolution ladder with a neutral receipt-directory + optional `base_root` resolution. The bisect
gains a caller-callback integration form (`first_divergence_pairs`) and probes-versus-linear
economics; nothing in the subpackage imports torch or the rest of `saturn_pub`. The corresponding
original mechanics tests were adapted to the public import namespace, dropping fixtures that pointed
at private host paths or real private result trees. The subpackage is kept self-contained so it can
later move into a neutral base package shared by more than one toolkit.

## Independent small implementations

Core transactions, native adapters, local storage, debugger, investigations, measured programs,
dependency reuse, shared-memory reference, and modular phase reference were written for this
repository. They draw on the author's documented state-transaction, causal-panel, measured-program, runtime,
branch-search, partial-evaluation, and phase-transport contracts without copying experiment harnesses.

The phase reference implements the mathematical modular law directly. It does not vendor the private phase
tensor format, its block formats, kernels, or MIT-licensed source. The shared-history reference has no Triton or
scheduler dependency. Optional upstream frameworks are installed separately and keep their own licenses.

All bundled owned code is Apache 2.0. No historical scientific result is granted by source extraction.

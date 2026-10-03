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

`trial.py` adapts the author's private *Instrument Trial*: a preregistered head-to-head between
standard interpretability instruments and consumer-gated certification, with an offline receipt
verifier. The frozen mechanical decision rules in `verify_bundle` (necessity threshold 0.15,
cosine-recovered threshold 0.99, and the per-case derivations) are ported from the source verifier.
Source SHA-256 of adapted files (Instrument Trial working tree):

- `src/instrument_trial/verify.py`: `9738f077e5c5732a96a7aebcf3801cbd01f587442b34d885e0e94f4dd3861f40`
- `README.md`: `11e57fa2d00f9ddcc73df84aa4ddeb69a0eb1c77ac96d2f26f5c09eb0c6b9854`
- `TRAPS.md`: `8ee4ab9733f25c6b1d3ca161f704b90737e8f4fc571e113dff9eafce25d21867`
- `GLOSSARY.md`: `4e20509d23049eabeffda127cb747ba596687cffbf1eb5605d4c78333aece5ca`
- `verdicts/expected.json`: `a693847b45e68a48d4e4102f07a0100310af9e423207cd6f34b5db5c67fcb41d`
- `verdicts/verdict-records.json`: `d157e39bf7f606a232826ca1ff704ab6d10d4830697e69cf5dee91006e21bf69`
- `bundle/manifest.json` (original pins): `0f656aa71bbe2f58bc9727c8b8a08194c6c6c902b3500a1b50f733c68b8e1d1b`

The public `arbitrate` / `Reading` / `DecisionRule` / `TrialRow` surface, the `Trap` checklist, and
the external-reader seam were written for this repository against the source's documented discipline;
they do not vendor the private arms, graders, or scheduler harness.

### Instrument Trial bundle redaction

The receipt bundle under `src/saturn_pub/trial_data/` is a redacted, re-pinned copy of the source
bundle. The source receipts carried private scheduler/host provenance (job scheduler identifiers,
host names, absolute `/home` and `/mnt` paths, source-tree names) and large base64 image blobs that
are not read by the verifier. For the public release a recursive scrubber removed every private
provenance key and subtree (for example the whole `execution_receipt` block), every string value
carrying a private path/host/codename, and every embedded image blob; `expected.json` was copied
verbatim; `verdict-records.json` had its source-receipt paths repointed to the public bundle. The
grading records were already free of private tokens. Because bytes changed, `manifest.json` was
re-pinned to the SHA-256 of the scrubbed files, and all six case verdicts (eleven arm verdicts) were
confirmed to still re-derive from the scrubbed bundle under the frozen rules. The bundle shrank from
roughly 22 MB to 1.6 MB. The receipts remain on public weights (FLUX.2 Klein 4B and Qwen-family
models); no private hostnames, paths, usernames, scheduler names, or codenames remain.

`certificate/gate.py` adapts the author's private autoregressive induction-circuit certificate,
a runtime-neutral fail-closed gate over preregistered relational-induction panels. The seven
per-panel gates, the two set-level gates, the Wilson upper-bound necessity/specificity tests, the
repair-replay fidelity test, and the content-seal are ported directly. Source SHA-256 (Saturn
research working tree):

- private induction-circuit certificate module: `4f6e1d6b388f53444773df7f46a575759acb7bdffbfa3f7b27c729c3fabad023`
- its mechanics test: `29b4c58182540219f79214d326d6527c1b4a7149c061947e0a5a3961891fd150`

Public changes rename the schema IDs and classes to the public namespace and replace the source's
hard-wired framework/accelerator consumer check with a declarable `ConsumerContract` (consumer
identity always checked; backend/device allow-lists optional and frozen in advance) so the same
policy certifies a CPU specimen and a GPU run. The source mechanics tests were adapted to the public
namespace and extended with consumer-contract, workload, and custody-binding tests.

`certificate/workload.py` and `certificate/panel.py` re-implement, against the public decoder
adapter and torch only, the relational-induction workload and the five-branch causal battery whose
private originals are a benchmark module (workload + exact direct-source extraction, SHA-256
`3be94a1b81f5e18c1ba620f2ee77565930074bbf5c835e2ef22971b08bf91568`) and a discovery/confirmation
worker (SHA-256 `ae02eb1110ce9a53bde100fa804af58b321bbce6ef78ca8c47141c1abec60e85`). The private
workload used a numpy RNG and a private model engine, job scheduler, and activation-capture library;
the public version uses a stdlib `random` RNG and applies the attention-edge deletion and
pre-`o_proj` repair as forward-hook contexts on the public decoder adapter's resident native model.
The source's compressed-envelope discovery/selection stage is intentionally omitted; the public
module certifies the full direct-source parent only. No scheduler, engine, or capture-library code
is vendored.

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

`saturn_pub/route.py` adapts the concept and payload-free discipline of the author's private
metadata-plus-delta route, which in the research tree was scoped to one diffusion family and built
on a diffusion-specific route register. Source SHA-256 of the adapted files:

- `src/saturn/metadata_delta_route.py`: `6d7cbecc6bde27885c15dd2fd84391bd53f74a4431a9dd0ef3a97e584a5827c1`.
- `src/saturn/temporal_route_register.py`: `edae9d9dcd28e4d9e44a868ef5d26d3ba92198965876253e6b67a80e9c47e611`.

What is kept is the genuinely new mechanism: a tensor-free, content-addressed catalog of candidate
deltas that is queried (`select`) before any heavy state is hydrated, a caller-owned resolver seam
with deduplicated, content-verified, byte-accounted hydration, and a custody receipt that embeds no
payloads. The public version is a thin, adapter-neutral layer over this repository's existing
`Session`/`Act`/`Receipt`: a `DeltaCard` binds a measured builtin operation to a recipient cut and
references its delta bytes by sha256 + shape + dtype; `apply_delta` composes the existing fork /
apply / continue / compare / restore lifecycle into one route-execution receipt whose
`rollback.verified_exact` is the public analog of the private demonstrated exact rollback. Public
changes rename the schema IDs to the `saturn-pub-*` family; drop the diffusion-only register builder
(`build_route_register_from_family_delta_report`) and the private route coordinate, contract,
knowledge-bundle, and codename vocabulary; generalize the selection coordinate to free-form tags so
the same card drives a decoder (`hidden`) and a FLUX block suffix (`image`/`text`); and verify
resolved bytes by exact content address rather than shape/dtype alone. No private register, report,
object-store client, or result archive is vendored; the module imports no torch to build or query a
route.

## Independent small implementations

Core transactions, native adapters, local storage, debugger, investigations, measured programs,
dependency reuse, shared-memory reference, and modular phase reference were written for this
repository. They draw on the author's documented state-transaction, causal-panel, measured-program, runtime,
branch-search, partial-evaluation, and phase-transport contracts without copying experiment harnesses.

The phase reference implements the mathematical modular law directly. It does not vendor the private phase
tensor format, its block formats, kernels, or MIT-licensed source. The shared-history reference has no Triton or
scheduler dependency. Optional upstream frameworks are installed separately and keep their own licenses.

The adapter-owned block-streamed residency (`adapters/_residency.py`, originally written for this
repository's FLUX adapters) was extended here to the decoder, Qwen, and Mamba LM adapters so a
model larger than the device can be stepped one native block at a time. This reuses `torch.func`
and the model's own forward and masking; it does not vendor the private paged key/value engine or
any scheduler, which remain outside this repository.

All bundled owned code is Apache 2.0. No historical scientific result is granted by source extraction.

# Changelog

All notable changes to `saturn-pub` (`src/saturn_pub`, repository root). The version number is set by the owner. Each release is one commit that sets the version in `pyproject.toml` and `src/saturn_pub/__init__.py`, with an annotated tag `saturn-pub-vX.Y.Z` on it. `mdb-runtime` has its own changelog and `mdb-vX.Y.Z` tags in `packages/mdb/CHANGELOG.md`.

Sections 0.4.3 and earlier were reconstructed from `git log -- pyproject.toml src/saturn_pub/__init__.py` and the commits in each range; they are summaries, not a full audit. Tags for 0.2.0 to 0.4.3 were created afterwards on the commits that first set each version. Earlier history: see `git log`.

## [Unreleased]

Nothing yet.

## [0.4.3] — 2026-11-17

Tag `saturn-pub-v0.4.3` (901422e).

- Public `cut_blocks`, `Blocks` and `StateCut.from_blocks`, so mdb checkpoints can store and rehydrate a native cut without saturn-pub internals. mdb pins `saturn-pub>=0.4.3`.

## [0.4.2] — 2026-11-16

Tag `saturn-pub-v0.4.2` (5493dd3). mdb pins `>=0.4.2`.

- `set_shared_store(root)`: an explicit shared pool for the store.
- Optional side-stream prefetch and host embedding for streamed residency; tiny modules stay out of the streamed prefetch chain.
- Stepping speedups with unchanged fingerprints, and audit fixes to them.
- One copy, one hash and one validation per call in the core and decoder adapter; mixing in float64.
- Read-only probe, port controllers and tensor blobs.

## [0.4.1] — 2026-10-28

Tag `saturn-pub-v0.4.1` (ddb6617).

- Qualified symbols joined to branch-bound dataflow slices and findings (`saturn_pub.symbols`).
- Execution reused by mdb and persisted investigations; MDB state tools and Jacobian workspace published on top of the public runtime.
- circuit-tracer interop reads selected activations through `active_features`; streamed Qwen `create_causal_mask` made robust across transformers versions.

## [0.4.0] — 2026-10-02

Tag `saturn-pub-v0.4.0` (79e7a11).

- Block-streamed residency for the decoder language-model adapters, with offline streamed==resident tests.
- Native Gemma-2 decoder adapter (soft-cap, sliding window, GeGLU) and a version-robust embedding scale.
- The metadata-delta route: select a measured delta before hydrating.
- `saturn_pub.certificate`: fail-closed induction-circuit certificate.
- circuit-tracer interop (`saturn_pub.interop.circuit_tracer`): native edge test, graph-vs-native agreement, group interventions, precomputed selection and carrier-path cap.

## [0.3.0] — 2026-10-02

Tag `saturn-pub-v0.3.0` (465e1ae).

- `saturn_pub.trial`: offline verifier and consumer-gated arbitration; redacted, re-pinned Instrument Trial receipt bundle.
- Stdlib-only evidence subpackage wired into the CLI, with offline evidence-plane tests.

## [0.2.0] — 2026-09-30

Tag `saturn-pub-v0.2.0` (f00e149). Initial public release of saturn-pub. Earlier history: see `git log`.

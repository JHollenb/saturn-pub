## What and why

<!-- One paragraph. Link the issue it fixes, if any. -->

## Package(s)

- [ ] `saturn-pub` (`src/saturn_pub`)
- [ ] `mdb-runtime` (`packages/mdb`)
- [ ] `mdb-lanes` / `mdb-experiments` / repository tooling

## Tests run

<!-- The test files you ran, from where, and in which environment (Python, torch, transformers). Targeted tests are expected; CI runs the full suites. -->

## Identities and outputs

- [ ] No serialized identity changed (record ids, fingerprints, receipts, schema names, addresses), or the change is intended and listed under `Changed (breaking)` with its migration
- [ ] Engine, custody, record or adapter change: bit-identity evidence included (hello-tiny record id, golden transcripts, `mdb.engine.selfcheck --tiny`)
- [ ] No verdict-affecting setting became readable from a config file

## Docs and changelog

- [ ] `packages/mdb/CHANGELOG.md` `[Unreleased]` entry for user-visible mdb changes
- [ ] Docs updated in this PR; mdb user docs carry no job ids, internal issue ids or `mdb-experiments` paths
- [ ] Version numbers untouched (the owner sets versions), or the owner asked for this bump

## Real-model check

- [ ] Not needed (tiny-model mechanics only)
- [ ] Needed: describe what a maintainer should run on real weights

See [CONTRIBUTING.md](../CONTRIBUTING.md).

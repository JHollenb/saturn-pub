# The metadata-delta route

A research run produces many candidate interventions — a different measured
delta at every site, stream, step, and family — and each one's heavy bytes live
somewhere far from the metadata that describes it. The metadata-delta route lets
you keep the catalog small and *choose before you pay*: query which delta to
apply from content-addressed metadata alone, and only then hydrate the one you
picked, verify its bytes, apply it, replay against the native run, and roll it
back exactly.

`saturn_pub.route` is a thin layer over this repository's existing lifecycle. It
does not repeat `saturn_pub.program` (which binds one measured arm's ordered Acts
to one parent and replays it). It adds the two things Program lacks: a
payload-free catalog of *many* candidate deltas with selection before hydration,
and a native-comparison-plus-exact-rollback receipt.

## `DeltaCard` — a measured delta saved as metadata

A `DeltaCard` is a tensor-free, content-addressed description of one measured
intervention delta:

- `recipient` — the identity it was measured against (`model_identity`,
  `execution`, `boundary`, `parent_fingerprint`);
- `tags` — a free-form selection coordinate (`family`, `site`, `stream`,
  `step`, `role`, or anything else), the only thing `select` reads;
- `operation` — a builtin `add`, `replace`, or `zero` on one state address;
- `artifact_refs` — for `add`/`replace`, the content address of the heavy delta
  tensor: `sha256`, `shape`, `dtype`, and a `source_handle` telling a resolver
  where to fetch the bytes. **The bytes themselves are never embedded.**

`DeltaCard.from_act(parent=..., act=...)` saves a measured builtin-operation
`Act` bound to a captured cut; the delta's content address is read from the Act's
own sealed parameters, so the card and the Act agree by construction. A card is
fingerprinted, round-trips through `to_dict`/`from_dict`, and refuses any
tensor-like payload in its metadata.

## `MetadataDeltaRoute` — select before you hydrate

A route indexes many cards by tag. `select(**tags)` returns matching cards from
metadata only — no artifact bytes are touched, and the move is counted. The
compact metadata projection reproduces a dense scan exactly (`project` is a
deterministic, payload-free view used for parity checks).

`hydrate(card, resolver)` is the only verb that touches bytes. The `resolver` is
caller-owned, so the catalog works against an in-memory store, a local
content-addressed directory, or a remote object store without the route owning
custody. Each resolved value is verified against its reference by **exact content
address** (sha256) plus shape and dtype before it is admitted, and repeated
roles are served from cache. `receipt()` reports the custody accounting —
`metadata_queries`, `artifact_requests` vs `artifact_hydrations` vs
`artifact_cache_hits`, `declared_bytes_requested` vs `hydrated_bytes`, and the
transport reduction — and always carries `raw_payloads_embedded: false`.

## `apply_delta` — replay vs native, roll back exactly

`apply_delta(session, acts, continuation_steps=..., evaluator=...)` forks a
candidate and a native branch from one parent, applies the hydrated delta,
continues both, compares them, and restores the candidate to the parent cut. It
reuses `Session.apply` / `continue_` / `compare` / `restore`; it does not
reimplement them. The returned route-execution receipt records whether the
candidate `changed_vs_native` and, under `rollback`, `verified_exact`,
`matches_parent`, and `restored_elements` — the per-element count of the exact
rollback.

## CLI

`saturn-pub route validate ROUTE.json` prints the custody receipt;
`saturn-pub route select ROUTE.json --tag family=qwen2 --tag step=0` lists the
selected card ids and their projection; `saturn-pub route show ROUTE.json CARD`
prints one card. All three are stdlib-only and touch no bytes.

Honest claim boundary: this is controlled mechanics. A route selects and applies
a measured delta and proves exact custody and exact rollback. It does not assert
that the delta is semantically meaningful or that two model families are
equivalent. The native downstream consumer stays the behavioral authority.

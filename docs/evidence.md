# The evidence plane

The evidence plane makes the experimental record itself a queryable,
self-auditing object: search backward from a symptom to the cut where it
started, hash-chain a claim to the evidence receipts it rests on, and index
every address a receipt cites so "who mentions this?" is one query, not a grep.

Everything in `saturn_pub.evidence` is standard-library only. It imports neither
`torch` nor the rest of `saturn_pub`; a live run is reached only through caller
callbacks. All four verbs share one custody idiom (`saturn_pub.evidence._canonical`):
canonical JSON — sorted keys, compact separators, `allow_nan=False` with
non-finite floats sanitized to `"NaN"`/`"Infinity"`/`"-Infinity"` strings — and
sha256 hash chains with tamper-evident re-verification. Equal content always
hashes to equal bytes, so a saved receipt re-verifies without re-running
anything.

Honest claim boundary: these are mechanics. A bisect receipt localizes *where* a
signal changes, not *why*; a citation records that something *was cited*, not
that the citation is correct; the index records *where* an address appears, not
what it means. The native downstream consumer stays the behavioral authority.

## `bisect` — search backward to the first divergent cut

`saturn_pub.evidence.bisect` runs binary search over an ordered lattice of
`CutHandle`s with a caller-supplied probe, and seals every probe into a
hash-chained ledger. Three verbs share one engine:

- `first_true(cuts, predicate)` — the earliest cut where a monotone predicate
  fires;
- `first_divergence(cuts, digest_a, digest_b)` — the earliest cut where two
  exact-replay arms stop agreeing, compared through content digests;
- `splice_bisect(cuts, splice_score, threshold)` — the smallest cut whose
  transfer score crosses a threshold.

Each lattice position is probed at most once, and probes are cached, so an
expensive native replay runs a logarithmic number of times. The result carries
`probe_count` against `linear_probe_count` (the lattice size); `probe_economics`
reports the saving. A non-monotone lattice is a *finding*, not an error:
`monotonicity_violation` is recorded on the result, never raised — a divergence
that "heals" by the last cut, or a predicate that is true then false, is surfaced
rather than hidden.

### Bisecting two `saturn_pub` branches

The callbacks are the only seam, so the bisect never imports the rest of the
toolkit. To locate where a native and a perturbed branch first diverge, build a
`CutHandle` lattice over the step index and pass a probe that replays each
retained `StateCut` arm (or `ReplayBundle` head) to a cut and returns its
fingerprint:

```python
from saturn_pub import Act
from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.evidence import first_divergence_pairs, make_cuts, probe_economics

adapter = QwenAdapter.tiny()
session = adapter.session([5, 7, 11])
session.continue_(2)
parent = session.capture()  # the cut both arms share at step 0

def digests(cut):
    native = session.fork(parent)
    perturbed = session.fork(parent)
    perturbed.apply(Act.zero("hidden"))
    if cut.index:
        native.continue_(cut.index)
        perturbed.continue_(cut.index)
    return (
        native.capture(retain=False).fingerprint,
        perturbed.capture(retain=False).fingerprint,
    )

result = first_divergence_pairs(make_cuts(24, prefix="decode-step"), digests)
print(result.first_true_index, probe_economics(result))
```

`first_divergence_pairs` is the integration form: one callback returns both arms'
digests for a cut, so an expensive replay is not run twice per position.
`examples/evidence_bisect.py` is the complete runnable version (measured offline:
24-step lattice, first divergence located in 7 probes vs 24 linear reads).

For multi-step routed systems, emit one cut per execution boundary in execution
order; `first_divergence` is only meaningful when the two arms share the intended
baseline and the lattice follows execution order.

### Verify a bisect receipt

```bash
saturn-pub evidence bisect demo lattice.json --out receipt.json
saturn-pub evidence bisect verify receipt.json   # exit 1 on a broken chain
```

`verify_ledger(probes)` re-hashes a saved probe chain and returns `False` (never
raises) on any tamper or broken link.

## `claims` — hash-chain a claim to its evidence

`saturn_pub.evidence.claims` is an append-only, hash-chained JSONL ledger of
typed claims (`measured` / `asserted` / `refuted` / `qualified` / `stale`). Each
row records the receipt it came from, an optional replay recipe, and a
dependency fingerprint over the exact bytes the claim rests on. `verify`
recomputes those fingerprints and flips drifted claims to `stale` — exactly once,
with restoration tracked — so a claim whose inputs changed stops presenting
itself as current evidence. A sidecar head pointer guards against silent tail
truncation; concurrent appends are serialized.

```bash
saturn-pub evidence claims ingest report.json \
  --claim-id my-claim --text '...' --registry claims.jsonl
saturn-pub evidence claims verify --registry claims.jsonl          # appends stale rows
saturn-pub evidence claims verify --check --registry claims.jsonl  # read-only: 0 ok / 3 drift / 2 chain broken
saturn-pub evidence claims history --claim-id my-claim --registry claims.jsonl
```

`--check` exists because CI must never mutate the ledger it judges.

`--harvest-declared` additionally ingests the `(path, sha256)` reference pairs a
receipt itself declares, tagged by hash semantics (`raw-bytes` vs
`canonical-json-seal` vs `unknown`) and by resolution (`absolute` / `job-dir` /
`base-root` / `missing`). Resolution is fail-closed: a candidate binds only by a
digest match, or — for an existing absolute path — by path identity, so a bare
basename never false-binds the wrong file. A reference that never resolved is
counted `unresolvable`, never `stale`: a dead pointer is not drift. Drift means a
*resolvable* fingerprint changed.

## `xref` — index the addresses receipts cite

`saturn_pub.evidence.xref` sweeps a tree of `*.json` receipts into a SQLite index
of every concrete `ar://` address occurrence, with the file's `path_class`
(report / run-receipt / observation / other) and best-effort model attribution
from the nearest enclosing model descriptor (`NULL` when none is found — never a
guess). Files over 256 MiB land in an explicit `skipped` table; a build over a
missing or empty root fails closed rather than purging the index.

```bash
saturn-pub evidence xref build --root results/ --db xref.sqlite
saturn-pub evidence xref query ar://decode/step/1/site/hidden --db xref.sqlite --dedup
saturn-pub evidence xref stats --db xref.sqlite --json
```

`--dedup` collapses report / run-receipt mirror pairs (the report row wins),
because mirrored files otherwise double every concentration count.

## `cite` — bind model and input at write time

Addresses alone are ambiguous — the same string can name a site in a 24-layer and
a 28-layer model. `saturn_pub.evidence.citations.make_citation` emits an
index-ready citation dict (marker key `saturn_pub_citation`) that binds
`model_id` / `revision` / `input_sha256` to a concrete address at write time;
the xref walk trusts it over best-effort structural attribution.

```bash
saturn-pub evidence cite make ar://decode/step/1/site/hidden \
  --model-id Example/Model-0.5B --quantity first-divergence
```

The citation grammar is kept in sync with the xref extractor by a test, so a
valid citation always indexes verbatim.

## Contracts the evidence plane keeps

- **Bind model and input to addresses.** Address strings alias across models and
  inputs; use citations.
- **Dead references are `unresolvable`, not `stale`.** Drift means a resolvable
  fingerprint changed.
- **Mirrored files double counts.** Deduplicate by path class before any
  concentration claim.
- **Verification writes must be opt-out-able.** `verify --check` exists because
  CI must never mutate the ledger it judges.
- **Non-monotone lattices are findings.** The bisect records the violation
  instead of pretending the search was clean.

The subpackage is intentionally self-contained (stdlib only, no `saturn_pub`
imports outside its own namespace) so it can later move into a neutral base
package shared by more than one toolkit.

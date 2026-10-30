# Discovery upgrades validation, 2026-10-30

Sol agents implemented physical native screening, durable frontiers and the
remote controller in parallel. Astra advised on ownership, replay and scientific
scope. Root integrated the installed CLI, qualified R3 execution and profiling.

## Executed qualification

`job-a61b5618e8db` succeeded in one finite mrun lease on Beast, CPU FP32,
one thread, frozen checkpoint
`9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e`.
Job elapsed time was 34.82 s; the instrumented worker took 21.33 s, including
client wait. Process-tree peak RSS was 1631.7 MB under the explicitly measured
2304 MB reservation. The native model prehook counted **27 physical forwards**
against a ceiling of 64. No training, oracle state or automatic retry was used.

The worker used installed wheels and sealed ISA sources. Offline analysis
verified all **169 MDB Python modules** were identical between the source tree,
the isolated local installation and the executed payload wheel. The retained
[source-bound analysis](discovery-upgrades-20261030.json) includes these hashes,
request/report/trace identities and raw measurement summaries. Its analyzer is
[analyze_discovery_upgrades.py](analyze_discovery_upgrades.py); the actual model
harness is [discovery_upgrades.py](discovery_upgrades.py).

| Feature | Observed result |
|---|---|
| Native physical batches | Actual B4/B3 calls; reordered rows, parked inputs, epoch refill and per-row output Acts checked against scalar controls |
| Batch throughput | Four rows: 11.85 ms scalar / 5.40 ms batched; reversed-order round: 10.16 / 5.13 ms; observed ratios 2.19× and 1.98× |
| All-head batch differences | Maximum absolute difference 2.288818359375e-5; within the declared 1e-4 absolute + 1e-5 relative numeric profile |
| Durable selected frontier | Fresh installed CLI started at actual `post.1` exit PC375 and executed only 205 remaining primitives; all 11 heads exactly matched native capture and separate unchanged native execution |
| Durable aliases | Engineering tests reconstruct shared storage, offsets, strides, transposes and zero-stride views; no pickle |
| Remote CLI | 11 commands: alias, capture, fork, select, zero-output Act, update/forward motion, inspect, instruction commit and close |
| Native intervention audit | All 11 edited replay heads exactly matched a separate hooked native forward |
| Remote retry | Same logical request/content returned the identical result without extra work; changed content with that ID was rejected through the actual deployed HTTP broker |
| Offline journals | 12 remote and 6 resume journal records verified |

Full native screening and scalar replay are different adapters. Component
breakpoints select scalar replay; B4 intermediate values are never inserted
into a B1 trace. Numeric B1/B4 agreement does **not** establish complete decoded
B1/B4 parity: those control summaries retain numerical comparisons rather than
all decoded decisions. Selected causal branches have exact scalar/native audits.
The first eager capture runs to completion before component replay can pause.

The saved native trace has 580 primitives. Its frozen `post.1` is `mlp_003`,
entry 365/exit 375, output `v000436`, bound to program
`0c565fe37bdd4e92f2a783269aa02ad8bf85379000b2ea8800c62573fe6fe129`.
Earlier qualification used a different grad-mode trace with 359/369 coordinates.
Use `module:post.1` and resolved site identities, not global numeric addresses.

## Measured bottleneck and correction

The original per-storage layout spent **67.297 s** saving one selected session
frontier. Its profiled capture/replay/save block made 988 `fsync` calls, consuming
64.218 s of self time. Default `bundle-v1` packs storage objects through the
existing public StateCut/LocalStore tensor codec. Actual file and directory
syncs remain intact; the session manifest still publishes after its native cut.
The same profiled block now makes 12 sync calls (1.468 s self), and saving the
session took **3.569 s**: an observed 18.86× reduction in save latency.

This is one cold before/after observation on the same host/profile, not a
randomized throughput benchmark. The earlier baseline later failed in CLI
dispatch; its completed save/profile remains valid baseline evidence.
Bundling trades fine-grained cross-cut deduplication for fewer publications.
Legacy `slots-v1` cuts remain readable and the writer is still selectable.
The [bottleneck guide](../docs/DISCOVERY-BOTTLENECKS.md) separates completed fixes
from further proposals.

## Exploratory variants

The exact byte window was `4881c7010000004881c701000000`; initial RDI was
252, 253, 254 or 255, RSI 3, other registers and flags zero. Raw bytes and
before-state were supplied, with no text prompt or semantic operation label.
Four contexts crossed four supplied `post.1` output scales: 0, 0.5, 1, 1.5.
The 16 rows took four physical B4 forwards and 26.35 ms including adapter work.

Across the four contexts, every nonidentity dose changed the same five
continuous heads: `regs`, `flags`, `imm`, `store_addr`, `store_value`.
The fresh trace's descendant graph independently reached exactly those five
heads. Six remaining heads were byte-identical. Decoded registers, flags,
length, store and syscall decisions stayed equal to their within-context
identity-dose result. Flag effects reached 34.796 logit units, with every retained
flag logit still at least 22.687 from its zero decision boundary.

This supports a consumer-incidence pattern and context-dependent numerical
response. It does not name an ISA operation, establish necessity, prove
generalization, or establish a universal MLP law. All full heads, margins,
input identities and physical-row mappings are retained for later contrasts.

## Failures, collection and limits

All earlier attempts remain immutable:

- `job-3be1d484619f`: batch/variant evidence completed; the harness incorrectly
  assumed old trace coordinates after freezing gradients. Fixed by catalog
  resolution rather than hardcoded event numbers.
- `job-f568e9a320d8`: old-layout durable save/profile completed; CLI resume
  exposed an argparse destination collision between the subcommand and repeated
  `--command`. Fixed and covered by installed CLI regressions.
- `job-7041ca178eb7`: bundled save and exact fresh-process suffix passed; remote
  registration lacked the public scheduler URL. The deployed agent predates
  automatic URL injection. The final sealed request explicitly supplies that
  endpoint while mrun continues to inject its narrow debugger credentials.

Final installed-wheel tests run outside the checkout with isolated imports:
**152 passed** across discovery, batch/storage/remote/CLI, bounded admission,
legacy protocol and distribution coverage. Ruff and whitespace checks passed.

MinIO publication reported `failed-queued` because its endpoint was unreachable.
The collector recovered the complete archive from worker logs and verified its
SHA-256 before extraction. Raw requests, logs, wheels, archives, cuts, profiler
rows and both client journals live in the domains workspace at
`artifacts/mdb-discovery-upgrades-20261030/`. No remote object-store publication
or general evidence-plane certification is claimed.

Durable restore selects one lane/epoch/branch. It retains actual work and budget
history; a new wall allowance requires explicit admission and all finite
ceilings. Completing saved heads needs no factory; another native forward
requires a caller-supplied factory checked against model/program/input identity.
Retries survive only while their worker is alive. RAM/OS effects still stop
before commit. This assay spans four contexts of one byte program and one MLP.

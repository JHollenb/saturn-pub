# Make route discovery cheaper and more informative

The current bottleneck is mostly research orchestration and evidence handling.
Four native inputs take about 5 ms to screen together; capture, durable saving,
fresh-process restoration and remote interaction each take seconds. The first
large improvement is implemented. The next improvements should amortize those
costs and choose contrasts that discriminate among explanations.

Implementation: [`562209f`](https://github.com/JHollenb/saturn-pub/commit/562209fc56d563cd345568c5ec04b10ff9656a87).
Measurements come from `job-a61b5618e8db`, CPU FP32/thread 1, frozen R3, with
the earlier completed save in `job-f568e9a320d8` as the baseline.
See [validation](../qualification/DISCOVERY-UPGRADES-VALIDATION.md), the
[source-bound rows](../qualification/discovery-upgrades-20261030.json) and
the [executed workflow](ROUTE-DISCOVERY-WORKFLOW.md).

## Measurements and proposed fixes

| Cost or inefficiency | Evidence | Action and status |
|---|---|---|
| Durable storage fan-out | Old save 67.30 s; 988 sync calls/64.22 s self in its profiled block. Bundled save 3.57 s; 12 sync calls/1.47 s self | **Implemented:** bundle backing storages through public CAS, preserving file/directory sync and manifest ordering. Observed save ratio 18.86× |
| Scalar native screening | Four scalar calls 10.16–11.85 ms versus one B4 call 5.13–5.40 ms | **Implemented:** native B4/B3 panels, row mapping, per-row Acts, parking/refill, scalar controls. Observed throughput ratio 1.98–2.19× |
| Cold capture/setup | Capture/create 4.03 s; profile includes 960 compiles costing 1.46 s self | **Next:** warm imports once in a finite resident lease; reuse frozen model bindings and immutable catalog/trace descriptions. Measure first versus subsequent capture separately |
| Serialization and metadata traversal | New profile still records 182,827 JSON `iterencode` calls; value `describe` has 1.58 s cumulative | **Next:** compact references to immutable trace/program/weights, bulk JSON encoding, cache descriptions keyed by immutable identity. Measure bytes/CPU per phase before altering the schema |
| Prefix and snapshot work | Prefix replay to MLP exit 261 ms; fresh-process restore plus suffix/commit 2.36 s | **Available:** fork/replay from a retained parent and restore a saved frontier. **Proposed:** retain only continuation-live tensors while storing historical observations separately |
| Command transport | Remote residency 9.84 s includes client waiting, reconnect and CLI startup | **Measure first:** command enqueue, transport, execution, response and idle durations. **Proposed:** submit a bounded panel/script once for worker-side execution rather than requiring a round trip per command |
| Repeated inputs and weak contrasts | 16 rows cover four contexts at one MLP; all retain the same decoded decisions | **Next:** cache exact repeated experiments, choose response-diverse contexts, include near-boundary contrasts, compare other MLPs and log coverage independently from replay count |
| Manual evidence joins | Full outputs, graph, journals and cuts exist; general atlas/evidence-plane ingestion is incomplete | **Next:** one indexed variant record joining those immutable artifacts, plus explicit incomplete/failed-arm statuses |

These timings have different scopes and must not be added as though each were
pure model compute. Cold before/after storage timing is one observation rather
than a randomized benchmark. Two opposite-order native timing rounds establish
a useful small-run trend, not a fleet-wide optimum. Hashing was about 51 ms
self in the new profile; copying was about 170 ms cumulative. Neither was the
original dominant cost. Do not remove identity checks to save those milliseconds.

Bundling reduces deduplication granularity: changing one storage may create a
new whole tensor bundle. A measured next storage variant could separate
immutable model roots, per-input roots and changing frontier storages into a
few bundles. Preserve complete alias groups across those boundaries. First
measure repeated warm saves and actual duplicate bytes; keep old cut formats
readable. The current implementation does not yet make this partition.

## Increase useful discoveries per forward

Use this finite loop:

1. **Enumerate structural sites and possible consumers.** Resolve generic
   MLP/math/update IDs from one sealed native trace. Keep the bit-axis scan,
   ATen clock and architectural instruction clock distinct.
2. **Screen native variants.** Batch compatible inputs and per-row doses. Keep
   identity-dose rows and all continuous heads, margins and physical mappings.
   Report component-stopped rows for scalar execution instead of bypassing stops.
3. **Choose contrasting responses.** Select cases differing in sign, dose shape,
   affected consumer, collateral or state continuation. Include stable and
   near-miss cases. Four repetitions of one context are not four contexts.
4. **Capture a scalar common parent.** Fork only the selected intervention
   branches, edit a typed endpoint and run its actual suffix. Reuse that parent
   for matched contrasts. Audit selected results with an independent native hook.
5. **Ask one discriminating causal question.** Restore an intermediate carrier,
   exchange one source or receiving condition, or compare a second MLP. Record
   alternatives and discrepancies instead of letting one scalar erase them.
6. **Continue on predicted state.** Compare current heads with later consumed
   state. A current numerical effect, a decision flip and persistent influence
   across fresh model calls are different observations.
7. **Append records and select the next cut.** Join source, carrier, writer and
   consumer evidence; rank the next reversible experiment by unresolved
   alternatives and expected execution cost.

No discovery score should require that every head or context changes. Failed
arms retain their work and outcomes. A broken instrument is recorded separately
from an absent or weak effect. Mechanical checks still enforce identities,
row ownership, finite execution and valid continuation.

## The next informative panel

The current 16-row panel scales `post.1` by 0, 0.5, 1, 1.5 across RDI 252–255.
All nonidentity doses affect `regs`, `flags`, `imm`, `store_addr`, `store_value`;
the exact fresh graph reaches those same five heads. Six other heads remain
byte-identical. Yet decoded state does not change. Flag responses reach 34.80
logit units, while all flag decisions remain at least 22.69 units from zero.
This is evidence of a numerical route even though a hard-output-only discovery
score would miss it.

Next, qualify matched native screens at `post.0` and `post.2` using the same
contexts/doses. A complete three-site panel has 48 logical rows/12 B4 calls
before optional reuse of identical controls. Predict structural consumer sets
from the trace first; retain disagreements with observed effects. Scalar-replay
the most contrasting responses and use an intermediate restoration to test
which carrier explains the difference. These additional sites have **not** yet
been qualified by this upgrade assay.

Then vary receiving state and input encoding around previously observed carry
contrasts. Hold one source fixed while changing its receiver, or exchange one
source while preserving the receiver. This distinguishes recurring routes,
portable displacement vectors and consumer-specific roles. They should become
separate table fields. A stable route can have context-dependent effects;
similar tensor directions need not produce identical decisions or collateral.

## Generic names and pattern discovery

The current runtime already provides program-bound site IDs, generic names,
aliases, occurrence ownership and immutable receipts. A cross-run indexed atlas
and automatic response-family discovery remain proposed work. The useful record
separates stable structure from an observation:

| Record | Identity and contents |
|---|---|
| Site | Checkpoint/program identity, structural site ID, module, unit, entry/exit |
| Route | Ordered source/carrier/writer/consumer addresses and graph scope; a structural path is a candidate until causally tested |
| Variant | Context/input digest, parent cut, site occurrence, intervention/sign/dose, attempt/row, receiving state, full effects and continuation |
| Symbol hypothesis | Generic ID plus proposed represented variable, applicability, source/consumer and supporting or conflicting variants |
| Family | Versioned grouping of response signatures with membership, uncertainty, exceptions and discriminating follow-up tests |

Keep generic handles such as a site ID or `route_<digest>` usable before any
semantic annotation. Human aliases can describe an association, such as
`post1_to_flags`, without asserting an ISA operation. A symbol hypothesis needs
evidence for a represented variable and its use; a tensor changing is not by
itself a named symbol. Store observations separately from interpretations so
later naming revisions do not rewrite raw records.

Cluster on **causal response signatures**: signed dose curves, consumer
incidence, receiving context, collateral and continuation. Include residual
reconstruction error and counterexamples when fitting a candidate equation.
Raw activation similarity can help propose a match; it does not establish
functional equivalence. Cluster membership alone does not prove a law.

There is no universal count at which we have enough variants. Track independent
contexts, sites, interventions, consumers and predicted-state horizons, along
with uncovered cells and contradictions. Fit candidate rules on discovery
contexts, predict a new contrast, and test it through the native consumer.
Keep the prediction and its errors. The best next row often separates two
explanations rather than increasing an existing cluster's sample count.

## Where graph algorithms help

Use static reachability to enumerate possible consumers; join it with observed
intervention responses. The current match between five structural descendants
and five affected heads is a bounded example. It does not establish which
intermediate path carries the effect, or that every reachable node is necessary.

Use dominators and min cuts to propose compact interception sets, then validate
them by typed interventions and restoration. Declare capacities explicitly—for
example, the execution cost of testing a cut or the number of independent
channels intercepted. Activation energy is not automatically a flow capacity.
Max-flow values on that chosen graph describe the graph model, not an intrinsic
amount of semantic computation.

Community/spectral methods can find repeated route shapes or response families
across contexts. Map those candidates back to exact source/carry/write/read
addresses and keep signed effects outside a nonnegative capacity model. The
graph chooses experiments; the native receiving computation supplies evidence.

## Engineering order

The next engineering slice should add phase telemetry and a finite worker-side
panel command, warm resident setup, and indexed variant output with generic
identities. Benchmark useful retained variants per second and per physical
forward, including setup, save and audit costs. Then optimize metadata and
snapshot retention using the new measurements. Expand batch/backend profiles
only when they are a measured bottleneck and retain scalar scientific audits.

That order improves both throughput and the quality of the table: fewer repeated
prefixes and manual joins, more controlled variants, and clearer evidence about
when a route or candidate rule applies.

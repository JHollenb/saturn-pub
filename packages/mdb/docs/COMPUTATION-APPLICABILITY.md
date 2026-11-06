# Choose a computation from source conditions

MDB now exposes an opt-in, source-only planner:

```python
from mdb.computation_applicability import evaluate

plan = evaluate(source_packet, intervention, execution_profile,
                output_spec, policy, coverage_index=coverage_index)
print(plan["selection"])
print(plan["decision_sha256"])
```

It reads no files, tensors or models. It fixes the decision before annotating
historical evidence. It rejects target/error/pass fields. Actual outcomes remain
`not-read-or-executed`; the existing numerical producer and saved reader supply
separate execution and evidence accounts.

The initial domain is deliberately narrow: the original inherited Qwen block18
packets from the [row-sign](QWEN-ROW-SIGN.md) and [cached-QKV](QWEN-CACHED-QKV.md)
experiments, original parameters, BF16 CUDA, all512 attended rows, original
positions, no cache, exact declared sources and flags. Descriptors are bound to
known metadata digests. This is no automatic generalization to another prompt,
layer, mask or GPU. The declared profile does not attest driver, architecture or
selected kernel; an executing runner still needs actual payload/device/custody
validation and hardware binding.

## What the installed replay did

The installed wheel passed 29 focused mechanics checks. Four actual metadata
replays and 28 metadata controls then ran without array/weight/outcome reads.
The normalizer authenticated two compact wrappers and 24 allowlisted metadata
members. It did not open numerical audits, target arrays or images.

| Fixed source and edit | Closed queries | Cross-sign permitted edges | Selected method |
|---|---:|---:|---|
| R/G/U packet, all negative | 512 | 0 | m0: cheap sign relation |
| R/G/U packet, prefix negative | 256 | 65,536 | m2: full functional block fallback |
| QKV packet, last row negative | 511 | 511 | m1: full-query cached-QKV recomputation |
| QKV packet, alternating64 | 64 | 65,536 | m1: full-query cached-QKV recomputation |

Every selected path retains all512 queries. No shortened attention batch or
cross-experiment port union occurs. The R/G/U packet has no cached QKV; the QKV
packet has no G/U. Full fallback requires explicit availability, all original
recipes and the same qualified declared domain. Missing response baselines
cause abstention rather than silent positive-state computation.

Changed source/profile/cache produces 12 abstentions across the controls.
Removing prediction ports produces four full-block fallbacks; removing a response
baseline produces four abstentions. Four unqualified continuation bindings leave
the local method unchanged and the continuation unqualified. Disabling fallback
abstains only on the prefix case. These are deterministic metadata checks,
not 28 new model experiments or numerical accuracy measurements.

Historical occurrence lookup matches three selected methods. The prefix case has
no exact m2 occurrence in this index: its old m0 counterexample remains m0. Prior
full-functional reference evidence is separate; the index never relabels that
failed shortcut or supplies selection authority. All four decision hashes are
unchanged when the coverage index is added.

## Replay the shipped metadata

Install MDB including this module. From `packages/mdb`, execute:

```bash
python -I -B docs/assets/qwen-applicability-20261106/prepare_packets.py replay \
  --inputs docs/assets/qwen-applicability-20261106/packets-v1 \
  --out /tmp/mdb-applicability-replay-new \
  --expected-input-manifest fa30ac0218f859a96df10ea80095f6211493c310422774fe442e9255497c61ca \
  --expected-module-sha256 820e971a6128aa7f49cd169734fd2a123e8a60924b25a6d08f6d0a46d33996f3
```

Use a fresh output directory. The command requires the installed source hash
and fails rather than importing a checkout fallback. It writes complete plans,
all28 controls, the count-only coverage index and a replay receipt. Root also
executed this command against the copied public asset layout.

Inspect the [plans](assets/qwen-applicability-20261106/replay-v1/plans.json),
[controls](assets/qwen-applicability-20261106/replay-v1/controls.json),
[read manifest](assets/qwen-applicability-20261106/packets-v1/read-manifest.json),
[independent review](assets/qwen-applicability-20261106/replay-review-v1.json),
[installed qualification](assets/qwen-applicability-20261106/installed-qualification-v1/qualification.json)
and [actual execution](assets/qwen-applicability-20261106/execution-v1/execution.json).
The [packet ABI](assets/qwen-applicability-20261106/packet-abi-v1.md) describes
the five input objects and returned schema. The [source](../src/mdb/computation_applicability.py)
imports only standard-library code and the existing supplied-condition matcher.

The symbol-table entry now has an explicit local applicability decision beside
its immutable computation and evidence records. Its finite numerical coverage,
raw-bit limitations and consumer continuation remain separate. The next original
experiment will send the retained, nonidentical cheap and corrected H19 states
through block19 after sealing both H20 predictions.

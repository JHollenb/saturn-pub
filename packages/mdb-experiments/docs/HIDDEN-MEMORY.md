# Reading and writing hidden memory

MDB exposes committed Qwen K/V rows as readable, writable model memory. A loaded `BenchSession` can show the token attached to each row, bounded numeric K/V contents, logical and physical addresses, and compact tensor summaries. It can atomically copy, zero, scale, or replace a selected layer/row/component rectangle, continue through the unchanged native model, compare two cuts, and restore the parent.

The current implementation covers Qwen2 sessions in journal publication mode at committed token boundaries. Addresses are native layer, token-row, and K/V coordinates. They are useful causal operands; they are not a universal decoder for model thoughts.

## One complete investigation

Run this inside the admitted mrun worker that already owns the resident model and tokenizer. Do not load a second copy for the debugger. The example uses the BF16 CUDA Qwen2.5-1.5B lane being qualified for this surface; pass the actual pinned checkpoint revision and durable worker-local root.

```python
from pathlib import Path

import torch

from mdb.bench import BenchSession

# `model` and `tokenizer` are caller-owned resident objects supplied by the worker.
model_id = "Qwen/Qwen2.5-1.5B"
revision = checkpoint_revision
assert next(model.parameters()).is_cuda
assert next(model.parameters()).dtype == torch.bfloat16

with BenchSession.from_resident(
    model,
    tokenizer,
    root=Path("/durable/mdb/hidden-memory-demo"),
    model_id=model_id,
    revision=revision,
    session_id="hidden-memory-demo",
    lane="virtual",
    publication_mode="journal",
) as bench:
    bench.prefill("Paris is the capital of")
    parent = bench.checkpoint("parent")

    selector = {
        "layers": [6],
        "row_start": len(bench.branch.token_ids) - 1,
        "row_stop": len(bench.branch.token_ids),
        "components": ["key", "value"],
    }

    read = bench.memory_read({**selector, "values": True, "max_values": 8})
    row = read["rows"][0]
    print(row["token_id"], row["token_text"])
    for address, component in row["components"].items():
        print(address, component["norm"], component["values"])

    zero_cut = bench.memory_write({"mode": "zero", "selector": selector})["cut_id"]
    print(bench.memory_diff({
        "left_cut": parent,
        "right_cut": zero_cut,
        **selector,
        "values": False,
    }))

    bench.restore(parent)
    scaled_cut = bench.memory_write({
        "mode": "scale",
        "value": 0.5,
        "selector": selector,
    })["cut_id"]

    bench.restore(zero_cut)
    restored_cut = bench.memory_write({
        "mode": "copy",
        "donor_cut": parent,
        "selector": selector,
    })["cut_id"]
    restored = bench.memory_diff({
        "left_cut": parent,
        "right_cut": restored_cut,
        **selector,
        "values": False,
    })
    assert all(
        component["equal"]
        for item in restored["rows"]
        for component in item["components"].values()
    )

    bench.restore(parent)
    probe = bench.memory_probe({
        "name": "half-dose-fixed-query",
        "intervention": {
            "mode": "copy",
            "donor_cut": scaled_cut,
            "selector": selector,
        },
        "query": " France or Germany?",
        "continuation_tokens": 4,
        "top_k": 5,
        "max_native_operations": 4096,
        "max_capture_bytes": 67108864,
    })
    report = probe["report"]
    print(report["native"]["answer_text"])
    print(report["candidate"]["answer_text"])
    assert report["selected_parent_restored_exact"]
    assert bench.branch.cut == parent
```

The probe runs untouched native memory, the candidate intervention, and exact parent-memory reinstall from one frozen recipient parent. Every arm receives the same caller-supplied fixed query and runs the same native greedy consumer. The report retains decoded answers, token IDs, logits hashes, top-token raw logits, endpoint closures, Act receipts, source identity, and finite budgets. To measure a signed candidate-minus-native first-answer margin, add:

```python
"observable": {
    "kind": "token_margin",
    "token_id": france_token_id,
    "against_token_id": germany_token_id,
}
```

The optional raw margin remains distinct from decoded native answers. The exported token IDs and native token history are the offline authority; decoded text is a tokenizer-identity-bound presentation of those IDs.

## Selectors, reads, writes, and diffs

A selector contains distinct native decoder `layers`, a half-open `row_start`/`row_stop` span over committed token history, and `components` containing `key`, `value`, or both.

`memory_read()` defaults to all layers, all committed rows, and both components. Each response returns at most 32 rows by default and 256 at most; `truncated` and `next_row_start` make a longer selection explicit and page it without changing the cut. Numeric output is bounded to 16 values per selected tensor by default and 256 at most. Set `values: false` to return shape, digest, norm, maximum absolute value, token text, and addresses without numeric contents. A read during an in-flight token must name an explicit committed `cut_id`; an implicit current-state read refuses that ambiguous boundary.

`memory_write()` supports these atomic transforms:

| Mode | Payload | Effect |
|---|---|---|
| `copy` | `donor_cut` | Copies the exact selected rectangle from a compatible cut with identical committed token history. |
| `zero` | none | Installs native-dtype zeros. |
| `scale` | finite scalar `value` | Multiplies the selection in native dtype. |
| `replace` | `values` keyed by every selected logical address | Installs exact finite arrays with shape `[1, kv_heads, 1, head_dim]`. |

Writes stage a private complete mount, verify selected-payload equality and masked locality, pass through Model OS authority, and publish a child StateCut only after commit. Any validation or commit failure restores the original branch and event lineage.

`memory_diff()` compares aligned cuts without checking out either cut. The cuts must have identical committed token history. Each component reports exact equality plus left, right, and signed `right - left` summaries, with the same bounded row paging as `memory_read()`.

## Present edits and historical reconstruction

A hidden-memory write changes state at the selected committed boundary. It does not reconstruct the history that originally produced that state. This matters when an earlier value has already affected later K/V rows, recurrent state, generated tokens, or other descendants: copying the old source row back at the current boundary can restore that row exactly while leaving those descendants changed.

`mdb.statecut_intent` records the intended meaning and binds the declared replay schedule. A `present-state-intervention` binds the current execution-history digest and intervention lineage, applies the edit at the observation cut, and plans only the declared future dependency closure. A `historical-counterfactual-reconstruction` additionally requires a rewind cut and an explicit replay schedule from that cut through the observation boundary. Every replay item says whether its input is caller-supplied or model-generated. A run may compose this record after execution when the actual candidate cut fingerprints are only known from the retained source Act and replay receipts; that post-hoc binding does not claim the plan drove execution. The edit-application cut is the source boundary being changed; donor origin belongs in the separate `provenance` field.

```python
from mdb.statecut_intent import (
    HISTORICAL_COUNTERFACTUAL_RECONSTRUCTION,
    ReplayScheduleItem,
    StateCutIntent,
    StateCutRef,
    plan_statecut_intent,
)

intent = StateCutIntent(
    intent_id="rewrite-reasoning-source-and-replay",
    intent_kind=HISTORICAL_COUNTERFACTUAL_RECONSTRUCTION,
    edit_application_cut=source_cut,
    rewind_cut=rewind_cut,             # may be the same cut as source_cut
    observation_cut=pre_answer_cut,
    replay_schedule=(
        ReplayScheduleItem(
            step_id="reasoning-token-1",
            sequence=0,
            input_authority="caller-supplied",
            input_fingerprint=reasoning_token_sha256,
            parent_cut=rewind_cut,
            result_cut=baseline_next_cut,
            dependency_cells=("writer", "store"),
        ),
        # Continue explicitly through pre_answer_cut.
    ),
    model_identity_fingerprint=model_sha256,
    state_schema_fingerprint=state_schema_sha256,
    numerical_contract=numerical_contract,
    dependency_contract_fingerprint=dependency_contract.fingerprint,
    changed_addresses=("reasoning/source",),
    execution_history_fingerprint=visible_history_sha256,
    intervention_lineage=(source_edit_act_id,),
    provenance={"donor_cut": donor_cut_ref},
    native_consumer="qwen.greedy",
    continuation_horizon=8,
)
plan = plan_statecut_intent(intent, dependency_contract)
```

The plan reports its declared dependency closure separately from receipt-bound parent links. An ordered list of cut identifiers declares a lineage; parent links carrying externally authenticated cut-receipt fingerprints enter `receipt_bound_parent_linked_closure`. When a schedule item carries that fingerprint, its execution receipt must name the same receipt. The planner does not turn a bare fingerprint into proof of ancestry. Execution receipts bind the declared cells and numerical contract but do not establish that the dependency graph contains every real dependency. Restore parity compares state at the selected boundary. Replay parity compares the subsequent native future, such as token IDs, logits and EOS. Both remain separate measurements, and neither is inferred from a successful plan.

## Debugger commands

Pause an attached compatible worker at a committed-token boundary before a write or probe. Inline JSON and `@FILE` are accepted:

```text
(mdb) checkpoint parent
(mdb) memory read @read.json
(mdb) memory write @zero.json
(mdb) checkpoint zero
(mdb) restore parent
(mdb) memory write @scale.json
(mdb) checkpoint scaled
(mdb) memory diff @diff.json
(mdb) restore parent
(mdb) memory probe @probe.json
```

For example, `scale.json` is:

```json
{
  "mode": "scale",
  "value": 0.5,
  "selector": {
    "layers": [6],
    "row_start": 4,
    "row_stop": 5,
    "components": ["key", "value"]
  }
}
```

The normal terminal view prints concise rows and signed probe effects. Use `--json` for the complete machine-readable report. The worker resolves `@current`, `@parent`, and checkpoint aliases in cut fields before dispatch. Unsupported families or adapters without native hooks refuse the operation. Writes and probes also refuse a running worker or a pending Act.

## Evidence boundary

The [October 5 Saturn latent causal handoff](https://github.com/JHollenb/saturn/blob/main/experiments/2026-10-05-latent-causal-handoff/FINDINGS.md) is the immediate evidence source for this surface. It retained formation over 14 identities in two contexts, separated K and V effects in a binding panel, and demonstrated two serial hidden-state handoffs in three competent fresh city contexts while visible intermediate tokens stayed fixed. Coherent base-state reinstall restored the original final answer.

Those are bounded causal uses of hidden memory through native consumers. They do not establish a unique address for every concept, universal thought decoding, or an autonomous model-owned reasoning loop. MDB makes the same kind of read, typed edit, paired native continuation, diff, and restore available through a stable standalone runtime surface.

The corresponding deployed Saturn CLI path is qualified by CUDA BF16 job [`job-d4f1e281b27a`](https://github.com/JHollenb/saturn/blob/main/results/hidden-memory-debugger/job-d4f1e281b27a/qualification.json). Native and exact-reinstall arms decoded ` Paris. It is`; all-layer subject-row value deletion produced ` Washington, D.C` with a `-4.8125` Paris-versus-Rome raw-margin shift, full-prefix value deletion produced ` a 20` with `-7.0625`, and a last-layer half-scale retained the answer with `+0.0625`. The [human transcript](https://github.com/JHollenb/saturn/blob/main/results/hidden-memory-debugger/job-d4f1e281b27a/client-recovery/debug-transcript.txt) and [experiment record](https://github.com/JHollenb/saturn/blob/main/experiments/2026-10-05-hidden-memory-debugger/README.md) retain the original client parsing failure, hash-bound same-worker correction, four-record/three-unique-panel variance, exact restoration, physical work ceilings, and offline closure.

## Integrity and budgets

Treat the parent cut, specimen identity, numerical contract, selector, intervention, fixed query, native consumer, and continuation horizon as part of every claim. A probe accepts 1–64 fixed query tokens and 1–32 continuation tokens. It rejects work above the caller's native-operation or capture-byte ceilings, non-finite input, context overflow, malformed selectors, incompatible donors, and non-journal sessions.

After all three probe arms, MDB restores the exact selected parent pointer and closure and verifies that sibling branch selections did not change. Exported reports bind the spec digest, Act and reinstall receipts, endpoint cuts, native steps, raw outputs, and source identity for offline verification.

See [MDB usage](USAGE.md), [experiment walkthroughs](EXPERIMENTS.md), and [qualification](QUALIFICATION.md) for session setup, retained evidence, and model-backed scope.

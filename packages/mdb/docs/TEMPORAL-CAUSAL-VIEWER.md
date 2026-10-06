# Dual-time causal path viewer

`mdb.temporal_causal_viewer` exports existing circuit evidence as one standalone,
offline HTML file. Token time is the horizontal axis and layer depth is the vertical
axis. Cells show recorded producer, carrier, read, commit, native, intervention and
pass-diff events using the vocabulary already present in the source artifact. Empty
cells remain visibly `missing`; a missing cell is never reported as a null effect.

```bash
python -m mdb.temporal_causal_viewer \
  circuit-report.json circuit-thought-trace.json \
  --output temporal-view.html
```

Supported inputs are:

- `saturn-bench-circuit-measurement-v1` reports or their `{report: ...}` result wrapper;
- individual `saturn-bench-circuit-observation-v1` rows;
- `saturn-thott-circuit-thought-v1` traces;
- `saturn-pass-causal-state-diff-v1` reports;
- `mdb-dual-time-causal-events-v1`, the generic typed event interchange described below.

Inputs are bound to a single model identity and numerical contract. Conflicting
bindings fail closed. Measurement rows must share their report's parent cut. The
viewer calls a non-native measurement row `causal_evidence` only when it is a
same-parent row, the native consumer ran, and a signed future effect exists. Native
baseline rows, thought-trace events and pass-diff
events remain observations. Their geometry, digests, receipts, cuts and payload
summaries remain inspectable in the cell detail pane.

The generic event adapter lets an experiment expose its own stages without importing
MDB runtime code:

```json
{
  "schema": "mdb-dual-time-causal-events-v1",
  "binding": {
    "model_identity": {"model_id": "..."},
    "input_binding": {"input_sha256": "...", "contract": "..."},
    "numerical_contract": {"dtype": "..."},
    "parent_cut": "cas://sha256/..."
  },
  "branches": [
    {"branch_id": "native", "role": "native", "arm_kind": "native"}
  ],
  "events": [
    {
      "event_id": "formation:token-4:layer-12",
      "token_time": 4,
      "layer_depth": 12,
      "stage": "formation",
      "event_kind": "producer",
      "branch_id": "native",
      "evidence_class": "observation",
      "logical_address": "...",
      "physical_address": {"module": "..."},
      "payload": {"sha256": "..."},
      "refs": {"receipt_ref": "cas://sha256/..."}
    }
  ]
}
```

Coordinates may be omitted or null; the exporter puts those events on an explicit `?`
axis. To mark a typed event `causal_evidence`, it must additionally contain
`same_parent: true`, `native_consumer_executed: true`, and a finite
`effect.signed_effect` or `effect.future_signed` value.

The two rerun buttons download a proposed JSON spec or command text. The HTML has no
network capability and does not attach to a bench, execute a model, or perform an
intervention. Run proposals require review and submission through a separately
qualified live bench.

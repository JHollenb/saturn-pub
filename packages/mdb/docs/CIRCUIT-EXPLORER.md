# MDB circuit explorer

MDB's circuit explorer is a small control plane for browsing native forward
routes and testing proposed branches.  It joins a sealed
`ForwardSymbolTable` trace to a qualified model/execution/input binding,
structural SSA edges, supplied roles, evidence, sealed predictions, and
measured branch outcomes.

The explorer is intentionally bounded.  A trace establishes that an operator
read or wrote a physical value at a captured native clock.  It does not give
that value a semantic name.  Roles such as `hidden`, `gate`, `reader`, or
`memory` are supplied by the experiment and are stored as
`supplied-unverified`.  Predictions are also supplied artifacts: the catalogue
seals their context and output digest, but it does not claim that they are
correct until a native result is recorded.

## Register a trace-backed circuit

```python
from mdb.circuit_explorer import CircuitCatalog

catalog = CircuitCatalog("circuits.json")
record = catalog.register(
    circuit_id="mamba://final-mixer/readout",
    trace=frozen_trace,                 # sealed mdb-forward-symbols-v1
    route_candidates=[
        {
            "route_id": "readout",
            "source_values": ["v041", "v042", "v043"],
            "target_values": ["v091"],
            "event_sequences": [8, 9],
            "roles": {"sources": ["H", "C", "G"], "consumer": "readout"},
        }
    ],
    supported_interventions=[
        {"intervention_id": "replace-hcg", "value_ids": ["v041", "v042", "v043"]}
    ],
    predictions=[
        {
            "prediction_id": "mamba-readout-dose-1",
            "route_id": "readout",
            # The experiment-local predictor sealed this digest before replay.
            # Use output_keys when the predictor covers only selected heads.
            "output_keys": ["regs", "imm", "store_addr", "store_value"],
            "output_digest": predicted_output_digest,
            "metadata": {"predictor": "source-derived-readout-v1"},
        }
    ],
    replay_handle={"backend": "mamba-forward"},
)
```

The adapter can also be used directly when a caller already has a public
`ForwardSymbolTable`:

```python
from mdb.circuit_explorer import TraceCircuitAdapter

record = TraceCircuitAdapter.from_table(table).to_record(
    circuit_id="toy://readout",
    route_candidates=[{"route_id": "out", "target_values": ["v003"]}],
)
```

The record stores the trace fingerprint and a derived context fingerprint.  It
does not copy model tensors or a Python predictor into the catalogue.

## Plan, execute, and record a branch

Plans are sealed records.  A plan contains the route, intervention declaration,
context binding, and prediction selected at planning time.  Use a different
`branch_id` to explore another dose or route; once a plan has an outcome, a
different plan body or a second different outcome for that plan is rejected.

```python
plan = catalog.plan(
    "mamba://final-mixer/readout",
    route_id="readout",
    intervention={
        "intervention_id": "replace-hcg",
        "value_ids": ["v041", "v042", "v043"],
        "dose": 1.0,
        "replacement_digests": replacement_digests,
    },
    prediction="mamba-readout-dose-1",
    branch_id="dose-1",
)
```

Execution is dispatched only through a callback explicitly registered by the
caller.  The callback protocol is keyword based:

```python
def replay(*, plan, record, context, trace, roots, replacements):
    result = branch_forward_many(
        trace, roots, replacements,
        dose=plan["intervention"]["dose"],
        native_output=native_output,
        source_identity={"circuit_id": record["circuit_id"]},
    )
    return {"receipt": result.receipt, "output": result.output}

from mdb.circuit_explorer import (
    BRANCH_FORWARD_CONTRACT,
    verify_branch_forward_result,
)

catalog.register_backend(
    "mamba-forward",
    replay,
    verifier=verify_branch_forward_result,
    contract=BRANCH_FORWARD_CONTRACT,
)
outcome = catalog.execute(
    plan, trace=frozen_trace, roots=roots, replacements=replacements
)
```

`execute()` checks every qualified binding before dispatch.  Supplying a trace,
model identity, execution contract, or input identity from a different context
raises `CircuitCatalogError`.  The callback receives the frozen trace and may
use any public execution API; MDB does not import a private Saturn runtime. An
unverified generic callback is refused. The branch verifier additionally checks
the actual public `Receipt`, exact root parent seal, the callback's dose against
the plan, intervention sites, replacement digests, every prefix/suffix event
against the frozen trace, and candidate output leaf digests.

`record_outcome()` can also be used directly when the caller performs replay
itself:

```python
outcome = catalog.record_outcome(
    plan,
    receipt=result.receipt,
    output=result.output,
)
```

The stored outcome includes the actual public receipt fingerprint, an output
descriptor digest, the prediction comparison (`matched`, `diverged`, or
`unavailable`), and the bound context.  Receipt and output payloads are not
used as replay roots.  They are result evidence only.

## CLI

The CLI reads and writes the same sealed JSON catalogue:

```console
mdb circuits list --catalog circuits.json
mdb circuits show mamba://final-mixer/readout --catalog circuits.json
mdb circuits plan mamba://final-mixer/readout --catalog circuits.json \
  --route-id readout --intervention '{"intervention_id":"replace-hcg","dose":1.0}'
mdb circuits record plan:0123... --catalog circuits.json \
  --outcome @measured-outcome.json
```

`plan` and `record` are catalogue operations.  There is no implicit execution
of a manifest or serialized callback from the CLI.  A native run must be
authorized by the caller through the explicit Python backend registration (or
another public host integration) before its receipt and output can be
recorded.

The catalogue keeps structural status, supplied-role status, prediction status,
and measurement status separate.  A divergent prediction is a measured trend
for the selected branch; it is not a claim that the structural route or the
instrument is absent.

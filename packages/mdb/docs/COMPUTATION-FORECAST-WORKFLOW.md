# Computation forecasts with actual ports and sealed execution

MDB can capture an actual module boundary, freeze an independent equation
forecast, release native execution, and retain numerical and collateral
comparisons as searchable computation records. The Python APIs share typed
operations with bounded scripts. Forecast callbacks are trusted Python code;
the module-entry guard is instrumentation rather than an OS sandbox.

## Actual module and parameter ports

Opt in when recording a CPU eval model:

```python
from mdb.forward_recording import record_forward
from mdb.forward_ports import NativePortCatalog

capture = record_forward(
    model, inputs, model_identity=checkpoint_sha256,
    execution={"backend": "cpu", "dtype": "float32"},
    input_identity=input_sha256, capture_ports=True,
)
ports = NativePortCatalog(capture.trace)
output = ports.resolve("module:mixers.2.dt_proj:return")
weight = ports.resolve("parameter:mixers.2.dt_proj.weight")
bias = ports.resolve("parameter:mixers.2.dt_proj.bias")
consumers = ports.first_consumers("module:mixers.2.dt_proj:return")
actual = capture.observed_values[output["value_id"]]
```

The return is the value observed by the module hook, including the bias add and
pre-existing return hooks. This namespace is separate from inferred mathematical
regions in `StructuralCatalog`. Existing selectors keep their meanings.

Inputs support `:args`, `:kwargs` and `:input`; nested values can be selected with
`path=(...)`. Convolutions and identity modules are represented even when they
do not match a known mathematical stage. Repeated calls require an explicit
`occurrence` or a `call:` selector. Shared modules retain their physical identity
and declared aliases; attribute spelling is not inferred from a hook.

`available_pc` locates the recorded boundary. Consumer results distinguish the
first direct SSA use from the first use after that boundary and flag prior
consumption. They do not promise a writable frontier for an alias or an already
consumed input. Named parameters and buffers are read-only declared roots.

`record_prefix(..., capture_ports=True)` uses the same option. An aborted call
has no fabricated completed return. Leaving the option off preserves the
existing recording format and numerical addresses.

```sh
mdb ports trace.json
mdb ports trace.json --select 'module:mixers.2.dt_proj:return'
```

## A scripted prospective transaction

The following small example forecasts ReLU and the final linear layer from a
genuine observed prefix. `inputs` is the ordinary positional tuple for the
CPU FP32 eval model `Linear(2,3) → ReLU → Linear(3,1)`.

```python
import numpy as np
from mdb.forecast_transaction import ForecastTransaction
from mdb.equation_comparison import compare_equations

def predict(context):
    incoming = context.frontiers["case"]["boundary"]["args"][0]
    result = np.maximum(incoming.astype(np.float64), 0)
    result = result @ context.weights["2.weight"].T + context.weights["2.bias"]
    return {"predictions": {"out": result}}

with ForecastTransaction(
    "./forecast-run", model=model, model_identity=checkpoint_sha256,
    input_identity=input_sha256, source_identity={"equation": equation_sha256},
    allowed_prefix={"stop_before": "1", "observe": ["0"]},
    denied_modules=("1", "2"),
) as transaction:
    transaction.run_script([
        {"op": "capture", "name": "case", "args": inputs},
        {"op": "predict", "name": "equation", "callback": predict},
        {"op": "seal"},
        {"op": "execute", "name": "native",
         "callback": lambda context: {"out": context.model(*inputs)},
         "analysis": lambda observed: compare_equations(
             transaction.prediction("equation"), observed)},
    ])
```

The transaction snapshots actual model weights and checks their identity. Each
prediction receives owned copies of its declared frontiers and weights. Native
calls to the instrumented model are denied during prediction and sealing;
attempts remain latched even if callback code catches the exception. Genuine
prefix capture is admitted only under the declared boundary contract.

Forecast arrays and optional intervention payloads are stored without pickle.
`ForecastPayload(value, dtype="float32", provenance={...})` declares the staging
cast. Record any earlier casts used to construct the value in its provenance.
`transaction.payload(forecast, name, as_tensor=True)` returns an owned CPU tensor
for a typed Act. Predictions cannot be edited after sealing.

The global seal is published durably before native release. Release checks the
sealed files and payloads again. Each execution result is persisted before its
optional analysis callback, so an analysis failure retains the native evidence.
Public receipts bind phase events and immutable content-addressed arrays.

Use multiple captures or predictions before one seal when all cases must be
forecast globally. For a raw-input account, declare `frontier={"kind": "raw-input"}`
instead of `allowed_prefix`, then use a `frontier` action to publish the allowed
inputs. This mode makes no native call to prepare a forecast. The fixed source
and weight identities remain part of the transaction declaration.

Scripts are finite lists of already-bound Python actions, not serialized code
or import expressions. Budget limits cover actions, captures, predictions,
executions, native calls, capture operations, bytes, files and cooperative wall
time. Hooks are removed on failure and close. Direct `.forward` calls, other
model objects and external side effects are outside the guard's authority.

```sh
mdb forecast verify ./forecast-run
mdb forecast show ./forecast-run
```

Offline verification is framework-free. Supply an expected seal and final event
head from independent custody when completeness against deletion is required;
a self-contained local ledger cannot prove that its final events were not removed.

## Numerical error and collateral

```python
from mdb.equation_comparison import compare_equations

report = compare_equations(
    predicted_heads, actual_heads,
    decision_rules={"regs": "positive", "flags": "positive", "length": "argmax"},
    baseline=actual_native_heads,
    predicted_baseline=predicted_native_heads,
    targets={"regs": [[0, 7, bit] for bit in range(64)]},
    components={"regs": predicted_register_terms},
    observed_components={"regs": measured_register_terms},
    include_residuals=True,
)
```

Residuals are observed minus predicted. Reports retain signed sums, means,
extrema, L2 norm, largest coordinates and optional complete signed arrays. A
decision rule is explicitly requested per field; no numerical tolerance turns a
scientific observation into a pass/fail gate. Shape, finite-value and element
budget checks protect the instrument.

The observed baseline and forecast baseline remain separate. Targets are explicit
tensor coordinates; every other coordinate is collateral when targets are
supplied. A correct destination with a changed RSI bit therefore remains visible.
Component closure measures whether supplied additive terms reconstruct an output,
independently of whether the forecast matches native execution. Native replay
parity and architectural ISA correctness are separate evidence supplied by the
caller; the comparison does not infer either from a small residual.

## Searchable applicability records

```python
from mdb.equation_comparison import EquationCatalog

catalog = EquationCatalog("./investigation/equations")
record = catalog.record(
    law_id="law_003", route_id="route_012", alias="late RSI reader response",
    binding={"model_identity": checkpoint_sha256, "execution": execution,
             "input_identity": input_sha256, "frontier": "raw-input",
             "horizon": 3, "source_identity": {"equation": equation_sha256}},
    report=report,
    conditions=[{"field": "architecture", "op": "eq", "value": "r3-mamba"}],
    tested_context={"rdi": 262, "arm": "U_swap", "instruction": 0},
    readers=["regs:6:1"],
    evidence={"forecast_seal": forecast_seal_sha256, "native_receipt": receipt_sha256},
)
matches = catalog.find(law_id="law_003", reader="regs:6:1")
```

Records are immutable, content-derived JSON objects published without overwrite.
Reload verifies both record and report seals. Source tensors stay in their named
custody store. A supplied equality, membership or inclusive-range condition can
match a query, fail to match or remain unknown when context is missing. Empty
conditions are unspecified. Matching a condition does not establish generalization:
`tested_context` records what was actually measured. No semantic ISA name is
required and no automatic rule induction is claimed.

```sh
mdb equations compare predictions.json outcomes.json --rules rules.json
mdb equations find --catalog ./investigation/equations --law-id law_003
mdb equations find --catalog ./investigation/equations --reader regs:6:1
```

The [earlier suffix experiment](EXPLICIT-COMPUTATION-FORECASTS.md) motivated these
features. The next [fixed raw-input sequence plan](../qualification/raw-isa-sequence-plan-20261030.json)
extends the independent equations backward to instruction bytes and forecasts
successive architectural updates before any native call. Its equations use the
complete checkpoint; its actual results must be reported separately after execution.

Validation for this upgrade: 111 installed-wheel tests passed outside the
workspace across native ports, transactions, equation records, prefix capture,
recording, interventions, symbols, discovery replay, catalogs and scripts. The
small Python transaction example above also ran from the installed wheel.

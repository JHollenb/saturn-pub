# Prepared native parents and experiment panels

Status: implemented, installed-wheel qualified on 2026-10-29. The API runs exact scalar ATen continuations from a root-produced live frontier. It supports the recorder's deterministic, functional CPU tensor contract.

```python
from mdb.forward_interventions import PreparedForwardParent
from mdb.forward_panels import ExperimentPanel

# capture is returned by mdb.forward_recording.record_forward.
# site is a produced SSA value in capture.trace; payload matches its contract.
with PreparedForwardParent(
    capture.trace, capture.initial_values, through_value=site,
    native_output=capture.output, observe_values=(site,),
) as parent:
    panel = ExperimentPanel(parent, {
        "native-copy": {"replacements": {site: capture.observed_values[site]}},
        "candidate": {"replacements": {site: payload}, "dose": 1.0},
    }, observe_values=(site,), max_branches=2)
    sealed = panel.manifest()  # Persist before executing candidates.
    rows = panel.run()
    for row in rows:
        if row.error is None:
            receipt = row.result.receipt.to_dict()
            output = row.result.output
```

Preparation owns copies of the trace, roots, native baseline and produced frontier. Frontier cloning preserves shared backing storage and view offsets. Each branch verifies its frontier and clones it before continuation. Caller edits cannot change the prepared owner. Interventions before the prepared boundary and unretained earlier observations are rejected. A new input, checkpoint, trace or execution contract requires a new parent.

The parent performs one complete exact root replay at preparation. Each candidate reuses its retained prefix proof and executes the original remaining operators, retaining full per-step output descriptors. Receipts retain the public branch operation and add `replay_reuse`, binding the StateCut, trace, roots, cut clock and actual operator counts. The public circuit verifier validates these bindings. The StateCut manifest describes the owned live closure; it does not promise durable process-independent restoration.

The panel owns replacement snapshots and seals all descriptors before its first candidate. It accepts declarative data, with no callback imports from serialized specifications. Each panel executes once and returns a `PanelRow` for each attempted candidate, including errors and wall time. Native consumer audits, controls, successor inputs and catalog insertion remain explicit adapter responsibilities. Preserve every completed row before doing later audits that may fail. The ISA launcher now also collects a separate bounded partial archive when an assay raises; partial evidence is never labelled a successful report.

Both paths retain the recorder's functional execution boundary. Mutable operators are refused. Seeded random operators are refused before execution because RNG closure is not implemented. An intervention replaces the declared SSA handle; it does not retroactively rewrite every alias. Branches execute serially under the prepared parent's lock. There is no physical branch batching or cross-job resident service.

## Qualification and measured limits

The archived [installed-wheel proof](../../../../artifacts/mdb-discovery-upgrade-20261029/installed-proof.json) runs 54 focused tests from outside the workspace, verifies installed source against wheel bytes, and checks metadata imports with Torch/private packages blocked. Tests cover native parity, multiple sites and doses, alias/view geometry, private-frontier corruption, caller mutation, returned-manifest mutation, sealed payloads, RNG refusal, clock tampering and candidate failure isolation.

The [R3 qualification](../../../../isa/outputs/prepared-forward-qualification-r3-mamba/README.md) uses twelve retained patches from the native square and tangent campaigns. All twelve root/prepared pairs match at six selected stages and all eleven native heads. Twelve separate complete model calls match the prepared outputs, and eight successors literally consume branch-predicted registers and flags. Every prepared branch executes 81 operators and reuses 490 of the 571-operator trace.

Warm scalar branch time falls from 0.692 to 0.132 seconds (5.26×). Preparation plus panel sealing costs 0.214 seconds, giving 2.00× across that panel. Lane order is deterministic and reversed between families; these are bounded observations, not a randomized hardware benchmark. The new four-branch row-9 panel costs more after preparation (0.91×), despite 5.24× marginal branch speed. Keep root replay available for small or one-off requests. Full-job speedup, physical batching, reduced tensor storage and automatic hypothesis selection remain unestablished.

# Instrument Trial: arbitrate a reading against the native consumer

A standard interpretability instrument — single-step activation patching, a linear or
dictionary probe, a cosine-similarity readout, an SAE ablation — produces a *reading*.
A reading is a candidate, not a verdict. The `saturn_pub.trial` module lets you take a
reading a standard instrument declared and let the **unchanged native consumer** (the
rest of the model, run to completion) decide whether that reading holds, inverts, or is
inconclusive — using saturn-pub's own Session / fork / compare / Receipt.

The module has two independent jobs. Importing it never imports torch.

## 1. Re-derive the shipped case verdicts offline

```python
from saturn_pub.trial import verify_bundle

result = verify_bundle()          # checks the bundle shipped inside the package
assert result["ok"]
```

or from the command line:

```bash
saturn-pub-trial-verify           # exit 0 = reproduced, 3 = mismatch, 2 = structural error
```

`verify_bundle` checks every receipt against its SHA-256 in `manifest.json`, then
re-derives six case verdicts (eleven arm verdicts) from the raw receipt scalars using the
frozen mechanical decision rules in the module (no interpretation layer), and compares
them to `expected.json`. It is stdlib-only and needs no GPU or model framework.

The six cases are the owner's preregistered head-to-head between standard instruments and
consumer-gated certification; in four of four graded non-reserve cases the standard arm
is a misreport while the consumer-gated arm matches ground truth. The receipts were
produced on public weights (FLUX.2 Klein 4B and Qwen-family models); the bundle shipped
here is a redacted, re-pinned copy (see [PROVENANCE](../PROVENANCE.md)).

## 2. Arbitrate your own reading

```python
from saturn_pub import Act
from saturn_pub.trial import Reading, DecisionRule, arbitrate

reading = Reading(
    instrument="single-site activation patching",
    claim="no necessary component: single-site ablation leaves the output intact",
    asserts_effect=False,      # the instrument says the carrier is NOT load-bearing
)

rule = DecisionRule(
    name="consumer-necessity", version="1",
    metric="target_removed",   # key the effect evaluator returns; higher = more "present"
    present_threshold=0.5, absent_threshold=0.15,
    require_sham_flat=True, sham_metric="sham_removed",
)

def effect(branches):          # consumer closure: read the finished branches
    native = branches["native"].read("output").tolist()[0]
    candidate = branches["candidate"].read("output").tolist()[0]
    out = {"target_removed": native - candidate}
    if "sham" in branches:
        out["sham_removed"] = native - branches["sham"].read("output").tolist()[0]
    return out

row = arbitrate(session, reading, rule,
                effect=effect, candidate=ablate_across_steps,
                sham=single_site_ablation, steps=3)
print(row.verdict)             # "invert", "agree", or "inconclusive"
```

`arbitrate` forks the parent into a native arm, a candidate arm (your instrument's
intervention), and an optional sham arm; runs the real remaining model on each; then
applies the **frozen** `DecisionRule` and compares the resulting class to what the reading
asserted:

| reading asserts | consumer shows | verdict |
| --- | --- | --- |
| effect absent | effect present | **invert** |
| effect present | effect absent | **invert** |
| matches | matches | **agree** |
| anything | ambiguous band | **inconclusive** |
| declined | anything | **inconclusive** |
| anything | a control failed | **inconclusive** |

Controls are part of the verdict. `require_exact_gate` (default on) reruns the unmodified
path and requires it to reproduce bit-for-bit before any result is read; `require_sham_flat`
requires the sham intervention's effect to stay within `sham_tolerance`. A result whose
controls fail is `inconclusive`, never a verdict.

### What `arbitrate` takes

- `session_or_adapter` — a live `Session`, or an `Adapter` plus a `parent` StateCut.
- `reading` — a `Reading` (see the seam below).
- `rule` — a `DecisionRule`; its `fingerprint` is recorded on the row.
- `effect` — a consumer-closure evaluator `(branches) -> metrics`. `branches` is
  `{"native", "candidate"[, "sham"]}` finished Sessions; it must return the rule's
  `metric` (and `sham_metric` when required). Higher metric = stronger consumer evidence
  that the thing the reading was about is real.
- `candidate` / `native` / `sham` — each an `Act`, a sequence of `Act`s (applied then run
  `steps`), or a driver callable `(branch) -> None` that drives a forked branch to the
  consumer-observable end (use a callable when the intervention is scheduled across
  steps, as a distributed store requires).
- `steps` — continuation budget for the default native arm and the exact-replay gate.

The returned `TrialRow` records the reading, the frozen rule, the verdict, the
classification, the consumer effect, the controls, the raw measurements, the parent cut,
and a content-sealed `Receipt`. `TrialRow.to_dict()` is a consumer-gated verdict row.

### Seam for an external reader (circuit-tracer)

A reading does not have to be declared by hand. `Reading.from_external` tags one produced
by another tool so its provenance stays explicit:

```python
reading = Reading.from_external(
    "circuit-tracer",
    claim="this edge is not load-bearing",
    asserts_effect=False,
)
```

This is the seam a future circuit-tracer `native_edge_test` integration would use: take an
edge attribution from the tracer, express it as a `Reading`, and arbitrate it by native
continuation exactly as above. The tracer itself is **not** implemented here; only the
seam is.

## The seven traps

The measurement failure modes that motivated the whole program, each hit in practice and
now gated against, are available as a programmatic checklist:

```python
from saturn_pub.trial import traps
for trap in traps():
    print(trap.number, trap.name, "->", trap.gate)
```

| # | Trap | The gate that kills it |
| --- | --- | --- |
| 1 | The latent-only false positive | Confirm every internal claim at an observable the model actually serves (the two-consumer rule). |
| 2 | Contrast collapse inflates alignment | Verify baseline contrast viability; report baseline-normalized progress, never raw alignment. |
| 3 | The scaffold dominates similarity | Test with semantic contrasts, or not at all. |
| 4 | A clean output is not a correct output | Output quality checks cannot substitute for output content checks. |
| 5 | Terminal readout masquerades as origin | Cross-check late-site interventions with delta-based screens first. |
| 6 | Relaxation artifacts read as structure | Derived graph objects are search guides, never certificates. |
| 7 | A live site is not a circuit | Behavioral liveness is a candidate; promotion needs the full battery and a control designed to kill it. |

## Runnable example

```bash
python examples/instrument_trial.py
```

It verifies the shipped bundle, then arbitrates the same distributed-store ablation on the
supplied redundant-writer circuit under two readings: a naive single-site reading inverts,
and a consumer-gated reading agrees. The arbiter is the consumer; the verdict depends only
on what the instrument claimed about the same measured effect. The report is written to
`outputs/instrument-trial/report.json`.

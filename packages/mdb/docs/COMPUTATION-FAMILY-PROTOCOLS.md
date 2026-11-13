# Compile and compare computation family protocols

The [protocol compiler](../qualification/compile_computation_family_protocols.py) turns two explicit family templates and their frozen historical selection metadata into a structured comparison. It runs without model frameworks, parameters, GPU work or fresh outcomes. It reproduces the development selections `PQP` for the [sequential text family](FLUX-PROSPECTIVE-WRITER-FAMILY.md) and `RA` for the [parallel image family](FLUX-PARALLEL-RELATION-FAMILY.md).

This is a research prototype under `qualification/`, rather than a new installed `mdb` command. It automates graph validation, candidate enumeration and selection bookkeeping. It does not infer operation graphs, identify semantic concepts, launch a model campaign or expand applicability automatically.

## Run the saved metadata comparison

From the research workspace root:

```sh
rtk proxy python3 mdb/qualification/compile_computation_family_protocols.py \
  --sequential-rule artifacts/mdb-computation-family-20261107/family-rule-v1.json \
  --parallel-rule artifacts/mdb-parallel-relation-family-20261107/family-rule-v2.json \
  --output /tmp/computation-protocols.json
```

The output is created exclusively. Choose another path for another execution. The published workspace result is `artifacts/mdb-computation-protocols-20261107/protocols-v1.json`; installed, framework-free qualification is recorded in `qualification-01/qualification-typed-ports-v1.json` under that directory. Eight focused tests cover DAG mechanics, anchors, incompatible declared metric domains and frozen metadata replay.

## What a protocol records

The templates make these fields explicit:

- The supplied source-formation DAG and its ancestor paths.
- Intervention ports and their actual projection stages.
- Held-return corners, recomputed candidates, baselines, source-copy anchors and separate references.
- Recipient gates, nonintervened state, address policy and source policy.
- The ideal real-arithmetic affine endpoint map and its fixed constants.
- The qualified finite operation convention and original full projection rule.
- Clock, receiving boundary, continuation and source-qualified metric domain.
- Historical scores, candidate menu, tie order and selected candidate.

Sequential attention has a formation path to its MLP return. Parallel attention and MLP features share an input without an attention-to-MLP path. Both held-port endpoint constructors are affine in real arithmetic, under their declared constants. Neither source formation nor downstream continuation inherits that affineness.

The sequential A/M ports are returned contributions after separate output projections. The parallel A/M ports are features before a shared output projection. Their letter names do not imply equal stage, width, energy or causal role. Symbolic widths point to the relevant model slots; this metadata compiler does not independently inspect model-width files.

## Avoid trivial or invalid selections

For the sequential family, enumerate eight R/A/M held corners, then classify QQQ as baseline and PPP as a source-copy anchor. The selectable menu contains six remaining held corners plus the two explicitly recomputed MLP actions QPM and PQM. Whole P text is a reference installed into Q image/nontext closure; it is not a natural complete P state. The sequential source is image-primed.

For the parallel family, enumerate Q, R, A and RA with Q MLP/gate held fixed. Q is the baseline; R/A/RA are candidates. Natural full P is a separate reference, not the all-P corner under the fixed Q receiving closure.

Historical selection excludes anchors and references even if their scores are smaller. Observable, units, reference and context scope must agree within a selection domain. This check validates the attached declarations. It does not independently prove the original scores' measurement provenance; the rule's bound evidence remains authoritative.

The two families have different observables and source/continuation policies. The compiler returns their selections separately and does not rank their effect magnitudes against each other.

## Use the comparison to plan the next campaign

The common experimental grammar is source formation → held-port endpoint → native-consumed continuation. Bind that grammar to a specific clock, address resolver, receiver and numerical profile. Freeze any proposed effect on untouched contexts, seal complete numerical predictions, execute native branches and retain all outcomes with their counterexamples.

The existing family workers implement those execution steps through public MDB/Saturn sessions and transactions. The prototype currently emits metadata, not worker scripts. A later integration can generate a finite MDB script from the protocol and ingest its independent audit, preserving the distinction between a correct numerical forecast, a transferred useful effect and an object-level semantic interpretation.

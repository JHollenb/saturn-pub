# Dependency-guided reuse

`mdb.dependency_reuse` produces a conservative, sealed plan for an ordered deterministic suffix. It tracks which state slots may differ between two bound parents and compares that set with each operation's declared reads, writes and invalidations. This is diagnostic planning; it does not execute a model, admit an Act, certify a footprint or authorize skipping native work.

```python
from mdb.dependency_reuse import plan_suffix_reuse, verify_reuse_plan

plan = plan_suffix_reuse(
    [
        {"operation": "single.7", "reads": ["image", "text", "context"],
         "writes": ["image", "text"], "footprint": "declared"},
        {"operation": "readout", "reads": ["image", "context"],
         "writes": ["velocity"], "footprint": "declared"},
        {"operation": "scheduler", "reads": ["latent", "velocity"],
         "writes": ["latent"], "footprint": "declared"},
    ],
    changed_slots=["latent"],
    model_identity="checkpoint-and-configuration-identity",
    execution_identity="deterministic-backend-dtype-contract",
    source_parent="complete-source-parent-fingerprint",
    target_parent="complete-target-parent-fingerprint",
)
assert verify_reuse_plan(plan)["verified"]
assert [row["decision"] for row in plan["rows"]] == [
    "reuse-candidate", "reuse-candidate", "recompute-candidate"
]
```

The example omits the actual model's full clock and closure slots for readability. Production inputs must include every changed parent slot and complete declared native footprints. The same model, deterministic execution program, non-payload model state and external inputs must be bound. Tensor proximity and semantic labels never authorize reuse. Unknown footprints conservatively taint the remaining plan. Affected reads propagate possible differences to every declared output. An unaffected overwrite clears the prior difference in its written slot; invalidation remains conservative.

## Qualified research occurrence

The November 4 Klein eye-restoration panel supplies two complete post-Act cuts that differ only in latent history. Its 15 native continuation receipts declare 13 single-block operations, readout and scheduler. On these real footprints, the planner identifies **14 reuse candidates and one scheduler recalculation**. Native checks at two image/text corners independently produced exactly the predicted velocity, all 131,072 terminal latent values and every decoded pixel. The middle two corners were forecast-state/native-VAE cases without native transformer suffix validation.

The terminal calculation explicitly preserves the native BF16 promotion and rounding boundaries. These results support the tested same-clock latent-only case, not arbitrary prompt/text edits. A changed working text bank invalidates subsequent attention consumers. A latent change before project also invalidates the projected image and its descendants.

- [Bound real-receipt plan](../../../../artifacts/mdb-scene-dependencies-20261104/dependency-reuse-plan.json)
- [Saved-only native audit](../../../../artifacts/mdb-scene-dependencies-20261104/analysis-v1.json)
- [Installed-wheel qualification](../../../../artifacts/mdb-scene-dependencies-20261104/installed-planner-qualification.json)
- [Implementation](../src/mdb/dependency_reuse.py) and [tests](../tests/test_dependency_reuse.py)

The wheel was built and tested outside the checkout with five unit tests and an import-block smoke that denied model frameworks and private workspace packages. The planner has not been deployed to the fleet and the CUDA panel did not import this newly built module. Applying it afterward to retained receipts is a planning integration check, not a new prospective skipping experiment or a measured speedup.

## Next runtime integration

Combine this plan with complete StateCut comparisons, native operation footprints and already qualified mathematical endpoint laws. A future executor should choose cached output reuse, endpoint prediction or native execution per operation, retain its reasons and record actual work/cost. It must preserve typed Act admission and validate representative complete futures. This module intentionally stops before that execution policy.

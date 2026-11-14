# Native branch change trial

The first slice of the [computation explanation proposal](COMPUTATION-EXPLANATIONS-PROPOSAL.md) works on Qwen2.5-1.5B. The new `mdb.analysis.changes.explain_changes` API distinguishes payload transport from routing changes, decomposes the receiving MLP into gate/up/interaction responses, and nominates a neuron whose restoration has a measured output effect. A conditional forecast for that same neuron in a reordered context precedes native execution and predicts the direction and approximate magnitude of its effect.

This is a Python prototype over two aligned native branches. The layer/source candidate is supplied from the historical temperature case. It does not automatically discover a complete circuit or assign a semantic name to a neuron.

## Relationship to existing MDB capabilities

MDB already offered routing/value and gate/up factorial experiments, consumer-connected native interventions, durable sealed forecasts and fresh-context evaluation through its earlier family workflows. The refactored engine already implemented the finite identities used here. See [family programs](COMPUTATION-FAMILY-PROGRAMS.md), [Qwen families](QWEN-COMPUTATION-FAMILIES.md) and [forecast transactions](COMPUTATION-FORECAST-WORKFLOW.md).

This trial adds a convenience report over two refactored-engine runs, a per-source/per-neuron view with explicit tails, a persistence callback in the engine's `whatif`, a CUDA-mask fix and a new bounded experiment. It does not establish a new category of mathematical explanation or an advance over the earlier scientific workflows. In particular, the callback does not replace the older `ForecastTransaction` model-entry quarantine. The novelty is integration and packaging within this engine; autonomous mechanism discovery remains unimplemented by this slice.

## Native outcomes

The receiver ends with `Description: freezing\nTemperature:` after icy/cold and fiery/hot demonstrations. The donor replaces freezing with boiling. Every intervention begins at the same layer-20 native parent; the baseline and edited branches consume the same receiver tokens. The edited source is row 18, and both KV groups are included.

The consumer is the final-row `hot minus cold` logit margin, with token IDs 4017 and 9255. No generated-token continuation is included.

| Branch | Native margin | Change from baseline |
| --- | ---: | ---: |
| Baseline | -4.750 | 0 |
| Copy receiver K/V back into itself | -4.750 | 0 |
| Donor V | +0.375 | +5.125 |
| Donor K | -4.625 | +0.125 |
| Donor K and V | +0.375 | +5.125 |

The sham logits are bitwise identical to baseline. V and K+V reach the same measured pairwise margin; this does not imply identical internal states or full logits.

## What the explanation adds

At layer 20, the V intervention has a nonzero payload term and exactly zero routing and routing/payload interaction terms. The K intervention has a nonzero routing term and zero payload/interaction terms. Their joint intervention includes all three terms.

At layer 21, the V intervention produces both changed routing and changed values. Their head-return vector norms are 3.96759 and 1.16027, with an interaction norm of 0.51873. These norms describe separate vectors and are not additive causal shares. The difference between a payload-only edit at its receiving block and its later mixed response is visible in the report.

The layer-20 MLP responds through changed gate and up arguments with substantial signed cancellation. The largest neuron under the fixed direct-readout ranking is `t21.L20.f[i=1279]`:

| Finite product term | Direct final-readout projection |
| --- | ---: |
| Gate response | +0.0414434 |
| Up response | +0.4452502 |
| Interaction | -0.2371587 |
| Total | +0.2495349 |

The shared final-readout scale is evaluated at the edited endpoint. These are direct residual-accounting projections, not isolated suffix effects. The report keeps every unlisted neuron/source in an aggregate tail and records measured native discrepancies separately.

## A nominated intervention and a fresh conditional forecast

The development explanation selects neuron 1279 without searching native repair outcomes. Restoring just that neuron's activation to the receiver baseline in the V-edited branch moves the native margin from +0.375 to 0. The effect is -0.375. A full mathematical suffix forecast predicts -0.4532007; the baseline local derivative predicts -0.4504484. The difference from the native result is retained.

The second context reverses the order of the two demonstrations. The same development-selected neuron is retained. After observing this context's V-edited branch, MDB forecasts the effect of restoring that neuron to this context's own baseline activation. A callback persists and flushes the sealed forecast before the native counterfactual measurement.

| Fresh context quantity | Margin or change |
| --- | ---: |
| V-edited branch margin | +1.500 |
| Predicted finite restoration change | -0.3815135 |
| Predicted local restoration change | -0.4493617 |
| Measured restoration change | -0.375 |
| Restored native margin | +1.125 |
| Native minus finite forecast | +0.0065135 |

A second native branch, replayed directly from the common parent with the V edit and neuron restoration, matches the what-if margin exactly. This is a conditional repair prediction using an observed fresh receiving state and that context's baseline activation. It is not a prediction from raw text or donor-free semantic generalization. Worst-case propagated numerical bounds remain unresolved.

## Implementation and checks

The API is in [changes.py](../src/mdb/analysis/changes.py). It reuses existing finite attention, normalization and SwiGLU identities, requires one adapter/checkpoint instance and identical token histories, and refuses direct edits of explained intermediates that would need their own displacement accounting. It checks that neither branch state changes. Tensor terms are returned separately from bounded JSON.

For a small mechanics example:

```python
import mdb
from mdb.analysis.changes import explain_changes
from mdb.analysis.forecast import Margin

dbg = mdb.open_tiny("qwen2", "bf16")
base = dbg.run([1, 5, 9, 3], name="base")
base.until("logits")
edited = dbg.run([1, 5, 9, 3], name="edited")
edited.set_cell(0, "v", 0.5, "add", rows=[1])
edited.until("logits")
result = explain_changes(base, edited, Margin(3, 4, 9), layer=0)
print(result.report["attention"]["sources"])
print(result.report["mlp"]["neurons"])
```

The existing `whatif` API now accepts an optional `before_native(record)` writer. Writer failure prevents measurement. The trial also exposed and fixed an existing bug in masked forecasts: the model's CUDA mask had to be moved to the CPU mathematical reference's device before `torch.where`.

[Tests](../tests/test_analysis_changes.py) cover Qwen2/Qwen3, FP32/BF16, K/V/joint/sham branches, native endpoint reconstruction, signed tails, refusal of incompatible comparisons, session preservation and the forecast writer barrier. The independent [artifact reader](../tools/verify_branch_changes_trial.py) imports neither a model nor MDB analysis. It recomputes attention and MLP finite terms from saved native operands and checks native margins, top/tail accounting and the separate repair execution. All 89 checked fields agree, with a maximum absolute report difference of 7.11e-15.

## Execution and custody

The completed finite mrun job is `job-7e3bd6a4fb0d`, on Beast's RTX 4080 with BF16 eager attention and batch one. It used one model acquisition and scalar suffix branches; the measured process elapsed time was 17.69 seconds, peak sampled RSS 2437.9 MB and peak sampled VRAM 3538.0 MB. These peaks are observations rather than a throughput or memory optimization claim.

The worker records CUDA reduction flags, model identity, source-wheel hashes, native parent state, donor cells, endpoint tensors, logits, forecasts and outcomes through the mrun outbox. Original model parameters remain external: `parameter-artifacts.json` binds the original safetensors archives, original headers, named tensor shapes/dtypes and SHA-256 hashes. [retrieve_weights.py](../tools/retrieve_weights.py) retrieves and verifies the original tensor bytes; one retrieval was exercised before model loading.

Two earlier attempts are preserved: `job-9ce936623940` failed before model loading because the host's configured uv cache path was broken; `job-d472c07e54ef` retained all four completed branch comparisons but stopped at the CUDA-mask forecast bug. The completed job includes both fixes, with automatic retries disabled.

Fetch the immutable outputs with:

```bash
common exec mrun artifacts fetch job-7e3bd6a4fb0d --dest ./branch-change-results
common exec python tools/verify_branch_changes_trial.py ./branch-change-results --out verification.json
```

Submit the same finite panel from this package directory with:

```bash
common exec python tools/submit_branch_changes_trial.py --record submission.json
```

The launcher builds and installs public wheels into a uv overlay; it does not inject checkout paths. The local results are under `results/branch-changes-20261113`. The standard Saturn evidence sweep cannot recognize the new raw report schema as a gate receipt or stream; its initial empty-target outcome is retained as an instrument limitation. A separate offline mechanics receipt binds the independently checked files and is used for the receipts-only sweep. No scientific success gate is added to the branch measurements.

## What remains

The trial demonstrates a useful connection from a finite decomposition to a nominated native intervention and a fresh conditional forecast. The report still uses a supplied site, direct readout ranking and explicit local equations. Recursive upstream producer expansion, general downstream edge ranking, automatic interface counterexamples, durable edited-value pointers and command/web integration remain the next stages of the proposal.

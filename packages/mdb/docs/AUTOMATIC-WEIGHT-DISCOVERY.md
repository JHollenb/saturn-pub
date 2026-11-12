# Automatic discovery of affine weight edits

MDB now has an experimental controller that selects an affine intervention site and its numerical edit from task feedback, then compiles the selected intervention into a removable parameter patch. The caller supplies a model, discovery evaluator, preservation examples and finite search budget. A selected layer, target parameter values, healthy weights and a published edit are not controller inputs.

The first experiments show automatic site/gain recovery on three constructed organisms and a useful but collateral-heavy intervention on a pretrained Qwen model. This is bounded numerical search over a supplied edit grammar. It does not use a frontier WSP reasoning agent or discover arbitrary semantic programs.

## What the controller does

`mdb.wsp_discovery.AffineDiscovery` infers the model's `torch.nn.Linear` sites, excluding its declared output embedding module. It observes input activation magnitudes during the baseline. Its finite sequence is:

1. Run the native discovery task and preservation evaluator.
2. Screen a deterministic sample of affine sites with write gain 0.5.
3. Refine the two or three best sites with gains 0 and 1.5.
4. Fit a quadratic to the observed dose scores and test its bounded optimum when concave.
5. Nominate input channels by activation magnitude times column norm; test suppression and amplification.
6. Select the best observed objective, retaining native on ties or when every intervention is worse.
7. Synthesize the selected affine parameter patch from the original coefficients.

Every round seals its candidate declarations before native evaluation through `WeightStudy`. Later rounds may use completed earlier discovery outcomes. These are adaptive discovery experiments; an unknown behavioral outcome is recorded as unknown, rather than presented as a numerical forecast. Separate held-out arms freeze the selected edit before their outcomes are released.

The available edits are an entire affine output gain and selected input-channel contribution gains. A write gain s compiles to W′=sW and b′=sb. A channel gain scales the corresponding columns of W. Virtual output edits and compiled parameters can differ under finite arithmetic; native comparisons remain measurements.

The module also provides `fit_output_delta` for a sample-space ridge solve of desired local write changes with preservation examples. The current controller does not automatically construct arbitrary desired-write targets or use this solver in its gain/channel loop.

## Run the constructed blind experiment

```sh
mdb wsp discover-demo ./blind-discovery --seed 731
```

The fixture constructs a five-write FP64 residual program, chooses a hidden site, and multiplies that site's matrix by eight. The controller receives output examples and an evaluator, with no fault-site label or healthy parameter bank. Site names are permuted. Preservation inputs lie outside the fault's input-sensitive direction. This makes the fault expressible by the search grammar; the organism and desired function are supplied, not learned.

The runs for seeds 731, 732 and 733 all recover the hidden site and a gain numerically equal to 1/8. Each uses sixteen baseline/candidate observations. On 48 fresh inputs per organism, native mean squared errors of 0.17939, 0.12285 and 0.12788 fall to below 6e-32 after compilation. Virtual and compiled outputs differ by at most 4.44e-16. Restoring the original write reproduces the native output exactly, and original parameter inventories are restored. Raw reports and sealed studies are under `artifacts/mdb-auto-discovery-20261110/blind-{731,732,733}` in the domains workspace.

## First pretrained experiment

Guarded Beast job `job-745584d1b1ce` uses original Qwen2.5-0.5B-Instruct, CPU FP32 eager execution and one resident model. The task asks for the last digit of a two-digit sum. Eight discovery problems supply arithmetic-oracle labels. Eight fixed general-language prompts provide a native-distribution preservation term. Eight different arithmetic problems are excluded from search and evaluated only after selection.

The objective is mean correct-token log probability minus mean preservation KL. The controller makes 23 baseline/candidate observations, then evaluates native, virtual, compiled and restored arms. It selects `model.layers.6.mlp.down_proj` with gain 0, suppressing that MLP's write. The target weights are synthesized from the selected intervention; they were not supplied.

| Retained held-out measurement | Native | Compiled edit |
|---|---:|---:|
| Correct full-vocabulary first-token choices | 1/8 | 3/8 |
| Mean correct-token log probability | -3.51870 | -2.16052 |
| Mean preservation KL in nats | 0 | 0.65455 |
| Combined objective | -3.51870 | -2.81507 |

The observation is a task improvement with material collateral distribution change. The preservation prompts were reused from discovery, so their KL is a measured development cost, not held-out locality certification. Eight task examples do not establish broad arithmetic improvement.

On the saved held-out observations, virtual and compiled edits have identical target-token log probabilities, first-token decisions and four generated continuations. Restoring the original write reproduces native retained log probabilities and generated tokens exactly. Original parameter inventories restore. These comparisons concern retained outputs, not an independently compared full vocabulary or complete K/V state.

Generation uses fresh native prefixes and each branch's own tokens and K/V, with four new tokens and sampling disabled. The checkpoint's inherited repetition penalty of 1.1 remains active. Raw-logit first-token scoring and processed generation therefore have different consumer policies; their choices must not be treated as interchangeable. The generated strings also continue after their first digit. No full-answer formatting or completion-quality improvement is certified.

The deterministic initial sampler chose twelve of 168 eligible affine sites. In this model's lexicographic module layout, all twelve happened to be MLP down projections. The trial therefore establishes discovery within that recorded subset, not broad coverage of attention and MLP operators. Channel nomination is a magnitude heuristic. Both fitted-dose proposals underperform the best tested grid edit, and those counterexamples remain in the raw records.

The worker completed in 198.45 seconds; the complete scheduler lease took 238.45 seconds, peaking at 3,971 MB RSS with no GPU allocation. Original parameters remain retrieval-bound. The selected patch still pays the ordinary dense model computation; no inference speedup is claimed.

The independent [saved-only reader](../qualification/audit_wsp_auto.py) verifies the archive inventory, forecast/outcome seals and selection policy, recomputes correctness and signed score differences, and compares the controls. The workspace custody bundle is `artifacts/mdb-auto-discovery-20261110/native-03`, including `publication.json`, `guarded-receipt.json`, `saved-audit.json` and the complete collected evidence. Two earlier setup failures are preserved under `native-01` and `native-02`; neither loaded the model or contributes scientific evidence.

The [qualification record](../qualification/automatic-affine-discovery-20261110.json) binds executed package sources, scheduler payload custody, the archive and saved reader, all three constructed results and 76 passing installed-package checks. The dedicated reader verifies 346 retained files. The standard Saturn evidence sweep and verification also complete, with target-streams-only coverage over 23 timeline records and no typed receipts or address citations; that sweep does not replace the native-output audit.

## Python interface

```python
from mdb.wsp_discovery import AffineDiscovery

controller = AffineDiscovery(
    model, "./discovery", binding=binding,
    input_identity=discovery_inputs_sha256,
    source_identity={"script_sha256": script_sha256},
    max_sites=32, shortlist=3, channels_per_site=2,
)
edit, patch, evidence = controller.discover(evaluate_discovery)
```

The evaluator returns `observations` containing a finite scalar `objective`, plus `work` and `state` mappings. Larger objective is preferred. Task correctness, preservation cost and the chosen tradeoff remain visible independently. The controller does not treat that scalar as a scientific existence test or terminal certification rule.

Use `synthesize_affine_patch` and `virtual_edit` separately when integrating a qualified native state adapter. Run real models through mrun admission. The current Qwen worker and staging launcher are [wsp_auto_qwen.py](../qualification/wsp_auto_qwen.py) and [submit_wsp_auto.py](../qualification/submit_wsp_auto.py).

The [selective repair follow-up](SELECTIVE-WEIGHT-REPAIR.md) uses this discovered site and suppression direction to fit preservation-aware low-rank corrections, with fresh task and preservation evaluation. Better operator coverage remains open. The present layer-wide deletion supplies a causal lead and an automatic edit; its collateral cost motivates the narrower implementation.

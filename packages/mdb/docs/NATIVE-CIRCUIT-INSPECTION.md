# Inspect and test a native circuit in one MDB Project

Status: implemented bounded console/API workflow; locally tested on an untrained, 22,720-parameter Qwen3 CPU model. No downloaded model, Beast job, trained semantic circuit or large-model qualification is implied. The [capability roadmap](NATIVE-CIRCUIT-CAPABILITIES.md) and [mathematical reference](NATIVE-CIRCUIT-MATH.md) distinguish what this implements from future proof work.

## Run the tiny installed-package walkthrough

Install the public packages with their model support, then run the example with a fresh output directory:

```sh
python -m pip install '.[ar]' './packages/mdb[autoregressive]'
python -I packages/mdb/examples/project-navigation/circuit_understanding.py \
  --out /absolute/new/tiny-circuit-investigation
```

The example sets one CPU thread, constructs the public two-layer random Qwen3, inspects original ports, verifies root-recipe materialization, computes a full first-readout adjoint, compares four native branches, tests an intentionally incomplete interface, saves a finding and reopens the Project. It fits no model or lens. Its report includes the actual backend/library versions and all numerical diagnostics.

For the actual console flow, [native-circuit.mdb](../examples/project-navigation/native-circuit.mdb) runs the same selection/inspection/comparison commands through `mdb public --tiny qwen3 --project NEW_DIRECTORY --script FILE`. It requires no copied result IDs: execution uses the current focus's latest sealed plan, and the finding binds its latest completed result.

Real-model runs remain a later admitted qualification. Beast's mrun service was unavailable during this implementation; only small local mechanics and tiny-model runs were performed.

## Start at a live occurrence

The same commands work in an attached `mdb public` console. To try random CPU Qwen3:

```sh
mdb public --tiny qwen3 --project ./tiny-investigation
```

Inside the console:

```text
step 1
project focus port:hidden
project explain
project seek
project explain --view attention
project explain --view mlp
project explain --view effect
project explain --view frequency
project parameter model.layers.0.mlp.gate_proj.weight --selector '[0]'
```

One transition embeds the pending token and stops before layer zero. `focus` creates a sealed slice-backed selection with the source clock, port, consumer and transition budget. In this decoder, the default consumer is the first native winner's logit; choose an explicit contrast when useful:

```text
project focus port:hidden --consumer '{"port":"logits","token":4,"contrast":5}'
```

IDs are vocabulary IDs on this random fixture, not meaningful words. Normal frames retain the selected focus and mark a historical occurrence after the active session moves. `project focus` shows it again. To select a saved focus or a result's original focus, use `project focus --record-id RECORD_ID`.

For graph navigation, use `project focus --slice-id SLICE_ID --node-id NODE_ID`. A retained port version hydrates through its cut. A virtual transition/port version replays the bounded prefix of its original `continue` receipt, verifies the transition history and clock, and remains inspect-only. Purely descriptive nodes have no invented writable state.

## What the views contain

| View | Actual content and scope |
| --- | --- |
| Default explanation | Exact slice node, declared adjacency, frontiers, target and links to saved results; framework-free |
| `seek` | Bounded value sample, descriptor and native clock from an isolated Session; active branch preserved |
| `attention` | Captured native eager probabilities, source row/token IDs, query-head/KV mapping and weighted-value rounding remainder |
| `mlp` | Original normalized input, gate/up/activation/product/write descriptors; selected argument values; original named-parameter references |
| `effect` | Original-module full remaining-layer first-readout adjoint; native-primal comparison; signed neuron product partials and gradient×activity diagnostics |
| `frequency` | Exact source-axis correlation terms for the observed attention row, complex samples, unlisted-sum descriptors and return closure |
| `parameter` | A bounded transient original tensor view bound to the model identity; no parameter file is saved |

Native expansion currently supports resident public Qwen2/Qwen3 `DecoderAdapter` at a batch-one pending-token layer. It reuses the adapter's original layer executor and cache construction. Other adapters still support metadata navigation and generic native panels when their writable/consumer surfaces permit them. Unsupported native expansion reports the required boundary rather than fabricating values.

The MLP window here is one pending token, so there is no nontrivial token-axis MLP spectrum. The frequency view is a source correlation, not the square token-window operator from the research captures. Product magnitude only selects displayed neurons. Gradient×activity is not a finite-output attribution. Autograd follows the operator backward convention, not derivatives of BF16 rounding or greedy decisions; its primal agreement is recorded.

## Plan and run a native comparison

```text
project compare --edit '{"port":"hidden","selector":[0,0,1],"value":0.0001}' --edit '{"port":"hidden","selector":[0,0,1],"value":-0.0001}' --linearize
project compare --run
project finding --title "Selected-coordinate response"
```

The plan is sealed before execution. `--run` chooses the latest plan for this exact focus; an explicit plan ID can be supplied. Plans and result inspection do not execute manifest callbacks. Native work requires the attached compatible public Session, existing edit authority, and no pending intervention. The default logits comparison runs to one own-token commit with a finite 128-transition ceiling. `--tokens N --steps BUDGET` requests up to four own-token commits; no teacher token is supplied.

One edit produces sham/A; two produce sham/A/B/both. Edits are applied in declared A-then-B order through public Acts, preserving source selectors, native casts and invalidations. Numeric `add`, `replace` and `scale` are supported. Overlapping scale/replace edits are not labelled a smooth additive rectangle.

Each completed arm retains source/result cuts, original receipts, native decision, selected scalar and own tokens. The first pending-token layer also retains finite gate/up/interaction/remainder and routing/payload/interaction/remainder accounting. Native remainders are measured and retained. The finite final output is separate from this local accounting.

`--linearize` requests an additional first-readout adjoint for hidden edits. It reports a local prediction only after matching the entire primal logits descriptor to native. Other ports, later-token horizons or incompatible numerical lanes retain an explicit unavailable diagnostic without erasing native arms. Partial arm failures preserve completed evidence and leave the original branch selected.

`project records` lists sealed results. `project record RECORD_ID` shows one in full. A finding without a component uses the current focus and the latest completed result for it. Result references are validated against that source component/cut, and the existing Project graph links focus, plans, arms, results and findings through knowledge edges, not invented execution dependencies.

## Test a proposed state interface

Select source/result pairs from measured comparison arms and declare argument and result coordinates. The Python example uses the public API:

```python
arms = [project.circuit_record(sid, ref) for ref in comparison["arms"]]
cases = [{"source": arm["input_cut"], "result": arm["result_cut"]}
         for arm in arms if arm["status"] == "measured"]
test = project.test_interface(
    sid, cases,
    arguments=[{"port": "hidden", "selector": [0, 0, 0]}],
    result={"port": "logits", "selector": [0, 4]},
    name="one-coordinate-hypothesis", live=owner.adapter,
)
```

Console: `project contract --cases @cases.json --arguments @arguments.json --result @result.json --title NAME`. All JSON arguments also accept literal JSON. Files are data only and bounded to 64 KiB.

The tool verifies branch ancestry and matches native continuation programs/horizons before grouping exact argument values. It records pairs with identical proposed arguments and different results. Different operation paths are separate cohorts; an intervening edit or ambiguous producer cannot masquerade as a native-only transition. A failed interface is a useful dependency witness. Zero witnesses with no repeated arguments is not evidence of sufficiency, and a finite sample never becomes a universal semantic proof automatically.

In the tiny walkthrough, editing hidden coordinate one leaves coordinate zero unchanged but changes the selected downstream logit. The one-coordinate interface therefore has concrete counterexamples. This illustrates omitted state on an untrained model, not discovery of a language concept.

## Reconstruct from token/history metadata

Attach an already sealed `NativeReplay` recipe for the selected retained identity:

```text
project replay-bind --recipe @recipe.json --occurrence embedded
project seek
```

The recipe must match the exact model/execution and include the focused occurrence's expected cut hash. Subsequent inspection regenerates that target from the recipe's token roots/history and verifies it; expected hashes never install state. Recipe use is scoped to its exact target, so inspecting another comparison endpoint does not reuse the original root accidentally. Other unsupported roots still need retained payloads.

The Project currently retains its ordinary public cut pages for branch persistence. Circuit records add descriptors, compact measurements, logical references and recipes, not a duplicate weight or activation bank. This implementation does not make every historical payload disposable. A recipe-qualified target can be inspected without loading its old tensor payload, while generic cache eviction and every-family regeneration remain future work.

## Offline and persistence boundaries

```sh
mdb project ./tiny-investigation focus
mdb project ./tiny-investigation explain
mdb project ./tiny-investigation records
mdb project ./tiny-investigation findings
mdb project ./tiny-investigation graph
```

These metadata operations need no Torch, Transformers, scheduler or model. An offline `seek` or native view refuses with an attached-session requirement. Existing native cut/receipt/schema fingerprints are unchanged. Sealed circuit sidecar records and the mutable current-focus pointer are optional tables alongside Project schema 3; read-only older projects remain readable and simply have no circuit focus.

Tests cover tamper detection, pending edits, wrong sessions, partial panels, virtual replay, interface counterexamples, isolated native branches, source-frequency closure, original-module adjoints, root hydration without target tensors and framework-free reopen. Installed-wheel examples must report their actual numerical lane; a compatibility smoke on another library version does not qualify the pinned large-model lane.

See the [tiny qualification reports and deferred pretrained protocol](../qualification/native-circuit-workflow-20261112/README.md) for the actual results, raw evidence locations and numerical profiles.

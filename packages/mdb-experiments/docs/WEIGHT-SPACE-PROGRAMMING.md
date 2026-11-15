# Weight space programming with MDB

MDB accompanies weight-space reading, repair and construction with exact parameter comparisons, removable edits, mathematical response accounts, sealed forecasts and per-arm native outcomes. It uses the existing `ForecastTransaction` storage and offline verifier. The experiment runner accepts trusted Python callbacks for a network, decoder, diffusion suffix or environment consumer.

The subsequent [automatic affine discovery controller](AUTOMATIC-WEIGHT-DISCOVERY.md) adds bounded site/gain/channel search and parameter synthesis. It has constructed-organism results and a small pretrained Qwen experiment; the original companion qualification below remains a distinct historical mechanics result.

This is an experimental companion toolkit. Its executed qualification is tiny CPU mechanics. No published WSP benchmark, Qwen3.5 runtime, general semantic symbol resolver or cmachine model adapter is qualified by these additions.

## Experiment surfaces

| Investigation | Available surface | Supplied by the experiment |
|---|---|---|
| Static weight reading | `weight_diff`, `parameter_inventory`, static study arms | Actual tensors, architecture, coordinate interpretation and proposed roles |
| Direct parameter repair | `ParameterPatch`, byte hashes, tied aliases and exact uninstall | Replacement tensors and original checkpoint identity |
| Network construction | A replacement patch over a declared initial network | Initial architecture and parameters, constructed tensors and evaluation environment |
| Direct write prediction | `linear_response`, `swiglu_response` | Bound weights and same-receiver operands |
| Weight by receiver analysis | `receiver_factorial` and existing response audits | Four genuine or prospectively predicted corners |
| Dose, ablation and controls | Independently frozen arms and scaled patches | Selected sites, control generation and their applicability |
| Activation mediation | Sealed prediction/payload arrays exposed to native callbacks | Bounded injection/removal through a compatible native adapter |
| Continued policy, text or image execution | Per-arm callbacks, state-policy declarations and native observations | Exact state restart or compatible parent, own-state continuation, real consumer and work ledger |
| Symbol-table integration | `bind_patch_ports`, per-arm `symbol_refs` and actual parameter names | A source-bound `NativePortCatalog` or public symbol records and evidence for proposed semantics |

The interfaces do not discover semantic labels automatically. A supplied source link is retained as supplied evidence. Static experiments can stop at the global seal and need not spend any native forward calls.

`bind_patch_ports(candidate, catalog)` joins a patch to an existing `NativePortCatalog`. It verifies the trace seal, model identity, original parameter bytes and declared aliases, then returns the actual parameter ports and first recorded consumers with the trace's input/execution identity. Pass the result's `parameters` as arm `symbol_refs` to retain those joins in the forecast seal. Recording a trace executes the original model; count that separately and do it only when the experimental feedback policy permits it. First-use dataflow remains structural evidence rather than a semantic or coordinate-level causal claim.

## Run the tiny example

Use a declared installed MDB environment with the discovery extra for numerical operations:

```sh
python -m pip install 'mdb-runtime[discovery]'
mdb wsp demo ./toy-wsp
mdb wsp verify ./toy-wsp/study
mdb wsp inspect ./toy-wsp/study/patches/edited
```

The demo constructs a two-action linear policy and a supplied terminal action. It forecasts all four arms before their first native execution. Native and zero-dose arms repeat action 0; the edited policy selects terminal action 1; restoring the original write reinstates repetition. Each nonterminal step consumes its own selected action. NumPy predicts states, Torch executes the policy, all parameter writes uninstall, and the ordinary forecast verifier audits the saved result.

This demonstrates the instrument on an organism whose mechanism and environment were supplied. It makes no claim about learned reasoning or reproducing the published Qwen repair. The output includes `report.json`, globally sealed forecasts, persisted edited tensors and native observations.

## Inspect what actually changed

```python
from mdb.wsp_weights import weight_diff

report = weight_diff(original_state, edited_state, support_limit=64)
```

`original_state` and `edited_state` are matching mappings of named Torch tensors. Mappings may lazily hydrate one tensor at a time from a sharded checkpoint. Shapes and dtypes must agree. Reports retain before/after raw-byte hashes, actually changed elements, numeric differences, FP64 delta norms and capped row/column support. Byte changes include signed zero. The denominator explicitly counts named tensor elements, including aliases; it is not silently presented as the logical model parameter count. No automatic SVD, rank-one assumption or scientific threshold is applied.

For two single-file checkpoints:

```sh
mdb wsp diff original.safetensors edited.safetensors --output weight-diff.json
```

For a live module, `parameter_inventory(model)` separately deduplicates identical parameter objects and retains their declared aliases. It hashes parameters, not buffers, caches or an environment. Full tensor hashing and finite checks are paid integrity work, even when the edit is sparse.

## Install and remove a candidate

```python
from mdb.wsp_weights import ParameterPatch

candidate = ParameterPatch(
    model, {"model.layers.15.mlp.down_proj.weight": edited_down_proj},
    model_identity={"checkpoint": pinned_checkpoint_sha256},
)
candidate.save("./candidate")

with candidate.installed(model):
    result = run_fresh_native_consumer(model)

zero = candidate.scaled(model, 0)
negative = candidate.scaled(model, -1)
restored_candidate = ParameterPatch.load(
    "./candidate", model, model_identity={"checkpoint": pinned_checkpoint_sha256},
)
```

The patch owns copies of the original and replacement tensors in memory. It validates every touched parameter before writing, supports identical tied `Parameter` aliases, and refuses distinct parameter objects sharing touched storage. Shape, dtype, alias topology, original bytes and target bytes are checked. Uninstall restores original parameter bytes even if native execution raises. Rebinding a parameter during execution invalidates restoration. Nested installs cannot silently overwrite another candidate's baseline.

Saving persists edited tensors and source descriptors, not original parameter copies. Existing lossless tensor archives preserve BF16 bits. Other floating types may be stored through exact FP32 widening and explicitly cast back on load; the target's original dtype and byte hash must still match. Parameter archive loading retains the existing reader's 256 MiB per-payload limit. Inspecting metadata checks its seals; loading verifies tensor payloads and source bindings. Independent expected identities remain necessary for external authenticity.

Dose 0 and dose 1 preserve the corresponding exact original and edited bytes. Other doses compute `original + dose * (edited - original)` in FP64 and round to the original dtype. This is a declared dose construction, not a claim that it matches the editing agent's original FP32 update sequence.

Restoring parameters does not restore K/V, recurrent buffers, RNG, optimizer state or an environment. A whole-checkpoint comparison should recompute its prefix under its own weights. A live hotpatch retaining old history must declare that history. A local mediation arm must use a compatible parent and the adapter's own intervention receipts.

## Seal and execute the experiment

```python
from mdb.wsp import WeightStudy

with WeightStudy(
    "./weight-study", model=model,
    binding={
        "model": {"checkpoint": pinned_checkpoint_sha256},
        "checkpoint": pinned_checkpoint_sha256,
        "execution": "declared backend and dtype",
        "consumer": "native decoder with supplied sampling policy",
        "horizon": "declared token or environment-step budget",
    },
    input_identity={"input_sha256": input_sha256},
    source_identity={"script_sha256": script_sha256},
) as study:
    study.tx.frontier("receiver", source_bound_inputs)
    study.add_arm(
        "edited", role="edited", patch=candidate, state_policy="restart",
        context={"prompt_family": "declared discovery family"},
        symbol_refs=source_bound_symbol_records,
        predictor=lambda context: {"predictions": predict_write(context)},
    )
    # Add native, dose, sham and relevant mediation controls before sealing.
    study.seal()

    def execute(context):
        # context.model has the declared patch; context.prediction and
        # context.payloads contain this arm's sealed values. context.frontiers
        # contains only the supplied roots and frozen arm declarations.
        state, reset_receipt = adapter.fresh_state(context.declaration)
        observed, work = adapter.continue_own_state(
            context.model, state, context.prediction, context.payloads,
        )
        return {"observations": observed, "work": work, "state": reset_receipt}

    study.run("edited", execute)
```

This is an adapter sketch, not a shipped arbitrary-model decoder. `predict_write`, `source_bound_inputs` and the state adapter belong to the experiment. Actual native entry is forbidden while forecasts are being prepared. For an observed local frontier, supply a separately captured source-bound parent; this WSP wrapper does not itself implement prefix capture. Existing public Sessions and `NativePortCatalog` provide the execution and port surfaces for compatible models.

Arms persist their edited tensor artifacts before the global seal. The original forecast transaction inventories those files, so native release also detects changed patch artifacts. Predictions, contexts, parameter manifests and candidate identities cannot be changed after release. Native results own their bytes before parameter uninstall. BF16 result tensors are retained as `bf16-bits-v1` wrappers with unsigned 16-bit arrays.

The study hashes all parameters at initialization, at sealing and before/after each native arm. Any parameter drift outside successful patch restoration stops execution. This detects mutation but does not promise rollback of an undeclared mutation. Per-arm hash byte counts are retained separately from consumer work. Initial/seal inventories, parameter-patch validation and cloning add further work. Model buffers and adapter-owned state are outside the parameter inventory.

Each successful callback returns observations, an explicit work ledger and a state receipt. Ordinary callback exceptions produce a retained `evidence-incomplete` row and allow subsequent arms to execute after parameter restoration. An exception that prevents a callback from returning also prevents it from returning its detailed work ledger; root/module entry counters remain available. Integrity or restoration failures stop the transaction. Nonfinite or malformed numerical evidence can also stop storage publication. Completed observations remain in custody.

`mechanics_status`, `instrument_status`, `trend_status` and `terminal_status` remain separate. There is no default benchmark threshold or all-controls-must-win promotion rule. A valid observation is not automatically a favorable scientific trend.

Closing the study writes `wsp-outcomes.json`, binding the final forecast event head, each native observation and its numerical payload files. `mdb wsp verify` checks this outcome manifest as well as the original forecast seal and event receipts. It reports declared arms that did not execute. Preserve the returned manifest identity in independent custody and use `--expected-manifest ID` and, where available, `--expected-seal ID` for externally pinned verification. Without independent pins, verification establishes local hash consistency. The existing generic `mdb forecast verify` retains its original scope; use the WSP verifier to check this wrapper's post-release outcomes.

## Compare direct and receiver-dependent responses

```python
from mdb.wsp_response import linear_response, swiglu_response, receiver_factorial
from mdb.response_closure import measure_error, audit_factorial

local = linear_response(down, edited_down, hidden)
terms = swiglu_response(gate, up, down, edited_gate, edited_up, edited_down, x)
factorial = receiver_factorial(base_at_base, edited_at_base, base_at_edited, edited_at_edited)
numerical_error = measure_error(local["edited"], measured_native_write)
```

Matrices are output-by-input and equations use FP64. `linear_response` separates baseline, edited write and direct change. `swiglu_response` separates output-map, gate, payload and gate-by-payload interaction. `receiver_factorial` returns weight, receiver, interaction and total signed arrays. Their finite-arithmetic remainders and downstream effects are measured separately; norms are not additive causal percentages. The existing equation catalog and response/factorial audits can retain those comparisons under model/input-bound conditions.

## Execution limits and qualification

The wrapper is single-writer trusted Python, not an OS sandbox. Module `__call__` hooks guard the supplied model. Direct `.forward`, functional tensor operations and other models can bypass those hooks. Premature intercepted entries remain latched even if a callback catches the exception. Root forward calls consume the transaction's `max_native_calls` budget; module entries are counted separately. They are not editing-agent request counts, which may include batches of many sequences. External consumers require their own physical-work ledger. Cooperative wall limits do not preempt callbacks.

`inspect`, `verify` and help require no model framework; numerical parameter operations need Torch and response equations need NumPy. Existing core schema and address identities are unchanged. The wrapper uses an explicitly externally managed forecast transaction so parameter mutation does not relax the original fixed-weight CPU FP32 transaction contract.

Use mrun admission for real-model or multistep scientific campaigns in the domains workspace. Bind retrieved parameters to immutable original archives with `parameter-artifacts.json` and a standalone retrieval script. This toolkit's patch archives are derived candidates, not substitutes for the original checkpoint retrieval contract. Qualify each architecture/backend/dtype and its cache/environment state adapter independently before making replay or continuation claims.

The separate [Qwen3.5 case study](../qualification/wsp_case_study/README.md) now reconstructs the published sparse edit, verifies both target tensor hashes, seals local write forecasts, and runs native substitution/restoration panels. Whole-projection substitution matches the edited model and restoration matches the original on three 64-token prompts and three 2,048-token exploratory prompts, including exact full first logits and own-token sequences. All greedy trajectories reach their caps; these panels establish bounded execution fidelity rather than completed loop repairs. Architecture-specific hybrid cache handling and backend limits are documented in that study. They do not extend the wrapper into a general Qwen3.5 Session adapter.

A sampled follow-up reproduces a correct completed rectangle answer under the edit and substitution, while the original and restoration cap. Pencil arithmetic remains capped, and a literal instruction takes substantially longer under the edit. Each context has one paired seed. Raw local targets expose 1.42–18.70% error relative to the smaller BF16 edit response despite roughly 0.16% endpoint error. The original native job fails during report serialization after all arms finish; a separately declared CPU recovery preserves and verifies its saved outcomes without native reexecution. See the case study for that later custody boundary and its limits.

The separate [paired workflow pilot](../qualification/wsp_workflow_comparison/README.md) uses the same named reasoning editor in three workflows on three blinded constructed networks, with two editing rounds and a shared native-call budget. All workflows recover the supplied functions to FP64 precision and preserve the healthy control. MDB uses 16 discovery calls versus 13 for competent PyTorch hooks, with 53.90 versus 0.0179 numerical-program seconds under the declared persistent-HDD profile. Its fewer output tokens are offset by additional input tokens. A separate NumPy reader checks retained outcomes and cost custody. This ceiling-limited toy pilot does not establish a practical advantage or rule out advantages on harder/pretrained tasks; its earlier numerical-kernel failures remain separate calibration evidence.

The [pretrained follow-up qualification](../qualification/wsp-native-workflow-comparison-20261111.json) evaluates actual down-projection weight gains on 48 new Qwen2.5-0.5B arithmetic prefixes. Original, hook-workflow and MDB selections get 10, 11 and 8 next digits correct. MDB has lower mean preservation KL than hooks (0.02879 versus 0.06349 nats), with higher program time (302.87 versus 105.94 seconds). Two kernel errors and one provider deadline remain explicit limitations; separate fixed-program replays validate the kernel fixes without revising primary outcomes. The editor uses ordinary hooks rather than MDB capture in this native panel, while all 27 MDB probes use installed parameter patches and sealed studies. No overall practical advantage, generation benchmark or universal workflow ranking is established.

The [storage component qualification](../qualification/wsp-workflow-storage-20261111.json) crosses medium and study grouping. Three separate HDD studies take a median 10.15183 seconds, versus 0.02176 seconds on tmpfs; one predeclared shared study reduces those times to 6.37529 and 0.01565 seconds. Native outputs and parameter restoration match throughout. Python `os.fsync` accounts for about 99.4% of the HDD median. Tmpfs volatility, later custody copying, shared-study semantics and residual bookkeeping cost remain separate; these component times are not end-to-end speedups.

The [November 10 qualification record](../qualification/wsp-mechanics-20261110.json) binds exact source and wheel hashes, 71 passing isolated installed-wheel checks, the four-arm toy outputs and the outcome/forecast identities. It used Torch 2.13.0 with the declared public Saturn-pub dependency outside the workspace. The supplied toy reaches completion only under the edit; restoring the original write restores repetition. Every state forecast is exact on those four arms and original parameter bytes are restored.

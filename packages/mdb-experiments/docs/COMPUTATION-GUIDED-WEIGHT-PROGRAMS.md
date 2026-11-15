# Compiling a computational request into weight edits

This experiment connects an authored semantic request, historical interpretation evidence, native causal interventions, a fitted weight correction and an unchanged model consumer. It extends the [selective repair machinery](SELECTIVE-WEIGHT-REPAIR.md) with an executable request and an explicit computation account. It is an experimental prototype, not an automatic semantic decompiler.

The workspace already has [neural recompilation experiments](../../../../saturn/experiments/2026-10-29-neural-recompilation/FINDINGS.md) that install conditional semantic changes and fold a hooked teacher into adapters through optimization. Those findings also retain locality losses after folding. This experiment adds a small public request/account interface, direct least-squares weight construction without optimizer steps, a matched random-response comparison and content-bound fresh-process application. It does not introduce semantic model editing for the first time. The earlier folding findings are a relevant precedent for interpreting any applicability loss here; they were not among the three records supplied to this run's frozen account.

The request is: **when the first operand's units digit is 7, use 3 at that symbol instead, while retaining other digit and position contexts.** This changes the intended arithmetic behavior. Success means following the authored rewrite; it does not mean improving ordinary addition accuracy.

## Interpretation actually used

The staged computation account pins Qwen2.5-1.5B-Instruct checkpoint SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`, the prompt prefix, absolute source-token position and execution profile. It includes content hashes and copies of three prior findings: causal reader roles, operand-row interaction constructors and the Cmachine dictionary/operand steering study.

The account supplies two writer candidates for `operand.a.column0`:

1. `model.layers.21.mlp.down_proj`: an additive residual write before the known layer-22 operand read. The desired displacement is the whole counterfactual residual difference leaving layer 21, added through this writer.
2. `model.layers.22.self_attn.v_proj`: the values delivered to known late readers. Here the desired displacement is the actual counterfactual value-projection difference.

`DigitRewrite.lower` resolves these writers from the account. There is no whole-model layer scan or activation-magnitude site nomination. The historical findings supply hypotheses; differences in their original prompt and numerical profiles do not silently become qualification for this new execution.

The input role, source/target digits, arithmetic oracle and source locations are supplied. Native contrast capture estimates a reusable response direction. Native development outcomes select a writer, preservation fit weight and dose. No frontier WSP reasoning agent is part of this numerical prototype.

## Experiment sequence

One mrun lease runs one resident model on CPU FP32 eager execution. The known fixed prefix has 45 tokens; the first operand's units digit is token 47. The matching cases restrict the second operand's units digit to 0, 1 or 2. Consequently, 7→3 changes no carry, and the first two answer digits are shared by the original and rewritten sums.

1. Capture sixteen paired native/counterfactual development contexts, eight position controls and eight general preservation prompts. The paired prompts differ in the requested source digit. Preserve native input keys, mean response displacements and a bounded sample of other-position/context inputs.
2. Test each response at its intended source and at the other operand's units position. Run the unchanged suffix. These experiments select between the two evidence-derived writer locations.
3. Fit rank-one corrections mapping the source inputs to the interpretation-derived displacement while penalizing responses on preservation keys. Test preservation weights 1 and 100, each at doses 0.5, 1 and 1.5.
4. Give a baseline six candidates at the same selected site, with the same source/preservation keys, fitting procedure and rank. Its response directions are random and matched to the semantic direction's norm. Actual fitted correction norms are retained for comparison.
5. Freeze the best observed candidate separately for each family. Only then evaluate new first operands, new position controls and new general prompts.
6. Compare native, conditional semantic program, compiled semantic weights, factored semantic response, wrong-location program, compiled random baseline and restoration. Retain failed candidates and signed effects.

Development selection scores the final answer digit from the full vocabulary after supplying the two unchanged answer digits. Its objective is mean rewritten-target log probability minus preservation KL on controls and general prompts. This supplied-prefix score is separate from final generation: each evaluation arm also produces three answer tokens from its own raw prompt and native K/V, with no answer digits supplied. Generation is greedy with repetition penalty 1.0.

The conditional state-program reference uses an externally bound row selection and a development-derived direction. Its controls receive no intervention by construction. The compiled parameter arms have no row mask, digit predicate, donor call or answer lookup in their forward path. Their applicability and collateral behavior must therefore be measured rather than inherited from the conditional reference.

The other-units-position control is informative about response transport but cannot by itself prove operand identity: addition is commutative, and a translated value at either operand can sometimes produce the same final digit. Separate controls put 7 in another column or in the second operand while leaving the first operand's units digit different from 7.

## Package interface and evidence

`mdb.wsp_program.DigitRewrite` expresses the literal requested transformation and lowers it against a supplied, model-bound account. [The worker](../qualification/wsp_program_qwen.py) executes its native interpretation and weight realization; [the launcher](../qualification/submit_wsp_program.py) pins the evidence and ships installed public packages. It reuses `WeightStudy`, `fit_low_rank_delta`, `virtual_low_rank_edit` and `ParameterPatch`.

Each phase seals declarations before native release. The account, recipe, historical sources, contrasts, fitted factors, all candidate outcomes, parameter patches and full evaluation logits remain in the evidence tree. Original pretrained constants remain retrieval-bound through `parameter-artifacts.json` and standalone `retrieve_weights.py`.

The [saved-only audit](../qualification/audit_wsp_program.py) verifies inventories, original publication, recipe derivation, role-source identities, split separation, both selection policies, actual norm matching, complete-vocabulary scores, exact generated answers, factored/compiled differences and restoration. It reports original-arithmetic accuracy separately from compliance with the new request.

The installed public-package mechanics suite passes 83 tests, including exact reconstruction of a frozen correction in a fresh model. The initial lease completed discovery and part of evaluation before reaching its scientific wall budget. Its outputs are retained under `artifacts/mdb-computation-program-20261110/native-01`; it is not a completed qualification.

The [replay worker](../qualification/wsp_program_replay.py) and [replay launcher](../qualification/submit_wsp_program_replay.py) separate application from discovery. A packet retains the closed capture/causal/discovery studies, request, interpretation account, fitted factors and original patch manifests. The new process fits nothing and makes no selection. It reconstructs both selected corrections from the original checkpoint and requires their complete parameter manifests, including edited tensor hashes, to match the frozen candidates before native execution. The packet also retains the original inputs, including the unchanged held-out split. Replaying that split after a time-limited partial evaluation supplies a fresh-process comparison; it is not a second independently held-out split.

## Measured result

Discovery selected the layer-21 down projection and a rank-one correction, preservation weight 100, dose 1.0. All six random-response candidates failed to produce any of the sixteen development rewrites. Their response norms matched the semantic response; the fitted correction norms match within 3.52e-8 relative at each corresponding preservation weight. Each family received the same six-candidate dose/fit budget. Their independently selected doses differ: the random baseline selected dose 1.5.

Completed replay job `job-f9ca124e108d` produces the following complete three-token answers from fresh raw prefixes. These are sixteen held-out first operands, eight held-out position controls and eight new general prompts. They are the same frozen evaluation set partly opened by the time-limited original run; there is no new fit or selection during replay.

| Arm | Requested rewritten sums, /16 | Original sums on target cases, /16 | Correct position controls, /8 | Mean control KL | Mean general-prompt KL |
|---|---:|---:|---:|---:|---:|
| Native | 0 | 16 | 8 | 0 | 0 |
| Conditional semantic program | 16 | 0 | 8 | 0 | 0 |
| Compiled semantic weights | 16 | 0 | 5 | 1.268251 | 0.000508 |
| Factored semantic response | 16 | 0 | 5 | 1.268248 | 0.000508 |
| Wrong-location conditional program | 0 | 12 | 8 | 0 | 0 |
| Best random-response weights | 0 | 15 | 8 | 0.000022 | 0.001589 |
| Original-write restoration | 0 | 16 | 8 | 0 | 0 |

For example, the edited model answers `347+342=` with `685`, which is exactly `343+342`. The unchanged model answers `689`. No modified prompt, external digit predicate, answer table or donor call is used by the compiled model.

KL is measured from the separate next-token scoring forwards, not from a comparison over every generated-token distribution. Supplied-prefix final-digit scoring preserves six of eight controls; own-state three-token generation preserves five. Each generated answer is scored within a three-token budget; stopping or prose after that budget is not evaluated.

The three control failures expose an applicability loss:

| Unmatched control | Expected unchanged sum | Compiled output | Endpoint interpretation |
|---|---:|---:|---|
| 120 + 127 | 247 | 243 | Exactly the result of rewriting the second operand's units 7 as 3 |
| 308 + 297 | 605 | 601 | Exactly the result of rewriting the second operand's units 7 as 3 |
| 471 + 371 | 842 | 802 | Consistent with rewriting one tens-column 7 as 3; commutativity prevents assigning which operand from the answer alone |

These endpoint matches support a **working inference** that the value transformation transfers more broadly than the requested operand/column binding. They do not identify the internal selector or establish a unique causal explanation for every failed control. The conditional program guarantees no writes to controls by its externally supplied selection; compiling the same response into a shared matrix does not inherit that guarantee. The next informative measurement is the fitted selector's response at separately controlled operand/column positions through native continuation, with sample roles retained. A richer native gate may be needed; rank-one affine expressibility has not been established for the complete requested condition.

The combined control/general preservation KL is 0.634379 nats, against 0.000805 for the selected random baseline. Reporting only the general prompts would hide the important structured collateral. Interpretation supplied a useful response direction in this assay; the result does not establish overall superiority across arbitrary edits or a safe bounded promotion.

## Numerical and custody checks

The factored and compiled paths have identical final-digit decisions and generated answer IDs on all twenty-four arithmetic rows. Maximum difference across all thirty-two saved full-vocabulary next-token logit rows is 2.93255e-5. Original-write restoration reproduces every saved native logit exactly and all generated IDs. The final parameter inventory restores exactly. No complete K/V-array equality claim is made.

The fresh process reconstructs both selected patch manifests exactly, including the edited tensor hashes, before native release. The three arms completed by the original run—native, conditional semantic program and compiled weights—have bit-identical saved logits and identical generated IDs in replay. This is reproduction of frozen candidates and inputs, not a second independent held-out sample.

The saved-only audit verifies 323 inventoried files, four closed studies (the three completed source phases plus the complete replay evaluation), both selection policies, role-account/recipe identities, split separation, actual norm matching, full-vocabulary logit reductions and complete generated answer IDs. The publication archive is 367,824,965 bytes, SHA-256 `78eb2135746f679ea075ac6f7ead9e55e579d37bf494f111e904798ce0e4900e`. Declared, sealed and executed replay payload hashes agree.

Replay scientific execution takes 246.33 seconds; its complete lease takes 273.97 seconds, with 8,853.7 MB peak RSS and no GPU allocation. Initial discovery took longer than planned and stopped after three evaluation arms when its declared scientific budget expired. Its failed scheduler status, all source rows and unexecuted-arm list remain unchanged. The source-phase audit verifies the completed capture, causal and discovery studies separately. The replay request performs no fitting, discovery or automatic retry.

The standard Saturn evidence run and verification finish with target-streams-only coverage over seven replay timeline records and no additional typed receipt or address-citation coverage. Numerical evidence comes from the dedicated reader. The [qualification record](../qualification/computation-guided-weight-program-20261110.json) binds source hashes, public wheels, scheduler custody, publication, saved audit, fresh-process comparison and 83 passing installed-package tests.

The workspace custody bundle is `artifacts/mdb-computation-program-20261110/replay-02`, with the incomplete original run retained under `native-01`. This prototype demonstrates a useful part of the programming loop: an explicit computational request supplies a response that can be realized in ordinary weights on new contexts. It also exposes the next missing part—reliable compilation of the request's contextual binding. Broader prompt transfer, carry changes, automatic role discovery and general repair remain separate experiments. Terminal certification is not evaluated.

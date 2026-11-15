# Compiling operand and column binding into weight edits

This follows the [computation-guided weight program](COMPUTATION-GUIDED-WEIGHT-PROGRAMS.md). Its rank-one correction implements an authored first-operand units-digit 7→3 rewrite, but also changes some second-operand and tens-column controls. The next question is whether we can retain its useful response while improving the condition under which ordinary weights apply it.

The experiment holds the original response direction fixed. It records the correction's scalar at every token, performs position-scoped causal cuts, and fits a new linear selector using explicit digit × operand × column counterexamples. The native model remains the downstream consumer. Role labels, addresses, arithmetic labels and the response are supplied; automatic semantic role discovery is not part of this assay.

## Instrument and construction

[`fit_response_selector`](../src/mdb/wsp_binding.py) solves a sample-space ridge problem with target coefficient one on positive inputs and zero on preservation inputs. Its left factor is exactly the supplied response direction; only the right factor is fitted. No optimizer or gradient steps run. The factors can be applied through the existing virtual response or compiled into an ordinary output-projection matrix with `synthesize_low_rank_patch`.

The specimen is Qwen2.5-1.5B-Instruct, CPU FP32 eager execution. It reuses the original layer-21 down-projection writer and exact checkpoint. The fixed 45-token prefix binds A tens/units to positions 46/47 and B tens/units to positions 50/51. The worker checks the token identities at each declared role on every input.

Four development blocks each compare 7 versus 3 at four separate roles, producing 32 contrasts. Three previously observed failures join development as diagnostics. The new fit adds structured negative keys from the other operand/column positions, nonmatching source rows and a bounded sample of other states to the prior preservation sample. Six candidates vary preservation strength and dose, with the write direction unchanged. Selection maximizes development rewritten-target final-digit log probability minus control KL, after supplying the unchanged two-digit answer prefix.

Fresh evaluation is a separate finite lease that reconstructs the selected correction from frozen factors and requires exact edited parameter manifests before release. Eight new blocks provide eight targets and 56 controls, with first operands excluded from all upstream and development cases. Complete three-token answers start from raw prompts and consume the model's own tokens and native KV. This measures a small fixed-format domain, with no carry change on target rewrites.

## Diagnostic cuts

The discovery reference applies the original factored correction everywhere, only at A units, or everywhere except A units. The latter two use external positional masks as causal instruments. They are not weight-only deployment candidates. Potential selector coefficients are retained even where a diagnostic mask prevents the write; the original model makes no correction despite recording its potential coefficient.

At layer 22 attention output and layer 23 residual output, the worker retains native downstream vectors at operand positions, the equals token and the last supplied answer digit. It also records selector coefficients during each model-generated step, separately from supplied-prefix scoring. These observations distinguish the fitted selector, the effective write scope and the later computation that consumes the write.

The first development comparison produces:

| Arm | Requested own-state rewrites /4 | Correct unchanged controls /31 |
|---|---:|---:|
| Native | 0 | 31 |
| Original full correction | 4 | 24 |
| Correction only at A units | 4 | 29 |
| Correction everywhere except A units | 0 | 28 |

Restricting the write to the intended source reduces the number of control errors from seven to two while retaining all four rewrites. Writes at other positions can also change controls without implementing the target rewrite. This supports investigating both applicability at the intended source and writes elsewhere; it does not isolate every control failure to a unique internal cause.

## Fresh answers and remaining collateral

Discovery selects preservation weight 1 and dose 1. The fixed response is fitted against 20 positive keys (16 historical plus four new) and 288 preservation keys (128 historical, 96 distinct structured role keys, and 64 other-position samples). Every candidate and the two preservation strengths remain available in the raw record.

The saved-only reader independently reconstructs the positive/preservation inputs from native captures, recomputes both selector solutions, verifies the frozen selection, checks scalar traces against the factors, and recomputes scores from complete-vocabulary arrays. Complete own-state answers on the fresh set are:

| Arm | Requested rewrites /8 | Correct controls /56 | Controls identical to native /56 | New errors among 54 initially correct controls |
|---|---:|---:|---:|---:|
| Native | 0 | 54 | 56 | 0 |
| Original compiled edit | 8 | 46 | 46 | 9 |
| New compiled selector | 8 | 53 | 55 | 1 |
| New factored selector | 8 | 53 | 55 | 1 |
| Restored native | 0 | 54 | 56 | 0 |

The original edit also repairs one pre-existing control error, so its net loss of eight correct answers differs from its nine newly introduced errors. The new selector retains both pre-existing native errors and introduces one new error. This distinction matters when the goal is to preserve the model's original computation outside the requested condition.

Mean control KL on the supplied-prefix final-digit forwards falls from 0.740322 to 0.000029786. That observation has a narrow scope. The remaining full-answer failure is `432+262`: native produces `694`, while both new realization paths produce `700`. The first generated digit changes, even though the final-digit distribution after supplying `69` remains close to native. Final-digit preservation is therefore an incomplete observable for full-answer preservation.

At this control's A-units source, the fitted scalar is approximately −0.09537 despite the digit being 2. This directly observes applicability outside the intended condition. Its causal contribution relative to writes at other positions has not yet been isolated for this new patch. The earlier scoped cuts apply to the original correction.

The new factored and compiled paths produce identical generated IDs on all 64 rows. Their maximum saved next-token logit difference is 1.33514e−5. Restoration recovers every saved native logit exactly and identical answer IDs. These numerical comparisons do not assert equality of complete KV arrays or every generated-token distribution.

The next discriminating cut is the remaining control: compare the new correction at A units alone with its writes at other positions, while capturing the native reader that chooses the first answer digit. The next candidate-ranking observable should cover all answer positions and own-state continuation. Reusing the opened evaluation cases for development requires a new evaluation split afterward.

## Execution and evidence

The [worker](../qualification/wsp_binding_qwen.py), [launcher](../qualification/submit_wsp_binding.py), [contrast protocol](../qualification/wsp_binding_protocol.py), and [saved-only reader](../qualification/audit_wsp_binding.py) ship sealed runtime wheels. Each scientific phase has one guarded mrun lease, strict preflight, one resident model, fixed physical chunks of at most 32, and no automatic retries or children. Arms restart from the unchanged checkpoint; editing the prefill can affect any token, so sharing an edited native prefix without qualification would change the intervention contract.

The first staging attempt was refused for an overlong queue note. The first admitted job, `job-07eaed01e415`, was killed during model loading: the CPU default batch ceiling prevented intent sizing and the fallback reservation was too small. Both attempts remain preserved. The corrected discovery, `job-2a86791b327a`, declares the prior measured batch-32 profile through mrun's policy override and receives a policy-sized reservation. The original replay `job-f9ca124e108d` supplies the profile evidence, including its 8,853.7 MB measured peak RSS. This is a correction to admission, not scientific evidence from the killed job.

An additional custody check found that the common launcher's compatibility `PYTHONPATH` exposed private workspace mrun sources while staging the initial runtime wheels. Those bytes were sealed and executed, but they did not come from the installed public mrun distribution. The initial discovery and evaluation remain measured evidence with that recorded provenance; they do not qualify an all-public stack. The [correction record](../../../../artifacts/mdb-role-selector-20261111/runtime-provenance-correction.json) retains the source hashes and affected jobs. The payload builder now rejects shadows of ordinary installed packages, and the launcher binds runtime package content digests into request identities. Public-runtime replay clears the compatibility path and reconstructs the same frozen factors and edited parameter manifests without fitting or selection.

The workspace artifact root is `artifacts/mdb-role-selector-20261111`. The earlier weight-program artifact tree remains unchanged. Full vocabulary scoring arrays, generated token IDs, selector traces, native consumer vectors, fitted factors, candidate outcomes, parameter manifests and standalone original-weight retrieval metadata are retained. Scientific outcomes and native mechanics are reported separately from terminal certification. The public saved-data reader `job-d1afaa296c3d` verifies 146 discovery files and 81 initial-evaluation files, with zero model loads or forwards. A prior reader's serialization error remains a failed instrument attempt, with its raw log retained.

Public-runtime replay `job-eff6e11ba683` completes successfully. The final public saved-data reader, `job-f471e27c5921`, verifies the discovery, initial evaluation and public replay, and confirms bit-identical saved full-vocabulary logits and identical generated IDs for every one of the five replay arms. Input bytes, selected factor bytes and edited parameter manifests also match. This reproduces the same frozen evaluation set in a new process and runtime package environment; it does not add an independent sample.

The public replay takes 321.40 seconds of scientific execution. The final reader takes 21.74 seconds for its complete lease, peaks at 940.3 MB RSS, and loads no model or performs any forward pass. Ten focused research-mechanics tests and fourteen payload/public-mrun tests pass; five relevant regressions also pass against installed wheels outside the workspace. Ruff passes. The [qualification record](../qualification/role-qualified-weight-program-20261111.json) binds executed sources, runtime wheels, declared/sealed/executed payload identities, archives, numerical audits, remaining failure and the provenance correction.

The standard Saturn evidence sweep and verification return zero for discovery and both evaluations, with target-streams-only coverage over ten/five/five timeline records. They add no typed receipt or address-citation coverage; the dedicated reader supplies the numerical evidence. Prompt transfer, carry-changing requests and automatic role discovery remain separate experiments; no terminal certification is claimed.

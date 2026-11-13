# Compose two discovered memory reads across fresh receivers

This assay follows [MDB causal read discovery](../../docs/CAUSAL-CIRCUIT-DISCOVERY.md)
and its [trained qualification](../CAUSAL-CIRCUIT-DISCOVERY-VALIDATION.md).
The question is whether a frozen two-read program predicts a joint intervention
when its first edit changes the receiving state of the second read.

The architectural motivation and existing proof are documented in the
[MDB blog post](../../../../../obsidian/blog/2026-11-07-000659-the-read-was-observed-before-the-future-was-tested.md).
Astra reviewed the mathematical design and concrete worker before admission.
The earlier FLUX work already tests frozen port constructors across contexts,
and the latest Qwen history experiment already establishes history-dependent
answers. The new comparison is receiver-aware sequential composition of the
two MDB-nominated reads, including another invocation after generated history.

## Frozen ports and fresh cases

The public `mdb.read_program.FrozenReadProgram` declares two physical sites:

- A: layer 19, KV head 0, first consumed generated row of the preceding answer.
- B: layer 22, KV head 0, last designated copula token within the current
  follow-up span.

Layer/head choices come from prior discovery. Both row-role constructors are
supplied by this experiment; no semantic role is automatically inferred.
Positions are resolved from committed history and recorded segment boundaries.
Missing copulas cause explicit abstention. The source word locates a row and
does not identify the represented variable.

Four fixed fresh contexts cross two lexical binding sets with whether the
previous question concerned the same or the other owner. The sets are
Nina/Samir with key/coin and Ada/Noah with book/cup. Each initial answer prefix
consumes two native-generated tokens. The first follow-up asks about the first
owner; a later follow-up asks about the other owner. No context is filtered by
native correctness. The full case plan is sealed with the source and request.

## Receiver-conditioned prediction

From one committed parent, observe native 00, A-only, and B-only worlds. Both
edits zero their selected historical value head while preserving all other
heads, keys, histories and pending input. Let the second layer's native
attention return in the A world be $o_A$. The conditional forecast is

\[
\widehat o_{AB}=o_A-\sum_{h:g(h)=0}
 p^A_{h,j_B}W_{O,h}v_{j_B}.
\]

Its stale-receiver comparator is

\[
\widehat o^{\mathrm{stale}}_{AB}=o_A-\sum_{h:g(h)=0}
 p^{00}_{h,j_B}W_{O,h}v_{j_B}.
\]

A changes the current query and current K/V append at layer 22. Its old
layer-22 history stays unchanged. Both probability calculations include the
current append and complete softmax denominator. Query/key equality is tested
between A and AB, where B changes only a past value, rather than between A and
00. Incoming layer-22 state equality and source-value equality are recorded.

The value subtraction is exact over real arithmetic. FP32 explanatory
contractions and native BF16 operation order can leave a numerical residual;
each complete forecast and consumed BF16 cast are saved before AB execution.

The receiving residual and SwiGLU are then recalculated, not added as
independent response directions. For predicted attention $\widehat o$ and
the actual A incoming state $x_A$,

\[
\widehat r=x_A+\operatorname{BF16}(\widehat o),\qquad
\widehat m=W_d\bigl[\operatorname{SiLU}(W_gN(\widehat r))
\odot W_uN(\widehat r)\bigr],\qquad
\widehat y=\widehat r+\widehat m.
\]

The worker explicitly applies native-width RMSNorm, gate/up/down maps, casts,
and residual additions. All dense MLP weights and work are paid. It saves the
actual interaction

\[
I_{22}=y_{AB}-y_A-y_B+y_{00}.
\]

Both forecasts are sealed before their replay or native AB reference is run.
An injected replay consumes the predicted attention return through the native
MLP and remaining model. The attention call still executes before the output
hook replaces its return, so injection demonstrates consumption, not omission
of dense attention.

## Recurrence and observables

The first comparison is a same-parent factorial. Each native 00/A/B/AB world
then consumes its own two generated tokens, appends the fixed second follow-up,
resolves its new preceding-answer row, and invokes the same intervention policy.
On the AB world's new parent, the complete factorial and prediction comparison
are repeated. Here 00 means no additional edit on an already altered history.
Cross-policy later outcomes may have different histories.

The first readout scores the current owner's correct object against the other
owner's object using a predeclared single-token contrast. Unrestricted greedy
answers and initial native correctness remain visible. Later token margins
are diagnostics of preference for those same tokens, not repeated answer
accuracy. The other-owner follow-up supplies a collateral binding observable.
No teacher-forced answer tokens enter the primary continuation.

## Execution and evidence

One frozen Qwen2.5-1.5B model runs BF16 native eager on CUDA under one strict
mrun lease, with no retries. The budget is at most 600 native token-step calls
plus four prefills. Prefill token count and 48 explicit dense forecast MLP
linear calls are reported separately. This is a receiver-assisted composition
test; economical query formation and full state-only prediction are untested.

Exact observer/sham and reinstall closure, head locality, source identity, and
the A/AB fixed-input contract protect instrument validity. Local errors,
semantic margins, interactions, wrong-row controls and continuation agreement
are retained as observations without an effect-size gate. Each completed
case and first bundle is saved before later work. Parameter recipes and a
standalone retriever retain source custody without copying pretrained weights.
Finite transport bounds are 262,144 files, 2 GiB expanded and 512 MiB compressed.

Code: [worker](worker.py), [launcher](launch.py),
[public resolver](../../src/mdb/read_program.py), and
[tiny mechanics](../../tests/test_read_composition_qualification.py).

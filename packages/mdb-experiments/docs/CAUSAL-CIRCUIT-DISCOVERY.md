# Discover causal memory-read circuits on evolving Qwen states

`bench.discover_circuit(spec)` connects native read observation, candidate
localization, head-local memory intervention, and native continuation in one
bounded transaction. Its output is a durable graph of receiver-specific reads
and measured intervention responses. It does not infer a complete semantic
chain of thought or replace dense model execution.

The initial implementation supports journal-mode Qwen2/Qwen2.5 Bench sessions
with `native-eager` attention. This preserves the ordered attention program
when a memory transaction changes physical ancestor/private segmentation.
Other model families retain their existing tracing and measurement interfaces.
Use mrun admission for pretrained-model experiments in the domains workspace.

## What the discovery pass computes

For a receiver position `i`, source row `j`, query head `h`, and its shared KV
head `g(h)`, the attention contribution has the mathematical form

\[
w_{i\leftarrow j,h}=a_{ij,h} W_{O,h}v_{j,g(h)}.
\]

Scoped hooks observe actual native `q_proj` and `o_proj` calls during token
stepping. The observer reconstructs RoPE queries, GQA grouping, and attention
over the complete visible history, including the current token. Candidate
values come from committed source rows. A declared search window limits the
candidate pool; it does not truncate the softmax normalizer or native context.

For each `(layer, KV head, source row)`, the nomination score is

\[
s_{\ell,g,j}=\sum_{h:g(h)=g}\|a_{ij,h}W_{O,h}v_{j,g}\|_2.
\]

The pass also retains the norm of the summed group write, individual reader
heads, attention weights, and exact source-value hashes. Native query/key hashes
bind each read to its evolving receiver. The same
stored payload can therefore be followed through different contextual reads.
Write projections use
FP32 contractions of native operands. Their sums are explanatory decompositions;
they do not promise the native low-precision reduction order. Reconstruction
error against the actual `o_proj` input is recorded per layer and backend.

Ranking nominates candidates. It is not evidence that a candidate matters to
the answer. Every selected candidate receives native continuation tests,
including small or zero measured effects. Search bounds and missed alternatives
remain explicit.

## Use the Python API

Start with an existing resident, frozen Qwen model and tokenizer:

```python
from mdb.bench import BenchSession

bench = BenchSession.from_resident(
    model, tokenizer, root="./circuit-session",
    attention_backend="native-eager", continuation_capacity=32,
)
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
bench.prefill(prompt_ids[:-1])
follow_up = tokenizer.encode("\nWhy?", add_special_tokens=False)

result = bench.discover_circuit({
    "name": "prompt-and-follow-up",
    "plan": [
        {"kind": "input", "token_ids": prompt_ids[-1:]},
        {"kind": "generate", "count": 2},
        {"kind": "input", "token_ids": follow_up},
        {"kind": "generate", "count": 2},
    ],
    "receiver_offsets": [0, 2 + len(follow_up)],
    "candidates_per_receiver": 2,
    "scales": [0.0, 0.5],
    "continuation_tokens": 1 + 2 + len(follow_up) + 2,
    "max_native_operations": 10000,
    "max_capture_bytes": 256 << 20,
})
report = result["report"]
```

The first receiver executes the last prompt token, so its output includes the
first answer decision. Prefilling the complete prompt would already have
selected that first answer token. Discovery observes newly executed reads; it
does not retrospectively explain the unrecorded prefill.

Plan offsets count token transitions, including fixed input tokens. The second
offset above executes the last follow-up input after two generated tokens, so
its readout includes the second answer decision. Input segments
are supplied by the caller. Every generated segment consumes the branch's own
greedy predictions, including predictions changed by an intervention. Future
baseline tokens are never supplied as branch inputs.

`continuation_tokens` caps each test starting with its receiver transition. A
short cap can end before a later follow-up; retain enough horizon for the
consumer you intend to measure. Generation executes a finite token count,
including any EOS tokens selected within that count; it does not silently adopt
a tokenizer-specific stopping policy.

## What is tested and retained

1. Run the declared plan while observing selected receiver occurrences.
2. Replay it without the observer and require exact native logits and complete
   endpoint-state parity. This checks observer mechanics.
3. Restore each receiver's committed parent. Scale only the nominated value
   head at its source row through the existing Model OS memory-write path.
   Seal a local attention-write prediction before executing the edited consumer:

   \[
   \widehat o'=o+(s-1)\sum_{h:g(h)=g}a_{ij,h}W_{O,h}v_{j,g}.
   \]

   Verify that its first query and complete key history remain exact, then
   retain the prediction's error against the native attention return. Rounding
   error is a diagnostic, not an effect-size gate. This forecasts the local
   write; later token outcomes remain measurements.
4. Execute each branch's own remaining plan. Retain full native logits through
   durable cuts, token decisions, step receipts, and per-offset logit changes.
5. Reinstall the original value row before the consumer executes and require
   exact reference logits and endpoint-state parity. This checks reversibility.
6. Restore the originally selected parent and preserve sibling selections.

The public hidden-memory selector still addresses complete rows. Discovery
constructs an exact full-row replacement while preserving all unselected heads.
Before/after payloads and untouched-head hashes make the narrower edit auditable.
The reinstall control reverses the edit before consumption; it is not a claim
that a downstream semantic computation was repaired.

An optional `observable` specifies a token margin:

```python
{"kind": "token_margin", "token_id": target, "against_token_id": alternative}
```

The report retains its signed change at every subsequent readout. With no
observable, it still reports native token/logit effects and leaves the task
margin unknown. Numerical differences after generated histories diverge are
marked as comparisons of different histories, rather than local approximation
errors at an identical input.

Candidate write directions also receive an unembedding readout. These are
vocabulary candidates with no fitted lens and no inferred intention, operand
role, or semantic operation. The causal tests concern the edited value and its
native future; they do not certify the meaning of a decoded word.

## Graph and debugger integration

```text
discover-circuit --spec @investigation.json
inspect circuits.discovery
```

The remote Python interface is `debug_session.discover_circuit(spec)`. The
operation requires a paused session with no pending Act, just like the existing
memory probe. The report reference is stored in the native session journal;
export retains the underlying cuts, payloads, receipts, and instrument source.

Save the returned report as JSON to browse its token-time/layer-depth evidence
without loading a model or tensor framework:

```sh
python -m mdb.circuit_discovery_view report.json --output circuit.html
```

This reuses the existing temporal causal viewer. Unknown task margins remain
unknown; geometric changes and vocabulary candidates stay visibly distinct
from qualified signed consumer effects.

The graph keeps these relationships distinct:

- `observed-native-read`: a measured read contribution at an exact receiver.
- `measured-intervention-response`: an edited carrier and its measured native
  future, including zero effects and each declared dose.
- `native-execution-order`: two receiver occurrences in the original trajectory.
- `same-payload-observation`: identical addressed value bytes seen at later
  receiver states; this is payload identity, not proof of a shared semantic role.

Observed reads alone do not certify causal dependence, and execution order does
not certify causal composition. Each carrier is
versioned by parent cut, layer, KV head, and source row. Matching layer/head
addresses across later states does not establish equal payloads or semantic
roles. Semantic CoT, unique/minimal circuits, cross-context generalization, and
economical emulation remain separate research questions.

## Freeze a candidate read program

The experimental `mdb.read_program.FrozenReadProgram` turns two previously
nominated sites into a frozen, digest-bound row-location rule. The standard
declaration is scoped to the Qwen2.5-1.5B composition assay: layer 19 / KV0 reads
the first generated row of the preceding answer prefix, and layer 22 / KV0 reads
the last designated copula within the current follow-up span.

```python
from mdb.read_program import FrozenReadProgram

program = FrozenReadProgram.standard()
a, b = program.resolve(
    committed_token_ids,
    preceding_answer_span=(answer_start, answer_stop),
    followup_span=(followup_start, followup_stop),
    copula_token_id=is_token_id,
)
if b is None:
    # The declared locator is inapplicable to this follow-up.
    ...
```

Both row-role constructors are supplied hypotheses. Prior discovery nominated
their physical sites; token words do not name the values' represented variables.
The resolver is framework-free, verifies its serialized declaration fingerprint,
and uses current committed history rather than fixed global row numbers.
It does not edit or run a model by itself. The
[prospective composition assay](../qualification/read_composition_20261107/PLAN.md)
tests these rules on fresh contexts and own-generated answer prefixes, while
keeping native receiver assistance and dense forecast MLP work explicit.
The [measured composition findings](../qualification/read_composition_20261107/FINDINGS.md)
report eight receivers: updating B's reader improves every local prediction,
while both forecasts match the same 24 native next-token decisions. Economical
formation of that changed reader remains open.
The [baseline-compiled weight producer](READ-WEIGHT-TRANSPORT.md) now predicts
that change without observing the edited receiver in a bounded eight-receiver
assay. Its [findings](../qualification/read_weight_transport_20261107/FINDINGS.md)
separate dense per-state compilation, scalar dose evaluation and the consumer's
still-paid native A output and incoming state.

## Bounds and validation

Plans contain at most 32 token transitions and eight inspected receiver
positions. Each receiver nominates at most four carriers. The default source
window is its latest 256 committed rows; explicit `row_start`/`row_stop` select
a smaller window. All native layers are searched by default. Operation and
capture bounds are checked before native work. Observer projections, vocabulary
readout, model weights, publication, and timing are accounted separately from
the native layer-token budget.

The tiny random BF16 Qwen tests verify native observation parity, head-local
edits, branch-owned token authority, fixed follow-up inputs, exact reinstall,
parent/sibling restoration, admission refusal, and hook-failure rollback. They
qualify mechanics; [qualification results](../qualification/CAUSAL-CIRCUIT-DISCOVERY-VALIDATION.md)
record pretrained-model findings separately with their
checkpoint, numerical lane, source hashes, raw rows, and custody.

Related interfaces: [hidden memory](HIDDEN-MEMORY.md),
[CoT register](COT-REGISTER.md),
[computation discovery](COMPUTATION-DISCOVERY-WORKFLOW.md), and
[original read-site experiments](../../../../saturn/experiments/2026-10-14-latent-bridge-two-hop/FINDINGS.md).

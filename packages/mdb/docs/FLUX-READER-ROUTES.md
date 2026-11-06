# A reader route is distributed, and its importance changes with context

The [kernel-qualified reader](FLUX-READER-ATTENTION.md) gives us an exact
consumer for captured Q/K/V: attention head → projection → gated residual →
normalization/modulation → MLP → next text state. We can now ask which parts
of its input carry a change without confusing that question with a numerical
reference that already misses the small effect.

Job `job-93d926083182` evaluated 116 new reader interventions in two known
contexts. It changed one head output at a time in both directions, and replaced
four bands of image-source K/V rows. The result is distributed and partly
transferable, with clear counterexamples to a fixed head-importance ranking.

## The two different causal questions

Let `F_c(H)` be the native text tail with receiving text/modulation fixed by
context c. Let `H0` and `H1` be native FlashAttention outputs for baseline and
changed image K/V, with text Q and text K/V held fixed. `P_i` selects the 128
channels of head i across all 512 text rows.

```text
complete change: Delta = F_c(H1) - F_c(H0)
head addition:   A_i = F_c(H0 + P_i(H1 - H0)) - F_c(H0)
head removal:    R_i = F_c(H1) - F_c(H1 - P_i(H1 - H0))
```

Addition asks what a head carries when changed alone. Removal asks what is lost
when that head is restored inside the complete change. Neither single-head
test establishes a minimal coalition or a universally necessary mechanism.

The reported signed alignment is `dot(A_i, Delta) / dot(Delta, Delta)`, or the
corresponding expression for `R_i`. This is projection onto the complete
context-specific change, not variance explained or a semantic success rate.
Orthogonal effects, magnitude, cosine, changed coordinates and reconstruction
error remain in the raw measurements.

The same add/remove construction is applied to four contiguous 256-row bands
within the 1024 image K/V rows. With the 512 text rows prepended, the bands are
`[512,768)`, `[768,1024)`, `[1024,1280)` and `[1280,1536)`. Both K and V change
inside each selected band. These are generic source addresses; no object
meaning is assigned to them. “Left” and “right” below are context IDs.

## What changed across the receiving contexts

| Question | Left context | Right context |
|---|---|---|
| Largest single-head addition alignment | Head 7: 26.82% | Head 14: 23.87% |
| Largest single-head removal alignment | Head 7: 28.36% | Head 21: 24.45% |
| Largest source-band addition alignment | Band 2: 48.76% | Band 2: 39.78% |
| Largest source-band removal alignment | Band 2: 44.29% | Band 1: 46.71% |

No single head reconstructs the complete text change. The strongest addition
has complete-effect error 94.79% on the left and 90.75% on the right. Head 7
remains useful on the right—14.38% addition alignment and 19.38% removal
alignment—but it is not the dominant component there.

The single-head addition alignments sum to 2.168 on the left and 1.978 on the
right; removal alignments sum to 2.333 and 2.312. These are not fractions that
should add to one. The counterfactuals overlap through a nonlinear, rounded
consumer. Adding their effects as independent pieces would be an incorrect
account of this computation.

The architecture and original matrices are shared. Captured attention payloads
and receiving states differ between contexts. This establishes context-dependent
importance; it does not yet isolate whether the ranking changes arise primarily
from Q/K/V payloads, incoming text/modulation, or their interaction. A crossed
payload/receiver experiment is needed for that attribution.

## Predictions were frozen before the second intervention panel

After the 58 left-context cases, MDB's `ForecastTransaction` stored the left
measurements and sealed predictions that the right context would preserve
the head and band rankings and single-head effect signs. Only then did the
58 right-context cases execute. The ledger's physical counters are unchanged
between left completion and the seal: 58 tails and 10 SDPA calls. The complete
frozen payload and signed outcomes remain available, including the misses.

| Frozen left → right prediction | Observed result |
|---|---|
| Same best head for addition | Miss: 7 → 14 |
| Same best head for removal | Miss: 7 → 21 |
| Same best source band for addition | Match: band 2 |
| Same best source band for removal | Miss: 2 → 1 |
| Same head-addition ranking | Spearman correlation 0.843; 2 of top 4 overlap |
| Same head-removal ranking | Spearman correlation 0.517; 1 of top 4 overlap |
| Same 24 single-head effect signs | 24/24 match |

Every measured addition alignment is positive in both contexts. Consequently a
constant-positive baseline also gets 24/24 signs. The sign agreement is an
observation, not strong standalone evidence of a discriminating prediction
rule. Band-rank correlations are 0.8 for addition and 0.6 for removal, across
only four bands and two contexts. There is no broad population claim.

The right intervention states had not executed before the seal in this
campaign, but both contexts and their baseline/full endpoints were already
known. This tests transfer to new interventions in known contexts, not fresh
seed or prompt generalization.

## MDB records the counterexamples too

CPU audit `job-9edd706026c1` verifies the original inventories, reconstructs all
116 input head recipes, verifies eight original head/text anchor archives,
checks the sealed transaction and recomputes every scalar comparison from
complete output tensors. It publishes 28 searchable route-rank records:
24 heads and four source bands. All matching-context queries resolve; all
changed-horizon queries abstain. Applicability conditions are supplied.

The records use generic component names such as `route.flux.reader.head7`,
qualified by model, parent state, step3/joint0, receiver and numerical contract.
They do not turn a physical head into a universal semantic symbol. A matching
lookup for the selected head returns its ranking miss as evidence.

From `artifacts/mdb-flux-reader-routes-20261105/`:

```bash
ROOT=compact-collection-01/extracted/binding-minimal-result
python -I -m mdb equations find --catalog "$ROOT/catalog/equations" \
  --applicable-to "$ROOT/catalog/context-selected.json"
python -I -m mdb equations find --catalog "$ROOT/catalog/equations" \
  --applicable-to "$ROOT/catalog/context-wrong-horizon.json"
```

The first returns `cmp.flux.reader-route.head7`; the second returns `[]`.
New artifacts retain standalone weight recipes/retrieval and an individual
evidence-file retriever. Head-mixture inputs are represented losslessly by
two source head archives, selected head IDs and an exact BF16 payload digest.
They need no redundant 3 MB input archive for every head intervention.

The native screen executed 20 FlashAttention calls, 348 linears, 116 LayerNorm
and 116 SwiGLU calls, with zero full model/block/image calls. Operator time
reached the last case at 21.537 seconds; runner time was 66.84 seconds, peak
RSS 1932.1 MB and peak VRAM 666 MB. This is not a comparative speedup claim.
The raw corpus has 286 files totaling 428,823,942 bytes, retained on Beast.
The collected compact archive is 183,414 bytes across 63 extracted files.

## Moving the record into a future

The [native continuation panel](FLUX-READER-ROUTE-FUTURE.md) now retains the
left-selected head 7 and band 2. It predicts all sixteen reader states exactly,
including four new dose states, before any full native block. Historical latent
and PNG controls replay exactly. Text-head edits first affect image at joint1;
source K/V edits affect image at joint0. Both persist to the final latent/PNG,
with small late-denoising effects and typed text/image disposal at commit.
Local rank, timed transport and final consumer effect remain separate fields
in the computation record.

# Fact editing: MEMIT, AlphaEdit and MLP memory overlays

MDB edits factual associations with two published methods: **MEMIT** (Meng et al., ICLR 2023)
and **AlphaEdit** (Fang et al., ICLR 2025). Both are applied as removable overlays on the
MLP `down_proj` modules: the model's parameters are never written, and removing the overlay
restores the native model bit for bit. Overlays can be used directly on a resident model or
installed into a `BenchSession`, where install and removal are recorded Acts.

`mdb.mlp_memory` provides the lower-level associative-memory overlay these methods grew out
of. It supports linear, Hebbian-quadratic and spectral-address keys, and is used for
mechanism experiments and relation-gated edits.

## Which method

| Need | Use |
|---|---|
| One batch of edits; best locality measured | `memit` |
| Several batches of edits applied one after another | `alphaedit` (measured better than MEMIT sequentially; worse in one batch at default threshold) |
| Keep the edited subject's *other* facts fixed (relation-specific edit) | `MLPMemory` with `fit_spectral_readers` keys (see below), or check explicitly |
| Causal experiments: install an edit at a cut, replay, compare futures | Any of the above via `bench.mlp_memory_install` |

## Quick start

```python
from mdb.memit import (AlphaEditHyperParams, MEMITHyperParams, alphaedit,
                       layer_second_moments, memit)

# 1. Key statistics C0 = E[k k^T] of each edit layer's down_proj input over generic text.
#    Compute once per model and cache.
C0, n_tokens = layer_second_moments(model, tok, wiki_paragraphs, [4, 5, 6, 7, 8],
                                    n_tokens=1_000_000)

requests = [{"id": "fr", "prompt": "The capital of {} is", "subject": "France",
             "target_new": " Tokyo"}]

patch, diag = memit(model, tok, requests, C0, MEMITHyperParams())
# or: patch, diag = alphaedit(model, tok, requests, C0, AlphaEditHyperParams())

with patch.install(model):           # forward hooks on down_proj; exit = exact undo
    ...                              # run anything

patch.save("edit.safetensors")       # content-addressed manifest + tensors
```

Sequential editing stacks batches. Pass the previous patch as `base`; the returned patch
contains every edit so far:

```python
p1, _ = alphaedit(model, tok, batch1, C0, hp)
p2, _ = alphaedit(model, tok, batch2, C0, hp, base=p1)    # p2 = batch1 + batch2
```

In a bench session, the patch is an MLP-memory Act:

```python
act = bench.mlp_memory_install(patch)   # lineage Act; later cuts record the patch digest
...
bench.mlp_memory_remove()               # witnesses bit-identical native probe logits
```

Cuts taken while an overlay is live record its digest. A cut can only be restored under the
overlay it was cut under. `BenchSession.resume` reinstalls a persisted overlay, and `close`
removes the hooks from the caller's model. History committed before an install keeps its
native K/V rows.

## The algorithms

Both methods share the reference `compute_z` and `compute_ks` steps
(github.com/kmeng01/memit, github.com/jianghoucheng/AlphaEdit).

1. **Target value.** For each request, optimize a residual delta added to the output of the
   last edit layer `L` at the subject's last token. The objective covers the bare prompt plus
   20 model-generated context prefixes:
   - the target's negative log-likelihood;
   - plus `kl_factor` × KL on `"{subject} is a"`, read at the subject token;
   - plus weight decay.

   The delta is norm-clamped to `clamp_norm_factor` × the original hidden norm, and the
   target is `z = h_L + delta`.
2. **Spread over layers.** For each edit layer `l`, lowest first:
   - the keys `K` are the `down_proj` input at the subject's last token, averaged over
     contexts, with the lower layers' updates already active;
   - the residual is `R = (z − h_L) / (layers remaining)`.
3. **Update.**
   - **MEMIT:** `ΔW = R Kᵀ (λ C0 + K Kᵀ)⁻¹`, with `λ = mom2_update_weight`.
   - **AlphaEdit:** `ΔW = Xᵀ`, with `X = solve(P (K Kᵀ + C_prev) + L2·I, P K Rᵀ)`, where:
     - `P = U₀U₀ᵀ` projects onto the singular directions of C0 below `nullspace_threshold`
       (directions preserved knowledge does not use);
     - `C_prev` accumulates the keys of earlier sequential batches.

     Updates therefore leave preserved keys approximately unchanged.

Defaults are EasyEdit's Qwen2.5 hyperparameters (`hparams/MEMIT/qwen2.5-7b.yaml`,
`hparams/AlphaEdit/qwen2.5-7b.yaml`):
- edit layers 4–8 (`down_proj`);
- 25 value steps at lr 0.5;
- weight decay 1e-3, clamp 4, KL 0.0625, λ = 15000;
- AlphaEdit only: `nullspace_threshold` 2e-2, `L2` 1.

Qwen2.5-1.5B and 7B both have 28 layers. Other model families need their own
hyperparameters.

### Deviations from the reference implementations

- The update is a forward hook on `down_proj` (`out + k @ ΔWᵀ`) instead of an in-place weight
  write. This is mathematically identical, and removal is exact.
- The AlphaEdit system `P (K Kᵀ + C_prev) + L2·I` is solved exactly by Woodbury
  (`alphaedit_solve`): it is `L2·I` plus a rank-n term, n = number of edit and earlier keys.
  This avoids the reference's dense d_ff × d_ff solve, which on Qwen2.5-1.5B took the job past
  15 GB of VRAM and got it killed. A unit test checks it against the dense solve to 1e-9.
- C0 is estimated from caller-supplied text with a token budget. The reference uses 100k
  Wikipedia records; the measurements below use 1M tokens of wikitext-103 train.
- `nullspace_threshold` is an absolute singular-value cutoff on C0, as in the reference.
  Its meaning depends on the model's activation scale, so check
  `null_space_projection(C0, thr)[1]` (the number of null dimensions) on a new model.

## Measured results

Qwen2.5-1.5B base, fp32, single seed. The edits shift each country's capital to the next
country's capital (12-country panel, and all 44 countries the model answers correctly).
Columns:
- **Efficacy:** the edit prompt "The capital of X is" now ends in the new capital.
- **Paraphrases:** T1 "X's capital city is", T2 "The capital city of X is", T3 "Q: What is the
  capital of X?".
- **Locality:** non-edited countries keep their capitals, across all four templates.
- **Same-subject specificity:** the edited country's language, flag, population, "People in X
  mostly speak" and "X's currency is the" answers are unchanged.

| Method | Edits | Efficacy | Paraphrases T1 / T2 / T3 | Locality | Same-subject specificity | Neutral KL | Job |
|---|---:|---:|---|---:|---|---:|---|
| MEMIT | 12 | 12/12 | 5 / 12 / 12 | 0.92 | 0.50–1.00 (mean 0.65) | 0.011 | job-2814d612dca2 |
| MEMIT | 44 | 44/44 | 25 / 44 / 44 | 0.94 | 0.36–1.00 | 0.037 | job-2814d612dca2 |
| AlphaEdit | 12 | 12/12 | 5 / 12 / 12 | 0.91 | 0.08–1.00 (mean 0.50) | 0.014 | job-0df4e1bc2ed1 |
| AlphaEdit | 44 | 44/44 | 30 / 44 / 44 | **0.62** | 0.16–1.00 (mean 0.42) | 0.021 | job-0df4e1bc2ed1 |
| MEMIT, sequential 4 × 11 | 44 | 39/44 | 20 / 38 / 36 | 0.75 | 0.34–1.00 (mean 0.53) | 0.056 | job-0df4e1bc2ed1 |
| AlphaEdit, sequential 4 × 11 | 44 | **44/44** | 26 / 44 / 44 | **0.88** | 0.25–1.00 (mean 0.49) | 0.036 | job-0df4e1bc2ed1 |
| Relation-gated `MLPMemory` (spectral-address keys, donor target) | 12 | 11/12 | 0 / 9 / 12 | 1.00 | 1.00 (all five) | 1.6e-8 | job-b70a76d5c0f5 |

Reading:

- **One batch: use MEMIT.** At the default `nullspace_threshold` 2e-2, batch AlphaEdit did not
  beat MEMIT on locality: 0.91 vs 0.92 at N = 12, and 0.62 vs 0.94 at N = 44. It also changed
  more of the edited subjects' other facts.
  - Its updates were 2–4× larger than MEMIT's in the lower edit layers (layer-4 ‖ΔW‖ 8.2 vs 2.0
    at N = 12).
  - The null space at this threshold holds 5,027–5,855 of 8,960 dimensions per layer, a
    threshold taken from the 7B configuration.
  - Untested hypothesis: the null space of generic-text keys does not exclude the
    same-relation keys of other countries, and low-energy directions are only weakly
    regularized (`L2` = 1). A threshold sweep would test it.
- **Sequential batches: use AlphaEdit.** This is the paper's main claim, and it reproduced.
  Over four batches of 11, AlphaEdit kept 44/44 edits with locality 0.88 and KL 0.036. MEMIT,
  which has no memory of earlier batches, lost 5 edits and fell to locality 0.75 and KL 0.056.
- **MEMIT and AlphaEdit generalize across paraphrases.** They are the only methods that move
  T1, where the subject is the first token.
- **Both change some of the edited subject's other facts.** About half of the language and
  currency answers moved at N = 12 with MEMIT. Their preservation term covers
  `"{subject} is a"` and the general key distribution, not the subject's other relations.
  Standard benchmarks (CounterFact neighborhood prompts) measure other subjects, not this.
- **Structural coupling.** "X's capital city is" and "X's currency is the" give the edit an
  identical key at the subject token. An edit that moves T1 also writes into the currency
  prompt.

## Relation-specific edits with `mlp_memory` (research API)

`mdb.mlp_memory.MLPMemory` adds `Δ(x) = φ(x)ᵀ B` to the MLP output at chosen layers.
`B` is a weighted ridge readout fitted sequentially per layer, with exact
Sherman–Morrison–Woodbury `insert`/`delete`. Key feature maps φ:

| Kind | φ | Use |
|---|---|---|
| linear (default) | normalized MLP input | subject-keyed associative memory |
| `QuadraticFeatures` | `(A x̂)⊙(G x̂)/√m`, random | Hebbian product construction |
| `fit_spectral_readers` → `AddressFeatures` | `(Σ p_s e^{iθ_s}) ⊙ (Σ p_r e^{iθ_r})` | conjunctive subject AND relation key |

`fit_spectral_readers` trains calibrated subject and relation posteriors: multinomial
logistic regression on z-scored keys, several wordings per relation, and a "none" class for
every other position. Each posterior is mapped onto spectral-address ABI phases
(`mdb.spectral_address`), and the two are bound by phase addition: the symbol phase plus the
role phase, as in `SpectralLogicalAddress`.

In real coordinates this is the gated bilinear (Hebbian) product. Phase carries the address,
and magnitude carries reader confidence, so the key abstains when either reader says "none".
With a donor target, this confines a subject swap to the chosen relation:
- 11/12 edits, locality 1.00, same-subject specificity 1.00 on lang, flag, pop, speak and currency;
- but weaker paraphrase generalization (21/24) and no T1.

`attribute_delta(..., also=paraphrases, keep=other-relation prompts)` optimizes
relation-specific values instead, the same idea as MEMIT's value step.

### Lessons (measured; details in the experiment record)

- **Store-layer keys at the subject token are about 97% additive** (subject + relation main
  effects; interaction 1–3% at layers 0–4).
- **Zero rows break a linear key.** A linear key cannot fire for (subject, relation) while
  staying silent for (subject, other relation) and (other subject, relation). That is an XOR:
  solvable only through the small interaction residue, at 4–10× gain, and held-out prompts
  collapse (locality 0.36). Do not add zero-target rows for the edited subject's other
  relations under a linear key.
- **A donor-transplant target is a subject swap.** It moves every attribute of the subject.
- **Readers must be trained on several wordings per relation.** One wording per relation
  confounds wording with relation.
- **Readers must keep magnitude.** Unit-modulus phases make the address fire on keys with no
  relation evidence.

Experiment record:
`saturn/experiments/2026-10-09-hebbian-store-mlp/` (`FINDINGS-*.md`, `PREREG-*.md`).

## Limits

- Qwen2-family decoders (`model.model.layers[i].mlp.down_proj`); other families need module
  mapping and their own hyperparameters.
- The bench path requires the BF16 bench ABI. Edits can be computed in fp32 and installed on
  any dtype.
- C0 estimation costs one forward pass over the stats corpus (about 3 minutes for 1M tokens
  on Qwen2.5-1.5B fp32 on a 4080). Cache it.
- Results above: one model, one relation family, single seed, at most 44 edits.

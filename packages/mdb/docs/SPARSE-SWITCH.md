# Sparse switch view

`mdb.sparse_switch` measures the per-position MLP gate *list* a model's output depends on, and scores a predictor's proposed list against it. It packages the measurement the 2026-11-03 forecast-cost Parts Z and AA use on Qwen2.5-1.5B open text.

## What it measures

A gated MLP writes `y = W_down (g * u)` with `g = act(W_gate h)` and `u = W_up h`. Against a reference gate `g_ref` (for example each token's mean gate over fit contexts), the switch at one position is `s = W_down((g - g_ref) * u)`. With a Fisher metric `F` at the MLP write, each neuron's diagonal Fisher score is

```
score_j = |(g_j - g_ref_j) * u_j| * sqrt(w_j^T F w_j),   w_j = W_down[:, j]
```

The oracle list is the top `m` neurons by this score. Keeping the true gate for those neurons and `g_ref` for the rest is the oracle sparse code. A predictor supplies its own scores, for example from a reconstructed residual pushed through the model's own gate. The view then reports, per position:

| metric | meaning |
|---|---|
| `recall` | share of the oracle top-m that the predicted top-m contains |
| `mass_recall` | oracle score mass inside the predicted list over the mass inside the oracle list |
| `flip_in` | share of oracle top-m neurons whose score error exceeds their margin above the (m+1)-th score |
| `flip_out` | share of oracle ranks m+1..2m whose score error exceeds their margin below the m-th score |
| `margin_ratio` | median score error over the oracle top-2m, divided by the score spread sc_(m/2) − sc_(2m) |

`fisher_energy` and `SparseSwitchView.energy_kept` give the Fisher energy a code keeps, `1 - Σ rᵀFr / Σ sᵀFs` with `r = s - ŝ`. That is Part Y's offline scoring rule. Part Z found it ranks codes correctly within one family, but not reliably across families.

## Use

```python
from mdb.sparse_switch import SparseSwitchView, fisher_energy, fisher_neuron_weights, switch_scores

w = fisher_neuron_weights(F, W_down)              # [I]
view = SparseSwitchView()
oracle = switch_scores(g, g_ref, u, w)            # [N, I]
proposed = switch_scores(g_hat, g_ref, u_hat, w)  # same shape
view.add_lists(layer, "predicted", 256, oracle, proposed)
view.add_switch_energy(layer, fisher_energy(s, F))
view.add_energy(layer, "predicted", fisher_energy(s - s_hat, F))
report = view.report()
```

The module needs only torch tensors, and torch is imported by the caller. Nothing in it fits parameters: every quantity is a closed-form function of the captured gates, the weights and the Fisher.

## Results (Part AA, job-e6996b016820)

Setup: Qwen2.5-1.5B, CUDA BF16 weights with an fp32 residual, WikiText, single seed. The view ran inside the job from a byte-identical copy of this module, sha256 5e00940b…. It computed every score, mask, list metric and Fisher energy below. Its neuron weights equal Part Z's formula exactly (max abs difference 0.0).

The predictor works in three steps:
1. reconstruct the gate input by closed-form ridge from the top 16 live head reads plus the four lower MLP writes;
2. push the reconstruction through the model's own RMSNorm, gate and up projections;
3. take the top 256 by the same score.

What it got:
- **List quality, held-out, mean over L10–27:** recall 0.48–0.50 against the oracle list, mass recall 0.72–0.73, margin ratio 0.66–0.76. The token-mean residual alone gives recall 0.14–0.17.
- **Block test with an exact memory**, as the share of the token-mean freeze's per-position KL gap closed:

  | Arm | All 28 layers | L10–18 |
  |---|---|---|
  | oracle list (Part Z) | 89% | — |
  | predicted list, true gate values | 57% | — |
  | predicted list, predicted values | 52% | — |
  | predicted gate at every neuron | — | 73% |
  | Part Z's linear coordinate ridge from the same kind of head reads | — | 21–24% |

- **The offline Fisher energy tracks the causal gain** across the predicted codes at L10–18.

Full tables: `FINDINGS-AA.md` in `saturn/experiments/2026-11-02-forecast-cost-and-compressed-addition`.

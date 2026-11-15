# Proof IDs

mdb tags ports, identities and numbers with proof IDs such as `{P05 P06}` (in `list`, `info
ports`, `backtrace`, `explain` and in the label of a `derived-exact` number: `[derived-exact
P19]`). An ID names the mathematical identity the value relies on: the operator a port computes,
or the decomposition an exact number is built from. The derivations are the developers' notes and
are not shipped with the package; this table is the key. An ID says which identity was used, not
that the number is correct: the authority label and the verdict say that.

| ID | Identity | Cited by mdb |
|---|---|---|
| P01 | Token lookup and complete forward expression DAG | yes |
| P02 | RMS denominator, full Jacobian, radial response | yes |
| P03 | Finite RMS change, including alignment and edit energy | yes |
| P04 | Affine Q/K/V, GQA, positional pair expansion | yes |
| P05 | Softmax contrasts and query invariance at fixed keys/visibility | yes |
| P06 | Full attention differential, centered values, source adjoint | yes |
| P07 | Finite score and allocation/payload/interaction splits | yes |
| P08 | Exact finite softmax reweighting and one-source closed form | yes |
| P09 | Zero-contribution cut versus mask-and-renormalize | yes |
| P10 | Full SwiGLU and state-conditioned operator factorization | yes |
| P11 | Rank-two quadratic form and its two eigenvalues | yes |
| P12 | Full nonlinear normalized-neuron read plane, gradient, Hessian | yes |
| P13 | Raw-residual Hessian: isotropic term plus rank-at-most-three correction | yes |
| P14 | Complete MLP Jacobian, residual skips, and consumer pullback | yes |
| P15 | Finite gate/up decomposition and source-component products | yes |
| P16 | SAE encoder/selection/decoder with centering, bias, and remainder |  |
| P17 | All feature/bias/remainder pair interactions and encoder writer couplings |  |
| P18 | Lossless successive-layer coordinates and prediction-error identity |  |
| P19 | Native readout, decision cells, competitor-margin bounds | yes |
| P20 | Recursive prediction error: propagated input error plus local mismatch | yes |
| P21 | Product convolution, derivative convolution, and normalized MLP operator |  |
| P22 | Two-axis attention transport, including changed route and payload |  |
| P23 | Finite nonwrapping causal cross-token transport cannot be Fourier diagonal |  |
| P24 | Shift commutator and nearest-circulant projection |  |
| P25 | Constant-signal/DC invariant and unequal source-use relation |  |
| P26 | RoPE score kernel's two-frequency transform and off-grid caveat | yes |
| P27 | Mixed gate/up derivatives, upstream terms, downstream curvature | yes |
| P28 | Path-gradient/Parseval completeness and mixed-Hessian rectangle | yes |
| P29 | Boolean Möbius inversion and Walsh–Hadamard representation | yes |
| P30 | Symbolic-state sufficiency, commuting transitions, semantic induction |  |
| P31 | Lipschitz composition and consumer readout bounds |  |
| P32 | Phase characters, Hebbian adjacency lookup, supplied BFS invariant |  |
| P33 | SPF affine exponent transport, scan/relative identities, period and phase-grid bound |  |

"Cited by mdb" marks the IDs that appear in the engine's source (`src/mdb`) and so in its output;
the others are listed so the numbering has no gaps.

The ones you meet most often:

- `P01` to `P15` label the port operators (see the Proofs column of the [Ports table](../README.md#ports)).
- `P14` (MLP Jacobian and residual skips) labels the linear term of `whatif` (`derived-local P14`).
- `P19` (native readout and margins) labels the exact split of a logit margin (`explain M[a,b]`,
  `reads`, `symbols`, `circuit`).
- `P20` (recursive prediction error) is the accounting behind every verdict: native change = exact
  change + rounding effect + residual (see [Verdicts](../README.md#verdicts)).

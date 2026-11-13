# Mathematics for native circuit inspection and composition

Status: self-contained mathematical reference for [future capabilities](NATIVE-CIRCUIT-CAPABILITIES.md), November 12, 2026. Identities apply under the assumptions stated here. The [workflow](NATIVE-CIRCUIT-WORKFLOW.md) is proposed; mathematical validity does not qualify a new native execution lane.

## Read planes and consumer-linked attention

For a SwiGLU neuron at normalized input `n`, write `a = gᵀn`, `b = uᵀn`, `φ = SiLU` and `f = φ(a)b`. Fixed affine biases can be included in `a,b` when present. Then

```text
∇f = b φ′(a) g + φ(a) u
∇²f = b φ″(a) ggᵀ + φ′(a)(guᵀ + ugᵀ).
```

Every dependence is through the span of `g,u`; the Hessian rank is at most two. If `R = [gᵀ; uᵀ]`, `R δn = 0` preserves the ideal neuron exactly. A requested coordinate change can be achieved by `R⁺δc` only if `δc ∈ image(R)`. This changes other neurons unless additional constraints preserve them. Raw residual edits also pass through normalization.

For `n = Γr / sqrt(ε + ||r||²/d)` with invertible gain `Γ` and `ε > 0`, feasible real normalized states satisfy `||Γ⁻¹n||² < d`. Their raw inverse is

```text
v = Γ⁻¹n
r = sqrt(ε / (1 − ||v||²/d)) v.
```

Substitution proves both the condition and the inverse. At `ε = 0`, feasible states lie on the corresponding sphere and raw scale is undetermined. Zero gains and BF16 reachability need separate treatment. No ideal nullspace theorem implies native bitwise or task invariance automatically.

For attention with fixed visibility, zero dropout, `p = softmax(l)` and `A = Σ_j p_j v_j`, differentiation gives

```text
δA = Σ_j p_j δv_j + Σ_j p_j(v_j − A) δl_j.
```

Proof: `δp_j = p_j(δl_j − Σ_k p_k δl_k)`; substitute into `δA = Σ_j(δp_j v_j + p_j δv_j)` and collect the mean. If `a = ∇_A M` includes the actual output projection and all downstream computation,

```text
∂M/∂l_j = p_j ⟨a, v_j − A⟩.
```

Identical values make routing changes ineffective. Large attention weights need not give large consumer effects. With fixed `P`, the source-value adjoint is `Pᵀ∇_A M`; an edit at source `s` in direction `d` has local effect `⟨d,(Pᵀ∇_A M)_s⟩`. Shared-KV adjoints sum over their query heads. Upstream edits additionally require query/key, value, normalization and skip paths.

For native neuron write `d_i f_i`, the product-coordinate derivative is `d_iᵀλ`, where `λ` is the suffix adjoint at the block output. This is contextual sensitivity, not a fixed importance label.

## Exact finite accounting

For two executions with `s_i = SiLU(g_i)` and ideal products `f_i = s_i u_i`, expanding the product proves

```text
Δf = u₀ Δs + s₀ Δu + Δs Δu
   = ((u₀+u₁)/2) Δs + ((s₀+s₁)/2) Δu.
```

The second equality demonstrates that branch attribution is convention-dependent even when the total is exact. `Δs` already includes gate nonlinearity; the explicit product interaction is not the entire nonlinear effect. If native `f_i = s_i u_i + e_i`, retain `Δe`.

Likewise, for `A_i = P_i V_i + e_i`,

```text
ΔA = ΔP V₀ + P₀ ΔV + ΔP ΔV + Δe.
```

For RMS normalization `n_i = Γρ_i r_i`, with `q_i = ε + ||r_i||²/d` and `ρ_i = q_i^(−1/2)`,

```text
Δn = Γ[ρ₁ Δr + (ρ₁−ρ₀) r₀]
ρ₁−ρ₀ = −(q₁−q₀) / [sqrt(q₀)sqrt(q₁)(sqrt(q₀)+sqrt(q₁))]
q₁−q₀ = [2r₀ᵀΔr + ||Δr||²]/d.
```

The scale identity follows by rationalizing the square-root difference. It exposes whole-residual coupling and finite edit energy rather than freezing the denominator.

For the unnormalized token DFT of length `N`, pointwise multiplication becomes circular convolution divided by `N`. Thus the finite SwiGLU identity is also an exact sum of three frequency convolutions. Parseval gives `δM = Re Σ_k ĝ_k* δx̂_k / N`, with channel contraction. Phases matter relative to other terms and the consumer. A unitary DFT removes this Parseval factor.

Frequency mixing alone does not establish semantic circuitry. A finite ordinarily causal operator with nonzero cross-token blocks cannot be block-circulant: causality zeros the first row's off-diagonal blocks, and circular translation would then zero every cross-token block. Token-varying local gains can also mix frequencies without cross-token transport. [Fourier diagonalization of circulants](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-31-eigenvectors-of-circulant-matrices-fourier-matrix/index.html) supplies the underlying standard operator relationship.

## Finite interactions and mixed derivatives

Let `M` be C² throughout `x + s a + t b`, for fixed additive directions `a,b` and `s,t ∈ [0,1]`. Then

```text
I_ab = M(x+a+b) − M(x+a) − M(x+b) + M(x)
     = ∫₀¹∫₀¹ aᵀ H_M(x+s a+t b) b ds dt.
```

Proof: integrate `∂²M(x+s a+t b)/∂s∂t` first over `t`, then over `s`. The four boundary values are exactly the finite difference. This is the bridge between a two-edit Möbius interaction and a mixed derivative. Same-parent nonadditive or noncommuting Acts do not automatically satisfy the assumptions. Native rounded contrasts remain valid observations, with a separate smooth-reference discrepancy.

For affine gate/up dependence on independent amplitudes `a,b`,

```text
f_ab = u φ″(g) g_a g_b + φ′(g)(g_a u_b + g_b u_a).
```

If the upstream inputs curve, add `u φ′(g) g_ab + φ(g) u_ab`. For downstream `M = H(f)`, also retain `M_ab = ⟨∇H,f_ab⟩ + f_aᵀ H_H f_b`. Product terms have the corresponding multiple-convolution representation; projecting a local MLP Hessian through a gradient omits downstream curvature. See [Integrated Hessians](https://jmlr.org/papers/v22/20-1223.html) for established interaction-attribution context.

For smooth fixed-prompt `V(λ) = V₀ + λd`, the fundamental theorem also yields `ΔM = ∫₀¹⟨∇M(V(λ)),d⟩dλ`. The changing derivative, rather than one baseline derivative, accounts for finite smooth change. Frequency allocation follows Parseval at each point and depends on path/baseline/basis. This is related to [Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html). Literal BF16 jumps and discrete token choices require separate native outcomes and closure.

## State sufficiency and composition

Let `α` describe a proposed symbolic interface, `T^ι` a native transition under intervention `ι`, and `ω` the corresponding abstract intervention. A deterministic high-level transition exists on the represented classes if and only if

```text
α(z) = α(z′) and ω(ι) = ω(ι′)
    imply α_next(T^ι(z)) = α_next(T^ι′(z′))
```

for all declared reachable compatible state/intervention pairs. Necessity follows because a function has one output for the same arguments. For sufficiency, define the output using any representative; the condition proves that it is well-defined. One neuron's read plane satisfies an immediate-output version of this condition, not generally a full-future version.

Suppose independently specified abstract steps `H_l` satisfy

```text
α_(l+1)(T_l^ι(z)) = H_l^ω(ι)(α_l(z)).
```

If initial states correspond, each selected intervention is compatible, covered states remain closed under the transitions, and the final readout has the specified meaning, induction gives trajectory refinement for every covered schedule. This is a bounded [causal-abstraction](https://jmlr.org/papers/v26/23-0058.html) claim. It does not imply a unique decomposition, bidirectional equivalence or arbitrary-language generalization. Abstract variables and operations must be grounded independently; copying the entire model into the abstract specification is vacuous.

With uniform local defects `ε_l` and Lipschitz bounds `L_l` on the compared abstract trajectories, `e_(l+1) ≤ L_l e_l + ε_l`. Iteration gives

```text
e_L ≤ (Π_j L_j)e₀ + Σ_i ε_i Π_(j>i)L_j.
```

Include readout error/amplification. If every output logit has error at most `E`, a reference winner's margin greater than `2E` against every competitor preserves argmax. A two-token margin certifies only that pair. Average empirical error does not provide these uniform guarantees. Own-token continuation needs the condition at every covered step and complete persistent state.

## Research sources

The equations above remain readable in an ordinary repository clone. The following original write-ups and raw evidence are workspace-local references; they are not runtime dependencies or claims that the notebook has been publicly published.

- [Synthesis and proposed semantic proof stack](/Users/jakeholl/domains/obsidian/blog/2026-11-12-172148-composing-native-mathematics-into-semantic-circuit-proofs.md).
- [Native read planes, RMS geometry and RoPE](/Users/jakeholl/domains/obsidian/blog/2026-11-12-161144-causal-attention-and-native-mlp-read-geometry.md).
- [Finite Qwen3 accounting and selected native hydration](/Users/jakeholl/domains/obsidian/blog/2026-11-12-161145-native-frequency-findings-and-content-dependent-circuits.md), [machine-readable audit](/Users/jakeholl/domains/artifacts/content-frequency-circuits-20261112/audit.json).
- [Consumer-connected Qwen2.5 experiment](/Users/jakeholl/domains/obsidian/blog/2026-11-12-155423-frequency-pairs-and-the-circuits-that-consume-them.md), [results](/Users/jakeholl/domains/artifacts/connected-frequency-consumer-20261112/results.json).
- [Systems/signals proofs and toy mechanics](/Users/jakeholl/domains/obsidian/blog/2026-11-12-165523-when-circuits-mix-frequencies-and-how-finite-edits-propagate.md), [Astra audit](/Users/jakeholl/domains/artifacts/systems-signals-proofs-20261112/astra-review.md).
- [Expanded native equations and intervention lattices](/Users/jakeholl/domains/obsidian/blog/2026-11-12-152729-expanding-native-computation-without-dropping-variables.md).
- [Earlier MLP-law composition and supplied graph algorithms](/Users/jakeholl/domains/obsidian/blog/2026-10-16-121731-mlp-laws-from-phase-circuits-to-graph-traversal.md), [proof audits](/Users/jakeholl/domains/model-experiments/experiments/2026-10-16-mlp-laws/audits/README.md).

The original experiments are separately scoped: Qwen3 block-local computation and hydration; Qwen2.5 full-prefill value edits and native output; deterministic synthetic checks; and supplied algorithm constructions. Their union is not a measured unified semantic circuit. Some independently checked statements overlap; repeated derivations do not create independent semantic evidence.

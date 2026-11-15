# Approximate FLUX velocity from a compact state

Follow-up: [read closure](FLUX-VELOCITY-READ-CLOSURE.md) separates own-state error from drift, measures a larger fresh two-pass quality gain, and supplies exact conditioning/local-reader counterexamples. The remaining structural target is a cheap nonlocal read from the candidate's own state.

The current bounded result omits one or two complete final denoiser passes after a charged native prefix. A pure NumPy actor receives current packed latent Z, previous velocity N and the four-step clock. Cached N produces a useful coarse update; small frozen channel/stencil maps with a feedback-selected scalar improve fresh-seed fidelity slightly. The actor receives no current/future native velocity or original denoiser parameters. Native prefix and VAE execution remain.

For the tested FLUX.2 Klein4B distilled CUDA BF16/512px profile:

```
R_k = (Z_k, N_(k-1), k)
N_hat_k = BF16(N_(k-1) + gain * correction_k(Z_k, N_(k-1)))
Z_hat_(k+1) = BF16(Z_k + BF16(BF16(sigma_(k+1)-sigma_k) * N_hat_k))
R_hat_(k+1) = (Z_hat_(k+1), N_hat_k, k+1)
```

The second omitted pass consumes its own predicted state. Gain0 exactly reinstalls cached velocity. Clock and rank-one state extrapolations worsen fidelity. Full-gain learned maps fit discovery examples but do not reliably generalize; damping to1/8 using old native-VAE feedback improves both image and velocity errors on two further fresh seeds.

Final test, two seeds × two related eye prompts: one-pass mean RGB MAE6.00871 cached,5.92002 damped stencil,57.42941 zero. Two-pass means14.49905,14.22590,68.78941. One-pass damped velocity relative L2 remains0.28421, so this is approximate continuation, not exact emulation. All four positive-dose rows improve and all four reversed one-pass rows worsen; seed/prompt breadth remains narrow.

Serving numeric inputs cost about1MiB FP32; linear/stencil banks267,264/533,504bytes. Original weights remain resident in the prefix worker. The learned program still performs its own small forward computation. Its millisecond arithmetic avoids the original denoiser at this suffix boundary; end-to-end acceleration has not been benchmarked.

The phase-aware runtime remains useful: `project → joint → single → readout → commit` identifies where a full native pass ends, which prior velocity is legal, and which exact finite arithmetic commits the approximation. It must label approximate writes and retain clock/numerical/domain applicability. Compact state does not include independently writable conditioning, so exact closure over arbitrary context changes is outside this interface.

Evidence: jobs `job-6d06f87a3f1a`, `job-b31068b78b60`, `job-2cd63c445200`. Across the three jobs: one resident load per job,448 native transitions per job, six sealed prediction cohorts and24 isolated actors. All corresponding native target passes occur after forecasts and VAE images seal. Local saved-state replay reproduces the final assay's complete predicted arrays exactly. No general claim follows from transaction integrity alone.

Workspace reports: [memory](../../../../artifacts/flux-velocity-memory-20261106/RESULTS.md), [student](../../../../artifacts/flux-velocity-student-20261106/RESULTS.md), [dose and audit](../../../../artifacts/flux-velocity-dose-20261106/RESULTS.md). These artifact links require the domains workspace. The next scientific distinction is local predictor error at its own state versus divergence from the canonical native trajectory, followed by richer functions versus added context operands. Existing [full-state closure](FLUX-STATE-CLOSURE.md) remains a separate result.

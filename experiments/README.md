# Reproducible experiments

These recipes run with the public package and public framework dependencies. Model
weights are acquired separately. No private checkout, fleet service, or job scheduler is required.
Generated states and results belong in `outputs/`, which is ignored by Git.

| Experiment | Process | Consumer |
|---|---|---|
| [FLUX.2 writer](flux2_writer/README.md) | Capture paired states, fit a writer, evaluate futures, restore, replay | Native rendered RGB |
| [AR family validation](family_validation/README.md) | Step each decoder family and Mamba-1 on real checkpoints; greedy decode and fresh-process replay vs native | Native logits and 16-token greedy decode |
| [LM block residency](lm_block_residency/README.md) | Step a decoder LM with frozen weights streamed one block at a time; streamed==resident bitwise where it fits, else vs an accelerate CPU-offload reference; fresh-process replay | Native 16-token greedy decode and final logits |
| [Diffusion family validation](diffusion_family_validation/README.md) | Step FLUX.2 Klein-4B/9B and FLUX.1-schnell resident or block-streamed; replay a mid-trajectory cut in a fresh process | Native diffusers per-step latents and VAE decode |
| [Debugging repair](debugging_repair/README.md) | Follow a deletion, native repair, repair suppression, and collateral; replay suffixes in fresh processes | Supplied neural circuit's threshold consumer |
| [Induction certificate](induction_certificate/README.md) | Preregister a full direct-source parent + control arms + sealed panels; run the relational-induction causal battery on two cached decoders and sign (or fail) the certificate | Native final norm + lm head on every branch |
| [Metadata-delta route](metadata_delta_route/README.md) | Save a measured delta as metadata, select it before hydrating, apply to a held-out parent, replay vs native, roll back exactly | Native carrier continuation and exact-rollback receipt |
| [circuit-tracer native edges](circuit_tracer/README.md) | Preregister a 56-prompt / 7-family panel; build circuit-tracer attribution graphs on Gemma-2-2B with Gemma Scope transcoders; re-run each top-20 feature edge and the whole group (zero-ablate, −2× steer) as native interventions on the real weights; compare to the graph's own predicted drop under three prediction modes; seal offline-re-derivable bundles | Native final norm + lm head (target log-prob and top-1) |

Follow-up measurements and the research papers that use these recipes are collected in
[related experiments](../docs/related-experiments.md).

Each recipe declares supplied inputs, fitted quantities, observations, exact mechanics,
and the scope of its claims. Reference results illustrate one measured development
context; they are not an expected universal outcome or a gate for your experiments.

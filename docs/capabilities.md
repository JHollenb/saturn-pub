# Capability and numerical boundaries

| Surface | V1 support | Boundary |
| --- | --- | --- |
| Core | JSON/tensor state, isolated branches, Acts, commit/restore, receipts | Trusted in-process runtime; no OS sandbox |
| Qwen | Qwen2/Qwen2.5, eager full attention, batch one, greedy token/layer cuts | Configured context limit; no padding, beam search, sliding windows |
| Decoder | gpt2, gpt_neox (Pythia), phi, llama, mistral, mixtral, gemma-1, qwen2, qwen3 native layer/operation cuts, eager attention, batch one, greedy | Configured context; no padding/beam; Mistral only while context <= sliding_window; Mixtral MoE bounded parity; unregistered/out-of-rule config refused |
| Mamba | Mamba-1 selective scan, per-layer causal-conv and recurrent-state cuts, batch one, greedy | No attention or positions; numeric (not bit-exact) cached-step vs full-scan parity; Mamba-2/SSM variants refused |
| Diffusion | DDIM eta-zero, native UNet steps, optional Stable Diffusion VAE | No CFG, multistep solvers, or SDXL |
| FLUX.2 Klein | Distilled 4B/9B native joint/single block cuts, deterministic flow Euler, native VAE | Batch one, text-only, guidance one; no reference KV, CFG, LoRA, or accelerate offload hooks during block execution; 9B-KV refused |
| FLUX.1 | schnell/dev native joint/single block cuts, T5+CLIP conditioner, flow Euler, native VAE | Batch one, text-only; dev guidance embed tested on tiny fixture only (no real dev checkpoint); no reference images, CFG, or LoRA |
| Block-streamed residency | Frozen host weights, one native block (FLUX transformer block, or decoder/Qwen/Mamba embedding, layer, norm, lm_head) copied to device per step; declared in the execution contract | Streamed output equals resident bitwise on the same device; the key/value (Mamba conv/recurrent) cache stays resident on the device, only frozen weights stream; host weights never move; pinning optional and off by default |
| Investigation | Same-parent panels, dose grids, custom control arms | Per-arm errors retained; no automatic semantic certificate |
| Program | Ordered measured pre-continuation Acts, exact parent/context binding | In-process; no learned/general compiler or cross-model portability |
| Training | Registered PyTorch mutable closure and isolated evaluation | Accepted snapshot in memory; rejected evidence retained |
| LocalStore | Verified JSON/safetensors execution cuts and immutable receipts | Local filesystem; no object store or remote transaction |
| DependencyGraph | Declared DAG dirty closure and exact local memoization | Caller owns dependency completeness and implementation versioning |
| Shared memory | One real Qwen ancestor plus private tails, portable segmented attention | Experimental; full attention work, no production speedup claim |
| PhaseLaw | Exact modular 32-bit transport and local-frame score reference | Quantized frequency law; floating decoded rotations |

CPU mechanics and native parity are tested. Other devices may run but need their own numerical
validation. The tests use tiny randomly initialized models; pretrained recipes are not a tested
semantic capability battery. Cross-kernel, cross-dtype, cross-version, and cross-device byte parity
is not assumed. Separate branches may choose different tokens near a close logit boundary.

Qwen's decomposed lane is checked against native logits within `rtol=1e-5, atol=1e-7` on tiny
FP32 CPU specimens. The generic decoder families and Mamba are checked against native logits
within `rtol=1e-4, atol=1e-5` on tiny FP32 CPU specimens (measured max absolute logit delta on the
tiny fixtures is a few ulps for the decoder families and ~2e-7 for Mamba); the Mamba single-token
zero-state step reproduces the native prefill bit-for-bit. Greedy layer/operation stepping matches
native `generate` token-for-token on the fixtures. Real-checkpoint numbers with job ids are in the
README. Same-program replay uses exact payload comparisons. DDIM deterministic
trajectory replay is checked for tensor equality on the same CPU program. Phase transport is
integer exact within the frequency quantization law; trigonometric decoding gets float tolerances.
Memory uses a different global-softmax reduction order, so native logits are compared numerically.
Memory accounting covers the supplied branch family; retained branches outside that list and
temporary numerical buffers require separate accounting. Empty private tails allocate lazily.

No claims of unlimited dense context, universal tensor semantics, semantic circuit portability,
general intelligence improvement, or serving acceleration are made. A consumer score is an
observation; a repeated direction is a trend; terminal certification requires a separate named
experiment with calibrated controls and held-out evidence.

The author's earlier private investigations into consumer-relative paths, causal panels, measured
programs, runtime state, partial evaluation, and memory inspired these interfaces. Their historical results are not bundled as a capability guarantee for this toolkit.

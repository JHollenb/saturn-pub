# Runtime and memory: public scope

The public repo contains small executable foundations from these investigations.
It does not contain the entire private workbench. Names below describe concrete
mechanisms, not guarantees about all historical prototypes.

The [usage guide](usage.md) explains which mechanics are automatic, which APIs must
be chosen explicitly, and provides runnable examples for each capability.

| Area | Included and runnable | Remaining outside this release |
|---|---|---|
| Runtime execution | Adapter transitions, typed state addresses, declared Acts, model/execution identity checks | Arbitrary model lowering and a production permission service |
| Change authority | Isolated preview, parent-bound commit, abort, exact restore, immutable receipts | Remote admission and deployment services |
| Branching building blocks | Nested forks, replay, retained cuts, named branch heads, durable local ancestry; [usage](usage.md#alternative-model-continuations) | Automated branch-search orchestration |
| Measured programs | `Program`: measured ordered Acts bound to an exact recipient parent and context | Learned address registry, general temporal compiler, measured-program atlases |
| Partial evaluation | `DependencyGraph`: declared dependencies, versioned implementations, dirty closure | Full private virtual machine and arbitrary program execution |
| Shared KV memory | `SharedQwenMemory`: pointer-shared live ancestor and private tails; `SharedQwenAdapter`: durable ancestor-plus-delta cuts under a separate segmented-attention ABI | CUDA virtual-address aliasing, eviction, page faults, allocator/scheduler integration |
| Phase transport | `PhaseLaw`: exact modular composition and local-frame scores; `PhaseOriginAdapter`: durable metadata-only origin-shift cuts with unchanged keys | Integration into the shared Qwen cache and a validated production phase ABI |
| Model debugger | Typed execution points/surfaces, boundary breakpoints, branch-local value watches, trace/diff/replay, intervention, commit/restore, save/load | Remote attach and Python-source breakpoints |
| Evidence map | Source/address/writer/carrier/consumer graph with separate divergence, repair, collateral and native-consumer rows | Automatic causal discovery or semantic verdicts |

Runtime authority lives in the Session; these modules do not introduce a second
neural executor. Read copies and declared write addresses make changes explicit.
Callable Acts and adapters remain trusted Python code, not an OS security sandbox.

Shared memory preserves the original causal ancestor and post-RoPE positions.
Both branches attend to that ancestor plus their own appended rows. Physical storage
sharing does not imply fewer attention rows, unlimited context, or faster serving.
The configured checkpoint position limit still applies. The segmented reference attention
can differ numerically from native attention, so the notebook reports the actual logit error.

The standard Qwen Session still does not turn its ordinary cache into shared memory.
`SharedQwenAdapter` is the explicit bridge: its cut stores the native ancestor and private
delta and replays them under its own segmented-attention numerical ABI. Session branches
clone those tensors in live RAM; they do not claim zero-copy residency. `LocalStore` may
deduplicate identical ancestor pages by content, which is a durable-storage property rather
than a live-memory property. Every attention step still consumes all ancestor and private
rows, so neither bridge claims reduced attention compute.

`PhaseOriginAdapter` similarly makes metadata-only phase movement capturable and restorable
while preserving local key bytes. Phase transport and memory sharing remain separate ABIs;
arbitrary independently compiled histories cannot be composed just by repositioning keys.

## Hands-on entry points

- [AR debugger notebook](../notebooks/01_debugger.ipynb): stop after a layer, inspect KV,
  fork native/candidate futures, intervene, commit, restore, and replay in a fresh process.
- [Diffusion debugger notebook](../notebooks/02_diffusion_debugger.ipynb): stop at a
  denoising step, branch a latent, compare numerical futures, and save/load a cut.
- [Runtime and memory notebook](../notebooks/03_runtime_memory.ipynb): shared ancestor
  pointers, private tails, native logit comparison, allocation accounting, and phase composition.
- [Full writer experiment](../experiments/flux2_writer/README.md): pretrained recipient
  images, construction provenance, rejected futures, exact rollback, and replay.

The tiny notebooks use random native models on CPU without downloads. They demonstrate
execution and memory mechanics. Their token IDs and tiny latents have no trained semantic
meaning. The writer experiment supplies the separate pretrained consumer demonstration.

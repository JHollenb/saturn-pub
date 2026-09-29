# Model OS and virtual memory: public scope

The public repo contains small executable foundations from these investigations.
It does not contain the entire private workbench. Names below describe concrete
mechanisms, not guarantees about all historical Saturn prototypes.

| Area | Included and runnable | Remaining outside this release |
|---|---|---|
| Model OS execution | Adapter transitions, typed state addresses, declared Acts, model/execution identity checks | Arbitrary model lowering and a production permission service |
| Change authority | Isolated preview, parent-bound commit, abort, exact restore, immutable receipts | Remote admission and deployment services |
| Branch history | Captured ancestry, retained futures, durable local cuts | Full Thott/Future Forest orchestration |
| Measured programs | `Program`: measured ordered Acts bound to an exact recipient parent and context | Learned Registrar, general temporal compiler, full Hyperion atlases |
| VM partial evaluation | `DependencyGraph`: declared dependencies, versioned implementations, dirty closure | Full private Saturn VM and arbitrary program execution |
| Shared KV memory | `SharedQwenMemory`: one actual ancestor, independent private tails, segmented global-softmax attention | CUDA virtual-address aliasing, eviction, page faults, allocator/scheduler integration |
| Phase transport | `PhaseLaw`: exact modular integer composition and local-frame reference scores | Integration into the shared Qwen cache and a validated production phase ABI |
| Model debugger | Boundary breakpoints, step/next, bounded continue, read, branch, intervention, commit/restore, save/load | Remote attach, Python-source breakpoints, value-change watchpoints |

Model OS authority lives in the Session; these modules do not introduce a second
neural executor. Read copies and declared write addresses make changes explicit.
Callable Acts and adapters remain trusted Python code, not an OS security sandbox.

Shared memory preserves the original causal ancestor and post-RoPE positions.
Both branches attend to that ancestor plus their own appended rows. Physical storage
sharing does not imply fewer attention rows, unlimited context, or faster serving.
The configured checkpoint position limit still applies. The portable attention reduction
can differ numerically from native attention, so the notebook reports the actual logit error.

The shared-memory object is a separate experimental consumer, not integrated into
the standard Qwen Session's durable cut format. Do not infer shared-cache durability
from the debugger's Qwen Session replay demonstration. Likewise, phase transport and
memory sharing are separate examples; arbitrary independently compiled histories cannot
be composed just by repositioning their keys.

## Hands-on entry points

- [AR debugger notebook](../notebooks/01_debugger.ipynb): stop after a layer, inspect KV,
  fork native/candidate futures, intervene, commit, restore, and replay in a fresh process.
- [Diffusion debugger notebook](../notebooks/02_diffusion_debugger.ipynb): stop at a
  denoising step, branch a latent, compare numerical futures, and save/load a cut.
- [Model OS and memory notebook](../notebooks/03_model_os_memory.ipynb): shared ancestor
  pointers, private tails, native logit comparison, allocation accounting, and phase composition.
- [Full writer experiment](../experiments/flux2_writer/README.md): pretrained recipient
  images, construction provenance, rejected futures, exact rollback, and replay.

The tiny notebooks use random native models on CPU without downloads. They demonstrate
execution and memory mechanics. Their token IDs and tiny latents have no trained semantic
meaning. The writer experiment supplies the separate pretrained consumer demonstration.

# Runnable examples

For the decision guide and automatic/explicit behavior, read [when and how to use each API](../docs/usage.md).
Run scripts from the checkout root. Results go to ignored `outputs/` directories.

| Script | When to run it | Install extra |
|---|---|---|
| `custom_adapter.py` | Add your own JSON-only execution machine | None |
| `ar_debug.py` | Capture, change a Qwen carrier, compare and rewind | `ar` |
| `debugger_replay.py` | Inspect native layers/KV and verify durable fresh-process replay | `ar` |
| `software_debugger.py` | Exercise typed microsteps, live watches, masked writes, replay bundle, and a causal map on a parameter-free graph | `torch` |
| `debugging_repair.py` | Resolve symbols, trace early deletion and later repair, suppress repair, measure collateral, and replay in fresh processes | `torch` |
| `explore_debugger.py` | Audit numerical versus token effects in tiny Qwen, edit timing in DDIM, repeated repair windows, and durable KV-history continuation | `ar,diffusion` |
| `diffusion_debug.py` | Compare branched DDIM latents | `diffusion` |
| `causal_panel.py` | Run a dose/control grid and compile one measured arm | `ar` |
| `induction_certificate.py` | Freeze a circuit policy + sealed panels, run the relational-induction causal battery on a tiny decoder, and read the signed gate verdict (random weights -> honest FAIL) | `ar` |
| `measured_program.py` | Replay an exact measured program and see unsupported-use refusal | `ar` |
| `dependency_reuse.py` | Inspect which declared cells are reused/recomputed | `ar` |
| `interop_saelens.py` | Find an SAE feature with TransformerLens/SAELens, ablate it on Saturn's aligned carrier, and replay the receipt | `interop` |
| `shared_memory.py` | Compare shared-prefix attention and separate phase mechanics | `ar` |
| `reversible_training.py` | Train a small writer with native feedback and reject/restore a harmful future | `ar,train` |
| `structural_rewind.py` | Reopen graph/rules/memory/cursor futures with an empty model and no optimizer | `train` |
| `flux2_build_writer.py` | Fit a writer from freshly paired pretrained block states | `diffusion` |
| `flux2_hotfix.py` | Evaluate a supplied writer through native images and replay | `diffusion,train` |

Example: `pip install -e '.[ar,train]'`, then `python examples/reversible_training.py`.
The tiny examples use random native CPU models; token IDs and tiny images have no learned
semantic meaning. The FLUX scripts require CUDA and checkpoints plus their declared CLI
arguments. Use the [one-command experiment](../experiments/flux2_writer/README.md) for
the complete pretrained writer construction, feedback, rollback, and replay workflow.

For cell-by-cell debugging, see the [notebooks](../notebooks/README.md).

# Hands-on notebooks

From the public checkout:

```bash
pip install -e '.[ar,diffusion,notebooks]'
jupyter lab notebooks/
```

Select the Python environment where you installed Saturn. All four notebooks run
on CPU without model downloads or a job scheduler; run their cells in order. Results are saved
under `outputs/notebooks/`. They use the exact command parser exposed by the terminal CLI.

| Notebook | What you can do |
|---|---|
| [01 — Model debugger](01_debugger.ipynb) | Set a layer breakpoint; continue/next; inspect tokens, hidden state and KV; fork; intervene; commit/rewind; verify fresh-process replay; run a CLI transcript |
| [02 — Diffusion debugger](02_diffusion_debugger.ipynb) | Stop at a denoising step; branch the latent; compare native numerical futures; rewind; save/load |
| [03 — Runtime and memory](03_runtime_memory.ipynb) | Inspect shared physical KV ancestry, private tails, native attention agreement, allocation accounting, and separate modular phase composition |
| [04 — Debugging repair](04_debugging_repair.ipynb) | Resolve supplied symbols, watch deletion and native repair, inspect writer provenance, compare recovery windows, suppress repair, and replay both suffixes in fresh processes |

For an interactive terminal, run `saturn-pub debug --family ar` or
`saturn-pub debug --family diffusion`. The notebooks call `Debugger.execute` so each
cell acts as an inspectable stop without blocking the notebook on terminal input.

These are mechanics demonstrations on random models or supplied circuits, not pretrained capability
results. See [public scope](../docs/runtime-memory-scope.md) and the
[pretrained writer experiment](../experiments/flux2_writer/README.md).

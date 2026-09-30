# Debugging repair

A complete controlled experiment using the public debugger, symbols, temporal traces,
cut storage, and the existing causal-path records. It requires only the `torch` extra.

```bash
pip install -e '.[torch]'
python examples/debugging_repair.py
```

Read `outputs/debugging-repair/walkthrough.md` and `report.json`. The output also contains
durable cuts and a sealed symbol sidecar. Two fresh processes reopen the early-deletion cut
and reproduce the repaired and suppressed-repair suffixes exactly.

The source, fixed linear weights, channel roles, and symbol locations are supplied. The
first carrier channel is the target and the second is collateral. The circuit has two native
residual writers that reconstruct the same source, followed by a threshold consumer.

| Arm | Target | Collateral | What it exposes |
| --- | ---: | ---: | --- |
| Native | 1 | 0 | Unedited continuation |
| No-op | 1 | 0 | Instrument control with no selected-port divergence |
| Early deletion | 1 | 0 | First-writer effect erased; second native writer repairs it |
| Delete both | 0 | 0 | Later repair result also erased |
| Late deletion | 0 | 0 | No later writer remains to repair the deletion |
| Late collateral | 1 | 1 | Target success does not imply collateral preservation |

All arms originate at one exact parent. The example asserts its parent stays unchanged,
links measurements to cuts and receipts, and retains complete intermediate event cuts.
It labels recovery of exact native carrier content separately from the final consumer output.

This is a no-download debugger mechanics demonstration, not a pretrained semantic result.
Use [the API guide](../../docs/debug-symbols-paths.md) to adapt the workflow to supported
native-model boundaries and evidence-backed local symbols.

## Audit and broader exploration

`python examples/explore_debugger.py` explores the same tools on tiny random native Qwen
and UNet/DDIM models, with `ar,diffusion` installed. It also tests repeated recovery windows
on a supplied integer organism. Results and source hashes go to
`outputs/debugger-exploration/report.json`; one Qwen candidate suffix reopens in a new process.

The audit found and corrected watch registration, event sealing/linkage, typed-change,
qualified-address, temporal-offset, and batched-provenance defects. Historical reproductions
remain in [audit-before-fixes.json](audit-before-fixes.json) and
[audit-extra-before-fixes.json](audit-extra-before-fixes.json). The regression suite records
corrected behavior separately. Macro-step provenance now expands recorded micro-receipts.

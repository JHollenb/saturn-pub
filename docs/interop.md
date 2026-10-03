# SAELens and TransformerLens interop

Saturn does not reimplement mechanistic-interpretability tooling. It works next to it.
TransformerLens and SAELens are good at exposing the residual stream and decoding it into
sparse features; Saturn is good at turning one intervention into a branchable, replayable
transaction on the native model. `examples/interop_saelens.py` shows the division of labor on
GPT-2 small, CPU, offline, from the local Hugging Face cache.

```bash
pip install -e '.[interop]'   # sae-lens==6.51.3, transformer-lens==3.9.0
python examples/interop_saelens.py
```

## Division of labor

1. **TransformerLens + SAELens (feature discovery).**
   `HookedTransformer.from_pretrained_no_processing("gpt2")` runs a prompt; the SAELens release
   `gpt2-small-res-jb` at `blocks.L.hook_resid_pre` decodes the last-position residual and names
   the top-active feature `f` with activation `a` and decoder direction `W_dec[f]`.
2. **Saturn (aligned intervention).**
   `saturn_pub.adapters.load("gpt2")` loads the same HF weights into the native decoder adapter.
   Step to the `layer:L` boundary and Saturn's residual carrier is the input to block `L`.

## The address alignment is measured, not assumed

Saturn's `layer:L` boundary carrier (last position) is TransformerLens `blocks.L.hook_resid_pre`
at the last position. The example verifies this numerically before intervening. Measured on GPT-2
small, layer 6, prompt "The Eiffel Tower is located in the city of":

- Carrier vs `blocks.6.hook_resid_pre[:, -1]`: max abs diff ~2e-5 (fp32, `no_processing`).

`from_pretrained_no_processing` is deliberate: no weight folding, so the residual stream stays
comparable to Saturn's raw native weights. The `gpt2-small-res-jb` SAE was trained on
`center_writing_weights=True` activations, so feature *magnitudes* here are approximate. That does
not touch the alignment, ablation-agreement, or replay claims, which use the same no-processing
residual, activation, and decoder direction on both sides.

## Ablate one feature, run the native suffix, ship a receipt

Subtract `a * W_dec[f]` from the carrier with an `Act`, then continue the native model:

```python
from saturn_pub import Act

act = Act.add("hidden", direction.reshape(1, 1, -1), dose=-activation)
candidate.apply(act)
adapter.generate(candidate, 8)
```

- Capture the parent `StateCut`; fork a native branch and an ablated candidate (optionally a dose
  grid via `Investigation`). Compare next-token logits/top-k and the generated continuation.
- **Cross-check:** a TransformerLens hook applying the same ablation agrees on next-token logits
  within ~2e-4 on GPT-2 small — the same magnitude as the intrinsic clean TransformerLens-vs-HF
  gap, because the two libraries use different block kernels. Bounded, not exact.
- **Custody:** `LocalStore` seals the parent and ablated cuts plus receipts. A fresh Python process
  reloads them and reproduces both continuations byte-for-byte. The claim ships with its receipt.

For the chosen headline, GPT-2 small predicts " London" after that prompt; removing feature 14344
flips the top token to " Paris" and the continuation from "London, and is the tallest building in"
to "Paris, France. It is the tallest". Prompts and layers tried are listed in the example and its
JSON report at `outputs/interop-saelens/report.json`.

## Fast test

`tests/test_interop.py` is skipped unless the `interop` extra is installed. It uses a tiny random
GPT-2 and a synthetic SAE-like direction (no download) to check the same contract against HF's own
kernels: Saturn's carrier equals `resid_pre[L]`, and a carrier ablation matches the identical
intervention applied through an HF forward hook.

## Attribution graphs

For circuit-tracer attribution graphs (Gemma Scope transcoders, Gemma-2-2B), see
[circuit-tracer interop](circuit-tracer.md): the same carrier-alignment discipline applied to
every graph edge and to whole top-k groups, with preregistered real-weight results.

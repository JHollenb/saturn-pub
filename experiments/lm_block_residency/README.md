# LM block-streamed residency on real checkpoints

`run.py` validates that a decoder LM adapter run in **block-streamed residency** -- frozen
weights parked in host memory, one native block (embedding, each decoder/mixer layer, final norm,
lm_head) copied to the execution device at a time -- reproduces the native decode, so a model
larger than device memory can be stepped and debugged layer by layer. The key/value (and Mamba
conv/recurrent) cache stays resident on the execution device; only frozen weights stream.

Each checkpoint declares a `compare` mode:

- **`compare: "resident"`** (the model fits the device). Build a resident adapter and a streamed
  adapter from the same checkpoint and assert, over an N-token greedy decode, that streamed is
  **bitwise identical** to resident: same generated token ids, bit-equal final logits, and a
  bit-equal `native_logits` comparator. Records peak VRAM and tokens/s for each mode and the VRAM
  reduction. This is the exactness contract: same dtype, same eager kernels, same device -- a
  byte-preserving `Tensor.to` and a non-mutating `functional_call` substitution leave nothing to
  diverge.
- **`compare: "reference"`** (the model does **not** fit the device). Build only the streamed
  adapter and reproduce an N-token greedy decode. The reference is HF Transformers
  `AutoModelForCausalLM.generate` with accelerate `device_map` CPU-offload (`max_memory` caps GPU
  use so the model genuinely offloads), eager attention, the same checkpoint and dtype, greedy
  (`do_sample=False, num_beams=1`); the check is **exact equality of the N generated token ids**.
  Records peak VRAM, tokens/s, and the resident weight footprint that exceeds the card.

Both modes also capture a mid-layer `StateCut` **under streaming**, save it to a `LocalStore`,
reload it in a brand-new interpreter, restore, continue, and check the continuation
token-for-token (**fresh-process replay**).

Settings: `tf32` disabled, batch one, greedy, eager attention, one checkpoint loaded at a time and
freed between entries (`del` + `gc` + `empty_cache`); the reference model is loaded and freed
before the streamed adapter so host memory holds one model at a time. Only `saturn_pub` and public
frameworks run; the job launch and collection scripts are not part of the public package.

## Reproduce with plain Python

```bash
pip install "saturn-pub[ar]" accelerate   # torch + transformers==5.14.1 + accelerate
python experiments/lm_block_residency/run.py \
    --checkpoints experiments/lm_block_residency/checkpoints.example.json \
    --device cuda --new-tokens 16 \
    --output outputs/lm-residency.json --workdir outputs/lm-residency-state
```

`checkpoints.example.json` entries are `{family, label, path, dtype, compare}`; `path` is any
Hugging Face id or local directory the frameworks can load offline. On a CPU-only host the
`reference` mode falls back to a plain full-model forward on the CPU (there is no accelerator to
offload to), which is still an independent comparator for the streamed decode.

`results/` holds the merged measured table and the job ids for one development context; it is not
an expected universal outcome or a gate for your experiments.

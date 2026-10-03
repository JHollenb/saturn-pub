# Induction-circuit certificate on real checkpoints

This recipe reproduces a bounded relational-induction source-circuit certificate on two
cached small decoders, using `saturn_pub` itself as the runtime. For each checkpoint it
wraps the native model with `saturn_pub.adapters.load`, runs the preregistered five-branch
causal battery over the **complete direct-source parent** (every attention layer and query
head) on two sealed arbitrary-token panels, and applies the frozen certificate gate from
`saturn_pub.certificate`. Every branch terminates at the model's own final norm and output
embedding (the native consumer).

## Reproduce with plain Python

```python
import torch
from transformers import AutoModelForCausalLM
from saturn_pub.adapters import load
from saturn_pub.certificate import InductionCircuitPolicy, InductionPanelSpec
from saturn_pub.certificate.panel import certify_decoder_induction

torch.backends.cuda.matmul.allow_tf32 = False
device = "cuda" if torch.cuda.is_available() else "cpu"
specs = [
    InductionPanelSpec(
        "confirmation-length12-seed20260812",
        seed=20260812,
        examples=192,
        length=12,
        classes=48,
        token_low=1000,
        token_high=20000,
    ),
    InductionPanelSpec(
        "confirmation-length16-seed20260813",
        seed=20260813,
        examples=192,
        length=16,
        classes=48,
        token_low=1000,
        token_high=20000,
    ),
]
for model_id in ("Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-360M"):
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="eager", dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    adapter = load(model)  # the native decoder adapter is the runtime
    bundle = certify_decoder_induction(adapter, specs, InductionCircuitPolicy(), batch_size=24)
    print(model_id, bundle["certificate"]["certified"], bundle["certificate"]["content_sha256"])
```

fp32, eager attention, one checkpoint resident at a time. The gate is stdlib-only; the
measurement helper needs `saturn-pub[ar]`. The private launch/collection scripts are not
part of this package.

## Measured result

- device: NVIDIA GeForce RTX 4080, fp32/cuda, eager attention, batch 24, tf32 off
- torch 2.13.0, transformers 5.14.1; 192 examples/panel, 48-class alphabet, two panels/model
- scheduler job: `job-ac4a913f9389` (full); `job-4807487ca252` (24-example smoke)
- policy: default `InductionCircuitPolicy` (seven per-panel gates, `min_panels=2`); the
  signed certificates and the full outcome metrics are in [`results/`](results/)

Both models certified (28/28 preregistered panel gates: 2 models x 2 panels x 7 gates):

| model | arch | panel | clean | source cut | match cut | repair | wrong donor | repair replay | certified |
|---|---|---|---:|---:|---:|---:|---:|:---:|:---:|
| Qwen2.5-0.5B | qwen2, 336 edges | length 12 | 0.9740 | 0.0000 | 0.9583 | 0.9740 | 0.0000 | exact | yes |
| Qwen2.5-0.5B | qwen2, 336 edges | length 16 | 0.9844 | 0.0052 | 0.9740 | 0.9844 | 0.0052 | exact | yes |
| SmolLM2-360M | llama, 480 edges | length 12 | 0.9635 | 0.0000 | 0.9219 | 0.9635 | 0.0000 | exact | yes |
| SmolLM2-360M | llama, 480 edges | length 16 | 0.9583 | 0.0000 | 0.9688 | 0.9583 | 0.0000 | exact | yes |

"repair replay = exact" means the full-parent repair reproduced the clean per-example
decision and candidate margin bit-for-bit (`repair_correctness_mismatch_fraction = 0`,
`repair_margin_mae = 0`) on every panel. The source cut collapses to chance (Wilson upper
bound at or below chance + 0.03), the neighbor-cue cut barely dents accuracy, and the
different-answer donor stays at chance.

Certificate content hashes:

- Qwen2.5-0.5B: `84dc2a27680bcf048f849ab66966c98c2a38663a70a61d8e6b55c7f0698cfc3d`
- SmolLM2-360M: `983b00ce660a0ddadba9a4997471a7b1812b3e6043f7b37bcb440881f85d105b`

## Claim boundary

The certificate is bounded to the recorded model bytes, the declared full direct-source
route family, the two sealed panels, the native lexical consumer, and the frozen
thresholds. The certified parent is a distributed family of edges (336 for Qwen2.5-0.5B,
480 for SmolLM2-360M), **not** a compact or globally minimal circuit, and the assay is
controlled one-step retrieval, not unrestricted natural-language reasoning. The
compressed-envelope search is out of scope here. These reference numbers illustrate one
measured context; they are not a gate for your own experiments. A failing gate would be a
legitimate result, reported rather than tuned away.

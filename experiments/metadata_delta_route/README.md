# Metadata-delta route on real checkpoints

`run.py` saves a measured intervention delta as content-addressed metadata, selects it from the
metadata alone, hydrates it through a content-verified resolver, applies it to a held-out recipient
parent, replays against the native run, and rolls back exactly. One flow serves both adapters:

- `--adapter decoder` steps a native HF decoder to its first layer boundary and treats the `hidden`
  carrier as the delta site.
- `--adapter flux2` steps FLUX.2 Klein past its third joint block and treats the `text` carrier as
  the delta site (real weights are block-streamed so the transformer fits a 16 GB card; the prompt
  is encoded by the public pipeline's encoder with the transformer detached).

Per run it records:

- **selected / card_fingerprint / route_fingerprint** -- the metadata-only selection that found the
  delta without touching any bytes.
- **dedup_ok** -- hydrating the same card twice resolves the bytes once (`artifact_requests` 2,
  `artifact_hydrations` 1, `artifact_cache_hits` 1 in the route receipt).
- **artifact_ref** -- the delta's content address (sha256 + shape + dtype + bytes); the resolver's
  bytes are admitted only after their sha256 matches it.
- **route_receipt.transport.estimated_reduction** -- metadata-plus-selected-delta vs a dense catalog.
- **changed_vs_native** and **effect_carrier_l2** -- the applied delta changes the state, and the
  carrier diverges from the native continuation by a measured L2.
- **rollback.verified_exact / matches_parent / restored_elements** -- the candidate restores to the
  recipient parent cut bit-for-bit; `restored_elements` counts every restored tensor element.

Settings: one model loaded at a time; the decoder runs fp32; FLUX.2 Klein steps in bfloat16 and is
block-streamed. Prompts for the decoder are fixed token ids (a mechanics check, not a tokenized
natural-language battery). Only `saturn_pub` and public frameworks run; the job launch and
collection scripts are not part of the public package. `results/` in this directory holds the
scrubbed reports and the job ids.

## Reproduce with plain Python

```bash
# Offline tiny smoke (no weights, CPU, a few seconds):
python run.py --adapter decoder --tiny --output outputs/decoder-tiny
python run.py --adapter flux2 --tiny --steps 3 --output outputs/flux2-tiny

# Real weights:
python run.py --adapter decoder --model Qwen/Qwen2.5-0.5B --device cuda \
    --dtype float32 --local-files-only --output outputs/qwen2.5-0.5b
python run.py --adapter flux2 --model /path/to/FLUX.2-klein-4B --device cuda \
    --residency streamed --steps 4 --output outputs/klein4b
```

Each run writes `report.json` and prints a `METADATA_DELTA_ROUTE_REPORT=` marker. The process exits
non-zero if any exactness check (exact rollback, deduplicated hydration, a measured change, no
embedded payloads) fails.

## Claim boundary

Exactness here is about the mechanics of one program on one device and dtype: a selected,
content-verified delta applied and rolled back exactly. The measured effect is a carrier change
signal, not a semantic label, and no cross-family or image-quality claim is made.

## Reference results

Measured on an RTX 4080 with cached weights, offline. One model at a time; scratch and outputs on
a scratch SSD; weights read-only. The scrubbed reports are in `results/`.

| Adapter | Model | Device / dtype | Delta carrier (bytes) | Hydration | Effect L2 | Exact rollback | Elements | Reduction | Wall | Peak VRAM | Job |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|
| decoder | Qwen2.5-0.5B | cuda / fp32 | `hidden` (3,584) | 1 resolve / 2 requests | 12.20 | verified_exact | 31,622 | 0.984 | 13.6 s | 1,897 MB | `job-926bdacce54c` |
| flux2 | FLUX.2 Klein-4B | cuda / bf16 streamed | `text` (3,145,728) | 1 resolve / 2 requests | 2,321.99 | verified_exact | 9,230,345 | 0.984 | 31.8 s | 11,310 MB | `job-465b710d1fcf` |

Both runs: `changed_vs_native` true, `raw_payloads_embedded` false, the restored fingerprint equals
the recipient parent, and the deduplicated resolver served the repeat request from cache. These
illustrate one measured context, not an expected universal outcome.

# mdb with TransformerLens and nnsight

`mdb.interop` lets the three tools work on one loaded Hugging Face model: one name table, values that cross between them, batched nnsight interventions written as mdb edits, and mdb's native re-measurement of what the other tools claim. Install the tools with the `interop` extra (`transformer-lens==4.4.0`, `nnsight==0.8.2`); on Beast they live in `/mnt/ssd1tb/cache/beast/mdb-interop/site-tl440-nn082` (a `uv pip install --target --no-deps` overlay on the `mrun` env, which has torch 2.13.0 and transformers 5.13.1).

## One name table

`mdb.interop.names` maps every port of the Qwen2/Qwen3 port program to the TransformerBridge hook (TransformerLens 4 removed `HookedTransformer`; the bridge wraps the Hugging Face module) and to the nnsight path:

```python
from mdb.interop import names
names.to_tl("L22.A")          # 'blocks.22.attn.hook_z'
names.to_nnsight("L22.A")     # 'model.model.layers[22].self_attn.o_proj.input'
names.from_tl("blocks.22.attn.o.hook_in")   # 'L22.A' (aliases accepted)
names.table()                 # rows for docs / tooling
```

| Port | TransformerLens 4 bridge hook | nnsight path under `model.model.layers[l]` | Layout difference |
|---|---|---|---|
| `h` | `blocks.l.hook_resid_pre` | `.input` | none |
| `a` | `blocks.l.ln1.hook_out` | `.input_layernorm.output` | `ln1.hook_normalized` is RMSNorm without the gain |
| `q` `k` `v` | `blocks.l.attn.{q,k,v}.hook_out` | `.self_attn.{q,k,v}_proj.output` | TL splits heads; mdb is flat |
| `qn` `kn` (Qwen3) | `blocks.l.attn.{q,k}_norm.hook_out` | `.self_attn.{q,k}_norm.output` | TL holds (batch, head, pos, d) |
| `qr` `kr` | `blocks.l.attn.hook_rot_{q,k}` | `.self_attn.source.attention_interface_1.inputs[0][1]`, `[2]` | mdb `kr` holds the cached rows too |
| `s` | `blocks.l.attn.hook_attn_scores` | none | mdb `s` adds the mask (finfo.min); equal on the causal part |
| `p` | `blocks.l.attn.hook_pattern` | `.self_attn.output[1]` | none |
| `A` | `blocks.l.attn.hook_z` | `.self_attn.o_proj.input` | TL splits heads |
| `o` | `blocks.l.hook_attn_out` | `.self_attn.o_proj.output` | none |
| `r` | `blocks.l.hook_resid_mid` | `.post_attention_layernorm.input` | none |
| `n` | `blocks.l.ln2.hook_out` | `.mlp.input` | `ln2.hook_normalized` is without the gain |
| `g` | `blocks.l.mlp.gate.hook_out` (`mlp.hook_pre`) | `.mlp.gate_proj.output` | none |
| `sg` | none | `.mlp.act_fn.output` | TL has no hook on silu(g) |
| `u` | `blocks.l.mlp.in.hook_out` (`hook_pre_linear`) | `.mlp.up_proj.output` | none |
| `f` | `blocks.l.mlp.hook_post` | `.mlp.down_proj.input` | none |
| `m` | `blocks.l.hook_mlp_out` | `.mlp.down_proj.output` | none |
| `h_out` | `blocks.l.hook_resid_post` | `.output` | none |

MEASURED (tests, tiny Qwen2 fp32/bf16 and Qwen3 bf16 on CPU): every mapped value is bitwise equal across `mdb.trace`, an unprocessed bridge over the same weights and nnsight. Real-model parity is in `results/interop-20261113/` and the report below.

## Two tools rewrite the module tree they are handed

- `TransformerBridge.boot_transformers(hf_model=m)` replaces `m`'s modules in place (tiny Qwen2: 33 modules become 155). The plain forward stays bitwise equal, but `mdb.trace` then refuses `p` because native eager attention no longer runs (issue I1).
- nnsight's `.source` replaces an attention module's forward body for good. mdb's eager interceptor never fires there again; before the fix `Run.read("t17.L0.qr")` raised `KeyError` on Qwen2.5-1.5B (issue I2).

So `mdb.interop.tl.bridge(model, tok)` boots the bridge on `shared_clone(model)` (a new module tree over the same parameter tensors, no weight copy), `mdb.interop.nn.read` sources a shared clone, and `mdb.attach` refuses a tree rewritten by either tool. Never call `enable_compatibility_mode()` on a bridge that shares weights with a model in use: weight processing writes the parameters.

## Values cross the boundary

```python
from mdb.interop import tl
_, rome_cache = tl.bridge(model, tok).run_with_cache(rome_ids)
tl.import_cache(dbg, "tlrome", rome_cache, rome_tokens)      # origin "tlrome"
run = dbg.run(paris_tokens)
# cell and port values from the import, in the REPL language or dispatch:
#   whatif cache.L22.v[kv1,row=11] = tlrome:cache.L22.v[kv1,row=11] consumer=M[9625,15344]
#   whatif L22.A[h6] = tlrome:t17.L22.A consumer=M[9625,15344]
with mdb.trace(model, ports=[...]) as t: model(ids)
cache = tl.trace_cache(t, head_dim=128)                        # {TL hook name: tensor}
```

`Debugger.import_values(name, {t<r>.L<l>.<port>: tensor}, tokens=..., provenance=...)` is the generic entry (nnsight saves go through it too). Imported values are labelled `supplied`: another program computed them. Values from weight-processed TransformerLens (compatibility mode) are refused except `p`, because folding and centring put them in another basis.

## Batched interventions written as mdb edits

```python
from mdb.interop import nn
E = nn.Edit
out = nn.batch_whatif(model, recipients, consumers, {
    "head": [E("L22.A[h6]", 0.0, "scale")],                       # last row, as mdb's whatif
    "vcell": [E("cache.L22.v[kv1,row=8]", "donor")],              # reader scope (mdb's meaning)
    "vcell_seq": [E("cache.L22.v[kv1,row=8]", "donor", scope="sequence")],  # activation patching
    "control": [E("cache.L22.v[kv1,row=8]", "self", donor_row=6)],
    "feature": [E("L23.m", per_prompt_vectors, "add")],           # (prompts, columns) tensor
}, donors=donor_prompts)
```

A cell edit in `reader` scope is what mdb means by a state-cell edit: only the last row reads the changed cell. nnsight realises it natively by patching the value (or key) row and putting every earlier row's `o_proj` input back to its clean value at that layer. `sequence` scope is ordinary activation patching: later rows read the patched cell too and the change spreads through their residuals. Prompts are grouped by length so no batch is padded.

## Gradients through nnsight (one batched forward and backward)

```python
out = nn.grads(model, prompts, ["L14.h", "L20.h", "L27.h_out"], consumers)   # equal lengths
out["grads"]["L14.h"]        # (batch, pos, d): dM_i/dh for every row of every prompt
```

`nn.grads` is the nnsight 0.8 idiom (read from the installed source, `nnsight/tracing/tracer.py` and `nnsight/intervention/backward.py`):

- The outermost trace hands back only names bound to `.save()`d objects. A name bound inside `with nm.trace(...)` to an unsaved object (a list built in the body, a value read inside `with M.backward():`) is dropped on exit, by design. Bind result containers inside the trace with `nnsight.save({})` or `nnsight.save([])`, or `.save()` each value you bind. Mutating a container created before the trace also works, because the body runs over a copy of the frame's locals that holds the same objects. That is a side effect, not the documented route (issue T7).
- `.grad` is served only inside `with M.backward():` and must be requested in reverse forward order. `L<l>.h_out` and `L<l+1>.h` are one tensor, so ask for it once.
- An input-only backward must not build weight gradients. `nn.grads` freezes the parameters for the call (and restores the flags) and makes the embedding output the grad leaf. On Qwen3-1.7B, leaving weights trainable cost about +3.4 GB (job-b3bc03382981).
- The prompts are independent rows of one batch, so a single backward of `sum_i M_i` gives each prompt its own gradient. Do not loop one trace per prompt.

## TransformerLens weight processing on RMSNorm models

`tl.processed(bridge_or_cache)` reads the flag TL sets on the bridge itself (`bridge._weights_processed`). Never read `bridge.model`: `TransformerBridge.__getattr__` falls through to the Hugging Face module, which has no flag. Before 2026-11-14 the helper read it there. It returned False for every bridge, but it was right for an `ActivationCache`, whose `.model` is the bridge (measured on tiny Qwen3). So `import_cache(..., bridge_model=bridge)` could import processed values without the refusal firing, and TP1 recorded `processed: false` for a bridge that had been processed. `tl.processing(...)` adds what TL's own gates did (TL 4.4.0 `ProcessWeights.process_weights`). `fold_ln` folds the norm gains for LN and RMS. Writing-weight centring runs only for `normalization_type` LN/LNPre without `final_rms`. So on Qwen2/Qwen3/Llama, compatibility mode folds gains and centres the unembedding but leaves the residual stream uncentred (`residual_centred: False`). If you want centred residuals on an RMSNorm model, subtract the mean yourself and label the result derived. TL will not do it.

Cache only what you read: `bridge.run_with_cache(ids, names_filter=lambda n: n.endswith("hook_resid_pre"))`. An unfiltered cache of a 1.7B model holds every hook. Combined with the fp32 CPU copy that processing needs, it peaked at 21–24 GB of RAM in TP1.

## mdb verifies another tool's claim natively

```python
from mdb.interop import verify
rows = verify.verify(dbg, run, [verify.Claim("L22.A[h6]", 0.0, "scale", "M[9625,15344]",
                                             claimed=-5.9, tool="transformer_lens")])
# each row: claimed (supplied), linear (derived-local), exact (derived-exact),
# native (measured, batch-1 stepped), verdict and floor (P20), claim_minus_native
```

Limits: mdb edits ports only at the pending row (issue I3), so position-wise activation patching at earlier rows has no native counterpart yet; native verification costs about 5 s per edit on a 1.5B model (issue I4).

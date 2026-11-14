# Branches, pools, holds, core files, transcoders, custody

Debugger objects built from the 2026-11-14 library-experiments gap audit (`obsidian/reviews/2026-11-14-mdb-library-experiments-gap-audit.md`). The audit found that experiment drivers re-implemented mdb's control flow: the branch loop, the population of donor values, frozen-pattern arms, offline replicas of a captured prefix and the transcoder identity. Each is now a named object with time and provenance. Everything below is tested on tiny CPU models (`tests/test_engine_refs.py`, `test_engine_holds.py`, `test_analysis_branches.py`, `test_analysis_pools.py`, `test_core_custody.py`, `test_engine_core_file.py`, `test_analysis_transcoder.py`, `test_debugger_helpers.py`); no real-model number is claimed here.

## P5: selectors and origins on `Run.set` (`engine/refs.py`)

```python
donor.until("t34.logits")
ref = donor.ref("t34.L20.v[kv6]")            # ValueRef: the selected block + its origin pointer
run.set("t34.L20.v[kv6]", ref)               # ONE masked edit, selection "kv6", origin = pointer
run.set("t@last.L20.p[h3,src=0:5]", 0.0)     # the REPL's selector, from Python
run.set("cache.L20.v[kv6,row=11]", donor.ref("cache.L20.v[kv6,row=11]"))   # a state cell
```

The selector is planned by the REPL's own planner (`breakpoints.plan`), so the Python edit and `set ADDR[sel] = ORIGIN:ADDR[sel]` stage the same mask, value and origin digest (tested). A ref must have the selected block's shape exactly; a number or tensor broadcasts. The pointer is `{origin, via, address, sha256, shape, dtype, where, cut}`; reading a ref moves nothing. Rows: `t<r>.`, `t@last.` or none (the cursor's row).

## P4: holds (`engine/holds.py`, `Run.hold`)

```python
base.until(f"t{last}.logits")
run.hold("t@last.L>20.p", source=base)            # attention weights of the answer row
run.hold("t@last.L>20.norm_scale", source=base)   # RMSNorm scales (attn and mlp); or attn_norm_scale / mlp_norm_scale
run.hold("cache.L20.v[kv6,row=11]", source=base)  # one state cell
run.holds; run.release("hold1"); holds.final_held_logits(run, row, base)
```

A hold stages, on every covered `(row, layer)` the run reaches, an ordinary `replace` edit whose origin is `{kind: "hold", hold, origin, address, sha256, cut, rule}`. A port hold replaces the port with the source's value at the same address. A norm-scale hold keeps the scale, not the output: `a = w_in * (h * rsqrt(mean(h_src^2) + eps)).to(dtype)` in the native RMSNorm's operation order, with `h` this run's layer input read by a probe that stops at it. Holding a run against itself is bit-identical (tested). The final norm is not a decoder port: `final_held_logits` gives the logits with its scale held, derived-exact from native `h_out`.

Measured on tiny models (CPU, 2026-11-14): a held arm equals exp4's hand-built arm1 (p held) and arm2 (p and norm scales held) bit for bit on qwen2 bf16 and qwen3 fp32. On qwen3 fp32 (4 layers, edit at L1.m, holds L2-3, final scale held) the held arm's finite difference over nnsight's frozen-pattern, frozen-norm gradient `grad . d` was 1.00001, 1.00018, 0.99998 at `|d| ~ 1e-5` (3 seeds), converging as `|d|` shrinks; the free arm gave 0.603, -1.851, 1.004. While writing that test we found that frozen scales must be per row: a last-row scale broadcast to every row moves the other rows' keys and shifted the gradient by 1-5%. exp4 saved full-row inputs, so it was not affected.

## P1: branches (`analysis/branches.py`)

```python
br = dbg.branches(run, Margin(last, a, b), at=f"t{last}.L20.h")     # checkpoint = branch point
br.add("R1", [edit(f"t{last}.L20.v[kv6]", donor.ref(f"t{last}.L20.v[kv6]"))])
br.add("R1-held", br.arm("R1").edits, holds=["t@last.L>20.p", "t@last.L>20.norm_scale"],
       final_scale="held")
br.add_null("rnd", "random", like="R1", seed=0)   # matched-norm Gaussian change, same address
br.add_null("self", "self", like="R1")            # self-patch: dM must be 0
br.add_null("ans", "answer", like="M1")           # matched norm along the readout direction
br.run()                                          # stepped; br.run(mode="batched", recheck=k) for cell arms
br.table(); br.compare("R1", "R1-held"); br.seal(); br.restore()
```

The branch point is a checkpoint. Each arm restores it, turns its holds on, stages its edits at their layers, runs to the consumer's logits, reads `M`, releases the holds and restores again. Every arm row holds `M` and `dM`, the argmax and its edits' manifests (origins included). It also holds its holds, `effect_floors`, and the `stability` band (stable above `k` floors, else kernel-sensitive; `k` comes from a calibration, the caller or 4). It records the `kernel_path` it was measured on. All arms share one floor: 2 spacings at the base margin's larger `|logit|` (`verdict.margin_floor`) plus `extra_floor`, or the caller's `floor`. `batched` measures state-cell arms through `sweep.cell_sweep` (kernel path `batched`; the measured null-vs-stepped delta is added to their floor) and re-measures the `k` largest stepped (the exp5 pattern). `subspace.interchange` now rides on `Branches`: its arms `M` and `M_whole` share the branch point's floor, and `keep=True` returns the object so nulls can be added. `seal()` writes a `branches` record with custody.

## P6: pools (`analysis/pools.py`)

```python
pool = dbg.pool(items, "cache.L20.v[kv6,row=@carrier]", labels=("country",),
                where=lambda item, run: item["correct"], name="carrier-v")
pool.mean_for(item)                      # leave-one-out: the item is excluded by construction
pool.class_mean_for(item, by="country")  # same-label mean, item excluded
pool.donor(item, rule="other:country")   # first other-label item in pool order
pool.random(item, seed=0)                # Gaussian with the item's norm
pool.seal(); pool.save(path); Pool.load(path)
```

Each item is one run of its token ids, stepped until the address is computed; `@role` binds per item. Every handed-out value is a `ValueRef` whose pointer names the pool, its record id once sealed, the rule, `n`, the items digest and `excluded`, so an edit says exactly which items went into its value. The TP1 self-inclusion bug and the hop-math dead `exclude` argument cannot recur: `mean_for` has no way to include the item. Values are stored as their native dtype's exact floats in a sha256-checked payload.

## P2: core files (`engine/core_file.py`)

```python
src.dump("core.pt", row=K, capture=(f"t{last}.logits",))   # once
run = dbg.run_from("core.pt")          # this debugger's model, its device and dtype
run.core, run.core_check               # provenance, null check (argmax equal, max |dlogit|)
```

`dump` writes the boundary state of `t<K>.start`: K/V of rows `0..K-1` of every layer, the prompt, and the stepped-prompt slot. It adds the captured values and a sealed `core-file` record with every tensor's sha256, the procedure, the path edits baked into the cache, and custody. `load` refuses a changed tensor. The reloaded run steps every row from `K` natively on the target adapter, with ports, edits, checkpoints, holds and branches. On the same model and dtype it is bit-identical to the source (tested, including an edit after reload). On another dtype it is a labelled replica (`run.core["replica"]`). Measured on tiny CPU models, with a bf16 dump reloaded on an exact fp32 upcast of the same weights: argmax equal, max `|dlogit|` 5.4e-4 (qwen2) and 1.1e-3 (qwen3). A model of another family or shape is refused.

## P3: transcoders in the decomposition family (`analysis/sae.py`, `components.py`)

```python
tset = sae.open_transcoders("lowl0", "/mnt/big/llm-models/qwen3-1.7b-transcoders-lowl0")
tc = tset.get(20)                                          # reads L20.n, writes L20.m
cb = components.transcoder_binding(run, "mlp20", f"t{last}.L20.m", tc)
components.component_table(run, cb, Margin(last, a, b))    # features + other + b_dec + error
run.set(f"t{last}.L20.m", cb.edited(remove=("feature[123]",)).reshape(1, 1, -1).to(dtype))
```

A `TranscoderBasis` has the mwhanna layout that SAELens's `mwhanna_transcoder` loader uses: `relu(W_enc n + b_enc)`, no `b_dec` on the input, `W_dec` rows as directions. The active set is chosen in float32 and the activations are recomputed in float64 from the native bits, as exp3/exp4 did by hand. `transcoder_binding` returns a `ComponentBinding` with `site="mlp-write"`, the same type the value-cell binding and `SubspaceBinding.decompose` return. Its error is the native `m` minus the reconstruction, so the partition closes to float64 roundoff. `component_table`, `edited`, `feature_value` and `features.activations` read it. `to_saelens()` builds SAELens's own `Transcoder`; its active set and decode match (tested). `describe()` carries SAELens `hook_name`/`hook_name_out`, and `node(row, f)` gives circuit-tracer node ids.

## P8: custody and record sinks (`core/custody.py`)

```python
dbg.custody(script=__file__, prereg="prereg.json", kernel_path="mdb-stepped")
sink = dbg.records_to("out/records.jsonl")   # every record sealed from now on, any kind
sink.records()                                # read back, each verified
```

While custody is declared, every sealed record gets `body.custody` = script and prereg sha256 (hashed when declared), library versions of what is imported (mdb, saturn_pub, torch, transformers, interop libraries) and the kernel path. This covers forecast, trace, watch-hits, circuit-map, supplied-forecast, branches, pool and core-file records. Undeclared, records are sealed exactly as before (REPL goldens and stored ids unchanged). `branches`, `pool` and `core-file` records always carry custody. A job payload can declare with no code change: `MDB_SCRIPT`, `MDB_PREREG`, `MDB_KERNEL_PATH`, `MDB_RECORDS`.

## Small helpers

- `verdict.margin_floor(logits, a, b, spacings=2, extra=0)` is the showcase `common.floor_of` on `verdict.spacing` (equal on 100 random bf16 cases, tested). `verdict.floors(effect, floor)` is the effect-in-floors column.
- `circuit_map.group_writer(adapter, layer, kv=None)` is a cached `group_writers` of the layer's `o_proj`. It is recomputed after an in-place weight edit.
- `sets.roles_from_text(tokenizer, text, {"hint": "the answer is", "q": (start, end), ...})` returns token rows per role from offset mappings. It replaces the four CoT `spans_for` copies.
- The direct-effect vector the drivers recomputed as `rho_F gamma (W_U[a] - W_U[b])` is `reads.readout(run, Margin(row, a, b)).vector` (with `.rho`, `.bias`, `.native`).

## Not built here

The gap audit lists P7 (Python `each`), P9 (feature lifetime strips), P10 (sealed suspect lists) and P12 (verdict statistics); none of them was in scope. A core file holds the cache at a row start, not a mid-row state. Holds on the final norm act through `final_held_logits` only, not as a run-wide switch.

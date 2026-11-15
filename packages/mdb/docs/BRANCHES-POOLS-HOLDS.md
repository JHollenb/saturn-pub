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

A hold stages, on every covered `(row, layer)` the run reaches, an ordinary `replace` edit whose origin is `{kind: "hold", hold, origin, address, sha256, cut, rule}`. A port hold replaces the port with the source's value at the same address. A norm-scale hold keeps the scale, not the output: `a = w_in * (h * rsqrt(mean(h_src^2) + eps)).to(dtype)` in the native RMSNorm's operation order, with `h` this run's layer input read by a probe that stops at it. Holding a run against itself is bit-identical (tested). A hold is the last write at its port, so staging an edit at a held port is refused (`ValueError`; `run.set(..., override_hold=True)` stages it anyway and records `origin.overridden_by`), and adding a hold over a staged edit at its port is refused. Auto hold names are monotonic (`hold1`, `hold2`, ... never reused after a release). `Branches` on a run with active holds builds its reference under the same holds, so `reference == M0` and the self-patch null reads 0; an arm row lists only the edits fired from the branch point on, and the path edits before it are recorded once under `at.path_edits`. Arm edits are staged in program order whatever the list order. The final norm is not a decoder port: `final_held_logits` gives the logits with its scale held, derived-exact from native `h_out`.

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
br.run()                                          # stepped (the authority)
br.run(mode="batched")                            # cell + port arms in batched forwards; recheck="band" by default
br.table(); br.compare("R1", "R1-held"); br.seal(); br.restore()
```

The branch point is a checkpoint. Each arm restores it, turns its holds on, stages its edits at their layers, runs to the consumer's logits, reads `M`, releases the holds and restores again. Every arm row holds `M` and `dM`, the argmax and its edits' manifests (origins included). It also holds its holds, `effect_floors`, and the `stability` band (stable above `k` floors, else kernel-sensitive; `k` comes from a calibration, the caller or 4). It records the `kernel_path` it was measured on. All arms share one floor: 2 spacings at the base margin's larger `|logit|` (`verdict.margin_floor`) plus `extra_floor`, or the caller's `floor`. `batched` measures arms in batched native forwards through `sweep.cell_sweep` (kernel path `batched`; twice the measured null-vs-stepped delta is added to their floor: a margin is a difference of two logits). Two kinds of arm batch: whole-cell `replace` edits of `cache.L<l>.k|v[kv<g>,row=<r>]`, and edits of the pending row's native-module ports (`a q k v n g sg u f A o m`: every port that is a `pre` or `out` hook on a layer submodule; not `h`, `r`, `h_out` or the eager-attention ports `qr s p`), any op, selector or mask, at any layer from the branch point on. A port arm is realised by a forward hook that applies the engine's own `PortEdit` (`edit.apply`, the same cast-once, one-rounded-op arithmetic) to that arm's batch element only, so each arm has its own value and mask and arms do not see each other. Measured on tiny qwen2 bf16 and qwen3 fp32: `f` and `m` port arms (donor block ref, selector scale, masked add, two layers in one arm) equal the stepped arms within the arm's floor from `L0.h` and from a later branch point, and equal nnsight's batched patch (`interop.nn.batch_whatif`) within the floor (`tests/test_analysis_branches.py`). Holds and `final_scale="held"` arms stay stepped-only. A cell value of shape `(d,)` for one kv head and one row is normalised to the block shape `(1, 1, 1, d)` when the arm is created (EX1), so the batched path and the stepped recheck accept the same arm.

**Recheck.** A batched forward is a different kernel path from the stepped run. The audited showcase (exp6, `analysis/stability.py`) found that this changes a verdict only for effects within about 4 floors (0 of 214 above). So the default is `recheck="band"`: only an arm whose batched effect could flip is re-measured stepped, `1/k <= |dM| / floor <= k` with `k` the stability margin (4, or the calibrated one). An arm above `k` floors is stable and an arm below `1/k` floors is clearly null; both keep their batched row. The lower edge `1/k` is a supplied rule (the multiplicative mirror of the measured upper edge), not a measurement; an arm with no floor (callable consumer) cannot be judged and is rechecked. `recheck=0` rechecks nothing, `"all"` every arm, and an int `k` the `k` arms of largest `|dM|` (the exp5 pattern, unchanged). Every batched row carries `recheck = {done, why}` (the why names the band edge and the arm's floors), and a rechecked row keeps `stepped` and `kernel_delta` (batched minus stepped). Measured on a tiny model, 40 m-port writes of growing size: `recheck="all"` re-measures 40 arms, `"band"` 3 (qwen2 bf16) and 5 (qwen3 fp32).

`subspace.interchange` now rides on `Branches`: its arms `M` and `M_whole` share the branch point's floor, and `keep=True` returns the object so nulls can be added. `seal()` writes a `branches` record with custody.

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

Each item is one run of its token ids, stepped until the address is computed; `@role` binds per item. Every handed-out value is a `ValueRef` whose pointer names the pool, its record id once sealed, the rule, `n`, the items digest and `excluded`, so an edit says exactly which items went into its value. The TP1 self-inclusion bug and the hop-math dead `exclude` argument cannot recur: `mean_for` has no way to include the item. `exclude="tokens"` (default) also drops items with the same prompt tokens under another id (a duplicated prompt); `exclude="id"` drops only the id, and the record counts `duplicates`. `Pool.load` refuses a file whose `items` differ from the payload ids, the sealed `items_sha256`, or the payload's labels and token digests. Labels read nested `labels` first, then the top-level key. Cache-cell pools record `kernel_path` `mdb-prefill-batched` and each item's `origin.prefill`. Values are stored as their native dtype's exact floats in a sha256-checked payload.

## P2: core files (`engine/core_file.py`)

```python
src.dump("core.pt", row=K, capture=(f"t{last}.logits",))   # once
run = dbg.run_from("core.pt")          # this debugger's model, its device and dtype
run.core, run.core_check               # provenance, null check (argmax equal, max |dlogit|)
```

`dump` writes the boundary state of `t<K>.start`: K/V of rows `0..K-1` of every layer, the prompt, and the stepped-prompt slot. It adds the captured values and a sealed `core-file` record with every tensor's sha256, the procedure, the path edits baked into the cache, and custody. `load` refuses a changed tensor, a prompt that differs from the sealed prompt sha256, and tokens that are not the sealed prompt prefix. `dump` refuses an explicit `capture` when the run committed edits after the split (the default capture is then dropped and `post_split_edits` counts them). The reloaded run steps every row from `K` natively on the target adapter, with ports, edits, checkpoints, holds and branches. On the same model and dtype it is bit-identical to the source (tested, including an edit after reload). On another dtype it is a labelled replica (`run.core["replica"]`, also when the model identity differs, e.g. Base vs Instruct). Measured on tiny CPU models, with a bf16 dump reloaded on an exact fp32 upcast of the same weights: argmax equal, max `|dlogit|` 5.4e-4 (qwen2) and 1.1e-3 (qwen3). A model of another family or shape is refused.

## P3: transcoders in the decomposition family (`analysis/sae.py`, `components.py`)

```python
tset = sae.open_transcoders("lowl0", "/mnt/big/llm-models/qwen3-1.7b-transcoders-lowl0")
tc = tset.get(20)                                          # reads L20.n, writes L20.m
cb = components.transcoder_binding(run, "mlp20", f"t{last}.L20.m", tc)
components.component_table(run, cb, Margin(last, a, b))    # features + other + b_dec + error
run.set(f"t{last}.L20.m", cb.edited(remove=("feature[123]",)).reshape(1, 1, -1).to(dtype))
```

A `TranscoderBasis` has the mwhanna layout that SAELens's `mwhanna_transcoder` loader uses: `relu(W_enc n + b_enc)`, no `b_dec` on the input, `W_dec` rows as directions. The active set is chosen in float32 (`W_enc`, `b_enc`, `n` widened exactly) and the activations are recomputed in float64 from the native bits, as exp3/exp4 did by hand. `transcoder_binding` returns a `ComponentBinding` with `site="mlp-write"`, the same type the value-cell binding and `SubspaceBinding.decompose` return. Its error is the native `m` minus the reconstruction, so the partition closes to float64 roundoff. `component_table`, `edited`, `feature_value` and `features.activations` read it. `to_saelens()` builds SAELens's own `Transcoder`; its active set and decode match (tested). The loader refuses a file with `W_skip` or a JumpReLU `threshold` (not modelled) and reads `config.json` or `config.yaml`. `describe()` carries SAELens `hook_name`/`hook_name_out`, and `node(row, f)` gives circuit-tracer node ids.

## P8: custody and record sinks (`core/custody.py`)

```python
dbg.custody(script=__file__, prereg="prereg.json", kernel_path="mdb-stepped")
sink = dbg.records_to("out/records.jsonl")   # every record sealed from now on, any kind
sink.records()                                # read back, each verified
```

While custody is declared, every sealed record gets `body.custody` = script and prereg sha256 (hashed when declared), library versions of what is imported (mdb, saturn_pub, torch, transformers, interop libraries) and the kernel path. This covers forecast, trace, watch-hits, circuit-map, supplied-forecast, branches, pool and core-file records. Undeclared, records are sealed exactly as before (REPL goldens and stored ids unchanged). `branches`, `pool` and `core-file` records always carry custody. A job payload can declare with no code change: any of `MDB_SCRIPT`, `MDB_PREREG`, `MDB_KERNEL_PATH` declares custody (an unreadable file raises on every seal, never once); `MDB_RECORDS` opens a sink.

## Small helpers

- `verdict.margin_floor(logits, a, b, spacings=2, extra=0)` is the showcase `common.floor_of` on `verdict.spacing` (equal on 100 random bf16 cases, tested). `verdict.floors(effect, floor)` is the effect-in-floors column.
- `circuit_map.group_writer(adapter, layer, kv=None)` is `group_writers` of the layer's `o_proj`, computed from the weight as it is on every call (no cache: `.data` edits are invisible to version counters).
- `sets.roles_from_text(tokenizer, text, {"hint": "the answer is", "q": (start, end), ...})` returns token rows per role from offset mappings. It replaces the four CoT `spans_for` copies.
- The direct-effect vector the drivers recomputed as `rho_F gamma (W_U[a] - W_U[b])` is `reads.readout(run, Margin(row, a, b)).vector` (with `.rho`, `.bias`, `.native`).

## Not built here

The gap audit lists P7 (Python `each`), P9 (feature lifetime strips), P10 (sealed suspect lists) and P12 (verdict statistics); none of them was in scope. A core file holds the cache at a row start, not a mid-row state. Holds on the final norm act through `final_held_logits` only, not as a run-wide switch.

## Multi-site feature watch (`analysis/features.py`, `engine/breakpoints.py`)

A `FeatureContext` is one site (one basis, one layer). `FeatureCondition(text, contexts)` also takes several contexts (a sequence or `{layer: context}`; `features.feature_sites({layer: basis}, row=r, sets=..., constants=...)` builds one per layer): the condition reads, at each stop, the context of that stop's layer, the watch stops only at those layers, keeps one memory per layer, and each hit names its stop (`hit["site"]`, `hit["layer"]`). `table.add_watch("t4.L*.h", program, mode="edge", compiled=cond)` therefore equals one watch `t4.L<l>.h` per layer run one by one (edge, level and change modes), with the same stop reasons (`watch`, `watch+breakpoint`) and reverse walk. Every context must declare the sets, labels and constants the text names. Tested on tiny qwen2 bf16 and qwen3 fp32 with a random `FeatureBasis` per layer (`tests/test_analysis_components.py`).

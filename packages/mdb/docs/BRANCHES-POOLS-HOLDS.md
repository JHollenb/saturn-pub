# Branches, pools, holds, checkpoints, transcoders, custody

Objects for the control flow an experiment script otherwise writes by hand: the branch loop, the population of donor values, frozen-pattern arms, offline replicas of a captured prefix and the transcoder identity. Each is a named debugger object with time and provenance. Everything below is tested on tiny CPU models (`tests/test_engine_refs.py`, `test_engine_holds.py`, `test_analysis_branches.py`, `test_analysis_pools.py`, `test_core_custody.py`, `test_engine_core_file.py`, `test_engine_checkpoint.py`, `test_analysis_transcoder.py`, `test_debugger_helpers.py`); real-model numbers here are marked as such.

## Selectors and origins on `Run.set` (`engine/refs.py`)

```python
donor.until("t34.logits")
ref = donor.ref("t34.L20.v[kv6]")            # ValueRef: the selected block + its origin pointer
run.set("t34.L20.v[kv6]", ref)               # ONE masked edit, selection "kv6", origin = pointer
run.set("t@last.L20.p[h3,src=0:5]", 0.0)     # the REPL's selector, from Python
run.set("cache.L20.v[kv6,row=11]", donor.ref("cache.L20.v[kv6,row=11]"))   # a state cell
```

The selector is planned by the REPL's own planner (`breakpoints.plan`), so the Python edit and `set ADDR[sel] = ORIGIN:ADDR[sel]` stage the same mask, value and origin digest (tested). A ref must have the selected block's shape exactly; a number or tensor broadcasts. The pointer is `{origin, via, address, sha256, shape, dtype, where, cut}`; reading a ref moves nothing. Rows: `t<r>.`, `t@last.` or none (the cursor's row).

## Holds (`engine/holds.py`, `Run.hold`)

```python
base.until(f"t{last}.logits")
run.hold("t@last.L>20.p", source=base)            # attention weights of the answer row
run.hold("t@last.L>20.norm_scale", source=base)   # RMSNorm scales (attn and mlp); or attn_norm_scale / mlp_norm_scale
run.hold("cache.L20.v[kv6,row=11]", source=base)  # one state cell
run.holds; run.release("hold1"); holds.final_held_logits(run, row, base)
```

A hold stages, on every covered `(row, layer)` the run reaches, an ordinary `replace` edit whose origin is `{kind: "hold", hold, origin, address, sha256, cut, rule}`. A port hold replaces the port with the source's value at the same address. A norm-scale hold keeps the scale, not the output: `a = w_in * (h * rsqrt(mean(h_src^2) + eps)).to(dtype)` in the native RMSNorm's operation order, with `h` this run's layer input read by a probe that stops at it. Holding a run against itself is bit-identical (tested). A hold is the last write at its port, so staging an edit at a held port is refused (`ValueError`; `run.set(..., override_hold=True)` stages it anyway and records `origin.overridden_by`), and adding a hold over a staged edit at its port is refused. Auto hold names are monotonic (`hold1`, `hold2`, ... never reused after a release). `Branches` on a run with active holds builds its reference under the same holds, so `reference == M0` and the self-patch null reads 0; an arm row lists only the edits fired from the branch point on, and the path edits before it are recorded once under `at.path_edits`. Arm edits are staged in program order whatever the list order. The final norm is not a decoder port: `final_held_logits` gives the logits with its scale held, derived-exact from native `h_out`.

Measured on tiny models (CPU, 2026-11-14): a held arm equals a hand-built arm that re-sets `p` (and, for the second, `p` and the norm scales) at every later layer, bit for bit on qwen2 bf16 and qwen3 fp32. On qwen3 fp32 (4 layers, edit at L1.m, holds L2-3, final scale held) the held arm's finite difference over nnsight's frozen-pattern, frozen-norm gradient `grad . d` was 1.00001, 1.00018, 0.99998 at `|d| ~ 1e-5` (3 seeds), converging as `|d|` shrinks; the free arm gave 0.603, -1.851, 1.004. While writing that test we found that frozen scales must be per row: a last-row scale broadcast to every row moves the other rows' keys and shifted the gradient by 1-5%. Holds keep per-row scales.

## Branches (`analysis/branches.py`)

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

The branch point is a checkpoint. Each arm restores it, turns its holds on, stages its edits at their layers, runs to the consumer's logits, reads `M`, releases the holds and restores again. Every arm row holds `M` and `dM`, the argmax and its edits' manifests (origins included). It also holds its holds, `effect_floors`, and the `stability` band (stable above `k` floors, else kernel-sensitive; `k` comes from a calibration, the caller or 4). It records the `kernel_path` it was measured on. All arms share one floor: 2 spacings at the base margin's larger `|logit|` (`verdict.margin_floor`) plus `extra_floor`, or the caller's `floor`. `batched` measures arms in batched native forwards through `sweep.cell_sweep` (kernel path `batched`; twice the measured null-vs-stepped delta is added to their floor: a margin is a difference of two logits). Two kinds of arm batch: whole-cell `replace` edits of `cache.L<l>.k|v[kv<g>,row=<r>]`, and edits of the pending row's native-module ports (`a q k v n g sg u f A o m`: every port that is a `pre` or `out` hook on a layer submodule; not `h`, `r`, `h_out` or the eager-attention ports `qr s p`), any op, selector or mask, at any layer from the branch point on. A port arm is realised by a forward hook that applies the engine's own `PortEdit` (`edit.apply`, the same cast-once, one-rounded-op arithmetic) to that arm's batch element only, so each arm has its own value and mask and arms do not see each other. Measured on tiny qwen2 bf16 and qwen3 fp32: `f` and `m` port arms (donor block ref, selector scale, masked add, two layers in one arm) equal the stepped arms within the arm's floor from `L0.h` and from a later branch point, and equal nnsight's batched patch (`interop.nn.batch_whatif`) within the floor (`tests/test_analysis_branches.py`). Holds and `final_scale="held"` arms stay stepped-only. A cell value of shape `(d,)` for one kv head and one row is normalised to the block shape `(1, 1, 1, d)` when the arm is created, so the batched path and the stepped recheck accept the same arm.

**Recheck.** A batched forward is a different kernel path from the stepped run. On two real models (Qwen3-0.6B-Base and Qwen2.5-0.5B, measured on the developers' GPU; see Verdict stability in the [README](../README.md#verdict-stability-across-realisations)) this changed a verdict only for effects within about 4 floors (0 of 214 above). So the default is `recheck="band"`: only an arm whose batched effect could flip is re-measured stepped, `1/k <= |dM| / floor <= k` with `k` the stability margin (4, or the calibrated one). An arm above `k` floors is stable and an arm below `1/k` floors is clearly null; both keep their batched row. The lower edge `1/k` is a supplied rule (the multiplicative mirror of the measured upper edge), not a measurement; an arm with no floor (callable consumer) cannot be judged and is rechecked. `recheck=0` rechecks nothing, `"all"` every arm, and an int `k` the `k` arms of largest `|dM|`. Every batched row carries `recheck = {done, why}` (the why names the band edge and the arm's floors), and a rechecked row keeps `stepped` and `kernel_delta` (batched minus stepped). Measured on a tiny model, 40 m-port writes of growing size: `recheck="all"` re-measures 40 arms, `"band"` 3 (qwen2 bf16) and 5 (qwen3 fp32).

`subspace.interchange` now rides on `Branches`: its arms `M` and `M_whole` share the branch point's floor, and `keep=True` returns the object so nulls can be added. `seal()` writes a `branches` record with custody.

**Probes.** `Branches(run, consumer, at=..., probes={"v": fn, ...})` takes named callables `probe(run) -> number | list`. Each stepped arm runs every probe after its logits and before the restore, so a probe reads another value of the arm's own continuation (a later cache cell, a port's norm). The base values are `br.base_probes`, and each arm row stores its values under `probes` (measured). Batched arms carry no probes, because their forward is not a `Run`. Tested against a hand run (`test_probes_read_each_arm_before_restore`).

**Large records.** When the arms (with edits and probes) exceed one record's bound, `seal()` keeps a compact per-arm table (`name, kind, M, dM, argmax, stability, effect_floors, kernel_path`), sets `arms_compact: true`, and stores the full rows under `arms_payload`, a sha256 pointer (`bounded.put_json`). With no project store that pointer is `state: "absent"`: the rows are held in memory on the run (`bounded.load_json(pointer, run=run)`) and, while a record sink is open, written to the sink file (custody and record sinks below, `sink.load_json`).

## Pools (`analysis/pools.py`)

```python
pool = dbg.pool(items, "cache.L20.v[kv6,row=@carrier]", labels=("country",),
                where=lambda item, run: item["correct"], name="carrier-v")
pool.mean_for(item)                      # leave-one-out: the item is excluded by construction
pool.class_mean_for(item, by="country")  # same-label mean, item excluded
pool.donor(item, rule="other:country")   # first other-label item in pool order
pool.random(item, seed=0)                # Gaussian with the item's norm
pool.seal(); pool.save(path); Pool.load(path)
```

Each item is one run of its token ids, stepped until the address is computed; `@role` binds per item. Every handed-out value is a `ValueRef` whose pointer names the pool, its record id once sealed, the rule, `n`, the items digest and `excluded`, so an edit says exactly which items went into its value. `mean_for` has no way to include the item itself (an item averaged into its own donor value inflates every effect). `exclude="tokens"` (default) also drops items with the same prompt tokens under another id (a duplicated prompt); `exclude="id"` drops only the id, and the record counts `duplicates`. `Pool.load` refuses a file whose `items` differ from the payload ids, the sealed `items_sha256`, or the payload's labels and token digests. Labels read nested `labels` first, then the top-level key. Cache-cell pools record `kernel_path` `mdb-prefill-batched` and each item's `origin.prefill`. Values are stored as their native dtype's exact floats in a sha256-checked payload.

## Checkpoints (`engine/checkpoint.py`; core files)

```python
rec = run.dump("cut.pt")                     # a file, at the cursor (a cut inside a row too)
info = dbg.checkpoint(run, "pre")            # a project: info["state"] is the record id
run.checkpoint("pre", durable=dbg.project)   # the same from a Run (or durable="cut.pt")

r = dbg.run_from("cut.pt")                   # or a record id / unique prefix, any process
r.from_checkpoint                            # {"checkpoint": id, "path": "unverified"}
r.verify()                                   # replay the recipe once; then "path verified"
r = dbg.restart("pre", verify=True)          # project name; raises unless the path verifies
r.core, r.lineage                            # provenance; what the fingerprints chain back to
dbg.branches(r, Margin(...), at=...)         # Branches, Pool, each run from it
dbg.pool([{"id": "a", "core": "cut.pt"}], "t@last.L20.h")
dbg.each([{"name": "x", "core": "cut.pt"}], "whatif ...")   # or dbg.each(items, line, core=)
```

In the REPL: `start from=ID|PATH [name=NAME] [verify=true|report]`.

A checkpoint is the run's state at a cut as content-addressed pointers to its tensors, not a
recipe to replay. It holds the native `StateCut` of each kept trail mark, slot by slot: per
layer K/V, the hidden carrier of a row in flight (a cut inside a row is the layer boundary
before the cursor), logits, the sampled token, the stepped-prompt slot, plus the cut's boundary,
execution point, surface and parent fingerprint. It also holds the cursor (a port in the
pending layer and the edits and cell edits staged there), the committed edits of every kept
transition and of the path before the first kept mark, all edit and mask tensors, the active
holds (each with the source values it reads from the cut's row on, so the reloaded run re-arms
it and continues held bit for bit; `holds="drop"` records them as dropped instead), the run's
recipe
(`Run.checkpoint()`, provenance and the input of `verify`) and optional captured values. A K/V
tensor is stored as parts: the prefix cache the engine already shares, then the appended rows in
power-of-two blocks aligned from its end. Every part is addressed by the sha256 of its bytes and
stored once, so cuts of one run share parts (measured: a 14-mark checkpoint of a 512-row fp32
Qwen3-0.6B cut held 117.4 MB for 117.4 MB of K/V).

A checkpoint is a snapshot, and that holds for re-armed holds too. A re-armed hold reads the donor
values its source held **at save time** (only the addresses the hold reads from the cut's row on,
and only those the source had computed then). Values the source computes after the save are not part
of the checkpoint, and the live source run is not saved. A re-armed hold that reaches an address the
source had not computed at save time is refused, as a live source refuses an uncomputed address:
`ValueError: <address> was not computed by the source when the checkpoint was saved`. To hold on a
later row, save again after the source has computed it.

Where it lives: a self-contained file (`torch.save` of plain data, loadable with
`weights_only`), or a project. In a project every part is a tensor blob of the store, the
state document is a JSON blob, and a sealed `checkpoint-state` record (schema `mdb-record-v3`)
is indexed under the session. `run_from` takes a path, a record id or a unique prefix of at
least 8 hex digits, a `Run.checkpoint(durable=...)` result, or a loaded checkpoint
(`dbg.load_checkpoint(ref)`). A string that names both an existing file and a record of the
project is refused as ambiguous (pass `Path(...)` for the file). `dbg.checkpoint(run, name)`
writes the replay recipe and the `checkpoint` record as before, and the state; it no longer
stores a whole native cut beside it (K/V would be stored twice), and cuts in projects written
before are still read. A value taken from a checkpoint by name (`NAME:ADDR`) rehydrates its
state (`via` `checkpoint`). `dbg.restart(name)` rehydrates the state when the project holds it
(`replay=True`, or a project written before, replays the recipe). It refuses with the recipe's
codes first: model identity, numeric scope, missing or corrupt blob.

The record holds the document's sha256, the sha256 of every part (`tensors.sha256`, 60 digests
per string, inline up to 3,600 parts; beyond that only `tensors.parts_sha256` and the sealed
document list them), the cut and root fingerprints, the prompt digest, the procedure, the
recipe's sha256, the lineage and custody. Loading checks the seal, the document digest, every
part's sha256 (file: all parts on load; project: each blob through the store when first read),
and that the document agrees with the record.

**Rehydration.** Every kept cut is rebuilt from its parts. Its fingerprint is recomputed from
the content and the recorded parent, and a cut that differs from the record is refused. The
run's trail is the kept marks with their committed edits. The cursor is put back with its port
and staged edits. Nothing before the cursor is recomputed. A cursor at a port inside a layer
costs one probe of that layer, as moving there costs on the source run. Tested bit for bit at
every port phase of a row (`start`, `emb`, `L0.p`, `L0.m`, `L0.h_out`, `L1.sg`, `fn`, `logits`,
`sample`), with an edit or cell edit staged at the cursor, and with a port hold and a
norm-scale hold active, on tiny Qwen2 bf16 and Qwen3 fp32. Each kept cut's parent must be the
previous kept mark's fingerprint.
`history`: `cursor` (only the cut), `row` (default: every boundary of the cut's row before it,
so the reloaded run rewinds inside that row), or `root` (back to the source's root: row starts
as cuts, the boundaries between them replayed on demand, as `retain_rows` does). A rewind before
the first kept mark is refused with that advice. So, with `cursor`, verbs that read the rows
before the cut through their ports (`symbols`, for one) refuse ("before the start of this
run"), although those rows' K/V are in the cache: use `row` or `root` for them.

Saving at an earlier boundary than the cursor (`at="t4.start"`, `row=4`) is refused when the
run edited after it, committed or staged. A cut never silently drops an edit. Saved that way,
the values the run computed later (default: the last prompt row's logits) are captured for the
null check.

**Lineage and identity.** A cut's fingerprint is the sha256 of the model identity, the
execution contract (dtype, device, numeric scope, library versions), the state content, the
execution point and the parent fingerprint. A rehydrated run has the source's fingerprints
(`run.lineage["root"] == "prompt"`), so checkpoints move between them both ways. A run
rehydrated from an earlier checkpoint of the same lineage restores a later one by replaying
from its first mark. `Run.restore` refuses a checkpoint before the run's first mark, or one
whose path before it differs (another lineage). The edits of the path before the first mark are
part of the run (`committed_edits`, `read_info()["upstream_edits"]`, `edited`,
`checkpoint()["path_edits"]`).

**Integrity, not authenticity.** The seal says a file is the record it names: a changed byte,
tensor, parent or document is refused, and the record id is a digest of the document, edit
history and lineage included. It does not say that history is true. The parent is recorded,
not re-derived, and anyone can strip the edits from a document and seal a new record for it:
the state loads (its tensors are genuine) and claims a clean path (measured: the continuation
equals the edited source, not a clean run). So every value and record of a run rehydrated from
a checkpoint carries `from checkpoint <id>, path unverified` until its path is verified by
recompute: `run.from_checkpoint`, the `from_checkpoint` of each read's provenance (the REPL
shows `[measured, from checkpoint ab12cd34…, path unverified]`), `procedure.from_checkpoint`
of every record that names its procedure, and `custody.from_checkpoint` of every record
sealed about it, from the REPL or the Python API (the sealer names its run:
`seal_record(run=...)`; a pool or sweep about several lists them under `checkpoints`). `fresh()` inherits the label. `run.verify()` (or `verify=True` on
`run_from`, `restart`, `start from=`) replays the recipe from the lineage root and checks it
reaches the same fingerprint (measured); it raises `CheckpointUnverified` otherwise, also when
it cannot check (another model, or a lineage rooted at another checkpoint with no `root=`), and
the run is then `path verified`. `verify="report"` records the result in `run.core["verified"]`
instead of raising. A downstream record cites the checkpoint id it trusted, so a resealed file
is a different id.

**Replicas.** Another dtype, device or weights of the same shape is refused unless
`replica=True`. A replica roots a new lineage at the cursor (`lineage["root"] ==
"checkpoint"`), its state cast once to the target dtype, labelled `run.core["replica"]`, and
null-checked against the captured values (`run.core_check`: argmax equal, max `|dlogit|`).
Values are captured only when the source computed them past the cut (save at an earlier
boundary); with none, `core_check` and the label say `no null check`. A model of another family
or shape is refused.

**Reference runs.** `Run.fresh()` is a new run at the root of the run's lineage: the prompt, or
the checkpoint again (no prefix recompute). `Branches` builds its reference with it, so
branches of a rehydrated run measure the same rows as on the source. `Pool` items and `each`
items take `core` (alias `from`), and `each` takes `core=` from Python for the items that bring
no prompt or tokens of their own (an item with its own starts from them; a pair donor is
rehydrated only from its own `core`). The prompt is the checkpoint's, `prefill` does not apply,
and the pool origin names the checkpoint record.

Measured on CPU (5 threads, Qwen3-0.6B, cut at `t<last>.L12.m`, history `row`):

| prompt | recompute | rehydrate file | rehydrate record | fresh process (file / record / restart) | verify |
|---|---|---|---|---|---|
| 512 rows fp32, 511 batched | 0.97 s | 0.38 s | 0.40 s | 0.40 / 0.62 / 0.48 s | 0.97 s |
| 96 rows bf16, 80 stepped | 9.99 s | 0.077 s | 0.13 s | 0.083 / 0.13 / 0.12 s | 11.0 s |
| 96 rows fp32, 80 stepped | 11.3 s | 0.096 s | 0.16 s | 0.25 / 0.50 / 0.17 s | 12.1 s |

`fresh()` of a loaded checkpoint took 0.035 to 0.10 s. Writing took 0.09 to 2.7 s for a file and
0.45 to 1.4 s for a project. Every continuation was bit-identical to the source's. Rehydration
cost is reading and hashing the parts once, plus one probe for a port cursor.

**Legacy core files.** `mdb-core-file-v1` files (`core_file.dump_v1`) still load
(`core_file.load` / `run_from`). They hold the K/V of rows `0..K-1` at `t<K>.start` with no
parent, so their run roots a foreign lineage (`lineage["root"] == "checkpoint"`). Another dtype
or weights needs `replica=True` there too.

## Transcoders in the decomposition family (`analysis/sae.py`, `components.py`)

```python
tset = sae.open_transcoders("lowl0", "/path/to/qwen3-1.7b-transcoders-lowl0")
tc = tset.get(20)                                          # reads L20.n, writes L20.m
cb = components.transcoder_binding(run, "mlp20", f"t{last}.L20.m", tc)
components.component_table(run, cb, Margin(last, a, b))    # features + other + b_dec + error
run.set(f"t{last}.L20.m", cb.edited(remove=("feature[123]",)).reshape(1, 1, -1).to(dtype))
```

A `TranscoderBasis` has the mwhanna layout that SAELens's `mwhanna_transcoder` loader uses: `relu(W_enc n + b_enc)`, no `b_dec` on the input, `W_dec` rows as directions. The active set is chosen in float32 (`W_enc`, `b_enc`, `n` widened exactly) and the activations are recomputed in float64 from the native bits. `transcoder_binding` returns a `ComponentBinding` with `site="mlp-write"`, the same type the value-cell binding and `SubspaceBinding.decompose` return. Its error is the native `m` minus the reconstruction, so the partition closes to float64 roundoff. `component_table`, `edited`, `feature_value` and `features.activations` read it. `to_saelens()` builds SAELens's own `Transcoder`; its active set and decode match (tested). The loader refuses a file with `W_skip` or a JumpReLU `threshold` (not modelled) and reads `config.json` or `config.yaml`. `describe()` carries SAELens `hook_name`/`hook_name_out`, and `node(row, f)` gives circuit-tracer node ids.

## Custody and record sinks (`core/custody.py`)

```python
dbg.custody(script=__file__, prereg="prereg.json", kernel_path="mdb-stepped")
sink = dbg.records_to("out/records.jsonl")   # every record sealed from now on, any kind
sink.records()                                # read back, each verified (v3 and v2 lines)
```

While custody is declared, every sealed record gets `body.custody` = script and prereg sha256 (hashed when declared), library versions of what is imported (mdb, saturn_pub, torch, transformers, interop libraries) and the kernel path. This covers forecast, trace, watch-hits, circuit-map, supplied-forecast, branches, pool, core-file and checkpoint-state records. Undeclared, records are sealed exactly as before (REPL goldens and stored ids unchanged). `branches`, `pool`, `core-file` and `checkpoint-state` records always carry custody. A job payload can declare with no code change: any of `MDB_SCRIPT`, `MDB_PREREG`, `MDB_CUSTODY_KERNEL_PATH` (older name `MDB_KERNEL_PATH`) declares custody (an unreadable file raises on every seal, never once); `MDB_RECORDS_SINK` (older name `MDB_RECORDS`) opens a sink, also when custody was declared in code first. The kernel path and the sink can also come from a config file (`[custody] kernel_path`, `[records] sink`), and `mdb engine` has `--records`, `--kernel-path` and `--prereg` ([CONFIG.md](CONFIG.md)). When a loaded config file sets a key that reaches a record (dtype, device, model, session, kernel path, SAE sources), every record carries custody with that file's sha256 ([CONFIG.md](CONFIG.md#provenance-what-a-record-says-about-config)).

Sink names. Several shards of one experiment that all open `records-<tag>.jsonl` and are collected into one directory would keep only the last shard's records, so names are unique per process. `dbg.records_to()` with no path writes `./records-<run>.jsonl`; a directory gives `<dir>/records-<run>.jsonl`; `{run}` in a name is replaced. `<run>` is `custody.run_tag()`: `MRUN_JOB_ID` when set, then the process start time and pid, so two shards or two steps of one job never share a file. An explicit file that already holds data and that this process did not open raises `FileExistsError` unless `append=True` (`MDB_RECORDS_APPEND=1` for the environment sink). Rerunning a driver into the same `out/records.jsonl` therefore needs `append=True`, a fresh directory or deleting the old file.

Payloads in the sink. A record too large for one line keeps a sha256 pointer to its full payload (`bounded.put_json`). With no project store the pointer is `state: "absent"` (a compact `Branches.seal` record's `arms_payload`, a pool's values, a trace). While a sink is open, the payload's bytes are also written to the sink as one `{"mdb_payload": "mdb-payload-v1", "sha256", "bytes", "raw"}` line, so per-arm probe values are sealed in the file. `sink.load_json(record["body"]["arms_payload"])` reads it back sha256-checked; `sink.records()` skips payload lines; `records_to(..., payloads=False)` turns them off. A sink never changes a record or its id.

Execution lines. A v3 record keeps the session's execution profile by sha256, so a sink writes the profile once, as one `{"mdb_execution": "mdb-execution-v1", "sha256", "execution"}` line before the first record that needs it; `sink.records()` uses those lines to return whole records, and `sink.executions()` returns them. To read a single line yourself use `records.load_record(line, execution=...)`; a plain `json.loads` gives the stored form. Format and migration of older sink files: [RECORDS.md](RECORDS.md).

## Small helpers

- `verdict.margin_floor(logits, a, b, spacings=2, extra=0)` is the margin noise floor on `verdict.spacing` (see Two floors in the [README](../README.md#two-floors)). `verdict.floors(effect, floor)` is the effect-in-floors column.
- `circuit_map.group_writer(adapter, layer, kv=None)` is `group_writers` of the layer's `o_proj`, computed from the weight as it is on every call (no cache: `.data` edits are invisible to version counters).
- `sets.roles_from_text(tokenizer, text, {"hint": "the answer is", "q": (start, end), ...})` returns token rows per role from offset mappings. It replaces the four CoT `spans_for` copies.
- The direct-effect vector the drivers recomputed as `rho_F gamma (W_U[a] - W_U[b])` is `reads.readout(run, Margin(row, a, b)).vector` (with `.rho`, `.bias`, `.native`).

## Not built here

Not built: `each` from Python over these objects, feature lifetime strips, sealed suspect lists and verdict statistics. A checkpoint re-arms active holds from the donor values snapshotted at save time, not from a live source run. There is no REPL verb to start from a checkpoint (`run_from` is Python). Holds on the final norm act through `final_held_logits` only, not as a run-wide switch.

## Multi-site feature watch (`analysis/features.py`, `engine/breakpoints.py`)

A `FeatureContext` is one site (one basis, one layer). `FeatureCondition(text, contexts)` also takes several contexts (a sequence or `{layer: context}`; `features.feature_sites({layer: basis}, row=r, sets=..., constants=...)` builds one per layer): the condition reads, at each stop, the context of that stop's layer, the watch stops only at those layers, keeps one memory per layer, and each hit names its stop (`hit["site"]`, `hit["layer"]`). `table.add_watch("t4.L*.h", program, mode="edge", compiled=cond)` therefore equals one watch `t4.L<l>.h` per layer run one by one (edge, level and change modes), with the same stop reasons (`watch`, `watch+breakpoint`) and reverse walk. Every context must declare the sets, labels and constants the text names. Tested on tiny qwen2 bf16 and qwen3 fp32 with a random `FeatureBasis` per layer (`tests/test_analysis_components.py`).

# mdb

mdb is a debugger for language models, in the spirit of gdb. It runs a loaded Hugging Face
decoder (Qwen2, Qwen3, Llama or Mistral; gemma is not supported, M32) and stops inside each native layer at named values called ports, where
you can read them, edit them, checkpoint and rewind. Every number it shows carries a label that
says how the number was obtained.

The experimental [branch-change analysis](docs/BRANCH-CHANGES-TRIAL.md) explains finite
attention routing/payload and MLP gate/up differences between aligned native runs. Its
first real-model trial connects a nominated neuron restoration to a persisted conditional
forecast. The [proposal](docs/COMPUTATION-EXPLANATIONS-PROPOSAL.md) describes the remaining
producer, consumer and circuit-discovery work.

This directory is the `mdb-runtime` distribution (import name `mdb`). It depends on `saturn-pub`,
whose `Session` runs the model. The older experiment code lives in `../mdb-lanes` and plugs in
through entry points (see [Lanes](#lanes)). Open issues are in [`issues.jsonl`](issues.jsonl)
(ids M1 onward); [Known limits](#known-limits) cites them.

Except for the section on a real model, every command and code block below was run on a random
tiny model on a CPU, offline, and the output shown is the real output. `...` marks a cut.

## Quick start

The REPL on a tiny Qwen2 (bf16, 2 layers, no tokenizer, so the prompt is token ids), fed with
`--script FILE`. `python -m mdb.repl` is the same entry point.

```
$ python -m mdb engine tiny:qwen2 --tokens "1 5 9 3"
(mdb) start 1 5 9 3
t3.L0.h  h = layer input (residual stream)  {P01}
(mdb) until L1.p
t3.L1.p  p = softmax(s, float32).to(dtype) (native attention weights)  {P05 P06 P07 P08 P09}
(mdb) print top=2
t3.L1.p  bfloat16 [1, 2, 1, 4]  (batch, head, row, src)
  norm 0.70607 min 0.24902 max 0.25 mean 0.24963 abs_max 0.25 [measured]
  top |v|: [0,0,0,0] 0.25, [0,0,0,3] 0.25 [measured]
  pointer occurrence ~stepped sha256 a3a32de6269f57e6 cut c1cd8fbec358
(mdb) backtrace
p <- s <- qr,kr <- q,k <- a <- h
#0  t3.L1.p  p = softmax(s, float32).to(dtype) (native attention weights)  {P05 P06 P07 P08 P09}  <- s
...  (#1 to #5 cut: s, qr, kr, q, k, a, h)
(mdb) checkpoint pre
checkpoint pre at t3.L1.p (736 bytes)
(mdb) set L1.m = 0 scale
staged at t3.L1.m: m scale 0 [supplied] (not fired yet)  (1 staged)
(mdb) until logits
t3.logits  logits = lm_head(fn)
(mdb) print logits top=3
t3.logits  bfloat16 [1, 32]  (batch, vocab)
  norm 0.39659 min -0.16699 max 0.13477 mean -0.0077629 abs_max 0.16699 [measured, under edit t3.L1.m]
  top |v|: [0,6] -0.16699, [0,7] -0.15234, [0,20] 0.13477 [measured, under edit t3.L1.m]
  pointer occurrence ~stepped sha256 88df038247cb2036 cut ba8ebd146ec3
(mdb) restart pre
t3.L1.p  p = softmax(s, float32).to(dtype) (native attention weights)  {P05 P06 P07 P08 P09}
(mdb) info edits
staged at t3.L1.p: none
```

`start` prefills all tokens but the last; the last token (row 3) is the one you step through.
`start prefill=K` batches only the first K rows and steps the rest one row at a time, so their
ports are stops too (see [Prefill split and position patching](#prefill-split-and-position-patching)).
`mdb engine MODEL [PROMPT]` names what to debug, as `gdb PROGRAM` does. `MODEL` is a checkpoint
directory, the name of a model in the local Hugging Face cache (`mdb engine qwen2.5-1.5b
"Weather: freezing"`; the name lookup was run against a real cache, the model was not loaded
here), `tiny:qwen2`, `tiny:qwen3`, `tiny:llama` or `tiny:mistral` (the same four as `--tiny` of `mdb engine` and `mdb serve`), or the address of a running `mdb serve` (see
[Live and remote sessions](#live-and-remote-sessions-mdb-serve)). Nothing is ever downloaded; a
name that matches no cached model, or several, is refused with the candidates. `--project DIR`
attaches a project, which makes checkpoints durable.

## Coming from TransformerLens or nnsight

mdb uses its own words for things you already know. Left: the usual term. Right: mdb's. Port names
are the Qwen2 ones in the [Ports](#ports) table; `L<L>` is the layer number. "Roughly" means the
tensor has another shape or sits at another position, so do not swap them blindly.

| TransformerLens / nnsight / pyvene | mdb |
|---|---|
| hook point, module hook | port: a named value inside one decoder layer, address `L<L>.<port>` |
| `blocks.L.hook_resid_pre` | `L<L>.h` (layer input) |
| `blocks.L.hook_resid_post` | `L<L>.h_out` (layer output) |
| `blocks.L.hook_resid_mid` | `L<L>.r` (`h + o`) |
| `blocks.L.attn.hook_pattern` | `L<L>.p` (per head, `p[h5]`; native attention weights) |
| `blocks.L.attn.hook_z` | roughly `L<L>.A`: `hook_z` is per head, before `W_O`; `A` is the heads concatenated, at the `o_proj` input (`A[h5]` selects one head's slice) |
| `blocks.L.hook_attn_out` | `L<L>.o` (after `o_proj`) |
| `blocks.L.attn.hook_q` / `hook_k` / `hook_v` | roughly `L<L>.q` / `k` / `v`: mdb's are flat, `[batch, row, head*head_dim]`, `q` and `k` before RoPE |
| `blocks.L.attn.hook_attn_scores` | `L<L>.s` (a replica checked against `p`, see Ports) |
| `blocks.L.mlp.hook_post` | `L<L>.f` (`down_proj` input) |
| `blocks.L.hook_mlp_out` | `L<L>.m` |
| activation cache (`run_with_cache`, `trace`) | reading ports with `print` / `x` at a stop, or `mdb.trace` in Python; one prompt, one row at a time |
| activation patching | `set ADDR = OTHERRUN:ADDR` for ports (`set L0.A[h1] = clean:L0.A[h1]`, see the example) and state cells; `whatif` and `compare` take the same value and measure the edit natively, then restore; at earlier positions after `start prefill=K`; the position x layer sweep is `sweep PORT = OTHERRUN --consumer C` |
| head or path ablation | `set L<L>.A[h5] *= 0` (selector `[h5]`); `whatif` for the effect without leaving the stop |
| direct logit attribution | `explain M[a,b]` (exact split of a margin, with a measured remainder) |
| KV cache | state cells `cache.L<L>.k` / `v[kv0,row=2]`; readable, editable, and takeable from another run |
| a dataset of prompts, `run_with_cache` over a batch | a prompt set: `each SET: OP LINE` runs any verb on every prompt, one run per prompt, with a summary (see [Prompt sets](#prompt-sets)); no batched forward (the engine is batch 1) |
| SAE features, dictionary units | none (issue M23) |

Example: patch one head's output from a clean run into a corrupted run and read a logit
difference. TransformerLens style, not run (patched at the last position: an mdb port edit acts
on the cursor's row; by default the earlier prompt rows are prefilled and live in the cache
cells, and `start prefill=K` makes them stepped rows with ports, see below):

```python
# pseudocode, not run
_, clean_cache = model.run_with_cache(clean_tokens)
def patch(z, hook):              # z: [batch, pos, head, d_head]
    z[:, -1, 1] = clean_cache[hook.name][:, -1, 1]
    return z
patched = model.run_with_hooks(corrupt_tokens, fwd_hooks=[("blocks.0.attn.hook_z", patch)])
diff = patched[0, -1, 6] - patched[0, -1, 7]
```

mdb, on the tiny Qwen2 (clean prompt `1 7 9 3`, corrupted `1 5 9 3`, only row 1 differs; the
logit difference is token 6 minus token 7 at the last row, bf16), real output with the long
lines cut:

```
$ python -m mdb engine tiny:qwen2 --tokens "1 5 9 3" --script FILE
(mdb) start name=corrupt 1 5 9 3
(mdb) checkpoint s0
(mdb) run name=clean 1 7 9 3
(mdb) until logits
(mdb) x logits 2 6
t3.logits  [6:8] of 32: -0.14355 -0.12451 [measured]
(mdb) frame corrupt:
(mdb) until logits
(mdb) x logits 2 6
t3.logits  [6:8] of 32: -0.16699 -0.15332 [measured]
(mdb) restart s0
(mdb) set L0.A[h1] = clean:L0.A[h1]
staged at t3.L0.A[h1]: A replace tensor [1, 1, 16] bfloat16 sha256 f111ca510505 from clean:t3.L0.A[h1] (live) on [h1] (8 elements) (not fired yet)  (1 staged)
  value from clean:t3.L0.A[h1]
  realised as one masked replace on the selected elements
(mdb) until logits
(mdb) x logits 2 6
t3.logits  [6:8] of 32: -0.1709 -0.14844 [measured, under edit t3.L0.A]
(mdb) whatif L1.m = clean:L1.m --consumer M[6,7]
whatif t3.L1.m replace tensor 7b16b2586286 from clean:t3.L1.m -> t3.M[6,7]  (mode rewind, ...)
  sealed  linear ΔM = -0.00021868 [derived-local P14]   exact ΔM = -0.00021847 [derived-exact]  ...
  native  ΔM = -0.00097656 [measured]   (base -0.022461 [measured] -> edited -0.023438 [measured])
  ...
(mdb) set L0.A[h1] = clean:L0.A[h0:2]
error (refused): clean:t3.L0.A[h=0:2] selects shape [1, 1, 16]; t3.L0.A[h1] selects [1, 1, 8]; select the same block on both sides (the same port kind, heads and row length)
```

Logit difference (token 6 minus token 7, from the printed values): clean -0.0190, corrupted
-0.0137, head 1 of L0 patched -0.0225. The patch moves it away from the clean value here; this is
a random tiny model, so the numbers only show the mechanics. The value of another run can name
another row or layer (`set L0.m = clean:t3.L1.m`); a missing row is the cursor's row, as for an
address of this run. Both selections must have the same shape. The edit records where its value
came from (`origin`: run, address, payload sha256, the clean run's cut) in its manifest, the
receipt and `info edits`; fetching it leaves the clean run where it was. With a project attached
the value is stored as a blob with the edited run's recipe, so a pointer to a value read under
the patch still resolves after the clean run is gone; without one such a pointer is refused and
says to attach with `project=`. A whole-residual patch reproduces the clean run bit for bit:
when only the last row differs, `set L1.h = clean:L1.h` gives the clean logits; when an earlier
row differs, add its cache cells (`set cache.L<l>.k|v[row=0:3] = clean:cache.L<l>.k|v[row=0:3]`
for every layer), as `tests/test_ops_port_origin.py` checks.

### Prefill split and position patching

`start prefill=K` (and `run prefill=K`, `Run(adapter, prompt, prefill=K)`) computes prompt rows
`0..K-1` in one batched call and steps rows `K..` one row at a time through the same transitions
a generated row takes, with the prompt's next token forced at `sample` instead of the argmax.
`prefill=0` steps every row; the default is all rows but the last. A stepped prompt row has
every port, so `until t1.L0.m`, `print`, `set tR.L.PORT = clean:tR.L.PORT`, `whatif` and
`compare` work there; on a row that was prefilled they are refused with a message that names
`start ... prefill=K` for a `K` no larger than the row.

The split is part of the procedure (issue M26): a batched and a stepped row can differ in the
last bits of their cached K/V (measured on the tiny fp32 Qwen2: logits differ by 3e-8 between
`prefill=0` and the default; M5 measured it on Qwen2.5-1.5B), so the checkpoint, the replay
recipe and every restore carry `prefill`; `where` and `info runs` show it when it is not the
default. A recipe written before the field existed reads as the default split.

`sweep PORT [rows=R0:R1] [layers=L0:L1] = ORIGIN --consumer C [top=K]` patches `PORT` from the
run `ORIGIN` at each (row, layer), runs natively to the consumer (a margin or one logit),
reads it and moves on; the session is restored once at the end and checked unchanged. The
result is one record: the layers x rows table of deltas (`measured`), the top cells by |Δ|,
and the native call counts. A table too large for the record is stored as a sha256-checked
blob with the top cells kept inline. Cells are visited from the latest layer entry back, each
from a trail mark below it, so no cell replays from row 0: a cell `(r, l)` with the consumer
in row `c` of an `L`-layer model costs `(L - l) + L(c - r)` layer calls, and the final restore
`(L - l0) + L(c - r0)`. In a prompt set, `each SET: sweep h rows=@last = @pair:h --consumer
M[$answer,$foil]` takes the origin from each item's pair; a pair of another length is a
per-item refusal.

Tiny Qwen2 (bf16), clean `1 7 9 3 6 2`, corrupted `1 5 9 3 6 2` (row 1 differs), both stepped
from row 0, real output (`tests/repl_golden/prompt_rows.mdb`):

```
(mdb) start name=clean prefill=0 1 7 9 3 6 2
(mdb) until t5.logits
(mdb) start prefill=0 1 5 9 3 6 2
prefill=0: no row batched, rows 0..5 stepped (tokens forced)
t0.L0.h  h = layer input (residual stream)  {P01}
(mdb) until t2.L1.h
(mdb) print t1.L0.m
t1.L0.m  bfloat16 [1, 1, 16]  (batch, row, d_model)
  norm 0.0017112 min -0.0005455 max 0.0007515 mean 0.00010715 abs_max 0.0007515 [measured]
(mdb) until t5.logits
(mdb) sweep h = clean --consumer t5.M[4,9]
sweep h = clean:h -> t5.M[4,9]  (12 cells; base -0.038208 [measured])
  prefill=0: no row batched, rows 0..5 stepped (tokens forced)
     t0 t1 t2 t3 t4 t5
L0    0 -8  0  0  0  0
L1    0 -9 -2 -1 -1 +5
cell: tenths of max |Δ| 0.014893 [measured] (9 caps); '.' < 1/20, '0' exact
  top |Δ|: t1.L1 -0.014893 [measured], t1.L0 -0.011353 [measured], t5.L1 0.0081177 [measured], ...
  cost: 90 layer calls, 35 probes, 201 other transitions [measured] (wall time in the record)
  session unchanged at t5.logits
```

Row 0 is the same token in both runs, so its cells are exactly 0; at `L0` only the differing
row 1 moves the margin (the other rows' layer inputs are the same embeddings). `whatif t1.L1.h
= clean:t1.L1.h --consumer t5.M[4,9]` measures the same -0.014893 natively, with an exact
prediction across rows. Patching the whole residual `tR.L2.h` from clean at every stepped row
of one layer gives the clean logits bit for bit when both runs use the same split
(`tests/test_engine_prefill.py`). Memory (measured, `Run.held_bytes()` at the last logits of a
12-layer tiny Qwen2, 24-token prompt): 31480 bytes in 407 trail marks with every row stepped
against 19136 bytes in 16 marks with the default split; a sweep leaves it unchanged. Time
(measured, same model, laptop CPU): a sweep of 4 rows x 12 layers made 1224 layer calls in
8.0 s; one row x 12 layers 20 rows before the consumer made 3210 in 18.8 s, so the cost is the
rows between the cell and the consumer.

mdb adds, and the others do not: a prediction sealed before the native measurement with the gap
split by cause (`forecast`, `verify`), an authority label and state on every number, and
reversible stepping inside a layer (`reverse-step`, `restart`). It lacks a batched forward
(prompt sets run one prompt at a time), model families beyond Qwen2 and Qwen3, and dictionary
features (M23).

## Embedding

```python
import mdb

dbg = mdb.open_tiny("qwen2")                 # or mdb.attach(model, tokenizer)
run = dbg.run([1, 5, 9, 3])                  # prefill 3 tokens; cursor at t3.L0.h
run.until("L1.p")                            # stop inside layer 1 at the attention weights
saved = run.checkpoint("pre")                # about 1 KB of plain data
run.set("L1.m", 0.0, "scale")                # stage an edit further down the same layer
print(run.until("logits"), run.read()[0, :3], run.read_info()["upstream_edits"])
print(run.restore(saved), run.edits)         # back at t3.L1.p, nothing staged
```

```
t3.logits tensor([-0.0233,  0.0430,  0.0220], dtype=torch.bfloat16) [{'row': 3, 'layer': 1, 'port': 'm', 'op': 'scale', 'staged': False}]
t3.L1.p []
```

`mdb.attach(model, tokenizer)` takes a loaded model; it refuses a model without eager attention
or with modules in training mode, and never changes the model. `dbg.repl(run)` opens the REPL
on the same run. Every REPL verb is also `dispatch(ctx, verb, args)` with JSON in and out.

Loader rule (`mdb.engine.models`): `load_local` passes the dtype to `from_pretrained` and moves
the module by device only; `cast_weights` (used by `tiny_model`) casts parameters, never buffers.
`module.to(dtype)` would round RoPE's float32 `inv_freq` to bf16, a different model from the one
Hugging Face runs (Qwen2.5-1.5B: one test margin moved a bf16 step, commit `2b6acc5`).

## Addresses

One string names a value everywhere: on the command line, in records and in the web viewer.

```
[origin:] t<row>[.s<step>].L<layer>.<port>[selectors][~program]  |  cache.L<layer>.k|v[selectors]
```

Selector keys are `h kv i src row d`; `a:b` is a half-open range and `a|b` a set. The stops of
one token row, in order: `start`, `emb`, `L<i>.<port>` for every layer and port, `fn`,
`logits`, `sample`. Five addresses:

| Address | Meaning |
|---|---|
| `L1.p` | port `p` of layer 1, in the cursor's row |
| `t3.L0.p[h1]` | head 1 of layer 0's attention weights, token row 3 |
| `cache.L0.k[kv0,row=2]` | key-cache cell of KV head 0 written by row 2, read now (`x cache.L0.k[kv0,row=2] 4`) |
| `logits` | the logits stop of the cursor's row |
| `L*.h_out` | a pattern (breakpoints, watches): the output of every layer |

An occurrence (`t2.L0.k`, the value written) and a cache cell (`cache.L0.k[row=2]`, read at
the current clock) are different types. An origin names another run: a live named run, a
checkpoint or a stored session (`boiling:cache.L0.v[kv0,row=1]`, `boiling:t3.L1.kr[kv0]`).

## State cells and other runs

`set cache.L<l>.k|v[selectors] = VALUE [replace|add|scale|mix DOSE]` stages a write into the KV
cache; `+= -= *= /=` are shorthand for `add` and `scale`. Rows must already be in the cache; the
row the pending layer writes is refused until the layer commits. The cell is realised before its
reader (`kr` for keys, `A` for values) unless the cursor has passed that boundary (`kr`, `p`),
in which case it lands after the read; the reply says which. `mix DOSE` is
`cell + DOSE * (value - cell)` in float64 from the native bits, rounded once to the cell dtype
(`mix 1` is the exact copy, `mix 0` leaves the cell).
`run name=boiling 1 7 9 3` starts a second run (`start` takes `name=` too); `frame boiling:` and
`frame main:` switch; `info runs` lists them. `ORIGIN:ADDR` takes a value from another run, a
checkpoint or a stored session of the same model and numeric scope: into a cell, as `v` or `kr`
occurrences or cache cells (`k`/`kn` occurrences are refused, before RoPE); into a port, as any
occurrence of the same selected shape (the activation patching example above). `seek b:t3.L1.p`
makes run `b` current, as `frame b:` does, and moves it there (a run name needs the full
occurrence, `b:t3.L1.p`, not `b:L1.p`); a target without a row (`seek L20.p`) is on the cursor's row.

```
(mdb) info runs
* main      t3.L0.h  4 tokens, staged 0
  boiling   t3.logits  4 tokens, staged 0
(mdb) set cache.L0.v[kv0,row=1] = boiling:cache.L0.v[kv0,row=1] mix 0.5
staged cell cache.L0.v[kv0,row=1]: cache.L0.v mix 0.5 [supplied] tensor [1, 1, 1, 8] bfloat16 ...
  realised before L0.A reads it (boundary L0.p); reads already made are not recomputed
  value from boiling:cache.L0.v[kv0,row=1]
```

`compare A_ADDR = V B_ADDR = V --consumer M[a,b]` and `whatif ADDR = V [mix D] --consumer ...`
take cell edits too (`tests/repl_golden/cells_cross_run.out`). Cell edits go into checkpoints,
restore, reverse and replay. Embedding: `run.set_cell(layer, "v", value, rows=[1], kv=[0])`;
`dbg.cell_value("boiling:cache.L0.v[kv0,row=1]", run=run)`. Reads of committed layers with no
edits come from one cached full-layer probe (64 layers kept), not a rebuild per port.

## Ports

One line per port, in native execution order, generated from the port program
(`mdb.engine.ports.program_for`); `info ports` and `list` print the same. The tables here are kept by hand. Proof IDs refer to
`obsidian/model-math/` (P01-P33). On Qwen3, `qr` and `kr` take `qn` and `kn`. Llama and Mistral have the Qwen2 list (`info program` on `tiny:llama`: `h a q k v qr kr s p A o r n g sg u f m h_out`); a projection without bias contributes no bias term, and a config outside a family's premises (a non-silu activation, Llama `mlp_bias`) is refused by `attach`, `trace` and `program_for`. The families come from one table, `mdb.engine.ports.FAMILIES`.
| Port | Read at | Operator | Proofs |
|---|---|---|---|
| `h` | decoder layer input | h = layer input (residual stream) | P01 |
| `a` | `input_layernorm` output | a = RMSNorm(h) * w_in | P02 P03 |
| `q` | `self_attn.q_proj` output | q = W_q a + b_q (pre-RoPE, flat) | P04 |
| `qn` (Qwen3) | `self_attn.q_norm` output | qn = RMSNorm_head(q) * w_q (per head, pre-RoPE) | P02 P04 |
| `k` | `self_attn.k_proj` output | k = W_k a + b_k (pre-RoPE, flat) | P04 |
| `kn` (Qwen3) | `self_attn.k_norm` output | kn = RMSNorm_head(k) * w_k (per kv head, pre-RoPE) | P02 P04 |
| `v` | `self_attn.v_proj` output | v = W_v a + b_v (flat) | P04 |
| `qr` | eager attention argument | qr = RoPE(q) (query argument of eager attention) | P04 |
| `kr` | eager attention argument | kr = [cache; RoPE(k)] (key argument of eager attention, cached rows included) | P04 |
| `s` | replica from `qr`, `kr` and the mask | s = scaling * qr repeat_kv(kr)^T + mask (replica of the native expression) | P05 P07 P08 P09 |
| `p` | eager attention weights | p = softmax(s, float32).to(dtype) (native attention weights) | P05 P06 P07 P08 P09 |
| `A` | `self_attn.o_proj` input | A = concat_heads(p repeat_kv([cache; v])) (o_proj input) | P06 P07 |
| `o` | `self_attn.o_proj` output | o = W_o A | P06 P07 |
| `r` | `post_attention_layernorm` input | r = h + o | P06 P07 P14 |
| `n` | `mlp` input | n = RMSNorm(r) * w_post (mlp input) | P02 P03 |
| `g` | `mlp.gate_proj` output | g = W_gate n | P10 P15 |
| `sg` | `mlp.act_fn` output | sg = silu(g) | P10 P15 |
| `u` | `mlp.up_proj` output | u = W_up n | P10 P15 |
| `f` | `mlp.down_proj` input | f = sg * u (down_proj input) | P10 P15 |
| `m` | `mlp.down_proj` output | m = W_down f | P14 |
| `h_out` | decoder layer output | h_out = r + m (layer output) | P14 |

`kr` is the only port that cannot be edited (it holds cached rows). `s` is not a native
intermediate: it is a replica of the native expression, checked against `p` bit for bit.

## Verbs

Taken from the op registry (`mdb.ops.registry()`), by hand; `info` or the registry is the
authority if they differ. The REPL, `--json` mode, the server's `POST /api/v1/s/SID/op` and
`dispatch` all go through it.
| Verb | Usage | Effect | What it does |
|---|---|---|---|
| `abort` | `abort` | mutate | Drop the staged edits of the pending layer (and the staged cell edits) |
| `advance` | `advance ADDR [budget=ROWS]` | navigate | Run to an address, stopping earlier at an armed breakpoint or watch |
| `backtrace` | `backtrace [ADDR]` | read | Producer chain from the focus port back to the layer input |
| `break` | `break PATTERN [if CONDITION]` | mutate | Set a breakpoint on an address pattern (L*.p, t5.L1.o, logits, cache.L0.v[row=2]); cache cells are evaluated at boundary stops only |
| `checkpoint` | `checkpoint [NAME]` | mutate | Name the cursor position (about 1 KB of plain data) |
| `checkpoints` | `checkpoints [offset=N] [limit=N]` | read | List named checkpoints (paged: offset=, limit=) |
| `circuit` | `circuit CONSUMER [top=K] [depth=D] \| circuit show ID [node=NODEID]` | mutate | Circuit records: build from the exact decomposition (nodes, edges, remainder), show as a walk (`node=` opens one stored page of a circuit that did not fit one record); `circuit test ID` is disabled and refuses (issue M4) |
| `commit` | `commit` | mutate | Commit the pending layer with its staged edits (cursor to its h_out) |
| `compare` | `compare A B [consumer=C]  (A, B: ADDR=VALUE [op] or a checkpoint name)` | navigate | Compare branches: base, A, B and A+B measured natively, effects and the interaction |
| `condition` | `condition N [EXPR]` | mutate | Set the condition of breakpoint N (a watch's predicate); no expression removes it |
| `continue` | `continue [N] [budget=ROWS]` | navigate | Run to the next armed breakpoint or watch hit (N: ignore the breakpoint stopped at N-1 more times, as gdb) |
| `delete` | `delete [ID ...]` | mutate | Delete breakpoints and watches (all when no ids are given) |
| `disable` | `disable [ID ...]` | mutate | Disable breakpoints and watches without deleting them (all when no ids) |
| `display` | `display [ADDR]` | mutate | Show an address after every stop (no address: list displays) |
| `down` | `down [N]` | navigate | Move the focus back toward its consumer, N times |
| `each` | `each SET [at=ADDR] [keep]: OP LINE [; OP LINE ...]` | native | Run one op line on every prompt of a set (one run per prompt): one result per prompt and a summary (n, mean, min, max, sign agreement, verdict counts) |
| `enable` | `enable [ID ...]` | mutate | Enable breakpoints and watches (all when no ids are given) |
| `explain` | `explain M[a,b]\|ADDR` | read | Exact accounting of a margin M[a,b] or of one port, with proof IDs and the measured remainder (cursor at the row's logits/sample) |
| `export` | `export` | not built yet | Export a walkthrough bundle of records and pointers |
| `finish` | `finish` | navigate | From a port inside a layer: run to that layer's h_out. From h_out or a token-level stop: run to this row's sample |
| `forecast` | `forecast [ADDR\|M[a,b]]` | mutate | Seal a prediction of the next stop (or a later stop of this row, or a margin M[a,b]) from values at the cursor, before native code runs |
| `frame` | `frame [ADDR \| cursor \| NAME:]` | navigate | Show or set the focus address (print/list/backtrace default to it); NAME: switches to the named run |
| `ignore` | `ignore N COUNT` | mutate | Ignore the next COUNT hits of breakpoint N (they are counted, not reported) |
| `info` | `info ports\|breakpoints\|watch\|display\|checkpoints\|edits\|program\|runs [offset=N] [limit=N]` | read | Show debugger tables or the program (lists are paged: offset=, limit=) |
| `list` | `list [LAYER]` | read | The layer's port program: operator text, proof IDs, cursor marked |
| `next` | `next [N]` | navigate | Step to the last port of the sublayer (r, then h_out) |
| `print` | `print [ADDR] [top=K] [slice=N]` | read | Reduce a value: shape, dtype, stats, top-k by \|value\|, pointer; `print M[a,b]` (also `x`, `display`) is the measured logit difference, refused before the logits stop |
| `promptset` | `set load NAME FILE \| set define NAME ITEMS \| set show NAME \| promptset list \| promptset result ID [ITEM]` | mutate | Prompt sets: load a JSONL file (not from the web), define a set inline, show or list sets, read a stored each result (the REPL's set load / set show) |
| `reads` | `reads SOURCE [rows=R\|A:B\|all] [consumer=C] [top=K] [causal=false]` | navigate | Who reads a value: attention reads of a source row with mass, exact contribution and a native causal verdict |
| `restart` | `restart NAME` | navigate | Return to a named checkpoint (rewinds or replays) |
| `reverse-continue` | `reverse-continue` | navigate | Run backwards to the previous breakpoint or watch hit |
| `reverse-finish` | `reverse-finish` | navigate | Step back to the boundary before the pending layer (its input) |
| `reverse-next` | `reverse-next [N]` | navigate | Step back to the previous sublayer end |
| `reverse-step` | `reverse-step [N] [unit] [budget=ROWS]` | navigate | Step back one port (or N units; layer/token units enter at most `budget` rows) |
| `run` | `run [name=NAME] [prefill=K] [TEXT \| ID ID ...] [budget=N]` | native | Like start, then continue to the first armed breakpoint or watch |
| `seek` | `seek ADDR \| seek ORIGIN:ADDR \| seek POINTER-JSON` | navigate | Move the session to an occurrence: forward, backward, or by replaying a stored recipe (ORIGIN:ADDR or an occurrence pointer) |
| `set` | `set ADDR = VALUE\|ADDR\|ORIGIN:ADDR [replace\|add\|scale\|mix DOSE]` | mutate | Stage an edit at a port of the pending layer, or of a state cell (cache.L<i>.k\|v[kv,row,d]): set ADDR = VALUE [op] |
| `start` | `start [name=NAME] [prefill=K] [TEXT \| ID ID ...]` | native | Prefill a prompt and stop at the first stepped row's first port (L0.h) |
| `step` | `step [N] [port\|sublayer\|layer\|token] [budget=ROWS]` | navigate | Step one port (or N units; layer/token units enter at most `budget` rows) |
| `sweep` | `sweep PORT [rows=R0:R1] [layers=L0:L1] = ORIGIN --consumer C [top=K]` | navigate | Patch PORT from another run at every (row, layer) of a range, measure the consumer natively for each, restore: one table of deltas (activation patching) |
| `symbol` | `symbol NAME = ADDR[,ADDR...]` | mutate | Declare a user symbol over cache rows: symbol NAME = ADDR[,ADDR...] |
| `symbols` | `symbols [consumer=C] [rows=R\|A:B\|all] [min_mass=X] [offset=N] [limit=N]` | mutate | Symbol table at the cursor: bindings with definition sites, lifetimes, readers, status and an Unbound row (stored as records with a project) |
| `tbreak` | `tbreak PATTERN [if CONDITION]` | mutate | Set a breakpoint that deletes itself after its first hit |
| `undisplay` | `undisplay [ID ...]` | mutate | Remove displays (all when no ids are given) |
| `until` | `until ADDR [budget=ROWS]` | navigate | Run to an address, stopping earlier at an armed breakpoint or watch |
| `up` | `up [N] [PORT]` | navigate | Move the focus to a producer (toward the layer input), N times; `up PORT` picks the producer by name, the result lists the alternatives |
| `verify` | `verify [ID ...]` | mutate | Compare open forecasts whose target the cursor has passed with the native values: native against the emulated prediction in dtype spacings, `confirmed` / `not confirmed` / `unresolved`, the rounding effect against exact, and the order proof (the op's own help text still describes the removed 3 x sqrt(n) floor, see Known limits) |
| `watch` | `watch ADDR [change\|level\|edge] [if PREDICATE]` | mutate | Watch an address: stop when it changes (or a predicate holds: level/edge); cache cells (cache.L<i>.k\|v) are evaluated at boundary stops only |
| `whatif` | `whatif ADDR=EXPR [replace\|add\|scale\|mix DOSE] consumer=M[a,b]\|logits[i]\|ADDR` | navigate | Predicted, then natively measured change of a consumer under one edit; the session is left exactly where it was |
| `where` | `where` | read | Show the cursor, its operator and the context tokens |
| `x` | `x [ADDR] [COUNT] [OFFSET]` | read | Examine raw values: a bounded slice of the selected, flattened value |

## Authority labels

Every number shown or stored carries exactly one label, and a state.

| Label | Meaning | Example |
|---|---|---|
| `measured` | read from native execution | `print top=2` at `t3.L1.p` |
| `derived-exact` | an exact identity on captured operands | the replica `s`; a value written by an `add` or `scale` edit |
| `derived-local` | a linearisation or quadrature only | the linear term of `whatif` |
| `bound` | a theorem with stated premises | the rounding bound of a `forecast`; never decides a verdict |
| `supplied` | a value handed in by a user or a stored record | the edit request `m scale 0` |

State is `resolved`, `unresolved` (could not be computed: never shown as zero) or `refused`
(a premise failed: the label is empty and a reason is given). A value downstream of an edit
stays `measured` and lists the edits above it (`[measured, under edit t3.L1.m]`).

## Passive capture: `mdb.trace`

`attach` drives the model. `trace` only watches a forward your experiment already runs (any
batch size, sdpa, `generate`, gradients on), with hooks at the stepper's sites, so a port name
means the same in both.

```python
import torch, mdb

model = mdb.open_tiny("qwen2").model         # any loaded Qwen2/Qwen3/Llama/Mistral causal LM
batch = torch.tensor([[1, 5, 9, 3], [2, 6, 10, 4]])
with mdb.trace(model, ports=["L1.p", "L1.s", "L1.m", "L0.h"]) as t, torch.no_grad():
    model(batch)                             # the experiment's own forward
print(t.calls, tuple(t["L1.m"].shape), tuple(t["L1.p"].shape))
print({k: t.info("L1.s")[k] for k in ("state", "authority", "kind", "replica_equal")})
```

```
1 (2, 4, 16) (2, 2, 4, 4)
{'state': 'resolved', 'authority': 'derived-exact', 'kind': 'replica', 'replica_equal': True}
```

- Passive: logits are bitwise equal with and without the trace (tested for qwen2 and qwen3,
  bf16 and fp32, eager and sdpa). On the stepper's native call every port is bitwise equal to
  `Run.read` (tested).
- One call is one outermost forward of `model.model`; with several use `t["L1.m", i]`. Values
  keep the batch dimension; the axes are in `t.info(address)["axes"]`.
- Refused, never wrong: `qr`, `kr`, `s`, `p` without eager attention (the reason is in
  `t.info(address)["reason"]`) or under attention dropout; ports of a layer that did not run;
  anything after the `max_bytes` cap (default 1 GiB).
- Hooks are removed when the block exits, also on error. A second active trace is refused.
  `project=` is not supported yet.

## Predictions: `forecast`, `verify`, `whatif`

`forecast` seals two predictions from values at the cursor, before native code runs, and
`verify` compares them with what native code then produced
(`tests/repl_golden/predict_forecast.out`):

- `exact`: the float64 chain of the port program's identities (`derived-exact`, real
  arithmetic on the captured operands), with its per-element worst-case rounding `bound`;
- `emulated`: the same chain with the run's dtype rounding at every point where eager HF
  rounds, each step run as eager HF writes it with the backend's own kernels at the native
  call's batch-1 shapes (`derived-exact` under the premise "rounding points of eager HF
  attention/MLP on this backend"; the premise list is in the record, `emulated.premises`).

Both sha256s are sealed (`digests`). The verdict is decided on native - emulated: `confirmed`
when every element is within the floor, `1 x spacing` (`supplied`), `spacing` being the dtype's
gap at the consumer's magnitude (a margin, a difference of two rounded logits: the sum of their
two spacings). The floor does not widen with depth. Measured: emulated equals native bit for bit
at every stop from the sealing stop to the logits on tiny Qwen2/Qwen3/Llama/Mistral (BF16 and
FP32, 2 to 28 layers) and on Qwen2.5-1.5B (CUDA, BF16 28 layers and an FP32 2-layer cut, 3
prompts). Above the floor `verify` says where emulation and native part: `not confirmed` when the
prediction is not the emulated chain of the sealed operands (an operand or the held prediction
differs from its sealed digest) or when a stage from that stop to the target does not reproduce
its native value from its native operands within the worst-case bound of its identity;
otherwise `unresolved` (rounding drift between stages, each stage being the native step), with
the parting stop and any native top-score tie the emulated softmax splits differently.
`rounding_effect` is native - exact (`measured`), and the sealed `rounding_origin` names the
first stop where exact and emulated part by more than a spacing; on Qwen2.5-1.5B BF16 that is
an L0 score (`t22.L0.s[h3]: score spacing 64 at |s| 1.51e+04`), and the exact chain is 35 to 78
spacings off at L0.m because rounded scores tie and the native softmax is not the softmax of the
real scores (M42). The worst-case bound against `exact` is printed only when finite
(information):

```
(mdb) forecast L0.p
forecast 1  t3.L0.p  sealed at t3.L0.h  record 96acb5fd3797bace
  method 7 identities t3.L0.a .. t3.L0.p (last softmax P05); chain not stored (no project)
  predicted max_abs 0.25152 [derived-exact] norm 0.70712 [derived-exact]  sha256 a8cee3618668d88c  bound 0.0026798 [bound] per element
(mdb) forecast M[4,9]
forecast 2  t3.M[4,9]  sealed at t3.L0.h  record 6d61cae9d3fa30fb
  method 39 identities t3.L0.a .. t3.logits (last readout_logits P19); chain not stored (no project)
  predicted -0.11452 [derived-exact P19]  bound 16.825 [bound]
(mdb) until t3.logits
t3.logits  logits = lm_head(fn)
(mdb) verify
forecast 1 t3.L0.p: confirmed  native max_abs 0.25195 [measured] norm 0.70781 [measured]  residual vs emulated 0 [measured] = 0 spacings <= floor 0.0019531 [supplied P20] = 1 bfloat16 spacing (supplied)  [P20, resolved]
  rounding effect native - exact 0.00091062 [measured] = 0.466 spacings; worst-case bound 0.0026798 [bound] (holds, information)
  rounding origin (sealed): exact and emulated agree within one spacing at every stop of the chain
  order: sealed at t3.L0.h (stop 2) before the target (stop 10); cut descends through 4 receipts
forecast 2 t3.M[4,9]: confirmed  native -0.11401 [measured]  residual vs emulated 0 [measured] = 0 spacings <= floor 0.00073242 [supplied P20] = 1 bfloat16 spacing (supplied)  [P20, resolved]
  rounding effect native - exact 0.00050802 [measured] = 0.694 spacings; worst-case bound 16.825 [bound] (holds, information)
  rounding origin (sealed): t3.L0.f: exact and emulated part by 1.83 spacings at |x| 0.00723
  order: sealed at t3.L0.h (stop 2) before the target (stop 41); cut descends through 4 receipts
```

`whatif` seals the predicted change of a consumer under one edit, measures it natively on a
private branch, and restores the session (`tests/repl_golden/predict_whatif.out`):

```
(mdb) whatif L1.m = 0.5 add --consumer M[4,9]
whatif t3.L1.m add 0.5 [supplied] -> t3.M[4,9]  (mode pending, record 6b04afb367ee5bf5)
  sealed  linear ΔM = 1.4572 [derived-local P14]   exact ΔM = 0.16513 [derived-exact]  bound 0.07029 [bound]  (operands forecast from t3.L1.a)
  native  ΔM = 0.16449 [measured]   (base -0.11401 [measured] -> edited 0.050476 [measured])
  error split (P20): linear 1.4572 [derived-local P14] + nonlinear -1.2921 [derived-exact P20] + operands 4.2358e-05 [derived-exact P20] + realisation -4.5182e-05 [derived-exact P20] + rounding -0.00063376 [measured P20] (within bound 0.01905 [bound])
  verdict (P20): effect  native ΔM 0.16449 [measured P20], exact ΔM (written value, measured operands) 0.16512 [derived-exact P20], residual -0.00063376 [measured P20]; floor 0.00097656 [measured P20] = max(|residual|, 2 x spacing 0.00048828 [measured P20])
  session unchanged at t3.L1.a  (5 moves, restored)
```

The exact change is `F(x1) - F(x0)` (P09 section 1): the base and the edited chains hold the
edited port at its value `x0` or write the edit there, and recompute the same stops (those the
edit reaches) in float64 from the same operands; the base is never the native value, which
would put the base run's rounding into the exact change (with a no-op edit `L1.m *= 1` the
exact change is 0, at any cursor). The sealed value uses the operands at the cursor. When the
consumer is ahead of the cursor, the stops past it exist only as the chain's own float64 values,
and the sealed line says `(operands forecast from t3.L1.a)` (`prediction.exact.operands` in the
result; the label stays `derived-exact`: an exact identity on the operands captured at the
cursor, not a linearisation). After the native run the chain is evaluated again at the consumer
on the measured operands of the unedited run (`exact` in the result, `requested` and `written`;
computed after the seal and never shown as sealed); the split gains `operands` (that value minus
the sealed one) and `realisation` and `rounding` are taken on the measured operands. The verdict
line says which exact change it holds native against: the written value on measured operands.
That number, the residual and the verdict are the same wherever the cursor was (tested at three
cursors for a port, a head-selected and a cross-run edit, to `spacing / 2^20`).

### Verdicts (P20)

`whatif`, `reads` and `compare` judge an effect from the P20 accounting
(`mdb.analysis.verdict`): native change = exact change (float64 chain on the measured operands,
written value, `derived-exact`) + residual (`measured`). The floor is
`max(|residual|, 2 x spacing)`, spacing being the gap between
adjacent values of the consumer's dtype at its magnitude. A change is an `effect` when the
native change is above the floor and the exact change has its sign and exceeds the residual;
`no effect above the noise floor` when both are within the floor; otherwise `unresolved`, with
the numbers. With no exact chain (as in `compare`) the native change is held against `2 x
spacing` alone and the line says so. Worst-case rounding bounds are still shown where they are
finite (`bound` label) and never decide a verdict. `verify` holds native against the emulated
prediction (above): `confirmed` within one spacing, else `not confirmed` or `unresolved` by where
emulation and native part, whatever the bound (`no-bound` is no longer a verdict).

## Following a value: `reads`, `symbols`, `circuit`, `compare`

`reads ADDR` lists who reads a source row: attention mass beside the exact contribution of
that read to a consumer, and a causal verdict measured by removing the read natively.
`symbols` prints the symbol table as bindings (where a value is defined, how long it lives,
who reads it), always with an `Unbound` row so the table sums to the whole. Cursor at
`t3.logits`; output cut at `...`:

```
(mdb) reads t1 --consumer M[4,9] top=3
reads t1 at t3.logits  rows t3  consumer t3.M[4,9]  (4 reads)
  t3.L1.h0   mass 0.24902 [measured]  effect 0.011455 [derived-exact P19]
    causal: Δ -0.015137 [measured], predicted -0.01456 [derived-exact], residual -0.00057636 [measured P20]; floor 0.00097656 [measured P20]
...
(mdb) symbol ctx = t1, cache.L1.v[row=2]
symbol ctx = t1, cache.L1.v[row=2]  (declared: ctx)
(mdb) symbols consumer=M[4,9]
symbols at t3.logits  consumer t3.M[4,9] = -0.11401 [measured]  min_mass 0.1 [supplied]
  ...  (two legend lines cut)
  ctx         read   mass 1.4971 [derived-exact] over 4 reads (mean 0.37427 [derived-exact])  to_consumer 0.0020898 [derived-exact P19]  life t3:## (t3.L0..t3.L1)
    defined t1, cache.L1.v[row=2]
    readers t3.L1.h0 mass 0.49805 [derived-exact] effect 0.013241 [derived-exact P19]; t3.L0.h1 mass 0.25 effect -0.0081352 [derived-exact P19]; ...
...
  Unbound     mass 0 [derived-exact]  readout -0.14463 [measured]
```

Mass is summed over the heads and layers that read the binding, so it can exceed 1. Causal
verdicts follow the rule in [Verdicts](#verdicts-p20).

`circuit CONSUMER` builds a record of exact contributions that closes on the consumer;
whatever is not listed stays in the record as `+N more` and `remainder` nodes; a subtree that was
dropped without expanding is recorded as not examined, and one that does not fit one record is a
`paged` pointer to a sha256-checked child page (`circuit show ID node=NODEID`). `circuit test` is
disabled (M4): it tested sufficiency by zero-ablating every node outside the circuit, which on the
real model is 26,924 nodes, moves the margin by 27.75 and always says "not sufficient"; the op
refuses with `circuit test is disabled: zero-ablating the complement says nothing on a real model
(issue M4)`. `compare` measures two edits separately and together:

```
(mdb) circuit M[4,9] top=2 depth=3
circuit 1  t3.M[4,9]  top 2 depth 3  (16 nodes, record 0b1ad30b3f694bac)
  t3.M[4,9]  -0.11401 [measured]
    embedding  t3.L0.h  -0.14578 [derived-exact P19]
    L1.attn  t3.L1.o  0.03064 [derived-exact P19]
...
    +3 more  t3.M[4,9] (unlisted)  0.00064862 [derived-exact]
    remainder  t3.M[4,9]  0.00047376 [measured] within 0.016137 [bound]
  closure 1.3878e-17 [derived-exact P19] within 2.0716e-15 [bound] (zero by construction: float64 summation only)
(mdb) compare L1.m=0.5 add L0.v = 0.25 add --consumer M[4,9]
compare A: L1.m=0.5 add  B: L0.v=0.25 add  -> t3.M[4,9]
  base -0.11401 [measured]  A 0.050476 [measured]  B -0.068237 [measured]  A+B 0.052124 [measured]
  effects A 0.16449 [measured]  B 0.045776 [measured]  A+B 0.16614 [measured]
  verdicts (P20 floor = 2 x dtype spacing; native only, no exact chain): A effect (floor 0.00097656 [measured P20])  B effect (floor 0.00097656 [measured P20])  A+B effect (floor 0.00097656 [measured P20])
  interaction (A+B) - A - B + base = -0.044128 [measured]
  order: A+B stages both sides' edits layer by layer; at one port A's edit is applied before B's (where both edit one port, the ops commute)
  session unchanged at t3.logits
```

`seek ADDR` moves the session to an occurrence: forward, backward, or by replaying a stored
recipe when a project is attached.

## Prompt sets

Every verb above answers for one prompt. `each SET: OP LINE` runs one op line on every prompt of
a set and adds a summary. The engine stays batch 1: each prompt gets its own run (`start`, then
`until AT`, by default `t@last.logits`), the line goes through the same op registry as when you
type it, and the run is released (`keep` holds the runs as `SET-ITEM`, for `frame recall-a:`).
A set is a JSONL file of items (`tests/fixtures/prompt_set_tiny.jsonl`):

```
{"name": "a", "tokens": [1, 5, 9, 3], "roles": {"subject": 1}, "vars": {"answer": 4, "foil": 9}, "pair": "b"}
```

`prompt` (text, needs a tokenizer) or `tokens`; `roles` name rows (`@last` is always defined);
`vars` name tokens (an id or a one-token text); `pair` names another item. A line uses them as a
template, bound per item before dispatch, so each item's result holds plain addresses:
`t@subject.L1.A[h1]` -> `t1.L1.A[h1]`, `[row=@subject]` and `rows=@subject` -> the row,
`M[$answer,$foil]` -> `M[4,9]`, and `@pair:ADDR` -> the value at ADDR (roles of the paired item)
in a clean run of the paired prompt at the same stop, the `ORIGIN:ADDR` of
[State cells and other runs](#state-cells-and-other-runs). A pair address with no row
(`@pair:L20.A[h5]`) means the paired item's last row whatever the two lengths, and the item's
row names the donor address used (`donor`); `t@ROLE` names the pair's own role row. An item whose template does not bind
(here `d` has no `subject`) or whose op is refused is listed with its reason and left out of the
statistics; the others run. Real output on the tiny Qwen2 (`tests/repl_golden/prompt_set.out`):

```
(mdb) set load recall tests/fixtures/prompt_set_tiny.jsonl
set recall: 4 items  (tests/fixtures/prompt_set_tiny.jsonl)
  a         1 5 9 3  @subject=1 $answer=4 $foil=9 pair b
...
(mdb) each recall: whatif L1.A[h1] *= 0 --consumer M[$answer,$foil]
each recall: whatif L1.A[h1] *= 0 --consumer M[$answer,$foil]
  4 items at t@last.logits; runs released; result each2
  item  native      exact       verdict
  a     -0.0083008  -0.0074347  effect
  b     0.0065918   0.0061747   effect
  c     0.034058    0.034368    effect
  d     0.024658    0.024872    effect
  labels: native measured, exact derived-exact
summary: none refused; statistics over all 4 items
  measured.delta  [measured]
    n 4  mean 0.014252  min -0.0083008  max 0.034058  sign + 3/4
  prediction.exact.value  [derived-exact]
    n 4  mean 0.014495  min -0.0074347  max 0.034368  sign + 3/4
  effect.verdict: effect 4
  +30 more in the record (promptset result each2 [ITEM])
(mdb) each recall: reads t@subject --consumer M[$answer,$foil] top=2
...
  item  read      mass     effect      causal     verdict
  a     t3.L1.h0  0.24902  0.011455    -0.015137  causal
  b     t3.L1.h1  0.25195  -0.0082406  0.012207   causal
  c     t4.L1.h1  0.19922  -0.0073077  0.0064697  causal
  d     refused: no role 'subject' (roles: last)
  labels: mass measured, effect derived-exact, causal measured
summary: 1 of 4 items refused (d) and excluded; statistics over the 3 others
...
(mdb) each recall: whatif cache.L0.v[row=@subject] = @pair:cache.L0.v[row=@subject] --consumer M[$answer,$foil]
...
  a     0.0041504   0.0046219   effect
```

The result of `each` is one summary record (at most 64 KiB): a row per item (its key numbers,
verdict, the sha256 of its result, the op's own record id when it seals one, the bytes its run
held) and statistics for every labelled number found at the same result path in all items that
ran: n, mean, min, max and how many share the most common sign, plus a count per value of every
verdict and state field. A statistic keeps its inputs' label when all share it; inputs with
different labels are summarised per label, never merged. The statistics are arithmetic on these
items only. Each item's full result is kept in memory: `promptset result each2 a` prints it. With
a writable project it is also a sha256-checked JSON blob in the project store (the item's row
has its `blob` pointer; without a project the row says the result is held in memory only), so
`promptset result RECORD a`, with RECORD the summary's `stored` id, reads it back in a new
process and refuses a changed blob. `promptset load` reads a file on the server, so the web
console and the `/op` route refuse it; send the set inline with `promptset define NAME ITEMS`
(items as a JSON list or JSONL). A
path such as `reads.0.mass` is "the first listed read", which can be a different head in each
item. With a project attached the summary is also stored as a `prompt-set` record. After `each`
the current run, its cursor, the run table and the session digest are what they were. From
Python: `dbg.each("recall", "reads t@subject --consumer M[$answer,$foil]")` (a loaded set, a
`.jsonl` path or a list of item dicts) returns `summary`, `results` (item -> result) and the
uncapped `statistics`.

Measured on the tiny Qwen2 (CPU, one process): a run holds 600 to 872 bytes at its logits, 1.4
to 1.8 KB after `whatif` and 1.8 to 2.1 KB after `reads` (`Run.held_bytes()`); one item takes
about 0.07 s for `whatif` or `reads`, 0.016 s for `print`; 40 items of `reads` took 3.7 s and
gave a 27.7 KB summary.

## Durable checkpoints

```python
dbg = mdb.attach(model, tok, project="runs/t")      # creates or reopens the project
run = dbg.run(tokens); run.until("L1.sg"); run.set("L1.u", tensor, "add")
dbg.checkpoint(run, "pre")                          # replay recipe + edit tensors as blobs
run = mdb.attach(model, tok, project="runs/t").restart("pre")  # later, another process, same scope
```

The REPL's `checkpoint NAME` / `restart NAME` do the same when a project is attached;
`mdb.resolve(pointer, project=..., session=dbg)` turns an occurrence pointer into a tensor.

## Live and remote sessions: `mdb serve`

`mdb serve --project DIR` serves a recorded project read-only on loopback, with no model and no
torch (it needs the `serve` extra). The browser at `/` shows the project's sessions with an
address bar and a command line, and these views: Timeline (token by layer, with a branch
rail), Graph, Symbols, Records and an Inspector whose reductions (`POST /api/v1/s/SID/reduce`)
never send a whole tensor. Every response is at most 64 KiB. The recorded browser reads lane-era
projects and engine-native ones (circuit and symbol-table records, no state cuts; the fixture
is built by `tests/test_serve_recorded.py`, issue M20).

With `--tiny FAMILY` or `--model DIR` the server also holds one live debugger, attached with
that project. The page gets a Live view with four tabs and an architecture strip:

- Pipeline: the current layer's ports, motion controls, breakpoints, edits, checkpoints, console.
- Symbols: one row per binding (definition sites, lifetime strip, readers with mass and exact
  effect, verdicts on request), an Unbound row, a consumer selector and a declare form.
- Circuits: build, list and show circuits as a layered graph with `+N more` and remainder
  nodes, a keyboard walk and seek on Enter (the `circuit test` control follows the op: disabled).
- Predict: a forecast column (hollow until verified), `whatif` with the sealed prediction and
  error split, and `compare`. Nothing stages an edit except an explicit button.
- Arch strip (above the tabs): layers as rows with the current layer's ports, computed ports,
  edits, checkpoints and the selected circuit.

Every action is the registry op of the same name (`POST /api/v1/s/SID/op`), run on one worker
thread in arrival order; `GET /api/v1/s/SID/live/ledger` lists the run's circuits, forecasts and
symbols. Build the web app with `npm run build` in `web/` before building a wheel. `mdb engine
http://HOST:PORT` is the same REPL prompt over the live session; the model stays in the server
process and the client needs only the standard library. Run as two processes (the server from
an environment with the `serve` extra):

```
$ python -m mdb serve --project runs/t --tiny qwen2 --port 8799
mdb serve: http://127.0.0.1:8799/  (project runs/t; live session qwen2-872f0e348e89)
$ python -m mdb engine http://127.0.0.1:8799 --tokens "1 5 9 3" --script walk.mdb
(mdb) start 1 5 9 3
t3.L0.h  h = layer input (residual stream)  {P01}
(mdb) until L1.p
t3.L1.p  p = softmax(s, float32).to(dtype) (native attention weights)  {P05 P06 P07 P08 P09}
(mdb) checkpoint pre
checkpoint pre at t3.L1.p (736 bytes)
$ echo "info checkpoints" | python -m mdb engine http://127.0.0.1:8799      # a second client
(mdb) info checkpoints
pre       t3.L1.p  staged 0, committed 0
```

(`walk.mdb` holds `until L1.p` and `checkpoint pre`; `--tokens` issues the `start`.) In the
remote prompt `quit` ends the client and leaves the session as it is, `source FILE` reads the
client's file, and Ctrl-C interrupts the op the server is running. From Python:
`mdb.repl.remote.Remote(url).op("print", address="p")` returns the JSON that `--json` prints.
The server has no authentication and binds loopback; use a tunnel (`ssh -L 8799:127.0.0.1:8799
host`). `--allow-remote` and `--allow-host` exist but add no authentication. One SIGTERM ends `mdb serve`
even with a browser's events socket open (`tests/test_serve_sigterm.py`). `--tiny` takes the same
four families as `mdb engine`. The `/op` route refuses `promptset load` (it reads a file on the
server); send a set with `promptset define`.

## Lanes

The experiment code is the `mdb-lanes` distribution (`mdb_lanes`); the engine never imports it.
It plugs in through the entry-point groups `mdb.lanes` (verbs: `mdb lanes VERB`, and a bare
`mdb VERB` after a notice on stderr) and `mdb.lanes.exports` (names such as `mdb.BenchSession`).
Lane-era documents under `docs/` and `examples/` describe that tool, not the engine above.
Without the plug-in `python -m mdb lanes` exits 2 with "mdb lanes has no commands: the mdb-lanes
plug-in is not installed". With `PYTHONPATH=$W/src:$W/packages/mdb/src:$W/packages/mdb-lanes/src` (`W` is the repository
root), `python -m mdb lanes` prints 46 verbs and `mdb.BenchSession.__module__` is `mdb_lanes.bench`.

## Tests

With `W` the repository root:

```
cd $W/packages/mdb && env PYTHONPATH=$W/src:$W/packages/mdb/src OMP_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 python -m pytest -q -p no:cacheprovider tests/test_engine_trace.py
```

`tests/test_import_contract.py` checks that the engine imports no lane module and that its
layers (`core`, `engine`, `analysis`, `ops`, `repl`, `serve`) import only downward; `import mdb`
loads no torch. Every automated test uses a random tiny CPU model (issue M15).

Run the test files of the module you touched, not the whole suite (it is large and loads many
tiny models; on a small laptop keep to one Python process at a time). Examples:
`tests/test_ops_registry.py` (the op table), `tests/test_engine_run.py` (the stepper),
`tests/test_analysis_whatif_modes.py` (whatif at any cursor), `tests/test_repl_golden.py` (the
transcripts in `tests/repl_golden/`; after an intended change regenerate one with
`MDB_UPDATE_GOLDEN=1 ... pytest tests/test_repl_golden.py -k NAME` and read the diff).
`tests/test_deep_models.py` runs every analysis op on a 40-layer and a 70-layer tiny model (M40:
no op result may exceed the record bound). The serve tests (`tests/test_serve_*.py`) need the
environment of the common harness (`fastapi`, `httpx`) and skip with `importorskip` elsewhere,
so a pass in the plain venv says nothing about the server. The tests of `saturn_pub` live in
`$W/tests` (run them from `$W` with `PYTHONPATH=$W/src`).

## Measured on a real model

Qwen2.5-1.5B, bf16, CUDA, eager attention, RTX 4080, torch 2.13.0+cu130, transformers 5.13.1,
through mrun; one prompt family (`Description: icy / Temperature: cold / ... / Temperature:`, 22
tokens, margin `M[" hot"," cold"]`). Every number is measured; logs are under
`results/engine-r5-20261113/` and `results/engine-r6-20261113/` (local, not tracked).

Cost and size (round 4, engine commit `b440ac6`, job-301fd6de7034, job-2c1cebccd74d; the
checkpoint size is from round 5, job-5e7d47663d29, revision 3472a33 plus the round-5 working tree):

- port step 2.8 ms; layer commit 5.2 ms; restore 4.8 ms; checkpoint 739 bytes (earlier 631);
  token 189 ms plain, 213 ms stepped; peak host memory 3.9 GB (the model load)
- `explain` 0.28 s; `symbols` 0.40 s; `whatif` 1.9 s; `circuit` depth 2 / 3: 0.61 s / 0.46 s;
  `reads` 0.20 s, or 5.4 s with causal verdicts (round 4; M6 later made the verdicts share one
  excursion, measured on a tiny model only)
- round 5, job-5e7d47663d29 (the session timings the job printed; the log does not name the
  device): load + attach 3.74 s, `checkpoint` 0.29 s, `restart` 0.01 s, `explain M[a,b]` 0.26 s,
  `reads t18 ... causal=false` 0.52 s, `whatif` of a cache cell 2.25 s (2.19 s with `mix 0.5`)
- round 6, job-c5130b0aaf7e: `verify` took 0.00 to 0.01 s per forecast, per its log lines

Rounding: exact math is not what the dtype computes (round 6, M42, job-c5130b0aaf7e, engine
revision `df7d715` plus working-tree `forecast.py`, `verdict.py`, `kinds.py`; 3 prompts, `forecast`
sealed at `L0.h` and at `L14.h`, then `verify`; bf16 28 layers and an fp32 2-layer cut of the same
model):

- the emulated prediction is bit-equal to native at every stop checked: 0.0 spacings in every
  `mdb` seal/verify pair of the log (all `confirmed`), bf16 and fp32
- the exact (float64) chain of the same operands is far off at `L0.m`: bf16 34.7, 48.4 and 78.0
  spacings for the three prompts (fp32 cut: 30.9 to 39.4); at the logits 1.4 to 1.6 spacings (bf16,
  sealed at `L0.h`)
- origin, bf16 prompt 0 (t21): `t21.L0.s[h3]`, a score of spacing 64 at |s| 1.51e+04; rounding ties
  the top scores, so the native softmax is not the softmax of the real scores. Prompt 1 names
  `t11.L0.p[h1]` (spacing 64 at |s| 9.47e+03)
- a float64 chain rounded once per sum is not enough (0 to 8 spacings off in bf16, 1 to 35 in fp32,
  M42 resolution)

Cross-run head patch (M43, job-59cba8de26fc, revision `f0a483a`): `whatif L20.A[h5] =
clean:L20.A[h5] --consumer M[" hot"," cold"]`, with the clean run on `boiling` and the edited run
on `freezing`, moves the margin from -4.75 to -0.375 (native dM 4.375, exact 4.2303, residual
0.14467, floor 0.25, `effect`). Before the fix this failed on the real model with "Expected all
tensors to be on the same device, but found at least two devices, cuda:0 and cpu" (job-176bc5bdcee7;
the donor block was on the GPU, the index on the host).

Whatif exact is the same at both cursors (job-c772b1dfb576, revision `8cc8e55`): `whatif L20.m =
clean:L20.m` sealed at `L0.h` (mode ahead) and again after `until logits` (mode rewind) both give
exact dM 0.5074 on measured operands (0.50742 in the verdict line of each) and `whatif L20.A[h5]
= clean:L20.A[h5]` gives 4.2145 at both; a no-op `L20.m *= 1` gives exact 0, native 0.

Prompt set (job-c772b1dfb576): six temperature prompts (freezing, boiling, frosty, scorching,
chilly, burning), margin `M[$answer,$foil]` 3.75 to 6.375, mean 4.7708, 6/6 positive. `each temp:
whatif L20.A[h5] *= 0` gives a mean delta of -1.6042 (min -2, max -0.9375), 6/6 negative, all six
`effect`. `each temp: whatif L20.A[h5] = @pair:L20.A[h5]` (head output from the opposite prompt)
gives a mean delta of -4.6719 (min -5.1875, max -4.375), 4/4 negative; the other two items
(frosty, scorching, prompts of 23 and 24 tokens) were refused because the pair address named a
prompt row; M53 (851c41a) changed that to the paired item's last row, and it was not rerun on the
real model. `each temp: reads t@last` found reads that are `unresolved` at the noise level (M51).

Other models: SmolLM2-360M (Llama family), bf16 CUDA on Beast, job-d16ba4fd134b (M24 resolution):
the selfcheck passed with 576 stop probes and 0 mismatches, port step 3.0 ms, stepped against the
full forward max |dlogit| 0.28. A real Mistral checkpoint was not run.

The stepped run is bitwise equal to Hugging Face's incremental program (prefill plus one
token). Against one full forward over the whole prompt the largest logit difference is 0.25
(issue M10).

## Known limits

Open issues and their evidence are in [`issues.jsonl`](issues.jsonl); ids below are its `issue`.

- `attach` needs eager attention, one sequence (batch 1) and frozen weights in eval mode;
  `trace` has none of these limits but cannot step or edit. Port programs exist for Qwen2,
  Qwen3, Llama and Mistral (`mdb.engine.ports.FAMILIES`); not gemma or gemma2 (M32). Real
  models run so far: Qwen2.5-1.5B and SmolLM2-360M (Llama). Qwen3 at real size and a real
  Mistral checkpoint are not run (M14, M24).
- A checkpoint is bound to its numeric scope (thread count, determinism flag, matmul
  precision, CUDA reduced-precision reduction flags); restoring under another scope is refused.
- Memory: trail marks share the state a step did not change, so a mark costs about what its step wrote (one layer's new K/V row, the hidden carrier, the logits); `Run.held_bytes()` measures it, and `retain_rows` bounds it further.
- Port stepping re-runs the layer from its input for every port. On the tiny qwen2, one layer
  stepped from `h` to `h_out` made 145 calls against 13 for a plain call; `lookahead="layer"`
  makes 26, but native code then has computed ports past the cursor
  (`run.status()["computed_past_cursor"]`).
- A stepped run equals Hugging Face's incremental program, not one full forward (M10).
- `circuit test` is disabled and refuses (M4): zero-ablating the complement of a circuit
  (26,924 nodes on the real model, job-eb4248434df4: retained delta 1.8125, complement delta
  27.75) always says "not sufficient". It comes back when a reference-run ablation (mean or donor
  run, now possible with `NAME:ADDR` values) is built.
- Running on mrun's native kernels is not supported; Hugging Face models are the supported
  path (M7, deferred; the proxy over the store seam was measured feasible, see the ledger).
- A port edit acts on the cursor's row only, and prompt rows prefilled in one call have no
  ports (M35); a replayed (batched-prefill) and a token-by-token run of the same prompt can
  differ by one bf16 step (M26, same cause class as the M5 answer). `prefill=K` and `sweep`
  address these, see [Prefill split and position patching](#prefill-split-and-position-patching);
  the ledger states whether the two issues are closed.
- A cell `whatif` is refused (prediction only; the native measurement runs) when the first
  affected row has not started at the cursor (M9).
- Verdicts: the floor is `max(|residual|, 2 x spacing)` with the 2 supplied, and after M46 the
  residual is edited-run rounding minus base-run rounding, so reads at the noise level are
  `unresolved` (on the real model two of the top three reads for the hot/cold margin, M51). The
  accumulation premise is conservative: on bf16 CUDA every bound for sums of 256 or more terms is
  `bound refused`, although a measurement on an RTX 4080 found the linear stages bitwise equal with
  the flag on and off (M50). Bounds never decide a verdict.
- On MPS (no float64) `whatif` fails with `a numeric claim needs a finite number` (M44); CPU and
  CUDA are the tested devices.
- No dictionary or interpretable unit (M23, by choice: a candidate is a reader role declared
  from a transplant result). Two circuit systems exist, the lane-era `discover_circuit` (ranks by
  write norm, picks the attention sink) and the engine's `circuit`; use the engine's (M25).
- The help text of two ops lags the code: `verify` still describes the removed 3 x sqrt(n) floor
  and `circuit` still says "test by native ablation" (`ops/predict.py`, `ops/inspect.py`; not in
  the ledger yet).
- Package hygiene (M13: `experiments/`, `examples/`, `results/`, `docs/` of the lane era still
  live in `packages/mdb`; bare `mdb run|open|attach` still dispatch to lanes), savings left out of
  the cast round because they change float64 digits (M16), and only tiny-model tests in the
  repository (M15: the three real-model faults of the last rounds, the 48 s per-port rebuild, the
  17.6 GB float64 weight copy and the rounded RoPE buffer, were visible only on a real model; the
  device mix, the op-result cap and the verify floor of rounds 5 and 6 as well).

## Not built yet

- `trace(project=)`.
- `export` (a walkthrough bundle of records and pointers): the verb answers "not built yet".
- `mdb serve --model` as an mrun job with the REPL connected through a tunnel (no remote parity
  run on the GPU machine yet), port programs for Qwen3 at real size and FLUX.2
  Klein (M14).
- gemma and gemma2 port programs (M32); the circuit-test replacement (M4); named units (M23).

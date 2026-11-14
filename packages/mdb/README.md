# mdb

mdb is a debugger for language models, in the spirit of gdb. It runs a loaded Hugging Face
decoder (Qwen2 or Qwen3) and stops inside each native layer at named values called ports, where
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
`mdb engine MODEL [PROMPT]` names what to debug, as `gdb PROGRAM` does. `MODEL` is a checkpoint
directory, the name of a model in the local Hugging Face cache (`mdb engine qwen2.5-1.5b
"Weather: freezing"`; the name lookup was run against a real cache, the model was not loaded
here), `tiny:qwen2` or `tiny:qwen3`, or the address of a running `mdb serve` (see
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
| activation patching | `set ADDR = OTHERRUN:ADDR` for ports (`set L0.A[h1] = clean:L0.A[h1]`, see the example) and state cells; `whatif` and `compare` take the same value and measure the edit natively, then restore |
| head or path ablation | `set L<L>.A[h5] *= 0` (selector `[h5]`); `whatif` for the effect without leaving the stop |
| direct logit attribution | `explain M[a,b]` (exact split of a margin, with a measured remainder) |
| KV cache | state cells `cache.L<L>.k` / `v[kv0,row=2]`; readable, editable, and takeable from another run |
| `run_with_cache` over a batch, batched forward | not available: batch 1 (issue M22) |
| SAE features, dictionary units | none (issue M23) |

Example: patch one head's output from a clean run into a corrupted run and read a logit
difference. TransformerLens style, not run (patched at the last position: an mdb port edit acts
on the cursor's row; the earlier prompt rows are prefilled and live in the cache cells):

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
  sealed  linear ΔM = -0.00021867 [derived-local P14]   exact ΔM = -0.00035917 [derived-exact]  ...
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

mdb adds, and the others do not: a prediction sealed before the native measurement with the gap
split by cause (`forecast`, `verify`), an authority label and state on every number, and
reversible stepping inside a layer (`reverse-step`, `restart`). It lacks batching (M22), model
families beyond Qwen2 and Qwen3, and dictionary features (M23).

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
occurrence of the same selected shape (the activation patching example above). `seek` does not switch runs
(use `frame NAME:`), and a target without a row (`seek L20.p`) is on the cursor's row.

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
`obsidian/model-math/` (P01-P33). On Qwen3, `qr` and `kr` take `qn` and `kn`.
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
| `circuit` | `circuit CONSUMER [top=K] [depth=D] \| circuit show ID` | mutate | Circuit records: build from the exact decomposition (nodes, edges, remainder), show as a walk; `circuit test ID` is disabled (issue M4) |
| `commit` | `commit` | mutate | Commit the pending layer with its staged edits (cursor to its h_out) |
| `compare` | `compare A B [consumer=C]  (A, B: ADDR=VALUE [op] or a checkpoint name)` | navigate | Compare branches: base, A, B and A+B measured natively, effects and the interaction |
| `condition` | `condition N [EXPR]` | mutate | Set the condition of breakpoint N (a watch's predicate); no expression removes it |
| `continue` | `continue [N] [budget=ROWS]` | navigate | Run to the next armed breakpoint or watch hit (N: ignore the breakpoint stopped at N-1 more times, as gdb) |
| `delete` | `delete [ID ...]` | mutate | Delete breakpoints and watches (all when no ids are given) |
| `disable` | `disable [ID ...]` | mutate | Disable breakpoints and watches without deleting them (all when no ids) |
| `display` | `display [ADDR]` | mutate | Show an address after every stop (no address: list displays) |
| `down` | `down [N]` | navigate | Move the focus back toward its consumer, N times |
| `enable` | `enable [ID ...]` | mutate | Enable breakpoints and watches (all when no ids are given) |
| `explain` | `explain M[a,b]\|ADDR` | read | Exact accounting of a margin M[a,b] or of one port, with proof IDs and the measured remainder (cursor at the row's logits/sample) |
| `export` | `export` | not built yet (B1) | Export a walkthrough bundle of records and pointers |
| `finish` | `finish` | navigate | From a port inside a layer: run to that layer's h_out. From h_out or a token-level stop: run to this row's sample |
| `forecast` | `forecast [ADDR\|M[a,b]]` | mutate | Seal a prediction of the next stop (or a later stop of this row, or a margin M[a,b]) from values at the cursor, before native code runs |
| `frame` | `frame [ADDR \| cursor \| NAME:]` | navigate | Show or set the focus address (print/list/backtrace default to it); NAME: switches to the named run |
| `ignore` | `ignore N COUNT` | mutate | Ignore the next COUNT hits of breakpoint N (they are counted, not reported) |
| `info` | `info ports\|breakpoints\|watch\|display\|checkpoints\|edits\|program\|runs [offset=N] [limit=N]` | read | Show debugger tables or the program (lists are paged: offset=, limit=) |
| `list` | `list [LAYER]` | read | The layer's port program: operator text, proof IDs, cursor marked |
| `next` | `next [N]` | navigate | Step to the last port of the sublayer (r, then h_out) |
| `print` | `print [ADDR] [top=K] [slice=N]` | read | Reduce a value: shape, dtype, stats, top-k by \|value\|, pointer |
| `reads` | `reads SOURCE [rows=R\|A:B\|all] [consumer=C] [top=K] [causal=false]` | navigate | Who reads a value: attention reads of a source row with mass, exact contribution and a native causal verdict |
| `restart` | `restart NAME` | navigate | Return to a named checkpoint (rewinds or replays) |
| `reverse-continue` | `reverse-continue` | navigate | Run backwards to the previous breakpoint or watch hit |
| `reverse-finish` | `reverse-finish` | navigate | Step back to the boundary before the pending layer (its input) |
| `reverse-next` | `reverse-next [N]` | navigate | Step back to the previous sublayer end |
| `reverse-step` | `reverse-step [N] [unit] [budget=ROWS]` | navigate | Step back one port (or N units; layer/token units enter at most `budget` rows) |
| `run` | `run [name=NAME] [TEXT \| ID ID ...] [budget=N]` | native | Like start, then continue to the first armed breakpoint or watch |
| `seek` | `seek ADDR \| seek ORIGIN:ADDR \| seek POINTER-JSON` | navigate | Move the session to an occurrence: forward, backward, or by replaying a stored recipe (ORIGIN:ADDR or an occurrence pointer) |
| `set` | `set ADDR = VALUE\|ADDR\|ORIGIN:ADDR [replace\|add\|scale\|mix DOSE]` | mutate | Stage an edit at a port of the pending layer, or of a state cell (cache.L<i>.k\|v[kv,row,d]): set ADDR = VALUE [op] |
| `start` | `start [name=NAME] [TEXT \| ID ID ...]` | native | Prefill a prompt and stop at the pending token's first port (L0.h) |
| `step` | `step [N] [port\|sublayer\|layer\|token] [budget=ROWS]` | navigate | Step one port (or N units; layer/token units enter at most `budget` rows) |
| `symbol` | `symbol NAME = ADDR[,ADDR...]` | mutate | Declare a user symbol over cache rows: symbol NAME = ADDR[,ADDR...] |
| `symbols` | `symbols [consumer=C] [rows=R\|A:B\|all] [min_mass=X] [offset=N] [limit=N]` | mutate | Symbol table at the cursor: bindings with definition sites, lifetimes, readers, status and an Unbound row (stored as records with a project) |
| `tbreak` | `tbreak PATTERN [if CONDITION]` | mutate | Set a breakpoint that deletes itself after its first hit |
| `undisplay` | `undisplay [ID ...]` | mutate | Remove displays (all when no ids are given) |
| `until` | `until ADDR [budget=ROWS]` | navigate | Run to an address, stopping earlier at an armed breakpoint or watch |
| `up` | `up [N] [PORT]` | navigate | Move the focus to a producer (toward the layer input), N times; `up PORT` picks the producer by name, the result lists the alternatives |
| `verify` | `verify [ID ...]` | mutate | Compare open forecasts whose target the cursor has passed with the native values (residual, bound, order) |
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

model = mdb.open_tiny("qwen2").model         # any loaded Qwen2/Qwen3 causal LM
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

`forecast` seals a prediction from values at the cursor, before native code runs; `verify`
compares it with what native code then produced (`tests/repl_golden/predict_forecast.out`). The
verdict is `confirmed` when, at every element, the measured residual (native - predicted, P20) is
within the floor, else `not confirmed`; both are `resolved`. The floor is
`3 x sqrt(n) x spacing`: `spacing` is the model dtype's gap at the consumer's magnitude (a margin
is a difference of two rounded logits: the sum of their two spacings), `n` the rounded
operations between the sealed operands and the target (their errors add as a random walk;
`derived-exact`, a count of the forecast's chain), and `3` is `supplied`: the largest measured
residual / (sqrt(n) x spacing) over 12264 correct forecasts on tiny Qwen2/Qwen3 (BF16 and FP32,
2 to 28 layers) was 2.18 (see the `coefficient.support` of the result). The residual is printed
in spacings too, so a pass near the floor is visible. An element's sealed worst-case bound
(`bound`, finite entries only) can tighten the floor at that element and never loosen it; it is
named in the line when it does. Spacing at an element's own magnitude is not used: a small
element of a sum rounds at the scale of its operands (measured up to 10^4 own spacings). The
whole-vector worst-case bound is printed only when finite:

```
(mdb) forecast L0.p
forecast 1  t3.L0.p  sealed at t3.L0.h  record f8059f3549dad257
  predicted max_abs 0.25152 [derived-exact] norm 0.70712 [derived-exact]  sha256 cb1cc1226f21049d  bound 0.0026798 [bound] per element
(mdb) until t3.logits
t3.logits  logits = lm_head(fn)
(mdb) verify
forecast 1 t3.L0.p: confirmed  native max_abs 0.25195 [measured] norm 0.70781 [measured]  residual 0.00091062 [measured] = 0.47 spacings <= floor 0.015502 [supplied P20] = 7.94 bfloat16 spacings (3 x sqrt(7 rounded ops), 3 supplied); element 4 limited by its sealed bound 0.0026096 [bound P20]  [P20, resolved]; worst-case bound 0.0026798 [bound] (holds, information)
```

`whatif` seals the predicted change of a consumer under one edit, measures it natively on a
private branch, and restores the session (`tests/repl_golden/predict_whatif.out`):

```
(mdb) whatif L1.m = 0.5 add --consumer M[4,9]
whatif t3.L1.m add 0.5 [supplied] -> t3.M[4,9]  (mode pending, record 5d511215876ce57d)
  sealed  linear ΔM = 1.4572 [derived-local P14]   exact ΔM = 0.16513 [derived-exact]  bound 0.07029 [bound]
  native  ΔM = 0.16449 [measured]   (base -0.11401 [measured] -> edited 0.050476 [measured])
  error split (P20): linear 1.4572 [derived-local P14] + nonlinear -1.2921 [derived-exact P20] + realisation -4.4976e-05 [derived-exact P20] + rounding -0.00059161 [measured P20] (within bound 0.061091 [bound])
  verdict (P20): effect  native ΔM 0.16449 [measured P20], exact ΔM 0.16508 [derived-exact P20], residual -0.00059161 [measured P20]; floor 0.00097656 [measured P20] = max(|residual|, 2 x spacing 0.00048828 [measured P20])
  session unchanged at t3.L1.a  (5 moves, restored)
```

### Verdicts (P20)

`whatif`, `reads` and `compare` judge an effect from the P20 accounting
(`mdb.analysis.verdict`): native change = exact change (float64 chain, `derived-exact`) +
residual (`measured`). The floor is `max(|residual|, 2 x spacing)`, spacing being the gap between
adjacent values of the consumer's dtype at its magnitude. A change is an `effect` when the
native change is above the floor and the exact change has its sign and exceeds the residual;
`no effect above the noise floor` when both are within the floor; otherwise `unresolved`, with
the numbers. With no exact chain (as in `compare`) the native change is held against `2 x
spacing` alone and the line says so. Worst-case rounding bounds are still shown where they are
finite (`bound` label) and never decide a verdict. `verify` follows the same rule: a forecast
is `confirmed` when its measured residual (native - predicted) is within the floor, `not
confirmed` otherwise, whatever the bound (`no-bound` is no longer a verdict).

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
whatever is not listed stays in the record as `+N more` and `remainder` nodes. `circuit test` is
disabled for now (issue M4). `compare` measures two edits separately and together:

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
host`). `--allow-remote` and `--allow-host` exist but add no authentication.

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

## Measured on a real model

Qwen2.5-1.5B, bf16, CUDA, eager attention, RTX 4080, torch 2.13.0+cu130, transformers 5.13.1,
engine commit `b440ac6`, via mrun (job-301fd6de7034, job-2c1cebccd74d). All numbers measured.

- port step 2.8 ms; layer commit 5.2 ms; restore 4.8 ms; checkpoint 631 bytes; token 189 ms
  plain, 213 ms stepped; peak host memory 3.9 GB (the model load)
- `explain` 0.28 s; `symbols` 0.40 s; `whatif` 1.9 s; `circuit` depth 2 / 3: 0.61 s / 0.46 s;
  `reads` 0.20 s, or 5.4 s with causal verdicts (issue M6)

The stepped run is bitwise equal to Hugging Face's incremental program (prefill plus one
token). Against one full forward over the whole prompt the largest logit difference is 0.25
(issue M10).

## Known limits

Open issues and their evidence are in [`issues.jsonl`](issues.jsonl); ids below are its `issue`.

- `attach` needs eager attention, one sequence (batch 1) and frozen weights in eval mode;
  `trace` has none of these limits but cannot step or edit. Port programs exist for Qwen2 and
  Qwen3 only, and Qwen3 at real size is not run (M14).
- A checkpoint is bound to its numeric scope (thread count, determinism flag, matmul
  precision, CUDA reduced-precision reduction flags); restoring under another scope is refused.
- Memory: trail marks share the state a step did not change, so a mark costs about what its step wrote (one layer's new K/V row, the hidden carrier, the logits); `Run.held_bytes()` measures it, and `retain_rows` bounds it further.
- Port stepping re-runs the layer from its input for every port. On the tiny qwen2, one layer
  stepped from `h` to `h_out` made 145 calls against 13 for a plain call; `lookahead="layer"`
  makes 26, but native code then has computed ports past the cursor
  (`run.status()["computed_past_cursor"]`).
- A stepped run equals Hugging Face's incremental program, not one full forward (M10).
- `circuit test` is disabled for now (issue M4).
- Open items are in `issues.jsonl`: M6, M9, M11-M17 (and any later ids), M7 deferred, M10 a
  known limit.
- Running on mrun's native kernels is not supported; Hugging Face models are the supported
  path (issue M7).

## Not built yet

- `trace(project=)`.
- `export` (B1), an engine-native recorded fixture for `mdb serve`, and `mdb serve --model` as
  an mrun job with the REPL on a tunnel (B5); port programs for Qwen3 at real size and
  FLUX.2 Klein (all M14).

# mdb

mdb is a debugger for language models, in the spirit of gdb. It runs a loaded Hugging Face
decoder (Qwen2 or Qwen3) and stops inside each native layer at named values called ports, where
you can read them, edit them, checkpoint and rewind. Every number it shows carries a label that
says how the number was obtained.

This directory is the `mdb-runtime` distribution (import name `mdb`). It depends on `saturn-pub`,
whose `Session` runs the model. The older experiment code lives in `../mdb-lanes` and plugs in
through entry points (see [Lanes](#lanes)).

Every command and code block below was run on a random tiny model on a CPU, offline, and the
output shown is the real output. `...` marks a cut, and the text says where.

## Quick start

The REPL on a tiny Qwen2 (bf16, 2 layers, no tokenizer, so the prompt is token ids).
`python -m mdb.repl` is the same entry point. The commands below were fed with `--script FILE`.

```
$ python -m mdb engine tiny:qwen2 --tokens "1 5 9 3"
(mdb) start 1 5 9 3
t3.L0.h  h = layer input (residual stream)  {P01}
(mdb) step
t3.L0.a  a = RMSNorm(h) * w_in  {P02 P03}
(mdb) until L1.p
t3.L1.p  p = softmax(s, float32).to(dtype) (native attention weights)  {P05 P06 P07 P08 P09}
(mdb) print top=2
t3.L1.p  bfloat16 [1, 2, 1, 4]  (batch, head, row, src)
  norm 0.70607 min 0.24902 max 0.25 mean 0.24963 abs_max 0.25 [measured]
  top |v|: [0,0,0,0] 0.25, [0,0,0,3] 0.25 [measured]
  pointer occurrence ~stepped sha256 a3a32de6269f57e6 cut 67c4899b500f
(mdb) print t3.L0.p[h1] top=2
t3.L0.p[h1]  bfloat16 [1, 2, 1, 4] -> [1, 1, 1, 4]  (batch, head, row, src)
  norm 0.50099 min 0.24805 max 0.25195 mean 0.25049 abs_max 0.25195 [measured]
  top |v|: [0,1,0,0] 0.25195, [0,1,0,3] 0.25195 [measured]
  pointer occurrence ~stepped sha256 bda254d8ec53233c
(mdb) x cache.L0.k[kv0,row=2] 4
cache.L0.k[kv0,row=2]  [0:4] of 8: 0.034912 -0.011963 -0.060059 -0.016113 [measured]
(mdb) backtrace
p <- s <- qr,kr <- q,k <- a <- h
#0  t3.L1.p  p = softmax(s, float32).to(dtype) (native attention weights)  {P05 P06 P07 P08 P09}  <- s
#1  t3.L1.s  s = scaling * qr repeat_kv(kr)^T + mask (replica of the native expression)  {P05 P07 P08 P09}  <- qr,kr
...  (#2 to #5 cut: qr, kr, q, k, a, h)
(mdb) checkpoint pre
checkpoint pre at t3.L1.p (628 bytes)
(mdb) set L1.m = 0 scale
staged at t3.L1.m: m scale 0 [supplied] (not fired yet)  (1 staged)
(mdb) tbreak L*.h_out
Temporary breakpoint 1 at L*.h_out
(mdb) continue
Temporary breakpoint 1, L*.h_out
t3.L1.h_out  h_out = r + m (layer output)  {P14}
(mdb) until logits
t3.logits  logits = lm_head(fn)
(mdb) print logits top=3
t3.logits  bfloat16 [1, 32]  (batch, vocab)
  norm 0.39659 min -0.16699 max 0.13477 mean -0.0077629 abs_max 0.16699 [measured, under edit t3.L1.m]
  top |v|: [0,6] -0.16699, [0,7] -0.15234, [0,20] 0.13477 [measured, under edit t3.L1.m]
  pointer omitted: edited run: occurrence origins (@cut+edit) are not wired yet
(mdb) restart pre
t3.L1.p  p = softmax(s, float32).to(dtype) (native attention weights)  {P05 P06 P07 P08 P09}
(mdb) info edits
staged at t3.L1.p: none
```

`start` prefills all tokens but the last; the last token (row 3) is the one you step through.

`mdb engine MODEL [PROMPT]` names what to debug, as `gdb PROGRAM` does. `MODEL` is a checkpoint
directory (run here on a saved tiny model), the name of a model in the local Hugging Face cache
(`mdb engine qwen2.5-1.5b "Weather: freezing"`; the name lookup was run against a real cache,
the model itself was not loaded here), `tiny:qwen2` or `tiny:qwen3`, or the address of a
running `mdb serve` (see [Live and remote sessions](#live-and-remote-sessions-mdb-serve)).
Nothing is ever downloaded; a name that matches no cached model, or several, is refused with
the candidates. `--project DIR` attaches a project, which makes checkpoints durable.

## Embedding

```python
import mdb
from mdb.ops import dispatch

dbg = mdb.open_tiny("qwen2")                 # or mdb.attach(model, tokenizer)
run = dbg.run([1, 5, 9, 3])                  # prefill 3 tokens; cursor at t3.L0.h
run.until("L1.p")                            # stop inside layer 1 at the attention weights
print(dispatch(dbg.ops(run), "print", {"top": 2})["stats"]["norm"])
saved = run.checkpoint("pre")                # about 1 KB of plain data
run.set("L1.m", 0.0, "scale")                # stage an edit further down the same layer
print(run.until("logits"), run.read()[0, :3], run.read_info()["upstream_edits"])
print(run.restore(saved), run.edits)         # back at t3.L1.p, nothing staged
```

```
{'value': 0.7060722464967372, 'authority': 'measured'}
t3.logits tensor([-0.0233,  0.0430,  0.0220], dtype=torch.bfloat16) [{'row': 3, 'layer': 1, 'port': 'm', 'op': 'scale', 'staged': False}]
t3.L1.p []
```

`mdb.attach(model, tokenizer)` takes a loaded model; it refuses a model without eager attention
or with modules in training mode, and never changes the model. `dbg.repl(run)` opens the REPL
on the same run. Every REPL verb is also `dispatch(ctx, verb, args)` with JSON in and out.

## Addresses

One string names a value everywhere: on the command line, in records and in the web viewer.

```
[origin:] t<row>[.s<step>].L<layer>.<port>[selectors][~program]  |  cache.L<layer>.k|v[selectors]
```

Selector keys are `h kv i src row d`; `a:b` is a half-open range and `a|b` a set. The stops of
one token row, in order: `start`, `emb`, `L<i>.<port>` for every layer and port, `fn`,
`logits`, `sample`. Five addresses from the session above:

| Address | Meaning |
|---|---|
| `L1.p` | port `p` of layer 1, in the cursor's row |
| `t3.L0.p[h1]` | head 1 of layer 0's attention weights, token row 3 |
| `cache.L0.k[kv0,row=2]` | key-cache cell of KV head 0 written by row 2, read now |
| `logits` | the logits stop of the cursor's row |
| `L*.h_out` | a pattern (breakpoints, watches): the output of every layer |

An occurrence (`t2.L0.k`, the value written) and a cache cell (`cache.L0.k[row=2]`, read at
the current clock) are different types. An origin names another run: a live named run, a
checkpoint or a stored session (`boiling:cache.L0.v[kv0,row=1]`, `boiling:t3.L1.kr[kv0]`).

## State cells and other runs

`set cache.L<l>.k|v[selectors] = VALUE [replace|add|scale|mix DOSE]` stages a write into the KV
cache. Rows must already be in the cache; the row the pending layer writes is refused. The cell
is realised before its reader (`kr` for keys, `A` for values) unless the cursor has passed the
boundary (`kr`, `p`), in which case it lands after that read. Shorthand: `+= -= *= /=`.
`mix DOSE` is `cell + DOSE * (value - cell)` in float64 from the native bits, rounded once to the
cell dtype (`mix 1` is the exact copy of the value, `mix 0` leaves the cell as it is).
`run name=boiling 1 7 9 3` starts a second run; `frame boiling:` and `frame main:` switch;
`info runs` lists them. `ORIGIN:ADDR` takes a value from that run, a checkpoint or a stored
session of the same model and numeric scope; `k`/`kn` occurrences are refused (before RoPE).

```
(mdb) compare cache.L0.v[kv0,row=1] = boiling:cache.L0.v[kv0,row=1] cache.L1.k[kv0,row=1] = boiling:cache.L1.k[kv0,row=1] --consumer M[4,9]
(mdb) whatif cache.L0.v[kv0,row=1] = boiling:cache.L0.v[kv0,row=1] --consumer M[4,9]
(mdb) set cache.L0.v[kv0,row=1] = boiling:cache.L0.v[kv0,row=1] mix 0.5
```

Cell edits go into checkpoints, restore, reverse and replay. Embedding: `run.set_cell(layer,
"v", value, rows=[1], kv=[0])`; `dbg.cell_value("boiling:cache.L0.v[kv0,row=1]", run=run)`.

## Ports

One line per port, in native execution order, generated from the port program
(`mdb.engine.ports.program_for`). `info ports` and `list` print the same program. Proof IDs
refer to `obsidian/model-math/` (P01-P33). On Qwen3, `qr` and `kr` take `qn` and `kn`.

<!-- BEGIN ports: generated by tests/test_readme.py -->
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
<!-- END ports -->

`kr` is the only port that cannot be edited (it holds cached rows). `s` is not a native
intermediate: it is a replica of the native expression, checked against `p` bit for bit.

## Verbs

Generated from the op registry (`mdb.ops.registry()`). The REPL, `--json` mode, the server's
`POST /op` and `dispatch` all go through it. `tests/test_readme.py` fails when a registered
verb is missing here; `python tests/test_readme.py` regenerates this table and the port table.

<!-- BEGIN verbs: generated by tests/test_readme.py -->
| Verb | Usage | Effect | What it does |
|---|---|---|---|
| `abort` | `abort` | mutate | Drop the staged edits of the pending layer (and the staged cell edits) |
| `advance` | `advance ADDR [budget=ROWS]` | navigate | Run to an address, stopping earlier at an armed breakpoint or watch |
| `backtrace` | `backtrace [ADDR]` | read | Producer chain from the focus port back to the layer input |
| `break` | `break PATTERN [if CONDITION]` | mutate | Set a breakpoint on an address pattern (L*.p, t5.L1.o, logits, cache.L0.v[row=2]); cache cells are evaluated at boundary stops only |
| `checkpoint` | `checkpoint [NAME]` | mutate | Name the cursor position (about 1 KB of plain data) |
| `checkpoints` | `checkpoints [offset=N] [limit=N]` | read | List named checkpoints (paged: offset=, limit=) |
| `circuit` | `circuit CONSUMER [top=K] [depth=D] \| circuit test ID \| circuit show ID` | mutate | Circuit records: build from the exact decomposition (nodes, edges, remainder), test by native ablation, show as a walk |
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
<!-- END verbs -->

## Authority labels

Every number shown or stored carries exactly one label, and a state.

| Label | Meaning | Example |
|---|---|---|
| `measured` | read from native execution | `print top=2` at `t3.L1.p` |
| `derived-exact` | an exact identity on captured operands | the replica `s`; a value written by an `add` or `scale` edit |
| `derived-local` | a linearisation or quadrature only | the linear term of `whatif` |
| `bound` | a theorem with stated premises | the rounding bound of a `forecast` |
| `supplied` | a value handed in by a user or a stored record | the edit request `m scale 0` |

State is `resolved`, `unresolved` (could not be computed: never shown as zero) or `refused`
(a premise failed: the label is empty and a reason is given). A value downstream of an edit
stays `measured` and lists the edits above it (`[measured, under edit t3.L1.m]`).

## Passive capture: `mdb.trace`

`attach` drives the model. `trace` only watches a forward your experiment already runs: any
batch size, sdpa, `generate`, gradients on. Hooks sit at the same sites as the stepper's, so a
port name means the same thing in both.

```python
import torch

import mdb

model = mdb.open_tiny("qwen2").model         # any loaded Qwen2/Qwen3 causal LM
batch = torch.tensor([[1, 5, 9, 3], [2, 6, 10, 4]])
with mdb.trace(model, ports=["L1.p", "L1.s", "L1.m", "L0.h"]) as t, torch.no_grad():
    model(batch)                             # the experiment's own forward
print(t.calls, tuple(t["L1.m"].shape), tuple(t["L1.p"].shape))
print({k: t.info("L1.s")[k] for k in ("state", "authority", "kind", "replica_equal")})
print(t.record()["calls"][0]["values"]["L1.m"]["sha256"][:16])
model.set_attn_implementation("sdpa")
with mdb.trace(model, ports=["L1.p", "L1.m"]) as t, torch.no_grad():
    model(batch)
print(t.info("L1.m")["state"], "|", t.info("L1.p")["reason"])
```

```
1 (2, 4, 16) (2, 2, 4, 4)
{'state': 'resolved', 'authority': 'derived-exact', 'kind': 'replica', 'replica_equal': True}
df7028d41d9aacde
resolved | attention implementation is 'sdpa': qr, kr, s and p exist only inside eager attention (load with attn_implementation='eager')
```

- Passive: hooks return nothing, so logits are bitwise equal with and without the trace
  (tested for qwen2 and qwen3, bf16 and fp32, eager and sdpa).
- Same values as the stepper: on the stepper's native call (the last token on top of the same
  cache) every port is bitwise equal to `Run.read` (tested, all ports, the same four cases).
- One call is one outermost forward of `model.model`. With several calls use `t["L1.m", i]`
  or `t.info(address, call=i)`. Values keep the batch dimension; the axes are in
  `t.info(address)["axes"]`.
- Refused, never wrong: `qr`, `kr`, `s`, `p` without eager attention; `s`, `p` under
  attention dropout; any port of a layer that did not run; everything after the `max_bytes`
  cap (default 1 GiB). Captures are detached copies on the model's device.
- Hooks are removed when the block exits, also when the forward raises. A second active
  trace is refused. `project=` is not supported by `trace` yet.

## Predictions: `forecast`, `verify`, `whatif`

`forecast` seals a prediction from values at the cursor, before native code runs; `verify`
compares it with what native code then produced (`tests/repl_golden/predict_forecast.out`):

```
(mdb) forecast L0.p
forecast 2  t3.L0.p  sealed at t3.L0.h  record 6a5b7ce96f898a51
  method 7 identities t3.L0.a .. t3.L0.p (last softmax P05)
  predicted max_abs 0.25152 [derived-exact] norm 0.70712 [derived-exact]  ...  bound 0.0026798 [bound] per element
(mdb) until t3.logits
(mdb) verify
forecast 2 t3.L0.p: within-bound  native max_abs 0.25195 [measured] norm 0.70781 [measured]  residual 0.00091045 [measured] <= bound 0.0026798 [bound]
  order: sealed at t3.L0.h (stop 2) before the target (stop 10); cut descends through 4 receipts
```

`whatif` seals the predicted change of a consumer under one edit, measures it natively on a
private branch, and restores the session (`tests/repl_golden/predict_whatif.out`):

```
(mdb) whatif L1.m = 0.5 add --consumer M[4,9]
whatif t3.L1.m add 0.5 [supplied] -> t3.M[4,9]  (mode pending, record 579b42095616cf72)
  sealed  linear ΔM = 1.4572 [derived-local P14]   exact ΔM = 0.16513 [derived-exact]  bound 0.07029 [bound]
  native  ΔM = 0.16449 [measured]   (base -0.11401 [measured] -> edited 0.050476 [measured])
  error split (P20): linear 1.4572 [derived-local P14] + nonlinear -1.2921 [derived-exact P20] + realisation -4.4976e-05 [derived-exact P20] + rounding -0.00059161 [measured P20] (within bound 0.061091 [bound])
  session unchanged at t3.L1.a  (5 moves, restored)
```

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
    causal: Δ -0.015137 [measured], predicted -0.01456 [derived-exact] (within rounding bound 0.029953 [bound])
...
(mdb) symbol ctx = t1, cache.L1.v[row=2]
symbol ctx = t1, cache.L1.v[row=2]  (declared: ctx)
(mdb) symbols consumer=M[4,9]
symbols at t3.logits  consumer t3.M[4,9] = -0.11401 [measured]  min_mass 0.1 [supplied]
  ctx         read   mass 1.4971 [derived-exact]  to_consumer 0.0020898 [derived-exact P19]  life t3:## (t3.L0..t3.L1)
    defined t1, cache.L1.v[row=2]
    readers t3.L1.h0 mass 0.49805 effect 0.013241 [derived-exact P19]; t3.L0.h1 mass 0.25 effect -0.0081352 [derived-exact P19]; ...
...
  Unbound     mass 0 [derived-exact]  readout -0.14463 [measured]
```

Mass is summed over the heads and layers that read the binding, so it can exceed 1.
`circuit CONSUMER` builds a record of exact contributions that closes on the consumer;
whatever is not listed stays in the record as `+N more` and `remainder` nodes. `circuit test`
measures the retained set and its complement natively, and `compare` measures two edits
separately and together:

```
(mdb) circuit M[4,9] top=2 depth=3
circuit 1  t3.M[4,9]  top 2 depth 3  (16 nodes, record bfdf9816699df9de)
  t3.M[4,9]  -0.11401 [measured]
    embedding  t3.L0.h  -0.14578 [derived-exact P19]
    L1.attn  t3.L1.o  0.03064 [derived-exact P19]
      L1.h0  t3.L1.A[h0]  0.025084 [derived-exact P07]
        t0  t3.L1.p[h0,src=0]  0.013624 [derived-exact P06]
...
    +3 more  t3.M[4,9] (unlisted)  0.00064862 [derived-exact]
    remainder  t3.M[4,9]  0.00047376 [measured]
  closure 1.3878e-17 [derived-exact P19] within 2.0716e-15 [bound]
(mdb) compare L1.m=0.5 add L0.v = 0.25 add --consumer M[4,9]
compare A: L1.m=0.5 add  B: L0.v=0.25 add  -> t3.M[4,9]
  base -0.11401 [measured]  A 0.050476 [measured]  B -0.068237 [measured]  A+B 0.052124 [measured]
  effects A 0.16449 [measured]  B 0.045776 [measured]  A+B 0.16614 [measured]
  interaction (A+B) - A - B + base = -0.044128 [measured]
  session unchanged at t3.logits
```

`seek ADDR` moves the session to an occurrence: forward, backward, or by replaying a stored
recipe when a project is attached.

## Durable checkpoints

```python
dbg = mdb.attach(model, tok, project="runs/t")      # creates or reopens the project
run = dbg.run(tokens); run.until("L1.sg"); run.set("L1.u", tensor, "add")
dbg.checkpoint(run, "pre")                          # replay recipe + edit tensors as blobs
# later, in another process with the same model and numeric scope:
run = mdb.attach(model, tok, project="runs/t").restart("pre")    # same cut fingerprint
```

The REPL's `checkpoint NAME` / `restart NAME` do the same when a project is attached.
`mdb.resolve(pointer, project=..., session=dbg)` turns an occurrence pointer back into a tensor.

## Live and remote sessions: `mdb serve`

`mdb serve --project DIR` serves a recorded project read-only on loopback, with no model and no
torch (it needs the `serve` extra). The browser at `/` shows the project's sessions with an
address bar and a command line, and these views: Timeline (token by layer, with a branch
rail), Graph, Symbols, Records and an Inspector whose reductions (`POST /reduce`) never send a
whole tensor. Every response is at most 64 KiB.

With `--tiny FAMILY` or `--model PATH` the server also holds one live debugger, attached with
that project. The page gets a Live view (the port pipeline of the current layer, step / next /
finish / until / continue / reverse controls, breakpoints, edits, checkpoints, a console), and
every action is the registry op of the same name, run on one worker thread in arrival order.
`mdb engine http://HOST:PORT` is the same REPL prompt over that session: the model stays in
the server process and the client needs only the standard library. Run here as two processes:

```
$ python -m mdb serve --project runs/t --tiny qwen2 --port 8799
mdb serve: http://127.0.0.1:8799/  (project runs/t; live session qwen2-8a16dcab78be)
$ python -m mdb engine http://127.0.0.1:8799 --tokens "1 5 9 3" --script walk.mdb
(mdb) start 1 5 9 3
t3.L0.h  h = layer input (residual stream)  {P01}
(mdb) until L1.p
t3.L1.p  p = softmax(s, float32).to(dtype) (native attention weights)  {P05 P06 P07 P08 P09}
(mdb) checkpoint pre
checkpoint pre at t3.L1.p (628 bytes)
$ echo "info checkpoints" | python -m mdb engine http://127.0.0.1:8799      # a second client
(mdb) info checkpoints
pre       t3.L1.p  staged 0, committed 0
```

(`walk.mdb` holds the two commands shown.) In the remote prompt `quit` ends the client and
leaves the session as it is, `source FILE` reads the client's file, and Ctrl-C interrupts the
op the server is running. From Python: `mdb.repl.remote.Remote(url).op("print", address="p")`
returns the JSON that `--json` prints. The server has no authentication and binds loopback;
reach another machine through a tunnel (`ssh -L 8799:127.0.0.1:8799 host`), not by binding it
to a network.

## Lanes

The experiment code is the `mdb-lanes` distribution (`mdb_lanes`). The engine never imports it.
It plugs in through two entry-point groups: `mdb.lanes` (CLI verbs, run as `mdb lanes VERB`)
and `mdb.lanes.exports` (names such as `mdb.BenchSession`, resolved on first use).

Without the plug-in:

```
$ python -m mdb lanes
mdb lanes has no commands: the mdb-lanes plug-in is not installed (no 'mdb.lanes' entry points found). Engine verbs: engine, serve, lanes, version.
```

From the source tree (`W` is the repository root; `packages/mdb-lanes/src` carries a
`*.dist-info`, so the entry points resolve without an install):

```
$ PYTHONPATH=$W/src:$W/packages/mdb/src:$W/packages/mdb-lanes/src python -m mdb lanes
analyze
atlas
...  (46 verbs in total)
xref
$ PYTHONPATH=$W/src:$W/packages/mdb/src:$W/packages/mdb-lanes/src python -c "import mdb; print(mdb.BenchSession.__module__)"
mdb_lanes.bench
```

## Tests

```
W=/path/to/repository
cd $W/packages/mdb && env PYTHONPATH=$W/src:$W/packages/mdb/src OMP_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 python -m pytest -q -p no:cacheprovider tests/test_engine_trace.py
```

That path has no lane code on it on purpose. `tests/test_import_contract.py` checks that the
engine imports no lane module and that its layers (`core`, `engine`, `analysis`, `ops`, `repl`,
`serve`) import only downward; `import mdb` loads no torch.

## Known limits

- `attach` needs eager attention, one sequence (batch 1) and frozen weights in eval mode.
  `trace` has none of these limits but cannot step or edit.
- Port programs exist for Qwen2 and Qwen3 only.
- A checkpoint is bound to its numeric scope (thread count, determinism flag, matmul
  precision). Restoring under another scope is refused and names the differing field.
- Port stepping is not free. It halts native code at the cursor by re-running the layer from
  its input for every port. Measured on the tiny models, counting the calls of the layer and
  its sub-modules while stepping one layer port by port from `h` to the committed `h_out`:
  qwen2 145 calls (132 in 18 stopping probes, 13 in the commit) against 13 for one plain layer
  call; qwen3 188 against 15. `dbg.run(..., lookahead="layer")` probes each layer once and
  serves ports from that capture: 26 calls on qwen2, 30 on qwen3, at the price that native
  code has computed ports past the cursor (`run.status()["computed_past_cursor"]`).
- A stepped run is bitwise equal to the same incremental program without hooks. On the tiny
  models the last row of a full-prompt forward also matched the stepper bitwise at every port
  of layer 1; this is not guaranteed for larger matrices.

## Not built yet

- `trace(project=)`. A cell `whatif` is refused when the first affected row has not started,
  and saturn receipts list port edits only (`run.committed_cells()` has the cells).
- `export` (B1).
- Web views for symbols, circuits and predictions in a live session (the ops work through the
  console), and running the server as an mrun job.

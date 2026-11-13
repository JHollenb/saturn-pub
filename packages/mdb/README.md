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
$ python -m mdb engine --tiny qwen2 --tokens "1 5 9 3"
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
For a local Hugging Face checkpoint directory: `python -m mdb.repl --model DIR --tokens "1 5 9 3"`
(run here on a saved tiny model; it never downloads). `--prompt TEXT` instead of `--tokens` uses
the checkpoint's tokenizer (not run here: the tiny model has none).

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
the current clock) are different types. Origins (`branch:`, `@cut+edit:`) parse but are not
wired to stored values yet.

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
| `abort` | `abort` | mutate | Drop the staged edits of the pending layer |
| `advance` | `advance ADDR [budget=ROWS]` | navigate | Run to an address, stopping earlier at an armed breakpoint or watch |
| `backtrace` | `backtrace [ADDR]` | read | Producer chain from the focus port back to the layer input |
| `break` | `break PATTERN [if CONDITION]` | mutate | Set a breakpoint on an address pattern (L*.p, t5.L1.o, logits, cache.L0.v[row=2]); cache cells are evaluated at boundary stops only |
| `checkpoint` | `checkpoint [NAME]` | mutate | Name the cursor position (about 1 KB of plain data) |
| `checkpoints` | `checkpoints [offset=N] [limit=N]` | read | List named checkpoints (paged: offset=, limit=) |
| `circuit` | `circuit` | not built yet (A6) | Circuit records: nodes, edges, consumer, unlisted remainder |
| `commit` | `commit` | mutate | Commit the pending layer with its staged edits (cursor to its h_out) |
| `compare` | `compare` | not built yet (A6) | Compare branches (sham, A, B, A+B, predicted-only) |
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
| `frame` | `frame [ADDR \| cursor]` | navigate | Show or set the focus address (print/list/backtrace default to it) |
| `ignore` | `ignore N COUNT` | mutate | Ignore the next COUNT hits of breakpoint N (they are counted, not reported) |
| `info` | `info ports\|breakpoints\|watch\|display\|checkpoints\|edits\|program [offset=N] [limit=N]` | read | Show debugger tables or the program (lists are paged: offset=, limit=) |
| `list` | `list [LAYER]` | read | The layer's port program: operator text, proof IDs, cursor marked |
| `next` | `next [N]` | navigate | Step to the last port of the sublayer (r, then h_out) |
| `print` | `print [ADDR] [top=K] [slice=N]` | read | Reduce a value: shape, dtype, stats, top-k by \|value\|, pointer |
| `reads` | `reads` | not built yet (A6) | Who reads a value: attention reads with mass and causal verdict |
| `restart` | `restart NAME` | navigate | Return to a named checkpoint (rewinds or replays) |
| `reverse-continue` | `reverse-continue` | navigate | Run backwards to the previous breakpoint or watch hit |
| `reverse-finish` | `reverse-finish` | navigate | Step back to the boundary before the pending layer (its input) |
| `reverse-next` | `reverse-next [N]` | navigate | Step back to the previous sublayer end |
| `reverse-step` | `reverse-step [N] [unit] [budget=ROWS]` | navigate | Step back one port (or N units; layer/token units enter at most `budget` rows) |
| `run` | `run [TEXT \| ID ID ...] [budget=N]` | native | Like start, then continue to the first armed breakpoint or watch |
| `seek` | `seek` | not built yet (A6) | Seek to an occurrence through recorded projects |
| `set` | `set ADDR = VALUE\|ADDR [replace\|add\|scale]` | mutate | Stage an edit at a port of the pending layer: set ADDR = VALUE [op] |
| `start` | `start [TEXT \| ID ID ...]` | native | Prefill a prompt and stop at the pending token's first port (L0.h) |
| `step` | `step [N] [port\|sublayer\|layer\|token] [budget=ROWS]` | navigate | Step one port (or N units; layer/token units enter at most `budget` rows) |
| `symbols` | `symbols` | not built yet (A6) | Symbol bindings at the cursor with readers and lifetimes |
| `tbreak` | `tbreak PATTERN [if CONDITION]` | mutate | Set a breakpoint that deletes itself after its first hit |
| `undisplay` | `undisplay [ID ...]` | mutate | Remove displays (all when no ids are given) |
| `until` | `until ADDR [budget=ROWS]` | navigate | Run to an address, stopping earlier at an armed breakpoint or watch |
| `up` | `up [N] [PORT]` | navigate | Move the focus to a producer (toward the layer input), N times; `up PORT` picks the producer by name, the result lists the alternatives |
| `verify` | `verify [ID ...]` | mutate | Compare open forecasts whose target the cursor has passed with the native values (residual, bound, order) |
| `watch` | `watch ADDR [change\|level\|edge] [if PREDICATE]` | mutate | Watch an address: stop when it changes (or a predicate holds: level/edge); cache cells (cache.L<i>.k\|v) are evaluated at boundary stops only |
| `whatif` | `whatif ADDR=EXPR [replace\|add\|scale] consumer=M[a,b]\|logits[i]\|ADDR` | navigate | Predicted, then natively measured change of a consumer under one edit; the session is left exactly where it was |
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

## Recorded projects: `mdb serve`

`mdb serve --project DIR` serves a recorded project read-only on loopback, with no model and no
torch (it needs the `serve` extra). Run here on a copy of the fixture project that
`../mdb-lanes/tests/fixtures/serve_fixture_project.py` writes:

```
$ python -m mdb serve --project project --port 8799
mdb serve: http://127.0.0.1:8799/  (project project)
$ curl -s http://127.0.0.1:8799/api/v1/sessions
{"bytes":579,"data":[{"adapter":"native-boundary-v2","cuts":2,"family":"qwen2","granularity":"operation","home":"t2.logits","id":"qwen",...},{"adapter":"native-decoder-v1","cuts":18,"family":"qwen3","granularity":"layer","home":"t2.L0.h","id":"tiny",...}]...
$ curl -s http://127.0.0.1:8799/api/v1/hello
{"bytes":887,"data":{"cap":65536,"capabilities":{"graph":{"available":true,"reason":null},"live":{"available":false,"reason":"recorded project; no engine attached"},...},"readOnly":true,...
```

(Output cut at `...`; the project path is shortened.) The browser at `/` shows the project's
sessions with an address bar and a command line, and these views: Timeline (token by layer,
with a branch rail), Graph, Symbols, Records and an Inspector whose reductions (`POST
/reduce`) never send a whole tensor. Every response is at most 64 KiB. Live stepping in the
browser is not built.

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

- `trace(project=)`, cache-cell edits (`set cache.L.v[...]`) and what-if values taken from
  another run (`boiling:cache.L20.v[...]`).
- `symbols`, `reads`, `circuit`, `compare`, `seek` (A6) and `export` (B1).
- Live stepping in the web viewer (B2-B4) and remote workers over mrun.

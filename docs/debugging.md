# Local debugger

For qualified symbols, native-operation provenance, temporal trajectory comparisons, and
a complete repair demonstration, see [symbols and causal paths](debug-symbols-paths.md).

`saturn-pub debug --family ar` starts a random offline Qwen model. Use `--family diffusion`
for a random UNet/DDIM trajectory. The process owns one resident adapter, named branches, and
a standalone Observer. There is no remote attachment protocol or hosted service.

For executable cell-by-cell walkthroughs, start with the
[debugger notebooks](../notebooks/README.md). The notebook command parser is identical
to the terminal CLI. `next` aliases `step`, and `where` aliases `inspect`.

```text
break layer:1
continue
where
capture parent
next
restore parent
delete all
until layer:2 4
```

`break PATTERN` adds a boundary breakpoint; shell-style wildcards are supported.
`breakpoints` lists them, and `delete PATTERN` or `delete all` removes them.
With breakpoints installed, bare `continue` advances until the next matching boundary,
with a default limit of 1,000 transitions. It advances before checking, so continuing
from a breakpoint makes forward progress. `continue N` advances exactly N transitions unless a
watch fires; installed boundary breakpoints affect bare `continue` and `until`, not an explicit
budget. Without stops installed, bare `continue`
retains its one-transition behavior; with watches installed, it searches up to the default limit.
`until PATTERN [MAX_STEPS]` performs a bounded search independently of installed breaks.
Its result reports `breakpoint`, `watchpoint`, or `step-limit`; completed transitions and receipts remain
available when a limit is reached. Native adapter errors propagate and do not masquerade
as a breakpoint. The debugger has no Python-source breakpoints.

## Typed watchpoints and traces

Watchpoints inspect a public state address after every micro-transition. `change` compares
content fingerprints against a branch-local baseline. `gt` and `lt` fire if any numeric element
satisfies the stated threshold; `nonfinite` detects NaN or infinity. Restoring a cut resets that
branch's baseline at the restored safe point, while other branches retain their own baselines.

```text
watch hidden change
watch logits gt 20
watch hidden nonfinite
watchpoints
continue
unwatch watch-1
trace 4
```

A fire captures one retained StateCut shared by all predicates that fired at that safe point.
Each typed WatchEvent includes the watch specification, branch, cut fingerprint, execution point,
before/after summaries, and transition receipt. The cut fingerprint is also a debugger cut name,
so it can be restored or replayed. Fires are diagnostic triage events. They do not decide whether
a mechanism exists, a branch is useful, or a candidate should be promoted.

Observer is also usable without the command parser:

```python
from saturn_pub.observers import Observer

observer = Observer()
observer.add("hidden", "change")
observer.sync(session, "candidate")
receipt = session.step()
events = observer.evaluate(session, "candidate", receipt=receipt)
```

`Session.step()` advances one adapter transition. The command debugger uses it internally even
for bounded `continue`, so no intermediate watchable point is skipped. `Session.continue_(N)`
retains its atomic legacy contract for callers that want one published N-transition operation.
`trace N` returns the Frame, receipt, and any watch events for each completed transition.

An AR session:

```text
step 2
inspect
capture parent
fork candidate parent
use candidate
zero hidden
continue 2
read tokens
use root
fork native parent
use native
continue 2
compare candidate
use root
commit candidate
restore parent
save outputs/debug parent
quit
```

For the two-layer tiny model, two steps reach the carrier after layer 0. Two further steps
run layer 1 and native readout. `read hidden` prints the actual copied tensor values;
`inspect` prints shapes, dtypes, and content fingerprints. `compare` compares saved payloads.
Scientific metrics belong in an evaluator; a state difference alone is not a capability claim.

`add hidden 0.1` adds a scalar expanded to the current tensor geometry. `replace ADDRESS JSON`
requires the exact current shape and dtype. Quote a JSON array containing spaces.
Unknown addresses, read-only state, or invalid shapes are refused by the adapter.

`load DIRECTORY DIGEST NAME` loads a durable cut, verifies payload and compatibility, and
names it without modifying the current branch. `restore NAME` then restores it explicitly.
`saturn-pub inspect DIRECTORY DIGEST` verifies only the descriptor without loading torch;
it does not verify the tensor blob. Full verification occurs on load.

Put commands into a text file and pass `--script FILE` for a reproducible debugger session.
Script errors stop execution with a nonzero exit; interactive errors leave the debugger open.

For a narrated demonstration with actual measurements and durable replay, run:

```bash
python examples/debugger_replay.py
```

This writes `outputs/debugger-replay/walkthrough.md` and the complete `report.json`.
It steps the native layers, inspects hidden RMS and each layer's KV length, compares two-token
futures, commits and restores, then checks every declared state payload against a fresh-process
replay from disk. Model weights are reconstructed from the same seed; the cut stores execution
state and model identity, not the weights. The example is a random-model mechanics check.

The checkout includes `examples/debug_commands.txt`:

```bash
saturn-pub debug --family ar --script examples/debug_commands.txt
```

`abort BRANCH` records a discard decision without changing the current parent. The candidate
remains inspectable as evidence but is refused by commit.

`replay CUT [STEPS] [BRANCH]` forks and advances a named future without moving the selected
branch's cursor. `diff BRANCH_OR_CUT` compares it with the selected branch. `inspect` reports
the adapter's ExecutionPoint and SurfaceManifest as well as slot fingerprints. A custom JSON
adapter does not need a `.device` attribute to use debugger save/load.

All commands delegate to Session, Act, and LocalStore. The same debugger is usable from Python:

```python
from saturn_pub.cli import Debugger

debugger = Debugger(session)
debugger.execute("capture parent")
```

Run `python examples/software_debugger.py` for one broad parameter-free demonstration of
execution points, surfaces, watches, masked live-state writes, preservation checks, trace/diff/
replay semantics, measured programs, causal-path evidence, content-addressed deduplication, and
a data-bearing ReplayBundle restored into a fresh store. Its arithmetic graph is caller supplied;
the report makes no learned semantic claim.

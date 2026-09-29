# Local debugger

`saturn-pub debug --family ar` starts a random offline Qwen model. Use `--family diffusion`
for a random UNet/DDIM trajectory. The process owns one resident adapter and named branches.
There is no remote attachment protocol or hosted service.

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
from a breakpoint makes forward progress. `continue N` always advances exactly N
transitions. Without breakpoints, bare `continue` retains its one-transition behavior.
`until PATTERN [MAX_STEPS]` performs a bounded search independently of installed breaks.
Its result reports `breakpoint` or `step-limit`; completed transitions and receipts remain
available when a limit is reached. Native adapter errors propagate and do not masquerade
as a breakpoint. The debugger has no Python-source breakpoints or value-change watchpoints.

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

All commands delegate to Session, Act, and LocalStore. The same debugger is usable from Python:

```python
from saturn_pub.cli import Debugger

debugger = Debugger(session)
debugger.execute("capture parent")
```

# Local debugger

`saturn-pub debug --family ar` starts a random offline Qwen model. Use `--family diffusion`
for a random UNet/DDIM trajectory. The process owns one resident adapter and named branches.
There is no remote attachment protocol or hosted service.

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

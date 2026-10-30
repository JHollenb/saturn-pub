# Discovery debugger

MDB now has an executable `discovery-v1` interface for investigating recorded
native computations. The Python API, command-line console and bounded script
runner share `DiscoverySession.call()`. Existing debugger command semantics are
unchanged.

The qualified lane is frozen R3 Mamba on CPU FP32, one CPU thread. Native
capture completes first; the debugger then executes the original ATen operations
from input and parameter roots. A breakpoint pauses this replay. It does not
pause the first eager capture. Resident lanes execute sequentially, and the
capability response advertises no physical batch profile.

## Use it

With a saved root recipe from the qualification run:

```sh
mdb discovery replay /path/to/discovery-result/root-recipe \
  --script examples/discovery/post1.mdb --output /tmp/post1-investigation

mdb discovery replay /path/to/discovery-result/root-recipe \
  --interactive --output /tmp/post1-console

mdb discovery inspect /tmp/post1-investigation/journal --last 3
```

`inspect` verifies the immutable journal chain without importing Torch. Replay
requires Torch. A root recipe contains a sealed trace and root tensors; loading
it starts a fresh numerical replay at event zero. It is not a saved native
frontier. Keep scientific runs under mrun admission in the domains workspace.

The example script installs a generic alias and breakpoint, captures and reads
the stopped MLP output, resumes through a deferred action, restores the cut,
forks a zero-output branch, and executes its actual suffix. No ISA mnemonic or
semantic interpretation of the MLP is required:

```text
interface mdb-discovery-v1
alias mlp_focus = module:post.1
break mlp_focus exit --id post1_exit
on-hit post1_exit
    capture at_mlp
    inspect mlp_focus:output
    continue --defer
end
step forward
```

For every typed operation, scripts also accept
`call {"operation":"...","payload":{...}}`. Finite `repeat N` blocks are
expanded before execution. Handler motion must use `continue --defer`.
Errors retain completed actions; budgets and effect blocks prevent deferred
continuation. Handlers remain bound to their hit lane even if console focus
changes. Alias rebinding leaves installed breakpoints pinned to their original
site and alias revision.

## Choose the clock

| Unit | What completes |
|---|---|
| `op` | One native ATen primitive |
| `math` | A declared stage, such as normalization, affine projection or GELU, including its views |
| `law` | The MLP output, before residual application |
| `update` | The residual/state-update region |
| `forward` | All raw output heads, with no architectural commit |
| `instruction` | One explicit decoded register/flag/PC commit under the bound ISA driver |

`step`, `next`, `into`, `finish` and `until` use the discovered scopes. Stop
checks run between primitives, including during larger motions. An interrupted
law reports `completed_unit: false` and retains executed progress. Reverse
exploration restores a retained cut; it does not invert an MLP.

On the retained R3 program, `mlp_003` is `post.1`: entry 359, exit 369,
activation input `v000412`, output `v000430`. Its residual is a separate update
ending at 370. These coordinates come from module/dataflow metadata and bind to
that program; they are not a universal model address.

## Resident research lanes

```python
from mdb.discovery import DiscoverySession
from mdb.discovery_isa import ISADriver
from mdb.discovery_runtime import ReplayForward
from mdb.forward_recording import record_forward

session = DiscoverySession(budgets={"max_model_forwards": 8})
def factory(*roots):
    capture = record_forward(model, roots, model_identity=checkpoint_sha256,
        execution={"backend": "cpu", "dtype": "float32"},
        input_identity=input_digest(roots), max_operations=8192)
    return ReplayForward(capture)

session.add_lane("a", factory,
    driver=ISADriver("4881c701000000" * 2,
                     [0, 0, 0, 0, 0, 0, 3, 252], max_instructions=2),
    reference_forwards=1, reference_operations=8192)
session.call("break", {"selector": "module:post.1", "edge": "exit", "id": "p"})
session.call("motion", {"action": "continue", "all": True, "stop_policy": "lane"})
session.call("capture", {"name": "parent"})
session.call("fork", {"name": "trial", "cut": "parent"})
session.call("select", {"name": "trial"})
session.call("act", {"selector": "module:post.1:output",
                     "value_spec": {"kind": "zeros_like"}})
```

The caller supplies an already bound frozen model and input digest function.
The qualification launcher packages the actual R3 implementation and checks its
checkpoint/source hashes. Model weights are shared by the capture factory;
each replay owns its tensor environment. Capture admission reserves the declared
operation bound before invoking the factory. Independent audit calls need their
own physical-work accounting, as in the qualification harness.

Each lane has input epochs, branches, a revision and separate model/instruction
clocks. `replace_input()` creates a new epoch and retains old history; it does
not silently reuse prepared state. A focused lane can advance while another
stays parked. `run --all --stop-policy lane` visits the frozen target set in
order. Group policy parks later lanes on the first stop; parked group state
currently requires explicit restoration. Group handler continuation does not
automatically release parked lanes.

Raw bytes and before-state are the only ISA model inputs. The driver fetches
15 bytes with zero padding, resets internal hidden state each call, thresholds
predicted state at `> 0`, and advances PC using predicted length. `step forward`
leaves uncommitted heads available; `step instruction` consumes them without
another forward. Predicted stores/syscalls stop before commit under the default
policy. No RAM or operating-system effect executor is supplied.

## Evidence and current boundaries

The qualification harness is [discovery_isa.py](../qualification/discovery_isa.py).
It uses one finite mrun lease, installed public wheels, four actual inputs,
fresh native controls and a separate hooked native audit of an edited branch.
It also launches the installed command-line example in a fresh Python process.
The [final measured qualification record](../qualification/discovery-final-20261030.json)
retains all-head parity, physical work, coordinates and source digests. See
[validation details](../qualification/DISCOVERY-VALIDATION.md) for checks and
artifact custody.

Live cuts preserve tensor aliases, cursor, Act history and architectural state.
Root recipes support replay from original roots; metadata journals support
offline inspection. Paused frontier restoration across process restart is not
implemented. Only change watches and immediate atomic Acts are implemented;
Act preview/abort, arbitrary predicates, native eager pause and physical batch
refill remain future adapters. The larger
[interface design](ISA-DEBUGGER-INTERFACE.md) retains those future contracts.
The new dialect runs in its own worker or root-recipe console process; wiring
it into legacy remote `mdb debug JOB_ID` attachment is still open.

Generic names distinguish observed computations and routes. They do not prove
that a site implements a semantic instruction. Compare a branch's downstream
outputs and continuation, retain collateral effects, and use a fresh native
audit when interpreting a causal intervention.

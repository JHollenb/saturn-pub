# A working route-discovery session

For the research loop and how MDB, Saturn, mrun and the symbol table fit together,
start with [Discovering computations with MDB and Saturn](COMPUTATION-DISCOVERY-WORKFLOW.md).
This guide supplies the executed commands and recovery details.

Implementation: [`562209fc56d563cd345568c5ec04b10ff9656a87`](https://github.com/JHollenb/saturn-pub/commit/562209fc56d563cd345568c5ec04b10ff9656a87).
This example follows the executed R3 qualification `job-a61b5618e8db`.
It uses actual model predictions and tensor operations. The
[validation record](../qualification/DISCOVERY-UPGRADES-VALIDATION.md) gives
all-head comparisons, timings, work counts, failures and custody limitations.

The useful division of work is: screen many inputs with native batches; inspect
selected computations with scalar replay; fork from a common parent; compare
effects and continue using predicted state. Generic site names work throughout.
Semantic operation names are optional research annotations.

## 1. Bind the specimen and input

Use an environment installed from the implementation commit with the
`mdb-runtime[discovery]` extra, plus its public `saturn-pub` and `mrun-pub`
dependencies. The qualification launcher additionally needs the ISA checkout
to package its frozen model implementation and checkpoint. Workers import
installed wheels rather than sibling source trees.

The actual test input was:

```python
from mdb.discovery_isa import ISADriver

driver = ISADriver(
    "4881c7010000004881c701000000",
    [0, 0, 0, 0, 0, 0, 3, 252],
    max_instructions=2,
)
roots = driver.inputs()
```

The register order is RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI. Flags start zero.
The model receives byte bits `[1,120]`, register bits `[1,8,64]` and flag bits
`[1,4]`, in that order. There is no natural-language prompt or hidden chain of
thought. Each call fetches a 15-byte window and starts internal hidden state
afresh. A subsequent instruction consumes the driver's predicted registers,
flags and predicted byte advance. A long byte buffer alone does not make one
model call execute its entire program.

The qualified model is frozen R3 Mamba, CPU FP32, one CPU thread, checkpoint
`9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e`.
The launcher seals model, source, input and request identities.

## 2. Screen compatible native inputs

Run this inside the admitted worker after binding the frozen `model`, the four
`drivers` and a native `capture`, as the qualification harness does:

```python
from mdb.discovery_batch import BatchProfile, BatchRow, FrozenNativeBatch, OutputAct

profile = BatchProfile(
    model_identity=checkpoint_sha256,
    program_identity="r3-frozen-native-inputs-v1",
    input_shapes=tuple(tuple(value.shape) for value in drivers[0].inputs()),
    head_shapes={name: tuple(value.shape) for name, value in capture.output.items()},
    max_batch_size=4,
    allowed_modules=("post.1",),
    atol=1e-4, rtol=1e-5,
    qualification="exploratory-screen-with-scalar-controls",
)
screen = FrozenNativeBatch(model, profile, max_physical_calls=16, max_logical_rows=64)
rows = [
    BatchRow(
        lane_id=f"context_{index:02d}", epoch_id="input_00",
        branch_id="post1_half", expected_revision=0,
        roots=driver.inputs(),
        acts=(OutputAct("post.1", "scale", 0.5),),
    )
    for index, driver in enumerate(drivers)
]
result = screen.screen(rows)
```

This makes one physical B4 forward, then returns separately bound B1 output
heads and a sealed row-to-attempt mapping. The executed panel crossed initial
RDI 252–255 with output scales 0, 0.5, 1, 1.5: sixteen logical variants in four
physical calls. Its full outputs and margins are retained in `variants.json`.
The standalone batch qualification also exercised row reordering, parking,
input-epoch refill and row-specific Acts against scalar controls.

Check `errors`, `skipped` and `scalar_required`; retain every attempt receipt.
A row needing component stops is returned for scalar execution. Treat output
rows as immutable screening observations. Before acting on a result, verify its
lane/epoch/branch/revision still matches the intended branch. The adapter does
not automatically publish results or commit architectural state.

The measured all-head B1/B4 maximum absolute difference was 2.2888e-5.
This was within the declared numerical profile; exact B1/B4 or complete decoded
parity was not established. Shortlisted interventions receive scalar/native
audits. B4 intermediate values never become roots of a B1 recorded trace.

## 3. Stop at a computation and fork

The worker creates a `DiscoverySession` with a caller-bound `record_forward`
factory, then calls:

```python
session.call("motion", {
    "action": "until", "selector": "module:post.1", "edge": "exit",
})
```

For this frozen trace, the catalog resolves `post.1` to `mlp_003`, entry 365,
exit 375, output `v000436`. Those coordinates belong to the exact program.
The generic selector avoids carrying coordinates between capture profiles.
Capture completes before replay begins; this stop pauses real ATen replay.

Once the finite worker advertises readiness, this is the actual remote script:

```text
interface mdb-discovery-v1
status
alias mlp_focus = module:post.1
capture remote_parent
fork remote_zero
select remote_zero
call {"operation":"act","payload":{"selector":"mlp_focus:output","value_spec":{"kind":"zeros_like"}}}
step update
step forward
inspect outputs
step instruction
call {"operation":"close"}
```

Attach with `mdb discovery attach JOB_ID --endpoint URL --script FILE --output
NEW_DIRECTORY` while the worker is alive. Existing mrun client authentication
is reused. Workers request `saturn_debug_credential_v1` and receive narrow
worker credentials from mrun. Keep credentials out of scripts and logs.
The deployed agent needs the public scheduler URL explicitly supplied in the
sealed launch command; the qualification launcher handles this.

`step update` completes the residual region; `step forward` produces raw heads;
`step instruction` commits the driver's decoded predicted state. They are
separate clocks. `step op`, `step math` and `step law` provide smaller scopes.
The existing [debugger guide](DISCOVERY-DEBUGGER.md) shows entry/exit breakpoints,
change watches and bounded on-hit handlers for longer sessions.

The executed remote branch matched a separate native zero-output hook on all
11 heads exactly. Its architectural commit moved RDI 252→253 and byte offset
0→7. This is one predicted instruction, not evidence that zeroing the MLP is
irrelevant in other contexts. Stores and syscalls still stop before commit;
MDB supplies no RAM or operating-system effect executor.

## 4. Save and resume the exact selected frontier

Before remote attachment, the qualified worker saved its current branch:

```python
from mdb.discovery_storage import save_session_frontier

saved = save_session_frontier(session, "research", "upgrade-result/store")
```

The equivalent console action is `save-frontier PATH`. The store path is on the
worker's filesystem. Collect the complete store before that worker disappears.
Default `bundle-v1` preserves unique backing storage plus view geometry,
cursor, Acts, ISA driver state, selected ownership, metadata and work counters.
It uses public StateCuts and Safetensors; restore executes no serialized code.

The actual selected session cut was
`f490fb444951660a4d2b2908264ebedb9b04666df77d4b257ee1b0fb2a05e12f`.
The collected store lives at:

```text
artifacts/mdb-discovery-upgrades-20261030/run-04/extracted/upgrade-result/store
```

Metadata inspection works without Torch:

```sh
mdb discovery inspect STORE --cut CUT_ID
mdb discovery inspect CLIENT_OUTPUT/cli/journal --last 3
```

The fresh-process resume script actually executed was:

```text
interface mdb-discovery-v1
status
inspect state
step forward
inspect outputs
step instruction
```

Invoked as `mdb discovery resume STORE CUT_ID --script FILE --output NEW_DIRECTORY`,
it resumed at PC375, ran only 205 suffix primitives and exactly matched all 11
native heads. It then committed the predicted instruction. No prefix or
`model.forward` was rerun in that child process.

That child ran on the same Linux CPU platform as the saving worker. The durable
profile checks Torch version, CPU architecture and endianness, and restores the
saved thread setting. The collected store can be inspected on the Mac; numerical
continuation must run on a compatible admitted worker. Cross-architecture replay
is not established by this result.

By default, elapsed and offline wall time count against the saved allowance.
An old cut may therefore be inspectable but require new admission to advance.
Inside a newly admitted mrun continuation worker, explicitly supply:

```sh
mdb discovery resume STORE CUT_ID --admission-id "$MRUN_JOB_ID" \
  --budget-json continuation.json --script resume.mdb --output NEW_DIRECTORY
```

`continuation.json` supplies all absolute finite ceilings:

```json
{"max_branches":16,"max_commands":128,"max_instructions":1000,
 "max_model_forwards":8,"max_native_operations":131072,"max_wall_ms":60000}
```

Historical work is retained and must fit those ceilings. The CLI can finish the
saved forward and commit its heads. Starting another model forward requires the
Python `resume_session_frontier(..., factory=qualified_factory)` API: the caller
must bind its model implementation explicitly. Model/program/execution and
actual new input roots are checked. Other branches, old live cut handles and
the old worker's retry cache are not reconstructed.

## 5. Reproduce the executed qualification

From the domains workspace, the tested control environment was
`/private/tmp/mdb-discovery-upgrades-installed-20261030/bin/python`.
Substitute an environment installed from the pinned commit when reproducing
elsewhere. Use a new run directory for each attempt:

```sh
common/bin/common exec env MRUN_URL=http://192.168.68.68:9025 \
  /private/tmp/mdb-discovery-upgrades-installed-20261030/bin/python \
  mdb/qualification/discovery_upgrades.py --isa-root /Users/jakeholl/domains/isa \
  --run-dir /tmp/mdb-example-run
```

The launcher returns a real job ID. Poll that ID with
`--run-dir /tmp/mdb-example-run --collect JOB_ID`. When it reports `ready:true`,
drive the session within its 90-second service allowance:

```sh
common/bin/common exec env MRUN_URL=http://192.168.68.68:9025 \
  /private/tmp/mdb-discovery-upgrades-installed-20261030/bin/python \
  mdb/qualification/discovery_upgrades.py --drive JOB_ID \
  --run-dir /tmp/mdb-example-run-client --endpoint http://192.168.68.68:9025
```

Then collect the same ID again after termination. `--drive` checks same-ID
retry/content-conflict behavior, invokes the installed remote CLI and closes
the worker. It does not synthesize model outputs. The worker independently
audits the edited branch before publishing its report/archive. The recorded
job is already closed; attaching to that historical ID cannot recreate it.

## 6. Turn observations into table records

Retain model/program/site IDs, input digest, context, common-parent cut,
intervention, physical attempt/row, complete head response, decision margins,
collateral and predicted-state continuation. Use generic aliases immediately.
Keep a site identity separate from the many context-qualified computations
observed at that site. Do not give all occurrences one inferred semantic name.

Current variants show that `post.1` reaches five heads, and its intervention
responses vary across contexts even when decoded decisions stay unchanged.
The [matched three-MLP follow-up](THREE-POST-MLP-CAMPAIGN.md) now compares the
other post MLPs and uses a final-mixer operand restoration to separate responses.
The [bottleneck and discovery plan](DISCOVERY-BOTTLENECKS.md) specifies how to
choose those experiments and when graph algorithms help.

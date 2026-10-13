# MDB experiment walkthroughs

Start with rewind and replay, then change one part of the state from the same
parent. MDB retains the original future, the candidate and the evidence needed
to repeat the comparison. These examples use the standalone MDB runtime;
real-model jobs also use installed mrun.

Commands below start in `packages/mdb` in the Saturn repository with MDB installed. See
[installation](INSTALL.md) and the [usage guide](USAGE.md). Real models run in
an admitted mrun worker. `BenchSession.from_resident` wraps the model already
loaded there and creates no additional lease.

| Experiment | Question | What to inspect |
|---|---|---|
| Layer rewind and replay | Does an in-flight token continue identically from a saved cut? | Token IDs, logit digests, cursor and compatible final cuts |
| Residual intervention | Does scaling one residual change the native continuation? | Abort closure, signed logit change, tokens and commit receipt |
| Three sibling futures | Which state is shared and which becomes private? | Owners, backing pages, logical bytes, resident bytes and parent preservation |
| Circuit measurement | Where does a residual perturbation affect state and a declared output? | Residual/memory deltas, signed token-margin effects and controls |
| Fitted J-space reader/writer | Does a sealed lens read the exact residual, and does a native write affect the later consumer? | Native readout parity, signed/dose controls, common-parent Acts, successor K/V transfer and exact reinstall |
| Diffusion block replay | Can we resume a denoiser block and reproduce the final image? | Actual scheduler/RNG closure, image digest, latent and block state |
| Portable export | Can another process continue without the original session directory? | Export byte audit, exact identity and resumed suffix |
| Paper E6/E20 | What can sparse features control, and does a residual seed act through formed K/V? | Frozen native protocols, same-run comparison, typed input cuts, controls and custody |

For the paper panels, use the dedicated [E6/E20 guide](../experiments/time_circuits/README.md).
E20 uses Qwen's FP32 native suffix; E6 uses a separately pinned Gemma
circuit-tracer consumer. Their research input cuts do not extend the public
BenchSession family or numerical contracts.

For J-space, use the [reader/writer API](JACOBIAN-WORKSPACE.md) and
[proof runbook](../experiments/jacobian-reader-writer/README.md). Tiny mechanics,
installed-wheel qualification and fitted real-model effects have separate records.

## Run the local mechanics demonstration

This creates a tiny random CPU model. It needs the autoregressive extra, no model
download and no scheduler. Vocabulary IDs have no semantic meaning here. Each
output directory must be new.

```sh
python -I "$PWD/examples/resident_lab.py" --tiny qwen2 --out ./results/lab-qwen
python -I "$PWD/examples/resident_lab.py" --tiny mamba --out ./results/lab-mamba
mdb verify --root ./results/lab-qwen/export
```

The program runs layer replay, a residual preview/abort/commit, three siblings and
a circuit panel. The last line reports `history_exact: true`,
`circuit_replay_exact: true` and `mechanics_status: "valid"`. Its `report.json`
retains raw rows, receipts, cuts, full memory inspections and circuit controls.
`session/` holds the original state; `export/` is independently byte-verifiable.
This demonstrates the control flow, not a reasoning circuit in a trained model.

The functions in [resident_lab.py](../examples/resident_lab.py) also accept an
already prepared real session. Load them without changing the runtime import path:

For a custom mrun worker, include `resident_lab.py` in its declared job payload;
repository examples are not installed as runtime modules in the MDB wheel.

```python
import runpy
lab = runpy.run_path("examples/resident_lab.py")
# Inside your admitted worker, after session.prefill(...):
result = lab["rewind_and_replay"](session, tokens=4)
```

## Layer rewind and replay on Qwen or Mamba

For a real finite CLI experiment, preview the launch before submitting it. Use a
new absolute directory on the selected worker's durable filesystem. The example
path below is the qualified fleet's ext4 volume; choose the appropriate durable
path on another fleet.

On the qualification fleet, select its physically materialized ext4 dependency
cache before the launch:

```sh
export MDB_WORKER_UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache
```

This path is on the worker host. Other fleets need their own available offline
cache; see [installation](INSTALL.md) for the cache contract.

```sh
mdb open --model qwen2.5-0.5b-instruct \
  --prompt 'The capital of France is' --lane virtual --capacity 16 \
  --root /mnt/hdd8tb/mdb/demos/qwen-history-01 \
  --commands examples/history-commands.json --dry-run

mdb open --model qwen2.5-0.5b-instruct \
  --prompt 'The capital of France is' --lane virtual --capacity 16 \
  --root /mnt/hdd8tb/mdb/demos/qwen-history-01 \
  --commands examples/history-commands.json
```

The CLI returns `job_id`, `root` and `request_sha256`. It submits one resident
worker and returns before the experiment completes. The script stops after the
first decoder layer, saves a common parent, generates four tokens on `baseline`,
then returns to `main` and generates the same budget on `replayed`. It finally
inspects memory, source frames, symbols and timing. A checkpoint label is a
description; the returned `cas://sha256/...` reference is the restore address.

Collect the completed job using its actual ID:

```sh
python -I "$PWD/examples/collect_session.py" JOB_ID --out ./results/qwen-history-01
```

The collector returns `collected: false` and exit code 2 while the job is still
running; collect again after completion. It preserves failed job logs too and
refuses to overwrite an existing result directory. It saves `job.json`,
`worker.log`, `execution-environment.json` when present and `report.json` when
the worker published a report. It does not download the worker's session store.

In `report.json`, each `results` row records `session_id`, `cut_id`, `op` and
`result`. Compare the two `generate` results' `token_id` and `logits_sha256`
sequences. They should be equal within the same numerical lane. For automatic
assertion of partial-cut replay, use `lab["rewind_and_replay"](session)` inside
your worker; it calls `restore`, `replay` and `compare` directly.

Replace the model with `mamba-130m` for Mamba v1. Qwen's `--lane native` is useful
for a separate native comparison. Keep the prompt and token budget equal and
use a new session root. Native and virtual Qwen outputs can differ; matching
within a lane is the qualified replay contract.

To continue the selected `replayed` branch in a new resident worker:

```sh
mdb resume --model qwen2.5-0.5b-instruct \
  --root /mnt/hdd8tb/mdb/demos/qwen-history-01 \
  --commands examples/resume-commands.json
```

Use resume after the first writer has closed. The session identity stays the
same; the job identity changes. Resume reads the stored lane and numerical
contract, so passing a different `--lane` does not convert a saved session.

## Preview and commit a residual intervention

Inside an admitted Qwen or Mamba worker with a prefilled session:

```python
result = lab["residual_intervention"](session, scale=0.9, tokens=4)
print(result["common_parent"])
print(result["abort_comparison"])
print(result["final_logit_delta"])
print(result["commit"]["receipt_ref"])
```

The helper pauses after one layer and retains a baseline branch. On a second
branch it previews a 10% residual reduction, aborts it and checks the restored
closure. It then previews the same edit, commits through Model OS and executes
four native token transitions. Both branches start at the same partial cut.

`final_logit_delta` contains L2, signed mean and maximum absolute change at the
final continuation step. Once generated tokens differ, later logits belong to
different contexts; this is a total continuation response, not an isolated
same-input causal estimate. Use the circuit panel's fixed inputs for that question.
Unchanged tokens or a zero effect remain observations. Retain no-op, random,
wrong-site and collateral controls before making a semantic claim.

For a minimal manual transaction:

```python
session.step("layer")
parent = session.checkpoint("before-edit")
preview = session.preview_edit("residual", 0.9, mode="scale")
session.abort_edit(preview["act_id"])
preview = session.preview_edit("residual", 0.9, mode="scale")
receipt = session.commit_edit(preview["act_id"])
session.generate(4)
session.restore(parent)
```

`preview_edit` stages an admitted payload; it does not itself run a future.
Execution waits for commit or abort. Non-finite, wrong-shape, unavailable-boundary
and undeclared edits are refused. Qwen KV pages are not writable through this
residual interface; the circuit instrument has a separate typed successor-KV
Act at committed token boundaries. Mamba formed recurrent-state edits have their
own native-consumer validation.

## Inspect three sibling futures and virtual memory

```python
result = lab["three_siblings"](session, tokens=2)
memory = result["memory_after"]
print(memory["unique_allocations"])
print(result["parent_comparison"])
```

The helper forks three private continuations from one retained parent and verifies
that the parent still compares exactly. Read the memory table's ownership and
backing links rather than adding every branch's visible bytes: shared ancestors
can appear in several logical views while occupying one allocation.

Qwen's memory report includes `branches`, `row_bytes`, `visible_bytes`,
`total_kv_bytes`, `budget_bytes`, `unique_allocations`, `backing_pages`,
`vm_events`, `evictions` and `hydrations`. Mamba and diffusion report
`visible_bytes`, `resident_bytes`, `unique_backed_bytes`, `unique_allocations`,
page `owners`/`views`, `events`, `evictions` and `hydrations`. Their schemas differ
because KV rows, recurrent state and image transients have different closures.
`weights_bytes` is separate. CUDA workspace telemetry includes weights and live
state, so it must not be added to those components as a separate allocation.

Choose `resident_kv_bytes` for Qwen or `resident_state_bytes` for other families
at `from_resident`, or `--kv-budget-bytes` / `--state-budget-bytes` in the CLI.
Start with a roomy budget, measure active and retained allocations, then use a
new session with a budget that fits one active closure but pressures inactive
siblings. An undersized budget can refuse admission. Eviction is not guaranteed
by merely creating three branches. Logical backed pages still assemble native
tensors; this is not hardware CUDA VMM.

For the already demonstrated forced-eviction and source-deleted continuation
panels, see [qualification](QUALIFICATION.md) and its compact memory records.

## Measure residual to memory to future response

Use a committed token boundary and journal publication. The current instrument
supports Qwen and Mamba memory mediation. Declare
two distinct vocabulary IDs rather than assuming that a word occupies one token:

```python
def one_token(text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Choose a single-token alternative: {text!r} -> {ids}")
    return ids[0]

spec = {
    "name": "capital-response",
    "sites": [0, 6],
    "contrasts": [{"name": "weaken", "mode": "scale", "value": 0.9}],
    "observable": {"kind": "token_margin", "token_id": one_token(" Paris"),
                   "against_token_id": one_token(" Rome")},
    "query_token_ids": [],
    "continuation_tokens": 2,
    "memory_mediation": False,
    "max_native_operations": 4096,
    "max_capture_bytes": 64 * 1024 * 1024,
    "seed": 0,
}
result = lab["circuit_panel"](session, spec)
report = result["measurement"]["report"]
print(result["measurement"]["report_ref"])
for row in report["rows"]:
    print(row["site"], row["role"], row["metrics"])
print(result["candidate_replay"]["exact"])
print(session.inspect("measurements"))
print(session.inspect("symbols"))
```

Site 0 is the embedding residual; site 6 is after six decoder layers. Every site
runs same-parent native, weakened-residual, norm-matched random, no-op and
residual-restore controls. The observable is the raw logit margin
`logit[Paris] - logit[Rome]`; its signed effect is the arm's margin minus the
reference margin. It is not an answer accuracy score or probability.

The report retains the residual trajectory, residual and successor-memory
differences, immediate and future signed effects, closure digests, cut links,
typed Act receipts and replayable evidence references. Capture and native-operation
budgets bound the panel. Strong, weak, zero and counter-direction effects all
remain in the record. The helper replays one candidate and checks exact closure.

For Mamba's state-mediated path, supply a fixed nonempty query input, for example
`query_token_ids = tokenizer.encode(". The country is", add_special_tokens=False)`,
and set `memory_mediation = True`. Keep the query within 32 tokens and increase
the declared operation budget if the conservative work bound requires it. These
tokens are supplied inputs, separately labeled from native greedy continuation.
The panel transfers/restores complete recurrent pages and flags while fixing
recipient input. Inapplicable rows retain their reason; they do not imply no effect.

Qwen uses the same fixed-query authority with its native BF16 CausalContextVM.
Omit `memory_selector` to transfer the complete visible successor K/V, or select
one bounded carrier with `{"layers": [6], "row_start": 12, "row_stop": 13,
"components": ["value"]}`. The half-open row span is over committed logical
history. The Act binds checkpoint/numerical ABI, complete geometry, history,
selector and pending-token policy; it privately stages the replacement and proves
exact masked locality before one atomic commit. A complete selection may reproduce
the paired future exactly. A partial selection is measured as a signed native
output and need not reproduce the complete candidate future. This operation does
not admit FP32 TokenMachine state into the BF16 lane.

This quantifies a prompt-bound state response. Hyperion may bind that measured
response to its context, but `semantic_binding` remains false. It does not count
reasoning circuits or certify the concept “capital.” Inspecting `frames` connects
retained source scopes to execution cuts; the logical frame stack is distinct
from a CPU or CUDA stack.

## Resume a diffusion block and save the image

Use a finite CLI script for either qualified image family:

```sh
mdb open --model flux2-klein-4b-distilled \
  --prompt 'A red cube on a white table.' --seed 17 \
  --height 512 --width 512 --steps 4 --guidance-scale 1.0 \
  --root /mnt/hdd8tb/mdb/demos/flux-block-01 \
  --commands examples/diffusion-commands.json
```

Replace the model with `illustrious-xl` for SDXL. The script retains a first-block
cut and runs through native VAE execution and image commit. Collect the job with
`collect_session.py` as above. Use a new root for each prompt/seed case.

For a caller-owned pipeline inside your admitted worker:

```python
from mdb import BenchSession
from PIL import Image

with BenchSession.from_resident(pipeline, root="./image-session") as session:
    session.prepare("A red cube on a white table.", seed=17, height=512,
                    width=512, num_inference_steps=4, guidance_scale=1.0)
    result = lab["diffusion_replay"](session)
    rgb = session.resolve_value("image")[0].float().clamp(0, 1).cpu().numpy()
    Image.fromarray((rgb * 255).round().astype("uint8")).save("image.png")
    session.export("./image-export")
    print(result)
```

The helper finishes from a block cut twice and requires equal image digests and
compatible final state. It restores the captured timetable, scheduler state,
RNG and transient block closure directly. Reusing only the seed is insufficient.
FLUX retains image/text streams; SDXL retains UNet skip and transient inputs.
The final `image` address is available after VAE and transaction commit.
`generate()` follows the prepared schedule; it takes no token budget.

## Export and recover without the source session

Inside a worker with the resident model and tokenizer still available:

```python
parent = session.checkpoint("portable-parent")
session.export("./portable-export")
expected = session.generate(4)
session.close()

from mdb import BenchSession
with BenchSession.resume(model, tokenizer, root="./portable-export") as restored:
    actual = restored.generate(4)
    assert [(r["token_id"], r["logits_sha256"]) for r in actual] == [
        (r["token_id"], r["logits_sha256"]) for r in expected]
```

This demonstrates an independent store in the same process. For fresh-process
continuation, close the writer, reload the exact model and environment in the next
admitted worker and resume the export there. The full qualification records also
demonstrate continuation after deleting the original private source store. Do not
delete your sole copy merely to try the API. The export includes retained tensor
closure and cuts, while model weights remain independently available.

`mdb verify --root ./portable-export` can audit the export without loading weights.
Run it where that directory is accessible; controller and worker paths differ.
Byte verification is separate from numerical compatibility and native continuation.

## Use the debugger console when the broker is available

Finite scripts and the Python examples above work without a live debugger broker.
The attach workflow needs deployed mrun debugger routes and worker credentials;
that deployment remains separately unqualified. On a fleet that provides them:

```sh
mdb open --model qwen2.5-0.5b-instruct --prompt 'The capital of France is' \
  --root /mnt/hdd8tb/mdb/demos/qwen-debug-01 --hold-seconds 300
mdb attach JOB_ID --endpoint http://YOUR_SCHEDULER:9025
```

The worker starts paused. In the console:

```text
status
capabilities
breakable
break layer 1
step layer
print residual --stats
inspect memory
frames
checkpoint before-edit
set residual 0.9 scale
preview
abort
cuts
```

Use `commit` after a new preview when you intend to keep the candidate, then
step the native continuation. `restore CUT_ID` navigates to the actual retained
cut reference printed by `cuts`. Check available boundaries and writes through
`capabilities`; family adapters determine what is valid at the current cursor.
The live window is bounded by `--hold-seconds`, and leaving out `--commands`
selects this workflow. No fleet refresh is needed for the finite examples.

## Preserve the result and sweep the evidence

After collecting a completed job locally:

```sh
mdb custody take ./results/qwen-history-01/report.json --job-dir ./results/qwen-history-01
mdb custody take ./results/qwen-history-01/worker.log --job-dir ./results/qwen-history-01
mdb custody verify --job-dir ./results/qwen-history-01
mdb evidence run ./results/qwen-history-01 --xref-root ./results/qwen-history-01
mdb evidence verify ./results/qwen-history-01
```

Also take custody of model/source identities, configuration, environment receipt,
images, export archives and requested tensor payloads before scratch storage is
removed. Do not rewrite the historical result when adding interpretation.
Evidence exit code 0 means a clean sweep, 3 signals a finding/drift to investigate,
and 2 signals broken/refused or incomplete input. `target-streams-only` means
the tool indexed streams without recognizing scientific receipts; it certifies
neither circuit quality nor an answer improvement.

## Follow up with a bounded mechanism experiment

To investigate changing scaffolding, repeat the circuit panel at several token
checkpoints and a small set of sites. Keep the output observable and intervention
dose fixed, retain the residual/state trajectories, and compare signed effects
across prompt contrasts. Then test another dose and a wrong-site or wrong-time
control. Qwen's weights and decoder graph remain fixed during inference; execution
state and effective causal contribution can change across forwards.

The useful result is a measured map of where and when a perturbation reaches
memory and the native output. A consistent state change with a zero output effect
is informative too. Generalizing a semantic circuit requires independent contexts,
an appropriate answer consumer and collateral observations beyond this walkthrough.

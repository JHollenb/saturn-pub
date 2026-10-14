# Start a local MDB debugger

From the repository root:

```sh
python -m pip install ./packages/mdb
mdb doctor
mdb help
mdb debug
```

`mdb debug` opens the existing debugger shell on a pure-Python Counter. There
is no job, scheduler, token, download or Torch requirement. Its integer state
and immutable cuts live in memory for this session. It teaches the debugger
commands; it does not run a trained model or produce a native model export.

Try this sequence in the console:

```text
checkpoint parent
break step 3
continue
inspect value
fork candidate
set value 10
preview
abort
restore parent
watch value >= 2
continue
reverse-continue value < 2
quit
```

The startup view includes six lines of command help. Each command prints a
bounded human result and the current frame, for example:

```text
(main) phase=counter step=3 value=3 cut=53ef96… stop=breakpoint
```

`inspect --full` prints the complete response; `inspect --json` prints a
machine-readable response for that command. Global `--json` prints command
responses without banners or extra frames; global `--full` preserves verbose
responses. `help` lists all commands. Capabilities are adapter-specific:
Counter has `step` boundaries and a writable `value`; native Qwen has
layer/token boundaries and residual/KV state.

## Tiny native Qwen, without a model download

Install the model extra, then start the same shell:

```sh
python -m pip install './packages/mdb[autoregressive]'
mdb debug --tiny qwen2 --root ./tiny-session
```

This creates two tiny layers with deterministic random CPU BF16 weights and
input IDs `[1,2,3]`. The weights are untrained and have no semantic vocabulary.
The first layer frame looks like:

```text
(main) phase=layers next_layer=1 tokens=3 pending=23 cut=53ef96…
```

Pending IDs and hashes depend on the runtime. Try:

```text
checkpoint parent
break layer 1
continue
inspect residual
set residual 0.9 scale
preview
abort
restore parent
step token 1
export ./tiny-export
quit
```

Motion checks the existing worker's breakpoints and watchpoints at actual
native boundaries. Each native transition retains a cut. `continue` generates
at most 32 tokens by default; `--max-new-tokens` changes that limit and
`--max-steps` bounds transitions per motion command. A pending edit must be
committed or aborted before execution. `reverse-continue PATH PREDICATE` walks
retained parent cuts and restores the original cut when no match is found.

The native root persists after exit. Resume with the same seed and environment:

```sh
mdb resume --tiny qwen2 --root ./tiny-session
mdb verify --root ./tiny-export
```

Offline verification checks exported bytes and mechanics without loading
weights. Restore and model execution still require matching weights, runtime
identity and numerical contract. This local mechanics demo adds no new
real-model qualification; the [qualification record](QUALIFICATION.md) names
the certified checkpoints and toolchains.

For a finite script, use `--script FILE`, or `--command 'step layer 1'` for one
command. `--export DIR` exports a native session after successful commands;
`--record FILE` saves a debugger receipt, including Counter sessions.

## Open an existing local checkpoint

```sh
mdb open --local /absolute/path/to/hf-checkpoint \
  --root ./session --prompt 'Explain gravity.'
mdb resume --local /absolute/path/to/hf-checkpoint --root ./session
```

`mdb debug --local` is the same entry point. The loader supports local
Hugging Face Qwen2/Qwen2.5 and Mamba v1 directories with `config.json`, weights
and tokenizer files. It uses local files only and disables remote code.
For a checkpoint without tokenizer files, use `--token-ids '1,2,3'` on first
open. Resume uses retained input and takes no new prompt or IDs.

The default device is CPU. `--device` selects an explicit caller-owned device.
Qwen requires BF16 and reports `use --dtype bf16` before loading incompatible
weights; the Python equivalent is `model.to(torch.bfloat16)` before opening a
resident session. Mamba defaults to FP32. The loader does not accept arbitrary
pickled `.pt` files. An unsupported architecture needs a native stateful adapter,
even if its framework supports forward hooks.

For other executors, use the [public backend interface](BACKENDS.md):
`ModelRunner` handles forward, capture and scoped interventions, while an
optional `DebuggerAdapter` declares stepping, checkpoint and edit semantics.
TransformerLens hooks can supply forward-time interventions; activation capture
alone does not establish resumable state or exact replay.

## Local diagnostics and managed jobs

`mdb doctor` reads installation metadata without importing model frameworks or
contacting a scheduler. `mdb doctor --endpoint URL` optionally checks the
scheduler's health; `mdb doctor JOB_ID` retains attached-job diagnostics.
The local CLI and `mdb debug JOB_ID` share one shell and protocol.

`mdb open --model MODEL` and job attachment retain mrun's acquisition, admission
and custody path. In the domains workspace, real-model scientific runs still
use admitted mrun workers. The local console supplies the caller-owned entry
point and tiny mechanics checks.

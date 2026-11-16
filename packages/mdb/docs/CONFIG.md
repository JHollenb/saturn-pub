# Configuration

mdb reads its defaults from TOML files and `MDB_*` environment variables, so the dtype, device, project, model locations, record sink and REPL defaults do not have to be typed on every command. A config file holds locations and machine defaults only. Settings that change a verdict stay in the call, where the script that custody hashes shows them (see [What a config file may not set](#what-a-config-file-may-not-set)).

```sh
mdb config init                  # write a commented ~/.config/mdb/config.toml
mdb config init --project runs/x # write runs/x/mdb.toml
mdb config show                  # every setting, its value and where it came from
mdb config explain model.dtype   # each source that sets it, highest first; the first wins
mdb config check                 # validate the files, the environment and repl.defaults
```

## Where settings come from

Highest precedence first:

| Layer | Where |
|---|---|
| 1. Command line | `mdb engine --dtype fp32`, `mdb serve --port 9000`, ... |
| 2. Environment | `MDB_MODEL_DTYPE=fp32` and the other variables in the [key reference](#key-reference) |
| 3. Named file | the file `MDB_CONFIG=PATH` names (it must exist) |
| 4. Project file | `<project>/mdb.toml` when a project is named (`--project DIR` or `MDB_PROJECT_DIR`); otherwise the first `mdb.toml`, or `pyproject.toml` with a `[tool.mdb]` table, found walking up from the working directory. The walk stops at the repository root (a directory with `.git`) or your home directory. |
| 5. User file | `$XDG_CONFIG_HOME/mdb/config.toml`, by default `~/.config/mdb/config.toml` |
| 6. Built-in defaults | listed below |

Each setting is resolved on its own: a project file that sets only `[model] dtype` leaves every other setting to the user file and the defaults. In a config file, relative paths are relative to the file's directory, and `~` is expanded. On the command line and in the environment, relative paths are relative to the working directory.

`mdb engine`, `mdb serve`, `python -m mdb.engine.selfcheck` and `mdb config` resolve settings when they start. Python code uses the same settings, with the same precedence, the first time mdb needs one. An argument you pass always wins. An argument you leave out (`None`) takes the configured value:

| Python call | Settings it reads when the argument is left out |
|---|---|
| `mdb.engine.models.load_local(path, dtype=, device=)` | `model.dtype`, `model.device`; `path` may be a name, root model or alias (`model.roots`, `model.aliases`) |
| `mdb.engine.models.tiny_model(family, dtype=)` | `model.dtype` |
| `mdb.open_tiny(family, dtype=, device=, project=, session=)` | `model.dtype`, `model.device`, and `attach`'s settings |
| `mdb.attach(model, tokenizer, project=, session=)` | `project.dir`; `project.session` when a project is attached. `project=False` attaches no project whatever the config says. |
| `custody.declare(kernel_path=)`, `dbg.custody(kernel_path=)` | `custody.kernel_path` |
| `custody.records_to(path, append=, payloads=)`, `dbg.records_to(...)` | `records.sink`, `records.append`, `records.payloads` |
| `sae load NAME` (REPL and op) | `sae.sources` |

Other functions are unchanged. The `kernel_path` of `Branches` and `forecast.verify` names the kernel that measured the arms, which is different from the custody label, so it is never read from a config file.

## Example

```toml
schema = "mdb-config-v1"

[model]
roots = ["/data/models"]
aliases = { q15 = "Qwen/Qwen2.5-1.5B", demo = "tiny:qwen2" }
dtype = "bf16"
device = "cuda"

[project]
dir = "runs/main"

[records]
sink = "records/"            # a directory: one records-<run>.jsonl per process

[custody]
kernel_path = "stepped"

[serve]
port = 8800
allow_hosts = ["mdb.lan"]

[repl]
init = ".mdbinit"
history = "~/.local/state/mdb/history"
defaults = { print = { top = 8 }, x = { count = 32 }, step = { budget = 16 } }

[storage]
shared_tensor_store = "/data/mdb-store"

[sae]
sources = { l12 = { path = "~/sae/qwen-l12", k = 50, layers = "0:28" } }
```

The same tables can live in `pyproject.toml` as `[tool.mdb.model]`, `[tool.mdb.records]`, and so on.

## Key reference

| Key | Type | Default | Environment | What it does |
|---|---|---|---|---|
| `model.roots` | list of paths | `[]` | `MDB_MODEL_ROOTS` (`:`-separated); older `LLM_MODELS_EXTRA_ROOT` | Directories searched for a model by name, together with the Hugging Face cache. A checkpoint is a directory with a `config.json` one or two levels down (`ROOT/NAME` or `ROOT/ORG/NAME`). If the same id is in the cache and a root, the cache wins. |
| `model.aliases` | table of text | `{}` | none | Short name to a checkpoint directory, a cached or rooted model name, or `tiny:FAMILY`. An alias names one target; an alias of an alias is refused. |
| `model.dtype` | `bf16` or `fp32` | `bf16` | `MDB_MODEL_DTYPE` | Weights dtype for `mdb engine`, `mdb serve`, the selfcheck and `mdb lanes public --local`. |
| `model.device` | text | `cpu` | `MDB_MODEL_DEVICE` | Torch device for the same commands, tiny models included. |
| `project.dir` | path | none | `MDB_PROJECT_DIR` | Project attached when no `--project` is given. `mdb serve` needs one from either place. |
| `project.session` | text | `auto` | `MDB_PROJECT_SESSION` | Session id in the project. `auto` is the family plus a digest of model identity and execution. The command-line flag is `--session`. |
| `records.sink` | text | none | `MDB_RECORDS_SINK`; older `MDB_RECORDS` | Record sink: every sealed record is appended there. A file, a directory (`records-<run>.jsonl` inside it), or a name containing `{run}`. See [BRANCHES-POOLS-HOLDS.md](BRANCHES-POOLS-HOLDS.md#custody-and-record-sinks-corecustodypy). |
| `records.append` | bool | `false` | `MDB_RECORDS_APPEND` | Add to a sink file another run wrote (otherwise refused). |
| `records.payloads` | bool | `true` | `MDB_RECORDS_PAYLOADS` | Also write large payloads to the sink. |
| `custody.kernel_path` | text | none | `MDB_CUSTODY_KERNEL_PATH`; older `MDB_KERNEL_PATH` | Kernel path recorded in custody. Setting it declares custody. |
| `serve.host` | text | `127.0.0.1` | `MDB_SERVE_HOST` | `mdb serve` bind host. A non-loopback host still needs `--allow-remote` on the command line. |
| `serve.port` | integer 1 to 65535 | `8765` | `MDB_SERVE_PORT` | `mdb serve` port. |
| `serve.allow_hosts` | list of text | `[]` | `MDB_SERVE_ALLOW_HOSTS` (comma-separated) | Extra `Host` headers the server accepts (`--allow-host`). |
| `repl.init` | path | none | `MDB_REPL_INIT` | Command file run before the start line, `--script` and the prompt (like `.gdbinit`). `--init FILE` overrides it and `--no-init` skips it. No file is read unless one is configured. |
| `repl.history` | path | none | `MDB_REPL_HISTORY` | File that keeps interactive command history. |
| `repl.defaults` | table `OP.ARG = value` | `{}` | none | Display and budget defaults for REPL commands, used when a command line does not give the argument. Only these arguments may be set: `print.top`, `print.slice`, `x.count`, the `budget` of `step`, `continue`, `until`, `advance`, `reverse-step` and `run`, `info.limit`, `checkpoints.limit`, `symbols.limit` and `sweep.max_calls`. `mdb config check` validates names and values. |
| `storage.shared_tensor_store` | path | none | `MDB_STORAGE_SHARED_TENSOR_STORE`; older `MDB_SHARED_TENSOR_STORE` | Shared tensor pool. It must be on the same filesystem as the projects that use it. mdb hands the resolved value to `saturn_pub.shared_store.set_shared_store`, so the environment is not changed. |
| `sae.sources` | table | `{}` | none | Named SAE suites: `NAME = { path, layers, k, port, cache, verify, layer }`. Only `path` is required. In the REPL, `sae load NAME` with no path loads one, and arguments on the command line override the entry. |

`mdb config explain KEY` also lists the environment variables of a key. Boolean variables accept `1`, `true`, `yes`, `on` and `0`, `false`, `no`, `off`. An older variable name works exactly like the current one. If both are set to different values, mdb stops with an error.

`MDB_SCRIPT` and `MDB_PREREG` (custody of one run) remain environment variables only.

### `.mdb-storage.json`

Before `[storage]`, a shared tensor store was configured by a `.mdb-storage.json` file in a parent directory of the project. saturn-pub still reads that file. It is deprecated: `mdb config check` and `mdb config explain storage.shared_tensor_store` report when one is in use. Order: `MDB_STORAGE_SHARED_TENSOR_STORE` or `MDB_SHARED_TENSOR_STORE`, then `[storage] shared_tensor_store`, then the nearest `.mdb-storage.json`. Outside mdb, saturn-pub uses an explicit `set_shared_store(root)` first, then `MDB_SHARED_TENSOR_STORE`, then the file.

## What a config file may not set

These keys are refused by name, in any section, with the reason:

| Key | Why it stays per call |
|---|---|
| `floor`, `floors`, `floor_spacings`, `margin_floor_spacings`, `stability_floor_spacings`, `stable_k`, `verdict` | A floor decides whether a reading counts as a verdict. The call that uses it writes it into the record body. Pass it per call (`margin_floor(spacings=...)`, the op argument). |
| `recheck` | It decides which branch arms are re-measured: `Branches.run(recheck=...)`. |
| `mode`, `batch`, `branches` | They decide how branch arms are measured: `Branches.run(mode=..., batch=...)`. |
| `lookahead` | It decides what a move inside a layer computes: `start lookahead=none\|layer\|auto`, `Debugger.run(lookahead=...)`. |
| `prefill` | It decides which prompt rows are stepped ports: `start prefill=K`. |
| `retain_rows` | It decides which cuts a run keeps for rewinds: `start retain_rows=N`, `Debugger.run(retain_rows=N)`. |
| `prereg` | A preregistration belongs to one run: `--prereg`, `MDB_PREREG` or `custody.declare(prereg=...)`. |
| `script` | Custody hashes the script that ran: `--script`, `MDB_SCRIPT`. |
| `seed` | A seed chooses the model or the samples: `--seed` per run. |
| `allow_remote` | Binding `mdb serve` beyond loopback has no authentication: type `--allow-remote` each time. |

Whoever opens a record sees the arguments of the script that custody hashed, but not a user-level file that changed them. That is why these settings are refused. The same names are refused inside `repl.defaults` and `sae.sources`. An unknown section, an unknown key and a value of the wrong type are also errors. Each error names the file and the key (`/home/me/.config/mdb/config.toml:model.dtyp: unknown key`).

## Provenance: what a record says about config

Some settings reach a record: they decide which model, numeric scope, session, SAE basis or custody label a record is about. These are `model.roots`, `model.aliases`, `model.dtype`, `model.device`, `project.session`, `custody.kernel_path` and `sae.sources`. The others are locations and display defaults: serve, repl, the records sink, the project directory and storage.

With no config file, or with config files that set none of the recorded keys, records are sealed exactly as before and their ids do not change. When a loaded file sets a recorded key, every sealed record carries custody, and the custody block has a `config` entry:

```json
"config": {
  "schema": "mdb-config-provenance-v1",
  "sources": [{"path": "/home/me/.config/mdb/config.toml", "sha256": "7900e0...", "layer": "user",
               "settings": {"model.dtype": "bf16"}}],
  "resolved": {"model.dtype": "bf16"},
  "origins": {"model.dtype": "user file /home/me/.config/mdb/config.toml"}
}
```

`sources` lists each file that sets a recorded key, with its path, the sha256 of its bytes and the recorded keys it sets. `resolved` and `origins` give the value that won for each of those keys and where it came from. A flag or variable may override a file's value. Because custody is part of the record body, such a record has a different id from the same record sealed without the file. Measured with torch 2.13 and transformers 5.13.1: `examples/00-hello-tiny` prints `record d0c1d6c733c4bfa1` with no config file and with a file holding only `[serve] port`, and `record 6bad22acbe76b92b` with a file holding only `[model] dtype = "bf16"` (same model, custody added). When custody was not declared, the block names the `__main__` script, as for branches and pools.

## Command-line flags

`mdb engine`: `--dtype`, `--device`, `--project`, `--session`, `--records PATH`, `--records-append`, `--no-payloads`, `--kernel-path NAME`, `--prereg FILE` (declares custody, with `--script` as the script), `--init FILE`, `--no-init`. With `--connect` these flags belong to the server and are refused. `MODEL` and `--model` take the same forms: a checkpoint directory, a cached or rooted model name, an alias, `tiny:FAMILY`, or an `http://` address.

`mdb serve`: `--project` (optional when `[project] dir` is set), `--session`, `--host`, `--port`, `--allow-host`, `--dtype`, `--device`. `--model` takes the same forms as `mdb engine` except an address.

`python -m mdb.engine.selfcheck`: `--generate N` and `--out PATH` are the new names of `--tokens N` and `--json PATH`. The old names still work and print a notice.

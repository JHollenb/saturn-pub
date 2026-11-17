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
| 4. Project file | `<project>/mdb.toml` when a project is named (`--project DIR` or `MDB_PROJECT_DIR`); otherwise the first `mdb.toml`, or `pyproject.toml` with a `[tool.mdb]` table, found walking up from the working directory. The walk stops at the repository root (a directory with `.git`) or your home directory; with neither above the working directory (a scratch directory under `/tmp`), only the working directory is looked at. A file you do not own is skipped. A project file is trusted less than the others: see [Trust model](#trust-model-what-a-project-file-may-set). |
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
defaults = { print = { top = 8 }, x = { count = 32 }, sweep = { max_calls = 100000 } }

[storage]
shared_tensor_store = "/data/mdb-store"
checkpoint_store = "s3://experiments/mdb-checkpoints"
endpoint = "http://127.0.0.1:19010"

[sae]
sources = { l12 = { path = "~/sae/qwen-l12", k = 50, layers = "0:28" } }
```

The same tables can live in `pyproject.toml` as `[tool.mdb.model]`, `[tool.mdb.records]`, and so on.

## Key reference

| Key | Type | Default | Environment | What it does |
|---|---|---|---|---|
| `model.roots` | list of paths | `[]` | `MDB_MODEL_ROOTS` (`:`-separated); older `LLM_MODELS_EXTRA_ROOT` | Directories searched for a model by name, together with the Hugging Face cache. A checkpoint is a directory with a `config.json` one or two levels down (`ROOT/NAME` or `ROOT/ORG/NAME`). If the same id is in the cache and a root, the cache wins. |
| `model.aliases` | table of text | `{}` | none | Short name to a checkpoint directory, a cached or rooted model name, or `tiny:FAMILY`. An alias names one target; an alias of an alias is refused, and so is an alias whose name starts with `tiny:`, `http://` or `https://`. `mdb engine` and `mdb serve` print `alias NAME -> TARGET (source)` when one is used. |
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
| `serve.allow_hosts` | list of text | `[]` | `MDB_SERVE_ALLOW_HOSTS` (comma-separated) | Extra `Host` headers the server accepts. Not from a project file. A configured list never satisfies `--allow-remote`, which needs `--allow-host` typed on the command line. |
| `repl.init` | path | none | `MDB_REPL_INIT` | Command file run before the start line, `--script` and the prompt (like `.gdbinit`). `--init FILE` overrides it and `--no-init` skips it. A project file's init file runs only from a trusted directory (`repl.trusted_projects`). Every init file that runs is hashed into custody, and it may not pass `lookahead=`, `prefill=`, `retain_rows=` or other verdict arguments. |
| `repl.trusted_projects` | list of paths | `[]` | `MDB_REPL_TRUSTED_PROJECTS` (`:`-separated) | Directories whose project files may run their `repl.init`. Not from a project file. |
| `repl.history` | path | none | `MDB_REPL_HISTORY` | File that keeps interactive command history. Not from a project file. If the file exists but is not a readline history file, history is turned off for the session and the file is left alone. |
| `repl.defaults` | table `OP.ARG = value` | `{}` | none | Display defaults for REPL commands, used when a command line does not give the argument. Only these arguments may be set: `print.top`, `print.slice`, `x.count`, `info.limit`, `checkpoints.limit` and `sweep.max_calls` (a cap that refuses a larger sweep; it never changes a sweep's result). Budgets and `symbols limit` are not configurable: a budget moves where a run stops, and `symbols limit` sets the Unbound row of a sealed symbol table. `mdb config check` validates names and values. Recorded in custody when set in a file. |
| `storage.shared_tensor_store` | path | none | `MDB_STORAGE_SHARED_TENSOR_STORE`; older `MDB_SHARED_TENSOR_STORE` | Shared tensor pool. It must be on the same filesystem as the projects that use it. mdb hands the resolved value to `saturn_pub.shared_store.set_shared_store`, so the environment is not changed. |
| `storage.checkpoint_store` | text | none | `MDB_STORAGE_CHECKPOINT_STORE`; also `MDB_CHECKPOINT_STORE` | Where project checkpoints are also published and where a record id the project lacks is fetched from: `s3://bucket/prefix` (needs the `s3` extra), `file:///dir` or a directory (an absolute path; a relative one resolves against the working directory). Credentials never go here: an `s3://` store reads them from the environment (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`; boto3's chain), and a URI that names any is refused. See [BRANCHES-POOLS-HOLDS.md](BRANCHES-POOLS-HOLDS.md#checkpoint-stores-corecheckpoint_storepy). |
| `storage.endpoint` | text | none (then `AWS_ENDPOINT_URL`) | `MDB_STORAGE_ENDPOINT` | Endpoint URL of an `s3://` checkpoint store (MinIO). Path-style addressing. |
| `storage.checkpoint_cache` | path | `$XDG_CACHE_HOME/mdb/checkpoints` | `MDB_STORAGE_CHECKPOINT_CACHE` | Local project that holds store checkpoints when no project is attached (`run.dump("s3://...")`, `run_from(record id)` with no project). |
| `sae.sources` | table | `{}` | none | Named SAE suites: `NAME = { path, layers, k, port, cache, verify, layer }`. Only `path` is required. In the REPL, `sae load NAME` with no path loads one, and arguments on the command line override the entry. |

`mdb config explain KEY` also lists the environment variables of a key. Boolean variables accept `1`, `true`, `yes`, `on` and `0`, `false`, `no`, `off`. An older variable name works exactly like the current one. If both are set to different values, mdb stops with an error.

`MDB_SCRIPT` and `MDB_PREREG` (custody of one run) remain environment variables only.

### `.mdb-storage.json`

Before `[storage]`, a shared tensor store was configured by a `.mdb-storage.json` file in a parent directory of the project. saturn-pub still reads that file. It is deprecated: `mdb config check` and `mdb config explain storage.shared_tensor_store` report when one is in use. Order: `MDB_STORAGE_SHARED_TENSOR_STORE` or `MDB_SHARED_TENSOR_STORE`, then `[storage] shared_tensor_store`, then the nearest `.mdb-storage.json`. Outside mdb, saturn-pub uses an explicit `set_shared_store(root)` first, then `MDB_SHARED_TENSOR_STORE`, then the file.

## Trust model: what a project file may set

The user file, `MDB_CONFIG`, the environment and the command line are yours. A project file may come from a repository you cloned, so it is read with less trust, as gdb treats a `.gdbinit` in the working directory:

| Kind of key | Project file | Keys |
|---|---|---|
| Settings and readable locations | yes | `model.*`, `project.session`, `records.append`, `records.payloads`, `custody.kernel_path`, `serve.host`, `serve.port`, `repl.defaults`, `sae.sources` |
| Paths mdb writes or runs | only inside the project file's directory (after `~`, `..` and symbolic links are resolved) | `project.dir`, `records.sink`, `storage.shared_tensor_store`, `repl.init` |
| Init file | parsed, but run only when the project directory is under a `repl.trusted_projects` entry of the user file (or the file is passed with `--init`); otherwise mdb prints a note and skips it | `repl.init` |
| Files mdb overwrites, hosts it trusts, the trust list itself | no (an error naming `file:key`) | `repl.history`, `serve.allow_hosts`, `repl.trusted_projects` |

Discovery reads only files you own. Outside a repository and outside your home directory, discovery looks only in the working directory. A `pyproject.toml` that is not valid TOML is skipped unless it contains a `[tool.mdb` table. In that case it is an error.

The settings are read once, when a command starts or when Python first needs one. A file edited afterwards does not change that process's settings or its records' config block.

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

Some settings reach a record: they decide which model, numeric scope, session, SAE basis or custody label a record is about, or which REPL commands run before the script. These are `model.roots`, `model.aliases`, `model.dtype`, `model.device`, `project.session`, `custody.kernel_path`, `sae.sources`, `repl.init` and `repl.defaults`. The others are locations: serve, the history file, the records sink, the project directory and storage.

With no config file, or with config files that set none of the recorded keys, records are sealed exactly as before and their ids do not change. When a loaded file sets a recorded key, every sealed record carries custody, and the custody block has a `config` entry:

```json
"config": {
  "schema": "mdb-config-provenance-v1",
  "sources": [{"path": "/home/me/.config/mdb/config.toml", "sha256": "7900e0...", "layer": "user",
               "settings": {"model.dtype": "\"bf16\""}}],
  "resolved": {"model.dtype": "\"bf16\""},
  "origins": {"model.dtype": "user file /home/me/.config/mdb/config.toml"}
}
```

`sources` lists each file that sets a recorded key, with its path, the sha256 of its bytes and the recorded keys it sets. `resolved` and `origins` give the setting that won for each of those keys and where it came from. Values are JSON text, so a number in a setting is never read as a measured claim. A flag or variable may override a file's value. When a Python call passes an argument that differs from the setting (`tiny_model("qwen2", "fp32")` under a `bf16` file), the block also has `call_arguments: {"model.dtype": ["\"fp32\""]}`; the record's `execution` says what ran.

When an init file runs (`repl.init`, `--init`), custody also gets `init: [{path, sha256}]`, with or without a config block. Because custody is part of the record body, such a record has a different id from the same record sealed without the file. Measured with torch 2.13 and transformers 5.13.1: `examples/00-hello-tiny` prints `record d0c1d6c733c4bfa1` with no config file and with a file holding only `[serve] port`, and a different id (`7f588d408421b1bf` in one run) with a file holding only `[model] dtype = "bf16"`: the same model, with custody added. That id depends on the file's path and bytes, so it differs between machines. When custody was not declared, the block names the `__main__` script, as for branches and pools.

## Command-line flags

`mdb engine`: `--dtype`, `--device`, `--project`, `--session`, `--records PATH`, `--records-append` / `--no-records-append`, `--payloads` / `--no-payloads` (the negative forms override a config file), `--kernel-path NAME`, `--prereg FILE` (declares custody, with `--script` as the script), `--init FILE`, `--no-init`. With `--connect` these flags belong to the server and are refused. `MODEL` and `--model` take the same forms: a checkpoint directory, a cached or rooted model name, an alias, `tiny:FAMILY`, or an `http://` address.

`mdb serve`: `--project` (optional when `[project] dir` is set), `--session`, `--host`, `--port`, `--allow-host`, `--dtype`, `--device`. `--model` takes the same forms as `mdb engine` except an address.

`python -m mdb.engine.selfcheck`: `--generate N` and `--out PATH` are the new names of `--tokens N` and `--json PATH`. The old names still work and print a notice.

# Contributing: repository rules

This page is for anyone changing code in this repository. It covers what is in the repository and what ships, how to set up, how versions and releases work, the rules a change must keep, how to add a model family or an mdb machine, and where bugs and feature requests go. User documentation is elsewhere: start at [docs/README.md](docs/README.md).

## Layout: what ships and what does not

| Path | Distribution / import | Ships? | What it is |
|---|---|---|---|
| `src/saturn_pub/` | `saturn-pub` / `saturn_pub` | yes | The runtime: typed contracts, sessions, store and custody primitives, native model adapters (`adapters/`), residency, evidence, training. Version in `pyproject.toml` and `saturn_pub.__version__`. |
| `packages/mdb/` | `mdb-runtime` / `mdb` | yes | The debugger (`mdb engine`, `mdb serve`, the Python API). Its own `pyproject.toml`, `README.md`, `docs/`, `examples/`, `tests/`, `web/`, `CHANGELOG.md`. |
| `packages/mdb-lanes/` | `mdb-lanes` / `mdb_lanes` | built in CI, not released | The plug-in holding the older lane-era commands (`mdb lanes VERB`), registered through the `mdb.lanes` entry-point group. The engine never imports it. |
| `packages/mdb-experiments/` | none | no | Internal research that uses mdb: `experiments/`, `results/`, audits, lane-era docs, and the internal ledgers (`issues.jsonl`, `investigations.jsonl`, `features.json`). Not built, not installed. Results here are custody records: do not edit them in place. |
| `tests/`, `examples/`, `notebooks/`, `docs/`, `experiments/` (root) | | | saturn-pub's tests, examples, notebooks, docs and validation experiments. |

The dependency direction is `mdb -> saturn_pub`, never the reverse. `mdb-runtime` declares `saturn-pub` in its metadata (`saturn-pub>=0.4.3,<0.5` on `main`) and imports only installed packages: no sibling-checkout discovery and no `sys.path` patching at run time. Private workspace packages (`saturn`, `common`, `common_harness`) are forbidden in both packages; `packages/mdb/tests/conftest.py` blocks them and `scripts/check_package_isolation.py` checks the built wheels. Anything that is an experiment, a result, an audit, a proposal or an internal tracker goes in `packages/mdb-experiments`, never in `packages/mdb`.

Inside mdb, layers import only downward: `core < engine < analysis < ops < interop < repl < serve < top`, and no engine module imports a lane module. `packages/mdb/tests/test_import_contract.py` enforces this. `import mdb` must not load torch, and framework-free commands (`mdb version`, reading records, `mdb serve --project`) must keep working without a model framework.

## Development setup

Python 3.10 or newer (CI runs 3.10 and 3.12; `.python-version` says 3.12). Use two environments: saturn-pub's development lock (`uv.lock` at the root) pins torch 2.14, while `mdb-runtime` pins torch 2.13.0 for its measured numbers. mdb is deliberately not a member of the root uv workspace.

saturn-pub, from the repository root:

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e '.[ar,diffusion,train,dev]'     # what CI installs
pytest -q tests/test_families.py               # targeted; see Testing
ruff check . && ruff format --check .
```

mdb, from `packages/mdb`:

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ../.. -e '.[autoregressive,serve,dev]'
python -m pytest -q tests/test_engine_run.py
python examples/00-hello-tiny/hello_tiny.py     # CPU, nothing downloaded, a few seconds
```

Without installing, the source trees work through `PYTHONPATH` (`W` is the repository root): `env PYTHONPATH=$W/src:$W/packages/mdb/src OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 python -m pytest -q -p no:cacheprovider tests/test_engine_trace.py`, run from `packages/mdb`. Add `$W/packages/mdb-lanes/src` for the lane plug-in. Without the `autoregressive` extra (torch and transformers) most mdb tests skip rather than fail, so a green run in a bare venv proves little; check the skip count with `-rs`.

Every automated test runs on random tiny CPU models (`mdb.engine.models.tiny_model`, `tiny:qwen2|qwen3|llama|mistral`; the `tiny_config` of each family in `saturn_pub.adapters.decoder`). Nothing downloads: CI sets `HF_HUB_OFFLINE=1`, and `mdb` loads checkpoints with `local_files_only=True`. External contributors never need a GPU or model weights.

The web viewer is built from `packages/mdb/web` (`npm ci && npm run build`) and is not checked in; build it before building an mdb wheel, or `mdb serve` answers `/` with `web_build_absent`.

## Versioning and releases

The **owner sets every version number**. Propose a version in a PR if you like; do not bump one on your own. Both packages are pre-1.0 and versioned semver-style as practiced so far:

- **Patch** (`0.2.x`): fixes and changes the owner judges small, including non-breaking features. 0.2.1 (stepping speedups, same output) and 0.2.2 (record schema v3, which changed every record id) were both patch releases by owner decision. A change that alters record ids or other serialized identities is still listed under `### Changed (breaking)` in the changelog with its migration path, whatever the version.
- **Minor** (`0.x.0`): a new series; it gets its own release branch.

mdb (`mdb-runtime`), described in full in [packages/mdb/docs/RELEASING.md](packages/mdb/docs/RELEASING.md):

- `main` is development. Its version is the next patch with `.devN` (now `0.2.4.dev0`), set in both `packages/mdb/pyproject.toml` and `mdb.__version__` (`packages/mdb/src/mdb/__init__.py`). Every change lands on `main` first.
- One release branch per minor series, `release/mdb-X.Y` (now `release/mdb-0.2`). Fixes for a released series are committed on `main` and then cherry-picked with `git cherry-pick -x`.
- A release is one commit that sets the version in both places and adds `## [X.Y.Z] — DATE` to `CHANGELOG.md`, plus an annotated tag `mdb-vX.Y.Z` on that commit. Tags are never moved or reused. Nothing is published to PyPI; users install from the tag with pip.
- `CHANGELOG.md` discipline: every user-visible change adds an entry under `## [Unreleased]` in the same commit, in the sections `Added`, `Changed`, `Changed (breaking)`, `Fixed`, `Deprecated`. When a change alters output a user may compare (record ids, a printed number), say what changed and what stayed the same, as the 0.2.2 and Unreleased entries do with the `examples/00-hello-tiny` record id. User-facing known issues go under "Known issues" there, not in the internal ledger alone.

saturn-pub has its own version (`pyproject.toml` at the root and `saturn_pub.__version__` in `src/saturn_pub/__init__.py`; now `0.4.3`). It has no tags or changelog of its own: its releases are the versions mdb pins. When an mdb change needs new saturn-pub API, bump saturn-pub (both places) and raise the lower bound of mdb's `saturn-pub` dependency **in the same commit** (as `saturn-pub 0.4.2 (set_shared_store); mdb pins >=0.4.2` did), and note it in mdb's changelog, so that any one ref installs a matching pair. If a released-series fix needs the saturn-pub change, cherry-pick both.

`mdb-lanes` stays at `0.1.0` and is not released.

## Branches, commits, worktrees

- Work on a branch; never commit straight to `main` or a `release/*` branch without the owner. For parallel work prefer a `git worktree` per branch over switching branches in one checkout.
- Maintainers do not push or merge for someone else; the owner merges.
- In a checkout others are also using, commit **by path**: `git commit -m "..." -- path/a path/b`. A bare `git commit` takes whatever anyone staged. For the same reason never run a bare `git stash` there (it stashes everyone's changes); use a worktree, or `git stash push -- <paths>`.
- Commit subjects name the package and the change (`mdb custody: open the MDB_RECORDS sink when custody is declared first`, `saturn_pub: ...`). A body that says what was measured, and on what, is welcome.
- Commits written with an AI assistant end with its attribution trailers (`Co-Authored-By: ...`, and a session link where the tool gives one). For human-only commits trailers are optional.

## Testing rules

- **Run the test files of the module you touch, not the whole suite.** The mdb suite is large and loads many tiny models; on a laptop keep to one Python process at a time. CI runs everything on every push and pull request.
- Run saturn-pub's suite (`tests/`, from the root) and mdb's suite (`packages/mdb/tests`, from `packages/mdb`) in **separate processes**. Collecting them together breaks their import guards and collides test-module names.
- Tests never read the developer's own configuration. `packages/mdb/tests/conftest.py` points `HOME` and `XDG_CONFIG_HOME` at empty temporary directories, unsets `MDB_CONFIG` and every `MDB_*` settings variable, and ignores any `mdb.toml` inside the checkout. Do not weaken that fixture, and do not write a test that reads `~/.config` or a real model cache.
- Golden REPL transcripts live in `packages/mdb/tests/repl_golden/`. After an intended output change, regenerate one with `MDB_UPDATE_GOLDEN=1 python -m pytest tests/test_repl_golden.py -k NAME` and read the diff before committing it.
- The serve tests need the `serve` extra and `httpx` and skip elsewhere, so a pass without them says nothing about the server.
- To test an installed wheel, install it into a fresh environment with its declared dependencies (saturn-pub included), copy `tests/` and `examples/` outside the checkout, and run Python with `-I` and no `PYTHONPATH`.

**Bit-identity.** A change to the engine stepping, port controllers, custody, records, hashing or an adapter must leave existing outputs byte-identical unless changing them is the point of the change. Show it with:

- `packages/mdb/tests/test_config_cli.py::test_hello_tiny_id_unchanged_by_a_locations_file_and_changed_by_a_recorded_key` and the record id `examples/00-hello-tiny` prints;
- the golden transcripts (`tests/test_repl_golden.py`);
- `python -m mdb.engine.selfcheck --tiny FAMILY --out report.json` for each tiny family (`"passed": true`: logits and receipts byte-identical, probes equal the one-pass capture, rewind identical);
- for saturn_pub, `tests/test_families.py`, `tests/test_step_speedups.py` and `tests/test_residency.py` (streamed vs resident).

If ids or receipts do change on purpose, the commit says so, the changelog says so under `Changed (breaking)`, and old records stay readable (see `mdb records migrate` and [RECORDS.md](packages/mdb/docs/RECORDS.md)).

## Invariants you must not break

- **Identities.** Record ids are sha256 of the stored form; state fingerprints, receipts, CAS addresses, schema names and address grammar are bound into existing records and measured artifacts. Preserve legacy serialized schema, address, CAS and numerical-contract identities. Do not change the Hyperion compiler bytes casually: measured artifacts bind its digest.
- **Custody.** Anything that can change what a record says (model roots and aliases, dtype, device, session, kernel path, SAE sources, REPL init or defaults) must be named in custody when it comes from a config file. Never let a record seal without saying what ran.
- **Verdict-affecting settings are never configurable.** Floors, `recheck`, branch `mode`/`batch`, `lookahead`, `prefill`, `retain_rows`, `prereg`, `script`, `seed` and `allow_remote` are refused by name in a config file and in an init file; they stay in the call, where the hashed script shows them. A new setting that can change a verdict joins that list ([CONFIG.md](packages/mdb/docs/CONFIG.md#what-a-config-file-may-not-set)).
- **Eager attention, eval mode.** Adapters and the engine require `attn_implementation="eager"` and refuse a module in training mode (`mdb.engine.ports.check_deterministic`). Loading casts parameters only, never buffers (RoPE `inv_freq` stays float32; see `mdb.engine.models.cast_weights` and `load_local`).
- **Fail closed.** A checkpoint outside a family's rules is refused with a message naming the rule; never accept a new family because its tensor shapes match. Sharing code does not make existing cuts interchangeable or qualify a new model/backend/dtype lane.
- **Examples are owner-approved.** `packages/mdb/examples/` holds only examples the owner approved. Put a new demo in `packages/mdb-experiments` and propose it.
- **User docs are about mdb, not about our research.** `packages/mdb/README.md`, `packages/mdb/docs/` and the examples explain an object, command or API. They carry no job ids, internal issue ids or paths into `packages/mdb-experiments` or other internal notes. A real-model number in them is labelled as measured on the developers' GPU.

## Adding a model family (a "port")

A decoder family is supported at two levels. **saturn-pub** can run it (stepped, forked, captured, restored) once `saturn_pub.adapters.decoder` has a family spec. **mdb** can step it port by port once `mdb.engine.ports.FAMILIES` has a row for it. A family can be in saturn-pub without being in mdb: today saturn-pub registers gpt2, gpt_neox, phi, llama, mistral, mixtral, gemma, gemma2, qwen2, qwen3 and mamba; mdb registers qwen2, qwen3, llama and mistral. Reference commits: `defcd69` (Gemma-2 in saturn-pub) and `f9208c1` (llama and mistral in mdb).

### 1. saturn-pub adapter (`src/saturn_pub/adapters/`)

1. Write a fail-closed validator `_validate_<family>(config)` in `decoder.py`. Call `_reject_common` (quantized and remote-code checkpoints) and, unless you implement scaled RoPE, `_reject_scaled_rope`. Refuse every config value whose semantics the adapter does not reproduce, naming the rule (activation, biases, tied embeddings, sliding window, softcaps, ...).
2. Add a `_FamilySpec` in `_import_families()`: HF model and config classes, the attribute names of the backbone, layer list, embedding and final norm, `position` (`"rotary"` or `"absolute"`, with `abs_pos`), the cache keyword, the validator, `moe` if routed, and a `tiny_config` that exercises the family's distinctive paths (Gemma-2's sets softcaps and a 16-token sliding window so the tests cross them).
3. Add the `model_type` to `SUPPORTED_FAMILIES` in `decoder.py` and `_DECODER_FAMILIES` in `adapters/__init__.py`, and update the docstring there.
4. If the family computes something outside the native decoder layer (Gemma-2's final-logit softcap lives in `Gemma2ForCausalLM`, not `lm_head`), reproduce it at readout and add a test proving it is load-bearing (the run without it must diverge).
5. A family that is not a decoder (Mamba) gets its own adapter module and its own entry in `adapters/__init__.py`, following `mamba.py` and [docs/adapters.md](docs/adapters.md#required-adapter-checks).
6. Tests in `tests/test_families.py`: the parametrized fixture over `SUPPORTED_FAMILIES` then runs stepped-vs-native logits, greedy-vs-native generate, fork isolation/intervention/replay, write guards, and fresh-process capture/save/continue for the new family automatically. Add a `test_refuses_<family>_<rule>` for each validator rule and any family-specific parity test. `test_load_dispatches_by_model_type` checks the two registries agree.
7. Docs: the family table in the root [README](README.md#supported-models) and the list in [docs/adapters.md](docs/adapters.md#shipped-native-adapters).

### 2. mdb port program (`packages/mdb/src/mdb/`)

mdb's port program (`engine/ports.py`) is written for one native block shape: RMSNorm, q/k/v projections with optional per-head q/k norm, RoPE, eager attention, `o_proj`, residual, RMSNorm, `act_fn(gate_proj) * up_proj`, `down_proj`, residual, with the Llama/Qwen module names.

**A Llama-like family** (same block shape and module names) is a table row:

1. `engine/ports.py`: add `Family("<model_type>", qk_norm=..., qkv_bias="always"|"config", premises=(...))` to `FAMILIES`. Premises are `(config attribute, allowed values, why)`; `config_refusal` refuses a loaded config outside them at `attach`, `trace` and `program_for`. `PROGRAMS` is built from the table.
2. `engine/models.py`: add a branch to `tiny_model` building a random tiny eager model of the family (seeded with `torch.manual_seed(seed)`, `attn_implementation="eager"`, cast with `cast_weights`), and describe it in the docstring.
3. `core/settings.py`: add the family to `TINY_FAMILIES` (the one list; the CLI, REPL, serve and selfcheck read it). Update the literal lists in docstrings and usage text (`cli.py` module docstring, `repl/__main__.py`, `engine/selfcheck.py`); `tests/test_tiny_families.py` checks the CLI help.
4. Tests: add the family to the family-parametrized lists (as of `f9208c1`: `test_engine_ports.py`, `test_analysis_finite.py`, `test_analysis_explain.py`, `test_engine_trace.py`, `test_repl_golden.py` `OTHER_FAMILIES`, plus the analysis and ops tests that commit touched); add a golden transcript `tests/repl_golden/<family>_session.mdb` and generate its `.out` with `MDB_UPDATE_GOLDEN=1`; a test that each premise refuses.
5. Run `python -m mdb.engine.selfcheck --tiny <family> --dtype bf16 --out ...` and `--dtype fp32`; both must pass.
6. Docs: `INSTALL.md` and the README name the tiny families; add a `CHANGELOG.md` entry.

**A Gemma-like family** (GeGLU activation, pre- and post-norms, softcaps, scaled embeddings) does not fit the shared program: `_decoder_program` would describe the wrong operator text, which is exactly what the premises exist to refuse. It needs its own port program (new `PortSpec`s for the extra norms and softcap, its own `Family` columns or a separate program builder), its own `explain` texts, and the same test and selfcheck list as above. Open a model port request first and agree the design with the maintainers.

### 3. Real-model validation (maintainers)

Tiny models prove mechanics, not a family. Before a family is called supported on real weights, a maintainer runs it on the developers' GPU host: for saturn-pub, `experiments/family_validation/run.py` (16-token greedy decode identical to native, fresh-process replay exact, max logit delta recorded); for mdb, `python -m mdb.engine.selfcheck --model PATH ...`. Results are recorded with the run's id, backend, device, dtype and source revision; the family table says "tiny fixture only" until then. External contributors stop at the tiny-model checklist.

### Port checklist

- [ ] Validator refuses every unreproduced config value, by name; one test per rule
- [ ] `_FamilySpec` with a `tiny_config` that exercises the family's distinctive paths
- [ ] `SUPPORTED_FAMILIES` and `_DECODER_FAMILIES` updated; `tests/test_families.py` passes for the family
- [ ] (mdb) `FAMILIES` row with premises, `tiny_model` branch, `TINY_FAMILIES`, docstring lists
- [ ] (mdb) parametrized test lists, golden transcript, selfcheck bf16 and fp32 pass
- [ ] Existing families' outputs unchanged (bit-identity checks above)
- [ ] README family table, docs/adapters.md, mdb CHANGELOG `[Unreleased]`
- [ ] Real-weight row only after maintainer validation; otherwise "tiny fixture only"

## Adding an mdb machine

A machine (`packages/mdb/src/mdb/machines/`) is a program that is not a Hugging Face decoder, debugged with the same verbs. The decoder engine stays specialised to transformer families; a machine implements the `Program` protocol in `machines/__init__.py`: `sites`, `step`/`next`/`reverse_step` over named clock domains, `snapshot`/`restore`, `poke`, typed `reads_of`/`writes_of` (edges `data`, `addr`, `ctrl`), `domains`/`define`. Patch verdicts are the shared `VERDICTS` (`effect`, `no effect`, `path-diverged`, `fault`, `nonterminating`). The reference is `machines/x86` (commit `3d0baff`): `binary.py` (loading and the backend), `address.py` (address grammar), `index.py` (execution index), `session.py` (`attach` and the verbs), `records.py` (records), `lanes.py` (batched lanes).

- [ ] A subpackage `machines/<name>/` whose `__init__` imports nothing heavy (`attach` imports the session lazily)
- [ ] Optional dependencies are imported inside functions and missing ones raise an `ImportError` that says what to install; tests `importorskip` them
- [ ] New record fields are declared in `core/kinds.py` (as x86 added its machine-binding paths to `SYMBOL_TABLE`) and documented in the schema doc ([SYMBOL-TABLE-SCHEMA.md](packages/mdb/docs/SYMBOL-TABLE-SCHEMA.md) "Machine bindings")
- [ ] `tests/test_machines_<name>.py` on a small hand-built program, with no reference data
- [ ] Experiment drivers and results go in `packages/mdb-experiments`, not in the machine package

## Bugs, model port requests and feature requests

**Everyone: GitHub Issues** on [JHollenb/saturn-pub](https://github.com/JHollenb/saturn-pub/issues). Pick a template: *Bug report* (versions, the smallest reproduction, preferably on a tiny model), *Model port request* (the `model_type`, a checkpoint, what differs from a registered family), or *Feature request* (what you want to do and why). Search first; a thumbs-up on an existing issue counts as a vote. Security problems are not filed publicly (see Security).

**Maintainers: the internal ledgers** in `packages/mdb-experiments/`, which carry measured evidence and run ids that do not belong in a public issue:

| Ledger | For | Fields |
|---|---|---|
| `issues.jsonl` | bugs and defects found in our own runs (ids with a prefix: `M*`, `C*`, `RG*`, ...) | `ts`, `issue`, `kind`, `status` (`open`, `fixed`, `closed`, `deferred`, `known-limit`), `severity`, `detail`, `evidence`, `next`; on closing, `resolution` saying what changed and what was measured (and `fixed` where used) |
| `investigations.jsonl` | research questions, not defects | `handle`, `timestamp`, `status` (`open`, `investigated`, `closed`, `follow-up`), `description`, `evidence`, `follow_up` |
| `features.json` | mdb feature requests (`F*`), one JSON document, schema `mdb-features-v1` | `id`, `title`, `status` (`requested`, `planned`, `in-progress`, `done`, `dropped`), `requested`, `requested_by`, `plan`, `summary`, `requirements`, `known_facts`, `open_questions` |

Append to the `.jsonl` ledgers one line at a time, and change an entry by editing its line as text. Never load and re-dump a ledger through a JSON round trip: it rewrites every line (key order, escaping, spacing) and destroys the history diff. Edit `features.json` by hand in the same spirit. When a GitHub issue leads to internal work, put the issue URL in the ledger entry's `evidence`; when an internal issue affects users, open or update a GitHub issue and add it to the changelog's "Known issues".

The maintainers' wider workspace also keeps a cross-project request list for Saturn tooling (`saturn/request.jsonl`, outside this repository: one JSON object per line with `handle`, `timestamp`, `vote count`, `description`; an existing request is upvoted or downvoted by editing its `vote count` instead of adding a duplicate). Requests specific to this repository go in GitHub Issues or the ledgers above.

## Documentation rules

- One page per subject; a new user-facing mdb page is listed in [packages/mdb/docs/README.md](packages/mdb/docs/README.md), a saturn-pub page in [docs/README.md](docs/README.md).
- Docs describe what the code does now. A change to a command, flag, record field or default updates its doc in the same commit.
- Numbers carry their authority: measured (and where: "on the developers' GPU", "laptop CPU"), derived, or asserted. Never present a tiny-model check as a real-model result.
- The mdb user-doc rule above (no job ids, issue ids or internal paths) applies to `packages/mdb`. Experiment reports and run logs live in `packages/mdb-experiments`.
- Markdown: one paragraph per line, no hard wraps in the middle of a sentence.

## Real-model runs

Real-model validation happens on the maintainers' GPU host through their job scheduler, not on contributors' machines and not in CI. Each run is recorded with its job id, backend, device, dtype and the source revisions used, and the id travels with the result in `packages/mdb-experiments` (or an example's `validation-job-<id>.log`). Real models run sequentially, one at a time per device, with strict preflight and no blind retries; a failure is investigated from its logs before rerunning. Long runs (over about 15 minutes) are reviewed independently before they are submitted. New research runs that depend on pretrained weights include a `retrieve_weights.py` and a `parameter-artifacts.json` that bind the named tensors to immutable checkpoint files and hashes (see `packages/mdb/AGENTS.md`).

If you contribute a change that needs a real-model check, say so in the PR and a maintainer will run it.

## Review

- Every change is reviewed before it is merged; the owner merges.
- Changes to custody, record format or ids, hashing, the engine stepper and port controllers, saturn_pub adapters, or residency get an **independent review**: someone (or a separate review pass) other than the author reads the diff against these rules and checks the bit-identity evidence, not just the test result.
- A PR states what it changes, which tests were run (files, and on what environment), whether any output identity changed, and whether a real-model check is still owed. The pull request template asks for these.

## Security

- **Config trust model.** The user's config file, `MDB_CONFIG`, the environment and the command line are trusted. A project file (`mdb.toml`, `pyproject.toml [tool.mdb]`) may come from a cloned repository and is trusted less: its `repl.init` runs only from a directory listed in the user file's `repl.trusted_projects`, and it may not set `repl.history`, `serve.allow_hosts` or `repl.trusted_projects`. Files the user does not own are skipped. Keep these rules when adding keys ([CONFIG.md](packages/mdb/docs/CONFIG.md#trust-model-what-a-project-file-may-set)).
- **`mdb serve` has no authentication.** It binds loopback; `--allow-remote` must be typed on every run and is refused in config files. Do not add a way around that, and do not describe serve as safe to expose.
- Never commit credentials, tokens, host names or model weights. Checkpoints load with `local_files_only=True`; remote-code (`auto_map`) and quantized checkpoints are refused.
- Report a vulnerability privately to the maintainer through GitHub (a private security advisory on the repository, if enabled), not in a public issue.

# Installing mdb

## Install

mdb (`mdb-runtime`, import name `mdb`) depends on `saturn-pub` (`>=0.4.2,<0.5` on `main`,
`>=0.4.1,<0.5` for 0.2.3), which is not on PyPI.
Both live in the [saturn-pub repository](https://github.com/JHollenb/saturn-pub); install them
together from a checkout:

```sh
git clone https://github.com/JHollenb/saturn-pub
cd saturn-pub
pip install . "./packages/mdb[autoregressive]"
# or, to edit the code:
pip install -e . -e "./packages/mdb[autoregressive]"
```

A clone is on `main`, the development branch. For a released version, `git checkout mdb-v0.2.3`
(the latest release tag) before installing, or install it from Git without cloning; name both
packages, because pip cannot find `saturn-pub` on PyPI:

```sh
pip install "saturn-pub @ git+https://github.com/JHollenb/saturn-pub@mdb-v0.2.3" \
  "mdb-runtime[autoregressive] @ git+https://github.com/JHollenb/saturn-pub@mdb-v0.2.3#subdirectory=packages/mdb"
```

`@release/mdb-0.2` instead of the tag follows the 0.2 series' fixes. Tags, branches and how
fixes reach a release: [RELEASING.md](RELEASING.md).

Python 3.10 or newer. mdb never searches for a source checkout at run time: it imports only
installed packages.

## Extras

| Extra | Adds | Needed for |
|---|---|---|
| (none) | `saturn-pub` only, no model framework | `mdb version`, `mdb serve` of a recorded project (with `serve`), reading records |
| `autoregressive` | torch 2.13.0, numpy, safetensors, transformers 5.14.1, accelerate 1.14.0 | `mdb engine`, `mdb.open_tiny`, `mdb.attach`, `mdb.trace`: anything that runs a model |
| `serve` | fastapi, uvicorn, pydantic, websockets, numpy | `mdb serve` (web viewer and live sessions) |
| `interop` | transformer-lens 4.4.0, nnsight 0.8.2 | `mdb.interop` (TransformerLens and nnsight names, imports, batched edits) |
| `transformerlens` | `autoregressive` plus transformer-lens 4.2.0 | an older pinned TransformerLens environment |
| `discovery` | torch, numpy, safetensors | framework-light analysis without transformers |
| `mrun`, `diffusion`, `all` | mrun-pub from a pinned public Git revision (plus diffusers and pillow for `diffusion`/`all`) | submitting jobs to an mrun scheduler; the diffusion pipelines of the lanes plug-in |
| `dev` | pytest, ruff, hatchling | running the tests and building |

The examples need further libraries (SAELens, pyvene, matplotlib); the table in
[examples/README.md](../examples/README.md) lists them.

## Check the install

```sh
mdb version                                   # 0.2.3 from the tag; X.Y.Z.devN on main
mdb engine tiny:qwen2 --tokens "1 5 9 3"      # a REPL on a random tiny model, CPU, no download
python examples/00-hello-tiny/hello_tiny.py   # from packages/mdb: the debugger loop in ~60 lines
```

`mdb engine MODEL` takes a checkpoint directory, the name of a model in the local Hugging Face
cache, `tiny:qwen2|qwen3|llama|mistral`, or the address of a running `mdb serve`. Nothing is ever
downloaded. A real model needs eager attention (`attn_implementation="eager"`, which `mdb engine`
sets when it loads a checkpoint) and runs on CPU or CUDA. On Apple MPS `whatif` fails (no float64).

## Configuration (optional)

mdb runs with no config file. To keep defaults (dtype, device, model locations, project, record sink, REPL init file) out of every command line, write a commented template and edit it:

```sh
mdb config init      # ~/.config/mdb/config.toml (or $XDG_CONFIG_HOME/mdb/config.toml)
mdb config show      # the resolved settings and where each came from
```

A project can carry its own `mdb.toml`. On Python 3.10 the TOML reader is `tomli`, which is installed as a dependency. See [CONFIG.md](CONFIG.md).

## The web viewer

`mdb serve` needs the `serve` extra. The browser app is built from `web/` and is not checked in:
run `npm ci && npm run build` in `web/` before building a wheel, or the server answers `/` with
`web_build_absent` (the API still works).

## Tests and building

From `packages/mdb`, with the `dev` and `autoregressive` extras:

```sh
python -m pytest -q tests/test_engine_run.py     # run the test files of the module you touch
uv build --out-dir dist                          # sdist + wheel; build the web app first
```

The test suite uses only random tiny CPU models. The serve tests need `fastapi` and `httpx` and are
skipped without them. To test an installed wheel, install it into a fresh environment, copy
`tests/` and `examples/` outside the checkout, and run Python with `-I` and no `PYTHONPATH`.

## Lanes plug-in

The commands of the older `mdb` tool (`mdb debug`, `mdb doctor`, `mdb help`, `mdb capabilities`,
`mdb verify`, ...) are not part of this package. They come from the separate `mdb-lanes`
plug-in (`packages/mdb-lanes` in the repository), which registers them through the `mdb.lanes`
entry-point group:

```sh
pip install ./packages/mdb-lanes
mdb lanes                 # lists its verbs; `mdb lanes VERB` runs one
```

Without the plug-in, `mdb lanes` prints "mdb lanes has no commands: the mdb-lanes plug-in is not
installed" and exits with status 2; a bare `mdb debug` exits 2 with "mdb lanes cannot run
'debug', which is not an engine verb: the mdb-lanes plug-in is not installed". The plug-in's own
documentation lives with the maintainers' internal experiments in the repository
(`packages/mdb-experiments/docs/`, e.g. `USAGE.md`, `LOCAL-DEBUGGER.md`, `BACKENDS.md`), not in
this package.

# Two packages, one repository

The repository root builds `saturn-pub`, imported as `saturn_pub`. The second
package lives at `packages/mdb`, builds `mdb-runtime`, and is imported as `mdb`.
Each has its own `pyproject.toml`, source tree, tests, examples and documentation.
MDB may depend on and import the public `saturn_pub` package through declared
package requirements. The dependency direction is `mdb -> saturn_pub`; the
Saturn-pub core remains independent of MDB. MDB declares `saturn-pub>=0.4.1,<0.5`
and retains its existing state and numerical contracts. The new public-session
adapter delegates execution to Saturn-pub; see [public-session projects](../packages/mdb/docs/SATURN-PUB-PROJECTS.md).
The [documentation index](README.md) covers both packages; start with the
[MDB overview](mdb.md) or [Saturn capability guide](usage.md).

## Choose and install a package

From the repository root:

```sh
python -m pip install '.[ar,diffusion,train]'
python -m pip install './packages/mdb[autoregressive]'
```

MDB installs the Saturn-pub metadata/execution core as a dependency. To use the
local versions together, install `pip install . ./packages/mdb`. Both bare cores
remain usable without model frameworks.
Saturn model extras use public dependencies. MDB's scheduler and diffusion extras
pin a public mrun-pub Git revision over HTTPS;
see [MDB installation](../packages/mdb/docs/INSTALL.md).
Use separate environments when reproducing their qualified model runs: MDB pins
Torch 2.13.0, while Saturn's current development lock pins Torch 2.14.0.

MDB commands documented relative to its checkout should run from `packages/mdb`:

```sh
cd packages/mdb
python -m pip install '.[autoregressive,dev]'
python -m pytest -q
python -I "$PWD/examples/resident_lab.py" --tiny qwen2 --out ./results/first-lab
```

Saturn's existing root installation, examples, notebooks and `uv.lock` keep their
original paths. MDB is deliberately not a member of the root uv workspace: its
private extras and qualification pins must not affect Saturn dependency resolution.

## Build and validate

With the `build` tool installed, from the repository root:

```sh
python -m pytest -q
python -m build --outdir dist/saturn
python -m build packages/mdb --outdir packages/mdb/dist
python scripts/check_package_isolation.py
```

Run MDB's tests in its own process/environment. Its test import guard permits
public `saturn_pub` imports and rejects private Saturn and common workspace
imports; collecting both suites together
would defeat their isolation and can cause test-module name collisions.
CI runs the suites separately on Python 3.10 and 3.12, then checks the built
distributions and bare installed imports. Saturn linting excludes the MDB tree;
MDB retains its own configuration and qualified source bytes.

Public MDB CI runs the autoregressive and dependency-free mechanics without
private Git access. Image execution and worker payload staging tests explicitly
skip when their optional mrun dependency is absent. The full integration suite
requires the dependencies from MDB's installation guide; a public mechanics pass
does not qualify that private integration or a deployed scheduler.

The isolation check verifies both wheels and source distributions, installs the
two bare wheels into a fresh environment outside the checkout, and checks each
API with model frameworks and private workspace packages forbidden. MDB may
import the installed Saturn-pub package; Saturn-pub is checked with MDB forbidden
to preserve the one-way dependency. Saturn distributions exclude the MDB package;
MDB wheels contain only MDB runtime code and declare any Saturn-pub dependency
instead of bundling its source.

## Migration and compatibility

MDB's original history through `70f2b33` was imported without squashing using a
Git subtree at `packages/mdb`. Its current uncommitted research-session and paper
adapter work was copied into that package before relocation. Runtime module bytes,
serialized schema names, checkpoint identities and extraction provenance remain
unchanged by the move. Existing qualification receipts remain historical evidence
for their recorded sources; relocation does not certify a new model/backend lane.

The domains workspace keeps `~/domains/mdb` as a compatibility symlink to
`saturn-pub/packages/mdb`. Local result archives and indexes remain ignored and
are retained with the relocated package. The original standalone checkout is
archived under `~/domains/backups/mdb-before-monorepo-*` for recovery.
That workspace symlink is not needed by installed code or other checkouts.

Maintainer extraction tools that consume the private Saturn research checkout
require an explicit `--upstream /path/to/saturn`. Do not rerun extraction as part
of ordinary installation or overwrite adapted runtime modules casually.

Shared contracts and cross-package backends are permitted. Prefer direct reuse
of public Saturn-pub APIs over copied implementations or a separate bridge solely
for package isolation. Declare dependencies, reconcile model-library pins, and
test installed-package integration before relying on a shared execution path.
Co-location does not make existing cuts interchangeable or qualify deployed
fleet debugger attachment.

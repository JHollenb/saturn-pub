# Releases and installing one

mdb (`mdb-runtime`) and the `saturn-pub` package it runs on are released from the
[saturn-pub repository](https://github.com/JHollenb/saturn-pub). Neither is on PyPI: a release is
a Git tag, and you install it with pip from that tag.

## Which ref to install

| Ref | What it is | Use it for |
|---|---|---|
| tag `mdb-vX.Y.Z` (latest: `mdb-v0.2.3`) | a released version; never moves | experiments, anything you cite or rerun |
| branch `release/mdb-X.Y` (now `release/mdb-0.2`) | the newest patch release of that minor series; moves when a fix is released | staying on a series and taking its fixes |
| branch `main` | development; version `X.Y.Z.devN` | changes not released yet (config files, new flags); may break |

The tags are listed in the repository (`git ls-remote --tags https://github.com/JHollenb/saturn-pub 'mdb-v*'`)
and each version's changes are in [CHANGELOG.md](../CHANGELOG.md).

## Installing a release

`mdb-runtime` depends on `saturn-pub`, which pip cannot find on PyPI, so name both, from the same
ref. `mdb-runtime` is in the `packages/mdb` subdirectory:

```sh
REF=mdb-v0.2.3
pip install "saturn-pub @ git+https://github.com/JHollenb/saturn-pub@$REF" \
  "mdb-runtime[autoregressive] @ git+https://github.com/JHollenb/saturn-pub@$REF#subdirectory=packages/mdb"
mdb version          # 0.2.3
```

`REF=release/mdb-0.2` installs the newest 0.2 patch release the same way. Installing
`mdb-runtime` alone from Git fails with "saturn-pub was not found in the package registry". The
extras are the same as from a checkout ([INSTALL.md](INSTALL.md#extras)).

From a clone, check out the ref and install both directories:

```sh
git clone https://github.com/JHollenb/saturn-pub && cd saturn-pub
git checkout mdb-v0.2.3          # or main for development
pip install . "./packages/mdb[autoregressive]"
```

`mdb-runtime` pins a range of `saturn-pub` versions (`saturn-pub>=0.4.1,<0.5` for 0.2.3; see
`dependencies` in `packages/mdb/pyproject.toml` of the ref). The `saturn-pub` at the same ref is
always inside that range.

## How releases are made

- **main is development.** Every change lands on `main` first. Its version is the next patch with
  a `.dev0` suffix (`0.2.4.dev0` after 0.2.3), in `packages/mdb/pyproject.toml` and
  `mdb.__version__`. `CHANGELOG.md` collects changes under `[Unreleased]`.
- **One release branch per minor version**, `release/mdb-X.Y`, started from a release commit on
  `main` (`release/mdb-0.2` starts at 0.2.2; 0.2.3 is the first release made on it). Users who
  want a stable series follow it.
- **Fixes are cherry-picked.** A fix for a released series is committed on `main`, then
  cherry-picked onto `release/mdb-X.Y` with `git cherry-pick -x` (the commit names its `main`
  original). New features and changes of behaviour stay on `main` until the owner cuts the next
  release from it. The owner sets every version number; before 1.0 a release cut from `main` may
  be a patch (0.2.1 and 0.2.2 shipped features and a record-id change as patches) or a new minor
  series with its own `release/mdb-X.Y` branch.
- **A patch release** is one commit on the release branch that sets the version in both places and
  adds a `## [X.Y.Z] — DATE` section to `CHANGELOG.md`; `main`'s changelog notes it too.
- **Tags** are annotated, named `mdb-vX.Y.Z`, on that commit of the release branch. A tag is never
  moved or reused.
- **saturn-pub** has its own version (`pyproject.toml` at the repository root, `saturn_pub.__version__`).
  When an mdb fix needs a saturn-pub change, both are cherry-picked; if `mdb-runtime` then needs
  the newer saturn-pub, its pin is raised in the same commit, so that one ref always installs a
  matching pair.

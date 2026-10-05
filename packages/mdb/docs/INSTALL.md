# Independent installation

MDB's installed runtime never searches for Saturn, Saturn-pub or a workspace
checkout. Maintainer extraction scripts are not part of the runtime wheel.

```sh
python -m pip install .
mdb capabilities
mdb verify --root /path/to/export
```

The base package has no model-library dependencies. For a caller-owned model,
install `.[autoregressive]` or `.[diffusion]`. Diffusion also installs mrun's native
checkpoint/phase adapter, even for a caller-owned pipeline. For scheduler launches, install
`.[all]`, or install a compatible mrun distribution separately. The optional
`mrun` extra pins private mrun and manalysis Git revisions; installing it requires
GitHub SSH access; the diffusion extra uses those same pins. Manalysis is a
transitive mrun dependency. No Saturn repository is required by either
installation route.

The pinned revisions are published Git objects: mrun `f002ce776e14` and
manalysis `2ea2eedca427`. The later local manalysis tracer change is not needed
by MDB and is excluded from the installation pin.

The pinned mrun client has an older `launch()` signature. MDB uses its guarded
`submit()` API when a declared CPU/disk reservation needs that compatibility
path, preserving CUDA needs, strict preflight, soft host preference and retry
policy. Unsupported admission controls are refused rather than dropped.

The model-backed qualification lane uses Python 3.14.6, Torch 2.13.0,
Transformers 5.14.1, Diffusers 0.39.0 and Accelerate 1.14.0. Other dependency
versions and the GPU driver are recorded by the worker bootstrap. Python 3.10+
supports package mechanics; a different runtime fingerprint can refuse an old
model cut even when its bytes verify.

Configure mrun's scheduler endpoint and credentials through mrun's normal
configuration. `MDB_HOME` selects writable local evidence indexes; it defaults
to `~/.local/share/mdb`. Model weights remain in mrun's custody stores.

Each launch stages deterministic wheels of the installed MDB and mrun packages,
the selected worker and explicitly declared files. The bootstrap checks source,
wheel, request and configuration hashes before executing the worker. It uses an
offline uv environment without an editable checkout or `PYTHONPATH`. The fleet
cache defaults can be overridden with `MDB_WORKER_UV_CACHE_DIR` and
`MDB_WORKER_TORCH_HOME`; their resolved values are bound into the request.
Workers default to uv symlink installation to avoid cloning large cached Torch
trees across fleet filesystems. The cache must remain available for the life of
the worker. `MDB_WORKER_UV_LINK_MODE` can select `copy`, `hardlink` or `clone`;
the selected mode is sealed and checked by bootstrap, and recorded in the
environment receipt. Startup timing is separate from model execution timing.

The qualification fleet's FUSE/NTFS volume later refused new export and cache
files despite available bytes. Remaining proofs use an ext4 durable root and
`MDB_WORKER_UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache`. A cache with linked archive
roots was not installable by uv; `qualification/materialize_cache.py` copies
the recorded runtime's selected archives into real ext4 directories. This is
an operational cache repair, not a requirement to copy model weights or run a
Saturn checkout. Keep the setup inventories and failed bootstrap records.

No fleet refresh, remote publication or service deployment is part of this
extraction. Live broker operation requires an independently qualified deployed
mrun service. Finite scripts exercise the same session operations without that
deployment.

## Maintainer validation

```sh
uv build --offline --out-dir dist
python -m pytest tests -q
PYTHONPATH=src python qualification/submit.py preflight mamba
```

For an independent wheel check, install the built wheel into a fresh environment,
copy only the tests and examples outside the source repository, remove
`PYTHONPATH`, and invoke Python with `-I`. Do not reuse the repository's pytest
configuration, which intentionally adds `src` for development tests. The test
import guard rejects Saturn and common-harness imports.

Extraction scripts capture an upstream source closure once. MDB-specific
adaptations are committed source changes; rerunning raw extraction can overwrite
them. `PROVENANCE.json` records original, extracted and final file hashes.

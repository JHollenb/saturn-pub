# Independent installation

In the Saturn monorepo, run these commands from `packages/mdb`. From the
repository root, the equivalent base install is `pip install ./packages/mdb`.
MDB retains its own distribution and depends on `saturn-pub>=0.4.0,<0.5`.
For development from this monorepo, install both local distributions together
from the repository root: `pip install . ./packages/mdb`.

MDB's installed runtime never searches for Saturn, Saturn-pub or a workspace
checkout. Maintainer extraction scripts are not part of the runtime wheel.

```sh
python -m pip install .
mdb debug
mdb doctor
mdb help
mdb capabilities
mdb verify --root /path/to/export
```

The base package has no model-library dependencies. For a caller-owned language
model or the tiny CPU Qwen console, install `.[autoregressive]` and run
`mdb debug --tiny qwen2`. The default `mdb debug` is a pure-Python Counter demo.
See the [local debugger guide](LOCAL-DEBUGGER.md) for opening local checkpoints;
these language-model entry points need no mrun installation or scheduler.
Qwen stateful execution requires BF16 and checks it before creating the session.
The model-library extra retains the qualified dependency pins below.

Diffusion also installs mrun-pub's native
checkpoint/phase adapter, even for a caller-owned pipeline. For scheduler launches,
install .[all] or .[mrun]. These extras use an immutable public HTTPS Git revision
of [mrun-pub](https://github.com/JHollenb/mrun-pub); no SSH access, manalysis,
common workspace, MLflow or MinIO installation is required. The import namespace
remains mrun.

For caller-owned Transformers or custom framework runners, use
`BenchSession.from_backend` from the [backend guide](BACKENDS.md).
`.[transformerlens]` supplies the independent TransformerLens 4.2.0 integration
environment. A custom dependency-free runner needs only the base MDB package.

MDB preserves guarded admission, CUDA needs, strict preflight, soft host preference
and retry policy. Unsupported admission controls are refused.

The model-backed qualification lane uses Python 3.14.6, Torch 2.13.0,
Transformers 5.14.1, Diffusers 0.39.0 and Accelerate 1.14.0. Other dependency
versions and the GPU driver are recorded by the worker bootstrap. Python 3.10+
supports package mechanics; a different runtime fingerprint can refuse an old
model cut even when its bytes verify.

Configure mrun's scheduler endpoint and credentials through mrun's normal
configuration. `MDB_HOME` selects writable local evidence indexes; it defaults
to `~/.local/share/mdb`. Model weights remain in mrun's custody stores.

Each launch stages deterministic wheels of the installed MDB, Saturn-pub and mrun packages,
the selected worker and explicitly declared files. The bootstrap checks source,
wheel, request and configuration hashes before executing the worker. It uses an
offline uv environment without an editable checkout or `PYTHONPATH`. The portable worker
cache defaults (~/.cache/mdb) can be overridden with `MDB_WORKER_UV_CACHE_DIR` and
`MDB_WORKER_TORCH_HOME`; their resolved values are bound into the request. Model roots come from\nMDB_WORKER_MODEL_ROOTS or LLM_MODELS_EXTRA_ROOT. MDB_WORKER_PYTHON overrides\nthe recorded qualification interpreter selection.
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

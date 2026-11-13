"""Stage and admit the mdb.attention_port FLUX.2 Klein smoke through guarded mrun launch.

Pattern copied from saturn/experiments/2026-11-11-flux-read-graph/stage_submit.py.
The payload wheel is built from whichever ``mdb`` the interpreter imports; set
PYTHONPATH to the checkout under test (mdb is an editable install) and confirm
the printed ``mdb_source``. Run:
  PYTHONPATH=<checkout>/packages/mdb/src <mdb venv>/python stage_attention_port_smoke.py RUN
  PYTHONPATH=<checkout>/packages/mdb/src <mdb venv>/python stage_attention_port_smoke.py RUN --submit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "attention_port_smoke_worker.py"
ENVIRONMENT = {"uv_link_mode": "symlink", "uv_cache_dir": "/mnt/hdd8tb/mdb/uv-cache",
               "torch_home": "/mnt/hdd8tb/mdb/torch", "extra_model_roots": "/mnt/big/llm-models", "python": "3.14.6"}
EXTRA_ENV = {"TMPDIR": "/mnt/hdd8tb/mdb/tmp", "CUDA_CACHE_PATH": "/mnt/hdd8tb/mdb/cuda-cache",
             "TORCHINDUCTOR_CACHE_DIR": "/mnt/hdd8tb/mdb/torchinductor-cache",
             "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
CAT = "a photorealistic black cat sitting beside a red ball in fresh snow at dawn, soft blue light"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write("\n")


def bind():
    for key, name in {"uv_cache_dir": "MDB_WORKER_UV_CACHE_DIR", "torch_home": "MDB_WORKER_TORCH_HOME",
                      "extra_model_roots": "MDB_WORKER_MODEL_ROOTS", "python": "MDB_WORKER_PYTHON",
                      "uv_link_mode": "MDB_WORKER_UV_LINK_MODE"}.items():
        os.environ[name] = ENVIRONMENT[key]
    os.environ.setdefault("MRUN_URL", "http://127.0.0.1:19025")
    from mdb_lanes.execution_environment import intent
    resolved = intent()
    if any(resolved[key] != value for key, value in ENVIRONMENT.items()):
        raise ValueError("execution environment intent differs")
    return resolved


def command():
    from mdb_lanes.payload import worker_command
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for name, value in EXTRA_ENV.items():
        args.insert(1, name + "=" + value)
    return args


def stage(run, head_file, vram_mb, ram_mb):
    import mdb
    environment = bind()
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    heads = json.loads(Path(head_file).read_text())["sets"]["cat:ball"]["own60"]
    config = dict(schema="mdb-attention-port-smoke-plan-v1", model="flux2-klein-4b-distilled",
                  model_path="/mnt/big/llm-models/FLUX.2-klein-4B", output_scratch="/mnt/hdd8tb/mdb/research",
                  height=256, width=256, steps=4, seed=4242, prompt=CAT, word="ball", heads=heads,
                  head_sets_sha256=sha(head_file), reference_job="job-a0bca63168b4",
                  reference_arm="z:own60:ball")
    run.mkdir(parents=True, exist_ok=False)
    save(run / "plan.json", config)
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [WORKER, "plan.json", preflight.name],
                     "path_token_allowlist": [config["model_path"], config["output_scratch"], "attention-port-smoke"]})
    extras = [(run / "plan.json", "plan.json"), (preflight, preflight.name)]
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    branches = 19
    request = dict(schema="mdb-attention-port-smoke-request-v1", config=config, model=config["model"], task_family="forward",
                   backend="diffusers", device="cuda", dtype="bfloat16", batch_size=1, height=256, width=256,
                   needs={"cuda": True}, execution_environment=environment,
                   config_bytes_sha256=hashlib.sha256(raw).hexdigest(),
                   source_hashes={name: sha(path) for path, name in [(HERE / WORKER, WORKER), *extras]},
                   scheduling_contract=one_lease_scheduling_contract(
                       model_loads=1, logical_operations=branches, logical_branches=branches, logical_checkpoint_replays=0,
                       max_local_work_items=branches,
                       checkpoint_policy="one resident Klein 4B lease; two prompts encoded first, encoder released; 19 short branches in process; no retries"),
                   resource_exception=dict(ram_mb=ram_mb, vram_mb=vram_mb, cpu_threads=4, disk_gb=1,
                                           est_wall_s=150, timeout_s=600,
                                           reason="Klein 4B BF16 B1 256px 4-step resident lane. job-a0bca63168b4 (same lane) measured 8738 MB peak allocated."))
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request, config_bytes=raw, additional_files=extras)
    save(run / "staged-command.json", {"command": command(), "request_sha256": request["request_sha256"],
                                        "worker_sha256": sha(run / "payload" / WORKER), "mdb_source": mdb.__file__})
    return dict(run=str(run), request_sha256=request["request_sha256"], mdb_source=mdb.__file__,
                worker_sha256=sha(run / "payload" / WORKER))


def submit(run, priority):
    bind()
    from mdb_lanes.job_support import guarded_mrun_launch
    request = json.loads((run / "request.json").read_text())
    if (run / "submission.json").exists():
        raise ValueError("already submitted")
    for name, digest in request["source_hashes"].items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("staged payload changed: " + name)
    resource = request["resource_exception"]
    job = guarded_mrun_launch(command(), request=request, experiment="mdb-attention-port-smoke",
                              task="forward", model=request["model"], payload=run / "payload", prefer_host="beast",
                              gpu=True, needs=request["needs"], ram_mb=resource["ram_mb"], vram_mb=resource["vram_mb"],
                              disk_gb=resource["disk_gb"], cpu_threads=resource["cpu_threads"],
                              est_wall_s=resource["est_wall_s"], timeout_s=resource["timeout_s"], priority=priority,
                              retry_on_kill=False, preflight=True,
                              note="mdb.attention_port smoke: FLUX.2 Klein 4B shams + per-head cut z:own60:ball vs job-a0bca63168b4")
    save(run / "submission.json", {"job_id": str(job)})
    return dict(job_id=str(job), run=str(run))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--head-file", type=Path,
                        default=Path("/Users/jakeholl/domains/saturn/experiments/2026-11-11-flux-read-graph/head_sets3.json"))
    parser.add_argument("--vram-mb", type=int, default=12000)
    parser.add_argument("--ram-mb", type=int, default=20000)
    parser.add_argument("--priority", type=int, default=70)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    print(json.dumps(submit(run, args.priority) if args.submit else stage(run, args.head_file, args.vram_mb, args.ram_mb), sort_keys=True))

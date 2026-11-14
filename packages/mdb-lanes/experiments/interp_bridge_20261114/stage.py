"""Stage and submit interp-bridge jobs (mrun, Beast). Offline except ``submit``; no model is loaded here.

  PY=/Users/jakeholl/domains/saturn-pub/packages/mdb/.venv/bin/python
  export PYTHONPATH=$W/packages/mdb-lanes/src:$W/packages/mdb/src:$W/src
  $PY stage.py stage RUN --job smoke [--vram-mb N]
  $PY stage.py stage RUN --job cuts --groups groups.json [--vram-mb N]
  $PY stage.py submit RUN [--attempt N]

Environment and payload layout follow ``qualification/flux_read_panel_stage.py`` (same Beast paths,
uv cache, Python 3.14.6 and pinned torch/diffusers/transformers), plus ``nnsight==0.8.2``. The
lane worker (``flux_read_panel_worker.py``) ships unchanged in the payload; the worker imports
its ``Lane`` so load, encoding, noise, conditioning and decode are the lane's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUAL = HERE.parents[1] / "qualification"
sys.path.insert(0, str(QUAL))
import flux_read_panel_stage as lane_stage  # noqa: E402

WORKER = "interp_bridge_worker.py"
LANE_WORKER = QUAL / "flux_read_panel_worker.py"
NNSIGHT = "nnsight==0.8.2"
# Without --offline uv resolves online; pin the two open-ended lane requirements to the versions the
# lane's jobs recorded (job-a34e4d6fe9dc software: numpy 2.5.4, pillow 13.0.0) so they cannot drift.
EXTRA_REQUIREMENTS = (NNSIGHT, "numpy==2.5.4", "pillow==13.0.0")
WORDS = ("red", "ball")
OTHER = {"red": "black", "ball": "snow"}
REFERENCE_HASHES = {   # final-latent sha256 of the lane's roots, seed 4242 (every job of mdb-flux-read-panels-20261112)
    "native:cat": "a3cef09bc04a61f4ba68adfb861de0c825349ea6a3cea3c71505772ee8cee605",
    "del:red": "c1f45449277b655dabc3b9c2f04f5aaf4bf05e5769154857d160078db8a2519c",
    "iso:red": "7bae8fef22723c016c927ba0388e4e64640851bdce3a88eac093a497536145cf",
    "del:ball": "5ae513b07654816e9a347fadcee32c938c5a81c408a60ccc27f05c22ea902b63",
    "iso:ball": "6c0076c6ed2402042a56526fe4c87cdca695a95d5ab730599d208f0b1f355c2c",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_config(job):
    return dict(schema="mdb-interp-bridge-run-v1", mode=job, model="flux2-klein-4b-distilled",
                model_path=lane_stage.MODEL_PATH, output_scratch=lane_stage.SCRATCH, height=256, width=256, steps=4,
                prompt=lane_stage.PROMPTS["cat"], words=list(WORDS), other=OTHER, seed=4242,
                reference_hashes=REFERENCE_HASHES)


def smoke_config():
    return dict(base_config("smoke"), cells=[[2, 7], [10, 13], [20, 21]], capture_steps=[0, 2],
                cut_cell=[10, 13], cut_cell_word="ball", timing_reps=3, grad_check_block=10, checkpoint=True,
                extras_joint_block=2, extras_single_block=10, max_wall_seconds=600)


def cuts_config(groups):
    """groups.json from analyze.py groups: {"branches": [{"id", "word", "cut", "partition", "group", "units"}]}."""
    g = json.loads(Path(groups).read_text())
    branches = list(g["branches"])
    for w in WORDS:
        for cut in ("zero", "mask"):
            branches.append(dict(id=f"{cut[0]}:none:{w}", word=w, cut=cut, partition=None, group=None, units=None))
    return dict(base_config("cuts"), branches=branches, groups_sha256=sha(groups), max_wall_seconds=540)


def worker_command():
    from mdb_lanes.payload import worker_command as wc
    args = wc(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for name, value in lane_stage.EXTRA_ENV.items():
        args.insert(1, name + "=" + value)
    if "--offline" in args:          # nnsight and its pure-Python dependencies are resolved into the Beast uv cache
        args.remove("--offline")
    i = args.index("python")
    args[i:i] = [x for r in EXTRA_REQUIREMENTS for x in ("--with", r)]
    return args


def stage(run, job, vram_mb, ram_mb, est_wall_s, timeout_s, groups=None):
    import mdb
    import mdb_lanes
    environment = dict(lane_stage.bind())
    environment["requirements"] = list(environment["requirements"]) + list(EXTRA_REQUIREMENTS)
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    run = Path(run)
    run.mkdir(parents=True, exist_ok=False)
    config = smoke_config() if job == "smoke" else cuts_config(groups)
    work = 60 if job == "smoke" else len(config["branches"]) + 5
    if work > 256:
        raise ValueError("more than 256 work items")
    preflight = run / "preflight.v1.json"
    lane_stage.save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [WORKER, preflight.name],
                                "path_token_allowlist": [lane_stage.MODEL_PATH, lane_stage.SCRATCH, "interp-bridge"]})
    extras = [(HERE / "bridge.py", "bridge.py"), (LANE_WORKER, "flux_read_panel_worker.py"), (preflight, preflight.name)]
    if groups:
        extras.append((Path(groups), "groups.json"))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    lanes_src = Path(mdb_lanes.__file__).resolve().parent
    source_hashes = {WORKER: sha(HERE / WORKER), **{name: sha(path) for path, name in extras},
                     **{f"mdb_lanes/{m}.py": sha(lanes_src / f"{m}.py")
                        for m in ("attention_port", "attention_panels", "flux_read_panels")}}
    request = dict(schema="mdb-interp-bridge-request-v1", config=config, model=config["model"], task_family="forward",
                   backend="diffusers", device="cuda", dtype="bfloat16", batch_size=1, height=256, width=256,
                   needs={"cuda": True}, execution_environment=environment,
                   config_bytes_sha256=hashlib.sha256(raw).hexdigest(), source_hashes=source_hashes,
                   scheduling_contract=one_lease_scheduling_contract(
                       model_loads=1, logical_operations=work, logical_branches=work, logical_checkpoint_replays=0,
                       max_local_work_items=256,
                       checkpoint_policy="one resident Klein 4B lease; prompts encoded first, encoder released; "
                                         "all arms, traces and backward passes in process; no retries"),
                   resource_exception=dict(ram_mb=ram_mb, vram_mb=vram_mb, cpu_threads=4, disk_gb=2,
                                           est_wall_s=est_wall_s, timeout_s=timeout_s,
                                           reason="Klein 4B BF16 B1 256px 4-step resident lane (lane peak 7991 MB "
                                                  "allocated, job-a34e4d6fe9dc); smoke adds a block-checkpointed "
                                                  "backward through one denoising step at a time"))
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    lane_stage.save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request, config_bytes=raw,
                 additional_files=extras)
    cmd = worker_command()
    lane_stage.save(run / "staged-command.json", {"command": cmd, "request_sha256": request["request_sha256"],
                                                  "worker_sha256": sha(run / "payload" / WORKER),
                                                  "mdb_source": mdb.__file__, "mdb_lanes_source": mdb_lanes.__file__,
                                                  "work_items": work})
    return dict(run=str(run), job=job, request_sha256=request["request_sha256"], mdb_source=mdb.__file__,
                mdb_lanes_source=mdb_lanes.__file__, worker_sha256=sha(run / "payload" / WORKER), work_items=work,
                command_tail=cmd[-8:])


def submit(run, priority, attempt=1):
    lane_stage.bind()
    from mdb_lanes.job_support import guarded_mrun_launch
    run = Path(run)
    request = json.loads((run / "request.json").read_text())
    staged = json.loads((run / "staged-command.json").read_text())
    if (run / "submission.json").exists():
        raise ValueError("already submitted")
    for name, digest in request["source_hashes"].items():
        if name.startswith("mdb_lanes/"):
            continue
        if sha(run / "payload" / name) != digest:
            raise ValueError("staged payload changed: " + name)
    resource = request["resource_exception"]
    experiment = "mdb-interp-bridge-" + request["config"]["mode"]
    if attempt > 1:
        experiment += f"-attempt{attempt}"
    job = guarded_mrun_launch(staged["command"], request=request, experiment=experiment, task="forward",
                              model=request["model"], payload=run / "payload", prefer_host="beast", gpu=True,
                              needs=request["needs"], ram_mb=resource["ram_mb"], vram_mb=resource["vram_mb"],
                              disk_gb=resource["disk_gb"], cpu_threads=resource["cpu_threads"],
                              est_wall_s=resource["est_wall_s"], timeout_s=resource["timeout_s"], priority=priority,
                              retry_on_kill=False, preflight=True,
                              note="mdb interp bridge (nnsight vs attention_port; attribution): " +
                                   request["config"]["mode"])
    lane_stage.save(run / "submission.json", {"job_id": str(job)})
    return dict(job_id=str(job), run=str(run))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stage")
    p.add_argument("run", type=Path)
    p.add_argument("--job", required=True, choices=("smoke", "cuts"))
    p.add_argument("--groups", type=Path)
    p.add_argument("--vram-mb", type=int, default=13000)
    p.add_argument("--ram-mb", type=int, default=24000)
    p.add_argument("--est-wall-s", type=int, default=420)
    p.add_argument("--timeout-s", type=int, default=840)
    p = sub.add_parser("submit")
    p.add_argument("run", type=Path)
    p.add_argument("--priority", type=int, default=70)
    p.add_argument("--attempt", type=int, default=1)
    a = parser.parse_args()
    if a.cmd == "stage":
        print(json.dumps(stage(a.run.resolve(), a.job, a.vram_mb, a.ram_mb, a.est_wall_s, a.timeout_s,
                               a.groups), sort_keys=True))
    else:
        print(json.dumps(submit(a.run.resolve(), a.priority, a.attempt), sort_keys=True))

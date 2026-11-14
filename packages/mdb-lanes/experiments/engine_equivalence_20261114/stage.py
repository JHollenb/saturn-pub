"""Stage and submit the lane-vs-engine equivalence smoke job (mrun, Beast). Offline except ``submit``.

  PY=/Users/jakeholl/domains/saturn-pub/packages/mdb/.venv/bin/python
  export PYTHONPATH=$W/packages/mdb-lanes/src:$W/packages/mdb/src:$W/src
  $PY stage.py stage RUN [--vram-mb N]
  $PY stage.py submit RUN

Environment, uv cache and worker command follow ``qualification/qwen_panel_stage.py`` (imported
from the lane's qualification directory, as the interp-bridge experiment does). The payload carries
the sealed FP32 read plan and seal from ``qualification/qwen_panels_20261112/full``.
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
import qwen_panel_stage as lane_stage  # noqa: E402

WORKER = "worker.py"
PLANS = QUAL / "qwen_panels_20261112" / "full"
PLAN = {"file": "read-float32.plan.json", "seal": "read-float32.seal.json",
        "plan_sha256": "8590b4f71741cd61367c933031b81023dcc0541d895b394eeef996d5c41c3fdc"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command():
    from mdb_lanes.payload import worker_command
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for name, value in lane_stage.EXTRA_ENV.items():
        args.insert(1, name + "=" + value)
    return args


def stage(run, vram_mb, ram_mb, est_wall_s, timeout_s):
    import mdb
    import mdb_lanes
    from mdb_lanes import qwen_panels as Q
    environment = lane_stage.bind()
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    run = Path(run)
    run.mkdir(parents=True, exist_ok=False)
    config = {"schema": "mdb-qwen-engine-equivalence-config-v1", "model": Q.MODEL, "model_path": Q.MODEL_PATH,
              "expected_weight_file_sha256": Q.WEIGHT_FILE_SHA256, "output_scratch": lane_stage.OUTPUT_SCRATCH,
              "plan": PLAN}
    preflight = run / "preflight.v1.json"
    lane_stage.save(preflight, {"schema_version": "mrun-preflight-config-v1",
                                "hashed_roots": [WORKER, preflight.name, PLAN["file"], PLAN["seal"]],
                                "path_token_allowlist": [Q.MODEL_PATH, lane_stage.OUTPUT_SCRATCH, "qwen-engine"]})
    extras = [(PLANS / PLAN["file"], PLAN["file"]), (PLANS / PLAN["seal"], PLAN["seal"]), (preflight, preflight.name)]
    if sha(PLANS / PLAN["file"]) != PLAN["plan_sha256"]:
        raise ValueError("plan file bytes differ from the sealed plan sha256")
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    lanes_src = Path(mdb_lanes.__file__).resolve().parent
    source_hashes = {WORKER: sha(HERE / WORKER), **{name: sha(path) for path, name in extras},
                     **{f"mdb_lanes/{m}.py": sha(lanes_src / f"{m}.py")
                        for m in ("attention_port", "attention_panels", "qwen_panels", "qwen_engine")}}
    work = 1 + 48 + 25 + 1
    request = dict(schema="mdb-qwen-engine-equivalence-request-v1", config=config, model=Q.MODEL,
                   task_family="forward", backend="transformers", device="cuda", dtype="float32", batch_size=1,
                   needs={"cuda": True}, execution_environment=environment,
                   config_bytes_sha256=hashlib.sha256(raw).hexdigest(), source_hashes=source_hashes,
                   scheduling_contract=one_lease_scheduling_contract(
                       model_loads=1, logical_operations=work, logical_branches=work, logical_checkpoint_replays=0,
                       max_local_work_items=256,
                       checkpoint_policy="one Qwen2.5-1.5B FP32 load; lane arms then one engine run and one "
                                         "whatif_many excursion in process; no retries"),
                   resource_exception=dict(ram_mb=ram_mb, vram_mb=vram_mb, cpu_threads=4, disk_gb=2,
                                           est_wall_s=est_wall_s, timeout_s=timeout_s,
                                           reason=("Qwen2.5-1.5B FP32 B1, 22-token prompt; lane peak 6528 MB "
                                                   "measured; the engine path's peak is not yet measured "
                                                   "(first FP32 engine run of this case), so the reservation "
                                                   "adds about 3 GB to the lane's 1.3x")))
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    lane_stage.save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request, config_bytes=raw,
                 additional_files=extras)
    cmd = command()
    lane_stage.save(run / "staged-command.json", {"command": cmd, "request_sha256": request["request_sha256"],
                                                  "worker_sha256": sha(run / "payload" / WORKER),
                                                  "mdb_source": mdb.__file__, "mdb_lanes_source": mdb_lanes.__file__})
    return dict(run=str(run), request_sha256=request["request_sha256"], mdb_source=mdb.__file__,
                mdb_lanes_source=mdb_lanes.__file__, worker_sha256=sha(run / "payload" / WORKER),
                payload=sorted(p.name for p in (run / "payload").iterdir()))


def submit(run, priority):
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
    job = guarded_mrun_launch(staged["command"], request=request, experiment="mdb-qwen-engine-equivalence",
                              task="forward", model=request["model"], payload=run / "payload", prefer_host="beast",
                              gpu=True, needs=request["needs"], ram_mb=resource["ram_mb"], vram_mb=resource["vram_mb"],
                              disk_gb=resource["disk_gb"], cpu_threads=resource["cpu_threads"],
                              est_wall_s=resource["est_wall_s"], timeout_s=resource["timeout_s"], priority=priority,
                              retry_on_kill=False, preflight=True,
                              note="mdb Qwen read panel: lane path vs engine path, FP32 bit equality")
    lane_stage.save(run / "submission.json", {"job_id": str(job)})
    return dict(job_id=str(job), run=str(run))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stage")
    p.add_argument("run", type=Path)
    p.add_argument("--vram-mb", type=int, default=9500)
    p.add_argument("--ram-mb", type=int, default=12000)
    p.add_argument("--est-wall-s", type=int, default=300)
    p.add_argument("--timeout-s", type=int, default=600)
    p = sub.add_parser("submit")
    p.add_argument("run", type=Path)
    p.add_argument("--priority", type=int, default=70)
    a = parser.parse_args()
    if a.cmd == "stage":
        print(json.dumps(stage(a.run.resolve(), a.vram_mb, a.ram_mb, a.est_wall_s, a.timeout_s), sort_keys=True))
    else:
        print(json.dumps(submit(a.run.resolve(), a.priority), sort_keys=True))

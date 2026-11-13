"""Build, seal, stage and submit Qwen value/read-cut panels (``qwen_panel_worker.py``) via guarded mrun.

Pattern copied from ``stage_attention_port_smoke.py``. The payload wheel is built from whichever
``mdb`` the interpreter imports: set PYTHONPATH to this checkout and confirm ``mdb_source``.

  # 1. smoke (plans need no attention data)
  python qwen_panel_stage.py plans  --profile smoke --out PLANS_SMOKE
  python qwen_panel_stage.py stage  RUN_SMOKE --plans PLANS_SMOKE
  python qwen_panel_stage.py submit RUN_SMOKE
  # 2. full (read plans use the smoke's FP32 receiver attention for label-free sets and random-head mass)
  python qwen_panel_stage.py plans  --profile full --out PLANS_FULL --attention SMOKE/float32/receiver-attention.json
  python qwen_panel_stage.py stage  RUN_FULL --plans PLANS_FULL --vram-mb N --ram-mb N
  python qwen_panel_stage.py submit RUN_FULL
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "qwen_panel_worker.py"
ENVIRONMENT = {"uv_link_mode": "symlink", "uv_cache_dir": "/mnt/hdd8tb/mdb/uv-cache",
               "torch_home": "/mnt/hdd8tb/mdb/torch", "extra_model_roots": "/mnt/big/llm-models", "python": "3.14.6"}
EXTRA_ENV = {"TMPDIR": "/mnt/hdd8tb/mdb/tmp", "CUDA_CACHE_PATH": "/mnt/hdd8tb/mdb/cuda-cache",
             "TORCHINDUCTOR_CACHE_DIR": "/mnt/hdd8tb/mdb/torchinductor-cache",
             "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
OUTPUT_SCRATCH = "/mnt/hdd8tb/mdb/research"
SHARED_MODULES = ("mdb", "mdb_lanes.attention_port", "mdb_lanes.attention_panels", "mdb_lanes.attention_ranking",
                  "mdb_lanes.analysis_metrics", "mdb_lanes.qwen_panels", "mdb_lanes.native_replay_recipe")
SMOKE_VALUE_ARMS = ("native-recipient", "native-donor", "sham-late", "replay-recipient", "coalition-L20-both")
JOB_BRANCHES = 256          # mdb.scheduling_contract.MAX_LOCAL_WORK_ITEMS: logical branches per job
READ_LEASE_CAP = 199        # value plan (56) + one read lease (<= 199 + its own native copy) fits one job


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_plan(directory, name, panel, note):
    """Canonical plan JSON (its sha256 is the plan hash) + seal."""
    plan_path = directory / f"{name}.plan.json"
    plan_path.write_text(panel.canonical())
    seal = panel.seal(predictions={"note": note})
    save(directory / f"{name}.seal.json", seal)
    if sha(plan_path) != panel.sha256():
        raise ValueError("canonical plan file does not hash to the plan sha256")
    return {"name": name, "file": plan_path.name, "seal": f"{name}.seal.json", "plan_sha256": panel.sha256(),
            "family": panel.family, "arms_in_plan": len(panel.arms), "branches": panel.branch_count}


def build_plans(out, profile, attention_file=None):
    from mdb_lanes import qwen_panels as Q
    out.mkdir(parents=True, exist_ok=False)
    index = {"profile": profile, "lanes": {}}
    attention = None
    if profile == "full":
        if attention_file is None:
            raise ValueError("full plans need --attention (the smoke's FP32 receiver attention)")
        record = json.loads(Path(attention_file).read_text())
        if record.get("lane") != "float32":
            raise ValueError("label-free sets and random mass come from the FP32 lane")
        attention = record["attention"]
        sets, summary = Q.read_sets(attention)
        save(out / "label-free-ranking.json", {"source": str(attention_file), "source_sha256": sha(attention_file),
                                               "sets": sets, "summary": summary})
        index["label_free"] = {"recovered": summary["recovered"], "named_rank": summary["named_rank"]}
    panels = {}
    for lane in Q.LANES:
        panels[f"value-{lane}"] = Q.value_plan(lane)
        plans = [write_plan(out, f"value-{lane}", panels[f"value-{lane}"],
                            "no numeric prediction; the 56-arm value_coalition.plan() panel, re-run")]
        if profile == "smoke":
            plans[0]["arms"] = list(SMOKE_VALUE_ARMS)
            panels[f"read-smoke-{lane}"] = Q.smoke_read_plan(lane)
            plans.append(write_plan(out, f"read-smoke-{lane}", panels[f"read-smoke-{lane}"],
                                    "smoke: one read-cut set (L20H5) with floors"))
        elif lane == "float32":
            panels["read-float32"] = Q.read_plan(lane, attention)
            plans.append(write_plan(out, "read-float32", panels["read-float32"],
                                    "no numeric prediction; named heads L20H5, L21H0, L20H11 and label-free top 3/12"))
        else:
            # BF16 read cuts: the smoke panel (L20H5 + all-cut, both items) measures the BF16 backend floor;
            # the full read panel runs in FP32 (the branch budget allows one full read panel).
            panels[f"read-smoke-{lane}"] = Q.smoke_read_plan(lane)
            plans.append(write_plan(out, f"read-smoke-{lane}", panels[f"read-smoke-{lane}"],
                                    "BF16 backend floors: one read-cut set (L20H5) with floors"))
        index["lanes"][lane] = plans
    if profile == "full":
        index["jobs"] = job_layout(index, panels)
    save(out / "index.json", index)
    return index


def job_layout(index, panels):
    """Pack (lane, plan, lease) work units into jobs of at most JOB_BRANCHES branches, lanes in order."""
    units = []
    for lane in ("float32", "bfloat16"):
        for plan in index["lanes"][lane]:
            panel = panels[plan["name"]]
            ref = {k: plan[k] for k in ("name", "file", "seal", "plan_sha256", "family")}
            if plan["family"] == "read" and panel.branch_count > READ_LEASE_CAP:
                leases = panel.leases(cap=READ_LEASE_CAP)
                for lease in leases:
                    extra = 0 if any(b == "native" for _, b in lease["branches"]) else 1
                    units.append((lane, dict(ref, lease={"cap": READ_LEASE_CAP, "index": lease["lease"],
                                                         "of": lease["of"], "lease_sha256": lease["lease_sha256"]}),
                                  len(lease["branches"]) + extra))
            else:
                units.append((lane, ref, panel.branch_count))
    jobs, current, size = [], [], 0
    for lane, ref, n in units:
        if n > JOB_BRANCHES:
            raise ValueError(f"{ref['name']}: {n} branches exceed one job")
        if size + n > JOB_BRANCHES:
            jobs.append(current)
            current, size = [], 0
        current.append({"lane": lane, "plan": ref, "branches": n})
        size += n
    if current:
        jobs.append(current)
    return jobs


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


def module_hashes():
    import importlib
    return {name: sha(importlib.import_module(name).__file__) for name in SHARED_MODULES}


def run_config(plans_dir, *, job=None, device="cuda", dry_run=False, output_scratch=OUTPUT_SCRATCH, model_path=None):
    from mdb_lanes import qwen_panels as Q
    index = json.loads((plans_dir / "index.json").read_text())
    lanes = []
    if "jobs" in index:
        if job is None or not 0 <= job < len(index["jobs"]):
            raise ValueError(f"these plans are split into {len(index['jobs'])} jobs; pass --job 0..{len(index['jobs']) - 1}")
        for unit in index["jobs"][job]:
            if not lanes or lanes[-1]["dtype"] != unit["lane"]:
                lanes.append({"dtype": unit["lane"], "plans": []})
            lanes[-1]["plans"].append(unit["plan"])
    else:
        for lane in Q.LANES:
            entry = {"dtype": lane, "plans": [{k: p[k] for k in ("name", "file", "seal", "plan_sha256", "family")}
                                              | ({"arms": p["arms"]} if "arms" in p else {})
                                              for p in index["lanes"][lane]]}
            if index["profile"] == "smoke":
                entry["replay_check"] = ["native-recipient", "coalition-L20-both"]
            lanes.append(entry)
    return {"schema": "mdb-qwen-panels-config-v1", "profile": index["profile"], "job": job, "model": Q.MODEL,
            "model_path": model_path or Q.MODEL_PATH, "expected_weight_file_sha256": Q.WEIGHT_FILE_SHA256,
            "output_scratch": output_scratch, "device": device, "dry_run": dry_run, "lanes": lanes,
            "module_sha256": module_hashes(), "native_recipes": True}


def stage(run, plans_dir, vram_mb, ram_mb, est_wall_s, timeout_s, job=None):
    import mdb
    environment = bind()
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    config = run_config(plans_dir, job=job)
    run.mkdir(parents=True, exist_ok=False)
    save(run / "plan.json", config)
    files = sorted(p for p in plans_dir.iterdir() if p.name.endswith((".plan.json", ".seal.json")))
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
                     "hashed_roots": [WORKER, "plan.json", preflight.name, *[p.name for p in files]],
                     "path_token_allowlist": [config["model_path"], config["output_scratch"], "qwen-panels"]})
    extras = [(run / "plan.json", "plan.json"), (preflight, preflight.name), *[(p, p.name) for p in files]]
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    branches = 0
    index = json.loads((plans_dir / "index.json").read_text())
    if job is not None:
        branches = sum(unit["branches"] for unit in index["jobs"][job])
    else:
        for lane in config["lanes"]:
            for p in lane["plans"]:
                plan = json.loads((plans_dir / p["file"]).read_text())
                branches += len(p.get("arms") or plan["arms"])
    request = dict(schema="mdb-qwen-panels-request-v1", config=config, model=config["model"], task_family="forward",
                   backend="transformers", device="cuda", dtype="float32+bfloat16", batch_size=1,
                   needs={"cuda": True}, execution_environment=environment,
                   config_bytes_sha256=hashlib.sha256(raw).hexdigest(),
                   source_hashes={name: sha(path) for path, name in [(HERE / WORKER, WORKER), *extras]},
                   scheduling_contract=one_lease_scheduling_contract(
                       model_loads=1, logical_operations=branches, logical_branches=branches,
                       logical_checkpoint_replays=0, max_local_work_items=JOB_BRANCHES,
                       checkpoint_policy=("one Qwen2.5-1.5B load (FP32); FP32 lane, then parameters cast in place "
                                          "to BF16 for the BF16 lane; prefix caches built once per lane and "
                                          "deep-copied per arm; no retries")),
                   resource_exception=dict(ram_mb=ram_mb, vram_mb=vram_mb, cpu_threads=4, disk_gb=2,
                                           est_wall_s=est_wall_s, timeout_s=timeout_s,
                                           reason=("Qwen2.5-1.5B FP32 then BF16, B1, 22-token prompt + 3 tokens. "
                                                   "main-02 (BF16, native session) peaked at 3444 MB VRAM / 2065 MB RSS; "
                                                   "FP32 doubles the weights; smoke measures the peak")))
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request, config_bytes=raw,
                 additional_files=extras)
    save(run / "staged-command.json", {"command": command(), "request_sha256": request["request_sha256"],
                                        "worker_sha256": sha(run / "payload" / WORKER), "mdb_source": mdb.__file__,
                                        "module_sha256": config["module_sha256"]})
    return dict(run=str(run), request_sha256=request["request_sha256"], mdb_source=mdb.__file__,
                worker_sha256=sha(run / "payload" / WORKER), branches=branches, module_sha256=config["module_sha256"])


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
    job = guarded_mrun_launch(command(), request=request, experiment="mdb-qwen-panels",
                              task="forward", model=request["model"], payload=run / "payload", prefer_host="beast",
                              gpu=True, needs=request["needs"], ram_mb=resource["ram_mb"], vram_mb=resource["vram_mb"],
                              disk_gb=resource["disk_gb"], cpu_threads=resource["cpu_threads"],
                              est_wall_s=resource["est_wall_s"], timeout_s=resource["timeout_s"], priority=priority,
                              retry_on_kill=False, preflight=True,
                              note=f"mdb Qwen panels ({request['config']['profile']}): value + read-cut panels, FP32 and BF16")
    save(run / "submission.json", {"job_id": str(job)})
    return dict(job_id=str(job), run=str(run))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("plans")
    p.add_argument("--profile", choices=("smoke", "full"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--attention", type=Path)
    s = sub.add_parser("stage")
    s.add_argument("run", type=Path)
    s.add_argument("--plans", type=Path, required=True)
    s.add_argument("--vram-mb", type=int, default=9000)
    s.add_argument("--ram-mb", type=int, default=12000)
    s.add_argument("--est-wall-s", type=int, default=240)
    s.add_argument("--timeout-s", type=int, default=600)
    s.add_argument("--job", type=int)
    u = sub.add_parser("submit")
    u.add_argument("run", type=Path)
    u.add_argument("--priority", type=int, default=70)
    args = parser.parse_args()
    if args.action == "plans":
        out = build_plans(args.out.resolve(), args.profile, args.attention)
    elif args.action == "stage":
        out = stage(args.run.resolve(), args.plans.resolve(), args.vram_mb, args.ram_mb, args.est_wall_s,
                    args.timeout_s, args.job)
    else:
        out = submit(args.run.resolve(), args.priority)
    print(json.dumps(out, sort_keys=True, default=str))

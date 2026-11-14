"""Stage and submit the predicted-write experiment's mrun jobs; build its sealed panel plans.

  PY=/Users/jakeholl/domains/saturn-pub/packages/mdb/.venv/bin/python
  export PYTHONPATH=<worktree>/packages/mdb-lanes/src:<worktree>/packages/mdb/src     # not <worktree>/src
  $PY stage.py capture RUN --profile smoke|full          # FLUX capture worker (pw_worker.py)
  $PY stage.py plans --capture DIR --out plans           # sets + sealed read plans (main, f3)
  $PY stage.py panel RUN --plans plans --plan main --lease 0   # flux_read_panel_worker.py, unchanged
  $PY stage.py qwen RUN                                  # qwen_pw_worker.py
  $PY stage.py submit RUN [--attempt N]

The read-panel lane's own ``flux_read_panel_stage.stage`` declares ``mdb/<module>.py`` source hashes,
which no longer exist after the mdb / mdb-lanes split; this script declares ``mdb_lanes/<module>.py``.
No model is loaded here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANES = HERE.parent.parent                                   # packages/mdb-lanes
QUAL = LANES / "qualification"
sys.path.insert(0, str(QUAL))
import flux_read_panel_stage as FS  # noqa: E402

WORDS = {"red": "cat", "black": "cat", "ball": "cat", "snow": "cat", "night": "car", "rain": "car"}
SEED = 4242
CLOSURE_CELLS = [[0, 1], [0, 4], [1, 7], [2, 14], [3, 22]]
VSWAP_ARMS = [
    {"name": "vswap-all", "cond": "native", "rows": "all", "mode": "delta"},
    {"name": "vswap-word", "cond": "native", "rows": "word", "mode": "delta"},
    {"name": "vswap-nonword", "cond": "native", "rows": "nonword", "mode": "delta"},
    {"name": "vswap-sham", "cond": "native", "rows": "all", "mode": "sham"},
    {"name": "iso-vswap-all", "cond": "iso", "rows": "all", "mode": "delta"},
]
PW_MODULES = ("attention_port", "attention_panels", "flux_read_panels", "predicted_write")
QWEN_MODULES = ("analysis_metrics", "qwen_panels", "predicted_write")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    FS.save(path, value)


def lanes_src():
    import mdb_lanes
    return Path(mdb_lanes.__file__).resolve().parent


def command(worker, extra=()):
    from mdb_lanes.payload import worker_command
    args = worker_command(worker)
    args.insert(args.index("python") + 1, "-B")
    for name, value in FS.EXTRA_ENV.items():
        args.insert(1, name + "=" + value)
    return args + list(extra)


def _request(schema, config, raw, environment, source_hashes, work, vram_mb, ram_mb, est_wall_s, timeout_s,
             backend, dtype, policy, reason, model):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    request = dict(schema=schema, config=config, model=model, task_family="forward", backend=backend, device="cuda",
                   dtype=dtype, batch_size=1, needs={"cuda": True}, execution_environment=environment,
                   config_bytes_sha256=hashlib.sha256(raw).hexdigest(), source_hashes=source_hashes,
                   scheduling_contract=one_lease_scheduling_contract(
                       model_loads=1, logical_operations=work, logical_branches=work, logical_checkpoint_replays=0,
                       max_local_work_items=FS.WORK_LIMIT, checkpoint_policy=policy),
                   resource_exception=dict(ram_mb=ram_mb, vram_mb=vram_mb, cpu_threads=4, disk_gb=3,
                                           est_wall_s=est_wall_s, timeout_s=timeout_s, reason=reason))
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    return request


def _stage(run, worker_path, config, extras, modules, work, vram_mb, ram_mb, est_wall_s, timeout_s, *,
           schema, backend, dtype, policy, reason, model, allow_tokens, extra_args=()):
    import mdb
    from mdb_lanes.payload import stage_worker
    environment = FS.bind()
    run = Path(run)
    run.mkdir(parents=True, exist_ok=False)
    if work > FS.WORK_LIMIT:
        raise ValueError(f"{work} work items exceed {FS.WORK_LIMIT}")
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker_path.name, preflight.name],
                     "path_token_allowlist": allow_tokens})
    extras = list(extras) + [(preflight, preflight.name)]
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    src = lanes_src()
    source_hashes = {worker_path.name: sha(worker_path), **{name: sha(path) for path, name in extras},
                     **{f"mdb_lanes/{m}.py": sha(src / f"{m}.py") for m in modules}}
    request = _request(schema, config, raw, environment, source_hashes, work, vram_mb, ram_mb, est_wall_s, timeout_s,
                       backend, dtype, policy, reason, model)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=worker_path, config=config, request=request, config_bytes=raw,
                 additional_files=extras)
    cmd = command(worker_path.name, extra_args)
    save(run / "staged-command.json", {"command": cmd, "request_sha256": request["request_sha256"],
                                        "worker_sha256": sha(run / "payload" / worker_path.name),
                                        "mdb_source": mdb.__file__, "lanes_source": str(src), "work_items": work})
    return dict(run=str(run), request_sha256=request["request_sha256"], worker=worker_path.name, work_items=work,
                mdb_source=mdb.__file__, lanes_source=str(src))


# ======================================================================================= capture

def capture_config(profile):
    words = {"snow": "cat"} if profile == "smoke" else dict(WORDS)
    closure_words = ["snow"] if profile == "smoke" else ["red", "snow"]
    swap_words = ["snow"] if profile == "smoke" else ["snow", "rain", "red", "ball"]
    return dict(schema="mdb-predicted-write-capture-v1", profile=profile, model="flux2-klein-4b-distilled",
                model_path=FS.MODEL_PATH, output_scratch=FS.SCRATCH, height=256, width=256, steps=4, seed=SEED,
                prompts=FS.PROMPTS, words=words, max_wall_seconds=600 if profile == "full" else 480,
                closure={"words": closure_words, "cells": CLOSURE_CELLS}, split=list(words),
                vswap={"words": swap_words, "arms": VSWAP_ARMS})


def stage_capture(run, profile, vram_mb, ram_mb, est_wall_s, timeout_s):
    config = capture_config(profile)
    n_words = len(config["words"])
    work = (n_words * (3 + 2 + 2) + len(config["closure"]["words"]) * len(CLOSURE_CELLS) * 5
            + len(config["vswap"]["words"]) * len(VSWAP_ARMS))
    extras = [(QUAL / "flux_read_panel_worker.py", "flux_read_panel_worker.py")]
    return _stage(run, HERE / "pw_worker.py", config, extras, PW_MODULES, work, vram_mb, ram_mb, est_wall_s, timeout_s,
                  schema="mdb-predicted-write-capture-request-v1", backend="diffusers", dtype="bfloat16",
                  policy="one resident Klein 4B lease; prompts encoded first, encoder released; roots, observer "
                         "runs, closure cut runs and value-swap arms in process; no retries",
                  reason="Klein 4B BF16 B1 256px 4-step resident lane (read-panel lane peak 7991 MB allocated) plus "
                         "FP32 observer tensors (~0.5 GB) and a CPU store of one run's q/k/v (~1.3 GB RAM)",
                  model="flux2-klein-4b-distilled", allow_tokens=[FS.MODEL_PATH, FS.SCRATCH, "predicted-write"])


# ======================================================================================= panels

def items_map():
    items = {w: {"prompt": pid, "word": w} for w, pid in WORDS.items()}
    items.update({pid: {"prompt": pid, "word": None} for pid in FS.PROMPTS})
    return items


def stage_panel(run, plans, plan, lease_index, vram_mb, ram_mb, est_wall_s, timeout_s, leases_file=None):
    from mdb_lanes.attention_panels import Panel
    from mdb_lanes.flux_read_panels import RUN_SCHEMA, lease_roots
    plans = Path(plans)
    plan_json = json.loads((plans / f"{plan}.plan.json").read_text())
    seal = json.loads((plans / f"{plan}.seal.json").read_text())
    leases_file = leases_file or f"{plan}.leases.json"
    leases = json.loads((plans / leases_file).read_text())
    panel = Panel.from_json(plan_json)
    if panel.sha256() != seal["plan_sha256"] or leases["plan_sha256"] != seal["plan_sha256"]:
        raise ValueError("plan, seal and leases disagree")
    lease = leases["leases"][lease_index]
    items = items_map()
    roots = lease_roots(panel, lease, items)
    work = len(lease["branches"]) + len(roots)
    suffix = leases_file.removeprefix(f"{plan}.leases").removesuffix(".json")
    tag = f"{plan}-{suffix + '-' if suffix else ''}{lease_index:02d}"
    passes = [dict(tag=tag, plan_file=f"plans/{plan}.plan.json", plan_sha256=seal["plan_sha256"],
                   seal_sha256=seal["seal_sha256"], lease=lease, precision="native", zero_method="delta",
                   reference_dir=None)]
    config = dict(schema=RUN_SCHEMA, job=tag, model="flux2-klein-4b-distilled", model_path=FS.MODEL_PATH,
                  output_scratch=FS.SCRATCH, height=256, width=256, steps=4, prompts=FS.PROMPTS, items=items,
                  contexts={str(SEED): {"seed": SEED}}, retain=[], max_wall_seconds=600, mode="panel",
                  passes=passes, hash_checkpoint=False, experiment="predicted_write_20261113")
    extras = [(plans / f"{plan}.plan.json", f"plans/{plan}.plan.json")]
    return _stage(run, QUAL / "flux_read_panel_worker.py", config, extras, ("attention_port", "attention_panels",
                                                                            "flux_read_panels"),
                  work, vram_mb, ram_mb, est_wall_s, timeout_s, schema="mdb-flux-read-panel-request-v1",
                  backend="diffusers", dtype="bfloat16",
                  policy="one resident Klein 4B lease; prompts encoded first, encoder released; reference roots then "
                         "sealed-lease branches in process; no retries",
                  reason="Klein 4B BF16 B1 256px 4-step resident lane; read-panel jobs measured 7991 MB peak allocated",
                  model="flux2-klein-4b-distilled", allow_tokens=[FS.MODEL_PATH, FS.SCRATCH, "flux-read-panel"])


# ========================================================================================= qwen

def stage_qwen(run, vram_mb, ram_mb, est_wall_s, timeout_s):
    from mdb_lanes import qwen_panels as Q
    config = dict(schema="mdb-predicted-write-qwen-v1", model=Q.MODEL, model_path=Q.MODEL_PATH,
                  expected_weight_file_sha256=Q.WEIGHT_FILE_SHA256, output_scratch=FS.SCRATCH, dtype="float32")
    return _stage(run, HERE / "qwen_pw_worker.py", config, [], QWEN_MODULES, 3, vram_mb, ram_mb, est_wall_s, timeout_s,
                  schema="mdb-predicted-write-qwen-request-v1", backend="transformers", dtype="float32",
                  policy="one Qwen2.5-1.5B FP32 load; two prefix caches and one receiver forward; no retries",
                  reason="Qwen2.5-1.5B FP32 eager B1, 22 tokens; Qwen panel jobs measured 6528 MB torch reserved",
                  model=Q.MODEL, allow_tokens=[Q.MODEL_PATH, FS.SCRATCH, "predicted-write-qwen"])


# ======================================================================================= submit

def submit(run, priority, attempt):
    FS.bind()
    from mdb_lanes.job_support import guarded_mrun_launch
    run = Path(run)
    request = json.loads((run / "request.json").read_text())
    staged = json.loads((run / "staged-command.json").read_text())
    if (run / "submission.json").exists():
        prev = json.loads((run / "submission.json").read_text())
        if attempt <= prev.get("attempt", 1):
            raise ValueError(f"already submitted as attempt {prev.get('attempt', 1)} ({prev.get('job_id')}); "
                             "pass --attempt N with N above it")
        (run / f"submission.attempt{prev.get('attempt', 1)}.json").write_text(json.dumps(prev, indent=1) + "\n")
    for name, digest in request["source_hashes"].items():
        if name.startswith("mdb_lanes/"):
            continue            # checked by the worker bootstrap against the shipped wheel
        if sha(run / "payload" / name) != digest:
            raise ValueError("staged payload changed: " + name)
    resource = request["resource_exception"]
    experiment = "mdb-predicted-write-" + run.name
    if attempt > 1:
        experiment += f"-attempt{attempt}"
    # Pin the declared reservation (measured peak x 1.3). Without the pin mrun's history clamp
    # lowered it to the sampled peak of an identical earlier run x 1.3 (7043 / 6503 MB), under
    # the model-load transient (tree VRAM 8022 MB): job-2eb545f0f377 and job-6c9b7875d02d
    # were killed_vram at load. The installed client's launch() does not forward a pin, so the
    # final submit call gets reservation["pin"] = True (server: reservation._is_pinned).
    import importlib
    mrun_submit = importlib.import_module("mrun.client.submit")      # the module (the package exports a function)
    original_submit = mrun_submit.submit

    def pinned_submit(*args, **kwargs):
        if kwargs.get("reservation"):
            kwargs["reservation"] = {**kwargs["reservation"], "pin": True}
        return original_submit(*args, **kwargs)
    mrun_submit.submit = pinned_submit
    job = guarded_mrun_launch(staged["command"], request=request, experiment=experiment, task="forward",
                              model=request["model"], payload=run / "payload", prefer_host="beast", gpu=True,
                              needs=request["needs"], ram_mb=resource["ram_mb"], vram_mb=resource["vram_mb"],
                              disk_gb=resource["disk_gb"], cpu_threads=resource["cpu_threads"],
                              est_wall_s=resource["est_wall_s"], timeout_s=resource["timeout_s"], priority=priority,
                              retry_on_kill=False, preflight=True,
                              note="mdb predicted-write experiment 2026-11-13: " + run.name)
    (run / "submission.json").write_text(json.dumps({"job_id": str(job), "attempt": attempt}, indent=1) + "\n")
    return dict(job_id=str(job), run=str(run))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture")
    p.add_argument("run", type=Path)
    p.add_argument("--profile", choices=("smoke", "full"), required=True)
    p.add_argument("--vram-mb", type=int, default=12000)
    p.add_argument("--ram-mb", type=int, default=20000)
    p.add_argument("--est-wall-s", type=int, default=420)
    p.add_argument("--timeout-s", type=int, default=900)
    p = sub.add_parser("plans")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("panel")
    p.add_argument("run", type=Path)
    p.add_argument("--plans", type=Path, required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--lease", type=int, required=True)
    p.add_argument("--leases-file", default=None,
                   help="leases file in --plans (default <plan>.leases.json); main.leases-r2.json repacks main "
                        "without replicated branches (flux_read_panels.check_lease refuses those)")
    p.add_argument("--vram-mb", type=int, default=10400)
    p.add_argument("--ram-mb", type=int, default=20000)
    p.add_argument("--est-wall-s", type=int, default=400)
    p.add_argument("--timeout-s", type=int, default=900)
    p = sub.add_parser("qwen")
    p.add_argument("run", type=Path)
    p.add_argument("--vram-mb", type=int, default=8487)
    p.add_argument("--ram-mb", type=int, default=12645)
    p.add_argument("--est-wall-s", type=int, default=240)
    p.add_argument("--timeout-s", type=int, default=600)
    p = sub.add_parser("submit")
    p.add_argument("run", type=Path)
    p.add_argument("--priority", type=int, default=70)
    p.add_argument("--attempt", type=int, default=1)
    a = parser.parse_args()
    if a.cmd == "capture":
        print(json.dumps(stage_capture(a.run.resolve(), a.profile, a.vram_mb, a.ram_mb, a.est_wall_s, a.timeout_s)))
    elif a.cmd == "plans":
        import plans as PL
        print(json.dumps(PL.write(a.capture, a.out), indent=1))
    elif a.cmd == "panel":
        print(json.dumps(stage_panel(a.run.resolve(), a.plans, a.plan, a.lease, a.vram_mb, a.ram_mb, a.est_wall_s,
                                     a.timeout_s, a.leases_file)))
    elif a.cmd == "qwen":
        print(json.dumps(stage_qwen(a.run.resolve(), a.vram_mb, a.ram_mb, a.est_wall_s, a.timeout_s)))
    else:
        print(json.dumps(submit(a.run.resolve(), a.priority, a.attempt)))

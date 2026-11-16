"""Submit worker.py (and demo.py) to Beast through mrun, through isa-bench's launcher (uv_sync=false, pydeps).

    python submit.py --programs bsort-O2 [--grid] [--demo] [--wait] [--fetch DIR]

Payload: src/mdb (this checkout), src/torchx86 and src/isa_bench/{__init__,state}.py (isa-bench, imported by
path, never copied into the repo), the two scripts, and pinned wheels in pydeps/. Data stays on Beast and
is read in place (``--ref``, ``--containers``).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MDB_SRC = HERE.parents[2] / "mdb/src/mdb"
ISA = Path(os.environ.get("ISA_BENCH_REPO", Path.home() / "domains/isa-bench"))
REF = "/mnt/hdd8tb/uv-archive-stage/isa-bench/reference/gt-66b4b67cc82e42b3"
CONT = "/mnt/hdd8tb/uv-archive-stage/isa-bench/binctr/gt-run0-job-98886bb1e7d0/containers"
PINS = ("capstone==5.0.6", "pyelftools==0.32")
EXECUTOR = {"host": "beast", "priority": 70, "uv_sync": False, "remote_work": "/mnt/big/mrun-scratch"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", nargs="+", default=[])
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--run6", nargs="+", default=[], help="programs for run 6 (run6.py)")
    ap.add_argument("--run6-args", nargs="*", default=[])
    ap.add_argument("--vram-mb", type=int, default=6000)
    ap.add_argument("--est-wall-s", type=int, default=420)
    ap.add_argument("--note", default="mdb x86 backend: binary-ground-truth method")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--fetch", type=Path)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ISA / "src"))
    from isa_bench import launch

    run_dir = Path(tempfile.mkdtemp(prefix="mdbx86-"))
    payload = run_dir / "payload"
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(MDB_SRC, payload / "src/mdb", ignore=ign)
    shutil.copytree(ISA / "models/torchx86/src/torchx86", payload / "src/torchx86", ignore=ign)
    (payload / "src/isa_bench").mkdir(parents=True)
    for f in ("__init__.py", "state.py"):
        shutil.copy2(ISA / "src/isa_bench" / f, payload / "src/isa_bench" / f)
    for f in ("worker.py", "demo.py", "run6.py"):
        shutil.copy2(HERE / f, payload / f)
    launch.stage_pydeps(payload, list(PINS))
    steps = []
    if a.programs:
        steps.append(["python", "worker.py", "--ref", REF, "--containers", CONT, "--programs", *a.programs,
                      "--device", "cuda", "--out", "result", *(["--grid"] if a.grid else [])])
    if a.demo:
        steps.append(["python", "demo.py", "--ref", REF, "--containers", CONT, "--device", "cuda",
                      "--out", "result/demo.txt"])
    if a.run6:
        steps.append(["python", "run6.py", "--ref", REF, "--containers", CONT, "--programs", *a.run6,
                      "--device", "cuda", "--out", "result", *a.run6_args])
    job = {"programs": a.programs, "grid": a.grid, "demo": a.demo, "run6": a.run6, "ref": REF, "containers": CONT}
    launch.write(payload / "job.json", job)
    # the launcher runs `uv run ... python <entry>`: a tiny shim runs run.sh's steps with that python
    (payload / "steps.py").write_text(
        "import os, subprocess, sys\nos.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'\n"
        f"for s in {json.dumps(steps)}:\n"
        "    subprocess.run([sys.executable, *s[1:]], check=True)\n")
    roots = sorted(p.name for p in payload.iterdir())
    launch.write(payload / "preflight.v1.json", {"path_token_allowlist": ["{env}", "result"], "hashed_roots": roots})
    launch.write(payload / "source-manifest.json",
                 {str(p.relative_to(payload)): launch.sha(p) for p in sorted(payload.rglob("*")) if p.is_file()})
    job_id = launch.submit(run_dir, payload, job, executor=EXECUTOR, experiment="mdb-binary-ground-truth",
                           task="mdb-x86-method", note=a.note, ram_mb=12000, cpu_threads=2, gpu=True,
                           vram_mb=a.vram_mb, est_wall_s=a.est_wall_s, timeout_s=600, entry=("steps.py",))
    if a.wait:
        meta = launch.wait(job_id, run_dir, max_wait_s=1200)
        print((run_dir / "job.log").read_text()[-6000:])
        if a.fetch and meta.get("state") == "succeeded":
            launch.fetch(job_id, a.fetch / job_id, executor=EXECUTOR)
            print(json.dumps({"fetched": str(a.fetch / job_id)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

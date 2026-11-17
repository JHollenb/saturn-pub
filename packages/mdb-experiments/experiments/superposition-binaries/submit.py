"""Submit study.py to Beast through mrun (isa-bench launcher; uv_sync=false, pinned pydeps).

    python submit.py --collect [--score-args ...] [--wait] [--fetch DIR]
    python submit.py --capture /mnt/big/mrun-scratch/<job>/result/capture --score-args --pooled ... [--wait]

Payload: src/mdb (this checkout), src/torchx86 + src/isa_bench/{__init__,state}.py (isa-bench, by path),
study.py, sup.py. The capture stays on Beast; --fetch copies only *.json back (the capture is ~180 MB).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
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
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--capture", default=None)
    ap.add_argument("--score-args", nargs=argparse.REMAINDER, default=[])
    ap.add_argument("--vram-mb", type=int, default=6000)
    ap.add_argument("--est-wall-s", type=int, default=420)
    ap.add_argument("--note", default="study S: controlled superposition on torchx86 gated-MLP f")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--fetch", type=Path)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ISA / "src"))
    from isa_bench import launch

    run_dir = Path(tempfile.mkdtemp(prefix="supS-"))
    payload = run_dir / "payload"
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(MDB_SRC, payload / "src/mdb", ignore=ign)
    shutil.copytree(ISA / "models/torchx86/src/torchx86", payload / "src/torchx86", ignore=ign)
    (payload / "src/isa_bench").mkdir(parents=True)
    for f in ("__init__.py", "state.py"):
        shutil.copy2(ISA / "src/isa_bench" / f, payload / "src/isa_bench" / f)
    for f in ("study.py", "sup.py"):
        shutil.copy2(HERE / f, payload / f)
    launch.stage_pydeps(payload, list(PINS))
    cmd = ["python", "study.py", "--device", "cuda", "--out", "result"]
    if a.collect:
        cmd += ["--collect", "--ref", REF, "--containers", CONT]
    if a.capture:
        cmd += ["--capture", a.capture]
    if a.score_args:
        cmd += ["--score", *a.score_args]
    steps = [cmd]
    job = {"study": "S", "steps": steps, "ref": REF, "containers": CONT}
    launch.write(payload / "job.json", job)
    (payload / "steps.py").write_text(
        "import os, subprocess, sys\nos.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'\n"
        f"for s in {json.dumps(steps)}:\n"
        "    subprocess.run([sys.executable, *s[1:]], check=True)\n")
    roots = sorted(p.name for p in payload.iterdir())
    launch.write(payload / "preflight.v1.json", {"path_token_allowlist": ["{env}", "result"], "hashed_roots": roots})
    launch.write(payload / "source-manifest.json",
                 {str(p.relative_to(payload)): launch.sha(p) for p in sorted(payload.rglob("*")) if p.is_file()})
    job_id = launch.submit(run_dir, payload, job, executor=EXECUTOR, experiment="mdb-superposition-S",
                           task="mdb-x86-superposition", note=a.note, ram_mb=12000, cpu_threads=2, gpu=True,
                           vram_mb=a.vram_mb, est_wall_s=a.est_wall_s, timeout_s=600, entry=("steps.py",))
    if a.wait:
        meta = launch.wait(job_id, run_dir, max_wait_s=1500)
        print((run_dir / "job.log").read_text()[-8000:])
        if a.fetch and meta.get("state") == "succeeded":
            dest = a.fetch / job_id
            dest.mkdir(parents=True, exist_ok=True)
            remote = f"beast:{launch.remote_result(job_id, EXECUTOR)}/"
            subprocess.run(["rsync", "-a", "--include=*/", "--include=*.json", "--exclude=*", remote, str(dest) + "/"],
                           check=True, timeout=600)
            print(json.dumps({"fetched": str(dest)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

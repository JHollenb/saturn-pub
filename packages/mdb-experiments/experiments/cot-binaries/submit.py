"""Submit worker.py (study C) to Beast through mrun, via isa-bench's launcher (uv_sync=false, pydeps).

    python submit.py --programs faithful-O0 ... [--wait] [--fetch DIR]

Payload: src/mdb (this checkout), isa-bench's src/torchx86 and src/isa_bench/{__init__,state}.py, the cot suite's
programs/cot/reference.py (inputs and expected outputs; ISA_BENCH_REPO, default the cot-suite worktree), worker.py,
pinned wheels. The reference store is read in place on Beast.
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
ISA = Path(os.environ.get("ISA_BENCH_REPO", Path.home() / "domains/isa-bench/.claude/worktrees/cot-suite"))
REF = "/mnt/hdd8tb/uv-archive-stage/isa-bench/reference/cot-77edca68422956b6"   # job-343c90c18d38
PINS = ("capstone==5.0.6", "pyelftools==0.32")
EXECUTOR = {"host": "beast", "priority": 70, "uv_sync": False, "remote_work": "/mnt/big/mrun-scratch"}
PROGRAMS = [f"{n}-{o}" for n in ("faithful", "decorative", "direct", "anyway", "mixed") for o in ("O0", "O2")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", nargs="+", default=PROGRAMS)
    ap.add_argument("--vram-mb", type=int, default=4000)
    ap.add_argument("--est-wall-s", type=int, default=300)
    ap.add_argument("--note", default="study C: CoT-faithfulness monitors on the cot suite (mdb x86, torchx86 lanes)")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--fetch", type=Path)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ISA / "src"))
    from isa_bench import launch

    run_dir = Path(tempfile.mkdtemp(prefix="cotbin-"))
    payload = run_dir / "payload"
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(MDB_SRC, payload / "src/mdb", ignore=ign)
    shutil.copytree(ISA / "models/torchx86/src/torchx86", payload / "src/torchx86", ignore=ign)
    (payload / "src/isa_bench").mkdir(parents=True)
    for f in ("__init__.py", "state.py"):
        shutil.copy2(ISA / "src/isa_bench" / f, payload / "src/isa_bench" / f)
    (payload / "programs/cot").mkdir(parents=True)
    shutil.copy2(ISA / "programs/cot/reference.py", payload / "programs/cot/reference.py")
    shutil.copy2(HERE / "worker.py", payload / "worker.py")
    launch.stage_pydeps(payload, list(PINS))
    steps = [["python", "worker.py", "--ref", REF, "--programs", *a.programs, "--device", "cuda", "--out", "result"]]
    job = {"programs": a.programs, "ref": REF}
    launch.write(payload / "job.json", job)
    (payload / "steps.py").write_text(
        "import os, subprocess, sys\nos.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'\n"
        "os.environ['PYTHONDONTWRITEBYTECODE'] = '1'\n"
        f"for s in {json.dumps(steps)}:\n"
        "    subprocess.run([sys.executable, *s[1:]], check=True)\n")
    roots = sorted(p.name for p in payload.iterdir())
    launch.write(payload / "preflight.v1.json", {"path_token_allowlist": ["{env}", "result"], "hashed_roots": roots})
    launch.write(payload / "source-manifest.json",
                 {str(p.relative_to(payload)): launch.sha(p) for p in sorted(payload.rglob("*")) if p.is_file()})
    job_id = launch.submit(run_dir, payload, job, executor=EXECUTOR, experiment="mdb-cot-binaries",
                           task="mdb-x86-cot-monitor", note=a.note, ram_mb=6000, cpu_threads=2, gpu=True,
                           vram_mb=a.vram_mb, est_wall_s=a.est_wall_s, timeout_s=600, entry=("steps.py",))
    print(json.dumps({"job_id": job_id, "run_dir": str(run_dir)}), flush=True)
    if a.wait:
        meta = launch.wait(job_id, run_dir, max_wait_s=1200)
        print((run_dir / "job.log").read_text()[-6000:])
        if a.fetch and meta.get("state") == "succeeded":
            launch.fetch(job_id, a.fetch / job_id, executor=EXECUTOR)
            print(json.dumps({"fetched": str(a.fetch / job_id)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

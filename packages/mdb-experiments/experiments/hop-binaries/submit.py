"""Submit hop_l.py (study L) to Beast through mrun, through isa-bench's launcher (priority 70, uv_sync=false).

    python submit.py --phase passive [--wait] [--fetch DIR]
    python submit.py --phase interchange --forecasts DIR [--wait] [--fetch DIR]

Payload: src/mdb (this checkout), torchx86 and isa_bench/{__init__,state}.py + isa_bench/gtkey (the hop-suite
isa-bench worktree, imported by path), programs/hop/{reference.py,tables.h}, hop_l.py, and pinned wheels in
pydeps/. The native reference store and binctr run-0 containers are read in place on Beast.
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
ISA = Path(os.environ.get("ISA_BENCH_REPO", Path.home() / "domains/isa-bench/.claude/worktrees/hop-suite"))
REF = "/mnt/hdd8tb/uv-archive-stage/isa-bench/reference/hop-5992b0b7b58baf3a/programs"     # job-89d89b1e541e
CONT = "/mnt/hdd8tb/uv-archive-stage/isa-bench/binctr/hop-run0-job-c37e6041d4d3/containers"  # job-c37e6041d4d3
TAGS = [f"{p}-{o}" for p in ("hop2", "arith", "hop3", "shortcut") for o in ("O0", "O2")]
PINS = ("capstone==5.0.6", "pyelftools==0.32")
EXECUTOR = {"host": "beast", "priority": 70, "uv_sync": False, "remote_work": "/mnt/big/mrun-scratch"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("passive", "interchange"), required=True)
    ap.add_argument("--programs", nargs="+", default=TAGS)
    ap.add_argument("--forecasts", type=Path, help="local directory of <tag>.passive.json (interchange phase)")
    ap.add_argument("--vram-mb", type=int, default=3000)
    ap.add_argument("--est-wall-s", type=int, default=300)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--fetch", type=Path)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ISA / "src"))
    from isa_bench import launch

    run_dir = Path(tempfile.mkdtemp(prefix="hopL-"))
    payload = run_dir / "payload"
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(MDB_SRC, payload / "src/mdb", ignore=ign)
    shutil.copytree(ISA / "models/torchx86/src/torchx86", payload / "src/torchx86", ignore=ign)
    (payload / "src/isa_bench").mkdir(parents=True)
    for f in ("__init__.py", "state.py"):
        shutil.copy2(ISA / "src/isa_bench" / f, payload / "src/isa_bench" / f)
    shutil.copytree(ISA / "src/isa_bench/gtkey", payload / "src/isa_bench/gtkey", ignore=ign)
    (payload / "programs/hop").mkdir(parents=True)
    for f in ("reference.py", "tables.h"):
        shutil.copy2(ISA / "programs/hop" / f, payload / "programs/hop" / f)
    shutil.copy2(HERE / "hop_l.py", payload / "hop_l.py")
    launch.stage_pydeps(payload, list(PINS))
    step = ["python", "hop_l.py", "--phase", a.phase, "--ref", REF, "--containers", CONT, "--programs-dir",
            "programs/hop", "--programs", *a.programs, "--device", "cuda", "--out", "result"]
    if a.phase == "interchange":
        shutil.copytree(a.forecasts, payload / "forecasts", ignore=shutil.ignore_patterns("*.interchange.json"))
        step += ["--forecasts", "forecasts"]
    job = {"phase": a.phase, "programs": a.programs, "ref": REF, "containers": CONT}
    launch.write(payload / "job.json", job)
    (payload / "steps.py").write_text(
        "import os, subprocess, sys\nos.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'\n"
        "os.environ['PYTHONPATH'] = os.pathsep.join([os.path.abspath('src'), os.path.abspath('pydeps'),"
        " os.environ.get('PYTHONPATH', '')])\n"
        f"subprocess.run([sys.executable, *{json.dumps(step[1:])}], check=True)\n")
    roots = sorted(p.name for p in payload.iterdir())
    launch.write(payload / "preflight.v1.json", {"path_token_allowlist": ["{env}", "result"], "hashed_roots": roots})
    launch.write(payload / "source-manifest.json",
                 {str(p.relative_to(payload)): launch.sha(p) for p in sorted(payload.rglob("*")) if p.is_file()})
    job_id = launch.submit(run_dir, payload, job, executor=EXECUTOR, experiment="mdb-hop-binaries",
                           task="mdb-x86-hop-l", note=f"study L {a.phase}", ram_mb=6000, cpu_threads=2, gpu=True,
                           vram_mb=a.vram_mb, est_wall_s=a.est_wall_s, timeout_s=600, entry=("steps.py",))
    print(json.dumps({"job_id": job_id, "run_dir": str(run_dir)}), flush=True)
    if a.wait:
        meta = launch.wait(job_id, run_dir, max_wait_s=1200)
        print((run_dir / "job.log").read_text()[-4000:])
        if a.fetch and meta.get("state") == "succeeded":
            launch.fetch(job_id, a.fetch / job_id, executor=EXECUTOR)
            print(json.dumps({"fetched": str(a.fetch / job_id)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

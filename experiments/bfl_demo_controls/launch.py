"""Submit one BFL demo-control experiment to a GPU worker through the public mrun client.

Reuses the saturn-pub diffusion-families submission pattern (one finite CUDA lease, offline
uv overlay on the operator ``mrun`` env, scratch/outputs on /mnt, weights from <model-root>) but
runs ``run.py`` directly so a non-zero worker exit fails the job (no exit-code swallowing).

Usage (from the worktree, with the mrun-pub client):
    MRUN_URL=http://<mrun-host>:9025 \
    /path/to/mrun-pub/.venv/bin/python experiments/bfl_demo_controls/launch.py \
        --experiment exp1 --vram-mb 14200 --ram-mb 28000 --wall-s 600 --timeout-s 1800
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path

from mrun.client.submit import launch

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
MODEL = "<model-root>/FLUX.2-klein-4B"
SCRATCH = "<scratch-root>/bfl-controls"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", choices=["exp1", "exp2", "exp2viab", "exp2b"], required=True)
    ap.add_argument("--vram-mb", type=int, required=True)
    ap.add_argument("--ram-mb", type=int, required=True)
    ap.add_argument("--wall-s", type=int, default=900)
    ap.add_argument("--timeout-s", type=int, default=2400)
    ap.add_argument("--priority", type=int, default=100)
    ap.add_argument("--size", type=int, default=0)
    args = ap.parse_args()

    label = f"{args.experiment}-{uuid.uuid4().hex[:6]}"
    out_dir = f"{SCRATCH}/{label}"

    staging = HERE / "staging" / label
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copytree(
        WORKTREE / "src/saturn_pub",
        staging / "src/saturn_pub",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copyfile(HERE / "run.py", staging / "run.py")
    source_sha256 = {
        str(p.relative_to(staging)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(staging.rglob("*"))
        if p.is_file()
    }
    (staging / "source_sha256.json").write_text(json.dumps(source_sha256, indent=2))

    size = args.size or (256 if args.experiment == "exp1" else 512)
    config = {
        "experiment": args.experiment,
        "model": MODEL,
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "size": size,
        "steps": 4,
        "output": out_dir,
        "execution": {"device": "cuda", "dtype": "bfloat16", "batch": 1, "guidance": 1.0},
        "source_sha256": source_sha256,
    }

    command = [
        "/usr/bin/env",
        "PYTHONUNBUFFERED=1",
        "PYTHONNOUSERSITE=1",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "DIFFUSERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false",
        "TQDM_DISABLE=1",
        "UV_CACHE_DIR=<model-root>/.uv-cache",
        "HF_HOME=<model-root>/hf",
        f"TORCH_HOME={SCRATCH}/cache/torch",
        f"XDG_CACHE_HOME={SCRATCH}/cache",
        f"TMPDIR={SCRATCH}/tmp",
        "PYTHONPATH=src",
        "uv",
        "run",
        "--no-project",
        "--offline",
        "--with",
        "diffusers==0.39.0",
        "--with",
        "transformers==5.14.1",
        "--with",
        "accelerate==1.14.0",
        "--with",
        "safetensors>=0.5",
        "--with",
        "numpy>=2.0",
        "--with",
        "Pillow>=9.0",
        "python",
        "-u",
        "run.py",
        "--experiment",
        args.experiment,
        "--model",
        MODEL,
        "--output",
        out_dir,
        "--size",
        str(size),
        "--steps",
        "4",
        "--source-sha256",
        hashlib.sha256(json.dumps(source_sha256, sort_keys=True).encode()).hexdigest(),
    ]

    note = (
        f"BFL demo control {args.experiment} on FLUX.2 Klein 4B (rev e7b7dc27), BF16, 4 steps, "
        f"{size}px, guidance 1.0. Smoke gate first (stepped-vs-native + StateCut resume + "
        f"zero-dose identity), abort on failure (non-zero exit fails job), else full panel. "
        f"Outputs under {out_dir} on <scratch-root>; weights from <model-root>; one model load. "
        f"Stop condition: single finite pass, no retry."
    )

    job = launch(
        command,
        experiment=f"bfl-demo-controls-{args.experiment}",
        payload=staging,
        prefer_host=None,
        needs={"cuda": True},
        config=config,
        ram_mb=args.ram_mb,
        vram_mb=args.vram_mb,
        cpu_threads=4,
        est_wall_s=args.wall_s,
        timeout_s=args.timeout_s,
        priority=args.priority,
        env_alias="mrun",
        retry_on_kill=False,
        preflight="warn",
        detach=True,
        note=note,
    )
    submission = {
        "job_id": job,
        "label": label,
        "experiment": args.experiment,
        "output_dir": out_dir,
        "size": size,
        "config": config,
    }
    (HERE / f"submission-{args.experiment}.json").write_text(json.dumps(submission, indent=2))
    print(json.dumps({"job_id": job, "label": label, "output_dir": out_dir}))


if __name__ == "__main__":
    main()
